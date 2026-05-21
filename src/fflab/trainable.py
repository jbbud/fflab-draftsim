from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .ai import DraftPolicy
from .config import FLEX_ELIGIBLE_POSITIONS, STARTER_POSITIONS
from .draft import DraftState

FEATURE_NAMES = [
    "season_points",
    "player_rank",
    "position_rank",
    "positional_dropoff",
    "need_tier",
    "flex_eligible",
    "round_progress",
    "pick_position",
    "remaining_position_scarcity",
    "starter_pressure",
    "bench_pressure",
]

DEFAULT_WEIGHT_VALUES = {
    "season_points": 2.0,
    "player_rank": 0.5,
    "position_rank": 0.35,
    "positional_dropoff": 0.4,
    "need_tier": 0.9,
    "flex_eligible": 0.05,
    "round_progress": 0.0,
    "pick_position": 0.0,
    "remaining_position_scarcity": 0.2,
    "starter_pressure": 0.25,
    "bench_pressure": 0.05,
}


@dataclass(frozen=True)
class DraftPolicyWeights:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHT_VALUES))
    intercept: float = 0.0
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)

    def vector(self) -> list[float]:
        return [float(self.weights.get(name, 0.0)) for name in FEATURE_NAMES]

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_names": FEATURE_NAMES,
            "weights": {name: float(self.weights.get(name, 0.0)) for name in FEATURE_NAMES},
            "intercept": float(self.intercept),
            "metadata": self.metadata,
            "training_config": self.training_config,
            "validation_summary": self.validation_summary,
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DraftPolicyWeights:
        raw_weights = payload.get("weights", {})
        weights = {
            name: float(raw_weights.get(name, DEFAULT_WEIGHT_VALUES.get(name, 0.0)))
            for name in FEATURE_NAMES
        }
        return cls(
            weights=weights,
            intercept=float(payload.get("intercept", 0.0)),
            version=int(payload.get("version", 1)),
            metadata=dict(payload.get("metadata", {})),
            training_config=dict(payload.get("training_config", {})),
            validation_summary=dict(payload.get("validation_summary", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> DraftPolicyWeights:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_payload(payload)


class WeightedDraftPolicy(DraftPolicy):
    def __init__(self, weights: DraftPolicyWeights, name: str = "weighted"):
        self.weights = weights
        self.name = name

    def choose_pick(self, state: DraftState, team_index: int) -> str:
        features = candidate_feature_frame(state, team_index)
        if features.empty:
            raise RuntimeError(f"team {team_index} has no legal picks")
        vector = pd.Series(self.weights.weights, dtype="float64")
        scores = features[FEATURE_NAMES].mul(vector, axis=1).sum(axis=1)
        scores = scores + self.weights.intercept
        best_index = scores.idxmax()
        return str(features.loc[best_index, "player_id"])


class MissingTrainedPolicyFallback(WeightedDraftPolicy):
    def __init__(self, path: str | Path):
        super().__init__(DraftPolicyWeights(), name=f"trained_missing:{path}")
        self.path = str(path)


def load_trained_policy(path: str | Path) -> WeightedDraftPolicy:
    policy_path = Path(path)
    if not policy_path.exists():
        return MissingTrainedPolicyFallback(policy_path)
    return WeightedDraftPolicy(
        DraftPolicyWeights.load(policy_path),
        name=f"trained:{policy_path}",
    )


def candidate_feature_frame(state: DraftState, team_index: int) -> pd.DataFrame:
    candidates = state.available_players_df()
    if candidates.empty:
        return pd.DataFrame(columns=["player_id", *FEATURE_NAMES])

    candidates = candidates[
        candidates["player_id"].map(lambda player_id: state.can_add_player(team_index, player_id))
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["player_id", *FEATURE_NAMES])

    players = state.players.copy()
    max_points = max(float(players["season_total_pts"].max()), 1.0)
    total_players = max(len(players) - 1, 1)
    players["global_rank"] = range(len(players))
    players["player_rank_feature"] = 1.0 - players["global_rank"] / total_players
    players["position_rank"] = players.groupby("position").cumcount()
    position_sizes = players.groupby("position")["player_id"].transform("count").clip(lower=1)
    players["position_rank_feature"] = 1.0 - (
        players["position_rank"] / (position_sizes - 1).where(position_sizes > 1, 1)
    )
    rank_features = players.set_index("player_id")[
        ["player_rank_feature", "position_rank_feature"]
    ]

    candidates = candidates.join(rank_features, on="player_id")
    candidates["season_points"] = candidates["season_total_pts"].astype(float) / max_points
    candidates["player_rank"] = candidates["player_rank_feature"].fillna(0.0)
    candidates["position_rank"] = candidates["position_rank_feature"].fillna(0.0)
    candidates["need_tier"] = candidates["position"].map(
        lambda position: state.need_tier(team_index, str(position)) / 2.0
    )
    candidates["flex_eligible"] = candidates["position"].isin(FLEX_ELIGIBLE_POSITIONS).astype(float)
    candidates["round_progress"] = state.current_round / max(state.rounds, 1)
    candidates["pick_position"] = (
        team_index / max(state.config.num_teams - 1, 1)
        if state.config.num_teams > 1
        else 0.0
    )

    remaining_counts = candidates.groupby("position")["player_id"].transform("count")
    total_counts = players.groupby("position")["player_id"].transform("count")
    total_counts_by_position = players.groupby("position")["player_id"].count().to_dict()
    candidates["remaining_position_scarcity"] = candidates["position"].map(
        lambda position: 1.0
        - (
            float(remaining_counts[candidates["position"].eq(position)].iloc[0])
            / max(float(total_counts_by_position.get(position, 1)), 1.0)
        )
    )

    candidates["positional_dropoff"] = 0.0
    for position, group in candidates.groupby("position", sort=False):
        ordered = group.sort_values("season_total_pts", ascending=False)
        next_points = ordered["season_total_pts"].shift(-1).fillna(0.0)
        dropoff = (ordered["season_total_pts"] - next_points).clip(lower=0.0) / max_points
        candidates.loc[ordered.index, "positional_dropoff"] = dropoff

    needs = state.roster_needs(team_index)
    starter_need = sum(needs.get(position, 0) for position in STARTER_POSITIONS) + needs.get(
        "FLEX", 0
    )
    starter_capacity = sum(
        state.config.roster_settings.get(position, 0)
        for position in (*STARTER_POSITIONS, "FLEX")
    )
    remaining_slots = max(state.config.total_roster_slots - state.roster_size(team_index), 0)
    bench_capacity = max(state.config.roster_settings.get("BENCH", 0), 1)
    candidates["starter_pressure"] = starter_need / max(starter_capacity, 1)
    candidates["bench_pressure"] = max(remaining_slots - starter_need, 0) / bench_capacity

    return candidates[["player_id", *FEATURE_NAMES]].reset_index(drop=True)
