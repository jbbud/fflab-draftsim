from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
import os

import numpy as np

from .ai import DraftPolicy
from .config import FLEX_ELIGIBLE_POSITIONS, LeagueConfig, STARTER_POSITIONS
from .draft import DraftState
from .scoring import ScoredData
from .training import (
    FLEX_POSITION_IDXS,
    POSITIONS,
    POS_TO_IDX,
    FastDraftData,
    FastDraftState,
    PolicySpec,
    build_fast_draft_data,
    choose_fast_pick,
    reward_for_team,
    simulate_fast_draft,
)

ProgressCallback = Callable[[str, int, int | None], None]
_STATIC_FEATURE_CACHE: dict[tuple[int, int, int, float, float], dict[str, np.ndarray]] = {}
_FAST_DATA_CACHE: dict[tuple[int, tuple[str, ...], int, int], FastDraftData] = {}
ScoredInput = ScoredData | Mapping[int, ScoredData]


def _get_fast_data(scored: ScoredData, config: LeagueConfig) -> FastDraftData:
    key = (
        id(scored),
        tuple(config.draftable_positions),
        len(scored.players),
        len(scored.weekly_scores),
    )
    cached = _FAST_DATA_CACHE.get(key)
    if cached is None:
        cached = build_fast_draft_data(scored, config)
        _FAST_DATA_CACHE[key] = cached
    return cached


def _normalize_scored_input(scored: ScoredInput) -> dict[int, ScoredData]:
    if isinstance(scored, Mapping):
        return {int(season): season_scored for season, season_scored in scored.items()}
    seasons = scored.players["season"].dropna().unique() if "season" in scored.players.columns else []
    season = int(seasons[0]) if len(seasons) == 1 else 0
    return {season: scored}


def _primary_scored(scored: ScoredInput, target_season: int | None = None) -> tuple[int, ScoredData]:
    scored_by_season = _normalize_scored_input(scored)
    if target_season is not None and target_season in scored_by_season:
        return target_season, scored_by_season[target_season]
    season = sorted(scored_by_season)[-1]
    return season, scored_by_season[season]


def _torch():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for neural draft policies. Install the training extra "
            "or use an existing heuristic/weighted policy."
        ) from exc
    torch.set_num_threads(1)
    return torch, nn


def neural_feature_names(week_count: int) -> list[str]:
    names = [f"week_{index + 1}" for index in range(week_count)]
    names.extend(
        [
            "season_total",
            "weekly_mean",
            "weekly_max",
            "weekly_std",
            "zero_week_rate",
            "top3_mean",
            "top5_mean",
            "late_season_total",
        ]
    )
    names.extend([f"position_{position}" for position in POSITIONS])
    names.extend(
        [
            "position_rank",
            "need_tier",
            "flex_eligible",
            "round_progress",
            "pick_position",
            "starter_pressure",
            "bench_pressure",
            "roster_best_mean",
            "roster_best_max",
            "candidate_gain_mean",
            "candidate_gain_max",
            "candidate_overlap_mean",
            "candidate_better_week_rate",
            "candidate_roster_corr",
        ]
    )
    return names


@dataclass(frozen=True)
class NeuralModelConfig:
    input_dim: int
    hidden_dim: int = 64


class NeuralDraftNet:
    @staticmethod
    def build(input_dim: int, hidden_dim: int = 64):
        torch, nn = _torch()
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 8), 1),
        )


@dataclass(frozen=True)
class NeuralPolicyArtifact:
    model_state: dict[str, Any]
    input_dim: int
    week_count: int
    hidden_dim: int
    max_weekly_point: float
    target_mean: float = 0.0
    target_std: float = 1.0
    version: int = 1
    metadata: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    validation_summary: dict[str, Any] | None = None

    def save(self, path: str | Path) -> None:
        torch, _ = _torch()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": self.version,
                "feature_names": neural_feature_names(self.week_count),
                "model_state": self.model_state,
                "input_dim": self.input_dim,
                "week_count": self.week_count,
                "hidden_dim": self.hidden_dim,
                "max_weekly_point": self.max_weekly_point,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "metadata": self.metadata or {},
                "training_config": self.training_config or {},
                "validation_summary": self.validation_summary or {},
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path) -> NeuralPolicyArtifact:
        torch, _ = _torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            model_state=payload["model_state"],
            input_dim=int(payload["input_dim"]),
            week_count=int(payload["week_count"]),
            hidden_dim=int(payload.get("hidden_dim", 64)),
            max_weekly_point=float(payload.get("max_weekly_point", 1.0)),
            target_mean=float(payload.get("target_mean", 0.0)),
            target_std=float(payload.get("target_std", 1.0)),
            version=int(payload.get("version", 1)),
            metadata=dict(payload.get("metadata", {})),
            training_config=dict(payload.get("training_config", {})),
            validation_summary=dict(payload.get("validation_summary", {})),
        )


@dataclass
class FastPolicyContext:
    neural_artifact: NeuralPolicyArtifact | None = None
    neural_model: Any | None = None
    trained_cache: dict[str, np.ndarray] = field(default_factory=dict)
    neural_cache: dict[str, tuple[NeuralPolicyArtifact, Any]] = field(default_factory=dict)
    score_cache: dict[tuple[int, int, int, tuple[int, ...]], np.ndarray] = field(default_factory=dict)


def _notify_progress(
    callback: ProgressCallback | None,
    name: str,
    advance: int = 1,
    total: int | None = None,
) -> None:
    if callback is not None:
        callback(name, advance, total)


def _build_model_from_artifact(artifact: NeuralPolicyArtifact) -> Any:
    model = NeuralDraftNet.build(artifact.input_dim, artifact.hidden_dim)
    model.load_state_dict(artifact.model_state)
    model.eval()
    return model


def _policy_cycle(config: LeagueConfig) -> tuple[str, ...]:
    return tuple(config.ai_policies or ("best_available", "scarcity", "balanced"))


def _policy_for_team(config: LeagueConfig, team_index: int) -> str:
    cycle = _policy_cycle(config)
    return cycle[team_index % len(cycle)]


def _target_counts(config: LeagueConfig) -> np.ndarray:
    return np.array(
        [config.roster_settings.get(position, 0) for position in POSITIONS],
        dtype=float,
    )


def _need_tier(state: FastDraftState, team_index: int, candidates: np.ndarray) -> np.ndarray:
    counts = state.roster_counts[team_index]
    targets = _target_counts(state.config)
    direct_need = (targets - counts).clip(min=0)
    positions = state.data.positions[candidates]
    tiers = np.zeros(len(candidates), dtype=float)
    direct = direct_need[positions] > 0
    tiers[direct] = 1.0

    flex_count = state.config.roster_settings.get("FLEX", 0)
    if flex_count > 0:
        base_required = sum(
            state.config.roster_settings.get(position, 0)
            for position in FLEX_ELIGIBLE_POSITIONS
        )
        drafted_flex = int(counts[FLEX_POSITION_IDXS].sum())
        if drafted_flex < base_required + flex_count:
            flex_mask = np.isin(positions, FLEX_POSITION_IDXS) & ~direct
            tiers[flex_mask] = 0.5
    return tiers


def _position_limit(config: LeagueConfig, position_index: int) -> int:
    position = POSITIONS[position_index]
    base = int(config.roster_settings.get(position, 0))
    if position in FLEX_ELIGIBLE_POSITIONS:
        base += int(config.roster_settings.get("FLEX", 0))
    extra = int(config.max_extra_per_position.get(position, 0))
    return max(base + extra, 0)


def _neural_legal_indices(state: FastDraftState, team_index: int) -> np.ndarray:
    legal = state.legal_indices(team_index)
    if len(legal) == 0:
        return legal
    counts = state.roster_counts[team_index]
    positions = state.data.positions[legal]
    allowed = np.array(
        [
            counts[position_index] < _position_limit(state.config, int(position_index))
            for position_index in positions
        ],
        dtype=bool,
    )
    filtered = legal[allowed]
    return filtered if len(filtered) > 0 else legal


def _position_ordered_candidates(
    state: FastDraftState,
    candidates: np.ndarray,
    position_index: int,
    limit: int,
) -> list[int]:
    local = candidates[state.data.positions[candidates] == position_index]
    if len(local) == 0:
        return []
    ordered = local[np.argsort(-state.data.season_points[local])]
    return [int(candidate) for candidate in ordered[:limit]]


def _flex_still_needed(state: FastDraftState, team_index: int) -> bool:
    counts = state.roster_counts[team_index]
    base_required = sum(
        state.config.roster_settings.get(position, 0)
        for position in FLEX_ELIGIBLE_POSITIONS
    )
    drafted_flex_eligible = int(counts[FLEX_POSITION_IDXS].sum())
    return drafted_flex_eligible < base_required + state.config.roster_settings.get("FLEX", 0)


def _include_specialist_candidates(state: FastDraftState, team_index: int) -> bool:
    counts = state.roster_counts[team_index]
    targets = _target_counts(state.config)
    direct_need = int((targets - counts).clip(min=0).sum())
    remaining_slots = max(int(state.config.total_roster_slots - state.roster_size[team_index]), 0)
    specialist_need = sum(
        max(int(state.config.roster_settings.get(position, 0)) - int(counts[POS_TO_IDX[position]]), 0)
        for position in ("K", "DEF")
    )
    return specialist_need > 0 and remaining_slots <= specialist_need + 1


def neural_lookahead_candidates(
    state: FastDraftState,
    team_index: int,
    candidates: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    per_position: int = 2,
) -> np.ndarray:
    if len(candidates) == 0:
        return candidates
    top_k = max(int(top_k), 1)
    selected: list[int] = []
    include_specialists = _include_specialist_candidates(state, team_index)
    scoring_candidates = candidates
    if not include_specialists:
        non_specialists = candidates[
            ~np.isin(state.data.positions[candidates], [POS_TO_IDX["K"], POS_TO_IDX["DEF"]])
        ]
        if len(non_specialists) > 0:
            scoring_candidates = non_specialists
            candidate_positions = {int(candidate): index for index, candidate in enumerate(candidates)}
            scores = np.array([scores[candidate_positions[int(candidate)]] for candidate in scoring_candidates])

    neural_ordered = scoring_candidates[np.argsort(-scores)]
    season_ordered = scoring_candidates[np.argsort(-state.data.season_points[scoring_candidates])]
    selected.extend(int(candidate) for candidate in neural_ordered[:top_k])
    selected.extend(int(candidate) for candidate in season_ordered[:top_k])

    counts = state.roster_counts[team_index]
    targets = _target_counts(state.config)
    for position in STARTER_POSITIONS:
        position_index = POS_TO_IDX[position]
        if targets[position_index] - counts[position_index] <= 0:
            continue
        if position in {"K", "DEF"} and not include_specialists:
            continue
        selected.extend(
            _position_ordered_candidates(
                state,
                candidates,
                position_index,
                per_position,
            )
        )

    if _flex_still_needed(state, team_index):
        flex_candidates = candidates[np.isin(state.data.positions[candidates], FLEX_POSITION_IDXS)]
        selected.extend(
            int(candidate)
            for candidate in flex_candidates[
                np.argsort(-state.data.season_points[flex_candidates])
            ][: max(per_position * 2, top_k)]
        )

    unique = list(dict.fromkeys(selected))
    return np.array(unique, dtype=int)


def _starter_pressure(state: FastDraftState, team_index: int) -> tuple[float, float]:
    counts = state.roster_counts[team_index]
    targets = _target_counts(state.config)
    starter_need = int((targets - counts).clip(min=0).sum())
    flex_need = state.config.roster_settings.get("FLEX", 0)
    starter_capacity = max(int(targets.sum()) + flex_need, 1)
    remaining_slots = max(
        int(state.config.total_roster_slots - state.roster_size[team_index]), 0
    )
    bench_capacity = max(state.config.roster_settings.get("BENCH", 0), 1)
    return (
        max(starter_need, 0) / starter_capacity,
        max(remaining_slots - starter_need, 0) / bench_capacity,
    )


def _roster_best_weekly(state: FastDraftState, team_index: int) -> np.ndarray:
    roster = state.rosters[team_index]
    if not roster:
        return np.zeros(len(state.data.weeks), dtype=float)
    return state.data.weekly_points[np.array(roster, dtype=int)].max(axis=0)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _safe_corr_many(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[1] < 2 or np.std(right) == 0:
        return np.zeros(left.shape[0], dtype=float)
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean()
    numerator = left_centered @ right_centered
    denominator = np.sqrt((left_centered * left_centered).sum(axis=1) * np.dot(right_centered, right_centered))
    output = np.zeros(left.shape[0], dtype=float)
    valid = denominator > 1e-12
    output[valid] = numerator[valid] / denominator[valid]
    return output


def _static_feature_blocks(
    data: FastDraftData,
    max_week: float,
    max_season: float,
) -> dict[str, np.ndarray]:
    key = (
        id(data),
        len(data.player_ids),
        len(data.weeks),
        round(float(max_week), 6),
        round(float(max_season), 6),
    )
    cached = _STATIC_FEATURE_CACHE.get(key)
    if cached is not None:
        return cached

    weekly = data.weekly_points / max_week
    top3 = np.sort(weekly, axis=1)[:, -min(3, weekly.shape[1]) :]
    top5 = np.sort(weekly, axis=1)[:, -min(5, weekly.shape[1]) :]
    late_width = min(4, weekly.shape[1])
    summaries = np.column_stack(
        [
            data.season_points / max_season,
            weekly.mean(axis=1),
            weekly.max(axis=1),
            weekly.std(axis=1),
            (data.weekly_points <= 0).mean(axis=1),
            top3.mean(axis=1),
            top5.mean(axis=1),
            weekly[:, -late_width:].sum(axis=1) / max(late_width, 1),
        ]
    )

    one_hot = np.zeros((len(data.player_ids), len(POSITIONS)), dtype=float)
    one_hot[np.arange(len(data.player_ids)), data.positions] = 1.0
    cached = {
        "weekly": weekly.astype("float32"),
        "summaries": summaries.astype("float32"),
        "one_hot": one_hot.astype("float32"),
    }
    _STATIC_FEATURE_CACHE[key] = cached
    return cached


def build_neural_candidate_features(
    state: FastDraftState,
    team_index: int,
    candidates: np.ndarray,
    max_weekly_point: float | None = None,
) -> np.ndarray:
    data = state.data
    if len(candidates) == 0:
        return np.zeros((0, len(neural_feature_names(len(data.weeks)))), dtype=float)

    max_week = max(
        float(max_weekly_point or 0.0),
        float(data.weekly_points.max()) if data.weekly_points.size else 0.0,
        1.0,
    )
    max_season = max(float(data.season_points.max()) if len(data.season_points) else 0.0, 1.0)
    static = _static_feature_blocks(data, max_week, max_season)
    weekly = static["weekly"][candidates]
    roster_best_raw = _roster_best_weekly(state, team_index)
    roster_best = roster_best_raw / max_week
    gains = np.maximum(weekly - roster_best.reshape(1, -1), 0.0)
    overlap = np.minimum(weekly, roster_best.reshape(1, -1))

    rows = [weekly]
    rows.append(static["summaries"][candidates])

    rows.append(static["one_hot"][candidates])

    need = _need_tier(state, team_index, candidates)
    starter_pressure, bench_pressure = _starter_pressure(state, team_index)
    roster_best_mean = float(roster_best.mean())
    roster_best_max = float(roster_best.max()) if len(roster_best) else 0.0
    corr = _safe_corr_many(weekly, roster_best)
    extra = np.column_stack(
        [
            data.position_rank_feature[candidates],
            need,
            data.flex_eligible[candidates].astype(float),
            np.full(len(candidates), state.current_round / max(state.config.total_roster_slots, 1)),
            np.full(len(candidates), team_index / max(state.config.num_teams - 1, 1)),
            np.full(len(candidates), starter_pressure),
            np.full(len(candidates), bench_pressure),
            np.full(len(candidates), roster_best_mean),
            np.full(len(candidates), roster_best_max),
            gains.mean(axis=1),
            gains.max(axis=1),
            overlap.mean(axis=1),
            (weekly > roster_best.reshape(1, -1)).mean(axis=1),
            corr,
        ]
    )
    rows.append(extra)
    return np.concatenate(rows, axis=1).astype("float32")


def _choose_neural_pure(
    artifact: NeuralPolicyArtifact,
    model: Any,
    state: FastDraftState,
    team_index: int,
    context: FastPolicyContext | None = None,
) -> int:
    candidates = _neural_legal_indices(state, team_index)
    if len(candidates) == 0:
        raise RuntimeError(f"team {team_index} has no legal picks")
    cache_key = (
        team_index,
        state.pick_index,
        hash(state.available.tobytes()),
        tuple(state.rosters[team_index]),
    )
    scores = context.score_cache.get(cache_key) if context is not None else None
    if scores is None:
        scores = _neural_scores_for_fast_state(artifact, model, state, team_index, candidates)
        if context is not None:
            context.score_cache[cache_key] = scores
    return int(candidates[int(np.argmax(scores))])


def _context_neural(
    policy_name: str,
    context: FastPolicyContext,
) -> tuple[NeuralPolicyArtifact, Any] | None:
    if context.neural_artifact is not None:
        if context.neural_model is None:
            context.neural_model = _build_model_from_artifact(context.neural_artifact)
        return context.neural_artifact, context.neural_model

    if not policy_name.lower().startswith("neural:"):
        return None
    path = policy_name.split(":", 1)[1]
    policy_path = Path(path)
    if not policy_path.exists():
        return None
    cache_key = str(policy_path)
    if cache_key not in context.neural_cache:
        artifact = NeuralPolicyArtifact.load(policy_path)
        context.neural_cache[cache_key] = (artifact, _build_model_from_artifact(artifact))
    return context.neural_cache[cache_key]


def _choose_policy_pick(
    state: FastDraftState,
    team_index: int,
    policy_name: str,
    context: FastPolicyContext,
) -> int:
    key = policy_name.strip().lower()
    if key.startswith("neural:"):
        neural = _context_neural(policy_name, context)
        if neural is not None:
            artifact, model = neural
            return _choose_neural_pure(artifact, model, state, team_index, context)
        return choose_fast_pick(state, team_index, PolicySpec(kind="best_available"))

    if key.startswith("trained:"):
        path = policy_name.split(":", 1)[1]
        if path not in context.trained_cache:
            from .trainable import DraftPolicyWeights

            policy_path = Path(path)
            if policy_path.exists():
                context.trained_cache[path] = np.array(
                    DraftPolicyWeights.load(policy_path).vector(),
                    dtype=float,
                )
            else:
                context.trained_cache[path] = np.array([], dtype=float)
        weights = context.trained_cache[path]
        if len(weights) > 0:
            return choose_fast_pick(
                state,
                team_index,
                PolicySpec(kind="weighted", weights=weights, name=policy_name),
            )
        return choose_fast_pick(state, team_index, PolicySpec(kind="best_available"))

    if key == "scarcity":
        return choose_fast_pick(state, team_index, PolicySpec(kind="scarcity", name=policy_name))
    if key == "balanced":
        return choose_fast_pick(state, team_index, PolicySpec(kind="balanced", name=policy_name))
    return choose_fast_pick(state, team_index, PolicySpec(kind="best_available", name=policy_name))


def _softmax_sample_candidates(
    candidates: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
    count: int,
    temperature: float,
) -> list[int]:
    if count <= 0 or len(candidates) == 0:
        return []
    count = min(int(count), len(candidates))
    temperature = max(float(temperature), 1e-6)
    scaled = (values.astype(float) - float(values.max())) / temperature
    weights = np.exp(np.clip(scaled, -60.0, 0.0))
    total = float(weights.sum())
    if total <= 0:
        indexes = rng.choice(len(candidates), size=count, replace=False)
    else:
        indexes = rng.choice(len(candidates), size=count, replace=False, p=weights / total)
    return [int(candidate) for candidate in candidates[indexes]]


def _exploratory_pick(
    state: FastDraftState,
    team_index: int,
    rng: np.random.Generator,
    temperature: float,
) -> int:
    legal = _neural_legal_indices(state, team_index)
    if len(legal) == 0:
        raise RuntimeError("no legal picks")
    need = _need_tier(state, team_index, legal)
    values = state.data.season_points_norm[legal] + need * 0.35
    return _softmax_sample_candidates(
        legal,
        values,
        rng,
        count=1,
        temperature=max(temperature, 0.05),
    )[0]


def _jitter_policy_cycle(
    policy_cycle: tuple[str, ...],
    rng: np.random.Generator,
    jitter: float,
    include_neural: bool,
) -> tuple[str, ...]:
    if jitter <= 0:
        return policy_cycle
    choices = ["best_available", "scarcity", "balanced"]
    if include_neural:
        choices.append("neural:override")
    output = []
    for policy in policy_cycle:
        output.append(str(rng.choice(choices)) if rng.random() < jitter else policy)
    return tuple(output)


def _choose_rollout_pick(
    state: FastDraftState,
    team_index: int,
    policy_name: str,
    context: FastPolicyContext,
    rng: np.random.Generator | None = None,
    behavior_epsilon: float = 0.0,
    opponent_temperature: float = 0.0,
) -> int:
    if rng is not None and rng.random() < behavior_epsilon:
        return _exploratory_pick(state, team_index, rng, opponent_temperature)
    return _choose_policy_pick(state, team_index, policy_name, context)


def _complete_fast_rollout(
    state: FastDraftState,
    target_team: int,
    policy_cycle: tuple[str, ...] | None = None,
    context: FastPolicyContext | None = None,
    override_slot: int | None = None,
    override_policy: str | None = None,
    rng: np.random.Generator | None = None,
    behavior_epsilon: float = 0.0,
    opponent_temperature: float = 0.0,
) -> float:
    cycle = policy_cycle or _policy_cycle(state.config)
    rollout_context = context or FastPolicyContext()
    while not state.is_complete:
        team_index = state.team_on_clock
        if team_index == override_slot and override_policy is not None:
            policy_name = "neural:override" if override_policy == "neural" else override_policy
        else:
            policy_name = cycle[team_index % len(cycle)]
        player_index = _choose_rollout_pick(
            state,
            team_index,
            policy_name,
            rollout_context,
            rng=rng,
            behavior_epsilon=behavior_epsilon,
            opponent_temperature=opponent_temperature,
        )
        state.draft(team_index, player_index)
    from .training import _fast_weekly_team_scores, _head_to_head_fast

    weekly_scores = _fast_weekly_team_scores(state.rosters, state.data, state.config)
    win_pct, points_for = _head_to_head_fast(weekly_scores, state.data.weeks, state.config)
    from .training import FastDraftResult

    result = FastDraftResult(
        rosters=[list(roster) for roster in state.rosters],
        weekly_team_scores=weekly_scores,
        roto_totals=weekly_scores.sum(axis=1),
        win_pct=win_pct,
        points_for=points_for,
    )
    return reward_for_team(result, state.config, target_team)


def draft_state_to_fast_state(state: DraftState) -> FastDraftState:
    weekly_scores = state.weekly_scores
    if weekly_scores is None:
        weeks = [1]
        weekly_scores = state.players[["player_id"]].copy()
        weekly_scores["week"] = 1
        weekly_scores["points_scored"] = state.players["season_total_pts"].astype(float)
    data = build_fast_draft_data(
        ScoredData(players=state.players, weekly_scores=weekly_scores),
        state.config,
    )
    fast_state = FastDraftState.create(data, state.config)
    player_to_index = {str(player_id): index for index, player_id in enumerate(data.player_ids)}
    fast_state.available[:] = True
    fast_state.rosters = [[] for _ in range(state.config.num_teams)]
    fast_state.roster_counts[:] = 0
    fast_state.roster_size[:] = 0
    for team_index in range(state.config.num_teams):
        for player_id in state.roster(team_index):
            player_index = player_to_index[str(player_id)]
            fast_state.available[player_index] = False
            fast_state.rosters[team_index].append(player_index)
            fast_state.roster_size[team_index] += 1
            fast_state.roster_counts[team_index, data.positions[player_index]] += 1
    fast_state.pick_index = state.pick_index
    return fast_state


class NeuralDraftPolicy(DraftPolicy):
    def __init__(
        self,
        artifact: NeuralPolicyArtifact,
        path: str | Path | None = None,
        top_k: int | None = None,
        budget_seconds: float = 0.8,
    ):
        torch, _ = _torch()
        self.artifact = artifact
        self.name = f"neural:{path}" if path else "neural"
        self.top_k = int(top_k if top_k is not None else (artifact.training_config or {}).get("top_k", 5))
        self.budget_seconds = budget_seconds
        self.model = _build_model_from_artifact(artifact)
        self.torch = torch

    def choose_pick(self, state: DraftState, team_index: int) -> str:
        fast_state = draft_state_to_fast_state(state)
        candidates = _neural_legal_indices(fast_state, team_index)
        if len(candidates) == 0:
            raise RuntimeError(f"team {team_index} has no legal picks")
        scores = self._score_candidates(fast_state, team_index, candidates)
        ordered = candidates[np.argsort(-scores)]
        best = int(ordered[0])
        if self.top_k <= 1 or self.budget_seconds <= 0:
            return str(fast_state.data.player_ids[best])

        rollout_candidates = neural_lookahead_candidates(
            fast_state,
            team_index,
            candidates,
            scores,
            self.top_k,
        )
        deadline = time.perf_counter() + self.budget_seconds
        best_reward = -float("inf")
        context = FastPolicyContext(
            neural_artifact=self.artifact,
            neural_model=self.model,
        )
        policy_cycle = _policy_cycle(fast_state.config)
        for candidate in rollout_candidates:
            if time.perf_counter() >= deadline:
                break
            rollout = fast_state.clone()
            rollout.draft(team_index, int(candidate))
            reward = _complete_fast_rollout(
                rollout,
                team_index,
                policy_cycle=policy_cycle,
                context=context,
                override_slot=team_index,
                override_policy="neural",
            )
            if reward > best_reward:
                best_reward = reward
                best = int(candidate)
        return str(fast_state.data.player_ids[best])

    def _score_candidates(
        self, state: FastDraftState, team_index: int, candidates: np.ndarray
    ) -> np.ndarray:
        features = build_neural_candidate_features(
            state,
            team_index,
            candidates,
            max_weekly_point=self.artifact.max_weekly_point,
        )
        if features.shape[1] != self.artifact.input_dim:
            features = _resize_features(features, self.artifact.input_dim)
        with self.torch.no_grad():
            tensor = self.torch.tensor(features, dtype=self.torch.float32)
            return self.model(tensor).squeeze(-1).cpu().numpy()


class MissingNeuralPolicyFallback(DraftPolicy):
    def __init__(self, path: str | Path):
        from .ai import BestAvailablePolicy

        self.path = str(path)
        self.name = f"neural_missing:{path}"
        self.fallback = BestAvailablePolicy()

    def choose_pick(self, state: DraftState, team_index: int) -> str:
        return self.fallback.choose_pick(state, team_index)


def _resize_features(features: np.ndarray, input_dim: int) -> np.ndarray:
    if features.shape[1] == input_dim:
        return features
    if features.shape[1] > input_dim:
        return features[:, :input_dim]
    pad = np.zeros((features.shape[0], input_dim - features.shape[1]), dtype=features.dtype)
    return np.concatenate([features, pad], axis=1)


def load_neural_policy(path: str | Path) -> DraftPolicy:
    policy_path = Path(path)
    if not policy_path.exists():
        return MissingNeuralPolicyFallback(policy_path)
    return NeuralDraftPolicy(NeuralPolicyArtifact.load(policy_path), path=policy_path)


@dataclass(frozen=True)
class NeuralTrainerConfig:
    samples: int = 5000
    epochs: int = 20
    seed: int = 11
    hidden_dim: int = 64
    batch_size: int = 128
    learning_rate: float = 0.001
    max_candidates_per_state: int = 6
    eval_slots: str = "all"
    top_k: int = 5
    rollout_budget: int | None = None
    behavior_epsilon: float = 0.12
    opponent_temperature: float = 0.12
    candidate_noise_std: float = 0.0
    rollouts_per_candidate: int = 1
    policy_mix_jitter: float = 0.0
    target_season: int | None = None


@dataclass(frozen=True)
class NeuralTrainingResult:
    artifact: NeuralPolicyArtifact
    history: list[dict[str, float]]
    training_rows: int
    baseline_reward: float
    neural_reward: float
    timings: dict[str, float] = field(default_factory=dict)


def _candidate_subset(
    state: FastDraftState,
    team_index: int,
    rng: np.random.Generator,
    max_candidates: int,
    context: FastPolicyContext | None = None,
    trainer_config: NeuralTrainerConfig | None = None,
) -> np.ndarray:
    legal = _neural_legal_indices(state, team_index)
    if len(legal) <= max_candidates:
        return legal
    trainer = trainer_config or NeuralTrainerConfig(max_candidates_per_state=max_candidates)
    selected: list[int] = []
    season_values = state.data.season_points_norm[legal]

    selected.extend(
        int(candidate)
        for candidate in legal[np.argsort(-state.data.season_points[legal])][
            : max(1, max_candidates // 4)
        ]
    )

    neural_scores = season_values.copy()
    if context is not None:
        neural = _context_neural("neural:override", context)
        if neural is not None:
            artifact, model = neural
            neural_scores = _neural_scores_for_fast_state(
                artifact,
                model,
                state,
                team_index,
                legal,
            )
    if trainer.candidate_noise_std > 0:
        neural_scores = neural_scores + rng.normal(0.0, trainer.candidate_noise_std, len(legal))
    selected.extend(
        int(candidate)
        for candidate in neural_lookahead_candidates(
            state,
            team_index,
            legal,
            neural_scores,
            top_k=max(1, max_candidates // 2),
        )
    )

    selected.extend(
        _softmax_sample_candidates(
            legal,
            season_values + _need_tier(state, team_index, legal) * 0.35,
            rng,
            count=max_candidates,
            temperature=max(trainer.opponent_temperature, 0.05),
        )
    )

    unique = list(dict.fromkeys(selected))
    if len(unique) < max_candidates:
        rest = np.array([candidate for candidate in legal if int(candidate) not in set(unique)], dtype=int)
        if len(rest) > 0:
            unique.extend(int(candidate) for candidate in rng.choice(rest, size=min(max_candidates - len(unique), len(rest)), replace=False))
    return np.array(unique[:max_candidates], dtype=int)


def _advance_behavior_pick(
    state: FastDraftState,
    rng: np.random.Generator,
    policy_cycle: tuple[str, ...],
    context: FastPolicyContext,
    trainer_config: NeuralTrainerConfig | None = None,
) -> None:
    trainer = trainer_config or NeuralTrainerConfig()
    team_index = state.team_on_clock
    legal = _neural_legal_indices(state, team_index)
    if len(legal) == 0:
        raise RuntimeError("no legal picks")
    pick = _choose_rollout_pick(
        state,
        team_index,
        policy_cycle[team_index % len(policy_cycle)],
        context,
        rng=rng,
        behavior_epsilon=trainer.behavior_epsilon,
        opponent_temperature=trainer.opponent_temperature,
    )
    state.draft(team_index, pick)


def generate_neural_training_samples(
    data: FastDraftData,
    config: LeagueConfig,
    trainer_config: NeuralTrainerConfig,
    champion_artifact: NeuralPolicyArtifact | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(trainer_config.seed)
    max_week = max(float(data.weekly_points.max()) if data.weekly_points.size else 0.0, 1.0)
    features: list[np.ndarray] = []
    targets: list[float] = []
    state = FastDraftState.create(data, config)
    base_policy_cycle = _policy_cycle(config)
    policy_cycle = _jitter_policy_cycle(
        base_policy_cycle,
        rng,
        trainer_config.policy_mix_jitter,
        include_neural=champion_artifact is not None,
    )
    context = FastPolicyContext(
        neural_artifact=champion_artifact,
        neural_model=_build_model_from_artifact(champion_artifact)
        if champion_artifact is not None
        else None,
    )

    while len(targets) < trainer_config.samples:
        if state.is_complete:
            state = FastDraftState.create(data, config)
            policy_cycle = _jitter_policy_cycle(
                base_policy_cycle,
                rng,
                trainer_config.policy_mix_jitter,
                include_neural=champion_artifact is not None,
            )
            continue
        team_index = state.team_on_clock
        candidates = _candidate_subset(
            state,
            team_index,
            rng,
            trainer_config.max_candidates_per_state,
            context=context,
            trainer_config=trainer_config,
        )
        if trainer_config.rollout_budget is not None and len(candidates) > trainer_config.rollout_budget:
            candidates = candidates[: max(int(trainer_config.rollout_budget), 1)]
        matrix = build_neural_candidate_features(
            state,
            team_index,
            candidates,
            max_weekly_point=max_week,
        )
        state_features: list[np.ndarray] = []
        state_rewards: list[float] = []
        for row_index, candidate in enumerate(candidates):
            rollout_rewards = []
            for _ in range(max(int(trainer_config.rollouts_per_candidate), 1)):
                rollout = state.clone()
                rollout.draft(team_index, int(candidate))
                reward = _complete_fast_rollout(
                    rollout,
                    team_index,
                    policy_cycle=policy_cycle,
                    context=context,
                    rng=rng,
                    behavior_epsilon=trainer_config.behavior_epsilon,
                    opponent_temperature=trainer_config.opponent_temperature,
                )
                rollout_rewards.append(reward)
                _notify_progress(progress_callback, "candidate rollouts", 1, None)
            reward = float(np.mean(rollout_rewards))
            state_features.append(matrix[row_index])
            state_rewards.append(reward)

        rewards = np.array(state_rewards, dtype="float32")
        spread = float(rewards.std())
        if spread > 1e-6:
            state_targets = (rewards - float(rewards.mean())) / spread
        else:
            state_targets = np.zeros_like(rewards)

        for row, target in zip(state_features, state_targets):
            if len(targets) >= trainer_config.samples:
                break
            features.append(row)
            targets.append(float(target))
            _notify_progress(progress_callback, "label samples", 1, trainer_config.samples)
        _advance_behavior_pick(state, rng, policy_cycle, context, trainer_config)

    return np.vstack(features).astype("float32"), np.array(targets, dtype="float32")


def train_neural_policy(
    scored: ScoredInput,
    config: LeagueConfig,
    trainer_config: NeuralTrainerConfig,
    base_artifact: NeuralPolicyArtifact | None = None,
    progress_callback: ProgressCallback | None = None,
) -> NeuralTrainingResult:
    timings: dict[str, float] = {}
    torch, nn = _torch()
    started = time.perf_counter()
    scored_by_season = _normalize_scored_input(scored)
    data_by_season = {
        season: _get_fast_data(season_scored, config)
        for season, season_scored in sorted(scored_by_season.items())
    }
    primary_season, primary_scored_value = _primary_scored(
        scored_by_season,
        trainer_config.target_season,
    )
    primary_data = data_by_season[primary_season]
    timings["prepare_data_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    sample_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    max_input_dim = max(
        len(neural_feature_names(len(data.weeks)))
        for data in data_by_season.values()
    )
    samples_per_season = max(4, trainer_config.samples // max(len(data_by_season), 1))
    for offset, (season, data) in enumerate(data_by_season.items()):
        season_samples = (
            trainer_config.samples - sum(len(part) for part in target_parts)
            if offset == len(data_by_season) - 1
            else samples_per_season
        )
        local_config = NeuralTrainerConfig(
            samples=max(season_samples, 1),
            epochs=trainer_config.epochs,
            seed=trainer_config.seed + offset * 1009,
            hidden_dim=trainer_config.hidden_dim,
            batch_size=trainer_config.batch_size,
            learning_rate=trainer_config.learning_rate,
            max_candidates_per_state=trainer_config.max_candidates_per_state,
            eval_slots=trainer_config.eval_slots,
            top_k=trainer_config.top_k,
            rollout_budget=trainer_config.rollout_budget,
            behavior_epsilon=trainer_config.behavior_epsilon,
            opponent_temperature=trainer_config.opponent_temperature,
            candidate_noise_std=trainer_config.candidate_noise_std,
            rollouts_per_candidate=trainer_config.rollouts_per_candidate,
            policy_mix_jitter=trainer_config.policy_mix_jitter,
            target_season=trainer_config.target_season,
        )
        x_part, y_part = generate_neural_training_samples(
            data,
            config,
            local_config,
            base_artifact,
            progress_callback=progress_callback,
        )
        sample_parts.append(_resize_features(x_part, max_input_dim))
        target_parts.append(y_part)
    x = np.vstack(sample_parts).astype("float32")
    y = np.concatenate(target_parts).astype("float32")
    timings["rollout_labeling_seconds"] = time.perf_counter() - started
    target_mean = float(y.mean())
    target_std = float(y.std() if y.std() > 1e-6 else 1.0)
    y_norm = (y - target_mean) / target_std

    torch.manual_seed(trainer_config.seed)
    hidden_dim = base_artifact.hidden_dim if base_artifact is not None else trainer_config.hidden_dim
    model = NeuralDraftNet.build(x.shape[1], hidden_dim)
    if base_artifact is not None and base_artifact.input_dim == x.shape[1]:
        model.load_state_dict(base_artifact.model_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config.learning_rate)
    loss_fn = nn.MSELoss()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm.reshape(-1, 1), dtype=torch.float32)
    rng = np.random.default_rng(trainer_config.seed)
    history: list[dict[str, float]] = []

    started = time.perf_counter()
    for epoch in range(1, trainer_config.epochs + 1):
        indexes = rng.permutation(len(x))
        total_loss = 0.0
        batches = 0
        for start in range(0, len(indexes), trainer_config.batch_size):
            batch = indexes[start : start + trainer_config.batch_size]
            optimizer.zero_grad()
            predictions = model(x_tensor[batch])
            loss = loss_fn(predictions, y_tensor[batch])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        history.append({"epoch": float(epoch), "loss": total_loss / max(batches, 1)})
        _notify_progress(progress_callback, "training epochs", 1, trainer_config.epochs)
    timings["torch_training_seconds"] = time.perf_counter() - started

    artifact = NeuralPolicyArtifact(
        model_state=model.state_dict(),
        input_dim=x.shape[1],
        week_count=max(len(data.weeks) for data in data_by_season.values()),
        hidden_dim=hidden_dim,
        max_weekly_point=max(
            max(float(data.weekly_points.max()) if data.weekly_points.size else 0.0, 1.0)
            for data in data_by_season.values()
        ),
        target_mean=target_mean,
        target_std=target_std,
        metadata={
            "policy_type": "weekly_vector_neural",
            "feature_names": neural_feature_names(max(len(data.weeks) for data in data_by_season.values())),
            "seasons": list(sorted(scored_by_season)),
            "target_season": primary_season,
        },
        training_config={
            "samples": trainer_config.samples,
            "epochs": trainer_config.epochs,
            "seed": trainer_config.seed,
            "hidden_dim": hidden_dim,
            "batch_size": trainer_config.batch_size,
            "learning_rate": trainer_config.learning_rate,
            "max_candidates_per_state": trainer_config.max_candidates_per_state,
            "eval_slots": trainer_config.eval_slots,
            "top_k": trainer_config.top_k,
            "rollout_budget": trainer_config.rollout_budget,
            "behavior_epsilon": trainer_config.behavior_epsilon,
            "opponent_temperature": trainer_config.opponent_temperature,
            "candidate_noise_std": trainer_config.candidate_noise_std,
            "rollouts_per_candidate": trainer_config.rollouts_per_candidate,
            "policy_mix_jitter": trainer_config.policy_mix_jitter,
            "seasons": list(sorted(scored_by_season)),
            "target_season": primary_season,
            "policy_cycle": list(_policy_cycle(config)),
            "base_policy": "provided" if base_artifact is not None else "none",
            "max_extra_per_position": dict(config.max_extra_per_position),
        },
    )
    started = time.perf_counter()
    baseline_reward = float(np.mean([
        _baseline_reward(
            data,
            config,
            trainer_config.eval_slots,
            neural_artifact=base_artifact,
        )
        for data in data_by_season.values()
    ]))
    timings["baseline_validation_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    neural_summary = evaluate_neural_artifact(
        primary_scored_value,
        config,
        artifact,
        drafts=max(1, len(_parse_eval_slots(trainer_config.eval_slots, config.num_teams))),
        eval_slots=trainer_config.eval_slots,
        progress_callback=progress_callback,
    )
    timings["validation_seconds"] = time.perf_counter() - started
    neural_reward = float(neural_summary["average_reward"])
    artifact = NeuralPolicyArtifact(
        model_state=artifact.model_state,
        input_dim=artifact.input_dim,
        week_count=artifact.week_count,
        hidden_dim=artifact.hidden_dim,
        max_weekly_point=artifact.max_weekly_point,
        target_mean=artifact.target_mean,
        target_std=artifact.target_std,
        metadata=artifact.metadata,
        training_config=artifact.training_config,
        validation_summary={
            "baseline_reward": baseline_reward,
            "neural_reward": neural_reward,
            "improvement": neural_reward - baseline_reward,
            "objective": config.draft_objective,
            "by_slot": neural_summary["by_slot"],
            "policy_cycle": list(_policy_cycle(config)),
            "timing_summary": timings,
        },
    )
    return NeuralTrainingResult(
        artifact=artifact,
        history=history,
        training_rows=len(x),
        baseline_reward=baseline_reward,
        neural_reward=neural_reward,
        timings=timings,
    )


def _parse_eval_slots(value: str, num_teams: int) -> list[int]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list(range(num_teams))
    if normalized == "spread":
        slots = {0, num_teams // 2, num_teams - 1}
        return sorted(slot for slot in slots if 0 <= slot < num_teams)
    slots = []
    for part in normalized.split(","):
        if part.strip():
            slot = int(part.strip())
            if 0 <= slot < num_teams:
                slots.append(slot)
    return slots or [0]


def _variant_config(config: LeagueConfig, num_teams: int) -> LeagueConfig:
    payload = config.model_dump()
    payload["num_teams"] = int(num_teams)
    payload["team_names"] = [f"Team {index + 1}" for index in range(int(num_teams))]
    return LeagueConfig.model_validate(payload)


def _variant_configs(config: LeagueConfig, league_sizes: tuple[int, ...]) -> list[LeagueConfig]:
    return [_variant_config(config, size) for size in league_sizes]


def _baseline_reward(
    data: FastDraftData,
    config: LeagueConfig,
    eval_slots: str,
    neural_artifact: NeuralPolicyArtifact | None = None,
) -> float:
    rewards = []
    context = FastPolicyContext(
        neural_artifact=neural_artifact,
        neural_model=_build_model_from_artifact(neural_artifact)
        if neural_artifact is not None
        else None,
    )
    for slot in _parse_eval_slots(eval_slots, config.num_teams):
        result = _simulate_policy_override_fast_draft(
            data,
            config,
            override_slot=None,
            override_policy=None,
            context=context,
        )
        rewards.append(reward_for_team(result, config, slot))
    return float(np.mean(rewards))


def _neural_scores_for_fast_state(
    artifact: NeuralPolicyArtifact,
    model: Any,
    state: FastDraftState,
    team_index: int,
    candidates: np.ndarray,
) -> np.ndarray:
    torch, _ = _torch()
    features = build_neural_candidate_features(
        state,
        team_index,
        candidates,
        max_weekly_point=artifact.max_weekly_point,
    )
    features = _resize_features(features, artifact.input_dim)
    with torch.no_grad():
        return model(torch.tensor(features, dtype=torch.float32)).squeeze(-1).cpu().numpy()


def _choose_neural_fast(
    artifact: NeuralPolicyArtifact,
    model: Any,
    state: FastDraftState,
    team_index: int,
    policy_cycle: tuple[str, ...] | None = None,
    context: FastPolicyContext | None = None,
    rollout_budget: int | None = None,
    candidate_pool_size: int | None = None,
) -> int:
    candidates = _neural_legal_indices(state, team_index)
    scores = _neural_scores_for_fast_state(artifact, model, state, team_index, candidates)
    ordered = candidates[np.argsort(-scores)]
    top_k = max(int(candidate_pool_size or (artifact.training_config or {}).get("top_k", 5)), 1)
    if top_k <= 1:
        return int(ordered[0])

    rollout_candidates = neural_lookahead_candidates(
        state,
        team_index,
        candidates,
        scores,
        top_k,
    )
    if rollout_budget is not None:
        rollout_candidates = rollout_candidates[: max(int(rollout_budget), 1)]
    best = int(ordered[0])
    best_reward = -float("inf")
    for candidate in rollout_candidates:
        rollout = state.clone()
        rollout.draft(team_index, int(candidate))
        reward = _complete_fast_rollout(
            rollout,
            team_index,
            policy_cycle=policy_cycle or _policy_cycle(state.config),
            context=context
            or FastPolicyContext(neural_artifact=artifact, neural_model=model),
            override_slot=team_index,
            override_policy="neural",
        )
        if reward > best_reward:
            best_reward = reward
            best = int(candidate)
    return best


def _simulate_policy_override_fast_draft(
    data: FastDraftData,
    config: LeagueConfig,
    override_slot: int | None,
    override_policy: str | None,
    context: FastPolicyContext | None = None,
    lookahead: bool = False,
    rollout_budget: int | None = None,
    candidate_pool_size: int | None = None,
) -> Any:
    policy_cycle = _policy_cycle(config)
    fast_context = context or FastPolicyContext()
    state = FastDraftState.create(data, config)
    while not state.is_complete:
        team_index = state.team_on_clock
        if team_index == override_slot and override_policy is not None:
            if override_policy == "neural":
                neural = _context_neural("neural:override", fast_context)
                if neural is None:
                    pick = choose_fast_pick(
                        state,
                        team_index,
                        PolicySpec(kind="best_available", name="best_available"),
                    )
                else:
                    artifact, model = neural
                    if lookahead:
                        pick = _choose_neural_fast(
                            artifact,
                            model,
                            state,
                            team_index,
                            policy_cycle=policy_cycle,
                            context=fast_context,
                            rollout_budget=rollout_budget,
                            candidate_pool_size=candidate_pool_size,
                        )
                    else:
                        pick = _choose_neural_pure(
                            artifact,
                            model,
                            state,
                            team_index,
                            fast_context,
                        )
            else:
                pick = _choose_policy_pick(state, team_index, override_policy, fast_context)
        else:
            pick = _choose_policy_pick(
                state,
                team_index,
                policy_cycle[team_index % len(policy_cycle)],
                fast_context,
            )
        state.draft(team_index, pick)

    from .training import FastDraftResult, _fast_weekly_team_scores, _head_to_head_fast

    weekly_scores = _fast_weekly_team_scores(state.rosters, data, config)
    win_pct, points_for = _head_to_head_fast(weekly_scores, data.weeks, config)
    return FastDraftResult(
        rosters=[list(roster) for roster in state.rosters],
        weekly_team_scores=weekly_scores,
        roto_totals=weekly_scores.sum(axis=1),
        win_pct=win_pct,
        points_for=points_for,
    )


def _simulate_neural_fast_draft(
    data: FastDraftData,
    config: LeagueConfig,
    artifact: NeuralPolicyArtifact,
    neural_slot: int,
    lookahead: bool = False,
    rollout_budget: int | None = None,
    candidate_pool_size: int | None = None,
) -> Any:
    context = FastPolicyContext(
        neural_artifact=artifact,
        neural_model=_build_model_from_artifact(artifact),
    )
    return _simulate_policy_override_fast_draft(
        data,
        config,
        override_slot=neural_slot,
        override_policy="neural",
        context=context,
        lookahead=lookahead,
        rollout_budget=rollout_budget,
        candidate_pool_size=candidate_pool_size,
    )


def evaluate_neural_artifact(
    scored: ScoredData,
    config: LeagueConfig,
    artifact: NeuralPolicyArtifact,
    drafts: int = 100,
    eval_slots: str = "all",
    lookahead: bool = False,
    rollout_budget: int | None = None,
    candidate_pool_size: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    data = _get_fast_data(scored, config)
    slots = _parse_eval_slots(eval_slots, config.num_teams)
    rewards = []
    by_slot: dict[int, list[float]] = {slot: [] for slot in slots}
    total = max(drafts, 1)
    for index in range(max(drafts, 1)):
        slot = slots[index % len(slots)]
        result = _simulate_neural_fast_draft(
            data,
            config,
            artifact,
            slot,
            lookahead=lookahead,
            rollout_budget=rollout_budget,
            candidate_pool_size=candidate_pool_size,
        )
        reward = reward_for_team(result, config, slot)
        rewards.append(reward)
        by_slot[slot].append(reward)
        _notify_progress(progress_callback, "validation drafts", 1, total)
    return {
        "drafts": max(drafts, 1),
        "average_reward": float(np.mean(rewards)),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "by_slot": {
            str(slot): float(np.mean(slot_rewards))
            for slot, slot_rewards in by_slot.items()
            if slot_rewards
        },
        "timing_summary": {
            "evaluation_seconds": time.perf_counter() - started,
            "lookahead": bool(lookahead),
        },
    }


def evaluate_neural_variants(
    scored: ScoredData,
    config: LeagueConfig,
    artifact: NeuralPolicyArtifact,
    league_sizes: tuple[int, ...],
    drafts: int = 100,
    eval_slots: str = "all",
    lookahead: bool = False,
    rollout_budget: int | None = None,
    candidate_pool_size: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    variants: dict[str, Any] = {}
    rewards: list[float] = []
    for size in league_sizes:
        variant = _variant_config(config, size)
        summary = evaluate_neural_artifact(
            scored,
            variant,
            artifact,
            drafts=drafts,
            eval_slots=eval_slots,
            lookahead=lookahead,
            rollout_budget=rollout_budget,
            candidate_pool_size=candidate_pool_size,
            progress_callback=progress_callback,
        )
        variants[str(size)] = summary
        rewards.append(float(summary["average_reward"]))
    return {
        "league_sizes": list(league_sizes),
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "variants": variants,
        "timing_summary": {
            "evaluation_seconds": time.perf_counter() - started,
            "lookahead": bool(lookahead),
        },
    }


@dataclass(frozen=True)
class NeuralImprovementConfig:
    generations: int = 5
    samples_per_generation: int = 6000
    epochs_per_generation: int = 12
    seed: int = 101
    batch_size: int = 256
    learning_rate: float = 0.001
    max_candidates_per_state: int = 12
    eval_slots: str = "all"
    top_k: int = 8
    robust: bool = False
    league_sizes: tuple[int, ...] = ()
    rollout_budget: int | None = None
    validation_drafts: int | None = None
    behavior_epsilon: float = 0.12
    opponent_temperature: float = 0.12
    candidate_noise_std: float = 0.0
    rollouts_per_candidate: int = 1
    policy_mix_jitter: float = 0.0
    target_season: int | None = None
    accept_4team_threshold: float | None = None
    threshold_league_size: int = 4
    robust_regression_tolerance: float = 0.02


@dataclass(frozen=True)
class NeuralImprovementResult:
    artifact: NeuralPolicyArtifact
    history: list[dict[str, Any]]
    initial_reward: float
    best_reward: float
    timings: dict[str, float] = field(default_factory=dict)


def _artifact_with_updates(
    artifact: NeuralPolicyArtifact,
    metadata: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    validation_summary: dict[str, Any] | None = None,
) -> NeuralPolicyArtifact:
    return NeuralPolicyArtifact(
        model_state=artifact.model_state,
        input_dim=artifact.input_dim,
        week_count=artifact.week_count,
        hidden_dim=artifact.hidden_dim,
        max_weekly_point=artifact.max_weekly_point,
        target_mean=artifact.target_mean,
        target_std=artifact.target_std,
        version=artifact.version,
        metadata=metadata if metadata is not None else artifact.metadata,
        training_config=training_config
        if training_config is not None
        else artifact.training_config,
        validation_summary=validation_summary
        if validation_summary is not None
        else artifact.validation_summary,
    )


def improve_neural_policy(
    scored: ScoredInput,
    config: LeagueConfig,
    base_artifact: NeuralPolicyArtifact,
    improvement_config: NeuralImprovementConfig,
    progress_callback: ProgressCallback | None = None,
) -> NeuralImprovementResult:
    run_started = time.perf_counter()
    timings: dict[str, float] = {}
    scored_by_season = _normalize_scored_input(scored)
    target_season = improvement_config.target_season or sorted(scored_by_season)[-1]
    league_sizes = improvement_config.league_sizes
    if improvement_config.robust and not league_sizes:
        league_sizes = (4, 8, 10, 12)
    training_configs = _variant_configs(config, league_sizes) if improvement_config.robust else [config]

    def evaluate_artifact(artifact: NeuralPolicyArtifact) -> dict[str, Any]:
        season_summaries: dict[str, Any] = {}
        rewards: list[float] = []
        if improvement_config.robust:
            default_drafts = max(
                1,
                sum(
                    len(_parse_eval_slots(improvement_config.eval_slots, c.num_teams))
                    for c in training_configs
                ),
            )
            for season, season_scored in scored_by_season.items():
                summary = evaluate_neural_variants(
                    season_scored,
                    config,
                    artifact,
                    league_sizes=league_sizes,
                    drafts=improvement_config.validation_drafts or default_drafts,
                    eval_slots=improvement_config.eval_slots,
                    lookahead=False,
                    rollout_budget=improvement_config.rollout_budget,
                    candidate_pool_size=improvement_config.top_k,
                    progress_callback=progress_callback,
                )
                season_summaries[str(season)] = summary
                rewards.append(float(summary["average_reward"]))
            output: dict[str, Any] = {
                "average_reward": float(np.mean(rewards)) if rewards else 0.0,
                "season_summaries": season_summaries,
                "variants": season_summaries.get(str(target_season), {}).get("variants", {}),
            }
        else:
            slots = _parse_eval_slots(improvement_config.eval_slots, config.num_teams)
            for season, season_scored in scored_by_season.items():
                summary = evaluate_neural_artifact(
                    season_scored,
                    config,
                    artifact,
                    drafts=improvement_config.validation_drafts or max(1, len(slots)),
                    eval_slots=improvement_config.eval_slots,
                    lookahead=False,
                    rollout_budget=improvement_config.rollout_budget,
                    candidate_pool_size=improvement_config.top_k,
                    progress_callback=progress_callback,
                )
                season_summaries[str(season)] = summary
                rewards.append(float(summary["average_reward"]))
            output = {
                "average_reward": float(np.mean(rewards)) if rewards else 0.0,
                "season_summaries": season_summaries,
                "by_slot": season_summaries.get(str(target_season), {}).get("by_slot", {}),
            }
        if improvement_config.accept_4team_threshold is not None and target_season in scored_by_season:
            target_summary = evaluate_neural_variants(
                scored_by_season[target_season],
                config,
                artifact,
                league_sizes=(improvement_config.threshold_league_size,),
                drafts=improvement_config.validation_drafts
                or max(1, improvement_config.threshold_league_size),
                eval_slots=improvement_config.eval_slots,
                lookahead=False,
                rollout_budget=improvement_config.rollout_budget,
                candidate_pool_size=improvement_config.top_k,
                progress_callback=progress_callback,
            )
            target_variant = target_summary["variants"].get(
                str(improvement_config.threshold_league_size),
                {},
            )
            output["target_threshold_reward"] = float(
                target_variant.get("average_reward", target_summary["average_reward"])
            )
            output["target_threshold_summary"] = target_summary
        return output

    def accept_candidate(
        candidate_summary: dict[str, Any],
        champion_summary: dict[str, Any],
    ) -> bool:
        candidate_reward = float(candidate_summary["average_reward"])
        champion_reward = float(champion_summary["average_reward"])
        threshold = improvement_config.accept_4team_threshold
        if threshold is None:
            return candidate_reward > champion_reward
        candidate_target = float(candidate_summary.get("target_threshold_reward", 0.0))
        champion_target = float(champion_summary.get("target_threshold_reward", 0.0))
        robust_floor = champion_reward * (1.0 - improvement_config.robust_regression_tolerance)
        if champion_target < threshold and candidate_target < threshold:
            return candidate_target > champion_target and candidate_reward >= robust_floor
        return candidate_target >= threshold and candidate_reward > champion_reward

    started = time.perf_counter()
    initial_summary = evaluate_artifact(base_artifact)
    timings["initial_validation_seconds"] = time.perf_counter() - started
    champion = base_artifact
    champion_summary = initial_summary
    best_reward = float(initial_summary["average_reward"])
    initial_reward = best_reward
    history: list[dict[str, Any]] = [
        {
            "generation": 0,
            "reward": best_reward,
            "accepted": True,
            "by_slot": initial_summary.get("by_slot", {}),
            "variants": initial_summary.get("variants", {}),
            "season_summaries": initial_summary.get("season_summaries", {}),
            "target_threshold_reward": initial_summary.get("target_threshold_reward"),
        }
    ]

    for generation in range(1, improvement_config.generations + 1):
        generation_started = time.perf_counter()
        candidate_artifact = champion
        training_rows = 0
        generation_training_seconds = 0.0
        samples_per_config = max(
            4,
            improvement_config.samples_per_generation // max(len(training_configs), 1),
        )
        for config_index, training_config_source in enumerate(training_configs):
            trainer_config = NeuralTrainerConfig(
                samples=samples_per_config,
                epochs=improvement_config.epochs_per_generation,
                seed=improvement_config.seed + generation + config_index,
                hidden_dim=candidate_artifact.hidden_dim,
                batch_size=improvement_config.batch_size,
                learning_rate=improvement_config.learning_rate,
                max_candidates_per_state=improvement_config.max_candidates_per_state,
                eval_slots=improvement_config.eval_slots,
                top_k=improvement_config.top_k,
                rollout_budget=improvement_config.rollout_budget,
                behavior_epsilon=improvement_config.behavior_epsilon,
                opponent_temperature=improvement_config.opponent_temperature,
                candidate_noise_std=improvement_config.candidate_noise_std,
                rollouts_per_candidate=improvement_config.rollouts_per_candidate,
                policy_mix_jitter=improvement_config.policy_mix_jitter,
                target_season=target_season,
            )
            started = time.perf_counter()
            candidate = train_neural_policy(
                scored,
                training_config_source,
                trainer_config,
                base_artifact=candidate_artifact,
                progress_callback=progress_callback,
            )
            generation_training_seconds += time.perf_counter() - started
            candidate_artifact = candidate.artifact
            training_rows += candidate.training_rows
        started = time.perf_counter()
        candidate_summary = evaluate_artifact(candidate_artifact)
        generation_validation_seconds = time.perf_counter() - started
        candidate_reward = float(candidate_summary["average_reward"])
        accepted = accept_candidate(candidate_summary, champion_summary)
        if accepted:
            champion = candidate_artifact
            champion_summary = candidate_summary
            best_reward = candidate_reward
        history.append(
            {
                "generation": generation,
                "reward": candidate_reward,
                "accepted": accepted,
                "best_reward": best_reward,
                "training_rows": training_rows,
                "by_slot": candidate_summary.get("by_slot", {}),
                "variants": candidate_summary.get("variants", {}),
                "season_summaries": candidate_summary.get("season_summaries", {}),
                "target_threshold_reward": candidate_summary.get("target_threshold_reward"),
                "training_seconds": generation_training_seconds,
                "validation_seconds": generation_validation_seconds,
                "elapsed_seconds": time.perf_counter() - generation_started,
            }
        )
        _notify_progress(progress_callback, "generations", 1, improvement_config.generations)

    started = time.perf_counter()
    final_summary = evaluate_artifact(champion)
    timings["final_validation_seconds"] = time.perf_counter() - started
    best_reward = float(final_summary["average_reward"])
    timings["total_seconds"] = time.perf_counter() - run_started
    metadata = {
        **(champion.metadata or {}),
        "policy_type": "weekly_vector_neural",
        "improvement_mode": "robust_self_play" if improvement_config.robust else "exact_league_self_play",
        "policy_cycle": list(_policy_cycle(config)),
        "league_sizes": list(league_sizes),
        "seasons": list(sorted(scored_by_season)),
        "target_season": target_season,
        "rollout_budget": improvement_config.rollout_budget,
        "validation_drafts": improvement_config.validation_drafts,
        "accept_4team_threshold": improvement_config.accept_4team_threshold,
    }
    training_config = {
        **(champion.training_config or {}),
        "improve_generations": improvement_config.generations,
        "samples_per_generation": improvement_config.samples_per_generation,
        "epochs_per_generation": improvement_config.epochs_per_generation,
        "improve_seed": improvement_config.seed,
        "policy_cycle": list(_policy_cycle(config)),
        "robust": improvement_config.robust,
        "league_sizes": list(league_sizes),
        "seasons": list(sorted(scored_by_season)),
        "target_season": target_season,
        "rollout_budget": improvement_config.rollout_budget,
        "validation_drafts": improvement_config.validation_drafts,
        "behavior_epsilon": improvement_config.behavior_epsilon,
        "opponent_temperature": improvement_config.opponent_temperature,
        "candidate_noise_std": improvement_config.candidate_noise_std,
        "rollouts_per_candidate": improvement_config.rollouts_per_candidate,
        "policy_mix_jitter": improvement_config.policy_mix_jitter,
        "accept_4team_threshold": improvement_config.accept_4team_threshold,
        "threshold_league_size": improvement_config.threshold_league_size,
        "timing_summary": timings,
    }
    validation_summary = {
        **(champion.validation_summary or {}),
        "initial_reward": initial_reward,
        "champion_reward": best_reward,
        "improvement": best_reward - initial_reward,
        "by_slot": final_summary.get("by_slot", {}),
        "variants": final_summary.get("variants", {}),
        "season_summaries": final_summary.get("season_summaries", {}),
        "target_threshold_reward": final_summary.get("target_threshold_reward"),
        "improvement_history": history,
        "objective": config.draft_objective,
        "policy_cycle": list(_policy_cycle(config)),
        "robust": improvement_config.robust,
        "league_sizes": list(league_sizes),
        "seasons": list(sorted(scored_by_season)),
        "target_season": target_season,
        "rollout_budget": improvement_config.rollout_budget,
        "validation_drafts": improvement_config.validation_drafts,
        "behavior_epsilon": improvement_config.behavior_epsilon,
        "opponent_temperature": improvement_config.opponent_temperature,
        "candidate_noise_std": improvement_config.candidate_noise_std,
        "rollouts_per_candidate": improvement_config.rollouts_per_candidate,
        "policy_mix_jitter": improvement_config.policy_mix_jitter,
        "accept_4team_threshold": improvement_config.accept_4team_threshold,
        "threshold_league_size": improvement_config.threshold_league_size,
        "timing_summary": timings,
    }
    champion = _artifact_with_updates(
        champion,
        metadata=metadata,
        training_config=training_config,
        validation_summary=validation_summary,
    )
    return NeuralImprovementResult(
        artifact=champion,
        history=history,
        initial_reward=initial_reward,
        best_reward=best_reward,
        timings=timings,
    )


def benchmark_neural_policy(
    scored: ScoredData,
    config: LeagueConfig,
    artifact: NeuralPolicyArtifact,
    drafts: int = 100,
    eval_slots: str = "all",
    policy_names: tuple[str, ...] = ("neural", "balanced", "scarcity", "best_available"),
    lookahead: bool = False,
    rollout_budget: int | None = None,
    candidate_pool_size: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    data = _get_fast_data(scored, config)
    slots = _parse_eval_slots(eval_slots, config.num_teams)
    comparisons: dict[str, dict[str, Any]] = {}
    total = len(policy_names) * max(drafts, 1)
    for policy_name in policy_names:
        rewards: list[float] = []
        by_slot: dict[int, list[float]] = {slot: [] for slot in slots}
        for index in range(max(drafts, 1)):
            slot = slots[index % len(slots)]
            context = FastPolicyContext(
                neural_artifact=artifact,
                neural_model=_build_model_from_artifact(artifact),
            )
            result = _simulate_policy_override_fast_draft(
                data,
                config,
                override_slot=slot,
                override_policy=policy_name,
                context=context,
                lookahead=lookahead,
                rollout_budget=rollout_budget,
                candidate_pool_size=candidate_pool_size,
            )
            reward = reward_for_team(result, config, slot)
            rewards.append(reward)
            by_slot[slot].append(reward)
            _notify_progress(progress_callback, "benchmark drafts", 1, total)
        comparisons[policy_name] = {
            "average_reward": float(np.mean(rewards)),
            "best_reward": float(np.max(rewards)),
            "worst_reward": float(np.min(rewards)),
            "by_slot": {
                str(slot): float(np.mean(slot_rewards))
                for slot, slot_rewards in by_slot.items()
                if slot_rewards
            },
        }
    return {
        "drafts": max(drafts, 1),
        "policy_cycle": list(_policy_cycle(config)),
        "comparisons": comparisons,
        "timing_summary": {
            "benchmark_seconds": time.perf_counter() - started,
            "lookahead": bool(lookahead),
        },
    }


def benchmark_neural_variants(
    scored: ScoredData,
    config: LeagueConfig,
    artifact: NeuralPolicyArtifact,
    league_sizes: tuple[int, ...] = (4, 8, 10, 12),
    drafts: int = 100,
    eval_slots: str = "all",
    lookahead: bool = False,
    rollout_budget: int | None = None,
    candidate_pool_size: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    variants: dict[str, Any] = {}
    for size in league_sizes:
        variant_config = _variant_config(config, size)
        variants[str(size)] = benchmark_neural_policy(
            scored,
            variant_config,
            artifact,
            drafts=drafts,
            eval_slots=eval_slots,
            lookahead=lookahead,
            rollout_budget=rollout_budget,
            candidate_pool_size=candidate_pool_size,
            progress_callback=progress_callback,
        )
    return {
        "league_sizes": list(league_sizes),
        "variants": variants,
        "timing_summary": {
            "benchmark_seconds": time.perf_counter() - started,
            "lookahead": bool(lookahead),
        },
    }


def save_neural_training_result(result: NeuralTrainingResult, output: str | Path) -> None:
    result.artifact.save(output)
