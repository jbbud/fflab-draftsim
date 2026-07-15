from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .config import FLEX_ELIGIBLE_POSITIONS, STARTER_POSITIONS
from .draft import DraftState


class DraftPolicy(ABC):
    name: str

    @abstractmethod
    def choose_pick(self, state: DraftState, team_index: int) -> str:
        """Return a player_id for the team on the clock."""

    def feature_context(self, state: DraftState, team_index: int) -> dict[str, object]:
        """Stable hook for future learning policies."""
        return {
            "policy": self.name,
            "round": state.current_round,
            "team_index": team_index,
            "roster_counts": state.roster_counts(team_index),
            "roster_needs": state.roster_needs(team_index),
        }

    def _candidates(self, state: DraftState, team_index: int) -> pd.DataFrame:
        df = state.available_players_df()
        if df.empty:
            raise RuntimeError("no available players remain")
        mask = df["player_id"].map(lambda player_id: state.can_add_player(team_index, player_id))
        df = df[mask].copy()
        if df.empty:
            raise RuntimeError(f"team {team_index} has no legal picks")
        return df


class BestAvailablePolicy(DraftPolicy):
    name = "best_available"

    def choose_pick(self, state: DraftState, team_index: int) -> str:
        df = self._candidates(state, team_index)
        df["need_bonus"] = df["position"].map(
            lambda position: 40.0 * state.need_tier(team_index, str(position))
        )
        df["draft_score"] = df["season_total_pts"] + df["need_bonus"]
        return str(df.sort_values("draft_score", ascending=False).iloc[0]["player_id"])


class ScarcityPolicy(DraftPolicy):
    name = "scarcity"

    def choose_pick(self, state: DraftState, team_index: int) -> str:
        df = self._candidates(state, team_index)
        round_number = state.current_round
        if round_number < state.rounds - 1:
            non_specialists = df[~df["position"].isin({"K", "DEF"})].copy()
            if not non_specialists.empty:
                df = non_specialists
        if round_number <= 4:
            scarce_pool = df[df["position"].isin({"RB", "WR"})].copy()
            if not scarce_pool.empty:
                df = scarce_pool

        def multiplier(position: str) -> float:
            if position in {"K", "DEF"} and round_number < state.rounds - 1:
                return 0.25
            if round_number <= 4 and position in {"RB", "WR"}:
                return 1.2
            if round_number < 5 and position in {"QB", "TE"}:
                return 0.82
            return 1.0

        df["need_bonus"] = df["position"].map(
            lambda position: 55.0 * state.need_tier(team_index, str(position))
        )
        df["draft_score"] = df.apply(
            lambda row: row["season_total_pts"] * multiplier(str(row["position"]))
            + row["need_bonus"],
            axis=1,
        )
        return str(df.sort_values("draft_score", ascending=False).iloc[0]["player_id"])


class BalancedPolicy(DraftPolicy):
    name = "balanced"

    def choose_pick(self, state: DraftState, team_index: int) -> str:
        df = self._candidates(state, team_index)
        counts = state.roster_counts(team_index)
        targets: dict[str, float] = {
            position: float(state.config.roster_settings.get(position, 0))
            for position in STARTER_POSITIONS
            if state.config.roster_settings.get(position, 0) > 0
        }
        flex = state.config.roster_settings.get("FLEX", 0)
        if flex > 0:
            for position in FLEX_ELIGIBLE_POSITIONS:
                targets[position] = targets.get(position, 0.0) + flex / 3.0

        def fill_ratio(position: str) -> float:
            target = targets.get(position, 0.0)
            if target <= 0:
                return 99.0
            return counts.get(position, 0) / target

        df["fill_ratio"] = df["position"].map(lambda position: fill_ratio(str(position)))
        min_ratio = df["fill_ratio"].min()
        thinnest = df[df["fill_ratio"].eq(min_ratio)].copy()
        return str(
            thinnest.sort_values("season_total_pts", ascending=False).iloc[0]["player_id"]
        )


POLICIES: dict[str, DraftPolicy] = {
    "best_available": BestAvailablePolicy(),
    "scarcity": ScarcityPolicy(),
    "balanced": BalancedPolicy(),
}


def get_policy(name: str) -> DraftPolicy:
    key = name.strip().lower()
    if key not in POLICIES:
        raise ValueError(f"unknown draft policy: {name}")
    return POLICIES[key]
