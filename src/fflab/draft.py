from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import FLEX_ELIGIBLE_POSITIONS, LeagueConfig, STARTER_POSITIONS


@dataclass(frozen=True)
class DraftPick:
    overall: int
    round_number: int
    pick_in_round: int
    team_index: int
    player_id: str
    player_name: str
    position: str


@dataclass
class TeamRoster:
    team_index: int
    player_ids: list[str] = field(default_factory=list)


def generate_snake_order(num_teams: int, rounds: int) -> list[int]:
    order: list[int] = []
    forward = list(range(num_teams))
    reverse = list(reversed(forward))
    for round_index in range(rounds):
        order.extend(forward if round_index % 2 == 0 else reverse)
    return order


class DraftState:
    def __init__(
        self,
        players: pd.DataFrame,
        config: LeagueConfig,
        weekly_scores: pd.DataFrame | None = None,
    ):
        self.config = config
        self.players = players.copy()
        self.weekly_scores = weekly_scores.copy() if weekly_scores is not None else None
        self.players["player_id"] = self.players["player_id"].astype(str)
        self.players["position"] = self.players["position"].astype(str).str.upper()
        if "season_total_pts" not in self.players.columns:
            self.players["season_total_pts"] = self.players.get("projected_total_pts", 0.0)
        if "projected_total_pts" not in self.players.columns:
            self.players["projected_total_pts"] = self.players["season_total_pts"]
        self.players = self.players[
            self.players["position"].isin(config.draftable_positions)
        ].copy()
        self.players = self.players.sort_values(
            "season_total_pts", ascending=False
        ).reset_index(drop=True)
        self._players_by_id = self.players.set_index("player_id", drop=False)
        self.rosters = [TeamRoster(index) for index in range(config.num_teams)]
        self.available_player_ids = set(self.players["player_id"].tolist())
        self.rounds = config.total_roster_slots
        self.order = generate_snake_order(config.num_teams, self.rounds)
        self.pick_index = 0
        self.picks: list[DraftPick] = []

    @property
    def is_complete(self) -> bool:
        return self.pick_index >= len(self.order)

    @property
    def current_round(self) -> int:
        return self.pick_index // self.config.num_teams + 1

    @property
    def pick_in_round(self) -> int:
        return self.pick_index % self.config.num_teams + 1

    @property
    def team_on_clock(self) -> int:
        if self.is_complete:
            raise RuntimeError("draft is complete")
        return self.order[self.pick_index]

    def team_name(self, team_index: int) -> str:
        return self.config.team_labels[team_index]

    def roster(self, team_index: int) -> list[str]:
        return list(self.rosters[team_index].player_ids)

    def roster_counts(self, team_index: int) -> dict[str, int]:
        counts = {position: 0 for position in STARTER_POSITIONS}
        for player_id in self.rosters[team_index].player_ids:
            position = str(self._players_by_id.loc[player_id, "position"])
            counts[position] = counts.get(position, 0) + 1
        return counts

    def roster_size(self, team_index: int) -> int:
        return len(self.rosters[team_index].player_ids)

    def can_add_player(self, team_index: int, player_id: str) -> bool:
        player_id = str(player_id)
        if player_id not in self.available_player_ids:
            return False
        if player_id not in self._players_by_id.index:
            return False
        position = str(self._players_by_id.loc[player_id, "position"])
        if position not in self.config.draftable_positions:
            return False
        return self.roster_size(team_index) < self.config.total_roster_slots

    def need_tier(self, team_index: int, position: str) -> int:
        counts = self.roster_counts(team_index)
        direct_need = self.config.roster_settings.get(position, 0) - counts.get(
            position, 0
        )
        if direct_need > 0:
            return 2
        if position in FLEX_ELIGIBLE_POSITIONS:
            base_required = sum(
                self.config.roster_settings.get(pos, 0)
                for pos in FLEX_ELIGIBLE_POSITIONS
            )
            drafted_flex_eligible = sum(counts.get(pos, 0) for pos in FLEX_ELIGIBLE_POSITIONS)
            flex_capacity = self.config.roster_settings.get("FLEX", 0)
            if drafted_flex_eligible < base_required + flex_capacity:
                return 1
        return 0

    def roster_needs(self, team_index: int) -> dict[str, int]:
        counts = self.roster_counts(team_index)
        needs = {
            position: max(self.config.roster_settings.get(position, 0) - counts[position], 0)
            for position in STARTER_POSITIONS
        }
        base_required = sum(
            self.config.roster_settings.get(pos, 0) for pos in FLEX_ELIGIBLE_POSITIONS
        )
        drafted_flex_eligible = sum(counts[pos] for pos in FLEX_ELIGIBLE_POSITIONS)
        needs["FLEX"] = max(
            self.config.roster_settings.get("FLEX", 0)
            - max(drafted_flex_eligible - base_required, 0),
            0,
        )
        needs["BENCH"] = max(
            self.config.total_roster_slots - self.roster_size(team_index) - sum(needs.values()),
            0,
        )
        return needs

    def available_players_df(self) -> pd.DataFrame:
        if not self.available_player_ids:
            return self.players.iloc[0:0].copy()
        return self.players[self.players["player_id"].isin(self.available_player_ids)].copy()

    def draft_player(self, team_index: int, player_id: str) -> DraftPick:
        if self.is_complete:
            raise RuntimeError("draft is complete")
        if team_index != self.team_on_clock:
            raise ValueError(f"team {team_index} is not on the clock")
        player_id = str(player_id)
        if not self.can_add_player(team_index, player_id):
            raise ValueError(f"player {player_id} cannot be drafted by team {team_index}")

        row = self._players_by_id.loc[player_id]
        self.rosters[team_index].player_ids.append(player_id)
        self.available_player_ids.remove(player_id)
        pick = DraftPick(
            overall=self.pick_index + 1,
            round_number=self.current_round,
            pick_in_round=self.pick_in_round,
            team_index=team_index,
            player_id=player_id,
            player_name=str(row["player_name"]),
            position=str(row["position"]),
        )
        self.picks.append(pick)
        self.pick_index += 1
        return pick
