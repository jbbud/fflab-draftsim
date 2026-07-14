from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import FLEX_ELIGIBLE_POSITIONS, STARTER_POSITIONS


@dataclass(frozen=True)
class LineupSlot:
    slot: str
    player_id: str
    player_name: str
    position: str
    points: float


@dataclass(frozen=True)
class LineupResult:
    week: int
    starters: list[LineupSlot]
    bench: list[LineupSlot]
    total_points: float


def get_optimal_weekly_lineup(
    roster: list[str],
    week: int,
    players: pd.DataFrame,
    weekly_scores: pd.DataFrame,
    roster_settings: dict[str, int],
) -> LineupResult:
    if not roster:
        return LineupResult(week=week, starters=[], bench=[], total_points=0.0)

    players_by_id = players.copy()
    players_by_id["player_id"] = players_by_id["player_id"].astype(str)
    players_by_id = players_by_id.set_index("player_id", drop=False)
    week_scores = weekly_scores[weekly_scores["week"].eq(week)].copy()
    week_scores["player_id"] = week_scores["player_id"].astype(str)
    score_column = "points_scored" if "points_scored" in week_scores.columns else "projected_points"
    score_by_id = week_scores.groupby("player_id")[score_column].sum().to_dict()

    candidates: list[LineupSlot] = []
    for player_id in roster:
        player_id = str(player_id)
        if player_id not in players_by_id.index:
            continue
        row = players_by_id.loc[player_id]
        candidates.append(
            LineupSlot(
                slot="",
                player_id=player_id,
                player_name=str(row["player_name"]),
                position=str(row["position"]),
                points=float(score_by_id.get(player_id, 0.0)),
            )
        )

    remaining = sorted(candidates, key=lambda item: item.points, reverse=True)
    starters: list[LineupSlot] = []

    def take(position: str, count: int) -> None:
        nonlocal remaining
        if count <= 0:
            return
        matching = [item for item in remaining if item.position == position]
        selected = matching[:count]
        for offset, item in enumerate(selected, start=1):
            label = position if count == 1 else f"{position}{offset}"
            starters.append(
                LineupSlot(
                    slot=label,
                    player_id=item.player_id,
                    player_name=item.player_name,
                    position=item.position,
                    points=item.points,
                )
            )
        selected_ids = {item.player_id for item in selected}
        remaining = [item for item in remaining if item.player_id not in selected_ids]

    for position in STARTER_POSITIONS:
        take(position, roster_settings.get(position, 0))

    flex_count = roster_settings.get("FLEX", 0)
    if flex_count > 0:
        flex_candidates = [
            item for item in remaining if item.position in FLEX_ELIGIBLE_POSITIONS
        ][:flex_count]
        for offset, item in enumerate(flex_candidates, start=1):
            label = "FLEX" if flex_count == 1 else f"FLEX{offset}"
            starters.append(
                LineupSlot(
                    slot=label,
                    player_id=item.player_id,
                    player_name=item.player_name,
                    position=item.position,
                    points=item.points,
                )
            )
        selected_ids = {item.player_id for item in flex_candidates}
        remaining = [item for item in remaining if item.player_id not in selected_ids]

    bench = [
        LineupSlot(
            slot="BENCH",
            player_id=item.player_id,
            player_name=item.player_name,
            position=item.position,
            points=item.points,
        )
        for item in remaining
    ]
    total = sum(item.points for item in starters)
    return LineupResult(week=week, starters=starters, bench=bench, total_points=total)
