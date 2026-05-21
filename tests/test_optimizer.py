from __future__ import annotations

import pandas as pd

from fflab.optimizer import get_optimal_weekly_lineup


def test_optimizer_fills_flex_after_required_slots() -> None:
    players = pd.DataFrame(
        [
            {"player_id": "rb1", "player_name": "RB 1", "position": "RB"},
            {"player_id": "rb2", "player_name": "RB 2", "position": "RB"},
            {"player_id": "wr1", "player_name": "WR 1", "position": "WR"},
            {"player_id": "te1", "player_name": "TE 1", "position": "TE"},
        ]
    )
    weekly = pd.DataFrame(
        [
            {"player_id": "rb1", "week": 1, "points_scored": 10},
            {"player_id": "rb2", "week": 1, "points_scored": 8},
            {"player_id": "wr1", "week": 1, "points_scored": 12},
            {"player_id": "te1", "week": 1, "points_scored": 5},
        ]
    )
    lineup = get_optimal_weekly_lineup(
        roster=["rb1", "rb2", "wr1", "te1"],
        week=1,
        players=players,
        weekly_scores=weekly,
        roster_settings={"RB": 1, "WR": 1, "TE": 1, "FLEX": 1},
    )
    assert lineup.total_points == 35
    assert [slot.player_id for slot in lineup.starters] == ["rb1", "wr1", "te1", "rb2"]
