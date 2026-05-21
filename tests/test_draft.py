from __future__ import annotations

import pandas as pd
import pytest

from fflab.config import LeagueConfig
from fflab.draft import DraftState, generate_snake_order


def test_snake_order_even_and_odd() -> None:
    assert generate_snake_order(4, 2) == [0, 1, 2, 3, 3, 2, 1, 0]
    assert generate_snake_order(3, 3) == [0, 1, 2, 2, 1, 0, 0, 1, 2]


def test_draft_state_prevents_unavailable_player() -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    players = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "QB One",
                "position": "QB",
                "season": 2020,
                "season_total_pts": 100,
            },
            {
                "player_id": "p2",
                "player_name": "QB Two",
                "position": "QB",
                "season": 2020,
                "season_total_pts": 90,
            },
        ]
    )
    state = DraftState(players, config)
    state.draft_player(0, "p1")
    with pytest.raises(ValueError):
        state.draft_player(1, "p1")
