from __future__ import annotations

import pandas as pd
import pytest

from fflab.config import LeagueConfig
from fflab.scoring import ScoredData
from fflab.web import (
    SESSIONS,
    create_draft_session,
    is_human_team,
    lineup_payload,
    make_human_pick,
    session_state_payload,
)


def tiny_scored_data() -> ScoredData:
    players = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "QB One",
                "position": "QB",
                "season": 2020,
                "season_total_pts": 20,
            },
            {
                "player_id": "p2",
                "player_name": "QB Two",
                "position": "QB",
                "season": 2020,
                "season_total_pts": 12,
            },
        ]
    )
    weekly = pd.DataFrame(
        [
            {"player_id": "p1", "week": 1, "points_scored": 20},
            {"player_id": "p2", "week": 1, "points_scored": 12},
        ]
    )
    return ScoredData(players=players, weekly_scores=weekly)


@pytest.fixture(autouse=True)
def clear_sessions() -> None:
    SESSIONS.clear()


def test_human_team_detection_uses_bot_substring() -> None:
    assert is_human_team("You")
    assert is_human_team("Jordan")
    assert not is_human_team("Alpha Bot")
    assert not is_human_team("robot manager")


def test_starting_draft_auto_advances_bot_and_stops_on_human_pick() -> None:
    config = LeagueConfig(
        num_teams=2,
        team_names=["Alpha Bot", "You"],
        roster_settings={"QB": 1},
    )
    session = create_draft_session(tiny_scored_data(), config, season=2020)
    payload = session_state_payload(session.id)

    assert payload["complete"] is False
    assert payload["currentPick"]["team"] == "You"
    assert payload["currentPick"]["human"] is True
    assert payload["picks"][0]["team"] == "Alpha Bot"
    assert payload["picks"][0]["player_id"] == "p1"
    assert payload["availablePlayers"][0]["player_id"] == "p2"


def test_human_pick_rejects_unavailable_player() -> None:
    config = LeagueConfig(
        num_teams=2,
        team_names=["Alpha Bot", "You"],
        roster_settings={"QB": 1},
    )
    session = create_draft_session(tiny_scored_data(), config, season=2020)

    with pytest.raises(ValueError, match="unavailable"):
        make_human_pick(session.id, "p1")


def test_human_pick_can_complete_draft_and_store_results() -> None:
    config = LeagueConfig(
        num_teams=2,
        team_names=["Alpha Bot", "You"],
        roster_settings={"QB": 1},
    )
    session = create_draft_session(tiny_scored_data(), config, season=2020)
    payload = make_human_pick(session.id, "p2")

    assert payload["complete"] is True
    assert payload["results"] is not None
    assert payload["results"]["roto"][0]["team"] == "Alpha Bot"
    assert payload["results"]["standings"][0]["team"] == "Alpha Bot"


def test_lineup_payload_returns_starters_bench_and_points() -> None:
    config = LeagueConfig(
        num_teams=2,
        team_names=["Alpha Bot", "You"],
        roster_settings={"QB": 1},
    )
    session = create_draft_session(tiny_scored_data(), config, season=2020)
    make_human_pick(session.id, "p2")

    lineup = lineup_payload(session.id, team_index=1, week=1)

    assert lineup["team"] == "You"
    assert lineup["week"] == 1
    assert lineup["total_points"] == 12
    assert lineup["starters"] == [
        {
            "slot": "QB",
            "player_id": "p2",
            "player_name": "QB Two",
            "position": "QB",
            "points": 12.0,
        }
    ]
    assert lineup["bench"] == []
