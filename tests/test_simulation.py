from __future__ import annotations

import pandas as pd

from fflab.config import LeagueConfig
from fflab.draft import DraftState
from fflab.simulation import evaluate_matchups, generate_round_robin_schedule, simulate_season


def test_round_robin_schedule_handles_odd_leagues() -> None:
    schedule = generate_round_robin_schedule(3, [1, 2, 3])
    assert len(schedule) == 3
    assert all(len(pairings) == 1 for pairings in schedule.values())
    teams_seen = {
        team for pairings in schedule.values() for matchup in pairings for team in matchup
    }
    assert teams_seen == {0, 1, 2}


def test_matchup_standings_sort_by_win_pct_then_points() -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    weekly = pd.DataFrame(
        [
            {"week": 1, "team_index": 0, "team": "Team 1", "team_score": 10.0},
            {"week": 1, "team_index": 1, "team": "Team 2", "team_score": 9.0},
            {"week": 2, "team_index": 0, "team": "Team 1", "team_score": 8.0},
            {"week": 2, "team_index": 1, "team": "Team 2", "team_score": 8.0},
        ]
    )
    standings = evaluate_matchups(weekly, config)
    assert standings.iloc[0]["team"] == "Team 1"
    assert standings.iloc[0]["wins"] == 1
    assert standings.iloc[0]["ties"] == 1


def test_simulate_season_roto_totals() -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
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
                "season_total_pts": 15,
            },
        ]
    )
    weekly = pd.DataFrame(
        [
            {"player_id": "p1", "week": 1, "points_scored": 10},
            {"player_id": "p1", "week": 2, "points_scored": 10},
            {"player_id": "p2", "week": 1, "points_scored": 8},
            {"player_id": "p2", "week": 2, "points_scored": 7},
        ]
    )
    state = DraftState(players, config)
    state.draft_player(0, "p1")
    state.draft_player(1, "p2")
    result = simulate_season(state, players, weekly, config)
    assert result.roto.iloc[0]["team"] == "Team 1"
    assert result.roto.iloc[0]["total_points"] == 20
