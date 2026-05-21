from __future__ import annotations

import pandas as pd

from fflab.config import LeagueConfig
from fflab.scoring import (
    score_defenses,
    score_kickers_from_pbp,
    score_offensive_players,
)


def test_offensive_scoring_standard_components() -> None:
    raw = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "QB One",
                "position": "QB",
                "season": 2020,
                "week": 1,
                "passing_yards": 250,
                "passing_tds": 2,
                "interceptions": 1,
                "rushing_yards": 20,
                "rushing_tds": 1,
                "receptions": 0,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "fumbles_lost": 1,
            }
        ]
    )
    scored = score_offensive_players(raw, LeagueConfig(), 2020)
    assert scored.loc[0, "points_scored"] == 22.0


def test_kicker_scoring_from_pbp() -> None:
    raw = pd.DataFrame(
        [
            {
                "kicker_player_id": "k1",
                "kicker_player_name": "Kick One",
                "season": 2020,
                "week": 1,
                "field_goal_result": "made",
                "kick_distance": 52,
            },
            {
                "kicker_player_id": "k1",
                "kicker_player_name": "Kick One",
                "season": 2020,
                "week": 1,
                "extra_point_result": "good",
            },
            {
                "kicker_player_id": "k1",
                "kicker_player_name": "Kick One",
                "season": 2020,
                "week": 1,
                "field_goal_result": "missed",
                "kick_distance": 35,
            },
        ]
    )
    scored = score_kickers_from_pbp(raw, LeagueConfig(), 2020)
    assert scored.loc[0, "points_scored"] == 5.0


def test_defense_scoring_from_schedule_and_pbp() -> None:
    schedules = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 1,
                "home_team": "CHI",
                "away_team": "GB",
                "home_score": 17,
                "away_score": 0,
            }
        ]
    )
    pbp = pd.DataFrame(
        [
            {"season": 2020, "week": 1, "defteam": "CHI", "sack": 1},
            {"season": 2020, "week": 1, "defteam": "CHI", "interception": 1},
            {
                "season": 2020,
                "week": 1,
                "defteam": "CHI",
                "touchdown": 1,
                "td_team": "CHI",
            },
        ]
    )
    scored = score_defenses(pbp, schedules, LeagueConfig(), 2020)
    chi = scored[scored["player_id"].eq("DEF_CHI")].iloc[0]
    assert chi["points_scored"] == 19.0
