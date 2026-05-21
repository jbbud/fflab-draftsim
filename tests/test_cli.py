from __future__ import annotations

import json

import pandas as pd

from fflab.cli import main


def test_cli_smoke_with_fixture(tmp_path) -> None:
    config = {
        "num_teams": 2,
        "roster_settings": {"QB": 1},
        "team_names": ["You", "Bot"],
    }
    config_path = tmp_path / "league.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    pd.DataFrame(
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
                "season_total_pts": 10,
            },
        ]
    ).to_csv(fixture / "players.csv", index=False)
    pd.DataFrame(
        [
            {"player_id": "p1", "week": 1, "points_scored": 20},
            {"player_id": "p2", "week": 1, "points_scored": 10},
        ]
    ).to_csv(fixture / "weekly_scores.csv", index=False)

    exit_code = main(
        [
            "draft",
            "--season",
            "2020",
            "--config",
            str(config_path),
            "--fixture-dir",
            str(fixture),
            "--auto",
        ]
    )
    assert exit_code == 0


def test_cli_train_and_evaluate_policy_with_fixture(tmp_path) -> None:
    config = {
        "num_teams": 2,
        "roster_settings": {"QB": 1},
        "team_names": ["Team 1", "Team 2"],
    }
    config_path = tmp_path / "league.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    pd.DataFrame(
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
                "season_total_pts": 10,
            },
        ]
    ).to_csv(fixture / "players.csv", index=False)
    pd.DataFrame(
        [
            {"player_id": "p1", "week": 1, "points_scored": 20},
            {"player_id": "p2", "week": 1, "points_scored": 10},
        ]
    ).to_csv(fixture / "weekly_scores.csv", index=False)

    output = tmp_path / "policy.json"
    train_exit = main(
        [
            "train",
            "--season",
            "2020",
            "--config",
            str(config_path),
            "--fixture-dir",
            str(fixture),
            "--episodes",
            "2",
            "--population",
            "4",
            "--output",
            str(output),
        ]
    )
    assert train_exit == 0
    assert output.exists()

    evaluate_exit = main(
        [
            "evaluate-policy",
            "--season",
            "2020",
            "--config",
            str(config_path),
            "--fixture-dir",
            str(fixture),
            "--policy",
            str(output),
            "--drafts",
            "2",
        ]
    )
    assert evaluate_exit == 0
