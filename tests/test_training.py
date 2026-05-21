from __future__ import annotations

import pandas as pd

from fflab.config import LeagueConfig
from fflab.draft import DraftState
from fflab.scoring import ScoredData
from fflab.simulation import simulate_season
from fflab.training import (
    PolicySpec,
    TrainerConfig,
    build_fast_draft_data,
    evaluate_policy_weights,
    simulate_fast_draft,
    train_policy,
)


def tiny_scored_data() -> ScoredData:
    players = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "QB One",
                "position": "QB",
                "season": 2020,
                "season_total_pts": 30,
            },
            {
                "player_id": "p2",
                "player_name": "QB Two",
                "position": "QB",
                "season": 2020,
                "season_total_pts": 20,
            },
        ]
    )
    weekly = pd.DataFrame(
        [
            {"player_id": "p1", "week": 1, "points_scored": 10},
            {"player_id": "p1", "week": 2, "points_scored": 20},
            {"player_id": "p2", "week": 1, "points_scored": 12},
            {"player_id": "p2", "week": 2, "points_scored": 8},
        ]
    )
    return ScoredData(players=players, weekly_scores=weekly)


def test_fast_simulator_matches_pandas_for_tiny_draft() -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    scored = tiny_scored_data()
    fast_data = build_fast_draft_data(scored, config)
    fast = simulate_fast_draft(
        fast_data,
        config,
        [PolicySpec(kind="best_available"), PolicySpec(kind="best_available")],
    )

    state = DraftState(scored.players, config)
    state.draft_player(0, "p1")
    state.draft_player(1, "p2")
    slow = simulate_season(state, scored.players, scored.weekly_scores, config)

    assert fast.roto_totals.tolist() == slow.roto.sort_values("team_index")[
        "total_points"
    ].tolist()


def test_trainer_smoke_produces_valid_policy() -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    result = train_policy(
        tiny_scored_data(),
        config,
        TrainerConfig(episodes=2, population=4, seed=1, eval_slots="all"),
    )

    assert result.weights.version == 1
    assert "season_points" in result.weights.weights
    assert result.history
    summary = evaluate_policy_weights(tiny_scored_data(), config, result.weights, drafts=2)
    assert summary["drafts"] == 2
    assert summary["average_reward"] > 0
