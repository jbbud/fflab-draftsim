from __future__ import annotations

import pandas as pd

from fflab.ai import get_policy
from fflab.config import LeagueConfig
from fflab.draft import DraftState
from fflab.trainable import (
    DraftPolicyWeights,
    WeightedDraftPolicy,
    candidate_feature_frame,
)


def test_candidate_features_mask_unavailable_players() -> None:
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
                "season_total_pts": 10,
            },
        ]
    )
    state = DraftState(players, config)
    state.draft_player(0, "p1")

    features = candidate_feature_frame(state, 1)

    assert features["player_id"].tolist() == ["p2"]
    assert set(features.columns).issuperset({"season_points", "need_tier"})


def test_weighted_policy_chooses_highest_weighted_legal_candidate() -> None:
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
                "season_total_pts": 10,
            },
        ]
    )
    state = DraftState(players, config)
    weights = DraftPolicyWeights(weights={"season_points": 10.0})
    policy = WeightedDraftPolicy(weights)

    assert policy.choose_pick(state, 0) == "p1"
    state.draft_player(0, "p1")
    assert policy.choose_pick(state, 1) == "p2"


def test_missing_trained_policy_falls_back_to_weighted_default() -> None:
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
                "season_total_pts": 10,
            },
        ]
    )
    state = DraftState(players, config)
    policy = get_policy("trained:not-a-real-policy.json")

    assert policy.choose_pick(state, 0) == "p1"
