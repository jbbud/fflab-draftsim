from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fflab.ai import get_policy
from fflab.cli import main
from fflab.config import LeagueConfig
from fflab.draft import DraftState
from fflab.neural import (
    NeuralImprovementConfig,
    NeuralDraftPolicy,
    NeuralPolicyArtifact,
    NeuralTrainerConfig,
    benchmark_neural_policy,
    benchmark_neural_variants,
    build_neural_candidate_features,
    evaluate_neural_artifact,
    generate_neural_training_samples,
    improve_neural_policy,
    neural_lookahead_candidates,
    save_neural_training_result,
    train_neural_policy,
    _safe_corr_many,
)
from fflab.scoring import ScoredData
from fflab.training import (
    FastDraftState,
    _fast_weekly_team_scores,
    _weekly_lineup_score,
    build_fast_draft_data,
)

torch = pytest.importorskip("torch")


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
            {
                "player_id": "p3",
                "player_name": "QB Three",
                "position": "QB",
                "season": 2020,
                "season_total_pts": 10,
            },
        ]
    )
    weekly = pd.DataFrame(
        [
            {"player_id": "p1", "week": 1, "points_scored": 10},
            {"player_id": "p1", "week": 2, "points_scored": 20},
            {"player_id": "p2", "week": 1, "points_scored": 12},
            {"player_id": "p2", "week": 2, "points_scored": 8},
            {"player_id": "p3", "week": 1, "points_scored": 2},
            {"player_id": "p3", "week": 2, "points_scored": 8},
        ]
    )
    return ScoredData(players=players, weekly_scores=weekly)


def multi_position_scored_data() -> ScoredData:
    rows = []
    weekly_rows = []
    specs = [
        ("qb", "QB", 12, 30.0),
        ("rb", "RB", 12, 18.0),
        ("wr", "WR", 12, 17.0),
        ("te", "TE", 10, 12.0),
        ("k", "K", 8, 8.0),
        ("def", "DEF", 8, 9.0),
    ]
    for prefix, position, count, base in specs:
        for index in range(1, count + 1):
            player_id = f"{prefix}{index}"
            total = 0.0
            for week in (1, 2):
                points = max(base - index + week, 1.0)
                total += points
                weekly_rows.append(
                    {
                        "player_id": player_id,
                        "week": week,
                        "points_scored": points,
                    }
                )
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": f"{position} {index}",
                    "position": position,
                    "season": 2020,
                    "season_total_pts": total,
                }
            )
    return ScoredData(players=pd.DataFrame(rows), weekly_scores=pd.DataFrame(weekly_rows))


def test_neural_features_include_weekly_scores_and_complementarity() -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1, "BENCH": 1})
    data = build_fast_draft_data(tiny_scored_data(), config)
    state = FastDraftState.create(data, config)
    state.draft(0, 0)
    candidates = state.legal_indices(1)

    features = build_neural_candidate_features(state, 1, candidates)

    assert features.shape[0] == 2
    assert features.shape[1] > len(data.weeks)
    assert np.all(features[:, : len(data.weeks)] >= 0)
    assert np.any(features[:, -7:] >= 0)


def test_vectorized_candidate_correlation_matches_numpy() -> None:
    left = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [5.0, 5.0, 5.0],
        ]
    )
    right = np.array([2.0, 3.0, 4.0])
    expected = np.array([1.0, -1.0, 0.0])

    np.testing.assert_allclose(_safe_corr_many(left, right), expected, atol=1e-6)


def test_vectorized_fast_weekly_scores_match_single_week_optimizer() -> None:
    config = LeagueConfig(
        num_teams=2,
        roster_settings={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1},
    )
    data = build_fast_draft_data(multi_position_scored_data(), config)
    rosters = [list(range(0, 12)), list(range(12, 24))]

    scores = _fast_weekly_team_scores(rosters, data, config)

    for team_index, roster in enumerate(rosters):
        for week_col in range(len(data.weeks)):
            assert scores[team_index, week_col] == pytest.approx(
                _weekly_lineup_score(roster, week_col, data, config)
            )


def test_stochastic_sample_generation_is_seeded_and_varies_by_seed() -> None:
    config = LeagueConfig(num_teams=4, roster_settings={"QB": 1, "RB": 1, "WR": 1, "TE": 1})
    data = build_fast_draft_data(multi_position_scored_data(), config)
    base_config = NeuralTrainerConfig(
        samples=12,
        seed=5,
        max_candidates_per_state=6,
        behavior_epsilon=0.8,
        opponent_temperature=0.25,
        candidate_noise_std=0.15,
        rollouts_per_candidate=2,
        policy_mix_jitter=0.5,
    )

    x1, y1 = generate_neural_training_samples(data, config, base_config)
    x2, y2 = generate_neural_training_samples(data, config, base_config)
    x3, y3 = generate_neural_training_samples(
        data,
        config,
        NeuralTrainerConfig(
            samples=12,
            seed=6,
            max_candidates_per_state=6,
            behavior_epsilon=0.8,
            opponent_temperature=0.25,
            candidate_noise_std=0.15,
            rollouts_per_candidate=2,
            policy_mix_jitter=0.5,
        ),
    )

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y1, y2)
    assert not (np.array_equal(x1, x3) and np.array_equal(y1, y3))


def test_neural_artifact_save_load_and_policy_legal_pick(tmp_path) -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    result = train_neural_policy(
        tiny_scored_data(),
        config,
        NeuralTrainerConfig(samples=8, epochs=1, hidden_dim=8, batch_size=4),
    )
    output = tmp_path / "neural.pt"
    save_neural_training_result(result, output)
    artifact = NeuralPolicyArtifact.load(output)

    state = DraftState(
        tiny_scored_data().players,
        config,
        weekly_scores=tiny_scored_data().weekly_scores,
    )
    policy = NeuralDraftPolicy(artifact, top_k=1, budget_seconds=0)
    pick = policy.choose_pick(state, 0)

    assert pick in {"p1", "p2", "p3"}
    state.draft_player(0, pick)
    assert policy.choose_pick(state, 1) in {"p1", "p2", "p3"} - {pick}


def test_neural_policy_loader_accepts_neural_prefix(tmp_path) -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    result = train_neural_policy(
        tiny_scored_data(),
        config,
        NeuralTrainerConfig(samples=8, epochs=1, hidden_dim=8, batch_size=4),
    )
    output = tmp_path / "neural.pt"
    save_neural_training_result(result, output)

    policy = get_policy(f"neural:{output}")
    state = DraftState(
        tiny_scored_data().players,
        config,
        weekly_scores=tiny_scored_data().weekly_scores,
    )

    assert policy.choose_pick(state, 0) in {"p1", "p2", "p3"}


def test_neural_policy_lookahead_respects_tiny_budget(tmp_path) -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    result = train_neural_policy(
        tiny_scored_data(),
        config,
        NeuralTrainerConfig(samples=8, epochs=1, hidden_dim=8, batch_size=4),
    )
    policy = NeuralDraftPolicy(result.artifact, top_k=3, budget_seconds=0.001)
    state = DraftState(
        tiny_scored_data().players,
        config,
        weekly_scores=tiny_scored_data().weekly_scores,
    )

    assert policy.choose_pick(state, 0) in {"p1", "p2", "p3"}


def test_neural_evaluation_uses_exact_policy_cycle_with_repeated_neural_entries(
    tmp_path,
) -> None:
    config = LeagueConfig(
        num_teams=3,
        roster_settings={"QB": 1},
        ai_policies=["neural:missing.pt", "balanced"],
    )
    result = train_neural_policy(
        tiny_scored_data(),
        config,
        NeuralTrainerConfig(samples=9, epochs=1, hidden_dim=8, batch_size=4),
    )
    summary = evaluate_neural_artifact(
        tiny_scored_data(),
        config,
        result.artifact,
        drafts=3,
        eval_slots="all",
    )

    assert set(summary["by_slot"]) == {"0", "1", "2"}
    assert summary["average_reward"] > 0
    assert result.artifact.training_config["policy_cycle"] == [
        "neural:missing.pt",
        "balanced",
    ]


def test_neural_improvement_keeps_champion_when_candidate_is_worse(
    monkeypatch,
) -> None:
    import fflab.neural as neural

    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    base = train_neural_policy(
        tiny_scored_data(),
        config,
        NeuralTrainerConfig(samples=8, epochs=1, hidden_dim=8, batch_size=4),
    )

    def fake_train_neural_policy(*args, **kwargs):
        return neural.NeuralTrainingResult(
            artifact=base.artifact,
            history=[],
            training_rows=1,
            baseline_reward=base.neural_reward,
            neural_reward=-1.0,
        )

    monkeypatch.setattr(neural, "train_neural_policy", fake_train_neural_policy)
    improved = improve_neural_policy(
        tiny_scored_data(),
        config,
        base.artifact,
        NeuralImprovementConfig(generations=1, samples_per_generation=4),
    )

    assert improved.best_reward == improved.initial_reward
    assert improved.history[-1]["accepted"] is False


def test_neural_threshold_gate_allows_interim_target_progress(monkeypatch) -> None:
    import fflab.neural as neural

    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    base = train_neural_policy(
        tiny_scored_data(),
        config,
        NeuralTrainerConfig(samples=8, epochs=1, hidden_dim=8, batch_size=4),
    )

    def fake_train_neural_policy(*args, **kwargs):
        return neural.NeuralTrainingResult(
            artifact=base.artifact,
            history=[],
            training_rows=1,
            baseline_reward=0.0,
            neural_reward=0.0,
        )

    returns = iter(
        [
            {"average_reward": 100.0, "variants": {"4": {"average_reward": 100.0}}},
            {"average_reward": 100.0, "variants": {"4": {"average_reward": 100.0}}},
            {"average_reward": 99.0, "variants": {"4": {"average_reward": 110.0}}},
            {"average_reward": 110.0, "variants": {"4": {"average_reward": 110.0}}},
            {"average_reward": 99.0, "variants": {"4": {"average_reward": 110.0}}},
            {"average_reward": 110.0, "variants": {"4": {"average_reward": 110.0}}},
        ]
    )

    monkeypatch.setattr(neural, "train_neural_policy", fake_train_neural_policy)
    monkeypatch.setattr(neural, "evaluate_neural_variants", lambda *args, **kwargs: next(returns))

    improved = improve_neural_policy(
        {2025: tiny_scored_data()},
        config,
        base.artifact,
        NeuralImprovementConfig(
            generations=1,
            samples_per_generation=4,
            robust=True,
            league_sizes=(4,),
            target_season=2025,
            accept_4team_threshold=120.0,
            robust_regression_tolerance=0.02,
        ),
    )

    assert improved.history[-1]["accepted"] is True
    assert improved.best_reward == pytest.approx(99.0)
    assert improved.artifact.validation_summary["target_threshold_reward"] == pytest.approx(110.0)


def test_neural_policy_benchmark_compares_slots_and_policies() -> None:
    config = LeagueConfig(num_teams=2, roster_settings={"QB": 1})
    result = train_neural_policy(
        tiny_scored_data(),
        config,
        NeuralTrainerConfig(samples=8, epochs=1, hidden_dim=8, batch_size=4),
    )
    summary = benchmark_neural_policy(
        tiny_scored_data(),
        config,
        result.artifact,
        drafts=2,
        eval_slots="all",
    )

    assert set(summary["comparisons"]) == {
        "neural",
        "balanced",
        "scarcity",
        "best_available",
    }
    assert set(summary["comparisons"]["neural"]["by_slot"]) == {"0", "1"}


def test_neural_lookahead_candidates_include_needed_positions() -> None:
    config = LeagueConfig(
        num_teams=4,
        roster_settings={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1},
    )
    data = build_fast_draft_data(multi_position_scored_data(), config)
    state = FastDraftState.create(data, config)
    candidates = state.legal_indices(0)
    scores = np.where(data.position_names[candidates] == "QB", 100.0, 0.0)

    pool = neural_lookahead_candidates(state, 0, candidates, scores, top_k=2)
    positions = set(data.position_names[pool])

    assert {"RB", "WR", "TE"}.issubset(positions)
    assert "K" not in positions
    assert "DEF" not in positions

    late_state = FastDraftState.create(data, config)
    for player_index in [
        int(np.flatnonzero(data.position_names == "QB")[0]),
        int(np.flatnonzero(data.position_names == "RB")[0]),
        int(np.flatnonzero(data.position_names == "WR")[0]),
        int(np.flatnonzero(data.position_names == "TE")[0]),
        int(np.flatnonzero(data.position_names == "RB")[1]),
    ]:
        late_state.available[player_index] = False
        late_state.rosters[0].append(player_index)
        late_state.roster_counts[0, data.positions[player_index]] += 1
        late_state.roster_size[0] += 1
    late_candidates = late_state.legal_indices(0)
    late_scores = np.where(data.position_names[late_candidates] == "QB", 100.0, 0.0)
    late_pool = neural_lookahead_candidates(late_state, 0, late_candidates, late_scores, top_k=2)
    late_positions = set(data.position_names[late_pool])

    assert {"K", "DEF"}.issubset(late_positions)


def test_neural_four_team_draft_does_not_fill_roster_with_qbs() -> None:
    config = LeagueConfig(
        num_teams=4,
        roster_settings={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        ai_policies=["neural:missing.pt"],
    )
    scored = multi_position_scored_data()
    result = train_neural_policy(
        scored,
        config,
        NeuralTrainerConfig(samples=12, epochs=1, hidden_dim=8, batch_size=4, top_k=2),
    )
    state = DraftState(scored.players, config, weekly_scores=scored.weekly_scores)
    policy = NeuralDraftPolicy(result.artifact, top_k=1, budget_seconds=0)

    while not state.is_complete:
        team_index = state.team_on_clock
        state.draft_player(team_index, policy.choose_pick(state, team_index))

    players = scored.players.set_index("player_id")
    for team_index in range(config.num_teams):
        positions = {str(players.loc[player_id, "position"]) for player_id in state.roster(team_index)}
        assert positions - {"QB"}


def test_neural_variant_benchmark_smoke() -> None:
    config = LeagueConfig(num_teams=4, roster_settings={"QB": 1, "RB": 1, "WR": 1, "TE": 1})
    result = train_neural_policy(
        multi_position_scored_data(),
        config,
        NeuralTrainerConfig(samples=12, epochs=1, hidden_dim=8, batch_size=4),
    )
    summary = benchmark_neural_variants(
        multi_position_scored_data(),
        config,
        result.artifact,
        league_sizes=(4, 8),
        drafts=2,
    )

    assert set(summary["variants"]) == {"4", "8"}
    assert "neural" in summary["variants"]["4"]["comparisons"]


def test_neural_cli_train_and_evaluate_with_fixture(tmp_path) -> None:
    config = {
        "num_teams": 2,
        "roster_settings": {"QB": 1},
        "team_names": ["Team 1", "Team 2"],
    }
    config_path = tmp_path / "league.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    tiny_scored_data().players.to_csv(fixture / "players.csv", index=False)
    tiny_scored_data().weekly_scores.to_csv(fixture / "weekly_scores.csv", index=False)
    output = tmp_path / "neural.pt"

    train_exit = main(
        [
            "train-neural",
            "--seasons",
            "2020,2021",
            "--target-season",
            "2021",
            "--config",
            str(config_path),
            "--fixture-dir",
            str(fixture),
            "--samples",
            "8",
            "--epochs",
            "1",
            "--hidden-dim",
            "8",
            "--batch-size",
            "4",
            "--behavior-epsilon",
            "0.2",
            "--opponent-temperature",
            "0.2",
            "--candidate-noise-std",
            "0.05",
            "--rollouts-per-candidate",
            "1",
            "--policy-mix-jitter",
            "0.1",
            "--rollout-budget",
            "2",
            "--candidate-pool-size",
            "2",
            "--profile",
            "--output",
            str(output),
        ]
    )
    assert train_exit == 0
    assert output.exists()

    eval_exit = main(
        [
            "evaluate-neural",
            "--seasons",
            "2020,2021",
            "--config",
            str(config_path),
            "--fixture-dir",
            str(fixture),
            "--policy",
            str(output),
            "--drafts",
            "2",
            "--lookahead",
            "--rollout-budget",
            "1",
            "--candidate-pool-size",
            "2",
            "--profile",
        ]
    )
    assert eval_exit == 0

    improve_output = tmp_path / "neural_champion.pt"
    improve_exit = main(
        [
            "improve-neural",
            "--seasons",
            "2020,2021",
            "--target-season",
            "2021",
            "--config",
            str(config_path),
            "--fixture-dir",
            str(fixture),
            "--base-policy",
            str(output),
            "--generations",
            "0",
            "--robust",
            "--league-sizes",
            "2",
            "--validation-drafts",
            "1",
            "--rollout-budget",
            "1",
            "--candidate-pool-size",
            "2",
            "--behavior-epsilon",
            "0.2",
            "--opponent-temperature",
            "0.2",
            "--candidate-noise-std",
            "0.05",
            "--rollouts-per-candidate",
            "1",
            "--policy-mix-jitter",
            "0.1",
            "--profile",
            "--output",
            str(improve_output),
        ]
    )
    assert improve_exit == 0
    assert improve_output.exists()

    benchmark_exit = main(
        [
            "benchmark-neural",
            "--seasons",
            "2020,2021",
            "--config",
            str(config_path),
            "--fixture-dir",
            str(fixture),
            "--policy",
            str(improve_output),
            "--drafts",
            "2",
            "--profile",
        ]
    )
    assert benchmark_exit == 0

    variant_exit = main(
        [
            "benchmark-neural-variants",
            "--seasons",
            "2020,2021",
            "--config",
            str(config_path),
            "--fixture-dir",
            str(fixture),
            "--policy",
            str(improve_output),
            "--drafts",
            "2",
            "--league-sizes",
            "2",
            "--lookahead",
            "--rollout-budget",
            "1",
            "--candidate-pool-size",
            "2",
            "--profile",
        ]
    )
    assert variant_exit == 0
