from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .ai import BestAvailablePolicy, get_policy
from .config import LeagueConfig, load_config
from .data import load_fixture_scored_data, load_scored_season
from .draft import DraftPick, DraftState
from .neural import (
    NeuralImprovementConfig,
    NeuralPolicyArtifact,
    NeuralTrainerConfig,
    benchmark_neural_policy,
    benchmark_neural_variants,
    evaluate_neural_artifact,
    improve_neural_policy,
    save_neural_training_result,
    train_neural_policy,
)
from .simulation import SimulationResult, simulate_season
from .trainable import DraftPolicyWeights
from .training import (
    TrainerConfig,
    evaluate_policy_weights,
    save_training_result,
    train_policy,
)

console = Console()


def _build_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _progress_callback(progress: Progress | None):
    if progress is None:
        return None
    tasks: dict[str, int] = {}

    def callback(name: str, advance: int = 1, total: int | None = None) -> None:
        if name not in tasks:
            tasks[name] = progress.add_task(name, total=total)
        elif total is not None:
            progress.update(tasks[name], total=total)
        progress.advance(tasks[name], advance)

    return callback


def _print_profile_table(title: str, timings: dict[str, float]) -> None:
    table = Table(title=title)
    table.add_column("Phase")
    table.add_column("Seconds", justify="right")
    for name, seconds in timings.items():
        table.add_row(name.replace("_", " "), f"{seconds:.3f}")
    console.print(table)


def _parse_league_sizes(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if part:
            sizes.append(int(part))
    return tuple(sizes)


def _parse_seasons(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    seasons = []
    for part in value.split(","):
        part = part.strip()
        if part:
            seasons.append(int(part))
    return tuple(seasons)


def _requested_neural_seasons(args: argparse.Namespace) -> tuple[int, ...]:
    seasons = _parse_seasons(getattr(args, "seasons", None))
    if seasons:
        return seasons
    season = getattr(args, "season", None)
    if season is not None:
        return (int(season),)
    raise ValueError("pass --season or --seasons")


def _load_scored_many(args: argparse.Namespace, config: LeagueConfig):
    seasons = _requested_neural_seasons(args)
    if len(seasons) == 1:
        return _load_scored(args, config)
    scored = {}
    fixture_dir = Path(args.fixture_dir) if args.fixture_dir else None
    for season in seasons:
        if fixture_dir is not None:
            season_fixture = fixture_dir / str(season)
            scored[season] = load_fixture_scored_data(
                season_fixture if season_fixture.exists() else fixture_dir
            )
        else:
            scored[season] = load_scored_season(season, config)
    return scored


def _primary_scored_for_display(scored):
    if isinstance(scored, dict):
        return scored[sorted(scored)[-1]]
    return scored


def _average_evaluation_summaries(summaries: dict[int, dict]) -> dict:
    rewards = [float(summary["average_reward"]) for summary in summaries.values()]
    return {
        "drafts": next(iter(summaries.values()))["drafts"],
        "average_reward": float(sum(rewards) / max(len(rewards), 1)),
        "best_reward": max(float(summary["best_reward"]) for summary in summaries.values()),
        "worst_reward": min(float(summary["worst_reward"]) for summary in summaries.values()),
        "by_slot": next(iter(summaries.values())).get("by_slot", {}),
        "season_summaries": summaries,
        "timing_summary": {
            "evaluation_seconds": sum(
                float(summary.get("timing_summary", {}).get("evaluation_seconds", 0.0))
                for summary in summaries.values()
            )
        },
    }


def _average_benchmark_summaries(summaries: dict[int, dict]) -> dict:
    first = next(iter(summaries.values()))
    comparisons = {}
    for policy_name in first["comparisons"]:
        values = [summary["comparisons"][policy_name] for summary in summaries.values()]
        by_slot = {}
        for slot in values[0].get("by_slot", {}):
            by_slot[slot] = sum(float(value["by_slot"].get(slot, 0.0)) for value in values) / len(values)
        comparisons[policy_name] = {
            "average_reward": sum(float(value["average_reward"]) for value in values) / len(values),
            "best_reward": max(float(value["best_reward"]) for value in values),
            "worst_reward": min(float(value["worst_reward"]) for value in values),
            "by_slot": by_slot,
        }
    return {
        "drafts": first["drafts"],
        "policy_cycle": first["policy_cycle"],
        "comparisons": comparisons,
        "season_summaries": summaries,
        "timing_summary": {
            "benchmark_seconds": sum(
                float(summary.get("timing_summary", {}).get("benchmark_seconds", 0.0))
                for summary in summaries.values()
            )
        },
    }


def _average_variant_summaries(summaries: dict[int, dict]) -> dict:
    first = next(iter(summaries.values()))
    variants = {}
    for size in first["league_sizes"]:
        size_key = str(size)
        size_summaries = {
            season: summary["variants"][size_key]
            for season, summary in summaries.items()
        }
        variants[size_key] = _average_benchmark_summaries(size_summaries)
    return {
        "league_sizes": first["league_sizes"],
        "variants": variants,
        "season_summaries": summaries,
        "timing_summary": {
            "benchmark_seconds": sum(
                float(summary.get("timing_summary", {}).get("benchmark_seconds", 0.0))
                for summary in summaries.values()
            )
        },
    }


def _players_table(players: pd.DataFrame, title: str, limit: int) -> Table:
    table = Table(title=title)
    table.add_column("ID", overflow="fold")
    table.add_column("Name")
    table.add_column("Pos")
    table.add_column("Pts", justify="right")
    for row in players.head(limit).itertuples(index=False):
        table.add_row(
            str(row.player_id),
            str(row.player_name),
            str(row.position),
            f"{float(row.season_total_pts):.2f}",
        )
    return table


def _roster_table(state: DraftState, team_index: int) -> Table:
    table = Table(title=f"{state.team_name(team_index)} Roster")
    table.add_column("Name")
    table.add_column("Pos")
    table.add_column("Season Pts", justify="right")
    players = state.players.set_index("player_id", drop=False)
    for player_id in state.roster(team_index):
        row = players.loc[player_id]
        table.add_row(
            str(row["player_name"]),
            str(row["position"]),
            f"{float(row['season_total_pts']):.2f}",
        )
    needs = state.roster_needs(team_index)
    table.caption = "Needs: " + ", ".join(
        f"{key} {value}" for key, value in needs.items() if value > 0
    )
    return table


def _print_pick(pick: DraftPick, state: DraftState) -> None:
    console.print(
        f"Pick {pick.overall} R{pick.round_number}.{pick.pick_in_round}: "
        f"{state.team_name(pick.team_index)} selected "
        f"{pick.player_name} ({pick.position})"
    )


def _select_user_pick(state: DraftState, team_index: int, board_limit: int) -> str:
    search = ""
    position_filter = ""
    while True:
        available = state.available_players_df()
        if search:
            available = available[
                available["player_name"].astype(str).str.contains(
                    search, case=False, na=False
                )
            ]
        if position_filter:
            available = available[available["position"].eq(position_filter)]

        console.print(_roster_table(state, team_index))
        console.print(
            _players_table(
                available,
                title=(
                    f"Available Players"
                    f"{' matching ' + search if search else ''}"
                    f"{' at ' + position_filter if position_filter else ''}"
                ),
                limit=board_limit,
            )
        )
        raw = console.input(
            "Enter player ID, `/search name`, `/pos RB`, `/clear`, or `/auto`: "
        ).strip()
        if not raw:
            continue
        if raw == "/auto":
            return BestAvailablePolicy().choose_pick(state, team_index)
        if raw.startswith("/search "):
            search = raw.removeprefix("/search ").strip()
            continue
        if raw.startswith("/pos "):
            position_filter = raw.removeprefix("/pos ").strip().upper()
            continue
        if raw == "/clear":
            search = ""
            position_filter = ""
            continue
        if state.can_add_player(team_index, raw):
            return raw
        console.print(f"[red]Cannot draft `{raw}`. Check the ID and roster capacity.[/red]")


def run_draft(
    players: pd.DataFrame,
    weekly_scores: pd.DataFrame,
    config: LeagueConfig,
    interactive_user: bool = True,
) -> DraftState:
    state = DraftState(players=players, config=config, weekly_scores=weekly_scores)
    policies = [get_policy(name) for name in config.ai_policies]

    while not state.is_complete:
        team_index = state.team_on_clock
        if team_index == 0 and interactive_user:
            player_id = _select_user_pick(state, team_index, config.max_draft_board)
        elif team_index == 0:
            player_id = BestAvailablePolicy().choose_pick(state, team_index)
        else:
            policy = policies[(team_index - 1) % len(policies)]
            player_id = policy.choose_pick(state, team_index)
        pick = state.draft_player(team_index, player_id)
        _print_pick(pick, state)
    return state


def _standings_table(result: SimulationResult) -> Table:
    table = Table(title="Head-to-Head Standings")
    table.add_column("Team")
    table.add_column("W", justify="right")
    table.add_column("L", justify="right")
    table.add_column("T", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("PF", justify="right")
    for row in result.standings.itertuples(index=False):
        table.add_row(
            str(row.team),
            str(row.wins),
            str(row.losses),
            str(row.ties),
            f"{float(row.win_pct):.3f}",
            f"{float(row.points_for):.2f}",
        )
    return table


def _roto_table(result: SimulationResult) -> Table:
    table = Table(title="Roto Leaderboard")
    table.add_column("Rank", justify="right")
    table.add_column("Team")
    table.add_column("Total Pts", justify="right")
    for row in result.roto.itertuples(index=False):
        table.add_row(str(row.rank), str(row.team), f"{float(row.total_points):.2f}")
    return table


def _load_scored(args: argparse.Namespace, config: LeagueConfig):
    if args.fixture_dir:
        return load_fixture_scored_data(Path(args.fixture_dir))
    return load_scored_season(args.season, config)


def command_draft(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    scored = _load_scored(args, config)
    if scored.players.empty:
        console.print("[red]No draftable players were loaded.[/red]")
        return 1

    state = run_draft(
        players=scored.players,
        weekly_scores=scored.weekly_scores,
        config=config,
        interactive_user=not args.auto,
    )
    result = simulate_season(
        state=state,
        players=scored.players,
        weekly_scores=scored.weekly_scores,
        config=config,
    )
    console.print(_roto_table(result))
    console.print(_standings_table(result))
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    scored = _load_scored(args, config)
    if scored.players.empty:
        console.print("[red]No draftable players were loaded.[/red]")
        return 1

    trainer_config = TrainerConfig(
        episodes=args.episodes,
        population=args.population,
        elite_fraction=args.elite_fraction,
        seed=args.seed,
        eval_slots=args.eval_slots,
    )
    result = train_policy(scored, config, trainer_config)
    save_training_result(result, args.output)

    table = Table(title="Training Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Output", str(args.output))
    table.add_row("Baseline Reward", f"{result.baseline_reward:.4f}")
    table.add_row("Best Reward", f"{result.best_reward:.4f}")
    table.add_row("Improvement", f"{result.best_reward - result.baseline_reward:.4f}")
    table.add_row("Episodes", str(args.episodes))
    table.add_row("Population", str(args.population))
    console.print(table)
    return 0


def command_evaluate_policy(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    scored = _load_scored(args, config)
    weights = DraftPolicyWeights.load(args.policy)
    summary = evaluate_policy_weights(
        scored,
        config,
        weights,
        drafts=args.drafts,
        eval_slots=args.eval_slots,
    )

    table = Table(title="Policy Evaluation")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Policy", str(args.policy))
    table.add_row("Drafts", str(summary["drafts"]))
    table.add_row("Average Reward", f"{summary['average_reward']:.4f}")
    table.add_row("Best Reward", f"{summary['best_reward']:.4f}")
    table.add_row("Worst Reward", f"{summary['worst_reward']:.4f}")
    console.print(table)

    slot_table = Table(title="Reward By Draft Slot")
    slot_table.add_column("Slot", justify="right")
    slot_table.add_column("Reward", justify="right")
    for slot, reward in summary["by_slot"].items():
        slot_table.add_row(slot, f"{reward:.4f}")
    console.print(slot_table)
    return 0


def command_train_neural(args: argparse.Namespace) -> int:
    timings: dict[str, float] = {}
    config = load_config(args.config)
    started = time.perf_counter()
    scored = _load_scored_many(args, config)
    timings["data_load_seconds"] = time.perf_counter() - started
    display_scored = _primary_scored_for_display(scored)
    if display_scored.players.empty:
        console.print("[red]No draftable players were loaded.[/red]")
        return 1

    max_candidates = args.candidate_pool_size or args.max_candidates_per_state
    trainer_config = NeuralTrainerConfig(
        samples=args.samples,
        epochs=args.epochs,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_candidates_per_state=max_candidates,
        eval_slots=args.eval_slots,
        top_k=args.top_k,
        rollout_budget=args.rollout_budget,
        behavior_epsilon=args.behavior_epsilon,
        opponent_temperature=args.opponent_temperature,
        candidate_noise_std=args.candidate_noise_std,
        rollouts_per_candidate=args.rollouts_per_candidate,
        policy_mix_jitter=args.policy_mix_jitter,
        target_season=args.target_season,
    )
    progress = _build_progress() if args.profile else None
    with (progress if progress is not None else nullcontext()):
        result = train_neural_policy(
            scored,
            config,
            trainer_config,
            progress_callback=_progress_callback(progress),
        )
    timings.update(result.timings)
    save_neural_training_result(result, args.output)

    table = Table(title="Neural Training Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Output", str(args.output))
    table.add_row("Rows", str(result.training_rows))
    table.add_row("Baseline Reward", f"{result.baseline_reward:.4f}")
    table.add_row("Neural Reward", f"{result.neural_reward:.4f}")
    table.add_row("Improvement", f"{result.neural_reward - result.baseline_reward:.4f}")
    table.add_row("Epochs", str(args.epochs))
    console.print(table)
    if args.profile:
        _print_profile_table("Neural Training Profile", timings)
    return 0


def command_evaluate_neural(args: argparse.Namespace) -> int:
    timings: dict[str, float] = {}
    config = load_config(args.config)
    started = time.perf_counter()
    scored = _load_scored_many(args, config)
    timings["data_load_seconds"] = time.perf_counter() - started
    artifact = NeuralPolicyArtifact.load(args.policy)
    progress = _build_progress() if args.profile else None
    with (progress if progress is not None else nullcontext()):
        if isinstance(scored, dict):
            summary = _average_evaluation_summaries({
                season: evaluate_neural_artifact(
                    season_scored,
                    config,
                    artifact,
                    drafts=args.drafts,
                    eval_slots=args.eval_slots,
                    lookahead=args.lookahead,
                    rollout_budget=args.rollout_budget,
                    candidate_pool_size=args.candidate_pool_size,
                    progress_callback=_progress_callback(progress),
                )
                for season, season_scored in scored.items()
            })
        else:
            summary = evaluate_neural_artifact(
                scored,
                config,
                artifact,
                drafts=args.drafts,
                eval_slots=args.eval_slots,
                lookahead=args.lookahead,
                rollout_budget=args.rollout_budget,
                candidate_pool_size=args.candidate_pool_size,
                progress_callback=_progress_callback(progress),
            )
    timings.update(summary.get("timing_summary", {}))

    table = Table(title="Neural Policy Evaluation")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Policy", str(args.policy))
    table.add_row("Drafts", str(summary["drafts"]))
    table.add_row("Average Reward", f"{summary['average_reward']:.4f}")
    table.add_row("Best Reward", f"{summary['best_reward']:.4f}")
    table.add_row("Worst Reward", f"{summary['worst_reward']:.4f}")
    console.print(table)

    slot_table = Table(title="Reward By Draft Slot")
    slot_table.add_column("Slot", justify="right")
    slot_table.add_column("Reward", justify="right")
    for slot, reward in summary["by_slot"].items():
        slot_table.add_row(slot, f"{reward:.4f}")
    console.print(slot_table)
    if args.profile:
        _print_profile_table("Neural Evaluation Profile", timings)
    return 0


def command_improve_neural(args: argparse.Namespace) -> int:
    timings: dict[str, float] = {}
    config = load_config(args.config)
    started = time.perf_counter()
    scored = _load_scored_many(args, config)
    timings["data_load_seconds"] = time.perf_counter() - started
    display_scored = _primary_scored_for_display(scored)
    if display_scored.players.empty:
        console.print("[red]No draftable players were loaded.[/red]")
        return 1

    base_artifact = NeuralPolicyArtifact.load(args.base_policy)
    candidate_pool = args.candidate_pool_size or args.max_candidates_per_state
    improvement_config = NeuralImprovementConfig(
        generations=args.generations,
        samples_per_generation=args.samples_per_generation,
        epochs_per_generation=args.epochs_per_generation,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_candidates_per_state=candidate_pool,
        eval_slots=args.eval_slots,
        top_k=args.candidate_pool_size or args.top_k,
        robust=args.robust,
        league_sizes=_parse_league_sizes(args.league_sizes),
        rollout_budget=args.rollout_budget,
        validation_drafts=args.validation_drafts,
        behavior_epsilon=args.behavior_epsilon,
        opponent_temperature=args.opponent_temperature,
        candidate_noise_std=args.candidate_noise_std,
        rollouts_per_candidate=args.rollouts_per_candidate,
        policy_mix_jitter=args.policy_mix_jitter,
        target_season=args.target_season,
        accept_4team_threshold=args.accept_4team_threshold,
    )
    progress = _build_progress() if args.profile else None
    with (progress if progress is not None else nullcontext()):
        result = improve_neural_policy(
            scored,
            config,
            base_artifact,
            improvement_config,
            progress_callback=_progress_callback(progress),
        )
    timings.update(result.timings)
    result.artifact.save(args.output)

    table = Table(title="Neural Improvement Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Base Policy", str(args.base_policy))
    table.add_row("Output", str(args.output))
    table.add_row("Initial Reward", f"{result.initial_reward:.4f}")
    table.add_row("Champion Reward", f"{result.best_reward:.4f}")
    table.add_row("Improvement", f"{result.best_reward - result.initial_reward:.4f}")
    table.add_row("Generations", str(args.generations))
    table.add_row("Robust", "yes" if args.robust else "no")
    console.print(table)

    history_table = Table(title="Generation History")
    history_table.add_column("Generation", justify="right")
    history_table.add_column("Reward", justify="right")
    history_table.add_column("Accepted")
    history_table.add_column("Best Reward", justify="right")
    history_table.add_column("Train s", justify="right")
    history_table.add_column("Validate s", justify="right")
    history_table.add_column("Elapsed s", justify="right")
    for row in result.history:
        history_table.add_row(
            str(row["generation"]),
            f"{float(row['reward']):.4f}",
            "yes" if row.get("accepted") else "no",
            f"{float(row.get('best_reward', row['reward'])):.4f}",
            f"{float(row.get('training_seconds', 0.0)):.2f}",
            f"{float(row.get('validation_seconds', 0.0)):.2f}",
            f"{float(row.get('elapsed_seconds', 0.0)):.2f}",
        )
    console.print(history_table)
    if args.profile:
        _print_profile_table("Neural Improvement Profile", timings)
    return 0


def command_benchmark_neural(args: argparse.Namespace) -> int:
    timings: dict[str, float] = {}
    config = load_config(args.config)
    started = time.perf_counter()
    scored = _load_scored_many(args, config)
    timings["data_load_seconds"] = time.perf_counter() - started
    artifact = NeuralPolicyArtifact.load(args.policy)
    progress = _build_progress() if args.profile else None
    with (progress if progress is not None else nullcontext()):
        if isinstance(scored, dict):
            summary = _average_benchmark_summaries({
                season: benchmark_neural_policy(
                    season_scored,
                    config,
                    artifact,
                    drafts=args.drafts,
                    eval_slots=args.eval_slots,
                    lookahead=args.lookahead,
                    rollout_budget=args.rollout_budget,
                    candidate_pool_size=args.candidate_pool_size,
                    progress_callback=_progress_callback(progress),
                )
                for season, season_scored in scored.items()
            })
        else:
            summary = benchmark_neural_policy(
                scored,
                config,
                artifact,
                drafts=args.drafts,
                eval_slots=args.eval_slots,
                lookahead=args.lookahead,
                rollout_budget=args.rollout_budget,
                candidate_pool_size=args.candidate_pool_size,
                progress_callback=_progress_callback(progress),
            )
    timings.update(summary.get("timing_summary", {}))

    table = Table(title="Exact-League Policy Benchmark")
    table.add_column("Policy")
    table.add_column("Average", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("Worst", justify="right")
    for policy_name, values in summary["comparisons"].items():
        table.add_row(
            policy_name,
            f"{float(values['average_reward']):.4f}",
            f"{float(values['best_reward']):.4f}",
            f"{float(values['worst_reward']):.4f}",
        )
    console.print(table)

    slot_table = Table(title="Reward By Policy And Draft Slot")
    slot_table.add_column("Policy")
    slots = sorted(
        {
            int(slot)
            for values in summary["comparisons"].values()
            for slot in values["by_slot"]
        }
    )
    for slot in slots:
        slot_table.add_column(str(slot), justify="right")
    for policy_name, values in summary["comparisons"].items():
        slot_table.add_row(
            policy_name,
            *[
                f"{float(values['by_slot'].get(str(slot), 0.0)):.2f}"
                for slot in slots
            ],
        )
    console.print(slot_table)
    if args.profile:
        _print_profile_table("Neural Benchmark Profile", timings)
    return 0


def command_benchmark_neural_variants(args: argparse.Namespace) -> int:
    timings: dict[str, float] = {}
    config = load_config(args.config)
    started = time.perf_counter()
    scored = _load_scored_many(args, config)
    timings["data_load_seconds"] = time.perf_counter() - started
    artifact = NeuralPolicyArtifact.load(args.policy)
    league_sizes = _parse_league_sizes(args.league_sizes) or (4, 8, 10, 12)
    progress = _build_progress() if args.profile else None
    with (progress if progress is not None else nullcontext()):
        if isinstance(scored, dict):
            summary = _average_variant_summaries({
                season: benchmark_neural_variants(
                    season_scored,
                    config,
                    artifact,
                    league_sizes=league_sizes,
                    drafts=args.drafts,
                    eval_slots=args.eval_slots,
                    lookahead=args.lookahead,
                    rollout_budget=args.rollout_budget,
                    candidate_pool_size=args.candidate_pool_size,
                    progress_callback=_progress_callback(progress),
                )
                for season, season_scored in scored.items()
            })
        else:
            summary = benchmark_neural_variants(
                scored,
                config,
                artifact,
                league_sizes=league_sizes,
                drafts=args.drafts,
                eval_slots=args.eval_slots,
                lookahead=args.lookahead,
                rollout_budget=args.rollout_budget,
                candidate_pool_size=args.candidate_pool_size,
                progress_callback=_progress_callback(progress),
            )
    timings.update(summary.get("timing_summary", {}))

    table = Table(title="Neural Variant Benchmark")
    table.add_column("League Size", justify="right")
    table.add_column("Neural", justify="right")
    table.add_column("Balanced", justify="right")
    table.add_column("Scarcity", justify="right")
    table.add_column("Best Available", justify="right")
    for size in summary["league_sizes"]:
        comparisons = summary["variants"][str(size)]["comparisons"]
        table.add_row(
            str(size),
            f"{float(comparisons['neural']['average_reward']):.4f}",
            f"{float(comparisons['balanced']['average_reward']):.4f}",
            f"{float(comparisons['scarcity']['average_reward']):.4f}",
            f"{float(comparisons['best_available']['average_reward']):.4f}",
        )
    console.print(table)
    if args.profile:
        _print_profile_table("Neural Variant Benchmark Profile", timings)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fflab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    draft = subparsers.add_parser("draft", help="run a retroactive snake draft")
    draft.add_argument("--season", type=int, required=True)
    draft.add_argument("--config", type=Path)
    draft.add_argument("--fixture-dir", type=Path)
    draft.add_argument(
        "--auto",
        action="store_true",
        help="auto-draft the user team with best available",
    )
    draft.set_defaults(func=command_draft)

    train = subparsers.add_parser("train", help="train a weighted draft policy")
    train.add_argument("--season", type=int, required=True)
    train.add_argument("--config", type=Path)
    train.add_argument("--fixture-dir", type=Path)
    train.add_argument("--episodes", type=int, default=50)
    train.add_argument("--population", type=int, default=32)
    train.add_argument("--elite-fraction", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--eval-slots", default="spread")
    train.add_argument("--output", type=Path, required=True)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser(
        "evaluate-policy", help="evaluate a trained weighted draft policy"
    )
    evaluate.add_argument("--season", type=int, required=True)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--fixture-dir", type=Path)
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--drafts", type=int, default=100)
    evaluate.add_argument("--eval-slots", default="all")
    evaluate.set_defaults(func=command_evaluate_policy)

    train_neural = subparsers.add_parser(
        "train-neural", help="train a weekly-vector neural draft policy"
    )
    train_neural.add_argument("--season", type=int)
    train_neural.add_argument("--seasons")
    train_neural.add_argument("--target-season", type=int)
    train_neural.add_argument("--config", type=Path)
    train_neural.add_argument("--fixture-dir", type=Path)
    train_neural.add_argument("--samples", type=int, default=5000)
    train_neural.add_argument("--epochs", type=int, default=20)
    train_neural.add_argument("--seed", type=int, default=11)
    train_neural.add_argument("--hidden-dim", type=int, default=64)
    train_neural.add_argument("--batch-size", type=int, default=128)
    train_neural.add_argument("--learning-rate", type=float, default=0.001)
    train_neural.add_argument("--max-candidates-per-state", type=int, default=6)
    train_neural.add_argument("--eval-slots", default="all")
    train_neural.add_argument("--top-k", type=int, default=5)
    train_neural.add_argument("--rollout-budget", type=int, default=None)
    train_neural.add_argument("--candidate-pool-size", type=int, default=None)
    train_neural.add_argument("--behavior-epsilon", type=float, default=0.12)
    train_neural.add_argument("--opponent-temperature", type=float, default=0.12)
    train_neural.add_argument("--candidate-noise-std", type=float, default=0.0)
    train_neural.add_argument("--rollouts-per-candidate", type=int, default=1)
    train_neural.add_argument("--policy-mix-jitter", type=float, default=0.0)
    train_neural.add_argument("--profile", action="store_true")
    train_neural.add_argument("--output", type=Path, required=True)
    train_neural.set_defaults(func=command_train_neural)

    evaluate_neural = subparsers.add_parser(
        "evaluate-neural", help="evaluate a weekly-vector neural draft policy"
    )
    evaluate_neural.add_argument("--season", type=int)
    evaluate_neural.add_argument("--seasons")
    evaluate_neural.add_argument("--config", type=Path)
    evaluate_neural.add_argument("--fixture-dir", type=Path)
    evaluate_neural.add_argument("--policy", type=Path, required=True)
    evaluate_neural.add_argument("--drafts", type=int, default=100)
    evaluate_neural.add_argument("--eval-slots", default="all")
    evaluate_neural.add_argument("--lookahead", action="store_true")
    evaluate_neural.add_argument("--rollout-budget", type=int, default=None)
    evaluate_neural.add_argument("--candidate-pool-size", type=int, default=None)
    evaluate_neural.add_argument("--profile", action="store_true")
    evaluate_neural.set_defaults(func=command_evaluate_neural)

    improve_neural = subparsers.add_parser(
        "improve-neural",
        help="iteratively improve a neural policy against the exact configured league",
    )
    improve_neural.add_argument("--season", type=int)
    improve_neural.add_argument("--seasons")
    improve_neural.add_argument("--target-season", type=int)
    improve_neural.add_argument("--config", type=Path)
    improve_neural.add_argument("--fixture-dir", type=Path)
    improve_neural.add_argument("--base-policy", type=Path, required=True)
    improve_neural.add_argument("--generations", type=int, default=5)
    improve_neural.add_argument("--samples-per-generation", type=int, default=6000)
    improve_neural.add_argument("--epochs-per-generation", type=int, default=12)
    improve_neural.add_argument("--seed", type=int, default=101)
    improve_neural.add_argument("--batch-size", type=int, default=256)
    improve_neural.add_argument("--learning-rate", type=float, default=0.001)
    improve_neural.add_argument("--max-candidates-per-state", type=int, default=12)
    improve_neural.add_argument("--eval-slots", default="all")
    improve_neural.add_argument("--top-k", type=int, default=8)
    improve_neural.add_argument("--rollout-budget", type=int, default=None)
    improve_neural.add_argument("--candidate-pool-size", type=int, default=None)
    improve_neural.add_argument("--validation-drafts", type=int, default=None)
    improve_neural.add_argument("--behavior-epsilon", type=float, default=0.12)
    improve_neural.add_argument("--opponent-temperature", type=float, default=0.12)
    improve_neural.add_argument("--candidate-noise-std", type=float, default=0.0)
    improve_neural.add_argument("--rollouts-per-candidate", type=int, default=1)
    improve_neural.add_argument("--policy-mix-jitter", type=float, default=0.0)
    improve_neural.add_argument("--accept-4team-threshold", type=float, default=None)
    improve_neural.add_argument("--profile", action="store_true")
    improve_neural.add_argument(
        "--robust",
        action="store_true",
        help="train and accept champions against multiple league sizes",
    )
    improve_neural.add_argument(
        "--league-sizes",
        default=None,
        help="comma-separated league sizes for robust mode, default 4,8,10,12",
    )
    improve_neural.add_argument("--output", type=Path, required=True)
    improve_neural.set_defaults(func=command_improve_neural)

    benchmark_neural = subparsers.add_parser(
        "benchmark-neural",
        help="compare neural and heuristic policies across exact league draft slots",
    )
    benchmark_neural.add_argument("--season", type=int)
    benchmark_neural.add_argument("--seasons")
    benchmark_neural.add_argument("--config", type=Path)
    benchmark_neural.add_argument("--fixture-dir", type=Path)
    benchmark_neural.add_argument("--policy", type=Path, required=True)
    benchmark_neural.add_argument("--drafts", type=int, default=100)
    benchmark_neural.add_argument("--eval-slots", default="all")
    benchmark_neural.add_argument("--lookahead", action="store_true")
    benchmark_neural.add_argument("--rollout-budget", type=int, default=None)
    benchmark_neural.add_argument("--candidate-pool-size", type=int, default=None)
    benchmark_neural.add_argument("--profile", action="store_true")
    benchmark_neural.set_defaults(func=command_benchmark_neural)

    benchmark_neural_variants = subparsers.add_parser(
        "benchmark-neural-variants",
        help="benchmark a neural policy across multiple league sizes",
    )
    benchmark_neural_variants.add_argument("--season", type=int)
    benchmark_neural_variants.add_argument("--seasons")
    benchmark_neural_variants.add_argument("--config", type=Path)
    benchmark_neural_variants.add_argument("--fixture-dir", type=Path)
    benchmark_neural_variants.add_argument("--policy", type=Path, required=True)
    benchmark_neural_variants.add_argument("--drafts", type=int, default=100)
    benchmark_neural_variants.add_argument("--eval-slots", default="all")
    benchmark_neural_variants.add_argument("--league-sizes", default="4,8,10,12")
    benchmark_neural_variants.add_argument("--lookahead", action="store_true")
    benchmark_neural_variants.add_argument("--rollout-budget", type=int, default=None)
    benchmark_neural_variants.add_argument("--candidate-pool-size", type=int, default=None)
    benchmark_neural_variants.add_argument("--profile", action="store_true")
    benchmark_neural_variants.set_defaults(func=command_benchmark_neural_variants)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
