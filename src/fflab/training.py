from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FLEX_ELIGIBLE_POSITIONS, LeagueConfig, STARTER_POSITIONS
from .draft import generate_snake_order
from .scoring import ScoredData
from .simulation import generate_round_robin_schedule
from .trainable import (
    DEFAULT_WEIGHT_VALUES,
    FEATURE_NAMES,
    DraftPolicyWeights,
)

POSITIONS = list(STARTER_POSITIONS)
POS_TO_IDX = {position: index for index, position in enumerate(POSITIONS)}
FLEX_POSITION_IDXS = np.array([POS_TO_IDX[pos] for pos in FLEX_ELIGIBLE_POSITIONS], dtype=int)
_SCHEDULE_CACHE: dict[tuple[int, tuple[int, ...]], dict[int, list[tuple[int, int]]]] = {}


@dataclass(frozen=True)
class TrainerConfig:
    episodes: int = 50
    population: int = 32
    elite_fraction: float = 0.25
    seed: int = 7
    eval_slots: str = "spread"
    initial_std: float = 0.75
    min_std: float = 0.05
    smoothing: float = 0.65
    opponents: tuple[str, ...] = ("best_available", "scarcity", "balanced")


@dataclass(frozen=True)
class FastDraftData:
    player_ids: np.ndarray
    player_names: np.ndarray
    position_names: np.ndarray
    positions: np.ndarray
    season_points: np.ndarray
    weekly_points: np.ndarray
    weeks: list[int]
    season_points_norm: np.ndarray
    player_rank_feature: np.ndarray
    position_rank_feature: np.ndarray
    flex_eligible: np.ndarray
    total_by_position: np.ndarray
    players: pd.DataFrame


@dataclass
class FastDraftState:
    data: FastDraftData
    config: LeagueConfig
    order: list[int]
    available: np.ndarray
    rosters: list[list[int]]
    roster_counts: np.ndarray
    roster_size: np.ndarray
    pick_index: int = 0

    @classmethod
    def create(cls, data: FastDraftData, config: LeagueConfig) -> FastDraftState:
        order = generate_snake_order(config.num_teams, config.total_roster_slots)
        return cls(
            data=data,
            config=config,
            order=order,
            available=np.ones(len(data.player_ids), dtype=bool),
            rosters=[[] for _ in range(config.num_teams)],
            roster_counts=np.zeros((config.num_teams, len(POSITIONS)), dtype=int),
            roster_size=np.zeros(config.num_teams, dtype=int),
            pick_index=0,
        )

    def clone(self) -> FastDraftState:
        return FastDraftState(
            data=self.data,
            config=self.config,
            order=list(self.order),
            available=self.available.copy(),
            rosters=[list(roster) for roster in self.rosters],
            roster_counts=self.roster_counts.copy(),
            roster_size=self.roster_size.copy(),
            pick_index=self.pick_index,
        )

    @property
    def is_complete(self) -> bool:
        return self.pick_index >= len(self.order)

    @property
    def team_on_clock(self) -> int:
        return self.order[self.pick_index]

    @property
    def current_round(self) -> int:
        return self.pick_index // self.config.num_teams + 1

    def legal_indices(self, team_index: int) -> np.ndarray:
        if self.roster_size[team_index] >= self.config.total_roster_slots:
            return np.array([], dtype=int)
        return np.flatnonzero(self.available)

    def draft(self, team_index: int, player_index: int) -> None:
        if self.is_complete:
            raise RuntimeError("draft is complete")
        if team_index != self.team_on_clock:
            raise ValueError("team is not on the clock")
        if not self.available[player_index]:
            raise ValueError("player is unavailable")
        if self.roster_size[team_index] >= self.config.total_roster_slots:
            raise ValueError("roster is full")
        self.available[player_index] = False
        self.rosters[team_index].append(int(player_index))
        self.roster_size[team_index] += 1
        self.roster_counts[team_index, self.data.positions[player_index]] += 1
        self.pick_index += 1


@dataclass(frozen=True)
class FastDraftResult:
    rosters: list[list[int]]
    weekly_team_scores: np.ndarray
    roto_totals: np.ndarray
    win_pct: np.ndarray
    points_for: np.ndarray


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    weights: np.ndarray | None = None
    name: str = ""


@dataclass(frozen=True)
class TrainingResult:
    weights: DraftPolicyWeights
    history: list[dict[str, float]]
    baseline_reward: float
    best_reward: float


def build_fast_draft_data(scored: ScoredData, config: LeagueConfig) -> FastDraftData:
    players = scored.players.copy()
    players["player_id"] = players["player_id"].astype(str)
    players["position"] = players["position"].astype(str).str.upper()
    players = players[players["position"].isin(config.draftable_positions)].copy()
    players = players.sort_values("season_total_pts", ascending=False).reset_index(drop=True)

    position_codes = np.array([POS_TO_IDX[pos] for pos in players["position"]], dtype=int)
    season_points = players["season_total_pts"].astype(float).to_numpy()
    max_points = max(float(season_points.max()) if len(season_points) else 0.0, 1.0)
    season_points_norm = season_points / max_points
    n_players = max(len(players) - 1, 1)
    player_rank_feature = 1.0 - (np.arange(len(players), dtype=float) / n_players)

    position_rank_feature = np.zeros(len(players), dtype=float)
    total_by_position = np.zeros(len(POSITIONS), dtype=int)
    for position, position_index in POS_TO_IDX.items():
        mask = position_codes == position_index
        indexes = np.flatnonzero(mask)
        total_by_position[position_index] = len(indexes)
        denom = max(len(indexes) - 1, 1)
        for rank, player_index in enumerate(indexes):
            position_rank_feature[player_index] = 1.0 - (rank / denom)

    weeks = sorted(int(week) for week in scored.weekly_scores["week"].dropna().unique())
    week_to_col = {week: col for col, week in enumerate(weeks)}
    player_to_index = {player_id: index for index, player_id in enumerate(players["player_id"])}
    weekly_points = np.zeros((len(players), len(weeks)), dtype=float)
    weekly = scored.weekly_scores.copy()
    weekly["player_id"] = weekly["player_id"].astype(str)
    for row in weekly.itertuples(index=False):
        player_index = player_to_index.get(str(row.player_id))
        week_col = week_to_col.get(int(row.week))
        if player_index is not None and week_col is not None:
            weekly_points[player_index, week_col] += float(row.points_scored)

    return FastDraftData(
        player_ids=players["player_id"].astype(str).to_numpy(),
        player_names=players["player_name"].astype(str).to_numpy(),
        position_names=players["position"].astype(str).to_numpy(),
        positions=position_codes,
        season_points=season_points,
        weekly_points=weekly_points,
        weeks=weeks,
        season_points_norm=season_points_norm,
        player_rank_feature=player_rank_feature,
        position_rank_feature=position_rank_feature,
        flex_eligible=np.isin(position_codes, FLEX_POSITION_IDXS),
        total_by_position=total_by_position,
        players=players,
    )


def default_weight_vector() -> np.ndarray:
    return np.array([DEFAULT_WEIGHT_VALUES[name] for name in FEATURE_NAMES], dtype=float)


def _target_counts(config: LeagueConfig) -> np.ndarray:
    return np.array(
        [config.roster_settings.get(position, 0) for position in POSITIONS],
        dtype=int,
    )


def _need_tier(state: FastDraftState, team_index: int, candidates: np.ndarray) -> np.ndarray:
    counts = state.roster_counts[team_index]
    targets = _target_counts(state.config)
    direct_need = (targets - counts).clip(min=0)
    tiers = np.zeros(len(candidates), dtype=float)
    candidate_positions = state.data.positions[candidates]
    direct = direct_need[candidate_positions] > 0
    tiers[direct] = 1.0

    flex_count = state.config.roster_settings.get("FLEX", 0)
    if flex_count > 0:
        base_required = sum(
            state.config.roster_settings.get(position, 0)
            for position in FLEX_ELIGIBLE_POSITIONS
        )
        drafted_flex = int(counts[FLEX_POSITION_IDXS].sum())
        flex_still_needed = drafted_flex < base_required + flex_count
        if flex_still_needed:
            flex_mask = np.isin(candidate_positions, FLEX_POSITION_IDXS) & ~direct
            tiers[flex_mask] = 0.5
    return tiers


def fast_candidate_features(
    state: FastDraftState, team_index: int, candidates: np.ndarray
) -> np.ndarray:
    data = state.data
    if len(candidates) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)

    features = np.zeros((len(candidates), len(FEATURE_NAMES)), dtype=float)
    features[:, 0] = data.season_points_norm[candidates]
    features[:, 1] = data.player_rank_feature[candidates]
    features[:, 2] = data.position_rank_feature[candidates]

    max_points = max(float(data.season_points.max()) if len(data.season_points) else 0.0, 1.0)
    dropoff = np.zeros(len(candidates), dtype=float)
    candidate_positions = data.positions[candidates]
    for position_index in range(len(POSITIONS)):
        local = np.flatnonzero(candidate_positions == position_index)
        if len(local) == 0:
            continue
        ordered_local = local[np.argsort(-data.season_points[candidates[local]])]
        ordered_points = data.season_points[candidates[ordered_local]]
        next_points = np.concatenate([ordered_points[1:], np.array([0.0])])
        dropoff[ordered_local] = np.maximum(ordered_points - next_points, 0.0) / max_points
    features[:, 3] = dropoff

    features[:, 4] = _need_tier(state, team_index, candidates)
    features[:, 5] = data.flex_eligible[candidates].astype(float)
    features[:, 6] = state.current_round / max(state.config.total_roster_slots, 1)
    features[:, 7] = team_index / max(state.config.num_teams - 1, 1)

    remaining_counts = np.bincount(
        data.positions[np.flatnonzero(state.available)], minlength=len(POSITIONS)
    )
    total_counts = np.maximum(data.total_by_position, 1)
    features[:, 8] = 1.0 - (remaining_counts[candidate_positions] / total_counts[candidate_positions])

    needs = _need_tier(state, team_index, candidates)
    counts = state.roster_counts[team_index]
    targets = _target_counts(state.config)
    starter_need = int((targets - counts).clip(min=0).sum())
    flex_need = state.config.roster_settings.get("FLEX", 0)
    starter_capacity = max(int(targets.sum()) + flex_need, 1)
    remaining_slots = max(
        int(state.config.total_roster_slots - state.roster_size[team_index]), 0
    )
    bench_capacity = max(state.config.roster_settings.get("BENCH", 0), 1)
    features[:, 9] = max(starter_need, 0) / starter_capacity
    features[:, 10] = max(remaining_slots - starter_need, 0) / bench_capacity
    return features


def _choose_weighted(state: FastDraftState, team_index: int, weights: np.ndarray) -> int:
    candidates = state.legal_indices(team_index)
    features = fast_candidate_features(state, team_index, candidates)
    scores = features @ weights
    return int(candidates[int(np.argmax(scores))])


def _choose_best_available(state: FastDraftState, team_index: int) -> int:
    candidates = state.legal_indices(team_index)
    features = fast_candidate_features(state, team_index, candidates)
    scores = state.data.season_points[candidates] + features[:, 4] * 80.0
    return int(candidates[int(np.argmax(scores))])


def _choose_scarcity(state: FastDraftState, team_index: int) -> int:
    candidates = state.legal_indices(team_index)
    round_number = state.current_round
    positions = state.data.positions[candidates]
    filtered = candidates
    if round_number < state.config.total_roster_slots - 1:
        non_specialists = filtered[
            ~np.isin(state.data.positions[filtered], [POS_TO_IDX["K"], POS_TO_IDX["DEF"]])
        ]
        if len(non_specialists) > 0:
            filtered = non_specialists
    if round_number <= 4:
        scarce = filtered[
            np.isin(state.data.positions[filtered], [POS_TO_IDX["RB"], POS_TO_IDX["WR"]])
        ]
        if len(scarce) > 0:
            filtered = scarce
    candidates = filtered
    positions = state.data.positions[candidates]
    multiplier = np.ones(len(candidates), dtype=float)
    multiplier[np.isin(positions, [POS_TO_IDX["RB"], POS_TO_IDX["WR"]]) & (round_number <= 4)] = 1.2
    multiplier[np.isin(positions, [POS_TO_IDX["QB"], POS_TO_IDX["TE"]]) & (round_number < 5)] = 0.82
    multiplier[np.isin(positions, [POS_TO_IDX["K"], POS_TO_IDX["DEF"]]) & (round_number < state.config.total_roster_slots - 1)] = 0.25
    need = fast_candidate_features(state, team_index, candidates)[:, 4]
    scores = state.data.season_points[candidates] * multiplier + need * 100.0
    return int(candidates[int(np.argmax(scores))])


def _choose_balanced(state: FastDraftState, team_index: int) -> int:
    candidates = state.legal_indices(team_index)
    counts = state.roster_counts[team_index].astype(float)
    targets = _target_counts(state.config).astype(float)
    flex = state.config.roster_settings.get("FLEX", 0)
    if flex > 0:
        for position in FLEX_ELIGIBLE_POSITIONS:
            targets[POS_TO_IDX[position]] += flex / 3.0
    candidate_positions = state.data.positions[candidates]
    ratios = np.full(len(candidates), 99.0, dtype=float)
    useful = targets[candidate_positions] > 0
    ratios[useful] = counts[candidate_positions[useful]] / targets[candidate_positions[useful]]
    thinnest = candidates[ratios == ratios.min()]
    scores = state.data.season_points[thinnest]
    return int(thinnest[int(np.argmax(scores))])


def choose_fast_pick(state: FastDraftState, team_index: int, spec: PolicySpec) -> int:
    if spec.kind == "weighted":
        if spec.weights is None:
            raise ValueError("weighted policy requires weights")
        return _choose_weighted(state, team_index, spec.weights)
    if spec.kind == "scarcity":
        return _choose_scarcity(state, team_index)
    if spec.kind == "balanced":
        return _choose_balanced(state, team_index)
    return _choose_best_available(state, team_index)


def _weekly_lineup_score(
    roster: list[int],
    week_col: int,
    data: FastDraftData,
    config: LeagueConfig,
) -> float:
    if not roster:
        return 0.0
    roster_array = np.array(roster, dtype=int)
    points = data.weekly_points[roster_array, week_col]
    selected = np.zeros(len(roster_array), dtype=bool)
    total = 0.0

    for position in STARTER_POSITIONS:
        count = config.roster_settings.get(position, 0)
        if count <= 0:
            continue
        position_index = POS_TO_IDX[position]
        local = np.flatnonzero((data.positions[roster_array] == position_index) & ~selected)
        if len(local) == 0:
            continue
        ordered = local[np.argsort(-points[local])][:count]
        selected[ordered] = True
        total += float(points[ordered].sum())

    flex_count = config.roster_settings.get("FLEX", 0)
    if flex_count > 0:
        local = np.flatnonzero(np.isin(data.positions[roster_array], FLEX_POSITION_IDXS) & ~selected)
        if len(local) > 0:
            ordered = local[np.argsort(-points[local])][:flex_count]
            selected[ordered] = True
            total += float(points[ordered].sum())
    return total


def _fast_weekly_team_scores(
    rosters: list[list[int]], data: FastDraftData, config: LeagueConfig
) -> np.ndarray:
    scores = np.zeros((config.num_teams, len(data.weeks)), dtype=float)
    for team_index, roster in enumerate(rosters):
        if not roster:
            continue
        roster_array = np.array(roster, dtype=int)
        points = data.weekly_points[roster_array]
        selected = np.zeros(points.shape, dtype=bool)
        team_scores = np.zeros(len(data.weeks), dtype=float)

        for position in STARTER_POSITIONS:
            count = config.roster_settings.get(position, 0)
            if count <= 0:
                continue
            position_index = POS_TO_IDX[position]
            local = np.flatnonzero(data.positions[roster_array] == position_index)
            if len(local) == 0:
                continue
            local_points = points[local]
            if len(local) <= count:
                selected[local, :] = True
                team_scores += local_points.sum(axis=0)
                continue
            ordered = np.argsort(-local_points, axis=0)[:count, :]
            cols = np.arange(points.shape[1]).reshape(1, -1)
            chosen_rows = local[ordered]
            selected[chosen_rows, cols] = True
            team_scores += local_points[ordered, cols].sum(axis=0)

        flex_count = config.roster_settings.get("FLEX", 0)
        if flex_count > 0:
            local = np.flatnonzero(np.isin(data.positions[roster_array], FLEX_POSITION_IDXS))
            if len(local) > 0:
                local_points = np.where(selected[local], -np.inf, points[local])
                ordered = np.argsort(-local_points, axis=0)[:flex_count, :]
                cols = np.arange(points.shape[1]).reshape(1, -1)
                chosen = local_points[ordered, cols]
                team_scores += np.where(np.isfinite(chosen), chosen, 0.0).sum(axis=0)

        scores[team_index] = team_scores
    return scores


def simulate_fast_draft(
    data: FastDraftData,
    config: LeagueConfig,
    policy_specs: list[PolicySpec],
) -> FastDraftResult:
    state = FastDraftState.create(data, config)
    while not state.is_complete:
        team_index = state.team_on_clock
        spec = policy_specs[team_index % len(policy_specs)]
        player_index = choose_fast_pick(state, team_index, spec)
        state.draft(team_index, player_index)

    weekly_scores = _fast_weekly_team_scores(state.rosters, data, config)
    roto_totals = weekly_scores.sum(axis=1)
    win_pct, points_for = _head_to_head_fast(weekly_scores, data.weeks, config)
    return FastDraftResult(
        rosters=[list(roster) for roster in state.rosters],
        weekly_team_scores=weekly_scores,
        roto_totals=roto_totals,
        win_pct=win_pct,
        points_for=points_for,
    )


def _head_to_head_fast(
    weekly_scores: np.ndarray, weeks: list[int], config: LeagueConfig
) -> tuple[np.ndarray, np.ndarray]:
    wins = np.zeros(config.num_teams, dtype=float)
    losses = np.zeros(config.num_teams, dtype=float)
    ties = np.zeros(config.num_teams, dtype=float)
    points_for = np.zeros(config.num_teams, dtype=float)
    schedule_key = (config.num_teams, tuple(weeks))
    schedule = _SCHEDULE_CACHE.get(schedule_key)
    if schedule is None:
        schedule = generate_round_robin_schedule(config.num_teams, weeks)
        _SCHEDULE_CACHE[schedule_key] = schedule
    week_to_col = {week: col for col, week in enumerate(weeks)}
    for week in weeks:
        col = week_to_col[week]
        for home, away in schedule[week]:
            home_score = weekly_scores[home, col]
            away_score = weekly_scores[away, col]
            points_for[home] += home_score
            points_for[away] += away_score
            if home_score > away_score:
                wins[home] += 1
                losses[away] += 1
            elif away_score > home_score:
                wins[away] += 1
                losses[home] += 1
            else:
                ties[home] += 1
                ties[away] += 1
    games = np.maximum(wins + losses + ties, 1.0)
    return (wins + ties * 0.5) / games, points_for


def reward_for_team(result: FastDraftResult, config: LeagueConfig, team_index: int) -> float:
    if config.draft_objective == "head_to_head":
        return float(result.win_pct[team_index] * 1000.0 + result.points_for[team_index] / 100.0)
    return float(result.roto_totals[team_index])


def parse_eval_slots(value: str, num_teams: int) -> list[int]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list(range(num_teams))
    if normalized == "spread":
        slots = {0, num_teams // 2, num_teams - 1}
        return sorted(slot for slot in slots if 0 <= slot < num_teams)
    slots = []
    for part in normalized.split(","):
        if part.strip():
            slot = int(part.strip())
            if 0 <= slot < num_teams:
                slots.append(slot)
    return slots or [0]


def _opponent_spec(name: str) -> PolicySpec:
    key = name.lower()
    if key == "scarcity":
        return PolicySpec(kind="scarcity", name="scarcity")
    if key == "balanced":
        return PolicySpec(kind="balanced", name="balanced")
    return PolicySpec(kind="best_available", name="best_available")


def evaluate_weight_vector(
    data: FastDraftData,
    config: LeagueConfig,
    weights: np.ndarray,
    eval_slots: list[int],
    opponents: tuple[str, ...] = ("best_available", "scarcity", "balanced"),
) -> float:
    rewards = []
    weighted = PolicySpec(kind="weighted", weights=weights, name="weighted")
    for slot in eval_slots:
        specs = [
            _opponent_spec(opponents[index % len(opponents)])
            for index in range(config.num_teams)
        ]
        specs[slot] = weighted
        result = simulate_fast_draft(data, config, specs)
        rewards.append(reward_for_team(result, config, slot))
    return float(np.mean(rewards))


def train_policy(
    scored: ScoredData,
    config: LeagueConfig,
    trainer_config: TrainerConfig,
) -> TrainingResult:
    data = build_fast_draft_data(scored, config)
    rng = np.random.default_rng(trainer_config.seed)
    eval_slots = parse_eval_slots(trainer_config.eval_slots, config.num_teams)
    mean = default_weight_vector()
    std = np.full(len(FEATURE_NAMES), trainer_config.initial_std, dtype=float)
    elite_count = max(1, int(round(trainer_config.population * trainer_config.elite_fraction)))
    best_weights = mean.copy()
    best_reward = evaluate_weight_vector(
        data, config, best_weights, eval_slots, trainer_config.opponents
    )
    baseline_reward = best_reward
    history: list[dict[str, float]] = []

    for episode in range(1, trainer_config.episodes + 1):
        samples = rng.normal(mean, std, size=(trainer_config.population, len(FEATURE_NAMES)))
        samples[0] = mean
        rewards = np.array(
            [
                evaluate_weight_vector(data, config, sample, eval_slots, trainer_config.opponents)
                for sample in samples
            ],
            dtype=float,
        )
        order = np.argsort(-rewards)
        elites = samples[order[:elite_count]]
        elite_rewards = rewards[order[:elite_count]]
        if float(elite_rewards[0]) > best_reward:
            best_reward = float(elite_rewards[0])
            best_weights = elites[0].copy()
        elite_mean = elites.mean(axis=0)
        elite_std = elites.std(axis=0)
        mean = trainer_config.smoothing * elite_mean + (1.0 - trainer_config.smoothing) * mean
        std = np.maximum(elite_std, trainer_config.min_std)
        history.append(
            {
                "episode": float(episode),
                "best_reward": float(best_reward),
                "population_best": float(rewards.max()),
                "population_mean": float(rewards.mean()),
            }
        )

    weights = DraftPolicyWeights(
        weights={
            name: float(best_weights[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        metadata={
            "policy_type": "weighted_evolutionary",
            "feature_names": FEATURE_NAMES,
        },
        training_config={
            "episodes": trainer_config.episodes,
            "population": trainer_config.population,
            "elite_fraction": trainer_config.elite_fraction,
            "seed": trainer_config.seed,
            "eval_slots": trainer_config.eval_slots,
            "opponents": list(trainer_config.opponents),
        },
        validation_summary={
            "baseline_reward": baseline_reward,
            "best_reward": best_reward,
            "improvement": best_reward - baseline_reward,
            "objective": config.draft_objective,
        },
    )
    return TrainingResult(
        weights=weights,
        history=history,
        baseline_reward=baseline_reward,
        best_reward=best_reward,
    )


def evaluate_policy_weights(
    scored: ScoredData,
    config: LeagueConfig,
    weights: DraftPolicyWeights,
    drafts: int = 100,
    eval_slots: str = "all",
) -> dict[str, Any]:
    data = build_fast_draft_data(scored, config)
    slots = parse_eval_slots(eval_slots, config.num_teams)
    vector = np.array(weights.vector(), dtype=float)
    rewards = []
    by_slot: dict[int, list[float]] = {slot: [] for slot in slots}
    for index in range(max(drafts, 1)):
        slot = slots[index % len(slots)]
        reward = evaluate_weight_vector(data, config, vector, [slot])
        rewards.append(reward)
        by_slot[slot].append(reward)
    return {
        "drafts": max(drafts, 1),
        "average_reward": float(np.mean(rewards)),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "by_slot": {
            str(slot): float(np.mean(slot_rewards))
            for slot, slot_rewards in by_slot.items()
            if slot_rewards
        },
    }


def save_training_result(result: TrainingResult, output: str | Path) -> None:
    result.weights.save(output)
