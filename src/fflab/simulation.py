from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import LeagueConfig
from .draft import DraftState
from .optimizer import get_optimal_weekly_lineup


@dataclass(frozen=True)
class SimulationResult:
    weekly_team_scores: pd.DataFrame
    roto: pd.DataFrame
    standings: pd.DataFrame


def generate_round_robin_schedule(
    num_teams: int, weeks: list[int]
) -> dict[int, list[tuple[int, int]]]:
    if num_teams < 2:
        raise ValueError("num_teams must be at least 2")
    teams: list[int | None] = list(range(num_teams))
    if num_teams % 2 == 1:
        teams.append(None)

    rounds: list[list[tuple[int, int]]] = []
    team_count = len(teams)
    rotation = teams[:]
    for round_index in range(team_count - 1):
        pairings: list[tuple[int, int]] = []
        for offset in range(team_count // 2):
            left = rotation[offset]
            right = rotation[team_count - 1 - offset]
            if left is None or right is None:
                continue
            if round_index % 2 == 0:
                pairings.append((left, right))
            else:
                pairings.append((right, left))
        rounds.append(pairings)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]

    return {
        week: rounds[index % len(rounds)]
        for index, week in enumerate(sorted(weeks))
    }


def build_weekly_team_scores(
    state: DraftState,
    players: pd.DataFrame,
    weekly_scores: pd.DataFrame,
    config: LeagueConfig,
) -> pd.DataFrame:
    weeks = sorted(int(week) for week in weekly_scores["week"].dropna().unique())
    rows: list[dict[str, object]] = []
    for week in weeks:
        for team_index in range(config.num_teams):
            lineup = get_optimal_weekly_lineup(
                roster=state.roster(team_index),
                week=week,
                players=players,
                weekly_scores=weekly_scores,
                roster_settings=config.roster_settings,
            )
            rows.append(
                {
                    "week": week,
                    "team_index": team_index,
                    "team": config.team_labels[team_index],
                    "team_score": lineup.total_points,
                }
            )
    return pd.DataFrame(rows)


def evaluate_roto(weekly_team_scores: pd.DataFrame) -> pd.DataFrame:
    if weekly_team_scores.empty:
        return pd.DataFrame(columns=["rank", "team", "total_points"])
    roto = (
        weekly_team_scores.groupby(["team_index", "team"], as_index=False)["team_score"]
        .sum()
        .rename(columns={"team_score": "total_points"})
        .sort_values("total_points", ascending=False)
        .reset_index(drop=True)
    )
    roto.insert(0, "rank", range(1, len(roto) + 1))
    return roto


def evaluate_matchups(
    weekly_team_scores: pd.DataFrame, config: LeagueConfig
) -> pd.DataFrame:
    if weekly_team_scores.empty:
        return pd.DataFrame(
            columns=[
                "team_index",
                "team",
                "wins",
                "losses",
                "ties",
                "win_pct",
                "points_for",
            ]
        )
    weeks = sorted(int(week) for week in weekly_team_scores["week"].unique())
    schedule = generate_round_robin_schedule(config.num_teams, weeks)
    score_lookup = {
        (int(row.week), int(row.team_index)): float(row.team_score)
        for row in weekly_team_scores.itertuples(index=False)
    }
    records = {
        index: {
            "team_index": index,
            "team": config.team_labels[index],
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "points_for": 0.0,
        }
        for index in range(config.num_teams)
    }

    for week in weeks:
        for home, away in schedule[week]:
            home_score = score_lookup.get((week, home), 0.0)
            away_score = score_lookup.get((week, away), 0.0)
            records[home]["points_for"] += home_score
            records[away]["points_for"] += away_score
            if home_score > away_score:
                records[home]["wins"] += 1
                records[away]["losses"] += 1
            elif away_score > home_score:
                records[away]["wins"] += 1
                records[home]["losses"] += 1
            else:
                records[home]["ties"] += 1
                records[away]["ties"] += 1

    standings = pd.DataFrame(records.values())
    games = standings["wins"] + standings["losses"] + standings["ties"]
    standings["win_pct"] = (standings["wins"] + standings["ties"] * 0.5) / games.where(
        games > 0, 1
    )
    standings = standings.sort_values(
        ["win_pct", "points_for"], ascending=[False, False]
    ).reset_index(drop=True)
    return standings


def simulate_season(
    state: DraftState,
    players: pd.DataFrame,
    weekly_scores: pd.DataFrame,
    config: LeagueConfig,
) -> SimulationResult:
    weekly_team_scores = build_weekly_team_scores(state, players, weekly_scores, config)
    return SimulationResult(
        weekly_team_scores=weekly_team_scores,
        roto=evaluate_roto(weekly_team_scores),
        standings=evaluate_matchups(weekly_team_scores, config),
    )
