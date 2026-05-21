from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .config import LeagueConfig


@dataclass(frozen=True)
class ScoredData:
    players: pd.DataFrame
    weekly_scores: pd.DataFrame


def _empty_weekly() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "player_id",
            "player_name",
            "position",
            "season",
            "week",
            "points_scored",
        ]
    )


def _column(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype="float64")


def _text_column(df: pd.DataFrame, names: Iterable[str], default: str = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def _coalesced_text(df: pd.DataFrame, names: Iterable[str], prefix: str) -> pd.Series:
    result = pd.Series("", index=df.index, dtype="object")
    for name in names:
        if name in df.columns:
            values = df[name].fillna("").astype(str)
            result = result.mask(result.eq(""), values)
    missing = result.eq("")
    if missing.any():
        result.loc[missing] = [f"{prefix}_{index}" for index in df.index[missing]]
    return result


def _season_series(df: pd.DataFrame, season: int | None = None) -> pd.Series:
    if "season" in df.columns:
        return pd.to_numeric(df["season"], errors="coerce").fillna(season or 0).astype(int)
    return pd.Series(season or 0, index=df.index, dtype="int64")


def _regular_season_only(df: pd.DataFrame) -> pd.DataFrame:
    if "season_type" not in df.columns:
        return df
    return df[df["season_type"].fillna("REG").astype(str).str.upper().eq("REG")].copy()


def score_offensive_players(
    raw: pd.DataFrame, config: LeagueConfig, season: int | None = None
) -> pd.DataFrame:
    if raw.empty:
        return _empty_weekly()

    df = _regular_season_only(raw).copy()
    if df.empty:
        return _empty_weekly()

    position = _text_column(df, ["position", "recent_position"]).str.upper()
    df = df[position.isin({"QB", "RB", "WR", "TE"})].copy()
    position = position.loc[df.index]
    if df.empty:
        return _empty_weekly()

    scoring = config.scoring
    if "fumbles_lost" in df.columns:
        fumbles_lost = _column(df, ["fumbles_lost"])
    else:
        fumbles_lost = (
            _column(df, ["rushing_fumbles_lost"])
            + _column(df, ["receiving_fumbles_lost"])
            + _column(df, ["sack_fumbles_lost"])
        )

    points = (
        _column(df, ["passing_yards"]) * scoring.passing.yards
        + _column(df, ["passing_tds", "passing_touchdowns"])
        * scoring.passing.touchdowns
        + _column(df, ["interceptions", "passing_interceptions"])
        * scoring.passing.interceptions
        + _column(df, ["passing_2pt_conversions", "passing_two_point_conversions"])
        * scoring.passing.two_point_conversions
        + _column(df, ["rushing_yards"]) * scoring.rushing.yards
        + _column(df, ["rushing_tds", "rushing_touchdowns"])
        * scoring.rushing.touchdowns
        + _column(df, ["rushing_2pt_conversions", "rushing_two_point_conversions"])
        * scoring.rushing.two_point_conversions
        + _column(df, ["receiving_yards"]) * scoring.receiving.yards
        + _column(df, ["receiving_tds", "receiving_touchdowns"])
        * scoring.receiving.touchdowns
        + _column(df, ["receptions"]) * scoring.receiving.receptions
        + _column(df, ["receiving_2pt_conversions", "receiving_two_point_conversions"])
        * scoring.receiving.two_point_conversions
        + fumbles_lost * scoring.misc.fumbles_lost
    )

    return pd.DataFrame(
        {
            "player_id": _coalesced_text(df, ["player_id", "gsis_id"], "OFF"),
            "player_name": _coalesced_text(
                df,
                ["player_display_name", "player_name", "player", "name"],
                "Player",
            ),
            "position": position.values,
            "season": _season_series(df, season).values,
            "week": _column(df, ["week"]).astype(int).values,
            "points_scored": points.round(4).values,
        }
    )


def score_kickers_from_stats(
    raw: pd.DataFrame, config: LeagueConfig, season: int | None = None
) -> pd.DataFrame:
    if raw.empty:
        return _empty_weekly()

    df = _regular_season_only(raw).copy()
    if df.empty:
        return _empty_weekly()
    if "position" in df.columns:
        df = df[df["position"].fillna("").astype(str).str.upper().eq("K")].copy()
    kicking_columns = {
        "field_goals_made",
        "fg_made",
        "fg_made_0_19",
        "fg_made_20_29",
        "fg_made_30_39",
        "fg_made_40_49",
        "fg_made_50_59",
        "fg_made_60_",
        "fg_made_60_plus",
        "field_goals_missed",
        "fg_missed",
        "extra_points_made",
        "pat_made",
        "xp_made",
        "extra_points_missed",
        "pat_missed",
        "xp_missed",
    }
    if df.empty or not kicking_columns.intersection(df.columns):
        return _empty_weekly()

    scoring = config.scoring.kicking
    bucket_columns = [
        "fg_made_0_19",
        "fg_made_20_29",
        "fg_made_30_39",
        "fg_made_40_49",
        "fg_made_50_59",
        "fg_made_60_",
        "fg_made_60_plus",
    ]
    has_buckets = any(column in df.columns for column in bucket_columns)
    if has_buckets:
        points = (
            (
                _column(df, ["fg_made_0_19"])
                + _column(df, ["fg_made_20_29"])
                + _column(df, ["fg_made_30_39"])
            )
            * scoring.field_goal_0_39
            + _column(df, ["fg_made_40_49"]) * scoring.field_goal_40_49
            + (
                _column(df, ["fg_made_50_59"])
                + _column(df, ["fg_made_60_", "fg_made_60_plus"])
            )
            * scoring.field_goal_50_plus
        )
    else:
        points = _column(df, ["field_goals_made", "fg_made"]) * scoring.field_goal_made

    points = (
        points
        + _column(df, ["field_goals_missed", "fg_missed"]) * scoring.field_goal_missed
        + _column(df, ["extra_points_made", "pat_made", "xp_made"])
        * scoring.extra_point_made
        + _column(df, ["extra_points_missed", "pat_missed", "xp_missed"])
        * scoring.extra_point_missed
    )

    return pd.DataFrame(
        {
            "player_id": _coalesced_text(df, ["player_id", "gsis_id"], "K"),
            "player_name": _coalesced_text(
                df,
                ["player_display_name", "player_name", "player", "name"],
                "Kicker",
            ),
            "position": "K",
            "season": _season_series(df, season).values,
            "week": _column(df, ["week"]).astype(int).values,
            "points_scored": points.round(4).values,
        }
    )


def score_kickers_from_pbp(
    raw: pd.DataFrame, config: LeagueConfig, season: int | None = None
) -> pd.DataFrame:
    if raw.empty:
        return _empty_weekly()

    df = _regular_season_only(raw).copy()
    if df.empty or not {"field_goal_result", "extra_point_result"}.intersection(df.columns):
        return _empty_weekly()

    raw_kicker_id = _text_column(df, ["kicker_player_id", "kicker_id"])
    raw_kicker_name = _text_column(df, ["kicker_player_name", "kicker_name"])
    has_kicker = raw_kicker_id.str.len().gt(0) | raw_kicker_name.str.len().gt(0)
    df = df[has_kicker].copy()
    if df.empty:
        return _empty_weekly()

    kicker_id = _coalesced_text(df, ["kicker_player_id", "kicker_id"], "K")
    kicker_name = _coalesced_text(df, ["kicker_player_name", "kicker_name"], "Kicker")
    scoring = config.scoring.kicking
    points = pd.Series(0.0, index=df.index, dtype="float64")

    fg_result = _text_column(df, ["field_goal_result"]).str.lower()
    distance = _column(df, ["kick_distance", "field_goal_distance"])
    fg_made = fg_result.eq("made")
    fg_missed = fg_result.isin({"missed", "blocked"})
    points = points.mask(fg_made & distance.ge(50), scoring.field_goal_50_plus)
    points = points.mask(
        fg_made & distance.ge(40) & distance.lt(50), scoring.field_goal_40_49
    )
    points = points.mask(
        fg_made & (distance.lt(40) | distance.eq(0)), scoring.field_goal_0_39
    )
    points = points + fg_missed.astype(float) * scoring.field_goal_missed

    xp_result = _text_column(df, ["extra_point_result"]).str.lower()
    points = points + xp_result.isin({"good", "made"}).astype(float) * (
        scoring.extra_point_made
    )
    points = points + xp_result.isin({"failed", "blocked"}).astype(float) * (
        scoring.extra_point_missed
    )

    out = pd.DataFrame(
        {
            "player_id": kicker_id.values,
            "player_name": kicker_name.values,
            "position": "K",
            "season": _season_series(df, season).values,
            "week": _column(df, ["week"]).astype(int).values,
            "points_scored": points.values,
        }
    )
    return (
        out.groupby(["player_id", "player_name", "position", "season", "week"], as_index=False)
        ["points_scored"]
        .sum()
    )


def _points_allowed_score(points_allowed: float, config: LeagueConfig) -> float:
    buckets = config.scoring.defense.points_allowed
    if points_allowed <= 0:
        return buckets.get("0", 0.0)
    if points_allowed <= 6:
        return buckets.get("1-6", 0.0)
    if points_allowed <= 13:
        return buckets.get("7-13", 0.0)
    if points_allowed <= 20:
        return buckets.get("14-20", 0.0)
    if points_allowed <= 27:
        return buckets.get("21-27", 0.0)
    if points_allowed <= 34:
        return buckets.get("28-34", 0.0)
    return buckets.get("35+", 0.0)


def _defense_points_allowed_rows(
    schedules: pd.DataFrame, season: int | None = None
) -> pd.DataFrame:
    if schedules.empty:
        return pd.DataFrame(columns=["team", "season", "week", "points_allowed"])
    required = {"home_team", "away_team", "home_score", "away_score", "week"}
    if not required.issubset(set(schedules.columns)):
        return pd.DataFrame(columns=["team", "season", "week", "points_allowed"])

    df = _regular_season_only(schedules).copy()
    if df.empty:
        return pd.DataFrame(columns=["team", "season", "week", "points_allowed"])

    home = pd.DataFrame(
        {
            "team": df["home_team"].astype(str),
            "season": _season_series(df, season),
            "week": pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int),
            "points_allowed": pd.to_numeric(df["away_score"], errors="coerce").fillna(0),
        }
    )
    away = pd.DataFrame(
        {
            "team": df["away_team"].astype(str),
            "season": _season_series(df, season),
            "week": pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int),
            "points_allowed": pd.to_numeric(df["home_score"], errors="coerce").fillna(0),
        }
    )
    return pd.concat([home, away], ignore_index=True)


def score_defenses(
    pbp: pd.DataFrame,
    schedules: pd.DataFrame,
    config: LeagueConfig,
    season: int | None = None,
) -> pd.DataFrame:
    base = _defense_points_allowed_rows(schedules, season)

    stat_columns = [
        "team",
        "season",
        "week",
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "touchdowns",
        "safeties",
        "blocked_kicks",
        "return_touchdowns",
    ]
    stats = pd.DataFrame(columns=stat_columns)

    if not pbp.empty and "defteam" in pbp.columns:
        df = _regular_season_only(pbp).copy()
        df = df[df["defteam"].notna()].copy()
        if not df.empty:
            team = df["defteam"].astype(str)
            touchdown = _column(df, ["touchdown"])
            td_team = _text_column(df, ["td_team", "touchdown_team"])
            defensive_td = (
                (_column(df, ["defensive_touchdown"]) > 0)
                | ((touchdown > 0) & td_team.eq(team))
            ).astype(float)
            return_td = (
                (_column(df, ["return_touchdown"]) > 0)
                | (_column(df, ["special_teams_touchdown"]) > 0)
            ).astype(float)
            play_stats = pd.DataFrame(
                {
                    "team": team,
                    "season": _season_series(df, season),
                    "week": _column(df, ["week"]).astype(int),
                    "sacks": (_column(df, ["sack"]) > 0).astype(float),
                    "interceptions": (
                        _column(df, ["interception", "pass_interception"]) > 0
                    ).astype(float),
                    "fumble_recoveries": (
                        _column(df, ["fumble_lost", "defteam_fumble_recovery"]) > 0
                    ).astype(float),
                    "touchdowns": defensive_td,
                    "safeties": (_column(df, ["safety"]) > 0).astype(float),
                    "blocked_kicks": (
                        _column(df, ["blocked_kick", "punt_blocked", "field_goal_blocked"])
                        > 0
                    ).astype(float),
                    "return_touchdowns": return_td,
                }
            )
            stats = play_stats.groupby(["team", "season", "week"], as_index=False).sum()

    if base.empty and stats.empty:
        return _empty_weekly()
    if base.empty:
        base = stats[["team", "season", "week"]].copy()
        base["points_allowed"] = 0.0

    merged = base.merge(stats, on=["team", "season", "week"], how="left")
    for column in stat_columns[3:]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    merged["points_allowed"] = pd.to_numeric(
        merged["points_allowed"], errors="coerce"
    ).fillna(0.0)

    scoring = config.scoring.defense
    points = (
        merged["points_allowed"].map(lambda value: _points_allowed_score(value, config))
        + merged["sacks"] * scoring.sacks
        + merged["interceptions"] * scoring.interceptions
        + merged["fumble_recoveries"] * scoring.fumble_recoveries
        + merged["touchdowns"] * scoring.touchdowns
        + merged["safeties"] * scoring.safeties
        + merged["blocked_kicks"] * scoring.blocked_kicks
        + merged["return_touchdowns"] * scoring.return_touchdowns
    )

    return pd.DataFrame(
        {
            "player_id": "DEF_" + merged["team"].astype(str),
            "player_name": merged["team"].astype(str) + " DEF",
            "position": "DEF",
            "season": merged["season"].astype(int),
            "week": merged["week"].astype(int),
            "points_scored": points.round(4),
        }
    )


def build_scored_data(
    offense: pd.DataFrame,
    kicking: pd.DataFrame,
    pbp: pd.DataFrame,
    schedules: pd.DataFrame,
    config: LeagueConfig,
    season: int,
) -> ScoredData:
    offensive_scores = score_offensive_players(offense, config, season)
    kicking_scores = score_kickers_from_stats(kicking, config, season)
    if kicking_scores.empty:
        kicking_scores = score_kickers_from_pbp(pbp, config, season)
    defense_scores = score_defenses(pbp, schedules, config, season)

    weekly = pd.concat(
        [offensive_scores, kicking_scores, defense_scores],
        ignore_index=True,
    )
    if weekly.empty:
        return ScoredData(
            players=pd.DataFrame(
                columns=[
                    "player_id",
                    "player_name",
                    "position",
                    "season",
                    "season_total_pts",
                ]
            ),
            weekly_scores=pd.DataFrame(
                columns=["player_id", "week", "points_scored"]
            ),
        )

    weekly = weekly[weekly["position"].isin(config.draftable_positions)].copy()
    if config.week_start is not None:
        weekly = weekly[weekly["week"] >= config.week_start].copy()
    if config.week_end is not None:
        weekly = weekly[weekly["week"] <= config.week_end].copy()

    weekly["points_scored"] = pd.to_numeric(
        weekly["points_scored"], errors="coerce"
    ).fillna(0.0)
    weekly = (
        weekly.groupby(
            ["player_id", "player_name", "position", "season", "week"], as_index=False
        )["points_scored"]
        .sum()
        .sort_values(["week", "points_scored"], ascending=[True, False])
        .reset_index(drop=True)
    )

    players = (
        weekly.groupby(["player_id", "player_name", "position", "season"], as_index=False)
        ["points_scored"]
        .sum()
        .rename(columns={"points_scored": "season_total_pts"})
        .sort_values("season_total_pts", ascending=False)
        .reset_index(drop=True)
    )
    weekly_scores = weekly[["player_id", "week", "points_scored"]].copy()
    return ScoredData(players=players, weekly_scores=weekly_scores)
