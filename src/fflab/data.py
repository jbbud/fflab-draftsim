from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .config import LeagueConfig
from .scoring import ScoredData, build_scored_data


@dataclass(frozen=True)
class RawSeasonData:
    offense: pd.DataFrame
    kicking: pd.DataFrame
    pbp: pd.DataFrame
    schedules: pd.DataFrame


class DataAdapter(Protocol):
    def load_raw(self, season: int) -> RawSeasonData:
        """Load raw data for a season."""


def _to_pandas(value: object) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _call_with_supported_kwargs(function: object, *args: object, **kwargs: object) -> object:
    signature = inspect.signature(function)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **supported)


def _filter_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    if df.empty or "season" not in df.columns:
        return df
    return df[pd.to_numeric(df["season"], errors="coerce").eq(season)].copy()


class NflreadpyAdapter:
    """Live nflverse adapter.

    `nflreadpy` is imported lazily so offline tests and fixture-based runs can use the
    engine without installing live data dependencies.
    """

    def load_raw(self, season: int) -> RawSeasonData:
        try:
            import nflreadpy as nfl
        except ImportError as exc:
            raise RuntimeError(
                "nflreadpy is not installed. Run `python -m pip install -e .` "
                "or use `--fixture-dir` for offline data."
            ) from exc

        offense = _to_pandas(
            _call_with_supported_kwargs(
                nfl.load_player_stats,
                [season],
                summary_level="week",
            )
        )

        kicking = pd.DataFrame()
        try:
            signature = inspect.signature(nfl.load_player_stats)
            if "stat_type" in signature.parameters:
                kicking = _to_pandas(
                    _call_with_supported_kwargs(
                        nfl.load_player_stats,
                        [season],
                        stat_type="kicking",
                        summary_level="week",
                    )
                )
        except (TypeError, ValueError, AttributeError):
            kicking = pd.DataFrame()

        pbp = pd.DataFrame()
        if hasattr(nfl, "load_pbp"):
            try:
                pbp = _to_pandas(nfl.load_pbp([season]))
            except Exception:
                pbp = pd.DataFrame()

        schedules = pd.DataFrame()
        if hasattr(nfl, "load_schedules"):
            try:
                schedules = _to_pandas(nfl.load_schedules([season]))
            except Exception:
                schedules = self._load_schedules_csv_fallback(nfl, season)

        return RawSeasonData(
            offense=offense,
            kicking=kicking,
            pbp=pbp,
            schedules=schedules,
        )

    def _load_schedules_csv_fallback(self, nfl: object, season: int) -> pd.DataFrame:
        try:
            from nflreadpy.config import DataFormat
            from nflreadpy.downloader import get_downloader

            schedules = _to_pandas(
                get_downloader().download(
                    "nflverse-data",
                    "schedules/games",
                    format=DataFormat.CSV,
                )
            )
            return _filter_season(schedules, season)
        except Exception:
            return pd.DataFrame()


def load_scored_season(
    season: int, config: LeagueConfig, adapter: DataAdapter | None = None
) -> ScoredData:
    data_adapter = adapter or NflreadpyAdapter()
    raw = data_adapter.load_raw(season)
    return build_scored_data(
        offense=raw.offense,
        kicking=raw.kicking,
        pbp=raw.pbp,
        schedules=raw.schedules,
        config=config,
        season=season,
    )


def load_fixture_scored_data(path: str | Path) -> ScoredData:
    fixture_dir = Path(path)
    players_path = fixture_dir / "players.csv"
    weekly_path = fixture_dir / "weekly_scores.csv"
    if not players_path.exists() or not weekly_path.exists():
        raise FileNotFoundError(
            "fixture directory must contain players.csv and weekly_scores.csv"
        )

    players = pd.read_csv(players_path)
    weekly_scores = pd.read_csv(weekly_path)
    return ScoredData(players=players, weekly_scores=weekly_scores)
