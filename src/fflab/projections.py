"""Normalize ESPN fantasy data for the hosted draft simulator."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Callable, Iterable


DRAFTABLE_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}
POSITION_ID_MAP = {
    0: "QB",
    2: "RB",
    4: "WR",
    6: "TE",
    16: "DEF",
    17: "K",
}
DEFAULT_WEEK_START = 1
DEFAULT_WEEK_END = 17
PROJECTED_WEEKLY_STAT_FILTERS = ("11{year}", "12{year}")


@dataclass(frozen=True)
class EspnSourceConfig:
    """Validated ESPN sync settings supplied by the browser or environment."""

    league_id: int
    year: int
    espn_s2: str | None = None
    swid: str | None = None
    week_start: int = DEFAULT_WEEK_START
    week_end: int = DEFAULT_WEEK_END
    batch_size: int = 50

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EspnSourceConfig:
        """Create a sync config from an API payload."""
        league_id = int(payload.get("league_id") or 0)
        year = int(payload.get("year") or datetime.now(timezone.utc).year)
        if league_id <= 0:
            raise ValueError("league_id is required")
        week_start = int(payload.get("week_start") or DEFAULT_WEEK_START)
        week_end = int(payload.get("week_end") or DEFAULT_WEEK_END)
        if week_start < 1 or week_end < week_start:
            raise ValueError("week_start/week_end are invalid")
        batch_size = max(int(payload.get("batch_size") or 50), 1)
        return cls(
            league_id=league_id,
            year=year,
            espn_s2=_blank_to_none(payload.get("espn_s2")),
            swid=_blank_to_none(payload.get("swid")),
            week_start=week_start,
            week_end=week_end,
            batch_size=batch_size,
        )


def _blank_to_none(value: object) -> str | None:
    """Convert blank-ish payload fields into None."""
    text = "" if value is None else str(value).strip()
    return text or None


def _number(value: object, default: float = 0.0) -> float:
    """Parse a numeric value, returning a fallback on bad input."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: object, default: int = 0) -> int:
    """Parse an integer value, returning a fallback on bad input."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_position(value: object) -> str:
    """Normalize ESPN position labels into the app's draft positions."""
    position = str(value or "").upper().strip()
    if position in {"D/ST", "DST", "DEFENSE"}:
        return "DEF"
    return position


def _position_from_id(value: object) -> str:
    """Map ESPN default position ids to draftable position labels."""
    if value is None:
        return ""
    return POSITION_ID_MAP.get(_integer(value, -999), "")


def _position_from_slots(value: object) -> str:
    """Infer a draftable position from ESPN eligible slot ids or labels."""
    if not isinstance(value, list):
        return ""
    for slot in value:
        position = normalize_position(slot)
        if position in DRAFTABLE_POSITIONS:
            return position
        position = _position_from_id(slot)
        if position in DRAFTABLE_POSITIONS:
            return position
    return ""


def _object_value(obj: object, name: str, default: object = None) -> object:
    """Read a field from either a dict or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stat_value(stat: object, name: str, default: object = None) -> object:
    """Read a stat field from either a dict or an object attribute."""
    if isinstance(stat, dict):
        return stat.get(name, default)
    return getattr(stat, name, default)


def _stat_projected_points(stat: object) -> float:
    """Extract projected points from the known stat field names."""
    if stat is None:
        return 0.0
    direct = _stat_value(stat, "projected_points")
    if direct is not None:
        return _number(direct)
    direct = _stat_value(stat, "projectedPoints")
    if direct is not None:
        return _number(direct)
    return _number(_stat_value(stat, "projected"))


def _stat_has_projected_points(stat: object) -> bool:
    """Return whether a stat object carries a projected-points field."""
    if stat is None:
        return False
    for name in ("projected_points", "projectedPoints", "projected"):
        if _stat_value(stat, name) is not None:
            return True
    return False


def _raw_player_payload(raw: object) -> dict[str, Any]:
    """Return the nested raw ESPN player dict when one is present."""
    if not isinstance(raw, dict):
        return {}
    player_pool = raw.get("playerPoolEntry")
    if isinstance(player_pool, dict) and isinstance(player_pool.get("player"), dict):
        return player_pool["player"]
    player = raw.get("player")
    if isinstance(player, dict):
        return player
    return raw


def _raw_player_id(raw: object) -> str:
    """Extract a player id from raw ESPN card payloads."""
    raw_player = _raw_player_payload(raw)
    player_id = raw_player.get("id") or raw_player.get("playerId")
    if not player_id and isinstance(raw, dict):
        player_id = raw.get("id") or raw.get("playerId")
    return str(player_id or "")


def _raw_player_stats(raw: object) -> list[dict[str, Any]]:
    """Return valid raw stat rows from a player-card payload."""
    raw_player = _raw_player_payload(raw)
    stats = raw_player.get("stats", [])
    if not isinstance(stats, list):
        return []
    return [stat for stat in stats if isinstance(stat, dict)]


def _merge_raw_stats(raw: dict[str, Any], stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge supplemental weekly stat rows into a raw player payload."""
    if not stats:
        return raw
    raw_player = _raw_player_payload(raw)
    if not raw_player:
        return raw
    existing = raw_player.setdefault("stats", [])
    if not isinstance(existing, list):
        raw_player["stats"] = existing = []
    seen = {
        (
            stat.get("id"),
            stat.get("scoringPeriodId"),
            stat.get("statSourceId"),
            stat.get("statSplitTypeId"),
        )
        for stat in existing
        if isinstance(stat, dict)
    }
    for stat in stats:
        key = (
            stat.get("id"),
            stat.get("scoringPeriodId"),
            stat.get("statSourceId"),
            stat.get("statSplitTypeId"),
        )
        if key not in seen:
            existing.append(stat)
            seen.add(key)
    return raw


def _raw_projected_points(raw: object, scoring_period: int, year: int) -> float | None:
    """Find projected points for one scoring period in raw ESPN stats."""
    candidates: list[tuple[int, float]] = []
    for stat in _raw_player_stats(raw):
        if _integer(stat.get("seasonId")) != year:
            continue
        if _integer(stat.get("scoringPeriodId"), -999) != scoring_period:
            continue
        if _integer(stat.get("statSourceId"), -1) != 1:
            continue
        if stat.get("appliedTotal") is None:
            continue
        split_type = _integer(stat.get("statSplitTypeId"), 999)
        split_priority = 0 if split_type in {0, 1} else 1
        candidates.append((split_priority, _number(stat.get("appliedTotal"))))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _raw_projection_stat_summary(raw: object, year: int, week_start: int, week_end: int) -> tuple[Counter[str], Counter[str]]:
    """Count raw projection rows for response metadata and diagnostics."""
    counts: Counter[str] = Counter()
    split_types: Counter[str] = Counter()
    for stat in _raw_player_stats(raw):
        if _integer(stat.get("seasonId")) != year:
            continue
        counts["stat_rows"] += 1
        scoring_period = _integer(stat.get("scoringPeriodId"), -999)
        stat_source = _integer(stat.get("statSourceId"), -1)
        split_type = _integer(stat.get("statSplitTypeId"), -999)
        if stat_source == 1:
            counts["projected_rows"] += 1
            split_types[str(split_type)] += 1
            if scoring_period == 0:
                counts["projected_season_rows"] += 1
            elif week_start <= scoring_period <= week_end:
                counts["projected_week_rows"] += 1
                if stat.get("appliedTotal") is not None:
                    counts["projected_week_rows_with_total"] += 1
                if split_type == 2:
                    counts["projected_week_split_type_2_rows"] += 1
            else:
                counts["projected_other_rows"] += 1
        elif stat_source == 0 and week_start <= scoring_period <= week_end:
            counts["actual_week_rows"] += 1
    return counts, split_types


def _normalized_position(player: object, raw: object | None = None) -> str:
    """Choose the best draftable position from object and raw player data."""
    position = normalize_position(_object_value(player, "position"))
    if position in DRAFTABLE_POSITIONS:
        return position

    for name in ("defaultPositionId", "default_position_id", "lineupSlotId", "lineup_slot_id"):
        position = _position_from_id(_object_value(player, name))
        if position in DRAFTABLE_POSITIONS:
            return position

    position = _position_from_slots(_object_value(player, "eligibleSlots"))
    if position in DRAFTABLE_POSITIONS:
        return position

    raw_player = _raw_player_payload(raw)
    for source in (raw_player, raw if isinstance(raw, dict) else {}):
        for name in ("defaultPositionId", "default_position_id", "lineupSlotId", "lineup_slot_id"):
            position = _position_from_id(source.get(name))
            if position in DRAFTABLE_POSITIONS:
                return position
        position = _position_from_slots(source.get("eligibleSlots"))
        if position in DRAFTABLE_POSITIONS:
            return position
    return ""


def _draft_rank_from_raw(raw: object) -> int:
    """Extract ESPN draft rank from raw player-card rank structures."""
    if not isinstance(raw, dict):
        return 0
    candidates: list[dict[str, Any]] = []
    candidates.append(raw)
    player_pool = raw.get("playerPoolEntry")
    if isinstance(player_pool, dict):
        candidates.append(player_pool)
        if isinstance(player_pool.get("player"), dict):
            candidates.append(player_pool["player"])
    player = raw.get("player")
    if isinstance(player, dict):
        candidates.append(player)

    for candidate in candidates:
        for name in ("rank", "overallRank", "draftRank", "draft_rank", "defaultDraftRank"):
            rank = _integer(candidate.get(name))
            if rank > 0:
                return rank

        rank_sets = (
            candidate.get("draftRanksByRankType")
            or candidate.get("draftRanksByRankTypeId")
            or candidate.get("draftRanks")
            or {}
        )
        if isinstance(rank_sets, dict):
            for key in ("PPR", "STANDARD", "HALF_PPR", "0", "1", "2"):
                rank_data = rank_sets.get(key)
                if isinstance(rank_data, dict):
                    rank = _integer(rank_data.get("rank") or rank_data.get("overallRank"))
                    if rank > 0:
                        return rank
            for rank_data in rank_sets.values():
                if isinstance(rank_data, dict):
                    rank = _integer(rank_data.get("rank") or rank_data.get("overallRank"))
                    if rank > 0:
                        return rank
    return 0


def _draft_rank(player: object, raw: object | None = None) -> int:
    """Extract draft rank, preferring the richer raw ESPN payload."""
    raw_rank = _draft_rank_from_raw(raw)
    if raw_rank > 0:
        return raw_rank
    for name in ("rank", "overallRank", "draftRank", "draft_rank", "defaultDraftRank"):
        rank = _integer(_object_value(player, name))
        if rank > 0:
            return rank
    return 0


def _pos_rank(player: object, raw: object | None = None) -> int:
    """Extract a player position rank from object or raw payload fields."""
    for name in ("posRank", "pos_rank", "positionalRanking", "positional_rank"):
        rank = _integer(_object_value(player, name))
        if rank > 0:
            return rank
    raw_player = _raw_player_payload(raw)
    candidates = [raw_player]
    if isinstance(raw, dict):
        candidates.append(raw)
        pool = raw.get("playerPoolEntry")
        if isinstance(pool, dict):
            candidates.append(pool)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for name in ("posRank", "pos_rank", "positionalRanking", "positional_rank"):
            rank = _integer(candidate.get(name))
            if rank > 0:
                return rank
    return 0


def _player_adp(player: object, raw: object | None = None) -> float:
    """Extract ESPN average draft position when available."""
    for name in ("averageDraftPosition", "average_draft_position", "adp"):
        value = _number(_object_value(player, name, None), -1.0)
        if value >= 0:
            return round(value, 2)
    raw_player = _raw_player_payload(raw)
    ownership = raw_player.get("ownership") if isinstance(raw_player, dict) else None
    if isinstance(ownership, dict):
        for name in ("averageDraftPosition", "average_draft_position", "adp"):
            value = _number(ownership.get(name), -1.0)
            if value >= 0:
                return round(value, 2)
    return -1.0


def _player_rank_sort_key(row: dict[str, Any]) -> tuple[bool, int, float, str]:
    """Build the stable display sort key for normalized players."""
    rank = _integer(row.get("espn_rank") or row.get("rank"))
    return (
        rank <= 0,
        rank if rank > 0 else 999999,
        -_number(row.get("projected_total_pts")),
        str(row.get("player_name", "")),
    )


def _fill_missing_position_ranks(player_rows: list[dict[str, Any]]) -> None:
    """Assign local position ranks when ESPN did not provide them."""
    by_position: dict[str, list[dict[str, Any]]] = {}
    for row in player_rows:
        by_position.setdefault(str(row.get("position") or ""), []).append(row)

    for rows in by_position.values():
        for index, row in enumerate(sorted(rows, key=_player_rank_sort_key), start=1):
            if _integer(row.get("pos_rank")) <= 0:
                row["pos_rank"] = index


def _assign_display_ranks(player_rows: list[dict[str, Any]]) -> None:
    """Replace sparse ESPN ranks with dense board ranks for display."""
    for index, row in enumerate(player_rows, start=1):
        espn_rank = _integer(row.get("espn_rank") or row.get("rank"))
        row["espn_rank"] = espn_rank
        row["rank"] = index


def _infer_bye_week(player: object, raw: object | None, week_start: int, week_end: int) -> int:
    """Infer a bye week from explicit fields or missing schedule weeks."""
    for name in ("byeWeek", "bye_week"):
        bye = _integer(_object_value(player, name))
        if bye > 0:
            return bye
    raw_player = _raw_player_payload(raw)
    for name in ("byeWeek", "bye_week"):
        bye = _integer(raw_player.get(name))
        if bye > 0:
            return bye

    schedule = _object_value(player, "schedule", None)
    if schedule is None and isinstance(raw_player.get("schedule"), dict):
        schedule = raw_player.get("schedule")
    if isinstance(schedule, dict) and schedule:
        game_weeks = {_integer(week) for week in schedule if _integer(week) > 0}
        for week in range(week_start, week_end + 1):
            if week not in game_weeks:
                return week
    return 0


def _normalize_player(
    player: object,
    week_start: int,
    week_end: int,
    year: int,
    raw: object | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], Counter[str]]:
    """Normalize one ESPN player and its weekly projection rows."""
    player_id = str(_object_value(player, "playerId", _object_value(player, "player_id", _object_value(player, "id", ""))))
    if not player_id:
        return None, [], Counter()

    position = _normalized_position(player, raw)
    if position not in DRAFTABLE_POSITIONS:
        return None, [], Counter()

    bye_week = _infer_bye_week(player, raw, week_start, week_end)
    raw_projection_counts, raw_projection_split_types = _raw_projection_stat_summary(raw, year, week_start, week_end)
    stats = _object_value(player, "stats", {}) or {}
    weekly_rows: list[dict[str, Any]] = []
    weekly_sources: list[str] = []
    for week in range(week_start, week_end + 1):
        stat = stats.get(week) if isinstance(stats, dict) else None
        if stat is None and isinstance(stats, dict):
            stat = stats.get(str(week))
        source = "missing"
        projected_points = 0.0
        if _stat_has_projected_points(stat):
            projected_points = _stat_projected_points(stat)
            source = "espn_weekly"
        else:
            raw_projected = _raw_projected_points(raw, week, year)
            if raw_projected is not None:
                projected_points = raw_projected
                source = "espn_raw_weekly"
        weekly_rows.append(
            {
                "player_id": player_id,
                "week": week,
                "projected_points": round(projected_points, 4),
            }
        )
        weekly_sources.append(source)

    projected_total = _number(_object_value(player, "projected_total_points"))
    if projected_total <= 0:
        season_stat = stats.get(0) if isinstance(stats, dict) else None
        if season_stat is None and isinstance(stats, dict):
            season_stat = stats.get("0")
        if _stat_has_projected_points(season_stat):
            projected_total = _stat_projected_points(season_stat)
    if projected_total <= 0:
        projected_total = _raw_projected_points(raw, 0, year) or 0.0

    weekly_source_counts = Counter(weekly_sources)
    weekly_sum = sum(row["projected_points"] for row in weekly_rows)
    if projected_total <= 0 and weekly_sum > 0:
        projected_total = weekly_sum
    if weekly_source_counts["espn_weekly"] + weekly_source_counts["espn_raw_weekly"] <= 0 and projected_total > 0 and weekly_rows:
        active_rows = [row for row in weekly_rows if row["week"] != bye_week]
        per_week = projected_total / len(active_rows or weekly_rows)
        for row in weekly_rows:
            row["projected_points"] = 0.0 if row["week"] == bye_week else round(per_week, 4)
        fallback_source = "season_total_bye_adjusted" if active_rows and len(active_rows) != len(weekly_rows) else "season_total_even_split"
        weekly_source_counts = Counter({fallback_source: len(weekly_rows)})
    weekly_source_counts.update({f"raw_{key}": value for key, value in raw_projection_counts.items()})
    weekly_source_counts.update({f"raw_projected_split_type_{key}": value for key, value in raw_projection_split_types.items()})

    espn_rank = _draft_rank(player, raw)
    row = {
        "player_id": player_id,
        "player_name": str(_object_value(player, "name", _object_value(player, "player_name", _object_value(player, "fullName", ""))) or player_id),
        "rank": espn_rank,
        "espn_rank": espn_rank,
        "position": position,
        "pro_team": str(_object_value(player, "proTeam", _object_value(player, "pro_team", "")) or ""),
        "pos_rank": _pos_rank(player, raw),
        "bye_week": bye_week,
        "injury_status": str(_object_value(player, "injuryStatus", _object_value(player, "injury_status", "")) or ""),
        "injured": bool(_object_value(player, "injured", False)),
        "active_status": str(_object_value(player, "active_status", _object_value(player, "activeStatus", "")) or ""),
        "percent_owned": round(_number(_object_value(player, "percent_owned", -1.0), -1.0), 2),
        "percent_started": round(_number(_object_value(player, "percent_started", -1.0), -1.0), 2),
        "adp": _player_adp(player, raw),
        "projected_total_pts": round(projected_total, 4),
        "projected_avg_pts": round(_number(_object_value(player, "projected_avg_points")), 4),
        "season_total_pts": round(projected_total, 4),
    }
    return row, weekly_rows, weekly_source_counts


def _chunked(values: list[int], size: int) -> Iterable[list[int]]:
    """Yield fixed-size batches for ESPN player-card requests."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _collect_player_ids(league: object) -> list[int]:
    """Collect player ids from league maps and roster objects."""
    ids: set[int] = set()
    player_map = _object_value(league, "player_map", {}) or {}
    if isinstance(player_map, dict):
        for key, value in player_map.items():
            key_id = _integer(key)
            value_id = _integer(value)
            if key_id != 0:
                ids.add(key_id)
            elif value_id != 0:
                ids.add(value_id)

    for team in _object_value(league, "teams", []) or []:
        for player in _object_value(team, "roster", []) or []:
            player_id = _object_value(player, "playerId")
            if player_id is not None:
                ids.add(_integer(player_id))
    return sorted(player_id for player_id in ids if player_id != 0)


def _players_from_response(response: object) -> list[object]:
    """Normalize espn_api player_info responses into a list."""
    if response is None:
        return []
    if isinstance(response, list):
        return response
    return [response]


def _pro_schedule(league: object) -> object | None:
    """Fetch the pro schedule when espn_api exposes the private helper."""
    if not hasattr(league, "_get_all_pro_schedule"):
        return None
    try:
        return league._get_all_pro_schedule()
    except Exception:  # noqa: BLE001
        return None


def _build_player_from_raw(raw: dict[str, Any], year: int, pro_schedule: object | None) -> object:
    """Build an espn_api Player object from raw data when possible."""
    try:
        from espn_api.football.player import Player
    except ImportError:
        return raw

    try:
        return Player(raw, year, pro_schedule)
    except Exception:  # noqa: BLE001
        return raw


def _fetch_weekly_projection_stats(
    league: object,
    player_ids: list[int],
    batch_size: int,
    *,
    year: int,
    week_start: int,
    week_end: int,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch supplemental weekly projection rows via kona_player_info."""
    espn_request = _object_value(league, "espn_request")
    if espn_request is None or not hasattr(espn_request, "league_get"):
        return {}

    weekly: dict[str, list[dict[str, Any]]] = {}
    weekly_batch_size = max(batch_size, 250)
    for week in range(week_start, week_end + 1):
        for batch in _chunked(player_ids, weekly_batch_size):
            params = {"view": "kona_player_info", "scoringPeriodId": week}
            filters = {"players": {"filterIds": {"value": batch}}}
            headers = {"x-fantasy-filter": json.dumps(filters)}
            try:
                data = espn_request.league_get(params=params, headers=headers)
            except Exception:  # noqa: BLE001
                continue
            for raw in data.get("players", []) if isinstance(data, dict) else []:
                player_id = _raw_player_id(raw)
                if not player_id:
                    continue
                for stat in _raw_player_stats(raw):
                    if _integer(stat.get("seasonId")) != year:
                        continue
                    if _integer(stat.get("scoringPeriodId"), -999) != week:
                        continue
                    if _integer(stat.get("statSourceId"), -1) != 1:
                        continue
                    if stat.get("appliedTotal") is None:
                        continue
                    weekly.setdefault(player_id, []).append(stat)
    return weekly


def _fetch_players(
    league: object,
    player_ids: list[int],
    batch_size: int,
    *,
    year: int,
    week_start: int,
    week_end: int,
) -> list[tuple[object, object | None]]:
    """Fetch player objects, preserving raw payloads when ESPN cards work."""
    players: list[tuple[object, object | None]] = []
    espn_request = _object_value(league, "espn_request")
    if espn_request is not None and hasattr(espn_request, "get_player_card"):
        pro_schedule = _pro_schedule(league)
        raw_failed = False
        additional_filters = [value.format(year=year) for value in PROJECTED_WEEKLY_STAT_FILTERS]
        weekly_stats = _fetch_weekly_projection_stats(
            league,
            player_ids,
            batch_size,
            year=year,
            week_start=week_start,
            week_end=week_end,
        )
        for batch in _chunked(player_ids, batch_size):
            try:
                try:
                    data = espn_request.get_player_card(
                        batch,
                        _integer(_object_value(league, "finalScoringPeriod"), week_end),
                        additional_filters=additional_filters,
                    )
                except TypeError:
                    data = espn_request.get_player_card(
                        batch,
                        _integer(_object_value(league, "finalScoringPeriod"), week_end),
                    )
            except Exception:  # noqa: BLE001
                raw_failed = True
                players = []
                break
            raw_players = data.get("players", []) if isinstance(data, dict) else []
            for raw in raw_players:
                _merge_raw_stats(raw, weekly_stats.get(_raw_player_id(raw), []))
                players.append((_build_player_from_raw(raw, year, pro_schedule), raw))
        if players and not raw_failed:
            return players

    for batch in _chunked(player_ids, batch_size):
        players.extend((player, None) for player in _players_from_response(league.player_info(playerId=batch)))
    return players


def _team_id(team: object, fallback: int) -> int:
    """Read a team id from a number, dict, or object."""
    if isinstance(team, int):
        return team
    return _integer(_object_value(team, "team_id", _object_value(team, "id", fallback)), fallback)


def _team_rows(league: object) -> list[dict[str, Any]]:
    """Normalize ESPN teams into the app's team row format."""
    rows: list[dict[str, Any]] = []
    for index, team in enumerate(_object_value(league, "teams", []) or [], start=1):
        rows.append(
            {
                "team_id": _team_id(team, index),
                "team_name": str(_object_value(team, "team_name", f"Team {index}") or f"Team {index}"),
                "team_abbrev": str(_object_value(team, "team_abbrev", "") or ""),
                "draft_projected_rank": _integer(_object_value(team, "draft_projected_rank")),
                "draft_slot": index,
            }
        )
    return rows


def _pick_round(pick: dict[str, Any]) -> int:
    """Read a round number from an ESPN draft pick dict."""
    return _integer(pick.get("roundId") or pick.get("round") or pick.get("round_num"))


def _pick_in_round(pick: dict[str, Any]) -> int:
    """Read a pick-in-round number from an ESPN draft pick dict."""
    return _integer(pick.get("roundPickNumber") or pick.get("round_pick") or pick.get("pick_in_round"))


def _pick_team_id(pick: dict[str, Any], *names: str) -> int:
    """Return the first positive team id found under the provided keys."""
    for name in names:
        team_id = _integer(pick.get(name))
        if team_id > 0:
            return team_id
    return 0


def _draft_data_candidates(league: object) -> list[dict[str, Any]]:
    """Collect possible raw draft payloads from league and request objects."""
    candidates: list[dict[str, Any]] = []
    for name in ("draft_detail", "draft_data", "raw_draft", "raw_league_data"):
        value = _object_value(league, name)
        if isinstance(value, dict):
            candidates.append(value)

    espn_request = _object_value(league, "espn_request")
    if espn_request is None:
        return candidates
    for method_name in ("get_league_draft", "get_league"):
        method = getattr(espn_request, method_name, None)
        if method is None:
            continue
        try:
            value = method()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, dict):
            candidates.append(value)
    return candidates


def _pick_order_from_data(data: dict[str, Any]) -> list[int]:
    """Extract first-round draft order team ids from a raw draft payload."""
    candidate_values = [
        data.get("pickOrder"),
        data.get("draftOrder"),
        data.get("draftDetail", {}).get("pickOrder") if isinstance(data.get("draftDetail"), dict) else None,
        data.get("draftDetail", {}).get("draftOrder") if isinstance(data.get("draftDetail"), dict) else None,
        data.get("settings", {}).get("draftSettings", {}).get("pickOrder")
        if isinstance(data.get("settings"), dict)
        else None,
    ]
    for candidate in candidate_values:
        if not isinstance(candidate, list):
            continue
        order: list[int] = []
        for item in candidate:
            team_id = _integer(item)
            if team_id <= 0 and isinstance(item, dict):
                team_id = _pick_team_id(item, "teamId", "id", "team_id")
            if team_id > 0 and team_id not in order:
                order.append(team_id)
        if order:
            return order
    return []


def _picks_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract raw pick rows from known ESPN draft payload shapes."""
    picks = data.get("draftDetail", {}).get("picks", []) if isinstance(data.get("draftDetail"), dict) else []
    if not isinstance(picks, list):
        picks = data.get("picks", [])
    return [pick for pick in picks if isinstance(pick, dict)]


def _draft_order_from_picks(picks: list[dict[str, Any]]) -> list[int]:
    """Infer draft order from the first round of raw pick rows."""
    valid = [
        pick for pick in picks
        if _pick_round(pick) > 0 and _pick_in_round(pick) > 0 and _pick_team_id(pick, "teamId", "team_id") > 0
    ]
    if not valid:
        return []
    first_round = min(_pick_round(pick) for pick in valid)
    order: list[int] = []
    for pick in sorted(valid, key=lambda item: (_pick_round(item), _pick_in_round(item))):
        if _pick_round(pick) != first_round:
            continue
        team_id = _pick_team_id(pick, "teamId", "team_id")
        if team_id not in order:
            order.append(team_id)
    return order


def _draft_slots_from_picks(picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw pick rows into current/original team slot rows."""
    slots: list[dict[str, Any]] = []
    for pick in sorted(picks, key=lambda item: (_pick_round(item), _pick_in_round(item))):
        round_num = _pick_round(pick)
        pick_in_round = _pick_in_round(pick)
        team_id = _pick_team_id(pick, "teamId", "team_id")
        if round_num <= 0 or pick_in_round <= 0 or team_id <= 0:
            continue
        slots.append(
            {
                "overall": _integer(pick.get("overallPickNumber")),
                "round": round_num,
                "pick_in_round": pick_in_round,
                "team_id": team_id,
                "original_team_id": _pick_team_id(
                    pick,
                    "originalTeamId",
                    "original_team_id",
                    "originalPickTeamId",
                    "owningTeamId",
                ),
            }
        )
    return slots


def _completed_draft_order(league: object) -> list[int]:
    """Infer draft order from completed espn_api draft pick objects."""
    draft = _object_value(league, "draft", []) or []
    picks = [
        pick for pick in draft
        if _integer(_object_value(pick, "round_num")) > 0 and _integer(_object_value(pick, "round_pick")) > 0
    ]
    if not picks:
        return []
    first_round = min(_integer(_object_value(pick, "round_num")) for pick in picks)
    order: list[int] = []
    for pick in sorted(picks, key=lambda item: (_integer(_object_value(item, "round_num")), _integer(_object_value(item, "round_pick")))):
        if _integer(_object_value(pick, "round_num")) != first_round:
            continue
        team_id = _team_id(_object_value(pick, "team"), 0)
        if team_id > 0 and team_id not in order:
            order.append(team_id)
    return order


def _ordered_teams_and_slots(league: object) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return teams in draft order plus normalized traded draft slots."""
    teams = _team_rows(league)
    if not teams:
        return [], []
    team_by_id = {row["team_id"]: row for row in teams}
    draft_order: list[int] = []
    raw_slots: list[dict[str, Any]] = []

    for data in _draft_data_candidates(league):
        draft_order = _pick_order_from_data(data)
        picks = _picks_from_data(data)
        if not draft_order:
            draft_order = _draft_order_from_picks(picks)
        if picks and not raw_slots:
            raw_slots = _draft_slots_from_picks(picks)
        if draft_order:
            break

    if not draft_order:
        draft_order = _completed_draft_order(league)

    if not draft_order:
        ranked = [row for row in teams if row["draft_projected_rank"] > 0]
        ranks = {row["draft_projected_rank"] for row in ranked}
        if len(ranked) == len(teams) and len(ranks) == len(teams):
            draft_order = [row["team_id"] for row in sorted(teams, key=lambda row: row["draft_projected_rank"])]

    if not draft_order:
        draft_order = [row["team_id"] for row in teams]

    ordered_ids: list[int] = []
    for team_id in draft_order:
        if team_id in team_by_id and team_id not in ordered_ids:
            ordered_ids.append(team_id)
    for row in teams:
        if row["team_id"] not in ordered_ids:
            ordered_ids.append(row["team_id"])

    ordered = [{**team_by_id[team_id], "draft_slot": index + 1} for index, team_id in enumerate(ordered_ids)]
    team_index_by_id = {row["team_id"]: index for index, row in enumerate(ordered)}
    draft_slots: list[dict[str, Any]] = []
    for raw_slot in raw_slots:
        current_team = team_index_by_id.get(raw_slot["team_id"])
        if current_team is None:
            continue
        round_num = _integer(raw_slot["round"])
        pick_in_round = _integer(raw_slot["pick_in_round"])
        original_team = team_index_by_id.get(_integer(raw_slot.get("original_team_id")))
        if original_team is None and 1 <= pick_in_round <= len(ordered):
            snake_order = list(range(len(ordered)))
            if round_num % 2 == 0:
                snake_order.reverse()
            original_team = snake_order[pick_in_round - 1]
        if original_team is None:
            original_team = current_team
        overall = _integer(raw_slot["overall"])
        if overall <= 0:
            overall = (round_num - 1) * len(ordered) + pick_in_round
        draft_slots.append(
            {
                "overall": overall,
                "round": round_num,
                "pick_in_round": pick_in_round,
                "original_team": original_team,
                "current_team": current_team,
            }
        )
    draft_slots.sort(key=lambda row: row["overall"])
    return ordered, draft_slots


def _schedule_week(matchup: dict[str, Any]) -> int:
    """Read the scoring week from a raw ESPN matchup row."""
    return _integer(
        matchup.get("matchupPeriodId")
        or matchup.get("matchup_period_id")
        or matchup.get("scoringPeriodId")
        or matchup.get("week")
    )


def _schedule_side_team_id(matchup: dict[str, Any], side: str) -> int:
    """Read the home or away team id from a raw matchup row."""
    data = matchup.get(side)
    if isinstance(data, dict):
        return _integer(data.get("teamId") or data.get("team_id") or data.get("id"))
    return _integer(data)


def _schedule_rows_from_matchup_api(league: object) -> list[dict[str, Any]]:
    """Fetch regular-season matchups from ESPN's matchup-score view."""
    espn_request = _object_value(league, "espn_request")
    if espn_request is None or not hasattr(espn_request, "league_get"):
        return []
    try:
        data = espn_request.league_get(params={"view": "mMatchupScore"})
    except Exception:  # noqa: BLE001
        return []
    raw_schedule = data.get("schedule", []) if isinstance(data, dict) else []
    rows: list[dict[str, Any]] = []
    for matchup in raw_schedule:
        if not isinstance(matchup, dict):
            continue
        week = _schedule_week(matchup)
        home_team_id = _schedule_side_team_id(matchup, "home")
        away_team_id = _schedule_side_team_id(matchup, "away")
        if week <= 0 or home_team_id <= 0 or away_team_id <= 0 or home_team_id == away_team_id:
            continue
        rows.append(
            {
                "week": week,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "source": "espn_matchup_score",
            }
        )
    return rows


def _schedule_rows_from_teams(league: object) -> list[dict[str, Any]]:
    """Build matchup rows from espn_api team schedule attributes."""
    rows: list[dict[str, Any]] = []
    for team in _object_value(league, "teams", []) or []:
        team_id = _team_id(team, 0)
        if team_id <= 0:
            continue
        for index, opponent in enumerate(_object_value(team, "schedule", []) or [], start=1):
            opponent_id = _team_id(opponent, 0)
            if opponent_id <= 0 or opponent_id == team_id:
                continue
            rows.append(
                {
                    "week": index,
                    "home_team_id": team_id,
                    "away_team_id": opponent_id,
                    "source": "espn_team_schedule",
                }
            )
    return rows


def _league_schedule(
    league: object,
    teams: list[dict[str, Any]],
    *,
    week_start: int,
    week_end: int,
) -> list[dict[str, Any]]:
    """Normalize league schedule rows into zero-based team-index matchups."""
    team_index_by_id = {row["team_id"]: index for index, row in enumerate(teams)}
    team_name_by_id = {row["team_id"]: row["team_name"] for row in teams}
    raw_rows = _schedule_rows_from_matchup_api(league) or _schedule_rows_from_teams(league)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for raw in raw_rows:
        week = _integer(raw.get("week"))
        if week < week_start or week > week_end:
            continue
        home_team_id = _integer(raw.get("home_team_id"))
        away_team_id = _integer(raw.get("away_team_id"))
        home_index = team_index_by_id.get(home_team_id)
        away_index = team_index_by_id.get(away_team_id)
        if home_index is None or away_index is None or home_index == away_index:
            continue
        key = (week, min(home_index, away_index), max(home_index, away_index))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"{week}:{home_index}:{away_index}",
                "week": week,
                "home_team_index": home_index,
                "away_team_index": away_index,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team": team_name_by_id.get(home_team_id, f"Team {home_index + 1}"),
                "away_team": team_name_by_id.get(away_team_id, f"Team {away_index + 1}"),
                "source": str(raw.get("source") or "espn_schedule"),
            }
        )
    return sorted(rows, key=lambda row: (row["week"], row["home_team_index"], row["away_team_index"]))


def _league_settings(league: object, team_count: int) -> dict[str, Any]:
    """Extract the league settings the browser needs after sync."""
    settings = _object_value(league, "settings")
    if settings is None:
        return {"team_count": team_count}
    return {
        "name": str(_object_value(settings, "name", "")),
        "team_count": _integer(_object_value(settings, "team_count"), team_count),
        "reg_season_count": _integer(_object_value(settings, "reg_season_count"), DEFAULT_WEEK_END),
        "playoff_team_count": _integer(_object_value(settings, "playoff_team_count")),
        "keeper_count": _integer(_object_value(settings, "keeper_count")),
        "scoring_format": _object_value(settings, "scoring_format", []),
        "position_slot_counts": _object_value(settings, "position_slot_counts", {}),
    }


def normalize_espn_league(
    league: object,
    *,
    year: int,
    week_start: int = DEFAULT_WEEK_START,
    week_end: int = DEFAULT_WEEK_END,
    batch_size: int = 50,
    player_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Normalize an espn_api League object into the browser cache payload."""
    ids = player_ids if player_ids is not None else _collect_player_ids(league)
    players = _fetch_players(
        league,
        ids,
        batch_size,
        year=year,
        week_start=week_start,
        week_end=week_end,
    )
    player_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    weekly_source_counts: Counter[str] = Counter()

    seen: set[str] = set()
    for player, raw in players:
        row, rows, sources = _normalize_player(player, week_start, week_end, year, raw)
        if row is None or row["player_id"] in seen:
            continue
        seen.add(row["player_id"])
        player_rows.append(row)
        weekly_rows.extend(rows)
        weekly_source_counts.update(sources)

    _fill_missing_position_ranks(player_rows)
    player_rows.sort(key=_player_rank_sort_key)
    _assign_display_ranks(player_rows)
    teams, draft_slots = _ordered_teams_and_slots(league)
    league_schedule = _league_schedule(league, teams, week_start=week_start, week_end=week_end)
    names = [team["team_name"] for team in teams]
    team_count = len(names) or _integer(_object_value(_object_value(league, "settings"), "team_count"), 0)
    source_counts = {
        key: value
        for key, value in weekly_source_counts.items()
        if not key.startswith("raw_")
    }
    raw_projection_stats = {
        key.removeprefix("raw_"): value
        for key, value in weekly_source_counts.items()
        if key.startswith("raw_") and not key.startswith("raw_projected_split_type_")
    }
    raw_projected_split_types = {
        key.removeprefix("raw_projected_split_type_"): value
        for key, value in weekly_source_counts.items()
        if key.startswith("raw_projected_split_type_")
    }
    return {
        "players": player_rows,
        "weekly_projections": weekly_rows,
        "league_settings": _league_settings(league, team_count),
        "teams": teams,
        "team_names": names,
        "draft_slots": draft_slots,
        "league_schedule": league_schedule,
        "projection_meta": {
            "weekly_projection_sources": dict(sorted(source_counts.items())),
            "has_espn_weekly_projections": bool(
                weekly_source_counts["espn_weekly"] + weekly_source_counts["espn_raw_weekly"]
            ),
            "raw_projection_stats": dict(sorted(raw_projection_stats.items())),
            "raw_projected_split_types": dict(sorted(raw_projected_split_types.items())),
            "espn_player_card_max_scoring_period": _integer(_object_value(league, "finalScoringPeriod"), week_end),
            "espn_player_card_additional_filters": [
                value.format(year=year) for value in PROJECTED_WEEKLY_STAT_FILTERS
            ],
            "week_start": week_start,
            "week_end": week_end,
        },
        "year": year,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def sync_projection_payload(
    payload: dict[str, Any],
    league_factory: Callable[..., object] | None = None,
) -> dict[str, Any]:
    """Create an ESPN league client and return normalized projection data."""
    config = EspnSourceConfig.from_payload(payload)
    if league_factory is None:
        try:
            from espn_api.football import League
        except ImportError as exc:
            raise RuntimeError(
                "espn_api is not installed. Run `python -m pip install -e .` "
                "before syncing ESPN projections."
            ) from exc
        league_factory = League

    league = league_factory(
        league_id=config.league_id,
        year=config.year,
        espn_s2=config.espn_s2,
        swid=config.swid,
    )
    return normalize_espn_league(
        league,
        year=config.year,
        week_start=config.week_start,
        week_end=config.week_end,
        batch_size=config.batch_size,
    )
