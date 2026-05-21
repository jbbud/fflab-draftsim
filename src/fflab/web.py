from __future__ import annotations

import argparse
import json
import threading
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .ai import BestAvailablePolicy, DraftPolicy, get_policy
from .config import LeagueConfig
from .data import load_scored_season
from .draft import DraftPick, DraftState
from .optimizer import LineupResult, get_optimal_weekly_lineup
from .scoring import ScoredData
from .simulation import SimulationResult, simulate_season


DEFAULT_GUI_CONFIG = {
    "num_teams": 10,
    "team_names": [
        "You",
        "Alpha Bot",
        "Beta Bot",
        "Gamma Bot",
        "Delta Bot",
        "Epsilon Bot",
        "Zeta Bot",
        "Eta Bot",
        "Theta Bot",
        "Iota Bot",
    ],
    "roster_settings": {
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 1,
        "K": 1,
        "DEF": 1,
        "BENCH": 6,
    },
    "ai_policies": ["best_available", "scarcity", "balanced"],
    "draft_objective": "roto",
}


@dataclass
class DraftSession:
    id: str
    season: int
    source: str
    config: LeagueConfig
    scored: ScoredData
    state: DraftState
    picks: list[dict[str, Any]] = field(default_factory=list)
    result: SimulationResult | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


SESSIONS: dict[str, DraftSession] = {}
SESSIONS_LOCK = threading.RLock()


def is_human_team(team_name: str) -> bool:
    return "bot" not in team_name.lower()


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def _player_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "player_id": str(row.player_id),
                "player_name": str(row.player_name),
                "position": str(row.position),
                "season_total_pts": round(float(row.season_total_pts), 2),
            }
        )
    return rows


def build_demo_scored_data(season: int, config: LeagueConfig) -> ScoredData:
    position_counts = {
        "QB": max(24, config.num_teams * 3),
        "RB": max(56, config.num_teams * 7),
        "WR": max(64, config.num_teams * 8),
        "TE": max(28, config.num_teams * 4),
        "K": max(24, config.num_teams * 3),
        "DEF": 32,
    }
    base_by_position = {
        "QB": 22.0,
        "RB": 15.0,
        "WR": 14.5,
        "TE": 10.5,
        "K": 8.0,
        "DEF": 8.5,
    }
    rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    for position, count in position_counts.items():
        for index in range(1, count + 1):
            player_id = f"{position}_{index:03d}" if position != "DEF" else f"DEF_T{index:02d}"
            player_name = (
                f"Team {index:02d} DEF"
                if position == "DEF"
                else f"{position} Demo {index:02d}"
            )
            position_decay = index * {
                "QB": 0.42,
                "RB": 0.28,
                "WR": 0.24,
                "TE": 0.22,
                "K": 0.12,
                "DEF": 0.15,
            }[position]
            base = max(base_by_position[position] - position_decay, 1.0)
            season_total = 0.0
            for week in range(1, 18):
                wave = ((index * 7 + week * 5) % 11) - 5
                spike = 6.0 if (index + week) % 13 == 0 and position in {"RB", "WR"} else 0.0
                points = max(base + wave * 0.75 + spike, 0.0)
                season_total += points
                weekly_rows.append(
                    {
                        "player_id": player_id,
                        "week": week,
                        "points_scored": round(points, 2),
                    }
                )
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "season": season,
                    "season_total_pts": round(season_total, 2),
                }
            )
    players = pd.DataFrame(rows).sort_values(
        "season_total_pts", ascending=False
    ).reset_index(drop=True)
    weekly_scores = pd.DataFrame(weekly_rows)
    return ScoredData(players=players, weekly_scores=weekly_scores)


def _load_scored_from_payload(payload: dict[str, Any], config: LeagueConfig) -> tuple[int, str, ScoredData]:
    season = int(payload.get("season") or 2025)
    source = str(payload.get("source", "demo"))
    if source == "live":
        scored = load_scored_season(season, config)
    else:
        source = "demo"
        scored = build_demo_scored_data(season, config)
    if scored.players.empty:
        raise ValueError("No draftable players were loaded.")
    return season, source, scored


def _policy_for_team(session: DraftSession, team_index: int) -> DraftPolicy:
    policies = session.config.ai_policies
    if not policies:
        return BestAvailablePolicy()
    policy_name = policies[team_index % len(policies)]
    return get_policy(policy_name)


def _pick_row(pick: DraftPick, state: DraftState, policy_name: str, human: bool) -> dict[str, Any]:
    return {
        "overall": pick.overall,
        "round": pick.round_number,
        "pick": pick.pick_in_round,
        "team_index": pick.team_index,
        "team": state.team_name(pick.team_index),
        "player_id": pick.player_id,
        "player": pick.player_name,
        "position": pick.position,
        "policy": policy_name,
        "human": human,
    }


def _finish_session_if_needed(session: DraftSession) -> None:
    if session.state.is_complete and session.result is None:
        session.result = simulate_season(
            state=session.state,
            players=session.scored.players,
            weekly_scores=session.scored.weekly_scores,
            config=session.config,
        )


def _auto_advance_bots(session: DraftSession) -> None:
    while not session.state.is_complete:
        team_index = session.state.team_on_clock
        team_name = session.state.team_name(team_index)
        if is_human_team(team_name):
            break
        policy = _policy_for_team(session, team_index)
        player_id = policy.choose_pick(session.state, team_index)
        pick = session.state.draft_player(team_index, player_id)
        session.picks.append(_pick_row(pick, session.state, policy.name, human=False))
    _finish_session_if_needed(session)


def create_draft_session(
    scored: ScoredData,
    config: LeagueConfig,
    season: int = 2019,
    source: str = "demo",
) -> DraftSession:
    session = DraftSession(
        id=uuid.uuid4().hex,
        season=season,
        source=source,
        config=config,
        scored=scored,
        state=DraftState(
            players=scored.players,
            config=config,
            weekly_scores=scored.weekly_scores,
        ),
    )
    with session.lock:
        _auto_advance_bots(session)
    with SESSIONS_LOCK:
        SESSIONS[session.id] = session
    return session


def start_draft_session(payload: dict[str, Any]) -> dict[str, Any]:
    config_text = payload.get("configText") or "{}"
    config_payload = json.loads(config_text)
    config = LeagueConfig.model_validate(config_payload)
    season, source, scored = _load_scored_from_payload(payload, config)
    session = create_draft_session(scored=scored, config=config, season=season, source=source)
    return session_state_payload(session.id)


def get_session(session_id: str) -> DraftSession:
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        raise KeyError("Unknown draft session.")
    return session


def make_human_pick(session_id: str, player_id: str) -> dict[str, Any]:
    session = get_session(session_id)
    with session.lock:
        if session.state.is_complete:
            raise ValueError("Draft is already complete.")
        team_index = session.state.team_on_clock
        team_name = session.state.team_name(team_index)
        if not is_human_team(team_name):
            raise ValueError(f"{team_name} is a Bot team and cannot receive a manual pick.")
        if not session.state.can_add_player(team_index, player_id):
            raise ValueError("That player is unavailable or the roster is full.")
        pick = session.state.draft_player(team_index, player_id)
        session.picks.append(_pick_row(pick, session.state, "human", human=True))
        _auto_advance_bots(session)
    return session_state_payload(session.id)


def run_auto_draft(scored: ScoredData, config: LeagueConfig) -> tuple[DraftState, list[dict[str, Any]]]:
    state = DraftState(
        players=scored.players,
        config=config,
        weekly_scores=scored.weekly_scores,
    )
    policies = [get_policy(name) for name in config.ai_policies]
    pick_rows: list[dict[str, Any]] = []
    while not state.is_complete:
        team_index = state.team_on_clock
        team_name = state.team_name(team_index)
        if is_human_team(team_name):
            player_id = BestAvailablePolicy().choose_pick(state, team_index)
            policy_name = "best_available"
            human = True
        else:
            policy = policies[team_index % len(policies)]
            player_id = policy.choose_pick(state, team_index)
            policy_name = policy.name
            human = False
        pick = state.draft_player(team_index, player_id)
        pick_rows.append(_pick_row(pick, state, policy_name, human=human))
    return state, pick_rows


def run_gui_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    config_text = payload.get("configText") or "{}"
    config_payload = json.loads(config_text)
    config = LeagueConfig.model_validate(config_payload)
    season, source, scored = _load_scored_from_payload(payload, config)

    state, picks = run_auto_draft(scored, config)
    result = simulate_season(state, scored.players, scored.weekly_scores, config)
    return {
        "source": source,
        "season": season,
        "draftPicks": picks,
        "roto": _records(result.roto),
        "standings": _records(result.standings),
        "weeklyTeamScores": _records(result.weekly_team_scores),
    }


def _roster_payload(session: DraftSession, team_index: int) -> dict[str, Any]:
    players_by_id = session.scored.players.set_index("player_id", drop=False)
    roster_rows: list[dict[str, Any]] = []
    for player_id in session.state.roster(team_index):
        row = players_by_id.loc[player_id]
        roster_rows.append(
            {
                "player_id": str(player_id),
                "player_name": str(row["player_name"]),
                "position": str(row["position"]),
                "season_total_pts": round(float(row["season_total_pts"]), 2),
            }
        )
    team_name = session.state.team_name(team_index)
    return {
        "team_index": team_index,
        "team": team_name,
        "human": is_human_team(team_name),
        "roster": roster_rows,
        "needs": session.state.roster_needs(team_index),
        "roster_size": session.state.roster_size(team_index),
    }


def _current_pick_payload(session: DraftSession) -> dict[str, Any] | None:
    if session.state.is_complete:
        return None
    team_index = session.state.team_on_clock
    team_name = session.state.team_name(team_index)
    return {
        "overall": session.state.pick_index + 1,
        "round": session.state.current_round,
        "pick": session.state.pick_in_round,
        "team_index": team_index,
        "team": team_name,
        "human": is_human_team(team_name),
    }


def _result_payload(session: DraftSession) -> dict[str, Any] | None:
    if session.result is None:
        return None
    return {
        "roto": _records(session.result.roto),
        "standings": _records(session.result.standings),
        "weeklyTeamScores": _records(session.result.weekly_team_scores),
    }


def session_state_payload(session_id: str) -> dict[str, Any]:
    session = get_session(session_id)
    with session.lock:
        available = session.state.available_players_df()
        weeks = sorted(
            int(week) for week in session.scored.weekly_scores["week"].dropna().unique()
        )
        teams = [
            _roster_payload(session, team_index)
            for team_index in range(session.config.num_teams)
        ]
        return {
            "id": session.id,
            "source": session.source,
            "season": session.season,
            "complete": session.state.is_complete,
            "currentPick": _current_pick_payload(session),
            "availablePlayers": _player_rows(available),
            "teams": teams,
            "picks": list(session.picks),
            "weeks": weeks,
            "results": _result_payload(session),
        }


def _lineup_rows(lineup: LineupResult) -> dict[str, list[dict[str, Any]]]:
    def row(slot: Any) -> dict[str, Any]:
        return {
            "slot": slot.slot,
            "player_id": slot.player_id,
            "player_name": slot.player_name,
            "position": slot.position,
            "points": round(float(slot.points), 2),
        }

    return {
        "starters": [row(slot) for slot in lineup.starters],
        "bench": [row(slot) for slot in lineup.bench],
    }


def lineup_payload(session_id: str, team_index: int, week: int) -> dict[str, Any]:
    session = get_session(session_id)
    with session.lock:
        if not session.state.is_complete:
            raise ValueError("Lineup audit is available after the draft completes.")
        if team_index < 0 or team_index >= session.config.num_teams:
            raise ValueError("Unknown team.")
        weeks = set(int(value) for value in session.scored.weekly_scores["week"].dropna().unique())
        if week not in weeks:
            raise ValueError("Unknown week.")
        lineup = get_optimal_weekly_lineup(
            roster=session.state.roster(team_index),
            week=week,
            players=session.scored.players,
            weekly_scores=session.scored.weekly_scores,
            roster_settings=session.config.roster_settings,
        )
        rows = _lineup_rows(lineup)
        return {
            "session_id": session.id,
            "team_index": team_index,
            "team": session.state.team_name(team_index),
            "week": week,
            "total_points": round(float(lineup.total_points), 2),
            "starters": rows["starters"],
            "bench": rows["bench"],
        }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fflab Draft Room</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172018;
      --muted: #637066;
      --line: #d5ded7;
      --panel: #ffffff;
      --soft: #edf4ee;
      --accent: #25614e;
      --accent-strong: #173f34;
      --gold: #b87922;
      --bad: #9d2d2d;
      --blue: #315f8f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(255,255,255,.10) 49%, rgba(255,255,255,.20) 50%, rgba(255,255,255,.10) 51%),
        repeating-linear-gradient(90deg, #e8f0e7 0, #e8f0e7 94px, #dfe9df 94px, #dfe9df 100px);
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.95);
      position: sticky;
      top: 0;
      z-index: 4;
    }
    h1 { margin: 0; font-size: 22px; line-height: 1.1; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 15px; line-height: 1.2; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 13px; line-height: 1.2; color: var(--muted); letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(330px, 430px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1520px;
      margin: 0 auto;
    }
    section, .band {
      background: rgba(255,255,255,.96);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .controls { padding: 16px; align-self: start; position: sticky; top: 78px; }
    .workspace { display: grid; gap: 18px; background: transparent; border: 0; }
    .band { padding: 16px; }
    label {
      display: block;
      margin: 12px 0 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    input, select, textarea, button {
      width: 100%;
      font: inherit;
      border-radius: 6px;
    }
    input, select, textarea {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 10px;
    }
    textarea {
      min-height: 300px;
      resize: vertical;
      font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre;
    }
    button {
      border: 0;
      background: var(--accent);
      color: white;
      padding: 10px 12px;
      font-weight: 750;
      cursor: pointer;
      white-space: nowrap;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: .45; cursor: not-allowed; }
    button.secondary {
      margin-top: 8px;
      background: var(--soft);
      color: var(--accent-strong);
      border: 1px solid var(--line);
    }
    button.small { padding: 7px 9px; font-size: 12px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .status { min-height: 22px; color: var(--muted); font-size: 14px; text-align: right; }
    .status.error { color: var(--bad); }
    .clock {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 14px;
      align-items: center;
      border-left: 5px solid var(--accent);
    }
    .clock.complete { border-left-color: var(--gold); }
    .clock strong { display: block; font-size: 22px; line-height: 1.1; }
    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 28px;
      padding: 4px 9px;
      border-radius: 999px;
      background: var(--soft);
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 750;
    }
    .pill.bot { background: #eef3fb; color: var(--blue); }
    .tools {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 120px;
      gap: 10px;
      margin-bottom: 10px;
    }
    .table-wrap {
      max-height: 460px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    table { width: 100%; border-collapse: collapse; min-width: 560px; font-size: 13px; }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f7faf7;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    td.number, th.number { text-align: right; }
    .split { display: grid; grid-template-columns: minmax(0, 1fr) minmax(290px, 360px); gap: 18px; }
    .tables { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .metric-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfdfb; }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .metric strong { display: block; font-size: 20px; line-height: 1.1; }
    .roster-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; }
    .team-box { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfdfb; min-width: 0; }
    .team-box.active { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
    .team-box ul { margin: 0; padding-left: 18px; color: var(--muted); font-size: 12px; }
    .team-box li { margin: 3px 0; }
    .hidden { display: none !important; }
    @media (max-width: 1100px) {
      main, .split, .tables, .metric-row { grid-template-columns: 1fr; }
      .controls { position: static; }
      header { align-items: flex-start; flex-direction: column; }
      .status { text-align: left; }
    }
  </style>
</head>
<body>
  <header>
    <h1>fflab Draft Room</h1>
    <div class="status" id="status">Ready</div>
  </header>
  <main>
    <section class="controls">
      <h2>Setup</h2>
      <div class="row">
        <div>
          <label for="season">Season</label>
          <input id="season" type="number" min="1999" max="2026" value="2019">
        </div>
        <div>
          <label for="source">Data</label>
          <select id="source">
            <option value="demo">Demo fixture</option>
            <option value="live">Live nflreadpy</option>
          </select>
        </div>
      </div>
      <label for="config">League JSON</label>
      <textarea id="config" spellcheck="false"></textarea>
      <p style="color: var(--muted); font-size: 12px; line-height: 1.35;">
        Bot policies can be <code>best_available</code>, <code>scarcity</code>,
        <code>balanced</code>, <code>trained:path/to/policy.json</code>, or
        <code>neural:path/to/policy.pt</code>.
      </p>
      <button id="start" type="button">Start Draft Room</button>
      <button id="reset" class="secondary" type="button">Reset Config</button>
    </section>
    <section class="workspace">
      <div class="band clock" id="clock">
        <div>
          <h2>On The Clock</h2>
          <strong id="clockTeam">No draft started</strong>
          <div id="clockMeta">Configure the league and start a draft room.</div>
        </div>
        <div class="pill" id="clockType">Idle</div>
      </div>

      <div class="split">
        <div class="band">
          <h2>Available Players</h2>
          <div class="tools">
            <input id="search" placeholder="Search player">
            <select id="positionFilter">
              <option value="">All positions</option>
              <option value="QB">QB</option>
              <option value="RB">RB</option>
              <option value="WR">WR</option>
              <option value="TE">TE</option>
              <option value="K">K</option>
              <option value="DEF">DEF</option>
            </select>
          </div>
          <div class="table-wrap"><table id="available"></table></div>
        </div>
        <div class="band">
          <h2>Current Team</h2>
          <div id="currentRoster"></div>
        </div>
      </div>

      <div class="band">
        <h2>Draft Log</h2>
        <div class="table-wrap"><table id="draftLog"></table></div>
      </div>

      <div class="band">
        <h2>All Rosters</h2>
        <div class="roster-grid" id="allRosters"></div>
      </div>

      <div class="band hidden" id="audit">
        <h2>Weekly Lineup Audit</h2>
        <div class="row">
          <div>
            <label for="auditTeam">Team</label>
            <select id="auditTeam"></select>
          </div>
          <div>
            <label for="auditWeek">Week</label>
            <select id="auditWeek"></select>
          </div>
        </div>
        <div class="metric-row" style="margin-top: 12px;">
          <div class="metric"><span>Team</span><strong id="auditTeamName">-</strong></div>
          <div class="metric"><span>Week</span><strong id="auditWeekLabel">-</strong></div>
          <div class="metric"><span>Optimal Points</span><strong id="auditTotal">-</strong></div>
          <div class="metric"><span>Rostered Players</span><strong id="auditCount">-</strong></div>
        </div>
        <div class="tables" style="margin-top: 14px;">
          <div>
            <h3>Starters</h3>
            <div class="table-wrap"><table id="starters"></table></div>
          </div>
          <div>
            <h3>Bench</h3>
            <div class="table-wrap"><table id="bench"></table></div>
          </div>
        </div>
      </div>

      <div class="band hidden" id="results">
        <h2>Season Results</h2>
        <div class="metric-row">
          <div class="metric"><span>Champion</span><strong id="champion">-</strong></div>
          <div class="metric"><span>Top Points</span><strong id="points">-</strong></div>
          <div class="metric"><span>Draft Picks</span><strong id="pickCount">-</strong></div>
          <div class="metric"><span>Weeks</span><strong id="weeks">-</strong></div>
        </div>
        <div class="tables" style="margin-top: 14px;">
          <div>
            <h3>Roto Leaderboard</h3>
            <div class="table-wrap"><table id="roto"></table></div>
          </div>
          <div>
            <h3>Head-to-Head Standings</h3>
            <div class="table-wrap"><table id="standings"></table></div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const defaultConfig = __DEFAULT_CONFIG__;
    let state = null;
    let sessionId = null;
    const status = document.getElementById("status");
    const config = document.getElementById("config");
    const start = document.getElementById("start");
    const reset = document.getElementById("reset");
    const search = document.getElementById("search");
    const positionFilter = document.getElementById("positionFilter");

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    function resetConfig() {
      config.value = JSON.stringify(defaultConfig, null, 2);
    }

    function setStatus(message, isError = false) {
      status.textContent = message;
      status.className = isError ? "status error" : "status";
    }

    function renderTable(id, rows, columns) {
      const table = document.getElementById(id);
      if (!rows || rows.length === 0) {
        table.innerHTML = "<tbody><tr><td>No rows yet</td></tr></tbody>";
        return;
      }
      const head = "<thead><tr>" + columns.map(col => `<th class="${col.number ? "number" : ""}">${escapeHtml(col.label)}</th>`).join("") + "</tr></thead>";
      const body = "<tbody>" + rows.map(row => {
        return "<tr>" + columns.map(col => {
          if (col.render) return col.render(row);
          const raw = row[col.key];
          const value = typeof raw === "number" ? (col.digits == null ? raw : raw.toFixed(col.digits)) : (raw ?? "");
          return `<td class="${col.number ? "number" : ""}">${escapeHtml(value)}</td>`;
        }).join("") + "</tr>";
      }).join("") + "</tbody>";
      table.innerHTML = head + body;
    }

    async function postJson(url, body) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Request failed");
      return payload;
    }

    async function startDraft() {
      start.disabled = true;
      setStatus("Loading data and creating draft room...");
      try {
        JSON.parse(config.value);
        state = await postJson("/api/draft/start", {
          season: document.getElementById("season").value,
          source: document.getElementById("source").value,
          configText: config.value
        });
        sessionId = state.id;
        setStatus(`Draft room ready for ${state.season} using ${state.source} data`);
        renderState();
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        start.disabled = false;
      }
    }

    async function makePick(playerId) {
      if (!sessionId) return;
      setStatus("Submitting pick...");
      try {
        state = await postJson("/api/draft/pick", { id: sessionId, player_id: playerId });
        setStatus(state.complete ? "Draft complete" : "Pick submitted");
        renderState();
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    function renderState() {
      renderClock();
      renderAvailable();
      renderCurrentRoster();
      renderDraftLog();
      renderAllRosters();
      renderResults();
    }

    function renderClock() {
      const clock = document.getElementById("clock");
      const clockTeam = document.getElementById("clockTeam");
      const clockMeta = document.getElementById("clockMeta");
      const clockType = document.getElementById("clockType");
      if (!state) {
        clock.classList.remove("complete");
        clockTeam.textContent = "No draft started";
        clockMeta.textContent = "Configure the league and start a draft room.";
        clockType.textContent = "Idle";
        clockType.className = "pill";
        return;
      }
      if (state.complete) {
        clock.classList.add("complete");
        clockTeam.textContent = "Draft complete";
        clockMeta.textContent = "Review season results or inspect any weekly lineup.";
        clockType.textContent = "Complete";
        clockType.className = "pill";
        return;
      }
      const pick = state.currentPick;
      clock.classList.remove("complete");
      clockTeam.textContent = pick.team;
      clockMeta.textContent = `Round ${pick.round}, pick ${pick.pick}, overall ${pick.overall}`;
      clockType.textContent = pick.human ? "Human pick" : "Bot pick";
      clockType.className = pick.human ? "pill" : "pill bot";
    }

    function renderAvailable() {
      const query = search.value.trim().toLowerCase();
      const position = positionFilter.value;
      const canDraft = state && state.currentPick && state.currentPick.human && !state.complete;
      const rows = (state?.availablePlayers || []).filter(player => {
        const matchesText = !query || player.player_name.toLowerCase().includes(query);
        const matchesPosition = !position || player.position === position;
        return matchesText && matchesPosition;
      }).slice(0, 250);
      renderTable("available", rows, [
        { key: "player_name", label: "Player" },
        { key: "position", label: "Pos" },
        { key: "season_total_pts", label: "Season Pts", number: true, digits: 2 },
        {
          label: "Pick",
          render: row => `<td><button class="small" ${canDraft ? "" : "disabled"} data-player-id="${escapeHtml(row.player_id)}">Draft</button></td>`
        }
      ]);
      document.querySelectorAll("[data-player-id]").forEach(button => {
        button.addEventListener("click", () => makePick(button.getAttribute("data-player-id")));
      });
    }

    function renderCurrentRoster() {
      const target = document.getElementById("currentRoster");
      if (!state) {
        target.innerHTML = "<p>No draft started.</p>";
        return;
      }
      const index = state.currentPick ? state.currentPick.team_index : 0;
      const team = state.teams[index] || state.teams[0];
      const needs = Object.entries(team.needs || {}).filter(([, value]) => value > 0)
        .map(([key, value]) => `${escapeHtml(key)} ${escapeHtml(value)}`).join(", ") || "Roster filled";
      const roster = team.roster.length
        ? `<ul>${team.roster.map(player => `<li>${escapeHtml(player.player_name)} (${escapeHtml(player.position)})</li>`).join("")}</ul>`
        : "<p>No players drafted yet.</p>";
      target.innerHTML = `<h3>${escapeHtml(team.team)}</h3><p>${escapeHtml(needs)}</p>${roster}`;
    }

    function renderDraftLog() {
      renderTable("draftLog", (state?.picks || []).slice().reverse(), [
        { key: "overall", label: "#", number: true },
        { key: "round", label: "Rd", number: true },
        { key: "pick", label: "Pick", number: true },
        { key: "team", label: "Team" },
        { key: "player", label: "Player" },
        { key: "position", label: "Pos" },
        { key: "policy", label: "Mode" }
      ]);
    }

    function renderAllRosters() {
      const target = document.getElementById("allRosters");
      if (!state) {
        target.innerHTML = "";
        return;
      }
      const active = state.currentPick ? state.currentPick.team_index : -1;
      target.innerHTML = state.teams.map(team => {
        const roster = team.roster.length
          ? team.roster.map(player => `<li>${escapeHtml(player.player_name)} (${escapeHtml(player.position)})</li>`).join("")
          : "<li>No players</li>";
        return `<div class="team-box ${team.team_index === active ? "active" : ""}">
          <h3>${escapeHtml(team.team)} ${team.human ? "" : "(Bot)"}</h3>
          <ul>${roster}</ul>
        </div>`;
      }).join("");
    }

    function renderResults() {
      const results = document.getElementById("results");
      const audit = document.getElementById("audit");
      if (!state?.complete || !state.results) {
        results.classList.add("hidden");
        audit.classList.add("hidden");
        return;
      }
      results.classList.remove("hidden");
      audit.classList.remove("hidden");
      const leader = state.results.roto[0] || {};
      document.getElementById("champion").textContent = leader.team || "-";
      document.getElementById("points").textContent = leader.total_points == null ? "-" : Number(leader.total_points).toFixed(1);
      document.getElementById("pickCount").textContent = String((state.picks || []).length);
      document.getElementById("weeks").textContent = String((state.weeks || []).length || "-");
      renderTable("roto", state.results.roto, [
        { key: "rank", label: "Rank", number: true },
        { key: "team", label: "Team" },
        { key: "total_points", label: "Total Pts", number: true, digits: 2 }
      ]);
      renderTable("standings", state.results.standings, [
        { key: "team", label: "Team" },
        { key: "wins", label: "W", number: true },
        { key: "losses", label: "L", number: true },
        { key: "ties", label: "T", number: true },
        { key: "win_pct", label: "Win %", number: true, digits: 3 },
        { key: "points_for", label: "PF", number: true, digits: 2 }
      ]);
      ensureAuditOptions();
      loadLineup();
    }

    function ensureAuditOptions() {
      const teamSelect = document.getElementById("auditTeam");
      const weekSelect = document.getElementById("auditWeek");
      const selectedTeam = teamSelect.value;
      const selectedWeek = weekSelect.value;
      teamSelect.innerHTML = state.teams.map(team => `<option value="${team.team_index}">${escapeHtml(team.team)}</option>`).join("");
      weekSelect.innerHTML = state.weeks.map(week => `<option value="${week}">Week ${week}</option>`).join("");
      if (selectedTeam) teamSelect.value = selectedTeam;
      if (selectedWeek) weekSelect.value = selectedWeek;
    }

    async function loadLineup() {
      if (!state?.complete || !sessionId) return;
      const teamIndex = document.getElementById("auditTeam").value || "0";
      const week = document.getElementById("auditWeek").value || (state.weeks[0] || "1");
      try {
        const response = await fetch(`/api/lineup?id=${encodeURIComponent(sessionId)}&team_index=${encodeURIComponent(teamIndex)}&week=${encodeURIComponent(week)}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Lineup lookup failed");
        document.getElementById("auditTeamName").textContent = payload.team;
        document.getElementById("auditWeekLabel").textContent = `Week ${payload.week}`;
        document.getElementById("auditTotal").textContent = Number(payload.total_points).toFixed(2);
        document.getElementById("auditCount").textContent = String(payload.starters.length + payload.bench.length);
        renderTable("starters", payload.starters, [
          { key: "slot", label: "Slot" },
          { key: "player_name", label: "Player" },
          { key: "position", label: "Pos" },
          { key: "points", label: "Pts", number: true, digits: 2 }
        ]);
        renderTable("bench", payload.bench, [
          { key: "slot", label: "Slot" },
          { key: "player_name", label: "Player" },
          { key: "position", label: "Pos" },
          { key: "points", label: "Pts", number: true, digits: 2 }
        ]);
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    start.addEventListener("click", startDraft);
    reset.addEventListener("click", resetConfig);
    search.addEventListener("input", renderAvailable);
    positionFilter.addEventListener("change", renderAvailable);
    document.getElementById("auditTeam").addEventListener("change", loadLineup);
    document.getElementById("auditWeek").addEventListener("change", loadLineup);
    resetConfig();
    renderState();
  </script>
</body>
</html>
"""


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "fflab-gui/0.2"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(
                HTML.replace("__DEFAULT_CONFIG__", json.dumps(DEFAULT_GUI_CONFIG))
            )
            return
        if parsed.path == "/api/config":
            self._send_json(DEFAULT_GUI_CONFIG)
            return
        if parsed.path == "/api/draft/state":
            self._handle_get_state(parsed.query)
            return
        if parsed.path == "/api/lineup":
            self._handle_get_lineup(parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            parsed = urlparse(self.path)
            if parsed.path == "/api/run":
                self._send_json(run_gui_simulation(payload))
                return
            if parsed.path == "/api/draft/start":
                self._send_json(start_draft_session(payload))
                return
            if parsed.path == "/api/draft/pick":
                session_id = str(payload.get("id", ""))
                player_id = str(payload.get("player_id", ""))
                self._send_json(make_human_pick(session_id, player_id))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except KeyError as exc:
            self._send_json({"error": str(exc).strip("'")}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_get_state(self, query: str) -> None:
        try:
            params = parse_qs(query)
            session_id = params.get("id", [""])[0]
            self._send_json(session_state_payload(session_id))
        except KeyError as exc:
            self._send_json({"error": str(exc).strip("'")}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_get_lineup(self, query: str) -> None:
        try:
            params = parse_qs(query)
            session_id = params.get("id", [""])[0]
            team_index = int(params.get("team_index", ["0"])[0])
            week = int(params.get("week", ["1"])[0])
            self._send_json(lineup_payload(session_id, team_index, week))
        except KeyError as exc:
            self._send_json({"error": str(exc).strip("'")}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fflab-gui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), GuiHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"fflab GUI running at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
