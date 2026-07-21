from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .projections import sync_projection_payload


DEFAULT_GUI_CONFIG: dict[str, Any] = {
    "num_teams": 14,
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
    "max_extra_per_position": {
        "QB": 1,
        "RB": 4,
        "WR": 4,
        "TE": 1,
        "K": 0,
        "DEF": 0,
    },
    "score_weights": {
        "vor": 200,
        "need": 34.68,
        "dropoff": 0.379,
        "handcuff": 1.0,
        "stack": 1.0,
        "rank": 75.99,
        "adp": 63.38,
        "backupPenalty": 1.58,
        "positionPreference": 0,
        "favoriteTeam": 0,
    },
    "score_weights_by_team": {},
    "position_preferences_by_team": {},
    "favorite_nfl_teams_by_team": {},
    "playoff_team_count": 6,
    "playoff_bye_count": 2,
    "human_team_index": 0,
}


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fflab Draft Simulator</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <div>
      <h1>fflab Draft Simulator</h1>
      <p id="syncMeta">No projections synced</p>
    </div>
    <div class="header-actions">
      <span id="onlineStatus" class="pill">Checking</span>
      <button id="exportData" type="button">Export</button>
      <button id="importData" type="button">Import</button>
      <button id="resetLocal" type="button" class="secondary">Reset Local</button>
    </div>
  </header>
  <main>
    <nav class="tabs" aria-label="Draft simulator tools">
      <button class="tab-button" type="button" data-tab="projectionTab">Projection Sync</button>
      <button class="tab-button" type="button" data-tab="setupTab">Draft Setup</button>
      <button class="tab-button" type="button" data-tab="tradingTab">Pick Trading</button>
      <button class="tab-button" type="button" data-tab="rostersTab">Rosters</button>
      <div class="nav-actions">
        <label class="nav-select">Your Team<select id="humanTeam"></select></label>
        <button id="newDraft" class="nav-action" type="button">New Draft Board</button>
      </div>
    </nav>
    <section id="tabPanels" class="tab-panels hidden">
      <section id="projectionTab" class="tab-panel hidden">
        <h2>Projection Sync</h2>
        <div class="grid two">
          <label>League ID<input id="leagueId" inputmode="numeric" placeholder="123456"></label>
          <label>Year<input id="year" inputmode="numeric" value="2026"></label>
        </div>
        <div class="grid two">
          <label>Week Start<input id="weekStart" inputmode="numeric" value="1"></label>
          <label>Week End<input id="weekEnd" inputmode="numeric" value="17"></label>
        </div>
        <label>SWID<input id="swid" autocomplete="off"></label>
        <label>ESPN S2<input id="espnS2" autocomplete="off" type="password"></label>
        <div class="grid two">
          <button id="syncEspn" type="button">Sync ESPN</button>
          <button id="loadDemo" type="button" class="secondary">Demo Data</button>
        </div>
      </section>
      <section id="setupTab" class="tab-panel hidden">
        <h2>Draft Setup</h2>
        <label class="short-field">Teams<input id="numTeams" inputmode="numeric"></label>
        <div class="grid two">
          <label>Playoff Teams<input id="playoffTeams" inputmode="numeric"></label>
          <label>First-Round Byes<input id="playoffByes" inputmode="numeric"></label>
        </div>
        <section class="score-weights">
          <h3>Bot Score Weights</h3>
          <label class="short-field">Team<select id="scoreWeightTeam"></select></label>
          <div class="weight-grid">
            <label>VOR<input id="weightVor" type="number" step="0.05"></label>
            <label>Need<input id="weightNeed" type="number" step="0.5"></label>
            <label>Dropoff<input id="weightDropoff" type="number" step="0.05"></label>
            <label>Handcuff<input id="weightHandcuff" type="number" step="0.25"></label>
            <label>Stack<input id="weightStack" type="number" step="0.25"></label>
            <label>Rank<input id="weightRank" type="number" step="0.5"></label>
            <label>ADP<input id="weightAdp" type="number" step="0.5"></label>
            <label>Backup Penalty<input id="weightBackupPenalty" type="number" step="0.05"></label>
            <label>Position Windows<input id="weightPositionPreference" type="number" step="0.25"></label>
            <label>Favorite Teams<input id="weightFavoriteTeam" type="number" step="0.25"></label>
          </div>
          <div class="grid two">
            <button id="saveScoreWeights" type="button" class="secondary">Save Team Weights</button>
            <button id="resetScoreWeights" type="button" class="secondary">Reset Team Weights</button>
          </div>
        </section>
        <section class="position-preferences">
          <h3>Draft Intel Position Windows</h3>
          <div class="preference-grid preference-head">
            <span>Pos</span><span>First Earliest</span><span>First Latest</span><span>Backup Earliest</span><span>Backup Latest</span>
          </div>
          <div class="preference-grid">
            <strong>QB</strong>
            <input id="prefQbFirstEarliest" type="number" min="1" step="1">
            <input id="prefQbFirstLatest" type="number" min="1" step="1">
            <input id="prefQbBackupEarliest" type="number" min="1" step="1">
            <input id="prefQbBackupLatest" type="number" min="1" step="1">
          </div>
          <div class="preference-grid">
            <strong>TE</strong>
            <input id="prefTeFirstEarliest" type="number" min="1" step="1">
            <input id="prefTeFirstLatest" type="number" min="1" step="1">
            <input id="prefTeBackupEarliest" type="number" min="1" step="1">
            <input id="prefTeBackupLatest" type="number" min="1" step="1">
          </div>
          <div class="preference-grid">
            <strong>K</strong>
            <input id="prefKFirstEarliest" type="number" min="1" step="1">
            <input id="prefKFirstLatest" type="number" min="1" step="1">
            <input id="prefKBackupEarliest" type="number" min="1" step="1">
            <input id="prefKBackupLatest" type="number" min="1" step="1">
          </div>
          <div class="preference-grid">
            <strong>DEF</strong>
            <input id="prefDefFirstEarliest" type="number" min="1" step="1">
            <input id="prefDefFirstLatest" type="number" min="1" step="1">
            <input id="prefDefBackupEarliest" type="number" min="1" step="1">
            <input id="prefDefBackupLatest" type="number" min="1" step="1">
          </div>
          <div class="grid two">
            <button id="savePositionPreferences" type="button" class="secondary">Save Position Windows</button>
            <button id="resetPositionPreferences" type="button" class="secondary">Reset Position Windows</button>
          </div>
        </section>
        <section class="favorite-teams">
          <h3>Favorite NFL Teams</h3>
          <div id="favoriteNflTeams" class="favorite-team-grid" role="group" aria-label="Favorite NFL teams"></div>
          <div class="grid two">
            <button id="saveFavoriteTeams" type="button" class="secondary">Save Favorite Teams</button>
            <button id="resetFavoriteTeams" type="button" class="secondary">Reset Favorite Teams</button>
          </div>
        </section>
      </section>
      <section id="tradingTab" class="tab-panel hidden">
        <h2>Pick Trade</h2>
        <div class="tab-grid">
          <div>
            <div class="trade-sheet">
              <div class="trade-side">
                <label>Team A<select id="tradeTeamA"></select></label>
                <strong>sends:</strong>
                <textarea id="tradePicksA" rows="7" placeholder="12&#10;#1&#10;#2.4"></textarea>
              </div>
              <div class="trade-side">
                <label>Team B<select id="tradeTeamB"></select></label>
                <strong>sends:</strong>
                <textarea id="tradePicksB" rows="7" placeholder="27&#10;#3&#10;#4.8"></textarea>
              </div>
            </div>
            <label>Notes<textarea id="tradeNotes" rows="3"></textarea></label>
            <div class="grid two">
              <button id="testTrade" type="button">Test Trade</button>
              <button id="saveTrade" type="button" class="secondary">Save</button>
            </div>
          </div>
          <div>
            <h3>Saved Pick Trades</h3>
            <div class="table-wrap compact"><table id="trades"></table></div>
          </div>
        </div>
      </section>
      <section id="rostersTab" class="tab-panel hidden">
        <h2>Rosters</h2>
        <label class="short-field">Team<select id="rosterTeam"></select></label>
        <div id="selectedRoster" class="roster-detail"></div>
        <section id="resultsPanel" class="results-block hidden">
          <h2>Projected Season</h2>
          <div class="metric-row">
            <div class="metric"><span>Champion</span><strong id="champion">-</strong></div>
            <div class="metric"><span>Projected Points</span><strong id="points">-</strong></div>
            <div class="metric"><span>Draft Picks</span><strong id="pickCount">-</strong></div>
            <div class="metric"><span>Weeks</span><strong id="weekCount">-</strong></div>
          </div>
          <div class="tables">
            <div>
              <h3>Standings</h3>
              <div class="table-wrap compact"><table id="standings"></table></div>
            </div>
            <div>
              <h3>Regular Matchups</h3>
              <div class="table-wrap compact"><table id="weeklyMatchups"></table></div>
            </div>
          </div>
          <div class="playoff-section">
            <h3>Mock Playoffs</h3>
            <div class="table-wrap compact"><table id="playoffMatchups"></table></div>
          </div>
        </section>
      </section>
    </section>
    <p id="status" class="status"></p>
    <section class="draft-layout">
      <section class="draft-main">
        <section class="band available-panel">
          <div id="clock" class="clock">
            <div>
              <strong id="clockTeam">No draft board</strong>
              <span id="clockMeta">Sync projections or load demo data.</span>
            </div>
            <div class="clock-actions">
              <button id="startDraft" type="button">Start Draft</button>
              <button id="autoPick" type="button" class="secondary">Auto Pick</button>
            </div>
          </div>
          <div class="board-tools">
            <div>
              <h2>Draft Board</h2>
              <p id="boardCount" class="board-count">No players loaded</p>
            </div>
            <div class="tools">
              <input id="search" placeholder="Search">
              <select id="positionFilter">
                <option value="">All</option>
                <option>QB</option><option>RB</option><option>WR</option>
                <option>TE</option><option>FLEX</option><option>K</option><option>DEF</option>
              </select>
            </div>
          </div>
          <div class="table-wrap board-wrap"><table id="available"></table></div>
        </section>
        <section class="band draft-log-panel">
          <h2>Draft Log</h2>
          <div class="table-wrap log-wrap"><table id="draftLog"></table></div>
        </section>
      </section>
      <section class="band user-roster-panel">
        <h2>Your Team</h2>
        <div id="currentRoster"></div>
      </section>
    </section>
  </main>
  <dialog id="importDialog">
    <form method="dialog">
      <h2>Import Local Data</h2>
      <textarea id="importPayload" rows="10"></textarea>
      <div class="grid two">
        <button id="confirmImport" value="default" type="button">Import</button>
        <button value="cancel" class="secondary">Cancel</button>
      </div>
    </form>
  </dialog>
  <script>window.FFLAB_DEFAULT_CONFIG = __DEFAULT_CONFIG__;</script>
  <script src="/static/dexie.min.js"></script>
  <script src="/static/app.js"></script>
</body>
</html>
"""


STATIC_ROOT = Path(__file__).with_name("static")
CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


def _clean_env_value(value: str) -> str:
    """Strip whitespace and one matching quote pair from a dotenv value."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def load_env_file(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE pairs into the process environment if unset."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _clean_env_value(value)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def load_default_env_files() -> None:
    """Load .env files from the current directory and repository root."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        load_env_file(resolved)


def _env_value(*names: str) -> str | None:
    """Return the first non-blank environment value for the provided names."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _env_int(*names: str) -> int | None:
    """Return the first configured environment value that parses as an int."""
    value = _env_value(*names)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def gui_config() -> dict[str, Any]:
    """Build browser-safe GUI defaults from static config and environment."""
    config = dict(DEFAULT_GUI_CONFIG)
    league_id = _env_value("LEAGUE_ID", "ESPN_LEAGUE_ID", "FFLAB_LEAGUE_ID", "league_id")
    if league_id:
        config["league_id"] = league_id
    year = _env_int("YEAR", "ESPN_YEAR", "FFLAB_YEAR", "year")
    if year:
        config["year"] = year
    week_start = _env_int("WEEK_START", "ESPN_WEEK_START", "FFLAB_WEEK_START", "week_start")
    if week_start:
        config["week_start"] = week_start
    week_end = _env_int("WEEK_END", "ESPN_WEEK_END", "FFLAB_WEEK_END", "week_end")
    if week_end:
        config["week_end"] = week_end
    return config


def payload_with_env_credentials(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill missing ESPN cookies from server environment variables."""
    resolved = dict(payload)
    if not str(resolved.get("espn_s2") or "").strip():
        espn_s2 = _env_value("ESPN_S2")
        if espn_s2:
            resolved["espn_s2"] = espn_s2
    if not str(resolved.get("swid") or "").strip():
        swid = _env_value("SWID", "ESPN_SWID")
        if swid:
            resolved["swid"] = swid
    return resolved


def projection_sync_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Sync projections and return a response without echoing private cookies."""
    safe_payload = {key: value for key, value in payload.items() if key not in {"espn_s2", "swid"}}
    result = sync_projection_payload(payload_with_env_credentials(payload))
    result["request"] = safe_payload
    return result


class GuiHandler(BaseHTTPRequestHandler):
    """HTTP handler for the hosted draft simulator UI and JSON endpoints."""

    server_version = "fflab-gui/1.0"

    def do_GET(self) -> None:
        """Serve the app shell, safe config, static assets, or a 404."""
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(HTML.replace("__DEFAULT_CONFIG__", json.dumps(gui_config())))
            return
        if parsed.path == "/api/config":
            self._send_json(gui_config())
            return
        if parsed.path.startswith("/static/"):
            self._send_static(parsed.path.removeprefix("/static/"))
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Handle projection sync and browser log POST endpoints."""
        try:
            parsed = urlparse(self.path)
            payload = self._read_json()

            if parsed.path == "/api/projections/sync":
                self._send_json(projection_sync_response(payload))
                return

            if parsed.path == "/api/log":
                print(f"[browser] {payload.get('label', 'log')}: {payload}")
                self._send_json({"ok": True})
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _read_json(self) -> dict[str, Any]:
        """Read the request body as a JSON object."""
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _send_static(self, relative_path: str) -> None:
        """Serve a whitelisted static file from the package static directory."""
        target = (STATIC_ROOT / relative_path).resolve()
        if STATIC_ROOT.resolve() not in target.parents or not target.is_file():
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send_bytes(target.read_bytes(), content_type=content_type)

    def _send_html(self, body: str) -> None:
        """Write a UTF-8 HTML response body."""
        self._send_bytes(body.encode("utf-8"), content_type="text/html; charset=utf-8")

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Serialize a payload as JSON with the provided HTTP status."""
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, content_type="application/json; charset=utf-8", status=status)

    def _send_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        """Write common response headers and a raw response body."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence default request logging; the browser log API is explicit."""
        return


def build_parser() -> argparse.ArgumentParser:
    """Create the local GUI server command-line parser."""
    parser = argparse.ArgumentParser(prog="fflab-gui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the local threaded HTTP server until interrupted."""
    load_default_env_files()
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
