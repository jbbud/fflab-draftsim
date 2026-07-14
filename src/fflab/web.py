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
    "max_extra_per_position": {
        "QB": 1,
        "RB": 4,
        "WR": 4,
        "TE": 1,
        "K": 0,
        "DEF": 0,
    },
    "position_start_rounds": {
        "QB": 6,
        "TE": 5,
        "DEF": 14,
        "K": 15,
    },
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
        <div class="grid two">
          <label>Teams<input id="numTeams" inputmode="numeric"></label>
          <label>Your Team<select id="humanTeam"></select></label>
        </div>
        <label>Team Names<textarea id="teamNames" rows="5"></textarea></label>
        <div class="grid four">
          <label>QB<input id="qbStart" inputmode="numeric"></label>
          <label>TE<input id="teStart" inputmode="numeric"></label>
          <label>DEF<input id="defStart" inputmode="numeric"></label>
          <label>K<input id="kStart" inputmode="numeric"></label>
        </div>
        <div class="grid two">
          <label>Playoff Teams<input id="playoffTeams" inputmode="numeric"></label>
          <label>First-Round Byes<input id="playoffByes" inputmode="numeric"></label>
        </div>
        <button id="newDraft" type="button">New Draft Board</button>
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
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def load_env_file(path: Path) -> dict[str, str]:
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
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _env_int(*names: str) -> int | None:
    value = _env_value(*names)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def gui_config() -> dict[str, Any]:
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
    safe_payload = {key: value for key, value in payload.items() if key not in {"espn_s2", "swid"}}
    result = sync_projection_payload(payload_with_env_credentials(payload))
    result["request"] = safe_payload
    return result


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "fflab-gui/1.0"

    def do_GET(self) -> None:
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
        try:
            parsed = urlparse(self.path)
            payload = self._read_json()
            if parsed.path == "/api/projections/sync":
                self._send_json(projection_sync_response(payload))
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _send_static(self, relative_path: str) -> None:
        target = (STATIC_ROOT / relative_path).resolve()
        if STATIC_ROOT.resolve() not in target.parents or not target.is_file():
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send_bytes(target.read_bytes(), content_type=content_type)

    def _send_html(self, body: str) -> None:
        self._send_bytes(body.encode("utf-8"), content_type="text/html; charset=utf-8")

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, content_type="application/json; charset=utf-8", status=status)

    def _send_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fflab-gui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
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
