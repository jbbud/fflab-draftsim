# fflab Draft Simulator

`fflab_draftsim` is an offline-first ESPN fantasy football draft simulator. It runs a small local Python server for ESPN sync, then the browser stores the draft room state in IndexedDB using a vendored Dexie-compatible bundle.

The goal is a free local Draft Dominator-style workflow: sync ESPN projections once, draft against bot teams, model pick trades, and review projected standings, weekly matchups, and mock playoffs.

## Current Features

- ESPN projection sync through `cwendt94/espn-api`.
- Private league support through `ESPN_S2` and `SWID` cookies.
- Offline browser cache for players, weekly projections, league schedule, draft slots, saved trades, draft picks, and session settings.
- Draft board with sortable columns for rank, name, position, NFL team, bye, projected points, position rank, ADP, injury status, and ownership.
- Position filter including `FLEX`.
- Defenses, kickers, and offensive players.
- Start Draft button so bots do not draft before you are ready.
- Bot drafting with configurable QB, TE, DEF, and K timing.
- Draft-pick trade testing and saving before the draft starts.
- ESPN league matchup schedule when available, with generated round robin fallback.
- Projected regular-season standings with PF and PA.
- Mock playoffs over the final three projection weeks with configurable playoff teams and first-round byes.
- Export, import, and reset for local browser data.

## Requirements

- Python 3.13 or newer.
- Network access only when syncing ESPN data.
- A modern browser with IndexedDB support.

## Install

From the repo root:

```powershell
python -m pip install -e .[dev]
```

The app dependencies are listed in [pyproject.toml](pyproject.toml):

- `espn_api`
- `pandas`
- `pydantic`
- `pytest` for development/tests

## ESPN Credentials

Public ESPN leagues may not need cookies. Private leagues usually need both cookies:

- `ESPN_S2`
- `SWID`

Copy the example file and fill in your values:

```powershell
Copy-Item .env.example .env
```

```env
ESPN_S2=your_espn_s2_cookie
SWID={your_swid_cookie}

LEAGUE_ID=123456
YEAR=2026
WEEK_START=1
WEEK_END=17
```

Safe defaults such as `LEAGUE_ID`, `YEAR`, `WEEK_START`, and `WEEK_END` are loaded into the GUI. Credentials are used server-side for sync if the browser form leaves them blank. Cookies are not saved to IndexedDB and are not echoed in API responses.

## Run The GUI

```powershell
python run_gui.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

Equivalent console script:

```powershell
fflab gui --port 8765
```

## Basic Workflow

1. Open the GUI.
2. Go to `Projection Sync`.
3. Enter or confirm league/year/week values.
4. Enter `SWID` and `ESPN S2` only if they are not already in `.env`.
5. Click `Sync ESPN`.
6. Go to `Draft Setup`.
7. Confirm team names, your team, bot timing, playoff teams, and first-round byes.
8. Add any pick trades in `Pick Trading`.
9. Click `Start Draft`.
10. Draft your players from the board; use `Auto Pick` to advance bot picks if needed.
11. After the draft completes, open `Rosters` to view rosters, standings, regular matchups, and mock playoffs.

## Projection Sync Details

The sync endpoint is:

```text
POST /api/projections/sync
```

It accepts:

- `league_id`
- `year`
- `week_start`
- `week_end`
- optional `espn_s2`
- optional `swid`

It returns normalized:

- `players`
- `weekly_projections`
- `league_settings`
- `teams`
- `team_names`
- `draft_slots`
- `league_schedule`
- `projection_meta`
- `synced_at`

Weekly projections come from ESPN weekly rows when available. If ESPN only provides a season total for a player, the app falls back to spreading that total across active weeks and setting the inferred bye to zero when possible.

## Draft Board

The board shows all undrafted cached players that match the current search and position filter. The count under `Draft Board` shows how many are visible out of all available players.

Columns:

- `Rank`: dense local board rank.
- `Name`
- `Pos`
- `Team`
- `Bye`
- `Proj`
- `Pos Rank`
- `ADP`: ESPN `ownership.averageDraftPosition`.
- `Inj`
- `Own %`
- `Pick`

The raw ESPN draft rank is preserved internally as `espn_rank`, but the visible rank is made dense so the board does not jump from, for example, 36 to 69.

## Draft Setup

The `Draft Setup` tab controls:

- Number of teams.
- Your team.
- Team names.
- Earliest bot draft rounds for QB, TE, DEF, and K.
- Playoff team count.
- First-round playoff byes.

Use `New Draft Board` after changing draft structure or team names. Playoff settings can be changed after a draft and results will recalculate.

## Pick Trading

Pick trades must be set before the draft starts.

The trade UI has two sides:

```text
Team A                 Team B
sends:                 sends:
Pick alpha             Pick alpha
Pick beta              Pick beta
```

Use `Test Trade` to apply a temporary trade for the current browser session. Use `Save` to persist the trade into IndexedDB.

Pick input formats:

- `12`: overall pick 12.
- `#1`: the selected team's pick in round 1.
- `#2.4`: round 2, pick 4.

If a team owns multiple picks in a round, use the `#round.pick` form to disambiguate.

## Results

The `Rosters` tab contains:

- A team selector and roster view.
- Projected standings.
- Regular-season matchup projections.
- Mock playoff bracket.

Regular-season matchups use ESPN's actual league schedule when the API returns it. If no schedule is available, the app generates a round-robin schedule.

The final three projection weeks are used as mock playoff weeks. Playoff seeds come from projected regular-season standings. Winners are determined by projected lineup score for that playoff week.

## Local Browser Data

The browser database is named:

```text
fflab_draftsim
```

Tables:

- `players`
- `weekly_projections`
- `league_schedule`
- `draft_slots`
- `pick_trades`
- `draft_picks`
- `sessions`

Use the GUI buttons:

- `Export`: save local data as JSON.
- `Import`: restore local data from JSON.
- `Reset Local`: clear the browser database and return to defaults.

## CLI

Start the local GUI:

```powershell
fflab gui --host 127.0.0.1 --port 8765
```

Fetch ESPN projections as JSON:

```powershell
fflab sync-projections --league-id 123456 --year 2026 --week-start 1 --week-end 17
```

For a private league:

```powershell
fflab sync-projections --league-id 123456 --year 2026 --espn-s2 "..." --swid "{...}"
```

## Tests

```powershell
python -m pytest -q
node --check src/fflab/static/app.js
```

On some Windows setups, pytest may warn that it cannot write `.pytest_cache`. That warning does not mean the tests failed.

## Troubleshooting

If sync fails:

- Confirm `LEAGUE_ID`, `YEAR`, `WEEK_START`, and `WEEK_END`.
- For private leagues, refresh `ESPN_S2` and `SWID`.
- Check that your ESPN account can open the league in a browser.

If the board is missing new fields such as ADP:

- Run a fresh ESPN sync. Older IndexedDB rows did not have every newer field.

If picks, trades, or rosters look stale:

- Use `Export` if you want a backup.
- Use `Reset Local`.
- Sync ESPN again.

If bots stop unexpectedly:

- Use `Auto Pick` once.
- If the draft state looks corrupted, reset local data and reload from a fresh sync.
