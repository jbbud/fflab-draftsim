# fflab Draft Simulator

`fflab` is an offline-first ESPN fantasy football draft simulator. It hosts a small Python web app, syncs ESPN projection and league data when requested, then keeps the draft room state in browser IndexedDB through a vendored Dexie-compatible bundle.

The app is built for a local Draft Dominator-style workflow: sync a league, tune bot behavior, model pick trades, draft against bots, and review projected rosters, standings, weekly matchups, and playoffs.

## Current Features

- ESPN projection sync through `cwendt94/espn-api`.
- Private league support through `ESPN_S2` and `SWID` cookies.
- Hosted UI through either `run_gui.py` locally or `api/index.py` for serverless-style hosting.
- Offline browser cache for players, weekly projections, league schedule, draft slots, pick trades, draft picks, and session settings.
- Draft board with search, position filtering, sortable columns, ADP, ownership, injury status, and dense local board ranks.
- Synced ESPN league team names, with `Team #X` fallback names when sync data is missing.
- ESPN draft slots when available, with a generated snake draft fallback.
- Pick trade testing and saving before the draft starts.
- Bot drafting with per-team score weights for VOR, need, dropoff, handcuff, stack, rank, and ADP.
- Robust per-pick normalization for VOR, rank, and ADP so equal weights operate on comparable scoring units.
- Fixed K/DEF timing delay against round 15; QB, RB, WR, and TE are not timing-discounted.
- ESPN league matchup schedule when available, with generated round-robin fallback.
- Projected regular-season standings with PF and PA.
- Mock playoffs over the final three projection weeks with configurable playoff teams and first-round byes.
- Export, import, and reset for local browser data.

## Requirements

- Python 3.13 or newer.
- Network access only when installing dependencies or syncing ESPN data.
- A modern browser with IndexedDB support.

## Install

From the repo root:

```powershell
python -m pip install -e .[dev]
```

Runtime dependency:

- `espn_api`

Development dependency:

- `pytest`

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

## Run Locally

```powershell
python run_gui.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

Equivalent console script:

```powershell
fflab-gui --port 8765
```

## Hosting Path

The active hosted app path is intentionally small:

- `run_gui.py` adds `src` to `sys.path` and calls `fflab.web:main`.
- `api/index.py` adds `src` to `sys.path` and exposes `fflab.web.GuiHandler` as `handler`.
- `src/fflab/web.py` serves the HTML shell, static assets, `/api/projections/sync`, and `/api/log`.
- `src/fflab/projections.py` talks to ESPN and normalizes projection, team, draft-slot, and schedule payloads.
- `src/fflab/static/app.js` owns the browser app, IndexedDB state, bot drafting, pick trading, results, and UI rendering.

Legacy CLI, optimizer, and simulation modules have been removed from the current hosting path.

## Basic Workflow

1. Open the GUI.
2. Go to `Projection Sync`.
3. Enter or confirm league/year/week values.
4. Enter `SWID` and `ESPN S2` only if they are not already in `.env`.
5. Click `Sync ESPN`.
6. Go to `Draft Setup`.
7. Confirm team count, your team, playoff teams, first-round byes, and bot score weights.
8. Add any pick trades in `Pick Trading`.
9. Click `New Draft Board` if you changed draft structure or trades.
10. Click `Start Draft`.
11. Draft your players from the board; use `Auto Pick` to advance bot picks if needed.
12. After the draft completes, open `Rosters` to view rosters, standings, regular matchups, and mock playoffs.

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
- Playoff team count.
- First-round playoff byes.
- Per-team bot score weights.

Team names come from the synced league. If a synced name is unavailable, the app displays `Team #X`.

The score-weight editor lets you pick a team and tune:

- `VOR`
- `Need`
- `Dropoff`
- `Handcuff`
- `Stack`
- `Rank`
- `ADP`
- `Backup Penalty`
- `Position Windows`
- `Favorite Teams`

VOR, rank, and ADP are normalized per bot pick with median/IQR scaling before weights are applied. This keeps equal weights roughly comparable while still allowing elite VOR outliers to matter.

Position-window and favorite-team preference weights default to zero, so those preference sections do not affect mock drafts unless you opt a team into them.

Kickers and defenses are timing-discounted against round 15. QB, RB, WR, and TE do not use a broad position start-round discount.

QB and TE backups receive extra opportunity-cost pressure in one-QB roster builds. After a team has filled its starter at QB or TE, the bot scorer subtracts roster-surplus, round-mismatch, and league-saturation penalties multiplied by the `Backup Penalty` weight, so similarly valued RB/WR depth is usually preferred unless the backup has clear falling value.

## Bot Scoring Notes

For each bot pick, the app evaluates legal candidates for the current roster and combines:

- normalized VOR
- starter and roster need
- positional dropoff before the team's next pick
- handcuff and stack bonuses
- normalized rank value
- normalized ADP value
- K/DEF timing discount
- QB/TE backup opportunity-cost penalties

The VOR replacement baseline scales with league size and roster settings. In a 14-team league, the default baseline is:

- `QB14`
- `RB35`
- `WR35`
- `TE14`
- `K1`
- `DEF1`

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

## Tests

```powershell
node --check src/fflab/static/app.js
node --check tools/train_standard_weights.mjs
python -m pytest -q
python -m compileall -q src api run_gui.py
```

On some Windows setups, pytest may warn that it cannot write `.pytest_cache`. That warning does not mean the tests failed.

## Training Bot Weights

The first-pass weight trainer is a dependency-free Node script:

```powershell
node tools/train_standard_weights.mjs --candidates 96 --seeds 12 --survivors 16 --holdout-seeds 48 --out weight-report.json --trace-out pick-trace.json
```

By default, the script loads `.env` and syncs the league referenced by `LEAGUE_ID`, `YEAR`, `WEEK_START`, `WEEK_END`, `ESPN_S2`, and `SWID`, using the same server-side credential flow as the GUI. Use `--data export.json` to train from a browser export instead, or `--demo` for an offline synthetic smoke run.

By default, the trainer optimizes for `--target-team 0`. Use `--target-teams all` to train one shared global weight vector across every team index, or pass an explicit subset such as `--target-teams 0,3,7`. Training stages sample target-team mini-batches by seed when multiple target teams are selected; tune that with `--target-team-sample-rate 0.3` and `--target-team-sample-min 1`. Final holdout evaluates every selected target team by default; pass `--full-holdout false` only when you want holdout to use the same sampling strategy.

The trainer prints stage progress to stderr with completed evaluation count, elapsed time, and ETA. The final JSON report is still printed to stdout and optionally written with `--out`.

The harness tunes these standard scorer weights:

- `vor`
- `rank`
- `adp`
- `need`
- `dropoff`
- `handcuff`
- `stack`
- `backupPenalty`

It does not train the `Position Windows` or `Favorite Teams` preference weights; those stay user-controlled and default to zero. It evaluates candidate weights for the selected target team or teams against baseline-weight opponents using seeded random search plus successive halving. The first objective is projected optimal-lineup season points, with a holdout seed set reported separately from tuning seeds.

Use `--trace-out` to write the promoted candidate's pick-level score breakdown, including VOR, rank, ADP, need, dropoff, handcuff, stack, backup penalty, timing, and selected player details.

## Troubleshooting

If sync fails:

- Confirm `LEAGUE_ID`, `YEAR`, `WEEK_START`, and `WEEK_END`.
- For private leagues, refresh `ESPN_S2` and `SWID`.
- Check that your ESPN account can open the league in a browser.

If the board is missing fields such as ADP, ownership, or injury status:

- Run a fresh ESPN sync. Older IndexedDB rows may not have every newer field.

If team names look generic:

- Run a fresh ESPN sync.
- If ESPN does not return names, the app falls back to `Team #X`.

If picks, trades, or rosters look stale:

- Use `Export` if you want a backup.
- Use `Reset Local`.
- Sync ESPN again.

If bots stop unexpectedly:

- Use `Auto Pick` once.
- If the draft state looks corrupted, reset local data and reload from a fresh sync.
