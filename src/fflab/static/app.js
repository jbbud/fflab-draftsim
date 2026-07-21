(() => {
  "use strict";

  const defaultConfig = window.FFLAB_DEFAULT_CONFIG || {};
  const TABLES = ["players", "weekly_projections", "league_schedule", "draft_slots", "pick_trades", "draft_picks", "sessions"];
  const db = new Dexie("fflab_draftsim");
  db.version(3).stores({
    players: "&player_id, rank, position, projected_total_pts",
    weekly_projections: "[player_id+week], player_id, week",
    league_schedule: "&id, week, home_team_index, away_team_index",
    draft_slots: "&overall, [round+pick_in_round], current_team, original_team",
    pick_trades: "&id, created_at",
    draft_picks: "&overall, player_id, team_index",
    sessions: "&id",
  });

  const state = {
    players: [],
    weekly: [],
    schedule: [],
    slots: [],
    trades: [],
    picks: [],
    session: null,
    results: null,
    draftBusy: false,
    availableSort: { key: "default", direction: "asc" },
    replacementBaselineByPosition: {},
  };

  // Resolve a DOM element by id.
  const $ = (id) => document.getElementById(id);
  const positionOrder = ["QB", "RB", "WR", "TE", "K", "DEF"];
  const flexPositions = new Set(["RB", "WR", "TE"]);
  const replacementFlexShareByPosition = { RB: 0.5, WR: 0.5, TE: 0 };
  const scoreWeightUnitVersion = "normalized-v1";
  const legacyVorWeightScale = 20;
  const positionPreferencePositions = ["QB", "TE", "K", "DEF"];
  const positionPreferenceFields = ["first_earliest", "first_latest", "backup_earliest", "backup_latest"];
  const positionPreferenceInputIds = {
    QB: {
      first_earliest: "prefQbFirstEarliest",
      first_latest: "prefQbFirstLatest",
      backup_earliest: "prefQbBackupEarliest",
      backup_latest: "prefQbBackupLatest",
    },
    TE: {
      first_earliest: "prefTeFirstEarliest",
      first_latest: "prefTeFirstLatest",
      backup_earliest: "prefTeBackupEarliest",
      backup_latest: "prefTeBackupLatest",
    },
    K: {
      first_earliest: "prefKFirstEarliest",
      first_latest: "prefKFirstLatest",
      backup_earliest: "prefKBackupEarliest",
      backup_latest: "prefKBackupLatest",
    },
    DEF: {
      first_earliest: "prefDefFirstEarliest",
      first_latest: "prefDefFirstLatest",
      backup_earliest: "prefDefBackupEarliest",
      backup_latest: "prefDefBackupLatest",
    },
  };
  const positionPreferenceMaxAdjustment = 24;
  const favoriteTeamScoreBonus = 8;
  const defaultScoreWeights = {
    vor: number(defaultConfig.score_weights?.vor, 25),
    need: number(defaultConfig.score_weights?.need, 20),
    dropoff: number(defaultConfig.score_weights?.dropoff, 0.6),
    handcuff: number(defaultConfig.score_weights?.handcuff, 1),
    stack: number(defaultConfig.score_weights?.stack, 1),
    rank: number(defaultConfig.score_weights?.rank, 25),
    adp: number(defaultConfig.score_weights?.adp, 25),
    positionPreference: number(defaultConfig.score_weights?.positionPreference, 0),
    favoriteTeam: number(defaultConfig.score_weights?.favoriteTeam, 0),
  };
  const scoreWeightFields = [
    { key: "vor", inputId: "weightVor" },
    { key: "need", inputId: "weightNeed" },
    { key: "dropoff", inputId: "weightDropoff" },
    { key: "handcuff", inputId: "weightHandcuff" },
    { key: "stack", inputId: "weightStack" },
    { key: "rank", inputId: "weightRank" },
    { key: "adp", inputId: "weightAdp" },
    { key: "positionPreference", inputId: "weightPositionPreference" },
    { key: "favoriteTeam", inputId: "weightFavoriteTeam" },
  ];

  // Escape dynamic text before injecting it into table or roster markup.
  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char]));
  }

  // Update the shared status message and error styling.
  function setStatus(message, isError = false) {
    $("status").textContent = message;
    $("status").className = isError ? "status error" : "status";
  }

  // Parse finite numbers from form fields and payload values.
  function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  // Parse whole-number values from form fields and payload values.
  function integer(value, fallback = 0) {
    return Math.trunc(number(value, fallback));
  }

  // Bound a numeric value to an inclusive range.
  function clamp(value, min, max) {
    return Math.max(min, Math.min(value, max));
  }

  // Format a fantasy points value for compact display.
  function formatPoints(value) {
    return number(value).toFixed(1);
  }

  // Sort players by board rank, then projected total points.
  function sortPlayers(players) {
    return players.slice().sort((a, b) => {
      const aRank = integer(a.rank);
      const bRank = integer(b.rank);
      if (aRank > 0 && bRank > 0) return aRank - bRank || number(b.projected_total_pts) - number(a.projected_total_pts);
      if (aRank > 0) return -1;
      if (bRank > 0) return 1;
      return number(b.projected_total_pts) - number(a.projected_total_pts);
    });
  }

  // Rebuild dense display ranks while preserving ESPN rank separately.
  function normalizeBoardRanks(players) {
    let changed = false;
    sortPlayers(players).forEach((player, index) => {
      const displayRank = index + 1;
      const espnRank = integer(player.espn_rank ?? player.rank);
      if (espnRank > 0 && integer(player.espn_rank) !== espnRank) {
        player.espn_rank = espnRank;
        changed = true;
      }
      if (integer(player.rank) !== displayRank) {
        player.rank = displayRank;
        changed = true;
      }
    });
    return changed;
  }

  // Fill blank position ranks from the local board order.
  function fillMissingPositionRanks(players) {
    const groups = new Map();
    let changed = false;
    for (const player of players) {
      const position = String(player.position || "");
      if (!position) continue;
      if (!groups.has(position)) groups.set(position, []);
      groups.get(position).push(player);
    }
    for (const rows of groups.values()) {
      sortPlayers(rows).forEach((player, index) => {
        if (integer(player.pos_rank) <= 0) {
          player.pos_rank = index + 1;
          changed = true;
        }
      });
    }
    return changed;
  }

  // Condense ESPN injury status text for the board column.
  function injuryCode(status) {
    const value = String(status || "").trim().toUpperCase().replace(/[\s-]+/g, "_");
    if (!value || value === "ACTIVE" || value === "NORMAL") return "";
    if (value === "QUESTIONABLE") return "Q";
    if (value === "DOUBTFUL") return "D";
    if (value === "OUT") return "O";
    if (["IR", "INJURY_RESERVE", "INJURED_RESERVE", "RESERVE_IR"].includes(value)) return "IR";
    if (value === "SUSPENSION" || value === "SUSPENDED") return "SUS";
    if (value === "PUP") return "PUP";
    return value.slice(0, 3);
  }

  // Compare nullable numeric or string values for sortable table headers.
  function compareValues(a, b, direction = "asc") {
    const aMissing = a == null || a === "";
    const bMissing = b == null || b === "";
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1;
    if (bMissing) return -1;
    const result = typeof a === "number" && typeof b === "number"
      ? a - b
      : String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
    return direction === "desc" ? -result : result;
  }

  // Deep-clone JSON-safe app state before storing it.
  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  // Merge stored score weights with the global scoring defaults.
  function normalizeScoreWeights(weights = {}) {
    const normalized = { ...defaultScoreWeights };
    for (const field of scoreWeightFields) {
      normalized[field.key] = number(weights?.[field.key], defaultScoreWeights[field.key]);
    }
    return normalized;
  }

  // Keep only per-team weight overrides that still belong to active teams.
  function normalizeScoreWeightsByTeam(weightsByTeam = {}, numTeams = 0) {
    const normalized = {};
    for (let index = 0; index < numTeams; index += 1) {
      normalized[index] = normalizeScoreWeights(weightsByTeam?.[index]);
    }
    return normalized;
  }

  // Keep only QB/TE/K/DEF Draft Intel windows with positive round values.
  function normalizePositionPreferences(preferences = {}) {
    const normalized = {};
    for (const position of positionPreferencePositions) {
      normalized[position] = {};
      for (const field of positionPreferenceFields) {
        const value = integer(preferences?.[position]?.[field], 0);
        normalized[position][field] = value > 0 ? value : null;
      }
      for (const prefix of ["first", "backup"]) {
        const earliestKey = `${prefix}_earliest`;
        const latestKey = `${prefix}_latest`;
        const earliest = normalized[position][earliestKey];
        const latest = normalized[position][latestKey];
        if (earliest != null && latest != null && earliest > latest) {
          normalized[position][earliestKey] = latest;
          normalized[position][latestKey] = earliest;
        }
      }
    }
    return normalized;
  }

  // Keep per-team Draft Intel windows aligned to active team indexes.
  function normalizePositionPreferencesByTeam(preferencesByTeam = {}, numTeams = 0) {
    const normalized = {};
    for (let index = 0; index < numTeams; index += 1) {
      normalized[index] = normalizePositionPreferences(preferencesByTeam?.[index]);
    }
    return normalized;
  }

  // Normalize favorite NFL team abbreviations for one fantasy team.
  function normalizeFavoriteNflTeams(teams = []) {
    const values = Array.isArray(teams) ? teams : [];
    return [...new Set(values
      .map((team) => String(team || "").trim().toUpperCase())
      .filter((team) => team && team !== "NONE"))]
      .sort((a, b) => a.localeCompare(b));
  }

  // Keep per-team favorite NFL team lists aligned to active team indexes.
  function normalizeFavoriteNflTeamsByTeam(teamsByTeam = {}, numTeams = 0) {
    const normalized = {};
    for (let index = 0; index < numTeams; index += 1) {
      normalized[index] = normalizeFavoriteNflTeams(teamsByTeam?.[index]);
    }
    return normalized;
  }

  // Convert saved raw-VOR score weights into normalized score units once.
  function migrateScoreWeightUnits(session) {
    if (!session || session.score_weight_units === scoreWeightUnitVersion) return { session, changed: false };

    const numTeams = sessionTeamCount(session);
    const scoreWeightsByTeam = normalizeScoreWeightsByTeam(session.score_weights_by_team || {}, numTeams);
    for (let index = 0; index < numTeams; index += 1) {
      const legacyWeights = session.score_weights_by_team?.[index] || {};
      scoreWeightsByTeam[index].vor = number(legacyWeights.vor, 1.25) * legacyVorWeightScale;
    }

    return {
      session: {
        ...session,
        score_weight_units: scoreWeightUnitVersion,
        score_weights_by_team: scoreWeightsByTeam,
      },
      changed: true,
    };
  }

  // Build one generated team name for missing synced league names.
  function fallbackTeamName(index) {
    return `Team #${index + 1}`;
  }

  // Prefer the team name fields returned by ESPN sync payloads.
  function syncedTeamName(team) {
    return String(team?.team_name || team?.name || team?.display_name || "").trim();
  }

  // Resolve team names from synced team records, synced name lists, then generated fallbacks.
  function teamNamesFromSources({ teams = [], teamNames = [], numTeams = 0 } = {}) {
    const sourceTeams = Array.isArray(teams) ? teams : [];
    const sourceNames = Array.isArray(teamNames) ? teamNames : [];
    return Array.from({ length: numTeams }, (_, index) => (
      syncedTeamName(sourceTeams[index])
      || String(sourceNames[index] || "").trim()
      || fallbackTeamName(index)
    ));
  }

  // Preserve synced team metadata while keeping display names and slots normalized.
  function teamsWithResolvedNames(session, names) {
    const sourceTeams = Array.isArray(session?.teams) ? session.teams : [];
    return names.map((name, index) => {
      const team = sourceTeams[index] || {};
      return {
        ...team,
        team_id: team.team_id == null ? index + 1 : team.team_id,
        team_name: name,
        draft_slot: integer(team.draft_slot, index + 1),
      };
    });
  }

  // Build fallback team names when synced league names are unavailable.
  function defaultTeamNames(numTeams) {
    return Array.from({ length: numTeams }, (_, index) => fallbackTeamName(index));
  }

  // Create the active session object from defaults and optional sync data.
  function defaultSession(overrides = {}) {
    const numTeams = integer(overrides.num_teams ?? defaultConfig.num_teams, 10);
    const names = teamNamesFromSources({
      teams: overrides.teams || [],
      teamNames: overrides.team_names || [],
      numTeams,
    });
    const leaguePlayoffTeams = integer(overrides.league_settings?.playoff_team_count ?? defaultConfig.playoff_team_count, Math.min(6, numTeams));
    const playoffTeams = clamp(integer(overrides.playoff_team_count ?? leaguePlayoffTeams, Math.min(6, numTeams)), 2, numTeams);
    const playoffByes = clamp(integer(overrides.playoff_bye_count ?? defaultConfig.playoff_bye_count, playoffTeams >= 6 ? 2 : 0), 0, Math.max(playoffTeams - 2, 0));
    return {
      id: "active",
      league_id: overrides.league_id || defaultConfig.league_id || "",
      year: integer(overrides.year ?? defaultConfig.year, new Date().getFullYear()),
      week_start: integer(overrides.week_start ?? defaultConfig.week_start, 1),
      week_end: integer(overrides.week_end ?? defaultConfig.week_end, 17),
      synced_at: overrides.synced_at || "",
      league_settings: overrides.league_settings || {},
      num_teams: numTeams,
      team_names: names.slice(0, numTeams),
      teams: teamsWithResolvedNames(overrides, names),
      draft_slots: clone(overrides.draft_slots || []),
      projection_meta: clone(overrides.projection_meta || {}),
      draft_started: Boolean(overrides.draft_started),
      roster_settings: clone(defaultConfig.roster_settings || {}),
      max_extra_per_position: clone(defaultConfig.max_extra_per_position || {}),
      score_weight_units: scoreWeightUnitVersion,
      score_weights_by_team: normalizeScoreWeightsByTeam(
        overrides.score_weights_by_team || defaultConfig.score_weights_by_team || {},
        numTeams
      ),
      position_preferences_by_team: normalizePositionPreferencesByTeam(
        overrides.position_preferences_by_team || defaultConfig.position_preferences_by_team || {},
        numTeams
      ),
      favorite_nfl_teams_by_team: normalizeFavoriteNflTeamsByTeam(
        overrides.favorite_nfl_teams_by_team || defaultConfig.favorite_nfl_teams_by_team || {},
        numTeams
      ),
      playoff_team_count: playoffTeams,
      playoff_bye_count: playoffByes,
      human_team_index: integer(overrides.human_team_index ?? defaultConfig.human_team_index, 0),
      source: overrides.source || "empty",
    };
  }

  // Normalize loaded sessions so team names only come from sync data or fallbacks.
  function normalizeSessionTeamNames(session) {
    const sourceSession = session || defaultSession();
    const numTeams = sessionTeamCount(sourceSession);
    const useSyncedNames = sourceSession.source !== "empty";
    const names = teamNamesFromSources({
      teams: useSyncedNames ? sourceSession.teams || [] : [],
      teamNames: useSyncedNames ? sourceSession.team_names || [] : [],
      numTeams,
    });
    return {
      ...sourceSession,
      num_teams: numTeams,
      team_names: names,
      teams: teamsWithResolvedNames(useSyncedNames ? sourceSession : { teams: [] }, names),
      position_preferences_by_team: normalizePositionPreferencesByTeam(sourceSession.position_preferences_by_team || {}, numTeams),
      favorite_nfl_teams_by_team: normalizeFavoriteNflTeamsByTeam(sourceSession.favorite_nfl_teams_by_team || {}, numTeams),
      human_team_index: clamp(integer(sourceSession.human_team_index, 0), 0, Math.max(numTeams - 1, 0)),
    };
  }

  // Count draft rounds from roster settings.
  function totalRosterSlots(session = state.session) {
    const settings = session?.roster_settings || {};
    return Object.values(settings).reduce((sum, value) => sum + integer(value), 0);
  }

  // Render a table with optional custom cell renderers and sort headers.
  function renderTable(id, rows, columns) {
    const table = $(id);
    if (!rows || rows.length === 0) {
      table.innerHTML = "<tbody><tr><td>No rows</td></tr></tbody>";
      return;
    }
    const head = `<thead><tr>${columns.map((col) => {
      const classes = [col.number ? "number" : "", col.className || ""].filter(Boolean).join(" ");
      const sortAttr = col.sortKey ? ` data-sort-key="${escapeHtml(col.sortKey)}"` : "";
      const indicator = col.sortKey && state.availableSort.key === col.sortKey
        ? (state.availableSort.direction === "asc" ? " ^" : " v")
        : "";
      const label = `${escapeHtml(col.label)}${indicator}`;
      return `<th class="${classes}"${sortAttr}>${col.sortKey ? `<button class="sort-header" type="button">${label}</button>` : label}</th>`;
    }).join("")}</tr></thead>`;
    const body = `<tbody>${rows.map((row) => `<tr>${columns.map((col) => {
      if (col.render) return col.render(row);
      const raw = row[col.key];
      const value = typeof raw === "number" ? (col.digits == null ? raw : raw.toFixed(col.digits)) : (raw ?? "");
      const classes = [col.number ? "number" : "", col.className || ""].filter(Boolean).join(" ");
      return `<td class="${classes}">${escapeHtml(value)}</td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
    table.innerHTML = head + body;
  }

  // Clear a set of IndexedDB tables.
  async function clearTables(tableNames) {
    for (const name of tableNames) await db.table(name).clear();
  }

  // Identify synthetic skipped picks used when no legal bot pick exists.
  function isSkippedPick(pick) {
    return Boolean(pick?.skipped || String(pick?.player_id || "").startsWith("__skip_"));
  }

  // Return player ids already drafted by non-skipped picks.
  function draftedPlayerIds(picks = state.picks) {
    const drafted = new Set();
    for (const pick of picks) {
      const playerId = String(pick.player_id || "");
      if (playerId && !isSkippedPick(pick)) drafted.add(playerId);
    }
    return drafted;
  }

  // Remove duplicate or malformed draft picks from a persisted pick list.
  function sanitizeDraftPicks(picks) {
    const clean = [];
    const removed = [];
    const seenOverall = new Set();
    const seenPlayers = new Set();
    const rows = picks.slice().sort((a, b) => integer(a.overall) - integer(b.overall));
    for (const pick of rows) {
      const overall = integer(pick.overall);
      const playerId = String(pick.player_id || "");
      if (!overall || !playerId || seenOverall.has(overall)) {
        removed.push(pick);
        continue;
      }
      if (!isSkippedPick(pick) && seenPlayers.has(playerId)) {
        removed.push(pick);
        continue;
      }
      seenOverall.add(overall);
      if (!isSkippedPick(pick)) seenPlayers.add(playerId);
      clean.push({ ...pick, overall });
    }
    return { clean, removed };
  }

  // Repair persisted draft picks before putting them into app state.
  async function repairDraftPicksFromDb() {
    const rows = await db.draft_picks.toArray();
    const { clean, removed } = sanitizeDraftPicks(rows);
    if (removed.length > 0) {
      await db.draft_picks.clear();
      if (clean.length > 0) await db.draft_picks.bulkPut(clean);
    }
    return { picks: clean, removed };
  }

  // Load all browser-persisted state and render the draft room.
  async function loadState() {
    await db.open();
    const storedSession = await db.sessions.get("active") || defaultSession();
    state.session = normalizeSessionTeamNames(storedSession);
    const teamNameMigrationChanged = JSON.stringify(storedSession) !== JSON.stringify(state.session);
    const weightMigration = migrateScoreWeightUnits(state.session);
    state.session = weightMigration.session;
    if (weightMigration.changed || teamNameMigrationChanged) await saveSession(state.session);
    const players = await db.players.toArray();
    const ranksChanged = normalizeBoardRanks(players);
    if (fillMissingPositionRanks(players) || ranksChanged) await db.players.bulkPut(players);
    state.players = sortPlayers(players);
    state.weekly = await db.weekly_projections.toArray();
    state.schedule = (await db.league_schedule.toArray()).sort((a, b) => integer(a.week) - integer(b.week));
    state.slots = (await db.draft_slots.toArray()).sort((a, b) => a.overall - b.overall);
    state.trades = (await db.pick_trades.toArray()).sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)));
    const repairedPicks = await repairDraftPicksFromDb();
    state.picks = repairedPicks.picks;
    state.results = simulateDraft();
    state.replacementBaselineByPosition = replacementPointsByPosition(state.players);
    writeProjectionInputs();
    writeSetupInputs();
    render();
    if (repairedPicks.removed.length > 0) {
      setStatus(`Removed ${repairedPicks.removed.length} duplicate draft pick${repairedPicks.removed.length === 1 ? "" : "s"} from local draft state.`);
    }
    await resumeBotDraftIfNeeded();
  }

  // Persist the active session and keep memory in sync with IndexedDB.
  async function saveSession(session = state.session) {
    state.session = normalizeSessionTeamNames(session);
    await db.sessions.put(state.session);
  }

  // Return the team currently selected in the score-weight editor.
  function selectedScoreWeightTeamIndex() {
    const fallback = integer(state.session?.human_team_index, 0);
    return clamp(integer($("scoreWeightTeam").value, fallback), 0, Math.max(integer(state.session?.num_teams, 1) - 1, 0));
  }

  // Return scoring weights for one team from the active session.
  function scoreWeightsForTeam(teamIndex, session = state.session) {
    return normalizeScoreWeights(session?.score_weights_by_team?.[teamIndex]);
  }

  // Return Draft Intel position windows for one team.
  function positionPreferencesForTeam(teamIndex, session = state.session) {
    return normalizePositionPreferences(session?.position_preferences_by_team?.[teamIndex]);
  }

  // Return favorite NFL teams for one mock drafter.
  function favoriteNflTeamsForTeam(teamIndex, session = state.session) {
    return normalizeFavoriteNflTeams(session?.favorite_nfl_teams_by_team?.[teamIndex]);
  }

  // Read the score-weight editor controls into a normalized weight object.
  function readScoreWeightInputs() {
    const weights = {};
    for (const field of scoreWeightFields) {
      weights[field.key] = number($(field.inputId).value, defaultScoreWeights[field.key]);
    }
    return normalizeScoreWeights(weights);
  }

  // Write one team's stored scoring weights into the editor controls.
  function writeScoreWeightInputs() {
    const teamIndex = selectedScoreWeightTeamIndex();
    const weights = scoreWeightsForTeam(teamIndex);
    for (const field of scoreWeightFields) {
      $(field.inputId).value = weights[field.key];
    }
  }

  // Persist the current score-weight editor values for the selected team.
  async function saveScoreWeightInputs({ announce = true } = {}) {
    const teamIndex = selectedScoreWeightTeamIndex();
    const scoreWeightsByTeam = normalizeScoreWeightsByTeam(
      state.session?.score_weights_by_team || {},
      integer(state.session?.num_teams, 0)
    );
    scoreWeightsByTeam[teamIndex] = readScoreWeightInputs();
    await saveSession({ ...state.session, score_weights_by_team: scoreWeightsByTeam });
    if (announce) setStatus(`Saved score weights for ${teamName(teamIndex)}.`);
  }

  // Restore default scoring weights for the selected team.
  async function resetScoreWeightInputs() {
    const teamIndex = selectedScoreWeightTeamIndex();
    const scoreWeightsByTeam = normalizeScoreWeightsByTeam(
      state.session?.score_weights_by_team || {},
      integer(state.session?.num_teams, 0)
    );
    scoreWeightsByTeam[teamIndex] = normalizeScoreWeights();
    await saveSession({ ...state.session, score_weights_by_team: scoreWeightsByTeam });
    writeScoreWeightInputs();
    setStatus(`Reset score weights for ${teamName(teamIndex)}.`);
  }

  // Read the Draft Intel position-window editor into normalized preferences.
  function readPositionPreferenceInputs() {
    const preferences = {};
    for (const position of positionPreferencePositions) {
      preferences[position] = {};
      for (const field of positionPreferenceFields) {
        const value = integer($(positionPreferenceInputIds[position][field]).value, 0);
        preferences[position][field] = value > 0 ? value : null;
      }
    }
    return normalizePositionPreferences(preferences);
  }

  // Write one team's Draft Intel position windows into the editor.
  function writePositionPreferenceInputs() {
    const teamIndex = selectedScoreWeightTeamIndex();
    const preferences = positionPreferencesForTeam(teamIndex);
    for (const position of positionPreferencePositions) {
      for (const field of positionPreferenceFields) {
        $(positionPreferenceInputIds[position][field]).value = preferences[position][field] || "";
      }
    }
  }

  // Persist the current Draft Intel position windows for the selected team.
  async function savePositionPreferenceInputs({ announce = true } = {}) {
    const teamIndex = selectedScoreWeightTeamIndex();
    const positionPreferencesByTeam = normalizePositionPreferencesByTeam(
      state.session?.position_preferences_by_team || {},
      integer(state.session?.num_teams, 0)
    );
    positionPreferencesByTeam[teamIndex] = readPositionPreferenceInputs();
    await saveSession({ ...state.session, position_preferences_by_team: positionPreferencesByTeam });
    if (announce) setStatus(`Saved position windows for ${teamName(teamIndex)}.`);
  }

  // Clear Draft Intel position windows for the selected team.
  async function resetPositionPreferenceInputs() {
    const teamIndex = selectedScoreWeightTeamIndex();
    const positionPreferencesByTeam = normalizePositionPreferencesByTeam(
      state.session?.position_preferences_by_team || {},
      integer(state.session?.num_teams, 0)
    );
    positionPreferencesByTeam[teamIndex] = normalizePositionPreferences();
    await saveSession({ ...state.session, position_preferences_by_team: positionPreferencesByTeam });
    writePositionPreferenceInputs();
    setStatus(`Reset position windows for ${teamName(teamIndex)}.`);
  }

  // Build favorite-team options from cached player NFL team abbreviations.
  function availableNflTeams() {
    return [...new Set(state.players
      .map((player) => String(player.pro_team || "").trim().toUpperCase())
      .filter((team) => team && team !== "NONE"))]
      .sort((a, b) => a.localeCompare(b));
  }

  // Refresh favorite-team choices from the current player cache.
  function refreshFavoriteNflTeamOptions() {
    const selected = new Set(Array.from(document.querySelectorAll("[data-favorite-nfl-team]:checked")).map((input) => input.value));
    $("favoriteNflTeams").innerHTML = availableNflTeams()
      .map((team) => `<label class="favorite-team-option"><input data-favorite-nfl-team type="checkbox" value="${escapeHtml(team)}" ${selected.has(team) ? "checked" : ""}>${escapeHtml(team)}</label>`)
      .join("");
  }

  // Read selected favorite NFL teams for the active mock drafter.
  function readFavoriteTeamInputs() {
    return normalizeFavoriteNflTeams(Array.from(document.querySelectorAll("[data-favorite-nfl-team]:checked")).map((input) => input.value));
  }

  // Write one mock drafter's favorite NFL teams into the multi-select.
  function writeFavoriteTeamInputs() {
    const teamIndex = selectedScoreWeightTeamIndex();
    const favorites = new Set(favoriteNflTeamsForTeam(teamIndex));
    refreshFavoriteNflTeamOptions();
    for (const input of document.querySelectorAll("[data-favorite-nfl-team]")) {
      input.checked = favorites.has(input.value);
    }
  }

  // Persist favorite NFL teams for the selected mock drafter.
  async function saveFavoriteTeamInputs({ announce = true } = {}) {
    const teamIndex = selectedScoreWeightTeamIndex();
    const favoritesByTeam = normalizeFavoriteNflTeamsByTeam(
      state.session?.favorite_nfl_teams_by_team || {},
      integer(state.session?.num_teams, 0)
    );
    favoritesByTeam[teamIndex] = readFavoriteTeamInputs();
    await saveSession({ ...state.session, favorite_nfl_teams_by_team: favoritesByTeam });
    if (announce) setStatus(`Saved favorite NFL teams for ${teamName(teamIndex)}.`);
  }

  // Clear favorite NFL teams for the selected mock drafter.
  async function resetFavoriteTeamInputs() {
    const teamIndex = selectedScoreWeightTeamIndex();
    const favoritesByTeam = normalizeFavoriteNflTeamsByTeam(
      state.session?.favorite_nfl_teams_by_team || {},
      integer(state.session?.num_teams, 0)
    );
    favoritesByTeam[teamIndex] = [];
    await saveSession({ ...state.session, favorite_nfl_teams_by_team: favoritesByTeam });
    writeFavoriteTeamInputs();
    setStatus(`Reset favorite NFL teams for ${teamName(teamIndex)}.`);
  }

  // Read draft setup controls into a normalized session object.
  function readSetupInputs() {
    const numTeams = Math.max(integer($("numTeams").value, state.session.num_teams), 2);
    const previousNumTeams = integer(state.session?.num_teams, numTeams);
    const playoffTeams = clamp(integer($("playoffTeams").value, state.session.playoff_team_count || Math.min(6, numTeams)), 2, numTeams);
    const playoffByes = clamp(integer($("playoffByes").value, state.session.playoff_bye_count || 0), 0, Math.max(playoffTeams - 2, 0));
    const names = teamNamesFromSources({
      teams: state.session.teams || [],
      teamNames: state.session.team_names || [],
      numTeams,
    });
    const teams = teamsWithResolvedNames(state.session, names);
    return {
      ...state.session,
      num_teams: numTeams,
      team_names: names,
      teams,
      draft_slots: previousNumTeams === numTeams ? clone(state.session.draft_slots || []) : [],
      score_weights_by_team: normalizeScoreWeightsByTeam(
        {
          ...(state.session.score_weights_by_team || {}),
          [selectedScoreWeightTeamIndex()]: readScoreWeightInputs(),
        },
        numTeams
      ),
      position_preferences_by_team: normalizePositionPreferencesByTeam(
        {
          ...(state.session.position_preferences_by_team || {}),
          [selectedScoreWeightTeamIndex()]: readPositionPreferenceInputs(),
        },
        numTeams
      ),
      favorite_nfl_teams_by_team: normalizeFavoriteNflTeamsByTeam(
        {
          ...(state.session.favorite_nfl_teams_by_team || {}),
          [selectedScoreWeightTeamIndex()]: readFavoriteTeamInputs(),
        },
        numTeams
      ),
      human_team_index: Math.max(Math.min(integer($("humanTeam").value, state.session.human_team_index), numTeams - 1), 0),
      playoff_team_count: playoffTeams,
      playoff_bye_count: playoffByes,
    };
  }

  // Write the active session into the setup form controls.
  function writeSetupInputs() {
    const session = state.session || defaultSession();
    $("numTeams").value = session.num_teams;
    const playoffTeams = clamp(integer(session.playoff_team_count, Math.min(6, session.num_teams)), 2, session.num_teams);
    $("playoffTeams").value = playoffTeams;
    $("playoffByes").value = clamp(integer(session.playoff_bye_count, 0), 0, Math.max(playoffTeams - 2, 0));
    refreshTradeTeamOptions();
    $("humanTeam").value = String(integer(session.human_team_index, 0));
    writeScoreWeightInputs();
    writePositionPreferenceInputs();
    writeFavoriteTeamInputs();
  }

  // Write the active projection sync settings into the sync form controls.
  function writeProjectionInputs() {
    const session = state.session || defaultSession();
    $("leagueId").value = session.league_id || defaultConfig.league_id || "";
    $("year").value = integer(session.year ?? defaultConfig.year, new Date().getFullYear());
    $("weekStart").value = integer(session.week_start ?? defaultConfig.week_start, 1);
    $("weekEnd").value = integer(session.week_end ?? defaultConfig.week_end, 17);
  }

  // Refresh all team selectors after session team names change.
  function refreshTradeTeamOptions() {
    const names = state.session?.team_names || [];
    const options = names.map((name, index) => `<option value="${index}">${escapeHtml(index + 1)}. ${escapeHtml(name)}</option>`).join("");
    const tradeTeamA = $("tradeTeamA").value;
    const tradeTeamB = $("tradeTeamB").value;
    const rosterTeam = $("rosterTeam").value;
    const scoreWeightTeam = $("scoreWeightTeam").value;
    const humanTeam = String(integer(state.session?.human_team_index, 0));
    $("tradeTeamA").innerHTML = options;
    $("tradeTeamB").innerHTML = options;
    $("humanTeam").innerHTML = options;
    $("rosterTeam").innerHTML = options;
    $("scoreWeightTeam").innerHTML = options;
    $("tradeTeamA").value = tradeTeamA || "0";
    $("tradeTeamB").value = tradeTeamB || String(Math.min(1, Math.max(names.length - 1, 0)));
    $("humanTeam").value = humanTeam;
    $("rosterTeam").value = rosterTeam || humanTeam;
    $("scoreWeightTeam").value = scoreWeightTeam || humanTeam;
    if ($("rosterTeam").value === "") $("rosterTeam").value = humanTeam;
    if ($("scoreWeightTeam").value === "") $("scoreWeightTeam").value = humanTeam;
  }

  // Generate a default snake draft slot list from the active roster size.
  function generateDraftSlots(session = state.session) {
    const slots = [];
    const numTeams = session.num_teams;
    const rounds = totalRosterSlots(session);
    let overall = 1;
    for (let round = 1; round <= rounds; round += 1) {
      const order = Array.from({ length: numTeams }, (_, index) => index);
      if (round % 2 === 0) order.reverse();
      for (let pickIndex = 0; pickIndex < order.length; pickIndex += 1) {
        const originalTeam = order[pickIndex];
        slots.push({
          overall,
          round,
          pick_in_round: pickIndex + 1,
          original_team: originalTeam,
          current_team: originalTeam,
        });
        overall += 1;
      }
    }
    return slots;
  }

  // Start from synced ESPN slots when available, otherwise generated slots.
  function baseDraftSlots(session = state.session) {
    const generated = generateDraftSlots(session);
    const espnSlots = Array.isArray(session.draft_slots) ? session.draft_slots : [];
    if (!espnSlots.length) return generated;
    const byOverall = new Map(espnSlots.map((slot) => [integer(slot.overall), slot]));
    return generated.map((slot) => {
      const espnSlot = byOverall.get(slot.overall);
      if (!espnSlot) return slot;
      return {
        ...slot,
        original_team: Math.max(Math.min(integer(espnSlot.original_team, slot.original_team), session.num_teams - 1), 0),
        current_team: Math.max(Math.min(integer(espnSlot.current_team, slot.current_team), session.num_teams - 1), 0),
      };
    });
  }

  // Apply saved and temporary pick trades to a base slot list.
  function applyTradesToSlots(baseSlots, trades) {
    const slots = baseSlots.map((slot) => ({ ...slot }));
    const lookup = new Map(slots.map((slot) => [`${slot.round}:${slot.original_team}`, slot]));
    for (const trade of trades) {
      tradeMoves(trade).forEach((move) => {
        const slot = lookup.get(`${move.ref.round}:${move.ref.original_team}`);
        if (slot) slot.current_team = move.to_team;
      });
    }
    return slots;
  }

  // Convert old and current trade shapes into uniform pick movements.
  function tradeMoves(trade) {
    if (Array.isArray(trade.team_a_pick_refs) || Array.isArray(trade.team_b_pick_refs)) {
      return [
        ...(trade.team_a_pick_refs || []).map((ref) => ({ ref, from_team: trade.team_a, to_team: trade.team_b })),
        ...(trade.team_b_pick_refs || []).map((ref) => ({ ref, from_team: trade.team_b, to_team: trade.team_a })),
      ];
    }
    return (trade.pick_refs || []).map((ref) => ({ ref, from_team: trade.from_team, to_team: trade.to_team }));
  }

  // Recompute draft slots from trades and optionally persist them.
  async function rebuildSlotsFromTrades({ persist = true } = {}) {
    state.slots = applyTradesToSlots(baseDraftSlots(state.session), state.trades);
    if (persist) {
      const persisted = applyTradesToSlots(baseDraftSlots(state.session), state.trades.filter((trade) => !trade.temporary));
      await db.draft_slots.clear();
      await db.draft_slots.bulkPut(persisted);
    }
  }

  // Reset picks and optionally trades after setup changes.
  async function resetDraftBoard({ keepTrades = false } = {}) {
    state.session = readSetupInputs();
    state.session.draft_started = false;
    await saveSession(state.session);
    await db.draft_picks.clear();
    state.picks = [];
    if (!keepTrades) {
      await db.pick_trades.clear();
      state.trades = [];
    }
    await rebuildSlotsFromTrades();
    state.results = null;
    await loadState();
  }

  // Persist playoff settings and rerender projected results.
  async function savePlayoffSettings() {
    const playoffTeams = clamp(integer($("playoffTeams").value, state.session.playoff_team_count || Math.min(6, state.session.num_teams)), 2, state.session.num_teams);
    const playoffByes = clamp(integer($("playoffByes").value, state.session.playoff_bye_count || 0), 0, Math.max(playoffTeams - 2, 0));
    state.session = {
      ...state.session,
      playoff_team_count: playoffTeams,
      playoff_bye_count: playoffByes,
    };
    await saveSession(state.session);
    writeSetupInputs();
    state.results = simulateDraft();
    renderResults();
    setStatus("Playoff settings updated.");
  }

  // Return the next unfilled draft slot.
  function currentPick() {
    const picked = new Set(state.picks.map((pick) => pick.overall));
    return state.slots.find((slot) => !picked.has(slot.overall)) || null;
  }

  // Return a team's next unfilled draft slot after the provided overall pick.
  function nextPickForTeam(teamIndex, afterOverall) {
    const picked = new Set(state.picks.map((pick) => integer(pick.overall)));
    const team = integer(teamIndex);
    return state.slots.find(
      (slot) =>
        integer(slot.current_team) === team &&
        integer(slot.overall) > integer(afterOverall, -1) &&
        !picked.has(integer(slot.overall))
    ) || null;
  }

  // Build a lookup of cached players keyed by id.
  function playerByIdMap() {
    return new Map(state.players.map((player) => [String(player.player_id), player]));
  }

  // Convert draft picks into per-team roster arrays.
  function rostersByTeam() {
    const players = playerByIdMap();
    const rosters = Array.from({ length: state.session?.num_teams || 0 }, () => []);
    const seenPlayers = new Set();
    for (const pick of state.picks) {
      if (isSkippedPick(pick)) continue;
      const playerId = String(pick.player_id);
      if (seenPlayers.has(playerId)) continue;
      seenPlayers.add(playerId);
      const player = players.get(String(pick.player_id));
      if (player && rosters[pick.team_index]) rosters[pick.team_index].push(player);
    }
    return rosters;
  }

  // Count rostered players by fantasy position.
  function rosterCounts(roster) {
    const counts = Object.fromEntries(positionOrder.map((position) => [position, 0]));
    for (const player of roster) counts[player.position] = (counts[player.position] || 0) + 1;
    return counts;
  }

  // Return the maximum rosterable players for one position.
  function positionLimit(position, session = state.session) {
    const starters = integer(session.roster_settings?.[position], 0);
    const extra = integer(session.max_extra_per_position?.[position], 0);
    return starters + extra;
  }

  // Check whether a player can legally fit on a team's roster.
  function canAddPlayer(teamIndex, player, rosters = rostersByTeam()) {
    if (!player) return false;
    if (draftedPlayerIds().has(String(player.player_id))) return false;
    const roster = rosters[teamIndex] || [];
    if (roster.length >= totalRosterSlots()) return false;
    const limit = positionLimit(player.position);
    if (limit > 0 && rosterCounts(roster)[player.position] >= limit) return false;
    return true;
  }

  // Score how strongly a position fills open starter or flex needs.
  function needTier(teamIndex, position, rosters = rostersByTeam()) {
    const roster = rosters[teamIndex] || [];
    const counts = rosterCounts(roster);
    const starters = integer(state.session.roster_settings?.[position], 0);

    const openStarterSlots = Math.max(starters - integer(counts[position], 0), 0);

    const flexCapacity = integer(state.session.roster_settings?.FLEX, 0);
    const flexWeightByPosition = {
      RB: 1,
      WR: 1,
      TE: 0.1,
    };

    const flexWeight = flexWeightByPosition[position] || 0;
    let flexValue = 0;

    if(flexWeight > 0 && flexCapacity > 0) {
      const eligiblePositions = ["RB", "WR", "TE"];
      const eligibleStarters = eligiblePositions.reduce((sum, pos) => sum + integer(state.session.roster_settings?.[pos], 0), 0);

      const eligibleDrafted = eligiblePositions.reduce((sum, pos) => sum + integer(counts[pos], 0), 0);

      const flexUsed = Math.max(eligibleDrafted - eligibleStarters, 0);
      const flexOpen = Math.max(flexCapacity - flexUsed, 0);

      flexValue = flexOpen * flexWeight;
    }
    return openStarterSlots + flexValue;
  }

  // List every cached player who has not been drafted.
  function undraftedPlayers() {
    const drafted = draftedPlayerIds();
    return state.players.filter((player) => !drafted.has(String(player.player_id)));
  }

  // List legal candidates for the team currently on the clock.
  function draftablePlayersForCurrentPick() {
    const pick = currentPick();
    if (!pick) return [];
    const rosters = rostersByTeam();
    return undraftedPlayers().filter((player) => canAddPlayer(pick.current_team, player, rosters));
  }

  // Sort one position by projected total points for lineup and bot scoring.
  function positionSortedPlayers(players, position) {
    return players
    .filter((p) => p.position === position)
    .sort((a, b) => number(b.projected_total_pts) - number(a.projected_total_pts));
  }

  // Resolve league size for calculations, tolerating older IndexedDB sessions.
  function sessionTeamCount(session = state.session) {
    return Math.max(
      integer(session?.num_teams, 0),
      Array.isArray(session?.team_names) ? session.team_names.length : 0,
      Array.isArray(session?.teams) ? session.teams.length : 0,
      integer(defaultConfig.num_teams, 0)
    );
  }

  // Return the one-based VOR replacement rank for one position.
  function replacementRankForPosition(position, session = state.session) {
    if (position === "K" || position === "DEF") return 1;
    const numTeams = sessionTeamCount(session);
    const starters = integer(session?.roster_settings?.[position], 0);
    const flexSlots = integer(session?.roster_settings?.FLEX, 0);
    const flexShare = replacementFlexShareByPosition[position] ?? 0;
    return Math.max(Math.ceil(numTeams * (starters + flexSlots * flexShare)), 1);
  }

  // Build replacement-level baselines for each position.
  function replacementPointsByPosition(players) {
    const out = {};
    const positions = [...new Set(players.map((p) => p.position))];

    for (const position of positions) {
      const sorted = positionSortedPlayers(players, position);
      if (sorted.length === 0) {
        out[position] = 0;
        continue;
      }
      const replacementIndex = replacementRankForPosition(position) - 1;
      const idx = Math.min(replacementIndex, sorted.length - 1);
      out[position] = number(sorted[idx]?.projected_total_pts ?? 0);
    }
    return out;
  }

// Score a player against the replacement baseline for their position.
function vorScore(player, replacementByPosition) {
  const replacement = replacementByPosition[player.position] ?? 0;
  return number(player.projected_total_pts) - replacement;
}

// Estimate how much same-position value may disappear before a team's next pick.
function dropoffScore(player, candidates, teamIndex) {
  // drop off is the difference between this player and the player at the same position who is expected to be available at the next pick for this team.
  // e.g. in scoring dropoff for a WR at pick 15 (#2.1), the dropoff is the difference between this WR and the WR expected to be available at pick 42 (#3.14).
  
  const sortedByPos = positionSortedPlayers(candidates, player.position);
  const currentPosIdx = sortedByPos.findIndex((p) => p.player_id === player.player_id);
  if (currentPosIdx < 0 || currentPosIdx >= sortedByPos.length - 1) return 0;

  const current = currentPick();
  const currentOverall = integer(current?.overall, -1);

  const nextTeamPick = nextPickForTeam(teamIndex, currentOverall);
  if (!nextTeamPick) return 0;

  const overallGap = Math.max(integer(nextTeamPick.overall) - currentOverall - 1, 0);

  // Global board = the players most likely to be drafted before our next turn.
  const globalBoard = sortPlayers(candidates);

  // How many players at this position are likely to disappear before we pick again?
  const expectedPosLoss = globalBoard
    .slice(0, overallGap)
    .filter((p) => p.position === player.position)
    .length;

  const nextPosIdx = Math.min(currentPosIdx + expectedPosLoss + 1, sortedByPos.length - 1);
  const playerAtNextPick = sortedByPos[nextPosIdx];

  if (!playerAtNextPick) return 0;

  const gap = Math.max(
    number(player.projected_total_pts) - number(playerAtNextPick.projected_total_pts),
    0
  );

  return gap;
}

// Add a roster-specific bonus for backup RBs behind a rostered starter.
function handcuffBonus(player, roster) {
  if (player.position !== "RB") return 0;

  const team = (player.pro_team || "").toUpperCase();
  if (!team) return 0;

  // Find the top projected RB on that NFL team.
  const teamRBs = state.players
  .filter((p) => p.position === "RB" && p.pro_team === team)
  .sort((a, b) => number(b.projected_total_pts) - number(a.projected_total_pts));

  if (teamRBs.length < 2) return 0;

  const starter = teamRBs[0];
  const backup = teamRBs.find((p) => p.player_id === player.player_id);
  if (!backup || starter.player_id === player.player_id) return 0;

  const rosterHasStarter = roster.some((p) => p.player_id === starter.player_id);
  if (!rosterHasStarter) return 0;

  // Small bonus, larger when the starter is more valuable.
  // e.g. if the starter is projected for 300 points and the backup for 100, the bonus is (300 - 100) * 0.075 = 15 points.
  // e.g. Bijan proj 315.1, backup Brian Robinson proj 89.7, bonus = (315.1 - 89.7) * 0.075 = 16.9 points.
  // thus for a team with Bijan, drafting Robinson is worth ~17 points of added value.
  return Math.max(number(starter.projected_total_pts) - number(backup.projected_total_pts), 0) * 0.075; 
}

// Add a roster-specific correlation bonus for useful NFL-team stacks.
function stackBonus(player, roster) {
  // Gives stack bonus for a player who has a teammate on the same NFL team in the roster so far.
  // e.g. if you have a WR from some team, then that team's QB is more valuable to you at a value of 8 additional points.
  // for QB + TE the bonus is 6 points, for RB + WR the bonus is 4 points.

  const team = (player.pro_team || "").toUpperCase();
  if (!team) return 0;

  const sameTeamRoster = roster.filter((p) => p.pro_team === team && p.player_id !== player.player_id);
  if (sameTeamRoster.length === 0) return 0;

  let bonus = 0;

  for (const mate of sameTeamRoster) {
    const pairA = player.position;
    const pairB = mate.position;

    // QB + WR
    if (
      (pairA === "QB" && (pairB === "WR")) ||
      (pairB === "QB" && (pairA === "WR"))
    ) {
      bonus = Math.max(bonus, 8.0);
    }

    //  QB + TE is weaker, but still useful as a minor correlation play
    if (
      (pairA === "QB" && pairB === "TE") ||
      (pairB === "QB" && pairA === "TE")
    ) {
      bonus = Math.max(bonus, 6.0);
    }

    // RB + WR is weaker, but still useful as a minor correlation play
    if (
      (pairA === "RB" && pairB === "WR") ||
      (pairA === "WR" && pairB === "RB")
    ) {
      bonus = Math.max(bonus, 4.0);
    }

  }

  return bonus;
}

// Convert board rank into a normalized positive score within the candidate pool.
function rankScore(player, candidatePool) {
  // lower rank is better, so we want to give a higher score for lower rank.
  // e.g. rank 1 should be worth more than rank 10.
  // also being the top ranked player on the board currently should have a spike as compared to being the second best, even

  const rank = integer(player.rank, 0);
  const poolRanks = [...new Set(
    candidatePool.map((p) => integer(p.rank, 0)).filter((r) => r > 0))]
    .sort((a, b) => a - b);

  const index = poolRanks.indexOf(rank);
  if (index < 0) return 0;

  return 1 / (index + 1);
}

// Convert ADP into a normalized positive score within the candidate pool.
function adpScore(player, candidatePool) {
  // same implementation as rankScore, but using ADP instead of rank.
  const adp = number(player.adp, 0);
  const poolAdps = [...new Set(
    candidatePool.map((p) => number(p.adp, 0)).filter((r) => r > 0))]
    .sort((a, b) => a - b);

  const index = poolAdps.indexOf(adp);
  if (index < 0) return 0;

  return 1 / (index + 1);
}

// Return an interpolated quantile from an already-sorted numeric list.
function quantile(sortedValues, percentile) {
  if (sortedValues.length === 0) return 0;
  if (sortedValues.length === 1) return sortedValues[0];

  const index = clamp(percentile, 0, 1) * (sortedValues.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

// Preserve sign while applying a power curve to normalized score distance.
function signedPower(value, gamma) {
  if (value === 0) return 0;
  return Math.sign(value) * Math.pow(Math.abs(value), gamma);
}

// Build a median/IQR normalizer so different score components share units.
function robustComponentNormalizer(values, gamma = 1) {
  const sorted = values
    .map((value) => number(value, NaN))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);

  if (sorted.length === 0) return () => 0;

  const median = quantile(sorted, 0.5);
  const q25 = quantile(sorted, 0.25);
  const q75 = quantile(sorted, 0.75);
  const spread = Math.max((q75 - q25) / 1.349, 1e-6);

  return (value) => {
    const z = (number(value) - median) / spread;
    return signedPower(clamp(z / 2.5, -3, 3), gamma);
  };
}

// Smoothly discount kickers and defenses until the final roster rounds.
function positionTimingMultiplier(position, round) {
  if (position !== "K" && position !== "DEF") return 1;

  const currentRound = Math.max(integer(round, 1), 1);
  const progress = clamp(currentRound / 15, 0, 1);
  return Math.max(Math.pow(progress, 6), 0.02);
}

// Keep elite-player value live for normal positions, but fully time K/DEF.
function untimedValueShare(position) {
  return position === "K" || position === "DEF" ? 0 : 0.5;
}

// Determine whether this pick would fill starter inventory or backup inventory.
function positionPreferenceStage(position, counts) {
  const starters = Math.max(integer(state.session?.roster_settings?.[position], 0), 1);
  return integer(counts?.[position], 0) < starters ? "first" : "backup";
}

// Convert one configured window into a small smooth score nudge.
function positionWindowAdjustment(window, round) {
  const earliest = window.earliest;
  const latest = window.latest;
  if (earliest != null && round < earliest) {
    const distance = clamp((earliest - round) / 3, 0, 1);
    return -positionPreferenceMaxAdjustment * Math.pow(distance, 1.25);
  }
  if (latest != null && round > latest) {
    const distance = clamp((round - latest) / 3, 0, 1);
    return positionPreferenceMaxAdjustment * Math.pow(distance, 1.1);
  }
  return 0;
}

// Softly nudge a bot toward or away from QB/TE/K/DEF Draft Intel windows.
function positionPreferenceAdjustment(teamIndex, player, pick, roster) {
  const position = String(player?.position || "");
  if (!positionPreferencePositions.includes(position)) return 0;

  const preferences = positionPreferencesForTeam(teamIndex);
  const counts = rosterCounts(roster);
  const stage = positionPreferenceStage(position, counts);
  const positionPrefs = preferences[position] || {};
  return positionWindowAdjustment({
    earliest: positionPrefs[`${stage}_earliest`],
    latest: positionPrefs[`${stage}_latest`],
  }, Math.max(integer(pick?.round, 1), 1));
}

// Add a small bonus when a mock drafter favors the player's NFL team.
function favoriteTeamAdjustment(teamIndex, player) {
  const proTeam = String(player?.pro_team || "").trim().toUpperCase();
  if (!proTeam) return 0;
  return favoriteNflTeamsForTeam(teamIndex).includes(proTeam) ? favoriteTeamScoreBonus : 0;
}

// Randomly sample from scored candidates while favoring higher scores.
function softmaxSample(items, scoreFn, temperature = 8) {
  // high temperature indicates more randomness, low temperature indicates more deterministic behavior.
  const scored = items.map((item) => scoreFn(item));
  const maxScore = Math.max(...scored);
  const weights = scored.map((score) => Math.exp((score - maxScore) / temperature));
  const total = weights.reduce((sum, w) => sum + w, 0);
  if (total <= 0) return items[0] || null;

  // for debug, print the scores and weights to terminal
/*fetch("/api/log", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    label: "softmax-debug",
    scores: scored,
    weights: weights,
    items: items.map((item) => ({
      player_id: item.player_id,
      player_name: item.player_name,
    })),
  }),
}).catch(() => {});*/


  let roll = Math.random() * total;
  let chosenIndex = items.length-1;

  for (let i = 0; i < items.length; i += 1) {
    roll -= weights[i];
    if (roll <= 0){
      chosenIndex=i;
      break;
    }
  }

  const ranked = items.map((item, index) => ({
    index, item, score: scored[index], weight: weights[index]
  })).sort((a, b) => b.weight - a.weight);
  const chosenRank = ranked.findIndex((entry) => entry.index === chosenIndex)+1;

  // if(chosenRank > 3){
  //   const payload = {
  //     label: "bot-pick-suboptimal",
  //     currentPick: currentPick(),
  //     chosenRank,
  //     chosenPlayer: {
  //       player_name: items[chosenIndex].player_name
  //     },
  //     all_items: ranked.map((entry) => ({
  //       player_name: entry.item.player_name,
  //       value: entry.score,
  //       vor: vorScore(entry.item, state.replacementBaselineByPosition),
  //     }))
  //   };

  //   fetch("/api/log", {
  //     method: "POST",
  //     headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify(payload),
  //   }).catch(() => {});
  // }

  return items[chosenIndex] || null;
}

// Choose a bot pick using value, roster need, ADP, rank, and roster synergy.
function chooseBotPick(teamIndex) {
  const pick = currentPick();
  if (!pick) return null;

  const rosters = rostersByTeam();
  const roster = rosters[teamIndex] || [];
  const candidates = draftablePlayersForCurrentPick();
  
  if (candidates.length === 0) return null;

  const candidatePool = sortPlayers(candidates).slice(0, 40);
  const replacementByPosition = state.replacementBaselineByPosition || replacementPointsByPosition(state.players);
  const weights = scoreWeightsForTeam(teamIndex);
  const rawComponentScores = candidatePool.map((player) => ({
    player_id: String(player.player_id),
    vor: vorScore(player, replacementByPosition),
    rank: rankScore(player, candidatePool),
    adp: adpScore(player, candidatePool),
  }));
  const rawComponentsByPlayerId = new Map(rawComponentScores.map((score) => [score.player_id, score]));
  const normalizeComponent = {
    vor: robustComponentNormalizer(rawComponentScores.map((score) => score.vor), 1.15),
    rank: robustComponentNormalizer(rawComponentScores.map((score) => score.rank), 1),
    adp: robustComponentNormalizer(rawComponentScores.map((score) => score.adp), 1),
  };

  // Score one candidate for the current bot pick.
  const scoreFn = (player) => {
    const need = needTier(teamIndex, player.position, rosters);
    const counts = rosterCounts(roster);
    //const benchPenalty =
    //  Math.max(1 + (counts[player.position] || 0) - integer(state.session.roster_settings?.[player.position], 0), 0) * 40;
    const rawComponents = rawComponentsByPlayerId.get(String(player.player_id)) || {
      vor: vorScore(player, replacementByPosition),
      rank: rankScore(player, candidatePool),
      adp: adpScore(player, candidatePool),
    };
    const vor = rawComponents.vor;
    const dropoff = dropoffScore(player, candidates, teamIndex);
    const handcuff = handcuffBonus(player, roster);
    const stack = stackBonus(player, roster);
    const vorVal = normalizeComponent.vor(vor);
    const rankValRaw = rawComponents.rank;
    const rankVal = normalizeComponent.rank(rankValRaw);
    const adpValRaw = rawComponents.adp;
    const adpVal = normalizeComponent.adp(adpValRaw);
    const timing = positionTimingMultiplier(player.position, pick.round);
    const positionPreference = positionPreferenceAdjustment(teamIndex, player, pick, roster);
    const favoriteTeam = favoriteTeamAdjustment(teamIndex, player);
    const coreShare = untimedValueShare(player.position);
    const timedShare = 1 - coreShare;
    const coreValue = (
      vorVal * weights.vor * coreShare
      + rankVal * weights.rank * coreShare
      + adpVal * weights.adp * coreShare
    );
    const timedValue = (
      vorVal * weights.vor * timedShare
      + need * weights.need
      + dropoff * weights.dropoff
      + handcuff * weights.handcuff
      + stack * weights.stack
      + rankVal * weights.rank * timedShare
      + adpVal * weights.adp * timedShare
      + positionPreference * weights.positionPreference
      + favoriteTeam * weights.favoriteTeam
    );

    const score = coreValue + timedValue * timing;
    /*fetch("/api/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        label: "softmax-debug",
        player_name: player.player_name,
        vor: vor,
        vorVal: vorVal,
        need: need,
        dropoff: dropoff,
        handcuff: handcuff,
        stack: stack,
        rank: player.rank,
        rankValRaw: rankValRaw,
        rankVal: rankVal,
        adp: player.adp,
        adpValRaw: adpValRaw,
        adpVal: adpVal,
        timing: timing,
        positionPreference: positionPreference,
        favoriteTeam: favoriteTeam,
        positionPreferenceWeight: weights.positionPreference,
        favoriteTeamWeight: weights.favoriteTeam,
        //coreShare: coreShare,
        coreValue: coreValue,
        timedValue: timedValue,
        //weights: weights,
        score: score,
      }),
    }).catch(() => {});//*/

    return score;
  };
  // Sample instead of argmax.
  return softmaxSample(candidatePool, scoreFn);
}

  // Update in-memory picks with one normalized draft pick.
  function upsertLocalDraftPick(draftPick) {
    state.picks = state.picks.filter((pick) => pick.overall !== draftPick.overall);
    state.picks.push(draftPick);
    state.picks.sort((a, b) => a.overall - b.overall);
  }

  // Persist one draft pick after checking for duplicate players.
  async function persistDraftPick(draftPick) {
    const normalized = {
      ...draftPick,
      overall: integer(draftPick.overall),
      player_id: String(draftPick.player_id),
    };

    const existingOverall = await db.draft_picks.get(normalized.overall);
    if (existingOverall) {
      throw new Error(`Pick ${normalized.overall} has already been made.`);
    }

    if (!isSkippedPick(normalized)) {
      const existingPicks = await db.draft_picks.toArray();
      const existingPlayer = existingPicks.find((pick) => (
        !isSkippedPick(pick)
        && String(pick.player_id) === normalized.player_id
        && integer(pick.overall) !== normalized.overall
      ));
      if (existingPlayer) {
        const player = playerByIdMap().get(normalized.player_id);
        throw new Error(`${player?.player_name || "That player"} was already drafted at pick ${existingPlayer.overall}.`);
      }
    }

    await db.draft_picks.put(normalized);
    upsertLocalDraftPick(normalized);
  }

  // Serialize draft actions so rapid clicks and bot picks cannot overlap.
  async function withDraftLock(action) {
    if (state.draftBusy) return false;
    state.draftBusy = true;
    render();
    try {
      await action();
      return true;
    } catch (error) {
      const repairedPicks = await repairDraftPicksFromDb();
      state.picks = repairedPicks.picks;
      setStatus(error.message, true);
      return false;
    } finally {
      state.draftBusy = false;
      state.results = simulateDraft();
      render();
    }
  }

  // Run a human or auto pick inside the draft lock.
  async function makePick(playerId, source = "human") {
    return withDraftLock(() => makePickUnlocked(playerId, source));
  }

  // Apply one pick assuming the caller already owns the draft lock.
  async function makePickUnlocked(playerId, source = "human") {
    const pick = currentPick();
    if (!pick) return;
    if (!state.session?.draft_started) {
      setStatus("Start the draft before making picks.", true);
      return;
    }
    if (state.picks.some((row) => String(row.player_id) === String(playerId))) {
      setStatus("That player has already been drafted.", true);
      return;
    }
    const player = state.players.find((row) => String(row.player_id) === String(playerId));
    if (!canAddPlayer(pick.current_team, player)) {
      setStatus("That player is unavailable for this roster.", true);
      return;
    }
    const draftPick = {
      overall: pick.overall,
      round: pick.round,
      pick_in_round: pick.pick_in_round,
      team_index: pick.current_team,
      player_id: String(player.player_id),
      source,
      timestamp: new Date().toISOString(),
    };
    await persistDraftPick(draftPick);
    await autoAdvanceBotsUnlocked();
  }

  // Advance bot picks inside the draft lock.
  async function autoAdvanceBots() {
    return withDraftLock(autoAdvanceBotsUnlocked);
  }

  // Continue drafting bot turns until the human team or draft end is reached.
  async function autoAdvanceBotsUnlocked() {
    if (!state.session?.draft_started) return;
    let guard = 0;
    while (guard < 500) {
      guard += 1;
      const pick = currentPick();
      if (!pick || pick.current_team === state.session.human_team_index) return;
      const player = chooseBotPick(pick.current_team);
      if (!player) {
        await persistDraftPick({
          overall: pick.overall,
          round: pick.round,
          pick_in_round: pick.pick_in_round,
          team_index: pick.current_team,
          player_id: `__skip_${pick.overall}`,
          source: "skip",
          timestamp: new Date().toISOString(),
          skipped: true,
          skip_reason: "No legal bot pick",
        });
        continue;
      }
      await persistDraftPick({
        overall: pick.overall,
        round: pick.round,
        pick_in_round: pick.pick_in_round,
        team_index: pick.current_team,
        player_id: String(player.player_id),
        source: "bot",
        timestamp: new Date().toISOString(),
      });
    }
  }

  // Auto-pick for the current slot, including the human team when requested.
  async function autoPickCurrent() {
    return withDraftLock(async () => {
      const pick = currentPick();
      if (!pick) return;
      if (!state.session?.draft_started) {
        setStatus("Start the draft before making picks.", true);
        return;
      }
      if (pick.current_team !== state.session.human_team_index) {
        await autoAdvanceBotsUnlocked();
        return;
      }
      const player = chooseBotPick(pick.current_team) || draftablePlayersForCurrentPick()[0];
      if (!player) {
        setStatus("No legal player is available for the current pick.", true);
        return;
      }
      await makePickUnlocked(player.player_id, pick.current_team === state.session.human_team_index ? "auto" : "bot");
    });
  }

  // Resume bot picks after reload when a bot is still on the clock.
  async function resumeBotDraftIfNeeded() {
    const pick = currentPick();
    if (state.session?.draft_started && pick && pick.current_team !== state.session.human_team_index) {
      await autoAdvanceBots();
    }
  }

  // Build a player-week projection lookup from cached weekly rows.
  function weeklyProjectionMap() {
    const map = new Map();
    for (const row of state.weekly) {
      map.set(`${row.player_id}:${row.week}`, number(row.projected_points));
    }
    return map;
  }

  // Pick the projected optimal lineup for one roster in one week.
  function optimalLineup(roster, week, scores) {
    let remaining = roster
      .map((player) => ({ ...player, points: scores.get(`${player.player_id}:${week}`) || 0 }))
      .sort((a, b) => b.points - a.points);
    const starters = [];
    // Fill positional starter slots before flex slots.
    const take = (position, count, slotName) => {
      const matching = remaining.filter((player) => player.position === position).slice(0, count);
      starters.push(...matching.map((player) => ({ ...player, slot: slotName })));
      const taken = new Set(matching.map((player) => player.player_id));
      remaining = remaining.filter((player) => !taken.has(player.player_id));
    };
    for (const position of positionOrder) take(position, integer(state.session.roster_settings?.[position], 0), position);
    const flexCount = integer(state.session.roster_settings?.FLEX, 0);
    if (flexCount > 0) {
      const flex = remaining.filter((player) => flexPositions.has(player.position)).slice(0, flexCount);
      starters.push(...flex.map((player) => ({ ...player, slot: "FLEX" })));
    }
    return {
      starters,
      total_points: starters.reduce((sum, player) => sum + number(player.points), 0),
    };
  }

  // Generate a round-robin schedule when ESPN does not provide matchups.
  function generateRoundRobinSchedule(numTeams, weeks) {
    const teams = Array.from({ length: numTeams }, (_, index) => index);
    if (numTeams % 2 === 1) teams.push(null);
    let rotation = teams.slice();
    const rounds = [];
    for (let roundIndex = 0; roundIndex < rotation.length - 1; roundIndex += 1) {
      const pairings = [];
      for (let offset = 0; offset < rotation.length / 2; offset += 1) {
        const left = rotation[offset];
        const right = rotation[rotation.length - 1 - offset];
        if (left == null || right == null) continue;
        pairings.push(roundIndex % 2 === 0 ? [left, right] : [right, left]);
      }
      rounds.push(pairings);
      rotation = [rotation[0], rotation[rotation.length - 1], ...rotation.slice(1, -1)];
    }
    return new Map(weeks.map((week, index) => [week, rounds[index % rounds.length] || []]));
  }

  // Select synced league schedule rows or fall back to generated pairings.
  function activeScheduleForWeeks(weeks) {
    const weekSet = new Set(weeks);
    const schedule = new Map();
    for (const row of state.schedule || []) {
      const week = integer(row.week);
      const home = integer(row.home_team_index);
      const away = integer(row.away_team_index);
      if (!weekSet.has(week) || home === away) continue;
      if (home < 0 || away < 0 || home >= state.session.num_teams || away >= state.session.num_teams) continue;
      if (!schedule.has(week)) schedule.set(week, []);
      schedule.get(week).push([home, away]);
    }
    return Array.from(schedule.values()).some((matchups) => matchups.length > 0)
      ? schedule
      : generateRoundRobinSchedule(state.session.num_teams, weeks);
  }

  // Label a playoff round based on teams still alive.
  function playoffRoundLabel(teamCount) {
    if (teamCount <= 2) return "Championship";
    if (teamCount <= 4) return "Semifinal";
    return "Quarterfinal";
  }

  // Resolve a head-to-head projected winner for one playoff matchup.
  function projectedWinner(home, away, week, scoreLookup) {
    const homeScore = scoreLookup.get(`${week}:${home.team_index}`) || 0;
    const awayScore = scoreLookup.get(`${week}:${away.team_index}`) || 0;
    return {
      homeScore,
      awayScore,
      winner: homeScore >= awayScore ? home : away,
    };
  }

  // Simulate the configured playoff bracket from regular-season standings.
  function simulatePlayoffs(standings, playoffWeeks, scoreLookup) {
    const playoffTeamCount = clamp(integer(state.session.playoff_team_count, Math.min(6, state.session.num_teams)), 2, state.session.num_teams);
    const playoffByeCount = clamp(integer(state.session.playoff_bye_count, 0), 0, Math.max(playoffTeamCount - 2, 0));
    let contenders = standings
      .slice(0, playoffTeamCount)
      .map((row) => ({ seed: row.rank, team_index: row.team_index, team: row.team }));
    const firstRoundByes = contenders.slice(0, playoffByeCount);
    let firstRoundParticipants = contenders.slice(playoffByeCount);
    const playoffMatchups = [];
    let champion = contenders[0] || null;

    for (const week of playoffWeeks) {
      if (contenders.length <= 1) break;
      const roundLabel = playoffRoundLabel(contenders.length);
      let participants = (week === playoffWeeks[0] ? firstRoundParticipants : contenders).slice().sort((a, b) => a.seed - b.seed);
      const roundByes = week === playoffWeeks[0] ? firstRoundByes.slice() : [];
      const winners = [];
      if (participants.length % 2 === 1) {
        const bye = participants.shift();
        if (bye) roundByes.push(bye);
      }
      for (const bye of roundByes) {
        playoffMatchups.push({
          week,
          round: roundLabel,
          home_seed: bye.seed,
          home: bye.team,
          home_score: "",
          away_seed: "",
          away: "BYE",
          away_score: "",
          winner: bye.team,
        });
      }
      while (participants.length >= 2) {
        const home = participants.shift();
        const away = participants.pop();
        const result = projectedWinner(home, away, week, scoreLookup);
        winners.push(result.winner);
        playoffMatchups.push({
          week,
          round: roundLabel,
          home_seed: home.seed,
          home: home.team,
          home_score: result.homeScore,
          away_seed: away.seed,
          away: away.team,
          away_score: result.awayScore,
          winner: result.winner.team,
        });
      }
      contenders = [...roundByes, ...winners].sort((a, b) => a.seed - b.seed);
      champion = contenders[0] || champion;
      firstRoundParticipants = contenders;
    }

    return { playoffMatchups, playoffChampion: champion };
  }

  // Simulate standings and playoffs from the current drafted rosters.
  function simulateDraft() {
    if (!state.session || state.picks.length !== state.slots.length || state.slots.length === 0) return null;
    const rosters = rostersByTeam();
    const weeks = Array.from(new Set(state.weekly.map((row) => integer(row.week)))).sort((a, b) => a - b);
    if (weeks.length === 0) return null;
    const playoffWeeks = weeks.length > 3 ? weeks.slice(-3) : [];
    const regularWeeks = playoffWeeks.length > 0 ? weeks.slice(0, -3) : weeks;
    const scores = weeklyProjectionMap();
    const weeklyTeamScores = [];
    for (const week of weeks) {
      for (let teamIndex = 0; teamIndex < state.session.num_teams; teamIndex += 1) {
        const lineup = optimalLineup(rosters[teamIndex] || [], week, scores);
        weeklyTeamScores.push({
          week,
          team_index: teamIndex,
          team: state.session.team_names[teamIndex],
          team_score: lineup.total_points,
        });
      }
    }
    const scoreLookup = new Map(weeklyTeamScores.map((row) => [`${row.week}:${row.team_index}`, row.team_score]));
    const schedule = activeScheduleForWeeks(regularWeeks);
    const matchupWeeks = regularWeeks.filter((week) => (schedule.get(week) || []).length > 0);
    const records = Array.from({ length: state.session.num_teams }, (_, index) => ({
      team_index: index,
      team: state.session.team_names[index],
      wins: 0,
      losses: 0,
      ties: 0,
      expected_wins: 0,
      points_for: 0,
      points_against: 0,
    }));
    const weeklyMatchups = [];
    for (const week of matchupWeeks) {
      for (const [home, away] of schedule.get(week)) {
        const homeScore = scoreLookup.get(`${week}:${home}`) || 0;
        const awayScore = scoreLookup.get(`${week}:${away}`) || 0;
        records[home].points_for += homeScore;
        records[away].points_for += awayScore;
        records[home].points_against += awayScore;
        records[away].points_against += homeScore;
        let winner = "Tie";
        if (homeScore > awayScore) {
          records[home].wins += 1;
          records[away].losses += 1;
          records[home].expected_wins += 1;
          winner = state.session.team_names[home];
        } else if (awayScore > homeScore) {
          records[away].wins += 1;
          records[home].losses += 1;
          records[away].expected_wins += 1;
          winner = state.session.team_names[away];
        } else {
          records[home].ties += 1;
          records[away].ties += 1;
          records[home].expected_wins += 0.5;
          records[away].expected_wins += 0.5;
        }
        weeklyMatchups.push({
          week,
          home: state.session.team_names[home],
          away: state.session.team_names[away],
          home_score: homeScore,
          away_score: awayScore,
          winner,
        });
      }
    }
    const standings = records
      .map((row) => {
        const games = row.wins + row.losses + row.ties || 1;
        return { ...row, win_pct: (row.wins + row.ties * 0.5) / games };
      })
      .sort((a, b) => b.win_pct - a.win_pct || b.points_for - a.points_for)
      .map((row, index) => ({ rank: index + 1, ...row }));
    const playoffs = playoffWeeks.length > 0 ? simulatePlayoffs(standings, playoffWeeks, scoreLookup) : { playoffMatchups: [], playoffChampion: null };
    return { weeks: [...matchupWeeks, ...playoffWeeks], regularWeeks: matchupWeeks, playoffWeeks, weeklyTeamScores, standings, weeklyMatchups, ...playoffs };
  }

  // Convert a draft slot into the compact trade-reference shape.
  function slotToRef(slot) {
    return {
      overall: slot.overall,
      round: slot.round,
      pick_in_round: slot.pick_in_round,
      original_team: slot.original_team,
    };
  }

  // Format a pick reference for saved-trade display.
  function pickRefLabel(ref) {
    return `#${ref.round}.${ref.pick_in_round}`;
  }

  // Guard trade edits after draft order has been locked by picks.
  function assertDraftOrderEditable() {
    if (state.session?.draft_started || state.picks.length > 0) {
      throw new Error("Pick trades must be set before the draft starts.");
    }
  }

  // Return a stable key for the current trade form inputs.
  function tradeFormKey() {
    return JSON.stringify({
      team_a: integer($("tradeTeamA").value, 0),
      team_b: integer($("tradeTeamB").value, 0),
      picks_a: $("tradePicksA").value.split(/[\n,;]+/).map((value) => value.trim()).filter(Boolean),
      picks_b: $("tradePicksB").value.split(/[\n,;]+/).map((value) => value.trim()).filter(Boolean),
      notes: $("tradeNotes").value.trim(),
    });
  }

  // Check whether either trade textarea currently has content.
  function hasPendingTradeForm() {
    return Boolean($("tradePicksA").value.trim() || $("tradePicksB").value.trim());
  }

  // Find an applied trade matching the current form contents.
  function matchingTradeForForm() {
    const key = tradeFormKey();
    return state.trades.find((trade) => trade.form_key === key) || null;
  }

  // Ensure a trade token refers to a pick currently owned by the side's team.
  function assertPickOwnedByTeam(slot, teamIndex, token) {
    if (!slot) throw new Error(`Unknown pick ${token}.`);
    if (slot.current_team !== teamIndex) {
      throw new Error(`${teamName(teamIndex)} does not currently own ${pickRefLabel(slot)}.`);
    }
  }

  // Parse one pick token from the trade form into a draft slot reference.
  function parsePickToken(token, teamIndex) {
    const value = token.trim();
    if (!value) return null;
    if (value.startsWith("#")) {
      const body = value.slice(1);
      const [roundText, pickText] = body.split(".");
      const round = integer(roundText);
      const pickInRound = pickText == null ? 0 : integer(pickText);
      let slot;
      if (pickInRound > 0) {
        slot = state.slots.find((row) => row.round === round && row.pick_in_round === pickInRound);
      } else {
        const owned = state.slots.filter((row) => row.round === round && row.current_team === teamIndex);
        const ownOriginal = owned.find((row) => row.original_team === teamIndex);
        if (owned.length > 1 && !ownOriginal) {
          throw new Error(`${teamName(teamIndex)} owns multiple round ${round} picks; use #${round}.pick.`);
        }
        slot = ownOriginal || owned[0];
      }
      assertPickOwnedByTeam(slot, teamIndex, value);
      return slotToRef(slot);
    }
    const overall = integer(value);
    const slot = state.slots.find((row) => row.overall === overall);
    assertPickOwnedByTeam(slot, teamIndex, value);
    return slotToRef(slot);
  }

  // Parse a newline-separated list of pick tokens for one team.
  function parseTradePickList(text, teamIndex) {
    const seen = new Set();
    return text
      .split(/[\n,;]+/)
      .map((token) => token.trim())
      .filter(Boolean)
      .map((token) => parsePickToken(token, teamIndex))
      .filter((ref) => {
        const key = `${ref.round}:${ref.original_team}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  }

  // Build a two-sided pick trade from the form.
  function buildTrade({ temporary = false } = {}) {
    assertDraftOrderEditable();
    const teamA = integer($("tradeTeamA").value, 0);
    const teamB = integer($("tradeTeamB").value, 0);
    if (teamA === teamB) throw new Error("Choose two different teams.");
    const teamAPicks = parseTradePickList($("tradePicksA").value, teamA);
    const teamBPicks = parseTradePickList($("tradePicksB").value, teamB);
    if (teamAPicks.length === 0 && teamBPicks.length === 0) {
      throw new Error("Add at least one pick to either side.");
    }
    return {
      id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
      label: `${teamName(teamA)} / ${teamName(teamB)}`,
      created_at: new Date().toISOString(),
      team_a: teamA,
      team_b: teamB,
      team_a_pick_refs: teamAPicks,
      team_b_pick_refs: teamBPicks,
      notes: $("tradeNotes").value.trim(),
      form_key: tradeFormKey(),
      temporary,
    };
  }

  // Apply a temporary trade preview without saving it.
  async function testTrade() {
    let trade;
    try {
      assertDraftOrderEditable();
      if (matchingTradeForForm()) {
        setStatus("That trade is already applied.");
        return;
      }
      trade = buildTrade({ temporary: true });
    } catch (error) {
      setStatus(error.message, true);
      return;
    }
    state.trades.push(trade);
    await rebuildSlotsFromTrades({ persist: false });
    setStatus("Trade tested for this session.");
    render();
  }

  // Persist the current trade form into IndexedDB.
  async function saveTrade() {
    let trade;
    try {
      assertDraftOrderEditable();
      const testedTrade = matchingTradeForForm();
      if (testedTrade?.temporary) {
        testedTrade.temporary = false;
        await db.pick_trades.put(testedTrade);
        await rebuildSlotsFromTrades({ persist: true });
        setStatus("Test trade saved locally.");
        render();
        return;
      }
      if (testedTrade) {
        setStatus("That trade is already saved.");
        return;
      }
      trade = buildTrade({ temporary: false });
    } catch (error) {
      setStatus(error.message, true);
      return;
    }
    state.trades.push(trade);
    await db.pick_trades.put(trade);
    await rebuildSlotsFromTrades({ persist: true });
    setStatus("Trade saved locally.");
    render();
  }

  // Delete a saved or temporary trade and rebuild draft slots.
  async function deleteTrade(id) {
    try {
      assertDraftOrderEditable();
    } catch (error) {
      setStatus(error.message, true);
      return;
    }
    const trade = state.trades.find((row) => row.id === id);
    state.trades = state.trades.filter((row) => row.id !== id);
    if (trade && !trade.temporary) await db.pick_trades.delete(id);
    await rebuildSlotsFromTrades({ persist: true });
    render();
  }

  // Normalize schedule rows from sync payloads into IndexedDB-ready objects.
  function normalizeLeagueSchedule(rows, teams) {
    const teamIndexById = new Map(teams.map((team, index) => [String(team.team_id), index]));
    const normalized = [];
    const seen = new Set();
    for (const row of rows || []) {
      const week = integer(row.week);
      const homeIndex = row.home_team_index == null
        ? teamIndexById.get(String(row.home_team_id))
        : integer(row.home_team_index);
      const awayIndex = row.away_team_index == null
        ? teamIndexById.get(String(row.away_team_id))
        : integer(row.away_team_index);
      if (week <= 0 || homeIndex == null || awayIndex == null || homeIndex === awayIndex) continue;
      if (homeIndex < 0 || awayIndex < 0 || homeIndex >= teams.length || awayIndex >= teams.length) continue;
      const key = `${week}:${Math.min(homeIndex, awayIndex)}:${Math.max(homeIndex, awayIndex)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      normalized.push({
        id: row.id || `${week}:${homeIndex}:${awayIndex}`,
        week,
        home_team_index: homeIndex,
        away_team_index: awayIndex,
        home_team_id: row.home_team_id == null ? teams[homeIndex]?.team_id : row.home_team_id,
        away_team_id: row.away_team_id == null ? teams[awayIndex]?.team_id : row.away_team_id,
        home_team: row.home_team || teams[homeIndex]?.team_name || fallbackTeamName(homeIndex),
        away_team: row.away_team || teams[awayIndex]?.team_name || fallbackTeamName(awayIndex),
        source: row.source || "espn_schedule",
      });
    }
    return normalized.sort((a, b) => integer(a.week) - integer(b.week) || integer(a.home_team_index) - integer(b.home_team_index));
  }

  // Save an ESPN or demo projection payload into local browser storage.
  async function saveProjectionPayload(payload, source) {
    const players = (payload.players || []).map((player) => ({
      ...player,
      player_id: String(player.player_id),
      rank: integer(player.rank),
      espn_rank: integer(player.espn_rank ?? player.rank),
      pos_rank: integer(player.pos_rank),
      adp: number(player.adp, -1),
      bye_week: integer(player.bye_week),
      projected_total_pts: number(player.projected_total_pts ?? player.season_total_pts),
    }));
    normalizeBoardRanks(players);
    fillMissingPositionRanks(players);
    const weekly = (payload.weekly_projections || []).map((row) => ({
      player_id: String(row.player_id),
      week: integer(row.week),
      projected_points: number(row.projected_points),
    }));
    const payloadNumTeams = payload.teams?.length || payload.team_names?.length || defaultConfig.num_teams || 10;
    const namesFromPayload = teamNamesFromSources({
      teams: payload.teams || [],
      teamNames: payload.team_names || [],
      numTeams: payloadNumTeams,
    });
    const teams = teamsWithResolvedNames({ teams: payload.teams || [] }, namesFromPayload);
    const names = teams.map((team) => team.team_name);
    const draftSlots = (payload.draft_slots || []).map((slot) => ({
      overall: integer(slot.overall),
      round: integer(slot.round),
      pick_in_round: integer(slot.pick_in_round),
      original_team: integer(slot.original_team),
      current_team: integer(slot.current_team),
    })).filter((slot) => slot.overall > 0 && slot.round > 0 && slot.pick_in_round > 0);
    const leagueSchedule = normalizeLeagueSchedule(payload.league_schedule || [], teams);
    const session = defaultSession({
      league_id: payload.request?.league_id || payload.league_id || "",
      year: payload.year || payload.request?.year || new Date().getFullYear(),
      week_start: payload.request?.week_start || defaultConfig.week_start || 1,
      week_end: payload.request?.week_end || defaultConfig.week_end || 17,
      synced_at: payload.synced_at || new Date().toISOString(),
      league_settings: payload.league_settings || {},
      num_teams: names.length,
      team_names: names,
      teams,
      draft_slots: draftSlots,
      projection_meta: payload.projection_meta || {},
      draft_started: false,
      score_weights_by_team: state.session?.score_weights_by_team || {},
      position_preferences_by_team: state.session?.position_preferences_by_team || {},
      favorite_nfl_teams_by_team: state.session?.favorite_nfl_teams_by_team || {},
      source,
    });
    await clearTables(["players", "weekly_projections", "league_schedule", "draft_slots", "pick_trades", "draft_picks"]);
    await db.players.bulkPut(players);
    await db.weekly_projections.bulkPut(weekly);
    if (leagueSchedule.length > 0) await db.league_schedule.bulkPut(leagueSchedule);
    await saveSession(session);
    state.session = session;
    state.players = sortPlayers(players);
    state.weekly = weekly;
    state.schedule = leagueSchedule;
    state.trades = [];
    state.picks = [];
    await rebuildSlotsFromTrades();
    await loadState();
  }

  // Lock draft slots and start bot drafting.
  async function startDraft() {
    if (state.draftBusy) return;
    if (!state.slots.length) {
      setStatus("Create a draft board before starting the draft.", true);
      return;
    }
    if (!currentPick()) return;
    if (hasPendingTradeForm() && !matchingTradeForForm()) {
      setStatus("Test or save the pending pick trade before starting the draft.", true);
      return;
    }
    await withDraftLock(async () => {
      await rebuildSlotsFromTrades({ persist: false });
      await db.draft_slots.clear();
      await db.draft_slots.bulkPut(state.slots);
      state.session.draft_started = true;
      await saveSession(state.session);
      await autoAdvanceBotsUnlocked();
      setStatus("Draft started.");
    });
  }

  // Build deterministic demo projections when ESPN sync is not available.
  function demoPayload() {
    const positions = { QB: 32, RB: 72, WR: 84, TE: 36, K: 32, DEF: 32 };
    const base = { QB: 285, RB: 210, WR: 200, TE: 145, K: 125, DEF: 130 };
    const players = [];
    const weekly = [];
    for (const [position, count] of Object.entries(positions)) {
      for (let index = 1; index <= count; index += 1) {
        const playerId = position === "DEF" ? `DEF_${index}` : `${position}_${index}`;
        const total = Math.max(base[position] - index * (position === "QB" ? 5.2 : 2.6), 20);
        players.push({
          player_id: playerId,
          player_name: position === "DEF" ? `Team ${index} DEF` : `${position} Projection ${index}`,
          rank: players.length + 1,
          position,
          pro_team: position === "DEF" ? `T${index}` : "",
          pos_rank: index,
          bye_week: ((index + position.length) % 14) + 4,
          injury_status: "",
          percent_owned: 0,
          percent_started: 0,
          projected_total_pts: Number(total.toFixed(2)),
          projected_avg_pts: Number((total / 17).toFixed(2)),
        });
        for (let week = 1; week <= 17; week += 1) {
          const wave = (((index * 7 + week * 5) % 11) - 5) * 0.35;
          weekly.push({
            player_id: playerId,
            week,
            projected_points: Number(Math.max(total / 17 + wave, 0).toFixed(2)),
          });
        }
      }
    }
    return {
      players,
      weekly_projections: weekly,
      team_names: defaultTeamNames(defaultConfig.num_teams || 10),
      league_settings: { name: "Demo", team_count: defaultConfig.num_teams || 10 },
      year: new Date().getFullYear(),
      synced_at: new Date().toISOString(),
      request: { source: "demo" },
    };
  }

  // Describe which weekly projection source was normalized.
  function weeklyProjectionLabel(meta) {
    const sources = meta?.weekly_projection_sources || {};
    const espnWeekly = integer(sources.espn_weekly) + integer(sources.espn_raw_weekly);
    const fallback = integer(sources.season_total_even_split) + integer(sources.season_total_bye_adjusted);
    if (espnWeekly > 0 && fallback > 0) return "mixed weekly/fallback projections";
    if (espnWeekly > 0) return "ESPN weekly projections";
    if (fallback > 0) return "season-total fallback projections";
    return "";
  }

  // Count raw ESPN weekly projection rows for sync status copy.
  function rawWeeklyProjectionCount(meta) {
    const stats = meta?.raw_projection_stats || {};
    return integer(stats.projected_week_rows_with_total || stats.projected_week_rows);
  }

  // Build the user-facing message after projection sync completes.
  function projectionSyncStatus(result) {
    const label = weeklyProjectionLabel(result.projection_meta);
    if (label === "season-total fallback projections") {
      const rawWeekly = rawWeeklyProjectionCount(result.projection_meta);
      if (rawWeekly > 0) {
        return `Synced ${result.players.length} players. Raw ESPN weekly projection rows were found (${rawWeekly}), but no usable weekly rows were normalized.`;
      }
      return `Synced ${result.players.length} players. ESPN did not return raw weekly projection rows, so season totals were spread across weeks.`;
    }
    return `Synced ${result.players.length} players${label ? ` with ${label}` : ""}.`;
  }

  // POST the sync form to the Python projection endpoint.
  async function syncEspn() {
    const payload = {
      league_id: $("leagueId").value.trim(),
      year: integer($("year").value, new Date().getFullYear()),
      week_start: integer($("weekStart").value, 1),
      week_end: integer($("weekEnd").value, 17),
      swid: $("swid").value.trim(),
      espn_s2: $("espnS2").value,
    };
    setStatus("Syncing ESPN projections...");
    const response = await fetch("/api/projections/sync", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Projection sync failed");
    $("espnS2").value = "";
    await saveProjectionPayload(result, "espn");
    setStatus(projectionSyncStatus(result));
  }

  // Export all local IndexedDB tables as a JSON file.
  async function exportLocalData() {
    const payload = {};
    for (const table of TABLES) payload[table] = await db.table(table).toArray();
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "fflab-draftsim.json";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  // Import a previously exported IndexedDB JSON snapshot.
  async function importLocalData() {
    const payload = JSON.parse($("importPayload").value || "{}");
    for (const table of TABLES) {
      await db.table(table).clear();
      if (Array.isArray(payload[table])) await db.table(table).bulkPut(payload[table]);
    }
    $("importDialog").close();
    await loadState();
    setStatus("Local data imported.");
  }

  // Clear all local browser data after user confirmation.
  async function resetLocalData() {
    if (!confirm("Reset all local draft simulator data in this browser?")) return;
    await clearTables(TABLES);
    await db.sessions.put(defaultSession());
    await loadState();
    setStatus("Local data reset.");
  }

  // Render current draft status and clock button states.
  function renderClock() {
    const pick = currentPick();
    const started = Boolean(state.session?.draft_started);
    $("startDraft").disabled = state.draftBusy || !state.slots.length || !pick || started;
    $("startDraft").textContent = started ? "Draft Started" : "Start Draft";
    $("autoPick").disabled = state.draftBusy || !pick || !started;
    if (!state.slots.length) {
      $("clockTeam").textContent = "No draft board";
      $("clockMeta").textContent = "Sync projections or load demo data.";
      return;
    }
    if (!pick) {
      $("clockTeam").textContent = "Draft complete";
      $("clockMeta").textContent = "Projected season results are available below.";
      return;
    }
    $("clockTeam").textContent = teamName(pick.current_team);
    const paused = state.draftBusy ? "Drafting... " : (started ? "" : "Draft paused. Start Draft when ready. ");
    $("clockMeta").textContent = `${paused}Round ${pick.round}, pick ${pick.pick_in_round}, overall ${pick.overall}`;
  }

  const availableSortDefaults = {
    rank: "asc",
    player_name: "asc",
    position: "asc",
    pro_team: "asc",
    bye_week: "asc",
    projected_total_pts: "desc",
    pos_rank: "asc",
    adp: "asc",
    injury_code: "asc",
    percent_owned: "desc",
  };

  // Return the comparable value for one available-board sort key.
  function availableSortValue(player, key) {
    if (key === "rank") return integer(player.rank) > 0 ? integer(player.rank) : null;
    if (key === "bye_week") return integer(player.bye_week) > 0 ? integer(player.bye_week) : null;
    if (key === "projected_total_pts") return number(player.projected_total_pts);
    if (key === "pos_rank") return integer(player.pos_rank) > 0 ? integer(player.pos_rank) : null;
    if (key === "adp") return number(player.adp, -1) >= 0 ? number(player.adp) : null;
    if (key === "injury_code") return injuryCode(player.injury_status);
    if (key === "percent_owned") return number(player.percent_owned, -1) >= 0 ? number(player.percent_owned) : null;
    return player[key] || "";
  }

  // Sort available board rows using the active sort selection.
  function sortAvailableRows(rows) {
    const { key, direction } = state.availableSort;
    if (!key || key === "default") return rows;
    return rows.slice().sort((a, b) => compareValues(availableSortValue(a, key), availableSortValue(b, key), direction));
  }

  // Attach click handlers to the available-board sortable headers.
  function bindAvailableSortHeaders() {
    document.querySelectorAll("#available [data-sort-key]").forEach((header) => {
      header.addEventListener("click", () => {
        const key = header.getAttribute("data-sort-key");
        if (state.availableSort.key === key) {
          state.availableSort.direction = state.availableSort.direction === "asc" ? "desc" : "asc";
        } else {
          state.availableSort = { key, direction: availableSortDefaults[key] || "asc" };
        }
        renderAvailable();
      });
    });
  }

  // Render the searchable, sortable available-player draft board.
  function renderAvailable() {
    const query = $("search").value.trim().toLowerCase();
    const position = $("positionFilter").value;
    const pick = currentPick();
    const canDraft = Boolean(!state.draftBusy && state.session?.draft_started && pick && pick.current_team === state.session.human_team_index);
    const available = undraftedPlayers();
    const filtered = available
      .filter((player) => {
        const matchesQuery = !query || player.player_name.toLowerCase().includes(query);
        const matchesPosition = !position || player.position === position || (position === "FLEX" && flexPositions.has(player.position));
        return matchesQuery && matchesPosition;
      });
    const rows = sortAvailableRows(filtered);
    $("boardCount").textContent = `Showing ${rows.length} of ${available.length} available players`;
    renderTable("available", rows, [
      { label: "Rank", sortKey: "rank", number: true, render: (row) => `<td class="number">${integer(row.rank) > 0 ? integer(row.rank) : ""}</td>` },
      { key: "player_name", label: "Name", sortKey: "player_name" },
      { key: "position", label: "Pos", sortKey: "position" },
      { key: "pro_team", label: "Team", sortKey: "pro_team" },
      { label: "Bye", sortKey: "bye_week", number: true, render: (row) => `<td class="number">${integer(row.bye_week) > 0 ? integer(row.bye_week) : ""}</td>` },
      { key: "projected_total_pts", label: "Proj", sortKey: "projected_total_pts", number: true, digits: 1 },
      { label: "Pos Rank", sortKey: "pos_rank", number: true, render: (row) => `<td class="number">${integer(row.pos_rank) > 0 ? integer(row.pos_rank) : ""}</td>` },
      { label: "ADP", sortKey: "adp", number: true, render: (row) => `<td class="number">${number(row.adp, -1) >= 0 ? number(row.adp).toFixed(1) : ""}</td>` },
      { label: "Inj", sortKey: "injury_code", className: "injury-col", render: (row) => `<td class="injury-col" title="${escapeHtml(row.injury_status || "")}">${escapeHtml(injuryCode(row.injury_status))}</td>` },
      { label: "Own %", sortKey: "percent_owned", number: true, render: (row) => `<td class="number">${number(row.percent_owned, -1) >= 0 ? number(row.percent_owned).toFixed(1) : ""}</td>` },
      { label: "Pick", render: (row) => `<td><button class="small" data-pick-player="${escapeHtml(row.player_id)}" ${canDraft && canAddPlayer(pick.current_team, row) ? "" : "disabled"}>Draft</button></td>` },
    ]);
    bindAvailableSortHeaders();
    document.querySelectorAll("[data-pick-player]").forEach((button) => {
      button.addEventListener("click", () => makePick(button.getAttribute("data-pick-player"), "human"));
    });
  }

  // Return a display name for a team index.
  function teamName(index) {
    return state.session.team_names[index] || fallbackTeamName(index);
  }

  // Summarize unfilled starter needs for a roster.
  function rosterNeedsText(index, roster) {
    const counts = rosterCounts(roster);
    return positionOrder
      .map((position) => [position, Math.max(integer(state.session.roster_settings?.[position], 0) - integer(counts[position], 0), 0)])
      .filter(([, count]) => count > 0)
      .map(([position, count]) => `${position} ${count}`)
      .join(", ") || "Starter slots filled";
  }

  // Build roster HTML for the current-team and selected-team panels.
  function rosterMarkup(index, { compact = false } = {}) {
    const roster = rostersByTeam()[index] || [];
    const needs = rosterNeedsText(index, roster);
    const rows = roster.length
      ? roster.map((player) => {
        const byeWeek = integer(player.bye_week);
        return `<li><span>${escapeHtml(player.player_name)}</span><strong>${escapeHtml(player.position)}</strong><span class="roster-bye">${byeWeek > 0 ? byeWeek : ""}</span></li>`;
      }).join("")
      : "<li class=\"roster-empty\">No players drafted.</li>";
    return `<h3>${escapeHtml(teamName(index))}</h3>
      <p class="roster-needs">${escapeHtml(needs)}</p>
      <ul class="${compact ? "compact-roster" : "roster-list"}"><li class="roster-heading"><span>Player</span><strong>Pos</strong><span class="roster-bye">Bye</span></li>${rows}</ul>`;
  }

  // Render the human team's compact roster panel.
  function renderCurrentRoster() {
    $("currentRoster").innerHTML = rosterMarkup(integer(state.session.human_team_index, 0), { compact: true });
  }

  // Render completed picks or scheduled picks in the draft log.
  function renderDraftLog() {
    const players = playerByIdMap();
    const rows = state.picks.length
      ? state.picks.slice().reverse().map((pick) => {
        const player = players.get(String(pick.player_id)) || {};
        return {
          ...pick,
          team: state.session.team_names[pick.team_index],
          player: pick.skipped ? "Skipped" : (player.player_name || pick.player_id),
          position: player.position || "",
        };
      })
      : state.slots.map((slot) => ({
        overall: slot.overall,
        round: slot.round,
        pick_in_round: slot.pick_in_round,
        team: state.session.team_names[slot.current_team],
        player: "",
        position: "",
        source: state.session?.draft_started ? "Upcoming" : "Scheduled",
      }));
    renderTable("draftLog", rows, [
      { key: "overall", label: "#", number: true },
      { key: "round", label: "Rd", number: true },
      { key: "pick_in_round", label: "Pick", number: true },
      { key: "team", label: "Team" },
      { key: "player", label: "Player" },
      { key: "position", label: "Pos" },
      { key: "source", label: "Mode" },
    ]);
  }

  // Render saved and temporary pick trades.
  function renderTrades() {
    const locked = Boolean(state.draftBusy || state.session?.draft_started || state.picks.length > 0);
    $("testTrade").disabled = locked;
    $("saveTrade").disabled = locked;
    const rows = state.trades.map((trade) => ({
      ...trade,
      mode: trade.temporary ? "Test" : "Saved",
      teamA: trade.team_a == null ? state.session.team_names[trade.from_team] : teamName(trade.team_a),
      teamB: trade.team_b == null ? state.session.team_names[trade.to_team] : teamName(trade.team_b),
      sendsA: trade.team_a_pick_refs == null
        ? (trade.pick_refs || []).map(pickRefLabel).join(", ")
        : (trade.team_a_pick_refs || []).map(pickRefLabel).join(", "),
      sendsB: (trade.team_b_pick_refs || []).map(pickRefLabel).join(", "),
    }));
    renderTable("trades", rows, [
      { key: "mode", label: "Mode" },
      { key: "teamA", label: "Team A" },
      { key: "sendsA", label: "A Sends" },
      { key: "teamB", label: "Team B" },
      { key: "sendsB", label: "B Sends" },
      { label: "", render: (row) => `<td><button class="small secondary" data-delete-trade="${escapeHtml(row.id)}" ${locked ? "disabled" : ""}>Delete</button></td>` },
    ]);
    document.querySelectorAll("[data-delete-trade]").forEach((button) => {
      button.addEventListener("click", () => deleteTrade(button.getAttribute("data-delete-trade")));
    });
  }

  // Render the roster selected in the Rosters tab.
  function renderSelectedRoster() {
    const index = Math.max(Math.min(integer($("rosterTeam").value, state.session.human_team_index), state.session.num_teams - 1), 0);
    $("selectedRoster").innerHTML = rosterMarkup(index);
  }

  // Render projected standings, matchups, and playoffs.
  function renderResults() {
    if (!state.results) {
      $("resultsPanel").classList.add("hidden");
      return;
    }
    $("resultsPanel").classList.remove("hidden");
    const leader = state.results.standings[0] || {};
    $("champion").textContent = state.results.playoffChampion?.team || leader.team || "-";
    $("points").textContent = leader.points_for == null ? "-" : formatPoints(leader.points_for);
    $("pickCount").textContent = String(state.picks.length);
    $("weekCount").textContent = String(state.results.weeks.length);
    renderTable("standings", state.results.standings, [
      { key: "rank", label: "Rank", number: true },
      { key: "team", label: "Team" },
      { key: "wins", label: "W", number: true },
      { key: "losses", label: "L", number: true },
      { key: "ties", label: "T", number: true },
      { key: "win_pct", label: "Win %", number: true, digits: 3 },
      { key: "points_for", label: "PF", number: true, digits: 1 },
      { key: "points_against", label: "PA", number: true, digits: 1 },
    ]);
    renderTable("weeklyMatchups", state.results.weeklyMatchups, [
      { key: "week", label: "Week", number: true },
      { key: "home", label: "Home" },
      { key: "home_score", label: "Home Proj", number: true, digits: 1 },
      { key: "away", label: "Away" },
      { key: "away_score", label: "Away Proj", number: true, digits: 1 },
      { key: "winner", label: "Winner" },
    ]);
    renderTable("playoffMatchups", state.results.playoffMatchups || [], [
      { key: "week", label: "Week", number: true },
      { key: "round", label: "Round" },
      { key: "home_seed", label: "Seed", number: true },
      { key: "home", label: "Team" },
      { key: "home_score", label: "Proj", number: true, digits: 1 },
      { key: "away_seed", label: "Seed", number: true },
      { key: "away", label: "Opponent" },
      { key: "away_score", label: "Proj", number: true, digits: 1 },
      { key: "winner", label: "Advances" },
    ]);
  }

  // Render top-level sync/cache metadata.
  function renderSyncMeta() {
    const synced = state.session?.synced_at ? new Date(state.session.synced_at).toLocaleString() : "never";
    const source = state.session?.source || "empty";
    const weekly = weeklyProjectionLabel(state.session?.projection_meta);
    const rawWeekly = rawWeeklyProjectionCount(state.session?.projection_meta);
    $("syncMeta").textContent = `${state.players.length} players cached | ${source} | synced ${synced}${weekly ? ` | ${weekly}` : ""}${rawWeekly ? ` | raw weekly rows ${rawWeekly}` : ""}`;
  }

  // Render browser online/offline status.
  function renderOnlineStatus() {
    const online = navigator.onLine;
    $("onlineStatus").textContent = online ? "Online" : "Offline";
    $("onlineStatus").className = online ? "pill" : "pill offline";
  }

  // Show one tab panel or collapse all panels.
  function setActiveTab(tabId) {
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.classList.toggle("active", button.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.classList.toggle("hidden", panel.id !== tabId);
    });
    $("tabPanels").classList.toggle("hidden", !tabId);
  }

  // Refresh all visible app panels from current state.
  function render() {
    renderSyncMeta();
    renderOnlineStatus();
    refreshTradeTeamOptions();
    writeFavoriteTeamInputs();
    renderClock();
    renderAvailable();
    renderCurrentRoster();
    renderSelectedRoster();
    renderDraftLog();
    renderTrades();
    renderResults();
  }

  // Attach browser event handlers once during startup.
  function bindEvents() {
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        const tabId = button.getAttribute("data-tab");
        setActiveTab(button.classList.contains("active") ? "" : tabId);
      });
    });
    $("syncEspn").addEventListener("click", async () => {
      try {
        await syncEspn();
      } catch (error) {
        setStatus(error.message, true);
      }
    });
    $("loadDemo").addEventListener("click", async () => {
      await saveProjectionPayload(demoPayload(), "demo");
      setStatus("Demo projections loaded locally.");
    });
    $("newDraft").addEventListener("click", () => resetDraftBoard());
    $("startDraft").addEventListener("click", async () => {
      try {
        await startDraft();
      } catch (error) {
        setStatus(error.message, true);
      }
    });
    $("autoPick").addEventListener("click", async () => {
      await autoPickCurrent();
    });
    $("testTrade").addEventListener("click", testTrade);
    $("saveTrade").addEventListener("click", saveTrade);
    $("playoffTeams").addEventListener("change", savePlayoffSettings);
    $("playoffByes").addEventListener("change", savePlayoffSettings);
    $("scoreWeightTeam").addEventListener("change", () => {
      writeScoreWeightInputs();
      writePositionPreferenceInputs();
      writeFavoriteTeamInputs();
    });
    $("saveScoreWeights").addEventListener("click", () => saveScoreWeightInputs());
    $("resetScoreWeights").addEventListener("click", resetScoreWeightInputs);
    $("savePositionPreferences").addEventListener("click", () => savePositionPreferenceInputs());
    $("resetPositionPreferences").addEventListener("click", resetPositionPreferenceInputs);
    $("saveFavoriteTeams").addEventListener("click", () => saveFavoriteTeamInputs());
    $("resetFavoriteTeams").addEventListener("click", resetFavoriteTeamInputs);
    $("rosterTeam").addEventListener("change", renderSelectedRoster);
    $("search").addEventListener("input", renderAvailable);
    $("positionFilter").addEventListener("change", renderAvailable);
    $("exportData").addEventListener("click", exportLocalData);
    $("importData").addEventListener("click", () => $("importDialog").showModal());
    $("confirmImport").addEventListener("click", async () => {
      try {
        await importLocalData();
      } catch (error) {
        setStatus(error.message, true);
      }
    });
    $("resetLocal").addEventListener("click", resetLocalData);
    window.addEventListener("online", renderOnlineStatus);
    window.addEventListener("offline", renderOnlineStatus);
  }

  bindEvents();
  loadState().catch((error) => setStatus(error.message, true));
})();
