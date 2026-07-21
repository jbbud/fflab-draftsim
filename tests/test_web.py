from __future__ import annotations

from pathlib import Path

from fflab import web


def test_projection_sync_response_excludes_private_cookie_fields(monkeypatch) -> None:
    def fake_sync(payload):
        assert payload["espn_s2"] == "secret"
        return {
            "players": [],
            "weekly_projections": [],
            "league_settings": {},
            "team_names": [],
            "synced_at": "2026-07-14T00:00:00+00:00",
        }

    monkeypatch.setattr(web, "sync_projection_payload", fake_sync)
    response = web.projection_sync_response(
        {"league_id": 1, "year": 2026, "espn_s2": "secret", "swid": "{abc}"}
    )

    assert response["request"] == {"league_id": 1, "year": 2026}
    assert "espn_s2" not in response
    assert "swid" not in response


def test_projection_sync_response_fills_missing_credentials_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ESPN_S2", "server-secret")
    monkeypatch.setenv("SWID", "{server-swid}")

    def fake_sync(payload):
        assert payload["espn_s2"] == "server-secret"
        assert payload["swid"] == "{server-swid}"
        return {
            "players": [],
            "weekly_projections": [],
            "league_settings": {},
            "team_names": [],
            "synced_at": "2026-07-14T00:00:00+00:00",
        }

    monkeypatch.setattr(web, "sync_projection_payload", fake_sync)
    response = web.projection_sync_response({"league_id": 1, "year": 2026})

    assert response["request"] == {"league_id": 1, "year": 2026}


def test_load_env_file_sets_missing_values_without_overriding(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        """
        ESPN_S2="from-file"
        SWID='{file-swid}'
        ESPN_SWID={alternate}
        """,
        encoding="utf-8",
    )
    monkeypatch.delenv("ESPN_S2", raising=False)
    monkeypatch.setenv("SWID", "{existing}")

    values = web.load_env_file(env_path)

    assert values["ESPN_S2"] == "from-file"
    assert values["SWID"] == "{file-swid}"
    assert values["ESPN_SWID"] == "{alternate}"
    assert web.os.environ["ESPN_S2"] == "from-file"
    assert web.os.environ["SWID"] == "{existing}"


def test_gui_config_includes_safe_env_defaults_without_credentials(monkeypatch) -> None:
    monkeypatch.setenv("LEAGUE_ID", "987654")
    monkeypatch.setenv("YEAR", "2026")
    monkeypatch.setenv("WEEK_START", "1")
    monkeypatch.setenv("WEEK_END", "17")
    monkeypatch.setenv("ESPN_S2", "server-secret")
    monkeypatch.setenv("SWID", "{server-swid}")

    config = web.gui_config()

    assert config["league_id"] == "987654"
    assert config["year"] == 2026
    assert config["week_start"] == 1
    assert config["week_end"] == 17
    assert "espn_s2" not in config
    assert "swid" not in config
    assert "ESPN_S2" not in config
    assert "SWID" not in config


def test_static_assets_include_dexie_schema_and_offline_logic() -> None:
    app = Path("src/fflab/static/app.js").read_text(encoding="utf-8")
    dexie = Path("src/fflab/static/dexie.min.js").read_text(encoding="utf-8")
    trainer = Path("tools/train_standard_weights.mjs").read_text(encoding="utf-8")

    assert 'new Dexie("fflab_draftsim")' in app
    assert "weekly_projections" in app
    assert "league_schedule" in app
    assert "draft_slots" in app
    assert "pick_trades" in app
    assert "baseDraftSlots" in app
    assert "bye_week" in app
    assert "FLEX" in app
    assert "setActiveTab" in app
    assert "renderSelectedRoster" in app
    assert "writeProjectionInputs" in app
    assert "parsePickToken" in app
    assert "testTrade" in app
    assert "availableSort" in app
    assert "injuryCode" in app
    assert "adp" in app
    assert "ADP" in app
    assert "undraftedPlayers" in app
    assert "draftablePlayersForCurrentPick" in app
    assert "draft_started" in app
    assert "startDraft" in app
    assert "draftBusy" in app
    assert "withDraftLock" in app
    assert "autoPickCurrent" in app
    assert "resumeBotDraftIfNeeded" in app
    assert "No legal bot pick" in app
    assert "sanitizeDraftPicks" in app
    assert "repairDraftPicksFromDb" in app
    assert "draftedPlayerIds" in app
    assert "was already drafted at pick" in app
    assert "rawWeeklyProjectionCount" in app
    assert "boardCount" in app
    assert "Showing ${rows.length} of ${available.length} available players" in app
    assert "ESPN did not return raw weekly projection rows" in app
    assert "assertPickOwnedByTeam" in app
    assert "Pick trades must be set before the draft starts." in app
    assert "Scheduled" in app
    assert "simulateDraft" in app
    assert "simulatePlayoffs" in app
    assert "savePlayoffSettings" in app
    assert "score_weights_by_team" in app
    assert "scoreWeightsForTeam" in app
    assert "saveScoreWeightInputs" in app
    assert "position_preferences_by_team" in app
    assert "positionPreferencePositions" in app
    assert '["QB", "TE", "K", "DEF"]' in app
    assert "normalizePositionPreferencesByTeam" in app
    assert "positionPreferencesForTeam" in app
    assert "positionPreferenceAdjustment" in app
    assert "positionPreferenceMaxAdjustment" in app
    assert "savePositionPreferenceInputs" in app
    assert "favorite_nfl_teams_by_team" in app
    assert "favoriteNflTeamsForTeam" in app
    assert "favoriteTeamAdjustment" in app
    assert "favoriteTeamScoreBonus" in app
    assert "saveFavoriteTeamInputs" in app
    assert 'team !== "NONE"' in app
    assert "positionPreferenceWeight" in app
    assert "favoriteTeamWeight" in app
    assert "backupPenaltyWeight" in app
    assert "nextPickForTeam" in app
    assert "state.slots.find" in app
    assert "positionTimingMultiplier" in app
    assert 'position !== "K" && position !== "DEF"' in app
    assert "currentRound / 15" in app
    assert "Math.pow(progress, 6)" in app
    assert "backupPositionPenalty" in app
    assert "rosterSurplusPenalty" in app
    assert "roundMismatchPenalty" in app
    assert "leagueSaturationPenalty" in app
    assert "preferredBenchDepth" in app
    assert "backupPenalty" in app
    assert "untimedValueShare" in app
    assert "coreShare" in app
    assert "timingStrengthByPosition" not in app
    assert "position_start_rounds" not in app
    assert "robustComponentNormalizer" in app
    assert "signedPower" in app
    assert "migrateScoreWeightUnits" in app
    assert "normalized-v1" in app
    assert "rankValRaw" in app
    assert "adpValRaw" in app
    assert "vorVal" in app
    assert "coreValue" in app
    assert "timedValue" in app
    assert "playoff_team_count" in app
    assert "playoff_bye_count" in app
    assert "points_against" in app
    assert "PA" in app
    assert "teamNamesFromSources" in app
    assert "fallbackTeamName" in app
    assert "Team #${index + 1}" in app
    assert '"teamNames"' not in app
    assert "global.Dexie = Dexie" in dexie

    assert 'data-tab="projectionTab"' in web.HTML
    assert 'id="rosterTeam"' in web.HTML
    assert 'id="playoffTeams"' in web.HTML
    assert 'id="playoffByes"' in web.HTML
    assert 'id="playoffMatchups"' in web.HTML
    assert 'id="tradeTeamA"' in web.HTML
    assert 'id="tradePicksA"' in web.HTML
    assert 'id="testTrade"' in web.HTML
    assert 'id="startDraft"' in web.HTML
    assert 'id="boardCount"' in web.HTML
    assert 'class="nav-actions"' in web.HTML
    assert 'class="nav-select">Your Team<select id="humanTeam"' in web.HTML
    assert 'id="newDraft" class="nav-action"' in web.HTML
    assert 'id="scoreWeightTeam"' in web.HTML
    assert 'id="saveScoreWeights"' in web.HTML
    assert 'id="weightVor"' in web.HTML
    assert 'id="weightBackupPenalty"' in web.HTML
    assert 'id="weightPositionPreference"' in web.HTML
    assert 'id="weightFavoriteTeam"' in web.HTML
    assert "Draft Intel Position Windows" in web.HTML
    assert 'id="prefQbFirstEarliest"' in web.HTML
    assert 'id="prefTeBackupLatest"' in web.HTML
    assert 'id="prefKFirstLatest"' in web.HTML
    assert 'id="prefDefBackupEarliest"' in web.HTML
    assert 'id="savePositionPreferences"' in web.HTML
    assert 'id="resetPositionPreferences"' in web.HTML
    assert "Favorite NFL Teams" in web.HTML
    assert 'id="favoriteNflTeams"' in web.HTML
    assert 'class="favorite-team-grid"' in web.HTML
    assert 'multiple size="8"' not in web.HTML
    assert 'id="saveFavoriteTeams"' in web.HTML
    assert 'id="resetFavoriteTeams"' in web.HTML
    assert 'id="teamNames"' not in web.HTML
    assert "Team Names" not in web.HTML
    assert 'id="qbStart"' not in web.HTML
    assert 'id="teStart"' not in web.HTML
    assert 'id="defStart"' not in web.HTML
    assert 'id="kStart"' not in web.HTML
    assert "position_start_rounds" not in web.HTML
    assert 'id="allRosters"' not in web.HTML

    css = Path("src/fflab/static/style.css").read_text(encoding="utf-8")
    assert ".sort-header" in css
    assert ".injury-col" in css
    assert ".clock-actions" in css
    assert ".playoff-section" in css
    assert ".weight-grid" in css
    assert "repeat(10" in css
    assert ".position-preferences" in css
    assert ".preference-grid" in css
    assert ".favorite-teams" in css
    assert ".favorite-team-grid" in css
    assert "repeat(8" in css
    assert ".nav-actions" in css
    assert ".nav-action" in css
    assert ".nav-select" in css
    assert ".grid.four" not in css

    assert "BASELINE_WEIGHTS" in trainer
    assert "backupPenalty: 1.58" in trainer
    assert "backupPenalty: [0, 2.5]" in trainer
    assert "positionPreference" not in trainer
    assert "favoriteTeam" not in trainer
    assert "targetTeams" in trainer
    assert "parseTargetTeams" in trainer
    assert "targetTeamSampleRate" in trainer
    assert "targetTeamSampleMin" in trainer
    assert "fullHoldout" in trainer
    assert "boolean(value" in trainer
    assert "targetTeamBatches" in trainer
    assert "sampleTargetTeams" in trainer
    assert "targetTeamSampleSize" in trainer
    assert "stageSeedSalt" in trainer
    assert "stage_one_target_team_batches" in trainer
    assert "stage_two_target_team_batches" in trainer
    assert "target_team_sample_rate" in trainer
    assert "target_team_sample_min" in trainer
    assert "full_holdout" in trainer
    assert "target_teams" in trainer
    assert "per_team" in trainer
    assert "syncEnvLeaguePayload" in trainer
    assert "payload_with_env_credentials" in trainer
    assert "ESPN league from .env" in trainer
    assert "demoPayload" in trainer
    assert "evaluateWeights" in trainer
    assert "createProgressLogger" in trainer
    assert "formatDuration" in trainer
    assert "batchEvaluationCount" in trainer
    assert "console.error(`[${label}] ${completed}/${safeTotal} evals" in trainer
    assert "sampleWeights" in trainer
    assert "stageOne" in trainer
    assert "stageTwo" in trainer
    assert "traceOut" in trainer
    assert "backupPositionPenalty" in trainer
    assert "backupPenaltyWeight" in trainer
    assert "LEAGUE_SATURATION_THRESHOLD" in trainer
    assert "PREFERRED_BENCH_DEPTH" in trainer
    assert "selected_player_name" in trainer
    assert "holdout_mean_delta" in trainer
    assert "selected target teams projected optimal-lineup season points" in trainer
