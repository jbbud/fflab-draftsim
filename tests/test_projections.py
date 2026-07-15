from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from fflab.projections import normalize_espn_league, sync_projection_payload


@dataclass
class FakePlayer:
    playerId: int
    name: str
    position: str
    projected_total_points: float = 0.0
    proTeam: str = ""
    posRank: int = 0
    positionalRanking: int = 0
    draftRank: int = 0
    averageDraftPosition: float = -1.0
    byeWeek: int = 0
    percent_owned: float = 0.0
    percent_started: float = 0.0
    stats: dict[int, dict[str, float]] = field(default_factory=dict)
    eligibleSlots: list[object] = field(default_factory=list)


@dataclass
class FakeTeam:
    team_name: str
    team_id: int
    team_abbrev: str = ""
    draft_projected_rank: int = 0
    roster: list[FakePlayer] = field(default_factory=list)
    schedule: list[object] = field(default_factory=list)


@dataclass
class FakeSettings:
    name: str = "League"
    team_count: int = 2
    reg_season_count: int = 17
    playoff_team_count: int = 4
    keeper_count: int = 0
    scoring_format: list[str] = field(default_factory=list)


class FakeLeague:
    def __init__(self) -> None:
        self.players = {
            1: FakePlayer(
                playerId=1,
                name="QB One",
                position="QB",
                projected_total_points=40.0,
                positionalRanking=3,
                draftRank=7,
                averageDraftPosition=12.34,
                byeWeek=11,
                stats={1: {"projected_points": 20.0}, 2: {"projected_points": 20.0}},
            ),
            2: FakePlayer(
                playerId=2,
                name="Defense",
                position="D/ST",
                projected_total_points=12.0,
                draftRank=12,
                averageDraftPosition=171.2,
                byeWeek=9,
                stats={1: {"projected_points": 6.0}, 2: {"projected_points": 6.0}},
            ),
            3: FakePlayer(playerId=3, name="Linebacker", position="LB"),
        }
        self.player_map = {1: "QB One", 2: "Defense", 3: "Linebacker"}
        self.teams = [FakeTeam("You", 1, "YOU"), FakeTeam("Bot", 2, "BOT")]
        self.teams[0].schedule = [self.teams[1], self.teams[1]]
        self.teams[1].schedule = [self.teams[0], self.teams[0]]
        self.settings = FakeSettings()
        self.draft_detail = {
            "settings": {"draftSettings": {"pickOrder": [2, 1]}},
            "draftDetail": {
                "picks": [
                    {"overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1, "teamId": 2},
                    {"overallPickNumber": 2, "roundId": 1, "roundPickNumber": 2, "teamId": 1},
                    {"overallPickNumber": 3, "roundId": 2, "roundPickNumber": 1, "teamId": 1},
                    {"overallPickNumber": 4, "roundId": 2, "roundPickNumber": 2, "teamId": 2},
                ]
            },
        }

    def player_info(self, playerId):
        return [self.players[player_id] for player_id in playerId]


class FakeDefenseLeague:
    def __init__(self) -> None:
        self.players = {
            -16033: FakePlayer(
                playerId=-16033,
                name="Ravens D/ST",
                position="",
                projected_total_points=18.0,
                eligibleSlots=["D/ST"],
                stats={1: {"projected_points": 9.0}, 2: {"projected_points": 9.0}},
            )
        }
        self.player_map = {-16033: "Ravens D/ST"}
        self.teams = []
        self.settings = FakeSettings(team_count=0)

    def player_info(self, playerId):
        return [self.players[player_id] for player_id in playerId]


class FakeSeasonOnlyLeague:
    def __init__(self) -> None:
        self.players = {
            4: FakePlayer(
                playerId=4,
                name="Season Only",
                position="RB",
                projected_total_points=34.0,
                byeWeek=2,
            )
        }
        self.player_map = {4: "Season Only"}
        self.teams = []
        self.settings = FakeSettings(team_count=0)

    def player_info(self, playerId):
        return [self.players[player_id] for player_id in playerId]


class FakeEspnRequest:
    def __init__(self, players):
        self.players = players
        self.additional_filters = []

    def get_player_card(self, player_ids, max_scoring_period, additional_filters=None):
        self.additional_filters.append(list(additional_filters or []))
        wanted = {int(player_id) for player_id in player_ids}
        return {"players": [player for player in self.players if int(player["id"]) in wanted]}


class FakeRawLeague:
    def __init__(self) -> None:
        self.espn_request = FakeEspnRequest(
            [
                {
                    "id": 10,
                    "fullName": "Weekly Back",
                    "eligibleSlots": [2],
                    "proTeamId": 2,
                    "defaultPositionId": 2,
                    "positionalRanking": 0,
                    "injuryStatus": "ACTIVE",
                    "jersey": "1",
                    "onTeamId": 0,
                    "playerPoolEntry": {
                        "player": {
                            "proTeamId": 2,
                            "defaultPositionId": 2,
                            "injuryStatus": "ACTIVE",
                            "injured": False,
                            "ownership": {},
                            "stats": [
                                {
                                    "seasonId": 2026,
                                    "scoringPeriodId": 0,
                                    "statSourceId": 1,
                                    "statSplitTypeId": 0,
                                    "appliedTotal": 99.0,
                                    "appliedAverage": 5.5,
                                    "stats": {},
                                    "appliedStats": {},
                                },
                                {
                                    "seasonId": 2026,
                                    "scoringPeriodId": 1,
                                    "statSourceId": 1,
                                    "statSplitTypeId": 2,
                                    "appliedTotal": 7.0,
                                    "appliedAverage": 7.0,
                                    "stats": {},
                                    "appliedStats": {},
                                },
                                {
                                    "seasonId": 2026,
                                    "scoringPeriodId": 2,
                                    "statSourceId": 1,
                                    "statSplitTypeId": 2,
                                    "appliedTotal": 13.0,
                                    "appliedAverage": 13.0,
                                    "stats": {},
                                    "appliedStats": {},
                                },
                            ],
                        }
                    },
                }
            ]
        )
        self.finalScoringPeriod = 2
        self.player_map = {10: "Weekly Back"}
        self.teams = []
        self.settings = FakeSettings(team_count=0)


class FakeWeeklyInfoRequest:
    def __init__(self) -> None:
        self.card_calls = []
        self.info_calls = []

    @staticmethod
    def _raw_player(stats):
        return {
            "id": 11,
            "fullName": "Info Weekly Back",
            "eligibleSlots": [2],
            "proTeamId": 2,
            "defaultPositionId": 2,
            "positionalRanking": 0,
            "injuryStatus": "ACTIVE",
            "jersey": "1",
            "onTeamId": 0,
            "playerPoolEntry": {
                "player": {
                    "id": 11,
                    "proTeamId": 2,
                    "defaultPositionId": 2,
                    "injuryStatus": "ACTIVE",
                    "injured": False,
                    "ownership": {},
                    "stats": stats,
                }
            },
        }

    def get_player_card(self, player_ids, max_scoring_period, additional_filters=None):
        self.card_calls.append((list(player_ids), max_scoring_period, list(additional_filters or [])))
        return {
            "players": [
                self._raw_player(
                    [
                        {
                            "id": "102026",
                            "seasonId": 2026,
                            "scoringPeriodId": 0,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "appliedTotal": 30.0,
                            "appliedAverage": 15.0,
                            "stats": {},
                            "appliedStats": {},
                        }
                    ]
                )
            ]
        }

    def league_get(self, params=None, headers=None):
        week = int(params["scoringPeriodId"])
        self.info_calls.append((week, headers))
        return {
            "players": [
                self._raw_player(
                    [
                        {
                            "id": f"112026{week}",
                            "seasonId": 2026,
                            "scoringPeriodId": week,
                            "statSourceId": 1,
                            "statSplitTypeId": 1,
                            "appliedTotal": 10.0 + week,
                            "appliedAverage": None,
                            "stats": {},
                            "appliedStats": {},
                        }
                    ]
                )
            ]
        }


class FakeWeeklyInfoLeague:
    def __init__(self) -> None:
        self.espn_request = FakeWeeklyInfoRequest()
        self.finalScoringPeriod = 2
        self.player_map = {11: "Info Weekly Back"}
        self.teams = []
        self.settings = FakeSettings(team_count=0)


class FakeScheduleRequest:
    def league_get(self, params=None, headers=None):
        assert params == {"view": "mMatchupScore"}
        return {
            "schedule": [
                {"matchupPeriodId": 1, "home": {"teamId": 2}, "away": {"teamId": 1}},
                {"matchupPeriodId": 2, "home": {"teamId": 1}, "away": {"teamId": 2}},
                {"matchupPeriodId": 3, "home": {"teamId": 1}, "away": {"teamId": -1}},
            ]
        }


class FakeApiScheduleLeague:
    def __init__(self) -> None:
        self.espn_request = FakeScheduleRequest()
        self.player_map = {}
        self.teams = [FakeTeam("You", 1, "YOU"), FakeTeam("Bot", 2, "BOT")]
        self.settings = FakeSettings()

    def player_info(self, playerId):
        return []


def test_normalize_espn_league_maps_players_and_weekly_projections() -> None:
    payload = normalize_espn_league(
        FakeLeague(),
        year=2026,
        week_start=1,
        week_end=2,
        batch_size=2,
    )

    assert [row["player_name"] for row in payload["players"]] == ["QB One", "Defense"]
    assert [row["rank"] for row in payload["players"]] == [1, 2]
    assert [row["espn_rank"] for row in payload["players"]] == [7, 12]
    assert [row["adp"] for row in payload["players"]] == [12.34, 171.2]
    assert payload["players"][0]["pos_rank"] == 3
    assert payload["players"][0]["bye_week"] == 11
    assert payload["players"][1]["position"] == "DEF"
    assert payload["players"][1]["pos_rank"] == 1
    assert payload["weekly_projections"] == [
        {"player_id": "1", "week": 1, "projected_points": 20.0},
        {"player_id": "1", "week": 2, "projected_points": 20.0},
        {"player_id": "2", "week": 1, "projected_points": 6.0},
        {"player_id": "2", "week": 2, "projected_points": 6.0},
    ]
    assert payload["team_names"] == ["Bot", "You"]
    assert [team["team_id"] for team in payload["teams"]] == [2, 1]
    assert payload["draft_slots"] == [
        {"overall": 1, "round": 1, "pick_in_round": 1, "original_team": 0, "current_team": 0},
        {"overall": 2, "round": 1, "pick_in_round": 2, "original_team": 1, "current_team": 1},
        {"overall": 3, "round": 2, "pick_in_round": 1, "original_team": 1, "current_team": 1},
        {"overall": 4, "round": 2, "pick_in_round": 2, "original_team": 0, "current_team": 0},
    ]
    assert payload["league_schedule"] == [
        {
            "id": "1:1:0",
            "week": 1,
            "home_team_index": 1,
            "away_team_index": 0,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_team": "You",
            "away_team": "Bot",
            "source": "espn_team_schedule",
        },
        {
            "id": "2:1:0",
            "week": 2,
            "home_team_index": 1,
            "away_team_index": 0,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_team": "You",
            "away_team": "Bot",
            "source": "espn_team_schedule",
        },
    ]
    assert payload["league_settings"]["team_count"] == 2
    assert payload["projection_meta"]["weekly_projection_sources"] == {"espn_weekly": 4}
    assert payload["projection_meta"]["raw_projection_stats"] == {}


def test_normalize_espn_league_prefers_api_matchup_schedule() -> None:
    payload = normalize_espn_league(
        FakeApiScheduleLeague(),
        year=2026,
        week_start=1,
        week_end=2,
        batch_size=2,
    )

    assert payload["league_schedule"] == [
        {
            "id": "1:1:0",
            "week": 1,
            "home_team_index": 1,
            "away_team_index": 0,
            "home_team_id": 2,
            "away_team_id": 1,
            "home_team": "Bot",
            "away_team": "You",
            "source": "espn_matchup_score",
        },
        {
            "id": "2:0:1",
            "week": 2,
            "home_team_index": 0,
            "away_team_index": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_team": "You",
            "away_team": "Bot",
            "source": "espn_matchup_score",
        },
    ]


def test_normalize_espn_league_keeps_negative_id_team_defenses() -> None:
    payload = normalize_espn_league(
        FakeDefenseLeague(),
        year=2026,
        week_start=1,
        week_end=2,
        batch_size=2,
    )

    assert payload["players"] == [
        {
            "player_id": "-16033",
            "player_name": "Ravens D/ST",
            "rank": 1,
            "espn_rank": 0,
            "position": "DEF",
            "pro_team": "",
            "pos_rank": 1,
            "bye_week": 0,
            "injury_status": "",
            "injured": False,
            "active_status": "",
            "percent_owned": 0.0,
            "percent_started": 0.0,
            "adp": -1.0,
            "projected_total_pts": 18.0,
            "projected_avg_pts": 0.0,
            "season_total_pts": 18.0,
        }
    ]


def test_normalize_espn_league_reads_weekly_projections_from_raw_espn_stats() -> None:
    payload = normalize_espn_league(
        FakeRawLeague(),
        year=2026,
        week_start=1,
        week_end=2,
        batch_size=2,
    )

    assert payload["weekly_projections"] == [
        {"player_id": "10", "week": 1, "projected_points": 7.0},
        {"player_id": "10", "week": 2, "projected_points": 13.0},
    ]
    assert payload["players"][0]["projected_total_pts"] == 99.0
    assert payload["projection_meta"]["weekly_projection_sources"] == {"espn_raw_weekly": 2}
    assert payload["projection_meta"]["has_espn_weekly_projections"] is True
    assert payload["projection_meta"]["raw_projection_stats"] == {
        "projected_rows": 3,
        "projected_season_rows": 1,
        "projected_week_rows": 2,
        "projected_week_rows_with_total": 2,
        "projected_week_split_type_2_rows": 2,
        "stat_rows": 3,
    }
    assert payload["projection_meta"]["raw_projected_split_types"] == {"0": 1, "2": 2}
    assert payload["projection_meta"]["espn_player_card_additional_filters"] == ["112026", "122026"]


def test_normalize_espn_league_fetches_weekly_projections_from_player_info() -> None:
    league = FakeWeeklyInfoLeague()
    payload = normalize_espn_league(
        league,
        year=2026,
        week_start=1,
        week_end=2,
        batch_size=2,
    )

    assert [week for week, _headers in league.espn_request.info_calls] == [1, 2]
    assert payload["weekly_projections"] == [
        {"player_id": "11", "week": 1, "projected_points": 11.0},
        {"player_id": "11", "week": 2, "projected_points": 12.0},
    ]
    assert payload["projection_meta"]["weekly_projection_sources"] == {"espn_raw_weekly": 2}
    assert payload["projection_meta"]["raw_projection_stats"]["projected_week_rows"] == 2
    assert payload["projection_meta"]["raw_projected_split_types"] == {"0": 1, "1": 2}


def test_normalize_espn_league_marks_season_total_weekly_fallback() -> None:
    payload = normalize_espn_league(
        FakeSeasonOnlyLeague(),
        year=2026,
        week_start=1,
        week_end=3,
        batch_size=2,
    )

    assert payload["weekly_projections"] == [
        {"player_id": "4", "week": 1, "projected_points": 17.0},
        {"player_id": "4", "week": 2, "projected_points": 0.0},
        {"player_id": "4", "week": 3, "projected_points": 17.0},
    ]
    assert payload["projection_meta"]["weekly_projection_sources"] == {"season_total_bye_adjusted": 3}
    assert payload["projection_meta"]["has_espn_weekly_projections"] is False
    assert payload["projection_meta"]["raw_projection_stats"] == {}


def test_sync_projection_payload_uses_factory_without_persisting_credentials() -> None:
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return FakeLeague()

    payload = sync_projection_payload(
        {
            "league_id": 123,
            "year": 2026,
            "swid": "{abc}",
            "espn_s2": "secret",
            "week_start": 1,
            "week_end": 1,
        },
        league_factory=factory,
    )

    assert calls == [
        {"league_id": 123, "year": 2026, "espn_s2": "secret", "swid": "{abc}"}
    ]
    assert "espn_s2" not in payload
    assert "swid" not in payload


def test_sync_projection_payload_requires_league_id() -> None:
    with pytest.raises(ValueError, match="league_id"):
        sync_projection_payload({"year": 2026}, league_factory=lambda **_: FakeLeague())
