from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

STARTER_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")
FLEX_ELIGIBLE_POSITIONS = ("RB", "WR", "TE")
ROSTER_KEYS = STARTER_POSITIONS + ("FLEX", "BENCH")


class PassingScoring(BaseModel):
    yards: float = 0.04
    touchdowns: float = 4.0
    interceptions: float = -2.0
    two_point_conversions: float = 2.0


class RushingScoring(BaseModel):
    yards: float = 0.1
    touchdowns: float = 6.0
    two_point_conversions: float = 2.0


class ReceivingScoring(BaseModel):
    yards: float = 0.1
    touchdowns: float = 6.0
    receptions: float = 1.0
    two_point_conversions: float = 2.0


class MiscScoring(BaseModel):
    fumbles_lost: float = -2.0


class KickingScoring(BaseModel):
    field_goal_made: float = 3.0
    field_goal_0_39: float = 3.0
    field_goal_40_49: float = 4.0
    field_goal_50_plus: float = 5.0
    field_goal_missed: float = -1.0
    extra_point_made: float = 1.0
    extra_point_missed: float = -1.0


class DefenseScoring(BaseModel):
    sacks: float = 1.0
    interceptions: float = 2.0
    fumble_recoveries: float = 2.0
    touchdowns: float = 6.0
    safeties: float = 2.0
    blocked_kicks: float = 2.0
    return_touchdowns: float = 0.0
    points_allowed: dict[str, float] = Field(
        default_factory=lambda: {
            "0": 10.0,
            "1-6": 7.0,
            "7-13": 4.0,
            "14-20": 1.0,
            "21-27": 0.0,
            "28-34": -1.0,
            "35+": -4.0,
        }
    )


class ScoringConfig(BaseModel):
    passing: PassingScoring = Field(default_factory=PassingScoring)
    rushing: RushingScoring = Field(default_factory=RushingScoring)
    receiving: ReceivingScoring = Field(default_factory=ReceivingScoring)
    misc: MiscScoring = Field(default_factory=MiscScoring)
    kicking: KickingScoring = Field(default_factory=KickingScoring)
    defense: DefenseScoring = Field(default_factory=DefenseScoring)


class LeagueConfig(BaseModel):
    num_teams: int = 10
    team_names: list[str] | None = None
    roster_settings: dict[str, int] = Field(
        default_factory=lambda: {
            "QB": 1,
            "RB": 2,
            "WR": 2,
            "TE": 1,
            "FLEX": 1,
            "K": 1,
            "DEF": 1,
            "BENCH": 6,
        }
    )
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    draft_objective: Literal["roto", "head_to_head"] = "roto"
    ai_policies: list[str] = Field(
        default_factory=lambda: ["best_available", "scarcity", "balanced"]
    )
    max_extra_per_position: dict[str, int] = Field(
        default_factory=lambda: {
            "QB": 1,
            "RB": 4,
            "WR": 4,
            "TE": 2,
            "K": 0,
            "DEF": 0,
        }
    )
    week_start: int | None = None
    week_end: int | None = None
    max_draft_board: int = 25

    @field_validator("num_teams")
    @classmethod
    def validate_num_teams(cls, value: int) -> int:
        if value < 2:
            raise ValueError("num_teams must be at least 2")
        return value

    @field_validator("roster_settings")
    @classmethod
    def validate_roster_settings(cls, value: dict[str, int]) -> dict[str, int]:
        normalized = {key.upper(): int(count) for key, count in value.items()}
        unknown = sorted(set(normalized) - set(ROSTER_KEYS))
        if unknown:
            raise ValueError(f"unknown roster positions: {', '.join(unknown)}")
        for key in ROSTER_KEYS:
            normalized.setdefault(key, 0)
        negatives = [key for key, count in normalized.items() if count < 0]
        if negatives:
            raise ValueError(f"negative roster counts: {', '.join(negatives)}")
        if sum(normalized.values()) <= 0:
            raise ValueError("roster must have at least one slot")
        return normalized

    @field_validator("max_extra_per_position")
    @classmethod
    def validate_max_extra_per_position(cls, value: dict[str, int]) -> dict[str, int]:
        normalized = {key.upper(): int(count) for key, count in value.items()}
        unknown = sorted(set(normalized) - set(STARTER_POSITIONS))
        if unknown:
            raise ValueError(f"unknown max extra positions: {', '.join(unknown)}")
        for key in STARTER_POSITIONS:
            normalized.setdefault(
                key,
                {
                    "QB": 1,
                    "RB": 4,
                    "WR": 4,
                    "TE": 2,
                    "K": 0,
                    "DEF": 0,
                }[key],
            )
        negatives = [key for key, count in normalized.items() if count < 0]
        if negatives:
            raise ValueError(f"negative max extra counts: {', '.join(negatives)}")
        return normalized

    @model_validator(mode="after")
    def validate_config(self) -> LeagueConfig:
        if self.team_names is not None and len(self.team_names) != self.num_teams:
            raise ValueError("team_names must match num_teams")
        if self.week_start is not None and self.week_start < 1:
            raise ValueError("week_start must be positive")
        if self.week_end is not None and self.week_end < 1:
            raise ValueError("week_end must be positive")
        if (
            self.week_start is not None
            and self.week_end is not None
            and self.week_end < self.week_start
        ):
            raise ValueError("week_end must be greater than or equal to week_start")
        if not self.ai_policies:
            raise ValueError("at least one AI policy must be configured")
        return self

    @property
    def team_labels(self) -> list[str]:
        if self.team_names:
            return self.team_names
        return [f"Team {index + 1}" for index in range(self.num_teams)]

    @property
    def total_roster_slots(self) -> int:
        return sum(self.roster_settings.values())

    @property
    def starter_slots(self) -> dict[str, int]:
        return {
            position: self.roster_settings.get(position, 0)
            for position in STARTER_POSITIONS + ("FLEX",)
        }

    @property
    def draftable_positions(self) -> set[str]:
        positions = {
            position
            for position in STARTER_POSITIONS
            if self.roster_settings.get(position, 0) > 0
        }
        if self.roster_settings.get("FLEX", 0) > 0:
            positions.update(FLEX_ELIGIBLE_POSITIONS)
        return positions


def load_config(path: str | Path | None) -> LeagueConfig:
    if path is None:
        return LeagueConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LeagueConfig.model_validate(payload)
