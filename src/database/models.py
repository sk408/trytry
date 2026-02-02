from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Team:
    team_id: int
    name: str
    abbreviation: str
    conference: Optional[str] = None


@dataclass
class Player:
    player_id: int
    name: str
    team_id: int
    position: Optional[str] = None
    is_injured: bool = False
    injury_note: Optional[str] = None


@dataclass
class PlayerStat:
    player_id: int
    opponent_team_id: int
    is_home: bool
    game_date: date
    points: float
    rebounds: float
    assists: float
    minutes: float


@dataclass
class Prediction:
    home_team_id: int
    away_team_id: int
    game_date: date
    predicted_spread: float
    predicted_total: float
    actual_spread: Optional[float] = None
    actual_total: Optional[float] = None
