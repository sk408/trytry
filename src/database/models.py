"""Data models for college basketball analytics.

Updated from NBA models to support college-specific fields:
- Division (D1, D2, D3)
- Conference (ACC, Big Ten, etc.)
- Gender (mens, womens)
- Class year (Freshman, Sophomore, etc.)
"""
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
    division: str = "D1"  # D1, D2, D3
    gender: str = "mens"  # mens, womens


@dataclass
class Player:
    player_id: int
    name: str
    team_id: int
    position: Optional[str] = None
    is_injured: bool = False
    injury_note: Optional[str] = None
    class_year: Optional[str] = None  # Freshman, Sophomore, Junior, Senior, Grad
    jersey_number: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None


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
    # Additional stats for college
    steals: float = 0.0
    blocks: float = 0.0
    turnovers: float = 0.0
    game_id: Optional[str] = None


@dataclass
class Prediction:
    home_team_id: int
    away_team_id: int
    game_date: date
    predicted_spread: float
    predicted_total: float
    actual_spread: Optional[float] = None
    actual_total: Optional[float] = None
    is_neutral_site: bool = False  # Important for tournament games
