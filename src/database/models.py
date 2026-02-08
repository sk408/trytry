from __future__ import annotations

from dataclasses import dataclass, field
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
    height: Optional[str] = None
    weight: Optional[str] = None
    age: Optional[int] = None
    experience: Optional[int] = None


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
    win_loss: Optional[str] = None
    personal_fouls: float = 0.0


@dataclass
class Prediction:
    home_team_id: int
    away_team_id: int
    game_date: date
    predicted_spread: float
    predicted_total: float
    actual_spread: Optional[float] = None
    actual_total: Optional[float] = None


@dataclass
class TeamMetrics:
    """Comprehensive team-level metrics from multiple NBA API endpoints."""
    team_id: int
    season: str
    # Record
    gp: int = 0
    w: int = 0
    l: int = 0
    w_pct: float = 0.0
    # NBA Estimated Metrics
    e_off_rating: Optional[float] = None
    e_def_rating: Optional[float] = None
    e_net_rating: Optional[float] = None
    e_pace: Optional[float] = None
    # Dashboard Advanced
    off_rating: Optional[float] = None
    def_rating: Optional[float] = None
    net_rating: Optional[float] = None
    pace: Optional[float] = None
    efg_pct: Optional[float] = None
    ts_pct: Optional[float] = None
    # Four Factors (team)
    ff_efg_pct: Optional[float] = None
    ff_fta_rate: Optional[float] = None
    ff_tm_tov_pct: Optional[float] = None
    ff_oreb_pct: Optional[float] = None
    # Four Factors (opponent)
    opp_efg_pct: Optional[float] = None
    opp_fta_rate: Optional[float] = None
    opp_tm_tov_pct: Optional[float] = None
    opp_oreb_pct: Optional[float] = None
    # Clutch
    clutch_net_rating: Optional[float] = None
    # Home / Road
    home_w: int = 0
    home_l: int = 0
    road_w: int = 0
    road_l: int = 0


@dataclass
class PlayerImpact:
    """Player on/off impact and estimated advanced metrics."""
    player_id: int
    team_id: int
    season: str
    # On/Off
    net_rating_diff: Optional[float] = None
    on_court_minutes: Optional[float] = None
    # Estimated Metrics
    e_usg_pct: Optional[float] = None
    e_off_rating: Optional[float] = None
    e_def_rating: Optional[float] = None
    e_net_rating: Optional[float] = None
