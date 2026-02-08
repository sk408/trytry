from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from src.analytics.stats_engine import (
    aggregate_projection,
    get_defensive_rating,
    get_opponent_defensive_factor,
    get_home_court_advantage,
)
from src.analytics.autotune import get_team_tuning
from src.database.db import get_conn


@dataclass
class MatchupPrediction:
    home_team_id: int
    away_team_id: int
    game_date: date
    predicted_spread: float
    predicted_total: float


def _rest_penalty(is_back_to_back: bool) -> float:
    return -1.0 if is_back_to_back else 0.0


def predict_matchup(
    home_team_id: int,
    away_team_id: int,
    home_players: Iterable[int],
    away_players: Iterable[int],
    game_date: Optional[date] = None,
    home_court: Optional[float] = None,
    is_back_to_back_home: bool = False,
    is_back_to_back_away: bool = False,
) -> MatchupPrediction:
    """
    Predict game spread and total using comprehensive stats.
    
    Factors considered:
    - Base scoring projections with opponent defensive adjustment
    - Dynamic home court advantage (team-specific, 1.5-5.0 range)
    - Home/away splits (30% weight)
    - Head-to-head history (30% weight)
    - Turnover differential (affects spread)
    - True Shooting % advantage (affects spread)
    - Offensive/Defensive rating matchup comparison (affects spread)
    - Rebound differential (affects spread)
    - 3PT shooting volume (affects total)
    - Defensive stats (affects total - good D = lower scores)
    - Offensive rebounds / second-chance points (affects total)
    - Pace estimation (affects total)
    - Back-to-back fatigue penalty
    - Recent form weighting (last 5 games weighted 60%, older 40%)
    """
    home_proj = aggregate_projection(home_players, opponent_team_id=away_team_id, is_home=True)
    away_proj = aggregate_projection(away_players, opponent_team_id=home_team_id, is_home=False)

    # ============ DYNAMIC HOME COURT ADVANTAGE ============
    # Use team-specific HCA unless explicitly overridden
    if home_court is None:
        home_court = get_home_court_advantage(home_team_id)
    
    # ============ OPPONENT DEFENSIVE ADJUSTMENT ============
    # Scale projected points by how well the opponent defends
    # Factor < 1.0 = opponent has good defense, reduces scoring
    # Factor > 1.0 = opponent has bad defense, increases scoring
    away_def_factor = get_opponent_defensive_factor(away_team_id)
    home_def_factor = get_opponent_defensive_factor(home_team_id)
    # Dampen: pull 50% toward 1.0 to avoid overcorrection
    away_def_factor = 1.0 + (away_def_factor - 1.0) * 0.5
    home_def_factor = 1.0 + (home_def_factor - 1.0) * 0.5
    
    home_base_pts = home_proj["points"] * away_def_factor
    away_base_pts = away_proj["points"] * home_def_factor

    # ============ AUTOTUNE CORRECTIONS ============
    # Apply per-team scoring corrections (if available from autotune)
    home_tuning = get_team_tuning(home_team_id)
    away_tuning = get_team_tuning(away_team_id)
    if home_tuning:
        home_base_pts += home_tuning["home_pts_correction"]
    if away_tuning:
        away_base_pts += away_tuning["away_pts_correction"]

    # ============ SPREAD CALCULATION ============
    
    # 1. Base spread from defense-adjusted scoring differential
    spread = (home_base_pts - away_base_pts) + home_court
    
    # 2. Rest penalty (back-to-back games)
    spread += _rest_penalty(is_back_to_back_home)
    spread -= _rest_penalty(is_back_to_back_away)
    
    # 3. Turnover differential impact
    # Each net turnover forced is worth ~1.0 points (scoring opportunity + denied possession)
    home_to_margin = home_proj.get("turnover_margin", 0)  # steals - turnovers
    away_to_margin = away_proj.get("turnover_margin", 0)
    turnover_advantage = home_to_margin - away_to_margin
    spread += turnover_advantage * 0.8  # ~0.8 points per net turnover advantage
    
    # 4. True Shooting % advantage
    # Higher efficiency = more points on same possessions
    home_ts = home_proj.get("ts_pct", 55.0)  # default to league avg ~55%
    away_ts = away_proj.get("ts_pct", 55.0)
    ts_advantage = home_ts - away_ts
    spread += ts_advantage * 0.25  # ~0.25 pts per 1% TS advantage
    
    # 5. Offensive/Defensive rating matchup comparison
    # Compares each team's offensive efficiency vs the opponent's defensive efficiency
    home_off_rating = home_proj.get("off_rating", 110.0)
    away_off_rating = away_proj.get("off_rating", 110.0)
    home_def_rating = get_defensive_rating(home_team_id)
    away_def_rating = get_defensive_rating(away_team_id)
    # Home offense vs away defense, and vice versa
    home_matchup_edge = home_off_rating - away_def_rating  # positive = home has edge
    away_matchup_edge = away_off_rating - home_def_rating
    rating_spread_adj = (home_matchup_edge - away_matchup_edge) * 0.1
    spread += rating_spread_adj
    
    # 6. Rebound differential impact
    # Teams that outrebound opponents control the possession battle
    home_reb = home_proj.get("rebounds", 0)
    away_reb = away_proj.get("rebounds", 0)
    spread += (home_reb - away_reb) * 0.15  # ~0.15 pts per rebound advantage
    
    # ============ TOTAL CALCULATION ============
    
    # 1. Base total from defense-adjusted scoring
    total = home_base_pts + away_base_pts
    
    # 2. Pacing boost from assists (more ball movement = more scoring)
    total += 0.08 * (home_proj.get("assists", 0) + away_proj.get("assists", 0))
    
    # 3. Three-point volume impact
    # High 3PT attempt rates = more variance but typically faster pace
    # League avg ~38% of shots are 3s
    home_fg3_rate = home_proj.get("fg3_rate", 38.0)
    away_fg3_rate = away_proj.get("fg3_rate", 38.0)
    combined_fg3_rate = (home_fg3_rate + away_fg3_rate) / 2
    # Teams shooting lots of 3s = slightly higher totals
    fg3_rate_above_avg = max(0, combined_fg3_rate - 38.0)
    total += fg3_rate_above_avg * 0.15  # +0.15 pts per 1% above avg 3PT rate
    
    # 4. Defensive strength impact (steals + blocks slow the game)
    # Good defensive teams = lower pace, fewer easy baskets
    combined_steals = home_proj.get("steals", 0) + away_proj.get("steals", 0)
    combined_blocks = home_proj.get("blocks", 0) + away_proj.get("blocks", 0)
    # League avg: ~7-8 steals, ~5 blocks per team per game
    steals_above_avg = max(0, combined_steals - 14.0)  # 14 = 7 per team avg
    blocks_above_avg = max(0, combined_blocks - 10.0)  # 10 = 5 per team avg
    total -= (steals_above_avg * 0.3 + blocks_above_avg * 0.25)
    
    # 5. Free throw rate impact
    # High FT rate = more stoppages, slower pace but more points
    home_ft_rate = home_proj.get("ft_rate", 25.0)  # ~25% FTA/FGA is avg
    away_ft_rate = away_proj.get("ft_rate", 25.0)
    combined_ft_rate = (home_ft_rate + away_ft_rate) / 2
    ft_rate_above_avg = max(0, combined_ft_rate - 25.0)
    total += ft_rate_above_avg * 0.1  # Slight boost for FT-heavy games
    
    # 6. Offensive rebound impact (second-chance points)
    # Each offensive rebound is worth ~1.1 expected points (extra possession)
    home_oreb = home_proj.get("oreb", 0)
    away_oreb = away_proj.get("oreb", 0)
    combined_oreb = home_oreb + away_oreb
    total += (combined_oreb - 20.0) * 0.4  # 20 = ~10 per team avg
    
    # 7. Pace adjustment
    # Fast-paced matchups produce more possessions and higher totals
    home_pace = home_proj.get("pace", 96.0)
    away_pace = away_proj.get("pace", 96.0)
    expected_pace = (home_pace + away_pace) / 2
    pace_factor = (expected_pace - 96.0) / 96.0  # % above/below league avg pace
    total *= (1 + pace_factor * 0.5)  # Dampen to avoid overcorrection

    return MatchupPrediction(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=game_date or date.today(),
        predicted_spread=spread,
        predicted_total=total,
    )


def save_prediction(prediction: MatchupPrediction) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (home_team_id, away_team_id, game_date, predicted_spread, predicted_total)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                prediction.home_team_id,
                prediction.away_team_id,
                prediction.game_date,
                prediction.predicted_spread,
                prediction.predicted_total,
            ),
        )
        conn.commit()
