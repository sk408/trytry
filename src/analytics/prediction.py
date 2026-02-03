from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from src.analytics.stats_engine import aggregate_projection
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
    home_court: float = 3.0,
    is_back_to_back_home: bool = False,
    is_back_to_back_away: bool = False,
) -> MatchupPrediction:
    """
    Predict game spread and total using comprehensive stats.
    
    Factors considered:
    - Base scoring projections (40% weight)
    - Home/away splits (30% weight)  
    - Head-to-head history (30% weight)
    - Turnover differential (affects spread)
    - True Shooting % advantage (affects spread)
    - 3PT shooting volume (affects total - more 3s = faster pace)
    - Defensive stats (affects total - good D = lower scores)
    - Back-to-back fatigue penalty
    """
    home_proj = aggregate_projection(home_players, opponent_team_id=away_team_id, is_home=True)
    away_proj = aggregate_projection(away_players, opponent_team_id=home_team_id, is_home=False)

    # ============ SPREAD CALCULATION ============
    
    # 1. Base spread from scoring differential
    spread = (home_proj["points"] - away_proj["points"]) + home_court
    
    # 2. Rest penalty (back-to-back games)
    spread += _rest_penalty(is_back_to_back_home)
    spread -= _rest_penalty(is_back_to_back_away)
    
    # 3. Turnover differential impact
    # Each net turnover forced is worth ~1.0 points (scoring opportunity + denied possession)
    # Home team's margin: (home steals - home turnovers) vs (away steals - away turnovers)
    home_to_margin = home_proj.get("turnover_margin", 0)  # steals - turnovers
    away_to_margin = away_proj.get("turnover_margin", 0)
    turnover_advantage = home_to_margin - away_to_margin
    spread += turnover_advantage * 0.8  # ~0.8 points per net turnover advantage
    
    # 4. True Shooting % advantage
    # Higher efficiency = more points on same possessions
    # A 5% TS difference on ~85 shots = ~4 point swing
    home_ts = home_proj.get("ts_pct", 55.0)  # default to league avg ~55%
    away_ts = away_proj.get("ts_pct", 55.0)
    ts_advantage = home_ts - away_ts
    spread += ts_advantage * 0.25  # ~0.25 pts per 1% TS advantage
    
    # ============ TOTAL CALCULATION ============
    
    # 1. Base total from combined scoring
    total = home_proj["points"] + away_proj["points"]
    
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
    # Above-average defense = lower totals
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
