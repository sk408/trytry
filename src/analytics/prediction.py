from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, Optional

from src.analytics.stats_engine import aggregate_projection
from src.database.db import get_conn


# ============ TUNABLE PREDICTION WEIGHTS ============
# All multipliers in one place for easy tuning via backtesting.

PREDICTION_CONFIG: Dict[str, float] = {
    # --- Spread factors ---
    "home_court_advantage": 2.5,       # College home court ~2.5 pts (vs NBA ~3.0)
    "back_to_back_penalty": -1.0,      # Points lost for back-to-back games
    "turnover_multiplier": 0.9,        # ~0.9 pts per net turnover advantage
    "ts_pct_multiplier": 0.20,         # ~0.20 pts per 1% TS% advantage
    "rebound_multiplier": 0.6,         # ~0.6 pts per net rebound advantage
    "assist_multiplier": 0.3,          # ~0.3 pts per assist advantage (offensive execution)
    "net_rating_multiplier": 0.15,     # ~0.15 pts per 1-pt net rating advantage
    "sos_multiplier": 1.5,             # ~1.5 pts per SOS tier advantage
    # --- Total factors ---
    "fg3_rate_multiplier": 0.12,       # +0.12 pts per 1% above avg 3PT rate
    "steals_multiplier": 0.35,         # Defensive impact on total (steals)
    "blocks_multiplier": 0.30,         # Defensive impact on total (blocks)
    "ft_rate_multiplier": 0.08,        # +0.08 pts per 1% above avg FT rate
    "second_chance_multiplier": 0.45,  # Extra pts per combined rebound above avg
    # --- Pace ---
    "college_avg_pace": 67.0,          # ~67 possessions per team per game
    # --- College averages (baselines) ---
    "avg_ts_pct": 54.0,               # College average TS%
    "avg_fg3_rate": 35.0,             # College average 3PT attempt rate
    "avg_ft_rate": 25.0,              # College average FT attempt rate
    "avg_steals_combined": 13.0,      # 6.5 per team
    "avg_blocks_combined": 7.0,       # 3.5 per team
    "avg_rebounds_per_team": 35.0,    # College average total RPG per team
    "avg_assists_per_team": 13.0,     # College average APG per team
    # --- Player projection weights ---
    "base_weight": 0.4,
    "location_weight": 0.3,
    "h2h_weight": 0.3,
    # --- Recency weights ---
    "last5_weight": 0.50,
    "last10_weight": 0.30,
    "season_weight": 0.20,
}


@dataclass
class MatchupPrediction:
    home_team_id: int
    away_team_id: int
    game_date: date
    predicted_spread: float
    predicted_total: float


def _rest_penalty(is_back_to_back: bool) -> float:
    return PREDICTION_CONFIG["back_to_back_penalty"] if is_back_to_back else 0.0


def predict_matchup(
    home_team_id: int,
    away_team_id: int,
    home_players: Iterable[int],
    away_players: Iterable[int],
    game_date: Optional[date] = None,
    home_court: Optional[float] = None,
    is_back_to_back_home: bool = False,
    is_back_to_back_away: bool = False,
    is_neutral_site: bool = False,  # Tournament games
    # Optional team-level metrics (from stats_engine team functions)
    home_pace: float = 0.0,
    away_pace: float = 0.0,
    home_net_rating: float = 0.0,
    away_net_rating: float = 0.0,
    home_sos: float = 0.0,
    away_sos: float = 0.0,
    # Optional per-matchup weight overrides (from autotuner DB)
    config_overrides: Optional[Dict[str, float]] = None,
) -> MatchupPrediction:
    """
    Predict game spread and total using comprehensive stats for college basketball.

    Factors considered:
    - Base scoring projections (weighted by base/location/h2h)
    - Turnover differential (affects spread)
    - True Shooting % advantage (affects spread)
    - Rebounding differential (affects spread + total via second-chance pts)
    - Assist differential (affects spread via offensive execution)
    - Net Rating differential (affects spread)
    - Strength of Schedule differential (affects spread)
    - Pace-adjusted total (faster pace -> higher total)
    - 3PT shooting volume (affects total)
    - Defensive stats (affects total)
    - Free-throw rate (affects total)
    - Neutral site / back-to-back adjustments

    When *config_overrides* is provided the values are merged on top of
    the global ``PREDICTION_CONFIG`` so that autotuned weights are used.
    """
    cfg = {**PREDICTION_CONFIG, **(config_overrides or {})}
    if home_court is None:
        home_court = cfg["home_court_advantage"]

    home_proj = aggregate_projection(home_players, opponent_team_id=away_team_id, is_home=True)
    away_proj = aggregate_projection(away_players, opponent_team_id=home_team_id, is_home=False)

    # ============ SPREAD CALCULATION ============

    # 1. Base spread from scoring differential + home court
    effective_home_court = 0.0 if is_neutral_site else home_court
    spread = (home_proj["points"] - away_proj["points"]) + effective_home_court

    # 2. Rest penalty (back-to-back games)
    spread += _rest_penalty(is_back_to_back_home)
    spread -= _rest_penalty(is_back_to_back_away)

    # 3. Turnover differential impact
    home_to_margin = home_proj.get("turnover_margin", 0)
    away_to_margin = away_proj.get("turnover_margin", 0)
    turnover_advantage = home_to_margin - away_to_margin
    spread += turnover_advantage * cfg["turnover_multiplier"]

    # 4. True Shooting % advantage
    home_ts = home_proj.get("ts_pct", cfg["avg_ts_pct"])
    away_ts = away_proj.get("ts_pct", cfg["avg_ts_pct"])
    ts_advantage = home_ts - away_ts
    spread += ts_advantage * cfg["ts_pct_multiplier"]

    # 5. Rebounding differential impact on spread
    # Each net rebound advantage is worth ~0.6 points (offensive rebounds create
    # extra possessions worth ~1.0 expected points, but only ~60% translate)
    home_reb = home_proj.get("rebounds", 0)
    away_reb = away_proj.get("rebounds", 0)
    rebound_advantage = home_reb - away_reb
    spread += rebound_advantage * cfg["rebound_multiplier"]

    # 6. Assist differential impact on spread
    # Teams with more assists run better offense and convert at higher rates
    home_ast = home_proj.get("assists", 0)
    away_ast = away_proj.get("assists", 0)
    assist_advantage = home_ast - away_ast
    spread += assist_advantage * cfg["assist_multiplier"]

    # 7. Net Rating differential (pace-adjusted efficiency)
    # Only applied when ratings are available (non-zero)
    if home_net_rating != 0.0 or away_net_rating != 0.0:
        net_rating_diff = home_net_rating - away_net_rating
        spread += net_rating_diff * cfg["net_rating_multiplier"]

    # 8. Strength of Schedule adjustment
    # A team that played a tougher schedule deserves more credit
    if home_sos != 0.0 or away_sos != 0.0:
        sos_advantage = home_sos - away_sos
        spread += sos_advantage * cfg["sos_multiplier"]

    # ============ TOTAL CALCULATION ============

    # 1. Base total from combined scoring
    total = home_proj["points"] + away_proj["points"]

    # 2. Pace-adjusted total scaling
    # When both teams play fast the total should be higher; slow = lower
    if home_pace > 0 and away_pace > 0:
        combined_pace = (home_pace + away_pace) / 2
        pace_factor = combined_pace / cfg["college_avg_pace"]
        total *= pace_factor
    else:
        # Fallback: use assist-based pacing boost when pace data unavailable
        total += 0.06 * (home_ast + away_ast)

    # 3. Three-point volume impact
    home_fg3_rate = home_proj.get("fg3_rate", cfg["avg_fg3_rate"])
    away_fg3_rate = away_proj.get("fg3_rate", cfg["avg_fg3_rate"])
    combined_fg3_rate = (home_fg3_rate + away_fg3_rate) / 2
    fg3_rate_above_avg = max(0, combined_fg3_rate - cfg["avg_fg3_rate"])
    total += fg3_rate_above_avg * cfg["fg3_rate_multiplier"]

    # 4. Defensive strength impact (steals + blocks slow the game)
    combined_steals = home_proj.get("steals", 0) + away_proj.get("steals", 0)
    combined_blocks = home_proj.get("blocks", 0) + away_proj.get("blocks", 0)
    steals_above_avg = max(0, combined_steals - cfg["avg_steals_combined"])
    blocks_above_avg = max(0, combined_blocks - cfg["avg_blocks_combined"])
    total -= (steals_above_avg * cfg["steals_multiplier"] + blocks_above_avg * cfg["blocks_multiplier"])

    # 5. Rebounding impact on total (second-chance points)
    # Above-average combined rebounding means more second-chance opportunities
    combined_reb = home_reb + away_reb
    reb_above_avg = max(0, combined_reb - cfg["avg_rebounds_per_team"] * 2)
    total += reb_above_avg * cfg["second_chance_multiplier"]

    # 6. Free throw rate impact
    home_ft_rate = home_proj.get("ft_rate", cfg["avg_ft_rate"])
    away_ft_rate = away_proj.get("ft_rate", cfg["avg_ft_rate"])
    combined_ft_rate = (home_ft_rate + away_ft_rate) / 2
    ft_rate_above_avg = max(0, combined_ft_rate - cfg["avg_ft_rate"])
    total += ft_rate_above_avg * cfg["ft_rate_multiplier"]

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
