"""3-signal blend for in-game predictions with time-varying weights."""

import logging
from functools import lru_cache
from typing import Dict, Any, Optional

from src.database import db
from src.analytics.prediction import predict_matchup

logger = logging.getLogger(__name__)

# Weight anchors: (game_minute, pregame_weight, pace_weight, quarter_history_weight)
WEIGHT_ANCHORS = [
    (0, 0.90, 0.10, 0.00),
    (12, 0.60, 0.25, 0.15),
    (24, 0.35, 0.45, 0.20),
    (36, 0.15, 0.65, 0.20),
    (43, 0.05, 0.85, 0.10),
    (48, 0.02, 0.95, 0.03),
]

LEAGUE_AVG_PPM = 112.0 / 48.0  # ~2.333


def _interpolate_weights(minute: float) -> Dict[str, float]:
    """Linearly interpolate weights between anchors."""
    minute = max(0, minute)

    if minute >= 48:
        return {"pregame": 0.02, "pace": 0.95, "quarter": 0.03}

    # Find surrounding anchors
    for i in range(len(WEIGHT_ANCHORS) - 1):
        m0, p0, pa0, q0 = WEIGHT_ANCHORS[i]
        m1, p1, pa1, q1 = WEIGHT_ANCHORS[i + 1]

        if m0 <= minute <= m1:
            if m1 == m0:
                t = 0
            else:
                t = (minute - m0) / (m1 - m0)

            pregame = p0 + t * (p1 - p0)
            pace = pa0 + t * (pa1 - pa0)
            quarter = q0 + t * (q1 - q0)
            return {"pregame": pregame, "pace": pace, "quarter": quarter}

    return {"pregame": 0.02, "pace": 0.95, "quarter": 0.03}


@lru_cache(maxsize=64)
def _cached_pregame(home_team_id: int, away_team_id: int, date_key: str = "") -> Dict[str, Any]:
    """Cached pregame prediction (date_key busts cache at midnight)."""
    result = predict_matchup(home_team_id, away_team_id, game_date=date_key)
    return result.__dict__


def _pace_extrapolation(home_score: float, away_score: float,
                         minutes_elapsed: float) -> Dict[str, float]:
    """Signal 2: Pace-based extrapolation.

    In OT, projects through the end of the current overtime period (5 min each).
    """
    if minutes_elapsed <= 0:
        return {"home": 112.0, "away": 112.0}

    home_ppm = home_score / minutes_elapsed
    away_ppm = away_score / minutes_elapsed

    if minutes_elapsed <= 48:
        remaining = 48 - minutes_elapsed
    else:
        # OT: project through end of current OT period (5 min each)
        ot_elapsed = minutes_elapsed - 48.0
        ot_remaining = 5.0 - (ot_elapsed % 5.0)
        remaining = ot_remaining if ot_remaining < 5.0 else 0.0

    pace_home = home_score + home_ppm * remaining
    pace_away = away_score + away_ppm * remaining

    # Early dampening (first 4 minutes)
    if minutes_elapsed < 4:
        alpha = minutes_elapsed / 4.0
        pace_home = alpha * pace_home + (1 - alpha) * LEAGUE_AVG_PPM * 48
        pace_away = alpha * pace_away + (1 - alpha) * LEAGUE_AVG_PPM * 48

    return {"home": pace_home, "away": pace_away}


def _quarter_history_lookup(home_team_id: int, away_team_id: int,
                             quarter: int, home_cum: float,
                             away_cum: float) -> Optional[Dict[str, float]]:
    """Signal 3: Historical games with similar score through completed quarters.

    Uses game_quarter_scores table which has columns:
        game_id, team_id, q1, q2, q3, q4, ot, final_score, game_date, is_home
    """
    if quarter < 1 or quarter > 4:
        return None

    # Build cumulative expression for the given number of completed quarters
    q_cols = ['q1', 'q2', 'q3', 'q4'][:quarter]
    h_cum_expr = ' + '.join(f'COALESCE(h.{c}, 0)' for c in q_cols)
    a_cum_expr = ' + '.join(f'COALESCE(a.{c}, 0)' for c in q_cols)

    sql = f"""
        SELECT h.game_id,
               ({h_cum_expr}) AS h_cum,
               ({a_cum_expr}) AS a_cum,
               h.final_score AS h_final,
               a.final_score AS a_final
        FROM game_quarter_scores h
        JOIN game_quarter_scores a
          ON h.game_id = a.game_id AND a.is_home = 0
        WHERE h.is_home = 1
          AND h.final_score IS NOT NULL
          AND a.final_score IS NOT NULL
          AND ABS(({h_cum_expr}) - ?) <= 4
          AND ABS(({a_cum_expr}) - ?) <= 4
        LIMIT 50
    """

    try:
        rows = db.fetch_all(sql, (home_cum, away_cum))
    except Exception:
        logger.debug("Quarter history lookup failed", exc_info=True)
        return None

    if not rows or len(rows) < 5:
        return None

    avg_home = sum(r["h_final"] for r in rows) / len(rows)
    avg_away = sum(r["a_final"] for r in rows) / len(rows)

    return {"home": avg_home, "away": avg_away}


def live_predict(home_team_id: int, away_team_id: int,
                 home_score: float, away_score: float,
                 minutes_elapsed: float,
                 quarter: int = 0,
                 home_cum_quarters: float = 0,
                 away_cum_quarters: float = 0,
                 pregame_spread: Optional[float] = None,
                 pregame_total: Optional[float] = None) -> Dict[str, Any]:
    """3-signal blend for in-game prediction.
    
    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID
        home_score: Current home score
        away_score: Current away score
        minutes_elapsed: Minutes elapsed in the game
        quarter: Completed quarters (0-4)
        home_cum_quarters: Home cumulative score through completed quarters
        away_cum_quarters: Away cumulative score through completed quarters
        pregame_spread: Override pregame spread (optional)
        pregame_total: Override pregame total (optional)
    """
    weights = _interpolate_weights(minutes_elapsed)

    # Signal 1: Pregame model
    from datetime import datetime
    pregame = _cached_pregame(home_team_id, away_team_id,
                              date_key=datetime.now().strftime("%Y-%m-%d"))
    pg_home = pregame.get("predicted_home_score", 112)
    pg_away = pregame.get("predicted_away_score", 112)

    if pregame_spread is not None:
        mid = (pg_home + pg_away) / 2
        pg_home = mid + pregame_spread / 2
        pg_away = mid - pregame_spread / 2

    if pregame_total is not None:
        ratio = pg_home / (pg_home + pg_away) if (pg_home + pg_away) > 0 else 0.5
        pg_home = pregame_total * ratio
        pg_away = pregame_total * (1 - ratio)

    # Signal 2: Pace extrapolation
    pace = _pace_extrapolation(home_score, away_score, minutes_elapsed)

    # Signal 3: Quarter history
    qh = _quarter_history_lookup(home_team_id, away_team_id,
                                  quarter, home_cum_quarters, away_cum_quarters)

    # If quarter history unavailable, redistribute to pace
    if qh is None:
        pace_w = weights["pace"] + weights["quarter"]
        quarter_w = 0.0
        pregame_w = weights["pregame"]
    else:
        pace_w = weights["pace"]
        quarter_w = weights["quarter"]
        pregame_w = weights["pregame"]

    # Normalize weights
    total_w = pregame_w + pace_w + quarter_w
    if total_w > 0:
        pregame_w /= total_w
        pace_w /= total_w
        quarter_w /= total_w

    # Blend
    blend_home = pg_home * pregame_w + pace["home"] * pace_w
    blend_away = pg_away * pregame_w + pace["away"] * pace_w

    if qh:
        blend_home += qh["home"] * quarter_w
        blend_away += qh["away"] * quarter_w

    blend_spread = blend_home - blend_away
    blend_total = blend_home + blend_away

    # Current state
    current_margin = home_score - away_score

    # Advisory signals
    advisories = []
    pg_spread = pregame.get("predicted_spread", 0)
    pg_total = pregame.get("predicted_total", 224)

    if minutes_elapsed >= 6:
        projected_margin = blend_spread
        if projected_margin > pg_spread + 4:
            advisories.append({"type": "spread", "signal": "Home covering"})
        elif projected_margin < pg_spread - 4:
            advisories.append({"type": "spread", "signal": "Away covering"})

    if blend_total > pg_total + 4:
        advisories.append({"type": "total", "signal": "OVER likely"})
    elif blend_total < pg_total - 4:
        advisories.append({"type": "total", "signal": "UNDER likely"})

    return {
        "home_projected": round(blend_home, 1),
        "away_projected": round(blend_away, 1),
        "spread": round(blend_spread, 1),
        "total": round(blend_total, 1),
        "current_margin": current_margin,
        "minutes_elapsed": minutes_elapsed,
        "weights": {
            "pregame": round(pregame_w, 3),
            "pace": round(pace_w, 3),
            "quarter_history": round(quarter_w, 3),
        },
        "signals": {
            "pregame": {"home": round(pg_home, 1), "away": round(pg_away, 1)},
            "pace": {"home": round(pace["home"], 1), "away": round(pace["away"], 1)},
            "quarter_history": {"home": round(qh["home"], 1), "away": round(qh["away"], 1)} if qh else None,
        },
        "advisories": advisories,
        "pregame_spread": pg_spread,
        "pregame_total": pg_total,
    }


def clear_pregame_cache():
    """Clear cached pregame predictions."""
    _cached_pregame.cache_clear()
