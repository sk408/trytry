"""Enhanced live in-game prediction engine.

Blends three signals whose weights shift as the game progresses:

1. **Pre-game model** (``predict_matchup``) – dominant early, fades as
   real data accumulates.
2. **Pace extrapolation** – projects the current score rate to 48
   minutes using team-specific historical PPM.
3. **Quarter-history lookup** – "when this team scores X through Q_n,
   what is their typical final score?" – only active when enough
   historical data exists in ``game_quarter_scores``.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from src.database.db import get_conn


# ---------------------------------------------------------------------------
# Data class returned to callers
# ---------------------------------------------------------------------------

@dataclass
class LivePrediction:
    # Blended final projections
    projected_home_score: float
    projected_away_score: float
    projected_total: float
    projected_spread: float  # home − away (positive = home favored)

    # Component projections (for display / transparency)
    pregame_home: float = 0.0
    pregame_away: float = 0.0
    pregame_total: float = 0.0
    pregame_spread: float = 0.0

    pace_home: float = 0.0
    pace_away: float = 0.0
    pace_total: float = 0.0
    pace_spread: float = 0.0

    quarter_history_home: Optional[float] = None
    quarter_history_away: Optional[float] = None
    quarter_history_total: Optional[float] = None

    # Context
    minutes_elapsed: float = 0.0
    blend_weights: Dict[str, float] = field(default_factory=dict)

    # Advisory signals
    over_under_signal: str = ""   # "OVER likely", "UNDER likely", "Close"
    spread_signal: str = ""       # "Home covering", "Away covering", "On track"

    # Quarter analysis text (human-readable)
    quarter_analysis_home: str = ""
    quarter_analysis_away: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _elapsed_minutes(period: int, clock: Optional[str]) -> float:
    """Total elapsed regulation minutes (0-48)."""
    period = max(0, period)
    if period == 0:
        return 0.0
    base = (period - 1) * 12.0
    qprog = _quarter_progress(clock)
    return min(base + qprog * 12.0, 48.0)


def _quarter_progress(clock: Optional[str]) -> float:
    """Fraction of current quarter completed (0.0–1.0)."""
    if not clock or ":" not in clock:
        return 1.0
    try:
        parts = clock.split(":")
        remaining = int(parts[0]) * 60 + int(parts[1])
        return 1.0 - remaining / 720.0  # 12*60
    except Exception:
        return 1.0


def _team_ppm(team_id: int, is_home: bool) -> float:
    """Points-per-minute from historical game logs (location-filtered)."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT SUM(ps.points), SUM(ps.minutes)
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ? AND ps.is_home = ?
            """,
            (team_id, int(is_home)),
        ).fetchone()
    if not row or not row[1] or row[1] <= 0:
        return 112.0 / 48.0  # league-average fallback
    return float(row[0]) / float(row[1])


# ---------------------------------------------------------------------------
# Pre-game prediction (cached for the lifetime of the process)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _cached_pregame(
    home_team_id: int,
    away_team_id: int,
    game_date_str: str,
) -> Tuple[float, float, float, float]:
    """Return (home_score, away_score, total, spread) from predict_matchup.

    Cached per ``(home, away, date)`` so it only runs once per game.
    """
    from src.analytics.prediction import predict_matchup

    # Gather roster with play probabilities instead of binary is_injured.
    # Players with play_probability >= 0.3 are included, weighted by that
    # probability in aggregate_projection.
    PLAY_PROB_THRESHOLD = 0.3
    home_pids: list[int] = []
    away_pids: list[int] = []
    home_pw: dict[int, float] = {}
    away_pw: dict[int, float] = {}

    with get_conn() as conn:
        for team_id, pids, pw in [
            (home_team_id, home_pids, home_pw),
            (away_team_id, away_pids, away_pw),
        ]:
            rows = conn.execute(
                "SELECT player_id, is_injured, injury_note FROM players "
                "WHERE team_id = ?",
                (team_id,),
            ).fetchall()
            for r in rows:
                pid, is_injured, injury_note = r[0], r[1], r[2] if len(r) > 2 else None
                if not is_injured:
                    pids.append(pid)
                    pw[pid] = 1.0
                else:
                    # Compute play probability for injured players
                    try:
                        from src.analytics.injury_intelligence import compute_play_probability
                        from src.data.sync_service import _normalise_status_level, _extract_injury_keyword
                        note = injury_note or ""
                        status_raw = note.split(":")[0].strip() if ":" in note else note
                        injury_text = note.split(":", 1)[1].strip() if ":" in note else note
                        if "(" in injury_text:
                            injury_text = injury_text[:injury_text.rfind("(")].strip()
                        status_level = _normalise_status_level(status_raw)
                        keyword = _extract_injury_keyword(injury_text)
                        prob_result = compute_play_probability(pid, "", status_level, keyword, conn)
                        pp = prob_result.composite_probability
                    except Exception:
                        pp = 0.0
                    if pp >= PLAY_PROB_THRESHOLD:
                        pids.append(pid)
                        pw[pid] = pp

    if not home_pids or not away_pids:
        return 110.0, 110.0, 220.0, 0.0

    pred = predict_matchup(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_players=home_pids,
        away_players=away_pids,
        game_date=date.fromisoformat(game_date_str) if game_date_str else None,
        home_player_weights=home_pw,
        away_player_weights=away_pw,
    )
    return (
        pred.predicted_home_score,
        pred.predicted_away_score,
        pred.predicted_total,
        pred.predicted_spread,
    )


# ---------------------------------------------------------------------------
# Quarter-history lookup
# ---------------------------------------------------------------------------

_MIN_QUARTER_HISTORY_GAMES = 5


def _quarter_history_projection(
    team_id: int,
    completed_quarters: int,
    cumulative_score: int,
) -> Optional[float]:
    """When *team_id* has scored *cumulative_score* through
    *completed_quarters* quarter(s), what is their average final score
    historically?

    Returns ``None`` when fewer than ``_MIN_QUARTER_HISTORY_GAMES``
    matching games exist.
    """
    if completed_quarters < 1 or cumulative_score <= 0:
        return None

    # Build the cumulative sum expression for the number of completed Qs
    # e.g. Q1 only → q1, Q1+Q2 → q1+q2, etc.
    q_cols = ["q1", "q2", "q3", "q4"][:completed_quarters]
    cumsum_expr = " + ".join(q_cols)

    # Allow a tolerance band of ±4 points
    low = cumulative_score - 4
    high = cumulative_score + 4

    sql = f"""
        SELECT AVG(final_score), COUNT(*)
        FROM game_quarter_scores
        WHERE team_id = ?
          AND q1 IS NOT NULL
          AND ({cumsum_expr}) BETWEEN ? AND ?
          AND final_score IS NOT NULL
    """
    with get_conn() as conn:
        row = conn.execute(sql, (team_id, low, high)).fetchone()

    if not row or not row[0] or row[1] < _MIN_QUARTER_HISTORY_GAMES:
        return None
    return float(row[0])


# ---------------------------------------------------------------------------
# Blend weights by elapsed minutes
# ---------------------------------------------------------------------------

def _blend_weights(
    minutes: float,
    has_quarter_history: bool,
) -> Dict[str, float]:
    """Return ``{"pregame": w1, "pace": w2, "quarter": w3}`` that sum
    to 1.0 based on how far into the game we are."""

    if minutes <= 0:
        return {"pregame": 1.0, "pace": 0.0, "quarter": 0.0}

    # Linear interpolation through anchor points
    # (minutes, pregame_weight, pace_weight, quarter_weight)
    anchors = [
        (0,  0.90, 0.10, 0.00),
        (12, 0.60, 0.25, 0.15),   # end Q1
        (24, 0.35, 0.45, 0.20),   # halftime
        (36, 0.15, 0.65, 0.20),   # end Q3
        (43, 0.05, 0.85, 0.10),   # Q4 ~5 min left
        (48, 0.02, 0.95, 0.03),   # end regulation
    ]

    # Find surrounding anchors
    for i in range(len(anchors) - 1):
        lo_m, lo_pg, lo_pa, lo_qh = anchors[i]
        hi_m, hi_pg, hi_pa, hi_qh = anchors[i + 1]
        if lo_m <= minutes <= hi_m:
            t = (minutes - lo_m) / (hi_m - lo_m) if hi_m != lo_m else 0.0
            w_pg = lo_pg + t * (hi_pg - lo_pg)
            w_pa = lo_pa + t * (hi_pa - lo_pa)
            w_qh = lo_qh + t * (hi_qh - lo_qh)
            break
    else:
        # Past 48 min (OT) – almost entirely pace
        w_pg, w_pa, w_qh = 0.02, 0.95, 0.03

    # If quarter history unavailable, redistribute its weight to pace
    if not has_quarter_history:
        w_pa += w_qh
        w_qh = 0.0

    # Normalise (should already sum to ~1.0 but be safe)
    total = w_pg + w_pa + w_qh
    if total > 0:
        w_pg /= total
        w_pa /= total
        w_qh /= total

    return {"pregame": round(w_pg, 3), "pace": round(w_pa, 3), "quarter": round(w_qh, 3)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def live_predict(
    home_team_id: int,
    away_team_id: int,
    home_score: int,
    away_score: int,
    period: int,
    clock: Optional[str],
    game_id: Optional[str] = None,
    game_date_str: str = "",
) -> LivePrediction:
    """Produce a blended live prediction for an in-progress game.

    Parameters
    ----------
    home_team_id, away_team_id : int
        Internal (NBA API) team IDs.
    home_score, away_score : int
        Current live scores.
    period : int
        Current period (1-4, 5+ for OT).
    clock : str | None
        Remaining time in current quarter, e.g. ``"08:32"``.
    game_id : str | None
        ESPN game ID – used for opportunistic quarter-score storage.
    game_date_str : str
        ISO date string (``"2026-02-07"``).  Used for pregame cache key.

    Returns
    -------
    LivePrediction
    """

    if not game_date_str:
        game_date_str = str(date.today())

    minutes = _elapsed_minutes(period, clock)

    # ---- 1. Pre-game prediction (cached) ----
    pg_home, pg_away, pg_total, pg_spread = _cached_pregame(
        home_team_id, away_team_id, game_date_str,
    )

    # ---- 2. Pace projection ----
    if minutes > 0:
        home_ppm = home_score / minutes
        away_ppm = away_score / minutes
        remaining = max(48.0 - minutes, 0.0)
        pace_home = home_score + home_ppm * remaining
        pace_away = away_score + away_ppm * remaining
    else:
        # Game hasn't started – fall back to historical PPM
        hist_home_ppm = _team_ppm(home_team_id, is_home=True)
        hist_away_ppm = _team_ppm(away_team_id, is_home=False)
        pace_home = hist_home_ppm * 48.0
        pace_away = hist_away_ppm * 48.0

    # Very early in Q1, pure pace extrapolation is unstable.
    # Dampen toward team-historical rates in the first ~4 minutes.
    if 0 < minutes < 4:
        hist_home_ppm = _team_ppm(home_team_id, is_home=True)
        hist_away_ppm = _team_ppm(away_team_id, is_home=False)
        hist_home = hist_home_ppm * 48.0
        hist_away = hist_away_ppm * 48.0
        alpha = minutes / 4.0  # 0→1 over first 4 minutes
        pace_home = alpha * pace_home + (1 - alpha) * hist_home
        pace_away = alpha * pace_away + (1 - alpha) * hist_away

    pace_total = pace_home + pace_away
    pace_spread = pace_home - pace_away

    # ---- 3. Quarter-history lookup ----
    completed_quarters = max(0, period - 1) if _quarter_progress(clock) < 0.95 else period
    # Cumulative score through completed quarters (use linescores if stored,
    # otherwise approximate from current score minus partial-quarter points).
    # Simplification: use current score as proxy when exact Q scores
    # aren't available (works perfectly at quarter breaks).
    qh_home = _quarter_history_projection(home_team_id, completed_quarters, home_score)
    qh_away = _quarter_history_projection(away_team_id, completed_quarters, away_score)

    has_qh = qh_home is not None or qh_away is not None
    # If only one team has history, fill the other from pace
    qh_home_val = qh_home if qh_home is not None else pace_home
    qh_away_val = qh_away if qh_away is not None else pace_away

    # ---- 4. Build quarter analysis text ----
    q_label = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}.get(completed_quarters, "")
    qa_home = ""
    qa_away = ""
    if qh_home is not None and q_label:
        qa_home = f"When scoring ~{home_score} through {q_label}, avg final: {qh_home:.1f}"
    if qh_away is not None and q_label:
        qa_away = f"When scoring ~{away_score} through {q_label}, avg final: {qh_away:.1f}"

    # ---- 5. Blend ----
    weights = _blend_weights(minutes, has_qh)

    w_pg = weights["pregame"]
    w_pa = weights["pace"]
    w_qh = weights["quarter"]

    blended_home = w_pg * pg_home + w_pa * pace_home + w_qh * qh_home_val
    blended_away = w_pg * pg_away + w_pa * pace_away + w_qh * qh_away_val
    blended_total = blended_home + blended_away
    blended_spread = blended_home - blended_away

    # ---- 6. Advisory signals ----
    # Compare blended total to pre-game total (or Vegas O/U if available)
    if blended_total > pg_total + 4:
        ou_signal = "OVER likely"
    elif blended_total < pg_total - 4:
        ou_signal = "UNDER likely"
    else:
        ou_signal = "Close"

    current_margin = home_score - away_score
    if minutes > 6:
        if current_margin > pg_spread + 4:
            spread_signal = "Home covering"
        elif current_margin < pg_spread - 4:
            spread_signal = "Away covering"
        else:
            spread_signal = "On track"
    else:
        spread_signal = "Too early"

    # ---- 7. Opportunistically store quarter scores ----
    if game_id:
        try:
            from src.data.sync_service import sync_quarter_scores
            sync_quarter_scores(game_id)
        except Exception:
            pass  # non-critical

    return LivePrediction(
        projected_home_score=round(blended_home, 1),
        projected_away_score=round(blended_away, 1),
        projected_total=round(blended_total, 1),
        projected_spread=round(blended_spread, 1),
        pregame_home=round(pg_home, 1),
        pregame_away=round(pg_away, 1),
        pregame_total=round(pg_total, 1),
        pregame_spread=round(pg_spread, 1),
        pace_home=round(pace_home, 1),
        pace_away=round(pace_away, 1),
        pace_total=round(pace_total, 1),
        pace_spread=round(pace_spread, 1),
        quarter_history_home=round(qh_home, 1) if qh_home is not None else None,
        quarter_history_away=round(qh_away, 1) if qh_away is not None else None,
        quarter_history_total=round(qh_home_val + qh_away_val, 1) if has_qh else None,
        minutes_elapsed=round(minutes, 1),
        blend_weights=weights,
        over_under_signal=ou_signal,
        spread_signal=spread_signal,
        quarter_analysis_home=qa_home,
        quarter_analysis_away=qa_away,
    )
