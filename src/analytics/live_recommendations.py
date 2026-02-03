from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

from src.database.db import get_conn


# College basketball uses halves, not quarters
# First half often slightly lower scoring as teams feel each other out
DEFAULT_HALF_WEIGHTS = [0.48, 0.52]  # H1/H2 weights


@dataclass
class LiveRecommendation:
    game_id: str
    home_team: str
    away_team: str
    status: str
    period: int
    clock: str
    home_score: int
    away_score: int
    projected_total: float
    projected_spread: float  # home minus away


def _half_progress(clock_str: str | None) -> float:
    """
    Returns fraction of current half completed (0-1).
    College basketball: 20 minute halves.
    Expects clock like '08:32'; treats None/'Final' as 1.0.
    """
    if not clock_str or ":" not in clock_str:
        return 1.0
    try:
        mins, secs = clock_str.split(":")
        remaining = int(mins) * 60 + int(secs)
        total = 20 * 60  # 20 minute halves in college
        return 1.0 - (remaining / total)
    except Exception:
        return 1.0


def _elapsed_minutes(period: int, clock: str | None) -> float:
    """Approximate total elapsed minutes in regulation (0-40 for college)."""
    period = max(1, period)
    # College uses halves (period 1 = 1st half, period 2 = 2nd half)
    base = (period - 1) * 20  # 20 minutes per half
    prog = _half_progress(clock)
    return min(base + prog * 20, 40.0)  # 40 minute game


# Keep old function name for compatibility
_quarter_progress = _half_progress


def _team_points_per_minute(team_id: int, fallback_ppg: float = 72.0) -> float:
    """
    Estimate scoring rate using historical player_stats.
    fallback_ppg is a league-ish average if no data.
    College basketball averages ~70-75 PPG vs NBA's ~112.
    """
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ps.points, ps.minutes
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ?
            """,
            conn,
            params=[team_id],
        )
    if df.empty:
        return fallback_ppg / 40.0  # 40 minute college game
    points = df["points"].sum()
    minutes = df["minutes"].sum()
    if minutes <= 0:
        return fallback_ppg / 40.0
    return (points / minutes)


def _compute_projection(
    home_score: int,
    away_score: int,
    period: int,
    clock: str | None,
    home_rate: float,
    away_rate: float,
    home_court: float = 1.5,
    is_neutral_site: bool = False,
):
    """
    Compute projected final score.
    College basketball: 40 minute games.
    """
    # Rates are points per minute
    elapsed = _elapsed_minutes(period, clock)
    remaining = max(40.0 - elapsed, 0.0)  # 40 minute college game
    projected_home = home_score + home_rate * remaining
    projected_away = away_score + away_rate * remaining
    # Neutral site games have no home court advantage
    effective_home_court = 0.0 if is_neutral_site else home_court
    projected_spread = (projected_home - projected_away) + effective_home_court
    projected_total = projected_home + projected_away
    return projected_total, projected_spread


def load_live_games() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT lg.*, th.abbreviation AS home_abbr, ta.abbreviation AS away_abbr
            FROM live_games lg
            JOIN teams th ON th.team_id = lg.home_team_id
            JOIN teams ta ON ta.team_id = lg.away_team_id
            ORDER BY lg.start_time_utc DESC
            """,
            conn,
        )
    return df


def build_live_recommendations(weights: List[float] | None = None) -> List[LiveRecommendation]:
    df = load_live_games()
    recs: List[LiveRecommendation] = []
    for row in df.itertuples():
        home_rate = _team_points_per_minute(int(row.home_team_id))
        away_rate = _team_points_per_minute(int(row.away_team_id))
        projected_total, projected_spread = _compute_projection(
            int(row.home_score or 0),
            int(row.away_score or 0),
            int(row.period or 0),
            row.clock,
            home_rate,
            away_rate,
        )
        recs.append(
            LiveRecommendation(
                game_id=row.game_id,
                home_team=row.home_abbr,
                away_team=row.away_abbr,
                status=row.status or "",
                period=int(row.period or 0),
                clock=row.clock or "",
                home_score=int(row.home_score or 0),
                away_score=int(row.away_score or 0),
                projected_total=projected_total,
                projected_spread=projected_spread,
            )
        )
    return recs
