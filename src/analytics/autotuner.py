"""
Autotuner for prediction weights.

Optimises the multipliers in PREDICTION_CONFIG on a per-team or global
basis by replaying historical games and minimising prediction error
using scipy's Nelder-Mead simplex optimiser.

The key to speed is *precomputation*: all per-game "feature differentials"
(rebound diff, assist diff, …) are computed once up-front so that each
iteration of the optimiser only needs to do cheap arithmetic -- no DB
queries inside the loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analytics.backtester import (
    _get_team_agg_stats_before,
    _get_opponent_ppg_before,
    calculate_injury_adjustment,
    find_similar_teams,
    get_actual_game_results,
)
from src.analytics.prediction import PREDICTION_CONFIG
from src.database.db import get_conn


# ============ Tunable weight keys (order matters for the param vector) ============

TUNABLE_KEYS: List[str] = [
    "home_court_advantage",
    "turnover_multiplier",
    "ts_pct_multiplier",
    "rebound_multiplier",
    "assist_multiplier",
    "net_rating_multiplier",
    "sos_multiplier",
    "fg3_rate_multiplier",
    "steals_multiplier",
    "blocks_multiplier",
    "ft_rate_multiplier",
    "second_chance_multiplier",
]

# Reasonable bounds to prevent nonsensical values after optimisation.
WEIGHT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "home_court_advantage":  (0.0, 6.0),
    "turnover_multiplier":   (0.0, 3.0),
    "ts_pct_multiplier":     (0.0, 1.0),
    "rebound_multiplier":    (0.0, 2.0),
    "assist_multiplier":     (0.0, 1.5),
    "net_rating_multiplier": (0.0, 0.5),
    "sos_multiplier":        (0.0, 5.0),
    "fg3_rate_multiplier":   (0.0, 0.5),
    "steals_multiplier":     (0.0, 1.0),
    "blocks_multiplier":     (0.0, 1.0),
    "ft_rate_multiplier":    (0.0, 0.3),
    "second_chance_multiplier": (0.0, 1.5),
}


# ============ Precomputed game features ============

@dataclass
class GameFeatures:
    """All the intermediate values needed to compute spread & total for one game."""
    # Spread components (raw differentials – multiplied by weights at eval time)
    scoring_diff: float = 0.0       # home_proj - away_proj (fixed)
    injury_adj: float = 0.0         # net injury adjustment (fixed)
    turnover_diff: float = 0.0
    ts_diff: float = 0.0
    reb_diff: float = 0.0
    ast_diff: float = 0.0
    net_rating_diff: float = 0.0
    sos_diff: float = 0.0
    # Total components
    base_total: float = 0.0         # home_proj + away_proj (fixed)
    combined_pace: float = 0.0
    fg3_above_avg: float = 0.0
    steals_above_avg: float = 0.0
    blocks_above_avg: float = 0.0
    reb_above_avg: float = 0.0
    ft_rate_above_avg: float = 0.0
    # Actual results
    actual_spread: float = 0.0
    actual_total: float = 0.0


@dataclass
class AutotuneResult:
    """Result of an autotune run."""
    team_id: Optional[int]         # None = global
    weights_before: Dict[str, float] = field(default_factory=dict)
    weights_after: Dict[str, float] = field(default_factory=dict)
    spread_error_before: float = 0.0
    spread_error_after: float = 0.0
    total_error_before: float = 0.0
    total_error_after: float = 0.0
    games_used: int = 0
    iterations: int = 0


# ============ DB helpers: load / save / clear ============

def load_team_weights(team_id: Optional[int]) -> Dict[str, float]:
    """
    Load tuned weight overrides for *team_id* (or global when ``None``).

    Returns only the overridden keys.  Callers should merge the result
    over ``PREDICTION_CONFIG`` to get a complete config dict.
    """
    with get_conn() as conn:
        if team_id is None:
            rows = conn.execute(
                "SELECT weight_key, weight_value FROM team_weights WHERE team_id IS NULL"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT weight_key, weight_value FROM team_weights WHERE team_id = ?",
                (team_id,),
            ).fetchall()
    return {k: v for k, v in rows}


def save_team_weights(
    team_id: Optional[int],
    weights: Dict[str, float],
    spread_err_before: float = 0.0,
    spread_err_after: float = 0.0,
    total_err_before: float = 0.0,
    total_err_after: float = 0.0,
) -> None:
    """Persist tuned weights, replacing any existing overrides for the team."""
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        if team_id is None:
            conn.execute("DELETE FROM team_weights WHERE team_id IS NULL")
        else:
            conn.execute("DELETE FROM team_weights WHERE team_id = ?", (team_id,))
        conn.executemany(
            """
            INSERT INTO team_weights
                (team_id, weight_key, weight_value, tuned_at,
                 spread_error_before, spread_error_after,
                 total_error_before, total_error_after)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (team_id, k, v, now,
                 spread_err_before, spread_err_after,
                 total_err_before, total_err_after)
                for k, v in weights.items()
            ],
        )
        conn.commit()


def clear_team_weights(team_id: Optional[int]) -> None:
    """Remove all weight overrides for a team (or global when ``None``)."""
    with get_conn() as conn:
        if team_id is None:
            conn.execute("DELETE FROM team_weights WHERE team_id IS NULL")
        else:
            conn.execute("DELETE FROM team_weights WHERE team_id = ?", (team_id,))
        conn.commit()


def get_effective_config(
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    Build a complete config by layering overrides on top of defaults:
      PREDICTION_CONFIG  <  global DB overrides  <  per-team overrides

    When both teams have a per-team override for the same key the two
    values are averaged.
    """
    cfg = dict(PREDICTION_CONFIG)

    # Layer 1: global DB overrides
    global_ov = load_team_weights(None)
    cfg.update(global_ov)

    # Layer 2: per-team overrides
    if home_team_id is not None or away_team_id is not None:
        home_ov = load_team_weights(home_team_id) if home_team_id else {}
        away_ov = load_team_weights(away_team_id) if away_team_id else {}

        all_keys = set(home_ov) | set(away_ov)
        for k in all_keys:
            if k in home_ov and k in away_ov:
                cfg[k] = (home_ov[k] + away_ov[k]) / 2
            elif k in home_ov:
                # Blend with current cfg at half-strength
                cfg[k] = (home_ov[k] + cfg.get(k, home_ov[k])) / 2
            else:
                cfg[k] = (away_ov[k] + cfg.get(k, away_ov[k])) / 2

    return cfg


# ============ Feature precomputation ============

def _precompute_game_features(
    games: List[dict],
    all_games_df: pd.DataFrame,
    progress: Callable[[str], None],
) -> List[GameFeatures]:
    """
    For each game dict (with home_team_id, away_team_id, game_date,
    home_score, away_score) precompute all intermediate values so the
    optimiser loop is cheap.
    """
    cfg = PREDICTION_CONFIG
    features: List[GameFeatures] = []

    for idx, g in enumerate(games):
        if idx % 25 == 0:
            progress(f"Precomputing features: game {idx + 1}/{len(games)}...")

        gd = g["game_date"]
        home_id = g["home_team_id"]
        away_id = g["away_team_id"]

        # ---- Base projections (from game-score history) ----
        before = all_games_df[all_games_df["game_date"] < gd]
        if before.empty:
            continue

        def _team_ppg(tid, as_home):
            h = before[before["home_team_id"] == tid]
            a = before[before["away_team_id"] == tid]
            scores = []
            if as_home is None or as_home:
                scores.extend(h["home_score"].tolist())
            if as_home is None or not as_home:
                scores.extend(a["away_score"].tolist())
            return float(np.mean(scores)) if scores else 0.0

        home_proj = _team_ppg(home_id, True) * 0.5 + _team_ppg(home_id, None) * 0.5
        away_proj = _team_ppg(away_id, False) * 0.5 + _team_ppg(away_id, None) * 0.5
        if home_proj == 0 or away_proj == 0:
            continue

        # Injury adjustment (fixed, doesn't depend on weights)
        home_adj, _ = calculate_injury_adjustment(home_id, gd, home_proj)
        away_adj, _ = calculate_injury_adjustment(away_id, gd, away_proj)
        home_proj += home_adj
        away_proj += away_adj

        # ---- Aggregated stats before this game ----
        home_agg = _get_team_agg_stats_before(home_id, gd)
        away_agg = _get_team_agg_stats_before(away_id, gd)

        if home_agg["games"] < 3 or away_agg["games"] < 3:
            continue

        gf = GameFeatures()
        gf.scoring_diff = home_proj - away_proj
        gf.injury_adj = home_adj - away_adj
        gf.base_total = home_proj + away_proj
        gf.actual_spread = float(g["home_score"]) - float(g["away_score"])
        gf.actual_total = float(g["home_score"]) + float(g["away_score"])

        # Turnover differential
        home_to = home_agg["stl"] - home_agg["tov"]
        away_to = away_agg["stl"] - away_agg["tov"]
        gf.turnover_diff = home_to - away_to

        # TS% differential
        gf.ts_diff = home_agg["ts_pct"] - away_agg["ts_pct"]

        # Rebound differential
        gf.reb_diff = home_agg["reb"] - away_agg["reb"]

        # Assist differential
        gf.ast_diff = home_agg["ast"] - away_agg["ast"]

        # Net rating differential
        home_opp = _get_opponent_ppg_before(home_id, gd)
        away_opp = _get_opponent_ppg_before(away_id, gd)
        hp, ap = home_agg["pace"], away_agg["pace"]
        if hp > 0 and ap > 0 and home_opp > 0 and away_opp > 0:
            h_ortg = (home_agg["pts"] / hp) * 100
            h_drtg = (home_opp / hp) * 100
            a_ortg = (away_agg["pts"] / ap) * 100
            a_drtg = (away_opp / ap) * 100
            gf.net_rating_diff = (h_ortg - h_drtg) - (a_ortg - a_drtg)

        # SOS differential
        if home_opp > 0 and away_opp > 0:
            avg_opp = (home_opp + away_opp) / 2
            gf.sos_diff = ((home_opp - avg_opp) - (away_opp - avg_opp)) / 10.0

        # Pace
        gf.combined_pace = (hp + ap) / 2 if hp > 0 and ap > 0 else 0.0

        # Total-side features (raw values, thresholded in objective using cfg baselines)
        combined_fg3a = home_agg.get("fg3a", 0) + away_agg.get("fg3a", 0)
        combined_fga = home_agg.get("fga", 0) + away_agg.get("fga", 0)
        combined_fg3_rate = (combined_fg3a / combined_fga * 100) if combined_fga > 0 else 0.0
        gf.fg3_above_avg = max(0.0, combined_fg3_rate - cfg["avg_fg3_rate"])

        combined_steals = home_agg["stl"] + away_agg["stl"]
        combined_blocks = home_agg["blk"] + away_agg["blk"]
        gf.steals_above_avg = max(0.0, combined_steals - cfg["avg_steals_combined"])
        gf.blocks_above_avg = max(0.0, combined_blocks - cfg["avg_blocks_combined"])

        combined_reb = home_agg["reb"] + away_agg["reb"]
        gf.reb_above_avg = max(0.0, combined_reb - cfg["avg_rebounds_per_team"] * 2)

        combined_fta = home_agg.get("fta", 0) + away_agg.get("fta", 0)
        combined_ft_rate = (combined_fta / combined_fga * 100) if combined_fga > 0 else 0.0
        gf.ft_rate_above_avg = max(0.0, combined_ft_rate - cfg["avg_ft_rate"])

        features.append(gf)

    progress(f"Precomputed features for {len(features)} games")
    return features


# ============ Objective function ============

def _objective(params: np.ndarray, features: List[GameFeatures]) -> float:
    """
    Mean combined error across all games for the trial weight vector.

    Error = |predicted_spread - actual_spread| + 0.5 * |predicted_total - actual_total|

    Total error is weighted at 0.5x because spreads are the primary
    betting target and total prediction is inherently noisier.
    """
    # Unpack params into a dict
    w = {k: float(params[i]) for i, k in enumerate(TUNABLE_KEYS)}

    total_err = 0.0
    for gf in features:
        # ---- Spread ----
        spread = (
            gf.scoring_diff
            + w["home_court_advantage"]
            + gf.injury_adj
            + gf.turnover_diff * w["turnover_multiplier"]
            + gf.ts_diff * w["ts_pct_multiplier"]
            + gf.reb_diff * w["rebound_multiplier"]
            + gf.ast_diff * w["assist_multiplier"]
            + gf.net_rating_diff * w["net_rating_multiplier"]
            + gf.sos_diff * w["sos_multiplier"]
        )

        # ---- Total ----
        total = gf.base_total
        if gf.combined_pace > 0:
            pace_factor = gf.combined_pace / PREDICTION_CONFIG["college_avg_pace"]
            total *= pace_factor
        total += gf.fg3_above_avg * w["fg3_rate_multiplier"]
        total -= (gf.steals_above_avg * w["steals_multiplier"]
                  + gf.blocks_above_avg * w["blocks_multiplier"])
        total += gf.reb_above_avg * w["second_chance_multiplier"]
        total += gf.ft_rate_above_avg * w["ft_rate_multiplier"]

        spread_err = abs(spread - gf.actual_spread)
        total_err_g = abs(total - gf.actual_total)
        total_err += spread_err + total_err_g * 0.5

    return total_err / len(features) if features else 999.0


def _compute_errors(params: np.ndarray, features: List[GameFeatures]) -> Tuple[float, float]:
    """Return (avg_spread_error, avg_total_error) separately for reporting."""
    w = {k: float(params[i]) for i, k in enumerate(TUNABLE_KEYS)}
    spread_errs, total_errs = [], []

    for gf in features:
        spread = (
            gf.scoring_diff + w["home_court_advantage"] + gf.injury_adj
            + gf.turnover_diff * w["turnover_multiplier"]
            + gf.ts_diff * w["ts_pct_multiplier"]
            + gf.reb_diff * w["rebound_multiplier"]
            + gf.ast_diff * w["assist_multiplier"]
            + gf.net_rating_diff * w["net_rating_multiplier"]
            + gf.sos_diff * w["sos_multiplier"]
        )
        total = gf.base_total
        if gf.combined_pace > 0:
            total *= gf.combined_pace / PREDICTION_CONFIG["college_avg_pace"]
        total += gf.fg3_above_avg * w["fg3_rate_multiplier"]
        total -= (gf.steals_above_avg * w["steals_multiplier"]
                  + gf.blocks_above_avg * w["blocks_multiplier"])
        total += gf.reb_above_avg * w["second_chance_multiplier"]
        total += gf.ft_rate_above_avg * w["ft_rate_multiplier"]

        spread_errs.append(abs(spread - gf.actual_spread))
        total_errs.append(abs(total - gf.actual_total))

    n = len(features) or 1
    return sum(spread_errs) / n, sum(total_errs) / n


def _clamp_weights(params: np.ndarray) -> np.ndarray:
    """Clamp each weight to its allowed range."""
    clamped = params.copy()
    for i, k in enumerate(TUNABLE_KEYS):
        lo, hi = WEIGHT_BOUNDS[k]
        clamped[i] = max(lo, min(hi, clamped[i]))
    return clamped


# ============ Public autotune entry points ============

def autotune_team(
    team_id: int,
    progress_cb: Optional[Callable[[str], None]] = None,
    max_iter: int = 500,
) -> AutotuneResult:
    """
    Optimise prediction weights for all games involving *team_id*.

    Returns an ``AutotuneResult`` with before/after errors and the
    tuned weights.  Does **not** persist to DB -- call ``save_team_weights``
    afterwards if the user accepts the result.
    """
    progress = progress_cb or (lambda _: None)
    progress(f"Starting autotune for team {team_id}...")

    # Load all game results
    all_games_df = get_actual_game_results()
    if all_games_df.empty:
        progress("[ERROR] No game data found")
        return AutotuneResult(team_id=team_id)

    # Filter to games involving this team
    mask = (all_games_df["home_team_id"] == team_id) | (all_games_df["away_team_id"] == team_id)
    team_games_df = all_games_df[mask].sort_values("game_date")

    games = team_games_df.to_dict("records")
    if len(games) < 5:
        progress(f"[ERROR] Only {len(games)} games found -- need at least 5")
        return AutotuneResult(team_id=team_id, games_used=len(games))

    progress(f"Found {len(games)} games for team {team_id}")

    # Precompute features
    features = _precompute_game_features(games, all_games_df, progress)
    if len(features) < 3:
        progress("[ERROR] Not enough games with sufficient data")
        return AutotuneResult(team_id=team_id, games_used=len(features))

    return _run_optimization(team_id, features, progress, max_iter)


def autotune_global(
    progress_cb: Optional[Callable[[str], None]] = None,
    max_iter: int = 500,
) -> AutotuneResult:
    """
    Optimise prediction weights across *all* games in the database.

    Returns an ``AutotuneResult`` (team_id=None for global).
    """
    progress = progress_cb or (lambda _: None)
    progress("Starting global autotune across all teams...")

    all_games_df = get_actual_game_results()
    if all_games_df.empty:
        progress("[ERROR] No game data found")
        return AutotuneResult(team_id=None)

    all_games_df = all_games_df.sort_values("game_date")
    games = all_games_df.to_dict("records")
    progress(f"Found {len(games)} total games")

    features = _precompute_game_features(games, all_games_df, progress)
    if len(features) < 5:
        progress("[ERROR] Not enough games with sufficient data")
        return AutotuneResult(team_id=None, games_used=len(features))

    return _run_optimization(None, features, progress, max_iter)


def _run_optimization(
    team_id: Optional[int],
    features: List[GameFeatures],
    progress: Callable[[str], None],
    max_iter: int,
) -> AutotuneResult:
    """Shared optimiser logic for per-team and global tuning."""

    # Starting point: current effective weights (may already include DB overrides)
    base_cfg = dict(PREDICTION_CONFIG)
    base_cfg.update(load_team_weights(team_id))

    x0 = np.array([base_cfg[k] for k in TUNABLE_KEYS], dtype=float)

    # Before errors
    spread_err_before, total_err_before = _compute_errors(x0, features)
    progress(
        f"Before tuning: avg spread error = {spread_err_before:.2f}, "
        f"avg total error = {total_err_before:.2f}"
    )

    # Track iteration progress
    iter_count = [0]

    def callback(xk):
        iter_count[0] += 1
        if iter_count[0] % 50 == 0:
            err = _objective(xk, features)
            progress(f"  Iteration {iter_count[0]}: combined error = {err:.3f}")

    progress(f"Running Nelder-Mead optimiser (max {max_iter} iterations)...")
    result = minimize(
        _objective,
        x0,
        args=(features,),
        method="Nelder-Mead",
        callback=callback,
        options={"maxiter": max_iter, "xatol": 0.005, "fatol": 0.05, "adaptive": True},
    )

    # Clamp to bounds
    best = _clamp_weights(result.x)

    spread_err_after, total_err_after = _compute_errors(best, features)
    progress(
        f"After tuning: avg spread error = {spread_err_after:.2f}, "
        f"avg total error = {total_err_after:.2f}"
    )

    weights_before = {k: float(x0[i]) for i, k in enumerate(TUNABLE_KEYS)}
    weights_after = {k: float(best[i]) for i, k in enumerate(TUNABLE_KEYS)}

    # Log per-weight changes
    for k in TUNABLE_KEYS:
        diff = weights_after[k] - weights_before[k]
        if abs(diff) > 0.001:
            progress(f"  {k}: {weights_before[k]:.4f} -> {weights_after[k]:.4f} ({diff:+.4f})")

    progress(
        f"[DONE] Optimisation finished in {result.nit} iterations, "
        f"{len(features)} games"
    )

    return AutotuneResult(
        team_id=team_id,
        weights_before=weights_before,
        weights_after=weights_after,
        spread_error_before=spread_err_before,
        spread_error_after=spread_err_after,
        total_error_before=total_err_before,
        total_error_after=total_err_after,
        games_used=len(features),
        iterations=result.nit,
    )
