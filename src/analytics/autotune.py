"""Per-team scoring corrections via grid search."""

import logging
import numpy as np
from typing import Dict, Any, Optional, Callable, List

from src.database import db
from src.analytics.weight_config import get_weight_config
from src.analytics.stats_engine import (
    aggregate_projection, player_splits, get_home_court_advantage, compute_fatigue,
)
from src.analytics.backtester import get_actual_game_results

logger = logging.getLogger(__name__)

# Parameters
DEFAULT_STRENGTH = 0.75
DEFAULT_MIN_THRESHOLD = 1.5
DEFAULT_MAX_ABS_CORRECTION = 6.0    # compromise: wider than original 4 but prevents wild swings
GRID_STEP = 0.25
GRID_RANGE = 8.0                    # search range; corrections clipped to MAX_ABS_CORRECTION


def _get_team_players(team_id: int, as_of_date: str = None) -> List[Dict]:
    """Get players for a team, optionally as of a date."""
    query = "SELECT * FROM players WHERE team_id = ?"
    rows = db.fetch_all(query, (team_id,))
    return [dict(r) for r in rows]


def _compute_composite_score(errors: List[float],
                             weights: Optional[List[float]] = None) -> Dict[str, float]:
    """Compute composite score from prediction errors, optionally weighted.

    Args:
        weights: Per-error recency weights (higher = more recent/important).
                 If None, all errors weighted equally (backward-compatible).
    """
    if not errors:
        return {"wrong_rate": 1.0, "mae": 99.0, "p90": 99.0, "composite": 999.0}

    abs_errors = np.array([abs(e) for e in errors])

    if weights is not None and len(weights) == len(errors):
        w = np.array(weights)
        w = w / w.sum()  # normalize to sum to 1
        wrong = sum(w[i] for i, e in enumerate(errors) if abs(e) > 0.5 and np.sign(e) != 0)
        wrong_rate = float(wrong)
        mae = float(np.dot(abs_errors, w))
    else:
        wrong = sum(1 for e in errors if abs(e) > 0.5 and np.sign(e) != 0)
        wrong_rate = wrong / len(errors) if errors else 1.0
        mae = float(np.mean(abs_errors))

    p90 = float(np.percentile(abs_errors, 90)) if len(abs_errors) >= 3 else mae * 1.5
    composite = 3.0 * wrong_rate + 1.0 * mae + 0.25 * p90

    return {"wrong_rate": wrong_rate, "mae": mae, "p90": p90, "composite": composite}


def autotune_team(team_id: int, games: List[Dict],
                  strength: float = DEFAULT_STRENGTH,
                  mode: str = "classic",
                  max_abs_correction: float = DEFAULT_MAX_ABS_CORRECTION,
                  callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Run autotune for a single team."""
    # Filter games for this team
    team_games = [g for g in games
                  if g["home_team_id"] == team_id or g["away_team_id"] == team_id]

    if mode == "walk_forward":
        team_games = team_games[-20:]

    total_found = len(team_games)

    # Skip first 5 games
    if len(team_games) <= 5:
        return {"team_id": team_id, "skipped": True, "reason": "insufficient_games",
                "games_found": total_found}

    team_games = team_games[5:]

    w = get_weight_config()

    # Compute errors for each game with recency decay.
    # Most recent game gets weight 1.0; older games decay by 0.92x each step.
    _AUTOTUNE_DECAY = 0.92
    n_games = len(team_games)
    decay_weights = [_AUTOTUNE_DECAY ** (n_games - 1 - i) for i in range(n_games)]

    home_errors = []  # when team is home: predicted - actual spread (from team's perspective)
    away_errors = []
    home_weights = []
    away_weights = []

    for gi, game in enumerate(team_games):
        try:
            home_tid = game["home_team_id"]
            away_tid = game["away_team_id"]
            is_home = (home_tid == team_id)

            # Compute projection using aggregate_projection (handles players internally)
            opp_tid = away_tid if is_home else home_tid
            proj = aggregate_projection(
                team_id, opp_tid,
                is_home=1 if is_home else 0,
                as_of_date=game["game_date"]
            )
            proj_pts = proj.get("points", 0)

            # Defensive factor
            opp_metrics = db.fetch_one(
                "SELECT def_rating FROM team_metrics WHERE team_id = ?",
                (away_tid if is_home else home_tid,)
            )
            if opp_metrics and opp_metrics["def_rating"]:
                def_factor = opp_metrics["def_rating"] / 110.0
                def_factor = 1.0 + (def_factor - 1.0) * w.def_factor_dampening
                proj_pts *= def_factor

            actual_score = game["home_score"] if is_home else game["away_score"]
            error = proj_pts - actual_score

            if is_home:
                home_errors.append(error)
                home_weights.append(decay_weights[gi])
            else:
                away_errors.append(error)
                away_weights.append(decay_weights[gi])

        except Exception as e:
            logger.debug(f"Error processing game {game.get('game_id')}: {e}")
            continue

    # Pre-correction baseline (weighted by recency)
    pre_home = _compute_composite_score(home_errors, home_weights)
    pre_away = _compute_composite_score(away_errors, away_weights)

    # Grid search for best home correction (weighted)
    best_home_shift = 0.0
    best_home_score = pre_home["composite"]

    for shift in np.arange(-GRID_RANGE, GRID_RANGE + GRID_STEP, GRID_STEP):
        corrected = [e - shift for e in home_errors]
        score = _compute_composite_score(corrected, home_weights)
        if score["composite"] < best_home_score:
            best_home_score = score["composite"]
            best_home_shift = shift

    # Grid search for best away correction (weighted)
    best_away_shift = 0.0
    best_away_score = pre_away["composite"]

    for shift in np.arange(-GRID_RANGE, GRID_RANGE + GRID_STEP, GRID_STEP):
        corrected = [e - shift for e in away_errors]
        score = _compute_composite_score(corrected, away_weights)
        if score["composite"] < best_away_score:
            best_away_score = score["composite"]
            best_away_shift = shift

    # Gradual confidence ramp: conservative early, full confidence only with solid sample
    n = len(team_games)
    if n < 10:
        confidence = n / 10.0 * 0.5        # 0–50% at 0–10 games
    elif n < 25:
        confidence = 0.5 + (n - 10) / 15.0 * 0.5  # 50–100% at 10–25 games
    else:
        confidence = 1.0
    home_shift = best_home_shift * strength * confidence
    away_shift = best_away_shift * strength * confidence

    # Min threshold
    home_correction = np.clip(home_shift, -max_abs_correction, max_abs_correction)
    away_correction = np.clip(-away_shift, -max_abs_correction, max_abs_correction)

    if abs(home_correction) < DEFAULT_MIN_THRESHOLD:
        home_correction = 0.0
    if abs(away_correction) < DEFAULT_MIN_THRESHOLD:
        away_correction = 0.0

    # Guardrail: reject if post-correction >= pre-correction
    post_home = _compute_composite_score([e - home_correction for e in home_errors])
    post_away = _compute_composite_score([e + away_correction for e in away_errors])

    if post_home["composite"] >= pre_home["composite"]:
        home_correction = 0.0
    if post_away["composite"] >= pre_away["composite"]:
        away_correction = 0.0

    # Save to DB
    db.execute("""
        INSERT INTO team_tuning (team_id, home_pts_correction, away_pts_correction, 
                                  games_analyzed, last_tuned_at, tuning_mode)
        VALUES (?, ?, ?, ?, datetime('now'), ?)
        ON CONFLICT(team_id) DO UPDATE SET
            home_pts_correction = excluded.home_pts_correction,
            away_pts_correction = excluded.away_pts_correction,
            games_analyzed = excluded.games_analyzed,
            last_tuned_at = excluded.last_tuned_at,
            tuning_mode = excluded.tuning_mode
    """, (team_id, round(home_correction, 2), round(away_correction, 2), n, mode))

    return {
        "team_id": team_id,
        "home_correction": round(home_correction, 2),
        "away_correction": round(away_correction, 2),
        "games_analyzed": n,
        "games_found": total_found,
        "pre_home_composite": round(pre_home["composite"], 3),
        "post_home_composite": round(post_home["composite"], 3),
        "pre_away_composite": round(pre_away["composite"], 3),
        "post_away_composite": round(post_away["composite"], 3),
        "mode": mode,
    }


def autotune_all(strength: float = DEFAULT_STRENGTH,
                 mode: str = "classic",
                 max_abs_correction: float = DEFAULT_MAX_ABS_CORRECTION,
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Run autotune for all teams."""
    games = get_actual_game_results()
    if not games:
        return {"error": "No games to autotune", "teams_tuned": 0}

    # Get all teams
    teams = db.fetch_all("SELECT team_id, abbreviation FROM teams")
    if not teams:
        return {"error": "No teams found", "teams_tuned": 0}

    if callback:
        callback(f"Autotuning {len(teams)} teams ({mode} mode, strength={strength})...")

    results = {}
    tuned_count = 0

    for i, team in enumerate(teams):
        tid = team["team_id"]
        abbr = team["abbreviation"]
        try:
            result = autotune_team(tid, games, strength=strength, mode=mode,
                                    max_abs_correction=max_abs_correction)
            results[abbr] = result
            if not result.get("skipped"):
                tuned_count += 1
        except Exception as e:
            logger.warning(f"Autotune failed for {abbr}: {e}")
            results[abbr] = {"error": str(e)}

        if callback and (i + 1) % 5 == 0:
            callback(f"Progress: {i + 1}/{len(teams)} teams")

    # ── Recenter corrections to remove systematic home/away bias ──
    # The base model systematically under-predicts home and over-predicts away.
    # Without recentering, autotune compensates with asymmetric corrections
    # that compound to massive home spread bias (+4-5 pts average).
    home_corrs = []
    away_corrs = []
    tuned_tids = []
    for team in teams:
        abbr = team["abbreviation"]
        r = results.get(abbr)
        if r and not r.get("skipped") and not r.get("error"):
            home_corrs.append(r["home_correction"])
            away_corrs.append(r["away_correction"])
            tuned_tids.append(team["team_id"])

    if home_corrs and away_corrs:
        mean_home = float(np.mean(home_corrs))
        mean_away = float(np.mean(away_corrs))
        if callback:
            callback(f"Recentering: mean_home={mean_home:+.2f}, mean_away={mean_away:+.2f}")
        for idx, tid in enumerate(tuned_tids):
            new_h = np.clip(home_corrs[idx] - mean_home, -max_abs_correction, max_abs_correction)
            new_a = np.clip(away_corrs[idx] - mean_away, -max_abs_correction, max_abs_correction)
            db.execute(
                "UPDATE team_tuning SET home_pts_correction = ?, away_pts_correction = ? WHERE team_id = ?",
                (round(float(new_h), 2), round(float(new_a), 2), tid)
            )

    if callback:
        callback(f"Autotune complete: {tuned_count}/{len(teams)} teams tuned")

    return {
        "teams_tuned": tuned_count,
        "total_teams": len(teams),
        "mode": mode,
        "strength": strength,
        "details": results,
    }


def clear_all_tuning(callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Clear all team tuning corrections."""
    db.execute("DELETE FROM team_tuning")
    if callback:
        callback("All team tuning corrections cleared")
    return {"status": "cleared"}
