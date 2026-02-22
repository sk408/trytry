"""VectorizedGames, Optuna TPE optimizer, per-team refinement, residual calibration."""

import logging
import random
from dataclasses import fields
from typing import List, Dict, Any, Optional, Callable

import numpy as np

from src.database import db
from src.analytics.weight_config import (
    WeightConfig, get_weight_config, save_weight_config,
    save_team_weights, OPTIMIZER_RANGES, invalidate_weight_cache,
)
from src.analytics.prediction import PrecomputedGame, predict_from_precomputed

logger = logging.getLogger(__name__)


class VectorizedGames:
    """Converts List[PrecomputedGame] into flat NumPy arrays for fast loss eval."""

    def __init__(self, games: List[PrecomputedGame]):
        n = len(games)
        self.n = n
        self.home_team_ids = np.array([g.home_team_id for g in games])
        self.away_team_ids = np.array([g.away_team_id for g in games])
        self.actual_spread = np.array([g.actual_home_score - g.actual_away_score for g in games])
        self.actual_total = np.array([g.actual_home_score + g.actual_away_score for g in games])

        self.home_pts_raw = np.array([g.home_proj.get("points", 0) for g in games])
        self.away_pts_raw = np.array([g.away_proj.get("points", 0) for g in games])
        self.home_def_factor_raw = np.array([g.home_def_factor_raw for g in games])
        self.away_def_factor_raw = np.array([g.away_def_factor_raw for g in games])
        self.home_tuning_corr = np.array([g.home_tuning_home_corr for g in games])
        self.away_tuning_corr = np.array([g.away_tuning_away_corr for g in games])
        self.home_fatigue = np.array([g.home_fatigue_penalty for g in games])
        self.away_fatigue = np.array([g.away_fatigue_penalty for g in games])
        self.home_court = np.array([g.home_court for g in games])

        # TO/reb diffs
        self.to_diff = np.array([
            g.away_proj.get("turnovers", 0) - g.home_proj.get("turnovers", 0)
            for g in games
        ])
        self.reb_diff = np.array([
            g.home_proj.get("rebounds", 0) - g.away_proj.get("rebounds", 0)
            for g in games
        ])

        # Ratings
        self.home_off = np.array([g.home_off for g in games])
        self.away_off = np.array([g.away_off for g in games])
        self.home_def = np.array([g.home_def for g in games])
        self.away_def = np.array([g.away_def for g in games])

        # Four Factors edges — compute from raw team values (home_ff has efg/tov/oreb/fta)
        self.ff_efg_edge = np.array([
            g.home_ff.get("efg", 0) - g.away_ff.get("efg", 0) for g in games])
        self.ff_tov_edge = np.array([
            g.away_ff.get("tov", 0) - g.home_ff.get("tov", 0) for g in games])  # positive = home turns over less
        self.ff_oreb_edge = np.array([
            g.home_ff.get("oreb", 0) - g.away_ff.get("oreb", 0) for g in games])
        self.ff_fta_edge = np.array([
            g.home_ff.get("fta", 0) - g.away_ff.get("fta", 0) for g in games])

        # Clutch
        self.clutch_diff = np.array([
            g.home_clutch.get("net_rating", 0) - g.away_clutch.get("net_rating", 0)
            for g in games
        ])

        # Hustle
        self.hustle_effort_diff = np.array([
            (g.home_hustle.get("deflections", 0) + g.home_hustle.get("contested", 0) * 0.3) -
            (g.away_hustle.get("deflections", 0) + g.away_hustle.get("contested", 0) * 0.3)
            for g in games
        ])

        # Totals
        self.combined_deflections = np.array([
            g.home_hustle.get("deflections", 0) + g.away_hustle.get("deflections", 0)
            for g in games
        ])
        self.avg_pace = np.array([(g.home_pace + g.away_pace) / 2.0 for g in games])
        self.combined_steals = np.array([
            g.home_proj.get("steals", 0) + g.away_proj.get("steals", 0)
            for g in games
        ])
        self.combined_blocks = np.array([
            g.home_proj.get("blocks", 0) + g.away_proj.get("blocks", 0)
            for g in games
        ])
        self.combined_oreb = np.array([
            g.home_proj.get("oreb", 0) + g.away_proj.get("oreb", 0)
            for g in games
        ])
        self.combined_fatigue = self.home_fatigue + self.away_fatigue

    def evaluate(self, w: WeightConfig) -> Dict[str, float]:
        """Vectorized loss evaluation. Returns spread_mae, total_mae, winner_pct, loss."""
        # Defensive factor
        away_def_f = 1.0 + (self.away_def_factor_raw - 1.0) * w.def_factor_dampening
        home_def_f = 1.0 + (self.home_def_factor_raw - 1.0) * w.def_factor_dampening

        home_base = self.home_pts_raw * away_def_f + np.clip(self.home_tuning_corr, -4.0, 4.0)
        away_base = self.away_pts_raw * home_def_f + np.clip(self.away_tuning_corr, -4.0, 4.0)

        # Spread
        spread = (home_base - away_base) + self.home_court
        spread -= (self.home_fatigue - self.away_fatigue)
        spread += self.to_diff * w.turnover_margin_mult
        spread += self.reb_diff * w.rebound_diff_mult

        # Rating matchup
        home_me = self.home_off - self.away_def
        away_me = self.away_off - self.home_def
        spread += (home_me - away_me) * w.rating_matchup_mult

        # Four Factors
        ff = (self.ff_efg_edge * w.ff_efg_weight + self.ff_tov_edge * w.ff_tov_weight +
              self.ff_oreb_edge * w.ff_oreb_weight + self.ff_fta_edge * w.ff_fta_weight) * w.four_factors_scale
        spread += ff

        # Clutch (vectorized: apply only when |spread| < threshold)
        clutch_mask = np.abs(spread) < w.clutch_threshold
        clutch_adj = np.clip(self.clutch_diff * w.clutch_scale, -w.clutch_cap, w.clutch_cap)
        spread += clutch_adj * clutch_mask

        # Hustle
        spread += self.hustle_effort_diff * w.hustle_effort_mult

        # Total
        total = home_base + away_base
        pace_factor = (self.avg_pace - w.pace_baseline) / w.pace_baseline
        total *= (1.0 + pace_factor * w.pace_mult)
        total -= (np.maximum(0, self.combined_steals - w.steals_threshold) * w.steals_penalty +
                  np.maximum(0, self.combined_blocks - w.blocks_threshold) * w.blocks_penalty)
        total += (self.combined_oreb - w.oreb_baseline) * w.oreb_mult
        defl_excess = np.maximum(0, self.combined_deflections - w.hustle_defl_baseline)
        total -= defl_excess * w.hustle_defl_penalty
        total -= self.combined_fatigue * w.fatigue_total_mult

        # Clamps
        spread = np.clip(spread, -w.spread_clamp, w.spread_clamp)
        total = np.clip(total, w.total_min, w.total_max)

        # Metrics
        spread_errors = np.abs(spread - self.actual_spread)
        total_errors = np.abs(total - self.actual_total)
        spread_mae = float(np.mean(spread_errors))
        total_mae = float(np.mean(total_errors))

        # Winner %
        pred_home_win = spread > 0.5
        pred_away_win = spread < -0.5
        actual_home_win = self.actual_spread > 0.5
        actual_away_win = self.actual_spread < -0.5
        push = np.abs(self.actual_spread) <= 3.0

        correct = ((pred_home_win & actual_home_win) |
                    (pred_away_win & actual_away_win) |
                    (push & ~pred_home_win & ~pred_away_win))
        winner_pct = float(np.mean(correct)) * 100.0

        # Composite loss
        loss = spread_mae * 1.0 + total_mae * 0.3 + (100.0 - winner_pct) * 0.1

        return {
            "spread_mae": spread_mae,
            "total_mae": total_mae,
            "winner_pct": winner_pct,
            "loss": loss,
        }


def optimize_weights(games: List[PrecomputedGame], n_trials: int = 200,
                     callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Run Optuna TPE optimization on global weights."""
    vg = VectorizedGames(games)

    # Baseline
    baseline_w = get_weight_config()
    baseline_result = vg.evaluate(baseline_w)
    baseline_loss = baseline_result["loss"]

    if callback:
        callback(f"Baseline: MAE={baseline_result['spread_mae']:.2f}, "
                 f"Winner={baseline_result['winner_pct']:.1f}%, Loss={baseline_loss:.3f}")

    best_w = baseline_w
    best_loss = baseline_loss

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {}
            for key, (lo, hi) in OPTIMIZER_RANGES.items():
                params[key] = trial.suggest_float(key, lo, hi)
            # Constraint
            params["espn_weight"] = 1.0 - params.get("espn_model_weight", 0.8)

            w = WeightConfig.from_dict({**baseline_w.to_dict(), **params})
            result = vg.evaluate(w)
            return result["loss"]

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def trial_callback(study, trial):
            if trial.number % 25 == 0 or trial.value < best_loss:
                if callback:
                    callback(f"Trial {trial.number}/{n_trials}: loss={trial.value:.3f}")

        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

        if study.best_value < baseline_loss:
            best_params = study.best_params
            best_params["espn_weight"] = 1.0 - best_params.get("espn_model_weight", 0.8)
            best_w = WeightConfig.from_dict({**baseline_w.to_dict(), **best_params})
            best_loss = study.best_value

    except ImportError:
        if callback:
            callback("Optuna not installed, using random search...")
        # Fallback random sampling
        for i in range(n_trials):
            params = {}
            for key, (lo, hi) in OPTIMIZER_RANGES.items():
                params[key] = random.uniform(lo, hi)
            params["espn_weight"] = 1.0 - params.get("espn_model_weight", 0.8)
            w = WeightConfig.from_dict({**baseline_w.to_dict(), **params})
            result = vg.evaluate(w)
            if result["loss"] < best_loss:
                best_w = w
                best_loss = result["loss"]
            if callback and (i + 1) % 25 == 0:
                callback(f"Random trial {i + 1}/{n_trials}: best_loss={best_loss:.3f}")

    # Save if improved
    if best_loss < baseline_loss:
        save_weight_config(best_w)
        invalidate_weight_cache()
        if callback:
            callback(f"Saved optimized weights (loss {baseline_loss:.3f} → {best_loss:.3f})")
    else:
        if callback:
            callback("No improvement found, keeping current weights")

    final = vg.evaluate(best_w)
    return {
        "baseline_loss": baseline_loss,
        "best_loss": best_loss,
        "improved": best_loss < baseline_loss,
        **final,
    }


def per_team_refinement(games: List[PrecomputedGame], n_trials: int = 100,
                        callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Per-team weight refinement (100 random trials per team)."""
    global_w = get_weight_config()
    vg_all = VectorizedGames(games)
    global_result = vg_all.evaluate(global_w)
    global_loss = global_result["loss"]

    # Get unique team IDs
    all_teams = set()
    for g in games:
        all_teams.add(g.home_team_id)
        all_teams.add(g.away_team_id)

    refined = 0
    results = {}

    for team_id in sorted(all_teams):
        # Filter games involving this team
        team_games = [g for g in games
                      if g.home_team_id == team_id or g.away_team_id == team_id]
        if len(team_games) < 10:
            continue

        # Sort by date, split train/holdout
        team_games.sort(key=lambda g: g.game_date)
        holdout_n = min(5, len(team_games) // 4)
        train_games = team_games[:-holdout_n] if holdout_n > 0 else team_games
        holdout_games = team_games[-holdout_n:] if holdout_n > 0 else []

        if not holdout_games:
            continue

        vg_holdout = VectorizedGames(holdout_games)
        global_holdout_loss = vg_holdout.evaluate(global_w)["loss"]

        best_team_w = global_w
        best_team_loss = global_holdout_loss

        tunable_keys = ["def_factor_dampening", "turnover_margin_mult",
                        "four_factors_scale", "pace_mult"]

        for _ in range(n_trials):
            params = global_w.to_dict()
            for key in tunable_keys:
                lo, hi = OPTIMIZER_RANGES.get(key, (params[key] * 0.8, params[key] * 1.2))
                # Perturb by ±20%
                delta = params[key] * 0.2
                val = params[key] + random.uniform(-delta, delta)
                params[key] = max(lo, min(hi, val))

            w = WeightConfig.from_dict(params)
            result = vg_holdout.evaluate(w)

            if result["loss"] < best_team_loss:
                best_team_w = w
                best_team_loss = result["loss"]

        # Keep only if better and not >30% worse than global
        if best_team_loss < global_holdout_loss and best_team_loss < global_loss * 1.3:
            save_team_weights(team_id, best_team_w)
            refined += 1
            results[team_id] = {"loss_before": global_holdout_loss, "loss_after": best_team_loss}

        if callback:
            callback(f"Team {team_id}: {'refined' if team_id in results else 'kept global'}")

    if callback:
        callback(f"Per-team refinement complete: {refined}/{len(all_teams)} teams refined")

    return {"teams_refined": refined, "total_teams": len(all_teams), "details": results}


def build_residual_calibration(games: List[PrecomputedGame],
                                callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Build residual calibration bins from prediction errors."""
    w = get_weight_config()

    # Spread bins
    spread_bins = [
        ("big_away", -30, -18), ("med_away", -18, -12), ("small_away", -12, -8),
        ("slight_away", -8, -4), ("toss_up", -4, 4), ("slight_home", 4, 8),
        ("small_home", 8, 12), ("med_home", 12, 18), ("big_home", 18, 30),
    ]
    total_bins = [
        ("very_low", 180, 200), ("low", 200, 210), ("below_avg", 210, 215),
        ("avg_low", 215, 220), ("avg_high", 220, 225), ("above_avg", 225, 230),
        ("high", 230, 240), ("very_high", 240, 270),
    ]

    spread_residuals = {label: [] for label, _, _ in spread_bins}
    total_residuals = {label: [] for label, _, _ in total_bins}

    for g in games:
        result = predict_from_precomputed(g, w, skip_residual=True)
        pred_spread = result["spread"]
        pred_total = result["total"]
        actual_spread = g.actual_home_score - g.actual_away_score
        actual_total = g.actual_home_score + g.actual_away_score

        # Spread bin
        for label, lo, hi in spread_bins:
            if lo <= pred_spread < hi:
                spread_residuals[label].append(pred_spread - actual_spread)
                break

        # Total bin
        for label, lo, hi in total_bins:
            if lo <= pred_total < hi:
                total_residuals[label].append(pred_total - actual_total)
                break

    # Create residual_calibration table
    db.execute("DROP TABLE IF EXISTS residual_calibration")
    db.execute("""
        CREATE TABLE residual_calibration (
            bin_label TEXT PRIMARY KEY,
            bin_low REAL, bin_high REAL,
            avg_residual REAL, sample_count INTEGER
        )
    """)

    for label, lo, hi in spread_bins:
        residuals = spread_residuals[label]
        avg = float(np.mean(residuals)) if len(residuals) >= 5 else 0.0
        db.execute(
            "INSERT INTO residual_calibration VALUES (?,?,?,?,?)",
            (label, lo, hi, avg, len(residuals))
        )

    # Total calibration
    db.execute("DROP TABLE IF EXISTS residual_calibration_total")
    db.execute("""
        CREATE TABLE residual_calibration_total (
            bin_label TEXT PRIMARY KEY,
            bin_low REAL, bin_high REAL,
            avg_residual REAL, sample_count INTEGER
        )
    """)

    for label, lo, hi in total_bins:
        residuals = total_residuals[label]
        avg = float(np.mean(residuals)) if len(residuals) >= 5 else 0.0
        db.execute(
            "INSERT INTO residual_calibration_total VALUES (?,?,?,?,?)",
            (label, lo, hi, avg, len(residuals))
        )

    if callback:
        callback("Residual calibration built")

    return {
        "spread_bins": {k: len(v) for k, v in spread_residuals.items()},
        "total_bins": {k: len(v) for k, v in total_residuals.items()},
    }


def compute_feature_importance(games: List[PrecomputedGame],
                                callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """Individual feature importance: disable each feature, measure loss delta."""
    w = get_weight_config()
    vg = VectorizedGames(games)
    baseline = vg.evaluate(w)["loss"]

    features = [
        ("def_factor_dampening", "def_factor_dampening", 0.0),
        ("turnover_margin_mult", "turnover_margin_mult", 0.0),
        ("rebound_diff_mult", "rebound_diff_mult", 0.0),
        ("rating_matchup_mult", "rating_matchup_mult", 0.0),
        ("four_factors_scale", "four_factors_scale", 0.0),
        ("clutch_scale", "clutch_scale", 0.0),
        ("hustle_effort_mult", "hustle_effort_mult", 0.0),
        ("pace_mult", "pace_mult", 0.0),
        ("fatigue_total_mult", "fatigue_total_mult", 0.0),
        ("espn_model_weight", "espn_model_weight", 1.0),
        ("ml_ensemble_weight", "ml_ensemble_weight", 0.0),
        ("steals_penalty", "steals_penalty", 0.0),
        ("blocks_penalty", "blocks_penalty", 0.0),
        ("oreb_mult", "oreb_mult", 0.0),
    ]

    results = []
    for name, attr, disabled_val in features:
        test_w = WeightConfig.from_dict(w.to_dict())
        setattr(test_w, attr, disabled_val)
        test_loss = vg.evaluate(test_w)["loss"]
        delta = test_loss - baseline
        results.append({"feature": name, "loss_delta": delta, "importance": abs(delta)})
        if callback:
            callback(f"{name}: delta={delta:+.3f}")

    results.sort(key=lambda x: x["importance"], reverse=True)
    return results
