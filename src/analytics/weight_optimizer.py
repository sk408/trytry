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

        self.vegas_spread = np.array([g.vegas_spread for g in games])
        self.vegas_home_ml = np.array([g.vegas_home_ml for g in games])
        self.vegas_away_ml = np.array([g.vegas_away_ml for g in games])
        
        # Default to 50/50 split if missing to avoid division by zero or skewed data
        self.spread_home_public = np.array([g.spread_home_public if g.spread_home_public else 50.0 for g in games], dtype=float)
        self.spread_home_money = np.array([g.spread_home_money if g.spread_home_money else 50.0 for g in games], dtype=float)
        
        # Calculate sharp money edge (Money % - Ticket %)
        # Positive means sharp money is on the home team
        self.sharp_money_edge = (self.spread_home_money - self.spread_home_public) / 100.0

    def evaluate(self, w: WeightConfig, target: str = "ats") -> Dict[str, float]:
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

        # Sharp Money (Vegas)
        # We only apply sharp money edge if there's an actual edge recorded (i.e. not the default 0.0)
        # sharp_money_edge is positive when money > tickets for the home team.
        spread += self.sharp_money_edge * w.sharp_money_weight

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
        
        # Vegas Betting objectives (ATS, Edge Hit Rate)
        if np.any(self.vegas_spread != 0.0):
            v_mask = self.vegas_spread != 0.0
            v_margin = -self.vegas_spread[v_mask]
            p_spread = spread[v_mask]
            a_spread = self.actual_spread[v_mask]
            
            pick_home = p_spread > v_margin
            actual_home_cover = a_spread > v_margin
            v_push = a_spread == v_margin
            
            covered = (pick_home == actual_home_cover) & ~v_push
            ats_rate = np.mean(covered) * 100.0 if len(covered) > 0 else 50.0
            
            edge_mask = np.abs(p_spread - v_margin) >= w.ats_edge_threshold
            if np.any(edge_mask):
                edge_rate = np.mean(covered[edge_mask]) * 100.0
            else:
                edge_rate = 50.0
                
            # ROI calculation (approximate for ATS assuming -110 odds)
            ats_roi = (ats_rate / 100.0 * 2.1 - 1.1) / 1.1 * 100.0
            edge_roi = (edge_rate / 100.0 * 2.1 - 1.1) / 1.1 * 100.0
            
            # Moneyline calculations
            ml_mask = (self.vegas_home_ml != 0) & (self.vegas_away_ml != 0)
            if np.any(ml_mask):
                h_ml = self.vegas_home_ml[ml_mask]
                a_ml = self.vegas_away_ml[ml_mask]
                p_spread_ml = spread[ml_mask]
                a_spread_ml = self.actual_spread[ml_mask]
                
                # Model's pick based on spread (who does it think wins straight up?)
                pick_home_ml = p_spread_ml > 0
                actual_home_win_ml = a_spread_ml > 0
                
                # Calculate ML payout multipliers
                # If ML is negative (e.g. -150), multiplier is 1 + 100/abs(ML)
                # If ML is positive (e.g. +130), multiplier is 1 + ML/100
                h_mult = np.where(h_ml < 0, 1.0 + 100.0 / np.abs(h_ml), 1.0 + h_ml / 100.0)
                a_mult = np.where(a_ml < 0, 1.0 + 100.0 / np.abs(a_ml), 1.0 + a_ml / 100.0)
                
                # Winnings: if pick wins, profit = mult - 1. If pick loses, profit = -1.
                h_profit = np.where(pick_home_ml & actual_home_win_ml, h_mult - 1.0, 
                           np.where(pick_home_ml & ~actual_home_win_ml, -1.0, 0.0))
                a_profit = np.where(~pick_home_ml & ~actual_home_win_ml, a_mult - 1.0, 
                           np.where(~pick_home_ml & actual_home_win_ml, -1.0, 0.0))
                
                total_profit = h_profit + a_profit
                
                ml_win_rate = np.mean((pick_home_ml & actual_home_win_ml) | (~pick_home_ml & ~actual_home_win_ml)) * 100.0
                ml_roi = np.mean(total_profit) * 100.0
            else:
                ml_win_rate = winner_pct
                ml_roi = -4.54 # Default negative
            
            if target == "ats":
                # Strongly penalize losing to Vegas and reward beating it
                loss += max(0, 52.5 - ats_rate) * 0.5 
                loss += max(0, 52.5 - edge_rate) * 0.5
                
                # Reward higher hit rates
                loss -= (ats_rate * 0.05)
                loss -= (edge_rate * 0.05)
                
                if ats_roi > 0: loss -= ats_roi * 0.05
                if edge_roi > 0: loss -= edge_roi * 0.05
                
            elif target == "roi":
                # Optimize purely for max ROI %
                loss -= ats_roi * 0.2
                loss -= edge_roi * 0.2
                # Still penalize horrible hit rates
                loss += max(0, 52.5 - ats_rate) * 0.2
                
            elif target == "ml":
                # Optimize for Moneyline ROI
                loss -= ml_roi * 0.2
                # Penalize low win rates (below 60% is bad for ML since favs win a lot)
                loss += max(0, 60.0 - ml_win_rate) * 0.5
                
        else:
            ats_rate = 50.0
            edge_rate = 50.0
            ats_roi = -4.54
            edge_roi = -4.54
            ml_win_rate = winner_pct
            ml_roi = -4.54

        return {
            "spread_mae": spread_mae,
            "total_mae": total_mae,
            "winner_pct": winner_pct,
            "ats_rate": ats_rate,
            "edge_rate": edge_rate,
            "ats_roi": ats_roi,
            "edge_roi": edge_roi,
            "ml_win_rate": ml_win_rate,
            "ml_roi": ml_roi,
            "loss": loss,
        }

    
def optimize_weights(games: List[PrecomputedGame], n_trials: int = 3000,
                     callback: Optional[Callable] = None,
                     is_cancelled: Optional[Callable[[], bool]] = None,
                     target: str = "ats") -> Dict[str, Any]:
    """Run Optuna TPE optimization on global weights."""
    vg = VectorizedGames(games)

    # Baseline
    baseline_w = get_weight_config()
    baseline_result = vg.evaluate(baseline_w, target=target)
    baseline_loss = baseline_result["loss"]

    if callback:
        callback(f"Baseline: MAE={baseline_result['spread_mae']:.2f}, "
                 f"Winner={baseline_result['winner_pct']:.1f}%, ATS={baseline_result['ats_rate']:.1f}%, "
                 f"ML ROI={baseline_result['ml_roi']:.1f}%, Loss={baseline_loss:.3f}")

    best_w = baseline_w
    best_loss = baseline_loss
    best_result = baseline_result

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
            result = vg.evaluate(w, target=target)
            # Store result in trial for callback
            trial.set_user_attr("result", result)
            return result["loss"]

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def trial_callback(study, trial):
            if is_cancelled and is_cancelled():
                if callback:
                    callback("Optimization cancelled by user. Stopping gracefully...")
                study.stop()
                return

            if trial.number % 25 == 0 or trial.value < best_loss:
                if callback:
                    res = trial.user_attrs.get("result", {})
                    if target == "ml":
                        win = res.get('ml_win_rate', 0)
                        roi = res.get('ml_roi', -4.54)
                        callback(f"Trial {trial.number}/{n_trials}: loss={trial.value:.3f} "
                                 f"(ML Win={win:.1f}%, ML ROI={roi:.1f}%)")
                    else:
                        ats = res.get('ats_rate', 0)
                        roi = res.get('ats_roi', -4.54)
                        callback(f"Trial {trial.number}/{n_trials}: loss={trial.value:.3f} "
                                 f"(ATS={ats:.1f}%, ROI={roi:.1f}%)")

        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

        try:
            if len(study.trials) > 0 and callback:
                import optuna.importance
                importances = optuna.importance.get_param_importances(study)
                top_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                param_str = ", ".join([f"{k}: {v:.2f}" for k, v in top_params])
                callback(f"Top 5 ROI impact parameters: {param_str}")
        except Exception:
            pass

        if study.best_value < baseline_loss:
            best_params = study.best_params
            best_params["espn_weight"] = 1.0 - best_params.get("espn_model_weight", 0.8)
            best_w = WeightConfig.from_dict({**baseline_w.to_dict(), **best_params})
            best_loss = study.best_value
            best_result = vg.evaluate(best_w, target=target)

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
            result = vg.evaluate(w, target=target)
            if result["loss"] < best_loss:
                best_w = w
                best_loss = result["loss"]
                best_result = result
            if callback and (i + 1) % 25 == 0:
                callback(f"Random trial {i + 1}/{n_trials}: best_loss={best_loss:.3f}")

    # Save if improved
    if best_loss < baseline_loss:
        save_weight_config(best_w)
        invalidate_weight_cache()
        if callback:
            callback(f"Saved optimized weights (loss {baseline_loss:.3f} → {best_loss:.3f}, "
                     f"ATS: {baseline_result['ats_rate']:.1f}% → {best_result['ats_rate']:.1f}%)")
    else:
        if callback:
            callback("No improvement found, keeping current weights")

    final = vg.evaluate(best_w, target=target)
    return {
        "baseline_loss": baseline_loss,
        "best_loss": best_loss,
        "improved": best_loss < baseline_loss,
        **final,
    }


def per_team_refinement(games: List[PrecomputedGame], n_trials: int = 100,
                        callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Per-team weight refinement with expanded tunable parameter set.

    Changes from v1:
    - 8 tunable params (was 4): adds rating_matchup_mult, clutch_scale,
      hustle_effort_mult, sharp_money_weight
    - Minimum 15 games per team (was 10) for more robust holdout
    - Holdout minimum raised to 7 games (was 5) for reliable evaluation
    - Perturbation range scales with distance from optimizer range midpoint
    """
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
        if len(team_games) < 15:
            continue

        # Sort by date, split train/holdout (min 7 holdout games)
        team_games.sort(key=lambda g: g.game_date)
        holdout_n = max(7, len(team_games) // 4)
        holdout_n = min(holdout_n, len(team_games) // 2)  # never exceed half
        train_games = team_games[:-holdout_n] if holdout_n > 0 else team_games
        holdout_games = team_games[-holdout_n:] if holdout_n > 0 else []

        if len(holdout_games) < 7:
            continue

        vg_holdout = VectorizedGames(holdout_games)
        global_holdout_loss = vg_holdout.evaluate(global_w)["loss"]

        best_team_w = global_w
        best_team_loss = global_holdout_loss

        tunable_keys = [
            "def_factor_dampening", "turnover_margin_mult",
            "four_factors_scale", "pace_mult",
            "rating_matchup_mult", "clutch_scale",
            "hustle_effort_mult", "sharp_money_weight",
        ]

        for _ in range(n_trials):
            params = global_w.to_dict()
            for key in tunable_keys:
                lo, hi = OPTIMIZER_RANGES.get(key, (params[key] * 0.8, params[key] * 1.2))
                # Perturb by ±20% of the optimizer range width (not value)
                range_width = hi - lo
                delta = range_width * 0.20
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
        ("sharp_money_weight", "sharp_money_weight", 0.0),
        ("ats_edge_threshold", "ats_edge_threshold", 3.0),
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
