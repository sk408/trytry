"""Sensitivity Analysis — sweep individual weights through extreme ranges
to discover hidden value outside conventional optimizer bounds.

Usage (CLI):
    python -m src.analytics.sensitivity --param rating_matchup_mult --min -10 --max 50 --steps 200
    python -m src.analytics.sensitivity --all --steps 100

Produces CSV + ASCII chart per parameter showing spread_mae, winner_pct, and loss
across the full sweep range, like a dyno "power curve" for prediction tuning.
"""

import csv
import logging
import os
import sys
import time
from dataclasses import fields
from typing import Dict, List, Optional, Callable, Any

import numpy as np

from src.analytics.weight_config import (
    WeightConfig, get_weight_config, save_weight_config,
    OPTIMIZER_RANGES, save_snapshot, invalidate_weight_cache,
)
from src.analytics.weight_optimizer import VectorizedGames
from src.analytics.prediction import PrecomputedGame, precompute_game_data

logger = logging.getLogger(__name__)

_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sensitivity")

# Default extreme sweep ranges per parameter type
# These intentionally go WAY beyond optimizer bounds to find hidden value
EXTREME_RANGES = {
    # Signs locked to match basketball logic — no sign flips allowed.
    # spread_clamp is fixed at 30 — not tunable (prevents compression cheating).
    "def_factor_dampening":  (0.1,    5.0),
    "turnover_margin_mult":  (0.0,   10.0),
    "rebound_diff_mult":     (0.0,   10.0),
    "rating_matchup_mult":   (0.0,   20.0),
    "four_factors_scale":    (10.0, 1500.0),
    "clutch_scale":          (0.0,    5.0),
    "hustle_effort_mult":    (0.0,    5.0),
    "pace_mult":             (0.0,   10.0),
    "fatigue_total_mult":    (0.0,    5.0),
    "espn_model_weight":     (0.0,    1.0),
    "espn_weight":           (0.0,    1.0),
    "ml_ensemble_weight":    (0.0,    5.0),
    "ml_disagree_damp":      (0.0,    3.0),
    "ff_efg_weight":         (0.0,    5.0),
    "ff_tov_weight":         (0.0,    8.0),
    "ff_oreb_weight":        (0.0,    5.0),
    "ff_fta_weight":         (0.0,    5.0),
    "fatigue_b2b":           (0.0,   15.0),
    "fatigue_3in4":          (0.0,   10.0),
    "fatigue_4in6":          (0.0,   10.0),
    "pace_baseline":         (85.0, 110.0),
    "steals_penalty":        (0.0,    3.0),
    "blocks_penalty":        (0.0,    3.0),
    "oreb_mult":             (0.0,    5.0),
    "sharp_money_weight":    (0.0,   10.0),
    "ats_edge_threshold":    (0.5,    8.0),
}

# Parameters NOT used by VectorizedGames.evaluate()
# These are only used in the full predict_matchup() pipeline or during precomputation.
# Including them in vectorized sweeps produces meaningless noise.
VECTORIZED_EXCLUDED = {
    "ml_ensemble_weight",   # ML ensemble blend — only in predict_matchup()
    "ml_disagree_damp",     # ML disagreement damping — only in predict_matchup()
    "espn_model_weight",    # ESPN model blend — only in predict_matchup()
    "espn_weight",          # ESPN BPI weight — only in predict_matchup()
    "fatigue_b2b",          # Fatigue sub-penalty — baked into precomputed data
    "fatigue_3in4",         # Fatigue sub-penalty — baked into precomputed data
    "fatigue_4in6",         # Fatigue sub-penalty — baked into precomputed data
}

# Parameters that are sweepable (float fields on WeightConfig)
SWEEPABLE_PARAMS = [f.name for f in fields(WeightConfig) if f.type is float or f.type == "float"]


def _load_games(callback: Optional[Callable] = None) -> List:
    """Load and precompute all game data for evaluation."""
    if callback:
        callback("Loading and precomputing games (this may take a few minutes)...")
    games = precompute_game_data(callback=callback)
    if not games:
        raise ValueError("No precomputed games returned. Check database.")
    if callback:
        callback(f"Ready: {len(games)} games for evaluation")
    return games


def sweep_parameter(param_name: str, min_val: float, max_val: float,
                    steps: int = 200, games: Optional[List[PrecomputedGame]] = None,
                    vg: Optional[VectorizedGames] = None,
                    callback: Optional[Callable] = None,
                    target: str = "ml") -> List[Dict]:
    """Sweep a single parameter through [min_val, max_val] in `steps` increments.

    Args:
        target: Optimization target for loss calculation — "ml", "value", "ats", or "roi".

    Returns a list of dicts with columns:
        param_value, spread_mae, total_mae, winner_pct, ats_rate, edge_rate,
        ats_roi, edge_roi, ml_win_rate, ml_roi, loss
    """
    if param_name not in SWEEPABLE_PARAMS:
        raise ValueError(f"Unknown parameter: {param_name}. Available: {SWEEPABLE_PARAMS}")

    if vg is None:
        if games is None:
            games = _load_games(callback)
        vg = VectorizedGames(games)

    base_w = get_weight_config()
    values = np.linspace(min_val, max_val, steps)
    results = []

    for i, val in enumerate(values):
        w_dict = base_w.to_dict()
        w_dict[param_name] = float(val)
        w = WeightConfig.from_dict(w_dict)
        metrics = vg.evaluate(w, target=target)
        results.append({
            "param_value": round(float(val), 6),
            "spread_mae": round(metrics.get("spread_mae", 0), 4),
            "total_mae": round(metrics.get("total_mae", 0), 4),
            "winner_pct": round(metrics.get("winner_pct", 0), 2),
            "ats_rate": round(metrics.get("ats_rate", 0), 2),
            "edge_rate": round(metrics.get("edge_rate", 0), 2),
            "ats_roi": round(metrics.get("ats_roi", 0), 2),
            "edge_roi": round(metrics.get("edge_roi", 0), 2),
            "ml_win_rate": round(metrics.get("ml_win_rate", 0), 2),
            "ml_roi": round(metrics.get("ml_roi", 0), 2),
            "dog_pick_rate": round(metrics.get("dog_pick_rate", 0), 2),
            "dog_hit_rate": round(metrics.get("dog_hit_rate", 0), 2),
            "dog_roi": round(metrics.get("dog_roi", 0), 2),
            "loss": round(metrics.get("loss", 0), 4),
        })
        if callback and (i + 1) % 50 == 0:
            callback(f"  {param_name}: {i + 1}/{steps} ({val:.3f} -> ML ROI={metrics.get('ml_roi', 0):.2f}%, DogROI={metrics.get('dog_roi', 0):.1f}%)")

    return results


def sweep_all_parameters(steps: int = 100,
                         callback: Optional[Callable] = None,
                         target: str = "ml") -> Dict[str, List[Dict]]:
    """Sweep every sweepable parameter and return results keyed by param name.

    Args:
        target: Optimization target for loss calculation — "ml", "value", "ats", or "roi".
    """
    games = _load_games(callback)
    vg = VectorizedGames(games)

    all_results = {}
    params = sorted(EXTREME_RANGES.keys())

    for pi, param_name in enumerate(params):
        lo, hi = EXTREME_RANGES[param_name]
        if callback:
            callback(f"[{pi + 1}/{len(params)}] Sweeping {param_name} ({lo} to {hi})...")
        results = sweep_parameter(param_name, lo, hi, steps=steps, vg=vg,
                                  callback=callback, target=target)
        all_results[param_name] = results

    return all_results


def export_sweep_csv(param_name: str, results: List[Dict], output_dir: str = None) -> str:
    """Write sweep results to a CSV file. Returns the file path."""
    out = output_dir or _OUTPUT_DIR
    os.makedirs(out, exist_ok=True)

    filename = f"sweep_{param_name}.csv"
    path = os.path.join(out, filename)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "param_value", "spread_mae", "total_mae", "winner_pct",
            "ats_rate", "edge_rate", "ats_roi", "edge_roi",
            "ml_win_rate", "ml_roi",
            "dog_pick_rate", "dog_hit_rate", "dog_roi",
            "loss"
        ])
        writer.writeheader()
        writer.writerows(results)

    return path


def format_ascii_chart(param_name: str, results: List[Dict],
                       metric: str = "spread_mae", width: int = 80, height: int = 20) -> str:
    """Generate an ASCII chart of the sweep results for terminal display."""
    values = [r["param_value"] for r in results]
    metrics = [r[metric] for r in results]

    min_v, max_v = min(values), max(values)
    min_m, max_m = min(metrics), max(metrics)

    # Find optimal point
    if metric in ("spread_mae", "total_mae", "loss"):
        best_idx = np.argmin(metrics)
    else:
        best_idx = np.argmax(metrics)
    best_val = values[best_idx]
    best_metric = metrics[best_idx]

    # Build chart
    chart_width = width - 12  # left margin for labels
    range_m = max_m - min_m if max_m != min_m else 1.0

    lines = []
    lines.append(f"  {param_name} -> {metric}")
    lines.append(f"  Range: [{min_v:.3f}, {max_v:.3f}]  Best: {best_val:.4f} ({metric}={best_metric:.4f})")
    lines.append(f"  {'-' * chart_width}")

    # Downsample to height rows
    step = max(1, len(results) // height)
    sampled = results[::step][:height]

    for r in sampled:
        val = r["param_value"]
        m = r[metric]
        bar_len = int((m - min_m) / range_m * (chart_width - 2)) if range_m > 0 else 0
        bar_len = max(0, min(chart_width - 2, bar_len))
        marker = "#" * bar_len
        is_best = abs(val - best_val) < (max_v - min_v) / len(results) * 2
        prefix = "*" if is_best else " "
        lines.append(f"{prefix}{val:>8.3f} |{marker}")

    lines.append(f"  {'-' * chart_width}")
    lines.append(f"  {min_m:<10.3f}{' ' * (chart_width - 20)}{max_m:>10.3f}")

    # Current default value
    current = getattr(WeightConfig(), param_name)
    lines.append(f"  Current default: {current}")
    if param_name in OPTIMIZER_RANGES:
        lo, hi = OPTIMIZER_RANGES[param_name]
        lines.append(f"  Optimizer range: [{lo}, {hi}]")

    return "\n".join(lines)


def run_full_analysis(steps: int = 100, callback: Optional[Callable] = None,
                      target: str = "ml") -> str:
    """Run sensitivity sweep on all parameters, export CSVs, print summary.

    Args:
        target: Optimization target for loss calculation — "ml", "value", "ats", or "roi".

    Returns the output directory path.
    """
    if callback:
        callback(f"Starting full sensitivity analysis (target={target})...")

    all_results = sweep_all_parameters(steps=steps, callback=callback, target=target)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # Export individual CSVs
    for param_name, results in all_results.items():
        export_sweep_csv(param_name, results)

    # Build summary
    summary_lines = ["=" * 80, "SENSITIVITY ANALYSIS SUMMARY", "=" * 80, ""]

    base_w = WeightConfig()
    for param_name in sorted(all_results.keys()):
        results = all_results[param_name]
        current_val = getattr(base_w, param_name)
        lo, hi = EXTREME_RANGES.get(param_name, (0, 1))

        # Find best value for each metric
        best_mae_idx = np.argmin([r["spread_mae"] for r in results])
        best_win_idx = np.argmax([r["winner_pct"] for r in results])
        best_loss_idx = np.argmin([r["loss"] for r in results])
        best_roi_idx = np.argmax([r.get("ats_roi", 0) for r in results])
        best_ml_roi_idx = np.argmax([r.get("ml_roi", -100) for r in results])
        best_dog_roi_idx = np.argmax([r.get("dog_roi", -100) for r in results])

        best_mae_val = results[best_mae_idx]["param_value"]
        best_mae = results[best_mae_idx]["spread_mae"]
        best_win_val = results[best_win_idx]["param_value"]
        best_win = results[best_win_idx]["winner_pct"]
        best_loss_val = results[best_loss_idx]["param_value"]
        best_loss = results[best_loss_idx]["loss"]
        best_roi_val = results[best_roi_idx]["param_value"]
        best_roi = results[best_roi_idx].get("ats_roi", 0)
        best_ml_roi_val = results[best_ml_roi_idx]["param_value"]
        best_ml_roi = results[best_ml_roi_idx].get("ml_roi", -100)
        best_dog_roi_val = results[best_dog_roi_idx]["param_value"]
        best_dog_roi = results[best_dog_roi_idx].get("dog_roi", -100)
        best_dog_hit = results[best_dog_roi_idx].get("dog_hit_rate", 0)

        # Evaluate at current default
        current_idx = np.argmin([abs(r["param_value"] - current_val) for r in results])
        current_mae = results[current_idx]["spread_mae"]
        current_win = results[current_idx]["winner_pct"]
        current_roi = results[current_idx].get("ats_roi", 0)
        current_ml_roi = results[current_idx].get("ml_roi", -100)
        current_dog_roi = results[current_idx].get("dog_roi", -100)

        # Check which best values fall outside optimizer range
        outside_flags = []
        if param_name in OPTIMIZER_RANGES:
            opt_lo, opt_hi = OPTIMIZER_RANGES[param_name]
            if best_loss_val < opt_lo or best_loss_val > opt_hi:
                outside_flags.append("Loss")
            if best_ml_roi_val < opt_lo or best_ml_roi_val > opt_hi:
                outside_flags.append("ML ROI")
            if best_dog_roi_val < opt_lo or best_dog_roi_val > opt_hi:
                outside_flags.append("DogROI")
            if best_win_val < opt_lo or best_win_val > opt_hi:
                outside_flags.append("Win%")
        flag = f" ** OUTSIDE OPT RANGE ({', '.join(outside_flags)})" if outside_flags else ""

        summary_lines.append(f"  {param_name}:{flag}")
        summary_lines.append(f"    Current={current_val:.4f} (Win={current_win:.1f}%, ML ROI={current_ml_roi:.1f}%, DogROI={current_dog_roi:.1f}%)")
        summary_lines.append(f"    Best Loss={best_loss:.4f} at {best_loss_val:.4f}")
        summary_lines.append(f"    Best ML ROI={best_ml_roi:.1f}% at {best_ml_roi_val:.4f}")
        summary_lines.append(f"    Best DogROI={best_dog_roi:.1f}% (Hit={best_dog_hit:.1f}%) at {best_dog_roi_val:.4f}")
        summary_lines.append(f"    Best Win={best_win:.1f}% at {best_win_val:.4f}")
        if param_name in OPTIMIZER_RANGES:
            summary_lines.append(f"    Optimizer range: [{opt_lo}, {opt_hi}]")
        summary_lines.append("")

    summary = "\n".join(summary_lines)

    # Write summary to file
    summary_path = os.path.join(_OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    if callback:
        callback(f"Sensitivity analysis complete. Output: {_OUTPUT_DIR}")
        callback(summary)

    return _OUTPUT_DIR


# ──────────────────────────────────────────────────────────────
# Read & Apply sweep optimals
# ──────────────────────────────────────────────────────────────

def read_sweep_optimals(metric: str = "loss",
                        output_dir: str = None) -> Dict[str, Dict]:
    """Read all sweep CSVs and extract the optimal value for each parameter.

    Args:
        metric: Which metric to optimize. Options: 'loss', 'spread_mae',
                'winner_pct', 'ml_win_rate', 'ml_roi', 'dog_roi',
                'dog_hit_rate', 'dog_pick_rate', 'ats_rate', etc.
        output_dir: Directory containing sweep CSVs (default: data/sensitivity/).

    Returns:
        Dict mapping parameter name to:
            {best_value, best_metric, current_value, improvement, is_active}
        Only includes parameters where the sweep showed measurable variation.
    """
    out = output_dir or _OUTPUT_DIR
    import glob
    files = sorted(glob.glob(os.path.join(out, "sweep_*.csv")))
    if not files:
        raise FileNotFoundError(f"No sweep CSVs found in {out}. Run --all first.")

    minimize = metric in ("spread_mae", "total_mae", "loss")
    # For metrics not in every CSV (older sweeps), fall back gracefully
    betting_metrics = {"ats_rate", "edge_rate", "ats_roi", "edge_roi",
                       "ml_win_rate", "ml_roi",
                       "dog_pick_rate", "dog_hit_rate", "dog_roi"}
    base_w = WeightConfig()
    optimals = {}

    for fpath in files:
        name = os.path.basename(fpath).replace("sweep_", "").replace(".csv", "")
        with open(fpath, "r") as f:
            rows = list(csv.DictReader(f))

        vals = [float(r["param_value"]) for r in rows]
        if metric in betting_metrics and metric not in rows[0]:
            logger.warning("Metric %s not in CSV %s — re-run sweeps to include it", metric, fpath)
            continue
        scores = [float(r.get(metric, 0)) for r in rows]

        # Check if parameter has any effect
        score_range = max(scores) - min(scores)
        is_active = score_range > 0.03  # tolerance for noise

        if minimize:
            best_idx = int(np.argmin(scores))
        else:
            best_idx = int(np.argmax(scores))

        best_val = vals[best_idx]
        best_score = scores[best_idx]
        current_val = getattr(base_w, name, None)

        # Evaluate current default
        if current_val is not None:
            cur_idx = int(np.argmin([abs(v - current_val) for v in vals]))
            cur_score = scores[cur_idx]
            if minimize:
                improvement = cur_score - best_score
            else:
                improvement = best_score - cur_score
        else:
            cur_score = best_score
            improvement = 0.0

        # Check if optimal is at edge of sweep range
        at_edge = best_idx < 3 or best_idx > len(vals) - 4

        optimals[name] = {
            "best_value": best_val,
            "best_metric": best_score,
            "current_value": current_val,
            "current_metric": cur_score,
            "improvement": improvement,
            "is_active": is_active,
            "at_edge": at_edge,
            "metric_range": score_range,
        }

    return optimals


def apply_sweep_optimals(metric: str = "loss",
                         only_active: bool = True,
                         snapshot_name: str = "pre_sweep_apply",
                         callback: Optional[Callable] = None) -> Dict[str, float]:
    """Apply best values from sweep CSVs to the prediction pipeline.

    1. Saves a snapshot of the current state (for easy rollback).
    2. Reads sweep CSVs and finds optimal values.
    3. Updates only 'active' parameters (those that actually affect the evaluator).
    4. Saves the new WeightConfig to the database.

    Args:
        metric: Optimize for 'loss' (default), 'ml_roi', 'dog_roi', etc.
        only_active: If True, skip parameters with no measurable effect.
        snapshot_name: Name for the pre-apply snapshot.
        callback: Progress callback function.

    Returns:
        Dict of {param_name: new_value} for all changed parameters.
    """
    # 1. Snapshot current state
    snap_path = save_snapshot(snapshot_name,
                              notes=f"Auto-saved before applying sweep optimals (metric={metric})")
    if callback:
        callback(f"Snapshot saved: {snap_path}")

    # 2. Read optimals
    optimals = read_sweep_optimals(metric=metric)

    # 3. Build new config
    w = get_weight_config()
    w_dict = w.to_dict()
    changes = {}

    if callback:
        callback(f"\n{'Parameter':<28} {'Current':>10} {'Optimal':>10} {'Change':>10} {'Active':>6}")
        callback("-" * 75)

    for name, info in sorted(optimals.items()):
        if only_active and not info["is_active"]:
            continue
        if name not in w_dict:
            continue

        old_val = w_dict[name]
        new_val = info["best_value"]

        # Skip if barely different
        if abs(new_val - old_val) < 0.0001:
            continue

        w_dict[name] = new_val
        changes[name] = new_val

        delta = new_val - old_val
        sign = "+" if delta > 0 else ""
        if callback:
            callback(f"  {name:<26} {old_val:>10.4f} {new_val:>10.4f} {sign}{delta:>9.4f} "
                     f"{'YES' if info['is_active'] else 'no':>6}")

    if not changes:
        if callback:
            callback("\nNo changes to apply.")
        return changes

    # 4. Save new weights
    new_w = WeightConfig.from_dict(w_dict)
    save_weight_config(new_w)
    invalidate_weight_cache()

    if callback:
        callback(f"\nApplied {len(changes)} weight changes.")
        callback(f"To rollback: restore snapshot '{snapshot_name}' from {snap_path}")

    return changes


# ──────────────────────────────────────────────────────────────
# Coordinate Descent — iterative multi-parameter optimization
# ──────────────────────────────────────────────────────────────

def coordinate_descent(params: Optional[List[str]] = None,
                       steps: int = 100,
                       max_rounds: int = 10,
                       convergence_threshold: float = 0.005,
                       callback: Optional[Callable] = None,
                       target: str = "ml") -> Dict[str, Any]:
    """Iteratively sweep each parameter and lock in the best value.

    This is far more effective than individual sweeps because it captures
    parameter interactions: after changing param A, the optimal for param B
    may shift. Repeats until convergence or max_rounds.

    Args:
        params: List of parameter names to optimize. Defaults to EXTREME_RANGES keys.
        steps: Sweep steps per parameter per round.
        max_rounds: Maximum descent rounds before stopping.
        convergence_threshold: Stop when round-over-round loss improvement < this.
        callback: Progress callback.
        target: Optimization target — "ats", "roi", or "ml".

    Returns:
        Dict with 'weights' (final WeightConfig dict), 'history' (per-round metrics),
        'changes' (param -> old, new), 'rounds' (count), 'final_loss' (best loss).
    """
    if params is None:
        params = sorted(p for p in EXTREME_RANGES.keys() if p not in VECTORIZED_EXCLUDED)

    games = _load_games(callback)
    vg = VectorizedGames(games)

    # Start from current weights
    w = get_weight_config()
    w_dict = w.to_dict()
    initial_metrics = vg.evaluate(w, target=target)
    initial_loss = initial_metrics["loss"]
    if callback:
        callback(f"Starting coordinate descent: {len(params)} params, {max_rounds} max rounds, target={target}")
        callback(f"Initial loss: {initial_loss:.4f}, ATS={initial_metrics['ats_rate']:.1f}%, "
                 f"ATS ROI={initial_metrics['ats_roi']:.1f}%, ML ROI={initial_metrics['ml_roi']:.1f}%")

    history = []
    all_changes = {}

    for round_num in range(1, max_rounds + 1):
        round_start_loss = vg.evaluate(WeightConfig.from_dict(w_dict), target=target)["loss"]
        improved_count = 0

        for pi, param_name in enumerate(params):
            lo, hi = EXTREME_RANGES.get(param_name, (-10, 10))
            values = np.linspace(lo, hi, steps)

            best_loss = float("inf")
            best_val = w_dict[param_name]

            for val in values:
                test_dict = w_dict.copy()
                test_dict[param_name] = float(val)
                test_w = WeightConfig.from_dict(test_dict)
                metrics = vg.evaluate(test_w, target=target)
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    best_val = float(val)

            old_val = w_dict[param_name]
            if abs(best_val - old_val) > 0.0001:
                w_dict[param_name] = best_val
                improved_count += 1
                if param_name not in all_changes:
                    all_changes[param_name] = {"old": old_val}
                all_changes[param_name]["new"] = best_val

            if callback and (pi + 1) % 5 == 0:
                callback(f"  Round {round_num}: {pi + 1}/{len(params)} params swept")

        round_end_metrics = vg.evaluate(WeightConfig.from_dict(w_dict), target=target)
        round_end_loss = round_end_metrics["loss"]
        improvement = round_start_loss - round_end_loss

        history.append({
            "round": round_num,
            "loss": round_end_loss,
            "spread_mae": round_end_metrics["spread_mae"],
            "winner_pct": round_end_metrics["winner_pct"],
            "ats_rate": round_end_metrics["ats_rate"],
            "ats_roi": round_end_metrics["ats_roi"],
            "ml_win_rate": round_end_metrics["ml_win_rate"],
            "ml_roi": round_end_metrics["ml_roi"],
            "improvement": improvement,
            "params_changed": improved_count,
        })

        if callback:
            callback(f"  Round {round_num} complete: loss={round_end_loss:.4f} "
                     f"(delta={improvement:+.4f}), {improved_count} params moved, "
                     f"MAE={round_end_metrics['spread_mae']:.2f}, ATS={round_end_metrics['ats_rate']:.1f}%, "
                     f"ATS ROI={round_end_metrics['ats_roi']:.1f}%, ML ROI={round_end_metrics['ml_roi']:.1f}%")

        if improvement < convergence_threshold:
            if callback:
                callback(f"Converged after {round_num} rounds (improvement {improvement:.6f} < {convergence_threshold})")
            break

    final_metrics = vg.evaluate(WeightConfig.from_dict(w_dict), target=target)

    return {
        "weights": w_dict,
        "history": history,
        "changes": all_changes,
        "rounds": len(history),
        "initial_loss": initial_loss,
        "final_loss": final_metrics["loss"],
        "final_mae": final_metrics["spread_mae"],
        "final_winner_pct": final_metrics["winner_pct"],
        "final_ats_rate": final_metrics["ats_rate"],
        "final_ats_roi": final_metrics["ats_roi"],
        "final_ml_roi": final_metrics["ml_roi"],
    }


# ──────────────────────────────────────────────────────────────
# Pairwise Interaction Grid
# ──────────────────────────────────────────────────────────────

def sweep_pairwise(param_a: str, param_b: str,
                   steps: int = 50,
                   games: Optional[List] = None,
                   vg: Optional[VectorizedGames] = None,
                   callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Sweep two parameters simultaneously on a 2D grid.

    Tests all combinations of param_a × param_b values to find the joint optimum
    and reveal interaction effects (synergy or antagonism).

    Returns:
        Dict with 'grid' (steps×steps array of loss values), 'a_values', 'b_values',
        'best_a', 'best_b', 'best_loss', 'independent_best_loss' (loss from individual optima).
    """
    if vg is None:
        if games is None:
            games = _load_games(callback)
        vg = VectorizedGames(games)

    lo_a, hi_a = EXTREME_RANGES.get(param_a, (-10, 10))
    lo_b, hi_b = EXTREME_RANGES.get(param_b, (-10, 10))
    a_vals = np.linspace(lo_a, hi_a, steps)
    b_vals = np.linspace(lo_b, hi_b, steps)

    base_w = get_weight_config()
    grid_loss = np.zeros((steps, steps))
    grid_mae = np.zeros((steps, steps))
    grid_win = np.zeros((steps, steps))

    total = steps * steps
    count = 0

    for i, va in enumerate(a_vals):
        for j, vb in enumerate(b_vals):
            w_dict = base_w.to_dict()
            w_dict[param_a] = float(va)
            w_dict[param_b] = float(vb)
            w = WeightConfig.from_dict(w_dict)
            m = vg.evaluate(w)
            grid_loss[i, j] = m["loss"]
            grid_mae[i, j] = m["spread_mae"]
            grid_win[i, j] = m["winner_pct"]
            count += 1

        if callback and (i + 1) % 10 == 0:
            callback(f"  Pairwise {param_a} x {param_b}: {i + 1}/{steps} rows")

    # Find joint optimum
    best_idx = np.unravel_index(np.argmin(grid_loss), grid_loss.shape)
    best_a = float(a_vals[best_idx[0]])
    best_b = float(b_vals[best_idx[1]])
    best_loss = float(grid_loss[best_idx])

    # Find individual optima for comparison
    # Sweep A alone at current B
    w_dict_a = base_w.to_dict()
    best_a_alone = w_dict_a[param_a]
    best_loss_a_alone = float("inf")
    for va in a_vals:
        w_dict_a[param_a] = float(va)
        m = vg.evaluate(WeightConfig.from_dict(w_dict_a))
        if m["loss"] < best_loss_a_alone:
            best_loss_a_alone = m["loss"]
            best_a_alone = float(va)

    w_dict_b = base_w.to_dict()
    best_b_alone = w_dict_b[param_b]
    best_loss_b_alone = float("inf")
    for vb in b_vals:
        w_dict_b[param_b] = float(vb)
        m = vg.evaluate(WeightConfig.from_dict(w_dict_b))
        if m["loss"] < best_loss_b_alone:
            best_loss_b_alone = m["loss"]
            best_b_alone = float(vb)

    # What would loss be if we applied both individual optima simultaneously?
    w_dict_both = base_w.to_dict()
    w_dict_both[param_a] = best_a_alone
    w_dict_both[param_b] = best_b_alone
    indep_loss = vg.evaluate(WeightConfig.from_dict(w_dict_both))["loss"]

    interaction = best_loss - indep_loss  # negative = synergy, positive = conflict

    return {
        "param_a": param_a,
        "param_b": param_b,
        "a_values": a_vals.tolist(),
        "b_values": b_vals.tolist(),
        "grid_loss": grid_loss,
        "grid_mae": grid_mae,
        "grid_win": grid_win,
        "best_a": best_a,
        "best_b": best_b,
        "best_loss": best_loss,
        "best_mae": float(grid_mae[best_idx]),
        "best_win": float(grid_win[best_idx]),
        "best_a_alone": best_a_alone,
        "best_b_alone": best_b_alone,
        "independent_loss": indep_loss,
        "interaction": interaction,
    }


def format_pairwise_ascii(result: Dict[str, Any], metric: str = "loss", size: int = 30) -> str:
    """Generate an ASCII heatmap of a pairwise sweep result."""
    pa = result["param_a"]
    pb = result["param_b"]
    grid = result[f"grid_{metric}"] if f"grid_{metric}" in result else result["grid_loss"]
    a_vals = result["a_values"]
    b_vals = result["b_values"]

    # Downsample grid to `size`
    step_a = max(1, len(a_vals) // size)
    step_b = max(1, len(b_vals) // size)
    g = grid[::step_a, ::step_b]
    av = a_vals[::step_a]
    bv = b_vals[::step_b]

    minimize = metric in ("loss", "spread_mae", "total_mae")
    vmin, vmax = float(np.min(g)), float(np.max(g))

    # Heatmap characters (light to dark)
    chars = " .:-=+*#%@"

    lines = [f"  {pa} (rows) x {pb} (cols) -> {metric}"]
    lines.append(f"  Joint best: {pa}={result['best_a']:.4f}, {pb}={result['best_b']:.4f} "
                 f"(loss={result['best_loss']:.4f})")
    interaction = result["interaction"]
    label = "SYNERGY" if interaction < -0.005 else ("CONFLICT" if interaction > 0.005 else "NEUTRAL")
    lines.append(f"  Interaction: {interaction:+.4f} ({label})")
    lines.append(f"  {' ':>10} {bv[0]:>8.2f} {'':>{len(bv)*1-16}}{bv[-1]:>8.2f}  <- {pb}")
    lines.append(f"  {' ':>10} {'_' * len(bv)}")

    for i, va in enumerate(av):
        row = ""
        for j in range(len(bv)):
            if vmax > vmin:
                if minimize:
                    norm = (g[i, j] - vmin) / (vmax - vmin)
                else:
                    norm = 1.0 - (g[i, j] - vmin) / (vmax - vmin)
            else:
                norm = 0.5
            idx = int(norm * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            row += chars[idx]
        lines.append(f"  {va:>10.3f}|{row}|")

    lines.append(f"  {'':>10} {'-' * len(bv)}")
    lines.append(f"  ^ {pa}")
    lines.append(f"  Legend: ' '=best  '@'=worst  (for {metric})")

    return "\n".join(lines)


def run_pairwise_analysis(params: Optional[List[str]] = None,
                          steps: int = 40,
                          callback: Optional[Callable] = None) -> str:
    """Run pairwise interaction analysis on all combinations of active parameters."""
    if params is None:
        # Use only the parameters that work in vectorized evaluation
        params = sorted(p for p in EXTREME_RANGES.keys() if p not in VECTORIZED_EXCLUDED)

    games = _load_games(callback)
    vg = VectorizedGames(games)

    # First, identify which params are actually active
    base_w = get_weight_config()
    active_params = []
    for p in params:
        lo, hi = EXTREME_RANGES.get(p, (-10, 10))
        lo_w = base_w.to_dict()
        hi_w = base_w.to_dict()
        lo_w[p] = lo
        hi_w[p] = hi
        lo_loss = vg.evaluate(WeightConfig.from_dict(lo_w))["loss"]
        hi_loss = vg.evaluate(WeightConfig.from_dict(hi_w))["loss"]
        if abs(lo_loss - hi_loss) > 0.03:
            active_params.append(p)

    if callback:
        callback(f"Active parameters: {len(active_params)} of {len(params)}")
        callback(f"Active: {', '.join(active_params)}")
        n_pairs = len(active_params) * (len(active_params) - 1) // 2
        callback(f"Running {n_pairs} pairwise grids ({steps}x{steps} each)...")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    summary_lines = ["=" * 80, "PAIRWISE INTERACTION SUMMARY", "=" * 80, ""]

    pair_count = 0
    total_pairs = len(active_params) * (len(active_params) - 1) // 2

    for i, pa in enumerate(active_params):
        for j, pb in enumerate(active_params):
            if j <= i:
                continue
            pair_count += 1
            if callback:
                callback(f"[{pair_count}/{total_pairs}] {pa} x {pb}...")

            result = sweep_pairwise(pa, pb, steps=steps, vg=vg, callback=callback)

            interaction = result["interaction"]
            label = "SYNERGY" if interaction < -0.005 else ("CONFLICT" if interaction > 0.005 else "NEUTRAL")

            summary_lines.append(f"  {pa} x {pb}:")
            summary_lines.append(f"    Joint best: {pa}={result['best_a']:.4f}, {pb}={result['best_b']:.4f}")
            summary_lines.append(f"    Joint loss={result['best_loss']:.4f}  "
                                 f"Indep loss={result['independent_loss']:.4f}  "
                                 f"Interaction={interaction:+.4f} ({label})")
            summary_lines.append(f"    MAE={result['best_mae']:.2f}  Win%={result['best_win']:.1f}%")
            summary_lines.append("")

            # Export CSV
            csv_path = os.path.join(_OUTPUT_DIR, f"pair_{pa}_x_{pb}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{pa}_value", f"{pb}_value", "loss", "spread_mae", "winner_pct"])
                a_vals = result["a_values"]
                b_vals = result["b_values"]
                gl = result["grid_loss"]
                gm = result["grid_mae"]
                gw = result["grid_win"]
                for ai in range(len(a_vals)):
                    for bi in range(len(b_vals)):
                        writer.writerow([f"{a_vals[ai]:.6f}", f"{b_vals[bi]:.6f}",
                                         f"{gl[ai, bi]:.4f}", f"{gm[ai, bi]:.4f}",
                                         f"{gw[ai, bi]:.2f}"])

    summary = "\n".join(summary_lines)
    summary_path = os.path.join(_OUTPUT_DIR, "pairwise_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    if callback:
        callback(f"\nPairwise analysis complete. Output: {_OUTPUT_DIR}")
        callback(summary)

    return _OUTPUT_DIR


# ──────────────────────────────────────────────────────────────
# 3-Parameter Triplet Interaction Sweep
# ──────────────────────────────────────────────────────────────

def sweep_triplet(param_a: str, param_b: str, param_c: str,
                  steps: int = 15,
                  vg: Optional[VectorizedGames] = None,
                  games: Optional[List] = None,
                  callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Sweep three parameters simultaneously on a 3D grid.

    Evaluates all steps³ combinations. With steps=15 that's 3375 evals per triplet
    (~3-4s with vectorized evaluation).

    Returns:
        Dict with best joint values, interaction strength vs independent optima,
        and vs best pairwise optima.
    """
    if vg is None:
        if games is None:
            games = _load_games(callback)
        vg = VectorizedGames(games)

    lo_a, hi_a = EXTREME_RANGES.get(param_a, (-10, 10))
    lo_b, hi_b = EXTREME_RANGES.get(param_b, (-10, 10))
    lo_c, hi_c = EXTREME_RANGES.get(param_c, (-10, 10))
    a_vals = np.linspace(lo_a, hi_a, steps)
    b_vals = np.linspace(lo_b, hi_b, steps)
    c_vals = np.linspace(lo_c, hi_c, steps)

    base_w = get_weight_config()

    best_loss = float("inf")
    best_a = best_b = best_c = 0.0
    best_mae = best_win = 0.0

    for i, va in enumerate(a_vals):
        for j, vb in enumerate(b_vals):
            for k, vc in enumerate(c_vals):
                w_dict = base_w.to_dict()
                w_dict[param_a] = float(va)
                w_dict[param_b] = float(vb)
                w_dict[param_c] = float(vc)
                m = vg.evaluate(WeightConfig.from_dict(w_dict))
                if m["loss"] < best_loss:
                    best_loss = m["loss"]
                    best_a, best_b, best_c = float(va), float(vb), float(vc)
                    best_mae = m["spread_mae"]
                    best_win = m["winner_pct"]

    # Find independent (single-param) optima applied together
    indep_best = {}
    for pname, vals in [(param_a, a_vals), (param_b, b_vals), (param_c, c_vals)]:
        ind_best_loss = float("inf")
        ind_best_val = base_w.to_dict()[pname]
        for v in vals:
            d = base_w.to_dict()
            d[pname] = float(v)
            l = vg.evaluate(WeightConfig.from_dict(d))["loss"]
            if l < ind_best_loss:
                ind_best_loss = l
                ind_best_val = float(v)
        indep_best[pname] = ind_best_val

    # Loss with all 3 independent optima applied simultaneously
    d_indep = base_w.to_dict()
    for pname, val in indep_best.items():
        d_indep[pname] = val
    indep_loss = vg.evaluate(WeightConfig.from_dict(d_indep))["loss"]

    # Find best pairwise among the 3 sub-pairs
    pair_losses = {}
    for p1, v1s, p2, v2s in [
        (param_a, a_vals, param_b, b_vals),
        (param_a, a_vals, param_c, c_vals),
        (param_b, b_vals, param_c, c_vals),
    ]:
        pair_best = float("inf")
        for v1 in v1s:
            for v2 in v2s:
                d = base_w.to_dict()
                d[p1] = float(v1)
                d[p2] = float(v2)
                l = vg.evaluate(WeightConfig.from_dict(d))["loss"]
                if l < pair_best:
                    pair_best = l
        pair_losses[f"{p1}+{p2}"] = pair_best

    best_pair_loss = min(pair_losses.values())

    interaction_vs_indep = best_loss - indep_loss  # negative = 3-way synergy vs independent
    interaction_vs_pair = best_loss - best_pair_loss  # negative = 3-way adds value over best pair

    return {
        "param_a": param_a, "param_b": param_b, "param_c": param_c,
        "best_a": best_a, "best_b": best_b, "best_c": best_c,
        "best_loss": best_loss, "best_mae": best_mae, "best_win": best_win,
        "indep_loss": indep_loss, "indep_best": indep_best,
        "pair_losses": pair_losses, "best_pair_loss": best_pair_loss,
        "interaction_vs_indep": interaction_vs_indep,
        "interaction_vs_pair": interaction_vs_pair,
    }


def run_triplet_analysis(anchor: str = "ff_tov_weight",
                         params: Optional[List[str]] = None,
                         steps: int = 15,
                         callback: Optional[Callable] = None) -> str:
    """Run 3-parameter triplet analysis: anchor × every pair of other active params.

    Outputs a single combined CSV (triplet_summary.csv) and text summary instead
    of individual files per triplet, to keep output manageable.

    Args:
        anchor: The anchored parameter tested in every triplet.
        params: List of params to pair with anchor. Defaults to all active params.
        steps: Grid resolution per axis (total evals = steps³ per triplet).
        callback: Progress callback.

    Returns:
        Output directory path.
    """
    if params is None:
        params = sorted(p for p in EXTREME_RANGES.keys()
                        if p not in VECTORIZED_EXCLUDED and p != anchor)

    games = _load_games(callback)
    vg = VectorizedGames(games)

    # Filter to active params (those that influence loss meaningfully)
    base_w = get_weight_config()
    active_params = []
    for p in params:
        lo, hi = EXTREME_RANGES.get(p, (-10, 10))
        lo_w = base_w.to_dict()
        hi_w = base_w.to_dict()
        lo_w[p] = lo
        hi_w[p] = hi
        lo_loss = vg.evaluate(WeightConfig.from_dict(lo_w))["loss"]
        hi_loss = vg.evaluate(WeightConfig.from_dict(hi_w))["loss"]
        if abs(lo_loss - hi_loss) > 0.03:
            active_params.append(p)

    n_triplets = len(active_params) * (len(active_params) - 1) // 2
    evals_per = steps ** 3
    total_evals = n_triplets * evals_per

    if callback:
        callback(f"Triplet analysis: {anchor} × {len(active_params)} active params")
        callback(f"  {n_triplets} triplets, {steps}³={evals_per} evals each, ~{total_evals:,} total")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    results = []
    triplet_idx = 0

    for i, pb in enumerate(active_params):
        for j, pc in enumerate(active_params):
            if j <= i:
                continue
            triplet_idx += 1
            if callback:
                callback(f"  [{triplet_idx}/{n_triplets}] {anchor} × {pb} × {pc} ...")

            r = sweep_triplet(anchor, pb, pc, steps=steps, vg=vg)
            results.append(r)

    # Sort by best joint loss
    results.sort(key=lambda r: r["best_loss"])

    # Write combined CSV
    csv_path = os.path.join(_OUTPUT_DIR, f"triplet_{anchor}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "param_b", "param_c",
            f"best_{anchor}", "best_param_b", "best_param_c",
            "joint_loss", "joint_mae", "joint_win_pct",
            "indep_loss", "best_pair_loss",
            "interaction_vs_indep", "interaction_vs_pair", "label",
        ])
        for rank, r in enumerate(results, 1):
            inter = r["interaction_vs_pair"]
            label = "SYNERGY" if inter < -0.005 else ("CONFLICT" if inter > 0.005 else "NEUTRAL")
            writer.writerow([
                rank, r["param_b"], r["param_c"],
                f"{r['best_a']:.6f}", f"{r['best_b']:.6f}", f"{r['best_c']:.6f}",
                f"{r['best_loss']:.4f}", f"{r['best_mae']:.4f}", f"{r['best_win']:.2f}",
                f"{r['indep_loss']:.4f}", f"{r['best_pair_loss']:.4f}",
                f"{r['interaction_vs_indep']:.4f}", f"{r['interaction_vs_pair']:.4f}",
                label,
            ])

    # Write text summary
    txt_path = os.path.join(_OUTPUT_DIR, f"triplet_{anchor}_summary.txt")
    lines = [
        "=" * 80,
        f"TRIPLET INTERACTION SUMMARY — anchor: {anchor}",
        f"  Steps per axis: {steps} ({steps}³ = {evals_per} evals/triplet)",
        f"  Triplets tested: {n_triplets}",
        f"  Current {anchor} = {getattr(base_w, anchor):.4f}",
        "=" * 80, "",
    ]

    base_loss = vg.evaluate(base_w)["loss"]
    lines.append(f"  Baseline loss (current weights): {base_loss:.4f}\n")

    for rank, r in enumerate(results, 1):
        inter_i = r["interaction_vs_indep"]
        inter_p = r["interaction_vs_pair"]
        label_i = "SYNERGY" if inter_i < -0.005 else ("CONFLICT" if inter_i > 0.005 else "NEUTRAL")
        label_p = "3WAY-SYNERGY" if inter_p < -0.005 else ("NO-EXTRA" if inter_p > -0.005 else "NEUTRAL")
        improvement = base_loss - r["best_loss"]

        lines.append(f"  #{rank}: {anchor} × {r['param_b']} × {r['param_c']}")
        lines.append(f"    Best values: {anchor}={r['best_a']:.4f}, "
                     f"{r['param_b']}={r['best_b']:.4f}, {r['param_c']}={r['best_c']:.4f}")
        lines.append(f"    Joint loss={r['best_loss']:.4f}  ({improvement:+.4f} vs baseline)")
        lines.append(f"    MAE={r['best_mae']:.2f}  Win%={r['best_win']:.1f}%")
        lines.append(f"    vs indep: {inter_i:+.4f} ({label_i})  "
                     f"vs best pair: {inter_p:+.4f} ({label_p})")
        lines.append(f"    Sub-pair losses: {', '.join(f'{k}={v:.4f}' for k, v in r['pair_losses'].items())}")
        lines.append("")

    summary_text = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    if callback:
        callback(f"\nTriplet analysis complete:")
        callback(f"  CSV: {csv_path}")
        callback(f"  Summary: {txt_path}")
        callback(f"\n  Top 5 triplets by joint loss:")
        for r in results[:5]:
            callback(f"    {anchor}={r['best_a']:.3f} × {r['param_b']}={r['best_b']:.3f} "
                     f"× {r['param_c']}={r['best_c']:.3f} -> loss={r['best_loss']:.4f} "
                     f"Win%={r['best_win']:.1f}%")

    return _OUTPUT_DIR


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Sensitivity sweep for prediction weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Sweep all params:         --all --steps 200 --chart
  Single param:             --param rating_matchup_mult --chart
  Show optimals:            --show --metric loss
  Apply optimals:           --apply --metric loss
  Coordinate descent:       --descent --steps 100
  Pairwise interactions:    --pairwise --steps 40
  Single pair:              --pairwise --param rating_matchup_mult --param2 def_factor_dampening
  Triplet analysis:         --triplet --anchor ff_tov_weight --steps 15
""")
    parser.add_argument("--param", type=str, help="Parameter name to sweep")
    parser.add_argument("--param2", type=str, help="Second parameter for pairwise sweep")
    parser.add_argument("--all", action="store_true", help="Sweep all parameters")
    parser.add_argument("--min", type=float, default=None, help="Sweep min value")
    parser.add_argument("--max", type=float, default=None, help="Sweep max value")
    parser.add_argument("--steps", type=int, default=200, help="Number of sweep steps")
    parser.add_argument("--chart", action="store_true", help="Print ASCII chart")
    parser.add_argument("--show", action="store_true",
                        help="Show optimal values from existing sweep CSVs without re-running")
    parser.add_argument("--apply", action="store_true",
                        help="Apply best sweep values to the pipeline (saves snapshot first)")
    parser.add_argument("--metric", type=str, default="loss",
                        choices=["loss", "spread_mae", "winner_pct", "ats_rate",
                                 "ats_roi", "edge_roi", "ml_win_rate", "ml_roi"],
                        help="Metric to optimize when using --show or --apply (default: loss)")
    parser.add_argument("--descent", action="store_true",
                        help="Run coordinate descent: iteratively sweep and lock in best values")
    parser.add_argument("--descent-rounds", type=int, default=10,
                        help="Max rounds for coordinate descent (default: 10)")
    parser.add_argument("--pairwise", action="store_true",
                        help="Run pairwise interaction analysis on active parameters")
    parser.add_argument("--triplet", action="store_true",
                        help="Run 3-parameter triplet interaction analysis")
    parser.add_argument("--anchor", type=str, default="ff_tov_weight",
                        help="Anchor parameter for triplet analysis (default: ff_tov_weight)")
    parser.add_argument("--target", type=str, default="ml",
                        choices=["ml", "value", "ats", "roi"],
                        help="Optimization target for loss calculation (default: ml)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    def cb(msg):
        print(msg)

    # --show: display optimals from existing CSVs
    if args.show:
        try:
            optimals = read_sweep_optimals(metric=args.metric)
        except FileNotFoundError as e:
            print(str(e))
            return

        print(f"\nOptimal values from sweep CSVs (metric={args.metric}):\n")
        print(f"  {'Parameter':<28} {'Current':>10} {'Optimal':>10} {'Improvement':>12} "
              f"{'Active':>7} {'Edge?':>6}")
        print(f"  {'-'*80}")

        for name, info in sorted(optimals.items(), key=lambda x: -x[1]['improvement']):
            imp = info['improvement']
            active = 'YES' if info['is_active'] else ''
            edge = 'EDGE' if info['at_edge'] else ''
            cur = info['current_value']
            best = info['best_value']
            if cur is not None:
                print(f"  {name:<28} {cur:>10.4f} {best:>10.4f} {imp:>+12.4f} "
                      f"{active:>7} {edge:>6}")
        return

    # --apply: apply optimals (but NOT if --descent is also set; descent handles apply itself)
    if args.apply and not args.descent:
        changes = apply_sweep_optimals(metric=args.metric, callback=cb)
        if changes:
            print(f"\nDone. {len(changes)} parameters updated.")
        return

    # --descent: iterative coordinate descent
    if args.descent:
        t0 = time.time()
        result = coordinate_descent(
            steps=min(args.steps, 100),  # cap steps per-param for speed
            max_rounds=args.descent_rounds,
            callback=cb,
            target=args.target,
        )
        elapsed = time.time() - t0

        print(f"\n{'='*60}")
        print(f"COORDINATE DESCENT RESULTS ({elapsed:.0f}s, target={args.target})")
        print(f"{'='*60}")
        print(f"  Rounds: {result['rounds']}")
        print(f"  Initial loss: {result['initial_loss']:.4f}")
        print(f"  Final loss:   {result['final_loss']:.4f}  "
              f"(improvement: {result['initial_loss'] - result['final_loss']:+.4f})")
        print(f"  Final MAE:    {result['final_mae']:.2f}")
        print(f"  Final Win%:   {result['final_winner_pct']:.1f}%")
        print(f"  Final ATS:    {result.get('final_ats_rate', 0):.1f}%  "
              f"ATS ROI: {result.get('final_ats_roi', 0):.1f}%  "
              f"ML ROI: {result.get('final_ml_roi', 0):.1f}%")
        print(f"\n  Parameter changes:")
        for name, info in sorted(result['changes'].items()):
            print(f"    {name:<28} {info['old']:>10.4f} -> {info['new']:>10.4f}")

        # Offer to apply
        print(f"\n  To apply these values:")
        print(f"    python -m src.analytics.sensitivity --descent --apply")
        if args.apply:
            snap_path = save_snapshot("pre_descent_apply",
                                      notes="Before coordinate descent apply")
            new_w = WeightConfig.from_dict(result["weights"])
            save_weight_config(new_w)
            invalidate_weight_cache()
            print(f"\n  Applied! Snapshot: {snap_path}")
        return

    # --triplet: 3-parameter interaction analysis
    if args.triplet:
        t0 = time.time()
        run_triplet_analysis(
            anchor=args.anchor,
            steps=min(args.steps, 20),  # cap at 20 (8000 evals/triplet)
            callback=cb,
        )
        elapsed = time.time() - t0
        print(f"\n  Completed in {elapsed:.0f}s")
        return

    # --pairwise: interaction analysis
    if args.pairwise:
        t0 = time.time()

        if args.param and args.param2:
            # Single pair
            games = _load_games(cb)
            vg = VectorizedGames(games)
            result = sweep_pairwise(args.param, args.param2,
                                    steps=min(args.steps, 50), vg=vg, callback=cb)
            elapsed = time.time() - t0
            print(f"\n  {args.param} x {args.param2} ({elapsed:.0f}s):")
            print(f"    Joint best: {args.param}={result['best_a']:.4f}, "
                  f"{args.param2}={result['best_b']:.4f}")
            print(f"    Joint loss={result['best_loss']:.4f}  "
                  f"Indep loss={result['independent_loss']:.4f}")
            interaction = result['interaction']
            label = "SYNERGY" if interaction < -0.005 else (
                "CONFLICT" if interaction > 0.005 else "NEUTRAL")
            print(f"    Interaction: {interaction:+.4f} ({label})")
            print(f"    MAE={result['best_mae']:.2f}  Win%={result['best_win']:.1f}%")
            if args.chart:
                print()
                print(format_pairwise_ascii(result))
        else:
            # All pairs of active params
            run_pairwise_analysis(steps=min(args.steps, 40), callback=cb)
            elapsed = time.time() - t0
            print(f"\n  Completed in {elapsed:.0f}s")
        return

    if args.all:
        run_full_analysis(steps=args.steps, callback=cb, target=args.target)
        return

    if not args.param:
        print(f"Available parameters: {', '.join(sorted(EXTREME_RANGES.keys()))}")
        return

    lo = args.min if args.min is not None else EXTREME_RANGES.get(args.param, (-10, 10))[0]
    hi = args.max if args.max is not None else EXTREME_RANGES.get(args.param, (-10, 10))[1]

    print(f"Sweeping {args.param} from {lo} to {hi} in {args.steps} steps...")
    t0 = time.time()

    results = sweep_parameter(args.param, lo, hi, steps=args.steps, callback=cb, target=args.target)
    path = export_sweep_csv(args.param, results)
    print(f"\nCSV exported: {path}  ({time.time() - t0:.1f}s)")

    # Print chart
    if args.chart:
        for metric in ("spread_mae", "winner_pct", "loss"):
            print()
            print(format_ascii_chart(args.param, results, metric=metric))
    else:
        # Always show summary
        best_mae_idx = np.argmin([r["spread_mae"] for r in results])
        best_win_idx = np.argmax([r["winner_pct"] for r in results])
        best = results[best_mae_idx]
        bestw = results[best_win_idx]
        current = getattr(WeightConfig(), args.param)
        print(f"\n  Current default: {current}")
        print(f"  Best MAE: {best['spread_mae']:.4f} at {args.param}={best['param_value']:.4f}")
        print(f"  Best Win%: {bestw['winner_pct']:.2f}% at {args.param}={bestw['param_value']:.4f}")
        if args.param in OPTIMIZER_RANGES:
            lo_o, hi_o = OPTIMIZER_RANGES[args.param]
            print(f"  Optimizer range: [{lo_o}, {hi_o}]")
            if best["param_value"] < lo_o or best["param_value"] > hi_o:
                print(f"  ** OPTIMAL VALUE IS OUTSIDE OPTIMIZER RANGE **")


if __name__ == "__main__":
    main()
