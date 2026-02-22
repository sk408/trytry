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

from src.analytics.weight_config import WeightConfig, get_weight_config, OPTIMIZER_RANGES
from src.analytics.weight_optimizer import VectorizedGames
from src.analytics.prediction import PrecomputedGame, precompute_game_data

logger = logging.getLogger(__name__)

_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sensitivity")

# Default extreme sweep ranges per parameter type
# These intentionally go WAY beyond optimizer bounds to find hidden value
EXTREME_RANGES = {
    "def_factor_dampening":  (-2.0,   5.0),
    "turnover_margin_mult":  (-5.0,  10.0),
    "rebound_diff_mult":     (-5.0,  10.0),
    "rating_matchup_mult":   (-5.0,  20.0),
    "four_factors_scale":    (-500.0, 1000.0),
    "clutch_scale":          (-2.0,  5.0),
    "hustle_effort_mult":    (-2.0,  5.0),
    "pace_mult":             (-5.0,  10.0),
    "fatigue_total_mult":    (-2.0,  5.0),
    "espn_model_weight":     (0.0,   1.0),
    "espn_weight":           (0.0,   1.0),
    "ml_ensemble_weight":    (-2.0,  5.0),
    "ml_disagree_damp":      (0.0,   3.0),
    "spread_clamp":          (3.0, 100.0),
    "ff_efg_weight":         (-2.0,  5.0),
    "ff_tov_weight":         (-2.0,  5.0),
    "ff_oreb_weight":        (-2.0,  5.0),
    "ff_fta_weight":         (-2.0,  5.0),
    "fatigue_b2b":           (-5.0,  15.0),
    "fatigue_3in4":          (-5.0,  10.0),
    "fatigue_4in6":          (-5.0,  10.0),
    "pace_baseline":         (85.0, 110.0),
    "steals_penalty":        (-1.0,  3.0),
    "blocks_penalty":        (-1.0,  3.0),
    "oreb_mult":             (-2.0,  5.0),
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
                    callback: Optional[Callable] = None) -> List[Dict]:
    """Sweep a single parameter through [min_val, max_val] in `steps` increments.

    Returns a list of dicts with columns:
        param_value, spread_mae, total_mae, winner_pct, loss
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
        metrics = vg.evaluate(w)
        results.append({
            "param_value": round(float(val), 6),
            "spread_mae": round(metrics["spread_mae"], 4),
            "total_mae": round(metrics["total_mae"], 4),
            "winner_pct": round(metrics["winner_pct"], 2),
            "loss": round(metrics["loss"], 4),
        })
        if callback and (i + 1) % 50 == 0:
            callback(f"  {param_name}: {i + 1}/{steps} ({val:.3f} → MAE={metrics['spread_mae']:.2f})")

    return results


def sweep_all_parameters(steps: int = 100,
                         callback: Optional[Callable] = None) -> Dict[str, List[Dict]]:
    """Sweep every sweepable parameter and return results keyed by param name."""
    games = _load_games(callback)
    vg = VectorizedGames(games)

    all_results = {}
    params = sorted(EXTREME_RANGES.keys())

    for pi, param_name in enumerate(params):
        lo, hi = EXTREME_RANGES[param_name]
        if callback:
            callback(f"[{pi + 1}/{len(params)}] Sweeping {param_name} ({lo} to {hi})...")
        results = sweep_parameter(param_name, lo, hi, steps=steps, vg=vg, callback=callback)
        all_results[param_name] = results

    return all_results


def export_sweep_csv(param_name: str, results: List[Dict], output_dir: str = None) -> str:
    """Write sweep results to a CSV file. Returns the file path."""
    out = output_dir or _OUTPUT_DIR
    os.makedirs(out, exist_ok=True)

    filename = f"sweep_{param_name}.csv"
    path = os.path.join(out, filename)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param_value", "spread_mae", "total_mae", "winner_pct", "loss"])
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
    lines.append(f"  {param_name} → {metric}")
    lines.append(f"  Range: [{min_v:.3f}, {max_v:.3f}]  Best: {best_val:.4f} ({metric}={best_metric:.4f})")
    lines.append(f"  {'─' * chart_width}")

    # Downsample to height rows
    step = max(1, len(results) // height)
    sampled = results[::step][:height]

    for r in sampled:
        val = r["param_value"]
        m = r[metric]
        bar_len = int((m - min_m) / range_m * (chart_width - 2)) if range_m > 0 else 0
        bar_len = max(0, min(chart_width - 2, bar_len))
        marker = "█" * bar_len
        is_best = abs(val - best_val) < (max_v - min_v) / len(results) * 2
        prefix = "★" if is_best else " "
        lines.append(f"{prefix}{val:>8.3f} |{marker}")

    lines.append(f"  {'─' * chart_width}")
    lines.append(f"  {min_m:<10.3f}{' ' * (chart_width - 20)}{max_m:>10.3f}")

    # Current default value
    current = getattr(WeightConfig(), param_name)
    lines.append(f"  Current default: {current}")
    if param_name in OPTIMIZER_RANGES:
        lo, hi = OPTIMIZER_RANGES[param_name]
        lines.append(f"  Optimizer range: [{lo}, {hi}]")

    return "\n".join(lines)


def run_full_analysis(steps: int = 100, callback: Optional[Callable] = None) -> str:
    """Run sensitivity sweep on all parameters, export CSVs, print summary.

    Returns the output directory path.
    """
    if callback:
        callback("Starting full sensitivity analysis...")

    all_results = sweep_all_parameters(steps=steps, callback=callback)

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

        best_mae_val = results[best_mae_idx]["param_value"]
        best_mae = results[best_mae_idx]["spread_mae"]
        best_win_val = results[best_win_idx]["param_value"]
        best_win = results[best_win_idx]["winner_pct"]
        best_loss_val = results[best_loss_idx]["param_value"]
        best_loss = results[best_loss_idx]["loss"]

        # Evaluate at current default
        current_idx = np.argmin([abs(r["param_value"] - current_val) for r in results])
        current_mae = results[current_idx]["spread_mae"]
        current_win = results[current_idx]["winner_pct"]

        # Check if optimizer range is too narrow
        if param_name in OPTIMIZER_RANGES:
            opt_lo, opt_hi = OPTIMIZER_RANGES[param_name]
            outside = (best_mae_val < opt_lo or best_mae_val > opt_hi or
                       best_win_val < opt_lo or best_win_val > opt_hi)
            flag = " ⚠️ OUTSIDE OPT RANGE" if outside else ""
        else:
            flag = ""

        summary_lines.append(f"  {param_name}:")
        summary_lines.append(f"    Current={current_val:.4f} (MAE={current_mae:.2f}, Win={current_win:.1f}%)")
        summary_lines.append(f"    Best MAE={best_mae:.2f} at {best_mae_val:.4f}")
        summary_lines.append(f"    Best Win={best_win:.1f}% at {best_win_val:.4f}")
        summary_lines.append(f"    Best Loss={best_loss:.4f} at {best_loss_val:.4f}{flag}")
        summary_lines.append("")

    summary = "\n".join(summary_lines)

    # Write summary to file
    summary_path = os.path.join(_OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    if callback:
        callback(f"Sensitivity analysis complete. Output: {_OUTPUT_DIR}")
        callback(summary)

    return _OUTPUT_DIR


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sensitivity sweep for prediction weights")
    parser.add_argument("--param", type=str, help="Parameter name to sweep")
    parser.add_argument("--all", action="store_true", help="Sweep all parameters")
    parser.add_argument("--min", type=float, default=None, help="Sweep min value")
    parser.add_argument("--max", type=float, default=None, help="Sweep max value")
    parser.add_argument("--steps", type=int, default=200, help="Number of sweep steps")
    parser.add_argument("--chart", action="store_true", help="Print ASCII chart")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    def cb(msg):
        print(msg)

    if args.all:
        run_full_analysis(steps=args.steps, callback=cb)
        return

    if not args.param:
        print(f"Available parameters: {', '.join(sorted(EXTREME_RANGES.keys()))}")
        return

    lo = args.min if args.min is not None else EXTREME_RANGES.get(args.param, (-10, 10))[0]
    hi = args.max if args.max is not None else EXTREME_RANGES.get(args.param, (-10, 10))[1]

    print(f"Sweeping {args.param} from {lo} to {hi} in {args.steps} steps...")
    t0 = time.time()

    results = sweep_parameter(args.param, lo, hi, steps=args.steps, callback=cb)
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
                print(f"  ⚠️  OPTIMAL VALUE IS OUTSIDE OPTIMIZER RANGE")


if __name__ == "__main__":
    main()
