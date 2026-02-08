"""Weight optimiser for the prediction engine.

Uses random search over the most impactful weights in ``WeightConfig``,
scoring each combination with a loss function computed from backtest
results.  The best weights are saved to the ``model_weights`` DB table
so they become the defaults for ``predict_matchup``.

Also provides:
- **Residual calibration** (spread-bin corrections)
- **Feature importance** analysis (toggle each factor to measure impact)
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from src.analytics.backtester import run_backtest, BacktestResults
from src.analytics.weight_config import (
    WeightConfig,
    get_weight_config,
    save_weights,
    set_weight_config,
)
from src.database.db import get_conn


# -----------------------------------------------------------------------
# Loss function
# -----------------------------------------------------------------------

def _loss(results: BacktestResults) -> float:
    """Compute a scalar loss from backtest results.

    Lower is better.  Combines:
    - Mean absolute spread error (primary)
    - Mean absolute total error (secondary)
    - Winner-pick accuracy bonus (inverted so lower = better)
    """
    if results.total_games == 0:
        return 999.0

    preds = results.predictions
    spread_mae = sum(abs(p.spread_error) for p in preds) / len(preds)
    total_mae = sum(abs(p.total_error) for p in preds) / len(preds)
    winner_pct = results.overall_spread_accuracy  # 0-100

    # Weighted loss: spread accuracy matters most
    # winner_pct is 0-100; invert so 60% accuracy → 40 penalty
    return spread_mae * 1.0 + total_mae * 0.3 + (100 - winner_pct) * 0.1


# -----------------------------------------------------------------------
# Search ranges for the top parameters
# -----------------------------------------------------------------------

# Each entry: (weight_name, min, max)
# Focused on the weights with the most impact on predictions.
TOP_WEIGHTS: List[Tuple[str, float, float]] = [
    ("def_factor_dampening",   0.25,  0.75),
    ("turnover_margin_mult",   0.15,  0.65),
    ("rebound_diff_mult",      0.02,  0.15),
    ("rating_matchup_mult",    0.02,  0.15),
    ("four_factors_scale",     0.10,  0.60),
    ("pace_mult",              0.08,  0.35),
    ("espn_model_weight",      0.60,  0.95),
    ("clutch_scale",           0.02,  0.10),
    ("hustle_effort_mult",     0.005, 0.05),
    ("fatigue_total_mult",     0.10,  0.60),
]


def _random_config(base: WeightConfig) -> WeightConfig:
    """Create a new config with randomly perturbed top weights."""
    d = base.to_dict()
    for name, lo, hi in TOP_WEIGHTS:
        d[name] = random.uniform(lo, hi)
    # Keep espn_weight = 1 - espn_model_weight
    d["espn_weight"] = round(1.0 - d["espn_model_weight"], 4)
    return WeightConfig.from_dict(d)


# -----------------------------------------------------------------------
# Public API – optimise
# -----------------------------------------------------------------------

@dataclass
class OptimiserResult:
    best_config: WeightConfig
    best_loss: float
    baseline_loss: float
    trials_run: int
    improvement_pct: float
    history: List[Tuple[int, float]] = field(default_factory=list)


def run_weight_optimiser(
    n_trials: int = 200,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> OptimiserResult:
    """Run random-search weight optimisation.

    1. Establish baseline loss with current (default) weights.
    2. For *n_trials*, sample random weights, run a fast backtest,
       and track the best score.
    3. Save the best weights to the DB.

    Args:
        n_trials: Number of random configurations to try.
        progress_cb: Optional progress callback.

    Returns:
        OptimiserResult with the best config and metrics.
    """
    progress = progress_cb or (lambda _: None)

    # ---- baseline ----
    progress("Establishing baseline with current weights...")
    baseline_cfg = WeightConfig()  # defaults
    set_weight_config(baseline_cfg)
    baseline_results = run_backtest(min_games_before=5)
    baseline_loss = _loss(baseline_results)
    progress(
        f"Baseline: loss={baseline_loss:.2f}, "
        f"winner={baseline_results.overall_spread_accuracy:.1f}%, "
        f"games={baseline_results.total_games}"
    )

    best_cfg = baseline_cfg
    best_loss = baseline_loss
    history: List[Tuple[int, float]] = [(0, baseline_loss)]

    # ---- random search ----
    t0 = time.time()
    for trial in range(1, n_trials + 1):
        candidate = _random_config(baseline_cfg)
        set_weight_config(candidate)

        try:
            results = run_backtest(min_games_before=5)
            trial_loss = _loss(results)
        except Exception as exc:
            progress(f"  Trial {trial} error: {exc}")
            continue

        if trial_loss < best_loss:
            best_loss = trial_loss
            best_cfg = candidate
            progress(
                f"  Trial {trial}/{n_trials}: NEW BEST loss={trial_loss:.2f} "
                f"(winner={results.overall_spread_accuracy:.1f}%)"
            )
        elif trial % 25 == 0:
            progress(
                f"  Trial {trial}/{n_trials}: loss={trial_loss:.2f} "
                f"(best so far={best_loss:.2f})"
            )

        history.append((trial, trial_loss))

    elapsed = time.time() - t0
    improvement = ((baseline_loss - best_loss) / baseline_loss * 100) if baseline_loss else 0

    # ---- persist best ----
    save_weights(best_cfg)
    set_weight_config(best_cfg)
    progress(
        f"Optimisation complete: {n_trials} trials in {elapsed:.0f}s, "
        f"loss {baseline_loss:.2f} → {best_loss:.2f} "
        f"({improvement:+.1f}% improvement)"
    )

    return OptimiserResult(
        best_config=best_cfg,
        best_loss=best_loss,
        baseline_loss=baseline_loss,
        trials_run=n_trials,
        improvement_pct=improvement,
        history=history,
    )


# -----------------------------------------------------------------------
# Residual calibration
# -----------------------------------------------------------------------

_CALIBRATION_TABLE = "residual_calibration"


def _ensure_calibration_table() -> None:
    with get_conn() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_CALIBRATION_TABLE} (
                bin_label     TEXT PRIMARY KEY,
                bin_low       REAL NOT NULL,
                bin_high      REAL NOT NULL,
                avg_residual  REAL NOT NULL,
                sample_count  INTEGER NOT NULL
            )
        """)
        conn.commit()


def build_residual_calibration(
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Dict]:
    """Run a backtest, bin predictions by predicted spread, and compute
    the average residual (predicted - actual) in each bin.

    Saves the calibration table to the DB and returns it.

    Bins: [-18,-12], (-12,-8], (-8,-4], (-4,-1], (-1,1], (1,4], (4,8],
          (8,12], (12,18]
    """
    progress = progress_cb or (lambda _: None)

    progress("Running backtest for calibration data...")
    results = run_backtest(min_games_before=5)
    if results.total_games == 0:
        progress("No games to calibrate")
        return {}

    bins = [
        ("big_away",    -18.0, -12.0),
        ("med_away",    -12.0,  -8.0),
        ("small_away",   -8.0,  -4.0),
        ("slight_away",  -4.0,  -1.0),
        ("toss_up",      -1.0,   1.0),
        ("slight_home",   1.0,   4.0),
        ("small_home",    4.0,   8.0),
        ("med_home",      8.0,  12.0),
        ("big_home",     12.0,  18.0),
    ]

    calibration: Dict[str, Dict] = {}
    _ensure_calibration_table()

    with get_conn() as conn:
        conn.execute(f"DELETE FROM {_CALIBRATION_TABLE}")

        for label, lo, hi in bins:
            preds_in_bin = [
                p for p in results.predictions
                if lo <= p.predicted_spread < hi
            ]
            if not preds_in_bin:
                continue

            avg_residual = sum(p.spread_error for p in preds_in_bin) / len(preds_in_bin)
            calibration[label] = {
                "bin_low": lo,
                "bin_high": hi,
                "avg_residual": round(avg_residual, 3),
                "sample_count": len(preds_in_bin),
            }
            conn.execute(
                f"INSERT INTO {_CALIBRATION_TABLE} (bin_label, bin_low, bin_high, avg_residual, sample_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (label, lo, hi, avg_residual, len(preds_in_bin)),
            )
            progress(
                f"  {label} [{lo:+.0f}, {hi:+.0f}): "
                f"n={len(preds_in_bin)}, avg_residual={avg_residual:+.2f}"
            )

        conn.commit()

    progress(f"Calibration complete: {len(calibration)} bins populated")
    return calibration


def load_residual_calibration() -> List[Dict]:
    """Load saved calibration bins from the DB."""
    _ensure_calibration_table()
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT bin_label, bin_low, bin_high, avg_residual, sample_count "
            f"FROM {_CALIBRATION_TABLE} ORDER BY bin_low"
        ).fetchall()
    return [
        {
            "bin_label": r[0], "bin_low": r[1], "bin_high": r[2],
            "avg_residual": r[3], "sample_count": r[4],
        }
        for r in rows
    ]


def apply_residual_calibration(predicted_spread: float) -> float:
    """Adjust a predicted spread using the saved residual calibration.

    Subtracts the average residual for the matching bin, effectively
    correcting for systematic over/under-prediction in that range.
    """
    calibration = load_residual_calibration()
    if not calibration:
        return predicted_spread

    for cal in calibration:
        if cal["bin_low"] <= predicted_spread < cal["bin_high"]:
            # Only apply if we have enough samples
            if cal["sample_count"] >= 5:
                return predicted_spread - cal["avg_residual"]
            break

    return predicted_spread


# -----------------------------------------------------------------------
# Feature importance analysis
# -----------------------------------------------------------------------

@dataclass
class FeatureImportance:
    feature_name: str
    baseline_loss: float
    disabled_loss: float
    impact: float          # disabled_loss - baseline_loss (positive = feature helps)
    impact_pct: float      # % change in loss when disabled


def run_feature_importance(
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[FeatureImportance]:
    """Measure each factor's contribution by disabling it and measuring
    the change in backtest loss.

    "Disabling" a factor means setting its weight to zero (or its
    dampening to 1.0 for the defensive factor).

    Returns a list sorted by impact (most helpful feature first).
    """
    progress = progress_cb or (lambda _: None)

    # ---- baseline ----
    progress("Running baseline backtest...")
    base_cfg = get_weight_config(force_reload=True)
    set_weight_config(base_cfg)
    base_results = run_backtest(min_games_before=5)
    base_loss = _loss(base_results)
    progress(f"Baseline loss: {base_loss:.2f}")

    # ---- features to test ----
    # Each: (display_name, weight_key, disabled_value)
    features = [
        ("Defensive dampening",     "def_factor_dampening",   1.0),   # 1.0 = no dampening
        ("Turnover margin",         "turnover_margin_mult",   0.0),
        ("Rebound differential",    "rebound_diff_mult",      0.0),
        ("Off/Def rating matchup",  "rating_matchup_mult",    0.0),
        ("Four Factors",            "four_factors_scale",      0.0),
        ("Clutch adjustment",       "clutch_scale",            0.0),
        ("Hustle spread",           "hustle_effort_mult",      0.0),
        ("Pace factor",             "pace_mult",               0.0),
        ("Steals disruption",       "steals_penalty",          0.0),
        ("Blocks disruption",       "blocks_penalty",          0.0),
        ("OREB impact",             "oreb_mult",               0.0),
        ("Hustle total",            "hustle_defl_penalty",     0.0),
        ("Fatigue total impact",    "fatigue_total_mult",      0.0),
        ("ESPN blending",           "espn_weight",             0.0),
    ]

    results: List[FeatureImportance] = []

    for i, (name, key, disabled_val) in enumerate(features, 1):
        progress(f"  Testing [{i}/{len(features)}] {name}...")

        # Create a modified config with this one feature disabled
        d = base_cfg.to_dict()
        d[key] = disabled_val
        # Keep ESPN weights consistent
        if key == "espn_weight":
            d["espn_model_weight"] = 1.0
        elif key == "espn_model_weight":
            d["espn_weight"] = 0.0
        test_cfg = WeightConfig.from_dict(d)
        set_weight_config(test_cfg)

        try:
            test_results = run_backtest(min_games_before=5)
            test_loss = _loss(test_results)
        except Exception:
            test_loss = base_loss  # treat failure as no change

        impact = test_loss - base_loss  # positive = disabling hurts = feature helps
        impact_pct = (impact / base_loss * 100) if base_loss else 0.0

        results.append(FeatureImportance(
            feature_name=name,
            baseline_loss=base_loss,
            disabled_loss=test_loss,
            impact=round(impact, 3),
            impact_pct=round(impact_pct, 2),
        ))

        direction = "HELPS" if impact > 0.05 else ("HURTS" if impact < -0.05 else "neutral")
        progress(
            f"    {name}: loss {base_loss:.2f} → {test_loss:.2f} "
            f"(impact: {impact:+.3f}, {direction})"
        )

    # Restore original config
    set_weight_config(base_cfg)

    # Sort by impact descending (most helpful first)
    results.sort(key=lambda x: x.impact, reverse=True)
    progress(f"Feature importance complete: {len(results)} features analysed")
    return results
