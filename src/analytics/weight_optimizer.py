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
from src.analytics.prediction import (
    PrecomputedGame,
    precompute_game_data,
    predict_from_precomputed,
)
from src.analytics.weight_config import (
    WeightConfig,
    get_weight_config,
    save_weights,
    save_team_weights,
    clear_team_weights,
    set_weight_config,
)
from src.database.db import get_conn

# Optional ML libraries (graceful degradation)
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    import xgboost as xgb
    import shap as _shap
    _HAS_ML = True
except ImportError:
    _HAS_ML = False


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


def _fast_loss(games: List[PrecomputedGame], w: WeightConfig) -> float:
    """Evaluate a weight config against precomputed games — pure arithmetic,
    zero DB I/O.  Returns the same loss metric as ``_loss``."""
    if not games:
        return 999.0

    n = len(games)
    spread_ae_sum = 0.0
    total_ae_sum = 0.0
    winner_correct = 0

    for g in games:
        pred_spread, pred_total = predict_from_precomputed(g, w)
        actual_spread = g.actual_home_score - g.actual_away_score
        actual_total = g.actual_home_score + g.actual_away_score

        spread_ae_sum += abs(pred_spread - actual_spread)
        total_ae_sum += abs(pred_total - actual_total)

        # Winner check
        if pred_spread > 0.5:
            pred_winner = "HOME"
        elif pred_spread < -0.5:
            pred_winner = "AWAY"
        else:
            pred_winner = "PUSH"

        if actual_spread > 0:
            actual_winner = "HOME"
        elif actual_spread < 0:
            actual_winner = "AWAY"
        else:
            actual_winner = "PUSH"

        if pred_winner == actual_winner or (pred_winner == "PUSH" and abs(actual_spread) <= 3):
            winner_correct += 1

    spread_mae = spread_ae_sum / n
    total_mae = total_ae_sum / n
    winner_pct = winner_correct / n * 100

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
    precomputed_games: Optional[List[PrecomputedGame]] = None,
) -> OptimiserResult:
    """Optimise prediction weights.

    Phase 1: Precompute all game data from the DB once.
    Phase 2: Evaluate weight combinations as pure in-memory arithmetic
             (no DB I/O per trial).

    Uses Optuna Bayesian optimisation (TPE sampler) when available,
    falling back to random search otherwise.

    Args:
        n_trials: Number of configurations to evaluate.
        progress_cb: Optional progress callback.
        precomputed_games: Optional pre-built list; skips Phase 1 if provided.

    Returns:
        OptimiserResult with the best config and metrics.
    """
    progress = progress_cb or (lambda _: None)

    # ── Phase 1: Precompute all game data (one-time DB cost) ──
    if precomputed_games is not None:
        games = precomputed_games
        progress(f"Using {len(games)} pre-supplied precomputed games")
    else:
        progress("Phase 1/2: Precomputing game data (one-time DB read)...")
        games = precompute_game_data(progress_cb=progress)
    if not games:
        progress("No games found for optimisation")
        return OptimiserResult(
            best_config=WeightConfig(),
            best_loss=999.0,
            baseline_loss=999.0,
            trials_run=0,
            improvement_pct=0.0,
        )

    # ── Baseline ──
    baseline_cfg = WeightConfig()  # defaults
    baseline_loss = _fast_loss(games, baseline_cfg)
    progress(
        f"Baseline: loss={baseline_loss:.2f}, "
        f"games={len(games)}"
    )

    # ── Phase 2: Fast optimisation (pure arithmetic) ──
    progress(f"Phase 2/2: Optimising over {n_trials} trials (in-memory, no DB)...")
    if _HAS_OPTUNA:
        progress("Using Optuna TPE sampler for Bayesian optimisation...")
        return _run_optuna_optimiser(baseline_cfg, baseline_loss, n_trials, progress, games)
    else:
        progress("Optuna not installed — using random search fallback...")
        return _run_random_optimiser(baseline_cfg, baseline_loss, n_trials, progress, games)


def _run_optuna_optimiser(
    baseline_cfg: WeightConfig,
    baseline_loss: float,
    n_trials: int,
    progress: Callable[[str], None],
    games: List[PrecomputedGame] | None = None,
) -> OptimiserResult:
    """Bayesian optimisation via Optuna TPE sampler.

    When *games* is provided, each trial uses ``_fast_loss`` (pure
    arithmetic).  Falls back to full backtests only if precomputed data
    is unavailable.
    """
    history: List[Tuple[int, float]] = [(0, baseline_loss)]
    use_fast = games is not None and len(games) > 0

    def objective(trial: "optuna.Trial") -> float:
        d = baseline_cfg.to_dict()
        for name, lo, hi in TOP_WEIGHTS:
            d[name] = trial.suggest_float(name, lo, hi)
        d["espn_weight"] = round(1.0 - d["espn_model_weight"], 4)
        cfg = WeightConfig.from_dict(d)

        if use_fast:
            return _fast_loss(games, cfg)

        # Fallback: full backtest (slow)
        set_weight_config(cfg)
        try:
            results = run_backtest(min_games_before=5)
            return _loss(results)
        except Exception as exc:
            progress(f"  Trial {trial.number + 1} error: {exc}")
            return 999.0

    def _callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        history.append((trial.number + 1, trial.value))
        if trial.value <= study.best_value:
            progress(
                f"  Trial {trial.number + 1}/{n_trials}: NEW BEST loss={trial.value:.2f}"
            )
        elif (trial.number + 1) % 25 == 0:
            progress(
                f"  Trial {trial.number + 1}/{n_trials}: loss={trial.value:.2f} "
                f"(best so far={study.best_value:.2f})"
            )

    t0 = time.time()
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_callback])
    elapsed = time.time() - t0

    best_loss = study.best_value
    improvement = ((baseline_loss - best_loss) / baseline_loss * 100) if baseline_loss else 0

    # Build and persist best config
    d = baseline_cfg.to_dict()
    d.update(study.best_params)
    d["espn_weight"] = round(1.0 - d.get("espn_model_weight", 0.8), 4)
    best_cfg = WeightConfig.from_dict(d)
    save_weights(best_cfg)
    set_weight_config(best_cfg)

    progress(
        f"Optuna optimisation complete: {n_trials} trials in {elapsed:.0f}s, "
        f"loss {baseline_loss:.2f} → {best_loss:.2f} ({improvement:+.1f}%)"
    )

    return OptimiserResult(
        best_config=best_cfg,
        best_loss=best_loss,
        baseline_loss=baseline_loss,
        trials_run=n_trials,
        improvement_pct=improvement,
        history=history,
    )


def _run_random_optimiser(
    baseline_cfg: WeightConfig,
    baseline_loss: float,
    n_trials: int,
    progress: Callable[[str], None],
    games: List[PrecomputedGame] | None = None,
) -> OptimiserResult:
    """Random-search fallback when Optuna is not installed."""
    best_cfg = baseline_cfg
    best_loss = baseline_loss
    history: List[Tuple[int, float]] = [(0, baseline_loss)]
    use_fast = games is not None and len(games) > 0

    t0 = time.time()
    for trial in range(1, n_trials + 1):
        candidate = _random_config(baseline_cfg)

        if use_fast:
            trial_loss = _fast_loss(games, candidate)
        else:
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
                f"  Trial {trial}/{n_trials}: NEW BEST loss={trial_loss:.2f}"
            )
        elif trial % 25 == 0:
            progress(
                f"  Trial {trial}/{n_trials}: loss={trial_loss:.2f} "
                f"(best so far={best_loss:.2f})"
            )

        history.append((trial, trial_loss))

    elapsed = time.time() - t0
    improvement = ((baseline_loss - best_loss) / baseline_loss * 100) if baseline_loss else 0

    save_weights(best_cfg)
    set_weight_config(best_cfg)
    progress(
        f"Random-search complete: {n_trials} trials in {elapsed:.0f}s, "
        f"loss {baseline_loss:.2f} → {best_loss:.2f} ({improvement:+.1f}%)"
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
# Per-team weight refinement with regressive validation
# -----------------------------------------------------------------------

# Subset of weights to refine per-team (most impactful, fewest params)
_TEAM_REFINE_WEIGHTS: List[Tuple[str, float, float]] = [
    ("def_factor_dampening",   0.25,  0.75),
    ("turnover_margin_mult",   0.15,  0.65),
    ("four_factors_scale",     0.10,  0.60),
    ("pace_mult",              0.08,  0.35),
]


@dataclass
class TeamRefinementResult:
    team_id: int
    team_abbr: str
    used_team_weights: bool   # True = per-team won, False = global won
    global_loss_recent: float  # loss on last 5 games with global weights
    team_loss_recent: float    # loss on last 5 games with per-team weights
    global_loss_train: float
    team_loss_train: float
    reason: str               # why this choice was made


def run_per_team_refinement(
    n_trials: int = 100,
    holdout_games: int = 5,
    max_worse_pct: float = 30.0,
    progress_cb: Optional[Callable[[str], None]] = None,
    precomputed_games: Optional[List[PrecomputedGame]] = None,
) -> List[TeamRefinementResult]:
    """Optimise weights per-team with regressive validation.

    For each team:
    1. Split that team's games into **training** (older) and **holdout**
       (most recent ``holdout_games``).
    2. Optimise 4 key weights on the training set (±20% from global).
    3. Evaluate both global and per-team weights on the holdout set.
    4. Keep per-team weights ONLY if they beat global on the holdout
       AND aren't wildly wrong (> ``max_worse_pct`` worse on any metric).

    Args:
        n_trials: Trials per team for the refinement search.
        holdout_games: Number of most recent games to hold out for validation.
        max_worse_pct: If per-team holdout loss is this % worse than global
                       on the holdout, discard per-team weights.
        progress_cb: Optional progress callback.
        precomputed_games: Optional pre-built list; skips Phase 1 if provided.

    Returns:
        List of ``TeamRefinementResult`` (one per team with enough data).
    """
    progress = progress_cb or (lambda _: None)

    # ── Phase 1: Precompute all game data ──
    if precomputed_games is not None:
        all_games = precomputed_games
        progress(f"Using {len(all_games)} pre-supplied precomputed games")
    else:
        progress("Phase 1/3: Precomputing game data...")
        all_games = precompute_game_data(progress_cb=progress)
    if not all_games:
        progress("No games found")
        return []

    global_cfg = get_weight_config(force_reload=True)

    # Group games by team (each game appears for both home & away team)
    from collections import defaultdict
    team_games: Dict[int, List[PrecomputedGame]] = defaultdict(list)
    for g in all_games:
        team_games[g.home_team_id].append(g)
        team_games[g.away_team_id].append(g)

    # Get team abbreviations
    with get_conn() as conn:
        team_abbrs = {
            r[0]: r[1]
            for r in conn.execute("SELECT team_id, abbreviation FROM teams").fetchall()
        }

    results: List[TeamRefinementResult] = []
    teams_sorted = sorted(team_games.keys(), key=lambda t: team_abbrs.get(t, "???"))
    total_teams = len(teams_sorted)

    progress(f"Phase 2/3: Refining weights for {total_teams} teams...")
    clear_team_weights()  # start fresh

    for idx, team_id in enumerate(teams_sorted):
        abbr = team_abbrs.get(team_id, "???")
        games = team_games[team_id]

        # Sort chronologically by game date
        games.sort(key=lambda g: g.game_date)

        if len(games) < holdout_games + 5:
            progress(f"  [{idx+1}/{total_teams}] {abbr}: skipped (only {len(games)} games)")
            continue

        # Split: train on older games, validate on most recent
        train = games[:-holdout_games]
        holdout = games[-holdout_games:]

        # Global loss on both sets
        global_loss_train = _fast_loss_team(train, team_id, global_cfg)
        global_loss_holdout = _fast_loss_team(holdout, team_id, global_cfg)

        # ── Per-team optimisation on training set ──
        best_team_cfg = global_cfg
        best_team_loss_train = global_loss_train

        base_dict = global_cfg.to_dict()
        for trial in range(n_trials):
            d = dict(base_dict)
            for name, lo, hi in _TEAM_REFINE_WEIGHTS:
                global_val = base_dict[name]
                # ±20% from global value, clamped to allowed range
                delta = global_val * 0.20
                trial_lo = max(lo, global_val - delta)
                trial_hi = min(hi, global_val + delta)
                d[name] = random.uniform(trial_lo, trial_hi)
            d["espn_weight"] = round(1.0 - d.get("espn_model_weight", 0.8), 4)
            candidate = WeightConfig.from_dict(d)

            trial_loss = _fast_loss_team(train, team_id, candidate)
            if trial_loss < best_team_loss_train:
                best_team_loss_train = trial_loss
                best_team_cfg = candidate

        # ── Regressive validation on holdout ──
        team_loss_holdout = _fast_loss_team(holdout, team_id, best_team_cfg)

        # Decision logic
        team_better = team_loss_holdout < global_loss_holdout
        wildly_wrong = (
            global_loss_holdout > 0
            and team_loss_holdout > global_loss_holdout * (1 + max_worse_pct / 100)
        )

        if team_better and not wildly_wrong:
            save_team_weights(team_id, best_team_cfg)
            reason = (
                f"Per-team wins: holdout {team_loss_holdout:.2f} vs "
                f"global {global_loss_holdout:.2f}"
            )
            used_team = True
        elif wildly_wrong:
            reason = (
                f"Per-team discarded (wildly wrong): holdout {team_loss_holdout:.2f} vs "
                f"global {global_loss_holdout:.2f} (>{max_worse_pct:.0f}% worse)"
            )
            used_team = False
        else:
            reason = (
                f"Global wins: holdout {global_loss_holdout:.2f} vs "
                f"per-team {team_loss_holdout:.2f}"
            )
            used_team = False

        results.append(TeamRefinementResult(
            team_id=team_id,
            team_abbr=abbr,
            used_team_weights=used_team,
            global_loss_recent=global_loss_holdout,
            team_loss_recent=team_loss_holdout,
            global_loss_train=global_loss_train,
            team_loss_train=best_team_loss_train,
            reason=reason,
        ))

        symbol = "+" if used_team else "-"
        progress(
            f"  [{idx+1}/{total_teams}] {abbr}: {symbol} "
            f"train {global_loss_train:.1f}→{best_team_loss_train:.1f}, "
            f"holdout {global_loss_holdout:.1f}→{team_loss_holdout:.1f} "
            f"{'(SAVED)' if used_team else '(global kept)'}"
        )

    adopted = sum(1 for r in results if r.used_team_weights)
    progress(
        f"Phase 3/3: Done — {adopted}/{len(results)} teams got per-team weights, "
        f"rest use global"
    )
    return results


def _fast_loss_team(
    games: List[PrecomputedGame],
    team_id: int,
    w: WeightConfig,
) -> float:
    """Evaluate loss for games involving a specific team."""
    if not games:
        return 999.0

    n = 0
    spread_ae_sum = 0.0
    total_ae_sum = 0.0
    winner_correct = 0

    for g in games:
        # Only count this game for the team in question
        if g.home_team_id != team_id and g.away_team_id != team_id:
            continue

        pred_spread, pred_total = predict_from_precomputed(g, w)
        actual_spread = g.actual_home_score - g.actual_away_score
        actual_total = g.actual_home_score + g.actual_away_score

        spread_ae_sum += abs(pred_spread - actual_spread)
        total_ae_sum += abs(pred_total - actual_total)

        if pred_spread > 0.5:
            pw = "HOME"
        elif pred_spread < -0.5:
            pw = "AWAY"
        else:
            pw = "PUSH"

        if actual_spread > 0:
            aw = "HOME"
        elif actual_spread < 0:
            aw = "AWAY"
        else:
            aw = "PUSH"

        if pw == aw or (pw == "PUSH" and abs(actual_spread) <= 3):
            winner_correct += 1
        n += 1

    if n == 0:
        return 999.0

    spread_mae = spread_ae_sum / n
    total_mae = total_ae_sum / n
    winner_pct = winner_correct / n * 100

    return spread_mae * 1.0 + total_mae * 0.3 + (100 - winner_pct) * 0.1


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


# -----------------------------------------------------------------------
# ML-based feature importance (XGBoost + SHAP)
# -----------------------------------------------------------------------

@dataclass
class MLFeatureImportance:
    feature_name: str
    shap_importance: float  # mean |SHAP value|
    direction: str  # "positive", "negative", or "mixed"


def run_ml_feature_importance(
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[MLFeatureImportance]:
    """Train XGBoost on the raw feature matrix and use SHAP to quantify
    each feature's contribution to predicting spread.

    Requires ``xgboost`` and ``shap`` (``pip install xgboost shap``).
    """
    if not _HAS_ML:
        raise ImportError(
            "XGBoost and SHAP are required for ML feature importance. "
            "Install with: pip install xgboost shap"
        )

    import numpy as np
    import pandas as pd

    progress = progress_cb or (lambda _: None)

    # Step 1 – collect feature matrix from historical games
    progress("Collecting feature matrix from historical games...")
    from src.analytics.backtester import (
        get_actual_game_results,
        get_team_profile,
        _get_roster_for_game,
    )
    from src.analytics.prediction import predict_matchup_detailed

    games_df = get_actual_game_results()
    if games_df.empty:
        progress("No game data available")
        return []

    games_df = games_df.sort_values("game_date")
    feature_rows: list[dict] = []
    targets: list[float] = []

    total = len(games_df)
    for idx, (_, game) in enumerate(games_df.iterrows()):
        if idx % 50 == 0:
            progress(f"  Processing game {idx + 1}/{total}...")

        game_date_raw = game["game_date"]
        if isinstance(game_date_raw, str):
            from datetime import date as _date
            gd = _date.fromisoformat(game_date_raw[:10])
        else:
            gd = game_date_raw

        home_id = int(game["home_team_id"])
        away_id = int(game["away_team_id"])

        # Skip if teams don't have enough history
        home_profile = get_team_profile(home_id, gd)
        away_profile = get_team_profile(away_id, gd)
        if home_profile["games"] < 5 or away_profile["games"] < 5:
            continue

        home_pids = _get_roster_for_game(home_id, gd)
        away_pids = _get_roster_for_game(away_id, gd)
        if not home_pids or not away_pids:
            continue

        try:
            detailed = predict_matchup_detailed(
                home_team_id=home_id,
                away_team_id=away_id,
                home_players=home_pids,
                away_players=away_pids,
                game_date=gd,
            )
            if detailed.features:
                feature_rows.append(detailed.features)
                actual_spread = float(game["home_score"]) - float(game["away_score"])
                targets.append(actual_spread)
        except Exception:
            continue

    if len(feature_rows) < 30:
        progress(f"Not enough games with features ({len(feature_rows)}), need at least 30")
        return []

    # Step 2 – build DataFrame
    X = pd.DataFrame(feature_rows).fillna(0.0)
    y = np.array(targets, dtype=float)
    progress(f"Feature matrix: {X.shape[0]} games x {X.shape[1]} features")

    # Step 3 – train XGBoost
    progress("Training XGBoost regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    # Step 4 – SHAP values
    progress("Computing SHAP values...")
    explainer = _shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_signed_shap = shap_values.mean(axis=0)

    # Step 5 – build result list
    ml_results: List[MLFeatureImportance] = []
    for i, col in enumerate(X.columns):
        signed = float(mean_signed_shap[i])
        direction = "positive" if signed > 0.01 else ("negative" if signed < -0.01 else "mixed")
        ml_results.append(MLFeatureImportance(
            feature_name=col,
            shap_importance=round(float(mean_abs_shap[i]), 4),
            direction=direction,
        ))

    ml_results.sort(key=lambda x: x.shap_importance, reverse=True)
    progress(f"ML feature importance complete: {len(ml_results)} features analysed")
    for r in ml_results[:5]:
        progress(f"  {r.feature_name}: SHAP={r.shap_importance:.4f} ({r.direction})")

    return ml_results


# -----------------------------------------------------------------------
# League-wide FFT error pattern analysis
# -----------------------------------------------------------------------

@dataclass
class FFTPattern:
    period_games: float      # every N games
    period_days: float       # approximate calendar days
    magnitude: float         # strength of pattern (0-1 normalised)
    phase: float             # phase angle
    description: str         # human-readable summary


def run_fft_error_analysis(
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[FFTPattern]:
    """Detect periodic patterns in league-wide prediction errors.

    Runs a full backtest, orders errors chronologically, and applies
    FFT to find dominant periodic components.  Only patterns affecting
    the **whole league** (not individual teams) and exceeding 2x the
    average spectral magnitude are reported.

    Requires ``numpy`` (already installed with pandas).
    """
    import numpy as np

    progress = progress_cb or (lambda _: None)

    # Step 1 – run backtest
    progress("Running backtest for error analysis...")
    results = run_backtest(min_games_before=5)

    if results.total_games < 50:
        progress(f"Not enough games ({results.total_games}), need at least 50 for FFT")
        return []

    # Sort predictions chronologically
    preds = sorted(results.predictions, key=lambda p: str(p.game_date))
    errors = np.array([p.spread_error for p in preds], dtype=float)
    N = len(errors)

    # Compute average days between games
    dates = [p.game_date for p in preds]
    if len(dates) > 1:
        total_span_days = (dates[-1] - dates[0]).days
        avg_days_per_game = total_span_days / (N - 1) if N > 1 else 1.0
    else:
        total_span_days = 0
        avg_days_per_game = 1.0

    progress(f"Analysing {N} game errors over {total_span_days} days...")

    # Step 2 – FFT
    fft_vals = np.fft.rfft(errors)
    magnitudes = np.abs(fft_vals)
    phases = np.angle(fft_vals)

    # Skip DC component (index 0 = mean error)
    if len(magnitudes) <= 1:
        progress("Not enough frequency bins for analysis")
        return []

    mag_no_dc = magnitudes[1:]
    mean_mag = float(mag_no_dc.mean())
    max_mag = float(mag_no_dc.max())
    threshold = mean_mag * 2.0

    # Step 3 – find significant peaks
    patterns: List[FFTPattern] = []
    for k in range(1, len(magnitudes)):
        if magnitudes[k] < threshold:
            continue

        period_games = N / k
        period_days = period_games * avg_days_per_game

        # Skip noise (very short) and unreliable (very long)
        if period_games < 3 or period_games > N / 2:
            continue

        norm_mag = float(magnitudes[k] / max_mag) if max_mag > 0 else 0.0

        # Human-readable label
        if period_days <= 4:
            desc = f"~{period_days:.0f}-day micro-cycle"
        elif period_days <= 10:
            desc = f"~weekly cycle ({period_days:.0f} days)"
        elif period_days <= 20:
            desc = f"~biweekly cycle ({period_days:.0f} days)"
        elif period_days <= 40:
            desc = f"~monthly cycle ({period_days:.0f} days)"
        else:
            desc = f"~{period_days:.0f}-day long-term cycle"

        patterns.append(FFTPattern(
            period_games=round(period_games, 1),
            period_days=round(period_days, 1),
            magnitude=round(norm_mag, 4),
            phase=round(float(phases[k]), 4),
            description=desc,
        ))

    # Sort by strength and keep top 5
    patterns.sort(key=lambda p: p.magnitude, reverse=True)
    patterns = patterns[:5]

    if patterns:
        progress(f"Found {len(patterns)} significant periodic error pattern(s):")
        for p in patterns:
            progress(f"  {p.description}: strength={p.magnitude:.2f}")
    else:
        progress("No significant periodic error patterns detected (good — errors are random)")

    return patterns


# -----------------------------------------------------------------------
# Combo: Global + Per-Team optimisation in one pass
# -----------------------------------------------------------------------

@dataclass
class ComboOptimiserResult:
    """Combined results from global + per-team optimisation."""
    global_result: OptimiserResult
    team_results: List[TeamRefinementResult]
    total_seconds: float = 0.0


def run_combo_optimiser(
    n_trials: int = 200,
    team_trials: int = 100,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ComboOptimiserResult:
    """Run global weight optimisation followed by per-team refinement.

    Precomputes game data once and shares it across both phases for
    maximum speed.

    Args:
        n_trials: Trials for the global optimiser.
        team_trials: Trials per team for refinement.
        progress_cb: Optional progress callback.

    Returns:
        ComboOptimiserResult with both sets of results.
    """
    import time as _time
    t0 = _time.perf_counter()
    progress = progress_cb or (lambda _: None)

    # ── Phase 1: Precompute once ──
    progress("[Combo 1/3] Precomputing game data (shared)...")
    games = precompute_game_data(progress_cb=progress)
    if not games:
        progress("No games found — aborting")
        return ComboOptimiserResult(
            global_result=OptimiserResult(
                best_config=WeightConfig(), best_loss=999.0,
                baseline_loss=999.0, trials_run=0, improvement_pct=0.0,
            ),
            team_results=[],
        )

    # ── Phase 2: Global optimisation ──
    progress(f"[Combo 2/3] Global optimisation ({n_trials} trials)...")
    global_result = run_weight_optimiser(
        n_trials=n_trials,
        progress_cb=progress,
        precomputed_games=games,
    )

    # ── Phase 3: Per-team refinement ──
    progress(f"[Combo 3/3] Per-team refinement ({team_trials} trials/team)...")
    team_results = run_per_team_refinement(
        n_trials=team_trials,
        progress_cb=progress,
        precomputed_games=games,
    )

    elapsed = _time.perf_counter() - t0
    adopted = sum(1 for r in team_results if r.used_team_weights)
    progress(
        f"Combo complete in {elapsed:.0f}s — "
        f"global improved {global_result.improvement_pct:+.1f}%, "
        f"{adopted}/{len(team_results)} teams got custom weights"
    )

    return ComboOptimiserResult(
        global_result=global_result,
        team_results=team_results,
        total_seconds=elapsed,
    )
