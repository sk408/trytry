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

import numpy as np

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


# ======================================================================
# Vectorized game data — flatten PrecomputedGame list into NumPy arrays
# for ~50-100x faster loss evaluation.
# ======================================================================

class VectorizedGames:
    """Pre-flattened arrays from a list of PrecomputedGame objects.

    All prediction arithmetic can be done on these arrays with NumPy
    instead of looping in Python.  Built once, reused for every trial.
    """
    __slots__ = (
        "n", "home_team_ids", "away_team_ids",
        "actual_spread", "actual_total",
        "home_pts_raw", "away_pts_raw",
        "home_def_factor_raw", "away_def_factor_raw",
        "home_tuning_corr", "away_tuning_corr",
        "home_fatigue", "away_fatigue",
        "home_court",
        "to_diff", "reb_diff",
        "home_off", "away_off", "home_def", "away_def",
        # Four Factors edges (precomputed, weight-independent parts)
        "ff_efg_edge", "ff_tov_edge", "ff_oreb_edge", "ff_fta_edge",
        # Clutch
        "clutch_diff",
        # Hustle
        "hustle_effort_diff",
        # Hustle total
        "combined_deflections",
        # Pace
        "avg_pace",
        # Total sub-components
        "combined_steals", "combined_blocks",
        "combined_oreb", "combined_fatigue",
        # Injury context
        "injury_count_diff", "injury_ppg_lost_diff", "injury_minutes_lost_diff",
    )

    def __init__(self, games: List[PrecomputedGame]) -> None:
        import math as _math

        def _num(v, default: float = 0.0) -> float:
            """Convert optional/NaN values to a safe float."""
            if v is None:
                return default
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return default
            if _math.isnan(fv) or _math.isinf(fv):
                return default
            return fv

        n = len(games)
        self.n = n

        self.home_team_ids = np.array([g.home_team_id for g in games], dtype=np.int64)
        self.away_team_ids = np.array([g.away_team_id for g in games], dtype=np.int64)
        self.actual_spread = np.array(
            [g.actual_home_score - g.actual_away_score for g in games], dtype=np.float64
        )
        self.actual_total = np.array(
            [g.actual_home_score + g.actual_away_score for g in games], dtype=np.float64
        )

        # Player projections (weight-independent)
        self.home_pts_raw = np.array([g.home_proj.get("points", 0) for g in games], dtype=np.float64)
        self.away_pts_raw = np.array([g.away_proj.get("points", 0) for g in games], dtype=np.float64)

        # Defensive factors
        self.home_def_factor_raw = np.array([g.home_def_factor_raw for g in games], dtype=np.float64)
        self.away_def_factor_raw = np.array([g.away_def_factor_raw for g in games], dtype=np.float64)

        # Tuning corrections
        self.home_tuning_corr = np.array([g.home_tuning_home_corr for g in games], dtype=np.float64)
        self.away_tuning_corr = np.array([g.away_tuning_away_corr for g in games], dtype=np.float64)

        # Fatigue
        self.home_fatigue = np.array([g.home_fatigue_penalty for g in games], dtype=np.float64)
        self.away_fatigue = np.array([g.away_fatigue_penalty for g in games], dtype=np.float64)

        # Home court
        self.home_court = np.array([g.home_court for g in games], dtype=np.float64)

        # Turnover diff
        self.to_diff = np.array(
            [g.home_proj.get("turnover_margin", 0) - g.away_proj.get("turnover_margin", 0)
             for g in games], dtype=np.float64
        )

        # Rebound diff
        self.reb_diff = np.array(
            [g.home_proj.get("rebounds", 0) - g.away_proj.get("rebounds", 0)
             for g in games], dtype=np.float64
        )

        # Off/Def ratings
        self.home_off = np.array([g.home_off for g in games], dtype=np.float64)
        self.away_off = np.array([g.away_off for g in games], dtype=np.float64)
        self.home_def = np.array([g.home_def for g in games], dtype=np.float64)
        self.away_def = np.array([g.away_def for g in games], dtype=np.float64)

        # Four Factors — precompute the weight-independent edge values
        def _ff_edge(key_a, key_b, home_key, away_key, sign=1):
            """Compute four-factor edge array."""
            vals = np.zeros(n, dtype=np.float64)
            for i, g in enumerate(games):
                hff = g.home_ff or {}
                aff = g.away_ff or {}
                ha = hff.get(key_a) or 0
                ab = aff.get(key_b) or 0
                aa = aff.get(key_a) or 0
                hb = hff.get(key_b) or 0
                vals[i] = sign * ((ha - ab) - (aa - hb))
            return vals

        self.ff_efg_edge = _ff_edge("efg_pct", "opp_efg_pct", "home", "away")
        self.ff_tov_edge = np.zeros(n, dtype=np.float64)
        for i, g in enumerate(games):
            hff = g.home_ff or {}
            aff = g.away_ff or {}
            h_tov = hff.get("tm_tov_pct") or 0
            a_tov = aff.get("tm_tov_pct") or 0
            h_opp_tov = hff.get("opp_tm_tov_pct") or 0
            a_opp_tov = aff.get("opp_tm_tov_pct") or 0
            self.ff_tov_edge[i] = (a_tov - h_opp_tov) - (h_tov - a_opp_tov)
        self.ff_oreb_edge = _ff_edge("oreb_pct", "opp_oreb_pct", "home", "away")
        self.ff_fta_edge = _ff_edge("fta_rate", "opp_fta_rate", "home", "away")

        # Clutch — net rating diff (use _num to handle None values safely)
        self.clutch_diff = np.array(
            [_num((g.home_clutch or {}).get("clutch_net_rating")) -
             _num((g.away_clutch or {}).get("clutch_net_rating"))
             for g in games], dtype=np.float64
        )

        # Hustle — effort differential for spread
        # effort = deflections + contested * hustle_contested_wt
        # We precompute without the contested weight since it varies
        # But hustle_contested_wt is fixed at 0.3 and not in TOP_WEIGHTS,
        # so we can bake it in.
        def _hustle_effort(g, wt=0.3):
            hh = g.home_hustle or {}
            ah = g.away_hustle or {}
            h_e = _num(hh.get("deflections")) + _num(hh.get("contested_shots")) * wt
            a_e = _num(ah.get("deflections")) + _num(ah.get("contested_shots")) * wt
            return h_e - a_e
        self.hustle_effort_diff = np.array(
            [_hustle_effort(g) for g in games], dtype=np.float64
        )

        # Hustle total — combined deflections
        self.combined_deflections = np.array(
            [_num((g.home_hustle or {}).get("deflections")) +
             _num((g.away_hustle or {}).get("deflections"))
             for g in games], dtype=np.float64
        )

        # Pace
        self.avg_pace = np.array(
            [(g.home_pace + g.away_pace) / 2 for g in games], dtype=np.float64
        )

        # Total sub-components
        self.combined_steals = np.array(
            [g.home_proj.get("steals", 0) + g.away_proj.get("steals", 0)
             for g in games], dtype=np.float64
        )
        self.combined_blocks = np.array(
            [g.home_proj.get("blocks", 0) + g.away_proj.get("blocks", 0)
             for g in games], dtype=np.float64
        )
        self.combined_oreb = np.array(
            [g.home_proj.get("oreb", 0) + g.away_proj.get("oreb", 0)
             for g in games], dtype=np.float64
        )
        self.combined_fatigue = self.home_fatigue + self.away_fatigue

        # Injury context — diffs (positive = home has *more* injury impact)
        self.injury_count_diff = np.array(
            [getattr(g, "home_injured_count", 0) - getattr(g, "away_injured_count", 0)
             for g in games], dtype=np.float64
        )
        self.injury_ppg_lost_diff = np.array(
            [getattr(g, "home_injury_ppg_lost", 0) - getattr(g, "away_injury_ppg_lost", 0)
             for g in games], dtype=np.float64
        )
        self.injury_minutes_lost_diff = np.array(
            [getattr(g, "home_injury_minutes_lost", 0) - getattr(g, "away_injury_minutes_lost", 0)
             for g in games], dtype=np.float64
        )

    def subset(self, mask: np.ndarray) -> "VectorizedGames":
        """Return a new VectorizedGames with only games where mask is True."""
        sub = object.__new__(VectorizedGames)
        sub.n = int(mask.sum())
        for attr in self.__slots__:
            if attr == "n":
                continue
            val = getattr(self, attr)
            setattr(sub, attr, val[mask])
        return sub


def _vectorized_loss(v: VectorizedGames, w: WeightConfig) -> float:
    """Evaluate weights against all games at once using NumPy arrays.

    Equivalent to calling ``predict_from_precomputed`` per game and
    computing the loss, but ~50-100x faster because there's no Python
    loop — everything is array arithmetic.
    """
    if v.n == 0:
        return 999.0

    # Defensive factor dampening
    adf = 1.0 + (v.away_def_factor_raw - 1.0) * w.def_factor_dampening
    hdf = 1.0 + (v.home_def_factor_raw - 1.0) * w.def_factor_dampening

    home_base = v.home_pts_raw * adf + v.home_tuning_corr
    away_base = v.away_pts_raw * hdf + v.away_tuning_corr

    fatigue_adj = v.home_fatigue - v.away_fatigue

    # ── Spread ──
    spread = (home_base - away_base) + v.home_court - fatigue_adj
    spread += v.to_diff * w.turnover_margin_mult
    spread += v.reb_diff * w.rebound_diff_mult

    # Rating matchup
    home_edge = v.home_off - v.away_def
    away_edge = v.away_off - v.home_def
    spread += (home_edge - away_edge) * w.rating_matchup_mult

    # Four Factors
    ff_adj = (
        v.ff_efg_edge * w.ff_efg_weight +
        v.ff_tov_edge * w.ff_tov_weight +
        v.ff_oreb_edge * w.ff_oreb_weight +
        v.ff_fta_edge * w.ff_fta_weight
    ) * w.four_factors_scale
    spread += ff_adj

    # Clutch (only when |spread| < threshold)
    clutch_adj = v.clutch_diff * w.clutch_scale
    clutch_adj = np.clip(clutch_adj, -w.clutch_cap, w.clutch_cap)
    clutch_mask = np.abs(spread) < w.clutch_threshold
    spread += np.where(clutch_mask, clutch_adj, 0.0)

    # Hustle spread
    spread += v.hustle_effort_diff * w.hustle_effort_mult

    # Clamp spread
    spread = np.clip(spread, -w.spread_clamp, w.spread_clamp)

    # ── Total ──
    total = home_base + away_base

    # Pace
    pace_factor = (v.avg_pace - w.pace_baseline) / w.pace_baseline
    total *= (1.0 + pace_factor * w.pace_mult)

    # Defensive disruption
    total -= (
        np.maximum(0.0, v.combined_steals - w.steals_threshold) * w.steals_penalty +
        np.maximum(0.0, v.combined_blocks - w.blocks_threshold) * w.blocks_penalty
    )

    # OREB
    total += (v.combined_oreb - w.oreb_baseline) * w.oreb_mult

    # Hustle total
    defl_excess = np.maximum(0.0, v.combined_deflections - w.hustle_defl_baseline)
    total -= defl_excess * w.hustle_defl_penalty

    # Fatigue total
    total -= v.combined_fatigue * w.fatigue_total_mult

    # Clamp total
    total = np.clip(total, w.total_min, w.total_max)

    # ── Loss ──
    spread_ae = np.abs(spread - v.actual_spread)
    total_ae = np.abs(total - v.actual_total)

    spread_mae = spread_ae.mean()
    total_mae = total_ae.mean()

    # Winner accuracy
    pred_home = spread > 0.5
    pred_away = spread < -0.5
    actual_home = v.actual_spread > 0
    actual_close = np.abs(v.actual_spread) <= 3
    push = ~pred_home & ~pred_away

    correct = (
        (pred_home & actual_home) |
        (pred_away & ~actual_home & (v.actual_spread != 0)) |
        (push & actual_close)
    )
    winner_pct = correct.sum() / v.n * 100

    return float(spread_mae * 1.0 + total_mae * 0.3 + (100 - winner_pct) * 0.1)


# Module-level cache for vectorized data
_vec_cache: Optional[VectorizedGames] = None
_vec_games_id: Optional[int] = None  # id() of the source list


def _get_vectorized(games: List[PrecomputedGame]) -> VectorizedGames:
    """Get or build vectorized arrays from a games list (cached by identity)."""
    global _vec_cache, _vec_games_id
    games_id = id(games)
    if _vec_cache is not None and _vec_games_id == games_id and _vec_cache.n == len(games):
        return _vec_cache
    _vec_cache = VectorizedGames(games)
    _vec_games_id = games_id
    return _vec_cache


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
    zero DB I/O.  Uses vectorized NumPy arrays for ~50-100x speed."""
    if not games:
        return 999.0
    v = _get_vectorized(games)
    return _vectorized_loss(v, w)


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
    # ML ensemble blending
    ("ml_ensemble_weight",     0.0,   0.6),
    ("ml_disagree_damp",       0.3,   1.0),
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

    # ── Baseline: use current best weights (from DB or defaults) ──
    baseline_cfg = get_weight_config(force_reload=True)
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

    # Pre-vectorize all game data once — reused across every trial
    vec = VectorizedGames(games) if use_fast else None

    def objective(trial: "optuna.Trial") -> float:
        d = baseline_cfg.to_dict()
        for name, lo, hi in TOP_WEIGHTS:
            d[name] = trial.suggest_float(name, lo, hi)
        d["espn_weight"] = round(1.0 - d["espn_model_weight"], 4)
        cfg = WeightConfig.from_dict(d)

        if vec is not None:
            return _vectorized_loss(vec, cfg)

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
        sampler=TPESampler(),  # random seed each run for fresh exploration
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_callback])
    elapsed = time.time() - t0

    best_loss = study.best_value
    improvement = ((baseline_loss - best_loss) / baseline_loss * 100) if baseline_loss else 0

    # Build best config from this run
    d = baseline_cfg.to_dict()
    d.update(study.best_params)
    d["espn_weight"] = round(1.0 - d.get("espn_model_weight", 0.8), 4)
    best_cfg = WeightConfig.from_dict(d)

    # Only persist if this run actually improved on the existing best
    if best_loss < baseline_loss:
        save_weights(best_cfg)
        set_weight_config(best_cfg)
        progress(
            f"Optuna optimisation complete: {n_trials} trials in {elapsed:.0f}s, "
            f"loss {baseline_loss:.2f} → {best_loss:.2f} ({improvement:+.1f}%) — SAVED"
        )
    else:
        progress(
            f"Optuna optimisation complete: {n_trials} trials in {elapsed:.0f}s, "
            f"best found {best_loss:.2f} did not beat existing {baseline_loss:.2f} — kept previous weights"
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

    # Pre-vectorize once
    vec = VectorizedGames(games) if use_fast else None

    t0 = time.time()
    for trial in range(1, n_trials + 1):
        candidate = _random_config(baseline_cfg)

        if vec is not None:
            trial_loss = _vectorized_loss(vec, candidate)
        elif use_fast:
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

    if best_loss < baseline_loss:
        save_weights(best_cfg)
        set_weight_config(best_cfg)
        progress(
            f"Random-search complete: {n_trials} trials in {elapsed:.0f}s, "
            f"loss {baseline_loss:.2f} → {best_loss:.2f} ({improvement:+.1f}%) — SAVED"
        )
    else:
        progress(
            f"Random-search complete: {n_trials} trials in {elapsed:.0f}s, "
            f"best found {best_loss:.2f} did not beat existing {baseline_loss:.2f} — kept previous weights"
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

        # Pre-vectorize this team's train/holdout sets for fast eval
        v_train = VectorizedGames(train)
        v_holdout = VectorizedGames(holdout)

        # Global loss on both sets
        global_loss_train = _vectorized_loss(v_train, global_cfg)
        global_loss_holdout = _vectorized_loss(v_holdout, global_cfg)

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

            trial_loss = _vectorized_loss(v_train, candidate)
            if trial_loss < best_team_loss_train:
                best_team_loss_train = trial_loss
                best_team_cfg = candidate

        # ── Regressive validation on holdout ──
        team_loss_holdout = _vectorized_loss(v_holdout, best_team_cfg)

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
    """Evaluate loss for games involving a specific team (vectorized)."""
    if not games:
        return 999.0

    v = _get_vectorized(games)
    mask = (v.home_team_ids == team_id) | (v.away_team_ids == team_id)
    n = int(mask.sum())

    if n == 0:
        return 999.0

    sub = v.subset(mask)
    return _vectorized_loss(sub, w)


# -----------------------------------------------------------------------
# Residual calibration
# -----------------------------------------------------------------------

_CALIBRATION_TABLE = "residual_calibration"
_CALIBRATION_TOTAL_TABLE = "residual_calibration_total"


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
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_CALIBRATION_TOTAL_TABLE} (
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
    results = run_backtest(min_games_before=5, progress_cb=progress_cb)
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

    # Total calibration bins (predicted total ~195-248)
    total_bins = [
        ("total_low",    195.0, 210.0),
        ("total_mid_low", 210.0, 220.0),
        ("total_mid",     220.0, 230.0),
        ("total_mid_high", 230.0, 240.0),
        ("total_high",    240.0, 260.0),
    ]
    total_calibration: Dict[str, Dict] = {}
    with get_conn() as conn:
        conn.execute(f"DELETE FROM {_CALIBRATION_TOTAL_TABLE}")
        for label, lo, hi in total_bins:
            preds_in_bin = [
                p for p in results.predictions
                if lo <= p.predicted_total < hi
            ]
            if not preds_in_bin:
                continue
            avg_residual = sum(p.total_error for p in preds_in_bin) / len(preds_in_bin)
            total_calibration[label] = {
                "bin_low": lo,
                "bin_high": hi,
                "avg_residual": round(avg_residual, 3),
                "sample_count": len(preds_in_bin),
            }
            conn.execute(
                f"INSERT INTO {_CALIBRATION_TOTAL_TABLE} (bin_label, bin_low, bin_high, avg_residual, sample_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (label, lo, hi, avg_residual, len(preds_in_bin)),
            )
            progress(
                f"  {label} total [{lo:.0f}, {hi:.0f}): "
                f"n={len(preds_in_bin)}, avg_residual={avg_residual:+.2f}"
            )
        conn.commit()

    progress(f"Calibration complete: {len(calibration)} spread bins, {len(total_calibration)} total bins")
    # Invalidate prediction module's calibration cache so it picks up new bins
    try:
        from src.analytics.prediction import reload_calibration_cache
        reload_calibration_cache()
    except Exception:
        pass
    return {"spread": calibration, "total": total_calibration}


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
    base_results = run_backtest(min_games_before=5, progress_cb=progress_cb)
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
# GROUPED feature importance (interaction effects)
# -----------------------------------------------------------------------

@dataclass
class GroupedFeatureImportance:
    group_name: str
    features_disabled: List[str]
    baseline_loss: float
    disabled_loss: float
    impact: float          # disabled_loss - baseline (positive = group helps)
    impact_pct: float


def run_grouped_feature_importance(
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[GroupedFeatureImportance]:
    """Measure *groups* of related features by disabling them together.

    Single-feature disruption can miss interaction effects: e.g. removing
    "four_factors" alone barely matters because "turnover_margin" and
    "rebound_diff" overlap.  Remove all three and the impact is larger.

    Returns a list sorted by group impact (most impactful group first).
    """
    progress = progress_cb or (lambda _: None)

    # ---- baseline ----
    progress("Running baseline backtest (grouped importance)...")
    base_cfg = get_weight_config(force_reload=True)
    set_weight_config(base_cfg)
    base_results = run_backtest(min_games_before=5, progress_cb=progress_cb)
    base_loss = _loss(base_results)
    progress(f"Baseline loss: {base_loss:.2f}")

    # ---- feature groups ----
    # Each group: (group_name, [(weight_key, disabled_value), ...])
    groups = [
        ("Efficiency & Four Factors", [
            ("four_factors_scale", 0.0),
            ("turnover_margin_mult", 0.0),
            ("rebound_diff_mult", 0.0),
        ]),
        ("Defensive Adjustments", [
            ("def_factor_dampening", 1.0),  # 1.0 = no dampening
            ("rating_matchup_mult", 0.0),
        ]),
        ("Pace & Volume", [
            ("pace_mult", 0.0),
            ("oreb_mult", 0.0),
        ]),
        ("Defensive Disruption", [
            ("steals_penalty", 0.0),
            ("blocks_penalty", 0.0),
            ("hustle_defl_penalty", 0.0),
        ]),
        ("Hustle & Effort", [
            ("hustle_effort_mult", 0.0),
            ("hustle_defl_penalty", 0.0),
        ]),
        ("Close Game / Clutch", [
            ("clutch_scale", 0.0),
            ("clutch_cap", 0.0),
        ]),
        ("Fatigue", [
            ("fatigue_total_mult", 0.0),
            ("fatigue_b2b", 0.0),
            ("fatigue_3in4", 0.0),
            ("fatigue_4in6", 0.0),
        ]),
        ("ESPN Blending", [
            ("espn_weight", 0.0),
            ("espn_model_weight", 1.0),
        ]),
        ("All Spread Adjustments", [
            ("turnover_margin_mult", 0.0),
            ("rebound_diff_mult", 0.0),
            ("rating_matchup_mult", 0.0),
            ("four_factors_scale", 0.0),
            ("clutch_scale", 0.0),
            ("hustle_effort_mult", 0.0),
        ]),
        ("All Total Adjustments", [
            ("pace_mult", 0.0),
            ("steals_penalty", 0.0),
            ("blocks_penalty", 0.0),
            ("oreb_mult", 0.0),
            ("hustle_defl_penalty", 0.0),
            ("fatigue_total_mult", 0.0),
        ]),
    ]

    results: List[GroupedFeatureImportance] = []

    for i, (group_name, overrides) in enumerate(groups, 1):
        keys_display = [k for k, _ in overrides]
        progress(f"  Testing group [{i}/{len(groups)}] {group_name} ({len(overrides)} weights)...")

        d = base_cfg.to_dict()
        for key, disabled_val in overrides:
            d[key] = disabled_val
        test_cfg = WeightConfig.from_dict(d)
        set_weight_config(test_cfg)

        try:
            test_results = run_backtest(min_games_before=5)
            test_loss = _loss(test_results)
        except Exception:
            test_loss = base_loss

        impact = test_loss - base_loss
        impact_pct = (impact / base_loss * 100) if base_loss else 0.0

        results.append(GroupedFeatureImportance(
            group_name=group_name,
            features_disabled=keys_display,
            baseline_loss=base_loss,
            disabled_loss=test_loss,
            impact=round(impact, 3),
            impact_pct=round(impact_pct, 2),
        ))

        direction = "HELPS" if impact > 0.1 else ("HURTS" if impact < -0.1 else "neutral")
        progress(
            f"    {group_name}: loss {base_loss:.2f} → {test_loss:.2f} "
            f"(impact: {impact:+.3f}, {direction})"
        )

    # Restore original config
    set_weight_config(base_cfg)

    results.sort(key=lambda x: x.impact, reverse=True)
    progress(f"Grouped feature importance complete: {len(results)} groups analysed")
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
    results = run_backtest(min_games_before=5, progress_cb=progress_cb)

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


# -----------------------------------------------------------------------
# Continuous optimiser — loops until cancelled, keeps only improvements
# -----------------------------------------------------------------------

@dataclass
class ContinuousOptResult:
    """Summary of a continuous optimisation session."""
    rounds_completed: int = 0
    global_improvements: int = 0
    best_global_loss: float = 999.0
    starting_loss: float = 999.0
    total_seconds: float = 0.0
    teams_refined: int = 0
    total_teams: int = 0


def run_continuous_optimiser(
    n_trials: int = 200,
    team_trials: int = 100,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ContinuousOptResult:
    """Loop global + per-team optimisation until cancelled.

    Each round:
    1. Run global weight optimisation (random seed, only saves if better)
    2. Run per-team refinement
    3. Report progress, check cancel, repeat

    The ``cancel_check`` callable should return True to stop the loop.
    Weights are only persisted when they beat the previous best, so
    stopping at any point is safe.
    """
    import time as _time
    t0 = _time.perf_counter()
    progress = progress_cb or (lambda _: None)
    cancelled = cancel_check or (lambda: False)

    result = ContinuousOptResult()

    # Precompute game data once — shared across all rounds
    progress("[Continuous] Precomputing game data (one-time)...")
    games = precompute_game_data(progress_cb=progress)
    if not games:
        progress("No games found — aborting")
        return result

    starting_cfg = get_weight_config(force_reload=True)
    starting_loss = _fast_loss(games, starting_cfg)
    result.starting_loss = starting_loss
    result.best_global_loss = starting_loss
    progress(f"[Continuous] Starting loss: {starting_loss:.2f} ({len(games)} games)")

    round_num = 0
    while not cancelled():
        round_num += 1
        round_t0 = _time.perf_counter()
        progress(f"\n{'='*50}")
        progress(f"[Round {round_num}] Starting global optimisation ({n_trials} trials)...")

        # ── Global optimisation ──
        try:
            global_result = run_weight_optimiser(
                n_trials=n_trials,
                progress_cb=progress,
                precomputed_games=games,
            )
            if global_result.best_loss < result.best_global_loss:
                result.best_global_loss = global_result.best_loss
                result.global_improvements += 1
        except Exception as exc:
            progress(f"  Global optimisation error: {exc}")

        if cancelled():
            break

        # ── Per-team refinement ──
        progress(f"[Round {round_num}] Per-team refinement ({team_trials} trials/team)...")
        try:
            team_results = run_per_team_refinement(
                n_trials=team_trials,
                progress_cb=progress,
                precomputed_games=games,
            )
            adopted = sum(1 for r in team_results if r.used_team_weights)
            result.teams_refined = adopted
            result.total_teams = len(team_results)
        except Exception as exc:
            progress(f"  Per-team refinement error: {exc}")

        result.rounds_completed = round_num
        round_elapsed = _time.perf_counter() - round_t0

        current_loss = _fast_loss(games, get_weight_config(force_reload=True))
        improvement = ((starting_loss - current_loss) / starting_loss * 100) if starting_loss else 0

        progress(
            f"[Round {round_num}] Done in {round_elapsed:.0f}s — "
            f"current loss: {current_loss:.2f} "
            f"(started at {starting_loss:.2f}, {improvement:+.1f}% overall)"
        )

    result.total_seconds = _time.perf_counter() - t0
    final_loss = _fast_loss(games, get_weight_config(force_reload=True))
    total_improvement = ((starting_loss - final_loss) / starting_loss * 100) if starting_loss else 0

    progress(f"\n{'='*50}")
    progress(
        f"[Continuous] Stopped after {result.rounds_completed} rounds in "
        f"{result.total_seconds:.0f}s"
    )
    progress(
        f"[Continuous] Loss: {starting_loss:.2f} → {final_loss:.2f} "
        f"({total_improvement:+.1f}% total improvement)"
    )
    progress(
        f"[Continuous] {result.global_improvements} global improvements saved, "
        f"{result.teams_refined}/{result.total_teams} teams refined"
    )

    return result
