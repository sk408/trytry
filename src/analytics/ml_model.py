"""ML Ensemble Model for NBA spread/total prediction.

Trains XGBoost regressors on a rich feature matrix extracted from
``PrecomputedGame`` objects.  The trained models are saved to disk
and loaded at prediction time to blend with the base (player-level)
model.

The key insight: the base model (player PPG x defensive factor + home
court) captures ~95% of the signal.  Linear adjustments on top add
only ±2-3 points and cannot find non-linear matchup interactions.
An XGBoost model can learn patterns like "fast-paced team vs slow
defensive team" or "elite 3PT shooting vs poor perimeter D" that
no amount of linear weight tuning will capture.

Usage::

    from src.analytics.ml_model import train_models, predict_ml

    # Train (once, after precomputation)
    result = train_models(precomputed_games, progress_cb=print)

    # Predict (at inference time)
    ml_spread, ml_total, confidence = predict_ml(features_dict)
"""
from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: graceful degradation if xgboost / shap not installed
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


# ====================================================================
#  Paths
# ====================================================================

_MODEL_DIR = Path("data/ml_models")
_SPREAD_MODEL_PATH = _MODEL_DIR / "spread_model.json"
_TOTAL_MODEL_PATH = _MODEL_DIR / "total_model.json"
_META_PATH = _MODEL_DIR / "model_meta.json"
_FEATURE_COLS_PATH = _MODEL_DIR / "feature_columns.json"


# ====================================================================
#  Feature extraction from PrecomputedGame
# ====================================================================

def _safe(val, default: float = 0.0) -> float:
    """Convert a possibly-None value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_features(g) -> Dict[str, float]:
    """Extract a flat feature dict from a ``PrecomputedGame``.

    Returns ~65 features covering raw projections, efficiency,
    ratings, matchup edges, four-factor components, clutch/hustle
    sub-metrics, fatigue, and key differentials.
    """
    hp = g.home_proj
    ap = g.away_proj
    f: Dict[str, float] = {}

    # ── Raw team projections ──
    for key in ("points", "rebounds", "assists", "steals", "blocks",
                "turnovers", "oreb", "dreb"):
        f[f"home_{key}"] = _safe(hp.get(key))
        f[f"away_{key}"] = _safe(ap.get(key))
        f[f"diff_{key}"] = f[f"home_{key}"] - f[f"away_{key}"]

    # ── Shooting efficiency ──
    for key in ("ts_pct", "fg3_rate", "ft_rate"):
        f[f"home_{key}"] = _safe(hp.get(key))
        f[f"away_{key}"] = _safe(ap.get(key))
        f[f"diff_{key}"] = f[f"home_{key}"] - f[f"away_{key}"]

    # ── Turnover margin ──
    f["home_to_margin"] = _safe(hp.get("turnover_margin"))
    f["away_to_margin"] = _safe(ap.get("turnover_margin"))
    f["diff_to_margin"] = f["home_to_margin"] - f["away_to_margin"]

    # ── Off/Def ratings ──
    f["home_off_rating"] = _safe(g.home_off)
    f["away_off_rating"] = _safe(g.away_off)
    f["home_def_rating"] = _safe(g.home_def)
    f["away_def_rating"] = _safe(g.away_def)
    f["home_net_rating"] = f["home_off_rating"] - f["home_def_rating"]
    f["away_net_rating"] = f["away_off_rating"] - f["away_def_rating"]
    f["diff_net_rating"] = f["home_net_rating"] - f["away_net_rating"]
    # Matchup edge: how does home offense fare vs away defense?
    f["home_matchup_edge"] = f["home_off_rating"] - f["away_def_rating"]
    f["away_matchup_edge"] = f["away_off_rating"] - f["home_def_rating"]
    f["diff_matchup_edge"] = f["home_matchup_edge"] - f["away_matchup_edge"]

    # ── Defensive factors (raw) ──
    f["home_def_factor_raw"] = _safe(g.home_def_factor_raw, 1.0)
    f["away_def_factor_raw"] = _safe(g.away_def_factor_raw, 1.0)

    # ── Pace ──
    f["home_pace"] = _safe(g.home_pace, 98.0)
    f["away_pace"] = _safe(g.away_pace, 98.0)
    f["avg_pace"] = (f["home_pace"] + f["away_pace"]) / 2
    f["diff_pace"] = f["home_pace"] - f["away_pace"]

    # ── Home court ──
    f["home_court"] = _safe(g.home_court, 3.0)

    # ── Fatigue ──
    f["home_fatigue"] = _safe(g.home_fatigue_penalty)
    f["away_fatigue"] = _safe(g.away_fatigue_penalty)
    f["diff_fatigue"] = f["home_fatigue"] - f["away_fatigue"]
    f["combined_fatigue"] = f["home_fatigue"] + f["away_fatigue"]

    # ── Four Factors components (individual edges) ──
    hff = g.home_ff or {}
    aff = g.away_ff or {}
    # eFG edge
    h_efg = _safe(hff.get("efg_pct"))
    a_efg = _safe(aff.get("efg_pct"))
    h_opp_efg = _safe(hff.get("opp_efg_pct"))
    a_opp_efg = _safe(aff.get("opp_efg_pct"))
    f["ff_efg_edge"] = (h_efg - a_opp_efg) - (a_efg - h_opp_efg)
    # TOV edge
    h_tov = _safe(hff.get("tm_tov_pct"))
    a_tov = _safe(aff.get("tm_tov_pct"))
    h_opp_tov = _safe(hff.get("opp_tm_tov_pct"))
    a_opp_tov = _safe(aff.get("opp_tm_tov_pct"))
    f["ff_tov_edge"] = (a_tov - h_opp_tov) - (h_tov - a_opp_tov)
    # OREB edge
    h_oreb_ff = _safe(hff.get("oreb_pct"))
    a_oreb_ff = _safe(aff.get("oreb_pct"))
    h_opp_oreb = _safe(hff.get("opp_oreb_pct"))
    a_opp_oreb = _safe(aff.get("opp_oreb_pct"))
    f["ff_oreb_edge"] = (h_oreb_ff - a_opp_oreb) - (a_oreb_ff - h_opp_oreb)
    # FTA rate edge
    h_fta = _safe(hff.get("fta_rate"))
    a_fta = _safe(aff.get("fta_rate"))
    h_opp_fta = _safe(hff.get("opp_fta_rate"))
    a_opp_fta = _safe(aff.get("opp_fta_rate"))
    f["ff_fta_edge"] = (h_fta - a_opp_fta) - (a_fta - h_opp_fta)

    # ── Clutch ──
    hc = g.home_clutch or {}
    ac = g.away_clutch or {}
    f["home_clutch_net"] = _safe(hc.get("clutch_net_rating"))
    f["away_clutch_net"] = _safe(ac.get("clutch_net_rating"))
    f["diff_clutch_net"] = f["home_clutch_net"] - f["away_clutch_net"]
    f["home_clutch_efg"] = _safe(hc.get("clutch_efg_pct"))
    f["away_clutch_efg"] = _safe(ac.get("clutch_efg_pct"))

    # ── Hustle ──
    hh = g.home_hustle or {}
    ah = g.away_hustle or {}
    f["home_deflections"] = _safe(hh.get("deflections"))
    f["away_deflections"] = _safe(ah.get("deflections"))
    f["diff_deflections"] = f["home_deflections"] - f["away_deflections"]
    f["home_contested"] = _safe(hh.get("contested_shots"))
    f["away_contested"] = _safe(ah.get("contested_shots"))
    f["home_loose_balls"] = _safe(hh.get("loose_balls_recovered"))
    f["away_loose_balls"] = _safe(ah.get("loose_balls_recovered"))

    return f


def build_training_data(
    games: list,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract feature matrix and target arrays from precomputed games.

    Returns:
        (X, y_spread, y_total) where X is a DataFrame, y_spread is
        actual home-away spread, y_total is actual combined score.
    """
    progress = progress_cb or (lambda _: None)
    rows: List[Dict[str, float]] = []
    spreads: List[float] = []
    totals: List[float] = []

    total_games = len(games)
    for i, g in enumerate(games):
        if i % 50 == 0:
            progress(f"Extracting features: {i + 1}/{total_games}...")
        feats = extract_features(g)
        rows.append(feats)
        spreads.append(g.actual_home_score - g.actual_away_score)
        totals.append(g.actual_home_score + g.actual_away_score)

    X = pd.DataFrame(rows).fillna(0.0)
    y_spread = np.array(spreads, dtype=np.float64)
    y_total = np.array(totals, dtype=np.float64)

    progress(f"Feature matrix: {X.shape[0]} games x {X.shape[1]} features")
    return X, y_spread, y_total


# ====================================================================
#  Training
# ====================================================================

@dataclass
class MLTrainingResult:
    """Results from training the ML ensemble models."""
    spread_train_mae: float = 0.0
    spread_val_mae: float = 0.0
    total_train_mae: float = 0.0
    total_val_mae: float = 0.0
    n_train: int = 0
    n_val: int = 0
    n_features: int = 0
    top_spread_features: List[Tuple[str, float]] = field(default_factory=list)
    top_total_features: List[Tuple[str, float]] = field(default_factory=list)
    shap_spread_features: List[Tuple[str, float]] = field(default_factory=list)
    shap_total_features: List[Tuple[str, float]] = field(default_factory=list)


def train_models(
    games: list,
    val_fraction: float = 0.2,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> MLTrainingResult:
    """Train XGBoost spread and total models on precomputed games.

    Uses a **time-based split**: the most recent *val_fraction* of games
    are held out for validation.  This prevents future data leakage.

    Saves models to ``data/ml_models/``.
    """
    if not _HAS_XGB:
        raise ImportError(
            "XGBoost is required for ML ensemble training. "
            "Install with: pip install xgboost"
        )

    progress = progress_cb or (lambda _: None)
    result = MLTrainingResult()

    # 1. Build feature matrix
    X, y_spread, y_total = build_training_data(games, progress_cb=progress)
    if len(X) < 30:
        progress(f"Not enough games ({len(X)}), need at least 30")
        return result

    # 2. Time-based split (games are pre-sorted by date)
    n_val = max(10, int(len(X) * val_fraction))
    n_train = len(X) - n_val
    X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
    y_spread_train, y_spread_val = y_spread[:n_train], y_spread[n_train:]
    y_total_train, y_total_val = y_total[:n_train], y_total[n_train:]

    progress(f"Split: {n_train} train / {n_val} validation (time-based)")
    result.n_train = n_train
    result.n_val = n_val
    result.n_features = X.shape[1]

    # 3. Save feature columns for inference
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = list(X.columns)
    with open(_FEATURE_COLS_PATH, "w") as fp:
        json.dump(feature_cols, fp)

    # 4. Train spread model
    progress("Training spread model...")
    spread_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    spread_model.fit(
        X_train, y_spread_train,
        eval_set=[(X_val, y_spread_val)],
        verbose=False,
    )
    spread_model.save_model(str(_SPREAD_MODEL_PATH))

    # Evaluate
    spread_train_pred = spread_model.predict(X_train)
    spread_val_pred = spread_model.predict(X_val)
    result.spread_train_mae = float(np.mean(np.abs(spread_train_pred - y_spread_train)))
    result.spread_val_mae = float(np.mean(np.abs(spread_val_pred - y_spread_val)))
    progress(
        f"  Spread MAE: train={result.spread_train_mae:.2f}, "
        f"val={result.spread_val_mae:.2f}"
    )

    # Feature importance (gain-based)
    importance = spread_model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    result.top_spread_features = [
        (feature_cols[int(k.replace("f", ""))] if k.startswith("f") else k, round(v, 2))
        for k, v in sorted_imp[:10]
    ]
    for name, gain in result.top_spread_features[:5]:
        progress(f"    {name}: gain={gain:.2f}")

    # 5. Train total model
    progress("Training total model...")
    total_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    total_model.fit(
        X_train, y_total_train,
        eval_set=[(X_val, y_total_val)],
        verbose=False,
    )
    total_model.save_model(str(_TOTAL_MODEL_PATH))

    total_train_pred = total_model.predict(X_train)
    total_val_pred = total_model.predict(X_val)
    result.total_train_mae = float(np.mean(np.abs(total_train_pred - y_total_train)))
    result.total_val_mae = float(np.mean(np.abs(total_val_pred - y_total_val)))
    progress(
        f"  Total MAE: train={result.total_train_mae:.2f}, "
        f"val={result.total_val_mae:.2f}"
    )

    importance = total_model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    result.top_total_features = [
        (feature_cols[int(k.replace("f", ""))] if k.startswith("f") else k, round(v, 2))
        for k, v in sorted_imp[:10]
    ]

    # 6. SHAP analysis (if available)
    if _HAS_SHAP:
        progress("Computing SHAP values for spread model...")
        try:
            explainer = _shap.TreeExplainer(spread_model)
            shap_vals = explainer.shap_values(X_val)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:10]
            result.shap_spread_features = [
                (feature_cols[i], round(float(mean_abs[i]), 4))
                for i in top_idx
            ]
            for name, imp in result.shap_spread_features[:5]:
                progress(f"    SHAP spread: {name} = {imp:.4f}")
        except Exception as exc:
            progress(f"    SHAP spread failed: {exc}")

        progress("Computing SHAP values for total model...")
        try:
            explainer = _shap.TreeExplainer(total_model)
            shap_vals = explainer.shap_values(X_val)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:10]
            result.shap_total_features = [
                (feature_cols[i], round(float(mean_abs[i]), 4))
                for i in top_idx
            ]
            for name, imp in result.shap_total_features[:5]:
                progress(f"    SHAP total: {name} = {imp:.4f}")
        except Exception as exc:
            progress(f"    SHAP total failed: {exc}")

    # 7. Save metadata
    meta = {
        "n_train": n_train,
        "n_val": n_val,
        "n_features": X.shape[1],
        "spread_train_mae": result.spread_train_mae,
        "spread_val_mae": result.spread_val_mae,
        "total_train_mae": result.total_train_mae,
        "total_val_mae": result.total_val_mae,
        "feature_columns": feature_cols,
        "trained_at": date.today().isoformat(),
    }
    with open(_META_PATH, "w") as fp:
        json.dump(meta, fp, indent=2)

    progress(
        f"ML models saved to {_MODEL_DIR}/ "
        f"(spread val MAE={result.spread_val_mae:.2f}, "
        f"total val MAE={result.total_val_mae:.2f})"
    )
    return result


# ====================================================================
#  Inference
# ====================================================================

# Module-level model cache (loaded once, reused)
_spread_model: Optional[xgb.XGBRegressor] = None
_total_model: Optional[xgb.XGBRegressor] = None
_feature_cols: Optional[List[str]] = None


def _ensure_models_loaded() -> bool:
    """Load models from disk if not already cached. Returns True if models are ready."""
    global _spread_model, _total_model, _feature_cols

    if _spread_model is not None and _total_model is not None:
        return True

    if not _HAS_XGB:
        return False

    if not _SPREAD_MODEL_PATH.exists() or not _TOTAL_MODEL_PATH.exists():
        return False

    if not _FEATURE_COLS_PATH.exists():
        return False

    try:
        _spread_model = xgb.XGBRegressor()
        _spread_model.load_model(str(_SPREAD_MODEL_PATH))

        _total_model = xgb.XGBRegressor()
        _total_model.load_model(str(_TOTAL_MODEL_PATH))

        with open(_FEATURE_COLS_PATH) as fp:
            _feature_cols = json.load(fp)

        return True
    except Exception:
        _spread_model = None
        _total_model = None
        _feature_cols = None
        return False


def reload_models() -> bool:
    """Force reload models from disk. Returns True if successful."""
    global _spread_model, _total_model, _feature_cols
    _spread_model = None
    _total_model = None
    _feature_cols = None
    return _ensure_models_loaded()


def is_ml_available() -> bool:
    """Check if ML models are trained and loadable."""
    return _ensure_models_loaded()


def predict_ml(features: Dict[str, float]) -> Tuple[float, float, float]:
    """Predict spread and total from a feature dict.

    Returns:
        (ml_spread, ml_total, confidence) where confidence is 0.0-1.0
        based on how many expected features are present.
    """
    if not _ensure_models_loaded():
        return 0.0, 0.0, 0.0

    # Build feature vector aligned to training columns
    row = {col: features.get(col, 0.0) for col in _feature_cols}
    X = pd.DataFrame([row])[_feature_cols]

    # Predict
    ml_spread = float(_spread_model.predict(X)[0])
    ml_total = float(_total_model.predict(X)[0])

    # Confidence: fraction of non-zero features present
    n_present = sum(1 for col in _feature_cols if features.get(col, 0.0) != 0.0)
    confidence = min(1.0, n_present / max(1, len(_feature_cols) * 0.6))

    return ml_spread, ml_total, confidence


def predict_ml_from_precomputed(g) -> Tuple[float, float, float]:
    """Convenience: extract features from PrecomputedGame and predict."""
    features = extract_features(g)
    return predict_ml(features)


def get_model_meta() -> Optional[Dict]:
    """Return model metadata (training stats, feature list) or None."""
    if not _META_PATH.exists():
        return None
    try:
        with open(_META_PATH) as fp:
            return json.load(fp)
    except Exception:
        return None
