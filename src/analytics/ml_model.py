"""XGBoost ML model: train, predict, SHAP, persistence."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data") / "ml_models"
SPREAD_MODEL_PATH = MODEL_DIR / "spread_model.json"
TOTAL_MODEL_PATH = MODEL_DIR / "total_model.json"
META_PATH = MODEL_DIR / "model_meta.json"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.json"

# Module-level cache
_spread_model = None
_total_model = None
_feature_cols = None
_loaded = False

# Hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
}


def _ensure_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_models_loaded():
    """Load models from disk once."""
    global _spread_model, _total_model, _feature_cols, _loaded
    if _loaded:
        return
    try:
        import xgboost as xgb
        if SPREAD_MODEL_PATH.exists() and TOTAL_MODEL_PATH.exists():
            _spread_model = xgb.XGBRegressor()
            _spread_model.load_model(str(SPREAD_MODEL_PATH))
            _total_model = xgb.XGBRegressor()
            _total_model.load_model(str(TOTAL_MODEL_PATH))
            if FEATURE_COLS_PATH.exists():
                with open(FEATURE_COLS_PATH, "r") as f:
                    _feature_cols = json.load(f)
            _loaded = True
            logger.info("ML models loaded successfully")
        else:
            logger.info("ML models not found, skipping load")
    except ImportError:
        logger.warning("xgboost not installed")
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")


def train_models(games: list, callback=None) -> Dict[str, Any]:
    """Train XGBoost spread and total models.

    Args:
        games: List of PrecomputedGame objects
        callback: Progress callback
    Returns:
        Training metadata
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed")
        return {"error": "xgboost not installed"}

    global _spread_model, _total_model, _feature_cols, _loaded

    if len(games) < 30:
        return {"error": f"Need at least 30 games, have {len(games)}"}

    if callback:
        callback("Extracting features from games...")

    # Extract features and targets
    feature_rows = []
    spread_targets = []
    total_targets = []

    for g in games:
        features = _extract_features_from_precomputed(g)
        if features is None:
            continue
        actual_spread = g.actual_home_score - g.actual_away_score
        actual_total = g.actual_home_score + g.actual_away_score
        feature_rows.append(features)
        spread_targets.append(actual_spread)
        total_targets.append(actual_total)

    if len(feature_rows) < 30:
        return {"error": f"Insufficient valid games: {len(feature_rows)}"}

    # Create DataFrame
    import pandas as pd
    X = pd.DataFrame(feature_rows)
    _feature_cols = list(X.columns)
    y_spread = np.array(spread_targets)
    y_total = np.array(total_targets)

    # Time-based split
    n_val = max(10, int(len(X) * 0.2))
    n_train = len(X) - n_val

    X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
    y_spread_train, y_spread_val = y_spread[:n_train], y_spread[n_train:]
    y_total_train, y_total_val = y_total[:n_train], y_total[n_train:]

    if callback:
        callback(f"Training spread model ({n_train} train, {n_val} val)...")

    # Train spread model
    _spread_model = xgb.XGBRegressor(
        **XGBOOST_PARAMS,
        early_stopping_rounds=20,
        random_state=42,
    )
    _spread_model.fit(
        X_train, y_spread_train,
        eval_set=[(X_val, y_spread_val)],
        verbose=False,
    )

    if callback:
        callback("Training total model...")

    # Train total model
    _total_model = xgb.XGBRegressor(
        **XGBOOST_PARAMS,
        early_stopping_rounds=20,
        random_state=42,
    )
    _total_model.fit(
        X_train, y_total_train,
        eval_set=[(X_val, y_total_val)],
        verbose=False,
    )

    # Evaluate
    spread_pred = _spread_model.predict(X_val)
    total_pred = _total_model.predict(X_val)
    spread_mae = float(np.mean(np.abs(spread_pred - y_spread_val)))
    total_mae = float(np.mean(np.abs(total_pred - y_total_val)))

    if callback:
        callback(f"Spread MAE: {spread_mae:.2f}, Total MAE: {total_mae:.2f}")

    # SHAP analysis
    shap_results = {}
    try:
        import shap
        if callback:
            callback("Computing SHAP values...")
        explainer = shap.TreeExplainer(_spread_model)
        shap_values = explainer.shap_values(X_val)
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_idx = np.argsort(-mean_abs)[:10]
        shap_results["spread_top_features"] = [
            {"feature": _feature_cols[i], "importance": float(mean_abs[i])}
            for i in top_idx
        ]

        explainer_t = shap.TreeExplainer(_total_model)
        shap_values_t = explainer_t.shap_values(X_val)
        mean_abs_t = np.mean(np.abs(shap_values_t), axis=0)
        top_idx_t = np.argsort(-mean_abs_t)[:10]
        shap_results["total_top_features"] = [
            {"feature": _feature_cols[i], "importance": float(mean_abs_t[i])}
            for i in top_idx_t
        ]
    except ImportError:
        logger.info("SHAP not available")
    except Exception as e:
        logger.warning(f"SHAP error: {e}")

    # Save models
    _ensure_dir()
    _spread_model.save_model(str(SPREAD_MODEL_PATH))
    _total_model.save_model(str(TOTAL_MODEL_PATH))

    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(_feature_cols, f)

    from datetime import datetime
    meta = {
        "n_train": n_train,
        "n_val": n_val,
        "n_features": len(_feature_cols),
        "spread_mae": spread_mae,
        "total_mae": total_mae,
        "feature_cols": _feature_cols,
        "trained_at": datetime.now().isoformat(),
        **shap_results,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    _loaded = True
    if callback:
        callback("ML training complete!")
    return meta


def predict_ml(features: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Predict spread and total from features.

    Returns dict with 'spread', 'total', 'confidence' or None.
    """
    _ensure_models_loaded()
    if _spread_model is None or _total_model is None or _feature_cols is None:
        return None

    import pandas as pd
    row = {col: features.get(col, 0.0) for col in _feature_cols}
    X = pd.DataFrame([row])

    spread = float(_spread_model.predict(X)[0])
    total = float(_total_model.predict(X)[0])

    n_present = sum(1 for col in _feature_cols if features.get(col, 0.0) != 0.0)
    confidence = n_present / (len(_feature_cols) * 0.6)
    confidence = min(1.0, confidence)

    return {"spread": spread, "total": total, "confidence": confidence}


def predict_ml_with_uncertainty(features: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Predict with uncertainty from per-tree variance."""
    _ensure_models_loaded()
    if _spread_model is None or _total_model is None or _feature_cols is None:
        return None

    import pandas as pd
    row = {col: features.get(col, 0.0) for col in _feature_cols}
    X = pd.DataFrame([row])

    result = predict_ml(features)
    if result is None:
        return None

    # Per-tree predictions for uncertainty
    try:
        import xgboost as xgb
        # Get leaf predictions from each tree
        spread_preds = []
        total_preds = []
        booster_s = _spread_model.get_booster()
        booster_t = _total_model.get_booster()
        dmat = xgb.DMatrix(X)

        # Use iteration range to get cumulative predictions
        n_trees = booster_s.num_boosted_rounds()
        if n_trees > 10:
            step = max(1, n_trees // 10)
            for i in range(step, n_trees + 1, step):
                sp = float(booster_s.predict(dmat, iteration_range=(0, i))[0])
                spread_preds.append(sp)
            for i in range(step, booster_t.num_boosted_rounds() + 1, step):
                tp = float(booster_t.predict(dmat, iteration_range=(0, i))[0])
                total_preds.append(tp)

            result["spread_std"] = float(np.std(spread_preds)) if spread_preds else 0.0
            result["total_std"] = float(np.std(total_preds)) if total_preds else 0.0
        else:
            result["spread_std"] = 0.0
            result["total_std"] = 0.0
    except Exception as e:
        logger.warning("ML uncertainty estimation failed: %s", e)
        result["spread_std"] = 0.0
        result["total_std"] = 0.0

    return result


def _extract_features_from_precomputed(g) -> Optional[Dict[str, float]]:
    """Extract ML features from a PrecomputedGame."""
    try:
        f = {}
        hp = g.home_proj
        ap = g.away_proj

        for stat in ["points", "rebounds", "assists", "steals", "blocks", "turnovers", "oreb", "dreb"]:
            hv = hp.get(stat, 0)
            av = ap.get(stat, 0)
            f[f"home_{stat}"] = hv
            f[f"away_{stat}"] = av
            f[f"diff_{stat}"] = hv - av

        # Shooting efficiency
        from src.analytics.stats_engine import compute_shooting_efficiency
        h_eff = compute_shooting_efficiency(hp)
        a_eff = compute_shooting_efficiency(ap)
        for stat in ["ts_pct", "fg3_rate", "ft_rate"]:
            f[f"home_{stat}"] = h_eff.get(stat, 0)
            f[f"away_{stat}"] = a_eff.get(stat, 0)
            f[f"diff_{stat}"] = h_eff.get(stat, 0) - a_eff.get(stat, 0)

        # TO margin
        f["home_to_margin"] = ap.get("turnovers", 0) - hp.get("turnovers", 0)
        f["away_to_margin"] = hp.get("turnovers", 0) - ap.get("turnovers", 0)
        f["diff_to_margin"] = f["home_to_margin"] - f["away_to_margin"]

        # Ratings
        f["home_off_rating"] = g.home_off
        f["away_off_rating"] = g.away_off
        f["home_def_rating"] = g.home_def
        f["away_def_rating"] = g.away_def
        f["home_net_rating"] = g.home_off - g.home_def
        f["away_net_rating"] = g.away_off - g.away_def
        f["diff_net_rating"] = f["home_net_rating"] - f["away_net_rating"]
        f["home_matchup_edge"] = g.home_off - g.away_def
        f["away_matchup_edge"] = g.away_off - g.home_def
        f["diff_matchup_edge"] = f["home_matchup_edge"] - f["away_matchup_edge"]
        f["home_def_factor_raw"] = g.home_def_factor_raw
        f["away_def_factor_raw"] = g.away_def_factor_raw

        # Pace
        f["home_pace"] = g.home_pace
        f["away_pace"] = g.away_pace
        f["avg_pace"] = (g.home_pace + g.away_pace) / 2
        f["diff_pace"] = g.home_pace - g.away_pace

        f["home_court"] = g.home_court

        # Fatigue
        f["home_fatigue"] = g.home_fatigue_penalty
        f["away_fatigue"] = g.away_fatigue_penalty
        f["diff_fatigue"] = g.home_fatigue_penalty - g.away_fatigue_penalty
        f["combined_fatigue"] = g.home_fatigue_penalty + g.away_fatigue_penalty

        # Four Factors
        f["ff_efg_edge"] = g.home_ff.get("efg_edge", 0)
        f["ff_tov_edge"] = g.home_ff.get("tov_edge", 0)
        f["ff_oreb_edge"] = g.home_ff.get("oreb_edge", 0)
        f["ff_fta_edge"] = g.home_ff.get("fta_edge", 0)

        # Clutch
        f["home_clutch_net"] = g.home_clutch.get("net_rating", 0)
        f["away_clutch_net"] = g.away_clutch.get("net_rating", 0)
        f["diff_clutch_net"] = f["home_clutch_net"] - f["away_clutch_net"]
        f["home_clutch_efg"] = g.home_clutch.get("efg_pct", 0)
        f["away_clutch_efg"] = g.away_clutch.get("efg_pct", 0)

        # Hustle
        f["home_deflections"] = g.home_hustle.get("deflections", 0)
        f["away_deflections"] = g.away_hustle.get("deflections", 0)
        f["diff_deflections"] = f["home_deflections"] - f["away_deflections"]
        f["home_contested"] = g.home_hustle.get("contested", 0)
        f["away_contested"] = g.away_hustle.get("contested", 0)
        f["home_loose_balls"] = g.home_hustle.get("loose_balls", 0)
        f["away_loose_balls"] = g.away_hustle.get("loose_balls", 0)

        # Injury
        f["home_injured_count"] = g.home_injured_count
        f["away_injured_count"] = g.away_injured_count
        f["diff_injured_count"] = g.home_injured_count - g.away_injured_count
        f["home_injury_ppg_lost"] = g.home_injury_ppg_lost
        f["away_injury_ppg_lost"] = g.away_injury_ppg_lost
        f["diff_injury_ppg_lost"] = g.home_injury_ppg_lost - g.away_injury_ppg_lost
        f["home_injury_minutes_lost"] = g.home_injury_minutes_lost
        f["away_injury_minutes_lost"] = g.away_injury_minutes_lost
        f["diff_injury_minutes_lost"] = g.home_injury_minutes_lost - g.away_injury_minutes_lost

        # Season phase
        f["home_games_played"] = g.home_games_played
        f["away_games_played"] = g.away_games_played
        f["min_games_played"] = min(g.home_games_played, g.away_games_played)
        f["games_played_diff"] = g.home_games_played - g.away_games_played

        # Roster change
        f["home_roster_changed"] = int(g.home_roster_changed)
        f["away_roster_changed"] = int(g.away_roster_changed)

        return f
    except Exception as e:
        logger.debug(f"Feature extraction failed: {e}")
        return None


def get_model_meta() -> Dict[str, Any]:
    """Load model metadata."""
    if META_PATH.exists():
        try:
            with open(META_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load model metadata: %s", e)
    return {}


def get_shap_importance() -> List[Dict[str, Any]]:
    """Compute SHAP feature importance for the spread model.

    Returns list of {feature, importance} dicts sorted by importance.
    """
    _ensure_models_loaded()
    if _spread_model is None:
        return []

    try:
        import shap
    except ImportError:
        # Fallback to XGBoost built-in feature importance
        logger.info("SHAP not installed, using XGBoost feature importance")
        return _get_xgb_importance()

    try:
        # Build a small synthetic sample for SHAP (use model's internal feature names)
        if _feature_cols is None:
            return _get_xgb_importance()

        # Create a zero-valued sample (SHAP explains deviation from mean prediction)
        sample = np.zeros((1, len(_feature_cols)))
        explainer = shap.TreeExplainer(_spread_model)
        shap_values = explainer.shap_values(sample)

        importance = []
        for i, col in enumerate(_feature_cols):
            importance.append({
                "feature": col,
                "importance": float(abs(shap_values[0][i])) if shap_values.ndim > 1 else 0.0,
            })
        importance.sort(key=lambda x: x["importance"], reverse=True)
        return importance
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return _get_xgb_importance()


def _get_xgb_importance() -> List[Dict[str, Any]]:
    """Fallback: XGBoost built-in feature importance."""
    if _spread_model is None or _feature_cols is None:
        return []
    try:
        scores = _spread_model.get_booster().get_score(importance_type="gain")
        result = []
        for col in _feature_cols:
            key = col
            # XGBoost may use f0, f1, ... or actual names
            imp = scores.get(key, 0.0)
            result.append({"feature": col, "importance": float(imp)})
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result
    except Exception as e:
        logger.warning(f"XGBoost importance failed: {e}")
        return []
