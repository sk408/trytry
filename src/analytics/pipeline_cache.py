"""Pickle cache with SHA-256 invalidation for precomputed data."""

import hashlib
import json
import logging
import pickle
import os
from pathlib import Path
from typing import Any, Optional

from src.database import db

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data") / "cache"


def _compute_state_hash() -> str:
    """SHA-256 hash of (model_weights, team_tuning, ml model meta)."""
    parts = []

    # model_weights
    rows = db.fetch_all("SELECT key, value FROM model_weights ORDER BY key")
    for r in rows:
        parts.append(f"{r['key']}={r['value']}")

    # team_tuning
    rows = db.fetch_all(
        "SELECT team_id, home_pts_correction, away_pts_correction FROM team_tuning ORDER BY team_id"
    )
    for r in rows:
        parts.append(f"t{r['team_id']}={r['home_pts_correction']},{r['away_pts_correction']}")

    # ML model meta
    meta_path = Path("data") / "ml_models" / "model_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                parts.append(f.read())
        except OSError:
            pass

    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def get_cache_path(prefix: str = "precomputed") -> Path:
    """Get the cache file path based on current state hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = _compute_state_hash()
    return CACHE_DIR / f"{prefix}_{h}.pkl"


def load_cached(prefix: str = "precomputed") -> Optional[Any]:
    """Load cached data if the state hash matches."""
    path = get_cache_path(prefix)
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
    return None


def save_cached(data: Any, prefix: str = "precomputed"):
    """Save data to cache with current state hash."""
    path = get_cache_path(prefix)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")


def invalidate_cache(prefix: str = "precomputed"):
    """Remove all cache files for a prefix."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob(f"{prefix}_*.pkl"):
            try:
                f.unlink()
            except OSError:
                pass
