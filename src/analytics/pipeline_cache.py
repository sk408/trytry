"""Smart caching layer for the full optimisation pipeline.

Tracks when each pipeline step was last run, detects whether new game
data has arrived, and caches expensive precomputed game data to disk
so the optimizer can skip the DB-loading phase entirely.

Usage::

    state = load_pipeline_state()
    if has_new_games(state):
        # need to re-run data-dependent steps
        ...
    if is_step_fresh(state, "optimize"):
        # can skip optimisation
        ...
"""
from __future__ import annotations

import hashlib
import json
import pickle
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.database.db import get_conn

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_STATE_FILE = _DATA_DIR / "pipeline_state.json"
_PRECOMPUTED_PICKLE = _DATA_DIR / "backtest_cache" / "precomputed_games.pkl"
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    last_sync_at: str = ""
    last_game_date_in_db: str = ""
    game_count_at_last_run: int = 0
    last_optimize_at: str = ""
    last_team_refine_at: str = ""
    last_calibrate_at: str = ""
    calibrate_config_hash: str = ""  # hash of model_weights + team_tuning when calibrate ran
    last_autotune_at: str = ""
    last_backtest_at: str = ""
    last_injury_history_at: str = ""
    precomputed_games_hash: str = ""


def load_pipeline_state() -> PipelineState:
    """Load persisted pipeline state from disk."""
    with _lock:
        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
                return PipelineState(**{
                    k: v for k, v in data.items()
                    if k in PipelineState.__dataclass_fields__
                })
            except Exception:
                return PipelineState()
        return PipelineState()


def save_pipeline_state(state: PipelineState) -> None:
    """Persist pipeline state to disk."""
    with _lock:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(
            json.dumps(asdict(state), indent=2),
            encoding="utf-8",
        )


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _calibration_config_hash() -> str:
    """Hash of model_weights + team_tuning; changes when weights/tuning change."""
    payload: dict = {}
    try:
        with get_conn() as conn:
            mw = conn.execute(
                "SELECT key, value FROM model_weights ORDER BY key"
            ).fetchall()
            payload["model_weights"] = [[r[0], r[1]] for r in mw]
    except Exception:
        payload["model_weights"] = []
    try:
        with get_conn() as conn:
            tt = conn.execute(
                "SELECT team_id, home_pts_correction, away_pts_correction "
                "FROM team_tuning ORDER BY team_id"
            ).fetchall()
            payload["team_tuning"] = [list(r) for r in tt]
    except Exception:
        payload["team_tuning"] = []
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def mark_step_done(state: PipelineState, step_name: str) -> PipelineState:
    """Update the timestamp for a completed step and save."""
    mapping = {
        "sync": "last_sync_at",
        "injury_history": "last_injury_history_at",
        "autotune": "last_autotune_at",
        "optimize": "last_optimize_at",
        "team_refine": "last_team_refine_at",
        "calibrate": "last_calibrate_at",
        "backtest": "last_backtest_at",
    }
    attr = mapping.get(step_name)
    if attr:
        setattr(state, attr, _now_iso())
    if step_name == "calibrate":
        state.calibrate_config_hash = _calibration_config_hash()
    save_pipeline_state(state)
    return state


def update_game_snapshot(state: PipelineState) -> PipelineState:
    """Query the DB for the current game count and last date, update state."""
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT game_id), MAX(game_date) FROM player_stats"
            ).fetchone()
            state.game_count_at_last_run = row[0] if row and row[0] else 0
            state.last_game_date_in_db = (row[1] if row and row[1] else "") or ""
    except Exception:
        pass
    save_pipeline_state(state)
    return state


# ---------------------------------------------------------------------------
# Freshness queries
# ---------------------------------------------------------------------------

def has_new_games(state: PipelineState) -> bool:
    """Return True if the DB has more games or a newer game date than last run."""
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT game_id), MAX(game_date) FROM player_stats"
            ).fetchone()
            if not row or not row[0]:
                return False
            current_count = row[0]
            current_date = row[1] or ""
    except Exception:
        return True  # If we can't check, assume new data

    if current_count != state.game_count_at_last_run:
        return True
    if current_date != state.last_game_date_in_db:
        return True
    return False


def is_step_fresh(state: PipelineState, step_name: str) -> bool:
    """Return True if *step_name* was run AFTER the last data change.

    A step is "fresh" when:
    1. It has a recorded last-run timestamp, AND
    2. No new games have appeared since it was last run.
    3. For "calibrate": model_weights + team_tuning hash must match
       (calibration is specific to the current weight config).
    """
    mapping = {
        "sync": "last_sync_at",
        "injury_history": "last_injury_history_at",
        "autotune": "last_autotune_at",
        "optimize": "last_optimize_at",
        "team_refine": "last_team_refine_at",
        "calibrate": "last_calibrate_at",
        "backtest": "last_backtest_at",
    }
    attr = mapping.get(step_name)
    if not attr:
        return False
    last_run = getattr(state, attr, "")
    if not last_run:
        return False  # never ran
    # If there are new games, step is stale
    if has_new_games(state):
        return False
    # Calibration is tied to weight config: if weights/tuning changed, recalibrate
    if step_name == "calibrate" and getattr(state, "calibrate_config_hash", ""):
        if _calibration_config_hash() != state.calibrate_config_hash:
            return False
    return True


# ---------------------------------------------------------------------------
# Precomputed game data pickle cache
# ---------------------------------------------------------------------------

def save_precomputed_games(games: list) -> str:
    """Pickle precomputed games to disk. Returns the hash key."""
    _PRECOMPUTED_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(games)
    h = hashlib.sha256(data).hexdigest()[:16]
    _PRECOMPUTED_PICKLE.write_bytes(data)
    return h


def load_precomputed_games(expected_hash: str = "") -> list | None:
    """Load precomputed games from pickle if the file exists and hash matches.

    If *expected_hash* is empty, loads regardless of hash.
    Returns None if the file doesn't exist or hash mismatches.
    """
    if not _PRECOMPUTED_PICKLE.exists():
        return None
    try:
        data = _PRECOMPUTED_PICKLE.read_bytes()
        if expected_hash:
            actual_hash = hashlib.sha256(data).hexdigest()[:16]
            if actual_hash != expected_hash:
                return None
        return pickle.loads(data)
    except Exception:
        return None


def clear_precomputed_cache() -> None:
    """Remove the pickled precomputed games file."""
    try:
        if _PRECOMPUTED_PICKLE.exists():
            _PRECOMPUTED_PICKLE.unlink()
    except Exception:
        pass
