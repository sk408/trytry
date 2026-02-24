"""12-step pipeline orchestrator with smart caching."""

import json
import logging
import os
import time
import threading
from typing import Dict, Any, Optional, Callable

from src.database import db
from src.config import get_config

logger = logging.getLogger(__name__)

PIPELINE_STATE_PATH = os.path.join("data", "pipeline_state.json")
_cancel_event = threading.Event()


def request_cancel():
    """Request pipeline cancellation."""
    _cancel_event.set()


def clear_cancel():
    """Clear cancellation flag."""
    _cancel_event.clear()


def is_cancelled() -> bool:
    return _cancel_event.is_set()


def _load_pipeline_state() -> Dict:
    if os.path.exists(PIPELINE_STATE_PATH):
        try:
            with open(PIPELINE_STATE_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load pipeline state: %s", e)
    return {}


def _save_pipeline_state(state: Dict):
    os.makedirs(os.path.dirname(PIPELINE_STATE_PATH), exist_ok=True)
    with open(PIPELINE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _is_fresh(step_name: str, max_age_hours: float = 24) -> bool:
    """Check if a sync step is fresh enough to skip."""
    row = db.fetch_one(
        "SELECT last_synced_at FROM sync_meta WHERE step_name = ?",
        (step_name,)
    )
    if not row or not row["last_synced_at"]:
        return False

    from datetime import datetime, timedelta
    try:
        last = datetime.fromisoformat(row["last_synced_at"])
        return (datetime.now() - last).total_seconds() < max_age_hours * 3600
    except Exception as e:
        logger.warning("Freshness check parse error for %s: %s", step_name, e)
        return False


def _has_new_data(step_name: str) -> bool:
    """Check if there's new data since last time this step ran."""
    row = db.fetch_one(
        "SELECT game_count_at_sync, last_game_date_at_sync FROM sync_meta WHERE step_name = ?",
        (step_name,)
    )
    if not row:
        return True

    from src.analytics.memory_store import InMemoryDataStore
    store = InMemoryDataStore()
    current_count, current_last = store.get_game_count_and_last_date()

    old_count = row["game_count_at_sync"] or 0
    old_last = row["last_game_date_at_sync"] or ""

    return current_count != old_count or current_last != old_last


def _mark_step_done(step_name: str):
    """Mark a step as completed in sync_meta."""
    from src.analytics.memory_store import InMemoryDataStore
    store = InMemoryDataStore()
    count, last_date = store.get_game_count_and_last_date()

    db.execute("""
        INSERT INTO sync_meta (step_name, last_synced_at, game_count_at_sync, last_game_date_at_sync)
        VALUES (?, datetime('now'), ?, ?)
        ON CONFLICT(step_name) DO UPDATE SET
            last_synced_at = excluded.last_synced_at,
            game_count_at_sync = excluded.game_count_at_sync,
            last_game_date_at_sync = excluded.last_game_date_at_sync
    """, (step_name, count, last_date))


def run_full_pipeline(callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Execute the 12-step pipeline."""
    clear_cancel()
    start_time = time.time()
    results = {}

    def emit(msg: str):
        if callback:
            callback(msg)
        logger.info(msg)

    try:
        # Step 1: Backup State
        emit("[Step 1/13] Creating snapshot backup...")
        from src.analytics.snapshots import create_snapshot
        snapshot_path = create_snapshot("auto")
        if snapshot_path:
            emit(f"  Backup saved to {os.path.basename(snapshot_path)}")
        else:
            emit("  Warning: Failed to create backup snapshot")
            
        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 2: Check pipeline state
        emit("[Step 2/13] Checking pipeline state...")
        state = _load_pipeline_state()
        results["state"] = "loaded"

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 3: Load data into memory
        emit("[Step 3/13] Loading data into memory...")
        from src.analytics.memory_store import InMemoryDataStore
        store = InMemoryDataStore()
        store.reload()
        results["memory_loaded"] = True
        emit(f"  Loaded {len(store.player_stats)} player stats, {len(store.teams)} teams")

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 4: Full data sync (6 sub-steps)
        emit("[Step 4/13] Running data sync...")
        from src.data.sync_service import full_sync
        sync_result = full_sync(callback=lambda msg: emit(f"  [Sync] {msg}"))
        results["sync"] = sync_result

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 5: Reload memory if new data
        emit("[Step 5/13] Checking for new data...")
        if _has_new_data("pipeline_reload"):
            store.reload()
            _mark_step_done("pipeline_reload")
            emit("  Memory store reloaded with new data")
        else:
            emit("  No new data, skipping reload")

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 6: Build injury history
        emit("[Step 6/13] Building injury history...")
        if not _is_fresh("injury_history", 168) or _has_new_data("injury_history"):
            from src.analytics.injury_history import infer_injuries_from_logs
            ih_result = infer_injuries_from_logs(callback=lambda msg: emit(f"  {msg}"))
            results["injury_history"] = ih_result
            _mark_step_done("injury_history")
        else:
            emit("  Injury history is fresh, skipping")

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 7: Injury intelligence backfill + roster change
        emit("[Step 7/13] Injury intelligence backfill...")
        from src.analytics.injury_intelligence import backfill_play_outcomes
        backfill_count = backfill_play_outcomes()
        results["backfill"] = backfill_count

        from src.analytics.stats_engine import detect_roster_change
        teams_list = db.fetch_all("SELECT team_id FROM teams")
        roster_changes = 0
        for t in teams_list:
            change = detect_roster_change(t["team_id"])
            if change.get("changed"):
                roster_changes += 1
        results["roster_changes"] = roster_changes
        emit(f"  Backfilled {backfill_count} outcomes, {roster_changes} roster changes")

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 8: Autotune
        emit("[Step 8/13] Autotuning teams...")
        if not _is_fresh("autotune", 168) or _has_new_data("autotune"):
            from src.analytics.autotune import autotune_all
            at_result = autotune_all(
                strength=0.75, mode="classic",
                callback=lambda msg: emit(f"  {msg}")
            )
            results["autotune"] = {"teams_tuned": at_result.get("teams_tuned", 0)}
            _mark_step_done("autotune")
        else:
            emit("  Autotune is fresh, skipping")

        if is_cancelled():
            return {"cancelled": True, **results}

        # --- Pre-compute game data once for steps 9-12 ---
        _needs_9 = not _is_fresh("ml_train", 168) or _has_new_data("ml_train")
        _needs_10 = not _is_fresh("weight_optimize", 168) or _has_new_data("weight_optimize")
        _needs_11 = not _is_fresh("team_refine", 168) or _has_new_data("team_refine")
        _needs_12 = not _is_fresh("residual_cal", 168) or _has_new_data("residual_cal")

        _precomputed_cache = None
        if _needs_9 or _needs_10 or _needs_11 or _needs_12:
            emit("  Precomputing game data (shared by steps 9-12)...")
            from src.analytics.prediction import precompute_game_data
            _precomputed_cache = precompute_game_data(
                callback=lambda msg: emit(f"  {msg}")
            )
            emit(f"  Precomputed {len(_precomputed_cache) if _precomputed_cache else 0} games (cached)")

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 9: Train ML models
        emit("[Step 9/13] Training ML models...")
        if _needs_9:
            from src.analytics.ml_model import train_models
            if _precomputed_cache and len(_precomputed_cache) >= 50:
                ml_result = train_models(
                    _precomputed_cache,
                    callback=lambda msg: emit(f"  {msg}")
                )
                results["ml_train"] = ml_result
            else:
                emit(f"  Not enough precomputed games for ML training ({len(_precomputed_cache) if _precomputed_cache else 0} < 50)")
            _mark_step_done("ml_train")
        else:
            emit("  ML models are fresh, skipping")

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 10: Global weight optimization
        emit("[Step 10/13] Optimizing weights (200 Optuna trials)...")
        if _needs_10:
            from src.analytics.weight_optimizer import optimize_weights
            if _precomputed_cache and len(_precomputed_cache) >= 20:
                opt_result = optimize_weights(
                    _precomputed_cache, n_trials=200,
                    callback=lambda msg: emit(f"  {msg}")
                )
                results["optimize"] = opt_result
            else:
                emit(f"  Not enough precomputed games for optimization")
            _mark_step_done("weight_optimize")
        else:
            emit("  Weights are fresh, skipping")

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 11: Per-team refinement
        emit("[Step 11/13] Per-team weight refinement...")
        if _needs_11:
            from src.analytics.weight_optimizer import per_team_refinement
            if _precomputed_cache and len(_precomputed_cache) >= 20:
                refine_result = per_team_refinement(
                    _precomputed_cache, n_trials=100,
                    callback=lambda msg: emit(f"  {msg}")
                )
                results["refinement"] = refine_result
            _mark_step_done("team_refine")
        else:
            emit("  Team refinement is fresh, skipping")

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 12: Residual calibration
        emit("[Step 12/13] Building residual calibration...")
        if _needs_12:
            from src.analytics.weight_optimizer import build_residual_calibration
            if _precomputed_cache:
                cal_result = build_residual_calibration(
                    _precomputed_cache,
                    callback=lambda msg: emit(f"  {msg}")
                )
                results["calibration"] = cal_result
            _mark_step_done("residual_cal")
        else:
            emit("  Residual calibration is fresh, skipping")

        if is_cancelled():
            return {"cancelled": True, **results}

        # Step 13: Validation backtest
        emit("[Step 13/13] Running validation backtest...")
        from src.analytics.backtester import run_backtest
        bt_result = run_backtest(
            use_cache=False,
            callback=lambda msg: emit(f"  {msg}")
        )
        # Store full backtest result so the UI worker can emit it to accuracy cards
        results["backtest"] = bt_result

        # Save state (exclude bulky per_game list from the state file)
        elapsed = time.time() - start_time
        state["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["elapsed_seconds"] = round(elapsed, 1)
        _state_summary = {}
        for k, v in results.items():
            if k == "sync":
                continue
            if k == "backtest" and isinstance(v, dict):
                # Keep summary only — strip per_game and per_team detail
                _state_summary[k] = {
                    sk: sv for sk, sv in v.items()
                    if sk not in ("per_game", "per_team")
                }
            else:
                _state_summary[k] = v
        state["results_summary"] = _state_summary
        _save_pipeline_state(state)

        emit(f"\nPipeline complete in {elapsed:.0f}s")
        results["elapsed_seconds"] = round(elapsed, 1)
        return results

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        emit(f"Pipeline error: {e}")
        results["error"] = str(e)
        return results
