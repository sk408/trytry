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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _save_pipeline_state(state: Dict):
    os.makedirs(os.path.dirname(PIPELINE_STATE_PATH), exist_ok=True)
    with open(PIPELINE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, cls=NumpyEncoder)


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

        # Reload memory after autotune writes new team corrections to DB,
        # so precomputed games below reflect the updated tuning values.
        if results.get("autotune"):
            store.reload()
            emit("  Memory reloaded with autotune corrections")

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
            # Stash for callers (e.g. run_overnight) to avoid re-precomputing
            results["_precomputed_cache"] = _precomputed_cache

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
        emit("[Step 10/13] Optimizing weights (3000 Optuna trials)...")
        if _needs_10:
            from src.analytics.weight_optimizer import optimize_weights
            if _precomputed_cache and len(_precomputed_cache) >= 20:
                opt_result = optimize_weights(
                    _precomputed_cache, n_trials=3000,
                    callback=lambda msg: emit(f"  {msg}"),
                    is_cancelled=is_cancelled
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
                    _precomputed_cache, n_trials=500,
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
            if k in ("sync", "_precomputed_cache"):
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


def run_overnight(max_hours: float = 8.0,
                  reset_weights: bool = False,
                  callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Run full pipeline once, then loop optimization steps until time runs out.

    Pass 1: full 13-step pipeline (data sync, autotune, ML, optimize, backtest)
    Pass 2+: optimize weights (3000) → per-team refine (500) → residual cal → backtest
    Each pass uses fresh random seeds. Precomputed game data is reused across passes.
    """
    clear_cancel()
    overall_start = time.time()
    deadline = overall_start + max_hours * 3600

    def emit(msg: str):
        if callback:
            callback(msg)
        logger.info(msg)

    def time_left():
        return max(0, deadline - time.time())

    def fmt_elapsed(secs):
        h, m = int(secs // 3600), int((secs % 3600) // 60)
        return f"{h}h {m}m" if h else f"{m}m {int(secs % 60)}s"

    emit(f"=== Overnight Optimization: {max_hours}h budget ===")

    # Optional weight reset
    if reset_weights:
        from src.analytics.weight_config import clear_all_weights
        clear_all_weights()
        emit("Weights reset to defaults.")

    # ── Pass 1: Full pipeline ──
    emit(f"\n--- Pass 1: Full Pipeline ---")
    pass1_start = time.time()
    results = run_full_pipeline(callback=callback)

    if results.get("error") or results.get("cancelled") or is_cancelled():
        return results

    pass1_elapsed = time.time() - pass1_start
    emit(f"Pass 1 complete in {fmt_elapsed(pass1_elapsed)} | {fmt_elapsed(time_left())} remaining")

    # Reuse precomputed cache from pass 1 if available, otherwise recompute
    precomputed = results.pop("_precomputed_cache", None)
    if not precomputed:
        emit("\nPrecomputing game data for optimization loops...")
        from src.analytics.prediction import precompute_game_data
        from src.analytics.memory_store import InMemoryDataStore
        store = InMemoryDataStore()
        store.reload()
        precomputed = precompute_game_data(callback=lambda msg: emit(f"  {msg}"))

    if not precomputed or len(precomputed) < 20:
        emit("Not enough game data for optimization loops. Stopping.")
        return results

    emit(f"Cached {len(precomputed)} games for reuse across passes\n")

    # Track best scores across all passes
    best_overall = results.get("backtest", {})
    pass_num = 1
    loop_times = []

    # ── Pass 2+: Optimization loops ──
    while time_left() > 0 and not is_cancelled():
        pass_num += 1
        loop_start = time.time()

        # Estimate if we have time for another full loop
        avg_loop = sum(loop_times) / len(loop_times) if loop_times else pass1_elapsed * 0.6
        if time_left() < avg_loop * 0.5:
            emit(f"\n~{fmt_elapsed(time_left())} remaining, not enough for another pass. Stopping.")
            break

        emit(f"\n--- Pass {pass_num}: Optimization Loop ({fmt_elapsed(time_left())} remaining) ---")

        try:
            # Step A: Global weight optimization
            emit(f"[Loop {pass_num}] Optimizing weights (3000 trials)...")
            from src.analytics.weight_optimizer import optimize_weights
            opt_result = optimize_weights(
                precomputed, n_trials=3000,
                callback=lambda msg: emit(f"  {msg}"),
                is_cancelled=is_cancelled
            )
            if is_cancelled():
                break

            improved_str = "IMPROVED" if opt_result.get("improved") else "no change"
            emit(f"  Global: {improved_str} (loss {opt_result.get('baseline_loss', 0):.3f} → {opt_result.get('best_loss', 0):.3f})")

            # Step B: Per-team refinement
            emit(f"[Loop {pass_num}] Per-team refinement (500 trials)...")
            from src.analytics.weight_optimizer import per_team_refinement
            refine_result = per_team_refinement(
                precomputed, n_trials=500,
                callback=lambda msg: emit(f"  {msg}")
            )
            if is_cancelled():
                break

            # Step C: Residual calibration
            emit(f"[Loop {pass_num}] Residual calibration...")
            from src.analytics.weight_optimizer import build_residual_calibration
            build_residual_calibration(
                precomputed,
                callback=lambda msg: emit(f"  {msg}")
            )
            if is_cancelled():
                break

            # Step D: Backtest to measure progress
            emit(f"[Loop {pass_num}] Validation backtest...")
            from src.analytics.backtester import run_backtest
            from src.analytics.weight_config import invalidate_weight_cache
            invalidate_weight_cache()
            bt = run_backtest(
                use_cache=False,
                callback=lambda msg: emit(f"  {msg}")
            )

            loop_elapsed = time.time() - loop_start
            loop_times.append(loop_elapsed)

            # Compare to best
            cur_loss = bt.get("spread_mae", 999) + bt.get("total_mae", 999) * 0.3
            best_loss = best_overall.get("spread_mae", 999) + best_overall.get("total_mae", 999) * 0.3
            if cur_loss < best_loss:
                best_overall = bt
                emit(f"  NEW BEST! MAE={bt.get('spread_mae', 0):.2f}, "
                     f"Win={bt.get('winner_pct', 0):.1f}%, ATS={bt.get('ats_rate', 0):.1f}%")
            else:
                emit(f"  No improvement (MAE={bt.get('spread_mae', 0):.2f} vs best {best_overall.get('spread_mae', 0):.2f})")

            emit(f"  Pass {pass_num} took {fmt_elapsed(loop_elapsed)} | "
                 f"avg {fmt_elapsed(sum(loop_times)/len(loop_times))}/pass")

        except Exception as e:
            logger.exception(f"Overnight loop {pass_num} error: {e}")
            emit(f"  Loop error: {e}")
            continue

    # ── Summary ──
    total_elapsed = time.time() - overall_start
    emit(f"\n{'='*60}")
    emit(f"Overnight complete: {pass_num} passes in {fmt_elapsed(total_elapsed)}")
    if best_overall:
        emit(f"Best result: MAE={best_overall.get('spread_mae', 0):.2f}, "
             f"Winner={best_overall.get('winner_pct', 0):.1f}%, "
             f"ATS={best_overall.get('ats_rate', 0):.1f}%, "
             f"ATS ROI={best_overall.get('ats_roi', 0):.1f}%")
    emit(f"{'='*60}")

    return {
        "passes": pass_num,
        "elapsed_seconds": round(total_elapsed, 1),
        "backtest": best_overall,
    }
