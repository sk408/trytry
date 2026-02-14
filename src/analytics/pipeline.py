"""Full optimisation pipeline orchestrator.

Chains every analytics step from data sync through validation backtest,
with smart caching to skip steps whose data hasn't changed.

Usage::

    from src.analytics.pipeline import run_full_pipeline
    summary = run_full_pipeline(
        n_trials=200,
        progress_cb=print,
        cancel_check=lambda: False,
    )
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional


def run_full_pipeline(
    n_trials: int = 200,
    team_trials: int = 100,
    autotune_strength: float = 0.75,
    autotune_mode: str = "classic",
    max_workers: int = 4,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    force_rerun: bool = False,
) -> Dict:
    """Run the full analytics pipeline end-to-end.

    Steps:
      1. Check pipeline state / detect new games
      2. Load data into memory store
      3. Full data sync (always runs — API cache makes it fast)
      4. Reload memory store if sync found new data
      5. Build injury history (skip if fresh)
      6. Injury intelligence backfill
      7. Autotune all teams (skip if fresh)
      8. Train ML ensemble models (skip if fresh)
      9. Global weight optimisation (skip if fresh)
     10. Per-team refinement (skip if fresh)
     11. Build residual calibration (skip if fresh)
     12. Validation backtest (always runs)

    Args:
        n_trials: Trials for global optimiser.
        team_trials: Trials per team for refinement.
        autotune_strength: Autotune strength parameter.
        max_workers: Backtest parallel workers.
        progress_cb: Progress callback.
        cancel_check: Returns True to abort between steps.
        force_rerun: Bypass all freshness checks.

    Returns:
        Summary dict with per-step timing and results.
    """
    # Lazy imports to avoid circular deps and keep startup fast
    from src.analytics.memory_store import get_memory_store
    from src.analytics.pipeline_cache import (
        load_pipeline_state,
        save_pipeline_state,
        has_new_games,
        is_step_fresh,
        mark_step_done,
        update_game_snapshot,
    )
    from src.analytics.prediction import precompute_game_data
    from src.analytics.weight_optimizer import (
        run_weight_optimiser,
        run_per_team_refinement,
        build_residual_calibration,
    )
    from src.analytics.weight_config import get_weight_config
    from src.analytics.backtester import run_backtest
    from src.database import migrations

    progress = progress_cb or (lambda _: None)
    cancelled = cancel_check or (lambda: False)
    summary: Dict = {"steps": {}, "total_seconds": 0.0}
    t_start = time.perf_counter()

    def _step_header(num: int, total: int, name: str) -> str:
        return f"[STEP {num}/{total}] {name}"

    def _skipped(num: int, total: int, name: str) -> None:
        msg = f"{_step_header(num, total, name)} -- SKIPPED (no new data since last run)"
        progress(msg)
        summary["steps"][name] = {"status": "skipped", "seconds": 0.0}

    def _check_cancel() -> bool:
        if cancelled():
            progress("Pipeline CANCELLED by user")
            summary["cancelled"] = True
            return True
        return False

    TOTAL_STEPS = 12
    migrations.init_db()

    # ── Step 1: Check pipeline state ──
    progress(_step_header(1, TOTAL_STEPS, "Check pipeline state"))
    t0 = time.perf_counter()
    state = load_pipeline_state()
    new_data = has_new_games(state) if not force_rerun else True
    progress(f"  New games detected: {'YES' if new_data else 'NO'}")
    if force_rerun:
        progress("  Force re-run enabled — all steps will execute")
    summary["steps"]["check_state"] = {
        "status": "done",
        "new_data": new_data,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if _check_cancel():
        return summary

    # ── Step 2: Load memory store ──
    progress(_step_header(2, TOTAL_STEPS, "Load data into memory"))
    t0 = time.perf_counter()
    store = get_memory_store()
    store.load(progress_cb=progress)
    summary["steps"]["load_memory"] = {
        "status": "done",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if _check_cancel():
        return summary

    # ── Step 3: Full data sync ──
    progress(_step_header(3, TOTAL_STEPS, "Full data sync"))
    t0 = time.perf_counter()
    try:
        from src.data.sync_service import full_sync
        full_sync(progress_cb=progress, force=force_rerun)
        state = mark_step_done(state, "sync")
    except Exception as exc:
        progress(f"  Sync error (continuing): {exc}")
    summary["steps"]["sync"] = {
        "status": "done",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if _check_cancel():
        return summary

    # ── Step 4: Reload memory if new data ──
    progress(_step_header(4, TOTAL_STEPS, "Reload memory (if new data)"))
    t0 = time.perf_counter()
    new_after_sync = has_new_games(state)
    if new_after_sync or force_rerun:
        progress("  Reloading memory store with fresh data...")
        store.reload(progress_cb=progress)
        state = update_game_snapshot(state)
    else:
        progress("  No new data — memory store still valid")
    summary["steps"]["reload_memory"] = {
        "status": "done",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if _check_cancel():
        return summary

    # ── Step 5: Build injury history ──
    if not force_rerun and is_step_fresh(state, "injury_history") and not new_after_sync:
        _skipped(5, TOTAL_STEPS, "Build injury history")
    else:
        progress(_step_header(5, TOTAL_STEPS, "Build injury history"))
        t0 = time.perf_counter()
        try:
            from src.analytics.injury_history import build_injury_history
            count = build_injury_history(progress_cb=progress)
            progress(f"  Built {count} injury records")
            state = mark_step_done(state, "injury_history")
        except Exception as exc:
            progress(f"  Injury history error: {exc}")
        summary["steps"]["injury_history"] = {
            "status": "done",
            "seconds": round(time.perf_counter() - t0, 1),
        }
    if _check_cancel():
        return summary

    # ── Step 6: Injury intelligence backfill ──
    progress(_step_header(6, TOTAL_STEPS, "Injury intelligence backfill"))
    t0 = time.perf_counter()
    try:
        from src.analytics.injury_intelligence import backfill_play_outcomes
        resolved = backfill_play_outcomes(progress_cb=progress)
        progress(f"  Backfilled {resolved} injury outcomes")
        state = mark_step_done(state, "injury_intel")
    except Exception as exc:
        progress(f"  Injury intelligence error: {exc}")
    summary["steps"]["injury_intel"] = {
        "status": "done",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if _check_cancel():
        return summary

    # ── Step 6b: Roster change detection (invalidate stale autotune) ──
    progress(_step_header(7, TOTAL_STEPS, "Roster change detection"))
    t0 = time.perf_counter()
    try:
        from src.analytics.stats_engine import detect_roster_change
        from src.analytics.autotune import clear_tuning
        from src.database.db import get_conn as _get_conn
        with _get_conn() as _conn:
            team_rows = _conn.execute("SELECT team_id, abbreviation FROM teams").fetchall()
        cleared_teams = []
        for t_row in team_rows:
            tid, abbr = int(t_row[0]), str(t_row[1])
            rc = detect_roster_change(tid)
            if rc["high_impact"]:
                clear_tuning(tid)
                cleared_teams.append(abbr)
                progress(f"  {abbr}: high-impact roster change — autotune cleared")
        if cleared_teams:
            progress(f"  Cleared autotune for {len(cleared_teams)} teams: {', '.join(cleared_teams)}")
        else:
            progress("  No high-impact roster changes detected")
    except Exception as exc:
        progress(f"  Roster change detection error: {exc}")
    summary["steps"]["roster_change"] = {
        "status": "done",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if _check_cancel():
        return summary

    # ── Step 7: Autotune all teams ──
    if not force_rerun and is_step_fresh(state, "autotune") and not new_after_sync:
        _skipped(7, TOTAL_STEPS, "Autotune all teams")
    else:
        progress(_step_header(7, TOTAL_STEPS, "Autotune all teams"))
        t0 = time.perf_counter()
        try:
            from src.analytics.autotune import autotune_all
            results = autotune_all(
                strength=autotune_strength,
                mode=autotune_mode,
                progress_cb=progress,
            )
            tuned = sum(1 for r in results if r.get("applied"))
            progress(f"  Autotune: {tuned}/{len(results)} teams got corrections")
            state = mark_step_done(state, "autotune")
        except Exception as exc:
            progress(f"  Autotune error: {exc}")
        summary["steps"]["autotune"] = {
            "status": "done",
            "seconds": round(time.perf_counter() - t0, 1),
        }
    if _check_cancel():
        return summary

    # ── Step 8: Train ML ensemble models ──
    if not force_rerun and is_step_fresh(state, "ml_train") and not new_after_sync:
        _skipped(8, TOTAL_STEPS, "Train ML ensemble models")
    else:
        progress(_step_header(8, TOTAL_STEPS, "Train ML ensemble models"))
        t0 = time.perf_counter()
        try:
            games = precompute_game_data(progress_cb=progress)
            from src.analytics.ml_model import train_models, reload_models
            ml_result = train_models(games, progress_cb=progress)
            reload_models()
            progress(
                f"  ML models: spread val MAE={ml_result.spread_val_mae:.2f}, "
                f"total val MAE={ml_result.total_val_mae:.2f}"
            )
            state = mark_step_done(state, "ml_train")
            summary["steps"]["ml_train"] = {
                "status": "done",
                "spread_val_mae": ml_result.spread_val_mae,
                "total_val_mae": ml_result.total_val_mae,
                "n_features": ml_result.n_features,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        except Exception as exc:
            progress(f"  ML training error: {exc}")
            summary["steps"]["ml_train"] = {
                "status": "error",
                "error": str(exc),
                "seconds": round(time.perf_counter() - t0, 1),
            }
    if _check_cancel():
        return summary

    # ── Step 9: Global weight optimisation ──
    if not force_rerun and is_step_fresh(state, "optimize") and not new_after_sync:
        _skipped(9, TOTAL_STEPS, "Global weight optimisation")
    else:
        progress(_step_header(9, TOTAL_STEPS, f"Global weight optimisation ({n_trials} trials)"))
        t0 = time.perf_counter()
        try:
            games = precompute_game_data(progress_cb=progress)
            opt_result = run_weight_optimiser(
                n_trials=n_trials,
                progress_cb=progress,
                precomputed_games=games,
            )
            progress(
                f"  Optimisation: loss {opt_result.baseline_loss:.2f} → "
                f"{opt_result.best_loss:.2f} ({opt_result.improvement_pct:+.1f}%)"
            )
            state = mark_step_done(state, "optimize")
            summary["steps"]["optimize"] = {
                "status": "done",
                "improvement_pct": opt_result.improvement_pct,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        except Exception as exc:
            progress(f"  Optimisation error: {exc}")
            summary["steps"]["optimize"] = {
                "status": "error",
                "error": str(exc),
                "seconds": round(time.perf_counter() - t0, 1),
            }
    if _check_cancel():
        return summary

    # ── Step 10: Per-team refinement ──
    if not force_rerun and is_step_fresh(state, "team_refine") and not new_after_sync:
        _skipped(10, TOTAL_STEPS, "Per-team refinement")
    else:
        progress(_step_header(10, TOTAL_STEPS, f"Per-team refinement ({team_trials} trials/team)"))
        t0 = time.perf_counter()
        try:
            games = precompute_game_data(progress_cb=progress)  # cache hit
            team_results = run_per_team_refinement(
                n_trials=team_trials,
                progress_cb=progress,
                precomputed_games=games,
            )
            adopted = sum(1 for r in team_results if r.used_team_weights)
            progress(f"  Per-team: {adopted}/{len(team_results)} teams refined")
            state = mark_step_done(state, "team_refine")
            summary["steps"]["team_refine"] = {
                "status": "done",
                "adopted": adopted,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        except Exception as exc:
            progress(f"  Per-team error: {exc}")
            summary["steps"]["team_refine"] = {
                "status": "error",
                "error": str(exc),
                "seconds": round(time.perf_counter() - t0, 1),
            }
    if _check_cancel():
        return summary

    # ── Step 11: Build residual calibration ──
    if not force_rerun and is_step_fresh(state, "calibrate") and not new_after_sync:
        _skipped(11, TOTAL_STEPS, "Build residual calibration")
    else:
        progress(_step_header(11, TOTAL_STEPS, "Build residual calibration"))
        t0 = time.perf_counter()
        try:
            cal = build_residual_calibration(progress_cb=progress)
            progress(f"  Calibration: {len(cal)} bins")
            state = mark_step_done(state, "calibrate")
        except Exception as exc:
            progress(f"  Calibration error: {exc}")
        summary["steps"]["calibrate"] = {
            "status": "done",
            "seconds": round(time.perf_counter() - t0, 1),
        }
    if _check_cancel():
        return summary

    # ── Step 12: Validation backtest ──
    progress(_step_header(12, TOTAL_STEPS, "Validation backtest"))
    t0 = time.perf_counter()
    try:
        bt_results = run_backtest(
            min_games_before=5,
            progress_cb=progress,
            max_workers=max_workers,
        )
        progress(
            f"  Backtest: {bt_results.total_games} games, "
            f"winner {bt_results.overall_spread_accuracy:.1f}%, "
            f"total-in-10 {bt_results.overall_total_accuracy:.1f}%"
        )
        state = mark_step_done(state, "backtest")
        summary["steps"]["backtest"] = {
            "status": "done",
            "games": bt_results.total_games,
            "winner_pct": bt_results.overall_spread_accuracy,
            "total_pct": bt_results.overall_total_accuracy,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        summary["backtest_results"] = bt_results
    except Exception as exc:
        progress(f"  Backtest error: {exc}")
        summary["steps"]["backtest"] = {
            "status": "error",
            "error": str(exc),
            "seconds": round(time.perf_counter() - t0, 1),
        }

    # ── Done ──
    total_elapsed = time.perf_counter() - t_start
    summary["total_seconds"] = round(total_elapsed, 1)
    state = update_game_snapshot(state)
    save_pipeline_state(state)
    progress(f"\nPipeline complete in {total_elapsed:.0f}s")
    return summary
