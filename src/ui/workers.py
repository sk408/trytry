"""QObject-based workers for background ops in the desktop UI.

6 sync workers + 12 accuracy/analysis workers.
Each worker runs in a QThread and emits progress/finished signals.
"""

import logging
import threading
from PySide6.QtCore import QObject, Signal, QThread, Qt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base worker
# ---------------------------------------------------------------------------

class BaseWorker(QObject):
    """Base background worker with progress & stop support."""
    progress = Signal(str)
    result = Signal(dict)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _check_stop(self) -> bool:
        return self._stop_event.is_set()

    def run(self):
        raise NotImplementedError


def _start_worker(worker: BaseWorker, on_progress=None, on_done=None, on_result=None):
    """Launch a worker on a new QThread. Returns the worker for stop().

    Uses QueuedConnection for all cross-thread signal→slot connections
    so that slots always run on the main/GUI thread, even if the
    callable is a lambda or free function (which have no QObject affinity).
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    _QC = Qt.ConnectionType.QueuedConnection
    if on_progress:
        worker.progress.connect(on_progress, _QC)
    if on_result:
        worker.result.connect(on_result, _QC)
    if on_done:
        worker.finished.connect(on_done, _QC)
    worker.finished.connect(thread.quit)
    # prevent GC
    worker._thread_ref = thread
    thread.start()
    return worker


# ---------------------------------------------------------------------------
# Sync workers
# ---------------------------------------------------------------------------

class SyncWorker(BaseWorker):
    """Runs one of the sync steps."""
    def __init__(self, step: str, force: bool = False):
        super().__init__()
        self.step = step
        self.force = force

    def run(self):
        try:
            from src.data.sync_service import (
                full_sync, sync_injuries_step, sync_injury_history,
                sync_team_metrics, sync_player_impact,
            )
            from src.data.image_cache import preload_images

            cb = lambda msg: self.progress.emit(msg)

            if self.step == "full":
                full_sync(callback=cb, force=self.force)
            elif self.step == "injuries":
                sync_injuries_step(callback=cb, force=self.force)
            elif self.step == "injury_history":
                sync_injury_history(callback=cb, force=self.force)
            elif self.step == "team_metrics":
                sync_team_metrics(callback=cb, force=self.force)
            elif self.step == "player_impact":
                sync_player_impact(callback=cb, force=self.force)
            elif self.step == "images":
                preload_images(callback=cb)
            else:
                self.progress.emit(f"Unknown step: {self.step}")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class InjuryWorker(BaseWorker):
    """Runs injury scraping."""
    def run(self):
        try:
            from src.data.injury_scraper import scrape_all_injuries
            self.progress.emit("Scraping injuries...")
            injuries = scrape_all_injuries()
            self.progress.emit(f"Found {len(injuries)} injuries")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


def start_sync_worker(step: str, on_progress=None, on_done=None, force: bool = False):
    return _start_worker(SyncWorker(step, force=force), on_progress, on_done)


def start_injury_worker(on_progress=None, on_done=None):
    return _start_worker(InjuryWorker(), on_progress, on_done)


class OddsSyncWorker(BaseWorker):
    """Runs odds backfill."""
    def run(self):
        try:
            from src.data.odds_sync import backfill_odds
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Syncing historical odds...")
            count = backfill_odds(callback=cb)
            self.progress.emit(f"Odds sync complete. Saved odds for {count} games.")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


def start_odds_sync_worker(on_progress=None, on_done=None):
    return _start_worker(OddsSyncWorker(), on_progress, on_done)


# ---------------------------------------------------------------------------
# Accuracy / Analysis workers
# ---------------------------------------------------------------------------

class BacktestWorker(BaseWorker):
    def run(self):
        try:
            from src.analytics.backtester import run_backtest
            self.progress.emit("Running backtest...")
            results = run_backtest(
                callback=lambda msg: self.progress.emit(msg)
            )
            self.result.emit(results)
            self.progress.emit("Backtest complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class OptimizerWorker(BaseWorker):
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import optimize_weights
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Optimizing weights (200 trials)...")
                optimize_weights(games, callback=cb)
                self.progress.emit("Optimization complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class CalibrationWorker(BaseWorker):
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import build_residual_calibration
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Building residual calibration...")
                build_residual_calibration(games, callback=cb)
                self.progress.emit("Calibration complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class FeatureImportanceWorker(BaseWorker):
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import compute_feature_importance
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Computing feature importance...")
                result = compute_feature_importance(games, callback=cb)
                self.result.emit({"feature_importance": result})
                self.progress.emit("Feature importance complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class MLFeatureWorker(BaseWorker):
    """ML/SHAP-based feature importance."""
    def run(self):
        try:
            from src.analytics.ml_model import get_shap_importance
            self.progress.emit("Computing ML feature importance...")
            result = get_shap_importance()
            self.result.emit({"ml_features": result})
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class GroupedFeatureWorker(BaseWorker):
    """Grouped feature importance."""
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import compute_feature_importance
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Computing grouped feature importance...")
                result = compute_feature_importance(games, callback=cb)
                # Group by category
                groups = {}
                for f in result:
                    name = f.get("feature", "")
                    if "factor" in name or "rating" in name:
                        cat = "Defense/Ratings"
                    elif "ff_" in name or "four_factors" in name:
                        cat = "Four Factors"
                    elif "hustle" in name or "clutch" in name:
                        cat = "Hustle/Clutch"
                    elif "fatigue" in name or "pace" in name:
                        cat = "Pace/Fatigue"
                    else:
                        cat = "Other"
                    groups.setdefault(cat, []).append(f)
                self.result.emit({"grouped_features": groups})
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class FFTWorker(BaseWorker):
    """FFT error pattern analysis."""
    def run(self):
        try:
            self.progress.emit("Running FFT analysis...")
            # FFT on backtest errors
            from src.analytics.backtester import run_backtest
            results = run_backtest()
            per_game = results.get("per_game", [])
            import numpy as np
            errors = [g.get("spread_error", 0) for g in per_game]
            if len(errors) > 10:
                fft = np.fft.rfft(errors)
                freqs = np.fft.rfftfreq(len(errors))
                magnitudes = np.abs(fft)
                top_idx = np.argsort(magnitudes)[-5:][::-1]
                fft_result = [
                    {"frequency": float(freqs[i]), "magnitude": float(magnitudes[i])}
                    for i in top_idx
                ]
                self.result.emit({"fft": fft_result})
                self.progress.emit(f"FFT: found {len(fft_result)} dominant frequencies")
            else:
                self.progress.emit("Not enough data for FFT")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class TeamRefineWorker(BaseWorker):
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import per_team_refinement
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Running per-team refinement...")
                per_team_refinement(games, callback=cb)
                self.progress.emit("Team refinement complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class ComboWorker(BaseWorker):
    """Global + per-team optimization."""
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import optimize_weights, per_team_refinement
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Phase 1: Global optimization...")
                optimize_weights(games, callback=cb)
                self.progress.emit("Phase 2: Per-team refinement...")
                per_team_refinement(games, callback=cb)
                self.progress.emit("Combo optimization complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class ContinuousWorker(BaseWorker):
    """Continuous optimization loop until stopped."""
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import optimize_weights
            cb = lambda msg: self.progress.emit(msg)
            iteration = 0
            while not self._check_stop():
                iteration += 1
                self.progress.emit(f"Continuous iteration {iteration}...")
                games = precompute_game_data(callback=cb)
                if games:
                    optimize_weights(games, callback=cb)
                if self._check_stop():
                    break
            self.progress.emit(f"Stopped after {iteration} iterations")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class PipelineWorker(BaseWorker):
    """Full 12-step pipeline."""
    def run(self):
        try:
            from src.analytics.pipeline import run_full_pipeline
            self.progress.emit("Starting full pipeline...")
            results = run_full_pipeline(
                callback=lambda msg: self.progress.emit(msg)
            )
            # Emit backtest results so accuracy cards get populated
            bt = results.get("backtest", {})
            if bt and bt.get("total_games", 0) > 0:
                self.result.emit(bt)
            self.progress.emit("Pipeline complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class MLTrainWorker(BaseWorker):
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.ml_model import train_models
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Training ML models...")
                train_models(games, callback=cb)
                self.progress.emit("ML training complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class DiagnosticCSVWorker(BaseWorker):
    """Export diagnostic CSV for worst-performing teams."""
    def __init__(self, backtest_results=None):
        super().__init__()
        self._bt_results = backtest_results

    def run(self):
        try:
            from src.analytics.backtester import export_diagnostic_csv
            from src.database.db import thread_local_db
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Generating diagnostic CSV...")
            with thread_local_db():
                path = export_diagnostic_csv(
                    backtest_results=self._bt_results,
                    callback=cb,
                )
            if path:
                self.result.emit({"csv_path": path})
            self.progress.emit("Diagnostic export complete")
        except Exception as e:
            import traceback
            logger.error("DiagnosticCSVWorker failed:\n%s", traceback.format_exc())
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


# ---------------------------------------------------------------------------
# Factory functions for accuracy workers
# ---------------------------------------------------------------------------

def start_backtest_worker(on_progress=None, on_result=None, on_done=None):
    return _start_worker(BacktestWorker(), on_progress, on_done, on_result)

def start_optimizer_worker(on_progress=None, on_done=None):
    return _start_worker(OptimizerWorker(), on_progress, on_done)

def start_calibration_worker(on_progress=None, on_done=None):
    return _start_worker(CalibrationWorker(), on_progress, on_done)

def start_feature_importance_worker(on_progress=None, on_done=None):
    return _start_worker(FeatureImportanceWorker(), on_progress, on_done)

def start_ml_feature_worker(on_progress=None, on_done=None):
    return _start_worker(MLFeatureWorker(), on_progress, on_done)

def start_grouped_feature_worker(on_progress=None, on_done=None):
    return _start_worker(GroupedFeatureWorker(), on_progress, on_done)

def start_fft_worker(on_progress=None, on_done=None):
    return _start_worker(FFTWorker(), on_progress, on_done)

def start_team_refine_worker(on_progress=None, on_done=None):
    return _start_worker(TeamRefineWorker(), on_progress, on_done)

def start_combo_worker(on_progress=None, on_done=None):
    return _start_worker(ComboWorker(), on_progress, on_done)

def start_continuous_worker(on_progress=None, on_done=None):
    return _start_worker(ContinuousWorker(), on_progress, on_done)

def start_pipeline_worker(on_progress=None, on_result=None, on_done=None):
    return _start_worker(PipelineWorker(), on_progress, on_done, on_result)

def start_ml_train_worker(on_progress=None, on_done=None):
    return _start_worker(MLTrainWorker(), on_progress, on_done)

def start_diagnostic_csv_worker(backtest_results=None, on_progress=None, on_result=None, on_done=None):
    return _start_worker(DiagnosticCSVWorker(backtest_results), on_progress, on_done, on_result)
