from __future__ import annotations

import logging
import threading
import traceback

from PySide6.QtCore import QObject, QThread, Signal

_log = logging.getLogger(__name__)

from src.data.sync_service import (
    SyncCancelled,
    full_sync,
    sync_injuries,
    sync_injury_history,
    sync_player_impact,
    sync_team_metrics,
)


class SyncWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    cancelled = Signal(str)
    error = Signal(str)

    def __init__(self, force: bool = False) -> None:
        super().__init__()
        self._cancel_flag = threading.Event()
        self._force = force

    def request_cancel(self) -> None:
        self._cancel_flag.set()

    def run(self) -> None:
        try:
            self._cancel_flag.clear()
            full_sync(progress_cb=self.progress.emit,
                      cancel_check=self._cancel_flag.is_set,
                      force=self._force)
            self.finished.emit("Sync complete")
        except SyncCancelled:
            self.cancelled.emit("Sync stopped — data synced so far is saved")
        except Exception as exc:  # pragma: no cover - UI path
            tb = traceback.format_exc()
            self.error.emit(f"{exc}\n{tb}")


class InjurySyncWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        try:
            self.progress.emit("Fetching injury reports...")
            count = sync_injuries(progress_cb=self.progress.emit)
            self.finished.emit(f"Injury sync complete: {count} players marked injured")
        except Exception as exc:  # pragma: no cover - UI path
            tb = traceback.format_exc()
            self.error.emit(f"{exc}\n{tb}")


class InjuryHistoryWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        try:
            self.progress.emit("Analyzing game logs to infer historical injuries...")
            count = sync_injury_history(progress_cb=self.progress.emit)
            self.finished.emit(f"Injury history built: {count} records")
        except Exception as exc:  # pragma: no cover - UI path
            tb = traceback.format_exc()
            self.error.emit(f"{exc}\n{tb}")


class TeamMetricsWorker(QObject):
    """Fetch team advanced metrics from NBA API."""

    progress = Signal(str)
    finished = Signal(str)
    cancelled = Signal(str)
    error = Signal(str)

    def __init__(self, force: bool = False) -> None:
        super().__init__()
        self._cancel_flag = threading.Event()
        self._force = force

    def request_cancel(self) -> None:
        self._cancel_flag.set()

    def run(self) -> None:
        try:
            self._cancel_flag.clear()
            self.progress.emit("Fetching team advanced metrics...")
            count = sync_team_metrics(progress_cb=self.progress.emit,
                                      cancel_check=self._cancel_flag.is_set,
                                      force=self._force)
            self.finished.emit(f"Team metrics sync complete: {count} teams updated")
        except SyncCancelled:
            self.cancelled.emit("Team metrics sync stopped — data synced so far is saved")
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"{exc}\n{tb}")


class PlayerImpactWorker(QObject):
    """Fetch player on/off and estimated impact metrics."""

    progress = Signal(str)
    finished = Signal(str)
    cancelled = Signal(str)
    error = Signal(str)

    def __init__(self, force: bool = False) -> None:
        super().__init__()
        self._cancel_flag = threading.Event()
        self._force = force

    def request_cancel(self) -> None:
        self._cancel_flag.set()

    def run(self) -> None:
        try:
            self._cancel_flag.clear()
            self.progress.emit("Fetching player impact metrics...")
            count = sync_player_impact(progress_cb=self.progress.emit,
                                       cancel_check=self._cancel_flag.is_set,
                                       force=self._force)
            self.finished.emit(f"Player impact sync complete: {count} players updated")
        except SyncCancelled:
            self.cancelled.emit("Player impact sync stopped — data synced so far is saved")
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"{exc}\n{tb}")


class ImageSyncWorker(QObject):
    """Download team logos and player headshots into the disk cache."""

    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        try:
            from src.data.image_cache import preload_team_logos, preload_player_photos

            self.progress.emit("Downloading team logos…")
            logos = preload_team_logos(progress_cb=self.progress.emit)
            self.progress.emit(f"Team logos: {logos} new downloads")

            self.progress.emit("Downloading player photos…")
            photos = preload_player_photos(progress_cb=self.progress.emit)
            self.progress.emit(f"Player photos: {photos} new downloads")

            self.finished.emit(
                f"Image sync complete — {logos} logos, {photos} photos downloaded"
            )
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"{exc}\n{tb}")


def start_sync_worker(on_progress, on_finished, on_error, on_cancelled=None, force=False):
    thread = QThread()
    worker = SyncWorker(force=force)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[arg-type]
    worker.progress.connect(on_progress)  # type: ignore[arg-type]
    worker.finished.connect(on_finished)  # type: ignore[arg-type]
    worker.error.connect(on_error)  # type: ignore[arg-type]
    if on_cancelled:
        worker.cancelled.connect(on_cancelled)  # type: ignore[arg-type]
    worker.finished.connect(thread.quit)  # type: ignore[arg-type]
    worker.cancelled.connect(thread.quit)  # type: ignore[arg-type]
    worker.error.connect(thread.quit)  # type: ignore[arg-type]
    thread.finished.connect(worker.deleteLater)  # type: ignore[arg-type]
    thread.finished.connect(thread.deleteLater)  # type: ignore[arg-type]
    return thread, worker


def start_injury_sync_worker(on_progress, on_finished, on_error):
    thread = QThread()
    worker = InjurySyncWorker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[arg-type]
    worker.progress.connect(on_progress)  # type: ignore[arg-type]
    worker.finished.connect(on_finished)  # type: ignore[arg-type]
    worker.error.connect(on_error)  # type: ignore[arg-type]
    worker.finished.connect(thread.quit)  # type: ignore[arg-type]
    worker.error.connect(thread.quit)  # type: ignore[arg-type]
    thread.finished.connect(worker.deleteLater)  # type: ignore[arg-type]
    thread.finished.connect(thread.deleteLater)  # type: ignore[arg-type]
    return thread, worker


def start_injury_history_worker(on_progress, on_finished, on_error):
    thread = QThread()
    worker = InjuryHistoryWorker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[arg-type]
    worker.progress.connect(on_progress)  # type: ignore[arg-type]
    worker.finished.connect(on_finished)  # type: ignore[arg-type]
    worker.error.connect(on_error)  # type: ignore[arg-type]
    worker.finished.connect(thread.quit)  # type: ignore[arg-type]
    worker.error.connect(thread.quit)  # type: ignore[arg-type]
    thread.finished.connect(worker.deleteLater)  # type: ignore[arg-type]
    thread.finished.connect(thread.deleteLater)  # type: ignore[arg-type]
    return thread, worker


def start_team_metrics_worker(on_progress, on_finished, on_error, on_cancelled=None, force=False):
    thread = QThread()
    worker = TeamMetricsWorker(force=force)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[arg-type]
    worker.progress.connect(on_progress)  # type: ignore[arg-type]
    worker.finished.connect(on_finished)  # type: ignore[arg-type]
    worker.error.connect(on_error)  # type: ignore[arg-type]
    if on_cancelled:
        worker.cancelled.connect(on_cancelled)  # type: ignore[arg-type]
    worker.finished.connect(thread.quit)  # type: ignore[arg-type]
    worker.cancelled.connect(thread.quit)  # type: ignore[arg-type]
    worker.error.connect(thread.quit)  # type: ignore[arg-type]
    thread.finished.connect(worker.deleteLater)  # type: ignore[arg-type]
    thread.finished.connect(thread.deleteLater)  # type: ignore[arg-type]
    return thread, worker


def start_player_impact_worker(on_progress, on_finished, on_error, on_cancelled=None, force=False):
    thread = QThread()
    worker = PlayerImpactWorker(force=force)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[arg-type]
    worker.progress.connect(on_progress)  # type: ignore[arg-type]
    worker.finished.connect(on_finished)  # type: ignore[arg-type]
    worker.error.connect(on_error)  # type: ignore[arg-type]
    if on_cancelled:
        worker.cancelled.connect(on_cancelled)  # type: ignore[arg-type]
    worker.finished.connect(thread.quit)  # type: ignore[arg-type]
    worker.cancelled.connect(thread.quit)  # type: ignore[arg-type]
    worker.error.connect(thread.quit)  # type: ignore[arg-type]
    thread.finished.connect(worker.deleteLater)  # type: ignore[arg-type]
    thread.finished.connect(thread.deleteLater)  # type: ignore[arg-type]
    return thread, worker


def start_image_sync_worker(on_progress, on_finished, on_error):
    thread = QThread()
    worker = ImageSyncWorker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[arg-type]
    worker.progress.connect(on_progress)  # type: ignore[arg-type]
    worker.finished.connect(on_finished)  # type: ignore[arg-type]
    worker.error.connect(on_error)  # type: ignore[arg-type]
    worker.finished.connect(thread.quit)  # type: ignore[arg-type]
    worker.error.connect(thread.quit)  # type: ignore[arg-type]
    thread.finished.connect(worker.deleteLater)  # type: ignore[arg-type]
    thread.finished.connect(thread.deleteLater)  # type: ignore[arg-type]
    return thread, worker


# ── Schedule fetch (lightweight, returns a DataFrame) ──


class ScheduleFetchWorker(QObject):
    """Fetch the NBA schedule off the UI thread.

    Emits *finished* with the resulting ``pd.DataFrame`` (possibly empty)
    or *error* with a message string.
    """
    finished = Signal(object)   # pd.DataFrame
    error = Signal(str)

    def __init__(self, include_future_days: int = 14, force_refresh: bool = False) -> None:
        super().__init__()
        self._future_days = include_future_days
        self._force = force_refresh

    def run(self) -> None:
        try:
            from src.data.sync_service import sync_schedule
            df = sync_schedule(
                include_future_days=self._future_days,
                force_refresh=self._force,
            )
            self.finished.emit(df)
        except Exception as exc:
            self.error.emit(str(exc))


def start_schedule_fetch_worker(on_finished, on_error,
                                include_future_days=14, force_refresh=False):
    """Convenience launcher — returns ``(thread, worker)``."""
    _log.info("[Schedule-Worker] Launching async schedule fetch (future_days=%d, force=%s)",
              include_future_days, force_refresh)
    thread = QThread()
    worker = ScheduleFetchWorker(include_future_days, force_refresh)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[arg-type]
    worker.finished.connect(on_finished)  # type: ignore[arg-type]
    worker.error.connect(on_error)  # type: ignore[arg-type]
    worker.finished.connect(thread.quit)  # type: ignore[arg-type]
    worker.error.connect(thread.quit)  # type: ignore[arg-type]
    thread.finished.connect(worker.deleteLater)  # type: ignore[arg-type]
    thread.finished.connect(thread.deleteLater)  # type: ignore[arg-type]
    return thread, worker
