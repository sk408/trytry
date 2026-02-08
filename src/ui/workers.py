from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal

from src.data.sync_service import full_sync, sync_injuries, sync_injury_history


class SyncWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        try:
            full_sync(progress_cb=self.progress.emit)
            self.finished.emit("Sync complete")
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


def start_sync_worker(on_progress, on_finished, on_error):
    thread = QThread()
    worker = SyncWorker()
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
