from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from src.ui.workers import start_sync_worker, start_injury_sync_worker, start_injury_history_worker


class Dashboard(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status = QLabel("Ready")
        self.status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(250)
        self._thread = None
        self._worker = None
        self._injury_thread = None
        self._injury_worker = None
        self._history_thread = None
        self._history_worker = None

        sync_button = QPushButton("Sync Data (teams/players/logs)")
        sync_button.clicked.connect(self._sync_data)  # type: ignore[arg-type]
        self.sync_button = sync_button

        injury_button = QPushButton("Sync Current Injuries")
        injury_button.clicked.connect(self._sync_injuries)  # type: ignore[arg-type]
        self.injury_button = injury_button

        history_button = QPushButton("Build Injury History")
        history_button.setToolTip("Infer historical injuries from game logs for backtesting")
        history_button.clicked.connect(self._build_injury_history)  # type: ignore[arg-type]
        self.history_button = history_button

        button_row = QHBoxLayout()
        button_row.addWidget(sync_button)
        button_row.addWidget(injury_button)
        button_row.addWidget(history_button)
        button_row.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("NBA Betting Analytics"))
        layout.addLayout(button_row)
        layout.addWidget(self.status)
        layout.addWidget(self.log)
        layout.addStretch()
        self.setLayout(layout)

    def _sync_data(self) -> None:
        if self._thread:
            return
        self.sync_button.setEnabled(False)
        self.status.setText("Starting sync...")
        self.log.clear()
        self._thread, self._worker = start_sync_worker(
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )
        self._thread.start()

    def _on_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)

    def _on_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)
        self.sync_button.setEnabled(True)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    def _on_error(self, msg: str) -> None:
        self.status.setText(f"Sync failed: {msg}")
        self.log.append(f"ERROR: {msg}")
        self.sync_button.setEnabled(True)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    def _sync_injuries(self) -> None:
        if self._injury_thread:
            return
        self.injury_button.setEnabled(False)
        self.status.setText("Syncing injuries...")
        self.log.append("--- Injury Sync Started ---")
        self._injury_thread, self._injury_worker = start_injury_sync_worker(
            on_progress=self._on_injury_progress,
            on_finished=self._on_injury_finished,
            on_error=self._on_injury_error,
        )
        self._injury_thread.start()

    def _on_injury_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)

    def _on_injury_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)
        self.log.append("--- Injury Sync Complete ---")
        self.injury_button.setEnabled(True)
        if self._injury_thread:
            self._injury_thread.quit()
            self._injury_thread.wait()
        self._injury_thread = None
        self._injury_worker = None

    def _on_injury_error(self, msg: str) -> None:
        self.status.setText(f"Injury sync failed: {msg}")
        self.log.append(f"ERROR: {msg}")
        self.injury_button.setEnabled(True)
        if self._injury_thread:
            self._injury_thread.quit()
            self._injury_thread.wait()
        self._injury_thread = None
        self._injury_worker = None

    def _build_injury_history(self) -> None:
        if self._history_thread:
            return
        self.history_button.setEnabled(False)
        self.status.setText("Building injury history from game logs...")
        self.log.append("--- Building Injury History ---")
        self._history_thread, self._history_worker = start_injury_history_worker(
            on_progress=self._on_history_progress,
            on_finished=self._on_history_finished,
            on_error=self._on_history_error,
        )
        self._history_thread.start()

    def _on_history_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)

    def _on_history_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)
        self.log.append("--- Injury History Built ---")
        self.history_button.setEnabled(True)
        if self._history_thread:
            self._history_thread.quit()
            self._history_thread.wait()
        self._history_thread = None
        self._history_worker = None

    def _on_history_error(self, msg: str) -> None:
        self.status.setText(f"Injury history build failed: {msg}")
        self.log.append(f"ERROR: {msg}")
        self.history_button.setEnabled(True)
        if self._history_thread:
            self._history_thread.quit()
            self._history_thread.wait()
        self._history_thread = None
        self._history_worker = None
