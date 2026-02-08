from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.analytics.autotune import autotune_all, autotune_team, get_all_tunings, clear_tuning
from src.database.db import get_conn


class _AutotuneWorker(QThread):
    """Background worker for running autotune (can be slow)."""
    progress = Signal(str)
    finished = Signal(list)

    def __init__(self, team_id: Optional[int], strength: float, parent=None):
        super().__init__(parent)
        self.team_id = team_id
        self.strength = strength

    def run(self):
        def _cb(msg: str):
            self.progress.emit(msg)

        try:
            if self.team_id is not None:
                result = autotune_team(self.team_id, strength=self.strength, progress_cb=_cb)
                self.finished.emit([result])
            else:
                results = autotune_all(strength=self.strength, progress_cb=_cb)
                self.finished.emit(results)
        except Exception as exc:
            self.progress.emit(f"Error: {exc}")
            self.finished.emit([])


class AutotuneView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[_AutotuneWorker] = None

        # Controls
        self.team_combo = QComboBox()
        self.team_combo.addItem("All Teams", None)
        self._populate_teams()

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 1.0)
        self.strength_spin.setSingleStep(0.05)
        self.strength_spin.setValue(0.75)

        self.run_btn = QPushButton("Run Autotune")
        self.run_btn.clicked.connect(self._run_autotune)  # type: ignore[arg-type]

        self.clear_btn = QPushButton("Clear All Tuning")
        self.clear_btn.clicked.connect(self._clear_tuning)  # type: ignore[arg-type]

        self.refresh_btn = QPushButton("Refresh Table")
        self.refresh_btn.clicked.connect(self._refresh_table)  # type: ignore[arg-type]

        # Progress log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)

        # Results table
        self.table = QTableWidget()

        # Layout
        controls = QGroupBox("Autotune Controls")
        form = QFormLayout()
        form.addRow("Team:", self.team_combo)
        form.addRow("Strength:", self.strength_spin)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.refresh_btn)
        form.addRow(btn_row)
        controls.setLayout(form)

        layout = QVBoxLayout()
        layout.addWidget(controls)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.log)
        layout.addWidget(QLabel("Current Tuning Corrections:"))
        layout.addWidget(self.table)
        self.setLayout(layout)

        # Load existing tuning on init
        self._refresh_table()

    def _populate_teams(self) -> None:
        try:
            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT team_id, abbreviation, name FROM teams ORDER BY abbreviation"
                ).fetchall()
            for tid, abbr, name in rows:
                self.team_combo.addItem(f"{abbr} - {name}", tid)
        except Exception:
            pass  # DB may not exist yet

    def _run_autotune(self) -> None:
        if self._worker and self._worker.isRunning():
            self.log.append("Autotune already running...")
            return

        team_id = self.team_combo.currentData()
        strength = self.strength_spin.value()
        label = self.team_combo.currentText()

        self.log.clear()
        self.log.append(f"Starting autotune for {label} (strength={strength:.2f})...")
        self.run_btn.setEnabled(False)

        self._worker = _AutotuneWorker(team_id, strength)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        self.log.append(msg)

    def _on_finished(self, results: list) -> None:
        self.run_btn.setEnabled(True)
        tuned = sum(1 for r in results if r.get("home_pts_correction", 0) != 0
                    or r.get("away_pts_correction", 0) != 0)
        self.log.append(f"\nDone: {len(results)} teams analyzed, {tuned} received corrections.")
        self._refresh_table()

    def _clear_tuning(self) -> None:
        team_id = self.team_combo.currentData()
        try:
            clear_tuning(team_id)
            label = self.team_combo.currentText() if team_id else "all teams"
            self.log.append(f"Cleared tuning for {label}.")
            self._refresh_table()
        except Exception as exc:
            self.log.append(f"Error clearing tuning: {exc}")

    def _refresh_table(self) -> None:
        try:
            tunings = get_all_tunings()
        except Exception:
            tunings = []

        headers = ["Team", "Home Adj", "Away Adj", "Games", "Avg Spread Err", "Avg Total Err", "Last Tuned"]
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        # Filter to teams with actual corrections
        active = [t for t in tunings if t["home_pts_correction"] != 0 or t["away_pts_correction"] != 0]
        self.table.setRowCount(len(active))

        for idx, t in enumerate(active):
            self.table.setItem(idx, 0, QTableWidgetItem(f"{t['abbr']} - {t['name']}"))
            self.table.setItem(idx, 1, QTableWidgetItem(f"{t['home_pts_correction']:+.2f}"))
            self.table.setItem(idx, 2, QTableWidgetItem(f"{t['away_pts_correction']:+.2f}"))
            self.table.setItem(idx, 3, QTableWidgetItem(str(t["games_analyzed"])))
            self.table.setItem(idx, 4, QTableWidgetItem(f"{t['avg_spread_error_before']:.1f}"))
            self.table.setItem(idx, 5, QTableWidgetItem(f"{t['avg_total_error_before']:.1f}"))
            self.table.setItem(idx, 6, QTableWidgetItem(t.get("last_tuned_at", "")[:16]))

        self.table.resizeColumnsToContents()
