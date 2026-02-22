"""Autotune tab — team selector, strength/mode/correction controls, corrections table."""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor

logger = logging.getLogger(__name__)


class _AutotuneWorker(QObject):
    """Background worker for autotune."""
    progress = Signal(str)
    finished = Signal()

    def __init__(self, team_id=None, strength=0.75, mode="classic", max_corr=10.0):
        super().__init__()
        self.team_id = team_id
        self.strength = strength
        self.mode = mode
        self.max_corr = max_corr

    def run(self):
        try:
            from src.analytics.autotune import autotune_team, autotune_all
            from src.analytics.backtester import get_actual_game_results
            cb = lambda msg: self.progress.emit(msg)
            if self.team_id:
                self.progress.emit(f"Autotuning team {self.team_id}...")
                games = get_actual_game_results()
                autotune_team(
                    self.team_id,
                    games,
                    strength=self.strength,
                    mode=self.mode,
                    max_abs_correction=self.max_corr,
                    callback=cb,
                )
                self.progress.emit("Done")
            else:
                self.progress.emit("Autotuning all teams...")
                autotune_all(
                    strength=self.strength,
                    mode=self.mode,
                    max_abs_correction=self.max_corr,
                    callback=cb,
                )
                self.progress.emit("All teams complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class AutotuneView(QWidget):
    """Autotune corrections view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._worker_thread = None
        self._worker = None

        layout = QVBoxLayout(self)

        header = QLabel("Autotune — Per-Team Corrections")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # Controls
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Team:"))
        self.team_combo = QComboBox()
        self.team_combo.addItem("All Teams", None)
        self.team_combo.setMinimumWidth(200)
        ctrl.addWidget(self.team_combo)

        ctrl.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["classic", "walk_forward"])
        ctrl.addWidget(self.mode_combo)

        ctrl.addWidget(QLabel("Strength:"))
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.1, 2.0)
        self.strength_spin.setValue(0.75)
        self.strength_spin.setSingleStep(0.05)
        ctrl.addWidget(self.strength_spin)

        ctrl.addWidget(QLabel("Max Corr:"))
        self.max_corr_spin = QDoubleSpinBox()
        self.max_corr_spin.setRange(1.0, 30.0)
        self.max_corr_spin.setValue(30.0)
        self.max_corr_spin.setSingleStep(0.5)
        ctrl.addWidget(self.max_corr_spin)

        layout.addLayout(ctrl)

        # Action buttons
        btn_layout = QHBoxLayout()
        run_btn = QPushButton("Run Autotune")
        run_btn.clicked.connect(self._on_run)
        btn_layout.addWidget(run_btn)

        clear_btn = QPushButton("Clear All Corrections")
        clear_btn.setProperty("class", "danger")
        clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(clear_btn)

        refresh_btn = QPushButton("Refresh Table")
        refresh_btn.clicked.connect(self._load_corrections)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Corrections table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Team", "Home Correction", "Away Correction", "Games Used",
        ])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        layout.addWidget(self.log)

        self._load_teams()
        self._load_corrections()

    def _load_teams(self):
        try:
            from src.database import db
            teams = db.fetch_all(
                "SELECT team_id, abbreviation FROM teams ORDER BY abbreviation"
            )
            for t in teams:
                self.team_combo.addItem(t["abbreviation"], t["team_id"])
        except Exception:
            pass

    def _load_corrections(self):
        """Load current corrections from DB."""
        try:
            from src.database import db
            rows = db.fetch_all(
                "SELECT tt.team_id, t.abbreviation, "
                "       tt.home_pts_correction, tt.away_pts_correction, tt.games_analyzed "
                "FROM team_tuning tt "
                "LEFT JOIN teams t ON tt.team_id = t.team_id "
                "ORDER BY t.abbreviation"
            )
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                self.table.setItem(r, 0, QTableWidgetItem(row.get("abbreviation", "")))

                home_corr = float(row.get("home_pts_correction", 0))
                away_corr = float(row.get("away_pts_correction", 0))

                home_item = QTableWidgetItem(f"{home_corr:+.2f}")
                away_item = QTableWidgetItem(f"{away_corr:+.2f}")

                # Color-code: positive = green, negative = red
                if home_corr > 0:
                    home_item.setForeground(QColor(34, 197, 94))
                elif home_corr < 0:
                    home_item.setForeground(QColor(239, 68, 68))

                if away_corr > 0:
                    away_item.setForeground(QColor(34, 197, 94))
                elif away_corr < 0:
                    away_item.setForeground(QColor(239, 68, 68))

                self.table.setItem(r, 1, home_item)
                self.table.setItem(r, 2, away_item)
                self.table.setItem(r, 3, QTableWidgetItem(
                    str(row.get("games_analyzed", 0))
                ))
        except Exception as e:
            logger.error(f"Load corrections: {e}")

    def _on_run(self):
        if self._worker_thread and self._worker_thread.isRunning():
            return
        self.log.clear()
        team_id = self.team_combo.currentData()
        self._worker = _AutotuneWorker(
            team_id=team_id,
            strength=self.strength_spin.value(),
            mode=self.mode_combo.currentText(),
            max_corr=self.max_corr_spin.value(),
        )
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        # QueuedConnection ensures the slot runs on the main/GUI thread,
        # not on the worker thread (which would crash Qt).
        self._worker.progress.connect(
            self._on_progress, Qt.ConnectionType.QueuedConnection
        )
        self._worker.finished.connect(
            self._on_done, Qt.ConnectionType.QueuedConnection
        )
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.start()

    def _on_progress(self, msg: str):
        """Append a progress message to the log (runs on main thread)."""
        self.log.append(f'<span style="color:#94a3b8">{msg}</span>')

    def _on_done(self):
        self._load_corrections()
        if self.main_window:
            self.main_window.set_status("Autotune complete")

    def _on_clear(self):
        try:
            from src.analytics.autotune import clear_all_tuning
            clear_all_tuning()
            self._load_corrections()
            self.log.append('<span style="color:#22c55e">All corrections cleared</span>')
        except Exception as e:
            self.log.append(f'<span style="color:#ef4444">Error: {e}</span>')
