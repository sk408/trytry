"""Live Games tab — auto-refresh 30s, color-coded rows."""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

logger = logging.getLogger(__name__)

_GREEN = QColor(34, 197, 94)
_PURPLE = QColor(168, 85, 247)
_WHITE = QColor(226, 232, 240)


class LiveView(QWidget):
    """Live games auto-refreshing view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        layout = QVBoxLayout(self)

        header = QLabel("Live Games")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # Controls
        ctrl = QHBoxLayout()
        self.auto_cb = QCheckBox("Auto-refresh (30s)")
        self.auto_cb.setChecked(True)
        self.auto_cb.toggled.connect(self._toggle_auto)
        ctrl.addWidget(self.auto_cb)

        refresh_btn = QPushButton("Refresh Now")
        refresh_btn.clicked.connect(self._refresh)
        ctrl.addWidget(refresh_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Status", "Away", "Score", "Home", "Score",
            "Spread", "O/U", "Recommendation",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(30_000)

        # Deferred initial load — let UI render first
        QTimer.singleShot(200, self._refresh)

    def _toggle_auto(self, checked: bool):
        if checked:
            self.timer.start(30_000)
        else:
            self.timer.stop()

    def _refresh(self):
        """Fetch live scores and recommendations."""
        try:
            from src.data.live_scores import fetch_live_scores
            games = fetch_live_scores()
        except Exception as e:
            logger.error(f"Live fetch error: {e}")
            games = []

        # Try recommendations
        recs = {}
        try:
            from src.analytics.live_recommendations import get_live_recommendations
            for g in games:
                gid = g.get("game_id", "")
                if g.get("status") == "LIVE":
                    try:
                        r = get_live_recommendations(
                            home_team_id=g.get("home_team_id", 0),
                            away_team_id=g.get("away_team_id", 0),
                            home_score=float(g.get("home_score", 0)),
                            away_score=float(g.get("away_score", 0)),
                            minutes_elapsed=float(g.get("minutes_elapsed", 0)),
                            quarter=int(g.get("quarter", 0)),
                        )
                        if r:
                            recs[gid] = r
                    except Exception as e:
                        logger.warning("Live rec failed for game %s: %s", gid, e)
        except Exception as e:
            logger.error("Live recommendations failed: %s", e, exc_info=True)

        self.table.setRowCount(len(games))
        for row, g in enumerate(games):
            status = g.get("status", "")
            away = g.get("away_team", "")
            home = g.get("home_team", "")
            away_score = str(g.get("away_score", ""))
            home_score = str(g.get("home_score", ""))

            spread = g.get("spread", "")
            ou = g.get("over_under", "")
            rec = ""
            gid = g.get("game_id", "")
            if gid in recs:
                rec_list = recs[gid]
                rec = "; ".join(r.get("text", "") for r in rec_list[:2])

            items = [status, away, away_score, home, home_score,
                     str(spread), str(ou), rec]

            # Row color
            if status == "LIVE":
                color = _GREEN
            elif status == "FINAL":
                color = _PURPLE
            else:
                color = _WHITE

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setForeground(color)
                self.table.setItem(row, col, item)

        if self.main_window:
            self.main_window.set_status(f"Live: {len(games)} games")
