"""Schedule tab — schedule table with team logos and day-of-week dates."""

import logging
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit,
)
from PySide6.QtCore import Qt, Signal, QDate, QSize, QTimer

logger = logging.getLogger(__name__)

_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class ScheduleView(QWidget):
    """NBA schedule browser with game selection."""

    game_selected = Signal(int, int)  # home_team_id, away_team_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._schedule = []

        layout = QVBoxLayout(self)

        header = QLabel("Schedule")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate())
        self.date_from.setCalendarPopup(True)
        ctrl.addWidget(self.date_from)

        ctrl.addWidget(QLabel("Days:"))
        self.days_label = QLabel("14")
        ctrl.addWidget(self.days_label)

        load_btn = QPushButton("Load Schedule")
        load_btn.clicked.connect(self._load)
        ctrl.addWidget(load_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Date", "Time", "Away", "Home", "Action",
        ])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setIconSize(QSize(24, 24))
        layout.addWidget(self.table)

        # Deferred initial load — let UI render first
        QTimer.singleShot(400, self._load)

    def _load(self):
        """Fetch schedule from NBA CDN."""
        try:
            from src.data.nba_fetcher import fetch_nba_cdn_schedule
            self._schedule = fetch_nba_cdn_schedule()
        except Exception as e:
            logger.error(f"Schedule load: {e}")
            self._schedule = []

        start = self.date_from.date().toPython().isoformat()
        end = (self.date_from.date().toPython() + timedelta(days=14)).isoformat()

        filtered = [
            g for g in self._schedule
            if g.get("game_date") and start <= g["game_date"] <= end
        ]

        self.table.setRowCount(len(filtered))
        for row, g in enumerate(filtered):
            self.table.setRowHeight(row, 30)

            # Date with day of week  (e.g. "Wed, Feb 26")
            raw_date = g.get("game_date", "")
            date_display = raw_date
            try:
                dt = datetime.strptime(raw_date, "%Y-%m-%d")
                day_name = _DAY_NAMES[dt.weekday()]
                date_display = f"{day_name}, {dt.strftime('%b %d').lstrip('0')}"
            except Exception as e:
                logger.warning("Date format failed for '%s': %s", raw_date, e)
            self.table.setItem(row, 0, QTableWidgetItem(date_display))

            self.table.setItem(row, 1, QTableWidgetItem(g.get("game_time", "")))

            # Away team with logo
            away_item = QTableWidgetItem(g.get("away_team", ""))
            away_id = g.get("away_team_id")
            if away_id:
                try:
                    from src.ui.widgets.image_utils import get_team_logo
                    logo = get_team_logo(int(away_id), 24)
                    if logo:
                        away_item.setData(Qt.ItemDataRole.DecorationRole, logo)
                except Exception as e:
                    logger.warning("Away team logo load failed for %s: %s", away_id, e)
            self.table.setItem(row, 2, away_item)

            # Home team with logo
            home_item = QTableWidgetItem(g.get("home_team", ""))
            home_id = g.get("home_team_id")
            if home_id:
                try:
                    from src.ui.widgets.image_utils import get_team_logo
                    logo = get_team_logo(int(home_id), 24)
                    if logo:
                        home_item.setData(Qt.ItemDataRole.DecorationRole, logo)
                except Exception as e:
                    logger.warning("Home team logo load failed for %s: %s", home_id, e)
            self.table.setItem(row, 3, home_item)

            predict_btn = QPushButton("Predict")
            home_id = g.get("home_team_id")
            away_id = g.get("away_team_id")
            predict_btn.clicked.connect(
                lambda checked, h=home_id, a=away_id: self._emit_game(h, a)
            )
            self.table.setCellWidget(row, 4, predict_btn)

        if self.main_window:
            self.main_window.set_status(f"Schedule: {len(filtered)} games loaded")

    def _emit_game(self, home_id, away_id):
        """Emit game_selected signal to switch to matchup tab."""
        if home_id and away_id:
            self.game_selected.emit(home_id, away_id)
            if self.main_window:
                # Switch to matchup tab
                self.main_window.tabs.setCurrentIndex(4)
