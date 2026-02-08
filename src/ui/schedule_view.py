from __future__ import annotations

from datetime import date

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.data.nba_fetcher import get_current_season
from src.data.image_cache import get_team_logo_pixmap
from src.data.sync_service import sync_schedule
from src.database.db import get_conn


def _relative_date_label(game_date: date, today: date) -> str:
    """Return a human-friendly relative date label.

    Examples: 'Today', 'Tomorrow', '+2 days', 'Yesterday', '-3 days'.
    """
    delta = (game_date - today).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Tomorrow"
    if delta == -1:
        return "Yesterday"
    if delta > 1:
        return f"+{delta} days"
    # delta < -1
    return f"{delta} days"


class ScheduleView(QWidget):
    # Signal emitted when user double-clicks a game: (home_team_id, away_team_id)
    game_selected = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self._games_data: list[dict] = []

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self._on_double_click)  # type: ignore[arg-type]

        self.refresh_button = QPushButton("Load schedule")
        self.refresh_button.clicked.connect(self.refresh)  # type: ignore[arg-type]

        self.open_matchup_btn = QPushButton("Open in Matchups")
        self.open_matchup_btn.clicked.connect(self._open_selected)  # type: ignore[arg-type]

        self.season_label = QLabel(f"Season: {get_current_season()}")

        header = QHBoxLayout()
        header.addWidget(QLabel("Schedule"))
        header.addWidget(self.season_label)
        header.addStretch()
        header.addWidget(self.open_matchup_btn)
        header.addWidget(self.refresh_button)

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def refresh(self) -> None:
        try:
            df = sync_schedule(include_future_days=14)
        except Exception as exc:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {exc}"))
            self._games_data = []
            return
        if df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self._games_data = []
            return

        today = date.today()

        # Store game data for lookup when user clicks a row
        self._games_data = []
        for _, row in df.iterrows():
            raw_date = row.get("game_date", "")
            # Normalise to a date object so we can sort and label
            if isinstance(raw_date, date):
                gd = raw_date
            else:
                try:
                    gd = date.fromisoformat(str(raw_date)[:10])
                except (ValueError, TypeError):
                    gd = today  # fallback

            self._games_data.append({
                "home_team_id": int(row.get("home_team_id", 0) or 0),
                "away_team_id": int(row.get("away_team_id", 0) or 0),
                "game_date_obj": gd,
                "game_date_str": gd.strftime("%a %m/%d"),
                "date_label": _relative_date_label(gd, today),
                "home_abbr": str(row.get("home_abbr", "")),
                "away_abbr": str(row.get("away_abbr", "")),
                "game_time": str(row.get("game_time", "")),
                "arena": str(row.get("arena", "")),
            })

        # Sort: today first, then ascending future, then most-recent past
        def _sort_key(g: dict) -> tuple:
            delta = (g["game_date_obj"] - today).days
            if delta == 0:
                return (0, 0)
            if delta > 0:
                return (1, delta)          # closest future first
            return (2, abs(delta))         # most-recent past first

        self._games_data.sort(key=_sort_key)

        self.table.clear()
        headers = ["Day", "Date", "Away", "@", "Home", "Time", "Arena"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self._games_data))
        self.table.setIconSize(QSize(22, 22))

        bold_font = QFont()
        bold_font.setBold(True)
        today_bg = QColor(45, 80, 140)       # subtle blue highlight for today
        tomorrow_bg = QColor(55, 70, 95)     # dimmer highlight for tomorrow

        for row_idx, g in enumerate(self._games_data):
            day_item = QTableWidgetItem(g["date_label"])
            date_item = QTableWidgetItem(g["game_date_str"])

            # Away team with logo
            away_item = QTableWidgetItem(g["away_abbr"])
            away_tid = g.get("away_team_id", 0)
            if away_tid:
                away_item.setIcon(QIcon(get_team_logo_pixmap(away_tid, 22)))

            at_item = QTableWidgetItem("@")

            # Home team with logo
            home_item = QTableWidgetItem(g["home_abbr"])
            home_tid = g.get("home_team_id", 0)
            if home_tid:
                home_item.setIcon(QIcon(get_team_logo_pixmap(home_tid, 22)))

            time_item = QTableWidgetItem(g["game_time"])
            arena_item = QTableWidgetItem(g["arena"])

            row_items = [day_item, date_item, away_item, at_item,
                         home_item, time_item, arena_item]

            # Highlight today / tomorrow rows
            if g["date_label"] == "Today":
                for item in row_items:
                    item.setFont(bold_font)
                    item.setBackground(today_bg)
            elif g["date_label"] == "Tomorrow":
                for item in row_items:
                    item.setBackground(tomorrow_bg)

            self.table.setItem(row_idx, 0, day_item)
            self.table.setItem(row_idx, 1, date_item)
            self.table.setItem(row_idx, 2, away_item)
            self.table.setItem(row_idx, 3, at_item)
            self.table.setItem(row_idx, 4, home_item)
            self.table.setItem(row_idx, 5, time_item)
            self.table.setItem(row_idx, 6, arena_item)

        self.table.resizeColumnsToContents()

    def _on_double_click(self, index) -> None:
        row = index.row()
        if 0 <= row < len(self._games_data):
            g = self._games_data[row]
            home_id = g.get("home_team_id", 0)
            away_id = g.get("away_team_id", 0)
            if home_id and away_id:
                self.game_selected.emit(home_id, away_id)

    def _open_selected(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._games_data):
            g = self._games_data[row]
            home_id = g.get("home_team_id", 0)
            away_id = g.get("away_team_id", 0)
            if home_id and away_id:
                self.game_selected.emit(home_id, away_id)
