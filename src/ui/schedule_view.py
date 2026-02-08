from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from src.data.nba_fetcher import get_current_season
from src.data.sync_service import sync_schedule
from src.database.db import get_conn


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

        # Store game data for lookup when user clicks a row
        self._games_data = []
        for _, row in df.iterrows():
            self._games_data.append({
                "home_team_id": int(row.get("home_team_id", 0) or 0),
                "away_team_id": int(row.get("away_team_id", 0) or 0),
                "game_date": str(row.get("game_date", "")),
                "home_abbr": str(row.get("home_abbr", "")),
                "away_abbr": str(row.get("away_abbr", "")),
                "game_time": str(row.get("game_time", "")),
                "arena": str(row.get("arena", "")),
            })

        self.table.clear()
        headers = ["Date", "Away", "@", "Home", "Time", "Arena"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self._games_data))

        for row_idx, g in enumerate(self._games_data):
            self.table.setItem(row_idx, 0, QTableWidgetItem(g["game_date"]))
            self.table.setItem(row_idx, 1, QTableWidgetItem(g["away_abbr"]))
            self.table.setItem(row_idx, 2, QTableWidgetItem("@"))
            self.table.setItem(row_idx, 3, QTableWidgetItem(g["home_abbr"]))
            self.table.setItem(row_idx, 4, QTableWidgetItem(g["game_time"]))
            self.table.setItem(row_idx, 5, QTableWidgetItem(g["arena"]))
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
