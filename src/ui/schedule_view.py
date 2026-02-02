from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from src.data.nba_fetcher import get_current_season
from src.data.sync_service import sync_schedule
from src.database.db import get_conn


def _team_lookup() -> dict[int, str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT team_id, abbreviation FROM teams").fetchall()
    return {tid: abbr for tid, abbr in rows}


class ScheduleView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.table = QTableWidget()
        self.refresh_button = QPushButton("Load schedule")
        self.refresh_button.clicked.connect(self.refresh)  # type: ignore[arg-type]
        self.season_label = QLabel(f"Season: {get_current_season()}")

        header = QHBoxLayout()
        header.addWidget(QLabel("Schedule"))
        header.addWidget(self.season_label)
        header.addStretch()
        header.addWidget(self.refresh_button)

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def refresh(self) -> None:
        lookup = _team_lookup()
        try:
            df = sync_schedule(include_future_days=14)
        except Exception as exc:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {exc}"))
            return
        if df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        df["team"] = df["team_id"].map(lookup)
        # opponent_abbr is already an abbreviation from the API
        df["opponent"] = df["opponent_abbr"]
        df = df[["game_date", "team", "opponent", "is_home"]]

        self.table.clear()
        headers = ["Date", "Team", "Opponent", "Home?"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(df))

        for row_idx, row in enumerate(df.itertuples(index=False)):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(row.game_date)))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(row.team)))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(row.opponent)))
            self.table.setItem(row_idx, 3, QTableWidgetItem("Yes" if row.is_home else "No"))
        self.table.resizeColumnsToContents()
