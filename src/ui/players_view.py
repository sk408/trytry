from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.data.image_cache import get_player_photo_pixmap
from src.database.db import get_conn


def _load_players_df() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT p.player_id, p.name, p.position, t.abbreviation AS team, 
                   p.is_injured, p.injury_note
            FROM players p
            LEFT JOIN teams t ON p.team_id = t.team_id
            ORDER BY t.abbreviation, p.name
            """,
            conn,
        )
    return df


def _load_injured_with_stats() -> pd.DataFrame:
    """Load injured players with their stats for impact assessment."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT 
                p.player_id, 
                p.name, 
                p.position, 
                t.abbreviation AS team,
                p.injury_note,
                COALESCE(AVG(ps.points), 0) as ppg,
                COALESCE(AVG(ps.minutes), 0) as mpg,
                COALESCE(AVG(ps.rebounds), 0) as rpg,
                COALESCE(AVG(ps.assists), 0) as apg,
                COUNT(ps.game_date) as games
            FROM players p
            LEFT JOIN teams t ON p.team_id = t.team_id
            LEFT JOIN player_stats ps ON p.player_id = ps.player_id
            WHERE p.is_injured = 1
            GROUP BY p.player_id, p.name, p.position, t.abbreviation, p.injury_note
            ORDER BY t.abbreviation, COALESCE(AVG(ps.minutes), 0) DESC
            """,
            conn,
        )
    return df


def _get_position_display(position: str) -> str:
    """Get readable position display."""
    pos = (position or "").upper().strip()
    if not pos:
        return "?"
    if pos in ("PG", "SG", "G"):
        return "G"
    if pos in ("SF", "PF", "F"):
        return "F"
    if pos == "C":
        return "C"
    if "-" in pos:
        return pos.split("-")[0]
    return pos[:2] if len(pos) > 2 else pos


class PlayersView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        
        # All players table (left side)
        self.all_table = QTableWidget()
        self.all_table.setAlternatingRowColors(True)
        self.all_table.verticalHeader().setVisible(False)
        self.all_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.all_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Injured players table (right side)
        self.injured_table = QTableWidget()
        self.injured_table.setAlternatingRowColors(True)
        self.injured_table.verticalHeader().setVisible(False)
        self.injured_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.injured_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)  # type: ignore[arg-type]
        
        self.injured_count_label = QLabel("Injured: 0")
        self.injured_count_label.setStyleSheet("font-weight: bold; color: #ef4444;")

        title = QLabel("Players")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")

        # Header
        header = QHBoxLayout()
        header.addWidget(title)
        header.addWidget(self.injured_count_label)
        header.addStretch()
        header.addWidget(self.refresh_button)

        # Left side - All Players
        all_box = QGroupBox("All Players")
        all_layout = QVBoxLayout()
        all_layout.addWidget(self.all_table)
        all_box.setLayout(all_layout)
        
        # Right side - Injured Players
        injured_box = QGroupBox("Injured Players (by Impact)")
        injured_layout = QVBoxLayout()
        injured_layout.addWidget(self.injured_table)
        injured_box.setLayout(injured_layout)
        
        # Split view
        split = QHBoxLayout()
        split.addWidget(all_box, stretch=2)
        split.addWidget(injured_box, stretch=1)

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addLayout(split)
        self.setLayout(layout)

        self.refresh()

    def refresh(self) -> None:
        # Load all players
        df = _load_players_df()
        self._populate_all_table(df)
        
        # Load injured players with stats
        injured_df = _load_injured_with_stats()
        self._populate_injured_table(injured_df)
        
        # Update count
        injured_count = len(injured_df)
        self.injured_count_label.setText(f"Injured: {injured_count}")

    def _populate_all_table(self, df: pd.DataFrame) -> None:
        self.all_table.clear()
        headers = ["Name", "Pos", "Team", "Status"]
        self.all_table.setColumnCount(len(headers))
        self.all_table.setHorizontalHeaderLabels(headers)
        self.all_table.setRowCount(len(df))

        for row_idx, row in enumerate(df.itertuples(index=False)):
            name_item = QTableWidgetItem(str(row.name))
            pos_item = QTableWidgetItem(str(row.position or ""))
            team_item = QTableWidgetItem(str(row.team or ""))
            status = "INJURED" if row.is_injured else "Active"
            status_item = QTableWidgetItem(status)
            
            if row.is_injured:
                name_item.setForeground(QColor("#ef4444"))
                status_item.setForeground(QColor("#ef4444"))
            
            self.all_table.setItem(row_idx, 0, name_item)
            self.all_table.setItem(row_idx, 1, pos_item)
            self.all_table.setItem(row_idx, 2, team_item)
            self.all_table.setItem(row_idx, 3, status_item)

        self.all_table.resizeColumnsToContents()

    def _populate_injured_table(self, df: pd.DataFrame) -> None:
        self.injured_table.clear()
        headers = ["", "Name", "Pos", "Team", "PPG", "MPG", "Injury"]
        self.injured_table.setColumnCount(len(headers))
        self.injured_table.setHorizontalHeaderLabels(headers)
        self.injured_table.setRowCount(len(df))
        self.injured_table.setIconSize(QSize(30, 30))

        for row_idx, row in enumerate(df.itertuples(index=False)):
            self.injured_table.setRowHeight(row_idx, 34)
            ppg = float(row.ppg) if row.ppg else 0.0
            mpg = float(row.mpg) if row.mpg else 0.0
            pos = _get_position_display(row.position)

            # Player headshot
            photo_item = QTableWidgetItem()
            photo_pm = get_player_photo_pixmap(int(row.player_id), 28)
            photo_item.setIcon(QIcon(photo_pm))
            self.injured_table.setItem(row_idx, 0, photo_item)

            name_item = QTableWidgetItem(str(row.name))
            pos_item = QTableWidgetItem(pos)
            team_item = QTableWidgetItem(str(row.team or ""))
            ppg_item = QTableWidgetItem(f"{ppg:.1f}")
            mpg_item = QTableWidgetItem(f"{mpg:.1f}")
            injury_item = QTableWidgetItem(str(row.injury_note or "Unknown"))
            
            # Highlight high-impact injuries (players with >15 MPG)
            if mpg >= 25:
                for it in (name_item, pos_item, ppg_item, mpg_item, injury_item):
                    it.setForeground(QColor("#ef4444"))
                name_item.setToolTip("HIGH IMPACT - Key player")
            elif mpg >= 15:
                for it in (name_item, pos_item):
                    it.setForeground(QColor("#f59e0b"))
                name_item.setToolTip("MODERATE IMPACT - Rotation player")
            
            self.injured_table.setItem(row_idx, 1, name_item)
            self.injured_table.setItem(row_idx, 2, pos_item)
            self.injured_table.setItem(row_idx, 3, team_item)
            self.injured_table.setItem(row_idx, 4, ppg_item)
            self.injured_table.setItem(row_idx, 5, mpg_item)
            self.injured_table.setItem(row_idx, 6, injury_item)

        self.injured_table.setColumnWidth(0, 36)
        self.injured_table.resizeColumnsToContents()
