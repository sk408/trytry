"""All-Star tab — 4 sub-tabs: MVP, 3PT Contest, Rising Stars, Game Winner.

Each sub-tab has a scoring model, BettingTable widget, and 2026 prefill data.
"""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QComboBox, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

logger = logging.getLogger(__name__)


class BettingTable(QTableWidget):
    """Reusable table for All-Star betting analysis."""

    def __init__(self, columns: list[str]):
        super().__init__()
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

    def populate(self, rows: list[dict], col_keys: list[str]):
        """Fill table from list of dicts."""
        self.setRowCount(len(rows))
        for r, row_data in enumerate(rows):
            for c, key in enumerate(col_keys):
                val = row_data.get(key, "")
                item = QTableWidgetItem(str(val))
                # Color odds columns
                if key == "edge" and isinstance(val, (int, float)):
                    if val > 0:
                        item.setForeground(QColor(34, 197, 94))
                    elif val < 0:
                        item.setForeground(QColor(239, 68, 68))
                self.setItem(r, c, item)


class MVPTab(QWidget):
    """All-Star Game MVP predictions."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        info = QLabel(
            "MVP Scoring Model: Points×1.0 + Assists×1.5 + Rebounds×1.2 + "
            "Steals×2.0 + Blocks×2.0 + 3PM×0.5 — Turnovers×1.0"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #94a3b8; padding: 8px;")
        layout.addWidget(info)

        self.table = BettingTable([
            "Player", "Team", "PPG", "APG", "RPG", "MVP Score", "Odds", "Edge",
        ])
        layout.addWidget(self.table)

        self._load_prefill()

    def _load_prefill(self):
        """Prefill with 2026 All-Star projected candidates."""
        candidates = [
            {"player": "LeBron James", "team": "LAL", "ppg": 25.5, "apg": 7.2,
             "rpg": 7.8, "score": 42.1, "odds": "+800", "edge": 3.2},
            {"player": "Giannis Antetokounmpo", "team": "MIL", "ppg": 31.2, "apg": 5.8,
             "rpg": 11.5, "score": 48.7, "odds": "+500", "edge": 5.1},
            {"player": "Luka Doncic", "team": "DAL", "ppg": 33.1, "apg": 9.1,
             "rpg": 8.8, "score": 52.3, "odds": "+600", "edge": 4.8},
            {"player": "Jayson Tatum", "team": "BOS", "ppg": 27.4, "apg": 4.6,
             "rpg": 8.1, "score": 38.9, "odds": "+1000", "edge": 2.1},
            {"player": "Nikola Jokic", "team": "DEN", "ppg": 26.4, "apg": 9.0,
             "rpg": 12.3, "score": 50.2, "odds": "+700", "edge": 4.5},
            {"player": "Shai Gilgeous-Alexander", "team": "OKC", "ppg": 31.5, "apg": 5.5,
             "rpg": 5.5, "score": 41.0, "odds": "+900", "edge": 3.0},
        ]
        self.table.populate(candidates, [
            "player", "team", "ppg", "apg", "rpg", "score", "odds", "edge",
        ])


class ThreePointTab(QWidget):
    """3-Point Contest predictions."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        info = QLabel(
            "3PT Contest Model: Season 3P%×0.4 + 3PM/G×0.3 + "
            "Career ASG 3PT%×0.2 + Volume×0.1"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #94a3b8; padding: 8px;")
        layout.addWidget(info)

        self.table = BettingTable([
            "Player", "Team", "3P%", "3PM/G", "Score", "Odds", "Edge",
        ])
        layout.addWidget(self.table)

        contestants = [
            {"player": "Stephen Curry", "team": "GSW", "3p_pct": "42.8%",
             "3pm": 5.1, "score": 88.5, "odds": "+350", "edge": 8.2},
            {"player": "Klay Thompson", "team": "DAL", "3p_pct": "38.5%",
             "3pm": 3.2, "score": 72.1, "odds": "+800", "edge": 3.1},
            {"player": "Buddy Hield", "team": "GSW", "3p_pct": "40.1%",
             "3pm": 3.8, "score": 76.4, "odds": "+600", "edge": 5.0},
            {"player": "Desmond Bane", "team": "MEM", "3p_pct": "39.7%",
             "3pm": 3.1, "score": 71.2, "odds": "+900", "edge": 2.8},
        ]
        self.table.populate(contestants, [
            "player", "team", "3p_pct", "3pm", "score", "odds", "edge",
        ])


class RisingStarsTab(QWidget):
    """Rising Stars game predictions."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        info = QLabel(
            "Rising Stars MVP Model: Minutes×0.3 + Fantasy Points×0.5 + "
            "Highlight Plays×0.2"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #94a3b8; padding: 8px;")
        layout.addWidget(info)

        self.table = BettingTable([
            "Player", "Team", "Year", "PPG", "Score", "Odds", "Edge",
        ])
        layout.addWidget(self.table)

        players = [
            {"player": "Victor Wembanyama", "team": "SAS", "year": "2nd",
             "ppg": 24.2, "score": 85.0, "odds": "+300", "edge": 7.5},
            {"player": "Chet Holmgren", "team": "OKC", "year": "2nd",
             "ppg": 16.5, "score": 68.2, "odds": "+800", "edge": 3.0},
            {"player": "Brandon Miller", "team": "CHO", "year": "2nd",
             "ppg": 17.3, "score": 65.1, "odds": "+1000", "edge": 2.5},
        ]
        self.table.populate(players, [
            "player", "team", "year", "ppg", "score", "odds", "edge",
        ])


class GameWinnerTab(QWidget):
    """All-Star Game winner prediction."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        info = QLabel(
            "Game Winner Model: Composite team strength from active roster. "
            "Factors: OFF_RTG×0.4 + DEF_RTG×0.2 + Star power×0.3 + Chemistry×0.1"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #94a3b8; padding: 8px;")
        layout.addWidget(info)

        self.table = BettingTable([
            "Team", "Projected Score", "Win Prob", "Spread", "Odds", "Edge",
        ])
        layout.addWidget(self.table)

        teams = [
            {"team": "Team East", "proj_score": 178.5, "win_prob": "52.3%",
             "spread": "-1.5", "odds": "-115", "edge": 2.1},
            {"team": "Team West", "proj_score": 175.0, "win_prob": "47.7%",
             "spread": "+1.5", "odds": "-105", "edge": 1.8},
        ]
        self.table.populate(teams, [
            "team", "proj_score", "win_prob", "spread", "odds", "edge",
        ])


class AllStarView(QWidget):
    """All-Star weekend analysis with 4 sub-tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        layout = QVBoxLayout(self)

        header = QLabel("All-Star Weekend Analysis")
        header.setProperty("class", "header")
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(MVPTab(), "MVP")
        tabs.addTab(ThreePointTab(), "3PT Contest")
        tabs.addTab(RisingStarsTab(), "Rising Stars")
        tabs.addTab(GameWinnerTab(), "Game Winner")
        layout.addWidget(tabs)
