"""Scrolling play-by-play feed with mini team logos and color-coded entries."""

import logging
from typing import Optional, Dict, List

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QHBoxLayout, QFrame,
)

from src.ui.widgets.nba_colors import get_team_colors
from src.ui.widgets.image_utils import get_team_logo, make_placeholder_logo

logger = logging.getLogger(__name__)


class _PlayItem(QFrame):
    """Single play-by-play entry with mini logo, clock, and description."""

    def __init__(self, play: Dict, home_team_id: int = None, away_team_id: int = None,
                 home_abbr: str = "", away_abbr: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("playItem")

        is_scoring = play.get("scoringPlay", False)
        team_id = play.get("team_id")
        text = play.get("text", "")
        clock_val = ""
        clock_raw = play.get("clock", {})
        if isinstance(clock_raw, dict):
            clock_val = clock_raw.get("displayValue", "")
        else:
            clock_val = str(clock_raw)
        period = play.get("period", {})
        period_num = period.get("number", 0) if isinstance(period, dict) else 0
        home_score = play.get("homeScore", "")
        away_score = play.get("awayScore", "")

        # Determine team color
        primary_color = "#64748b"
        abbr = ""
        if team_id:
            tid = int(team_id) if team_id else None
            if tid:
                primary_color, _ = get_team_colors(tid)
                if tid == home_team_id:
                    abbr = home_abbr
                elif tid == away_team_id:
                    abbr = away_abbr

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Mini team logo (20px)
        logo_label = QLabel()
        logo_label.setFixedSize(22, 22)
        if team_id:
            tid = int(team_id) if team_id else None
            logo = get_team_logo(tid, 22) if tid else None
            if logo:
                logo_label.setPixmap(logo)
            else:
                placeholder = make_placeholder_logo(abbr, 22, primary_color)
                logo_label.setPixmap(placeholder)
        layout.addWidget(logo_label)

        # Clock / period
        period_label = f"Q{period_num}" if period_num <= 4 else f"OT{period_num - 4}"
        clock_label = QLabel(f"{period_label} {clock_val}")
        clock_label.setFixedWidth(70)
        clock_label.setStyleSheet("color: #64748b; font-size: 10px;")
        layout.addWidget(clock_label)

        # Play text
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        if is_scoring:
            text_label.setStyleSheet(f"color: {primary_color}; font-weight: 600; font-size: 11px;")
        else:
            text_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(text_label, 1)

        # Score after play
        if home_score or away_score:
            score_label = QLabel(f"{away_score}-{home_score}")
            score_label.setFixedWidth(50)
            score_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if is_scoring:
                score_label.setStyleSheet(f"color: {primary_color}; font-weight: 700; font-size: 11px;")
            else:
                score_label.setStyleSheet("color: #475569; font-size: 10px;")
            layout.addWidget(score_label)

        # Styling
        bg = "#111b27" if not is_scoring else "#0d1f12"
        border_color = primary_color if is_scoring else "transparent"
        self.setStyleSheet(f"""
            #playItem {{
                background: {bg};
                border-left: 3px solid {border_color};
                border-radius: 3px;
                margin: 1px 0;
            }}
        """)


class PlayFeedWidget(QWidget):
    """Scrolling play-by-play feed with mini logos and scoring highlights."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._home_team_id: Optional[int] = None
        self._away_team_id: Optional[int] = None
        self._home_abbr = ""
        self._away_abbr = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("  PLAY-BY-PLAY")
        header.setStyleSheet("""
            background: #1e293b; color: #94a3b8; font-size: 10px;
            font-weight: 700; letter-spacing: 1px; padding: 4px 8px;
            text-transform: uppercase;
        """)
        header.setFixedHeight(24)
        layout.addWidget(header)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: #0a1628; }")

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(2, 2, 2, 2)
        self._container_layout.setSpacing(1)
        self._container_layout.addStretch()
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

    def set_teams(self, home_team_id: int, away_team_id: int,
                  home_abbr: str, away_abbr: str):
        self._home_team_id = home_team_id
        self._away_team_id = away_team_id
        self._home_abbr = home_abbr
        self._away_abbr = away_abbr

    def set_plays(self, plays: List[Dict], max_count: int = 80):
        """Replace all plays in the feed."""
        # Clear existing
        while self._container_layout.count() > 1:
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Flatten nested play structures
        flat_plays = []
        for play in plays:
            items = play.get("items", [play])
            for item in items:
                if isinstance(item, dict) and item.get("text"):
                    flat_plays.append(item)

        # Show most recent plays (reversed: newest at top)
        for play in reversed(flat_plays[-max_count:]):
            item_widget = _PlayItem(
                play,
                home_team_id=self._home_team_id,
                away_team_id=self._away_team_id,
                home_abbr=self._home_abbr,
                away_abbr=self._away_abbr,
            )
            # Insert before the stretch
            self._container_layout.insertWidget(self._container_layout.count() - 1, item_widget)

    def clear(self):
        while self._container_layout.count() > 1:
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
