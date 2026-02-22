"""Visual scoreboard widget with team logos, scores, quarter breakdown, and game clock."""

import logging
from typing import Optional, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QLinearGradient, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from src.ui.widgets.nba_colors import get_team_colors
from src.ui.widgets.image_utils import get_team_logo, make_placeholder_logo

logger = logging.getLogger(__name__)


class ScoreboardWidget(QWidget):
    """TV-style scoreboard with team logos, scores, quarters, and clock."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setMaximumHeight(200)

        # State
        self._away_abbr = ""
        self._home_abbr = ""
        self._away_team_id: Optional[int] = None
        self._home_team_id: Optional[int] = None
        self._away_score = 0
        self._home_score = 0
        self._away_quarters: List[int] = []
        self._home_quarters: List[int] = []
        self._status_text = ""
        self._status_state = ""
        self._clock = ""
        self._period = 0

    def update_data(self, *, away_abbr: str, home_abbr: str,
                    away_team_id: int = None, home_team_id: int = None,
                    away_score: int = 0, home_score: int = 0,
                    away_quarters: list = None, home_quarters: list = None,
                    status_text: str = "", status_state: str = "",
                    clock: str = "", period: int = 0):
        """Set scoreboard data and trigger repaint."""
        self._away_abbr = away_abbr
        self._home_abbr = home_abbr
        self._away_team_id = away_team_id
        self._home_team_id = home_team_id
        self._away_score = away_score
        self._home_score = home_score
        self._away_quarters = away_quarters or []
        self._home_quarters = home_quarters or []
        self._status_text = status_text
        self._status_state = status_state
        self._clock = clock
        self._period = period
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background gradient
        grad = QLinearGradient(0, 0, w, 0)
        away_clr, _ = get_team_colors(self._away_team_id) if self._away_team_id else ("#3b82f6", "#1e293b")
        home_clr, _ = get_team_colors(self._home_team_id) if self._home_team_id else ("#3b82f6", "#1e293b")
        grad.setColorAt(0, QColor(away_clr).darker(350))
        grad.setColorAt(0.45, QColor("#0f1923"))
        grad.setColorAt(0.55, QColor("#0f1923"))
        grad.setColorAt(1, QColor(home_clr).darker(350))
        p.fillRect(0, 0, w, h, grad)

        # Bottom border accent
        p.fillRect(0, h - 3, w // 2, 3, QColor(away_clr))
        p.fillRect(w // 2, h - 3, w // 2, 3, QColor(home_clr))

        mid_x = w // 2
        logo_size = min(64, h - 40)
        score_font = QFont("Segoe UI", 36, QFont.Weight.Bold)
        name_font = QFont("Segoe UI", 14, QFont.Weight.DemiBold)
        small_font = QFont("Segoe UI", 11)
        tiny_font = QFont("Segoe UI", 9)

        # ── Away side (left) ──
        away_logo = self._get_logo(self._away_team_id, self._away_abbr, logo_size)
        if away_logo:
            lx = mid_x - 240 - logo_size
            ly = (h - logo_size) // 2 - 10
            p.drawPixmap(lx, ly, away_logo)

        # Away abbreviation
        p.setFont(name_font)
        p.setPen(QColor("#94a3b8"))
        p.drawText(mid_x - 240 - logo_size - 5, h - 25, self._away_abbr)

        # Away score
        p.setFont(score_font)
        p.setPen(QColor("#e2e8f0"))
        away_score_text = str(self._away_score)
        fm = p.fontMetrics()
        sw = fm.horizontalAdvance(away_score_text)
        p.drawText(mid_x - 70 - sw, h // 2 + 14, away_score_text)

        # ── Center divider ──
        p.setPen(QColor("#334155"))
        p.drawLine(mid_x, 10, mid_x, h - 30)
        # "VS" or dash
        p.setFont(QFont("Segoe UI", 10))
        p.setPen(QColor("#64748b"))
        p.drawText(mid_x - 6, h // 2 + 4, "—")

        # ── Home side (right) ──
        home_logo = self._get_logo(self._home_team_id, self._home_abbr, logo_size)
        if home_logo:
            lx = mid_x + 240
            ly = (h - logo_size) // 2 - 10
            p.drawPixmap(lx, ly, home_logo)

        # Home abbreviation
        p.setFont(name_font)
        p.setPen(QColor("#94a3b8"))
        home_abbr_w = fm.horizontalAdvance(self._home_abbr)
        p.drawText(mid_x + 240 + logo_size + 5, h - 25, self._home_abbr)

        # Home score
        p.setFont(score_font)
        p.setPen(QColor("#e2e8f0"))
        p.drawText(mid_x + 70, h // 2 + 14, str(self._home_score))

        # ── Quarter scores (bottom center) ──
        max_quarters = max(len(self._away_quarters), len(self._home_quarters), 4)
        q_start_x = mid_x - (max_quarters * 28) // 2
        q_y = h - 50

        p.setFont(tiny_font)
        # Quarter headers
        for qi in range(max_quarters):
            qx = q_start_x + qi * 28
            label = f"Q{qi + 1}" if qi < 4 else f"OT{qi - 3}"
            p.setPen(QColor("#64748b"))
            p.drawText(qx, q_y - 2, label)

        # Total header
        p.drawText(q_start_x + max_quarters * 28 + 8, q_y - 2, "T")

        # Away quarter scores
        p.setPen(QColor("#94a3b8"))
        for qi in range(len(self._away_quarters)):
            qx = q_start_x + qi * 28
            p.drawText(qx, q_y + 12, str(self._away_quarters[qi]))
        p.setPen(QColor("#e2e8f0"))
        p.drawText(q_start_x + max_quarters * 28 + 4, q_y + 12, str(self._away_score))

        # Home quarter scores
        p.setPen(QColor("#94a3b8"))
        for qi in range(len(self._home_quarters)):
            qx = q_start_x + qi * 28
            p.drawText(qx, q_y + 24, str(self._home_quarters[qi]))
        p.setPen(QColor("#e2e8f0"))
        p.drawText(q_start_x + max_quarters * 28 + 4, q_y + 24, str(self._home_score))

        # ── Status / Clock (top center) ──
        p.setFont(small_font)
        if self._status_state == "in":
            # Live indicator
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#22c55e"))
            p.drawEllipse(mid_x - 50, 12, 8, 8)
            p.setPen(QColor("#22c55e"))
            period_text = f"Q{self._period}" if self._period <= 4 else f"OT{self._period - 4}"
            p.drawText(mid_x - 38, 20, f"LIVE  {period_text}  {self._clock}")
        elif self._status_state == "post":
            p.setPen(QColor("#94a3b8"))
            p.drawText(mid_x - 25, 20, "FINAL")
        else:
            p.setPen(QColor("#64748b"))
            p.drawText(mid_x - 40, 20, self._status_text[:20])

        p.end()

    def _get_logo(self, team_id: Optional[int], abbr: str, size: int) -> Optional[QPixmap]:
        """Get team logo or fallback to placeholder."""
        if team_id:
            logo = get_team_logo(team_id, size)
            if logo:
                return logo
        primary, _ = get_team_colors(team_id) if team_id else ("#3b82f6", "#1e293b")
        return make_placeholder_logo(abbr, size, primary)
