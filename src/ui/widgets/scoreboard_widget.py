"""Visual scoreboard widget with team logos, scores, quarter breakdown, and game clock."""

import logging
from typing import Optional, Dict, List

from PySide6.QtCore import Qt, QPointF, Property, QVariantAnimation, QObject, QEasingCurve, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QLinearGradient, QPen, QPixmap, QPolygonF, QPainterPath
from PySide6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QGraphicsDropShadowEffect

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
        self._away_timeouts = -1
        self._home_timeouts = -1
        self._away_bonus = False
        self._home_bonus = False

        # Animation state
        self._away_flash_alpha = 0.0
        self._home_flash_alpha = 0.0

        # Animations
        self._anim_away = QVariantAnimation(self)
        self._anim_away.setDuration(800)
        self._anim_away.setStartValue(0.8)
        self._anim_away.setEndValue(0.0)
        self._anim_away.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._anim_away.valueChanged.connect(self._set_away_flash_alpha)

        self._anim_home = QVariantAnimation(self)
        self._anim_home.setDuration(800)
        self._anim_home.setStartValue(0.8)
        self._anim_home.setEndValue(0.0)
        self._anim_home.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._anim_home.valueChanged.connect(self._set_home_flash_alpha)

    def _set_away_flash_alpha(self, val):
        self._away_flash_alpha = val
        self.update()

    def _set_home_flash_alpha(self, val):
        self._home_flash_alpha = val
        self.update()

    def update_data(self, *, away_abbr: str, home_abbr: str,
                    away_team_id: int = None, home_team_id: int = None,
                    away_score: int = 0, home_score: int = 0,
                    away_quarters: list = None, home_quarters: list = None,
                    status_text: str = "", status_state: str = "",
                    clock: str = "", period: int = 0,
                    away_timeouts: int = -1, home_timeouts: int = -1,
                    away_bonus: bool = False, home_bonus: bool = False):
        """Set scoreboard data and trigger repaint. Triggers animations on score change."""
        if self._away_score > 0 and away_score > self._away_score:
            self._anim_away.start()
        if self._home_score > 0 and home_score > self._home_score:
            self._anim_home.start()

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
        self._away_timeouts = away_timeouts
        self._home_timeouts = home_timeouts
        self._away_bonus = away_bonus
        self._home_bonus = home_bonus
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid_x = w / 2.0

        away_clr, _ = get_team_colors(self._away_team_id) if self._away_team_id else ("#3b82f6", "#1e293b")
        home_clr, _ = get_team_colors(self._home_team_id) if self._home_team_id else ("#ef4444", "#1e293b")

        # Background - Dark glass panel
        p.fillRect(0, 0, w, h, QColor(15, 20, 30, 240))
        
        # Angled polygons for modern TV look
        skew = 30.0
        
        # Away Polygon
        away_poly = QPolygonF([
            QPointF(0, 0),
            QPointF(mid_x - skew, 0),
            QPointF(mid_x, h),
            QPointF(0, h)
        ])
        
        # Base Away Gradient
        grad_away = QLinearGradient(0, 0, mid_x, h)
        grad_away.setColorAt(0, QColor(away_clr).darker(300))
        grad_away.setColorAt(1, QColor(away_clr).darker(450))
        
        p.setBrush(grad_away)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(away_poly)
        
        # Blend in flash color if animating
        if self._away_flash_alpha > 0:
            flash_clr = QColor(away_clr)
            flash_clr.setAlphaF(self._away_flash_alpha)
            p.setBrush(flash_clr)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(away_poly)

        # Home Polygon
        home_poly = QPolygonF([
            QPointF(mid_x - skew, 0),
            QPointF(w, 0),
            QPointF(w, h),
            QPointF(mid_x, h)
        ])
        
        # Base Home Gradient
        grad_home = QLinearGradient(mid_x, 0, w, h)
        grad_home.setColorAt(0, QColor(home_clr).darker(300))
        grad_home.setColorAt(1, QColor(home_clr).darker(450))
        
        p.setBrush(grad_home)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(home_poly)

        # Blend in flash color if animating
        if self._home_flash_alpha > 0:
            flash_clr = QColor(home_clr)
            flash_clr.setAlphaF(self._home_flash_alpha)
            p.setBrush(flash_clr)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(home_poly)
            
        # Center separator accent
        p.setPen(QPen(QColor(255, 255, 255, 100), 2))
        p.drawLine(QPointF(mid_x - skew, 0), QPointF(mid_x, h))

        # Bottom glowing borders
        p.fillRect(0, h - 4, int(mid_x - (skew/2)), 4, QColor(away_clr))
        p.fillRect(int(mid_x - (skew/2)), h - 4, int(w - (mid_x - (skew/2))), 4, QColor(home_clr))

        # Typography
        score_font = QFont("Oswald", 48, QFont.Weight.Bold)
        name_font = QFont("Oswald", 18, QFont.Weight.DemiBold)
        small_font = QFont("Oswald", 12)
        tiny_font = QFont("Oswald", 10)

        # ── Away side (left) ──
        logo_size = 72
        away_logo = self._get_logo(self._away_team_id, self._away_abbr, logo_size)
        if away_logo:
            lx = mid_x - 300 - logo_size
            ly = (h - logo_size) // 2 - 10
            p.drawPixmap(int(lx), int(ly), away_logo)

        # Text with subtle drop shadows
        def draw_text_with_shadow(painter, x, y, text, font, color):
            painter.setFont(font)
            # Shadow
            painter.setPen(QColor(0, 0, 0, 180))
            painter.drawText(int(x + 2), int(y + 2), text)
            # Text
            painter.setPen(color)
            painter.drawText(int(x), int(y), text)

        fm_name = p.fontMetrics()

        # Away abbreviation
        p.setFont(name_font)
        away_name_w = fm_name.horizontalAdvance(self._away_abbr)
        away_name_x = (mid_x - 300 - logo_size / 2) - away_name_w / 2
        draw_text_with_shadow(p, away_name_x, h - 30, self._away_abbr, name_font, QColor("#e2e8f0"))

        # Away Bonus and Timeouts
        if self._away_bonus:
            draw_text_with_shadow(p, (mid_x - 300 - logo_size / 2) - 15, h - 10, "BONUS", tiny_font, QColor("#fbbf24"))
        
        if self._away_timeouts >= 0:
            to_start_x = (mid_x - 300 - logo_size / 2) - (7 * 8) / 2
            for t in range(7):
                to_x = to_start_x + (t * 8)
                to_y = h - 20
                if t < self._away_timeouts:
                    p.fillRect(int(to_x), int(to_y), 5, 2, QColor("#ffffff"))
                else:
                    p.fillRect(int(to_x), int(to_y), 5, 2, QColor(255, 255, 255, 50))

        # Away score
        p.setFont(score_font)
        fm_score = p.fontMetrics()
        away_score_text = str(self._away_score)
        sw = fm_score.horizontalAdvance(away_score_text)
        draw_text_with_shadow(p, mid_x - 120 - sw, h // 2 + 20, away_score_text, score_font, QColor("#ffffff"))

        # ── Center VS ──
        draw_text_with_shadow(p, mid_x - 20, h // 2 + 10, "VS", small_font, QColor("#94a3b8"))

        # ── Home side (right) ──
        home_logo = self._get_logo(self._home_team_id, self._home_abbr, logo_size)
        if home_logo:
            lx = mid_x + 300
            ly = (h - logo_size) // 2 - 10
            p.drawPixmap(int(lx), int(ly), home_logo)

        # Home abbreviation
        p.setFont(name_font)
        home_name_w = fm_name.horizontalAdvance(self._home_abbr)
        home_name_x = (mid_x + 300 + logo_size / 2) - home_name_w / 2
        draw_text_with_shadow(p, home_name_x, h - 30, self._home_abbr, name_font, QColor("#e2e8f0"))

        # Home Bonus and Timeouts
        if self._home_bonus:
            draw_text_with_shadow(p, (mid_x + 300 + logo_size / 2) - 15, h - 10, "BONUS", tiny_font, QColor("#fbbf24"))
        
        if self._home_timeouts >= 0:
            to_start_x = (mid_x + 300 + logo_size / 2) - (7 * 8) / 2
            for t in range(7):
                to_x = to_start_x + (t * 8)
                to_y = h - 20
                if t < self._home_timeouts:
                    p.fillRect(int(to_x), int(to_y), 5, 2, QColor("#ffffff"))
                else:
                    p.fillRect(int(to_x), int(to_y), 5, 2, QColor(255, 255, 255, 50))

        # Home score
        draw_text_with_shadow(p, mid_x + 100, h // 2 + 20, str(self._home_score), score_font, QColor("#ffffff"))

        # ── Quarter scores (bottom center) ──
        max_quarters = max(len(self._away_quarters), len(self._home_quarters), 4)
        q_start_x = mid_x - (max_quarters * 32) // 2
        q_y = h - 45

        # Background for quarter scores
        bg_rect = QRectF(q_start_x - 10, q_y - 12, max_quarters * 32 + 50, 40)
        p.fillRect(bg_rect, QColor(0, 0, 0, 150))

        # Quarter headers
        for qi in range(max_quarters):
            qx = q_start_x + qi * 32
            label = f"Q{qi + 1}" if qi < 4 else f"OT{qi - 3}"
            draw_text_with_shadow(p, qx, q_y, label, tiny_font, QColor("#64748b"))

        # Total header
        draw_text_with_shadow(p, q_start_x + max_quarters * 32 + 10, q_y, "T", tiny_font, QColor("#64748b"))

        # Away quarter scores
        for qi in range(len(self._away_quarters)):
            qx = q_start_x + qi * 32
            draw_text_with_shadow(p, qx, q_y + 14, str(self._away_quarters[qi]), tiny_font, QColor("#94a3b8"))
        draw_text_with_shadow(p, q_start_x + max_quarters * 32 + 10, q_y + 14, str(self._away_score), tiny_font, QColor(away_clr))

        # Home quarter scores
        for qi in range(len(self._home_quarters)):
            qx = q_start_x + qi * 32
            draw_text_with_shadow(p, qx, q_y + 26, str(self._home_quarters[qi]), tiny_font, QColor("#94a3b8"))
        draw_text_with_shadow(p, q_start_x + max_quarters * 32 + 10, q_y + 26, str(self._home_score), tiny_font, QColor(home_clr))

        # ── Status / Clock (top center) ──
        # Top banner for time
        p.fillRect(int(mid_x - 80), 0, 160, 32, QColor(10, 15, 25, 220))
        p.setPen(QColor(away_clr))
        p.drawLine(int(mid_x - 80), 32, int(mid_x), 32)
        p.setPen(QColor(home_clr))
        p.drawLine(int(mid_x), 32, int(mid_x + 80), 32)

        if self._status_state == "in":
            # Live indicator
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#22c55e"))
            p.drawEllipse(int(mid_x - 65), 12, 8, 8)
            period_text = f"Q{self._period}" if self._period <= 4 else f"OT{self._period - 4}"
            draw_text_with_shadow(p, mid_x - 45, 22, f"LIVE  {period_text}  {self._clock}", small_font, QColor("#ffffff"))
        elif self._status_state == "post":
            draw_text_with_shadow(p, mid_x - 20, 22, "FINAL", small_font, QColor("#94a3b8"))
        else:
            draw_text_with_shadow(p, mid_x - 40, 22, self._status_text[:20], small_font, QColor("#64748b"))

        p.end()

    def _get_logo(self, team_id: Optional[int], abbr: str, size: int) -> Optional[QPixmap]:
        """Get team logo or fallback to placeholder."""
        if team_id:
            logo = get_team_logo(team_id, size)
            if logo:
                return logo
        primary, _ = get_team_colors(team_id) if team_id else ("#3b82f6", "#1e293b")
        return make_placeholder_logo(abbr, size, primary)
