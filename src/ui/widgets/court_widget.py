"""Animated half-court widget — shows shot chart, ball animation, and live play overlay."""

import math
import logging
from typing import Optional, Dict, List

from PySide6.QtCore import (
    Qt, QPointF, QRectF, QTimer, QPropertyAnimation,
    Property, QEasingCurve, Signal,
)
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QRadialGradient, QPainterPath, QLinearGradient
)
from PySide6.QtWidgets import QWidget

from src.ui.widgets.nba_colors import get_team_colors

logger = logging.getLogger(__name__)

# Court dimensions (NBA half-court in feet: 47 × 50)
_COURT_W = 50.0
_COURT_H = 47.0
_HOOP_X = 25.0
_HOOP_Y = 5.25  # from baseline
_THREE_RADIUS = 23.75
_FT_LINE_Y = 19.0
_KEY_WIDTH = 16.0
_RESTRICTED_RADIUS = 4.0


class CourtWidget(QWidget):
    """Animated NBA half-court with shot chart and ball animation."""

    play_clicked = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 240)

        # State
        self._home_team_id: Optional[int] = None
        self._away_team_id: Optional[int] = None
        self._shots: List[Dict] = []  # {x, y, made, team_id, text}
        self._last_play_text = ""
        self._last_play_team_id: Optional[int] = None

        # Ball animation state
        self._ball_pos = QPointF(25, 30)
        self._ball_visible = False
        self._ball_target = QPointF(25, 5.25)
        self._basket_glow = 0.0  # 0=off, 1=full glow
        self._score_flash_text = ""

        # Animation timers
        self._ball_anim = None
        self._glow_timer = QTimer(self)
        self._glow_timer.setSingleShot(True)
        self._glow_timer.timeout.connect(self._end_glow)

        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)

    # ── Properties for animation ──
    def _get_ball_x(self): return self._ball_pos.x()
    def _set_ball_x(self, v):
        self._ball_pos.setX(v)
        self.update()
    ball_x = Property(float, _get_ball_x, _set_ball_x)

    def _get_ball_y(self): return self._ball_pos.y()
    def _set_ball_y(self, v):
        self._ball_pos.setY(v)
        self.update()
    ball_y = Property(float, _get_ball_y, _set_ball_y)

    def _get_glow(self): return self._basket_glow
    def _set_glow(self, v):
        self._basket_glow = v
        self.update()
    glow = Property(float, _get_glow, _set_glow)

    def set_teams(self, home_team_id: int, away_team_id: int):
        self._home_team_id = home_team_id
        self._away_team_id = away_team_id
        self.update()

    def add_play(self, play: Dict):
        """Add a play-by-play item. Animate if it's a scoring play."""
        is_scoring = play.get("scoringPlay", False)
        is_shooting = play.get("shootingPlay", False)
        team_id = play.get("team_id")
        text = play.get("text", "")
        coord = play.get("coordinate", {})

        self._last_play_text = text
        self._last_play_team_id = team_id

        # ESPN coordinates are already in half-court feet:
        #   x: 0-50 (sideline to sideline, 0=left, 50=right)
        #   y: 0-47 (baseline to half-court, 0=baseline near hoop)
        # Sentinel values (~-2.1e8) appear for jump balls / non-shot plays.
        cx = coord.get("x", 25) if coord else 25
        cy = coord.get("y", 20) if coord else 20

        # Discard sentinel / garbage values (ESPN uses ~-2.1e8 for n/a)
        if cx < -100 or cx > 200 or cy < -100 or cy > 200:
            cx, cy = 25, 20  # default to mid-court

        shot_x = max(0, min(cx, _COURT_W))      # clamp to court width
        shot_y = max(0, min(cy, _COURT_H - 1))   # clamp to court depth

        if is_shooting or is_scoring:
            self._shots.append({
                "x": shot_x, "y": shot_y,
                "made": is_scoring,
                "team_id": team_id,
                "text": text[:40],
            })
            # Keep only last 30 shots
            if len(self._shots) > 30:
                self._shots = self._shots[-30:]

        if is_scoring:
            self._animate_score(shot_x, shot_y, text)

        self.update()

    def clear_shots(self):
        """Clear all shot markers."""
        self._shots.clear()
        self._last_play_text = ""
        self.update()

    def _animate_score(self, from_x: float, from_y: float, text: str):
        """Animate ball from shot location to basket."""
        self._ball_pos = QPointF(from_x, from_y)
        self._ball_visible = True
        self._score_flash_text = text[:30]

        # Animate ball X
        if self._ball_anim and self._ball_anim.state() == QPropertyAnimation.State.Running:
            self._ball_anim.stop()

        self._ball_anim_x = QPropertyAnimation(self, b"ball_x")
        self._ball_anim_x.setDuration(600)
        self._ball_anim_x.setStartValue(from_x)
        self._ball_anim_x.setEndValue(_HOOP_X)
        self._ball_anim_x.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._ball_anim_x.start()

        anim_y = QPropertyAnimation(self, b"ball_y")
        anim_y.setDuration(600)
        anim_y.setStartValue(from_y)
        anim_y.setEndValue(_HOOP_Y)
        anim_y.setEasingCurve(QEasingCurve.Type.OutBounce)
        anim_y.finished.connect(self._on_ball_arrived)
        anim_y.start()
        self._ball_anim = anim_y  # prevent GC

    def _on_ball_arrived(self):
        """Ball reached basket — glow effect."""
        self._ball_visible = False
        self._basket_glow = 1.0

        glow_anim = QPropertyAnimation(self, b"glow")
        glow_anim.setDuration(800)
        glow_anim.setStartValue(1.0)
        glow_anim.setEndValue(0.0)
        glow_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        glow_anim.start()
        self._glow_anim = glow_anim  # prevent GC

        self._flash_timer.start(2000)
        self.update()

    def _end_glow(self):
        self._basket_glow = 0.0
        self.update()

    def _end_flash(self):
        self._score_flash_text = ""
        self.update()

    # ── Coordinate mapping ──
    def _court_to_px(self, cx: float, cy: float) -> QPointF:
        """Map court coords (feet) to widget pixel coords."""
        w, h = self.width(), self.height()
        margin = 10
        pw = w - 2 * margin
        ph = h - 2 * margin
        px = margin + (cx / _COURT_W) * pw
        py = margin + (cy / _COURT_H) * ph
        return QPointF(px, py)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor("#0a1628"))

        margin = 10
        court_rect = QRectF(margin, margin, w - 2 * margin, h - 2 * margin)

        # Court floor
        p.setPen(Qt.PenStyle.NoPen)
        # Use a hardwood-like gradient or a stylized dark court
        floor_grad = QLinearGradient(0, 0, 0, h)
        floor_grad.setColorAt(0, QColor("#1e293b"))
        floor_grad.setColorAt(1, QColor("#0f172a"))
        p.setBrush(floor_grad)
        p.drawRect(court_rect)

        # Court lines
        line_pen = QPen(QColor(255, 255, 255, 100), 2.0)
        p.setPen(line_pen)

        # Baseline
        bl = self._court_to_px(0, 0)
        br = self._court_to_px(_COURT_W, 0)
        p.drawLine(bl, br)

        # Half-court line
        hl = self._court_to_px(0, _COURT_H)
        hr = self._court_to_px(_COURT_W, _COURT_H)
        p.drawLine(hl, hr)

        # Sidelines
        p.drawLine(bl, hl)
        p.drawLine(br, hr)

        # Center circle (at half court)
        px_per_foot_x = (w - 2 * margin) / _COURT_W
        px_per_foot_y = (h - 2 * margin) / _COURT_H
        center_rx = 6.0 * px_per_foot_x
        center_ry = 6.0 * px_per_foot_y
        center_rect = QRectF(hl.x() + (w - 2 * margin) / 2 - center_rx, hl.y() - center_ry, center_rx * 2, center_ry * 2)
        
        home_clr, _ = get_team_colors(self._home_team_id) if self._home_team_id else ("#3b82f6", "")
        p.setBrush(QColor(home_clr))
        # Draw bottom half of center circle
        p.drawChord(center_rect, 0, 180 * 16)
        
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Key / paint area
        key_left = (_COURT_W - _KEY_WIDTH) / 2.0
        key_right = key_left + _KEY_WIDTH
        kl_bl = self._court_to_px(key_left, 0)
        kr_bl = self._court_to_px(key_right, 0)
        kl_ft = self._court_to_px(key_left, _FT_LINE_Y)
        kr_ft = self._court_to_px(key_right, _FT_LINE_Y)
        
        # Fill the paint area with team color slightly transparent
        paint_color = QColor(home_clr)
        paint_color.setAlpha(40)
        p.fillRect(QRectF(kl_bl, kr_ft), paint_color)

        p.setPen(line_pen)
        p.drawLine(kl_bl, kl_ft)
        p.drawLine(kr_bl, kr_ft)
        p.drawLine(kl_ft, kr_ft)

        # Free throw circle (top half)
        ft_center = self._court_to_px(_HOOP_X, _FT_LINE_Y)
        ft_radius_px_x = 6.0 * px_per_foot_x
        ft_radius_px_y = 6.0 * px_per_foot_y
        ft_rect = QRectF(ft_center.x() - ft_radius_px_x, ft_center.y() - ft_radius_px_y,
                         ft_radius_px_x * 2, ft_radius_px_y * 2)
        p.drawArc(ft_rect, 0, 180 * 16)  # top half
        
        # Free throw circle (bottom half dashed)
        dash_pen = QPen(QColor(255, 255, 255, 100), 2.0)
        dash_pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(dash_pen)
        p.drawArc(ft_rect, 180 * 16, 180 * 16)

        # 3-point arc
        arc_pen = QPen(QColor(255, 255, 255, 100), 2.0)
        p.setPen(arc_pen)
        hoop_px = self._court_to_px(_HOOP_X, _HOOP_Y)
        three_rx = _THREE_RADIUS * px_per_foot_x
        three_ry = _THREE_RADIUS * px_per_foot_y
        arc_rect = QRectF(hoop_px.x() - three_rx, hoop_px.y() - three_ry,
                          three_rx * 2, three_ry * 2)
        # Draw arc from ~-68° to ~248° (roughly front-facing arc)
        start_angle = int(-68 * 16)
        span_angle = int(316 * 16)
        p.drawArc(arc_rect, start_angle, span_angle)

        # Corner 3 lines (straight parts, 14ft from baseline on each side)
        left_corner_top = self._court_to_px(3, 14)
        left_corner_bot = self._court_to_px(3, 0)
        right_corner_top = self._court_to_px(_COURT_W - 3, 14)
        right_corner_bot = self._court_to_px(_COURT_W - 3, 0)
        p.drawLine(left_corner_bot, left_corner_top)
        p.drawLine(right_corner_bot, right_corner_top)

        # Restricted area arc
        ra_rx = _RESTRICTED_RADIUS * px_per_foot_x
        ra_ry = _RESTRICTED_RADIUS * px_per_foot_y
        ra_rect = QRectF(hoop_px.x() - ra_rx, hoop_px.y() - ra_ry, ra_rx * 2, ra_ry * 2)
        p.setPen(QPen(QColor("#334155"), 1))
        p.drawArc(ra_rect, 0, 180 * 16)

        # Basket (hoop)
        hoop_r = max(3, int(0.75 * px_per_foot_x))
        if self._basket_glow > 0:
            glow_color = QColor("#22c55e")
            glow_color.setAlphaF(self._basket_glow * 0.6)
            glow_grad = QRadialGradient(hoop_px, hoop_r * 6)
            glow_grad.setColorAt(0, glow_color)
            glow_grad.setColorAt(1, QColor(0, 0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(glow_grad)
            p.drawEllipse(hoop_px, hoop_r * 6, hoop_r * 6)

        p.setPen(QPen(QColor("#ef4444"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(hoop_px, hoop_r, hoop_r)

        # Backboard
        bb_w = 3.0 * px_per_foot_x
        bb_y = self._court_to_px(0, 4).y()
        p.setPen(QPen(QColor("#94a3b8"), 2))
        p.drawLine(QPointF(hoop_px.x() - bb_w / 2, bb_y),
                   QPointF(hoop_px.x() + bb_w / 2, bb_y))

        # ── Shot markers ──
        for shot in self._shots:
            sx, sy = shot["x"], shot["y"]
            sp = self._court_to_px(sx, sy)
            if shot["made"]:
                clr = QColor("#22c55e")
                clr.setAlpha(180)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(clr)
                p.drawEllipse(sp, 4, 4)
            else:
                clr = QColor("#ef4444")
                clr.setAlpha(140)
                p.setPen(QPen(clr, 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                # X mark
                p.drawLine(QPointF(sp.x() - 3, sp.y() - 3), QPointF(sp.x() + 3, sp.y() + 3))
                p.drawLine(QPointF(sp.x() + 3, sp.y() - 3), QPointF(sp.x() - 3, sp.y() + 3))

        # ── Ball animation ──
        if self._ball_visible:
            bp = self._court_to_px(self._ball_pos.x(), self._ball_pos.y())
            ball_r = max(5, int(1.0 * px_per_foot_x))
            ball_grad = QRadialGradient(bp, ball_r)
            ball_grad.setColorAt(0, QColor("#f97316"))
            ball_grad.setColorAt(1, QColor("#c2410c"))
            p.setPen(QPen(QColor("#7c2d12"), 1))
            p.setBrush(ball_grad)
            p.drawEllipse(bp, ball_r, ball_r)
            # Ball stripes
            p.setPen(QPen(QColor("#1c1917"), 0.5))
            p.drawLine(QPointF(bp.x() - ball_r * 0.7, bp.y()),
                       QPointF(bp.x() + ball_r * 0.7, bp.y()))

        # ── Score flash overlay ──
        if self._score_flash_text:
            p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            p.setPen(QColor("#22c55e"))
            text_rect = QRectF(10, h - 35, w - 20, 25)
            p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._score_flash_text)

        # ── Last play text (top) ──
        if self._last_play_text and not self._score_flash_text:
            p.setFont(QFont("Segoe UI", 9))
            p.setPen(QColor("#94a3b8"))
            text_rect = QRectF(10, h - 30, w - 20, 20)
            p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter,
                       self._last_play_text[:60])

        p.end()
