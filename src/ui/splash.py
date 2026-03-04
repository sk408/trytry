"""Lightweight splash screen matching the broadcast theme."""

import os

from PySide6.QtCore import Qt, QRectF, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen, QLinearGradient, QScreen
from PySide6.QtWidgets import QSplashScreen, QApplication, QGraphicsOpacityEffect


class SplashScreen(QSplashScreen):
    """Dark-themed splash with status messages and progress bar."""

    _BG = QColor("#0b0f19")
    _ACCENT = QColor("#00e5ff")
    _TEXT = QColor("#e2e8f0")
    _MUTED = QColor("#94a3b8")
    _SURFACE = QColor(20, 30, 45, 200)
    _TRACK = QColor(30, 41, 59, 180)

    def __init__(self):
        super().__init__()
        # Match the main window's minimum size (1200×800)
        self.setFixedSize(1200, 800)
        self.setWindowFlags(
            Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self._status_text = ""
        self._progress = 0.0  # 0.0 – 1.0
        self._font_family = "Segoe UI"

        # Load Oswald early so the splash can use it
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        oswald_path = os.path.join(font_dir, "Oswald.ttf")
        if os.path.exists(oswald_path):
            fid = QFontDatabase.addApplicationFont(oswald_path)
            if fid != -1:
                families = QFontDatabase.applicationFontFamilies(fid)
                if families:
                    self._font_family = families[0]

        # Center on the primary screen
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(x, y)

    def set_status(self, msg: str, progress: float | None = None):
        """Update the status text and optional progress (0.0–1.0)."""
        self._status_text = msg
        if progress is not None:
            self._progress = max(0.0, min(1.0, progress))
        self.repaint()

    def drawContents(self, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Vertical offset to center the content block in the larger canvas
        cy = h // 2

        # Background
        painter.fillRect(self.rect(), self._BG)

        # Accent bar at top
        painter.fillRect(0, 0, w, 3, self._ACCENT)

        # Subtle inner panel
        margin = 60
        panel_rect = self.rect().adjusted(margin, margin + 10, -margin, -margin)
        painter.setBrush(self._SURFACE)
        painter.setPen(QPen(QColor(255, 255, 255, 25)))
        painter.drawRoundedRect(panel_rect, 10, 10)

        # Title
        title_font = QFont(self._font_family, 44, QFont.Bold)
        title_font.setLetterSpacing(QFont.AbsoluteSpacing, 5.0)
        painter.setFont(title_font)
        painter.setPen(self._TEXT)
        title_y = cy - 70
        title_rect = QRectF(0, title_y, w, 80)
        painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignTop, "NBA PREDICTIONS")

        # Accent line under title
        line_y = title_y + 80
        line_w = 280
        painter.setPen(QPen(self._ACCENT, 2))
        painter.drawLine((w - line_w) // 2, line_y, (w + line_w) // 2, line_y)

        # Subtitle
        sub_font = QFont(self._font_family, 14)
        sub_font.setLetterSpacing(QFont.AbsoluteSpacing, 5.0)
        painter.setFont(sub_font)
        painter.setPen(self._MUTED)
        sub_rect = QRectF(0, line_y + 14, w, 30)
        painter.drawText(sub_rect, Qt.AlignHCenter | Qt.AlignTop, "GAME PREDICTION SYSTEM")

        # Progress bar
        bar_margin = 120
        bar_y = h - 100
        bar_h = 4
        bar_w = w - bar_margin * 2
        bar_radius = 2.0

        # Track
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._TRACK)
        painter.drawRoundedRect(QRectF(bar_margin, bar_y, bar_w, bar_h), bar_radius, bar_radius)

        # Fill with gradient
        if self._progress > 0:
            fill_w = bar_w * self._progress
            grad = QLinearGradient(bar_margin, 0, bar_margin + fill_w, 0)
            grad.setColorAt(0, QColor("#00b8cc"))
            grad.setColorAt(1, self._ACCENT)
            painter.setBrush(grad)
            painter.drawRoundedRect(QRectF(bar_margin, bar_y, fill_w, bar_h), bar_radius, bar_radius)

        # Status text below progress bar
        status_font = QFont("Segoe UI", 11)
        painter.setFont(status_font)
        painter.setPen(self._ACCENT)
        status_rect = QRectF(bar_margin, bar_y + 12, bar_w, 30)
        painter.drawText(
            status_rect, Qt.AlignHCenter | Qt.AlignTop, self._status_text
        )

        # Bottom accent bar
        painter.fillRect(0, h - 3, w, 3, self._ACCENT)

    def crossfade_to(self, window, duration=600):
        """Fade out splash while the main window fades in underneath."""
        # Position the main window behind the splash, start it transparent
        win_effect = QGraphicsOpacityEffect(window)
        win_effect.setOpacity(0.0)
        window.setGraphicsEffect(win_effect)
        window.show()

        # Fade the main window in
        self._win_effect = win_effect  # prevent GC
        win_anim = QPropertyAnimation(win_effect, b"opacity")
        win_anim.setDuration(duration)
        win_anim.setStartValue(0.0)
        win_anim.setEndValue(1.0)
        win_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._win_anim = win_anim

        # Fade the splash out
        splash_effect = QGraphicsOpacityEffect(self)
        splash_effect.setOpacity(1.0)
        self.setGraphicsEffect(splash_effect)
        self._splash_effect = splash_effect
        splash_anim = QPropertyAnimation(splash_effect, b"opacity")
        splash_anim.setDuration(duration)
        splash_anim.setStartValue(1.0)
        splash_anim.setEndValue(0.0)
        splash_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._splash_anim = splash_anim

        # When done, clean up
        def _finish():
            self.close()
            window.setGraphicsEffect(None)

        splash_anim.finished.connect(_finish)

        win_anim.start()
        splash_anim.start()
