"""Lightweight splash screen matching the broadcast theme."""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen
from PySide6.QtWidgets import QSplashScreen


class SplashScreen(QSplashScreen):
    """Dark-themed splash with status messages."""

    _BG = QColor("#0b0f19")
    _ACCENT = QColor("#00e5ff")
    _TEXT = QColor("#e2e8f0")
    _MUTED = QColor("#94a3b8")
    _SURFACE = QColor(20, 30, 45, 200)

    def __init__(self):
        super().__init__()
        self.setFixedSize(440, 280)
        self.setWindowFlags(
            Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self._status_text = ""
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

    def set_status(self, msg: str):
        """Update the status text shown at the bottom."""
        self._status_text = msg
        self.repaint()

    def drawContents(self, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(self.rect(), self._BG)

        # Accent bar at top
        painter.fillRect(0, 0, w, 3, self._ACCENT)

        # Subtle inner panel
        panel_margin = 24
        panel_rect = self.rect().adjusted(
            panel_margin, panel_margin + 10, -panel_margin, -panel_margin
        )
        painter.setBrush(self._SURFACE)
        painter.setPen(QPen(QColor(255, 255, 255, 25)))
        painter.drawRoundedRect(panel_rect, 8, 8)

        # Title
        title_font = QFont(self._font_family, 26, QFont.Bold)
        title_font.setLetterSpacing(QFont.AbsoluteSpacing, 2.0)
        painter.setFont(title_font)
        painter.setPen(self._TEXT)
        title_rect = self.rect().adjusted(0, 55, 0, 0)
        painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignTop, "NBA PREDICTIONS")

        # Accent line under title
        line_y = 105
        line_w = 160
        painter.setPen(QPen(self._ACCENT, 2))
        painter.drawLine((w - line_w) // 2, line_y, (w + line_w) // 2, line_y)

        # Subtitle
        sub_font = QFont(self._font_family, 11)
        sub_font.setLetterSpacing(QFont.AbsoluteSpacing, 3.0)
        painter.setFont(sub_font)
        painter.setPen(self._MUTED)
        sub_rect = self.rect().adjusted(0, 115, 0, 0)
        painter.drawText(sub_rect, Qt.AlignHCenter | Qt.AlignTop, "GAME PREDICTION SYSTEM")

        # Status text at bottom (Segoe UI intentional — regular weight for readability)
        status_font = QFont("Segoe UI", 10)
        painter.setFont(status_font)
        painter.setPen(self._ACCENT)
        status_rect = self.rect().adjusted(30, 0, -30, -25)
        painter.drawText(
            status_rect, Qt.AlignBottom | Qt.AlignHCenter, self._status_text
        )

        # Bottom accent bar
        painter.fillRect(0, h - 3, w, 3, self._ACCENT)
