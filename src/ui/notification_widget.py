"""Notification bell widget for the desktop status bar.

Shows an unread badge and, when clicked, opens a dropdown panel listing
recent notifications.
"""
from __future__ import annotations

from datetime import datetime
from typing import List

from PySide6.QtCore import Qt, QPoint, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.notifications.models import Notification
from src.notifications import service as notif_service


# ── Severity -> colour mapping ──
_SEVERITY_COLOURS = {
    Notification.SEVERITY_CRITICAL: "#ef4444",
    Notification.SEVERITY_WARNING: "#f59e0b",
    Notification.SEVERITY_INFO: "#3b82f6",
}

_CATEGORY_ICONS = {
    Notification.CATEGORY_INJURY: "\u26A0",        # warning sign
    Notification.CATEGORY_MATCHUP: "\U0001F3C0",   # basketball
    Notification.CATEGORY_INSIGHT: "\U0001F4CA",    # chart
}


class NotificationBell(QPushButton):
    """A bell button that shows an unread count badge.

    Sits in the status bar.  Clicking it toggles the notification panel.
    """

    clicked_bell = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("\U0001F514", parent)  # bell emoji
        self._unread = 0
        self.setFixedSize(36, 28)
        self.setToolTip("Notifications")
        self.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 16px; }"
            "QPushButton:hover { background: #243b53; border-radius: 4px; }"
        )
        self.clicked.connect(self.clicked_bell.emit)  # type: ignore[arg-type]

    @property
    def unread_count(self) -> int:
        return self._unread

    @unread_count.setter
    def unread_count(self, value: int) -> None:
        self._unread = max(0, value)
        self.update()  # trigger repaint

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self._unread <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Badge circle
        badge_size = 16
        x = self.width() - badge_size - 1
        y = 1
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ef4444"))
        painter.drawEllipse(x, y, badge_size, badge_size)
        # Badge text
        painter.setPen(QColor("#ffffff"))
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)
        text = str(self._unread) if self._unread < 100 else "99+"
        painter.drawText(x, y, badge_size, badge_size, Qt.AlignmentFlag.AlignCenter, text)
        painter.end()


class _NotificationCard(QFrame):
    """A single notification row inside the panel."""

    def __init__(self, notification: Notification, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.notification = notification
        colour = _SEVERITY_COLOURS.get(notification.severity, "#3b82f6")
        icon = _CATEGORY_ICONS.get(notification.category, "\u2139")  # info symbol

        bg = "#1c2e42" if notification.read else "#243b53"
        self.setStyleSheet(
            f"QFrame {{ background: {bg}; border-left: 3px solid {colour}; "
            f"border-radius: 4px; margin: 2px 0; padding: 6px 8px; }}"
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)

        # Title row: icon + title + time
        top = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setFixedWidth(18)
        title_lbl = QLabel(notification.title)
        title_lbl.setStyleSheet("font-weight: 600; font-size: 12px; color: #e2e8f0;")
        title_lbl.setWordWrap(True)
        time_lbl = QLabel(_relative_time(notification.created_at))
        time_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        top.addWidget(icon_lbl)
        top.addWidget(title_lbl, stretch=1)
        top.addWidget(time_lbl)

        # Body
        body_lbl = QLabel(notification.body)
        body_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        body_lbl.setWordWrap(True)

        layout.addLayout(top)
        layout.addWidget(body_lbl)
        self.setLayout(layout)


class NotificationPanel(QFrame):
    """Drop-down panel listing recent notifications."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setFixedSize(380, 420)
        self.setStyleSheet(
            "NotificationPanel { background: #172333; border: 1px solid #2a3f55; "
            "border-radius: 8px; }"
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QHBoxLayout()
        title = QLabel("Notifications")
        title.setStyleSheet("font-size: 14px; font-weight: 600; color: #e2e8f0;")
        self.mark_all_btn = QPushButton("Mark all read")
        self.mark_all_btn.setStyleSheet(
            "QPushButton { color: #3b82f6; background: transparent; border: none; "
            "font-size: 11px; } QPushButton:hover { text-decoration: underline; }"
        )
        self.mark_all_btn.clicked.connect(self._mark_all_read)  # type: ignore[arg-type]
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.mark_all_btn)

        # Scrollable notification list
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(4)
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll.setWidget(self.scroll_content)

        # Empty-state label
        self.empty_lbl = QLabel("No notifications yet")
        self.empty_lbl.setStyleSheet("color: #64748b; font-size: 12px; padding: 20px;")
        self.empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(header)
        layout.addWidget(self.scroll, stretch=1)
        layout.addWidget(self.empty_lbl)
        self.setLayout(layout)

    def refresh(self) -> None:
        """Reload notifications from the service."""
        # Clear existing cards
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        notifications = notif_service.get_recent(limit=30)

        if not notifications:
            self.empty_lbl.setVisible(True)
            self.scroll.setVisible(False)
            return

        self.empty_lbl.setVisible(False)
        self.scroll.setVisible(True)

        for n in notifications:
            card = _NotificationCard(n)
            self.scroll_layout.addWidget(card)
        self.scroll_layout.addStretch()

    def _mark_all_read(self) -> None:
        notif_service.mark_all_read()
        self.refresh()

    def show_at(self, global_pos: QPoint) -> None:
        """Show the panel positioned below the bell button."""
        self.refresh()
        # Position near bell, but clamp to screen bounds and flip above when needed.
        screen = QApplication.screenAt(global_pos) or QApplication.primaryScreen()
        geo = screen.availableGeometry() if screen else None
        x = global_pos.x() - self.width() + 36
        y = global_pos.y()
        if geo is not None:
            if y + self.height() > geo.bottom():
                y = max(geo.top(), y - self.height() - 36)
            x = max(geo.left() + 4, min(x, geo.right() - self.width() - 4))
            y = max(geo.top() + 4, min(y, geo.bottom() - self.height() - 4))
        self.move(x, y)
        self.show()


def _relative_time(iso_str: str) -> str:
    """Human-friendly relative time string."""
    try:
        dt = datetime.fromisoformat(iso_str)
        delta = datetime.now() - dt
        secs = int(delta.total_seconds())
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return ""
