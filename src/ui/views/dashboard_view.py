"""Dashboard tab — 4 stat cards, 6 sync buttons, activity log."""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QGridLayout, QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve

from src.ui.workers import start_sync_worker, start_injury_worker, start_nuke_resync_worker

logger = logging.getLogger(__name__)


class StatCard(QFrame):
    """Broadcast-styled stat card widget."""

    # Different accent colors per card
    _ACCENTS = ["#00e5ff", "#22c55e", "#a78bfa", "#f59e0b"]
    _idx = 0

    def __init__(self, label: str, value: str = "0"):
        super().__init__()
        accent = StatCard._ACCENTS[StatCard._idx % len(StatCard._ACCENTS)]
        StatCard._idx += 1

        self.setProperty("class", "broadcast-card")
        self.setMinimumHeight(90)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(2)

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(
            f"font-size: 32px; font-weight: 700; color: {accent}; "
            f"font-family: 'Oswald'; background: transparent;"
        )
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_label = QLabel(label.upper())
        self.text_label.setStyleSheet(
            "font-size: 10px; font-weight: 700; letter-spacing: 2px; "
            "color: #64748b; background: transparent;"
        )
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.value_label)
        layout.addWidget(self.text_label)

        # Opacity effect deferred to animate_in() to avoid QFont warnings
        self._opacity_effect = None

    def animate_in(self, delay_ms: int = 0):
        """Fade in with optional delay for stagger effect."""
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)
        self._anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._anim.setDuration(500)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(lambda: self.setGraphicsEffect(None))
        if delay_ms > 0:
            QTimer.singleShot(delay_ms, self._anim.start)
        else:
            self._anim.start()

    def set_value(self, value: str):
        self.value_label.setText(value)


class DashboardView(QWidget):
    """Dashboard tab content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QLabel("Dashboard")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # Stat cards — reset accent index so colors are consistent on re-creation
        StatCard._idx = 0
        cards_layout = QGridLayout()
        self.teams_card = StatCard("Teams")
        self.players_card = StatCard("Players")
        self.games_card = StatCard("Game Logs")
        self.injuries_card = StatCard("Injured")

        cards_layout.addWidget(self.teams_card, 0, 0)
        cards_layout.addWidget(self.players_card, 0, 1)
        cards_layout.addWidget(self.games_card, 0, 2)
        cards_layout.addWidget(self.injuries_card, 0, 3)
        layout.addLayout(cards_layout)

        # Sync buttons
        btn_layout = QHBoxLayout()
        btns = [
            ("Full Sync", self._on_full_sync),
            ("Force Full Sync", self._on_force_sync),
            ("Nuke & Resync", self._on_nuke_resync),
            ("Injuries", self._on_injuries),
            ("Injury History", self._on_injury_history),
            ("Team Metrics", self._on_team_metrics),
            ("Player Impact", self._on_player_impact),
            ("Images", self._on_images),
        ]
        for text, handler in btns:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            if text == "Nuke & Resync":
                btn.setProperty("class", "danger")
                btn.setToolTip("Delete ALL synced data and re-fetch everything from scratch")
            btn_layout.addWidget(btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("class", "danger")
        self.stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

        # Activity log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(300)
        layout.addWidget(self.log)

        layout.addStretch()

        # Current worker
        self._worker = None

        # Load initial stats and trigger entrance animations
        self._refresh_stats()
        for i, card in enumerate([
            self.teams_card, self.players_card, self.games_card, self.injuries_card
        ]):
            card.animate_in(delay_ms=i * 120)

    def _refresh_stats(self):
        """Refresh stat card values from DB."""
        try:
            from src.database import db
            self.teams_card.set_value(
                str(db.fetch_one("SELECT COUNT(*) as c FROM teams")["c"])
            )
            self.players_card.set_value(
                str(db.fetch_one("SELECT COUNT(*) as c FROM players")["c"])
            )
            self.games_card.set_value(
                str(db.fetch_one("SELECT COUNT(*) as c FROM player_stats")["c"])
            )
            try:
                self.injuries_card.set_value(
                    str(db.fetch_one("SELECT COUNT(*) as c FROM injuries")["c"])
                )
            except Exception as e:
                logger.warning("Injuries count query failed: %s", e)
                self.injuries_card.set_value("0")
        except Exception as e:
            logger.debug(f"Stats refresh error: {e}")

    def _append_log(self, msg: str):
        """Append colored HTML to activity log."""
        color = "#94a3b8"
        if "error" in msg.lower():
            color = "#ef4444"
        elif "complete" in msg.lower() or "done" in msg.lower():
            color = "#22c55e"
        elif "progress" in msg.lower() or "step" in msg.lower():
            color = "#3b82f6"

        import html
        self.log.append(f'<span style="color:{color}">{html.escape(msg)}</span>')

    def _on_full_sync(self):
        self.log.clear()
        self._worker = start_sync_worker(
            "full", self._append_log, self._on_sync_done
        )

    def _on_force_sync(self):
        self.log.clear()
        self._append_log("Starting FORCE sync — bypassing all freshness checks...")
        self._worker = start_sync_worker(
            "full", self._append_log, self._on_sync_done, force=True
        )

    def _on_nuke_resync(self):
        self.log.clear()
        self._append_log("NUKING all synced data and re-fetching from scratch...")
        self._worker = start_nuke_resync_worker(self._append_log, self._on_sync_done)

    def _on_injuries(self):
        self.log.clear()
        self._worker = start_injury_worker(self._append_log, self._on_sync_done)

    def _on_injury_history(self):
        self.log.clear()
        self._worker = start_sync_worker(
            "injury_history", self._append_log, self._on_sync_done
        )

    def _on_team_metrics(self):
        self.log.clear()
        self._worker = start_sync_worker(
            "team_metrics", self._append_log, self._on_sync_done
        )

    def _on_player_impact(self):
        self.log.clear()
        self._worker = start_sync_worker(
            "player_impact", self._append_log, self._on_sync_done
        )

    def _on_images(self):
        self.log.clear()
        self._worker = start_sync_worker(
            "images", self._append_log, self._on_sync_done
        )

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self._append_log("Stop requested")

    def _on_sync_done(self):
        self._append_log("Operation complete")
        self._refresh_stats()
        if self.main_window:
            self.main_window.set_status("Sync complete")
