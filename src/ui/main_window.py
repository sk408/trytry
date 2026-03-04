"""PySide6 Desktop GUI — Main window with 10 tabs."""

import logging
import os
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QApplication, QWidget,
    QVBoxLayout, QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFontDatabase, QFont
from src.ui.theme import GLOBAL_STYLESHEET

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window with 10 tabs."""

    def __init__(self):
        super().__init__()

        # Load bundled Oswald font
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        oswald_path = os.path.join(font_dir, "Oswald.ttf")
        if os.path.exists(oswald_path):
            font_id = QFontDatabase.addApplicationFont(oswald_path)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    app_font = QFont(families[0], 10)
                    QApplication.setFont(app_font)

        self.setWindowTitle("NBA Game Prediction System")
        self.setMinimumSize(1200, 800)
        from src.ui.theme import setup_theme
        setup_theme(self)

        # Central widget with tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Initialize tabs (lazy import to avoid circular deps)
        self._init_tabs()

        # Tab crossfade transition
        self._tab_fade_effect = None
        self._tab_fade_anim = None
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Notification bell
        self._init_notifications()

    def _init_tabs(self):
        """Create all tabs."""
        from src.ui.views.dashboard_view import DashboardView
        from src.ui.views.gamecast_view import GamecastView
        from src.ui.views.players_view import PlayersView
        from src.ui.views.matchup_view import MatchupView
        from src.ui.views.schedule_view import ScheduleView
        from src.ui.views.accuracy_view import AccuracyView
        from src.ui.views.automatic_view import AutomaticView
        from src.ui.views.sensitivity_view import SensitivityView
        from src.ui.views.overview_view import OverviewView
        from src.ui.views.tools_view import ToolsView

        self.dashboard = DashboardView(self)
        self.overview = OverviewView(self)
        self.gamecast = GamecastView(self)
        self.players = PlayersView(self)
        self.matchup = MatchupView(self)
        self.schedule = ScheduleView(self)
        self.accuracy = AccuracyView(self)
        self.automatic = AutomaticView(self)
        self.sensitivity = SensitivityView(self)
        self.tools = ToolsView(self)

        self.tabs.addTab(self.dashboard, "Dashboard")
        self.tabs.addTab(self.overview, "Today's Overview")
        self.tabs.addTab(self.gamecast, "Gamecast")
        self.tabs.addTab(self.players, "Players")
        self.tabs.addTab(self.matchup, "Matchups")
        self.tabs.addTab(self.schedule, "Schedule")
        self.tabs.addTab(self.accuracy, "Accuracy")
        self.tabs.addTab(self.automatic, "Automatic")
        self.tabs.addTab(self.sensitivity, "Sensitivity")
        self.tabs.addTab(self.tools, "Tools")

        # Connect schedule game selection to matchup tab
        self.schedule.game_selected.connect(self._on_schedule_game_selected)

    def _on_schedule_game_selected(self, home_id: int, away_id: int):
        """Switch to matchup tab and set selected teams."""
        # Set teams on matchup view
        for i in range(self.matchup.home_combo.count()):
            if self.matchup.home_combo.itemData(i) == home_id:
                self.matchup.home_combo.setCurrentIndex(i)
                break
        for i in range(self.matchup.away_combo.count()):
            if self.matchup.away_combo.itemData(i) == away_id:
                self.matchup.away_combo.setCurrentIndex(i)
                break
        # Switch to matchup tab
        self.tabs.setCurrentWidget(self.matchup)
        self.matchup._on_predict()

    def _init_notifications(self):
        """Set up notification bell in the tab bar corner."""
        from src.ui.notification_widget import NotificationBell
        self.notif_bell = NotificationBell(self)
        self.tabs.setCornerWidget(self.notif_bell, Qt.Corner.TopRightCorner)

    def _on_tab_changed(self, index: int):
        """Apply a quick fade-in when switching tabs."""
        widget = self.tabs.widget(index)
        if not widget:
            return
        # Stop any running tab animation before starting a new one
        if self._tab_fade_anim is not None:
            self._tab_fade_anim.stop()
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.3)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(250)
        anim.setStartValue(0.3)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        # Store refs so they aren't garbage collected mid-animation
        self._tab_fade_effect = effect
        self._tab_fade_anim = anim
        anim.finished.connect(lambda w=widget: w.setGraphicsEffect(None))
        anim.start()

    def set_status(self, msg: str):
        """Update status bar message."""
        self.status_bar.showMessage(msg)

    def closeEvent(self, event):
        """Clean up on close — signal optimizer to save results before exiting."""
        logger.info("Application closing")
        if hasattr(self, 'gamecast'):
            self.gamecast._stop_websocket()

        # If an optimization/pipeline worker is running, signal cancellation
        # so Optuna finishes its current trial and the save gate runs.
        if hasattr(self, 'automatic') and self.automatic._current_worker:
            worker = self.automatic._current_worker
            if worker.isRunning():
                from src.analytics.pipeline import request_cancel
                logger.info("Optimization running — requesting graceful stop…")
                self.status_bar.showMessage("Saving optimization results…")
                request_cancel()
                worker.stop()
                # Wait up to 10s for the save gate to finish
                if worker._thread_ref is not None:
                    worker._thread_ref.wait(10000)

        event.accept()
