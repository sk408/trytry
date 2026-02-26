"""PySide6 Desktop GUI — Main window with 10 tabs."""

import logging
import os
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QApplication, QWidget, QVBoxLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QFont

from src.ui.theme import GLOBAL_STYLESHEET

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window with 10 tabs."""

    def __init__(self):
        super().__init__()
        
        # Load custom fonts
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
        from src.ui.views.snapshots_view import SnapshotsView
        from src.ui.views.autotune_view import AutotuneView
        from src.ui.views.sensitivity_view import SensitivityView
        from src.ui.views.admin_view import AdminView
        from src.ui.views.overview_view import OverviewView

        self.dashboard = DashboardView(self)
        self.overview = OverviewView(self)
        self.gamecast = GamecastView(self)
        self.players = PlayersView(self)
        self.matchup = MatchupView(self)
        self.schedule = ScheduleView(self)
        self.accuracy = AccuracyView(self)
        self.automatic = AutomaticView(self)
        self.snapshots = SnapshotsView(self)
        self.autotune = AutotuneView(self)
        self.sensitivity = SensitivityView(self)
        self.admin = AdminView(self)

        self.tabs.addTab(self.dashboard, "Dashboard")
        self.tabs.addTab(self.overview, "Today's Overview")
        self.tabs.addTab(self.gamecast, "Gamecast")
        self.tabs.addTab(self.players, "Players")
        self.tabs.addTab(self.matchup, "Matchups")
        self.tabs.addTab(self.schedule, "Schedule")
        self.tabs.addTab(self.accuracy, "Accuracy")
        self.tabs.addTab(self.automatic, "Automatic")
        self.tabs.addTab(self.snapshots, "Snapshots")
        self.tabs.addTab(self.autotune, "Autotune")
        self.tabs.addTab(self.sensitivity, "Sensitivity")
        self.tabs.addTab(self.admin, "Admin")

    def _init_notifications(self):
        """Set up notification bell in the tab bar corner."""
        from src.ui.notification_widget import NotificationBell
        self.notif_bell = NotificationBell(self)
        self.tabs.setCornerWidget(self.notif_bell, Qt.Corner.TopRightCorner)

    def set_status(self, msg: str):
        """Update status bar message."""
        self.status_bar.showMessage(msg)

    def closeEvent(self, event):
        """Clean up on close."""
        logger.info("Application closing")
        event.accept()
