from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from src.database import migrations
from src.ui.accuracy_view import AccuracyView
from src.ui.dashboard import Dashboard
from src.ui.admin_view import AdminView
from src.ui.live_view import LiveView
from src.ui.matchup_view import MatchupView
from src.ui.players_view import PlayersView
from src.ui.schedule_view import ScheduleView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NBA Betting Analytics")
        self.resize(1200, 800)

        tabs = QTabWidget()
        tabs.addTab(Dashboard(), "Dashboard")
        tabs.addTab(PlayersView(), "Players")
        tabs.addTab(MatchupView(), "Matchups")
        tabs.addTab(ScheduleView(), "Schedule")
        tabs.addTab(LiveView(), "Live")
        tabs.addTab(AccuracyView(), "Accuracy")
        tabs.addTab(AdminView(), "Admin")

        self.setCentralWidget(tabs)


def run_app() -> None:
    migrations.init_db()
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
