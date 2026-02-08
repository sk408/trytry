from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from src.database import migrations
from src.ui.accuracy_view import AccuracyView
from src.ui.autotune_view import AutotuneView
from src.ui.dashboard import Dashboard
from src.ui.admin_view import AdminView
from src.ui.gamecast_view import GamecastView
from src.ui.live_view import LiveView
from src.ui.matchup_view import MatchupView
from src.ui.players_view import PlayersView
from src.ui.schedule_view import ScheduleView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NBA Betting Analytics")
        self.resize(1200, 800)

        self.tabs = QTabWidget()

        self.matchup_view = MatchupView()
        self.schedule_view = ScheduleView()

        self.tabs.addTab(Dashboard(), "Dashboard")
        self.tabs.addTab(PlayersView(), "Players")
        self.tabs.addTab(self.matchup_view, "Matchups")
        self.tabs.addTab(self.schedule_view, "Schedule")
        self.tabs.addTab(LiveView(), "Live")
        self.tabs.addTab(GamecastView(), "Gamecast")
        self.tabs.addTab(AccuracyView(), "Accuracy")
        self.tabs.addTab(AutotuneView(), "Autotune")
        self.tabs.addTab(AdminView(), "Admin")

        # Connect schedule → matchup navigation
        self.schedule_view.game_selected.connect(self._on_schedule_game_selected)

        self.setCentralWidget(self.tabs)

    def _on_schedule_game_selected(self, home_team_id: int, away_team_id: int) -> None:
        """When a game is selected in the schedule, switch to matchup tab and load it."""
        self.tabs.setCurrentWidget(self.matchup_view)
        self.matchup_view.load_game(home_team_id, away_team_id)


def run_app() -> None:
    migrations.init_db()
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
