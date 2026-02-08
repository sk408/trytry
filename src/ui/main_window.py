from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QWidget,
)

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

# ── Global stylesheet ─────────────────────────────────────────────────
# Modern dark theme with a professional sports-analytics aesthetic.

GLOBAL_STYLESHEET = """
/* ── Palette ──
   bg-base:     #0f1923
   bg-surface:  #172333
   bg-card:     #1c2e42
   bg-hover:    #243b53
   accent:      #3b82f6
   accent-dim:  #2563eb
   success:     #10b981
   warning:     #f59e0b
   danger:      #ef4444
   text:        #e2e8f0
   text-muted:  #94a3b8
   border:      #2a3f55
*/

/* ── Base ── */
QMainWindow, QWidget {
    background-color: #0f1923;
    color: #e2e8f0;
    font-family: "Segoe UI", "Inter", "SF Pro Display", sans-serif;
    font-size: 13px;
}

/* ── Tab bar ── */
QTabWidget::pane {
    border: 1px solid #2a3f55;
    border-radius: 6px;
    background: #0f1923;
}
QTabBar {
    qproperty-drawBase: 0;
}
QTabBar::tab {
    background: #172333;
    color: #94a3b8;
    border: 1px solid #2a3f55;
    border-bottom: none;
    padding: 8px 18px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: #1c2e42;
    color: #e2e8f0;
    border-bottom: 2px solid #3b82f6;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background: #1c2e42;
    color: #cbd5e1;
}

/* ── Buttons ── */
QPushButton {
    background-color: #1c2e42;
    color: #e2e8f0;
    border: 1px solid #2a3f55;
    padding: 7px 16px;
    border-radius: 6px;
    font-weight: 500;
    min-height: 18px;
}
QPushButton:hover {
    background-color: #243b53;
    border-color: #3b82f6;
}
QPushButton:pressed {
    background-color: #2563eb;
}
QPushButton:disabled {
    background-color: #172333;
    color: #475569;
    border-color: #1e2d3d;
}
QPushButton[cssClass="primary"] {
    background-color: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
    font-weight: 600;
}
QPushButton[cssClass="primary"]:hover {
    background-color: #3b82f6;
}
QPushButton[cssClass="danger"] {
    background-color: #7f1d1d;
    border-color: #991b1b;
    color: #fecaca;
}
QPushButton[cssClass="danger"]:hover {
    background-color: #991b1b;
    border-color: #ef4444;
}

/* ── Tables ── */
QTableWidget {
    background-color: #172333;
    alternate-background-color: #1c2e42;
    gridline-color: #2a3f55;
    border: 1px solid #2a3f55;
    border-radius: 6px;
    selection-background-color: #243b53;
    selection-color: #e2e8f0;
}
QTableWidget::item {
    padding: 4px 8px;
    border-bottom: 1px solid #1e2d3d;
}
QHeaderView::section {
    background-color: #1c2e42;
    color: #94a3b8;
    padding: 6px 8px;
    border: none;
    border-bottom: 2px solid #2a3f55;
    border-right: 1px solid #2a3f55;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
}

/* ── Group boxes ── */
QGroupBox {
    background-color: #172333;
    border: 1px solid #2a3f55;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    background-color: #1c2e42;
    border: 1px solid #2a3f55;
    border-radius: 4px;
    color: #3b82f6;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Inputs ── */
QComboBox {
    background-color: #1c2e42;
    color: #e2e8f0;
    border: 1px solid #2a3f55;
    padding: 6px 10px;
    border-radius: 6px;
    min-height: 20px;
}
QComboBox:hover { border-color: #3b82f6; }
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #1c2e42;
    color: #e2e8f0;
    border: 1px solid #2a3f55;
    selection-background-color: #243b53;
}
QDoubleSpinBox, QSpinBox {
    background-color: #1c2e42;
    color: #e2e8f0;
    border: 1px solid #2a3f55;
    padding: 5px 8px;
    border-radius: 6px;
}
QCheckBox {
    spacing: 6px;
    color: #e2e8f0;
}
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 2px solid #2a3f55;
    border-radius: 4px;
    background: #172333;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #3b82f6;
}

/* ── Text areas / logs ── */
QTextEdit {
    background-color: #0d1520;
    color: #94a3b8;
    border: 1px solid #2a3f55;
    border-radius: 6px;
    padding: 6px;
    font-family: "Cascadia Code", "Consolas", "Fira Code", monospace;
    font-size: 12px;
}

/* ── Progress bar ── */
QProgressBar {
    background-color: #172333;
    border: 1px solid #2a3f55;
    border-radius: 4px;
    text-align: center;
    color: #e2e8f0;
    height: 16px;
    font-size: 11px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563eb, stop:1 #3b82f6);
    border-radius: 3px;
}

/* ── List widgets ── */
QListWidget {
    background-color: #172333;
    alternate-background-color: #1c2e42;
    border: 1px solid #2a3f55;
    border-radius: 6px;
}
QListWidget::item {
    padding: 4px 8px;
    border-bottom: 1px solid #1e2d3d;
}
QListWidget::item:selected {
    background-color: #243b53;
}

/* ── Scroll bars ── */
QScrollBar:vertical {
    background: #172333;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #2a3f55;
    min-height: 30px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover { background: #3b82f6; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #172333;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #2a3f55;
    min-width: 30px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover { background: #3b82f6; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Scroll area ── */
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }

/* ── Status bar ── */
QStatusBar {
    background: #172333;
    color: #94a3b8;
    border-top: 1px solid #2a3f55;
    padding: 4px;
    font-size: 12px;
}

/* ── Tooltips ── */
QToolTip {
    background-color: #1c2e42;
    color: #e2e8f0;
    border: 1px solid #3b82f6;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
}

/* ── Splitter ── */
QSplitter::handle {
    background: #2a3f55;
}
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }

/* ── Form labels ── */
QFormLayout { spacing: 8px; }
"""


class MainWindow(QMainWindow):
    """Main application window with thread registry and graceful shutdown."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NBA Betting Analytics")
        self.resize(1280, 860)

        # Active thread registry for graceful shutdown
        self._active_threads: set[QThread] = set()

        self.tabs = QTabWidget()

        self.matchup_view = MatchupView()
        self.schedule_view = ScheduleView()

        self.tabs.addTab(Dashboard(), "  Dashboard  ")
        self.tabs.addTab(PlayersView(), "  Players  ")
        self.tabs.addTab(self.matchup_view, "  Matchups  ")
        self.tabs.addTab(self.schedule_view, "  Schedule  ")
        self.tabs.addTab(LiveView(), "  Live  ")
        self.tabs.addTab(GamecastView(), "  Gamecast  ")
        self.tabs.addTab(AccuracyView(), "  Accuracy  ")
        self.tabs.addTab(AutotuneView(), "  Autotune  ")
        self.tabs.addTab(AdminView(), "  Admin  ")

        # Connect schedule -> matchup navigation
        self.schedule_view.game_selected.connect(self._on_schedule_game_selected)

        self.setCentralWidget(self.tabs)

        # Status bar with Exit button
        sb = QStatusBar()
        sb.showMessage("Ready")
        exit_btn = QPushButton("Exit App")
        exit_btn.setProperty("cssClass", "danger")
        exit_btn.setFixedWidth(90)
        exit_btn.clicked.connect(self.close)  # type: ignore
        sb.addPermanentWidget(exit_btn)
        self.setStatusBar(sb)

    # ── Thread registry ──

    def register_thread(self, thread: QThread) -> None:
        """Register a background thread for shutdown tracking."""
        self._active_threads.add(thread)
        thread.finished.connect(lambda t=thread: self._active_threads.discard(t))  # type: ignore

    def unregister_thread(self, thread: QThread) -> None:
        self._active_threads.discard(thread)

    # ── Navigation ──

    def _on_schedule_game_selected(self, home_team_id: int, away_team_id: int) -> None:
        """When a game is selected in the schedule, switch to matchup tab and load it."""
        self.tabs.setCurrentWidget(self.matchup_view)
        self.matchup_view.load_game(home_team_id, away_team_id)

    # ── Graceful shutdown ──

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Flush caches, stop threads, and quit gracefully."""
        running = [t for t in self._active_threads if t.isRunning()]
        if running:
            reply = QMessageBox.question(
                self,
                "Background Tasks Running",
                f"{len(running)} background task(s) are still running.\n"
                "Exit anyway? (tasks will be stopped)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        # Stop all running threads
        for t in running:
            t.quit()
            t.wait(5000)  # 5 second timeout

        # Flush in-memory store
        try:
            from src.analytics.memory_store import get_memory_store
            get_memory_store().flush()
        except Exception:
            pass

        # Clear module-level caches
        try:
            from src.analytics.backtester import clear_advanced_profile_cache
            clear_advanced_profile_cache()
        except Exception:
            pass

        event.accept()
        QApplication.quit()


def run_app() -> None:
    migrations.init_db()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    # Slightly larger default font
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
