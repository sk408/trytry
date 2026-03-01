"""Players tab — split view: injured players (enhanced) + all players with photos."""

import logging
import re
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QLineEdit, QFrame, QTreeWidget, QTreeWidgetItem,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor, QIcon, QPixmap

logger = logging.getLogger(__name__)

_KEY_COLOR = QColor(239, 68, 68)        # Red — KEY player
_ROTATION_COLOR = QColor(251, 191, 36)  # Amber — ROTATION
_BENCH_COLOR = QColor(148, 163, 184)    # Slate

_STATUS_COLORS = {
    "Out": QColor(239, 68, 68),          # Red
    "Doubtful": QColor(239, 115, 68),    # Orange-red
    "Questionable": QColor(251, 191, 36),# Amber
    "Day-To-Day": QColor(251, 191, 36),  # Amber
    "GTD": QColor(251, 191, 36),         # Amber
    "Probable": QColor(34, 197, 94),     # Green
}


def _estimate_play_probability(status: str, expected_return: str, detail: str) -> tuple:
    """Estimate probability the player plays next game + short label.

    Returns (probability_float, label_string, color).
    """
    low_status = status.lower()
    low_detail = detail.lower()

    # Season-ending keywords
    season_ending = any(kw in low_detail for kw in (
        "season-ending", "remainder of the", "rest of the 2025",
        "rest of the 2026", "torn acl", "torn achilles",
        "indefinitely", "suspended",
    ))
    # ESPN uses "Oct 1" as expected return for season-ending injuries
    if expected_return and expected_return.strip().lower().startswith("oct"):
        season_ending = True

    if season_ending:
        return (0.0, "OUT FOR SEASON", QColor(239, 68, 68))

    # Check if detail mentions surgery without explicit season-ending
    if "surgery" in low_detail and "successful" not in low_detail:
        return (0.0, "OUT (surgery)", QColor(239, 68, 68))

    # Parse expected return date
    return_days = _days_until_return(expected_return)

    if low_status in ("out", "o"):
        if return_days is not None:
            if return_days <= 0:
                return (0.15, "OUT (back soon)", QColor(239, 115, 68))
            elif return_days <= 3:
                return (0.10, f"OUT ~{return_days}d", QColor(239, 68, 68))
            elif return_days <= 14:
                return (0.02, f"OUT ~{return_days}d", QColor(239, 68, 68))
            else:
                wks = return_days // 7
                return (0.0, f"OUT ~{wks}wk", QColor(239, 68, 68))
        return (0.0, "OUT", QColor(239, 68, 68))

    if low_status == "doubtful":
        return (0.15, "DOUBTFUL 15%", QColor(239, 115, 68))

    if low_status in ("questionable", "gtd"):
        return (0.50, "QUESTIONABLE 50%", QColor(251, 191, 36))

    if low_status == "day-to-day":
        if return_days is not None and return_days <= 1:
            return (0.65, "DAY-TO-DAY 65%", QColor(251, 191, 36))
        return (0.55, "DAY-TO-DAY 55%", QColor(251, 191, 36))

    if low_status == "probable":
        return (0.85, "PROBABLE 85%", QColor(34, 197, 94))

    return (0.25, status.upper(), QColor(148, 163, 184))


def _days_until_return(expected_return: str) -> int | None:
    """Parse ESPN return date like 'Feb 26' or 'Mar 4' -> days from now."""
    if not expected_return or len(expected_return) < 4:
        return None
    try:
        now = datetime.now()
        m = re.match(r"([A-Za-z]+)\s+(\d+)", expected_return)
        if m:
            month_str, day_str = m.group(1), m.group(2)
            for fmt in ("%b %d %Y", "%B %d %Y"):
                for year in (now.year, now.year + 1):
                    try:
                        dt = datetime.strptime(f"{month_str} {day_str} {year}", fmt)
                        delta = (dt - now).days
                        if delta >= -30:
                            return max(0, delta)
                    except ValueError:
                        continue
    except Exception as e:
        logger.warning("Return date parse failed for '%s': %s", raw, e)
    return None


class _SyncInjuriesWorker(QObject):
    """Background worker for injury sync."""
    progress = Signal(str)
    finished = Signal(int)

    def run(self):
        try:
            from src.data.injury_scraper import sync_injuries
            count = sync_injuries(callback=lambda msg: self.progress.emit(msg))
            self.finished.emit(count)
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self.finished.emit(0)


class PlayersView(QWidget):
    """All players + injured by impact with photos and play probability."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._worker_thread = None
        self._worker = None
        layout = QVBoxLayout(self)

        header = QLabel("Players")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # Search + sync + refresh
        ctrl = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search players...")
        self.search.textChanged.connect(self._filter_players)
        ctrl.addWidget(self.search)

        self.sync_btn = QPushButton("Sync Injuries")
        self.sync_btn.setToolTip("Re-scrape injury data from ESPN / CBS")
        self.sync_btn.clicked.connect(self._sync_injuries)
        ctrl.addWidget(self.sync_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load)
        ctrl.addWidget(refresh_btn)

        self._sync_status = QLabel("")
        self._sync_status.setStyleSheet("color: #64748b; font-size: 10px;")
        ctrl.addWidget(self._sync_status)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Splitter: injured on top, all players below
        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Injured players panel ──
        injured_frame = QFrame()
        inj_layout = QVBoxLayout(injured_frame)
        inj_layout.setContentsMargins(0, 0, 0, 0)
        self._inj_header = QLabel("Injured Players (by Impact)")
        inj_layout.addWidget(self._inj_header)

        self.injured_table = QTableWidget()
        self.injured_table.setColumnCount(8)
        self.injured_table.setHorizontalHeaderLabels([
            "", "Player", "Team", "Status", "Availability", "Return", "Injury", "Impact",
        ])
        hdr = self.injured_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.injured_table.setColumnWidth(0, 34)
        self.injured_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.injured_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.injured_table.verticalHeader().setVisible(False)
        self.injured_table.setShowGrid(False)
        self.injured_table.setAlternatingRowColors(True)
        inj_layout.addWidget(self.injured_table)
        splitter.addWidget(injured_frame)

        # ── All players panel (grouped by team) ──
        all_frame = QFrame()
        all_layout = QVBoxLayout(all_frame)
        all_layout.setContentsMargins(0, 0, 0, 0)
        self._all_header = QLabel("All Players")
        all_layout.addWidget(self._all_header)

        self.all_tree = QTreeWidget()
        self.all_tree.setColumnCount(4)
        self.all_tree.setHeaderLabels(["Player", "Position", "Status", ""])
        all_hdr = self.all_tree.header()
        all_hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        all_hdr.setStretchLastSection(False)
        all_hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.all_tree.setColumnWidth(3, 0)  # hidden spacer
        self.all_tree.setIndentation(20)
        self.all_tree.setRootIsDecorated(True)
        self.all_tree.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)
        self.all_tree.setSelectionBehavior(QTreeWidget.SelectionBehavior.SelectRows)
        self.all_tree.setAlternatingRowColors(True)
        all_layout.addWidget(self.all_tree)
        splitter.addWidget(all_frame)

        layout.addWidget(splitter)

        self._all_players = []
        self._load()

    # ──────────────────────── DATA LOADING ────────────────────────

    def _load(self):
        """Load players and injuries from DB."""
        try:
            from src.database import db

            # All players
            self._all_players = db.fetch_all(
                "SELECT p.player_id, p.name, t.abbreviation, p.position, "
                "       p.is_injured, p.injury_note "
                "FROM players p LEFT JOIN teams t ON p.team_id = t.team_id "
                "ORDER BY t.abbreviation, p.name LIMIT 600"
            )
            self._populate_all()

            # Injured players with MPG and return date
            injured = db.fetch_all(
                "SELECT i.player_id, i.player_name, t.abbreviation, i.status, "
                "       i.reason, i.expected_return, i.injury_keyword, "
                "       COALESCE(("
                "           SELECT AVG(ps.minutes) FROM player_stats ps "
                "           WHERE ps.player_id = i.player_id "
                "           ORDER BY ps.game_date DESC LIMIT 10"
                "       ), 0) as mpg "
                "FROM injuries i "
                "LEFT JOIN teams t ON i.team_id = t.team_id "
                "ORDER BY mpg DESC"
            )
            self._populate_injured(injured)
            self._inj_header.setText(
                f"Injured Players ({len(injured)} — by Impact)"
            )

        except Exception as e:
            logger.error(f"Players load error: {e}")

    # ──────────────────────── INJURED TABLE ────────────────────────

    def _populate_injured(self, injured: list):
        """Fill injured table with impact classification and play probability."""
        self.injured_table.setRowCount(len(injured))
        for r, inj in enumerate(injured):
            mpg = float(inj.get("mpg", 0))
            if mpg >= 25:
                impact = "KEY"
                impact_color = _KEY_COLOR
            elif mpg >= 15:
                impact = "ROTATION"
                impact_color = _ROTATION_COLOR
            else:
                impact = "BENCH"
                impact_color = _BENCH_COLOR

            status = inj.get("status", "Out")
            expected_return = inj.get("expected_return", "") or ""
            detail = inj.get("reason", "") or ""
            player_id = inj.get("player_id")

            prob, avail_label, avail_color = _estimate_play_probability(
                status, expected_return, detail,
            )

            self.injured_table.setRowHeight(r, 32)

            # Col 0: Player photo
            photo_item = QTableWidgetItem()
            if player_id:
                try:
                    from src.ui.widgets.image_utils import get_player_photo
                    pixmap = get_player_photo(int(player_id), 28, circle=True)
                    if pixmap:
                        photo_item.setData(
                            Qt.ItemDataRole.DecorationRole, pixmap
                        )
                except Exception as e:
                    logger.warning("Injured player photo load failed for %s: %s", player_id, e)
            self.injured_table.setItem(r, 0, photo_item)

            # Col 1: Name
            self.injured_table.setItem(
                r, 1, QTableWidgetItem(inj.get("player_name", ""))
            )

            # Col 2: Team
            self.injured_table.setItem(
                r, 2, QTableWidgetItem(inj.get("abbreviation", ""))
            )

            # Col 3: Status (colored)
            status_item = QTableWidgetItem(status)
            status_item.setForeground(
                _STATUS_COLORS.get(status, QColor(148, 163, 184))
            )
            self.injured_table.setItem(r, 3, status_item)

            # Col 4: Availability (play probability)
            avail_item = QTableWidgetItem(avail_label)
            avail_item.setForeground(avail_color)
            self.injured_table.setItem(r, 4, avail_item)

            # Col 5: Expected return
            self.injured_table.setItem(
                r, 5, QTableWidgetItem(expected_return)
            )

            # Col 6: Injury detail (truncated)
            injury_text = detail[:80] + "..." if len(detail) > 80 else detail
            self.injured_table.setItem(r, 6, QTableWidgetItem(injury_text))

            # Col 7: Impact
            impact_item = QTableWidgetItem(f"{impact} ({mpg:.0f}m)")
            impact_item.setForeground(impact_color)
            self.injured_table.setItem(r, 7, impact_item)

    # ──────────────────────── ALL PLAYERS TABLE ────────────────────

    def _populate_all(self, filter_text: str = ""):
        """Fill all-players tree grouped by team, with optional filter."""
        from collections import OrderedDict

        ft = filter_text.lower()
        rows = (
            [p for p in self._all_players if ft in p["name"].lower()]
            if ft else self._all_players
        )

        # Group by team (already sorted by abbreviation from query)
        teams: OrderedDict[str, list] = OrderedDict()
        for p in rows:
            team = p.get("abbreviation", "???")
            teams.setdefault(team, []).append(p)

        self.all_tree.clear()
        total = 0
        for team_abbr, players in teams.items():
            # Team header row
            team_item = QTreeWidgetItem(self.all_tree)
            active = sum(1 for p in players if not p.get("is_injured"))
            injured = len(players) - active
            label = f"{team_abbr}  ({len(players)} players"
            if injured:
                label += f", {injured} injured"
            label += ")"
            team_item.setText(0, label)
            font = team_item.font(0)
            font.setBold(True)
            team_item.setFont(0, font)
            total += len(players)

            # Player rows under team
            for p in players:
                child = QTreeWidgetItem(team_item)

                # Col 0: Name (with photo icon)
                child.setText(0, p["name"])
                pid = p.get("player_id")
                if pid:
                    try:
                        from src.ui.widgets.image_utils import get_player_photo
                        pixmap = get_player_photo(int(pid), 22, circle=True)
                        if pixmap:
                            child.setIcon(0, QIcon(pixmap))
                    except Exception:
                        pass

                # Col 1: Position
                child.setText(1, p.get("position", ""))

                # Col 2: Status
                is_inj = p.get("is_injured")
                note = p.get("injury_note", "") or ""
                if is_inj:
                    child.setText(2, note[:50] if note else "Injured")
                    child.setForeground(2, QColor(239, 68, 68))
                else:
                    child.setText(2, "Active")
                    child.setForeground(2, QColor(34, 197, 94))

        # Expand all when filtering (so matches are visible), collapse when not
        if ft:
            self.all_tree.expandAll()
        else:
            self.all_tree.collapseAll()

        self._all_header.setText(f"All Players ({total})")

    def _filter_players(self, text: str):
        self._populate_all(text)

    # ──────────────────────── INJURY SYNC ────────────────────────

    def _sync_injuries(self):
        """Run injury sync on background thread."""
        if self._worker_thread and self._worker_thread.isRunning():
            return
        self.sync_btn.setEnabled(False)
        self._sync_status.setText("Syncing injuries...")

        self._worker = _SyncInjuriesWorker()
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(
            self._on_sync_progress, Qt.ConnectionType.QueuedConnection
        )
        self._worker.finished.connect(
            self._on_sync_done, Qt.ConnectionType.QueuedConnection
        )
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.start()

    def _on_sync_progress(self, msg: str):
        self._sync_status.setText(msg)

    def _on_sync_done(self, count: int):
        self.sync_btn.setEnabled(True)
        self._sync_status.setText(f"Synced {count} injuries")
        self._load()
