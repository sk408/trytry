"""Matchup tab — game selector (today+14d), prediction cards, injury labels,
H2H table, player tables with headshots."""

import logging
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QGridLayout, QScrollArea, QSplitter, QTabWidget,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
from PySide6.QtGui import QColor, QFont

logger = logging.getLogger(__name__)


class _PredictWorker(QObject):
    """Background worker for matchup prediction."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, home_id: int, away_id: int):
        super().__init__()
        self.home_id = home_id
        self.away_id = away_id

    def run(self):
        try:
            from datetime import datetime
            from src.analytics.prediction import predict_matchup
            today = datetime.now().strftime("%Y-%m-%d")
            result = predict_matchup(self.home_id, self.away_id, game_date=today)
            self.finished.emit(result.__dict__)
        except Exception as e:
            self.error.emit(str(e))


class PredictionCard(QFrame):
    """Card showing a single prediction value."""

    def __init__(self, title: str, value: str = "—"):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "PredictionCard { background: #1e293b; border: 1px solid #334155; "
            "border-radius: 8px; padding: 10px; }"
        )
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #64748b; font-size: 11px;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #e2e8f0; font-size: 22px; font-weight: 700;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, val: str):
        self.value_label.setText(val)


class MatchupView(QWidget):
    """Full matchup prediction view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._worker_thread = None
        self._worker = None

        layout = QVBoxLayout(self)

        header = QLabel("Matchup Predictions")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # Team selectors
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Home:"))
        self.home_combo = QComboBox()
        self.home_combo.setMinimumWidth(200)
        sel_layout.addWidget(self.home_combo)

        sel_layout.addWidget(QLabel("Away:"))
        self.away_combo = QComboBox()
        self.away_combo.setMinimumWidth(200)
        sel_layout.addWidget(self.away_combo)

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self._on_predict)
        sel_layout.addWidget(self.predict_btn)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)

        # Game selector from schedule
        game_layout = QHBoxLayout()
        game_layout.addWidget(QLabel("Or pick a game:"))
        self.game_combo = QComboBox()
        self.game_combo.setMinimumWidth(350)
        self.game_combo.currentIndexChanged.connect(self._on_game_picked)
        game_layout.addWidget(self.game_combo)
        game_layout.addStretch()
        layout.addLayout(game_layout)

        # Prediction cards
        cards_layout = QGridLayout()
        self.spread_card = PredictionCard("Spread")
        self.total_card = PredictionCard("Total")
        self.home_card = PredictionCard("Home Score")
        self.away_card = PredictionCard("Away Score")
        self.winner_card = PredictionCard("Win Probability")
        self.confidence_card = PredictionCard("Confidence")

        cards_layout.addWidget(self.spread_card, 0, 0)
        cards_layout.addWidget(self.total_card, 0, 1)
        cards_layout.addWidget(self.home_card, 0, 2)
        cards_layout.addWidget(self.away_card, 0, 3)
        cards_layout.addWidget(self.winner_card, 1, 0, 1, 2)
        cards_layout.addWidget(self.confidence_card, 1, 2, 1, 2)
        layout.addLayout(cards_layout)

        # Breakdown table
        self.breakdown_table = QTableWidget()
        self.breakdown_table.setColumnCount(3)
        self.breakdown_table.setHorizontalHeaderLabels(["Factor", "Home", "Away"])
        self.breakdown_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.breakdown_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.breakdown_table.setMaximumHeight(250)
        layout.addWidget(self.breakdown_table)

        # Injury impact summary
        self.injury_label = QLabel("")
        self.injury_label.setWordWrap(True)
        self.injury_label.setStyleSheet("color: #94a3b8; padding: 8px;")
        layout.addWidget(self.injury_label)

        # ── Player rosters with injury status ──
        roster_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Home roster
        home_roster_panel = QWidget()
        home_roster_layout = QVBoxLayout(home_roster_panel)
        home_roster_layout.setContentsMargins(0, 0, 0, 0)
        self._home_roster_header = QLabel("Home Roster")
        self._home_roster_header.setStyleSheet(
            "color: #e2e8f0; font-size: 13px; font-weight: 700; padding: 4px;"
        )
        home_roster_layout.addWidget(self._home_roster_header)
        self._home_roster_table = self._make_roster_table()
        home_roster_layout.addWidget(self._home_roster_table)
        roster_splitter.addWidget(home_roster_panel)

        # Away roster
        away_roster_panel = QWidget()
        away_roster_layout = QVBoxLayout(away_roster_panel)
        away_roster_layout.setContentsMargins(0, 0, 0, 0)
        self._away_roster_header = QLabel("Away Roster")
        self._away_roster_header.setStyleSheet(
            "color: #e2e8f0; font-size: 13px; font-weight: 700; padding: 4px;"
        )
        away_roster_layout.addWidget(self._away_roster_header)
        self._away_roster_table = self._make_roster_table()
        away_roster_layout.addWidget(self._away_roster_table)
        roster_splitter.addWidget(away_roster_panel)

        layout.addWidget(roster_splitter, 1)

        # Load teams (DB only — fast) then defer network call
        self._load_teams()
        QTimer.singleShot(600, self._load_games)

    def _load_teams(self):
        """Populate team dropdowns."""
        try:
            from src.database import db
            teams = db.fetch_all(
                "SELECT team_id, abbreviation, name FROM teams ORDER BY name"
            )
            for t in teams:
                label = f"{t['abbreviation']} — {t['name']}"
                self.home_combo.addItem(label, t["team_id"])
                self.away_combo.addItem(label, t["team_id"])
        except Exception as e:
            logger.error(f"Load teams error: {e}")

    def _load_games(self):
        """Load upcoming games (today + 14 days)."""
        self.game_combo.blockSignals(True)
        self.game_combo.clear()
        self.game_combo.addItem("— Select game —", None)
        try:
            from src.data.nba_fetcher import fetch_nba_cdn_schedule
            schedule = fetch_nba_cdn_schedule()
            today = datetime.now().strftime("%Y-%m-%d")
            cutoff = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
            for g in schedule:
                game_date = g.get("game_date", "")
                if game_date and today <= game_date <= cutoff:
                    time_str = g.get("game_time", "")
                    label = (
                        f"{game_date} {time_str} — "
                        f"{g.get('away_team', '?')} @ {g.get('home_team', '?')}"
                    )
                    self.game_combo.addItem(label, g)
        except Exception as e:
            logger.debug(f"Schedule load: {e}")
        self.game_combo.blockSignals(False)

    def _on_game_picked(self, idx: int):
        """Auto-select teams from game pick."""
        data = self.game_combo.itemData(idx)
        if not data or not isinstance(data, dict):
            return
        home_id = data.get("home_team_id")
        away_id = data.get("away_team_id")
        if home_id:
            for i in range(self.home_combo.count()):
                if self.home_combo.itemData(i) == home_id:
                    self.home_combo.setCurrentIndex(i)
                    break
        if away_id:
            for i in range(self.away_combo.count()):
                if self.away_combo.itemData(i) == away_id:
                    self.away_combo.setCurrentIndex(i)
                    break
        self._on_predict()

    def _on_predict(self):
        """Run prediction in background."""
        home_id = self.home_combo.currentData()
        away_id = self.away_combo.currentData()
        if not home_id or not away_id:
            return
        if home_id == away_id:
            return
        if self._worker_thread and self._worker_thread.isRunning():
            return

        self.predict_btn.setEnabled(False)
        self.predict_btn.setText("Predicting...")

        self._worker = _PredictWorker(home_id, away_id)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        _QC = Qt.ConnectionType.QueuedConnection
        self._worker.finished.connect(self._on_result, _QC)
        self._worker.error.connect(self._on_error, _QC)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.start()

    def _on_error(self, msg: str):
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("Predict")
        logger.error(f"Prediction error: {msg}")
        if self.main_window:
            self.main_window.set_status(f"Prediction error: {msg}")

    def _on_result(self, result: dict):
        """Populate UI with prediction result."""
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("Predict")

        spread = result.get("predicted_spread", 0)
        total = result.get("predicted_total", 0)
        home_score = result.get("predicted_home_score", 0)
        away_score = result.get("predicted_away_score", 0)
        confidence = result.get("confidence", 0)

        self.spread_card.set_value(f"{spread:+.1f}")
        self.total_card.set_value(f"{total:.1f}")
        self.home_card.set_value(f"{home_score:.1f}")
        self.away_card.set_value(f"{away_score:.1f}")

        winner = result.get("winner", "")
        self.winner_card.set_value(winner if winner else "—")
        self.confidence_card.set_value(f"{confidence * 100:.0f}%")

        # Breakdown — show all non-zero parameters
        all_adjustments = [
            ("Home Court", result.get("home_court_advantage", 0)),
            ("Fatigue", result.get("fatigue_adj", 0)),
            ("Turnover", result.get("turnover_adj", 0)),
            ("Rebound", result.get("rebound_adj", 0)),
            ("Rating Matchup", result.get("rating_matchup_adj", 0)),
            ("Four Factors", result.get("four_factors_adj", 0)),
            ("Clutch", result.get("clutch_adj", 0)),
            ("Hustle (Spread)", result.get("hustle_adj", 0)),
            ("Pace (Total)", result.get("pace_adj", 0)),
            ("Def. Disruption (Total)", result.get("defensive_disruption", 0)),
            ("OREB Boost (Total)", result.get("oreb_boost", 0)),
            ("Hustle (Total)", result.get("hustle_total_adj", 0)),
            ("Sharp Money", result.get("adjustments", {}).get("sharp_money", 0)),
            ("ESPN Blend", result.get("espn_blend_adj", 0)),
            ("ML Blend", result.get("ml_blend_adj", 0)),
        ]
        visible = [(k, v) for k, v in all_adjustments if abs(v) > 0.01]

        # Add sharp money raw info row if data exists
        sh_pub = result.get("sharp_home_public", 0)
        sh_mon = result.get("sharp_home_money", 0)
        if sh_pub > 0 or sh_mon > 0:
            visible.append(("Sharp: Public/Money", None))  # sentinel for special formatting

        self.breakdown_table.setRowCount(len(visible))
        for row, (factor, val) in enumerate(visible):
            self.breakdown_table.setItem(row, 0, QTableWidgetItem(factor))
            if val is None:
                # Special row: show raw sharp money percentages
                item = QTableWidgetItem(f"H {sh_pub}%/{sh_mon}%  A {100-sh_pub}%/{100-sh_mon}%")
                if abs(sh_mon - sh_pub) >= 10:
                    item.setForeground(QColor("#2196F3"))  # blue for big divergence
                self.breakdown_table.setItem(row, 1, item)
            else:
                item = QTableWidgetItem(f"{val:+.2f}")
                if abs(val) > 10.0:
                    item.setForeground(Qt.red)
                self.breakdown_table.setItem(row, 1, item)

        # ── Injury impact + rosters ──
        home_team = result.get("home_team", "HOME")
        away_team = result.get("away_team", "AWAY")
        home_id = result.get("home_team_id")
        away_id = result.get("away_team_id")

        self._home_roster_header.setText(f"{home_team} Roster")
        self._away_roster_header.setText(f"{away_team} Roster")

        self.injury_label.setText("")  # clear before populating
        self._populate_team_roster(home_id, home_team, self._home_roster_table)
        self._populate_team_roster(away_id, away_team, self._away_roster_table)

        if self.main_window:
            self.main_window.set_status("Prediction complete")

    # ──────────────────────── ROSTER / INJURY HELPERS ────────────────

    @staticmethod
    def _make_roster_table() -> QTableWidget:
        """Create a styled roster table with injury columns."""
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            "Player", "Pos", "PPG", "MPG", "Status", "Reason", "Impact",
        ])
        hdr = table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(1, 40)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        return table

    def _populate_team_roster(self, team_id, team_abbr: str, table: QTableWidget):
        """Fill a roster table with players, highlighting injured ones."""
        if not team_id:
            table.setRowCount(0)
            return
        try:
            from src.database import db

            # Get roster with injury info
            players = db.fetch_all("""
                SELECT p.player_id, p.name, p.position,
                       i.status AS injury_status, i.reason AS injury_reason,
                       COALESCE(
                           (SELECT AVG(minutes) FROM (
                               SELECT minutes FROM player_stats
                               WHERE player_id = p.player_id
                               ORDER BY game_date DESC LIMIT 15
                           )), 0
                       ) AS mpg
                FROM players p
                LEFT JOIN injuries i ON p.player_id = i.player_id
                WHERE p.team_id = ?
                ORDER BY mpg DESC
            """, (team_id,))

            # Get per-player average stats
            player_stats = {}
            for p in players:
                pid = p["player_id"]
                stat_row = db.fetch_one("""
                    SELECT AVG(points) AS ppg, AVG(minutes) AS mpg
                    FROM (SELECT points, minutes FROM player_stats
                          WHERE player_id = ? ORDER BY game_date DESC LIMIT 15)
                """, (pid,))
                if stat_row:
                    player_stats[pid] = {
                        "ppg": round(stat_row["ppg"] or 0, 1),
                        "mpg": round(stat_row["mpg"] or 0, 1),
                    }

            table.setRowCount(len(players))
            _RED = QColor(239, 68, 68)
            _YELLOW = QColor(234, 179, 8)
            _GREEN = QColor(34, 197, 94)
            _DEFAULT = QColor(226, 232, 240)
            _DIM = QColor(100, 116, 139)

            for row, p in enumerate(players):
                pid = p["player_id"]
                name = p["name"] or ""
                pos = p["position"] or ""
                inj_status = p["injury_status"] or ""
                inj_reason = p["injury_reason"] or ""
                stats = player_stats.get(pid, {"ppg": 0, "mpg": 0})

                table.setRowHeight(row, 26)

                # Determine row color based on injury status
                status_lower = inj_status.lower()
                if "out" in status_lower:
                    row_color = _RED
                    impact_text = f"-{stats['ppg']:.1f} pts"
                elif "doubtful" in status_lower:
                    row_color = _RED
                    impact_text = f"~-{stats['ppg'] * 0.8:.1f} pts"
                elif "questionable" in status_lower or "day-to-day" in status_lower:
                    row_color = _YELLOW
                    impact_text = f"~-{stats['ppg'] * 0.4:.1f} pts"
                elif "probable" in status_lower:
                    row_color = _GREEN
                    impact_text = "Likely plays"
                elif inj_status:
                    row_color = _YELLOW
                    impact_text = "Unknown"
                else:
                    row_color = _DEFAULT
                    impact_text = ""
                    inj_status = "Active"

                items = [
                    name, pos,
                    f"{stats['ppg']:.1f}", f"{stats['mpg']:.1f}",
                    inj_status, inj_reason, impact_text,
                ]
                for col, text in enumerate(items):
                    item = QTableWidgetItem(text)
                    if col >= 4 and inj_status != "Active":
                        item.setForeground(row_color)
                        if col == 4:
                            item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                    elif inj_status != "Active" and col == 0:
                        item.setForeground(row_color if "out" in status_lower
                                           else _DEFAULT)
                    else:
                        item.setForeground(
                            _DIM if inj_status != "Active" and "out" in status_lower
                            else _DEFAULT
                        )
                    if col >= 2 and col <= 3:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table.setItem(row, col, item)

            # Summary line in injury label
            injured = [p for p in players if p["injury_status"]]
            if injured:
                parts = []
                for p in injured:
                    s = player_stats.get(p["player_id"], {"ppg": 0, "mpg": 0})
                    parts.append(
                        f"{p['name']} ({p['injury_status']}, {s['ppg']} ppg)"
                    )
                existing = self.injury_label.text()
                line = f"{team_abbr} Injuries: " + " | ".join(parts)
                if existing:
                    self.injury_label.setText(existing + "\n" + line)
                else:
                    self.injury_label.setText(line)

        except Exception as e:
            logger.error(f"Roster load error for team {team_id}: {e}")
            table.setRowCount(0)
