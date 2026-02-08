from __future__ import annotations

from typing import Dict

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.analytics.stats_engine import get_team_matchup_stats, get_scheduled_games, TeamMatchupStats
from src.analytics.prediction import predict_matchup
from src.database.db import get_conn


def get_matchup_backtest(home_id: int, away_id: int) -> Dict:
    """
    Use fully aggregated game results (team-level) to avoid undercounting due to trades.
    """
    from src.analytics.backtester import get_actual_game_results

    games = get_actual_game_results()
    if games.empty:
        return {
            "home_abbr": "HOME",
            "away_abbr": "AWAY",
            "home_games_at_home": 0,
            "away_games_on_road": 0,
            "home_avg_home": 0.0,
            "away_avg_road": 0.0,
            "h2h_games": [],
            "home_record_at_home": {"wins": 0, "losses": 0},
            "away_record_on_road": {"wins": 0, "losses": 0},
        }

    with get_conn() as conn:
        teams = pd.read_sql(
            "SELECT team_id, abbreviation FROM teams WHERE team_id IN (?, ?)",
            conn,
            params=[home_id, away_id],
        )
    abbrs = {int(r["team_id"]): r["abbreviation"] for _, r in teams.iterrows()}
    home_abbr = abbrs.get(home_id, "HOME")
    away_abbr = abbrs.get(away_id, "AWAY")

    home_home = games[games["home_team_id"] == home_id]
    away_road = games[games["away_team_id"] == away_id]

    result = {
        "home_abbr": home_abbr,
        "away_abbr": away_abbr,
        "home_games_at_home": len(home_home),
        "away_games_on_road": len(away_road),
        "home_avg_home": float(home_home["home_score"].mean()) if not home_home.empty else 0.0,
        "away_avg_road": float(away_road["away_score"].mean()) if not away_road.empty else 0.0,
        "h2h_games": [],
        "home_record_at_home": {"wins": 0, "losses": 0},
        "away_record_on_road": {"wins": 0, "losses": 0},
    }

    # Home record at home
    for _, g in home_home.iterrows():
        if g["home_score"] > g["away_score"]:
            result["home_record_at_home"]["wins"] += 1
        elif g["away_score"] > g["home_score"]:
            result["home_record_at_home"]["losses"] += 1

        if int(g["away_team_id"]) == away_id:
            result["h2h_games"].append({
                "date": str(g["game_date"]),
                "home_score": float(g["home_score"]),
                "away_score": float(g["away_score"]),
                "winner": home_abbr if g["home_score"] > g["away_score"] else away_abbr,
            })

    # Away record on road
    for _, g in away_road.iterrows():
        if g["away_score"] > g["home_score"]:
            result["away_record_on_road"]["wins"] += 1
        elif g["home_score"] > g["away_score"]:
            result["away_record_on_road"]["losses"] += 1

    return result


def _teams_df() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT team_id, abbreviation, name FROM teams ORDER BY abbreviation", conn)


class MatchupView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        
        # Game selector
        self.game_combo = QComboBox()
        self.game_combo.currentIndexChanged.connect(self._on_game_selected)  # type: ignore[arg-type]
        
        self.home_combo = QComboBox()
        self.away_combo = QComboBox()
        
        # Prediction summary
        self.spread_label = QLabel("--")
        self.spread_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.total_label = QLabel("--")
        self.total_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.home_proj_label = QLabel("--")
        self.away_proj_label = QLabel("--")
        
        # Injury impact labels
        self.home_injury_label = QLabel("No injuries")
        self.away_injury_label = QLabel("No injuries")
        
        # Backtest accuracy labels
        self.home_record_label = QLabel("--")
        self.away_record_label = QLabel("--")
        self.h2h_label = QLabel("--")
        self.h2h_table = QTableWidget()
        self.h2h_table.setMaximumHeight(100)
        
        # Player tables
        self.home_table = QTableWidget()
        self.away_table = QTableWidget()

        self.refresh_button = QPushButton("Load Games")
        self.refresh_button.clicked.connect(self.refresh)  # type: ignore[arg-type]
        self.predict_button = QPushButton("Analyze Matchup")
        self.predict_button.clicked.connect(self.predict)  # type: ignore[arg-type]

        # Game selection from schedule
        game_row = QHBoxLayout()
        game_row.addWidget(QLabel("Select Game:"))
        game_row.addWidget(self.game_combo, stretch=1)
        game_row.addWidget(self.refresh_button)

        # Manual team selection
        form = QFormLayout()
        form.addRow("Home team", self.home_combo)
        form.addRow("Away team", self.away_combo)
        
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.predict_button)
        btn_row.addStretch()

        # Prediction summary box
        pred_box = QGroupBox("Prediction Summary")
        pred_layout = QHBoxLayout()
        
        spread_col = QVBoxLayout()
        spread_col.addWidget(QLabel("Spread (home)"))
        spread_col.addWidget(self.spread_label)
        
        total_col = QVBoxLayout()
        total_col.addWidget(QLabel("Over/Under"))
        total_col.addWidget(self.total_label)
        
        home_col = QVBoxLayout()
        home_col.addWidget(QLabel("Home Projected"))
        home_col.addWidget(self.home_proj_label)
        home_col.addWidget(self.home_injury_label)
        
        away_col = QVBoxLayout()
        away_col.addWidget(QLabel("Away Projected"))
        away_col.addWidget(self.away_proj_label)
        away_col.addWidget(self.away_injury_label)
        
        pred_layout.addLayout(spread_col)
        pred_layout.addLayout(total_col)
        pred_layout.addLayout(home_col)
        pred_layout.addLayout(away_col)
        pred_box.setLayout(pred_layout)
        
        # Historical accuracy box
        history_box = QGroupBox("Historical Performance (Backtest)")
        history_layout = QVBoxLayout()
        
        records_row = QHBoxLayout()
        home_rec_col = QVBoxLayout()
        home_rec_col.addWidget(QLabel("Home Team Record (at home)"))
        home_rec_col.addWidget(self.home_record_label)
        
        away_rec_col = QVBoxLayout()
        away_rec_col.addWidget(QLabel("Away Team Record (on road)"))
        away_rec_col.addWidget(self.away_record_label)
        
        h2h_col = QVBoxLayout()
        h2h_col.addWidget(QLabel("Head-to-Head This Season"))
        h2h_col.addWidget(self.h2h_label)
        
        records_row.addLayout(home_rec_col)
        records_row.addLayout(away_rec_col)
        records_row.addLayout(h2h_col)
        
        history_layout.addLayout(records_row)
        history_layout.addWidget(self.h2h_table)
        history_box.setLayout(history_layout)
        
        # Player breakdown tables
        tables_layout = QHBoxLayout()
        
        home_box = QGroupBox("Home Team Players")
        home_box_layout = QVBoxLayout()
        home_box_layout.addWidget(self.home_table)
        home_box.setLayout(home_box_layout)
        
        away_box = QGroupBox("Away Team Players")
        away_box_layout = QVBoxLayout()
        away_box_layout.addWidget(self.away_table)
        away_box.setLayout(away_box_layout)
        
        tables_layout.addWidget(home_box)
        tables_layout.addWidget(away_box)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(game_row)
        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addWidget(pred_box)
        layout.addWidget(history_box)
        layout.addLayout(tables_layout)
        self.setLayout(layout)
        
        self._games_data = []  # Store game data for selection
        self.refresh()

    def refresh(self) -> None:
        # Load teams for manual selection
        teams = _teams_df()
        self.home_combo.clear()
        self.away_combo.clear()
        for _, row in teams.iterrows():
            label = f"{row['abbreviation']} - {row['name']}"
            self.home_combo.addItem(label, int(row["team_id"]))
            self.away_combo.addItem(label, int(row["team_id"]))
        
        # Load scheduled games
        self.game_combo.blockSignals(True)
        self.game_combo.clear()
        self.game_combo.addItem("-- Select a game --", None)
        
        try:
            self._games_data = get_scheduled_games()
        except Exception:
            self._games_data = []
        
        for game in self._games_data:
            date_str = str(game.get("game_date", ""))
            label = f"{date_str}: {game['away_abbr']} @ {game['home_abbr']}"
            self.game_combo.addItem(label, game)
        
        self.game_combo.blockSignals(False)

    def _on_game_selected(self, index: int) -> None:
        """When a game is selected from dropdown, populate home/away teams."""
        if index <= 0:
            return
        game = self.game_combo.currentData()
        if not game:
            return
        
        # Find and select home team
        home_id = game.get("home_team_id")
        away_id = game.get("away_team_id")
        
        for i in range(self.home_combo.count()):
            if self.home_combo.itemData(i) == home_id:
                self.home_combo.setCurrentIndex(i)
                break
        
        for i in range(self.away_combo.count()):
            if self.away_combo.itemData(i) == away_id:
                self.away_combo.setCurrentIndex(i)
                break
        
        # Auto-analyze
        self.predict()

    def predict(self) -> None:
        if self.home_combo.currentData() is None or self.away_combo.currentData() is None:
            self.spread_label.setText("Select teams")
            return
        home_id = int(self.home_combo.currentData())
        away_id = int(self.away_combo.currentData())
        if home_id == away_id:
            self.spread_label.setText("Different teams")
            return

        # Get comprehensive stats for both teams
        home_stats = get_team_matchup_stats(home_id, opponent_team_id=away_id, is_home=True)
        away_stats = get_team_matchup_stats(away_id, opponent_team_id=home_id, is_home=False)
        
        if not home_stats.players or not away_stats.players:
            self.spread_label.setText("No data")
            self.total_label.setText("Run sync")
            return
        
        # Use full prediction engine with all advanced factors
        home_player_ids = [p.player_id for p in home_stats.players if not p.is_injured]
        away_player_ids = [p.player_id for p in away_stats.players if not p.is_injured]
        pred = predict_matchup(
            home_team_id=home_id,
            away_team_id=away_id,
            home_players=home_player_ids,
            away_players=away_player_ids,
        )
        spread = pred.predicted_spread
        total = pred.predicted_total
        home_proj = pred.predicted_home_score
        away_proj = pred.predicted_away_score
        
        # Update summary labels
        self.spread_label.setText(f"{spread:+.1f}")
        self.total_label.setText(f"{total:.1f}")
        self.home_proj_label.setText(f"{home_proj:.1f}")
        self.away_proj_label.setText(f"{away_proj:.1f}")
        
        # Update injury impact labels
        self._update_injury_labels(home_stats, self.home_injury_label)
        self._update_injury_labels(away_stats, self.away_injury_label)
        
        # Update backtest / historical performance
        self._update_backtest(home_id, away_id)
        
        # Populate player tables
        self._populate_table(self.home_table, home_stats, away_id, is_home=True)
        self._populate_table(self.away_table, away_stats, home_id, is_home=False)
    
    def _update_backtest(self, home_id: int, away_id: int) -> None:
        """Update the historical performance section."""
        try:
            data = get_matchup_backtest(home_id, away_id)
        except Exception:
            self.home_record_label.setText("No data")
            self.away_record_label.setText("No data")
            self.h2h_label.setText("No data")
            return
        
        home_abbr = data["home_abbr"]
        away_abbr = data["away_abbr"]
        
        # Home team at home record
        hw = data["home_record_at_home"]["wins"]
        hl = data["home_record_at_home"]["losses"]
        home_pct = (hw / (hw + hl) * 100) if (hw + hl) > 0 else 0
        self.home_record_label.setText(
            f"{home_abbr}: {hw}-{hl} ({home_pct:.0f}%) | Avg {data['home_avg_home']:.1f} pts"
        )
        if home_pct >= 60:
            self.home_record_label.setStyleSheet("color: green; font-weight: bold;")
        elif home_pct <= 40:
            self.home_record_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.home_record_label.setStyleSheet("")
        
        # Away team on road record
        aw = data["away_record_on_road"]["wins"]
        al = data["away_record_on_road"]["losses"]
        away_pct = (aw / (aw + al) * 100) if (aw + al) > 0 else 0
        self.away_record_label.setText(
            f"{away_abbr}: {aw}-{al} ({away_pct:.0f}%) | Avg {data['away_avg_road']:.1f} pts"
        )
        if away_pct >= 60:
            self.away_record_label.setStyleSheet("color: green; font-weight: bold;")
        elif away_pct <= 40:
            self.away_record_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.away_record_label.setStyleSheet("")
        
        # Head-to-head
        h2h = data["h2h_games"]
        if h2h:
            home_wins = sum(1 for g in h2h if g["winner"] == home_abbr)
            away_wins = len(h2h) - home_wins
            self.h2h_label.setText(f"{home_abbr} {home_wins} - {away_wins} {away_abbr}")
            
            # Populate H2H table
            self.h2h_table.clear()
            self.h2h_table.setColumnCount(4)
            self.h2h_table.setHorizontalHeaderLabels(["Date", "Score", "Winner", "Total"])
            self.h2h_table.setRowCount(len(h2h))
            
            for row, game in enumerate(h2h):
                score = f"{int(game['away_score'])}-{int(game['home_score'])}"
                total = int(game["home_score"]) + int(game["away_score"])
                items = [game["date"], score, game["winner"], str(total)]
                for col, val in enumerate(items):
                    self.h2h_table.setItem(row, col, QTableWidgetItem(val))
            
            self.h2h_table.resizeColumnsToContents()
            self.h2h_table.setVisible(True)
        else:
            self.h2h_label.setText("No games this season")
            self.h2h_table.setRowCount(0)
            self.h2h_table.setVisible(False)

    def _update_injury_labels(self, stats: TeamMatchupStats, label: QLabel) -> None:
        """Update injury impact label for a team."""
        injured = [p for p in stats.players if p.is_injured and p.mpg > 0]
        
        if not injured:
            label.setText("No injuries")
            label.setStyleSheet("color: green;")
            return
        
        # Calculate total lost points
        lost_points = sum(p.ppg for p in injured)
        
        # Determine impact level and get position info
        key_injuries = [p for p in injured if p.mpg >= 25]
        rotation_injuries = [p for p in injured if 15 <= p.mpg < 25]
        
        def get_pos_abbr(pos: str) -> str:
            """Get short position abbreviation."""
            pos = pos.upper()
            if "G" in pos or "GUARD" in pos:
                return "G"
            if "C" in pos or "CENTER" in pos:
                return "C"
            return "F"
        
        if key_injuries:
            # Show name and position for key injuries
            parts = []
            for p in key_injuries[:2]:
                last_name = p.name.split()[-1]
                pos = get_pos_abbr(p.position)
                parts.append(f"{last_name}({pos})")
            names = ", ".join(parts)
            label.setText(f"KEY OUT: {names} (-{lost_points:.0f} PPG)")
            label.setStyleSheet("color: darkred; font-weight: bold;")
        elif rotation_injuries:
            parts = []
            for p in rotation_injuries[:2]:
                last_name = p.name.split()[-1]
                pos = get_pos_abbr(p.position)
                parts.append(f"{last_name}({pos})")
            names = ", ".join(parts)
            label.setText(f"OUT: {names} (-{lost_points:.0f} PPG)")
            label.setStyleSheet("color: red;")
        else:
            label.setText(f"{len(injured)} minor injuries")
            label.setStyleSheet("color: orange;")

    def _populate_table(self, table: QTableWidget, stats: TeamMatchupStats, opp_id: int, is_home: bool) -> None:
        headers = ["Player", "Pos", "PPG", "RPG", "APG", "MPG", "Home", "Away", "vs Opp", "Proj", "Status"]
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        
        # Only show top 12 by minutes
        players = [p for p in stats.players if p.mpg > 0][:12]
        table.setRowCount(len(players))
        
        for row_idx, p in enumerate(players):
            # Calculate this player's projected contribution
            base = p.ppg * 0.4
            loc = (p.ppg_home if is_home else p.ppg_away) * 0.3
            vs = (p.ppg_vs_opp if p.games_vs_opp > 0 else p.ppg) * 0.3
            proj = base + loc + vs
            
            status = "INJ" if p.is_injured else "OK"
            
            items = [
                p.name,
                p.position,
                f"{p.ppg:.1f}",
                f"{p.rpg:.1f}",
                f"{p.apg:.1f}",
                f"{p.mpg:.1f}",
                f"{p.ppg_home:.1f}",
                f"{p.ppg_away:.1f}",
                f"{p.ppg_vs_opp:.1f}" if p.games_vs_opp > 0 else "--",
                f"{proj:.1f}",
                status,
            ]
            
            for col_idx, val in enumerate(items):
                item = QTableWidgetItem(val)
                if p.is_injured:
                    item.setForeground(Qt.GlobalColor.red)
                table.setItem(row_idx, col_idx, item)
        
        table.resizeColumnsToContents()
