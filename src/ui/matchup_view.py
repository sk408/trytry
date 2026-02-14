from __future__ import annotations

from typing import Dict

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.analytics.stats_engine import get_team_matchup_stats, TeamMatchupStats
from src.analytics.prediction import predict_matchup
from src.data.image_cache import get_team_logo_pixmap, get_player_photo_pixmap
from src.data.sync_service import sync_schedule
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


def _pred_card(title: str, value: str = "--", accent: str = "#3b82f6") -> QFrame:
    """Small prediction metric card."""
    card = QFrame()
    card.setStyleSheet(
        f"QFrame {{ background: #1c2e42; border: 1px solid #2a3f55;"
        f"  border-radius: 8px; border-left: 3px solid {accent}; }}"
    )
    lay = QVBoxLayout()
    lay.setContentsMargins(12, 8, 12, 8)
    t = QLabel(title)
    t.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 600;"
                     " text-transform: uppercase; letter-spacing: 0.5px;")
    v = QLabel(value)
    v.setObjectName("card_value")
    v.setStyleSheet(f"color: {accent}; font-size: 24px; font-weight: 700;")
    lay.addWidget(t)
    lay.addWidget(v)
    card.setLayout(lay)
    return card


class MatchupView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        
        # Game selector
        self.game_combo = QComboBox()
        self.game_combo.currentIndexChanged.connect(self._on_game_selected)  # type: ignore[arg-type]
        
        self.home_combo = QComboBox()
        self.away_combo = QComboBox()
        
        # Prediction summary cards
        self.spread_card = _pred_card("Spread (Home)", "--", "#3b82f6")
        self.total_card = _pred_card("Over / Under", "--", "#f59e0b")
        self.home_proj_card = _pred_card("Home Projected", "--", "#10b981")
        self.away_proj_card = _pred_card("Away Projected", "--", "#ef4444")

        # Convenience refs to the value labels inside cards
        self.spread_label = self.spread_card.findChild(QLabel, "card_value")
        self.total_label = self.total_card.findChild(QLabel, "card_value")
        self.home_proj_label = self.home_proj_card.findChild(QLabel, "card_value")
        self.away_proj_label = self.away_proj_card.findChild(QLabel, "card_value")
        
        # Injury impact labels
        self.home_injury_label = QLabel("No injuries")
        self.home_injury_label.setStyleSheet("color: #10b981; font-size: 12px;")
        self.away_injury_label = QLabel("No injuries")
        self.away_injury_label.setStyleSheet("color: #10b981; font-size: 12px;")
        
        # Backtest accuracy labels
        self.home_record_label = QLabel("--")
        self.away_record_label = QLabel("--")
        self.h2h_label = QLabel("--")
        self.h2h_table = QTableWidget()
        self.h2h_table.setMaximumHeight(120)
        self.h2h_table.setAlternatingRowColors(True)
        self.h2h_table.verticalHeader().setVisible(False)
        
        # Player tables
        self.home_table = QTableWidget()
        self.home_table.setAlternatingRowColors(True)
        self.home_table.verticalHeader().setVisible(False)
        self.away_table = QTableWidget()
        self.away_table.setAlternatingRowColors(True)
        self.away_table.verticalHeader().setVisible(False)

        self.refresh_button = QPushButton("Load Games")
        self.refresh_button.clicked.connect(self.refresh)  # type: ignore[arg-type]
        self.predict_button = QPushButton("  Analyze Matchup")
        self.predict_button.setProperty("cssClass", "primary")
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

        # Team logo labels (shown next to projected scores)
        self.home_logo_lbl = QLabel()
        self.home_logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.home_logo_lbl.setFixedSize(48, 48)
        self.away_logo_lbl = QLabel()
        self.away_logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.away_logo_lbl.setFixedSize(48, 48)

        # Prediction summary – cards in a row
        pred_box = QGroupBox("Prediction Summary")
        pred_layout = QHBoxLayout()
        pred_layout.setSpacing(12)

        # Spread + Total cards
        pred_layout.addWidget(self.spread_card)
        pred_layout.addWidget(self.total_card)

        # Home projected: logo + card + injury
        home_col = QVBoxLayout()
        home_top = QHBoxLayout()
        home_top.addWidget(self.home_logo_lbl)
        home_top.addWidget(self.home_proj_card, stretch=1)
        home_col.addLayout(home_top)
        home_col.addWidget(self.home_injury_label)
        pred_layout.addLayout(home_col)

        # Away projected: logo + card + injury
        away_col = QVBoxLayout()
        away_top = QHBoxLayout()
        away_top.addWidget(self.away_logo_lbl)
        away_top.addWidget(self.away_proj_card, stretch=1)
        away_col.addLayout(away_top)
        away_col.addWidget(self.away_injury_label)
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
        
        # Load scheduled games (today + 14 days, sorted chronologically)
        self.game_combo.blockSignals(True)
        self.game_combo.clear()
        self.game_combo.addItem("-- Select a game --", None)
        
        self._games_data = []
        try:
            from datetime import date as _date
            sched_df = sync_schedule(include_future_days=14)
            if not sched_df.empty:
                sched_df = sched_df.copy()
                sched_df["game_date"] = pd.to_datetime(sched_df["game_date"]).dt.date
                today = _date.today()
                sched_df = sched_df[sched_df["game_date"] >= today]
                sort_cols = ["game_date", "game_time"] if "game_time" in sched_df.columns else ["game_date"]
                sched_df = sched_df.sort_values(sort_cols)
                sched_df = sched_df.drop_duplicates(subset=["game_date", "home_team_id", "away_team_id"])
                for _, r in sched_df.iterrows():
                    game = {
                        "game_date": r["game_date"],
                        "home_team_id": int(r["home_team_id"]),
                        "away_team_id": int(r["away_team_id"]),
                        "home_abbr": r.get("home_abbr", ""),
                        "away_abbr": r.get("away_abbr", ""),
                        "game_time": str(r.get("game_time", "") or ""),
                    }
                    self._games_data.append(game)
        except Exception:
            pass
        
        for game in self._games_data:
            gd = game["game_date"]
            date_fmt = gd.strftime("%a %m/%d") if hasattr(gd, "strftime") else str(gd)
            time_str = game.get("game_time", "")
            time_part = f" - {time_str}" if time_str and time_str != "TBD" else ""
            label = f"{date_fmt}: {game['away_abbr']} @ {game['home_abbr']}{time_part}"
            self.game_combo.addItem(label, game)
        
        self.game_combo.blockSignals(False)

    def _on_game_selected(self, index: int) -> None:
        """When a game is selected from dropdown, populate home/away teams."""
        if index <= 0:
            return
        game = self.game_combo.currentData()
        if not game:
            return
        
        home_id = game.get("home_team_id")
        away_id = game.get("away_team_id")
        self._select_teams_and_predict(home_id, away_id)

    def load_game(self, home_team_id: int, away_team_id: int) -> None:
        """
        Public method to load a specific matchup (called externally, e.g. from schedule view).
        Ensures teams are loaded, selects them, and runs prediction.
        """
        # Make sure team combos are populated
        if self.home_combo.count() == 0:
            self.refresh()
        self._select_teams_and_predict(home_team_id, away_team_id)

    def _select_teams_and_predict(self, home_id: int, away_id: int) -> None:
        """Select the given teams in the combos and auto-analyze."""
        if not home_id or not away_id:
            return

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
        
        # Use full prediction engine; include players with play_probability >= 0.2
        # (Injury Intelligence) instead of binary is_injured
        _min_prob = 0.2
        home_player_ids = [
            p.player_id for p in home_stats.players
            if getattr(p, "play_probability", 1.0) >= _min_prob
        ]
        away_player_ids = [
            p.player_id for p in away_stats.players
            if getattr(p, "play_probability", 1.0) >= _min_prob
        ]
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
        
        # Update summary labels (negate spread for betting convention:
        # negative = home favored, positive = home underdog)
        display_spread = -spread
        self.spread_label.setText(f"{display_spread:+.1f}")
        self.total_label.setText(f"{total:.1f}")
        self.home_proj_label.setText(f"{home_proj:.1f}")
        self.away_proj_label.setText(f"{away_proj:.1f}")

        # Team logos
        self.home_logo_lbl.setPixmap(get_team_logo_pixmap(home_id, 44))
        self.away_logo_lbl.setPixmap(get_team_logo_pixmap(away_id, 44))

        # Color the spread: green if home favored (negative), red if away favored (positive)
        if display_spread < -2:
            self.spread_label.setStyleSheet("color: #10b981; font-size: 24px; font-weight: 700;")
        elif display_spread > 2:
            self.spread_label.setStyleSheet("color: #ef4444; font-size: 24px; font-weight: 700;")
        else:
            self.spread_label.setStyleSheet("color: #f59e0b; font-size: 24px; font-weight: 700;")
        
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
            self.home_record_label.setStyleSheet("color: #10b981; font-weight: bold;")
        elif home_pct <= 40:
            self.home_record_label.setStyleSheet("color: #ef4444; font-weight: bold;")
        else:
            self.home_record_label.setStyleSheet("color: #e2e8f0;")
        
        # Away team on road record
        aw = data["away_record_on_road"]["wins"]
        al = data["away_record_on_road"]["losses"]
        away_pct = (aw / (aw + al) * 100) if (aw + al) > 0 else 0
        self.away_record_label.setText(
            f"{away_abbr}: {aw}-{al} ({away_pct:.0f}%) | Avg {data['away_avg_road']:.1f} pts"
        )
        if away_pct >= 60:
            self.away_record_label.setStyleSheet("color: #10b981; font-weight: bold;")
        elif away_pct <= 40:
            self.away_record_label.setStyleSheet("color: #ef4444; font-weight: bold;")
        else:
            self.away_record_label.setStyleSheet("color: #e2e8f0;")
        
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
        """Update injury impact label for a team, now with play probabilities."""
        injured = [p for p in stats.players if p.is_injured and p.mpg > 0]

        if not injured:
            label.setText("No injuries")
            label.setStyleSheet("color: #10b981; font-size: 12px;")
            return

        def get_pos_abbr(pos: str) -> str:
            pos = pos.upper()
            if "G" in pos or "GUARD" in pos:
                return "G"
            if "C" in pos or "CENTER" in pos:
                return "C"
            return "F"

        # Weighted lost points (scaled by absent fraction)
        lost_points = sum(p.ppg * (1.0 - getattr(p, "play_probability", 0.0)) for p in injured)

        # Separate definite outs from uncertain
        definite_out = [p for p in injured if getattr(p, "play_probability", 0.0) <= 0]
        uncertain = [p for p in injured if 0 < getattr(p, "play_probability", 0.0) < 1.0]

        parts = []
        for p in sorted(definite_out, key=lambda x: x.mpg, reverse=True)[:2]:
            last_name = p.name.split()[-1]
            pos = get_pos_abbr(p.position)
            parts.append(f"{last_name}({pos}) OUT")

        for p in sorted(uncertain, key=lambda x: x.mpg, reverse=True)[:2]:
            last_name = p.name.split()[-1]
            pp = getattr(p, "play_probability", 0.0)
            status = getattr(p, "injury_status", "?")
            parts.append(f"{last_name} {status} {pp*100:.0f}%")

        text = ", ".join(parts)
        if lost_points > 0:
            text += f" (-{lost_points:.0f} exp PPG)"

        if any(p.mpg >= 25 for p in definite_out):
            label.setText(f"KEY: {text}")
            label.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 12px;")
        elif uncertain:
            label.setText(text)
            label.setStyleSheet("color: #f59e0b; font-size: 12px;")
        elif definite_out:
            label.setText(text)
            label.setStyleSheet("color: #f59e0b; font-size: 12px;")
        else:
            label.setText(f"{len(injured)} minor injuries")
            label.setStyleSheet("color: #94a3b8; font-size: 12px;")

    def _populate_table(self, table: QTableWidget, stats: TeamMatchupStats, opp_id: int, is_home: bool) -> None:
        headers = ["", "Player", "Pos", "PPG", "RPG", "APG", "MPG", "Home", "Away", "vs Opp", "Proj", "Status"]
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        
        # Only show top 12 by minutes
        players = [p for p in stats.players if p.mpg > 0][:12]
        table.setRowCount(len(players))
        table.setIconSize(self._player_icon_size())
        
        for row_idx, p in enumerate(players):
            table.setRowHeight(row_idx, 36)

            # Calculate this player's projected contribution
            base = p.ppg * 0.4
            loc = (p.ppg_home if is_home else p.ppg_away) * 0.3
            vs = (p.ppg_vs_opp if p.games_vs_opp > 0 else p.ppg) * 0.3
            proj = base + loc + vs
            
            # Rich injury status with play probability
            pp = getattr(p, "play_probability", 1.0)
            inj_status = getattr(p, "injury_status", "")
            if not p.is_injured:
                status = "OK"
            elif pp <= 0:
                status = "OUT"
            elif inj_status:
                status = f"{inj_status} {pp*100:.0f}%"
            else:
                status = "INJ"

            # Player headshot in column 0
            photo_pm = get_player_photo_pixmap(p.player_id, 30)
            photo_item = QTableWidgetItem()
            photo_item.setIcon(QIcon(photo_pm))
            table.setItem(row_idx, 0, photo_item)

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
                    if pp <= 0:
                        item.setForeground(QColor("#ef4444"))        # red — OUT
                    elif pp < 0.5:
                        item.setForeground(QColor("#f97316"))        # orange — unlikely
                    elif pp < 0.8:
                        item.setForeground(QColor("#eab308"))        # yellow — uncertain
                    else:
                        item.setForeground(QColor("#22d3ee"))        # cyan — probable
                table.setItem(row_idx, col_idx + 1, item)  # +1 for photo column
        
        table.setColumnWidth(0, 40)  # photo column
        table.resizeColumnsToContents()

    @staticmethod
    def _player_icon_size():
        from PySide6.QtCore import QSize
        return QSize(30, 30)
