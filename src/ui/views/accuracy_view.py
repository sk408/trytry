"""Accuracy tab — 12 QObject workers, summary cards, team accuracy table."""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QFrame, QGridLayout, QComboBox, QTabWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from src.ui.workers import (
    start_backtest_worker, start_optimizer_worker,
    start_calibration_worker, start_feature_importance_worker,
    start_ml_train_worker, start_team_refine_worker,
    start_fft_worker, start_combo_worker,
    start_continuous_worker, start_pipeline_worker,
    start_ml_feature_worker, start_grouped_feature_worker,
    start_diagnostic_csv_worker, start_odds_sync_worker,
)

logger = logging.getLogger(__name__)


class AccuracyView(QWidget):
    """Backtest accuracy, optimization, and analysis tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._current_worker = None
        self._last_backtest_results = None

        layout = QVBoxLayout(self)

        header = QLabel("Model Accuracy & Optimization")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # Action buttons (row 1)
        btn_row1 = QHBoxLayout()
        actions1 = [
            ("Backtest", self._on_backtest),
            ("Optimize Weights", self._on_optimize),
            ("Calibrate", self._on_calibrate),
            ("Feature Importance", self._on_feature_importance),
        ]
        for text, handler in actions1:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn_row1.addWidget(btn)
        layout.addLayout(btn_row1)

        # Action buttons (row 2)
        btn_row2 = QHBoxLayout()
        actions2 = [
            ("Train ML", self._on_ml_train),
            ("Team Refinement", self._on_team_refine),
            ("FFT Analysis", self._on_fft),
            ("Full Pipeline", self._on_pipeline),
            ("Sync Odds", self._on_sync_odds),
        ]
        for text, handler in actions2:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn_row2.addWidget(btn)

        self.diag_btn = QPushButton("Export Diagnostic CSV")
        self.diag_btn.setToolTip("Export game-by-game CSV for bottom 10 teams with all metrics")
        self.diag_btn.clicked.connect(self._on_diagnostic_csv)
        btn_row2.addWidget(self.diag_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("class", "danger")
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row2.addWidget(self.stop_btn)
        layout.addLayout(btn_row2)

        # Summary cards (row 1)
        cards_layout = QGridLayout()
        self.winner_card = self._make_card("Winner Accuracy", "—")
        self.spread5_card = self._make_card("Spread ≤5", "—")
        self.total10_card = self._make_card("Total ≤10", "—")
        self.games_card = self._make_card("Games Tested", "—")
        self.mae_card = self._make_card("Spread MAE", "—")
        self.total_mae_card = self._make_card("Total MAE", "—")

        cards_layout.addWidget(self.winner_card, 0, 0)
        cards_layout.addWidget(self.spread5_card, 0, 1)
        cards_layout.addWidget(self.total10_card, 0, 2)
        cards_layout.addWidget(self.games_card, 0, 3)
        cards_layout.addWidget(self.mae_card, 0, 4)
        cards_layout.addWidget(self.total_mae_card, 0, 5)
        layout.addLayout(cards_layout)

        # Bias / Home-Away cards (row 2)
        bias_layout = QGridLayout()
        self.home_acc_card = self._make_card("Home Pred Acc", "—")
        self.away_acc_card = self._make_card("Away Pred Acc", "—")
        self.spread_bias_card = self._make_card("Spread Bias", "—")
        self.total_bias_card = self._make_card("Total Bias", "—")
        bias_layout.addWidget(self.home_acc_card, 0, 0)
        bias_layout.addWidget(self.away_acc_card, 0, 1)
        bias_layout.addWidget(self.spread_bias_card, 0, 2)
        bias_layout.addWidget(self.total_bias_card, 0, 3)
        layout.addLayout(bias_layout)

        # Tabbed breakdown area
        self.breakdown_tabs = QTabWidget()

        # --- Team accuracy table ---
        self.team_table = QTableWidget()
        self.team_table.setColumnCount(9)
        self.team_table.setHorizontalHeaderLabels([
            "Team", "Games", "Winner%", "Home W%", "Away W%",
            "Spread ≤5%", "Total ≤10%", "Avg Spread Err", "Avg Total Err",
        ])
        self.team_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.team_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.team_table.setSortingEnabled(True)
        self.breakdown_tabs.addTab(self.team_table, "Per Team")

        # --- Spread range table ---
        self.spread_range_table = QTableWidget()
        self.spread_range_table.setColumnCount(5)
        self.spread_range_table.setHorizontalHeaderLabels([
            "Actual Spread Range", "Games", "Winner%", "Within 5%", "Avg Error",
        ])
        self.spread_range_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.spread_range_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.breakdown_tabs.addTab(self.spread_range_table, "By Spread Range")

        # --- Total range table ---
        self.total_range_table = QTableWidget()
        self.total_range_table.setColumnCount(4)
        self.total_range_table.setHorizontalHeaderLabels([
            "Actual Total Range", "Games", "Within 10%", "Avg Error",
        ])
        self.total_range_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.total_range_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.breakdown_tabs.addTab(self.total_range_table, "By Total Range")

        # --- Home / Away summary ---
        self.home_away_table = QTableWidget()
        self.home_away_table.setColumnCount(2)
        self.home_away_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.home_away_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.home_away_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.breakdown_tabs.addTab(self.home_away_table, "Home vs Away")

        layout.addWidget(self.breakdown_tabs)

        # Log output
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        layout.addWidget(self.log)

    def _make_card(self, title: str, value: str) -> QFrame:
        """Create a summary card."""
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; "
            "border-radius: 8px; padding: 10px; }"
        )
        fl = QVBoxLayout(frame)
        fl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        val_label = QLabel(value)
        val_label.setObjectName(f"{title}_value")
        val_label.setStyleSheet("font-size: 20px; font-weight: 700; color: #3b82f6;")
        val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 10px; color: #64748b;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        fl.addWidget(val_label)
        fl.addWidget(title_label)
        return frame

    def _update_card(self, frame: QFrame, value: str):
        """Update a card's value label."""
        val_label = frame.findChildren(QLabel)[0]
        if val_label:
            val_label.setText(value)

    def _append_log(self, msg: str):
        color = "#94a3b8"
        if "error" in msg.lower():
            color = "#ef4444"
        elif "complete" in msg.lower() or "done" in msg.lower():
            color = "#22c55e"
        elif "%" in msg or "progress" in msg.lower():
            color = "#3b82f6"
        self.log.append(f'<span style="color:{color}">{msg}</span>')

    def _on_results(self, results: dict):
        """Handle backtest results (full dict from run_backtest)."""
        self._last_backtest_results = results
        # ---------- Summary cards ----------
        self._update_card(
            self.winner_card,
            f"{results.get('overall_spread_accuracy', 0):.1f}%"
        )
        self._update_card(
            self.spread5_card,
            f"{results.get('spread_within_5_pct', 0):.1f}%"
        )
        self._update_card(
            self.total10_card,
            f"{results.get('total_within_10_pct', 0):.1f}%"
        )
        self._update_card(
            self.games_card,
            str(results.get("total_games", 0))
        )
        self._update_card(
            self.mae_card,
            f"{results.get('avg_spread_error', 0):.2f}"
        )
        self._update_card(
            self.total_mae_card,
            f"{results.get('avg_total_error', 0):.1f}"
        )

        # ---------- Bias / Home-Away cards ----------
        ha = results.get("home_away", {})
        self._update_card(self.home_acc_card,
                          f"{ha.get('home_pred_accuracy', 0):.1f}%")
        self._update_card(self.away_acc_card,
                          f"{ha.get('away_pred_accuracy', 0):.1f}%")
        bias = results.get("bias", {})
        sb = bias.get("spread_bias", 0)
        tb = bias.get("total_bias", 0)
        sb_color = "#22c55e" if abs(sb) < 1 else ("#f59e0b" if abs(sb) < 3 else "#ef4444")
        tb_color = "#22c55e" if abs(tb) < 3 else ("#f59e0b" if abs(tb) < 7 else "#ef4444")
        self._update_card(self.spread_bias_card, f"{sb:+.2f}")
        self._update_card(self.total_bias_card, f"{tb:+.1f}")
        # Color the bias cards
        for card, color in [(self.spread_bias_card, sb_color),
                            (self.total_bias_card, tb_color)]:
            vl = card.findChildren(QLabel)[0]
            if vl:
                vl.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {color};")

        # ---------- Per-team table ----------
        per_team = results.get("per_team", {})
        teams = sorted(per_team.keys())
        self.team_table.setSortingEnabled(False)
        self.team_table.setRowCount(len(teams))
        for row, team in enumerate(teams):
            t = per_team[team]
            items = [
                (team, None),
                (str(t.get("games", 0)), t.get("games", 0)),
                (f"{t.get('winner_pct', 0):.1f}%", t.get("winner_pct", 0)),
                (f"{t.get('home_winner_pct', 0):.1f}%", t.get("home_winner_pct", 0)),
                (f"{t.get('away_winner_pct', 0):.1f}%", t.get("away_winner_pct", 0)),
                (f"{t.get('spread_accuracy', 0):.1f}%", t.get("spread_accuracy", 0)),
                (f"{t.get('total_accuracy', 0):.1f}%", t.get("total_accuracy", 0)),
                (f"{t.get('avg_spread_error', 0):.2f}", t.get("avg_spread_error", 0)),
                (f"{t.get('avg_total_error', 0):.1f}", t.get("avg_total_error", 0)),
            ]
            for col, (text, sort_val) in enumerate(items):
                item = QTableWidgetItem(text)
                if sort_val is not None:
                    item.setData(Qt.ItemDataRole.UserRole, sort_val)
                self.team_table.setItem(row, col, item)
                # Color-code winner% column
                if col == 2:
                    pct = t.get("winner_pct", 0)
                    if pct >= 65:
                        item.setForeground(QColor("#22c55e"))
                    elif pct < 50:
                        item.setForeground(QColor("#ef4444"))
        self.team_table.setSortingEnabled(True)

        # ---------- Spread range table ----------
        spread_ranges = results.get("spread_ranges", [])
        self.spread_range_table.setRowCount(len(spread_ranges))
        for row, sr in enumerate(spread_ranges):
            self.spread_range_table.setItem(row, 0, QTableWidgetItem(sr["range"]))
            self.spread_range_table.setItem(row, 1, QTableWidgetItem(str(sr["games"])))
            self.spread_range_table.setItem(row, 2, QTableWidgetItem(f"{sr['winner_pct']:.1f}%"))
            self.spread_range_table.setItem(row, 3, QTableWidgetItem(f"{sr['within_5_pct']:.1f}%"))
            self.spread_range_table.setItem(row, 4, QTableWidgetItem(f"{sr['avg_error']:.2f}"))

        # ---------- Total range table ----------
        total_ranges = results.get("total_ranges", [])
        self.total_range_table.setRowCount(len(total_ranges))
        for row, tr in enumerate(total_ranges):
            self.total_range_table.setItem(row, 0, QTableWidgetItem(tr["range"]))
            self.total_range_table.setItem(row, 1, QTableWidgetItem(str(tr["games"])))
            self.total_range_table.setItem(row, 2, QTableWidgetItem(f"{tr['within_10_pct']:.1f}%"))
            self.total_range_table.setItem(row, 3, QTableWidgetItem(f"{tr['avg_error']:.1f}"))

        # ---------- Home / Away detail table ----------
        ha_rows = [
            ("Actual home wins", str(ha.get("actual_home_wins", 0))),
            ("Actual away wins", str(ha.get("actual_away_wins", 0))),
            ("Home win rate (actual)", f"{ha.get('overall_home_win_rate', 0):.1f}%"),
            ("We predicted home win", str(ha.get("pred_home_wins", 0))),
            ("We predicted away win", str(ha.get("pred_away_wins", 0))),
            ("Home pred accuracy", f"{ha.get('home_pred_accuracy', 0):.1f}%"),
            ("Away pred accuracy", f"{ha.get('away_pred_accuracy', 0):.1f}%"),
            ("Spread bias (+ = home lean)",
             f"{bias.get('spread_bias', 0):+.2f}  (pred avg {bias.get('avg_pred_spread', 0):+.2f} vs actual {bias.get('avg_actual_spread', 0):+.2f})"),
            ("Total bias (+ = over)",
             f"{bias.get('total_bias', 0):+.1f}  (pred avg {bias.get('avg_pred_total', 0):.1f} vs actual {bias.get('avg_actual_total', 0):.1f})"),
        ]
        self.home_away_table.setRowCount(len(ha_rows))
        for row, (metric, val) in enumerate(ha_rows):
            self.home_away_table.setItem(row, 0, QTableWidgetItem(metric))
            self.home_away_table.setItem(row, 1, QTableWidgetItem(val))

    def _on_done(self):
        self._append_log("Operation complete")
        if self.main_window:
            self.main_window.set_status("Accuracy operation complete")

    def _on_backtest(self):
        self.log.clear()
        self._current_worker = start_backtest_worker(
            self._append_log, self._on_results, self._on_done
        )

    def _on_optimize(self):
        self.log.clear()
        self._current_worker = start_optimizer_worker(self._append_log, self._on_done)

    def _on_calibrate(self):
        self.log.clear()
        self._current_worker = start_calibration_worker(self._append_log, self._on_done)

    def _on_feature_importance(self):
        self.log.clear()
        self._current_worker = start_feature_importance_worker(
            self._append_log, self._on_done
        )

    def _on_ml_train(self):
        self.log.clear()
        self._current_worker = start_ml_train_worker(self._append_log, self._on_done)

    def _on_team_refine(self):
        self.log.clear()
        self._current_worker = start_team_refine_worker(self._append_log, self._on_done)

    def _on_fft(self):
        self.log.clear()
        self._current_worker = start_fft_worker(self._append_log, self._on_done)

    def _on_pipeline(self):
        self.log.clear()
        self._current_worker = start_pipeline_worker(
            self._append_log, self._on_results, self._on_done
        )

    def _on_sync_odds(self):
        self.log.clear()
        self._current_worker = start_odds_sync_worker(self._append_log, self._on_done)

    def _on_diagnostic_csv(self):
        self.log.clear()
        self._append_log("Starting diagnostic CSV export for worst 10 teams...")
        self._current_worker = start_diagnostic_csv_worker(
            backtest_results=self._last_backtest_results,
            on_progress=self._append_log,
            on_result=self._on_csv_result,
            on_done=self._on_done,
        )

    def _on_csv_result(self, result):
        """Handle diagnostic CSV export result."""
        path = result.get("csv_path", "")
        if path:
            import os
            abs_path = os.path.abspath(path)
            self._append_log(f"CSV saved: {abs_path}")
        else:
            self._append_log("No CSV generated")

    def _on_stop(self):
        if self._current_worker:
            self._current_worker.stop()
        self._append_log("Stop requested")
