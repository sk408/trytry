"""Overview tab — today's games, odds, and predictions."""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt

from src.ui.workers import start_overview_worker

logger = logging.getLogger(__name__)

class OverviewView(QWidget):
    """Overview of today's games."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._current_worker = None

        layout = QVBoxLayout(self)

        header = QLabel("Today's Games Overview")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # Controls
        ctrl = QHBoxLayout()
        run_btn = QPushButton("Refresh Overview")
        run_btn.clicked.connect(self._on_refresh)
        ctrl.addWidget(run_btn)
        
        self.status_lbl = QLabel("")
        ctrl.addWidget(self.status_lbl)
        
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Matchup", "Status", "Score", 
            "Pred Spread", "Vegas Spread", "Pred Total", "Vegas Total", "Sharp Edge"
        ])
        
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 8):
            header_view.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
            
        layout.addWidget(self.table)
        
    def _on_refresh(self):
        if self._current_worker and self._current_worker.isRunning():
            self.status_lbl.setText("Already refreshing...")
            return

        self.table.setRowCount(0)
        self.status_lbl.setText("Refreshing...")
        self._current_worker = start_overview_worker(
            on_progress=self._on_progress,
            on_result=self._on_result,
            on_done=self._on_done
        )

    def _on_progress(self, msg: str):
        self.status_lbl.setText(msg)

    def _on_result(self, data: dict):
        try:
            games = data.get("games", [])
            self.table.setRowCount(len(games))
            for r, g in enumerate(games):
                matchup = f"{g.get('away_team', '?')} @ {g.get('home_team', '?')}"
                status = str(g.get('status', ''))
                if g.get('clock'):
                    status += f" ({g['clock']})"

                home_sc = g.get('home_score', 0) or 0
                away_sc = g.get('away_score', 0) or 0
                score = f"{away_sc} - {home_sc}" if home_sc or away_sc else "-"

                pred = g.get('prediction') or {}
                odds = g.get('odds') or {}

                p_spread = pred.get('spread')
                try:
                    p_spread_str = f"{float(p_spread):+.1f}" if p_spread is not None else "-"
                except (TypeError, ValueError):
                    p_spread_str = str(p_spread)

                v_spread_raw = odds.get('spread')
                v_spread = str(v_spread_raw) if v_spread_raw else "-"

                p_tot = pred.get('total')
                try:
                    p_tot_str = f"{float(p_tot):.1f}" if p_tot is not None else "-"
                except (TypeError, ValueError):
                    p_tot_str = str(p_tot)

                v_tot = odds.get('over_under')
                v_tot_str = str(v_tot) if v_tot is not None else "-"

                sharp_adj = pred.get('sharp_money_adj', 0.0) or 0.0
                try:
                    sharp_adj = float(sharp_adj)
                    sharp_str = f"{sharp_adj:+.2f} pts" if sharp_adj else "-"
                except (TypeError, ValueError):
                    sharp_adj = 0.0
                    sharp_str = "-"

                self.table.setItem(r, 0, QTableWidgetItem(matchup))
                self.table.setItem(r, 1, QTableWidgetItem(status))
                self.table.setItem(r, 2, QTableWidgetItem(score))

                item_ps = QTableWidgetItem(p_spread_str)
                item_ps.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, 3, item_ps)

                item_vs = QTableWidgetItem(v_spread)
                item_vs.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, 4, item_vs)

                item_pt = QTableWidgetItem(p_tot_str)
                item_pt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, 5, item_pt)

                item_vt = QTableWidgetItem(v_tot_str)
                item_vt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, 6, item_vt)

                item_sh = QTableWidgetItem(sharp_str)
                item_sh.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if sharp_adj > 0.5:
                    item_sh.setForeground(Qt.GlobalColor.darkGreen)
                elif sharp_adj < -0.5:
                    item_sh.setForeground(Qt.GlobalColor.darkRed)
                self.table.setItem(r, 7, item_sh)
        except Exception as e:
            logger.error(f"Overview result display failed: {e}", exc_info=True)
            self.status_lbl.setText(f"Display error: {e}")

    def _on_done(self):
        try:
            self.status_lbl.setText("Ready")
            self._current_worker = None
            if self.main_window:
                self.main_window.set_status("Overview refreshed.")
        except Exception as e:
            logger.error(f"Overview _on_done failed: {e}", exc_info=True)
