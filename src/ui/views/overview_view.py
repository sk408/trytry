"""Overview tab — today's games with team logos, odds, and predictions."""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QFont

from src.ui.workers import start_overview_worker

logger = logging.getLogger(__name__)


class OverviewView(QWidget):
    """Overview of today's games with team logos and broadcast styling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._current_worker = None
        self._team_id_map = {}  # abbreviation -> team_id

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

        # Table with logo columns
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "", "Away", "@", "Home", "",
            "Status", "Pred Spread", "Vegas Spread", "Pred Total", "Sharp",
        ])
        self.table.setIconSize(Qt.QSize(28, 28) if hasattr(Qt, 'QSize') else __import__('PySide6.QtCore', fromlist=['QSize']).QSize(28, 28))
        from PySide6.QtCore import QSize
        self.table.setIconSize(QSize(28, 28))

        header_view = self.table.horizontalHeader()
        # Logo columns fixed width
        for col in (0, 4):
            header_view.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(col, 36)
        # @ column narrow
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 24)
        # Team name columns stretch
        for col in (1, 3):
            header_view.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        # Data columns resize to contents
        for col in (5, 6, 7, 8, 9):
            header_view.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        layout.addWidget(self.table)

        # Build team_id map for logo lookup
        self._build_team_map()

    def _build_team_map(self):
        """Build abbreviation -> team_id map for logo lookup."""
        try:
            from src.database import db
            teams = db.fetch_all("SELECT team_id, abbreviation FROM teams")
            for t in teams:
                self._team_id_map[t["abbreviation"]] = t["team_id"]
        except Exception:
            pass

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
        try:
            self.status_lbl.setText(str(msg))
        except RuntimeError:
            pass

    def _on_result(self, data: dict):
        try:
            games = data.get("games", [])
            self.table.setRowCount(len(games))
            from src.ui.widgets.image_utils import get_team_logo, make_placeholder_logo
            from src.ui.widgets.nba_colors import get_team_colors

            for r, g in enumerate(games):
                away_abbr = g.get('away_team', '?')
                home_abbr = g.get('home_team', '?')
                away_tid = self._team_id_map.get(away_abbr)
                home_tid = self._team_id_map.get(home_abbr)

                self.table.setRowHeight(r, 40)

                # Away logo
                away_logo_item = QTableWidgetItem()
                if away_tid:
                    logo = get_team_logo(away_tid, 28)
                    if logo:
                        away_logo_item.setData(Qt.ItemDataRole.DecorationRole, logo)
                    else:
                        primary, _ = get_team_colors(away_tid)
                        away_logo_item.setData(Qt.ItemDataRole.DecorationRole,
                                               make_placeholder_logo(away_abbr, 28, primary))
                self.table.setItem(r, 0, away_logo_item)

                # Away team name
                away_item = QTableWidgetItem(away_abbr)
                away_item.setFont(QFont("Oswald", 13, QFont.Weight.DemiBold))
                self.table.setItem(r, 1, away_item)

                # @ separator
                at_item = QTableWidgetItem("@")
                at_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                at_item.setForeground(QColor("#64748b"))
                self.table.setItem(r, 2, at_item)

                # Home team name
                home_item = QTableWidgetItem(home_abbr)
                home_item.setFont(QFont("Oswald", 13, QFont.Weight.DemiBold))
                self.table.setItem(r, 3, home_item)

                # Home logo
                home_logo_item = QTableWidgetItem()
                if home_tid:
                    logo = get_team_logo(home_tid, 28)
                    if logo:
                        home_logo_item.setData(Qt.ItemDataRole.DecorationRole, logo)
                    else:
                        primary, _ = get_team_colors(home_tid)
                        home_logo_item.setData(Qt.ItemDataRole.DecorationRole,
                                               make_placeholder_logo(home_abbr, 28, primary))
                self.table.setItem(r, 4, home_logo_item)

                # Status with score
                status = str(g.get('status', ''))
                home_sc = g.get('home_score', 0) or 0
                away_sc = g.get('away_score', 0) or 0
                if home_sc or away_sc:
                    status_text = f"{away_sc}-{home_sc}"
                    if g.get('clock'):
                        status_text += f" {g['clock']}"
                elif g.get('clock'):
                    status_text = g['clock']
                else:
                    status_text = status

                status_item = QTableWidgetItem(status_text)
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # Color code: live=green, final=muted, pre=cyan
                state = str(g.get('status_state', '')).lower()
                if state == 'in' or (home_sc and 'final' not in status.lower()):
                    status_item.setForeground(QColor("#22c55e"))
                    status_item.setFont(QFont("Oswald", 11, QFont.Weight.Bold))
                elif 'final' in status.lower():
                    status_item.setForeground(QColor("#64748b"))
                else:
                    status_item.setForeground(QColor("#00e5ff"))
                self.table.setItem(r, 5, status_item)

                # Prediction data
                pred = g.get('prediction') or {}
                odds = g.get('odds') or {}

                # Pred spread with edge coloring
                p_spread = pred.get('spread')
                try:
                    p_spread_str = f"{float(p_spread):+.1f}" if p_spread is not None else "-"
                except (TypeError, ValueError):
                    p_spread_str = str(p_spread)

                item_ps = QTableWidgetItem(p_spread_str)
                item_ps.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_ps.setFont(QFont("Oswald", 12, QFont.Weight.Bold))
                self.table.setItem(r, 6, item_ps)

                # Vegas spread
                v_spread_raw = odds.get('spread')
                v_spread = str(v_spread_raw) if v_spread_raw else "-"
                item_vs = QTableWidgetItem(v_spread)
                item_vs.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, 7, item_vs)

                # Pred total
                p_tot = pred.get('total')
                try:
                    p_tot_str = f"{float(p_tot):.1f}" if p_tot is not None else "-"
                except (TypeError, ValueError):
                    p_tot_str = str(p_tot)
                item_pt = QTableWidgetItem(p_tot_str)
                item_pt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, 8, item_pt)

                # Sharp edge with color coding
                sh_pub = pred.get('sharp_home_public', 0) or 0
                sh_mon = pred.get('sharp_home_money', 0) or 0
                if sh_pub > 0 and sh_mon > 0:
                    edge = sh_mon - sh_pub
                    sharp_str = f"{edge:+.0f}%"
                    sharp_adj = float(edge)
                else:
                    sharp_adj = 0.0
                    sharp_str = "-"

                item_sh = QTableWidgetItem(sharp_str)
                item_sh.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_sh.setFont(QFont("Oswald", 11, QFont.Weight.Bold))
                if sharp_adj >= 5:
                    item_sh.setForeground(QColor("#22c55e"))
                elif sharp_adj <= -5:
                    item_sh.setForeground(QColor("#ef4444"))
                self.table.setItem(r, 9, item_sh)

                # Color the edge between pred spread and vegas spread
                if p_spread is not None and v_spread_raw:
                    try:
                        edge_val = float(p_spread) - float(v_spread_raw)
                        if abs(edge_val) >= 2.0:
                            item_ps.setForeground(QColor("#22c55e") if edge_val > 0 else QColor("#ef4444"))
                    except (TypeError, ValueError):
                        pass

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
