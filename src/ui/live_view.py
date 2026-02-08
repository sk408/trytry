from __future__ import annotations

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.analytics.live_recommendations import build_live_recommendations
from src.data.image_cache import get_team_logo_pixmap
from src.data.sync_service import sync_live_scores
from src.database.db import get_conn

# ── Colors ──
_LIVE_BG = QColor(20, 60, 40)        # green-tinted row for in-progress
_FINAL_BG = QColor(40, 30, 50)       # muted purple-tinted row for final
_SCHED_BG = QColor(28, 46, 66)       # default surface for scheduled
_LIVE_TEXT = QColor(16, 185, 129)     # green accent
_FINAL_TEXT = QColor(148, 163, 184)   # muted gray


def _abbr_to_tid_map() -> dict:
    """Build abbreviation → team_id map."""
    try:
        with get_conn() as conn:
            rows = conn.execute("SELECT team_id, abbreviation FROM teams").fetchall()
        return {abbr: tid for tid, abbr in rows}
    except Exception:
        return {}


class LiveView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._abbr_map: dict = {}
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(False)  # we do our own row colors
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        self.refresh_button = QPushButton("  Refresh")
        self.refresh_button.setProperty("cssClass", "primary")
        self.refresh_button.clicked.connect(self.refresh)  # type: ignore[arg-type]

        self.last_update_lbl = QLabel("")
        self.last_update_lbl.setStyleSheet("color: #64748b; font-size: 11px;")

        title = QLabel("Live Games & Recommendations")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.last_update_lbl)
        header.addWidget(self.refresh_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)
        layout.addLayout(header)
        layout.addWidget(self.table)
        self.setLayout(layout)

        # Auto-refresh every 30 seconds
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self.refresh)
        self._auto_timer.start(30_000)

        self.refresh()

    def refresh(self) -> None:
        try:
            sync_live_scores()
        except Exception as exc:  # pragma: no cover
            self._render([], error=str(exc))
            return
        recs = build_live_recommendations()
        self._render(recs)

        from datetime import datetime
        self.last_update_lbl.setText(f"Updated {datetime.now().strftime('%I:%M:%S %p')}")

    def _render(self, recs, error: str | None = None) -> None:
        self.table.clear()
        headers = ["Status", "Home", "Away", "Score", "Period", "Clock",
                    "Proj Spread", "Proj Total"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(recs) if not error else 1)
        self.table.setIconSize(QSize(24, 24))

        if error:
            item = QTableWidgetItem(f"Error: {error}")
            item.setForeground(QColor("#ef4444"))
            self.table.setItem(0, 0, item)
            self.table.resizeColumnsToContents()
            return

        # Lazy-load abbreviation map
        if not self._abbr_map:
            self._abbr_map = _abbr_to_tid_map()

        bold = QFont()
        bold.setBold(True)

        for row_idx, rec in enumerate(recs):
            self.table.setRowHeight(row_idx, 32)
            is_live = rec.status.lower() in ("in_progress", "live", "in progress")
            is_final = rec.status.lower() in ("final", "post")
            bg = _LIVE_BG if is_live else _FINAL_BG if is_final else _SCHED_BG

            status_text = "LIVE" if is_live else "FINAL" if is_final else rec.status
            score = f"{rec.home_score} - {rec.away_score}"

            values = [
                status_text,
                rec.home_team,
                rec.away_team,
                score,
                f"Q{rec.period}" if rec.period else "--",
                rec.clock or "--",
                f"{rec.projected_spread:+.1f}",
                f"{rec.projected_total:.1f}",
            ]

            for col_idx, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setBackground(bg)

                # Team logo icons on Home / Away columns
                if col_idx == 1:  # Home
                    tid = self._abbr_map.get(rec.home_team)
                    if tid:
                        item.setIcon(QIcon(get_team_logo_pixmap(tid, 24)))
                elif col_idx == 2:  # Away
                    tid = self._abbr_map.get(rec.away_team)
                    if tid:
                        item.setIcon(QIcon(get_team_logo_pixmap(tid, 24)))

                # Status column styling
                if col_idx == 0:
                    item.setFont(bold)
                    if is_live:
                        item.setForeground(_LIVE_TEXT)
                    elif is_final:
                        item.setForeground(_FINAL_TEXT)

                # Score column bold
                if col_idx == 3:
                    item.setFont(bold)

                # Spread coloring
                if col_idx == 6:
                    spread_val = rec.projected_spread
                    if spread_val < -3:
                        item.setForeground(QColor("#10b981"))  # home favored
                    elif spread_val > 3:
                        item.setForeground(QColor("#ef4444"))  # away favored

                self.table.setItem(row_idx, col_idx, item)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
