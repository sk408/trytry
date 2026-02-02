from __future__ import annotations

from PySide6.QtCore import Qt
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
from src.data.sync_service import sync_live_scores


class LiveView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.table = QTableWidget()
        self.refresh_button = QPushButton("Refresh live scores")
        self.refresh_button.clicked.connect(self.refresh)  # type: ignore[arg-type]

        header = QHBoxLayout()
        header.addWidget(QLabel("Live Games & Recommendations"))
        header.addStretch()
        header.addWidget(self.refresh_button)

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.refresh()

    def refresh(self) -> None:
        # Fetch latest live scores then rebuild recs
        try:
            sync_live_scores()
        except Exception as exc:  # pragma: no cover - network dependent
            self._render([], error=str(exc))
            return
        recs = build_live_recommendations()
        self._render(recs)

    def _render(self, recs, error: str | None = None) -> None:
        self.table.clear()
        headers = ["Status", "Home", "Away", "Score", "Period", "Clock", "Proj Spread (home)", "Proj Total"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(recs) if not error else 1)

        if error:
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {error}"))
            self.table.resizeColumnsToContents()
            return

        for row_idx, rec in enumerate(recs):
            score = f"{rec.home_score}-{rec.away_score}"
            self.table.setItem(row_idx, 0, QTableWidgetItem(rec.status))
            self.table.setItem(row_idx, 1, QTableWidgetItem(rec.home_team))
            self.table.setItem(row_idx, 2, QTableWidgetItem(rec.away_team))
            self.table.setItem(row_idx, 3, QTableWidgetItem(score))
            self.table.setItem(row_idx, 4, QTableWidgetItem(str(rec.period)))
            self.table.setItem(row_idx, 5, QTableWidgetItem(rec.clock))
            self.table.setItem(row_idx, 6, QTableWidgetItem(f"{rec.projected_spread:+.1f}"))
            self.table.setItem(row_idx, 7, QTableWidgetItem(f"{rec.projected_total:.1f}"))

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
