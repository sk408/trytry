from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from src.database.db import DB_PATH
from src.database import migrations


class AdminView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status = QLabel("Admin tools: reset database")
        self.status.setAlignment(Qt.AlignmentFlag.AlignLeft)

        reset_btn = QPushButton("Delete database file and re-init")
        reset_btn.clicked.connect(self._reset_db)  # type: ignore[arg-type]

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Database Maintenance"))
        layout.addWidget(reset_btn)
        layout.addWidget(self.status)
        layout.addStretch()
        self.setLayout(layout)

    def _reset_db(self) -> None:
        try:
            if DB_PATH.exists():
                DB_PATH.unlink()
            else:
                # Also handle potential alternative paths if moved
                maybe = Path("data") / "nba_analytics.db"
                if maybe.exists():
                    maybe.unlink()
            migrations.init_db()
            self.status.setText("Database reset and reinitialized. Run Sync Data next.")
        except Exception as exc:  # pragma: no cover - UI path
            self.status.setText(f"Reset failed: {exc}")
