from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.database.db import DB_PATH
from src.database import migrations


class AdminView(QWidget):
    def __init__(self) -> None:
        super().__init__()

        title = QLabel("Admin Tools")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")

        # ── Database info card ──
        self.db_path_lbl = QLabel(f"Path: {DB_PATH}")
        self.db_path_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.db_size_lbl = QLabel("")
        self.db_size_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self._update_db_info()

        db_box = QGroupBox("Database")
        db_layout = QVBoxLayout()
        db_layout.addWidget(self.db_path_lbl)
        db_layout.addWidget(self.db_size_lbl)

        reset_btn = QPushButton("  Delete Database & Re-initialize")
        reset_btn.setProperty("cssClass", "danger")
        reset_btn.setToolTip("Permanently deletes all data. You will need to re-sync everything.")
        reset_btn.clicked.connect(self._reset_db)  # type: ignore[arg-type]

        self.status = QLabel("")
        self.status.setStyleSheet("color: #94a3b8; padding: 4px 0;")

        btn_row = QHBoxLayout()
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()

        db_layout.addSpacing(8)
        db_layout.addLayout(btn_row)
        db_layout.addWidget(self.status)
        db_box.setLayout(db_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)
        layout.addWidget(title)
        layout.addSpacing(8)
        layout.addWidget(db_box)
        layout.addStretch()
        self.setLayout(layout)

    def _update_db_info(self) -> None:
        if DB_PATH.exists():
            size_mb = DB_PATH.stat().st_size / (1024 * 1024)
            self.db_size_lbl.setText(f"Size: {size_mb:.1f} MB")
        else:
            self.db_size_lbl.setText("Size: database not found")

    def _reset_db(self) -> None:
        reply = QMessageBox.warning(
            self,
            "Confirm Database Reset",
            "This will permanently delete all synced data.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            if DB_PATH.exists():
                DB_PATH.unlink()
            else:
                maybe = Path("data") / "nba_analytics.db"
                if maybe.exists():
                    maybe.unlink()
            migrations.init_db()
            self.status.setText("Database reset and reinitialized. Run Sync Data next.")
            self.status.setStyleSheet("color: #10b981; padding: 4px 0;")
            self._update_db_info()
        except Exception as exc:
            self.status.setText(f"Reset failed: {exc}")
            self.status.setStyleSheet("color: #ef4444; padding: 4px 0;")
