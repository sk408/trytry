import logging
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QCursor

from src.analytics.snapshots import create_snapshot, list_snapshots, restore_snapshot, delete_snapshot

logger = logging.getLogger(__name__)

class SnapshotsView(QWidget):
    """Tab to manage database and tuning snapshots/backups."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Header Area
        header_layout = QHBoxLayout()
        header = QLabel("Snapshots & Backups")
        header.setProperty("class", "header")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        self.btn_refresh = QPushButton("↻ Refresh")
        self.btn_refresh.clicked.connect(self.load_data)
        header_layout.addWidget(self.btn_refresh)
        
        self.btn_manual = QPushButton("➕ Create Manual Backup")
        self.btn_manual.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_manual.clicked.connect(self._on_create_manual)
        header_layout.addWidget(self.btn_manual)
        
        layout.addLayout(header_layout)

        # Info Label
        info = QLabel(
            "Snapshots backup your current tuning weights and local database. "
            "The Automatic pipeline creates an 'auto' snapshot before running. "
            "Restore a previous snapshot if a new update performs worse."
        )
        info.setStyleSheet("color: #94a3b8; font-size: 13px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Date / Time", "Type", "Accuracy %", "Skill Score", "ATS Record", "Size (MB)"
        ])
        
        # Table styling
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                gridline-color: #334155;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #94a3b8;
                padding: 6px;
                border: none;
                border-right: 1px solid #334155;
                border-bottom: 1px solid #334155;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QTableWidget::item:selected {
                background-color: #3b82f6;
                color: white;
            }
        """)
        layout.addWidget(self.table)

        # Action Buttons
        actions_layout = QHBoxLayout()
        
        self.btn_restore = QPushButton("↺ Restore Selected")
        self.btn_restore.setStyleSheet("background-color: #d97706; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px;")
        self.btn_restore.clicked.connect(self._on_restore)
        self.btn_restore.setEnabled(False)
        actions_layout.addWidget(self.btn_restore)
        
        self.btn_delete = QPushButton("🗑 Delete Selected")
        self.btn_delete.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px;")
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_delete.setEnabled(False)
        actions_layout.addWidget(self.btn_delete)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        # Load data initially
        self.snapshots = []
        self.load_data()

    def load_data(self):
        self.table.setRowCount(0)
        self.snapshots = list_snapshots()
        self.table.setRowCount(len(self.snapshots))
        
        for row, snap in enumerate(self.snapshots):
            # Format datetime nicely
            dt_str = snap.get("timestamp", "")
            if "T" in dt_str:
                dt_str = dt_str.replace("T", " ")[:19] # YYYY-MM-DD HH:MM:SS
                
            type_str = str(snap.get("type", "unknown")).upper()
            
            metrics = snap.get("metrics", {})
            acc = metrics.get("accuracy")
            acc_str = f"{acc:.1f}%" if acc is not None else "-"
            
            skill = metrics.get("skill_score")
            skill_str = f"{skill:.1f}" if skill is not None else "-"
            
            ats = metrics.get("ats_record")
            ats_str = str(ats) if ats is not None else "-"
            
            size_mb = snap.get("size_mb", 0)
            size_str = f"{size_mb:.1f} MB"

            items = [
                QTableWidgetItem(dt_str),
                QTableWidgetItem(type_str),
                QTableWidgetItem(acc_str),
                QTableWidgetItem(skill_str),
                QTableWidgetItem(ats_str),
                QTableWidgetItem(size_str),
            ]
            
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                # Type coloring
                if col == 1:
                    if type_str == "MANUAL":
                        item.setForeground(QColor("#3b82f6")) # Blue
                    elif type_str == "AUTO":
                        item.setForeground(QColor("#22c55e")) # Green
                self.table.setItem(row, col, item)

    def _on_selection_changed(self):
        has_selection = len(self.table.selectedItems()) > 0
        self.btn_restore.setEnabled(has_selection)
        self.btn_delete.setEnabled(has_selection)

    def _get_selected_filename(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        row = selected_rows[0].row()
        if 0 <= row < len(self.snapshots):
            return self.snapshots[row].get("filename")
        return None

    def _on_create_manual(self):
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            create_snapshot("manual")
            self.load_data()
        finally:
            QApplication.restoreOverrideCursor()

    def _on_restore(self):
        filename = self._get_selected_filename()
        if not filename:
            return
            
        reply = QMessageBox.question(
            self, 'Confirm Restore',
            f"Are you sure you want to restore '{filename}'?\n\n"
            f"This will OVERWRITE your current database and tuning weights. "
            f"This action cannot be undone unless you have another backup.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            try:
                success = restore_snapshot(filename)
                if success:
                    QMessageBox.information(self, "Restore Complete", "Snapshot restored successfully.")
                else:
                    QMessageBox.warning(self, "Restore Failed", "Failed to restore snapshot. See logs for details.")
            finally:
                QApplication.restoreOverrideCursor()

    def _on_delete(self):
        filename = self._get_selected_filename()
        if not filename:
            return
            
        reply = QMessageBox.question(
            self, 'Confirm Delete',
            f"Are you sure you want to permanently delete '{filename}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            delete_snapshot(filename)
            self.load_data()
