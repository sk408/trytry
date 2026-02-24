"""Admin tab — DB path/size, delete + reinitialize, worker threads."""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QMessageBox, QTextEdit, QSpinBox, QComboBox,
    QCheckBox,
)
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)


class AdminView(QWidget):
    """Database administration panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        layout = QVBoxLayout(self)

        header = QLabel("Administration")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # DB info card
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_frame.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; "
            "border-radius: 8px; padding: 16px; }"
        )
        info_layout = QVBoxLayout(info_frame)

        self.db_path_label = QLabel("Database: —")
        self.db_path_label.setStyleSheet("color: #e2e8f0; font-size: 13px;")
        info_layout.addWidget(self.db_path_label)

        self.db_size_label = QLabel("Size: —")
        self.db_size_label.setStyleSheet("color: #94a3b8; font-size: 13px;")
        info_layout.addWidget(self.db_size_label)

        self.table_counts_label = QLabel("")
        self.table_counts_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.table_counts_label.setWordWrap(True)
        info_layout.addWidget(self.table_counts_label)

        layout.addWidget(info_frame)

        # Weights info
        weights_frame = QFrame()
        weights_frame.setFrameShape(QFrame.Shape.StyledPanel)
        weights_frame.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; "
            "border-radius: 8px; padding: 16px; }"
        )
        w_layout = QVBoxLayout(weights_frame)
        
        self.weights_toggle_btn = QPushButton("Current Weights (Click to expand)")
        self.weights_toggle_btn.setStyleSheet(
            "text-align: left; background: transparent; color: #00e5ff; font-weight: bold; border: none; padding: 0px;"
        )
        self.weights_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.weights_toggle_btn.clicked.connect(self._toggle_weights)
        w_layout.addWidget(self.weights_toggle_btn)
        
        self.weights_text = QTextEdit()
        self.weights_text.setReadOnly(True)
        self.weights_text.setMaximumHeight(200)
        self.weights_text.setVisible(False)
        w_layout.addWidget(self.weights_text)
        layout.addWidget(weights_frame)

        # Performance settings
        perf_frame = QFrame()
        from PySide6.QtWidgets import QSizePolicy
        perf_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        perf_frame.setMinimumHeight(160)
        perf_frame.setFrameShape(QFrame.Shape.StyledPanel)
        perf_frame.setStyleSheet(
            "QFrame { background: #1e293b; border: 1px solid #334155; "
            "border-radius: 8px; padding: 16px; }"
        )
        perf_layout = QVBoxLayout(perf_frame)
        perf_label = QLabel("⚙️ Performance")
        perf_label.setStyleSheet("color: #e2e8f0; font-size: 14px; font-weight: 700;")
        perf_layout.addWidget(perf_label)

        thread_row = QHBoxLayout()
        thread_label = QLabel("Worker Threads:")
        thread_label.setStyleSheet("color: #94a3b8; font-size: 13px;")
        thread_row.addWidget(thread_label)

        self.thread_spin = QSpinBox()
        self.thread_spin.setMinimum(1)
        self.thread_spin.setMaximum(32)
        self.thread_spin.setFixedWidth(70)
        self.thread_spin.setStyleSheet(
            "QSpinBox { background: #0f172a; color: #e2e8f0; border: 1px solid #475569; "
            "border-radius: 4px; padding: 4px; font-size: 13px; }"
        )
        thread_row.addWidget(self.thread_spin)

        thread_desc = QLabel("Threads for backtest, optimization, and parallel workloads")
        thread_desc.setStyleSheet("color: #64748b; font-size: 11px;")
        thread_row.addWidget(thread_desc)
        thread_row.addStretch()
        perf_layout.addLayout(thread_row)

        log_row = QHBoxLayout()
        log_label = QLabel("Log Level:")
        log_label.setStyleSheet("color: #94a3b8; font-size: 13px;")
        log_row.addWidget(log_label)

        self.log_combo = QComboBox()
        self.log_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_combo.setFixedWidth(100)
        self.log_combo.setStyleSheet(
            "QComboBox { background: #0f172a; color: #e2e8f0; border: 1px solid #475569; "
            "border-radius: 4px; padding: 4px; font-size: 13px; }"
        )
        log_row.addWidget(self.log_combo)
        log_row.addStretch()
        perf_layout.addLayout(log_row)

        self.oled_checkbox = QCheckBox("OLED Dark Mode (Pure Black Backgrounds)")
        self.oled_checkbox.setStyleSheet(
            "QCheckBox { color: #e2e8f0; font-size: 13px; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #475569; border-radius: 3px; background: #0f172a; }"
            "QCheckBox::indicator:checked { background: #2563eb; border-color: #2563eb; }"
        )
        perf_layout.addWidget(self.oled_checkbox)

        save_perf_btn = QPushButton("Save Settings")
        save_perf_btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: white; border-radius: 6px; "
            "padding: 6px 16px; font-weight: 600; } "
            "QPushButton:hover { background: #1d4ed8; }"
        )
        save_perf_btn.clicked.connect(self._on_save_perf)
        perf_layout.addWidget(save_perf_btn)

        layout.addWidget(perf_frame)

        # Danger zone
        danger_frame = QFrame()
        danger_frame.setFrameShape(QFrame.Shape.StyledPanel)
        danger_frame.setStyleSheet(
            "QFrame { background: #1e293b; border: 2px solid #ef4444; "
            "border-radius: 8px; padding: 16px; }"
        )
        d_layout = QVBoxLayout(danger_frame)
        d_label = QLabel("⚠️ Danger Zone")
        d_label.setStyleSheet("color: #ef4444; font-size: 14px; font-weight: 700;")
        d_layout.addWidget(d_label)

        btn_layout = QHBoxLayout()

        reset_btn = QPushButton("Delete & Reinitialize Database")
        reset_btn.setProperty("class", "danger")
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)

        clear_weights_btn = QPushButton("Clear All Weights")
        clear_weights_btn.setProperty("class", "danger")
        clear_weights_btn.clicked.connect(self._on_clear_weights)
        btn_layout.addWidget(clear_weights_btn)

        btn_layout.addStretch()
        d_layout.addLayout(btn_layout)
        layout.addWidget(danger_frame)

        layout.addStretch()

        # Refresh button
        refresh_btn = QPushButton("Refresh Info")
        refresh_btn.clicked.connect(self._refresh)
        layout.addWidget(refresh_btn)

        self._refresh()

    def _toggle_weights(self):
        is_visible = self.weights_text.isVisible()
        self.weights_text.setVisible(not is_visible)
        if not is_visible:
            self.weights_toggle_btn.setText("Current Weights (Click to collapse)")
        else:
            self.weights_toggle_btn.setText("Current Weights (Click to expand)")

    def _refresh(self):
        """Refresh DB info."""
        try:
            from src.config import get as get_setting
            from src.database.db import get_db_size
            from src.database.migrations import get_table_counts

            db_path = get_setting("db_path")
            self.db_path_label.setText(f"Database: {db_path}")

            size_str = get_db_size()
            self.db_size_label.setText(f"Size: {size_str}")

            counts = get_table_counts()
            parts = [f"{table}: {count}" for table, count in sorted(counts.items())]
            self.table_counts_label.setText("Tables: " + " | ".join(parts))

        except Exception as e:
            self.db_path_label.setText(f"Error: {e}")

        # Load weights
        try:
            from src.analytics.weight_config import get_weight_config
            wc = get_weight_config()
            d = wc.to_dict()
            lines = [f"{k}: {v}" for k, v in sorted(d.items())]
            self.weights_text.setPlainText("\n".join(lines))
        except Exception:
            self.weights_text.setPlainText("(No weights loaded)")

        # Load performance settings
        try:
            from src.config import get as get_setting
            threads = get_setting("worker_threads", 4)
            self.thread_spin.setValue(int(threads))
            log_level = get_setting("log_level", "INFO")
            idx = self.log_combo.findText(str(log_level).upper())
            if idx >= 0:
                self.log_combo.setCurrentIndex(idx)
            oled_mode = get_setting("oled_mode", False)
            self.oled_checkbox.setChecked(bool(oled_mode))
        except Exception:
            pass

    def _on_save_perf(self):
        """Save performance settings."""
        try:
            from src.config import set_value
            threads = self.thread_spin.value()
            set_value("worker_threads", threads)
            log_level = self.log_combo.currentText()
            set_value("log_level", log_level)
            
            oled_mode = self.oled_checkbox.isChecked()
            set_value("oled_mode", oled_mode)

            # Apply log level immediately
            import logging as _logging
            level = getattr(_logging, log_level, _logging.INFO)
            _logging.getLogger().setLevel(level)
            
            # Apply theme immediately
            from src.ui.theme import setup_theme
            if self.main_window:
                setup_theme(self.main_window)
                self.main_window.set_status(f"Settings saved: {threads} threads, log level {log_level}, OLED {'On' if oled_mode else 'Off'}")
                
                # Iterate through all widgets and force style update
                for widget in self.main_window.findChildren(QWidget):
                    widget.style().unpolish(widget)
                    widget.style().polish(widget)
                    widget.update()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def _on_reset(self):
        """Delete and reinitialize the database."""
        reply = QMessageBox.warning(
            self,
            "Confirm Reset",
            "This will DELETE all data and reinitialize the database.\n\n"
            "This action cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            from src.database.migrations import reset_db
            reset_db()
            self._refresh()
            if self.main_window:
                self.main_window.set_status("Database reset complete")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Reset failed: {e}")

    def _on_clear_weights(self):
        """Clear all weights."""
        reply = QMessageBox.warning(
            self,
            "Clear Weights",
            "Clear global weights and per-team refinements?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            from src.analytics.weight_config import WeightConfig, save_weight_config
            wc = WeightConfig()  # Defaults
            save_weight_config(wc)
            self._refresh()
            if self.main_window:
                self.main_window.set_status("Weights cleared")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Clear failed: {e}")
