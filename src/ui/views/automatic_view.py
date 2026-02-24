"""Automatic tab — one-click 12-step pipeline execution and logs."""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QCheckBox, QScrollArea, QProgressBar,
    QGridLayout
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QTextCursor

from src.ui.workers import start_pipeline_worker
from src.analytics.pipeline import request_cancel

logger = logging.getLogger(__name__)


class StepIndicator(QFrame):
    """A visual indicator for a single pipeline step."""
    
    def __init__(self, step_num: int, title: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            StepIndicator {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.header_layout = QHBoxLayout()
        
        self.num_label = QLabel(f"{step_num}.")
        self.num_label.setStyleSheet("color: #64748b; font-weight: bold; font-size: 14px;")
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #94a3b8; font-weight: 500; font-size: 13px;")
        
        self.status_icon = QLabel("○")
        self.status_icon.setStyleSheet("color: #64748b; font-size: 14px;")
        
        self.header_layout.addWidget(self.num_label)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.status_icon)
        
        layout.addLayout(self.header_layout)
        
    def set_state(self, state: str):
        """state can be: 'pending', 'active', 'done', 'error', 'skipped'"""
        if state == "active":
            self.setStyleSheet("StepIndicator { background-color: #1e3a8a; border: 1px solid #3b82f6; border-radius: 6px; }")
            self.title_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 13px;")
            self.status_icon.setText("↻")
            self.status_icon.setStyleSheet("color: #60a5fa; font-size: 16px;")
        elif state == "done":
            self.setStyleSheet("StepIndicator { background-color: #064e3b; border: 1px solid #10b981; border-radius: 6px; }")
            self.title_label.setStyleSheet("color: #d1fae5; font-weight: bold; font-size: 13px;")
            self.status_icon.setText("✓")
            self.status_icon.setStyleSheet("color: #34d399; font-size: 16px; font-weight: bold;")
        elif state == "skipped":
            self.setStyleSheet("StepIndicator { background-color: #1e293b; border: 1px solid #475569; border-radius: 6px; }")
            self.title_label.setStyleSheet("color: #94a3b8; font-style: italic; font-size: 13px;")
            self.status_icon.setText("⏭")
            self.status_icon.setStyleSheet("color: #94a3b8; font-size: 16px;")
        elif state == "error":
            self.setStyleSheet("StepIndicator { background-color: #450a0a; border: 1px solid #ef4444; border-radius: 6px; }")
            self.title_label.setStyleSheet("color: #fee2e2; font-weight: bold; font-size: 13px;")
            self.status_icon.setText("✕")
            self.status_icon.setStyleSheet("color: #f87171; font-size: 16px; font-weight: bold;")
        else: # pending
            self.setStyleSheet("StepIndicator { background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; }")
            self.title_label.setStyleSheet("color: #94a3b8; font-weight: 500; font-size: 13px;")
            self.status_icon.setText("○")
            self.status_icon.setStyleSheet("color: #64748b; font-size: 14px;")


class AutomaticView(QWidget):
    """Automatic Pipeline runner tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._current_worker = None
        
        # Step matching mapping (log text to step index)
        self.step_map = {
            "Step 1/13": 0, "Step 2/13": 1, "Step 3/13": 2, "Step 4/13": 3,
            "Step 5/13": 4, "Step 6/13": 5, "Step 7/13": 6, "Step 8/13": 7,
            "Step 9/13": 8, "Step 10/13": 9, "Step 11/13": 10, "Step 12/13": 11,
            "Step 13/13": 12
        }
        self.current_step_idx = -1

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Header Area
        header_layout = QHBoxLayout()
        header = QLabel("Automatic Update Pipeline")
        header.setProperty("class", "header")
        header_layout.addWidget(header)
        
        self.force_cb = QCheckBox("Force Full Update (Bypass all caches and re-fetch everything)")
        self.force_cb.setStyleSheet("color: #cbd5e1; font-size: 13px;")
        header_layout.addStretch()
        header_layout.addWidget(self.force_cb)
        layout.addLayout(header_layout)

        # Action Buttons
        controls_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("▶ Run Full Pipeline")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #16a34a; }
            QPushButton:disabled { background-color: #064e3b; color: #94a3b8; }
        """)
        self.run_btn.clicked.connect(self._start_pipeline)
        controls_layout.addWidget(self.run_btn)
        
        self.cancel_btn = QPushButton("⏹ Stop Pipeline")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #dc2626; }
            QPushButton:disabled { background-color: #450a0a; color: #94a3b8; }
        """)
        self.cancel_btn.clicked.connect(self._stop_pipeline)
        self.cancel_btn.setEnabled(False)
        controls_layout.addWidget(self.cancel_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)

        # Visual Steps Grid
        steps_container = QFrame()
        steps_container.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 10px;")
        grid = QGridLayout(steps_container)
        
        step_titles = [
            "Backup State", "Check State", "Load Memory", "Fetch Data & Odds", "Memory Reload",
            "Injury History", "Play Outcomes", "Autotune Teams", "Train ML Models",
            "Global Weights", "Team Refinement", "Calibration", "Final Backtest"
        ]
        
        self.step_widgets = []
        for i, title in enumerate(step_titles):
            sw = StepIndicator(i + 1, title)
            self.step_widgets.append(sw)
            row = i // 4
            col = i % 4
            grid.addWidget(sw, row, col)
            
        layout.addWidget(steps_container)

        # Terminal Log Output
        log_label = QLabel("Terminal Output")
        log_label.setStyleSheet("color: #94a3b8; font-size: 12px; text-transform: uppercase;")
        layout.addWidget(log_label)

        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        # Monospace font for terminal feel
        font = QFont("Consolas", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.terminal.setFont(font)
        self.terminal.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #22c55e;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.terminal, stretch=1)
        
        self._reset_steps()

    def _reset_steps(self):
        for sw in self.step_widgets:
            sw.set_state("pending")
        self.current_step_idx = -1

    def _append_log(self, msg: str):
        # Determine color and parse state
        color = "#22c55e" # Default green
        if "error" in msg.lower() or "failed" in msg.lower():
            color = "#ef4444"
        elif "skipping" in msg.lower():
            color = "#94a3b8"
            if self.current_step_idx >= 0:
                self.step_widgets[self.current_step_idx].set_state("skipped")
        elif msg.startswith("[Step"):
            color = "#3b82f6" # Blue for step headers
            # Parse step index
            for marker, idx in self.step_map.items():
                if marker in msg:
                    # Mark previous as done if active
                    if self.current_step_idx >= 0 and self.current_step_idx != idx:
                        prev_state = self.step_widgets[self.current_step_idx].styleSheet()
                        if "#1e3a8a" in prev_state: # If it was active, mark done
                            self.step_widgets[self.current_step_idx].set_state("done")
                            
                    self.current_step_idx = idx
                    self.step_widgets[idx].set_state("active")
                    break

        # Append to terminal
        self.terminal.moveCursor(QTextCursor.MoveOperation.End)
        self.terminal.insertHtml(f'<div style="color:{color}; font-family:Consolas; white-space:pre-wrap;">{msg}</div><br>')
        self.terminal.moveCursor(QTextCursor.MoveOperation.End)

    def _on_results(self, results: dict):
        """Called when pipeline worker yields final result."""
        if results.get("error"):
            self._append_log(f"Pipeline Failed: {results['error']}")
            if self.current_step_idx >= 0:
                self.step_widgets[self.current_step_idx].set_state("error")
        else:
            # Mark final step as done
            if self.current_step_idx >= 0:
                 self.step_widgets[self.current_step_idx].set_state("done")
            self._append_log("✨ Automatic Update Pipeline Complete!")
            
        # Optional: Switch to accuracy tab so user sees the new results
        if self.main_window and hasattr(self.main_window, "accuracy"):
            # Update accuracy tab cards
            self.main_window.accuracy._on_results(results.get("backtest", {}))

    def _on_done(self):
        """Worker thread finished."""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.force_cb.setEnabled(True)
        if self.main_window:
            self.main_window.set_status("Pipeline execution finished")

    def _start_pipeline(self):
        self.terminal.clear()
        self._reset_steps()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.force_cb.setEnabled(False)
        
        self._append_log("Starting automatic pipeline initialization...")
        
        # If force is checked, we should probably clear caches. 
        # The pipeline currently doesn't natively accept a "force" arg directly to run_full_pipeline, 
        # so we will use the data sync force mechanism and clear sync_meta.
        if self.force_cb.isChecked():
            from src.data.sync_service import clear_sync_cache
            clear_sync_cache()
            self._append_log("Force mode enabled: Caches cleared.")

        # Start the worker
        self._current_worker = start_pipeline_worker(
            on_progress=self._append_log,
            on_result=self._on_results,
            on_done=self._on_done
        )

    def _stop_pipeline(self):
        self._append_log("Cancellation requested... waiting for current step to abort.")
        request_cancel()
        if self._current_worker:
            self._current_worker.stop()
        self.cancel_btn.setEnabled(False)
        if self.current_step_idx >= 0:
            self.step_widgets[self.current_step_idx].set_state("error")
