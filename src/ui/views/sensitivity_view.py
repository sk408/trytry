"""Sensitivity Analysis tab — explore parameter ranges and coordinate descent."""

import logging
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QTextEdit, QCheckBox, QDoubleSpinBox,
    QGroupBox,
)
from PySide6.QtCore import Qt

from src.ui.workers import start_sensitivity_worker, start_coordinate_descent_worker

logger = logging.getLogger(__name__)

class SensitivityView(QWidget):
    """Sensitivity Analysis View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._current_worker = None

        layout = QVBoxLayout(self)

        header = QLabel("Sensitivity Analysis")
        header.setProperty("class", "header")
        layout.addWidget(header)

        desc = QLabel(
            "Sweep individual weights through extreme ranges to discover hidden value outside conventional optimizer bounds.\n"
            "This produces CSV files and terminal-style ASCII charts in the data/sensitivity folder."
        )
        layout.addWidget(desc)

        # ── Sweep Controls ──
        sweep_group = QGroupBox("Parameter Sweep")
        sweep_layout = QVBoxLayout(sweep_group)

        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItem("All Parameters", "all")

        # Load parameters
        try:
            from src.analytics.sensitivity import EXTREME_RANGES
            for p in sorted(EXTREME_RANGES.keys()):
                self.param_combo.addItem(p, p)
        except Exception:
            pass

        self.param_combo.setMinimumWidth(200)
        ctrl.addWidget(self.param_combo)

        ctrl.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox()
        self.target_combo.addItem("Moneyline", "ml")
        self.target_combo.addItem("Value (underdog)", "value")
        self.target_combo.addItem("ATS (spread betting)", "ats")
        self.target_combo.addItem("ROI (max return)", "roi")
        ctrl.addWidget(self.target_combo)

        ctrl.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 500)
        self.steps_spin.setValue(100)
        self.steps_spin.setSingleStep(10)
        ctrl.addWidget(self.steps_spin)

        ctrl.addStretch()
        sweep_layout.addLayout(ctrl)

        # Sweep buttons
        sweep_btns = QHBoxLayout()
        run_btn = QPushButton("Run Sweep")
        run_btn.clicked.connect(self._on_run)
        sweep_btns.addWidget(run_btn)

        open_folder_btn = QPushButton("Open Output Folder")
        open_folder_btn.clicked.connect(self._on_open_folder)
        sweep_btns.addWidget(open_folder_btn)
        sweep_btns.addStretch()
        sweep_layout.addLayout(sweep_btns)

        layout.addWidget(sweep_group)

        # ── Coordinate Descent Controls ──
        cd_group = QGroupBox("Coordinate Descent")
        cd_layout = QVBoxLayout(cd_group)

        cd_desc = QLabel(
            "Iteratively optimize each parameter while holding others fixed. "
            "Captures parameter interactions that single sweeps miss."
        )
        cd_desc.setWordWrap(True)
        cd_layout.addWidget(cd_desc)

        cd_ctrl = QHBoxLayout()

        cd_ctrl.addWidget(QLabel("Target:"))
        self.cd_target_combo = QComboBox()
        self.cd_target_combo.addItem("ATS (spread betting)", "ats")
        self.cd_target_combo.addItem("ROI (max return)", "roi")
        self.cd_target_combo.addItem("Moneyline", "ml")
        cd_ctrl.addWidget(self.cd_target_combo)

        cd_ctrl.addWidget(QLabel("Steps:"))
        self.cd_steps_spin = QSpinBox()
        self.cd_steps_spin.setRange(20, 500)
        self.cd_steps_spin.setValue(100)
        self.cd_steps_spin.setSingleStep(10)
        cd_ctrl.addWidget(self.cd_steps_spin)

        cd_ctrl.addWidget(QLabel("Max Rounds:"))
        self.cd_rounds_spin = QSpinBox()
        self.cd_rounds_spin.setRange(1, 50)
        self.cd_rounds_spin.setValue(10)
        cd_ctrl.addWidget(self.cd_rounds_spin)

        cd_ctrl.addWidget(QLabel("Convergence:"))
        self.cd_conv_spin = QDoubleSpinBox()
        self.cd_conv_spin.setRange(0.0001, 1.0)
        self.cd_conv_spin.setValue(0.005)
        self.cd_conv_spin.setDecimals(4)
        self.cd_conv_spin.setSingleStep(0.001)
        cd_ctrl.addWidget(self.cd_conv_spin)

        self.cd_apply_check = QCheckBox("Apply results")
        self.cd_apply_check.setChecked(True)
        self.cd_apply_check.setToolTip("Save optimized weights and create a snapshot when done")
        cd_ctrl.addWidget(self.cd_apply_check)

        cd_ctrl.addStretch()
        cd_layout.addLayout(cd_ctrl)

        cd_btns = QHBoxLayout()
        cd_run_btn = QPushButton("Run Coordinate Descent")
        cd_run_btn.clicked.connect(self._on_run_cd)
        cd_btns.addWidget(cd_run_btn)
        cd_btns.addStretch()
        cd_layout.addLayout(cd_btns)

        layout.addWidget(cd_group)

        # ── Common controls ──
        common_btns = QHBoxLayout()
        stop_btn = QPushButton("Stop")
        stop_btn.setProperty("class", "danger")
        stop_btn.clicked.connect(self._on_stop)
        common_btns.addWidget(stop_btn)
        common_btns.addStretch()
        layout.addLayout(common_btns)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        # Use monospace font for ASCII charts
        font = self.log.font()
        font.setFamily("Consolas")
        font.setStyleHint(font.StyleHint.Monospace)
        self.log.setFont(font)
        layout.addWidget(self.log)

    def _append_log(self, msg: str):
        self.log.append(msg)
        # scroll to bottom
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_run(self):
        if self._current_worker and self._current_worker.isRunning():
            self._append_log("Analysis is already running.")
            return

        param = self.param_combo.currentData()
        steps = self.steps_spin.value()
        target = self.target_combo.currentData()

        self.log.clear()
        self._append_log(f"Starting sensitivity analysis (param: {param}, steps: {steps}, target: {target})...")

        self._current_worker = start_sensitivity_worker(param, steps, target=target,
                                                        on_progress=self._append_log, on_done=self._on_done)

    def _on_run_cd(self):
        if self._current_worker and self._current_worker.isRunning():
            self._append_log("Analysis is already running.")
            return

        target = self.cd_target_combo.currentData()
        steps = self.cd_steps_spin.value()
        max_rounds = self.cd_rounds_spin.value()
        convergence = self.cd_conv_spin.value()
        apply_results = self.cd_apply_check.isChecked()

        self.log.clear()
        self._append_log(f"Starting coordinate descent (target={target}, steps={steps}, "
                         f"max_rounds={max_rounds}, convergence={convergence}, "
                         f"apply={'yes' if apply_results else 'dry run'})...")

        self._current_worker = start_coordinate_descent_worker(
            steps=steps, max_rounds=max_rounds, convergence=convergence,
            target=target, apply_results=apply_results,
            on_progress=self._append_log, on_done=self._on_done
        )

    def _on_stop(self):
        if self._current_worker:
            self._append_log("Stopping analysis gracefully...")
            self._current_worker.stop()

    def _on_done(self):
        self._append_log("Done.")
        if self.main_window:
            self.main_window.set_status("Sensitivity Analysis complete.")

    def _on_open_folder(self):
        import subprocess
        import sys

        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "sensitivity")
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        if sys.platform == "win32":
            os.startfile(out_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", out_dir])
        else:
            subprocess.Popen(["xdg-open", out_dir])
