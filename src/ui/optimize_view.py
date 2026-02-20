"""Unified Optimize tab — one-click improvement with before/after comparison.

Includes regression guard: auto-rollback, parameter blacklisting,
variable diff logging, and suggested parameter adjustments.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.analytics.optimize_guard import (
    WeightBackup,
    OptimizeRunRecord,
    check_blacklist_before_run,
    evaluate_run,
    generate_suggestions,
    suggest_adjusted_params,
    apply_suggested_params,
    format_run_summary,
    load_history,
    get_regression_history,
    clear_blacklist,
    composite_score,
)


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """Frozen metrics from a backtest run."""
    timestamp: float = 0.0
    total_games: int = 0
    winner_pct: float = 0.0
    avg_spread_error: float = 0.0
    total_in_10_pct: float = 0.0
    avg_total_error: float = 0.0
    team_data: Dict[str, Dict] = field(default_factory=dict)  # abbr → metrics

    @staticmethod
    def from_backtest(bt) -> "Snapshot":
        s = Snapshot(timestamp=time.time())
        if bt is None:
            return s
        s.total_games = bt.total_games
        s.winner_pct = bt.overall_spread_accuracy
        s.total_in_10_pct = bt.overall_total_accuracy
        if bt.predictions:
            s.avg_spread_error = sum(abs(p.spread_error) for p in bt.predictions) / len(bt.predictions)
            s.avg_total_error = sum(abs(p.total_error) for p in bt.predictions) / len(bt.predictions)
        for tid, ta in bt.team_accuracy.items():
            s.team_data[ta.team_abbr] = {
                "games": ta.games_analyzed,
                "winner_pct": ta.spread_accuracy,
                "avg_spread_err": ta.avg_spread_error,
                "total_in_10": ta.total_accuracy,
                "avg_total_err": ta.avg_total_error,
            }
        return s


# ---------------------------------------------------------------------------
#  Workers
# ---------------------------------------------------------------------------

class _SnapshotWorker(QObject):
    """Run a backtest to capture current accuracy (pre-optimization)."""
    progress = Signal(str)
    finished = Signal(object)  # Snapshot
    error = Signal(str)

    def run(self) -> None:
        try:
            from src.analytics.backtester import run_backtest, load_backtest_cache
            self.progress.emit("Capturing baseline accuracy (backtest)...")
            # Try cache first (valid for 480 min = 8 hrs — stale-ish is fine for "before")
            bt = load_backtest_cache(None, None, True, max_age_minutes=480)
            if bt is None:
                bt = run_backtest(
                    min_games_before=5,
                    progress_cb=self.progress.emit,
                    max_workers=4,
                )
            snap = Snapshot.from_backtest(bt)
            self.finished.emit(snap)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class _PipelineWorker(QObject):
    """Run the full optimisation pipeline."""
    progress = Signal(str)
    step_update = Signal(int, str, str)  # step_index, status, detail
    finished = Signal(object)  # pipeline summary dict
    error = Signal(str)

    def __init__(self, n_trials: int = 200, team_trials: int = 100,
                 max_workers: int = 4, force_rerun: bool = False):
        super().__init__()
        self.n_trials = n_trials
        self.team_trials = team_trials
        self.max_workers = max_workers
        self.force_rerun = force_rerun
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self) -> None:
        try:
            from src.analytics.pipeline import run_full_pipeline
            summary = run_full_pipeline(
                n_trials=self.n_trials,
                team_trials=self.team_trials,
                max_workers=self.max_workers,
                progress_cb=self._on_progress,
                cancel_check=lambda: self._cancel,
                force_rerun=self.force_rerun,
            )
            self.finished.emit(summary)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    def _on_progress(self, msg: str):
        self.progress.emit(msg)
        # Parse step headers to update step indicators
        if msg.startswith("[STEP "):
            try:
                parts = msg.split("]", 1)
                nums = parts[0].replace("[STEP ", "").strip()
                step_num = int(nums.split("/")[0])
                step_name = parts[1].strip() if len(parts) > 1 else ""
                if "SKIPPED" in step_name:
                    self.step_update.emit(step_num - 1, "skipped", step_name)
                else:
                    self.step_update.emit(step_num - 1, "running", step_name)
            except (ValueError, IndexError):
                pass


# ---------------------------------------------------------------------------
#  Metric card helpers
# ---------------------------------------------------------------------------

def _make_metric_card(title: str, object_name: str = "") -> tuple[QFrame, QLabel, QLabel]:
    """Create a styled card with a title, a big value label, and a delta label."""
    card = QFrame()
    card.setFrameShape(QFrame.StyledPanel)
    card.setStyleSheet(
        "QFrame { background: #23272e; border-radius: 8px; padding: 8px; }"
    )
    card.setMinimumWidth(160)
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    vbox = QVBoxLayout(card)
    vbox.setContentsMargins(12, 10, 12, 10)
    vbox.setSpacing(2)

    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
    title_lbl.setAlignment(Qt.AlignCenter)

    value_lbl = QLabel("--")
    value_lbl.setAlignment(Qt.AlignCenter)
    value_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: #e6edf3;")

    delta_lbl = QLabel("")
    delta_lbl.setAlignment(Qt.AlignCenter)
    delta_lbl.setStyleSheet("font-size: 12px;")

    vbox.addWidget(title_lbl)
    vbox.addWidget(value_lbl)
    vbox.addWidget(delta_lbl)
    return card, value_lbl, delta_lbl


def _format_delta(before: float, after: float, higher_is_better: bool, suffix: str = "") -> str:
    """Return a colored delta string like '▲ +2.3%' or '▼ -1.1 pts'."""
    diff = after - before
    if abs(diff) < 0.01:
        return '<span style="color:#8b949e;">— no change</span>'
    improved = (diff > 0) == higher_is_better
    arrow = "▲" if diff > 0 else "▼"
    color = "#3fb950" if improved else "#f85149"
    sign = "+" if diff > 0 else ""
    return f'<span style="color:{color};">{arrow} {sign}{diff:.2f}{suffix}</span>'


# ---------------------------------------------------------------------------
#  Step tracker widget
# ---------------------------------------------------------------------------

PIPELINE_STEPS = [
    "Check State",
    "Load Memory",
    "Data Sync",
    "Reload Memory",
    "Injury History",
    "Injury Intel",
    "Roster Check",
    "Autotune",
    "ML Training",
    "Weight Opt",
    "Per-Team Refine",
    "Calibration",
    "Backtest",
]


class _StepTracker(QWidget):
    """A horizontal or vertical list of step indicators."""
    def __init__(self, steps: List[str], parent=None):
        super().__init__(parent)
        self._labels: List[QLabel] = []
        self._icons: List[QLabel] = []
        layout = QGridLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        for i, name in enumerate(steps):
            icon = QLabel("○")
            icon.setFixedWidth(18)
            icon.setAlignment(Qt.AlignCenter)
            icon.setStyleSheet("font-size: 14px; color: #484f58;")
            lbl = QLabel(name)
            lbl.setStyleSheet("font-size: 11px; color: #8b949e;")
            layout.addWidget(icon, i, 0)
            layout.addWidget(lbl, i, 1)
            self._icons.append(icon)
            self._labels.append(lbl)

    def set_step_status(self, index: int, status: str):
        if 0 <= index < len(self._icons):
            if status == "running":
                self._icons[index].setText("◉")
                self._icons[index].setStyleSheet("font-size: 14px; color: #58a6ff;")
                self._labels[index].setStyleSheet("font-size: 11px; color: #e6edf3; font-weight: bold;")
            elif status == "done":
                self._icons[index].setText("✓")
                self._icons[index].setStyleSheet("font-size: 14px; color: #3fb950;")
                self._labels[index].setStyleSheet("font-size: 11px; color: #8b949e;")
            elif status == "skipped":
                self._icons[index].setText("–")
                self._icons[index].setStyleSheet("font-size: 14px; color: #484f58;")
                self._labels[index].setStyleSheet("font-size: 11px; color: #484f58;")
            elif status == "error":
                self._icons[index].setText("✗")
                self._icons[index].setStyleSheet("font-size: 14px; color: #f85149;")
                self._labels[index].setStyleSheet("font-size: 11px; color: #f85149;")

    def mark_all_done(self, summary_steps: dict):
        """Update all steps from a pipeline summary dict."""
        step_keys = [
            "check_state", "load_memory", "sync", "reload_memory",
            "injury_history", "injury_intel", "roster_change",
            "autotune", "ml_train", "optimize", "team_refine",
            "calibrate", "backtest",
        ]
        for i, key in enumerate(step_keys):
            info = summary_steps.get(key, {})
            st = info.get("status", "")
            if st == "skipped":
                self.set_step_status(i, "skipped")
            elif st == "error":
                self.set_step_status(i, "error")
            elif st == "done":
                self.set_step_status(i, "done")

    def reset(self):
        for i in range(len(self._icons)):
            self._icons[i].setText("○")
            self._icons[i].setStyleSheet("font-size: 14px; color: #484f58;")
            self._labels[i].setStyleSheet("font-size: 11px; color: #8b949e;")


# ---------------------------------------------------------------------------
#  Main widget
# ---------------------------------------------------------------------------

class OptimizeView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._thread: Optional[QThread] = None
        self._worker: Optional[QObject] = None
        self._before: Optional[Snapshot] = None
        self._after: Optional[Snapshot] = None
        self._backup: Optional[WeightBackup] = None
        self._last_record: Optional[OptimizeRunRecord] = None
        self._suggested_params: Dict[str, float] = {}
        self._running = False
        self._current_step = -1

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 12, 16, 12)

        # ── Title row ──
        title = QLabel("Optimize Predictions")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #e6edf3;")
        subtitle = QLabel(
            "One click to sync data, retrain ML models, tune weights, "
            "and calibrate — then see before vs. after accuracy."
        )
        subtitle.setStyleSheet("color: #8b949e; font-size: 12px;")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        # ── Controls row ──
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        self.optimize_btn = QPushButton("  ⚡  Optimize Everything  ")
        self.optimize_btn.setStyleSheet(
            "QPushButton { background: #238636; color: white; font-size: 15px; "
            "font-weight: bold; padding: 10px 28px; border-radius: 6px; }"
            "QPushButton:hover { background: #2ea043; }"
            "QPushButton:disabled { background: #21262d; color: #484f58; }"
        )
        self.optimize_btn.clicked.connect(self._on_optimize_clicked)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background: #da3633; color: white; padding: 10px 16px; "
            "border-radius: 6px; font-weight: bold; }"
            "QPushButton:disabled { background: #21262d; color: #484f58; }"
        )
        self.cancel_btn.clicked.connect(self._on_cancel)

        self.force_checkbox = QCheckBox("Force re-run (ignore caches)")
        self.force_checkbox.setStyleSheet("color: #8b949e;")

        trials_lbl = QLabel("Trials:")
        trials_lbl.setStyleSheet("color: #8b949e;")
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(20, 2000)
        self.trials_spin.setValue(200)
        self.trials_spin.setSingleStep(50)
        self.trials_spin.setFixedWidth(80)

        ctrl.addWidget(self.optimize_btn)
        ctrl.addWidget(self.cancel_btn)
        ctrl.addSpacing(20)
        ctrl.addWidget(self.force_checkbox)
        ctrl.addSpacing(10)
        ctrl.addWidget(trials_lbl)
        ctrl.addWidget(self.trials_spin)
        ctrl.addStretch()

        root.addLayout(ctrl)

        # ── Progress bar ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: none; background: #21262d; }"
            "QProgressBar::chunk { background: #58a6ff; }"
        )
        root.addWidget(self.progress_bar)

        # ── Central content: cards + step tracker side by side ──
        content = QHBoxLayout()
        content.setSpacing(16)

        # Left side: before/after metric cards
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(12)

        # BEFORE section
        before_header = QLabel("BEFORE")
        before_header.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        cards_layout.addWidget(before_header)

        before_row = QHBoxLayout()
        before_row.setSpacing(8)
        c1, self.b_games, _ = _make_metric_card("Games")
        c2, self.b_winner, _ = _make_metric_card("Winner %")
        c3, self.b_spread, _ = _make_metric_card("Avg Spread Err")
        c4, self.b_total10, _ = _make_metric_card("Total in 10 %")
        c5, self.b_totalerr, _ = _make_metric_card("Avg Total Err")
        for card in [c1, c2, c3, c4, c5]:
            before_row.addWidget(card)
        cards_layout.addLayout(before_row)

        # AFTER section
        after_header = QLabel("AFTER")
        after_header.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        cards_layout.addWidget(after_header)

        after_row = QHBoxLayout()
        after_row.setSpacing(8)
        c1, self.a_games, self.d_games = _make_metric_card("Games")
        c2, self.a_winner, self.d_winner = _make_metric_card("Winner %")
        c3, self.a_spread, self.d_spread = _make_metric_card("Avg Spread Err")
        c4, self.a_total10, self.d_total10 = _make_metric_card("Total in 10 %")
        c5, self.a_totalerr, self.d_totalerr = _make_metric_card("Avg Total Err")
        for card in [c1, c2, c3, c4, c5]:
            after_row.addWidget(card)
        cards_layout.addLayout(after_row)

        cards_layout.addStretch()
        content.addWidget(cards_widget, stretch=3)

        # Right side: step tracker
        step_box = QGroupBox("Pipeline Steps")
        step_box.setStyleSheet(
            "QGroupBox { font-size: 12px; font-weight: bold; color: #8b949e; "
            "border: 1px solid #30363d; border-radius: 6px; padding-top: 18px; }"
            "QGroupBox::title { subcontrol-position: top left; padding: 4px 8px; }"
        )
        step_layout = QVBoxLayout(step_box)
        self.step_tracker = _StepTracker(PIPELINE_STEPS)
        step_layout.addWidget(self.step_tracker)
        step_layout.addStretch()
        step_box.setFixedWidth(200)
        content.addWidget(step_box, stretch=0)

        root.addLayout(content)

        # ── Regression guard banner (hidden until regression detected) ──
        self.regression_banner = QFrame()
        self.regression_banner.setStyleSheet(
            "QFrame { background: #3d1418; border: 2px solid #f85149; "
            "border-radius: 8px; padding: 12px; }"
        )
        self.regression_banner.setVisible(False)
        banner_layout = QVBoxLayout(self.regression_banner)
        banner_layout.setSpacing(6)

        self.regression_title = QLabel("⚠️  REGRESSION DETECTED — Changes Rolled Back")
        self.regression_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #f85149;"
        )
        banner_layout.addWidget(self.regression_title)

        self.regression_detail = QLabel("")
        self.regression_detail.setWordWrap(True)
        self.regression_detail.setStyleSheet("color: #f0a0a0; font-size: 12px;")
        banner_layout.addWidget(self.regression_detail)

        self.regression_suggestions = QTextEdit()
        self.regression_suggestions.setReadOnly(True)
        self.regression_suggestions.setMaximumHeight(120)
        self.regression_suggestions.setStyleSheet(
            "QTextEdit { background: #1a1a24; color: #e6c07b; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "border: 1px solid #f8514950; border-radius: 4px; }"
        )
        self.regression_suggestions.setVisible(False)
        banner_layout.addWidget(self.regression_suggestions)

        # Buttons row inside banner
        banner_btns = QHBoxLayout()
        self.apply_suggested_btn = QPushButton("💡 Apply Suggested Parameters")
        self.apply_suggested_btn.setStyleSheet(
            "QPushButton { background: #1f6feb; color: white; padding: 6px 14px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #388bfd; }"
        )
        self.apply_suggested_btn.setVisible(False)
        self.apply_suggested_btn.clicked.connect(self._on_apply_suggested)
        banner_btns.addWidget(self.apply_suggested_btn)

        self.view_diffs_btn = QPushButton("📋 View Weight Changes")
        self.view_diffs_btn.setStyleSheet(
            "QPushButton { background: #30363d; color: #c9d1d9; padding: 6px 14px; "
            "border-radius: 4px; }"
            "QPushButton:hover { background: #484f58; }"
        )
        self.view_diffs_btn.clicked.connect(self._on_view_diffs)
        self.view_diffs_btn.setVisible(False)
        banner_btns.addWidget(self.view_diffs_btn)

        banner_btns.addStretch()
        banner_layout.addLayout(banner_btns)
        root.addWidget(self.regression_banner)

        # ── Success banner (hidden until improvement confirmed) ──
        self.success_banner = QFrame()
        self.success_banner.setStyleSheet(
            "QFrame { background: #0d2818; border: 2px solid #3fb950; "
            "border-radius: 8px; padding: 12px; }"
        )
        self.success_banner.setVisible(False)
        success_layout = QVBoxLayout(self.success_banner)
        self.success_title = QLabel("✓ Optimization Improved Results")
        self.success_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #3fb950;"
        )
        success_layout.addWidget(self.success_title)
        self.success_detail = QLabel("")
        self.success_detail.setWordWrap(True)
        self.success_detail.setStyleSheet("color: #7ee787; font-size: 12px;")
        success_layout.addWidget(self.success_detail)
        root.addWidget(self.success_banner)

        # ── History button row ──
        hist_row = QHBoxLayout()
        self.history_btn = QPushButton("📊 Run History")
        self.history_btn.setStyleSheet(
            "QPushButton { background: #30363d; color: #c9d1d9; padding: 6px 14px; "
            "border-radius: 4px; }"
            "QPushButton:hover { background: #484f58; }"
        )
        self.history_btn.clicked.connect(self._on_show_history)

        self.clear_blacklist_btn = QPushButton("🗑 Clear Blacklist")
        self.clear_blacklist_btn.setStyleSheet(
            "QPushButton { background: #30363d; color: #c9d1d9; padding: 6px 14px; "
            "border-radius: 4px; }"
            "QPushButton:hover { background: #484f58; }"
        )
        self.clear_blacklist_btn.clicked.connect(self._on_clear_blacklist)

        hist_row.addWidget(self.history_btn)
        hist_row.addWidget(self.clear_blacklist_btn)
        hist_row.addStretch()
        root.addLayout(hist_row)

        # ── Per-team comparison table ──
        team_box = QGroupBox("Per-Team Before / After")
        team_box.setStyleSheet(
            "QGroupBox { font-size: 12px; font-weight: bold; color: #8b949e; "
            "border: 1px solid #30363d; border-radius: 6px; padding-top: 18px; }"
            "QGroupBox::title { subcontrol-position: top left; padding: 4px 8px; }"
        )
        tb_layout = QVBoxLayout(team_box)
        self.team_table = QTableWidget()
        self.team_table.setAlternatingRowColors(True)
        self.team_table.verticalHeader().setVisible(False)
        self.team_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.team_table.setSelectionBehavior(QTableWidget.SelectRows)
        tb_layout.addWidget(self.team_table)
        root.addWidget(team_box)

        # ── Pipeline details / step summary table ──
        detail_box = QGroupBox("Step Details")
        detail_box.setStyleSheet(
            "QGroupBox { font-size: 12px; font-weight: bold; color: #8b949e; "
            "border: 1px solid #30363d; border-radius: 6px; padding-top: 18px; }"
            "QGroupBox::title { subcontrol-position: top left; padding: 4px 8px; }"
        )
        det_layout = QVBoxLayout(detail_box)
        self.detail_table = QTableWidget()
        self.detail_table.setAlternatingRowColors(True)
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setEditTriggers(QTableWidget.NoEditTriggers)
        det_layout.addWidget(self.detail_table)
        root.addWidget(detail_box)

        # ── Log ──
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #8b949e; font-size: 12px;")
        root.addWidget(self.status_lbl)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        self.log.setStyleSheet(
            "QTextEdit { background: #0d1117; color: #c9d1d9; font-family: Consolas, monospace; "
            "font-size: 11px; border: 1px solid #30363d; border-radius: 4px; }"
        )
        root.addWidget(self.log)

    # ── Button handlers ──────────────────────────────────────────────────

    def _on_optimize_clicked(self) -> None:
        if self._running:
            return
        self._running = True
        self._before = None
        self._after = None
        self._last_record = None
        self._current_step = -1
        log.info("[Optimize] 'Optimize Everything' clicked — starting 2-phase pipeline")

        # ── Pre-run blacklist check ──
        is_bl, bl_warnings, _ = check_blacklist_before_run(
            self.trials_spin.value(), self.trials_spin.value()
        )
        if is_bl:
            log.warning("[Optimize] Blacklist match detected — warning user")
            msg = "\n".join(bl_warnings)
            reply = QMessageBox.warning(
                self,
                "Blacklisted Configuration",
                f"{msg}\n\nDo you still want to proceed?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                self._running = False
                return

        # ── Capture weight backup before anything changes ──
        self._backup = WeightBackup()
        self._backup.capture()

        # Reset UI
        self.optimize_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.step_tracker.reset()
        self.log.clear()
        self.team_table.setRowCount(0)
        self.detail_table.setRowCount(0)
        self._reset_cards()
        self.regression_banner.setVisible(False)
        self.success_banner.setVisible(False)
        self.regression_suggestions.setVisible(False)
        self.apply_suggested_btn.setVisible(False)
        self.view_diffs_btn.setVisible(False)

        self.status_lbl.setText("Phase 1/2 — Capturing baseline accuracy...")
        self.log.append("Starting optimisation...\n")
        if is_bl:
            self.log.append("⚠️ NOTE: This config was previously blacklisted — proceeding anyway\n")

        # Phase 1: capture "before" snapshot
        self._start_snapshot()

    def _on_cancel(self) -> None:
        if self._worker and hasattr(self._worker, "cancel"):
            self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self.log.append("\nCancellation requested — stopping after current step...")

    # ── Phase 1: Before snapshot ─────────────────────────────────────────

    def _start_snapshot(self) -> None:
        self._cleanup_thread()
        log.info("[Optimize] Phase 1 — starting baseline snapshot worker")
        thread = QThread()
        worker = _SnapshotWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_log)
        worker.finished.connect(self._on_snapshot_done)
        worker.error.connect(self._on_snapshot_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_thread_refs)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_snapshot_done(self, snap: Snapshot) -> None:
        self._before = snap
        self._fill_before_cards(snap)
        log.info("[Optimize] Phase 1 done — baseline: %d games, %.1f%% winner, "
                 "%.2f avg spread err", snap.total_games, snap.winner_pct, snap.avg_spread_error)
        self.log.append(f"Baseline captured: {snap.total_games} games, "
                        f"{snap.winner_pct:.1f}% winner accuracy\n")

        # Phase 2: run pipeline
        self.status_lbl.setText("Phase 2/2 — Running optimisation pipeline...")
        self._start_pipeline()

    def _on_snapshot_error(self, msg: str) -> None:
        log.warning("[Optimize] Phase 1 failed — %s", msg.split('\n')[0])
        self.log.append(f"Baseline capture failed: {msg}")
        self.log.append("Proceeding without baseline...\n")
        # Still run the pipeline even without a baseline
        self.status_lbl.setText("Phase 2/2 — Running optimisation pipeline...")
        self._start_pipeline()

    # ── Phase 2: Pipeline ────────────────────────────────────────────────

    def _start_pipeline(self) -> None:
        self._cleanup_thread()
        log.info("[Optimize] Phase 2 — starting pipeline worker (trials=%d, force=%s)",
                 self.trials_spin.value(), self.force_checkbox.isChecked())
        thread = QThread()
        worker = _PipelineWorker(
            n_trials=self.trials_spin.value(),
            team_trials=self.trials_spin.value(),
            max_workers=4,
            force_rerun=self.force_checkbox.isChecked(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_log)
        worker.step_update.connect(self._on_step_update)
        worker.finished.connect(self._on_pipeline_done)
        worker.error.connect(self._on_pipeline_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_thread_refs)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_step_update(self, index: int, status: str, detail: str) -> None:
        # Mark previous step as done if we advanced
        if self._current_step >= 0 and index > self._current_step:
            self.step_tracker.set_step_status(self._current_step, "done")
        self._current_step = index
        self.step_tracker.set_step_status(index, status)

    def _on_pipeline_done(self, summary: dict) -> None:
        self.progress_bar.setVisible(False)
        self.cancel_btn.setEnabled(False)

        # Mark all steps from summary
        self.step_tracker.mark_all_done(summary.get("steps", {}))

        total_s = summary.get("total_seconds", 0)
        log.info("[Optimize] Phase 2 done — pipeline completed in %.0fs", total_s)
        self.log.append(f"\nPipeline completed in {total_s:.0f} seconds.")

        # Extract after snapshot from the pipeline's backtest results
        bt = summary.get("backtest_results")
        if bt:
            self._after = Snapshot.from_backtest(bt)
            self._fill_after_cards(self._after)
            if self._before:
                self._fill_deltas()
                self._fill_team_comparison()
                # Log the before/after comparison for traceability
                b, a = self._before, self._after
                log.info("[Optimize] BEFORE: winner=%.1f%%, spread_err=%.2f, total_err=%.2f",
                         b.winner_pct, b.avg_spread_error, b.avg_total_error)
                log.info("[Optimize] AFTER:  winner=%.1f%%, spread_err=%.2f, total_err=%.2f",
                         a.winner_pct, a.avg_spread_error, a.avg_total_error)
                log.info("[Optimize] DELTA:  winner=%+.1f%%, spread_err=%+.2f, total_err=%+.2f",
                         a.winner_pct - b.winner_pct,
                         a.avg_spread_error - b.avg_spread_error,
                         a.avg_total_error - b.avg_total_error)

                # ── Regression guard evaluation ──
                if self._backup:
                    record = evaluate_run(
                        before_snap=self._before,
                        after_snap=self._after,
                        backup=self._backup,
                        n_trials=self.trials_spin.value(),
                        team_trials=self.trials_spin.value(),
                        force_rerun=self.force_checkbox.isChecked(),
                    )
                    self._last_record = record
                    self._show_guard_result(record)
                else:
                    log.warning("[Optimize] No backup available — skipping regression guard")

        # Fill step detail table
        self._fill_step_table(summary)

        if not (self._last_record and self._last_record.was_regression):
            self.status_lbl.setText(
                f"Done in {total_s:.0f}s — "
                + (self._summary_line() if self._before and self._after else "optimization complete")
            )
        self._finish()

    def _show_guard_result(self, record: OptimizeRunRecord) -> None:
        """Show regression or success banner based on guard evaluation."""
        if record.was_regression:
            # ── REGRESSION ──
            self.regression_banner.setVisible(True)
            self.success_banner.setVisible(False)

            reasons_html = "<br>".join(record.regression_reasons) if record.regression_reasons else "Composite score decreased"
            self.regression_detail.setText(
                f"Composite score: {record.score_before:.2f} → {record.score_after:.2f} "
                f"(Δ{record.score_delta:+.2f})<br><br>{reasons_html}<br><br>"
                f"All weights and ML models have been restored to their pre-optimization state."
            )

            # Show suggestions
            suggestions = generate_suggestions(record)
            if suggestions:
                self.regression_suggestions.setVisible(True)
                self.regression_suggestions.setPlainText("\n".join(suggestions))

            # Check if we have suggested params
            suggested = suggest_adjusted_params(record)
            if suggested:
                self._suggested_params = suggested
                self.apply_suggested_btn.setVisible(True)
                self.apply_suggested_btn.setText(
                    f"💡 Apply {len(suggested)} Suggested Parameter Adjustments"
                )
            self.view_diffs_btn.setVisible(True)

            self.status_lbl.setText(
                f"⚠️ REGRESSION — rolled back (score {record.score_before:.2f} → {record.score_after:.2f})"
            )
            self.status_lbl.setStyleSheet("color: #f85149; font-size: 12px; font-weight: bold;")

            # Log full summary to the log pane
            self.log.append("\n" + "=" * 60)
            self.log.append("⚠️  REGRESSION DETECTED — ALL CHANGES ROLLED BACK")
            self.log.append("=" * 60)
            self.log.append(format_run_summary(record))
            if record.weight_diffs:
                self.log.append("\nWeight changes that caused regression:")
                for param, diff in sorted(record.weight_diffs.items(),
                                           key=lambda x: abs(x[1].get("delta", 0)),
                                           reverse=True):
                    self.log.append(
                        f"  {param}: {diff['before']:.6f} → {diff['after']:.6f} "
                        f"(Δ{diff['delta']:+.6f}, {diff.get('pct_change', 0):+.1f}%)"
                    )
        else:
            # ── SUCCESS ──
            self.success_banner.setVisible(True)
            self.regression_banner.setVisible(False)
            self.success_detail.setText(
                f"Composite score: {record.score_before:.2f} → {record.score_after:.2f} "
                f"(Δ{record.score_delta:+.2f})"
            )
            self.status_lbl.setStyleSheet("color: #3fb950; font-size: 12px; font-weight: bold;")
            self.log.append(f"\n✓ Optimization improved results — changes kept "
                            f"(score {record.score_before:.2f} → {record.score_after:.2f})")

    def _on_pipeline_error(self, msg: str) -> None:
        self.progress_bar.setVisible(False)
        log.error("[Optimize] Pipeline error: %s", msg.split('\n')[0])
        self.log.append(f"\nPipeline error: {msg}")
        self.status_lbl.setText("Pipeline failed — see log for details")
        # Restore weights on error too
        if self._backup:
            log.info("[Optimize] Restoring weights after pipeline error")
            self._backup.restore()
            self.log.append("\n⚠️ Weights restored to pre-optimization state due to error.")
        self._finish()

    # ── Card population ──────────────────────────────────────────────────

    def _reset_cards(self) -> None:
        for lbl in [self.b_games, self.b_winner, self.b_spread, self.b_total10, self.b_totalerr,
                     self.a_games, self.a_winner, self.a_spread, self.a_total10, self.a_totalerr]:
            lbl.setText("--")
        for lbl in [self.d_games, self.d_winner, self.d_spread, self.d_total10, self.d_totalerr]:
            lbl.setText("")

    def _fill_before_cards(self, s: Snapshot) -> None:
        self.b_games.setText(str(s.total_games))
        self.b_winner.setText(f"{s.winner_pct:.1f}%")
        self.b_spread.setText(f"{s.avg_spread_error:.2f}")
        self.b_total10.setText(f"{s.total_in_10_pct:.1f}%")
        self.b_totalerr.setText(f"{s.avg_total_error:.2f}")

    def _fill_after_cards(self, s: Snapshot) -> None:
        self.a_games.setText(str(s.total_games))
        self.a_winner.setText(f"{s.winner_pct:.1f}%")
        self.a_spread.setText(f"{s.avg_spread_error:.2f}")
        self.a_total10.setText(f"{s.total_in_10_pct:.1f}%")
        self.a_totalerr.setText(f"{s.avg_total_error:.2f}")

    def _fill_deltas(self) -> None:
        b, a = self._before, self._after
        if not b or not a:
            return
        self.d_games.setText(
            _format_delta(b.total_games, a.total_games, True)
            if a.total_games != b.total_games else ""
        )
        self.d_winner.setText(_format_delta(b.winner_pct, a.winner_pct, True, "%"))
        self.d_spread.setText(_format_delta(b.avg_spread_error, a.avg_spread_error, False, " pts"))
        self.d_total10.setText(_format_delta(b.total_in_10_pct, a.total_in_10_pct, True, "%"))
        self.d_totalerr.setText(_format_delta(b.avg_total_error, a.avg_total_error, False, " pts"))

    def _summary_line(self) -> str:
        b, a = self._before, self._after
        if not b or not a:
            return ""
        wd = a.winner_pct - b.winner_pct
        sd = a.avg_spread_error - b.avg_spread_error
        parts = []
        if abs(wd) >= 0.05:
            parts.append(f"winner {'▲' if wd > 0 else '▼'}{abs(wd):.1f}%")
        if abs(sd) >= 0.01:
            parts.append(f"spread err {'▼' if sd < 0 else '▲'}{abs(sd):.2f}")
        return ", ".join(parts) if parts else "no significant change"

    # ── Team comparison table ────────────────────────────────────────────

    def _fill_team_comparison(self) -> None:
        b, a = self._before, self._after
        if not b or not a:
            return

        all_abbrs = sorted(set(b.team_data.keys()) | set(a.team_data.keys()))
        headers = [
            "Team", "Games",
            "Winner % (Before)", "Winner % (After)", "Δ Winner",
            "Spread Err (Before)", "Spread Err (After)", "Δ Spread",
        ]
        self.team_table.setColumnCount(len(headers))
        self.team_table.setHorizontalHeaderLabels(headers)
        self.team_table.setRowCount(len(all_abbrs))
        self.team_table.setSortingEnabled(False)

        for row, abbr in enumerate(all_abbrs):
            bd = b.team_data.get(abbr, {})
            ad = a.team_data.get(abbr, {})

            self.team_table.setItem(row, 0, QTableWidgetItem(abbr))
            self.team_table.setItem(row, 1, QTableWidgetItem(
                str(ad.get("games", bd.get("games", 0)))))

            bw = bd.get("winner_pct", 0)
            aw = ad.get("winner_pct", 0)
            self.team_table.setItem(row, 2, QTableWidgetItem(f"{bw:.1f}%"))
            self.team_table.setItem(row, 3, QTableWidgetItem(f"{aw:.1f}%"))
            delta_w = aw - bw
            di = QTableWidgetItem(f"{delta_w:+.1f}%")
            if delta_w > 0.5:
                di.setForeground(QColor("#3fb950"))
            elif delta_w < -0.5:
                di.setForeground(QColor("#f85149"))
            self.team_table.setItem(row, 4, di)

            bs = bd.get("avg_spread_err", 0)
            as_ = ad.get("avg_spread_err", 0)
            self.team_table.setItem(row, 5, QTableWidgetItem(f"{bs:.2f}"))
            self.team_table.setItem(row, 6, QTableWidgetItem(f"{as_:.2f}"))
            delta_s = as_ - bs
            ds = QTableWidgetItem(f"{delta_s:+.2f}")
            if delta_s < -0.1:
                ds.setForeground(QColor("#3fb950"))  # lower error = better
            elif delta_s > 0.1:
                ds.setForeground(QColor("#f85149"))
            self.team_table.setItem(row, 7, ds)

        self.team_table.setSortingEnabled(True)
        self.team_table.resizeColumnsToContents()

    # ── Step detail table ────────────────────────────────────────────────

    def _fill_step_table(self, summary: dict) -> None:
        steps = summary.get("steps", {})
        headers = ["Step", "Status", "Time", "Details"]
        self.detail_table.setColumnCount(len(headers))
        self.detail_table.setHorizontalHeaderLabels(headers)
        self.detail_table.setRowCount(len(steps))
        self.detail_table.setSortingEnabled(False)

        for row, (name, info) in enumerate(steps.items()):
            self.detail_table.setItem(row, 0, QTableWidgetItem(name))

            status = info.get("status", "?")
            si = QTableWidgetItem(status)
            if status == "done":
                si.setForeground(QColor("#3fb950"))
            elif status == "skipped":
                si.setForeground(QColor("#484f58"))
            elif status == "error":
                si.setForeground(QColor("#f85149"))
            self.detail_table.setItem(row, 1, si)

            secs = info.get("seconds", 0)
            self.detail_table.setItem(row, 2, QTableWidgetItem(f"{secs:.1f}s"))

            detail = ""
            if "improvement_pct" in info:
                detail = f"{info['improvement_pct']:+.1f}% weight improvement"
            elif "spread_val_mae" in info:
                detail = f"spread MAE={info['spread_val_mae']:.2f}, total MAE={info['total_val_mae']:.2f}"
            elif "adopted" in info:
                detail = f"{info['adopted']} teams adopted refined weights"
            elif "games" in info:
                detail = f"{info['games']} games — {info.get('winner_pct', 0):.1f}% winner"
            elif "error" in info:
                detail = info["error"][:80]
            elif info.get("new_data") is not None:
                detail = "new data detected" if info["new_data"] else "no new data"
            self.detail_table.setItem(row, 3, QTableWidgetItem(detail))

        self.detail_table.resizeColumnsToContents()

    # ── Guard action handlers ───────────────────────────────────────────

    def _on_apply_suggested(self) -> None:
        """Apply the suggested parameter adjustments from the regression guard."""
        if not hasattr(self, "_suggested_params") or not self._suggested_params:
            return
        reply = QMessageBox.question(
            self,
            "Apply Suggested Parameters",
            f"This will modify {len(self._suggested_params)} weight parameters to "
            f"halfway between the 'before' and 'after' values.\n\n"
            f"Parameters to adjust:\n"
            + "\n".join(f"  • {k}: → {v:.4f}" for k, v in list(self._suggested_params.items())[:8])
            + ("\n  ..." if len(self._suggested_params) > 8 else "")
            + "\n\nApply these changes?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            apply_suggested_params(self._suggested_params)
            self.log.append(f"\n💡 Applied {len(self._suggested_params)} suggested parameter adjustments.")
            self.log.append("Run optimization again to test the new parameters.\n")
            self.apply_suggested_btn.setEnabled(False)
            self.apply_suggested_btn.setText("✓ Parameters Applied")

    def _on_view_diffs(self) -> None:
        """Show a dialog with the full weight diff table."""
        if not self._last_record or not self._last_record.weight_diffs:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Weight Changes (Regression)")
        dlg.setMinimumSize(700, 500)
        layout = QVBoxLayout(dlg)

        info = QLabel(
            f"Showing {len(self._last_record.weight_diffs)} parameters that changed.\n"
            f"Sorted by largest absolute change."
        )
        info.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(info)

        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        headers = ["Parameter", "Before", "After", "Delta", "% Change"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)

        diffs = sorted(self._last_record.weight_diffs.items(),
                       key=lambda x: abs(x[1].get("delta", 0)), reverse=True)
        table.setRowCount(len(diffs))

        for row, (param, diff) in enumerate(diffs):
            table.setItem(row, 0, QTableWidgetItem(param))
            table.setItem(row, 1, QTableWidgetItem(f"{diff['before']:.6f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{diff['after']:.6f}"))
            delta_item = QTableWidgetItem(f"{diff['delta']:+.6f}")
            delta_item.setForeground(QColor("#f85149") if diff['delta'] != 0 else QColor("#8b949e"))
            table.setItem(row, 3, delta_item)
            pct_item = QTableWidgetItem(f"{diff.get('pct_change', 0):+.1f}%")
            table.setItem(row, 4, pct_item)

        table.resizeColumnsToContents()
        layout.addWidget(table)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        layout.addWidget(btn_box)
        dlg.exec()

    def _on_show_history(self) -> None:
        """Show a dialog with optimization run history."""
        history = load_history()
        if not history:
            QMessageBox.information(self, "Run History", "No optimization runs recorded yet.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Optimization Run History")
        dlg.setMinimumSize(800, 500)
        layout = QVBoxLayout(dlg)

        # Summary label
        total = len(history)
        regressions = sum(1 for r in history if r.get("was_regression"))
        info = QLabel(
            f"{total} runs recorded • {regressions} regressions • "
            f"{total - regressions} improvements"
        )
        info.setStyleSheet("color: #8b949e; font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(info)

        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSortingEnabled(True)
        headers = [
            "Time", "Trials", "Result",
            "Winner % Before", "Winner % After", "Δ Winner",
            "Spread Err Before", "Spread Err After", "Δ Spread",
            "Score Before", "Score After", "Δ Score",
        ]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(history))

        for row_idx, rec in enumerate(reversed(history)):
            import time as _time
            ts = _time.strftime("%m/%d %H:%M", _time.localtime(rec.get("timestamp", 0)))
            table.setItem(row_idx, 0, QTableWidgetItem(ts))
            table.setItem(row_idx, 1, QTableWidgetItem(str(rec.get("n_trials", "?"))))

            result = "⚠️ REGRESSION" if rec.get("was_regression") else "✓ Improved"
            ri = QTableWidgetItem(result)
            ri.setForeground(QColor("#f85149") if rec.get("was_regression") else QColor("#3fb950"))
            table.setItem(row_idx, 2, ri)

            table.setItem(row_idx, 3, QTableWidgetItem(f"{rec.get('before_winner_pct', 0):.1f}%"))
            table.setItem(row_idx, 4, QTableWidgetItem(f"{rec.get('after_winner_pct', 0):.1f}%"))
            dw = rec.get("delta_winner_pct", 0)
            dwi = QTableWidgetItem(f"{dw:+.1f}%")
            dwi.setForeground(QColor("#3fb950") if dw > 0 else QColor("#f85149") if dw < 0 else QColor("#8b949e"))
            table.setItem(row_idx, 5, dwi)

            table.setItem(row_idx, 6, QTableWidgetItem(f"{rec.get('before_avg_spread_error', 0):.2f}"))
            table.setItem(row_idx, 7, QTableWidgetItem(f"{rec.get('after_avg_spread_error', 0):.2f}"))
            ds = rec.get("delta_avg_spread_error", 0)
            dsi = QTableWidgetItem(f"{ds:+.2f}")
            dsi.setForeground(QColor("#3fb950") if ds < 0 else QColor("#f85149") if ds > 0 else QColor("#8b949e"))
            table.setItem(row_idx, 8, dsi)

            table.setItem(row_idx, 9, QTableWidgetItem(f"{rec.get('score_before', 0):.2f}"))
            table.setItem(row_idx, 10, QTableWidgetItem(f"{rec.get('score_after', 0):.2f}"))
            sd = rec.get("score_delta", 0)
            sdi = QTableWidgetItem(f"{sd:+.2f}")
            sdi.setForeground(QColor("#3fb950") if sd > 0 else QColor("#f85149") if sd < 0 else QColor("#8b949e"))
            table.setItem(row_idx, 11, sdi)

        table.resizeColumnsToContents()
        layout.addWidget(table)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        layout.addWidget(btn_box)
        dlg.exec()

    def _on_clear_blacklist(self) -> None:
        """Clear the configuration blacklist."""
        reply = QMessageBox.question(
            self,
            "Clear Blacklist",
            "This will remove all blacklisted configurations, allowing previously\n"
            "regressive parameter combinations to be tried again.\n\n"
            "Clear the blacklist?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            clear_blacklist()
            self.log.append("\n🗑 Blacklist cleared — all configurations can be tried again.\n")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _on_log(self, msg: str) -> None:
        self.log.append(msg)
        # Auto-scroll
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _finish(self) -> None:
        self._running = False
        self.optimize_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            log.debug("[Optimize-Thread] Cleaning up previous thread (disconnecting stale finished signal)")
            try:
                self._thread.finished.disconnect(self._cleanup_thread_refs)
            except RuntimeError:
                pass
            if self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(3000)
            self._thread = None
            self._worker = None

    def _cleanup_thread_refs(self) -> None:
        """Called on thread.finished — only clear refs if sender is current thread."""
        sender = self.sender()
        if sender is not None and sender is not self._thread:
            log.debug("[Optimize-Thread] Ignoring stale thread.finished signal (sender != current thread)")
            return  # stale signal from a previous thread — ignore
        log.debug("[Optimize-Thread] Thread finished — clearing refs")
        self._thread = None
        self._worker = None
