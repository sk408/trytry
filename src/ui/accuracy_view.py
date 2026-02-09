from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.analytics.backtester import (
    run_backtest,
    BacktestResults,
    load_backtest_cache,
    get_backtest_cache_age,
)
from src.analytics.weight_optimizer import (
    run_weight_optimiser,
    run_per_team_refinement,
    run_combo_optimiser,
    build_residual_calibration,
    load_residual_calibration,
    run_feature_importance,
    run_ml_feature_importance,
    run_fft_error_analysis,
    OptimiserResult,
    ComboOptimiserResult,
    TeamRefinementResult,
    FeatureImportance,
    MLFeatureImportance,
    FFTPattern,
)
from src.analytics.pipeline import run_full_pipeline
from src.analytics.weight_config import get_weight_config, clear_weights, clear_team_weights
from src.database.db import get_conn


class BacktestWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # BacktestResults
    error = Signal(str)
    
    def __init__(
        self,
        home_team_filter: int | None = None,
        away_team_filter: int | None = None,
        use_injury_adjustment: bool = True,
        max_workers: int = 4,
    ):
        super().__init__()
        self.home_team_filter = home_team_filter
        self.away_team_filter = away_team_filter
        self.use_injury_adjustment = use_injury_adjustment
        self.max_workers = max_workers
    
    def run(self) -> None:
        try:
            results = run_backtest(
                min_games_before=5,
                home_team_filter=self.home_team_filter,
                away_team_filter=self.away_team_filter,
                progress_cb=self.progress.emit,
                use_injury_adjustment=self.use_injury_adjustment,
                max_workers=self.max_workers,
            )
            self.finished.emit(results)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class OptimiserWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # OptimiserResult
    error = Signal(str)

    def __init__(self, n_trials: int = 200):
        super().__init__()
        self.n_trials = n_trials

    def run(self) -> None:
        try:
            result = run_weight_optimiser(
                n_trials=self.n_trials,
                progress_cb=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class CalibrationWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # dict
    error = Signal(str)

    def run(self) -> None:
        try:
            result = build_residual_calibration(progress_cb=self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class FeatureImportanceWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # List[FeatureImportance]
    error = Signal(str)

    def run(self) -> None:
        try:
            result = run_feature_importance(progress_cb=self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class MLFeatureImportanceWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # List[MLFeatureImportance]
    error = Signal(str)

    def run(self) -> None:
        try:
            result = run_ml_feature_importance(progress_cb=self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class GroupedImportanceWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # List[GroupedFeatureImportance]
    error = Signal(str)

    def run(self) -> None:
        try:
            from src.analytics.weight_optimizer import run_grouped_feature_importance
            result = run_grouped_feature_importance(progress_cb=self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class FFTAnalysisWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # List[FFTPattern]
    error = Signal(str)

    def run(self) -> None:
        try:
            result = run_fft_error_analysis(progress_cb=self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class TeamRefineWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # List[TeamRefinementResult]
    error = Signal(str)

    def __init__(self, n_trials: int = 100):
        super().__init__()
        self.n_trials = n_trials

    def run(self) -> None:
        try:
            result = run_per_team_refinement(
                n_trials=self.n_trials,
                progress_cb=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class ComboOptimiserWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # ComboOptimiserResult
    error = Signal(str)

    def __init__(self, n_trials: int = 200, team_trials: int = 100):
        super().__init__()
        self.n_trials = n_trials
        self.team_trials = team_trials

    def run(self) -> None:
        try:
            result = run_combo_optimiser(
                n_trials=self.n_trials,
                team_trials=self.team_trials,
                progress_cb=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class ContinuousOptWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # ContinuousOptResult
    error = Signal(str)

    def __init__(self, n_trials: int = 200, team_trials: int = 100):
        super().__init__()
        self.n_trials = n_trials
        self.team_trials = team_trials
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from src.analytics.weight_optimizer import run_continuous_optimiser
            result = run_continuous_optimiser(
                n_trials=self.n_trials,
                team_trials=self.team_trials,
                progress_cb=self.progress.emit,
                cancel_check=lambda: self._cancel,
            )
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class FullPipelineWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # dict summary
    error = Signal(str)

    def __init__(
        self,
        n_trials: int = 200,
        team_trials: int = 100,
        autotune_strength: float = 0.75,
        max_workers: int = 4,
        force_rerun: bool = False,
    ):
        super().__init__()
        self.n_trials = n_trials
        self.team_trials = team_trials
        self.autotune_strength = autotune_strength
        self.max_workers = max_workers
        self.force_rerun = force_rerun
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            result = run_full_pipeline(
                n_trials=self.n_trials,
                team_trials=self.team_trials,
                autotune_strength=self.autotune_strength,
                max_workers=self.max_workers,
                progress_cb=self.progress.emit,
                cancel_check=lambda: self._cancel,
                force_rerun=self.force_rerun,
            )
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class MLTrainWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # MLTrainingResult
    error = Signal(str)

    def run(self) -> None:
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.ml_model import train_models, reload_models
            games = precompute_game_data(progress_cb=self.progress.emit)
            if not games:
                self.error.emit("No precomputed games available. Run a data sync first.")
                return
            result = train_models(games, progress_cb=self.progress.emit)
            reload_models()
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


def _teams_df() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT team_id, abbreviation, name FROM teams ORDER BY abbreviation", conn)


class AccuracyView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._thread = None
        self._worker = None
        
        # Team filters - separate home and away
        self.home_team_combo = QComboBox()
        self.home_team_combo.addItem("Any Home Team", None)
        self.away_team_combo = QComboBox()
        self.away_team_combo.addItem("Any Away Team", None)
        
        # Summary labels
        self.total_games_label = QLabel("--")
        self.total_games_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.spread_accuracy_label = QLabel("--")
        self.spread_accuracy_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.total_accuracy_label = QLabel("--")
        self.total_accuracy_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.avg_spread_err_label = QLabel("--")
        self.avg_total_err_label = QLabel("--")
        
        # Team accuracy table
        self.team_table = QTableWidget()
        self.team_table.setAlternatingRowColors(True)
        self.team_table.verticalHeader().setVisible(False)
        self.team_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.team_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Recent predictions table
        self.predictions_table = QTableWidget()
        self.predictions_table.setAlternatingRowColors(True)
        self.predictions_table.verticalHeader().setVisible(False)
        self.predictions_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.predictions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Status log
        self.status = QLabel("Select home/away teams (or Any) and click 'Run Backtest'")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(100)
        
        # Buttons
        self.run_button = QPushButton("  Run Backtest")
        self.run_button.setProperty("cssClass", "primary")
        self.run_button.clicked.connect(self.run_backtest)  # type: ignore[arg-type]
        self.refresh_teams_btn = QPushButton("Refresh Teams")
        self.refresh_teams_btn.clicked.connect(self._load_teams)  # type: ignore[arg-type]
        
        # Injury adjustment checkbox
        self.use_injuries_checkbox = QCheckBox("Use Injury Adjustments")
        self.use_injuries_checkbox.setChecked(True)
        self.use_injuries_checkbox.setToolTip(
            "Adjust predictions based on which players were out for each game.\n"
            "Requires 'Build Injury History' to be run first from Dashboard."
        )

        # Workers spinner
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        self.workers_spin.setSuffix(" threads")
        self.workers_spin.setToolTip(
            "Number of parallel threads for backtesting.\n"
            "More threads = faster, but uses more CPU.\n"
            "4 is a good default for most machines."
        )

        # Cache checkbox
        self.use_cache_checkbox = QCheckBox("Use cached results")
        self.use_cache_checkbox.setChecked(False)
        self.use_cache_checkbox.setToolTip(
            "If recent backtest results exist for these settings,\n"
            "load them instantly instead of re-running."
        )
        self.cache_age_label = QLabel("")
        self.cache_age_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        # Update cache age when filters change
        self.home_team_combo.currentIndexChanged.connect(self._update_cache_age_label)  # type: ignore
        self.away_team_combo.currentIndexChanged.connect(self._update_cache_age_label)  # type: ignore
        self.use_injuries_checkbox.stateChanged.connect(self._update_cache_age_label)  # type: ignore
        
        # ──── Model optimisation buttons ────
        self.optimise_btn = QPushButton("  Optimize Weights")
        self.optimise_btn.setToolTip(
            "Random-search optimisation over the top 10 prediction weights.\n"
            "Runs ~200 backtest trials to find the best weight combination."
        )
        self.optimise_btn.clicked.connect(self._run_optimiser)  # type: ignore[arg-type]

        self.calibrate_btn = QPushButton("  Build Calibration")
        self.calibrate_btn.setToolTip(
            "Compute residual calibration table by spread-prediction bin.\n"
            "Corrects systematic over/under-prediction in each range."
        )
        self.calibrate_btn.clicked.connect(self._run_calibration)  # type: ignore[arg-type]

        self.feature_btn = QPushButton("  Feature Importance")
        self.feature_btn.setToolTip(
            "Measure each prediction factor's impact on accuracy\n"
            "by disabling them one at a time."
        )
        self.feature_btn.clicked.connect(self._run_feature_importance)  # type: ignore[arg-type]

        self.clear_weights_btn = QPushButton("Reset Weights")
        self.clear_weights_btn.setToolTip("Clear optimised weights and revert to defaults.")
        self.clear_weights_btn.clicked.connect(self._clear_weights)  # type: ignore[arg-type]

        # Trials spinner for Optuna optimisation
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(50, 2000)
        self.trials_spin.setValue(200)
        self.trials_spin.setSingleStep(50)
        self.trials_spin.setSuffix(" trials")
        self.trials_spin.setToolTip(
            "Number of weight combinations to evaluate.\n"
            "More trials = better results but takes longer.\n"
            "200 is a good default; 500+ for thorough optimisation."
        )

        self.ml_feature_btn = QPushButton("  ML Feature Importance")
        self.ml_feature_btn.setToolTip(
            "XGBoost + SHAP analysis of feature contributions.\n"
            "Trains a model on the raw feature matrix and computes SHAP values."
        )
        self.ml_feature_btn.clicked.connect(self._run_ml_feature_importance)  # type: ignore[arg-type]

        self.team_refine_btn = QPushButton("  Per-Team Refinement")
        self.team_refine_btn.setToolTip(
            "Refine weights for each team individually.\n"
            "Validates against most recent 5 games to pick\n"
            "whichever (global or per-team) predicts better."
        )
        self.team_refine_btn.clicked.connect(self._run_team_refinement)  # type: ignore[arg-type]

        self.grouped_feature_btn = QPushButton("  Grouped Importance")
        self.grouped_feature_btn.setToolTip(
            "Test groups of related features together.\n"
            "Catches interaction effects that single-feature\n"
            "disruption misses."
        )
        self.grouped_feature_btn.clicked.connect(self._run_grouped_importance)  # type: ignore[arg-type]

        self.fft_btn = QPushButton("  Error Patterns (FFT)")
        self.fft_btn.setToolTip(
            "Detect periodic patterns in prediction errors\n"
            "using Fourier analysis on league-wide data."
        )
        self.fft_btn.clicked.connect(self._run_fft_analysis)  # type: ignore[arg-type]

        self.ml_train_btn = QPushButton("  Train ML Model")
        self.ml_train_btn.setProperty("cssClass", "primary")
        self.ml_train_btn.setToolTip(
            "Train XGBoost spread/total models on historical game features.\n"
            "The ML model is blended with the base model for ensemble predictions.\n"
            "Shows training/validation MAE and top SHAP features."
        )
        self.ml_train_btn.clicked.connect(self._run_ml_train)  # type: ignore[arg-type]

        self.combo_opt_btn = QPushButton("  Optimize All")
        self.combo_opt_btn.setProperty("cssClass", "primary")
        self.combo_opt_btn.setToolTip(
            "Run global weight optimisation THEN per-team refinement\n"
            "in a single pass with shared precomputed data."
        )
        self.combo_opt_btn.clicked.connect(self._run_combo_optimiser)  # type: ignore[arg-type]

        self.continuous_opt_btn = QPushButton("  Continuous Optimize")
        self.continuous_opt_btn.setProperty("cssClass", "primary")
        self.continuous_opt_btn.setToolTip(
            "Loop global + per-team optimisation indefinitely.\n"
            "Each round explores new weight combinations (random seed).\n"
            "Only saves when it finds improvements.\n"
            "Click Cancel to stop — best results are kept."
        )
        self.continuous_opt_btn.clicked.connect(self._run_continuous_opt)  # type: ignore[arg-type]

        self.pipeline_btn = QPushButton("  Full Pipeline")
        self.pipeline_btn.setProperty("cssClass", "primary")
        self.pipeline_btn.setToolTip(
            "Run EVERYTHING: sync, injury history, autotune, global optimisation,\n"
            "per-team refinement, calibration, and validation backtest.\n"
            "Skips steps if data hasn't changed since last run."
        )
        self.pipeline_btn.clicked.connect(self._run_full_pipeline)  # type: ignore[arg-type]

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("cssClass", "danger")
        self.cancel_btn.setToolTip("Cancel the currently running pipeline")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_pipeline)  # type: ignore[arg-type]

        self.force_rerun_checkbox = QCheckBox("Force re-run")
        self.force_rerun_checkbox.setToolTip(
            "Bypass smart caching — re-run all steps\n"
            "even if data hasn't changed."
        )

        # Feature importance / calibration results table
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # Layout - filter row
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Home Team:"))
        filter_layout.addWidget(self.home_team_combo)
        filter_layout.addWidget(QLabel("Away Team:"))
        filter_layout.addWidget(self.away_team_combo)
        filter_layout.addWidget(self.refresh_teams_btn)
        filter_layout.addWidget(self.use_injuries_checkbox)
        filter_layout.addWidget(self.use_cache_checkbox)
        filter_layout.addWidget(self.cache_age_label)
        filter_layout.addStretch()
        filter_layout.addWidget(QLabel("Workers:"))
        filter_layout.addWidget(self.workers_spin)
        filter_layout.addWidget(self.run_button)
        
        self._load_teams()
        self._update_cache_age_label()
        
        # Summary box with metric cards
        summary_box = QGroupBox("Overall Accuracy")
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(12)

        def _metric_card(title_text: str, value_lbl: QLabel, accent: str) -> QFrame:
            card = QFrame()
            card.setStyleSheet(
                f"QFrame {{ background: #1c2e42; border: 1px solid #2a3f55;"
                f"  border-radius: 8px; border-top: 3px solid {accent}; }}"
            )
            lay = QVBoxLayout()
            lay.setContentsMargins(12, 8, 12, 8)
            t = QLabel(title_text)
            t.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 600;"
                            " text-transform: uppercase;")
            value_lbl.setStyleSheet(
                f"color: {accent}; font-size: 22px; font-weight: 700;"
            )
            lay.addWidget(t)
            lay.addWidget(value_lbl)
            card.setLayout(lay)
            return card

        summary_layout.addWidget(_metric_card("Games", self.total_games_label, "#3b82f6"))
        summary_layout.addWidget(_metric_card("Winner %", self.spread_accuracy_label, "#10b981"))
        summary_layout.addWidget(_metric_card("Avg Spread Err", self.avg_spread_err_label, "#f59e0b"))
        summary_layout.addWidget(_metric_card("Total in 10 %", self.total_accuracy_label, "#8b5cf6"))
        summary_layout.addWidget(_metric_card("Avg Total Err", self.avg_total_err_label, "#ef4444"))
        summary_box.setLayout(summary_layout)

        # ──── Model optimisation section ────
        opt_box = QGroupBox("Model Optimisation")
        opt_layout = QVBoxLayout()

        # Row 1: Main optimisation buttons
        opt_row1 = QHBoxLayout()
        opt_row1.addWidget(self.optimise_btn)
        opt_row1.addWidget(self.team_refine_btn)
        opt_row1.addWidget(self.combo_opt_btn)
        opt_row1.addWidget(self.ml_train_btn)
        opt_row1.addWidget(QLabel("Trials:"))
        opt_row1.addWidget(self.trials_spin)
        opt_row1.addWidget(self.clear_weights_btn)
        opt_row1.addStretch()

        # Row 2: Pipeline + analysis buttons
        opt_row2 = QHBoxLayout()
        opt_row2.addWidget(self.pipeline_btn)
        opt_row2.addWidget(self.continuous_opt_btn)
        opt_row2.addWidget(self.cancel_btn)
        opt_row2.addWidget(self.force_rerun_checkbox)
        opt_row2.addWidget(self.calibrate_btn)
        opt_row2.addWidget(self.feature_btn)
        opt_row2.addWidget(self.grouped_feature_btn)
        opt_row2.addWidget(self.ml_feature_btn)
        opt_row2.addWidget(self.fft_btn)
        opt_row2.addStretch()

        opt_layout.addLayout(opt_row1)
        opt_layout.addLayout(opt_row2)
        opt_box.setLayout(opt_layout)
        
        # Team accuracy box
        team_box = QGroupBox("Accuracy by Team")
        team_box_layout = QVBoxLayout()
        team_box_layout.addWidget(self.team_table)
        team_box.setLayout(team_box_layout)
        
        # Recent predictions box
        pred_box = QGroupBox("Recent Predictions vs Actual")
        pred_box_layout = QVBoxLayout()
        pred_box_layout.addWidget(self.predictions_table)
        pred_box.setLayout(pred_box_layout)

        # Optimisation / importance results
        results_box = QGroupBox("Optimisation Results")
        results_box_layout = QVBoxLayout()
        results_box_layout.addWidget(self.results_table)
        results_box.setLayout(results_box_layout)
        
        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(filter_layout)
        layout.addWidget(summary_box)
        layout.addWidget(opt_box)
        layout.addWidget(team_box)
        layout.addWidget(pred_box)
        layout.addWidget(results_box)
        layout.addWidget(self.status)
        layout.addWidget(self.log)
        self.setLayout(layout)

    def _load_teams(self) -> None:
        self.home_team_combo.clear()
        self.home_team_combo.addItem("Any Home Team", None)
        self.away_team_combo.clear()
        self.away_team_combo.addItem("Any Away Team", None)
        try:
            teams = _teams_df()
            for _, row in teams.iterrows():
                label = f"{row['abbreviation']} - {row['name']}"
                self.home_team_combo.addItem(label, int(row["team_id"]))
                self.away_team_combo.addItem(label, int(row["team_id"]))
        except Exception:
            pass

    def _update_cache_age_label(self) -> None:
        """Show how old the cached results are for the current filter settings."""
        home_filter = self.home_team_combo.currentData()
        away_filter = self.away_team_combo.currentData()
        use_injuries = self.use_injuries_checkbox.isChecked()
        age = get_backtest_cache_age(home_filter, away_filter, use_injuries)
        if age is None:
            self.cache_age_label.setText("(no cache)")
        elif age < 1:
            self.cache_age_label.setText("(cached <1 min ago)")
        elif age < 60:
            self.cache_age_label.setText(f"(cached {int(age)} min ago)")
        else:
            hours = age / 60
            self.cache_age_label.setText(f"(cached {hours:.1f} hr ago)")

    def run_backtest(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        
        self._set_buttons_enabled(False)
        self.status.setText("Running backtest...")
        self.log.clear()
        self.log.append("Starting backtest analysis...")
        
        home_filter = self.home_team_combo.currentData()
        away_filter = self.away_team_combo.currentData()
        use_injuries = self.use_injuries_checkbox.isChecked()
        
        # ── Check cache first ──
        if self.use_cache_checkbox.isChecked():
            cached = load_backtest_cache(
                home_filter, away_filter, use_injuries, max_age_minutes=1440
            )
            if cached is not None:
                self.log.append("Loaded results from cache (instant)")
                self._on_finished(cached)
                self._update_cache_age_label()
                return
            else:
                self.log.append("No valid cache found — running fresh backtest")
        
        # Log what we're filtering
        if home_filter and away_filter:
            self.log.append(f"Filtering: specific matchup (home vs away)")
        elif home_filter:
            self.log.append(f"Filtering: team as HOME only")
        elif away_filter:
            self.log.append(f"Filtering: team as AWAY only")
        else:
            self.log.append("Analyzing all games")
        
        self.log.append(f"Injury adjustments: {'ENABLED' if use_injuries else 'DISABLED'}")
        
        self._thread = QThread()
        self._worker = BacktestWorker(
            home_team_filter=home_filter,
            away_team_filter=away_filter,
            use_injury_adjustment=use_injuries,
            max_workers=self.workers_spin.value(),
        )
        self._worker.moveToThread(self._thread)
        
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_finished)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        
        self._thread.start()
    
    def _cleanup_thread(self) -> None:
        """Clean up thread after it finishes."""
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
    
    def _on_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)
    
    def _on_finished(self, results: BacktestResults) -> None:
        self._display_results(results)
        self.status.setText(f"Backtest complete: {results.total_games} games analyzed")
        self._set_buttons_enabled(True)
        self._update_cache_age_label()
        if self._thread:
            self._thread.quit()
            self._thread.wait()
    
    def _on_error(self, msg: str) -> None:
        self.status.setText("Operation failed")
        self.log.append(f"ERROR: {msg}")
        self._set_buttons_enabled(True)
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── Optimiser ────

    def _run_optimiser(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.log.clear()
        self.log.append("Starting weight optimisation (this may take a while)...")
        self.status.setText("Optimising weights...")

        self._thread = QThread()
        self._worker = OptimiserWorker(n_trials=self.trials_spin.value())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_optimiser_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_optimiser_done(self, result: OptimiserResult) -> None:
        self.log.append(
            f"\nOptimisation complete — loss {result.baseline_loss:.2f} → {result.best_loss:.2f} "
            f"({result.improvement_pct:+.1f}%)"
        )
        self.status.setText("Weight optimisation done!")
        self._set_buttons_enabled(True)
        # Show best weights in results table
        cfg = result.best_config.to_dict()
        self._show_dict_table("Weight", "Value", {k: f"{v:.4f}" for k, v in cfg.items()})
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── Calibration ────

    def _run_calibration(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.log.clear()
        self.log.append("Building residual calibration table...")
        self.status.setText("Building calibration...")

        self._thread = QThread()
        self._worker = CalibrationWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_calibration_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_calibration_done(self, calibration: dict) -> None:
        self.log.append(f"\nCalibration complete: {len(calibration)} bins populated")
        self.status.setText("Calibration done!")
        self._set_buttons_enabled(True)
        # Show calibration bins in results table
        headers = ["Bin", "Range", "Avg Residual", "Samples"]
        rows = []
        for label, data in calibration.items():
            rows.append([
                label,
                f"[{data['bin_low']:+.0f}, {data['bin_high']:+.0f})",
                f"{data['avg_residual']:+.3f}",
                str(data['sample_count']),
            ])
        self._show_list_table(headers, rows)
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── Feature importance ────

    def _run_feature_importance(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.log.clear()
        self.log.append("Running feature importance analysis...")
        self.status.setText("Analysing features...")

        self._thread = QThread()
        self._worker = FeatureImportanceWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_feature_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_feature_done(self, features: list) -> None:
        self.log.append(f"\nFeature importance complete: {len(features)} features analysed")
        self.status.setText("Feature importance done!")
        self._set_buttons_enabled(True)
        headers = ["Feature", "Baseline Loss", "Disabled Loss", "Impact", "Impact %", "Verdict"]
        rows = []
        for f in features:
            verdict = "HELPS" if f.impact > 0.05 else ("HURTS" if f.impact < -0.05 else "neutral")
            rows.append([
                f.feature_name,
                f"{f.baseline_loss:.2f}",
                f"{f.disabled_loss:.2f}",
                f"{f.impact:+.3f}",
                f"{f.impact_pct:+.2f}%",
                verdict,
            ])
        self._show_list_table(headers, rows)
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── Grouped Feature Importance ────

    def _run_grouped_importance(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.log.clear()
        self.log.append("Running grouped feature importance (interaction effects)...")
        self.status.setText("Testing feature groups...")

        self._thread = QThread()
        self._worker = GroupedImportanceWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_grouped_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_grouped_done(self, groups: list) -> None:
        self.log.append(f"\nGrouped importance complete: {len(groups)} groups analysed")
        self.status.setText("Grouped importance done!")
        self._set_buttons_enabled(True)
        headers = ["Group", "Weights Disabled", "Baseline Loss", "Disabled Loss",
                    "Impact", "Impact %", "Verdict"]
        rows = []
        for g in groups:
            verdict = "HELPS" if g.impact > 0.1 else ("HURTS" if g.impact < -0.1 else "neutral")
            rows.append([
                g.group_name,
                ", ".join(g.features_disabled),
                f"{g.baseline_loss:.2f}",
                f"{g.disabled_loss:.2f}",
                f"{g.impact:+.3f}",
                f"{g.impact_pct:+.2f}%",
                verdict,
            ])
        self._show_list_table(headers, rows)
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── ML Feature Importance ────

    def _run_ml_feature_importance(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.log.clear()
        self.log.append("Running ML feature importance (XGBoost + SHAP)...")
        self.status.setText("Analysing ML features...")

        self._thread = QThread()
        self._worker = MLFeatureImportanceWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_ml_feature_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_ml_feature_done(self, features: list) -> None:
        self.log.append(f"\nML feature importance complete: {len(features)} features analysed")
        self.status.setText("ML feature importance done!")
        self._set_buttons_enabled(True)
        headers = ["Feature", "SHAP Importance", "Direction"]
        rows = []
        for f in features:
            rows.append([
                f.feature_name,
                f"{f.shap_importance:.4f}",
                f.direction,
            ])
        self._show_list_table(headers, rows)
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── FFT Error Analysis ────

    def _run_fft_analysis(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.log.clear()
        self.log.append("Running FFT error pattern analysis...")
        self.status.setText("Analysing error patterns...")

        self._thread = QThread()
        self._worker = FFTAnalysisWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_fft_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_fft_done(self, patterns: list) -> None:
        count = len(patterns)
        self.log.append(
            f"\nFFT analysis complete: {count} pattern(s) detected"
            if count else "\nFFT analysis complete: no significant patterns (good)"
        )
        self.status.setText("FFT analysis done!")
        self._set_buttons_enabled(True)
        if patterns:
            headers = ["Description", "Period (games)", "Period (days)", "Strength"]
            rows = []
            for p in patterns:
                rows.append([
                    p.description,
                    f"{p.period_games:.1f}",
                    f"{p.period_days:.1f}",
                    f"{p.magnitude:.2f}",
                ])
            self._show_list_table(headers, rows)
        else:
            self._show_dict_table(
                "Result", "Value",
                {"Status": "No significant periodic error patterns detected (good)"},
            )
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── Train ML Model ────

    def _run_ml_train(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.log.clear()
        self.log.append("Training ML ensemble models (XGBoost spread + total)...")
        self.status.setText("Training ML models...")

        self._thread = QThread()
        self._worker = MLTrainWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_ml_train_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_ml_train_done(self, result) -> None:
        self.log.append(
            f"\nML training complete!\n"
            f"  Spread: train MAE={result.spread_train_mae:.2f}, "
            f"val MAE={result.spread_val_mae:.2f}\n"
            f"  Total:  train MAE={result.total_train_mae:.2f}, "
            f"val MAE={result.total_val_mae:.2f}\n"
            f"  {result.n_train} train / {result.n_val} validation games, "
            f"{result.n_features} features"
        )
        self.status.setText("ML training done!")
        self._set_buttons_enabled(True)

        # Show SHAP features if available, otherwise gain-based
        if result.shap_spread_features:
            headers = ["Feature", "SHAP Spread", "SHAP Total"]
            rows = []
            shap_total_map = {name: imp for name, imp in result.shap_total_features}
            for name, imp in result.shap_spread_features:
                rows.append([
                    name,
                    f"{imp:.4f}",
                    f"{shap_total_map.get(name, 0.0):.4f}",
                ])
            self._show_list_table(headers, rows)
        elif result.top_spread_features:
            headers = ["Feature", "Spread Gain", "Total Gain"]
            rows = []
            total_map = {name: gain for name, gain in result.top_total_features}
            for name, gain in result.top_spread_features:
                rows.append([
                    name,
                    f"{gain:.2f}",
                    f"{total_map.get(name, 0.0):.2f}",
                ])
            self._show_list_table(headers, rows)
        else:
            self._show_dict_table("Metric", "Value", {
                "Spread Val MAE": f"{result.spread_val_mae:.2f}",
                "Total Val MAE": f"{result.total_val_mae:.2f}",
                "Training Samples": str(result.n_train),
                "Validation Samples": str(result.n_val),
                "Features": str(result.n_features),
            })

        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── Per-team refinement ────

    def _run_team_refinement(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.log.clear()
        self.log.append("Running per-team weight refinement with regressive validation...")
        self.status.setText("Per-team refinement...")

        self._thread = QThread()
        self._worker = TeamRefineWorker(n_trials=self.trials_spin.value())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_team_refine_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_team_refine_done(self, results: list) -> None:
        adopted = sum(1 for r in results if r.used_team_weights)
        self.log.append(
            f"\nPer-team refinement complete: {adopted}/{len(results)} teams "
            f"got custom weights"
        )
        self.status.setText(f"Per-team refinement done: {adopted} teams refined")
        self._set_buttons_enabled(True)

        headers = ["Team", "Decision", "Global (holdout)", "Per-Team (holdout)", "Reason"]
        rows = []
        for r in sorted(results, key=lambda x: x.team_abbr):
            decision = "Per-Team" if r.used_team_weights else "Global"
            rows.append([
                r.team_abbr,
                decision,
                f"{r.global_loss_recent:.2f}",
                f"{r.team_loss_recent:.2f}",
                r.reason,
            ])
        self._show_list_table(headers, rows)

        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── Combo optimiser ────

    def _run_combo_optimiser(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.log.clear()
        self.log.append("Starting combo optimisation (global + per-team)...")
        self.status.setText("Combo optimisation running...")

        self._thread = QThread()
        self._worker = ComboOptimiserWorker(
            n_trials=self.trials_spin.value(),
            team_trials=self.trials_spin.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_combo_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_combo_done(self, result: ComboOptimiserResult) -> None:
        gr = result.global_result
        adopted = sum(1 for r in result.team_results if r.used_team_weights)
        self.log.append(
            f"\nCombo complete in {result.total_seconds:.0f}s\n"
            f"  Global: loss {gr.baseline_loss:.2f} → {gr.best_loss:.2f} "
            f"({gr.improvement_pct:+.1f}%)\n"
            f"  Per-team: {adopted}/{len(result.team_results)} teams refined"
        )
        self.status.setText("Combo optimisation done!")
        self._set_buttons_enabled(True)

        # Show team results in table
        headers = ["Team", "Decision", "Global (holdout)", "Per-Team (holdout)", "Reason"]
        rows = []
        for r in sorted(result.team_results, key=lambda x: x.team_abbr):
            decision = "Per-Team" if r.used_team_weights else "Global"
            rows.append([
                r.team_abbr, decision,
                f"{r.global_loss_recent:.2f}", f"{r.team_loss_recent:.2f}",
                r.reason,
            ])
        self._show_list_table(headers, rows)

        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── Continuous optimiser ────

    def _run_continuous_opt(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.cancel_btn.setEnabled(True)
        self.log.clear()
        self.log.append("Starting continuous optimisation (runs until cancelled)...")
        self.log.append("Click Cancel to stop — best results are always kept.\n")
        self.status.setText("Continuous optimisation running...")

        self._thread = QThread()
        self._worker = ContinuousOptWorker(
            n_trials=self.trials_spin.value(),
            team_trials=self.trials_spin.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_continuous_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_continuous_done(self, result) -> None:
        self.cancel_btn.setEnabled(False)
        self.log.append(
            f"\nContinuous optimisation stopped.\n"
            f"  Rounds completed: {result.rounds_completed}\n"
            f"  Global improvements saved: {result.global_improvements}\n"
            f"  Loss: {result.starting_loss:.2f} → {result.best_global_loss:.2f}\n"
            f"  Teams refined: {result.teams_refined}/{result.total_teams}\n"
            f"  Total time: {result.total_seconds:.0f}s"
        )
        self.status.setText(
            f"Continuous opt done: {result.rounds_completed} rounds, "
            f"loss {result.starting_loss:.2f} → {result.best_global_loss:.2f}"
        )
        self._set_buttons_enabled(True)

        self._show_dict_table("Metric", "Value", {
            "Rounds Completed": str(result.rounds_completed),
            "Global Improvements": str(result.global_improvements),
            "Starting Loss": f"{result.starting_loss:.2f}",
            "Best Global Loss": f"{result.best_global_loss:.2f}",
            "Teams Refined": f"{result.teams_refined}/{result.total_teams}",
            "Total Time": f"{result.total_seconds:.0f}s",
        })

        if self._thread:
            self._thread.quit()
            self._thread.wait()

    # ──── Full pipeline ────

    def _run_full_pipeline(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.cancel_btn.setEnabled(True)
        self.log.clear()
        self.log.append("Starting full optimisation pipeline...")
        self.status.setText("Full pipeline running...")

        self._thread = QThread()
        self._worker = FullPipelineWorker(
            n_trials=self.trials_spin.value(),
            team_trials=self.trials_spin.value(),
            max_workers=self.workers_spin.value(),
            force_rerun=self.force_rerun_checkbox.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)  # type: ignore
        self._worker.progress.connect(self._on_progress)  # type: ignore
        self._worker.finished.connect(self._on_pipeline_done)  # type: ignore
        self._worker.error.connect(self._on_error)  # type: ignore
        self._thread.finished.connect(self._cleanup_thread)  # type: ignore
        self._thread.start()

    def _on_pipeline_done(self, summary: dict) -> None:
        self.cancel_btn.setEnabled(False)
        total_s = summary.get("total_seconds", 0)
        self.log.append(f"\nFull pipeline complete in {total_s:.0f}s")

        # Show per-step summary
        headers = ["Step", "Status", "Time (s)", "Details"]
        rows = []
        for name, info in summary.get("steps", {}).items():
            details = ""
            if "improvement_pct" in info:
                details = f"{info['improvement_pct']:+.1f}% improvement"
            elif "spread_val_mae" in info:
                details = (
                    f"spread MAE={info['spread_val_mae']:.2f}, "
                    f"total MAE={info['total_val_mae']:.2f}, "
                    f"{info.get('n_features', '?')} features"
                )
            elif "adopted" in info:
                details = f"{info['adopted']} teams refined"
            elif "games" in info:
                details = f"{info['games']} games, {info.get('winner_pct', 0):.1f}% winner"
            elif "error" in info:
                details = info["error"][:60]
            elif info.get("new_data") is not None:
                details = "new data" if info["new_data"] else "no new data"
            rows.append([
                name, info.get("status", "?"),
                f"{info.get('seconds', 0):.1f}", details,
            ])
        self._show_list_table(headers, rows)

        # If backtest results are included, display them
        bt = summary.get("backtest_results")
        if bt:
            self._display_results(bt)

        self.status.setText(f"Full pipeline done in {total_s:.0f}s")
        self._set_buttons_enabled(True)

        if self._thread:
            self._thread.quit()
            self._thread.wait()

    def _cancel_pipeline(self) -> None:
        if self._worker and hasattr(self._worker, "cancel"):
            self._worker.cancel()
            self.log.append("Cancellation requested — will stop after current step...")
            self.cancel_btn.setEnabled(False)

    # ──── Clear weights ────

    def _clear_weights(self) -> None:
        clear_weights()
        clear_team_weights()
        self.log.append("Optimised weights cleared — defaults will be used.")
        self.status.setText("Weights reset to defaults")

    # ──── Table helpers ────

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.run_button.setEnabled(enabled)
        self.optimise_btn.setEnabled(enabled)
        self.calibrate_btn.setEnabled(enabled)
        self.feature_btn.setEnabled(enabled)
        self.clear_weights_btn.setEnabled(enabled)
        self.ml_feature_btn.setEnabled(enabled)
        self.ml_train_btn.setEnabled(enabled)
        self.team_refine_btn.setEnabled(enabled)
        self.fft_btn.setEnabled(enabled)
        self.grouped_feature_btn.setEnabled(enabled)
        self.combo_opt_btn.setEnabled(enabled)
        self.continuous_opt_btn.setEnabled(enabled)
        self.pipeline_btn.setEnabled(enabled)
        if enabled:
            self.cancel_btn.setEnabled(False)

    def _show_dict_table(self, key_header: str, val_header: str, data: dict) -> None:
        self.results_table.clear()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels([key_header, val_header])
        self.results_table.setRowCount(len(data))
        for i, (k, v) in enumerate(data.items()):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(v)))
        self.results_table.resizeColumnsToContents()

    def _show_list_table(self, headers: list, rows: list) -> None:
        self.results_table.clear()
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(rows))
        for r, row_data in enumerate(rows):
            for c, val in enumerate(row_data):
                item = QTableWidgetItem(str(val))
                # Color-code verdict column for feature importance
                if headers[-1] == "Verdict" and c == len(row_data) - 1:
                    if val == "HELPS":
                        item.setForeground(QColor("#10b981"))
                    elif val == "HURTS":
                        item.setForeground(QColor("#ef4444"))
                self.results_table.setItem(r, c, item)
        self.results_table.resizeColumnsToContents()

    def _display_results(self, results: BacktestResults) -> None:
        # Update summary
        self.total_games_label.setText(str(results.total_games))
        self.spread_accuracy_label.setText(f"{results.overall_spread_accuracy:.1f}%")
        self.total_accuracy_label.setText(f"{results.overall_total_accuracy:.1f}%")
        
        # Calculate average errors
        if results.predictions:
            avg_spread_err = sum(abs(p.spread_error) for p in results.predictions) / len(results.predictions)
            avg_total_err = sum(abs(p.total_error) for p in results.predictions) / len(results.predictions)
            self.avg_spread_err_label.setText(f"{avg_spread_err:.1f}")
            self.avg_total_err_label.setText(f"{avg_total_err:.1f}")
        
        # Color code accuracy
        spread_color = "#10b981" if results.overall_spread_accuracy >= 55 else \
                       "#f59e0b" if results.overall_spread_accuracy >= 50 else "#ef4444"
        self.spread_accuracy_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {spread_color};"
        )
        
        total_color = "#8b5cf6" if results.overall_total_accuracy >= 55 else \
                      "#f59e0b" if results.overall_total_accuracy >= 50 else "#ef4444"
        self.total_accuracy_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {total_color};"
        )
        
        self.log.append(f"Total games: {results.total_games}")
        self.log.append(f"Winner correct: {results.overall_spread_accuracy:.1f}%")
        self.log.append(f"Total within 10: {results.overall_total_accuracy:.1f}%")
        
        # Populate team accuracy table
        self._populate_team_table(results)
        
        # Populate predictions table
        self._populate_predictions_table(results)

    def _populate_team_table(self, results: BacktestResults) -> None:
        headers = ["Team", "Record", "Games", "Spread %", "Avg Spread Err", "Total %", "Avg Total Err"]
        self.team_table.clear()
        self.team_table.setColumnCount(len(headers))
        self.team_table.setHorizontalHeaderLabels(headers)
        
        # Sort by spread accuracy descending
        teams = sorted(
            results.team_accuracy.values(),
            key=lambda t: t.spread_accuracy if t.games_analyzed > 0 else 0,
            reverse=True,
        )
        
        # Filter to teams with games
        teams = [t for t in teams if t.games_analyzed > 0]
        self.team_table.setRowCount(len(teams))
        
        for row_idx, ta in enumerate(teams):
            record = f"{ta.wins}-{ta.losses}"
            items = [
                ta.team_abbr,
                record,
                str(ta.games_analyzed),
                f"{ta.spread_accuracy:.1f}%",
                f"{ta.avg_spread_error:.1f}",
                f"{ta.total_accuracy:.1f}%",
                f"{ta.avg_total_error:.1f}",
            ]
            
            for col_idx, val in enumerate(items):
                item = QTableWidgetItem(val)
                # Color code spread accuracy
                if col_idx == 3:  # Spread %
                    if ta.spread_accuracy >= 60:
                        item.setForeground(QColor("#10b981"))
                    elif ta.spread_accuracy < 45:
                        item.setForeground(QColor("#ef4444"))
                self.team_table.setItem(row_idx, col_idx, item)
        
        self.team_table.resizeColumnsToContents()

    def _populate_predictions_table(self, results: BacktestResults) -> None:
        headers = [
            "Date", "Matchup", "Final Score",
            "Pred Winner", "Actual Winner", "Correct?",
            "Pred Score", "Score Diff",
            "Pred Total", "Actual Total", "Total Diff",
            "Injuries",
        ]
        self.predictions_table.clear()
        self.predictions_table.setColumnCount(len(headers))
        self.predictions_table.setHorizontalHeaderLabels(headers)
        
        # Show most recent 50 predictions
        preds = sorted(results.predictions, key=lambda p: str(p.game_date), reverse=True)[:50]
        self.predictions_table.setRowCount(len(preds))
        
        for row_idx, p in enumerate(preds):
            matchup = f"{p.away_abbr} @ {p.home_abbr}"
            final_score = f"{int(p.actual_away_score)}-{int(p.actual_home_score)}"
            
            # Predicted winner display
            if p.predicted_winner == "HOME":
                pred_winner = p.home_abbr
            elif p.predicted_winner == "AWAY":
                pred_winner = p.away_abbr
            else:
                pred_winner = "Close"
            
            # Actual winner display
            if p.actual_winner == "HOME":
                actual_winner = p.home_abbr
            elif p.actual_winner == "AWAY":
                actual_winner = p.away_abbr
            else:
                actual_winner = "Tie"
            
            # Predicted score
            pred_score = f"{int(p.predicted_away_score)}-{int(p.predicted_home_score)}"
            
            # Score difference (how far off were we?)
            score_diff = f"H:{p.home_score_error:+.0f} A:{p.away_score_error:+.0f}"
            
            # Total difference
            total_diff = f"{p.total_error:+.0f}"
            
            # Injury summary
            injury_parts = []
            if p.home_injuries:
                # Show first 2 names with adjustment
                names = ", ".join(n.split()[-1] for n in p.home_injuries[:2])
                injury_parts.append(f"{p.home_abbr}: {names}")
            if p.away_injuries:
                names = ", ".join(n.split()[-1] for n in p.away_injuries[:2])
                injury_parts.append(f"{p.away_abbr}: {names}")
            injury_text = " | ".join(injury_parts) if injury_parts else "-"
            
            items = [
                str(p.game_date),
                matchup,
                final_score,
                pred_winner,
                actual_winner,
                "Yes" if p.winner_correct else "No",
                pred_score,
                score_diff,
                f"{p.predicted_total:.0f}",
                f"{p.actual_total:.0f}",
                total_diff,
                injury_text,
            ]
            
            for col_idx, val in enumerate(items):
                item = QTableWidgetItem(val)
                # Color winner correct column
                if col_idx == 5:
                    item.setForeground(QColor("#10b981") if p.winner_correct else QColor("#ef4444"))
                # Color total diff - green if within 10
                if col_idx == 10:
                    if abs(p.total_error) <= 10:
                        item.setForeground(QColor("#10b981"))
                    elif abs(p.total_error) <= 20:
                        item.setForeground(QColor("#f59e0b"))
                    else:
                        item.setForeground(QColor("#ef4444"))
                # Color injuries column if there are injuries
                if col_idx == 11 and injury_text != "-":
                    item.setForeground(QColor("#ef4444"))
                self.predictions_table.setItem(row_idx, col_idx, item)
        
        self.predictions_table.resizeColumnsToContents()
