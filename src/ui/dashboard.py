from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui.workers import (
    start_sync_worker,
    start_injury_sync_worker,
    start_injury_history_worker,
    start_team_metrics_worker,
    start_player_impact_worker,
    start_image_sync_worker,
)


def _stat_card(title: str, value: str, accent: str = "#3b82f6") -> QFrame:
    """Create a small stat card widget."""
    card = QFrame()
    card.setStyleSheet(
        f"QFrame {{ background: #1c2e42; border: 1px solid #2a3f55;"
        f"  border-radius: 8px; border-top: 3px solid {accent}; }}"
    )
    layout = QVBoxLayout()
    layout.setContentsMargins(14, 10, 14, 10)
    lbl_title = QLabel(title)
    lbl_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;"
                            " text-transform: uppercase; letter-spacing: 0.5px;")
    lbl_value = QLabel(value)
    lbl_value.setObjectName("stat_value")
    lbl_value.setStyleSheet(f"color: {accent}; font-size: 22px; font-weight: 700;")
    layout.addWidget(lbl_title)
    layout.addWidget(lbl_value)
    card.setLayout(layout)
    return card


class Dashboard(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status = QLabel("Ready")
        self.status.setStyleSheet("color: #94a3b8; padding: 4px 0;")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)
        self._thread = None
        self._worker = None
        self._injury_thread = None
        self._injury_worker = None
        self._history_thread = None
        self._history_worker = None
        self._metrics_thread = None
        self._metrics_worker = None
        self._impact_thread = None
        self._impact_worker = None
        self._image_thread = None
        self._image_worker = None

        # ── Title ──
        title = QLabel("NBA Betting Analytics")
        title.setStyleSheet(
            "font-size: 24px; font-weight: 700; color: #e2e8f0;"
            " padding: 8px 0; letter-spacing: -0.5px;"
        )
        subtitle = QLabel("Sync data, manage injuries, and build models")
        subtitle.setStyleSheet("color: #64748b; font-size: 13px; padding-bottom: 8px;")

        # ── Stat cards ──
        self.teams_card = _stat_card("Teams", "--", "#3b82f6")
        self.players_card = _stat_card("Players", "--", "#10b981")
        self.games_card = _stat_card("Game Logs", "--", "#f59e0b")
        self.injured_card = _stat_card("Injured", "--", "#ef4444")

        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        for card in (self.teams_card, self.players_card, self.games_card, self.injured_card):
            cards_row.addWidget(card)
        cards_row.addStretch()

        # ── Buttons ──
        sync_button = QPushButton("  Sync Data")
        sync_button.setProperty("cssClass", "primary")
        sync_button.setToolTip("Fetch teams, players, and game logs from NBA API")
        sync_button.clicked.connect(self._sync_data)  # type: ignore[arg-type]
        self.sync_button = sync_button

        injury_button = QPushButton("  Sync Injuries")
        injury_button.setToolTip("Fetch current injury report")
        injury_button.clicked.connect(self._sync_injuries)  # type: ignore[arg-type]
        self.injury_button = injury_button

        history_button = QPushButton("  Build Injury History")
        history_button.setToolTip("Infer historical injuries from game logs for backtesting")
        history_button.clicked.connect(self._build_injury_history)  # type: ignore[arg-type]
        self.history_button = history_button

        metrics_button = QPushButton("  Sync Team Metrics")
        metrics_button.setToolTip("Fetch team advanced metrics (four factors, clutch, hustle)")
        metrics_button.clicked.connect(self._sync_team_metrics)  # type: ignore[arg-type]
        self.metrics_button = metrics_button

        impact_button = QPushButton("  Sync Player Impact")
        impact_button.setToolTip("Fetch player on/off and estimated impact metrics")
        impact_button.clicked.connect(self._sync_player_impact)  # type: ignore[arg-type]
        self.impact_button = impact_button

        image_button = QPushButton("  Sync Images")
        image_button.setToolTip("Download team logos and player headshots (runs in background)")
        image_button.clicked.connect(self._sync_images)  # type: ignore[arg-type]
        self.image_button = image_button

        stop_button = QPushButton("  Stop Sync")
        stop_button.setToolTip("Stop the current sync after the current step completes")
        stop_button.setStyleSheet(
            "QPushButton { background: #ef4444; color: white; font-weight: 600; }"
            "QPushButton:hover { background: #dc2626; }"
            "QPushButton:disabled { background: #555; color: #888; }"
        )
        stop_button.clicked.connect(self._stop_sync)  # type: ignore[arg-type]
        stop_button.setVisible(False)
        self.stop_button = stop_button

        force_check = QCheckBox("Force Recheck")
        force_check.setToolTip(
            "Bypass all freshness caches — re-fetch everything from the NBA API "
            "even if data was synced recently.  Also clears the schedule cache."
        )
        force_check.setStyleSheet(
            "QCheckBox { color: #f59e0b; font-weight: 600; font-size: 12px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; }"
        )
        self.force_check = force_check

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addWidget(sync_button)
        button_row.addWidget(injury_button)
        button_row.addWidget(history_button)
        button_row.addWidget(metrics_button)
        button_row.addWidget(impact_button)
        button_row.addWidget(image_button)
        button_row.addWidget(stop_button)
        button_row.addWidget(force_check)
        button_row.addStretch()

        # ── Log header ──
        log_header = QLabel("Activity Log")
        log_header.setStyleSheet(
            "color: #94a3b8; font-size: 11px; font-weight: 600;"
            " text-transform: uppercase; letter-spacing: 0.5px; padding-top: 6px;"
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(cards_row)
        layout.addSpacing(4)
        layout.addLayout(button_row)
        layout.addWidget(self.status)
        layout.addWidget(log_header)
        layout.addWidget(self.log, stretch=1)
        self.setLayout(layout)

        # Populate stat cards on init
        self._update_stat_cards()

    def _sync_data(self) -> None:
        if self._thread:
            return
        force = self.force_check.isChecked()
        self._set_syncing(True)
        self.sync_button.setEnabled(False)
        self.status.setText("Starting sync (force)..." if force else "Starting sync...")
        self.log.clear()
        if force:
            self._log_warning("Force recheck — bypassing all freshness caches")
            self._clear_caches()
        self._thread, self._worker = start_sync_worker(
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
            on_cancelled=self._on_cancelled,
            force=force,
        )
        self._thread.start()

    def _on_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)

    def _on_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self._log_success(msg)
        self.sync_button.setEnabled(True)
        self._set_syncing(False)
        self._update_stat_cards()
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    def _on_cancelled(self, msg: str) -> None:
        self.status.setText(msg)
        self._log_warning(msg)
        self.sync_button.setEnabled(True)
        self._set_syncing(False)
        self._update_stat_cards()
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    def _on_error(self, msg: str) -> None:
        self.status.setText(f"Sync failed: {msg}")
        self._log_error(msg)
        self.sync_button.setEnabled(True)
        self._set_syncing(False)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    def _sync_injuries(self) -> None:
        if self._injury_thread:
            return
        self.injury_button.setEnabled(False)
        self.status.setText("Syncing injuries...")
        self.log.append("--- Injury Sync Started ---")
        self._injury_thread, self._injury_worker = start_injury_sync_worker(
            on_progress=self._on_injury_progress,
            on_finished=self._on_injury_finished,
            on_error=self._on_injury_error,
        )
        self._injury_thread.start()

    def _on_injury_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)

    def _on_injury_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self._log_success(msg)
        self._log_success("--- Injury Sync Complete ---")
        self.injury_button.setEnabled(True)
        self._update_stat_cards()
        if self._injury_thread:
            self._injury_thread.quit()
            self._injury_thread.wait()
        self._injury_thread = None
        self._injury_worker = None

    def _on_injury_error(self, msg: str) -> None:
        self.status.setText(f"Injury sync failed: {msg}")
        self._log_error(msg)
        self.injury_button.setEnabled(True)
        if self._injury_thread:
            self._injury_thread.quit()
            self._injury_thread.wait()
        self._injury_thread = None
        self._injury_worker = None

    def _build_injury_history(self) -> None:
        if self._history_thread:
            return
        self.history_button.setEnabled(False)
        self.status.setText("Building injury history from game logs...")
        self.log.append("--- Building Injury History ---")
        self._history_thread, self._history_worker = start_injury_history_worker(
            on_progress=self._on_history_progress,
            on_finished=self._on_history_finished,
            on_error=self._on_history_error,
        )
        self._history_thread.start()

    def _on_history_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)

    def _on_history_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self._log_success(msg)
        self._log_success("--- Injury History Built ---")
        self.history_button.setEnabled(True)
        if self._history_thread:
            self._history_thread.quit()
            self._history_thread.wait()
        self._history_thread = None
        self._history_worker = None

    def _on_history_error(self, msg: str) -> None:
        self.status.setText(f"Injury history build failed: {msg}")
        self._log_error(msg)
        self.history_button.setEnabled(True)
        if self._history_thread:
            self._history_thread.quit()
            self._history_thread.wait()
        self._history_thread = None
        self._history_worker = None

    # ── Team metrics sync ──

    def _sync_team_metrics(self) -> None:
        if self._metrics_thread:
            return
        force = self.force_check.isChecked()
        self._set_syncing(True)
        self.metrics_button.setEnabled(False)
        self.status.setText("Syncing team metrics...")
        self.log.append("--- Team Metrics Sync Started ---")
        if force:
            self._log_warning("Force recheck — bypassing freshness")
        self._metrics_thread, self._metrics_worker = start_team_metrics_worker(
            on_progress=self._on_metrics_progress,
            on_finished=self._on_metrics_finished,
            on_error=self._on_metrics_error,
            on_cancelled=self._on_metrics_cancelled,
            force=force,
        )
        self._metrics_thread.start()

    def _on_metrics_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)

    def _on_metrics_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self._log_success(msg)
        self._log_success("--- Team Metrics Sync Complete ---")
        self.metrics_button.setEnabled(True)
        self._set_syncing(False)
        if self._metrics_thread:
            self._metrics_thread.quit()
            self._metrics_thread.wait()
        self._metrics_thread = None
        self._metrics_worker = None

    def _on_metrics_cancelled(self, msg: str) -> None:
        self.status.setText(msg)
        self._log_warning(msg)
        self.metrics_button.setEnabled(True)
        self._set_syncing(False)
        if self._metrics_thread:
            self._metrics_thread.quit()
            self._metrics_thread.wait()
        self._metrics_thread = None
        self._metrics_worker = None

    def _on_metrics_error(self, msg: str) -> None:
        self.status.setText(f"Team metrics sync failed: {msg}")
        self._log_error(msg)
        self.metrics_button.setEnabled(True)
        self._set_syncing(False)
        if self._metrics_thread:
            self._metrics_thread.quit()
            self._metrics_thread.wait()
        self._metrics_thread = None
        self._metrics_worker = None

    # ── Player impact sync ──

    def _sync_player_impact(self) -> None:
        if self._impact_thread:
            return
        force = self.force_check.isChecked()
        self._set_syncing(True)
        self.impact_button.setEnabled(False)
        self.status.setText("Syncing player impact...")
        self.log.append("--- Player Impact Sync Started ---")
        if force:
            self._log_warning("Force recheck — bypassing freshness")
        self._impact_thread, self._impact_worker = start_player_impact_worker(
            on_progress=self._on_impact_progress,
            on_finished=self._on_impact_finished,
            on_error=self._on_impact_error,
            on_cancelled=self._on_impact_cancelled,
            force=force,
        )
        self._impact_thread.start()

    def _on_impact_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)

    def _on_impact_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self._log_success(msg)
        self._log_success("--- Player Impact Sync Complete ---")
        self.impact_button.setEnabled(True)
        self._set_syncing(False)
        if self._impact_thread:
            self._impact_thread.quit()
            self._impact_thread.wait()
        self._impact_thread = None
        self._impact_worker = None

    def _on_impact_cancelled(self, msg: str) -> None:
        self.status.setText(msg)
        self._log_warning(msg)
        self.impact_button.setEnabled(True)
        self._set_syncing(False)
        if self._impact_thread:
            self._impact_thread.quit()
            self._impact_thread.wait()
        self._impact_thread = None
        self._impact_worker = None

    def _on_impact_error(self, msg: str) -> None:
        self.status.setText(f"Player impact sync failed: {msg}")
        self._log_error(msg)
        self.impact_button.setEnabled(True)
        self._set_syncing(False)
        if self._impact_thread:
            self._impact_thread.quit()
            self._impact_thread.wait()
        self._impact_thread = None
        self._impact_worker = None

    # ── Image sync ──

    def _sync_images(self) -> None:
        if self._image_thread:
            return
        self.image_button.setEnabled(False)
        self.status.setText("Downloading images…")
        self.log.append("--- Image Sync Started ---")
        self._image_thread, self._image_worker = start_image_sync_worker(
            on_progress=self._on_image_progress,
            on_finished=self._on_image_finished,
            on_error=self._on_image_error,
        )
        self._image_thread.start()

    def _on_image_progress(self, msg: str) -> None:
        self.status.setText(msg)
        self.log.append(msg)

    def _on_image_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self._log_success(msg)
        self._log_success("--- Image Sync Complete ---")
        self.image_button.setEnabled(True)
        if self._image_thread:
            self._image_thread.quit()
            self._image_thread.wait()
        self._image_thread = None
        self._image_worker = None

    def _on_image_error(self, msg: str) -> None:
        self.status.setText(f"Image sync failed: {msg}")
        self._log_error(msg)
        self.image_button.setEnabled(True)
        if self._image_thread:
            self._image_thread.quit()
            self._image_thread.wait()
        self._image_thread = None
        self._image_worker = None

    # ── Stop / cancel ──

    def _stop_sync(self) -> None:
        """Request cancellation of whatever sync is currently running."""
        self.stop_button.setEnabled(False)
        self.stop_button.setText("  Stopping...")
        self.status.setText("Cancelling after current step...")
        # Signal whichever worker is active
        for worker in (self._worker, self._metrics_worker, self._impact_worker):
            if worker and hasattr(worker, "request_cancel"):
                worker.request_cancel()

    def _set_syncing(self, active: bool) -> None:
        """Show/hide the stop button based on whether a sync is running."""
        self.stop_button.setVisible(active)
        self.stop_button.setEnabled(active)
        if active:
            self.stop_button.setText("  Stop Sync")

    # ── helpers ──

    def _clear_caches(self) -> None:
        """Flush in-memory caches so the next sync re-fetches everything."""
        try:
            from src.data.sync_service import clear_schedule_cache
            clear_schedule_cache()
        except Exception:
            pass

    def _log_info(self, msg: str) -> None:
        self.log.append(f'<span style="color:#94a3b8;">{msg}</span>')

    def _log_success(self, msg: str) -> None:
        self.log.append(f'<span style="color:#10b981;">{msg}</span>')

    def _log_warning(self, msg: str) -> None:
        self.log.append(f'<span style="color:#f59e0b;">{msg}</span>')

    def _log_error(self, msg: str) -> None:
        self.log.append(f'<span style="color:#ef4444;">ERROR: {msg}</span>')

    def _update_stat_cards(self) -> None:
        """Refresh the stat-card numbers from the database."""
        try:
            from src.database.db import get_conn

            with get_conn() as conn:
                teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
                players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
                logs = conn.execute("SELECT COUNT(*) FROM player_stats").fetchone()[0]
                injured = conn.execute(
                    "SELECT COUNT(*) FROM players WHERE is_injured = 1"
                ).fetchone()[0]

            self.teams_card.findChild(QLabel, "stat_value").setText(str(teams))
            self.players_card.findChild(QLabel, "stat_value").setText(str(players))
            self.games_card.findChild(QLabel, "stat_value").setText(f"{logs:,}")
            self.injured_card.findChild(QLabel, "stat_value").setText(str(injured))
        except Exception:
            pass  # DB may not exist yet
