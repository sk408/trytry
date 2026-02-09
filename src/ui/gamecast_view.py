"""Desktop Gamecast tab – live game detail with in-game predictions.

Shows game selector, live score, blended prediction panel,
odds, box score, and play-by-play feed.  Auto-refreshes via QTimer.

All network I/O and heavy computation runs in a background thread to
keep the UI responsive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, QTimer, QObject, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.data.gamecast import (
    BoxScore,
    GameInfo,
    GameOdds,
    PlayEvent,
    compute_bonus_status,
    get_box_score,
    get_game_odds,
    get_live_games,
    get_play_by_play,
)
from src.data.image_cache import get_team_logo_pixmap
from src.analytics.live_prediction import LivePrediction, live_predict
from src.database.db import get_conn


# ── background fetch result ───────────────────────────────────────────

@dataclass
class _PollResult:
    """Data bundle returned by the background poll worker."""
    game: Optional[GameInfo] = None
    odds: Optional[GameOdds] = None
    box: Optional[BoxScore] = None
    plays: List[PlayEvent] = field(default_factory=list)
    all_plays: List[PlayEvent] = field(default_factory=list)
    prediction: Optional[LivePrediction] = None
    prediction_error: str = ""
    # For post-game analysis
    postgame_data: Optional[dict] = None


class _PollWorker(QObject):
    """Fetches all gamecast data in a background thread."""
    finished = Signal(object)  # _PollResult

    def __init__(
        self,
        game: GameInfo,
        last_play_id: str = "",
    ):
        super().__init__()
        self.game = game
        self.last_play_id = last_play_id

    def run(self) -> None:
        result = _PollResult(game=self.game)
        game = self.game

        # 1. Odds
        try:
            result.odds = get_game_odds(game.game_id)
        except Exception:
            pass

        # 2. Box score
        try:
            result.box = get_box_score(game.game_id)
        except Exception:
            pass

        # 3. Play-by-play (incremental)
        try:
            result.plays = get_play_by_play(game.game_id, self.last_play_id)
        except Exception:
            pass

        # 4. All plays for bonus (only for live games)
        if game.status == "in_progress" and result.plays:
            try:
                result.all_plays = get_play_by_play(game.game_id, "")
            except Exception:
                pass

        # 5. Live prediction (DB queries + math)
        home_id = _abbr_to_team_id(game.home_abbr)
        away_id = _abbr_to_team_id(game.away_abbr)

        if home_id and away_id:
            game_date = game.start_time[:10] if game.start_time else ""

            if game.status == "final":
                # Post-game analysis data
                try:
                    from src.analytics.live_prediction import _cached_pregame
                    pg_home, pg_away, pg_total, pg_spread = _cached_pregame(
                        home_id, away_id, game_date,
                    )
                    result.postgame_data = {
                        "pg_home": pg_home, "pg_away": pg_away,
                        "pg_total": pg_total, "pg_spread": pg_spread,
                        "home_id": home_id, "away_id": away_id,
                    }
                except Exception:
                    result.postgame_data = {
                        "pg_home": 110.0, "pg_away": 110.0,
                        "pg_total": 220.0, "pg_spread": 0.0,
                        "home_id": home_id, "away_id": away_id,
                    }
            else:
                try:
                    result.prediction = live_predict(
                        home_team_id=home_id,
                        away_team_id=away_id,
                        home_score=game.home_score,
                        away_score=game.away_score,
                        period=game.period,
                        clock=game.clock,
                        game_id=game.game_id,
                        game_date_str=game_date,
                    )
                except Exception as exc:
                    result.prediction_error = str(exc)
        else:
            result.prediction_error = "teams not in DB"

        self.finished.emit(result)


class _GamesListWorker(QObject):
    """Fetches the ESPN scoreboard game list in a background thread."""
    finished = Signal(list)  # List[GameInfo]

    def run(self) -> None:
        try:
            games = get_live_games()
        except Exception:
            games = []
        self.finished.emit(games)


class _ScoreboardRefreshWorker(QObject):
    """Fetches updated scoreboard + full game detail in one background pass."""
    finished = Signal(object)  # _PollResult

    def __init__(self, game_id: str, last_play_id: str):
        super().__init__()
        self.game_id = game_id
        self.last_play_id = last_play_id

    def run(self) -> None:
        # 1. Get updated scoreboard to find the latest game state
        try:
            games = get_live_games()
        except Exception:
            self.finished.emit(_PollResult())
            return

        updated = next(
            (g for g in games if g.game_id == self.game_id), None,
        )
        if not updated:
            self.finished.emit(_PollResult())
            return

        # 2. Fetch all details in this same thread
        result = _PollResult(game=updated)

        # Odds
        try:
            result.odds = get_game_odds(updated.game_id)
        except Exception:
            pass

        # Box score
        try:
            result.box = get_box_score(updated.game_id)
        except Exception:
            pass

        # Play-by-play
        try:
            result.plays = get_play_by_play(updated.game_id, self.last_play_id)
        except Exception:
            pass

        if updated.status == "in_progress" and result.plays:
            try:
                result.all_plays = get_play_by_play(updated.game_id, "")
            except Exception:
                pass

        # Prediction
        home_id = _abbr_to_team_id(updated.home_abbr)
        away_id = _abbr_to_team_id(updated.away_abbr)

        if home_id and away_id:
            game_date = updated.start_time[:10] if updated.start_time else ""

            if updated.status == "final":
                try:
                    from src.analytics.live_prediction import _cached_pregame
                    pg_home, pg_away, pg_total, pg_spread = _cached_pregame(
                        home_id, away_id, game_date,
                    )
                    result.postgame_data = {
                        "pg_home": pg_home, "pg_away": pg_away,
                        "pg_total": pg_total, "pg_spread": pg_spread,
                        "home_id": home_id, "away_id": away_id,
                    }
                except Exception:
                    result.postgame_data = {
                        "pg_home": 110.0, "pg_away": 110.0,
                        "pg_total": 220.0, "pg_spread": 0.0,
                        "home_id": home_id, "away_id": away_id,
                    }
            else:
                try:
                    result.prediction = live_predict(
                        home_team_id=home_id,
                        away_team_id=away_id,
                        home_score=updated.home_score,
                        away_score=updated.away_score,
                        period=updated.period,
                        clock=updated.clock,
                        game_id=updated.game_id,
                        game_date_str=game_date,
                    )
                except Exception as exc:
                    result.prediction_error = str(exc)
        else:
            result.prediction_error = "teams not in DB"

        self.finished.emit(result)


# ── helpers ──────────────────────────────────────────────────────────

# ESPN uses shorter abbreviations for a few teams.  Map to the NBA-API
# values stored in our ``teams`` table.
_ESPN_TO_NBA_ABBR = {
    "GS": "GSW",
    "SA": "SAS",
    "NY": "NYK",
    "NO": "NOP",
    "UTAH": "UTA",
    "WSH": "WAS",
}


_ABBR_CACHE: dict[str, Optional[int]] = {}


def _abbr_to_team_id(abbr: str) -> Optional[int]:
    """Resolve an ESPN abbreviation to our internal team_id.

    Handles known ESPN ↔ NBA-API abbreviation mismatches (e.g. GS → GSW).
    Results are cached in-memory so the DB is only hit once per abbreviation.
    """
    if abbr in _ABBR_CACHE:
        return _ABBR_CACHE[abbr]
    canonical = _ESPN_TO_NBA_ABBR.get(abbr, abbr)
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT team_id FROM teams WHERE abbreviation = ?", (canonical,)
            ).fetchone()
        result = int(row[0]) if row else None
    except Exception:
        result = None
    _ABBR_CACHE[abbr] = result
    return result


def _bold(text: str) -> str:
    return f"<b>{text}</b>"


# ── main widget ──────────────────────────────────────────────────────

class GamecastView(QWidget):
    """Full gamecast tab for the desktop application."""

    _POLL_LIVE_MS = 15_000   # 15 s for in-progress games
    _POLL_IDLE_MS = 60_000   # 60 s otherwise

    def __init__(self) -> None:
        super().__init__()

        # ── state ──
        self._current_game: Optional[GameInfo] = None
        self._last_play_id = ""
        self._poll_thread: Optional[QThread] = None
        self._poll_worker: Optional[_PollWorker] = None

        # ── game selector ──
        self.game_combo = QComboBox()
        self.game_combo.setMinimumWidth(360)
        self.game_combo.currentIndexChanged.connect(self._on_game_selected)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_games)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Select Game:"))
        selector_row.addWidget(self.game_combo, stretch=1)
        selector_row.addWidget(self.refresh_btn)

        # ── score header ──
        self.away_logo_lbl = QLabel()
        self.away_logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.away_logo_lbl.setFixedSize(56, 56)

        self.away_name_lbl = QLabel("")
        self.away_name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.away_score_lbl = QLabel("0")
        self.away_score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.away_score_lbl.setStyleSheet("font-size: 36px; font-weight: 800;")

        self.status_lbl = QLabel("")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.home_name_lbl = QLabel("")
        self.home_name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.home_score_lbl = QLabel("0")
        self.home_score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.home_score_lbl.setStyleSheet("font-size: 36px; font-weight: 800;")

        self.home_logo_lbl = QLabel()
        self.home_logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.home_logo_lbl.setFixedSize(56, 56)

        self.quarter_scores_lbl = QLabel("")
        self.quarter_scores_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quarter_scores_lbl.setStyleSheet("color: gray; font-size: 11px;")

        # Bonus / team-foul strip
        self.bonus_lbl = QLabel("")
        self.bonus_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bonus_lbl.setStyleSheet("font-size: 11px;")

        score_layout = QHBoxLayout()
        score_layout.addWidget(self.away_logo_lbl)
        for w in (self.away_name_lbl, self.away_score_lbl,
                  self.status_lbl,
                  self.home_score_lbl, self.home_name_lbl):
            score_layout.addWidget(w, stretch=1)
        score_layout.addWidget(self.home_logo_lbl)

        score_box = QGroupBox("Score")
        score_inner = QVBoxLayout()
        score_inner.addLayout(score_layout)
        score_inner.addWidget(self.quarter_scores_lbl)
        score_inner.addWidget(self.bonus_lbl)
        score_box.setLayout(score_inner)

        # ── live prediction panel ──
        self.pregame_lbl = QLabel("Pre-game: --")
        self.live_proj_lbl = QLabel("Live Projection: --")
        self.live_proj_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.pace_lbl = QLabel("Pace: --")
        self.quarter_hist_lbl = QLabel("")
        self.signal_lbl = QLabel("")
        self.blend_bar = QProgressBar()
        self.blend_bar.setRange(0, 100)
        self.blend_bar.setFormat("Pre-game %v% ← blend → Pace")
        self.blend_bar.setTextVisible(True)
        self.blend_bar.setMaximumHeight(18)

        pred_box = QGroupBox("Live Prediction")
        pred_layout = QVBoxLayout()
        pred_layout.addWidget(self.pregame_lbl)
        pred_layout.addWidget(self.live_proj_lbl)
        pred_layout.addWidget(self.pace_lbl)
        pred_layout.addWidget(self.quarter_hist_lbl)
        pred_layout.addWidget(self.signal_lbl)
        pred_layout.addWidget(self.blend_bar)
        pred_box.setLayout(pred_layout)

        # ── odds panel ──
        self.spread_lbl = QLabel("Spread: --")
        self.ou_lbl = QLabel("O/U: --")
        self.ml_lbl = QLabel("ML: --")
        self.win_pct_lbl = QLabel("Win%: --")

        odds_box = QGroupBox("Live Odds (ESPN)")
        odds_layout = QVBoxLayout()
        for w in (self.spread_lbl, self.ou_lbl, self.ml_lbl, self.win_pct_lbl):
            odds_layout.addWidget(w)
        odds_box.setLayout(odds_layout)

        # ── top row: prediction + odds side by side ──
        top_row = QHBoxLayout()
        top_row.addWidget(pred_box, stretch=2)
        top_row.addWidget(odds_box, stretch=1)

        # ── box score tables ──
        self.home_box_table = QTableWidget()
        self.away_box_table = QTableWidget()

        home_box_group = QGroupBox("Home Box Score")
        hbl = QVBoxLayout()
        hbl.addWidget(self.home_box_table)
        home_box_group.setLayout(hbl)

        away_box_group = QGroupBox("Away Box Score")
        abl = QVBoxLayout()
        abl.addWidget(self.away_box_table)
        away_box_group.setLayout(abl)

        box_splitter = QSplitter(Qt.Orientation.Horizontal)
        box_splitter.addWidget(away_box_group)
        box_splitter.addWidget(home_box_group)

        # ── play-by-play ──
        self.play_list = QListWidget()
        self.play_list.setAlternatingRowColors(True)

        play_box = QGroupBox("Play-by-Play (newest first)")
        play_layout = QVBoxLayout()
        play_layout.addWidget(self.play_list)
        play_box.setLayout(play_layout)

        # ── assemble into scrollable layout ──
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.addLayout(selector_row)
        inner_layout.addWidget(score_box)
        inner_layout.addLayout(top_row)
        inner_layout.addWidget(box_splitter)
        inner_layout.addWidget(play_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── auto-refresh timer ──
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(self._POLL_IDLE_MS)

        # initial load
        self._refresh_games()

    # ────────────────────────────────────────────────────────────────
    # Game list management
    # ────────────────────────────────────────────────────────────────

    def _refresh_games(self) -> None:
        """Reload game list from ESPN scoreboard in a background thread."""
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("Loading…")

        self._games_thread = QThread()
        self._games_worker = _GamesListWorker()
        self._games_worker.moveToThread(self._games_thread)
        self._games_thread.started.connect(self._games_worker.run)
        self._games_worker.finished.connect(self._on_games_loaded)
        self._games_worker.finished.connect(self._games_thread.quit)
        self._games_thread.finished.connect(self._cleanup_games_thread)
        self._games_thread.start()

    def _on_games_loaded(self, games: list) -> None:
        """Populate game combo from background-fetched list (main thread)."""
        saved_game_id = self._current_game.game_id if self._current_game else None

        self.game_combo.blockSignals(True)
        self.game_combo.clear()
        self.game_combo.addItem("-- Select a game --", None)

        _status_order = {"in_progress": 0, "scheduled": 1, "final": 2}
        games.sort(key=lambda g: _status_order.get(g.status, 9))

        for g in games:
            if g.status == "in_progress":
                tag = f"LIVE Q{g.period} {g.clock} ({g.away_score}-{g.home_score})"
            elif g.status == "final":
                tag = f"FINAL ({g.away_score}-{g.home_score})"
            else:
                tag = g.start_time[:16] if g.start_time else "Upcoming"
            label = f"{g.away_abbr} @ {g.home_abbr} — {tag}"
            self.game_combo.addItem(label, g)

        self.game_combo.blockSignals(False)

        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("Refresh")

        # Re-select previously selected game
        if saved_game_id:
            for i in range(self.game_combo.count()):
                data = self.game_combo.itemData(i)
                if data and getattr(data, "game_id", None) == saved_game_id:
                    self.game_combo.setCurrentIndex(i)
                    self._load_game_detail(data)
                    return

    def _cleanup_games_thread(self) -> None:
        if hasattr(self, "_games_worker") and self._games_worker:
            self._games_worker.deleteLater()
            self._games_worker = None
        if hasattr(self, "_games_thread") and self._games_thread:
            self._games_thread.deleteLater()
            self._games_thread = None

    def _on_game_selected(self, index: int) -> None:
        if index <= 0:
            self._current_game = None
            self._clear_detail()
            self._timer.setInterval(self._POLL_IDLE_MS)
            return
        game = self.game_combo.itemData(index)
        if not game:
            return
        self._current_game = game
        self._last_play_id = ""
        interval = self._POLL_LIVE_MS if game.status == "in_progress" else self._POLL_IDLE_MS
        self._timer.setInterval(interval)
        self._load_game_detail(game)

    # ────────────────────────────────────────────────────────────────
    # Detail loading (background thread)
    # ────────────────────────────────────────────────────────────────

    def _load_game_detail(self, game: GameInfo) -> None:
        self._current_game = game
        # Update score immediately (no I/O, just labels)
        self._update_score(game)
        # Kick off background fetch for everything else
        self._start_poll_worker(game)

    def _start_poll_worker(self, game: GameInfo) -> None:
        """Launch a background thread to fetch odds/box/plays/prediction."""
        # Don't stack up workers
        if self._poll_thread is not None and self._poll_thread.isRunning():
            return

        self._poll_thread = QThread()
        self._poll_worker = _PollWorker(game, self._last_play_id)
        self._poll_worker.moveToThread(self._poll_thread)
        self._poll_thread.started.connect(self._poll_worker.run)
        self._poll_worker.finished.connect(self._on_poll_result)
        self._poll_worker.finished.connect(self._poll_thread.quit)
        self._poll_thread.finished.connect(self._cleanup_poll)
        self._poll_thread.start()

    def _cleanup_poll(self) -> None:
        if self._poll_worker:
            self._poll_worker.deleteLater()
            self._poll_worker = None
        if self._poll_thread:
            self._poll_thread.deleteLater()
            self._poll_thread = None

    def _on_poll_result(self, result: _PollResult) -> None:
        """Apply fetched data to the UI (runs on main thread)."""
        game = result.game
        if not game:
            return
        # Update score from the freshest game data
        self._update_score(game)
        # Prediction
        if result.prediction:
            self._render_prediction(result.prediction, game)
        elif result.postgame_data:
            self._render_final_analysis_data(result.postgame_data, game)
        elif result.prediction_error:
            self.pregame_lbl.setText(f"Pre-game: {result.prediction_error}")
        # Odds
        self._apply_odds(result.odds, game)
        # Box score
        self._fill_box_table(self.home_box_table, result.box.home_players if result.box else [])
        self._fill_box_table(self.away_box_table, result.box.away_players if result.box else [])
        # Plays
        self._apply_plays(result.plays, result.all_plays, game)
        # Timer interval
        if game.status == "in_progress":
            self._timer.setInterval(self._POLL_LIVE_MS)
        else:
            self._timer.setInterval(self._POLL_IDLE_MS)

    def _update_score(self, game: GameInfo) -> None:
        self.away_name_lbl.setText(f"{game.away_abbr}\n{game.away_team}")
        self.home_name_lbl.setText(f"{game.home_abbr}\n{game.home_team}")

        # Team logos
        away_tid = _abbr_to_team_id(game.away_abbr)
        home_tid = _abbr_to_team_id(game.home_abbr)
        if away_tid:
            self.away_logo_lbl.setPixmap(get_team_logo_pixmap(away_tid, 52))
        if home_tid:
            self.home_logo_lbl.setPixmap(get_team_logo_pixmap(home_tid, 52))
        self.away_score_lbl.setText(str(game.away_score))
        self.home_score_lbl.setText(str(game.home_score))

        if game.status == "in_progress":
            self.status_lbl.setText(f"<b style='color:#e74c3c;'>LIVE</b><br>Q{game.period} {game.clock}")
        elif game.status == "final":
            self.status_lbl.setText("<b>FINAL</b>")
        else:
            self.status_lbl.setText("Upcoming")

        # Quarter scores line
        if game.away_linescores or game.home_linescores:
            parts = []
            for i, (a, h) in enumerate(
                zip(game.away_linescores, game.home_linescores)
            ):
                qlabel = f"Q{i+1}" if i < 4 else f"OT{i-3}"
                parts.append(f"{qlabel}: {a}-{h}")
            self.quarter_scores_lbl.setText("  |  ".join(parts))
        else:
            self.quarter_scores_lbl.setText("")

    def _render_final_analysis_data(self, pgdata: dict, game: GameInfo) -> None:
        """Show post-game comparison using pre-fetched data (no I/O)."""
        ha = game.home_abbr
        aa = game.away_abbr
        pg_home = pgdata["pg_home"]
        pg_away = pgdata["pg_away"]
        pg_total = pgdata["pg_total"]
        pg_spread = pgdata["pg_spread"]

        actual_spread = game.home_score - game.away_score
        actual_total = game.home_score + game.away_score
        spread_err = abs(pg_spread - actual_spread)
        total_err = abs(pg_total - actual_total)
        winner_ok = (pg_spread > 0 and actual_spread > 0) or (pg_spread < 0 and actual_spread < 0)

        self.pregame_lbl.setText(
            f"Pre-game model: {ha} {pg_spread:+.1f}  |  Total {pg_total:.1f}  "
            f"({ha} {pg_home:.0f} – {aa} {pg_away:.0f})"
        )

        winner_icon = "✓" if winner_ok else "✗"
        self.live_proj_lbl.setText(
            f"Final Analysis: {ha} {actual_spread:+d}  |  Total {actual_total}  "
            f"| Winner {winner_icon}  |  Spread Err {spread_err:.1f}  |  Total Err {total_err:.1f}"
        )
        if winner_ok:
            self.live_proj_lbl.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            self.live_proj_lbl.setStyleSheet("color: #e74c3c; font-weight: bold;")

        self.pace_lbl.setText("")
        self.quarter_hist_lbl.setText("")

        self.signal_lbl.setText(
            f"Spread error: {spread_err:.1f} pts  |  Total error: {total_err:.1f} pts"
        )
        self.signal_lbl.setStyleSheet("color: gray;")

        self.blend_bar.setValue(100)
        self.blend_bar.setFormat("Game Complete")

    def _render_prediction(self, lp: LivePrediction, game: GameInfo) -> None:
        ha = game.home_abbr
        aa = game.away_abbr

        self.pregame_lbl.setText(
            f"Pre-game model: {ha} {lp.pregame_spread:+.1f}  |  Total {lp.pregame_total:.1f}  "
            f"({ha} {lp.pregame_home:.0f} – {aa} {lp.pregame_away:.0f})"
        )

        self.live_proj_lbl.setText(
            f"Live Projection: {ha} {lp.projected_spread:+.1f}  |  Total {lp.projected_total:.1f}  "
            f"({ha} {lp.projected_home_score:.0f} – {aa} {lp.projected_away_score:.0f})"
        )
        self.live_proj_lbl.setStyleSheet("")  # reset any prior final-analysis styling

        self.pace_lbl.setText(
            f"At current pace: Total {lp.pace_total:.1f}  "
            f"({ha} {lp.pace_home:.0f} – {aa} {lp.pace_away:.0f})"
        )

        # Quarter history
        qh_parts = []
        if lp.quarter_analysis_home:
            qh_parts.append(f"{game.home_abbr}: {lp.quarter_analysis_home}")
        if lp.quarter_analysis_away:
            qh_parts.append(f"{game.away_abbr}: {lp.quarter_analysis_away}")
        self.quarter_hist_lbl.setText("\n".join(qh_parts) if qh_parts else "")

        # Signals
        signal_parts = []
        if lp.over_under_signal:
            signal_parts.append(lp.over_under_signal)
        if lp.spread_signal:
            signal_parts.append(lp.spread_signal)
        self.signal_lbl.setText("  |  ".join(signal_parts))
        if "OVER" in lp.over_under_signal:
            self.signal_lbl.setStyleSheet("color: #27ae60; font-weight: bold;")
        elif "UNDER" in lp.over_under_signal:
            self.signal_lbl.setStyleSheet("color: #e74c3c; font-weight: bold;")
        else:
            self.signal_lbl.setStyleSheet("color: gray;")

        # Blend bar: 0 = all pre-game, 100 = all pace
        pace_pct = int(lp.blend_weights.get("pace", 0) * 100)
        self.blend_bar.setValue(pace_pct)
        pg_pct = int(lp.blend_weights.get("pregame", 0) * 100)
        qh_pct = int(lp.blend_weights.get("quarter", 0) * 100)
        self.blend_bar.setFormat(
            f"Pre-game {pg_pct}%  |  Pace {pace_pct}%  |  Quarter-Hist {qh_pct}%"
        )

    def _apply_odds(self, odds: Optional[GameOdds], game: GameInfo) -> None:
        """Update odds labels from pre-fetched data (no I/O)."""
        if not odds:
            self.spread_lbl.setText("Spread: N/A")
            self.ou_lbl.setText("O/U: N/A")
            self.ml_lbl.setText("ML: N/A")
            self.win_pct_lbl.setText("Win%: N/A")
            return

        ha = game.home_abbr
        aa = game.away_abbr

        self.spread_lbl.setText(
            f"Spread: {ha} {odds.spread:+.1f} ({odds.spread_odds})"
            if odds.spread else "Spread: N/A"
        )
        self.ou_lbl.setText(
            f"O/U: {odds.over_under:.1f}  (O {odds.over_odds} / U {odds.under_odds})"
            if odds.over_under else "O/U: N/A"
        )
        self.ml_lbl.setText(f"ML: {aa} {odds.away_ml}  |  {ha} {odds.home_ml}")
        self.win_pct_lbl.setText(
            f"Win%: {aa} {odds.away_win_pct:.1f}%  |  {ha} {odds.home_win_pct:.1f}%"
        )

    # ── box score ──

    @staticmethod
    def _fill_box_table(table: QTableWidget, players) -> None:
        from PySide6.QtGui import QColor
        headers = ["Player", "MIN", "PTS", "REB", "AST", "FG", "3PT", "FT", "PF"]
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(players))
        for r, p in enumerate(players):
            for c, val in enumerate([
                f"{p.name} ({p.position})", p.minutes,
                str(p.points), str(p.rebounds), str(p.assists),
                p.fg, p.fg3, p.ft, str(p.fouls),
            ]):
                item = QTableWidgetItem(val)
                # Highlight high foul counts
                if c == 8:  # PF column
                    if p.fouls >= 5:
                        item.setForeground(QColor("#e74c3c"))
                    elif p.fouls >= 4:
                        item.setForeground(QColor("#f39c12"))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    # ── play-by-play ──

    def _apply_plays(
        self, plays: List[PlayEvent], all_plays: List[PlayEvent], game: GameInfo,
    ) -> None:
        """Update play-by-play list from pre-fetched data (no I/O)."""
        from PySide6.QtGui import QColor

        if not plays:
            return

        home_abbr = game.home_abbr
        away_abbr = game.away_abbr

        for play in reversed(plays):
            foul_tag = ""
            if play.is_foul:
                if play.flagrant_foul:
                    foul_tag = " [FLAGRANT]"
                elif play.technical_foul:
                    foul_tag = " [TECH]"
                elif play.shooting_foul:
                    foul_tag = " [FTs]"
                elif play.offensive_foul:
                    foul_tag = " [OFF FOUL]"
                else:
                    foul_tag = " [FOUL]"

            team_tag = f"[{play.team}]" if play.team else "     "
            text = (
                f"Q{play.period} {play.clock}  {team_tag:<6} "
                f"{play.text}{foul_tag}  "
                f"({away_abbr} {play.score_away} - {play.score_home} {home_abbr})"
            )
            item = QListWidgetItem(text)

            if play.is_foul:
                if play.flagrant_foul:
                    item.setForeground(QColor("#e74c3c"))
                elif play.technical_foul:
                    item.setForeground(QColor("#9b59b6"))
                else:
                    item.setForeground(QColor("#f39c12"))
            elif play.event_type and ("made shot" in play.event_type or "free throw" in play.event_type):
                item.setForeground(QColor("#27ae60"))
            elif play.team == home_abbr:
                item.setForeground(QColor("#3498db"))
            elif play.team == away_abbr:
                item.setForeground(QColor("#e67e22"))

            self.play_list.insertItem(0, item)

        self._last_play_id = plays[0].event_id

        # Bonus status from pre-fetched all_plays
        if all_plays and game.status == "in_progress":
            try:
                bonus = compute_bonus_status(
                    all_plays, home_abbr, away_abbr, game.period,
                )
                self._render_bonus(bonus, home_abbr, away_abbr)
            except Exception:
                pass

        while self.play_list.count() > 200:
            self.play_list.takeItem(self.play_list.count() - 1)

    # ── bonus rendering ──

    def _render_bonus(self, bonus: dict, home_abbr: str, away_abbr: str) -> None:
        """Update the bonus / team-foul strip below the score."""
        away_f = bonus.get("away_fouls_q", 0)
        home_f = bonus.get("home_fouls_q", 0)
        away_bonus = bonus.get("away_in_bonus", False)
        home_bonus = bonus.get("home_in_bonus", False)

        parts = []
        away_txt = f"{away_abbr} Fouls: {away_f}"
        if away_bonus:
            away_txt += "  <b style='color:#f39c12;'>BONUS</b>"
        parts.append(away_txt)

        home_txt = f"{home_abbr} Fouls: {home_f}"
        if home_bonus:
            home_txt += "  <b style='color:#f39c12;'>BONUS</b>"
        parts.append(home_txt)

        self.bonus_lbl.setText("  |  ".join(parts))

    # ── clear ──

    def _clear_detail(self) -> None:
        for lbl in (self.away_score_lbl, self.home_score_lbl):
            lbl.setText("0")
        for lbl in (self.away_name_lbl, self.home_name_lbl):
            lbl.setText("")
        self.status_lbl.setText("")
        self.quarter_scores_lbl.setText("")
        self.bonus_lbl.setText("")
        self.pregame_lbl.setText("Pre-game: --")
        self.live_proj_lbl.setText("Live Projection: --")
        self.pace_lbl.setText("Pace: --")
        self.quarter_hist_lbl.setText("")
        self.signal_lbl.setText("")
        self.blend_bar.setValue(0)
        self.spread_lbl.setText("Spread: --")
        self.ou_lbl.setText("O/U: --")
        self.ml_lbl.setText("ML: --")
        self.win_pct_lbl.setText("Win%: --")
        self.home_box_table.setRowCount(0)
        self.away_box_table.setRowCount(0)
        self.play_list.clear()
        self._last_play_id = ""

    # ── polling ──

    def _poll(self) -> None:
        """Called by QTimer – refresh the current game in a background thread."""
        if not self._current_game:
            return

        # If a background fetch is already running, skip this tick
        if self._poll_thread is not None and self._poll_thread.isRunning():
            return

        # Launch a lightweight thread to get the updated scoreboard entry
        # then fetch full game detail in the same thread
        self._refresh_thread = QThread()
        self._refresh_worker = _ScoreboardRefreshWorker(
            self._current_game.game_id, self._last_play_id,
        )
        self._refresh_worker.moveToThread(self._refresh_thread)
        self._refresh_thread.started.connect(self._refresh_worker.run)
        self._refresh_worker.finished.connect(self._on_scoreboard_refresh)
        self._refresh_worker.finished.connect(self._refresh_thread.quit)
        self._refresh_thread.finished.connect(self._cleanup_refresh)
        self._refresh_thread.start()

    def _on_scoreboard_refresh(self, result: _PollResult) -> None:
        """Handle scoreboard + detail refresh from background thread."""
        if result.game:
            self._current_game = result.game
            self._on_poll_result(result)

    def _cleanup_refresh(self) -> None:
        if hasattr(self, "_refresh_worker") and self._refresh_worker:
            self._refresh_worker.deleteLater()
            self._refresh_worker = None
        if hasattr(self, "_refresh_thread") and self._refresh_thread:
            self._refresh_thread.deleteLater()
            self._refresh_thread = None
