"""Gamecast tab — immersive live game view with scoreboard, court, plays, odds.

Architecture:
- All ESPN network calls run on background threads (never block UI)
- In-memory cache stores parsed game data per game_id with TTL
- Background preloader fetches all games after scoreboard loads
- Smart polling: live=15s, pre-game=60s, final=never (single load)
- Game switching is instant when data is cached
"""

import logging
import time
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QTabWidget,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QRunnable, QThreadPool
from PySide6.QtGui import QColor, QImage, QPixmap

from src.ui.widgets.scoreboard_widget import ScoreboardWidget
from src.ui.widgets.court_widget import CourtWidget
from src.ui.widgets.play_feed_widget import PlayFeedWidget
from src.ui.widgets.info_panel_widget import InfoPanelWidget
from src.ui.widgets.nba_colors import get_team_colors
from src.ui.widgets.image_utils import get_team_logo, get_player_photo

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# In-memory caches
# ──────────────────────────────────────────────────────────────

_espn_headshot_cache: Dict[tuple, QPixmap] = {}   # (url, size) → QPixmap

_CACHE_TTL_LIVE = 12        # seconds — live games refresh often
_CACHE_TTL_PRE = 55         # seconds — pre-game (odds may update)
_CACHE_TTL_FINAL = 86400    # seconds — final games almost never change


class _GameCache:
    """Thread-safe in-memory cache for parsed game data."""

    def __init__(self):
        self._data: Dict[str, dict] = {}       # game_id -> parsed data
        self._ts: Dict[str, float] = {}         # game_id -> timestamp
        self._state: Dict[str, str] = {}        # game_id -> "pre"/"in"/"post"

    def get(self, game_id: str) -> Optional[dict]:
        """Return cached data if still fresh, else None."""
        if game_id not in self._data:
            return None
        age = time.time() - self._ts.get(game_id, 0)
        ttl = self._ttl_for(game_id)
        if age > ttl:
            return None
        return self._data[game_id]

    def put(self, game_id: str, data: dict, state: str = ""):
        """Store parsed data with current timestamp."""
        self._data[game_id] = data
        self._ts[game_id] = time.time()
        if state:
            self._state[game_id] = state

    def is_final(self, game_id: str) -> bool:
        return self._state.get(game_id) == "post"

    def _ttl_for(self, game_id: str) -> float:
        s = self._state.get(game_id, "")
        if s == "post":
            return _CACHE_TTL_FINAL
        elif s == "in":
            return _CACHE_TTL_LIVE
        return _CACHE_TTL_PRE

    def clear(self):
        self._data.clear()
        self._ts.clear()
        self._state.clear()


_cache = _GameCache()

# ──────────────────────────────────────────────────────────────
# Background workers
# ──────────────────────────────────────────────────────────────


def _fetch_and_parse(game_id: str) -> dict:
    """Network call + parse — runs off main thread. Returns parsed dict."""
    from src.data.gamecast import fetch_espn_game_summary
    summary = fetch_espn_game_summary(game_id)

    pickcenter = summary.get("pickcenter", [])
    odds = {}
    if pickcenter:
        od = pickcenter[0]
        odds = {
            "spread": od.get("details", ""),
            "over_under": od.get("overUnder"),
            "home_moneyline": od.get("homeTeamOdds", {}).get("moneyLine"),
            "away_moneyline": od.get("awayTeamOdds", {}).get("moneyLine"),
            "provider": od.get("provider", {}).get("name", ""),
        }

    predictor = summary.get("predictor", {})
    home_win_pct = 50.0
    if predictor:
        home_win_pct = float(
            predictor.get("homeTeam", {}).get("gameProjection", 50.0)
        )

    # Detect game state from header
    header = summary.get("header", {})
    comps = header.get("competitions", [{}])
    comp = comps[0] if comps else {}
    status_state = comp.get("status", {}).get("type", {}).get("state", "")

    return {
        "summary": summary,
        "odds": odds,
        "boxscore": summary.get("boxscore", {}),
        "plays": summary.get("plays", []),
        "home_win_pct": home_win_pct,
        "status_state": status_state,
    }


class _GameFetchWorker(QObject):
    """Fetches one game's data on a background thread, with cache integration."""
    finished = Signal(str, dict)   # game_id, data
    error = Signal(str, str)       # game_id, message

    def __init__(self, game_id: str):
        super().__init__()
        self.game_id = game_id

    def run(self):
        try:
            data = _fetch_and_parse(self.game_id)
            state = data.get("status_state", "")
            _cache.put(self.game_id, data, state)
            self.finished.emit(self.game_id, data)
        except Exception as e:
            self.error.emit(self.game_id, str(e))


class _ScoreboardWorker(QObject):
    """Fetches ESPN scoreboard on background thread."""
    finished = Signal(list)
    error = Signal(str)

    def run(self):
        try:
            from src.data.gamecast import fetch_espn_scoreboard
            games = fetch_espn_scoreboard()
            self.finished.emit(games)
        except Exception as e:
            self.error.emit(str(e))


class _PreloadRunnable(QRunnable):
    """QRunnable that preloads a single game into the cache (fire-and-forget)."""

    def __init__(self, game_id: str):
        super().__init__()
        self.game_id = game_id
        self.setAutoDelete(True)

    def run(self):
        try:
            if _cache.get(self.game_id) is not None:
                return  # already cached
            data = _fetch_and_parse(self.game_id)
            state = data.get("status_state", "")
            _cache.put(self.game_id, data, state)
            logger.debug(f"Preloaded game {self.game_id} ({state})")
        except Exception as e:
            logger.debug(f"Preload failed for {self.game_id}: {e}")


# ──────────────────────────────────────────────────────────────
# Main View
# ──────────────────────────────────────────────────────────────


class GamecastView(QWidget):
    """Immersive live game detail view with background loading."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        # Worker tracking (prevent GC + ensure clean lifecycle)
        self._active_threads: list = []   # keep refs to prevent GC
        self._current_game_id: Optional[str] = None
        self._home_team_id = None
        self._away_team_id = None
        self._home_abbr = ""
        self._away_abbr = ""
        self._known_play_count = 0
        self._game_ids: list = []         # all game IDs in combo order
        self._team_id_cache: Dict[str, Optional[int]] = {}  # abbr -> team_id

        # Thread pool for preloading (limit concurrency to avoid ESPN rate-limit)
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(3)

        self._build_ui()

        # Polling timer — smart intervals per game state
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._poll)

        # Deferred initial load (let UI render first)
        QTimer.singleShot(100, self._load_games)

    # ──────────────────────── UI CONSTRUCTION ────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Game selector bar ──
        sel = QHBoxLayout()
        sel.setSpacing(8)

        lbl = QLabel("GAME")
        lbl.setStyleSheet(
            "color: #94a3b8; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
        )
        sel.addWidget(lbl)

        self.game_combo = QComboBox()
        self.game_combo.setMinimumWidth(320)
        self.game_combo.currentIndexChanged.connect(self._on_game_selected)
        sel.addWidget(self.game_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setProperty("class", "outline")
        self.refresh_btn.setFixedWidth(80)
        self.refresh_btn.clicked.connect(self._load_games)
        sel.addWidget(self.refresh_btn)

        self._live_dot = QLabel("●")
        self._live_dot.setStyleSheet("color: #334155; font-size: 16px;")
        sel.addWidget(self._live_dot)

        self._status_hint = QLabel("")
        self._status_hint.setStyleSheet("color: #64748b; font-size: 10px;")
        sel.addWidget(self._status_hint)

        sel.addStretch()
        root.addLayout(sel)

        # ── Scoreboard ──
        self.scoreboard = ScoreboardWidget()
        root.addWidget(self.scoreboard)

        # ── Middle: Court + Info | Box Score ──
        mid_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Court + Info stacked vertically
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.court = CourtWidget()
        self.court.setMinimumHeight(200)
        left_layout.addWidget(self.court, 3)

        self.info_panel = InfoPanelWidget()
        left_layout.addWidget(self.info_panel, 2)

        mid_splitter.addWidget(left_panel)

        # Right: Box score with team tabs
        box_panel = QWidget()
        box_layout = QVBoxLayout(box_panel)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(0)

        box_header = QLabel("  BOX SCORE")
        box_header.setStyleSheet(
            "background: #1e293b; color: #94a3b8; font-size: 10px; "
            "font-weight: 700; letter-spacing: 1px; padding: 4px 8px;"
        )
        box_header.setFixedHeight(24)
        box_layout.addWidget(box_header)

        self.box_tabs = QTabWidget()
        self.box_tabs.setDocumentMode(True)
        self._away_box = self._make_box_table()
        self._home_box = self._make_box_table()
        self.box_tabs.addTab(self._away_box, "AWAY")
        self.box_tabs.addTab(self._home_box, "HOME")
        box_layout.addWidget(self.box_tabs)

        mid_splitter.addWidget(box_panel)
        mid_splitter.setStretchFactor(0, 2)
        mid_splitter.setStretchFactor(1, 3)
        root.addWidget(mid_splitter, 3)

        # ── Bottom: Play-by-play feed ──
        self.play_feed = PlayFeedWidget()
        self.play_feed.setMinimumHeight(120)
        self.play_feed.setMaximumHeight(260)
        root.addWidget(self.play_feed, 2)

    def _make_box_table(self) -> QTableWidget:
        """Create a box score table with player photo column."""
        table = QTableWidget()
        table.setColumnCount(11)
        table.setHorizontalHeaderLabels([
            "", "Player", "MIN", "PTS", "REB", "AST", "STL", "BLK", "FG", "3PT", "+/-",
        ])
        # Interactive resize so users can drag column widths
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        # Photo column fixed, Player column wider
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(0, 32)
        table.setColumnWidth(1, 120)  # Player name
        for col in range(2, 11):
            table.setColumnWidth(col, 50)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        return table

    # ──────────────────────── SCOREBOARD LOADING ────────────────────

    def _load_games(self):
        """Load today's games into combo — runs ESPN scoreboard on background thread."""
        self.refresh_btn.setEnabled(False)
        self._status_hint.setText("Loading games...")
        self._launch_worker(_ScoreboardWorker(), self._on_scoreboard_loaded,
                            self._on_scoreboard_error)

    def _on_scoreboard_loaded(self, games: list):
        """Populate combo (main thread) then preload all games in background."""
        self.refresh_btn.setEnabled(True)
        self._status_hint.setText("")

        self.game_combo.blockSignals(True)
        prev_id = self._current_game_id
        self.game_combo.clear()
        self._game_ids.clear()

        for g in games:
            status = g.get("status", "")
            short_detail = g.get("short_detail", "") or status
            away = g.get("away_team", "?")
            home = g.get("home_team", "?")
            a_score = g.get("away_score", 0)
            h_score = g.get("home_score", 0)
            state = g.get("state", "")

            # Convert ET times to local for scheduled games
            display_detail = short_detail
            if state == "pre" and "ET" in short_detail:
                try:
                    from datetime import datetime
                    from zoneinfo import ZoneInfo
                    # Parse "7:00 PM ET" → convert to local
                    time_str = short_detail.replace(" ET", "").strip()
                    et_time = datetime.strptime(time_str, "%I:%M %p")
                    et_tz = ZoneInfo("America/New_York")
                    local_tz = ZoneInfo("America/Los_Angeles")
                    now = datetime.now()
                    et_dt = now.replace(hour=et_time.hour, minute=et_time.minute,
                                        second=0, microsecond=0, tzinfo=et_tz)
                    local_dt = et_dt.astimezone(local_tz)
                    display_detail = local_dt.strftime("%I:%M %p PT").lstrip("0")
                except Exception:
                    display_detail = short_detail

            if a_score or h_score:
                label = f"{away} {a_score}  @  {home} {h_score}  —  {display_detail}"
            else:
                label = f"{away}  @  {home}  —  {display_detail}"
            espn_id = str(g.get("espn_id", ""))
            self.game_combo.addItem(label, espn_id)
            self._game_ids.append(espn_id)

        self.game_combo.blockSignals(False)

        # Restore previous selection or pick first
        if prev_id and prev_id in self._game_ids:
            self.game_combo.setCurrentIndex(self._game_ids.index(prev_id))
        elif self.game_combo.count() > 0:
            self._on_game_selected(0)

        # ── Preload all games in background ──
        self._preload_all_games()

    def _on_scoreboard_error(self, msg: str):
        self.refresh_btn.setEnabled(True)
        self._status_hint.setText("Failed to load games")
        logger.error(f"Scoreboard error: {msg}")

    def _preload_all_games(self):
        """Queue background preload for every game (thread pool, max 3 parallel)."""
        for gid in self._game_ids:
            if _cache.is_final(gid):
                continue  # already finalized, skip
            self._pool.start(_PreloadRunnable(gid))

    # ──────────────────────── GAME SELECTION ────────────────────────

    def _on_game_selected(self, idx):
        if idx < 0:
            return
        game_id = self.game_combo.itemData(idx)
        if not game_id:
            return
        self._current_game_id = str(game_id)
        self._known_play_count = 0
        self.court.clear_shots()

        # Try cache first → instant switch
        cached = _cache.get(self._current_game_id)
        if cached:
            self._apply_data(cached)
        else:
            self._status_hint.setText("Loading...")
            self._fetch_game(self._current_game_id)

        # Start smart polling
        self._start_smart_timer()

    # ──────────────────────── FETCH / POLL ────────────────────────

    def _fetch_game(self, game_id: str):
        """Launch a background fetch for a single game."""
        worker = _GameFetchWorker(game_id)
        worker.finished.connect(self._on_game_fetched)
        worker.error.connect(self._on_game_fetch_error)
        self._launch_worker(worker)

    def _on_game_fetched(self, game_id: str, data: dict):
        """Game data arrived — if it's still the selected game, update UI."""
        if game_id == self._current_game_id:
            self._status_hint.setText("")
            self._apply_data(data)

    def _on_game_fetch_error(self, game_id: str, msg: str):
        if game_id == self._current_game_id:
            self._status_hint.setText("Load failed — retrying...")
        logger.error(f"Fetch error for {game_id}: {msg}")

    def _poll(self):
        """Timer callback — refresh only if game is not final."""
        gid = self._current_game_id
        if not gid:
            return
        if _cache.is_final(gid):
            self.live_timer.stop()
            self._status_hint.setText("Final")
            return
        self._fetch_game(gid)

    def _start_smart_timer(self):
        """Set polling interval based on game state, or stop if final."""
        gid = self._current_game_id
        if not gid:
            return
        if _cache.is_final(gid):
            self.live_timer.stop()
            self._status_hint.setText("Final")
            return
        cached = _cache.get(gid)
        state = cached.get("status_state", "") if cached else ""
        if state == "in":
            self.live_timer.start(15_000)
        elif state == "pre":
            self.live_timer.start(60_000)
        else:
            # Unknown/scheduled — moderate interval
            self.live_timer.start(30_000)

    # ──────────────────────── THREAD MANAGEMENT ────────────────────

    def _launch_worker(self, worker: QObject, *connect_pairs):
        """Launch a QObject worker on a new QThread. Keeps refs alive, auto-cleans.

        connect_pairs: up to 2 callables — [0] connected to worker.finished,
        [1] connected to worker.error.  Uses QueuedConnection so slots always
        run on the main/GUI thread.
        """
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        _QC = Qt.ConnectionType.QueuedConnection

        # Connect provided slots with explicit QueuedConnection
        if len(connect_pairs) >= 1:
            if hasattr(worker, "finished"):
                worker.finished.connect(connect_pairs[0], _QC)
        if len(connect_pairs) >= 2:
            if hasattr(worker, "error"):
                worker.error.connect(connect_pairs[1], _QC)

        # Auto-cleanup: when thread finishes, schedule deletion
        def _cleanup():
            thread.deleteLater()
            worker.deleteLater()
            try:
                self._active_threads.remove((thread, worker))
            except ValueError:
                pass

        # Ensure thread quits when worker finishes
        if hasattr(worker, "finished"):
            worker.finished.connect(thread.quit)
        if hasattr(worker, "error"):
            worker.error.connect(thread.quit)
        thread.finished.connect(_cleanup, _QC)

        self._active_threads.append((thread, worker))
        thread.start()

    # ──────────────────────── DATA → UI ────────────────────────

    def _apply_data(self, data: dict):
        """Update all UI widgets from parsed game data. Runs on main thread."""
        summary = data.get("summary", {})
        odds = data.get("odds", {})
        boxscore = data.get("boxscore", {})
        plays = data.get("plays", [])
        home_win_pct = data.get("home_win_pct", 50.0)

        # ── Parse header ──
        header = summary.get("header", {})
        competitions = header.get("competitions", [{}])
        comp = competitions[0] if competitions else {}
        competitors = comp.get("competitors", [])
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), {})

        home_abbr = home_comp.get("team", {}).get("abbreviation", "HOME")
        away_abbr = away_comp.get("team", {}).get("abbreviation", "AWAY")
        home_score = int(home_comp.get("score", 0) or 0)
        away_score = int(away_comp.get("score", 0) or 0)

        status_detail = comp.get("status", {})
        status_text = status_detail.get("type", {}).get("description", "")
        status_state = status_detail.get("type", {}).get("state", "")
        period = int(status_detail.get("period", 0) or 0)
        clock_str = status_detail.get("displayClock", "0:00")

        # Resolve NBA team IDs (cached per abbreviation)
        home_team_id = self._resolve_team_id(home_abbr)
        away_team_id = self._resolve_team_id(away_abbr)
        self._home_team_id = home_team_id
        self._away_team_id = away_team_id
        self._home_abbr = home_abbr
        self._away_abbr = away_abbr

        # ── Live indicator ──
        if status_state == "in":
            self._live_dot.setStyleSheet("color: #22c55e; font-size: 16px;")
        elif status_state == "post":
            self._live_dot.setStyleSheet("color: #94a3b8; font-size: 16px;")
        else:
            self._live_dot.setStyleSheet("color: #334155; font-size: 16px;")

        # ── Scoreboard ──
        away_quarters = []
        home_quarters = []
        for ls_comp in competitors:
            linescores = ls_comp.get("linescores", [])
            quarters = [int(q.get("displayValue", 0) or 0) for q in linescores]
            if ls_comp.get("homeAway") == "home":
                home_quarters = quarters
            else:
                away_quarters = quarters

        self.scoreboard.update_data(
            away_abbr=away_abbr, home_abbr=home_abbr,
            away_team_id=away_team_id, home_team_id=home_team_id,
            away_score=away_score, home_score=home_score,
            away_quarters=away_quarters, home_quarters=home_quarters,
            status_text=status_text, status_state=status_state,
            clock=clock_str, period=period,
        )

        # ── Court ──
        if home_team_id and away_team_id:
            self.court.set_teams(home_team_id, away_team_id)

        # Feed new plays to court animation (only new ones)
        flat_plays = self._flatten_plays(plays)
        new_plays = flat_plays[self._known_play_count:]
        for play in new_plays:
            self.court.add_play(play)
        self._known_play_count = len(flat_plays)

        # ── Info panel (prediction + odds) ──
        model_spread = self._update_prediction(
            home_team_id, away_team_id,
            home_score, away_score,
            status_state, period, clock_str,
        )
        self.info_panel.update_odds(odds)

        # Win probability — prefer model spread; fall back to ESPN predictor
        if model_spread is not None and model_spread != 0:
            # Convert spread to approximate win probability using logistic curve
            # A ~10-pt spread ≈ 75% win probability
            import math
            home_win_pct = 100.0 / (1.0 + math.exp(-0.15 * model_spread))
        elif status_state == "post":
            # Finished game — use actual result
            if home_score > away_score:
                home_win_pct = 100.0
            elif away_score > home_score:
                home_win_pct = 0.0
            else:
                home_win_pct = 50.0
        # else: home_win_pct keeps the ESPN predictor value (may be 50 default)

        home_clr, _ = get_team_colors(home_team_id) if home_team_id else ("#3b82f6", "")
        away_clr, _ = get_team_colors(away_team_id) if away_team_id else ("#ef4444", "")
        self.info_panel.update_win_probability(
            home_win_pct, home_clr, away_clr, home_abbr, away_abbr,
        )

        # ── Box score tabs ──
        self.box_tabs.setTabText(0, away_abbr)
        self.box_tabs.setTabText(1, home_abbr)
        self._fill_box_table(boxscore, competitors, is_home=False, table=self._away_box)
        self._fill_box_table(boxscore, competitors, is_home=True, table=self._home_box)

        # ── Play feed ──
        self.play_feed.set_teams(home_team_id, away_team_id, home_abbr, away_abbr)
        self.play_feed.set_plays(plays)

        # ── Update timer based on actual state ──
        if status_state == "in":
            self.live_timer.start(15_000)
        elif status_state == "post":
            self.live_timer.stop()
            self._status_hint.setText("Final")
        else:
            self.live_timer.start(60_000)

    # ──────────────────────── HELPERS ────────────────────────

    @staticmethod
    def _fetch_espn_headshot(url: str, size: int) -> Optional[QPixmap]:
        """Download ESPN player headshot and return as circular QPixmap."""
        cache_key = (url, size)
        if cache_key in _espn_headshot_cache:
            return _espn_headshot_cache[cache_key]
        try:
            import requests
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return None
            img = QImage()
            img.loadFromData(resp.content)
            if img.isNull():
                return None
            pixmap = QPixmap.fromImage(img).scaled(
                size, size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            from src.ui.widgets.image_utils import _make_circle_pixmap
            pixmap = _make_circle_pixmap(pixmap)
            _espn_headshot_cache[cache_key] = pixmap
            return pixmap
        except Exception:
            return None

    def _resolve_team_id(self, abbr: str) -> Optional[int]:
        """Lookup NBA team_id from abbreviation with per-session cache."""
        if abbr in self._team_id_cache:
            return self._team_id_cache[abbr]
        try:
            from src.database import db
            row = db.fetch_one(
                "SELECT team_id FROM teams WHERE abbreviation = ?", (abbr,)
            )
            tid = row["team_id"] if row else None
        except Exception:
            tid = None
        self._team_id_cache[abbr] = tid
        return tid

    def _flatten_plays(self, plays) -> list:
        """Flatten nested ESPN play structure into a flat list."""
        flat = []
        if not isinstance(plays, list):
            return flat
        for play in plays:
            items = play.get("items", [play])
            for item in items:
                if isinstance(item, dict) and item.get("text"):
                    flat.append(item)
        return flat

    def _update_prediction(self, home_team_id, away_team_id,
                           home_score, away_score,
                           status_state, period, clock_str):
        """Run live prediction and update info panel.

        Returns predicted spread or None (used for win-prob bar).
        """
        if not home_team_id or not away_team_id:
            self.info_panel.update_prediction(None)
            return None

        quarter = 0
        minutes_elapsed = 0.0
        if status_state == "in":
            quarter = max(0, period - 1)
            try:
                parts = clock_str.split(":")
                mins_left = int(parts[0]) if parts else 0
                secs_left = int(parts[1]) if len(parts) > 1 else 0
                if period <= 4:
                    # Regulation quarter = 12 minutes
                    period_len = 12 * 60
                    minutes_elapsed = (period - 1) * 12.0
                else:
                    # OT period = 5 minutes; regulation = 48 min total
                    period_len = 5 * 60
                    minutes_elapsed = 48.0 + (period - 5) * 5.0
                elapsed_in_q = period_len - (mins_left * 60 + secs_left)
                minutes_elapsed += max(0, elapsed_in_q) / 60.0
            except Exception:
                minutes_elapsed = min(quarter, 4) * 12.0
            # Cap completed-quarters at 4 for history lookup (OT has no q5+ data)
            quarter = min(quarter, 4)
        elif status_state == "post":
            quarter = 4
            # Account for OT in final time so pace signal is accurate
            ot_periods = max(0, period - 4) if period > 4 else 0
            minutes_elapsed = 48.0 + ot_periods * 5.0

        try:
            from src.analytics.live_prediction import live_predict
            pred = live_predict(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_score=home_score,
                away_score=away_score,
                quarter=quarter,
                minutes_elapsed=minutes_elapsed,
            )
            self.info_panel.update_prediction(
                pred, self._home_abbr, self._away_abbr
            )
            return pred.get("spread", None)
        except Exception:
            logger.error("Prediction failed", exc_info=True)
            self.info_panel.update_prediction(None)
            return None

    def _fill_box_table(self, boxscore, competitors, is_home: bool, table: QTableWidget):
        """Populate a box score table for one team, with player photos."""
        box_teams = boxscore.get("players", [])
        if not isinstance(box_teams, list):
            return

        target_key = "home" if is_home else "away"
        team_box = None
        for tb in box_teams:
            team_info = tb.get("team", {})
            for comp in competitors:
                if comp.get("homeAway") == target_key:
                    comp_team_id = comp.get("team", {}).get("id")
                    tb_team_id = team_info.get("id")
                    if str(comp_team_id) == str(tb_team_id):
                        team_box = tb
                        break
            if team_box:
                break

        if not team_box:
            idx = 1 if is_home else 0
            if idx < len(box_teams):
                team_box = box_teams[idx]

        if not team_box:
            return

        statistics = team_box.get("statistics", [])
        if not statistics:
            return
        stat_block = statistics[0]
        labels = stat_block.get("labels", [])
        athletes = stat_block.get("athletes", [])

        table.setRowCount(len(athletes))
        for r, ath in enumerate(athletes):
            athlete_info = ath.get("athlete", {})
            name = athlete_info.get("displayName", "")
            player_id = athlete_info.get("id", "")
            stats = ath.get("stats", [])
            stat_map = dict(zip(labels, stats)) if len(labels) == len(stats) else {}

            table.setRowHeight(r, 30)

            # Column 0: Player photo (28px)
            photo_item = QTableWidgetItem()
            if player_id:
                try:
                    pixmap = get_player_photo(int(player_id), 28, circle=True)
                    if not pixmap:
                        # Fallback: use ESPN headshot URL
                        headshot_url = athlete_info.get("headshot", {}).get("href", "")
                        if headshot_url:
                            pixmap = self._fetch_espn_headshot(headshot_url, 28)
                    if pixmap:
                        photo_item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
                except Exception:
                    pass
            table.setItem(r, 0, photo_item)

            # Column 1: Name
            table.setItem(r, 1, QTableWidgetItem(name))

            # Columns 2-10: Stats
            stat_keys = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "FG", "3PT", "+/-"]
            for c, key in enumerate(stat_keys):
                val = stat_map.get(key, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(r, c + 2, item)
