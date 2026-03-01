"""Gamecast tab — immersive live game view with scoreboard, court, plays, odds.

Architecture:
- All ESPN network calls run on background threads (never block UI)
- In-memory cache stores parsed game data per game_id with TTL
- Background preloader fetches all games after scoreboard loads
- Smart polling: live=15s, pre-game=60s, final=never (single load)
- Game switching is instant when data is cached
"""

import logging
import threading
import time
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QTabWidget, QScrollArea,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QRunnable, QThreadPool
from PySide6.QtGui import QColor, QImage, QPixmap

from src.ui.widgets.scoreboard_widget import ScoreboardWidget
from src.ui.widgets.court_widget import CourtWidget
from src.ui.widgets.play_feed_widget import PlayFeedWidget
from src.ui.widgets.info_panel_widget import InfoPanelWidget, _CollapsibleSection
from src.ui.widgets.nba_colors import get_team_colors
from src.ui.widgets.image_utils import get_team_logo, get_player_photo

logger = logging.getLogger(__name__)

# Import ESPN→NBA abbreviation mapping from the data layer
from src.data.gamecast import normalize_espn_abbr  # noqa: E402

# ──────────────────────────────────────────────────────────────
# In-memory caches
# ──────────────────────────────────────────────────────────────

_espn_headshot_cache: Dict[tuple, QPixmap] = {}   # (url, size) → QPixmap
_espn_headshot_data: Dict[str, bytes] = {}            # url → raw image bytes
_headshot_lock = threading.Lock()                      # guards both headshot caches

_CACHE_TTL_LIVE = 8         # seconds — live games refresh often
_CACHE_TTL_PRE = 55         # seconds — pre-game (odds may update)
_CACHE_TTL_FINAL = 86400    # seconds — final games almost never change

class _GameCache:
    """Thread-safe in-memory cache for parsed game data."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}       # game_id -> parsed data
        self._ts: Dict[str, float] = {}         # game_id -> timestamp
        self._state: Dict[str, str] = {}        # game_id -> "pre"/"in"/"post"

    def get(self, game_id: str) -> Optional[dict]:
        """Return cached data if still fresh, else None."""
        with self._lock:
            if game_id not in self._data:
                return None
            age = time.time() - self._ts.get(game_id, 0)
            ttl = self._ttl_for(game_id)
            if age > ttl:
                return None
            return self._data[game_id]

    def put(self, game_id: str, data: dict, state: str = ""):
        """Store parsed data with current timestamp."""
        with self._lock:
            self._data[game_id] = data
            self._ts[game_id] = time.time()
            if state:
                self._state[game_id] = state

    def is_final(self, game_id: str) -> bool:
        with self._lock:
            state = self._state.get(game_id)
            if state == "post":
                return True
            if game_id in self._data:
                if self._data[game_id].get("status_state") == "post":
                    self._state[game_id] = "post"
                    return True
            return False

    def _ttl_for(self, game_id: str) -> float:
        s = self._state.get(game_id, "")
        if s == "post":
            return _CACHE_TTL_FINAL
        elif s == "in":
            return _CACHE_TTL_LIVE
        return _CACHE_TTL_PRE

    def clear(self):
        with self._lock:
            self._data.clear()
            self._ts.clear()
            self._state.clear()


_cache = _GameCache()

# ──────────────────────────────────────────────────────────────
# Background workers
# ──────────────────────────────────────────────────────────────


def _fetch_and_parse(game_id: str) -> dict:
    """Network call + parse — runs off main thread. Returns parsed dict."""
    from src.data.gamecast import fetch_espn_game_summary, normalize_espn_abbr, get_actionnetwork_odds
    
    # Retry logic since this runs on a background thread anyway
    import time
    summary = {}
    for attempt in range(3):
        try:
            summary = fetch_espn_game_summary(game_id)
            if summary and "header" in summary:
                break
        except Exception as e:
            if attempt == 2:
                logger.error(f"Failed to fetch summary for {game_id} after 3 attempts: {e}")
            else:
                time.sleep(1)

    # Detect game state from header
    header = summary.get("header", {})
    comps = header.get("competitions", [{}])
    comp = comps[0] if comps else {}
    status_state = comp.get("status", {}).get("type", {}).get("state", "")

    competitors = comp.get("competitors", [])
    home_c = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away_c = next((c for c in competitors if c.get("homeAway") == "away"), {})
    home_abbr = normalize_espn_abbr(home_c.get("team", {}).get("abbreviation", ""))
    away_abbr = normalize_espn_abbr(away_c.get("team", {}).get("abbreviation", ""))

    odds = {}
    try:
        odds = get_actionnetwork_odds(home_abbr, away_abbr)
    except Exception:
        pass

    if not odds or not odds.get("spread"):
        pickcenter = summary.get("pickcenter", [])
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
            # First check if we're shutting down (crude but effective)
            import sys
            if not getattr(sys, "modules", None):
                return
                
            if _cache.get(self.game_id) is not None:
                cached = _cache.get(self.game_id)
                if cached.get("status_state") != "in":
                    return  # already cached
            data = _fetch_and_parse(self.game_id)
            state = data.get("status_state", "")
            _cache.put(self.game_id, data, state)
            logger.debug(f"Preloaded game {self.game_id} ({state})")
        except Exception as e:
            logger.debug(f"Preload failed for {self.game_id}: {e}")


# ──────────────────────────────────────────────────────────────
# WebSocket → Qt bridge
# ──────────────────────────────────────────────────────────────


class _WebSocketBridge(QObject):
    """Thread-safe bridge: FastcastWebSocket thread → Qt main thread signals."""
    data_changed = Signal()
    ws_error = Signal(str)


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
        self._pending_headshots: set = set()  # URLs already queued for bg fetch

        # Thread pool for preloading (limit concurrency to avoid ESPN rate-limit)
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(3)

        # WebSocket for real-time live updates
        self._fastcast_ws = None
        self._ws_bridge = _WebSocketBridge()
        self._ws_bridge.data_changed.connect(self._on_ws_data_changed)
        self._ws_bridge.ws_error.connect(self._on_ws_error)

        # Debounce timer for WebSocket-triggered refetches (avoids hammering ESPN)
        self._ws_debounce = QTimer(self)
        self._ws_debounce.setSingleShot(True)
        self._ws_debounce.setInterval(2000)  # 2s debounce
        self._ws_debounce.timeout.connect(self._ws_refetch)

        self._build_ui()

        # Polling timer — smart intervals per game state
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._poll)

        # Scoreboard refresh timer — keeps dropdown labels (scores) up to date
        self._scoreboard_timer = QTimer(self)
        self._scoreboard_timer.timeout.connect(self._load_games)

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

        # ── Middle: Info Cards | Box Score ──
        self._mid_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Info Cards in scroll area (full height, no court)
        self.info_panel = InfoPanelWidget()
        self._info_scroll = QScrollArea()
        self._info_scroll.setWidget(self.info_panel)
        self._info_scroll.setWidgetResizable(True)
        self._info_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._info_scroll.setStyleSheet("QScrollArea { background: transparent; }"
                                  "QScrollBar:vertical { width: 6px; background: transparent; }"
                                  "QScrollBar::handle:vertical { background: rgba(255,255,255,0.2); border-radius: 3px; }"
                                  "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }")
        self._mid_splitter.addWidget(self._info_scroll)

        # Right: Box score (collapsible) with team tabs
        self._box_section = _CollapsibleSection("Box Score")
        self.box_tabs = QTabWidget()
        self.box_tabs.setDocumentMode(True)
        self._away_box = self._make_box_table()
        self._home_box = self._make_box_table()
        self.box_tabs.addTab(self._away_box, "AWAY")
        self.box_tabs.addTab(self._home_box, "HOME")
        self._box_section.add_widget(self.box_tabs, 1)

        self._mid_splitter.addWidget(self._box_section)
        self._mid_splitter.setStretchFactor(0, 2)
        self._mid_splitter.setStretchFactor(1, 3)
        self._box_section.toggled.connect(self._on_box_toggled)
        self._box_expanded_sizes = None
        root.addWidget(self._mid_splitter, 3)

        # ── Bottom: Shot Chart + Play-by-play side by side ──
        self._bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._court_section = _CollapsibleSection("Shot Chart")
        self.court = CourtWidget()
        self.court.setFixedHeight(220)
        self._court_section.add_widget(self.court, 0)
        self._bottom_splitter.addWidget(self._court_section)

        self.play_feed = PlayFeedWidget()
        self.play_feed.setMinimumHeight(120)
        self._bottom_splitter.addWidget(self.play_feed)
        self._bottom_splitter.setStretchFactor(0, 0)
        self._bottom_splitter.setStretchFactor(1, 1)
        root.addWidget(self._bottom_splitter, 2)

    def _make_box_table(self) -> QTableWidget:
        """Create a box score table with player photo column."""
        table = QTableWidget()
        table.setColumnCount(12)
        table.setHorizontalHeaderLabels([
            "", "Player", "MIN", "PTS", "REB", "AST", "STL", "BLK", "FG", "3PT", "PF", "+/-",
        ])
        # Interactive resize so users can drag column widths
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        # Photo column fixed, Player column wider
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(0, 32)
        table.setColumnWidth(1, 120)  # Player name
        for col in range(2, 12):
            table.setColumnWidth(col, 50)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        return table

    def _on_box_toggled(self, collapsed: bool):
        """Handle box score collapse/expand — switch info cards to 2x2 grid."""
        if collapsed:
            self._box_expanded_sizes = self._mid_splitter.sizes()
            self._box_section.setMaximumWidth(24)
            self.info_panel.set_grid_mode(True)
        else:
            self._box_section.setMaximumWidth(16777215)
            self.info_panel.set_grid_mode(False)
            if self._box_expanded_sizes:
                self._mid_splitter.setSizes(self._box_expanded_sizes)
            else:
                total = sum(self._mid_splitter.sizes())
                self._mid_splitter.setSizes([int(total * 0.4), int(total * 0.6)])

    # ──────────────────────── SCOREBOARD LOADING ────────────────────

    def _load_games(self):
        """Load today's games into combo — runs ESPN scoreboard on background thread."""
        self.refresh_btn.setEnabled(False)
        
        if self.game_combo.count() == 0:
            self._status_hint.setText("Loading games...")
        else:
            self._status_hint.setText("Refreshing...")

        self._launch_worker(_ScoreboardWorker(), self._on_scoreboard_loaded,
                            self._on_scoreboard_error)

    def _on_scoreboard_loaded(self, games: list):
        """Populate combo (main thread) then preload all games in background."""
        self.refresh_btn.setEnabled(True)
        self._status_hint.setText("")

        self.game_combo.blockSignals(True)
        prev_id = self._current_game_id
        
        # Keep track of current index if we have one
        current_idx = self.game_combo.currentIndex()

        self.game_combo.clear()
        self._game_ids.clear()
        
        has_added_games = False

        # Add an initial "Select Game" to prevent it from automatically triggering the first item
        self.game_combo.addItem("— Select Game —", None)

        game_states = []
        for g in games:
            status = g.get("status", "")
            short_detail = g.get("short_detail", "") or status
            away = normalize_espn_abbr(g.get("away_team", "?"))
            home = normalize_espn_abbr(g.get("home_team", "?"))
            a_score = g.get("away_score", 0)
            h_score = g.get("home_score", 0)
            state = g.get("state", "")
            game_states.append(state)

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
                    local_tz = datetime.now().astimezone().tzinfo
                    now = datetime.now()
                    et_dt = now.replace(hour=et_time.hour, minute=et_time.minute,
                                        second=0, microsecond=0, tzinfo=et_tz)
                    local_dt = et_dt.astimezone(local_tz)
                    display_detail = local_dt.strftime("%I:%M %p").lstrip("0")
                except Exception:
                    display_detail = short_detail

            if a_score or h_score:
                label = f"{away} {a_score}  @  {home} {h_score}  —  {display_detail}"
            else:
                label = f"{away}  @  {home}  —  {display_detail}"
            espn_id = str(g.get("espn_id", ""))
            self.game_combo.addItem(label, espn_id)
            self._game_ids.append(espn_id)
            has_added_games = True

        self.game_combo.blockSignals(False)

        # Restore previous selection or pick first valid one if not previously selected
        if prev_id and prev_id in self._game_ids:
            # +1 because of "Select Game" at index 0
            self.game_combo.setCurrentIndex(self._game_ids.index(prev_id) + 1)
        elif current_idx > 0 and self.game_combo.count() > current_idx:
            self.game_combo.setCurrentIndex(current_idx)
        elif not prev_id:
             # Just leave it on "Select Game" so it doesn't auto-load
             self.game_combo.setCurrentIndex(0)

        # ── Preload all games in background ──
        self._preload_all_games()

        # ── Manage scoreboard refresh timer based on game states ──
        if any(s == "in" for s in game_states):
            self._scoreboard_timer.start(20_000)   # 20s when live games
        elif any(s == "pre" for s in game_states):
            self._scoreboard_timer.start(120_000)  # 2min when all pre-game
        else:
            self._scoreboard_timer.stop()           # all final, stop

    def _on_scoreboard_error(self, msg: str):
        self.refresh_btn.setEnabled(True)
        self._status_hint.setText("Failed to load games")
        logger.error(f"Scoreboard error: {msg}")

    def _preload_all_games(self):
        """Queue background preload for every game (thread pool, max 3 parallel)."""
        for gid in self._game_ids:
            if _cache.is_final(gid):
                continue  # already finalized, skip
                
            # If it's cached and pre-game, don't continually pre-load unless we need to
            cached = _cache.get(gid)
            if cached and cached.get("status_state") == "pre":
                 continue
                 
            self._pool.start(_PreloadRunnable(gid))

    # ──────────────────────── GAME SELECTION ────────────────────────

    def _on_game_selected(self, idx):
        if idx < 0:
            return
        game_id = self.game_combo.itemData(idx)
        if not game_id:
            # Selected the empty "Select Game" option
            self._current_game_id = None
            self.live_timer.stop()
            self._stop_websocket()
            self._status_hint.setText("")
            self.court.clear_shots()
            self.info_panel.update_prediction(None)
            self.info_panel.update_odds({})
            self.info_panel.update_win_probability(50.0)
            
            # Clear scoreboard and other parts
            self.scoreboard.update_data(
                home_abbr="Home Team",
                away_abbr="Away Team",
                home_score=0,
                away_score=0,
                period=0,
                clock="12:00",
                status_text="Select a game",
                status_state="pre",
            )
            self.info_panel.update_odds({})
            self.play_feed.clear()
            self.box_tabs.setTabText(0, "AWAY")
            self.box_tabs.setTabText(1, "HOME")
            self._away_box.setRowCount(0)
            self._home_box.setRowCount(0)
            return
            
        game_id = str(game_id)
            
        # Prevent re-fetching if we're already viewing this game
        if game_id == self._current_game_id:
            # If timer is active and game id matches, no-op.
            if self.live_timer.isActive() or _cache.is_final(game_id):
                return
            
        self._current_game_id = game_id
        self._known_play_count = 0
        self._stop_websocket()  # stop WS for previous game
        self.court.clear_shots()

        # Try cache first → instant switch
        cached = _cache.get(self._current_game_id)
        if cached:
            self._apply_data(cached)
            state = cached.get("status_state", "")
            if state == "post":
                self._status_hint.setText("Final")
                self.live_timer.stop()
            else:
                self._status_hint.setText("Refreshing...")
                self._fetch_game(self._current_game_id)
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
            state = data.get("status_state", "")
            if state == "post":
                self._status_hint.setText("Final")
            elif state == "in" and self._fastcast_ws and self._fastcast_ws.is_connected:
                self._status_hint.setText("WS")
            else:
                self._status_hint.setText("")
            self._apply_data(data)
            
            # Since state might have changed, ensure timer is updated
            self._start_smart_timer()

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
            self._stop_websocket()
            self._status_hint.setText("Final")
            return

        cached = _cache.get(gid)
        state = cached.get("status_state", "") if cached else ""
        if state == "post":
            _cache.put(gid, cached, state="post")
            self.live_timer.stop()
            self._stop_websocket()
            self._status_hint.setText("Final")
            return

        if state == "in":
            # Start WebSocket for real-time updates
            self._start_websocket(gid)
            # Polling is a slow fallback when WS is active (30s)
            # or fast (10s) if WS failed to connect
            ws_ok = self._fastcast_ws and self._fastcast_ws.is_connected
            self.live_timer.start(30_000 if ws_ok else 10_000)
        elif state == "pre":
            self._stop_websocket()
            self.live_timer.start(60_000)
        else:
            self._stop_websocket()
            self.live_timer.start(30_000)

    # ──────────────────────── WEBSOCKET ────────────────────────

    def _start_websocket(self, game_id: str):
        """Start Fastcast WebSocket for a live game (no-op if already connected)."""
        if (self._fastcast_ws and self._fastcast_ws.game_id == game_id
                and self._fastcast_ws.is_connected):
            return  # already connected to this game

        self._stop_websocket()
        try:
            from src.data.gamecast import FastcastWebSocket
            self._fastcast_ws = FastcastWebSocket(
                game_id,
                on_data_changed=lambda: self._ws_bridge.data_changed.emit(),
                on_error=lambda e: self._ws_bridge.ws_error.emit(e),
            )
            self._fastcast_ws.start()
            logger.info(f"Fastcast WS started for {game_id}")
        except Exception as e:
            logger.warning(f"Failed to start Fastcast WS: {e}")
            self._fastcast_ws = None

    def _stop_websocket(self):
        """Stop any active WebSocket connection."""
        if self._fastcast_ws:
            self._fastcast_ws.stop()
            self._fastcast_ws = None
        self._ws_debounce.stop()

    def _on_ws_data_changed(self):
        """WebSocket detected a change — schedule a debounced refetch."""
        if not self._ws_debounce.isActive():
            self._ws_debounce.start()

    def _on_ws_error(self, msg: str):
        """WebSocket failed — revert to fast polling as fallback."""
        logger.warning(f"Fastcast WS error, reverting to fast polling: {msg}")
        gid = self._current_game_id
        if gid and not _cache.is_final(gid):
            self.live_timer.start(10_000)

    def _ws_refetch(self):
        """Debounce timer fired — do the actual refetch now."""
        gid = self._current_game_id
        if gid and not _cache.is_final(gid):
            self._fetch_game(gid)

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
        home_name = home_comp.get("team", {}).get("displayName", home_abbr)
        away_name = away_comp.get("team", {}).get("displayName", away_abbr)
        home_score = int(home_comp.get("score", 0) or 0)
        away_score = int(away_comp.get("score", 0) or 0)

        # Timeouts remaining
        home_timeouts = home_comp.get("timeoutsRemaining", -1)
        away_timeouts = away_comp.get("timeoutsRemaining", -1)

        # Bonus indicator
        home_bonus = home_comp.get("linescores", [{}])[-1].get("isBonus", False) if home_comp.get("linescores") else False
        away_bonus = away_comp.get("linescores", [{}])[-1].get("isBonus", False) if away_comp.get("linescores") else False

        # Calculate team fouls from box score
        home_fouls = 0
        away_fouls = 0
        box_players = boxscore.get("players", []) if isinstance(boxscore, dict) else []
        for tb in box_players:
            tb_tid = str(tb.get("team", {}).get("id", ""))
            side = "home" if tb_tid == home_comp.get("team", {}).get("id", "") else "away"
            stats_blocks = tb.get("statistics", [])
            if not stats_blocks:
                continue
            labels = stats_blocks[0].get("labels", [])
            pf_index = labels.index("PF") if "PF" in labels else -1
            if pf_index >= 0:
                for ath in stats_blocks[0].get("athletes", []):
                    stats = ath.get("stats", [])
                    if len(stats) > pf_index:
                        pf_str = stats[pf_index]
                        try:
                            f = int(pf_str)
                            if side == "home":
                                home_fouls += f
                            else:
                                away_fouls += f
                        except ValueError:
                            pass

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
            away_abbr=away_name, home_abbr=home_name,
            away_team_id=away_team_id, home_team_id=home_team_id,
            away_score=away_score, home_score=home_score,
            away_quarters=away_quarters, home_quarters=home_quarters,
            status_text=status_text, status_state=status_state,
            clock=clock_str, period=period,
            away_timeouts=away_timeouts, home_timeouts=home_timeouts,
            away_bonus=away_bonus, home_bonus=home_bonus,
            away_fouls=away_fouls, home_fouls=home_fouls
        )

        # ── Court ──
        if home_team_id and away_team_id:
            self.court.set_teams(home_team_id, away_team_id)

        # Feed new plays to court animation (only new ones)
        home_espn_id = home_comp.get("team", {}).get("id", "")
        away_espn_id = away_comp.get("team", {}).get("id", "")
        flat_plays = self._flatten_plays(plays)
        is_initial_load = (self._known_play_count == 0)
        new_plays = flat_plays[self._known_play_count:]

        for play in new_plays:
            # Resolve ESPN fields to clean values for court widget
            espn_tid = str(play.get("team", {}).get("id", ""))
            if espn_tid == str(home_espn_id):
                play["team_id"] = home_team_id
            elif espn_tid == str(away_espn_id):
                play["team_id"] = away_team_id
            clock_raw = play.get("clock", {})
            if isinstance(clock_raw, dict):
                play["clock"] = clock_raw.get("displayValue", "")
            per_raw = play.get("period", {})
            if isinstance(per_raw, dict):
                play["period"] = per_raw.get("number", 0)
            self.court.add_play(play)

        if new_plays and not is_initial_load:
            last_play = new_plays[-1]
            text = last_play.get("text", "")
            
            # Find the last substitution in the new batch, if any
            last_sub_text = None
            for p in new_plays:
                if "enters the game" in p.get("text", "").lower():
                    last_sub_text = p.get("text", "")

            if "timeout" in text.lower() and status_state == "in":
                self.scoreboard.start_timeout(75)
            elif last_sub_text:
                self.scoreboard.show_substitution(last_sub_text, boxscore)
                if hasattr(self.scoreboard, 'stop_timeout'):
                    self.scoreboard.stop_timeout()
            else:
                if hasattr(self.scoreboard, 'stop_timeout'):
                    self.scoreboard.stop_timeout()
        elif is_initial_load and flat_plays:
            # Check if we should show a timeout that is currently active
            last_play = flat_plays[-1]
            text = last_play.get("text", "")
            if "timeout" in text.lower() and status_state == "in":
                p_clock = last_play.get("clock", "")
                p_period = last_play.get("period", 0)
                is_recent = True
                if p_period == period and p_clock and clock_str:
                    try:
                        def parse_sec(c):
                            if ":" in c:
                                parts = c.split(":")
                                return int(parts[0])*60 + int(parts[1])
                            return float(c)
                        if abs(parse_sec(p_clock) - parse_sec(clock_str)) > 15:
                            is_recent = False
                    except:
                        pass
                elif p_period != period:
                    is_recent = False
                
                if is_recent:
                    self.scoreboard.start_timeout(75)
                else:
                    if hasattr(self.scoreboard, 'stop_timeout'):
                        self.scoreboard.stop_timeout()
            else:
                if hasattr(self.scoreboard, 'stop_timeout'):
                    self.scoreboard.stop_timeout()

        self._known_play_count = len(flat_plays)

        # ── Info panel (prediction + odds + flow) ──
        model_spread = self._update_prediction(
            home_team_id, away_team_id,
            home_score, away_score,
            status_state, period, clock_str,
        )
        self.info_panel.update_odds(odds)

        # Calculate Game Flow Stats
        home_drives = 0
        away_drives = 0
        home_drives_scored = 0
        away_drives_scored = 0
        current_run_team = None
        current_run_pts = 0

        for p in flat_plays:
            text = p.get("text", "").lower()
            espn_tid = str(p.get("team", {}).get("id", ""))
            team = home_team_id if espn_tid == home_espn_id else (away_team_id if espn_tid == away_espn_id else None)
            scoring = p.get("scoringPlay")
            
            if "driving" in text:
                if team == home_team_id:
                    home_drives += 1
                    if scoring:
                        home_drives_scored += 1
                elif team == away_team_id:
                    away_drives += 1
                    if scoring:
                        away_drives_scored += 1
                        
            if not scoring:
                continue
            
            pts = 0
            if "free throw" in text:
                pts = 1
            elif "3-pt" in text or "three point" in text:
                pts = 3
            else:
                pts = 2
                
            if team == current_run_team:
                current_run_pts += pts
            else:
                current_run_team = team
                current_run_pts = pts
                
        run_text = ""
        if current_run_pts >= 5 and current_run_team:
            run_team_abbr = home_abbr if current_run_team == home_team_id else away_abbr
            run_text = f"{run_team_abbr} on a {current_run_pts}-0 run"
            
        # Calculate Game Possessions from Team Box Score
        box_teams = boxscore.get("teams", []) if isinstance(boxscore, dict) else []
        home_poss = 0
        away_poss = 0
        home_poss_scored = 0
        away_poss_scored = 0
        for tb in box_teams:
            tb_tid = str(tb.get("team", {}).get("id", ""))
            side = "home" if tb_tid == home_espn_id else "away"
            stats_blocks = tb.get("statistics", [])
            if not stats_blocks:
                continue
            stats_dict = {}
            for s in stats_blocks:
                if "abbreviation" in s:
                    stats_dict[s["abbreviation"]] = s.get("displayValue")
                elif "label" in s:
                    stats_dict[s["label"]] = s.get("displayValue")
            
            try:
                fg = stats_dict.get("FG", "0-0").split("-")
                fgm = int(fg[0]) if len(fg) > 0 and fg[0].isdigit() else 0
                fga = int(fg[1]) if len(fg) > 1 and fg[1].isdigit() else 0
                
                ft = stats_dict.get("FT", "0-0").split("-")
                ftm = int(ft[0]) if len(ft) > 0 and ft[0].isdigit() else 0
                fta = int(ft[1]) if len(ft) > 1 and ft[1].isdigit() else 0
                
                orb = int(stats_dict.get("OR", stats_dict.get("Offensive Rebounds", 0)))
                tov = int(stats_dict.get("TO", stats_dict.get("Turnovers", 0)))
                
                poss = round(fga + 0.44 * fta - orb + tov)
                scored = round(fgm + (ftm / 2))
                
                if side == "home":
                    home_poss = poss
                    home_poss_scored = scored
                else:
                    away_poss = poss
                    away_poss_scored = scored
            except Exception:
                pass
            
        self.info_panel.update_flow_stats(f"{home_drives_scored}/{home_drives}", f"{away_drives_scored}/{away_drives}", run_text, f"{home_poss_scored}/{home_poss}", f"{away_poss_scored}/{away_poss}")

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
        self.play_feed.set_teams(
            home_team_id, away_team_id, home_abbr, away_abbr,
            home_espn_id, away_espn_id,
        )
        self.play_feed.set_plays(plays)

        # ── Delayed headshot refresh — if photos were queued, schedule
        # a single re-apply so they appear once downloaded (especially for
        # finished games where the live timer is stopped).
        if getattr(self, '_needs_headshot_refresh', False):
            self._needs_headshot_refresh = False
            self._last_data = data
            if not getattr(self, '_headshot_timer_active', False):
                self._headshot_timer_active = True
                QTimer.singleShot(4000, self._refresh_for_headshots)

        # ── Update timer based on actual state ──
        if status_state == "in":
            self.live_timer.start(10_000)
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
        with _headshot_lock:
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
            with _headshot_lock:
                _espn_headshot_cache[cache_key] = pixmap
            return pixmap
        except Exception:
            return None

    def _queue_headshot_fetch(self, url: str, size: int):
        """Schedule a headshot download on the thread pool (non-blocking).

        The image will appear on the next UI refresh once cached.
        """
        cache_key = (url, size)
        with _headshot_lock:
            if cache_key in _espn_headshot_cache:
                return
        if url in self._pending_headshots:
            return
        self._pending_headshots.add(url)

        class _HeadshotRunnable(QRunnable):
            def __init__(self, u, s):
                super().__init__()
                self.url = u
                self.size = s
                self.setAutoDelete(True)

            def run(self):
                try:
                    import requests as _req
                    resp = _req.get(self.url, timeout=5)
                    if resp.status_code == 200 and resp.content:
                        # Store raw bytes only — QPixmap MUST be created on
                        # the main/GUI thread (Qt requirement).
                        with _headshot_lock:
                            _espn_headshot_data[self.url] = resp.content
                except Exception:
                    pass

        self._pool.start(_HeadshotRunnable(url, size))

    def _resolve_team_id(self, abbr: str) -> Optional[int]:
        """Lookup NBA team_id from abbreviation with per-session cache.

        Automatically translates ESPN abbreviations (GS, SA, NY, etc.)
        to NBA-standard abbreviations (GSW, SAS, NYK, etc.) before lookup.
        """
        nba_abbr = normalize_espn_abbr(abbr)
        if nba_abbr in self._team_id_cache:
            return self._team_id_cache[nba_abbr]
        try:
            from src.database import db
            row = db.fetch_one(
                "SELECT team_id FROM teams WHERE abbreviation = ?", (nba_abbr,)
            )
            tid = row["team_id"] if row else None
        except Exception:
            tid = None
        self._team_id_cache[nba_abbr] = tid
        # Also cache the original ESPN abbreviation so repeat lookups are fast
        if abbr != nba_abbr:
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

    def _refresh_for_headshots(self):
        """One-shot delayed refresh so background-downloaded headshots appear."""
        self._headshot_timer_active = False
        data = getattr(self, '_last_data', None)
        if data:
            self._apply_data(data)

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
            is_active = ath.get("active", False)
            athlete_info = ath.get("athlete", {})
            name = athlete_info.get("displayName", "")
            
            # Active players styling
            if is_active:
                name_display = f"🟢 {name}"
            else:
                name_display = name

            player_id = athlete_info.get("id", "")
            stats = ath.get("stats", [])
            stat_map = dict(zip(labels, stats)) if len(labels) == len(stats) else {}

            table.setRowHeight(r, 30)

            # Column 0: Player photo (28px)
            photo_item = QTableWidgetItem()
            if player_id:
                try:
                    pixmap = None
                    headshot_url = athlete_info.get("headshot", {}).get("href", "")

                    # Try to load from local NBA cache first
                    try:
                        from src.database import db
                        row = db.fetch_one("SELECT player_id FROM players WHERE name = ?", (name,))
                        local_pid = row["player_id"] if row else None
                        if local_pid:
                            from src.ui.widgets.image_utils import get_player_photo
                            pixmap = get_player_photo(local_pid, 28, circle=True)
                    except Exception:
                        pass

                    # Tier 1: main-thread pixmap cache (already converted)
                    if not pixmap and headshot_url:
                        with _headshot_lock:
                            pixmap = _espn_headshot_cache.get((headshot_url, 28))

                    # Tier 2: raw bytes downloaded in background → convert on main thread
                    if not pixmap and headshot_url:
                        with _headshot_lock:
                            raw_bytes = _espn_headshot_data.get(headshot_url)
                        if raw_bytes:
                            from src.ui.widgets.image_utils import _make_circle_pixmap
                            img = QImage()
                            img.loadFromData(raw_bytes)
                            if not img.isNull():
                                pix = QPixmap.fromImage(img).scaled(
                                    28, 28,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation,
                                )
                                pixmap = _make_circle_pixmap(pix)
                                with _headshot_lock:
                                    _espn_headshot_cache[(headshot_url, 28)] = pixmap

                    if pixmap:
                        photo_item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
                    else:
                        from src.ui.widgets.image_utils import get_player_photo
                        # Try to load a generic fallback, say a transparent pixel, or ignore
                        pass
                    if not pixmap and headshot_url:
                        # Queue background download; appears on next refresh
                        self._queue_headshot_fetch(headshot_url, 28)
                        self._needs_headshot_refresh = True
                except Exception:
                    pass
            table.setItem(r, 0, photo_item)

            # Column 1: Name
            table.setItem(r, 1, QTableWidgetItem(name_display))

            # Columns 2-11: Stats
            stat_keys = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "FG", "3PT", "PF", "+/-"]
            for c, key in enumerate(stat_keys):
                val = stat_map.get(key, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if key == "PF" and val and str(val).isdigit() and int(val) > 0:
                    item.setForeground(QColor("#ef4444"))
                table.setItem(r, c + 2, item)
