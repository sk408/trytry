"""Desktop Gamecast tab – live game detail with in-game predictions.

Shows game selector, live score, blended prediction panel,
odds, box score, and play-by-play feed.  Auto-refreshes via QTimer.

All network I/O and heavy computation runs in a background thread to
keep the UI responsive.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


# ── Pacific time helper ───────────────────────────────────────────────

from datetime import datetime as _datetime, timezone as _tz

def _to_pacific_str(iso_utc: str) -> str:
    """Convert an ISO-8601 UTC timestamp (e.g. ESPN ``date`` field) to a
    short Pacific-time string like ``7:30 PM PST``.

    Falls back gracefully if the string can't be parsed.
    """
    if not iso_utc:
        return "TBD"
    try:
        cleaned = iso_utc.replace("Z", "+00:00")
        dt = _datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz.utc)
        # zoneinfo (Python 3.9+)
        try:
            from zoneinfo import ZoneInfo
            pacific = dt.astimezone(ZoneInfo("America/Los_Angeles"))
        except (ImportError, KeyError):
            # manual fallback: PST = UTC-8, PDT ≈ Mar-Nov = UTC-7
            from datetime import timedelta
            utc_dt = dt.astimezone(_tz.utc)
            off = -7 if 3 <= utc_dt.month <= 11 else -8
            pacific = utc_dt + timedelta(hours=off)
        return pacific.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return iso_utc[:16] if len(iso_utc) >= 16 else iso_utc


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
        if _is_allstar_game(game):
            # All-Star exhibition — use custom roster-based prediction
            try:
                result.prediction = _allstar_predict(
                    home_name=game.home_team,
                    away_name=game.away_team,
                    home_score=game.home_score,
                    away_score=game.away_score,
                    period=game.period,
                    clock=game.clock,
                )
                if game.status == "final":
                    lp = result.prediction
                    result.postgame_data = {
                        "pg_home": lp.pregame_home,
                        "pg_away": lp.pregame_away,
                        "pg_total": lp.pregame_total,
                        "pg_spread": lp.pregame_spread,
                        "home_id": 0, "away_id": 0,
                    }
                    result.prediction = None
            except Exception as exc:
                result.prediction_error = f"All-Star: {exc}"
        else:
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


class _AllGamesRefreshWorker(QObject):
    """Pre-fetches ESPN data for ALL games on the scoreboard.

    Fetches odds + box score for every game in parallel (using a small
    thread pool).  Play-by-play and live-prediction are computed only for
    the currently selected game.  This means the result cache always has
    fresh odds/box for every game so switching is instant.
    """
    finished = Signal(list, dict)  # (List[GameInfo], {game_id: _PollResult})

    def __init__(
        self,
        selected_game_id: Optional[str],
        play_id_cache: dict,
        skip_game_ids: Optional[set] = None,
    ):
        super().__init__()
        self.selected_game_id = selected_game_id or ""
        self.play_id_cache = dict(play_id_cache)  # snapshot
        self.skip_game_ids = skip_game_ids or set()

    # -- per-game fetch (runs inside thread-pool) -------------------------

    def _fetch_game(self, game: GameInfo) -> tuple:
        """Fetch ESPN data for one game; returns (game_id, _PollResult).

        Returns ``(game_id, None)`` for games that don't need re-fetching:
        * **Final** games whose data will never change.
        * **Scheduled** games that have already been fetched once (odds
          won't move until tipoff, and there's no box/plays yet).
        """
        if game.game_id in self.skip_game_ids:
            return game.game_id, None

        result = _PollResult(game=game)
        is_selected = game.game_id == self.selected_game_id

        # Odds
        try:
            result.odds = get_game_odds(game.game_id)
        except Exception:
            pass

        # Box score
        try:
            result.box = get_box_score(game.game_id)
        except Exception:
            pass

        # Play-by-play — only for the selected game (saves API calls)
        if is_selected:
            last_pid = self.play_id_cache.get(game.game_id, "")
            try:
                result.plays = get_play_by_play(game.game_id, last_pid)
            except Exception:
                pass
            if game.status == "in_progress" and result.plays:
                try:
                    result.all_plays = get_play_by_play(game.game_id, "")
                except Exception:
                    pass

            # Prediction — only for the selected game
            self._compute_prediction(result, game)

        return game.game_id, result

    @staticmethod
    def _compute_prediction(result: _PollResult, game: GameInfo) -> None:
        if _is_allstar_game(game):
            try:
                result.prediction = _allstar_predict(
                    home_name=game.home_team,
                    away_name=game.away_team,
                    home_score=game.home_score,
                    away_score=game.away_score,
                    period=game.period,
                    clock=game.clock,
                )
                if game.status == "final":
                    lp = result.prediction
                    result.postgame_data = {
                        "pg_home": lp.pregame_home,
                        "pg_away": lp.pregame_away,
                        "pg_total": lp.pregame_total,
                        "pg_spread": lp.pregame_spread,
                        "home_id": 0, "away_id": 0,
                    }
                    result.prediction = None
            except Exception as exc:
                result.prediction_error = f"All-Star: {exc}"
            return

        home_id = _abbr_to_team_id(game.home_abbr)
        away_id = _abbr_to_team_id(game.away_abbr)
        if not (home_id and away_id):
            result.prediction_error = "teams not in DB"
            return
        game_date = game.start_time[:10] if game.start_time else ""
        if game.status == "final":
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

    # -- entry point ------------------------------------------------------

    def run(self) -> None:
        try:
            games = get_live_games()
        except Exception:
            self.finished.emit([], {})
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_results: dict[str, _PollResult] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._fetch_game, g): g for g in games}
            for fut in as_completed(futures):
                try:
                    gid, result = fut.result()
                    if result is not None:  # None = skipped final game
                        all_results[gid] = result
                except Exception:
                    pass

        self.finished.emit(games, all_results)


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


def _abbr_to_team_id(abbr: str) -> Optional[int]:
    """Resolve an ESPN abbreviation to our internal team_id.

    Handles known ESPN ↔ NBA-API abbreviation mismatches (e.g. GS → GSW).
    Uses ``team_cache`` — zero DB calls after the first lookup.
    """
    from src.analytics.cache import team_cache
    canonical = _ESPN_TO_NBA_ABBR.get(abbr, abbr)
    return team_cache.get_id(canonical)


def _bold(text: str) -> str:
    return f"<b>{text}</b>"


# ── All-Star Game prediction helpers ──────────────────────────────────

# Map ESPN team display names / abbreviations to our ASG_TEAMS roster keys.
# ESPN typically uses names like "World", "Team Stars", "Team Stripes",
# and abbreviations like "WLD", "STR", "STP" (or variations).
_ALLSTAR_NAME_MAP: Dict[str, str] = {
    # Full display names
    "world": "Team World",
    "team world": "Team World",
    "stars": "USA Stars (Young)",
    "team stars": "USA Stars (Young)",
    "usa stars": "USA Stars (Young)",
    "stripes": "USA Stripes (Veterans)",
    "team stripes": "USA Stripes (Veterans)",
    "usa stripes": "USA Stripes (Veterans)",
    # Common ESPN abbreviations
    "wld": "Team World",
    "str": "USA Stars (Young)",
    "stp": "USA Stripes (Veterans)",
}


def _is_allstar_game(game: GameInfo) -> bool:
    """Return True if this game appears to be an All-Star exhibition."""
    names = (
        game.home_team.lower(),
        game.away_team.lower(),
        game.home_abbr.lower(),
        game.away_abbr.lower(),
    )
    keywords = ("world", "stars", "stripes", "all-star", "allstar",
                "wld", "str", "stp")
    return any(kw in n for n in names for kw in keywords)


def _resolve_allstar_team(name_or_abbr: str) -> Optional[str]:
    """Map an ESPN team name/abbreviation to an ASG_TEAMS key."""
    low = name_or_abbr.lower().strip()
    if low in _ALLSTAR_NAME_MAP:
        return _ALLSTAR_NAME_MAP[low]
    # Partial match
    for key, roster_key in _ALLSTAR_NAME_MAP.items():
        if key in low or low in key:
            return roster_key
    return None


def _allstar_predict(
    home_name: str,
    away_name: str,
    home_score: int,
    away_score: int,
    period: int,
    clock: Optional[str],
) -> LivePrediction:
    """Compute a prediction for an All-Star exhibition game.

    Uses player stats from the database rather than team-level models,
    since All-Star teams don't exist in our ``teams`` table.
    Each round-robin game is 12 minutes, so projected totals are scaled.
    """
    from src.ui.allstar_view import ASG_TEAMS, _find_player_ids, _load_player_stats

    home_key = _resolve_allstar_team(home_name) or _resolve_allstar_team(
        home_name.split()[-1] if home_name else ""
    )
    away_key = _resolve_allstar_team(away_name) or _resolve_allstar_team(
        away_name.split()[-1] if away_name else ""
    )

    # Fallback if we can't map — return zeroed prediction
    if not home_key or not away_key:
        return LivePrediction(
            projected_home_score=0, projected_away_score=0,
            projected_total=0, projected_spread=0,
            pregame_home=0, pregame_away=0, pregame_total=0, pregame_spread=0,
        )

    # Load rosters and stats (recent 15 games for recency)
    home_roster = ASG_TEAMS.get(home_key, [])
    away_roster = ASG_TEAMS.get(away_key, [])
    home_ids = _find_player_ids(home_roster)
    away_ids = _find_player_ids(away_roster)
    home_stats = _load_player_stats(home_ids, recent_n=15)
    away_stats = _load_player_stats(away_ids, recent_n=15)

    def _team_strength(stats: list) -> float:
        if not stats:
            return 0
        n = len(stats)
        ppg = sum(s.get("ppg", 0) for s in stats) / n
        fg = sum(s.get("fg_pct", 0) for s in stats) / n
        fg3 = sum(s.get("fg3_pct", 0) for s in stats) / n
        apg = sum(s.get("apg", 0) for s in stats) / n
        rpg = sum(s.get("rpg", 0) for s in stats) / n
        pm = sum(s.get("plus_minus", 0) for s in stats) / n
        return ppg * 2.0 + fg3 * 0.5 + fg * 0.3 + apg * 1.5 + rpg * 0.8 + pm * 1.0

    h_str = _team_strength(home_stats)
    a_str = _team_strength(away_stats)
    total_str = h_str + a_str if (h_str + a_str) > 0 else 1.0

    # All-Star round-robin games are 12 minutes (one quarter).
    # All-Star pace is typically much higher (~3.0 PPM per team vs ~2.3 regular).
    GAME_MINUTES = 12.0
    ALLSTAR_PPM = 3.0  # points per minute per team (All-Star pace)
    expected_total = GAME_MINUTES * ALLSTAR_PPM * 2  # ~72 combined for 12-min game

    h_proj = expected_total * (h_str / total_str)
    a_proj = expected_total * (a_str / total_str)
    pregame_spread = h_proj - a_proj

    # If the game is in progress, blend with pace extrapolation
    elapsed = 0.0
    if period > 0 and clock:
        try:
            parts = clock.split(":")
            remaining = int(parts[0]) * 60 + int(parts[1])
            elapsed = GAME_MINUTES - remaining / 60.0
        except Exception:
            elapsed = 0.0

    if elapsed > 0.5:  # at least 30 seconds played
        game_frac = elapsed / GAME_MINUTES
        # Pace projection from actual scores
        pace_h = home_score / game_frac if game_frac > 0 else h_proj
        pace_a = away_score / game_frac if game_frac > 0 else a_proj
        # Blend: pregame fades, pace grows
        live_wt = min(0.95, game_frac * 1.2)
        pre_wt = 1.0 - live_wt
        final_h = pre_wt * h_proj + live_wt * pace_h
        final_a = pre_wt * a_proj + live_wt * pace_a
    else:
        final_h = h_proj
        final_a = a_proj
        live_wt = 0.0
        pre_wt = 1.0

    proj_total = final_h + final_a
    proj_spread = final_h - final_a
    h_win_prob = h_str / total_str

    # Signals
    if proj_spread > 3:
        spread_sig = f"Home ({home_key.split('(')[0].strip()}) favored"
    elif proj_spread < -3:
        spread_sig = f"Away ({away_key.split('(')[0].strip()}) favored"
    else:
        spread_sig = "Close matchup"

    return LivePrediction(
        projected_home_score=round(final_h, 1),
        projected_away_score=round(final_a, 1),
        projected_total=round(proj_total, 1),
        projected_spread=round(proj_spread, 1),
        pregame_home=round(h_proj, 1),
        pregame_away=round(a_proj, 1),
        pregame_total=round(h_proj + a_proj, 1),
        pregame_spread=round(pregame_spread, 1),
        pace_home=round(final_h, 1),
        pace_away=round(final_a, 1),
        pace_total=round(proj_total, 1),
        pace_spread=round(proj_spread, 1),
        minutes_elapsed=elapsed,
        blend_weights={"pregame": pre_wt, "pace": live_wt, "quarter": 0.0},
        spread_signal=spread_sig,
        over_under_signal=(
            f"Win prob: {h_win_prob * 100:.0f}% / {(1 - h_win_prob) * 100:.0f}%"
        ),
    )


# ── main widget ──────────────────────────────────────────────────────

class GamecastView(QWidget):
    """Full gamecast tab for the desktop application."""

    _POLL_LIVE_MS = 20_000   # 20 s when any live games on the board
    _POLL_IDLE_MS = 120_000  # 2 min when all games are final / scheduled
    _BG_THREAD_TIMEOUT_S = 90  # abandon bg thread if running longer than this

    def __init__(self) -> None:
        super().__init__()

        # ── state ──
        self._current_game: Optional[GameInfo] = None
        self._last_play_id = ""
        self._games_thread: Optional[QThread] = None
        self._games_worker: Optional[_GamesListWorker] = None
        self._poll_thread: Optional[QThread] = None
        self._poll_worker: Optional[_PollWorker] = None
        # Background all-games poller
        self._bg_thread: Optional[QThread] = None
        self._bg_worker: Optional[_AllGamesRefreshWorker] = None
        self._bg_started_at: float = 0.0  # monotonic timestamp
        # Orphaned threads whose signals have been disconnected but whose
        # OS threads are still running.  Preventing GC from destroying
        # a QThread while its OS thread is active avoids the
        # "QThread: Destroyed while thread is still running" crash.
        self._orphaned_threads: list = []
        # Per-game result cache: show the last-known state instantly when
        # switching back to a game, while fresh data loads in background.
        self._result_cache: dict[str, _PollResult] = {}
        self._play_id_cache: dict[str, str] = {}  # game_id -> last_play_id
        self._plays_cache: dict[str, List[PlayEvent]] = {}  # game_id -> all plays (newest first)

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

        # ── auto-refresh timer (pre-fetches ALL games) ──
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_all_games)
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

        # Abandon previous refresh if still running
        self._abandon_games_thread()

        self._games_thread = QThread()
        self._games_worker = _GamesListWorker()
        self._games_worker.moveToThread(self._games_thread)
        self._games_thread.started.connect(self._games_worker.run)
        self._games_worker.finished.connect(self._on_games_loaded)
        self._games_worker.finished.connect(self._games_thread.quit)
        self._games_thread.finished.connect(self._cleanup_games_thread)
        self._games_thread.start()

    def _abandon_games_thread(self) -> None:
        """Disconnect the current games-list worker/thread so stale
        signals are ignored, then let the thread finish naturally."""
        if self._games_worker is not None:
            try:
                self._games_worker.finished.disconnect(self._on_games_loaded)
            except RuntimeError:
                pass
        if self._games_thread is not None:
            try:
                self._games_thread.finished.disconnect(self._cleanup_games_thread)
            except RuntimeError:
                pass
            # Stash the thread so GC can't destroy it while still running.
            self._stash_orphan(self._games_thread, self._games_worker)
        self._games_worker = None
        self._games_thread = None

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
                period_lbl = "P" if _is_allstar_game(g) else "Q"
                tag = f"LIVE {period_lbl}{g.period} {g.clock} ({g.away_score}-{g.home_score})"
            elif g.status == "final":
                tag = f"FINAL ({g.away_score}-{g.home_score})"
            else:
                tag = _to_pacific_str(g.start_time) if g.start_time else "Upcoming"
            prefix = "[ASG] " if _is_allstar_game(g) else ""
            label = f"{prefix}{g.away_abbr} @ {g.home_abbr} — {tag}"
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
                    break

        # Immediately pre-fetch ESPN data for all games so the cache is
        # warm by the time the user clicks a game.
        self._poll_all_games()

    def _cleanup_games_thread(self) -> None:
        thread = self.sender()
        if thread is not None and thread is not self._games_thread:
            thread.deleteLater()   # stale signal from an abandoned thread
            return
        if self._games_worker:
            self._games_worker.deleteLater()
            self._games_worker = None
        if self._games_thread:
            self._games_thread.deleteLater()
            self._games_thread = None

    # ── orphan management ──────────────────────────────────────────

    def _stash_orphan(self, thread: QThread, worker: Optional[QObject] = None) -> None:
        """Keep *thread* (and its *worker*) alive until the thread finishes.

        Without this, setting ``self._xxx_thread = None`` lets Python's
        GC destroy the underlying C++ QThread while its OS thread is
        still running, which triggers the fatal
        ``QThread: Destroyed while thread is still running`` message.

        We connect the thread's ``finished`` signal so both objects
        call ``deleteLater()`` once the thread actually stops, and we
        purge already-finished orphans every time a new one is stashed.
        """
        if worker is not None:
            thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._orphaned_threads.append(thread)
        if worker is not None:
            self._orphaned_threads.append(worker)
        # Purge finished orphans to prevent unbounded growth.
        # Workers get deleteLater'd when their thread finishes, so
        # we only need to keep threads (and their paired workers)
        # while the thread is alive.
        alive: list = []
        for obj in self._orphaned_threads:
            if isinstance(obj, QThread):
                if obj.isRunning():
                    alive.append(obj)
            # Keep workers only while their thread is still alive.
            # Since thread.finished → worker.deleteLater, once the
            # thread finishes the worker is already queued for
            # deletion and can be dropped from this list.
        self._orphaned_threads = alive

    def _on_game_selected(self, index: int) -> None:
        if index <= 0:
            self._current_game = None
            self._clear_detail()
            self._timer.setInterval(self._POLL_IDLE_MS)
            return
        game = self.game_combo.itemData(index)
        if not game:
            return

        # Save outgoing game's play-by-play position
        if self._current_game and self._last_play_id:
            self._play_id_cache[self._current_game.game_id] = self._last_play_id

        # Abandon in-flight workers for the *previous* game so the new
        # game can start loading immediately and stale bg results don't
        # overwrite the new game's data.
        self._abandon_poll_worker()
        self._abandon_bg_thread()

        self._current_game = game
        # Reset play position — _render_cached_result will set it from the
        # plays cache if one exists, otherwise the worker fetches everything.
        self._last_play_id = ""
        interval = self._POLL_LIVE_MS if game.status == "in_progress" else self._POLL_IDLE_MS
        self._timer.setInterval(interval)
        self._load_game_detail(game)

        # Kick off a fresh bg poll immediately so the new game gets its
        # prediction computed right away instead of waiting for the timer.
        self._poll_all_games()

    # ────────────────────────────────────────────────────────────────
    # Detail loading (background thread)
    # ────────────────────────────────────────────────────────────────

    def _load_game_detail(self, game: GameInfo) -> None:
        self._current_game = game

        # Clear play-by-play from the previous game
        self.play_list.clear()

        # Update score immediately (no I/O, just labels)
        self._update_score(game)

        # If we've seen this game before, render the cached result
        # instantly so there's zero delay.  The background thread will
        # refresh everything and replace stale data when it arrives.
        cached = self._result_cache.get(game.game_id)
        if cached:
            self._render_cached_result(cached, game)

        # Kick off background fetch for fresh data
        self._start_poll_worker(game)

    def _abandon_poll_worker(self) -> None:
        """Disconnect the current poll worker so its result is ignored,
        then let its thread finish naturally (no data race)."""
        if self._poll_worker is not None:
            try:
                self._poll_worker.finished.disconnect(self._on_poll_result)
            except RuntimeError:
                pass  # already disconnected
        if self._poll_thread is not None:
            try:
                self._poll_thread.finished.disconnect(self._cleanup_poll)
            except RuntimeError:
                pass
            # Stash the thread so GC can't destroy it while still running.
            self._stash_orphan(self._poll_thread, self._poll_worker)
        self._poll_worker = None
        self._poll_thread = None

    def _start_poll_worker(self, game: GameInfo) -> None:
        """Launch a background thread to fetch odds/box/plays/prediction."""
        # If an old worker is still running, abandon it (ignore its results)
        if self._poll_thread is not None and self._poll_thread.isRunning():
            self._abandon_poll_worker()

        self._poll_thread = QThread()
        self._poll_worker = _PollWorker(game, self._last_play_id)
        self._poll_worker.moveToThread(self._poll_thread)
        self._poll_thread.started.connect(self._poll_worker.run)
        self._poll_worker.finished.connect(self._on_poll_result)
        self._poll_worker.finished.connect(self._poll_thread.quit)
        self._poll_thread.finished.connect(self._cleanup_poll)
        self._poll_thread.start()

    def _cleanup_poll(self) -> None:
        thread = self.sender()
        if thread is not None and thread is not self._poll_thread:
            thread.deleteLater()   # stale signal from an abandoned thread
            return
        if self._poll_worker:
            self._poll_worker.deleteLater()
            self._poll_worker = None
        if self._poll_thread:
            self._poll_thread.deleteLater()
            self._poll_thread = None

    def _render_cached_result(self, result: _PollResult, game: GameInfo) -> None:
        """Instantly display previously cached data for a game.

        Renders prediction, odds, box score, and play-by-play from cache
        so the user sees content immediately.  The background thread will
        refresh everything and add any new plays when it arrives.
        """
        if result.prediction:
            self._render_prediction(result.prediction, game)
        elif result.postgame_data:
            self._render_final_analysis_data(result.postgame_data, game)
        elif result.prediction_error:
            self.pregame_lbl.setText(f"Pre-game: {result.prediction_error}")
        self._apply_odds(result.odds, game)
        self._fill_box_table(
            self.home_box_table,
            result.box.home_players if result.box else [],
        )
        self._fill_box_table(
            self.away_box_table,
            result.box.away_players if result.box else [],
        )

        # Restore cached play-by-play so the list isn't empty while the
        # background thread fetches fresh / incremental plays.
        cached_plays = self._plays_cache.get(game.game_id, [])
        if cached_plays:
            self._apply_plays(cached_plays, [], game)

    def _on_poll_result(self, result: _PollResult) -> None:
        """Apply fetched data to the UI (runs on main thread)."""
        game = result.game
        if not game:
            return
        # Ignore stale results if user already switched to another game
        if self._current_game and game.game_id != self._current_game.game_id:
            # Still cache it — useful if the user switches back
            self._result_cache[game.game_id] = result
            return

        # Cache for instant display when switching back to this game
        self._result_cache[game.game_id] = result

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

        is_asg = _is_allstar_game(game)
        if game.status == "in_progress":
            period_lbl = f"P{game.period}" if is_asg else f"Q{game.period}"
            self.status_lbl.setText(
                f"<b style='color:#e74c3c;'>LIVE</b><br>{period_lbl} {game.clock}"
            )
        elif game.status == "final":
            self.status_lbl.setText("<b>FINAL</b>")
        else:
            self.status_lbl.setText("All-Star" if is_asg else "Upcoming")

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
        """Update play-by-play list from pre-fetched data (no I/O).

        Also accumulates plays into ``_plays_cache`` so they can be
        rendered instantly when the user switches back to this game.
        """
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

        # Accumulate plays into the per-game cache (newest first).
        # ``plays`` from the API is newest-first; prepend to existing cache.
        gid = game.game_id
        existing = self._plays_cache.get(gid, [])
        self._plays_cache[gid] = (list(plays) + existing)[:200]

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

    # ── all-games background poller ──

    def _poll_all_games(self) -> None:
        """Timer-driven: pre-fetch ESPN data for every game on the board.

        Runs the all-games worker in a background thread.  Results update
        ``_result_cache`` for *all* games so switching is always instant.
        Completed games that already have cached data are skipped entirely.
        """
        if self._bg_thread is not None and self._bg_thread.isRunning():
            # If the thread has been running too long, abandon it to
            # prevent a permanently blocked refresh cycle.
            elapsed = _time.monotonic() - self._bg_started_at
            if elapsed < self._BG_THREAD_TIMEOUT_S:
                return  # previous cycle still running — skip
            self._abandon_bg_thread()

        # Build play-ID snapshot including the current game's position
        play_ids = dict(self._play_id_cache)
        if self._current_game and self._last_play_id:
            play_ids[self._current_game.game_id] = self._last_play_id

        # Games we can skip hitting ESPN for:
        #  - Final games with cached data (score/odds/box will never change)
        #  - Scheduled games already fetched once (no box/plays; odds are
        #    static until tipoff — one fetch is enough)
        skip_ids: set[str] = set()
        for gid, r in self._result_cache.items():
            if not r.game:
                continue
            if r.game.status == "final" and (r.odds is not None or r.box is not None):
                skip_ids.add(gid)
            elif r.game.status == "scheduled":
                skip_ids.add(gid)

        self._bg_thread = QThread()
        self._bg_worker = _AllGamesRefreshWorker(
            selected_game_id=(
                self._current_game.game_id if self._current_game else None
            ),
            play_id_cache=play_ids,
            skip_game_ids=skip_ids,
        )
        self._bg_worker.moveToThread(self._bg_thread)
        self._bg_thread.started.connect(self._bg_worker.run)
        self._bg_worker.finished.connect(self._on_all_games_result)
        self._bg_worker.finished.connect(self._bg_thread.quit)
        self._bg_thread.finished.connect(self._cleanup_bg)
        self._bg_started_at = _time.monotonic()
        self._bg_thread.start()

    def _on_all_games_result(
        self, games: list, all_results: dict,
    ) -> None:
        """Process pre-fetched data for every game."""
        if not games and not all_results:
            return

        # Detect status transitions (scheduled → in_progress, etc.).
        # If a game was skipped because our cache said "scheduled" but the
        # fresh scoreboard says "in_progress", evict the stale cache so the
        # next poll cycle fetches real data for it.
        for g in games:
            cached = self._result_cache.get(g.game_id)
            if (
                cached and cached.game
                and cached.game.status == "scheduled"
                and g.status != "scheduled"
            ):
                # Game just tipped off — drop the stale entry so it's
                # no longer in the skip set on the next cycle.
                del self._result_cache[g.game_id]

        # 1. Cache results for ALL games (instant switching)
        self._result_cache.update(all_results)

        # 2. Update combo box labels with fresh scores / statuses
        self._update_combo_from_scoreboard(games)

        # 3. Adjust timer based on whether any games are live
        has_live = any(g.status == "in_progress" for g in games)
        self._timer.setInterval(
            self._POLL_LIVE_MS if has_live else self._POLL_IDLE_MS,
        )

        # 4. Render the selected game's fresh result
        if self._current_game:
            result = all_results.get(self._current_game.game_id)
            if result and result.game:
                self._current_game = result.game
                self._on_poll_result(result)

    def _update_combo_from_scoreboard(self, games: list) -> None:
        """Update combo text & stored data with fresh scores (no rebuild)."""
        game_map = {g.game_id: g for g in games}
        self.game_combo.blockSignals(True)
        for i in range(1, self.game_combo.count()):
            data = self.game_combo.itemData(i)
            if not data or not hasattr(data, "game_id"):
                continue
            updated = game_map.get(data.game_id)
            if not updated:
                continue
            # Refresh the stored GameInfo object
            self.game_combo.setItemData(i, updated)
            # Rebuild the label
            if updated.status == "in_progress":
                plbl = "P" if _is_allstar_game(updated) else "Q"
                tag = (
                    f"LIVE {plbl}{updated.period} {updated.clock} "
                    f"({updated.away_score}-{updated.home_score})"
                )
            elif updated.status == "final":
                tag = f"FINAL ({updated.away_score}-{updated.home_score})"
            else:
                tag = (
                    _to_pacific_str(updated.start_time)
                    if updated.start_time else "Upcoming"
                )
            prefix = "[ASG] " if _is_allstar_game(updated) else ""
            self.game_combo.setItemText(
                i, f"{prefix}{updated.away_abbr} @ {updated.home_abbr} — {tag}",
            )
        self.game_combo.blockSignals(False)

    def _abandon_bg_thread(self) -> None:
        """Disconnect the current bg worker/thread so stale results
        are ignored, then let the thread finish naturally."""
        if self._bg_worker is not None:
            try:
                self._bg_worker.finished.disconnect(self._on_all_games_result)
            except RuntimeError:
                pass
        if self._bg_thread is not None:
            try:
                self._bg_thread.finished.disconnect(self._cleanup_bg)
            except RuntimeError:
                pass
            self._stash_orphan(self._bg_thread, self._bg_worker)
        self._bg_worker = None
        self._bg_thread = None

    def _cleanup_bg(self) -> None:
        thread = self.sender()
        if thread is not None and thread is not self._bg_thread:
            thread.deleteLater()   # stale signal from an abandoned thread
            return
        if self._bg_worker:
            self._bg_worker.deleteLater()
            self._bg_worker = None
        if self._bg_thread:
            self._bg_thread.deleteLater()
            self._bg_thread = None
