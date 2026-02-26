"""Core prediction engine: predict_matchup(), MatchupPrediction, PrecomputedGame."""

import hashlib
import logging
import os
import pickle
from dataclasses import dataclass, field, fields as dc_fields
from typing import Dict, Any, Optional, List, Tuple

from src.database import db
from src.config import get_season
from src.analytics.weight_config import (
    WeightConfig, get_weight_config, load_team_weights,
)
from src.analytics.stats_engine import (
    aggregate_projection, get_home_court_advantage, compute_fatigue,
    compute_shooting_efficiency, _LEAGUE_AVG_PPG, _PACE_FALLBACK, _RATING_FALLBACK,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchupPrediction:
    """Full prediction result."""
    home_team_id: int = 0
    away_team_id: int = 0
    home_team: str = ""
    away_team: str = ""
    game_date: str = ""
    predicted_spread: float = 0.0
    predicted_total: float = 0.0
    predicted_home_score: float = 0.0
    predicted_away_score: float = 0.0
    home_court_advantage: float = 0.0
    home_fatigue: float = 0.0
    away_fatigue: float = 0.0
    fatigue_adj: float = 0.0
    turnover_adj: float = 0.0
    rebound_adj: float = 0.0
    rating_matchup_adj: float = 0.0
    four_factors_adj: float = 0.0
    clutch_adj: float = 0.0
    hustle_adj: float = 0.0
    espn_blend_adj: float = 0.0
    ml_blend_adj: float = 0.0
    pace_adj: float = 0.0
    defensive_disruption: float = 0.0
    oreb_boost: float = 0.0
    hustle_total_adj: float = 0.0
    home_proj: Dict[str, float] = field(default_factory=dict)
    away_proj: Dict[str, float] = field(default_factory=dict)
    adjustments: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    winner: str = ""
    ml_spread: float = 0.0
    ml_total: float = 0.0
    espn_home_pct: float = 50.0
    injury_impact: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrecomputedGame:
    """Zero-DB-access replay structure for optimization."""
    game_date: str = ""
    home_team_id: int = 0
    away_team_id: int = 0
    actual_home_score: float = 0.0
    actual_away_score: float = 0.0
    home_proj: Dict[str, float] = field(default_factory=dict)
    away_proj: Dict[str, float] = field(default_factory=dict)
    home_court: float = 3.0
    away_def_factor_raw: float = 1.0
    home_def_factor_raw: float = 1.0
    home_tuning_home_corr: float = 0.0
    away_tuning_away_corr: float = 0.0
    home_fatigue_penalty: float = 0.0
    away_fatigue_penalty: float = 0.0
    home_off: float = 110.0
    away_off: float = 110.0
    home_def: float = 110.0
    away_def: float = 110.0
    home_pace: float = 98.0
    away_pace: float = 98.0
    home_ff: Dict[str, float] = field(default_factory=dict)
    away_ff: Dict[str, float] = field(default_factory=dict)
    home_clutch: Dict[str, float] = field(default_factory=dict)
    away_clutch: Dict[str, float] = field(default_factory=dict)
    home_hustle: Dict[str, float] = field(default_factory=dict)
    away_hustle: Dict[str, float] = field(default_factory=dict)
    home_injured_count: float = 0.0
    away_injured_count: float = 0.0
    home_injury_ppg_lost: float = 0.0
    away_injury_ppg_lost: float = 0.0
    home_injury_minutes_lost: float = 0.0
    away_injury_minutes_lost: float = 0.0
    home_return_discount: float = 1.0
    away_return_discount: float = 1.0
    home_games_played: int = 0
    away_games_played: int = 0
    home_roster_changed: bool = False
    away_roster_changed: bool = False
    vegas_spread: float = 0.0
    vegas_home_ml: int = 0
    vegas_away_ml: int = 0
    spread_home_public: int = 0
    spread_home_money: int = 0


# ──────────────────────────────────────────────────────────────
# Precompute disk + memory cache
# ──────────────────────────────────────────────────────────────

_PRECOMPUTE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
_PRECOMPUTE_CACHE_FILE = os.path.join(_PRECOMPUTE_CACHE_DIR, "precomputed_games.pkl")
_CONTEXT_CACHE_FILE = os.path.join(_PRECOMPUTE_CACHE_DIR, "precompute_context.pkl")

# In-memory caches (persist across calls within the same process)
_mem_pc_cache: Optional[Dict[str, "PrecomputedGame"]] = None
_mem_pc_schema: Optional[str] = None
_mem_ctx_cache: Optional[Dict[str, Any]] = None
_mem_ctx_game_count: Optional[int] = None


def _precompute_schema_version() -> str:
    """Hash of PrecomputedGame field names — auto-invalidates when fields change."""
    names = tuple(f.name for f in dc_fields(PrecomputedGame))
    return hashlib.md5(str(names).encode()).hexdigest()[:12]


def _game_cache_key(home_team_id: int, away_team_id: int, game_date: str) -> str:
    """Unique key for a game (NBA teams can't play twice on the same date)."""
    return f"{home_team_id}_{away_team_id}_{game_date}"


def _load_pc_cache() -> Dict[str, "PrecomputedGame"]:
    """Load precompute cache from memory or disk. Returns empty dict on miss."""
    global _mem_pc_cache, _mem_pc_schema
    schema = _precompute_schema_version()

    # In-memory hit
    if _mem_pc_cache is not None and _mem_pc_schema == schema:
        return _mem_pc_cache

    # Disk hit
    try:
        if os.path.exists(_PRECOMPUTE_CACHE_FILE):
            with open(_PRECOMPUTE_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if data.get("schema") == schema:
                _mem_pc_cache = data["games"]
                _mem_pc_schema = schema
                logger.info("Loaded precompute cache from disk (%d games)", len(_mem_pc_cache))
                return _mem_pc_cache
            else:
                logger.info("Precompute cache schema mismatch — will rebuild")
    except Exception as e:
        logger.warning("Failed to load precompute cache: %s", e)

    return {}


def _save_pc_cache(cache: Dict[str, "PrecomputedGame"]):
    """Persist precompute cache to disk and update in-memory copy."""
    global _mem_pc_cache, _mem_pc_schema
    schema = _precompute_schema_version()
    os.makedirs(_PRECOMPUTE_CACHE_DIR, exist_ok=True)
    try:
        with open(_PRECOMPUTE_CACHE_FILE, "wb") as f:
            pickle.dump({"schema": schema, "games": cache}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved precompute cache to disk (%d games)", len(cache))
    except Exception as e:
        logger.warning("Failed to save precompute cache: %s", e)
    _mem_pc_cache = cache
    _mem_pc_schema = schema


def invalidate_precompute_cache():
    """Clear all precompute caches (games + context, memory + disk)."""
    global _mem_pc_cache, _mem_pc_schema, _mem_ctx_cache, _mem_ctx_game_count
    _mem_pc_cache = None
    _mem_pc_schema = None
    _mem_ctx_cache = None
    _mem_ctx_game_count = None
    for path in (_PRECOMPUTE_CACHE_FILE, _CONTEXT_CACHE_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    logger.info("Invalidated all precompute caches")


from src.analytics.cache import team_cache


# ──────────────────────────────────────────────────────────────
# Precompute context: historical rosters + inferred injuries
# ──────────────────────────────────────────────────────────────

def _load_ctx_cache(game_count: int) -> Optional[Dict[str, Any]]:
    """Load precompute context from memory or disk if game count matches."""
    global _mem_ctx_cache, _mem_ctx_game_count
    if _mem_ctx_cache is not None and _mem_ctx_game_count == game_count:
        return _mem_ctx_cache
    try:
        if os.path.exists(_CONTEXT_CACHE_FILE):
            with open(_CONTEXT_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if data.get("game_count") == game_count:
                _mem_ctx_cache = data["ctx"]
                _mem_ctx_game_count = game_count
                logger.info("Loaded precompute context from disk (%d games)", game_count)
                return _mem_ctx_cache
    except Exception as e:
        logger.warning("Failed to load context cache: %s", e)
    return None


def _save_ctx_cache(ctx: Dict[str, Any], game_count: int):
    """Persist precompute context to disk."""
    global _mem_ctx_cache, _mem_ctx_game_count
    os.makedirs(_PRECOMPUTE_CACHE_DIR, exist_ok=True)
    try:
        with open(_CONTEXT_CACHE_FILE, "wb") as f:
            pickle.dump({"game_count": game_count, "ctx": ctx}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved precompute context to disk (%d games)", game_count)
    except Exception as e:
        logger.warning("Failed to save context cache: %s", e)
    _mem_ctx_cache = ctx
    _mem_ctx_game_count = game_count


def _build_precompute_context(games: List[Dict], force: bool = False) -> Dict[str, Any]:
    """Build lookup tables for historical roster + injury inference.

    One bulk SQL query determines which team each player was on for every
    game (handles trades automatically — team membership follows the data).
    Then for each game we can infer injuries by comparing the recent active
    roster vs who actually played.

    Cached to disk — only rebuilt when new games are added.
    Returns a context dict that is read-only and thread-safe.
    """
    from collections import defaultdict

    game_count = len(games)

    # Try cache first
    if not force:
        cached = _load_ctx_cache(game_count)
        if cached is not None:
            return cached

    # 1) game_id → (home_team_id, away_team_id) from actual results
    game_team_map: Dict[str, tuple] = {}
    for g in games:
        gid = g.get("game_id")
        if gid:
            game_team_map[gid] = (g["home_team_id"], g["away_team_id"])

    # 2) Player info (names, positions) — one query
    player_info: Dict[int, Dict] = {}
    for r in db.fetch_all("SELECT player_id, name, position FROM players"):
        player_info[r["player_id"]] = {
            "player_id": r["player_id"],
            "name": r["name"],
            "position": r["position"],
        }

    # 3) All player game appearances — the single expensive query
    rows = db.fetch_all("""
        SELECT player_id, game_id, game_date, is_home,
               points, minutes
        FROM player_stats
        ORDER BY game_date
    """)

    # team_game_players[(team_id, game_date)] = set of player_ids who played
    team_game_players: Dict[tuple, set] = defaultdict(set)
    # player season stats for injury impact estimation (full season — all data)
    player_season_stats: Dict[int, List[Dict]] = defaultdict(list)

    for r in rows:
        gid = r["game_id"]
        if gid not in game_team_map:
            continue

        htid, atid = game_team_map[gid]
        team_id = htid if r["is_home"] else atid
        pid = r["player_id"]
        gdate = r["game_date"]

        team_game_players[(team_id, gdate)].add(pid)
        player_season_stats[pid].append({
            "pts": r["points"] or 0,
            "mins": r["minutes"] or 0,
        })

    # 4) Team game dates (sorted) for binary search in injury inference
    team_dates: Dict[int, List[str]] = defaultdict(list)
    for (tid, gdate) in team_game_players:
        team_dates[tid].append(gdate)
    for tid in team_dates:
        team_dates[tid] = sorted(set(team_dates[tid]))

    # 5) Player average stats (full season — user wants all available data)
    player_avg: Dict[int, Dict[str, float]] = {}
    for pid, stats in player_season_stats.items():
        n = len(stats)
        if n > 0:
            player_avg[pid] = {
                "ppg": sum(s["pts"] for s in stats) / n,
                "mpg": sum(s["mins"] for s in stats) / n,
            }

    result = {
        "game_team_map": dict(game_team_map),
        "team_game_players": dict(team_game_players),
        "team_dates": dict(team_dates),
        "player_info": player_info,
        "player_avg": player_avg,
    }
    _save_ctx_cache(result, game_count)
    return result


def _get_historical_roster(team_id: int, game_date: str,
                           ctx: Dict[str, Any]) -> List[Dict]:
    """Get the roster for a team on a given date from precompute context.

    Uses actual game data — handles trades automatically because team
    membership is determined by which team a player appeared in game stats for.
    """
    pids = ctx["team_game_players"].get((team_id, game_date), set())
    info = ctx["player_info"]
    return [info.get(pid, {"player_id": pid, "name": "Unknown", "position": "F"})
            for pid in pids]


def _infer_historical_injuries(team_id: int, game_date: str,
                               ctx: Dict[str, Any]) -> Dict[str, float]:
    """Infer injuries for a historical game by comparing recent roster vs actual.

    If a player played in the last 5 games for this team but didn't play in
    THIS game, they were effectively injured/unavailable — a fact that was
    known at game time.  Uses full-season averages for impact estimation.
    """
    import bisect

    team_dates = ctx["team_dates"].get(team_id, [])
    tgp = ctx["team_game_players"]
    pavg = ctx["player_avg"]

    # Find last 5 game dates for this team BEFORE this game
    idx = bisect.bisect_left(team_dates, game_date)
    recent_dates = team_dates[max(0, idx - 5):idx]

    if not recent_dates:
        return {"injured_count": 0, "injury_ppg_lost": 0.0, "injury_minutes_lost": 0.0}

    # Expected roster: union of players from last 5 games
    expected = set()
    for d in recent_dates:
        expected |= tgp.get((team_id, d), set())

    # Actual roster: who played in THIS game
    actual = tgp.get((team_id, game_date), set())

    # Missing = inferred injured/unavailable
    missing = expected - actual

    count = 0
    ppg_lost = 0.0
    min_lost = 0.0
    for pid in missing:
        avg = pavg.get(pid, {})
        mpg = avg.get("mpg", 0)
        if mpg >= 10:  # Only count rotation players (10+ min/game)
            count += 1
            ppg_lost += avg.get("ppg", 0)
            min_lost += mpg

    return {
        "injured_count": count,
        "injury_ppg_lost": ppg_lost,
        "injury_minutes_lost": min_lost,
    }


def _get_team_metrics(team_id: int) -> Dict[str, float]:
    """Fetch team metrics as a flat dict, using cache."""
    cached = team_cache.get(team_id, "metrics")
    if cached is not None:
        return cached

    season = get_season()
    row = db.fetch_one(
        "SELECT * FROM team_metrics WHERE team_id = ? AND season = ?",
        (team_id, season)
    )
    result = dict(row) if row else {}
    team_cache.set(team_id, "metrics", result)
    return result


def _get_tuning(team_id: int) -> Dict[str, float]:
    """Get per-team autotune corrections, using cache."""
    cached = team_cache.get(team_id, "tuning")
    if cached is not None:
        return cached

    row = db.fetch_one(
        "SELECT home_pts_correction, away_pts_correction FROM team_tuning WHERE team_id = ?",
        (team_id,)
    )
    if row:
        result = {"home_pts_correction": row["home_pts_correction"],
                  "away_pts_correction": row["away_pts_correction"]}
    else:
        result = {"home_pts_correction": 0.0, "away_pts_correction": 0.0}
    team_cache.set(team_id, "tuning", result)
    return result


def _clamp(low: float, val: float, high: float) -> float:
    return max(low, min(high, val))


def _get_espn_predictor(home_abbr: str, away_abbr: str) -> Dict[str, float]:
    """Try to get ESPN predictor for the matchup."""
    try:
        from src.data.gamecast import fetch_espn_scoreboard
        games = fetch_espn_scoreboard()
        for g in games:
            if ((g.get("home_team", "").upper() == home_abbr.upper() and
                 g.get("away_team", "").upper() == away_abbr.upper())):
                from src.data.gamecast import get_espn_predictor
                return get_espn_predictor(g["espn_id"])
    except Exception as e:
        logger.warning("ESPN predictor fetch failed: %s", e)
    return {"home_win_pct": 50.0, "away_win_pct": 50.0}


def _get_residual_correction(spread: float, total: float) -> Tuple[float, float]:
    """Look up residual calibration corrections."""
    spread_corr = 0.0
    total_corr = 0.0

    # Spread bins (exclusive lower bound to prevent overlap, except first bin)
    spread_bins = [
        ("big_away", -30, -18), ("med_away", -18, -12), ("small_away", -12, -8),
        ("slight_away", -8, -4), ("toss_up", -4, 4), ("slight_home", 4, 8),
        ("small_home", 8, 12), ("med_home", 12, 18), ("big_home", 18, 30),
    ]
    for idx, (label, lo, hi) in enumerate(spread_bins):
        if (lo < spread <= hi) if idx > 0 else (lo <= spread <= hi):
            row = db.fetch_one(
                "SELECT avg_residual FROM residual_calibration WHERE bin_label = ?",
                (label,)
            )
            if row and row["avg_residual"] is not None:
                spread_corr = row["avg_residual"]
            break

    # Total bins
    total_bins = [
        ("very_low", 180, 200), ("low", 200, 210), ("below_avg", 210, 215),
        ("avg_low", 215, 220), ("avg_high", 220, 225), ("above_avg", 225, 230),
        ("high", 230, 240), ("very_high", 240, 270),
    ]
    for label, lo, hi in total_bins:
        if lo <= total < hi:
            row = db.fetch_one(
                "SELECT avg_residual FROM residual_calibration_total WHERE bin_label = ?",
                (label,)
            )
            if row and row["avg_residual"] is not None:
                total_corr = row["avg_residual"]
            break

    return spread_corr, total_corr


def predict_matchup(home_team_id: int, away_team_id: int, game_date: str,
                    as_of_date: Optional[str] = None,
                    injured_players: Optional[Dict[int, float]] = None,
                    skip_espn: bool = False,
                    skip_ml: bool = False) -> MatchupPrediction:
    """Full prediction pipeline (Steps 0-11)."""
    pred = MatchupPrediction(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=game_date,
    )

    # Team abbreviations
    home_row = db.fetch_one("SELECT abbreviation, name FROM teams WHERE team_id = ?", (home_team_id,))
    away_row = db.fetch_one("SELECT abbreviation, name FROM teams WHERE team_id = ?", (away_team_id,))
    pred.home_team = home_row["abbreviation"] if home_row else str(home_team_id)
    pred.away_team = away_row["abbreviation"] if away_row else str(away_team_id)
    home_name = home_row["name"] if home_row else ""
    away_name = away_row["name"] if away_row else ""

    if injured_players is None:
        injured_players = {}

    # ── Step 0: Weight Resolution ──
    # Per-team weights blended proportional to each team's games_analyzed
    home_w = load_team_weights(home_team_id)
    away_w = load_team_weights(away_team_id)
    if home_w and away_w:
        # Use games_analyzed from team_tuning as a proxy for per-team data depth
        home_games = db.fetch_one(
            "SELECT games_analyzed FROM team_tuning WHERE team_id = ?", (home_team_id,))
        away_games = db.fetch_one(
            "SELECT games_analyzed FROM team_tuning WHERE team_id = ?", (away_team_id,))
        hg = home_games["games_analyzed"] if home_games else 0
        ag = away_games["games_analyzed"] if away_games else 0
        w = home_w.blend(away_w, self_games=hg, other_games=ag)
    elif home_w:
        w = home_w
    elif away_w:
        w = away_w
    else:
        w = get_weight_config()

    # ── Step 1: Player-Level Projections ──
    home_proj = aggregate_projection(home_team_id, away_team_id, is_home=1,
                                     as_of_date=as_of_date, injured_players=injured_players)
    away_proj = aggregate_projection(away_team_id, home_team_id, is_home=0,
                                     as_of_date=as_of_date, injured_players=injured_players)
    pred.home_proj = {k: v for k, v in home_proj.items() if not k.startswith("_")}
    pred.away_proj = {k: v for k, v in away_proj.items() if not k.startswith("_")}

    # ── Step 2: Home Court Advantage ──
    home_court = get_home_court_advantage(home_team_id)
    pred.home_court_advantage = home_court

    # ── Step 3: Opponent Defensive Adjustment ──
    home_metrics = _get_team_metrics(home_team_id)
    away_metrics = _get_team_metrics(away_team_id)

    league_avg = _LEAGUE_AVG_PPG
    away_opp_pts = away_metrics.get("opp_pts", league_avg) or league_avg
    home_opp_pts = home_metrics.get("opp_pts", league_avg) or league_avg

    away_def_factor_raw = away_opp_pts / league_avg if league_avg > 0 else 1.0
    home_def_factor_raw = home_opp_pts / league_avg if league_avg > 0 else 1.0

    away_def_factor = 1.0 + (away_def_factor_raw - 1.0) * w.def_factor_dampening
    home_def_factor = 1.0 + (home_def_factor_raw - 1.0) * w.def_factor_dampening

    home_base_pts = home_proj["points"] * away_def_factor
    away_base_pts = away_proj["points"] * home_def_factor

    # ── Step 4: Autotune Corrections (individual ±8, net spread ±8) ──
    home_tuning = _get_tuning(home_team_id)
    away_tuning = _get_tuning(away_team_id)
    _TUNE_CAP = 8.0
    ht_corr = _clamp(-_TUNE_CAP, home_tuning["home_pts_correction"], _TUNE_CAP)
    at_corr = _clamp(-_TUNE_CAP, away_tuning["away_pts_correction"], _TUNE_CAP)
    # Cap the NET spread effect of autotune (prevents compounding home/away bias)
    net_tune = ht_corr - at_corr
    if abs(net_tune) > _TUNE_CAP:
        scale = _TUNE_CAP / abs(net_tune)
        ht_corr *= scale
        at_corr *= scale
    home_base_pts += ht_corr
    away_base_pts += at_corr

    # ── Step 5: Fatigue Detection ──
    home_fatigue = compute_fatigue(home_team_id, game_date, w=w)
    away_fatigue = compute_fatigue(away_team_id, game_date, w=w)
    pred.home_fatigue = home_fatigue["penalty"]
    pred.away_fatigue = away_fatigue["penalty"]
    fatigue_adj = home_fatigue["penalty"] - away_fatigue["penalty"]
    pred.fatigue_adj = fatigue_adj

    # ── Step 6: Spread Calculation ──
    spread = (home_base_pts - away_base_pts) + home_court
    spread -= fatigue_adj

    # Turnover differential
    home_to = home_proj.get("turnovers", 0)
    away_to = away_proj.get("turnovers", 0)
    to_adj = (away_to - home_to) * w.turnover_margin_mult
    spread += to_adj
    pred.turnover_adj = to_adj

    # Rebound differential
    home_reb = home_proj.get("rebounds", 0)
    away_reb = away_proj.get("rebounds", 0)
    reb_adj = (home_reb - away_reb) * w.rebound_diff_mult
    spread += reb_adj
    pred.rebound_adj = reb_adj

    # Off/Def rating matchup — cross-team matchup (aligned with VectorizedGames)
    home_off = home_metrics.get("off_rating", _RATING_FALLBACK) or _RATING_FALLBACK
    away_off = away_metrics.get("off_rating", _RATING_FALLBACK) or _RATING_FALLBACK
    home_def = home_metrics.get("def_rating", _RATING_FALLBACK) or _RATING_FALLBACK
    away_def = away_metrics.get("def_rating", _RATING_FALLBACK) or _RATING_FALLBACK

    home_matchup_edge = home_off - away_def   # home offense vs away defense
    away_matchup_edge = away_off - home_def   # away offense vs home defense
    rating_adj = (home_matchup_edge - away_matchup_edge) * w.rating_matchup_mult
    spread += rating_adj
    pred.rating_matchup_adj = rating_adj

    # Four Factors — raw team diffs (positive = home advantage)
    home_ff_efg = home_metrics.get("ff_efg_pct", 0) or 0
    away_ff_efg = away_metrics.get("ff_efg_pct", 0) or 0
    home_ff_tov = home_metrics.get("ff_tm_tov_pct", 0) or 0
    away_ff_tov = away_metrics.get("ff_tm_tov_pct", 0) or 0
    home_ff_oreb = home_metrics.get("ff_oreb_pct", 0) or 0
    away_ff_oreb = away_metrics.get("ff_oreb_pct", 0) or 0
    home_ff_fta = home_metrics.get("ff_fta_rate", 0) or 0
    away_ff_fta = away_metrics.get("ff_fta_rate", 0) or 0

    # Higher eFG/OREB/FTA = better for team; lower TOV = better
    efg_edge = home_ff_efg - away_ff_efg
    tov_edge = away_ff_tov - home_ff_tov       # positive = home turns over less
    oreb_edge = home_ff_oreb - away_ff_oreb
    fta_edge = home_ff_fta - away_ff_fta

    ff_adj = (efg_edge * w.ff_efg_weight + tov_edge * w.ff_tov_weight +
              oreb_edge * w.ff_oreb_weight + fta_edge * w.ff_fta_weight) * w.four_factors_scale
    spread += ff_adj
    pred.four_factors_adj = ff_adj

    # Sharp Money
    sharp_money_adj = 0.0
    if w.sharp_money_weight != 0.0:
        odds_row = db.fetch_one("SELECT spread_home_public, spread_home_money FROM game_odds WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?", (game_date, home_team_id, away_team_id))
        if odds_row and odds_row["spread_home_public"] is not None and odds_row["spread_home_money"] is not None:
            sh_pub = odds_row["spread_home_public"]
            sh_mon = odds_row["spread_home_money"]
            sharp_edge = (sh_mon - sh_pub) / 100.0
            
            # Time decay for current day betting (regress influence if far from typical 7 PM ET game time)
            from datetime import datetime, timedelta
            today_str = datetime.now().strftime("%Y-%m-%d")
            decay = 1.0
            if game_date == today_str:
                now_et = datetime.utcnow() - timedelta(hours=5) # Approx Eastern Time
                # Scale from 0.0 at noon ET to 1.0 at 7 PM ET
                if now_et.hour < 12:
                    decay = 0.0
                elif now_et.hour >= 19:
                    decay = 1.0
                else:
                    # Linear scale between 12 and 19 (7 hours)
                    hours_past_noon = (now_et.hour - 12) + (now_et.minute / 60.0)
                    decay = hours_past_noon / 7.0
                    
            sharp_money_adj = sharp_edge * w.sharp_money_weight * decay
            spread += sharp_money_adj
    pred.adjustments["sharp_money"] = sharp_money_adj

    # Clutch (only if close game)
    clutch_adj = 0.0
    if abs(spread) < w.clutch_threshold:
        home_clutch_net = home_metrics.get("clutch_net_rating", 0) or 0
        away_clutch_net = away_metrics.get("clutch_net_rating", 0) or 0
        clutch_diff = (home_clutch_net - away_clutch_net) * w.clutch_scale
        clutch_adj = _clamp(-w.clutch_cap, clutch_diff, w.clutch_cap)
        spread += clutch_adj
    pred.clutch_adj = clutch_adj

    # Hustle (normalize season totals to per-game)
    home_gp = max(1, home_metrics.get("gp", 1) or 1)
    away_gp = max(1, away_metrics.get("gp", 1) or 1)
    home_defl = (home_metrics.get("deflections", 0) or 0) / home_gp
    away_defl = (away_metrics.get("deflections", 0) or 0) / away_gp
    home_contested = (home_metrics.get("contested_shots", 0) or 0) / home_gp
    away_contested = (away_metrics.get("contested_shots", 0) or 0) / away_gp

    home_effort = home_defl + home_contested * w.hustle_contested_wt
    away_effort = away_defl + away_contested * w.hustle_contested_wt
    hustle_adj = (home_effort - away_effort) * w.hustle_effort_mult
    spread += hustle_adj
    pred.hustle_adj = hustle_adj

    # ── Step 7: Total Calculation ──
    total = home_base_pts + away_base_pts

    # Pace adjustment
    home_pace = home_metrics.get("pace", _PACE_FALLBACK) or _PACE_FALLBACK
    away_pace = away_metrics.get("pace", _PACE_FALLBACK) or _PACE_FALLBACK
    expected_pace = (home_pace + away_pace) / 2.0
    pace_factor = (expected_pace - w.pace_baseline) / w.pace_baseline
    total *= (1.0 + pace_factor * w.pace_mult)
    pred.pace_adj = pace_factor * w.pace_mult * total

    # Defensive disruption
    combined_steals = home_proj.get("steals", 0) + away_proj.get("steals", 0)
    combined_blocks = home_proj.get("blocks", 0) + away_proj.get("blocks", 0)
    def_disruption = (max(0, combined_steals - w.steals_threshold) * w.steals_penalty +
                      max(0, combined_blocks - w.blocks_threshold) * w.blocks_penalty)
    total -= def_disruption
    pred.defensive_disruption = def_disruption

    # Offensive rebound boost
    combined_oreb = home_proj.get("oreb", 0) + away_proj.get("oreb", 0)
    oreb_boost = (combined_oreb - w.oreb_baseline) * w.oreb_mult
    total += oreb_boost
    pred.oreb_boost = oreb_boost

    # Hustle total (deflections above baseline reduce total)
    combined_defl = home_defl + away_defl
    hustle_total = 0.0
    if combined_defl > w.hustle_defl_baseline:
        hustle_total = (combined_defl - w.hustle_defl_baseline) * w.hustle_defl_penalty
        total -= hustle_total
    pred.hustle_total_adj = hustle_total

    # Fatigue total impact
    combined_fatigue = home_fatigue["penalty"] + away_fatigue["penalty"]
    total -= combined_fatigue * w.fatigue_total_mult

    # ── Step 8: ESPN Predictor Blend (80/20) ──
    if not skip_espn:
        try:
            espn = _get_espn_predictor(pred.home_team, pred.away_team)
            espn_pct = espn.get("home_win_pct", 50.0)
            pred.espn_home_pct = espn_pct
            if espn_pct != 50.0:
                espn_edge = espn_pct - 50.0
                espn_implied = espn_edge * w.espn_spread_scale
                pre_blend = spread
                spread = spread * w.espn_model_weight + espn_implied * w.espn_weight

                # Disagreement dampening
                if ((pre_blend > 0.5 and espn_implied < -0.5) or
                        (pre_blend < -0.5 and espn_implied > 0.5)):
                    spread *= w.espn_disagree_damp

                pred.espn_blend_adj = spread - pre_blend
        except Exception as e:
            logger.warning("ESPN blend failed: %s", e)

    # ── Step 9: ML Ensemble Blend ──
    if not skip_ml and w.ml_ensemble_weight > 0:
        try:
            from src.analytics.ml_model import predict_ml_with_uncertainty
            # Build injury context for ML features
            _injury_ctx = _build_injury_context(home_team_id, away_team_id)
            features = _build_ml_features(home_proj, away_proj, home_metrics, away_metrics,
                                          home_court, home_fatigue, away_fatigue,
                                          efg_edge, tov_edge, oreb_edge, fta_edge,
                                          home_off, away_off, home_def, away_def,
                                          home_pace, away_pace, home_defl, away_defl,
                                          home_contested, away_contested,
                                          injury_context=_injury_ctx)
            ml_result = predict_ml_with_uncertainty(features)
            if ml_result and ml_result.get("confidence", 0) > 0.3:
                ml_spread = ml_result["spread"]
                ml_total = ml_result["total"]
                ml_wt = w.ml_ensemble_weight

                # Early-season dampening
                home_gp = home_metrics.get("gp", 0) or 0
                away_gp = away_metrics.get("gp", 0) or 0
                min_gp = min(home_gp, away_gp)
                if min_gp < 15:
                    ml_wt *= min_gp / 15.0

                # Disagreement dampening
                if abs(ml_spread - spread) > w.ml_disagree_threshold:
                    ml_wt *= w.ml_disagree_damp

                # Uncertainty dampening
                spread_std = ml_result.get("spread_std", 0)
                total_std = ml_result.get("total_std", 0)
                uncertainty_scale = max(0.35, min(1.0,
                    1.0 / (1.0 + (spread_std / 12.0) + (total_std / 20.0))))
                ml_wt *= uncertainty_scale

                base_wt = 1.0 - ml_wt
                pre_ml_spread = spread
                pre_ml_total = total
                spread = base_wt * spread + ml_wt * ml_spread
                total = base_wt * total + ml_wt * ml_total

                pred.ml_spread = ml_spread
                pred.ml_total = ml_total
                pred.ml_blend_adj = spread - pre_ml_spread
        except Exception as e:
            logger.debug(f"ML blend skipped: {e}")

    # ── Step 10: Sanity Clamps ──
    spread = _clamp(-w.spread_clamp, spread, w.spread_clamp)
    total = _clamp(w.total_min, total, w.total_max)

    # ── Step 11: Residual Calibration ──
    try:
        spread_corr, total_corr = _get_residual_correction(spread, total)
        spread -= spread_corr
        total -= total_corr
    except Exception as e:
        logger.warning("Residual calibration failed: %s", e)

    # ── Final: Derive Individual Scores ──
    pred.predicted_spread = round(spread, 1)
    pred.predicted_total = round(total, 1)
    pred.predicted_home_score = round((total + spread) / 2, 1)
    pred.predicted_away_score = round((total - spread) / 2, 1)
    pred.winner = pred.home_team if spread > 0 else pred.away_team
    pred.confidence = min(1.0, abs(spread) / 15.0)

    return pred


def precompute_game_data(callback=None, force=False) -> List[PrecomputedGame]:
    """Build a list of PrecomputedGame objects for optimization/backtest.

    Uses a persistent disk + memory cache so that already-computed historical
    games are never reprocessed.  Only truly new games (added since last run)
    go through the expensive per-game projection pipeline.

    Pass ``force=True`` to discard the cache and recompute everything.
    """
    from src.analytics.backtester import get_actual_game_results
    from src.database.db import thread_local_db
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.config import get_config
    import threading

    # ── Load cache ──
    cache = {} if force else _load_pc_cache()

    all_games = get_actual_game_results()
    if not all_games:
        if callback:
            callback("No game results found")
        return []

    # Filter teams with < 5 games
    from collections import Counter
    team_games = Counter()
    for g in all_games:
        team_games[g.get("home_team_id", 0)] += 1
        team_games[g.get("away_team_id", 0)] += 1
    valid_teams = {tid for tid, cnt in team_games.items() if cnt >= 5}
    games = [g for g in all_games
             if g.get("home_team_id") in valid_teams and g.get("away_team_id") in valid_teams]

    # ── Determine which games still need computing ──
    valid_keys = set()
    new_games = []
    for g in games:
        key = _game_cache_key(g["home_team_id"], g["away_team_id"], g["game_date"])
        valid_keys.add(key)
        if key not in cache:
            new_games.append(g)

    if not new_games:
        # Everything cached — fast path
        result = [cache[k] for k in valid_keys if k in cache]
        result.sort(key=lambda pg: pg.game_date)
        if callback:
            callback(f"Loaded {len(result)} precomputed games from cache (0 new)")
        return result

    cached_count = len(valid_keys) - len(new_games)
    if callback:
        callback(f"Precomputing {len(new_games)} new games ({cached_count} cached)...")

    # Build historical context ONCE (rosters, trades, injury inference)
    # Uses all_games (not filtered) so every game_id is in the map.
    if callback:
        callback("Building historical roster & injury context...")
    ctx = _build_precompute_context(all_games)

    cfg = get_config()
    max_workers = cfg.get("worker_threads", 4)
    _pc_lock = threading.Lock()
    completed_count = [0]

    def _precompute_one(g):
        """Process one game with a thread-local DB."""
        with thread_local_db():
            htid = g["home_team_id"]
            atid = g["away_team_id"]
            gdate = g["game_date"]

            # Historical roster from context (handles trades — team membership
            # follows actual game data, not the current players table).
            home_roster = _get_historical_roster(htid, gdate, ctx)
            away_roster = _get_historical_roster(atid, gdate, ctx)

            # Projections (with historical rosters)
            home_proj = aggregate_projection(htid, atid, is_home=1, as_of_date=gdate,
                                             roster=home_roster)
            away_proj = aggregate_projection(atid, htid, is_home=0, as_of_date=gdate,
                                             roster=away_roster)

            # Home court
            home_court = get_home_court_advantage(htid)

            # Metrics
            hm = _get_team_metrics(htid)
            am = _get_team_metrics(atid)

            league_avg = _LEAGUE_AVG_PPG
            away_opp_pts = am.get("opp_pts", league_avg) or league_avg
            home_opp_pts = hm.get("opp_pts", league_avg) or league_avg
            away_def_raw = away_opp_pts / league_avg if league_avg > 0 else 1.0
            home_def_raw = home_opp_pts / league_avg if league_avg > 0 else 1.0

            # Tuning
            ht = _get_tuning(htid)
            at = _get_tuning(atid)

            # Fatigue (pass WeightConfig for tunable b2b/3in4/4in6 penalties)
            from src.analytics.weight_config import get_weight_config as _gwc
            _w = _gwc()
            hfat = compute_fatigue(htid, gdate, w=_w)
            afat = compute_fatigue(atid, gdate, w=_w)

            # Ratings
            home_off = hm.get("off_rating", _RATING_FALLBACK) or _RATING_FALLBACK
            away_off = am.get("off_rating", _RATING_FALLBACK) or _RATING_FALLBACK
            home_def = hm.get("def_rating", _RATING_FALLBACK) or _RATING_FALLBACK
            away_def = am.get("def_rating", _RATING_FALLBACK) or _RATING_FALLBACK

            # Pace
            home_pace = hm.get("pace", _PACE_FALLBACK) or _PACE_FALLBACK
            away_pace = am.get("pace", _PACE_FALLBACK) or _PACE_FALLBACK

            # Four Factors — store raw team values for predict_from_precomputed
            h_efg = hm.get("ff_efg_pct", 0) or 0
            a_efg = am.get("ff_efg_pct", 0) or 0

            h_tov = hm.get("ff_tm_tov_pct", 0) or 0
            a_tov = am.get("ff_tm_tov_pct", 0) or 0

            h_oreb = hm.get("ff_oreb_pct", 0) or 0
            a_oreb = am.get("ff_oreb_pct", 0) or 0

            h_fta = hm.get("ff_fta_rate", 0) or 0
            a_fta = am.get("ff_fta_rate", 0) or 0

            # Edge values for ML features (compatibility)
            h_opp_efg = hm.get("opp_efg_pct", 0) or 0
            a_opp_efg = am.get("opp_efg_pct", 0) or 0
            efg_e = h_efg - a_efg

            h_opp_tov = hm.get("opp_tm_tov_pct", 0) or 0
            a_opp_tov = am.get("opp_tm_tov_pct", 0) or 0
            tov_e = a_tov - h_tov

            h_opp_oreb = hm.get("opp_oreb_pct", 0) or 0
            a_opp_oreb = am.get("opp_oreb_pct", 0) or 0
            oreb_e = h_oreb - a_oreb

            h_opp_fta = hm.get("opp_fta_rate", 0) or 0
            a_opp_fta = am.get("opp_fta_rate", 0) or 0
            fta_e = h_fta - a_fta

            # Clutch
            h_clutch = {"net_rating": hm.get("clutch_net_rating", 0) or 0,
                        "efg_pct": hm.get("clutch_efg_pct", 0) or 0}
            a_clutch = {"net_rating": am.get("clutch_net_rating", 0) or 0,
                        "efg_pct": am.get("clutch_efg_pct", 0) or 0}

            # Hustle (normalize season totals to per-game)
            h_gp = max(1, hm.get("gp", 1) or 1)
            a_gp = max(1, am.get("gp", 1) or 1)
            h_hustle = {"deflections": (hm.get("deflections", 0) or 0) / h_gp,
                        "contested": (hm.get("contested_shots", 0) or 0) / h_gp,
                        "loose_balls": (hm.get("loose_balls_recovered", 0) or 0) / h_gp}
            a_hustle = {"deflections": (am.get("deflections", 0) or 0) / a_gp,
                        "contested": (am.get("contested_shots", 0) or 0) / a_gp,
                        "loose_balls": (am.get("loose_balls_recovered", 0) or 0) / a_gp}

            # Inferred historical injuries — compare recent active roster vs
            # who actually played in this game.  If a rotation player was
            # playing last week but not today, they're injured.  Uses only
            # data that was known at game time.
            h_inj = _infer_historical_injuries(htid, gdate, ctx)
            a_inj = _infer_historical_injuries(atid, gdate, ctx)
            inj_ctx = {
                "home_injured_count": h_inj["injured_count"],
                "away_injured_count": a_inj["injured_count"],
                "home_injury_ppg_lost": h_inj["injury_ppg_lost"],
                "away_injury_ppg_lost": a_inj["injury_ppg_lost"],
                "home_injury_minutes_lost": h_inj["injury_minutes_lost"],
                "away_injury_minutes_lost": a_inj["injury_minutes_lost"],
            }

            return PrecomputedGame(
                game_date=gdate,
                home_team_id=htid,
                away_team_id=atid,
                actual_home_score=g.get("home_score", 0),
                actual_away_score=g.get("away_score", 0),
                vegas_spread=g.get("vegas_spread") or 0.0,
                vegas_home_ml=g.get("vegas_home_ml") or 0,
                vegas_away_ml=g.get("vegas_away_ml") or 0,
                spread_home_public=g.get("spread_home_public") or 0,
                spread_home_money=g.get("spread_home_money") or 0,
                home_proj={k: v for k, v in home_proj.items() if not k.startswith("_")},
                away_proj={k: v for k, v in away_proj.items() if not k.startswith("_")},
                home_court=home_court,
                away_def_factor_raw=away_def_raw,
                home_def_factor_raw=home_def_raw,
                home_tuning_home_corr=ht["home_pts_correction"],
                away_tuning_away_corr=at["away_pts_correction"],
                home_fatigue_penalty=hfat["penalty"],
                away_fatigue_penalty=afat["penalty"],
                home_off=home_off,
                away_off=away_off,
                home_def=home_def,
                away_def=away_def,
                home_pace=home_pace,
                away_pace=away_pace,
                home_ff={"efg": h_efg, "tov": h_tov,
                         "oreb": h_oreb, "fta": h_fta},
                away_ff={"efg": a_efg, "tov": a_tov,
                         "oreb": a_oreb, "fta": a_fta},
                home_clutch=h_clutch,
                away_clutch=a_clutch,
                home_hustle=h_hustle,
                away_hustle=a_hustle,
                home_injured_count=inj_ctx.get("home_injured_count", 0),
                away_injured_count=inj_ctx.get("away_injured_count", 0),
                home_injury_ppg_lost=inj_ctx.get("home_injury_ppg_lost", 0),
                away_injury_ppg_lost=inj_ctx.get("away_injury_ppg_lost", 0),
                home_injury_minutes_lost=inj_ctx.get("home_injury_minutes_lost", 0),
                away_injury_minutes_lost=inj_ctx.get("away_injury_minutes_lost", 0),
                home_games_played=hm.get("gp", 0) or 0,
                away_games_played=am.get("gp", 0) or 0,
            )

    # ── Only compute new games ──
    new_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_precompute_one, g): g for g in new_games}
        for future in as_completed(futures):
            game = futures[future]
            try:
                pg = future.result()
                if pg is not None:
                    new_results.append(pg)
            except Exception as e:
                logger.warning("Skipping game %s: %s", game.get("game_id"), e)

            with _pc_lock:
                completed_count[0] += 1
                c = completed_count[0]
                if callback and c % 25 == 0:
                    callback(f"Precomputed {c}/{len(new_games)} games")

    # ── Merge new results into cache and persist ──
    for pg in new_results:
        key = _game_cache_key(pg.home_team_id, pg.away_team_id, pg.game_date)
        cache[key] = pg

    if new_results:
        _save_pc_cache(cache)

    # Return only the games that are currently valid
    result = [cache[k] for k in valid_keys if k in cache]
    result.sort(key=lambda pg: pg.game_date)

    if callback:
        callback(f"Precomputed {len(result)} games total ({len(new_results)} new, {cached_count} cached)")
    return result


def predict_from_precomputed(g: PrecomputedGame, w: WeightConfig,
                             skip_residual: bool = True) -> Dict[str, float]:
    """Identical logic to predict_matchup() but on PrecomputedGame.

    Returns dict with 'spread', 'total', 'home_score', 'away_score'.
    """
    # Defensive adjustment
    away_def_f = 1.0 + (g.away_def_factor_raw - 1.0) * w.def_factor_dampening
    home_def_f = 1.0 + (g.home_def_factor_raw - 1.0) * w.def_factor_dampening

    home_base = g.home_proj.get("points", 0) * away_def_f
    away_base = g.away_proj.get("points", 0) * home_def_f

    # Autotune corrections (individual ±8, net spread ±8, matching predict_matchup)
    _TUNE_CAP_OPT = 8.0
    ht_c = _clamp(-_TUNE_CAP_OPT, g.home_tuning_home_corr, _TUNE_CAP_OPT)
    at_c = _clamp(-_TUNE_CAP_OPT, g.away_tuning_away_corr, _TUNE_CAP_OPT)
    net_t = ht_c - at_c
    if abs(net_t) > _TUNE_CAP_OPT:
        sc = _TUNE_CAP_OPT / abs(net_t)
        ht_c *= sc
        at_c *= sc
    home_base += ht_c
    away_base += at_c

    # Spread
    spread = (home_base - away_base) + g.home_court
    spread -= (g.home_fatigue_penalty - g.away_fatigue_penalty)

    # Turnover differential
    home_to = g.home_proj.get("turnovers", 0)
    away_to = g.away_proj.get("turnovers", 0)
    spread += (away_to - home_to) * w.turnover_margin_mult

    # Rebound diff
    home_reb = g.home_proj.get("rebounds", 0)
    away_reb = g.away_proj.get("rebounds", 0)
    spread += (home_reb - away_reb) * w.rebound_diff_mult

    # Rating matchup — cross-team matchup (aligned with VectorizedGames)
    home_me = g.home_off - g.away_def   # home offense vs away defense
    away_me = g.away_off - g.home_def   # away offense vs home defense
    spread += (home_me - away_me) * w.rating_matchup_mult

    # Four Factors — raw team diffs
    hff = g.home_ff
    aff = g.away_ff
    efg_e = hff.get("efg", 0) - aff.get("efg", 0)
    tov_e = aff.get("tov", 0) - hff.get("tov", 0)    # positive = home turns over less
    oreb_e = hff.get("oreb", 0) - aff.get("oreb", 0)
    fta_e = hff.get("fta", 0) - aff.get("fta", 0)
    ff_adj = (efg_e * w.ff_efg_weight + tov_e * w.ff_tov_weight +
              oreb_e * w.ff_oreb_weight + fta_e * w.ff_fta_weight) * w.four_factors_scale
    spread += ff_adj

    # Clutch
    if abs(spread) < w.clutch_threshold:
        h_clutch = g.home_clutch.get("net_rating", 0)
        a_clutch = g.away_clutch.get("net_rating", 0)
        clutch_diff = (h_clutch - a_clutch) * w.clutch_scale
        spread += _clamp(-w.clutch_cap, clutch_diff, w.clutch_cap)

    # Hustle
    h_eff = g.home_hustle.get("deflections", 0) + g.home_hustle.get("contested", 0) * w.hustle_contested_wt
    a_eff = g.away_hustle.get("deflections", 0) + g.away_hustle.get("contested", 0) * w.hustle_contested_wt
    spread += (h_eff - a_eff) * w.hustle_effort_mult

    # Sharp Money
    sh_pub = g.spread_home_public if g.spread_home_public else 50.0
    sh_mon = g.spread_home_money if g.spread_home_money else 50.0
    sharp_money_edge = (sh_mon - sh_pub) / 100.0
    spread += sharp_money_edge * w.sharp_money_weight

    # Total
    total = home_base + away_base

    avg_pace = (g.home_pace + g.away_pace) / 2.0
    pace_factor = (avg_pace - w.pace_baseline) / w.pace_baseline
    total *= (1.0 + pace_factor * w.pace_mult)

    # Defensive disruption
    comb_steals = g.home_proj.get("steals", 0) + g.away_proj.get("steals", 0)
    comb_blocks = g.home_proj.get("blocks", 0) + g.away_proj.get("blocks", 0)
    total -= (max(0, comb_steals - w.steals_threshold) * w.steals_penalty +
              max(0, comb_blocks - w.blocks_threshold) * w.blocks_penalty)

    # OREB boost
    comb_oreb = g.home_proj.get("oreb", 0) + g.away_proj.get("oreb", 0)
    total += (comb_oreb - w.oreb_baseline) * w.oreb_mult

    # Hustle total
    comb_defl = g.home_hustle.get("deflections", 0) + g.away_hustle.get("deflections", 0)
    if comb_defl > w.hustle_defl_baseline:
        total -= (comb_defl - w.hustle_defl_baseline) * w.hustle_defl_penalty

    # Fatigue total
    comb_fatigue = g.home_fatigue_penalty + g.away_fatigue_penalty
    total -= comb_fatigue * w.fatigue_total_mult

    # Clamps
    spread = _clamp(-w.spread_clamp, spread, w.spread_clamp)
    total = _clamp(w.total_min, total, w.total_max)

    # Residual calibration (skip during optimization)
    if not skip_residual:
        try:
            s_corr, t_corr = _get_residual_correction(spread, total)
            spread -= s_corr
            total -= t_corr
        except Exception as e:
            logger.warning("Residual calibration (fast) failed: %s", e)

    home_score = (total + spread) / 2.0
    away_score = (total - spread) / 2.0
    return {
        "spread": spread,
        "total": total,
        "home_score": home_score,
        "away_score": away_score,
    }


def _build_injury_context(home_team_id: int, away_team_id: int) -> Dict[str, float]:
    """Build injury impact context for ML features from current injury data."""
    ctx = {}
    for side, tid in [("home", home_team_id), ("away", away_team_id)]:
        try:
            injured = db.fetch_all("""
                SELECT p.player_id, i.status,
                       COALESCE(p.ppg, 0) as ppg,
                       COALESCE(p.mpg, 0) as mpg
                FROM injuries i
                JOIN players p ON i.player_id = p.player_id
                WHERE p.team_id = ?
            """, (tid,))
            count = 0
            ppg_lost = 0.0
            min_lost = 0.0
            for inj in injured:
                status = (inj["status"] or "").lower()
                if status in ("out", "o", "doubtful"):
                    count += 1
                    ppg_lost += float(inj["ppg"] or 0)
                    min_lost += float(inj["mpg"] or 0)
                elif status in ("questionable", "gtd", "day-to-day"):
                    count += 0.5
                    ppg_lost += float(inj["ppg"] or 0) * 0.5
                    min_lost += float(inj["mpg"] or 0) * 0.5
            ctx[f"{side}_injured_count"] = count
            ctx[f"{side}_injury_ppg_lost"] = ppg_lost
            ctx[f"{side}_injury_minutes_lost"] = min_lost
        except Exception:
            ctx[f"{side}_injured_count"] = 0
            ctx[f"{side}_injury_ppg_lost"] = 0
            ctx[f"{side}_injury_minutes_lost"] = 0
    return ctx


def _build_ml_features(home_proj, away_proj, home_m, away_m,
                        home_court, home_fat, away_fat,
                        efg_edge, tov_edge, oreb_edge, fta_edge,
                        home_off, away_off, home_def, away_def,
                        home_pace, away_pace, home_defl, away_defl,
                        home_contested, away_contested,
                        injury_context=None) -> Dict[str, float]:
    """Build the ~88 feature dict for ML model."""
    f = {}

    # Counting stats
    for stat in ["points", "rebounds", "assists", "steals", "blocks", "turnovers", "oreb", "dreb"]:
        hv = home_proj.get(stat, 0)
        av = away_proj.get(stat, 0)
        f[f"home_{stat}"] = hv
        f[f"away_{stat}"] = av
        f[f"diff_{stat}"] = hv - av

    # Shooting efficiency
    h_eff = compute_shooting_efficiency(home_proj)
    a_eff = compute_shooting_efficiency(away_proj)
    for stat in ["ts_pct", "fg3_rate", "ft_rate"]:
        f[f"home_{stat}"] = h_eff.get(stat, 0)
        f[f"away_{stat}"] = a_eff.get(stat, 0)
        f[f"diff_{stat}"] = h_eff.get(stat, 0) - a_eff.get(stat, 0)

    # TO margin
    home_to = home_proj.get("turnovers", 0)
    away_to = away_proj.get("turnovers", 0)
    f["home_to_margin"] = away_to - home_to
    f["away_to_margin"] = home_to - away_to
    f["diff_to_margin"] = f["home_to_margin"] - f["away_to_margin"]

    # Ratings
    f["home_off_rating"] = home_off
    f["away_off_rating"] = away_off
    f["home_def_rating"] = home_def
    f["away_def_rating"] = away_def
    f["home_net_rating"] = home_off - home_def
    f["away_net_rating"] = away_off - away_def
    f["diff_net_rating"] = f["home_net_rating"] - f["away_net_rating"]
    f["home_matchup_edge"] = home_off - away_def
    f["away_matchup_edge"] = away_off - home_def
    f["diff_matchup_edge"] = f["home_matchup_edge"] - f["away_matchup_edge"]

    # Defensive factors
    league_avg = _LEAGUE_AVG_PPG
    f["home_def_factor_raw"] = (home_m.get("opp_pts", league_avg) or league_avg) / league_avg
    f["away_def_factor_raw"] = (away_m.get("opp_pts", league_avg) or league_avg) / league_avg

    # Pace
    f["home_pace"] = home_pace
    f["away_pace"] = away_pace
    f["avg_pace"] = (home_pace + away_pace) / 2
    f["diff_pace"] = home_pace - away_pace

    # Home court
    f["home_court"] = home_court

    # Fatigue
    f["home_fatigue"] = home_fat["penalty"]
    f["away_fatigue"] = away_fat["penalty"]
    f["diff_fatigue"] = home_fat["penalty"] - away_fat["penalty"]
    f["combined_fatigue"] = home_fat["penalty"] + away_fat["penalty"]

    # Four Factors edges
    f["ff_efg_edge"] = efg_edge
    f["ff_tov_edge"] = tov_edge
    f["ff_oreb_edge"] = oreb_edge
    f["ff_fta_edge"] = fta_edge

    # Clutch
    f["home_clutch_net"] = home_m.get("clutch_net_rating", 0) or 0
    f["away_clutch_net"] = away_m.get("clutch_net_rating", 0) or 0
    f["diff_clutch_net"] = f["home_clutch_net"] - f["away_clutch_net"]
    f["home_clutch_efg"] = home_m.get("clutch_efg_pct", 0) or 0
    f["away_clutch_efg"] = away_m.get("clutch_efg_pct", 0) or 0

    # Hustle
    f["home_deflections"] = home_defl
    f["away_deflections"] = away_defl
    f["diff_deflections"] = home_defl - away_defl
    f["home_contested"] = home_contested
    f["away_contested"] = away_contested
    h_gp = max(1, home_m.get("gp", 1) or 1)
    a_gp = max(1, away_m.get("gp", 1) or 1)
    home_loose = (home_m.get("loose_balls_recovered", 0) or 0) / h_gp
    away_loose = (away_m.get("loose_balls_recovered", 0) or 0) / a_gp
    f["home_loose_balls"] = home_loose
    f["away_loose_balls"] = away_loose

    # Injury context — use provided data or zeros
    ic = injury_context or {}
    f["home_injured_count"] = ic.get("home_injured_count", 0)
    f["away_injured_count"] = ic.get("away_injured_count", 0)
    f["diff_injured_count"] = f["home_injured_count"] - f["away_injured_count"]
    f["home_injury_ppg_lost"] = ic.get("home_injury_ppg_lost", 0)
    f["away_injury_ppg_lost"] = ic.get("away_injury_ppg_lost", 0)
    f["diff_injury_ppg_lost"] = f["home_injury_ppg_lost"] - f["away_injury_ppg_lost"]
    f["home_injury_minutes_lost"] = ic.get("home_injury_minutes_lost", 0)
    f["away_injury_minutes_lost"] = ic.get("away_injury_minutes_lost", 0)
    f["diff_injury_minutes_lost"] = f["home_injury_minutes_lost"] - f["away_injury_minutes_lost"]

    # Season phase
    home_gp = home_m.get("gp", 0) or 0
    away_gp = away_m.get("gp", 0) or 0
    f["home_games_played"] = home_gp
    f["away_games_played"] = away_gp
    f["min_games_played"] = min(home_gp, away_gp)
    f["games_played_diff"] = home_gp - away_gp

    # Roster change
    f["home_roster_changed"] = 0
    f["away_roster_changed"] = 0

    return f
