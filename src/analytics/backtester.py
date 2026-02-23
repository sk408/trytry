"""Historical game replay, caching, metrics."""

import hashlib
import json
import logging
import os
import threading
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from src.database import db
from src.database.db import thread_local_db
from src.analytics.cache import start_session_caches, stop_session_caches
from src.analytics.prediction import predict_matchup
from src.analytics.prediction_quality import compute_quality_metrics, compute_progression, compute_vegas_comparison
from src.analytics.weight_config import get_weight_config
from src.config import get_config

logger = logging.getLogger(__name__)

_progress_lock = threading.Lock()

BACKTEST_CACHE_DIR = os.path.join("data", "backtest_cache")
BACKTEST_CACHE_TTL = 3600  # 60 minutes


def get_actual_game_results(team_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Aggregate player_stats by (game_id, is_home) to reconstruct game scores.

    Uses is_home + opponent_team_id (recorded at game time) instead of
    current team_id from players table, so traded players are attributed
    correctly to the team they played for at the time.
    """
    query = """
        SELECT ps.game_id, ps.game_date, ps.is_home,
               ps.opponent_team_id,
               SUM(ps.points) as total_pts, COUNT(*) as player_count
        FROM player_stats ps
        GROUP BY ps.game_id, ps.is_home
        ORDER BY ps.game_date ASC
    """
    rows = db.fetch_all(query)

    # Group by game_id — two rows per game (is_home=0 and is_home=1)
    games_dict: Dict[str, Dict] = {}
    for r in rows:
        gid = r["game_id"]
        if gid not in games_dict:
            games_dict[gid] = {"game_id": gid, "game_date": r["game_date"]}

        if r["is_home"]:
            # Home players' opponent_team_id = away team
            games_dict[gid]["home_score"] = r["total_pts"]
            games_dict[gid]["away_team_id"] = r["opponent_team_id"]
            games_dict[gid]["home_count"] = r["player_count"]
        else:
            # Away players' opponent_team_id = home team
            games_dict[gid]["away_score"] = r["total_pts"]
            games_dict[gid]["home_team_id"] = r["opponent_team_id"]
            games_dict[gid]["away_count"] = r["player_count"]

    results = []
    for gid, g in games_dict.items():
        home_score = g.get("home_score", 0)
        away_score = g.get("away_score", 0)
        # Sanity filter — require at least 4 players per side and 40 pts
        if home_score < 40 or away_score < 40:
            continue
        if g.get("home_count", 0) < 4 or g.get("away_count", 0) < 4:
            continue
        if "home_team_id" not in g or "away_team_id" not in g:
            continue

        if team_id and g["home_team_id"] != team_id and g["away_team_id"] != team_id:
            continue

        spread = home_score - away_score
        if spread > 0.5:
            winner = "HOME"
        elif spread < -0.5:
            winner = "AWAY"
        else:
            winner = "PUSH"

        g["home_score"] = home_score
        g["away_score"] = away_score
        g["actual_spread"] = spread
        g["actual_total"] = home_score + away_score
        g["winner"] = winner
        results.append(g)

    results.sort(key=lambda x: x["game_date"])
    return results


def _get_cache_hash() -> str:
    """SHA-256 hash of model_weights + team_tuning + ml model meta."""
    w = get_weight_config()
    parts = [json.dumps(w.to_dict(), sort_keys=True)]

    # Team tuning
    rows = db.fetch_all("SELECT * FROM team_tuning ORDER BY team_id")
    parts.append(json.dumps([dict(r) for r in rows], sort_keys=True))

    # ML model meta
    meta_path = os.path.join("data", "ml_models", "model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            parts.append(f.read())

    h = hashlib.sha256("".join(parts).encode()).hexdigest()[:16]
    return h


def _get_cache_path() -> str:
    h = _get_cache_hash()
    return os.path.join(BACKTEST_CACHE_DIR, f"bt_{h}.json")


def get_backtest_cache_age() -> Optional[float]:
    """Returns age in minutes of the cache file, or None if no cache."""
    path = _get_cache_path()
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    return age / 60.0


def _load_cache() -> Optional[Dict]:
    path = _get_cache_path()
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > BACKTEST_CACHE_TTL:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(data: Dict):
    os.makedirs(BACKTEST_CACHE_DIR, exist_ok=True)
    path = _get_cache_path()
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Failed to save backtest cache: {e}")


def run_backtest(team_id: Optional[int] = None,
                 use_cache: bool = True,
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Run backtest on historical games. Returns per-game and aggregate metrics."""
    # Check cache
    if use_cache and team_id is None:
        cached = _load_cache()
        if cached:
            # Inject quality metrics if missing from old cache
            if "quality_metrics" not in cached:
                quality = compute_quality_metrics(cached.get("per_game", []), cached.get("per_team", {}))
                progression = compute_progression(cached.get("per_game", []))
                vegas_comparison = compute_vegas_comparison(cached.get("per_game", []))
                quality["progression"] = progression
                quality["vegas_comparison"] = vegas_comparison
                cached["quality_metrics"] = quality
                _save_cache(cached)
            if callback:
                callback("Using cached backtest results")
            return cached

    games = get_actual_game_results(team_id)
    if not games:
        return {"error": "No games found", "total_games": 0}

    if callback:
        callback(f"Backtesting {len(games)} games...")

    start_session_caches()

    # Per-game results
    per_game = []
    completed = 0
    error_counts: Dict[str, int] = {}
    _error_lock = threading.Lock()

    # Configurable thread count from settings
    cfg = get_config()
    max_workers = cfg.get("worker_threads", 4)

    def process_game(game):
        nonlocal completed
        # Each thread gets its own in-memory DB copy — no lock contention
        with thread_local_db():
            try:
                home_tid = game["home_team_id"]
                away_tid = game["away_team_id"]

                result = predict_matchup(
                    home_tid, away_tid,
                    game_date=game["game_date"],
                    as_of_date=game["game_date"],
                    skip_espn=True,
                )

                pred_spread = result.predicted_spread
                pred_total = result.predicted_total
                pred_home = result.predicted_home_score
                pred_away = result.predicted_away_score

                if pred_spread is None or pred_total is None:
                    logger.warning("Backtest game %s: predict returned None spread/total",
                                   game.get("game_id"))
                    return None

                actual_spread = game["actual_spread"]
                actual_total = game["actual_total"]

                spread_error = abs(pred_spread - actual_spread)
                total_error = abs(pred_total - actual_total)
                home_error = abs(pred_home - game["home_score"])
                away_error = abs(pred_away - game["away_score"])

                # Winner check
                pred_winner = "HOME" if pred_spread > 0.5 else ("AWAY" if pred_spread < -0.5 else "PUSH")
                winner_correct = (pred_winner == game["winner"] or
                                  (game["winner"] == "PUSH" and abs(pred_spread) <= 3))

                return {
                    "game_id": game["game_id"],
                    "game_date": game["game_date"],
                    "home_team_id": home_tid,
                    "away_team_id": away_tid,
                    "pred_spread": round(pred_spread, 2),
                    "actual_spread": actual_spread,
                    "spread_error": round(spread_error, 2),
                    "pred_total": round(pred_total, 1),
                    "actual_total": actual_total,
                    "total_error": round(total_error, 1),
                    "home_score_error": round(home_error, 1),
                    "away_score_error": round(away_error, 1),
                    "winner_correct": winner_correct,
                    "spread_within_5": spread_error <= 5,
                    "total_within_10": total_error <= 10,
                    # Adjs for feature attribution
                    "fatigue_adj": getattr(result, "fatigue_adj", 0),
                    "rest_adv": getattr(result, "home_fatigue", 0) - getattr(result, "away_fatigue", 0),
                    "rating_matchup_adj": getattr(result, "rating_matchup_adj", 0),
                    "clutch_adj": getattr(result, "clutch_adj", 0),
                    "ml_blend_adj": getattr(result, "ml_blend_adj", 0),
                }
            except Exception as e:
                error_msg = str(e)
                # Classify and count errors, always print full traceback
                logger.error("Backtest error for game %s:\n%s",
                             game.get("game_id"), traceback.format_exc())
                with _error_lock:
                    error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
                return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_game, g): g for g in games}
        for future in as_completed(futures):
            result = future.result()
            if result:
                per_game.append(result)
            with _progress_lock:
                completed += 1
                if callback and completed % 20 == 0:
                    callback(f"Progress: {completed}/{len(games)} games")

    stop_session_caches()

    # Log error summary
    if error_counts:
        total_errors = sum(error_counts.values())
        logger.error("Backtest: %d/%d games failed. Error breakdown:", total_errors, len(games))
        for msg, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            logger.error("  [%d×] %s", count, msg)

    # Sort by date
    per_game.sort(key=lambda x: x["game_date"])

    # Aggregate
    if not per_game:
        return {"error": "No valid results", "total_games": 0}

    total = len(per_game)
    correct = sum(1 for g in per_game if g["winner_correct"])
    within_5 = sum(1 for g in per_game if g["spread_within_5"])
    within_10 = sum(1 for g in per_game if g["total_within_10"])

    avg_spread_err = sum(g["spread_error"] for g in per_game) / total
    avg_total_err = sum(g["total_error"] for g in per_game) / total

    # ---- Team abbreviation lookup ---
    team_abbrev: Dict[int, str] = {}
    try:
        rows = db.fetch_all("SELECT team_id, abbreviation FROM teams")
        team_abbrev = {r["team_id"]: r["abbreviation"] for r in rows}
    except Exception:
        pass

    # ---- Per-team metrics ---
    team_results: Dict[int, Dict] = {}
    for g in per_game:
        for tid_key, is_home in [("home_team_id", True), ("away_team_id", False)]:
            tid = g[tid_key]
            if tid not in team_results:
                team_results[tid] = {
                    "games": 0, "correct": 0, "spread_errors": [],
                    "total_errors": [], "within_5": 0, "within_10": 0,
                    "home_correct": 0, "home_games": 0,
                    "away_correct": 0, "away_games": 0,
                }
            tr = team_results[tid]
            tr["games"] += 1
            tr["correct"] += 1 if g["winner_correct"] else 0
            tr["spread_errors"].append(g["spread_error"])
            tr["total_errors"].append(g["total_error"])
            tr["within_5"] += 1 if g["spread_within_5"] else 0
            tr["within_10"] += 1 if g["total_within_10"] else 0
            if is_home:
                tr["home_games"] += 1
                tr["home_correct"] += 1 if g["winner_correct"] else 0
            else:
                tr["away_games"] += 1
                tr["away_correct"] += 1 if g["winner_correct"] else 0

    per_team = {}
    for tid, tr in team_results.items():
        n = tr["games"]
        abbr = team_abbrev.get(tid, str(tid))
        per_team[abbr] = {
            "team_id": tid,
            "games": n,
            "spread_accuracy": round(tr["within_5"] / n * 100, 1),
            "total_accuracy": round(tr["within_10"] / n * 100, 1),
            "avg_spread_error": round(sum(tr["spread_errors"]) / n, 2),
            "avg_total_error": round(sum(tr["total_errors"]) / n, 1),
            "wins": tr["correct"],
            "losses": n - tr["correct"],
            "winner_pct": round(tr["correct"] / n * 100, 1),
            "home_winner_pct": round(tr["home_correct"] / tr["home_games"] * 100, 1) if tr["home_games"] else 0,
            "away_winner_pct": round(tr["away_correct"] / tr["away_games"] * 100, 1) if tr["away_games"] else 0,
        }

    # ---- Home vs Away global breakdown ---
    home_games = [g for g in per_game]
    home_correct = sum(1 for g in per_game
                       if g["winner_correct"] and g.get("pred_spread", 0) > 0)
    away_correct = sum(1 for g in per_game
                       if g["winner_correct"] and g.get("pred_spread", 0) < 0)
    home_predicted = sum(1 for g in per_game if g.get("pred_spread", 0) > 0)
    away_predicted = sum(1 for g in per_game if g.get("pred_spread", 0) < 0)

    # Home team won / lost counts
    actual_home_wins = sum(1 for g in per_game if g["actual_spread"] > 0)
    actual_away_wins = total - actual_home_wins
    pred_home_wins_correct = sum(1 for g in per_game
                                  if g.get("pred_spread", 0) > 0 and g["actual_spread"] > 0)
    pred_away_wins_correct = sum(1 for g in per_game
                                  if g.get("pred_spread", 0) < 0 and g["actual_spread"] < 0)

    home_away_breakdown = {
        "actual_home_wins": actual_home_wins,
        "actual_away_wins": actual_away_wins,
        "pred_home_wins": home_predicted,
        "pred_away_wins": away_predicted,
        "home_pred_accuracy": round(pred_home_wins_correct / home_predicted * 100, 1) if home_predicted else 0,
        "away_pred_accuracy": round(pred_away_wins_correct / away_predicted * 100, 1) if away_predicted else 0,
        "overall_home_win_rate": round(actual_home_wins / total * 100, 1),
    }

    # ---- Spread error bins ---
    spread_bins = [
        ("0-3 (nail-biter)", 0, 3),
        ("3-7 (comfortable)", 3, 7),
        ("7-12 (blowout)", 7, 12),
        ("12+ (landslide)", 12, 999),
    ]
    spread_range_analysis = []
    for label, lo, hi in spread_bins:
        bucket = [g for g in per_game if lo <= abs(g["actual_spread"]) < hi]
        if bucket:
            bc = sum(1 for g in bucket if g["winner_correct"])
            bw5 = sum(1 for g in bucket if g["spread_within_5"])
            avg_err = sum(g["spread_error"] for g in bucket) / len(bucket)
            spread_range_analysis.append({
                "range": label,
                "games": len(bucket),
                "winner_pct": round(bc / len(bucket) * 100, 1),
                "within_5_pct": round(bw5 / len(bucket) * 100, 1),
                "avg_error": round(avg_err, 2),
            })

    # ---- Total score bins ---
    total_bins = [
        ("Under 200", 0, 200),
        ("200-215", 200, 215),
        ("215-230", 215, 230),
        ("230-245", 230, 245),
        ("245+", 245, 9999),
    ]
    total_range_analysis = []
    for label, lo, hi in total_bins:
        bucket = [g for g in per_game if lo <= g["actual_total"] < hi]
        if bucket:
            bw10 = sum(1 for g in bucket if g["total_within_10"])
            avg_err = sum(g["total_error"] for g in bucket) / len(bucket)
            total_range_analysis.append({
                "range": label,
                "games": len(bucket),
                "within_10_pct": round(bw10 / len(bucket) * 100, 1),
                "avg_error": round(avg_err, 1),
            })

    # ---- Bias check: do we systematically over/under predict? ---
    avg_pred_spread = sum(g["pred_spread"] for g in per_game) / total
    avg_actual_spread = sum(g["actual_spread"] for g in per_game) / total
    avg_pred_total = sum(g["pred_total"] for g in per_game) / total
    avg_actual_total = sum(g["actual_total"] for g in per_game) / total

    bias = {
        "avg_pred_spread": round(avg_pred_spread, 2),
        "avg_actual_spread": round(avg_actual_spread, 2),
        "spread_bias": round(avg_pred_spread - avg_actual_spread, 2),
        "avg_pred_total": round(avg_pred_total, 1),
        "avg_actual_total": round(avg_actual_total, 1),
        "total_bias": round(avg_pred_total - avg_actual_total, 1),
    }

    quality = compute_quality_metrics(per_game, per_team)
    progression = compute_progression(per_game)
    vegas_comparison = compute_vegas_comparison(per_game)
    quality["progression"] = progression
    quality["vegas_comparison"] = vegas_comparison

    summary = {
        "total_games": total,
        "overall_spread_accuracy": round(correct / total * 100, 1),
        "overall_total_accuracy": round(within_10 / total * 100, 1),
        "avg_spread_error": round(avg_spread_err, 2),
        "avg_total_error": round(avg_total_err, 1),
        "spread_within_5_pct": round(within_5 / total * 100, 1),
        "total_within_10_pct": round(within_10 / total * 100, 1),
        "per_team": per_team,
        "per_game": per_game,
        "home_away": home_away_breakdown,
        "spread_ranges": spread_range_analysis,
        "total_ranges": total_range_analysis,
        "bias": bias,
        "quality_metrics": quality,
    }

    # Cache (global only)
    if team_id is None:
        _save_cache(summary)

    if callback:
        callback(f"Backtest complete: {total} games, Winner={correct / total * 100:.1f}%, "
                 f"Spread MAE={avg_spread_err:.2f}, Total MAE={avg_total_err:.1f}")

    return summary


# ---------------------------------------------------------------------------
# Diagnostic CSV export — detailed game-by-game breakdown for weak teams
# ---------------------------------------------------------------------------

def export_diagnostic_csv(
    backtest_results: Optional[Dict] = None,
    n_worst: int = 30,
    output_dir: str = "data",
    callback: Optional[Callable] = None,
) -> str:
    """Export detailed diagnostic CSV for the N worst-predicted teams.

    For each game involving one of the bottom teams, re-runs predict_matchup to
    capture every intermediate adjustment value, team metric, projection, and
    weight used — providing full visibility into *why* a prediction failed.

    Returns the path to the written CSV.
    """
    import csv
    from dataclasses import fields as dc_fields

    # 1. Get or run backtest
    if backtest_results is None:
        if callback:
            callback("Running backtest for diagnostic export...")
        backtest_results = run_backtest(use_cache=True, callback=callback)

    per_team = backtest_results.get("per_team", {})
    per_game = backtest_results.get("per_game", [])
    if not per_team or not per_game:
        if callback:
            callback("No backtest data available")
        return ""

    # 2. Rank teams by winner accuracy (ascending = worst first)
    ranked = sorted(per_team.items(), key=lambda kv: kv[1].get("winner_pct", 0))
    worst_teams = ranked[:n_worst]
    worst_abbrevs = {t[0] for t in worst_teams}

    # Resolve abbreviation -> team_id
    abbr_to_id: Dict[str, int] = {}
    for abbr, data in worst_teams:
        tid = data.get("team_id")
        if tid:
            abbr_to_id[abbr] = tid
    worst_ids = set(abbr_to_id.values())

    if callback:
        summary_lines = [f"  {abbr}: {data.get('winner_pct', 0):.1f}% winner accuracy, "
                         f"MAE={data.get('avg_spread_error', 0):.2f}"
                         for abbr, data in worst_teams]
        callback(f"Bottom {n_worst} teams:\n" + "\n".join(summary_lines))

    # 3. Filter games involving worst teams
    target_games = [
        g for g in per_game
        if g["home_team_id"] in worst_ids or g["away_team_id"] in worst_ids
    ]
    target_games.sort(key=lambda g: g["game_date"])

    if callback:
        callback(f"Re-predicting {len(target_games)} games with full diagnostics...")

    # 4. Team abbreviation lookup
    team_abbrev: Dict[int, str] = {}
    try:
        rows = db.fetch_all("SELECT team_id, abbreviation FROM teams")
        team_abbrev = {r["team_id"]: r["abbreviation"] for r in rows}
    except Exception:
        pass

    # 5. Re-run predictions capturing all detail
    csv_rows = []
    start_session_caches()

    for i, g in enumerate(target_games):
        htid = g["home_team_id"]
        atid = g["away_team_id"]
        gdate = g["game_date"]

        try:
            pred = predict_matchup(
                htid, atid,
                game_date=gdate,
                as_of_date=gdate,
                skip_espn=True,
            )

            # Get the raw metrics/projections used (re-fetch to include in CSV)
            from src.analytics.stats_engine import (
                aggregate_projection, get_home_court_advantage,
                compute_fatigue, _LEAGUE_AVG_PPG, _PACE_FALLBACK, _RATING_FALLBACK,
            )
            from src.analytics.prediction import _get_team_metrics, _get_tuning

            hm = _get_team_metrics(htid)
            am = _get_team_metrics(atid)
            ht = _get_tuning(htid)
            at = _get_tuning(atid)

            # Weights used (load before fatigue so we can pass w for b2b/3in4/4in6)
            from src.analytics.weight_config import load_team_weights, get_weight_config
            home_w = load_team_weights(htid)
            away_w = load_team_weights(atid)
            if home_w and away_w:
                w = home_w.blend(away_w)
            elif home_w:
                w = home_w
            elif away_w:
                w = away_w
            else:
                w = get_weight_config()

            hfat = compute_fatigue(htid, gdate, w=w)
            afat = compute_fatigue(atid, gdate, w=w)

            home_abbr = team_abbrev.get(htid, str(htid))
            away_abbr = team_abbrev.get(atid, str(atid))

            # Determine which focus teams to emit rows for
            _focus_teams = []
            if htid in worst_ids:
                _focus_teams.append(home_abbr)
            if atid in worst_ids and atid != htid:
                _focus_teams.append(away_abbr)
            if not _focus_teams:
                _focus_teams.append(home_abbr)

            row = {
                # ── Game identity ──
                "game_date": gdate,
                "game_id": g["game_id"],
                "home_team": home_abbr,
                "away_team": away_abbr,
                "focus_team": _focus_teams[0],  # replaced below for multi-focus

                # ── Actual results ──
                "actual_home_score": (g.get("actual_total", 0) + g.get("actual_spread", 0)) / 2 if g.get("actual_total") else 0,
                "actual_away_score": (g.get("actual_total", 0) - g.get("actual_spread", 0)) / 2 if g.get("actual_total") else 0,
                "actual_spread": g["actual_spread"],
                "actual_total": g["actual_total"],

                # ── Predicted results ──
                "pred_home_score": pred.predicted_home_score,
                "pred_away_score": pred.predicted_away_score,
                "pred_spread": pred.predicted_spread,
                "pred_total": pred.predicted_total,
                "pred_winner": pred.winner,

                # ── Errors ──
                "spread_error": g["spread_error"],
                "total_error": g["total_error"],
                "winner_correct": g["winner_correct"],
                "spread_within_5": g["spread_within_5"],

                # ── Step-by-step adjustments ──
                "home_court_adv": pred.home_court_advantage,
                "fatigue_adj": pred.fatigue_adj,
                "home_fatigue": pred.home_fatigue,
                "away_fatigue": pred.away_fatigue,
                "turnover_adj": pred.turnover_adj,
                "rebound_adj": pred.rebound_adj,
                "rating_matchup_adj": pred.rating_matchup_adj,
                "four_factors_adj": pred.four_factors_adj,
                "clutch_adj": pred.clutch_adj,
                "hustle_adj": pred.hustle_adj,
                "pace_adj": pred.pace_adj,
                "defensive_disruption": pred.defensive_disruption,
                "oreb_boost": pred.oreb_boost,
                "hustle_total_adj": pred.hustle_total_adj,
                "ml_blend_adj": pred.ml_blend_adj,
                "ml_spread": pred.ml_spread,
                "ml_total": pred.ml_total,
                "confidence": pred.confidence,

                # ── Home projections ──
                "home_proj_pts": pred.home_proj.get("points", 0),
                "home_proj_reb": pred.home_proj.get("rebounds", 0),
                "home_proj_ast": pred.home_proj.get("assists", 0),
                "home_proj_to": pred.home_proj.get("turnovers", 0),
                "home_proj_stl": pred.home_proj.get("steals", 0),
                "home_proj_blk": pred.home_proj.get("blocks", 0),
                "home_proj_oreb": pred.home_proj.get("oreb", 0),
                "home_proj_fg_pct": pred.home_proj.get("fg_pct", 0),
                "home_proj_fg3_pct": pred.home_proj.get("fg3_pct", 0),
                "home_proj_ft_pct": pred.home_proj.get("ft_pct", 0),

                # ── Away projections ──
                "away_proj_pts": pred.away_proj.get("points", 0),
                "away_proj_reb": pred.away_proj.get("rebounds", 0),
                "away_proj_ast": pred.away_proj.get("assists", 0),
                "away_proj_to": pred.away_proj.get("turnovers", 0),
                "away_proj_stl": pred.away_proj.get("steals", 0),
                "away_proj_blk": pred.away_proj.get("blocks", 0),
                "away_proj_oreb": pred.away_proj.get("oreb", 0),
                "away_proj_fg_pct": pred.away_proj.get("fg_pct", 0),
                "away_proj_fg3_pct": pred.away_proj.get("fg3_pct", 0),
                "away_proj_ft_pct": pred.away_proj.get("ft_pct", 0),

                # ── Home team metrics ──
                "home_off_rating": hm.get("off_rating", 0),
                "home_def_rating": hm.get("def_rating", 0),
                "home_net_rating": (hm.get("off_rating", 0) or 0) - (hm.get("def_rating", 0) or 0),
                "home_pace": hm.get("pace", 0),
                "home_opp_pts": hm.get("opp_pts", 0),
                "home_ff_efg": hm.get("ff_efg_pct", 0),
                "home_ff_tov": hm.get("ff_tm_tov_pct", 0),
                "home_ff_oreb": hm.get("ff_oreb_pct", 0),
                "home_ff_fta_rate": hm.get("ff_fta_rate", 0),
                "home_opp_efg": hm.get("opp_efg_pct", 0),
                "home_opp_tov": hm.get("opp_tm_tov_pct", 0),
                "home_opp_oreb": hm.get("opp_oreb_pct", 0),
                "home_opp_fta_rate": hm.get("opp_fta_rate", 0),
                "home_clutch_net": hm.get("clutch_net_rating", 0),
                "home_deflections_pg": (hm.get("deflections", 0) or 0) / max(1, hm.get("gp", 1) or 1),
                "home_contested_pg": (hm.get("contested_shots", 0) or 0) / max(1, hm.get("gp", 1) or 1),
                "home_gp": hm.get("gp", 0),

                # ── Away team metrics ──
                "away_off_rating": am.get("off_rating", 0),
                "away_def_rating": am.get("def_rating", 0),
                "away_net_rating": (am.get("off_rating", 0) or 0) - (am.get("def_rating", 0) or 0),
                "away_pace": am.get("pace", 0),
                "away_opp_pts": am.get("opp_pts", 0),
                "away_ff_efg": am.get("ff_efg_pct", 0),
                "away_ff_tov": am.get("ff_tm_tov_pct", 0),
                "away_ff_oreb": am.get("ff_oreb_pct", 0),
                "away_ff_fta_rate": am.get("ff_fta_rate", 0),
                "away_opp_efg": am.get("opp_efg_pct", 0),
                "away_opp_tov": am.get("opp_tm_tov_pct", 0),
                "away_opp_oreb": am.get("opp_oreb_pct", 0),
                "away_opp_fta_rate": am.get("opp_fta_rate", 0),
                "away_clutch_net": am.get("clutch_net_rating", 0),
                "away_deflections_pg": (am.get("deflections", 0) or 0) / max(1, am.get("gp", 1) or 1),
                "away_contested_pg": (am.get("contested_shots", 0) or 0) / max(1, am.get("gp", 1) or 1),
                "away_gp": am.get("gp", 0),

                # ── Autotune corrections ──
                "home_tune_home_corr": ht["home_pts_correction"],
                "home_tune_away_corr": ht.get("away_pts_correction", 0),
                "away_tune_home_corr": at.get("home_pts_correction", 0),
                "away_tune_away_corr": at["away_pts_correction"],

                # ── Fatigue detail ──
                "home_fatigue_penalty": hfat["penalty"],
                "away_fatigue_penalty": afat["penalty"],
                "home_fatigue_b2b": hfat.get("is_b2b", False),
                "away_fatigue_b2b": afat.get("is_b2b", False),

                # ── Key weights applied ──
                "w_def_factor_dampening": w.def_factor_dampening,
                "w_turnover_margin_mult": w.turnover_margin_mult,
                "w_rebound_diff_mult": w.rebound_diff_mult,
                "w_rating_matchup_mult": w.rating_matchup_mult,
                "w_four_factors_scale": w.four_factors_scale,
                "w_ff_efg_weight": w.ff_efg_weight,
                "w_ff_tov_weight": w.ff_tov_weight,
                "w_ff_oreb_weight": w.ff_oreb_weight,
                "w_ff_fta_weight": w.ff_fta_weight,
                "w_clutch_scale": w.clutch_scale,
                "w_hustle_effort_mult": w.hustle_effort_mult,
                "w_pace_mult": w.pace_mult,
                "w_ml_ensemble_weight": w.ml_ensemble_weight,
            }
            # Emit one row per focus team (both home & away when both in worst set)
            for ft in _focus_teams:
                r = dict(row)
                r["focus_team"] = ft
                csv_rows.append(r)

        except Exception as e:
            logger.warning("Diagnostic skip game %s: %s", g.get("game_id"), e)

        if callback and (i + 1) % 20 == 0:
            callback(f"Diagnostic: {i + 1}/{len(target_games)} games processed")

    stop_session_caches()

    if not csv_rows:
        if callback:
            callback("No diagnostic rows generated")
        return ""

    # 6. Write CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "diagnostic_worst_teams.csv")
    fieldnames = list(csv_rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    if callback:
        callback(f"Diagnostic CSV written: {csv_path} ({len(csv_rows)} rows, "
                 f"{len(fieldnames)} columns)")
    logger.info("Diagnostic CSV: %s (%d rows)", csv_path, len(csv_rows))
    return csv_path
