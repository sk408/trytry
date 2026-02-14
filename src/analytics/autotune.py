"""Per-team autotune with distribution-aware spread-shift optimisation.

Two modes:
- **classic** – uses all historical games for each team.
- **walk_forward** – uses only the most recent rolling window.

The optimiser minimises a composite score that blends:
  * winner-correct rate (highest weight)
  * mean absolute spread error
  * 90th-percentile tail error

A per-team acceptance guardrail rejects corrections that don't improve
the composite score on the source data.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.config import get_default_workers as _get_default_workers

from src.analytics.backtester import get_actual_game_results
from src.analytics.stats_engine import (
    aggregate_projection,
    get_opponent_defensive_factor,
)
from src.database.db import get_conn


# ════════════════════════════════════════════════════════════════════════
#  Internal: simulate a game prediction from player logs
# ════════════════════════════════════════════════════════════════════════


def _predict_game_player_level(
    team_id: int,
    opponent_id: int,
    game_date: str,
    is_home: bool,
) -> Optional[float]:
    """Simulate what predict_matchup() would produce for a historical game
    using the actual players who played that game.

    Uses *current* full-season stats (standard backtesting approach).
    Returns the defense-adjusted projected points for *team_id*, or None
    if no player data is available for that game.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ps.player_id
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ? AND ps.game_date = ?
            """,
            (team_id, str(game_date)),
        ).fetchall()

    player_ids = [r[0] for r in rows]
    if not player_ids:
        return None

    # Use full-season stats (no before_date filtering)
    proj = aggregate_projection(
        player_ids, opponent_team_id=opponent_id, is_home=is_home
    )

    # Apply opponent defensive factor (same dampening as predict_matchup)
    def_factor = get_opponent_defensive_factor(opponent_id)
    def_factor = 1.0 + (def_factor - 1.0) * 0.5
    adjusted_pts = proj["points"] * def_factor

    return adjusted_pts


# ════════════════════════════════════════════════════════════════════════
#  Distribution-aware scoring helpers
# ════════════════════════════════════════════════════════════════════════


def _winner_side(spread: float) -> int:
    """Return +1 home, -1 away, 0 push-ish using betting threshold."""
    if spread > 0.5:
        return 1
    if spread < -0.5:
        return -1
    return 0


def _distribution_metrics(
    pairs: List[tuple[float, float]], shift: float
) -> dict:
    """Compute error distribution metrics after applying *shift*."""
    if not pairs:
        return {"wrong_rate": 0.0, "mae": 0.0, "p90": 0.0}
    abs_errs: List[float] = []
    wrong = 0
    for pred_spread, actual_spread in pairs:
        shifted = pred_spread + shift
        err = shifted - actual_spread
        abs_errs.append(abs(err))
        if _winner_side(shifted) != _winner_side(actual_spread):
            wrong += 1
    abs_errs_sorted = sorted(abs_errs)
    idx90 = int(0.9 * (len(abs_errs_sorted) - 1)) if len(abs_errs_sorted) > 1 else 0
    p90 = abs_errs_sorted[idx90] if abs_errs_sorted else 0.0
    return {
        "wrong_rate": wrong / len(pairs),
        "mae": sum(abs_errs) / len(abs_errs),
        "p90": p90,
    }


def _combined_distribution_score(
    pairs: List[tuple[float, float]], shift: float
) -> float:
    """Lower is better; weighted blend of correctness + miss magnitude + tail risk."""
    if not pairs:
        return 0.0
    m = _distribution_metrics(pairs, shift)
    return (3.0 * m["wrong_rate"]) + (1.0 * m["mae"]) + (0.25 * m["p90"])


def _optimise_spread_shift(
    pairs: List[tuple[float, float]],
    max_abs_shift: float,
) -> tuple[float, dict]:
    """Grid-search over shift values for the best composite score."""
    if not pairs:
        return 0.0, {"best_shift": 0.0}
    step = 0.25
    lo = -abs(max_abs_shift)
    hi = abs(max_abs_shift)
    best_shift = 0.0
    best_score = _combined_distribution_score(pairs, 0.0)
    cur = lo
    while cur <= hi + 1e-9:
        s = _combined_distribution_score(pairs, cur)
        if s < best_score:
            best_score = s
            best_shift = cur
        cur += step
    return best_shift, {"best_shift": best_shift, "best_score": best_score}


# ════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════


def autotune_team(
    team_id: int,
    strength: float = 0.75,
    min_threshold: float = 1.5,
    mode: str = "classic",
    max_abs_correction: float = 9.0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Analyse a team's historical games, compare player-level predictions
    to actual results, and compute per-team scoring corrections.

    Args:
        team_id: The team to tune.
        strength: 0.0-1.0 how aggressively to apply the correction.
        min_threshold: Minimum average error (pts) to store a correction.
        mode: ``"classic"`` (all games) or ``"walk_forward"`` (recent window).
        max_abs_correction: Maximum absolute correction value.
        progress_cb: Optional callback for progress messages.

    Returns:
        Dict with correction values and diagnostic metrics.
    """
    progress = progress_cb or (lambda _: None)

    all_games = get_actual_game_results()
    if all_games.empty:
        progress("No game data found")
        return _empty_result(team_id)

    # Games where this team was home / away
    home_games = all_games[all_games["home_team_id"] == team_id].copy()
    away_games = all_games[all_games["away_team_id"] == team_id].copy()
    home_games = home_games.sort_values("game_date")
    away_games = away_games.sort_values("game_date")

    # Skip the first 5 games (not enough data to predict reliably)
    home_games = home_games.iloc[5:] if len(home_games) > 5 else home_games.iloc[0:0]
    away_games = away_games.iloc[5:] if len(away_games) > 5 else away_games.iloc[0:0]

    home_score_errors: List[float] = []
    away_score_errors: List[float] = []
    home_spread_pairs: List[tuple[float, float]] = []
    away_spread_pairs: List[tuple[float, float]] = []
    home_spread_errors_all: List[float] = []
    away_spread_errors_all: List[float] = []
    home_total_errors_all: List[float] = []
    away_total_errors_all: List[float] = []
    spread_errors: List[float] = []
    total_errors: List[float] = []

    total_to_process = len(home_games) + len(away_games)
    processed = 0

    # ----- HOME GAMES -----
    for _, game in home_games.iterrows():
        gd = game["game_date"]
        away_id = int(game["away_team_id"])
        actual_home = float(game["home_score"])
        actual_away = float(game["away_score"])

        pred_home = _predict_game_player_level(team_id, away_id, gd, is_home=True)
        pred_away = _predict_game_player_level(away_id, team_id, gd, is_home=False)

        if pred_home is not None:
            home_score_errors.append(pred_home - actual_home)
        if pred_home is not None and pred_away is not None:
            pred_spread = pred_home - pred_away
            pred_total = pred_home + pred_away
            actual_spread = actual_home - actual_away
            actual_total = actual_home + actual_away
            se = pred_spread - actual_spread
            spread_errors.append(se)
            home_spread_pairs.append((pred_spread, actual_spread))
            home_spread_errors_all.append(se)
            home_total_errors_all.append(pred_total - actual_total)
            total_errors.append(pred_total - actual_total)

        processed += 1
        if processed % 10 == 0:
            progress(f"  Processed {processed}/{total_to_process} games...")

    # ----- AWAY GAMES -----
    for _, game in away_games.iterrows():
        gd = game["game_date"]
        home_id = int(game["home_team_id"])
        actual_home = float(game["home_score"])
        actual_away = float(game["away_score"])

        pred_away = _predict_game_player_level(team_id, home_id, gd, is_home=False)
        pred_home = _predict_game_player_level(home_id, team_id, gd, is_home=True)

        if pred_away is not None:
            away_score_errors.append(pred_away - actual_away)
        if pred_home is not None and pred_away is not None:
            pred_spread = pred_home - pred_away
            pred_total = pred_home + pred_away
            actual_spread = actual_home - actual_away
            actual_total = actual_home + actual_away
            se = pred_spread - actual_spread
            spread_errors.append(se)
            away_spread_pairs.append((pred_spread, actual_spread))
            away_spread_errors_all.append(se)
            away_total_errors_all.append(pred_total - actual_total)
            total_errors.append(pred_total - actual_total)

        processed += 1
        if processed % 10 == 0:
            progress(f"  Processed {processed}/{total_to_process} games...")

    # ----- COMPUTE CORRECTIONS -----
    n = len(home_score_errors) + len(away_score_errors)
    if n == 0:
        progress("  No games with player data found")
        return _empty_result(team_id)

    if mode == "walk_forward":
        window = 20
        home_source = home_spread_pairs[-window:] if home_spread_pairs else []
        away_source = away_spread_pairs[-window:] if away_spread_pairs else []
    else:
        home_source = home_spread_pairs
        away_source = away_spread_pairs

    # Distribution-aware spread shift optimisation
    home_shift, home_stats = _optimise_spread_shift(
        home_source, max_abs_shift=max_abs_correction,
    )
    away_shift, away_stats = _optimise_spread_shift(
        away_source, max_abs_shift=max_abs_correction,
    )

    # Confidence dampening for small sample sizes
    confidence = min(1.0, n / 15.0)
    home_shift *= strength * confidence
    away_shift *= strength * confidence

    home_correction = max(-max_abs_correction, min(max_abs_correction, home_shift))
    away_correction = max(-max_abs_correction, min(max_abs_correction, -away_shift))

    # Acceptance guardrail: only keep correction if it improves the score
    if home_source or away_source:
        before_score = (
            _combined_distribution_score(home_source, 0.0)
            + _combined_distribution_score(away_source, 0.0)
        )
        after_score = (
            _combined_distribution_score(home_source, home_correction)
            + _combined_distribution_score(away_source, -away_correction)
        )
        if after_score >= before_score:
            home_correction = 0.0
            away_correction = 0.0
            progress(
                "  Guardrail: rejected tuning "
                f"(dist_score {before_score:.3f} -> {after_score:.3f})"
            )
        else:
            progress(
                "  Accepted tuning "
                f"(home_shift={home_stats['best_shift']:+.2f}, "
                f"away_shift={away_stats['best_shift']:+.2f}, "
                f"dist_score {before_score:.3f} -> {after_score:.3f})"
            )

    # Only store if error exceeds minimum threshold
    if abs(home_correction) < min_threshold:
        home_correction = 0.0
    if abs(away_correction) < min_threshold:
        away_correction = 0.0

    # Diagnostic stats (before vs after corrections)
    avg_spread_err = (
        sum(abs(e) for e in spread_errors) / len(spread_errors)
        if spread_errors else 0.0
    )
    avg_total_err = (
        sum(abs(e) for e in total_errors) / len(total_errors)
        if total_errors else 0.0
    )
    spread_after_abs = (
        [abs(se + home_correction) for se in home_spread_errors_all]
        + [abs(se - away_correction) for se in away_spread_errors_all]
    )
    total_after_abs = (
        [abs(te + home_correction) for te in home_total_errors_all]
        + [abs(te + away_correction) for te in away_total_errors_all]
    )
    avg_spread_err_after = (
        sum(spread_after_abs) / len(spread_after_abs)
        if spread_after_abs else avg_spread_err
    )
    avg_total_err_after = (
        sum(total_after_abs) / len(total_after_abs)
        if total_after_abs else avg_total_err
    )

    result = {
        "team_id": team_id,
        "home_pts_correction": round(home_correction, 2),
        "away_pts_correction": round(away_correction, 2),
        "games_analyzed": n,
        "avg_spread_error_before": round(avg_spread_err, 2),
        "avg_total_error_before": round(avg_total_err, 2),
        "avg_spread_error_after": round(avg_spread_err_after, 2),
        "avg_total_error_after": round(avg_total_err_after, 2),
        "tuning_mode": mode,
        "tuning_version": "v2_walk_forward" if mode == "walk_forward" else "v2_classic",
        "tuning_sample_size": len(home_source) + len(away_source),
    }

    _save_tuning(result)
    progress(
        f"  Done: {n} games, home_adj={home_correction:+.2f}, "
        f"away_adj={away_correction:+.2f}, "
        f"spread_mae {avg_spread_err:.2f}->{avg_spread_err_after:.2f}, "
        f"total_mae {avg_total_err:.2f}->{avg_total_err_after:.2f}"
    )
    return result


def autotune_all(
    strength: float = 0.75,
    min_threshold: float = 1.5,
    mode: str = "classic",
    max_abs_correction: float = 9.0,
    require_global_improvement: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
    max_workers: int | None = None,
) -> List[Dict]:
    """Run autotune for every team in the database.

    Uses ``max_workers`` threads to process teams in parallel.
    """
    if max_workers is None:
        max_workers = _get_default_workers()
    progress = progress_cb or (lambda _: None)

    baseline_spread_pct = None
    baseline_snapshot: list[tuple] = []
    if require_global_improvement:
        baseline_snapshot = _snapshot_team_tuning_rows()
        try:
            from src.analytics.backtester import run_backtest
            base_bt = run_backtest(min_games_before=5, progress_cb=None, max_workers=max_workers)
            baseline_spread_pct = float(base_bt.overall_spread_accuracy)
            progress(f"Global guardrail baseline spread%: {baseline_spread_pct:.2f}")
        except Exception as exc:
            progress(f"Global guardrail baseline check failed: {exc}")
            require_global_improvement = False

    with get_conn() as conn:
        teams = conn.execute(
            "SELECT team_id, abbreviation FROM teams ORDER BY abbreviation"
        ).fetchall()

    if not teams:
        progress("No teams found in database")
        return []

    n_workers = max(1, min(max_workers, len(teams)))
    progress(f"Autotuning {len(teams)} teams with {n_workers} workers (mode={mode})...")

    results: List[Dict] = []
    completed = 0
    _lock = threading.Lock()

    def _tune_one(tid: int, abbr: str) -> Dict:
        nonlocal completed
        res = autotune_team(
            tid,
            strength=strength,
            min_threshold=min_threshold,
            mode=mode,
            max_abs_correction=max_abs_correction,
        )
        with _lock:
            completed += 1
            progress(
                f"[{completed}/{len(teams)}] {abbr}: "
                f"home={res['home_pts_correction']:+.2f}, "
                f"away={res['away_pts_correction']:+.2f}, "
                f"spread_mae {res.get('avg_spread_error_before', 0):.2f}->"
                f"{res.get('avg_spread_error_after', 0):.2f}"
            )
        return res

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_tune_one, tid, abbr): (tid, abbr)
            for tid, abbr in teams
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                tid, abbr = futures[future]
                progress(f"  {abbr} autotune error: {exc}")
                results.append(_empty_result(tid))

    tuned_count = sum(
        1 for r in results
        if r["home_pts_correction"] != 0 or r["away_pts_correction"] != 0
    )
    progress(
        f"Autotune complete: {len(teams)} teams analyzed, "
        f"{tuned_count} received corrections"
    )

    # Optional global rollback guardrail
    if require_global_improvement and baseline_spread_pct is not None:
        try:
            from src.analytics.backtester import run_backtest
            tuned_bt = run_backtest(
                min_games_before=5, progress_cb=None, max_workers=max_workers
            )
            tuned_spread_pct = float(tuned_bt.overall_spread_accuracy)
            progress(
                f"Global guardrail post-tune spread%: {tuned_spread_pct:.2f} "
                f"(baseline {baseline_spread_pct:.2f})"
            )
            if tuned_spread_pct < baseline_spread_pct:
                _restore_team_tuning_rows(baseline_snapshot)
                progress(
                    "Global guardrail: reverted autotune because spread% got worse "
                    f"({baseline_spread_pct:.2f} -> {tuned_spread_pct:.2f})"
                )
                for r in results:
                    r["home_pts_correction"] = 0.0
                    r["away_pts_correction"] = 0.0
        except Exception as exc:
            progress(f"Global guardrail post-check failed: {exc}")

    return results


# ════════════════════════════════════════════════════════════════════════
#  DB helpers
# ════════════════════════════════════════════════════════════════════════


def _empty_result(team_id: int) -> Dict:
    return {
        "team_id": team_id,
        "home_pts_correction": 0.0,
        "away_pts_correction": 0.0,
        "games_analyzed": 0,
        "avg_spread_error_before": 0.0,
        "avg_total_error_before": 0.0,
        "avg_spread_error_after": 0.0,
        "avg_total_error_after": 0.0,
        "tuning_mode": "classic",
        "tuning_version": "v1_classic",
        "tuning_sample_size": 0,
    }


def _save_tuning(result: Dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO team_tuning
                (team_id, home_pts_correction, away_pts_correction,
                 games_analyzed, avg_spread_error_before, avg_total_error_before,
                 last_tuned_at, tuning_mode, tuning_version, tuning_sample_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["team_id"],
                result["home_pts_correction"],
                result["away_pts_correction"],
                result["games_analyzed"],
                result["avg_spread_error_before"],
                result["avg_total_error_before"],
                datetime.utcnow().isoformat(),
                result.get("tuning_mode", "classic"),
                result.get("tuning_version", "v1_classic"),
                result.get("tuning_sample_size", result.get("games_analyzed", 0)),
            ),
        )
        conn.commit()
    # Invalidate cache for this team
    _tuning_cache.pop(result["team_id"], None)


def _snapshot_team_tuning_rows() -> list[tuple]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT team_id, home_pts_correction, away_pts_correction,
                   games_analyzed, avg_spread_error_before, avg_total_error_before,
                   last_tuned_at, tuning_mode, tuning_version, tuning_sample_size
            FROM team_tuning
            ORDER BY team_id
            """
        ).fetchall()


def _restore_team_tuning_rows(rows: list[tuple]) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM team_tuning")
        for r in rows:
            conn.execute(
                """
                INSERT INTO team_tuning
                    (team_id, home_pts_correction, away_pts_correction,
                     games_analyzed, avg_spread_error_before, avg_total_error_before,
                     last_tuned_at, tuning_mode, tuning_version, tuning_sample_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]),
            )
        conn.commit()
    _clear_tuning_cache()


# ════════════════════════════════════════════════════════════════════════
#  Tuning cache (avoids repeated DB lookups during backtesting)
# ════════════════════════════════════════════════════════════════════════

_tuning_cache: Dict[int, tuple] = {}  # team_id -> (timestamp, dict|None)
_TUNING_CACHE_TTL: float = 600.0  # 10 minutes


def _clear_tuning_cache() -> None:
    _tuning_cache.clear()


def get_team_tuning(team_id: int) -> Optional[Dict]:
    """Load per-team tuning corrections.  Returns None if no tuning exists."""
    import time as _time
    now = _time.monotonic()
    entry = _tuning_cache.get(team_id)
    if entry is not None:
        ts, val = entry
        if now - ts < _TUNING_CACHE_TTL:
            return val

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT team_id, home_pts_correction, away_pts_correction, games_analyzed,
                   avg_spread_error_before, avg_total_error_before, last_tuned_at,
                   tuning_mode, tuning_version, tuning_sample_size
            FROM team_tuning WHERE team_id = ?
            """,
            (team_id,),
        ).fetchone()
    if not row:
        result = None
    else:
        result = {
            "team_id": row[0],
            "home_pts_correction": row[1],
            "away_pts_correction": row[2],
            "games_analyzed": row[3],
            "avg_spread_error_before": row[4],
            "avg_total_error_before": row[5],
            "last_tuned_at": row[6],
            "tuning_mode": row[7] if len(row) > 7 else "classic",
            "tuning_version": row[8] if len(row) > 8 else "v1_classic",
            "tuning_sample_size": row[9] if len(row) > 9 else row[3],
        }
    _tuning_cache[team_id] = (now, result)
    return result


def get_all_tunings() -> List[Dict]:
    """Return all saved team tunings with team abbreviation/name."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                tt.team_id,
                tt.home_pts_correction,
                tt.away_pts_correction,
                tt.games_analyzed,
                tt.avg_spread_error_before,
                tt.avg_total_error_before,
                tt.last_tuned_at,
                tt.tuning_mode,
                tt.tuning_version,
                tt.tuning_sample_size,
                t.abbreviation,
                t.name
            FROM team_tuning tt
            JOIN teams t ON t.team_id = tt.team_id
            ORDER BY t.abbreviation
            """
        ).fetchall()
    return [
        {
            "team_id": r[0],
            "home_pts_correction": r[1],
            "away_pts_correction": r[2],
            "games_analyzed": r[3],
            "avg_spread_error_before": r[4],
            "avg_total_error_before": r[5],
            "last_tuned_at": r[6],
            "tuning_mode": r[7] if len(r) > 7 else "classic",
            "tuning_version": r[8] if len(r) > 8 else "v1_classic",
            "tuning_sample_size": r[9] if len(r) > 9 else r[3],
            "abbr": r[10],
            "name": r[11],
        }
        for r in rows
    ]


def clear_tuning(team_id: Optional[int] = None) -> None:
    """Clear tuning for one team (if team_id given) or all teams."""
    with get_conn() as conn:
        if team_id is not None:
            conn.execute("DELETE FROM team_tuning WHERE team_id = ?", (team_id,))
        else:
            conn.execute("DELETE FROM team_tuning")
        conn.commit()
    if team_id is not None:
        _tuning_cache.pop(team_id, None)
    else:
        _clear_tuning_cache()
