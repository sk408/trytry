from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.analytics.backtester import get_actual_game_results
from src.analytics.stats_engine import (
    aggregate_projection,
    get_opponent_defensive_factor,
)
from src.database.db import get_conn


def _predict_game_player_level(
    team_id: int,
    opponent_id: int,
    game_date: str,
    is_home: bool,
) -> Optional[float]:
    """
    Simulate what predict_matchup() would produce for a historical game
    using the actual players who played that game.

    Returns the defense-adjusted projected points for *team_id*, or None
    if no player data is available for that game.
    """
    with get_conn() as conn:
        # Find players who actually played this game for this team
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

    # Run the same aggregate_projection() that predict_matchup() uses
    proj = aggregate_projection(
        player_ids, opponent_team_id=opponent_id, is_home=is_home
    )

    # Apply opponent defensive factor (same dampening as predict_matchup)
    def_factor = get_opponent_defensive_factor(opponent_id)
    def_factor = 1.0 + (def_factor - 1.0) * 0.5
    adjusted_pts = proj["points"] * def_factor

    return adjusted_pts


def autotune_team(
    team_id: int,
    strength: float = 0.75,
    min_threshold: float = 1.5,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Analyse a team's historical games, compare player-level predictions
    to actual results, and compute per-team scoring corrections.

    Args:
        team_id: The team to tune.
        strength: 0.0-1.0 how aggressively to apply the correction.
        min_threshold: Minimum average error (pts) to store a correction.
        progress_cb: Optional callback for progress messages.

    Returns:
        Dict with team_id, home_pts_correction, away_pts_correction,
        games_analyzed, avg_spread_error_before, avg_total_error_before.
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
            spread_errors.append(pred_spread - actual_spread)
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
            spread_errors.append(pred_spread - actual_spread)
            total_errors.append(pred_total - actual_total)

        processed += 1
        if processed % 10 == 0:
            progress(f"  Processed {processed}/{total_to_process} games...")

    # ----- COMPUTE CORRECTIONS -----
    n = len(home_score_errors) + len(away_score_errors)
    if n == 0:
        progress("  No games with player data found")
        return _empty_result(team_id)

    home_correction = 0.0
    if home_score_errors:
        home_correction = -(sum(home_score_errors) / len(home_score_errors))

    away_correction = 0.0
    if away_score_errors:
        away_correction = -(sum(away_score_errors) / len(away_score_errors))

    # Confidence dampening for small sample sizes
    confidence = min(1.0, n / 10.0)

    # Apply user-configured strength and confidence
    home_correction *= strength * confidence
    away_correction *= strength * confidence

    # Only store if error exceeds minimum threshold
    if abs(home_correction) < min_threshold:
        home_correction = 0.0
    if abs(away_correction) < min_threshold:
        away_correction = 0.0

    # Diagnostic stats (before corrections)
    avg_spread_err = (
        sum(abs(e) for e in spread_errors) / len(spread_errors)
        if spread_errors else 0.0
    )
    avg_total_err = (
        sum(abs(e) for e in total_errors) / len(total_errors)
        if total_errors else 0.0
    )

    result = {
        "team_id": team_id,
        "home_pts_correction": round(home_correction, 2),
        "away_pts_correction": round(away_correction, 2),
        "games_analyzed": n,
        "avg_spread_error_before": round(avg_spread_err, 2),
        "avg_total_error_before": round(avg_total_err, 2),
    }

    # Save to DB
    _save_tuning(result)
    progress(
        f"  Done: {n} games, home_adj={home_correction:+.2f}, "
        f"away_adj={away_correction:+.2f}"
    )
    return result


def autotune_all(
    strength: float = 0.75,
    min_threshold: float = 1.5,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """Run autotune for every team in the database."""
    progress = progress_cb or (lambda _: None)

    with get_conn() as conn:
        teams = conn.execute(
            "SELECT team_id, abbreviation FROM teams ORDER BY abbreviation"
        ).fetchall()

    if not teams:
        progress("No teams found in database")
        return []

    results: List[Dict] = []
    for idx, (tid, abbr) in enumerate(teams, start=1):
        progress(f"[{idx}/{len(teams)}] Tuning {abbr}...")
        res = autotune_team(
            tid,
            strength=strength,
            min_threshold=min_threshold,
            progress_cb=progress,
        )
        results.append(res)

    tuned_count = sum(1 for r in results if r["home_pts_correction"] != 0 or r["away_pts_correction"] != 0)
    progress(f"Autotune complete: {len(teams)} teams analyzed, {tuned_count} received corrections")
    return results


# ---- DB helpers ----


def _empty_result(team_id: int) -> Dict:
    return {
        "team_id": team_id,
        "home_pts_correction": 0.0,
        "away_pts_correction": 0.0,
        "games_analyzed": 0,
        "avg_spread_error_before": 0.0,
        "avg_total_error_before": 0.0,
    }


def _save_tuning(result: Dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO team_tuning
                (team_id, home_pts_correction, away_pts_correction,
                 games_analyzed, avg_spread_error_before, avg_total_error_before,
                 last_tuned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["team_id"],
                result["home_pts_correction"],
                result["away_pts_correction"],
                result["games_analyzed"],
                result["avg_spread_error_before"],
                result["avg_total_error_before"],
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


def get_team_tuning(team_id: int) -> Optional[Dict]:
    """Load per-team tuning corrections.  Returns None if no tuning exists."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM team_tuning WHERE team_id = ?", (team_id,)
        ).fetchone()
    if not row:
        return None
    return {
        "team_id": row[0],
        "home_pts_correction": row[1],
        "away_pts_correction": row[2],
        "games_analyzed": row[3],
        "avg_spread_error_before": row[4],
        "avg_total_error_before": row[5],
        "last_tuned_at": row[6],
    }


def get_all_tunings() -> List[Dict]:
    """Return all saved team tunings with team abbreviation/name."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT tt.*, t.abbreviation, t.name
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
            "abbr": r[7],
            "name": r[8],
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
