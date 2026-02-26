"""Infer injuries from missing games in player game logs."""

import logging
from typing import Dict, Any, Optional, List, Callable

from src.database import db

logger = logging.getLogger(__name__)


def infer_injuries_from_logs(callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Detect missed games by comparing team schedule vs player game logs.
    
    For each player, find team games where the player has no stats entry.
    Consecutive missed games form an "injury stint".
    Records individual game-date rows in injury_history.
    """
    # Get all teams (cached singleton)
    from src.analytics.stats_engine import get_team_abbreviations
    abbr_map = get_team_abbreviations()
    teams = [{"team_id": tid, "abbreviation": abbr} for tid, abbr in abbr_map.items()]
    if not teams:
        return {"stints_found": 0, "records": 0}

    total_stints = 0
    total_entries = 0

    for team in teams:
        tid = team["team_id"]
        abbr = team["abbreviation"]

        # Get team's game dates (from player_stats via players join)
        team_game_dates = db.fetch_all("""
            SELECT DISTINCT ps.game_date FROM player_stats ps
            JOIN players p ON ps.player_id = p.player_id
            WHERE p.team_id = ?
            ORDER BY ps.game_date ASC
        """, (tid,))
        game_dates = [r["game_date"] for r in team_game_dates]

        if len(game_dates) < 5:
            continue

        # Get all players on this team
        players = db.fetch_all("""
            SELECT DISTINCT ps.player_id, p.name as player_name
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.player_id
            WHERE p.team_id = ?
        """, (tid,))

        for player in players:
            pid = player["player_id"]

            # Get dates this player played
            played_rows = db.fetch_all("""
                SELECT DISTINCT ps.game_date FROM player_stats ps
                JOIN players p ON ps.player_id = p.player_id
                WHERE ps.player_id = ? AND p.team_id = ?
            """, (pid, tid))
            played_dates = set(r["game_date"] for r in played_rows)

            if not played_dates:
                continue

            # Find first and last game the player appeared in
            sorted_played = sorted(played_dates)
            first_game = sorted_played[0]
            last_game = sorted_played[-1]

            # Only consider dates in player's active window
            active_dates = [d for d in game_dates if first_game <= d <= last_game]

            # Find missed dates
            missed = [d for d in active_dates if d not in played_dates]

            if not missed:
                continue

            # Group consecutive missed games into stints
            stints = []
            current_stint_dates = []

            for date in sorted(missed):
                if not current_stint_dates:
                    current_stint_dates = [date]
                else:
                    # Check if consecutive (within team's schedule)
                    prev_date = current_stint_dates[-1]
                    prev_idx = game_dates.index(prev_date) if prev_date in game_dates else -1
                    curr_idx = game_dates.index(date) if date in game_dates else -1

                    if curr_idx == prev_idx + 1:
                        current_stint_dates.append(date)
                    else:
                        if len(current_stint_dates) >= 2:
                            stints.append(list(current_stint_dates))
                        current_stint_dates = [date]

            # Don't forget last stint
            if len(current_stint_dates) >= 2:
                stints.append(list(current_stint_dates))

            # Save individual game-date rows (only stints of 2+ games)
            for stint_dates in stints:
                total_stints += 1
                for game_date in stint_dates:
                    db.execute("""
                        INSERT INTO injury_history 
                            (player_id, team_id, game_date, was_out, reason)
                        VALUES (?, ?, ?, 1, ?)
                        ON CONFLICT DO NOTHING
                    """, (pid, tid, game_date, "inferred"))
                    total_entries += 1

        if callback:
            callback(f"Processed {abbr}")

    if callback:
        callback(f"Injury history complete: {total_stints} stints, {total_entries} total missed games")

    return {
        "stints_found": total_stints,
        "total_missed_games": total_entries,
        "records": total_entries,
    }


def get_player_injury_history(player_id: int) -> List[Dict]:
    """Get injury history for a player."""
    rows = db.fetch_all("""
        SELECT * FROM injury_history
        WHERE player_id = ?
        ORDER BY game_date DESC
    """, (player_id,))
    return [dict(r) for r in rows]


def get_games_missed_streak(player_id: int, team_id: int) -> int:
    """How many consecutive recent games has the player missed?"""
    # Get team's recent game dates
    recent_dates = db.fetch_all("""
        SELECT DISTINCT ps.game_date FROM player_stats ps
        JOIN players p ON ps.player_id = p.player_id
        WHERE p.team_id = ?
        ORDER BY ps.game_date DESC
        LIMIT 10
    """, (team_id,))

    if not recent_dates:
        return 0

    dates = [r["game_date"] for r in recent_dates]

    # Check from most recent backwards
    missed = 0
    for date in dates:
        played = db.fetch_one("""
            SELECT 1 FROM player_stats
            WHERE player_id = ? AND game_date = ?
        """, (player_id, date))
        if played:
            break
        missed += 1

    return missed
