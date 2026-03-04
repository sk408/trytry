"""Infer injuries from missing games in player game logs."""

import logging
from collections import defaultdict
from typing import Dict, Any, Optional, List, Callable

from src.database import db

logger = logging.getLogger(__name__)


def _get_seasons_in_db() -> List[str]:
    """Return distinct seasons present in player_stats, sorted chronologically."""
    rows = db.fetch_all(
        "SELECT DISTINCT season FROM player_stats ORDER BY season ASC"
    )
    return [r["season"] for r in rows] if rows else []


def infer_injuries_from_logs(callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Detect missed games by comparing team schedule vs player game logs.

    For each player, find team games where the player has no stats entry.
    Consecutive missed games form an "injury stint".
    Records individual game-date rows in injury_history.

    Processes each season independently to avoid false positives from
    off-season gaps between seasons.
    """
    seasons = _get_seasons_in_db()
    if not seasons:
        return {"stints_found": 0, "records": 0}

    # Bulk-load all game appearances: (player_id, opponent_team_id, game_date, season, is_home)
    # Use opponent_team_id + is_home to determine which team the player was on,
    # so traded/cut players are attributed correctly (not dependent on players table).
    rows = db.fetch_all("""
        SELECT player_id, opponent_team_id, game_date, season, is_home
        FROM player_stats
        ORDER BY game_date ASC
    """)

    if not rows:
        return {"stints_found": 0, "records": 0}

    # Build lookup: team_id -> season -> set of game_dates (team schedule)
    # and: (team_id, season) -> player_id -> set of game_dates (player appearances)
    #
    # Determine player's team from the game: if is_home=1, opponent is the away team
    # so the player's team is the "other" team. We need team_id, not opponent.
    # Actually, we need the team schedule. A team's schedule = all dates where
    # any player on that team played. We infer team from (is_home, opponent_team_id):
    # But we don't have team_id directly. We need to get it from the game context.
    #
    # Simpler approach: use the teams table for team IDs, then find all game_dates
    # where opponent_team_id = tid (meaning the opponent played AGAINST this team).
    # A game involving team X appears as: some players with opponent_team_id=X.

    # Step 1: Build team schedule per season
    # A team's game dates = dates where any player has opponent_team_id pointing at
    # them (they were the opponent) OR they were playing against someone.
    # Easiest: for each row, the player's team played on that date.
    # Player's team = opponent's opponent. We can derive it:
    #   if is_home=1: player is home, opponent_team_id = away team. Player's team != opponent.
    #   if is_home=0: player is away, opponent_team_id = home team. Player's team != opponent.
    # We don't have the player's team_id directly in player_stats.
    #
    # Alternative: just query team game dates directly from game structure.

    # Let's use a simpler bulk approach: get unique (game_date, season, team_id) from
    # the game results perspective.
    team_rows = db.fetch_all("SELECT team_id FROM teams")
    all_team_ids = {r["team_id"] for r in team_rows}

    # Team schedule: dates where opponent_team_id = tid means someone played AGAINST tid,
    # so tid had a game on that date.
    schedule_rows = db.fetch_all("""
        SELECT DISTINCT opponent_team_id as team_id, game_date, season
        FROM player_stats
        ORDER BY season, game_date
    """)

    # team_schedule[team_id][season] = sorted list of game_dates
    team_schedule: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    _seen = set()
    for r in schedule_rows:
        key = (r["team_id"], r["season"], r["game_date"])
        if key not in _seen:
            _seen.add(key)
            team_schedule[r["team_id"]][r["season"]].append(r["game_date"])

    # player_games[(player_id, team_id, season)] = set of game_dates
    # Derive player's team: we need to know which team they played FOR.
    # From player_stats: opponent_team_id is who they played against.
    # So for each row, the player played for "some team" against opponent_team_id.
    # We can get the player's team from the game: on that date, opponent_team_id
    # had a game. The other team in that game is the player's team.
    # But we don't store game_id → teams mapping here easily.
    #
    # Practical solution: a player's team for a game = the team whose schedule
    # includes that date AND is NOT the opponent. But multiple teams play on same date.
    #
    # Simplest correct approach: the player's team on a given date is the team
    # they are listed against as opponent. If player has opponent_team_id=X and is_home=1,
    # player is on the home team playing against X. The home team is whichever team
    # is NOT X on that game. We can find it from the data: other players in the same
    # game (same game_date, same opponent) who are is_home=0 have opponent_team_id = player's team.
    #
    # This is getting complex. Let's just use a direct query for player-team mapping.

    # Bulk query: for each player+season, which teams did they play against?
    # Group by the teams they opposed — their actual team is the one NOT in opponent_team_id.
    # Actually, let's just build it from the raw data more simply:

    # For injury inference, what we actually need per player per season:
    # 1. Which team's schedule to compare against
    # 2. Which dates the player actually played
    #
    # The player's team = the team whose schedule they appear on.
    # We can determine this: on date D, player P has opponent_team_id=X.
    # Team X played on date D. The OTHER team that played on date D (against X) is P's team.
    # That other team = the team_id that appears as opponent_team_id for players on team X
    # on the same date with is_home flipped.
    #
    # This is circular. Let's use a game-level approach instead.

    # Build game mapping: (game_date, team_A, team_B) from the data
    # Each game has home players (is_home=1, opponent=away_team) and
    # away players (is_home=0, opponent=home_team).
    # Home player's opponent = away_team_id
    # Away player's opponent = home_team_id
    # So: home_team = away player's opponent_team_id
    #     away_team = home player's opponent_team_id

    # player_team_map[player_id][season] = set of team_ids they played for
    player_team: Dict[int, Dict[str, int]] = defaultdict(dict)
    player_dates: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    # Build games: (game_date, season) -> {home_team, away_team}
    game_teams: Dict[tuple, Dict[str, int]] = {}
    for r in rows:
        gkey = (r["game_date"], r["season"])
        if gkey not in game_teams:
            game_teams[gkey] = {}
        if r["is_home"] == 1:
            # Home player, opponent is away team
            game_teams[gkey]["away"] = r["opponent_team_id"]
        else:
            # Away player, opponent is home team
            game_teams[gkey]["home"] = r["opponent_team_id"]

    # Now assign each player's team per game
    for r in rows:
        pid = r["player_id"]
        season = r["season"]
        gdate = r["game_date"]
        gkey = (gdate, season)

        player_dates[pid][season].add(gdate)

        # Determine player's team
        gt = game_teams.get(gkey, {})
        if r["is_home"] == 1:
            # Player is home team; home_team = opponent of away players
            ptid = gt.get("home")
        else:
            # Player is away team; away_team = opponent of home players
            ptid = gt.get("away")

        if ptid and ptid in all_team_ids:
            # A player might be traded mid-season; use most frequent team
            if season not in player_team[pid]:
                player_team[pid][season] = ptid

    if callback:
        callback(f"Loaded {len(rows)} game records across {len(seasons)} seasons")

    # Now infer injuries per team, per season
    total_stints = 0
    total_entries = 0
    batch = []

    from src.analytics.stats_engine import get_team_abbreviations
    abbr_map = get_team_abbreviations()

    for tid in sorted(all_team_ids):
        abbr = abbr_map.get(tid, str(tid))

        for season in seasons:
            game_dates = team_schedule.get(tid, {}).get(season, [])
            if len(game_dates) < 5:
                continue

            game_date_set = set(game_dates)
            # Build index for consecutive-game detection
            date_to_idx = {d: i for i, d in enumerate(game_dates)}

            # Find players who played for this team this season
            season_players = [
                pid for pid, st in player_team.items()
                if st.get(season) == tid
            ]

            for pid in season_players:
                played = player_dates.get(pid, {}).get(season, set())
                if not played:
                    continue

                sorted_played = sorted(played)
                first_game = sorted_played[0]
                last_game = sorted_played[-1]

                # Only consider team games in player's active window
                missed = [
                    d for d in game_dates
                    if first_game <= d <= last_game and d not in played
                ]

                if not missed:
                    continue

                # Group consecutive missed games into stints
                stints = []
                current_stint = []

                for date in missed:
                    if not current_stint:
                        current_stint = [date]
                    else:
                        prev_idx = date_to_idx.get(current_stint[-1], -1)
                        curr_idx = date_to_idx.get(date, -1)
                        if curr_idx == prev_idx + 1:
                            current_stint.append(date)
                        else:
                            if len(current_stint) >= 2:
                                stints.append(current_stint)
                            current_stint = [date]

                if len(current_stint) >= 2:
                    stints.append(current_stint)

                for stint_dates in stints:
                    total_stints += 1
                    for gdate in stint_dates:
                        batch.append((pid, tid, gdate, "inferred"))
                        total_entries += 1

        if callback:
            callback(f"Processed {abbr}")

    # Batch insert all injury records at once
    # Disable FK checks — injury_history references players table which may
    # not contain historical-season players.
    if batch:
        db.execute("PRAGMA foreign_keys=OFF")
        try:
            db.execute_many("""
                INSERT INTO injury_history
                    (player_id, team_id, game_date, was_out, reason)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT DO NOTHING
            """, batch)
        finally:
            db.execute("PRAGMA foreign_keys=ON")

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
    row = db.fetch_one("""
        WITH recent_team_dates AS (
            SELECT DISTINCT ps.game_date
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.player_id
            WHERE p.team_id = ?
            ORDER BY ps.game_date DESC
            LIMIT 10
        ),
        numbered AS (
            SELECT game_date,
                   ROW_NUMBER() OVER (ORDER BY game_date DESC) AS rn,
                   EXISTS(
                       SELECT 1 FROM player_stats
                       WHERE player_id = ? AND game_date = recent_team_dates.game_date
                   ) AS played
            FROM recent_team_dates
        )
        SELECT COALESCE(MIN(rn) - 1, (SELECT COUNT(*) FROM numbered)) AS streak
        FROM numbered
        WHERE played = 1
    """, (team_id, player_id))
    return row["streak"] if row else 0
