"""Sync service for college basketball data.

Uses ESPN API for on-demand, day-ahead data loading.
Only fetches data for games within the configured time window.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Callable, Iterable, List, Optional, Set

import pandas as pd

from src.data.injury_scraper import fetch_injuries
from src.data.live_scores import fetch_live_games
from src.data.college_fetcher import (
    fetch_teams,
    fetch_players,
    fetch_schedule,
    fetch_scoreboard,
    fetch_player_stats_from_game,
    get_current_season,
    DEFAULT_LEAGUE,
)
from src.database import migrations
from src.database.db import get_conn
from src.analytics.injury_history import build_injury_history


def sync_reference_data(
    progress_cb: Optional[Callable[[str], None]] = None,
    season: Optional[str] = None,
    league: str = DEFAULT_LEAGUE,
    team_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Sync teams and players to DB.
    
    For college basketball, we use on-demand loading:
    - If team_ids provided, only sync those teams
    - Otherwise, sync teams from today's scheduled games
    
    Returns:
        DataFrame of synced players
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)
    migrations.init_db()
    migrations.ensure_columns()
    
    # Get team IDs from today's games if not provided
    if team_ids is None:
        progress("Fetching today's scheduled games...")
        games = fetch_scoreboard(league=league)
        team_ids = list(set(
            [g["home_team_id"] for g in games] + [g["away_team_id"] for g in games]
        ))
        progress(f"Found {len(team_ids)} teams from {len(games)} games")
    
    if not team_ids:
        progress("No games today - no teams to sync")
        return pd.DataFrame(columns=["id", "full_name", "team_id", "position"])
    
    # Fetch teams
    progress("Fetching team info...")
    teams_df = fetch_teams(league=league, progress_cb=progress)
    # Filter to only teams we need
    teams_df = teams_df[teams_df["id"].isin(team_ids)]
    
    # Fetch players for these teams
    progress(f"Fetching rosters for {len(team_ids)} teams...")
    players_df = fetch_players(team_ids=team_ids, league=league, progress_cb=progress)
    
    if players_df.empty:
        progress("No player data retrieved")
        return pd.DataFrame(columns=["id", "full_name", "team_id", "position"])
    
    players_df = players_df.dropna(subset=["team_id"])
    players_df["team_id"] = players_df["team_id"].astype(int)

    with get_conn() as conn:
        # Insert/update teams
        for _, row in teams_df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO teams (team_id, name, abbreviation, conference)
                VALUES (?, ?, ?, ?)
                """,
                (int(row["id"]), row["full_name"], row["abbreviation"], row.get("conference", "")),
            )

        # Insert/update players
        for _, row in players_df.iterrows():
            conn.execute(
                """
                INSERT INTO players (player_id, name, team_id, position, is_injured, injury_note)
                VALUES (?, ?, ?, ?, 0, NULL)
                ON CONFLICT(player_id) DO UPDATE SET
                    name = excluded.name,
                    team_id = excluded.team_id,
                    position = excluded.position
                """,
                (int(row["id"]), row["full_name"], int(row["team_id"]), row.get("position", "")),
            )
        conn.commit()
    
    progress(f"Synced {len(teams_df)} teams and {len(players_df)} players")
    return players_df


def _team_abbr_lookup(conn) -> dict[str, int]:
    rows = conn.execute("SELECT abbreviation, team_id FROM teams").fetchall()
    return {abbr: tid for abbr, tid in rows}


def sync_player_stats_from_games(
    game_ids: List[str],
    league: str = DEFAULT_LEAGUE,
    progress_cb: Optional[Callable[[str], None]] = None,
    sleep_between: float = 0.5,
) -> int:
    """
    Sync player stats by fetching box scores from completed games.
    
    This is the primary method for getting player stats in college basketball,
    since ESPN doesn't have a direct player game log endpoint.
    
    Returns:
        Number of player stat records added
    """
    progress = progress_cb or (lambda _: None)
    total_added = 0
    
    for idx, game_id in enumerate(game_ids, start=1):
        if idx % 5 == 0:
            progress(f"Processing game {idx}/{len(game_ids)}...")
        
        try:
            stats_df = fetch_player_stats_from_game(game_id, league=league)
            if stats_df.empty:
                continue
            
            with get_conn() as conn:
                for _, row in stats_df.iterrows():
                    try:
                        conn.execute(
                            """
                            INSERT INTO player_stats
                                (player_id, opponent_team_id, is_home, game_date, points, rebounds, assists, minutes)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(player_id, opponent_team_id, game_date) DO NOTHING
                            """,
                            (
                                int(row["player_id"]),
                                int(row["opponent_team_id"]),
                                int(row["is_home"]),
                                row["game_date"],
                                float(row["points"]),
                                float(row["rebounds"]),
                                float(row["assists"]),
                                float(row["minutes"]),
                            ),
                        )
                        total_added += 1
                    except Exception as e:
                        progress(f"Error inserting stat for player {row.get('player_id')}: {e}")
                conn.commit()
            
        except Exception as e:
            progress(f"Failed to fetch game {game_id}: {e}")
        
        if sleep_between > 0:
            time.sleep(sleep_between)
    
    progress(f"Added {total_added} player stat records from {len(game_ids)} games")
    return total_added


def sync_recent_games(
    days_back: int = 7,
    league: str = DEFAULT_LEAGUE,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Sync player stats from recently completed games.
    
    Args:
        days_back: Number of days to look back for completed games
        league: League to sync
        progress_cb: Progress callback
    
    Returns:
        Number of stats added
    """
    progress = progress_cb or (lambda _: None)
    
    # Get completed games from recent days
    game_ids = []
    today = date.today()
    
    for day_offset in range(days_back):
        target_date = today - timedelta(days=day_offset)
        date_str = target_date.strftime("%Y%m%d")
        
        progress(f"Checking games from {target_date}...")
        games = fetch_scoreboard(league=league, dates=date_str)
        
        # Only include completed games
        completed = [g for g in games if g.get("status") == "post"]
        game_ids.extend([g["game_id"] for g in completed])
    
    progress(f"Found {len(game_ids)} completed games to process")
    
    if not game_ids:
        return 0
    
    return sync_player_stats_from_games(game_ids, league=league, progress_cb=progress)


def sync_injuries(progress_cb: Optional[Callable[[str], None]] = None) -> int:
    """
    Sync injury data from web sources.
    Returns number of injuries updated.
    """
    progress = progress_cb or (lambda _: None)
    
    # Fetch from multiple sources
    data = fetch_injuries(progress_cb=progress)
    if not data:
        progress("No injury data retrieved")
        return 0
    
    with get_conn() as conn:
        # First, clear all existing injuries
        progress("Clearing previous injury flags...")
        conn.execute("UPDATE players SET is_injured = 0, injury_note = NULL")
        
        updated = 0
        for entry in data:
            player_name = entry["player"]
            status = entry["status"]
            injury = entry["injury"]
            note = f"{status}: {injury}".strip()
            if entry.get("update"):
                note += f" ({entry['update']})"
            
            # Try exact name match first, then fuzzy match
            cursor = conn.execute(
                """
                UPDATE players
                SET is_injured = 1, injury_note = ?
                WHERE lower(name) = lower(?)
                """,
                (note, player_name),
            )
            if cursor.rowcount > 0:
                updated += cursor.rowcount
            else:
                # Try partial match (last name)
                parts = player_name.split()
                if len(parts) >= 2:
                    last_name = parts[-1]
                    cursor = conn.execute(
                        """
                        UPDATE players
                        SET is_injured = 1, injury_note = ?
                        WHERE name LIKE ?
                        AND is_injured = 0
                        """,
                        (note, f"%{last_name}%"),
                    )
                    updated += cursor.rowcount
        
        conn.commit()
        progress(f"Updated {updated} player injury records")
        return updated


def sync_schedule(
    season: Optional[str] = None,
    team_ids: List[int] | None = None,
    include_future_days: int = 1,  # Day-ahead by default
    league: str = DEFAULT_LEAGUE,
) -> pd.DataFrame:
    """Fetch schedule for upcoming games."""
    season = season or get_current_season()
    return fetch_schedule(
        league=league,
        team_ids=team_ids,
        include_future_days=include_future_days,
    )


def full_sync(
    season: Optional[str] = None,
    league: str = DEFAULT_LEAGUE,
    progress_cb: Optional[Callable[[str], None]] = None,
    days_back: int = 7,
) -> None:
    """
    Full sync for college basketball.
    
    Day-ahead loading approach:
    1. Get today's scheduled games
    2. Sync teams and rosters for those games
    3. Sync recent game stats for those teams
    4. Sync current injuries
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)
    
    progress("Sync: reference data (teams + rosters from today's games)")
    players_df = sync_reference_data(progress_cb=progress, season=season, league=league)
    
    if not players_df.empty:
        progress(f"Sync: recent game stats for {len(players_df)} players...")
        sync_recent_games(days_back=days_back, league=league, progress_cb=progress)
    
    progress("Sync: current injuries")
    try:
        sync_injuries(progress_cb=progress)
        progress("Current injuries synced")
    except Exception as exc:
        progress(f"Current injuries sync skipped: {exc}")
    
    progress("Building historical injury data...")
    try:
        count = sync_injury_history(progress_cb=progress)
        progress(f"Historical injury data built: {count} records")
    except Exception as exc:
        progress(f"Historical injury build skipped: {exc}")
    
    progress("Sync complete")


def sync_injury_history(progress_cb: Optional[Callable[[str], None]] = None) -> int:
    """
    Build historical injury data by inferring from game logs.
    This analyzes which rotation players missed games.
    """
    migrations.init_db()  # Ensure injury_history table exists
    return build_injury_history(progress_cb=progress_cb)


def sync_live_scores(
    game_date: str | None = None,
    league: str = DEFAULT_LEAGUE,
) -> None:
    """Sync live game scores to database."""
    migrations.init_db()
    games = fetch_live_games(game_date, league=league)
    if not games:
        return

    # Ensure the referenced teams exist to avoid FK errors
    required_ids = {g["home_team_id"] for g in games} | {g["away_team_id"] for g in games}
    _ensure_teams_exist(required_ids, league=league)

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO live_games
                (game_id, home_team_id, away_team_id, start_time_utc, status, period, clock, home_score, away_score, last_updated)
            VALUES
                (:game_id, :home_team_id, :away_team_id, :start_time_utc, :status, :period, :clock, :home_score, :away_score, :last_updated)
            ON CONFLICT(game_id) DO UPDATE SET
                status=excluded.status,
                period=excluded.period,
                clock=excluded.clock,
                home_score=excluded.home_score,
                away_score=excluded.away_score,
                last_updated=excluded.last_updated
            """,
            games,
        )
        conn.commit()


def _ensure_teams_exist(team_ids: Set[int], league: str = DEFAULT_LEAGUE) -> None:
    """Insert missing teams (id + minimal fields) to satisfy FK constraints."""
    if not team_ids:
        return
    with get_conn() as conn:
        existing = {row[0] for row in conn.execute("SELECT team_id FROM teams").fetchall()}
    missing = team_ids - existing
    if not missing:
        return

    teams_df = fetch_teams(league=league)
    subset = teams_df[teams_df["id"].isin(missing)]
    if subset.empty:
        return
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO teams (team_id, name, abbreviation, conference)
            VALUES (?, ?, ?, ?)
            """,
            subset[["id", "full_name", "abbreviation", "conference"]].itertuples(index=False),
        )
        conn.commit()


# Legacy function for compatibility
def sync_player_logs(
    player_ids: Iterable[int],
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    sleep_between: float = 0.6,
    max_retries: int = 2,
) -> None:
    """
    Legacy function - in college basketball, we sync stats from game box scores
    rather than individual player game logs.
    
    This function now triggers sync_recent_games instead.
    """
    progress = progress_cb or (lambda _: None)
    progress("Note: College basketball uses game box scores for stats")
    sync_recent_games(days_back=7, progress_cb=progress)
