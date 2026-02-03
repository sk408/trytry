"""Sync service for college basketball data.

Uses ESPN API for on-demand, day-ahead data loading.
Only fetches data for games within the configured time window.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Callable, Iterable, List, Optional, Set

import pandas as pd

from src.data.injury_scraper import get_all_injuries
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
        # Insert/update teams first
        for _, row in teams_df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO teams (team_id, name, abbreviation, conference)
                VALUES (?, ?, ?, ?)
                """,
                (int(row["id"]), row["full_name"], row["abbreviation"], row.get("conference", "")),
            )
        
        # Also ensure any team referenced by players exists (in case roster has different team)
        player_team_ids = set(players_df["team_id"].unique())
        existing_team_ids = set(row[0] for row in conn.execute("SELECT team_id FROM teams").fetchall())
        missing_team_ids = player_team_ids - existing_team_ids
        
        for tid in missing_team_ids:
            # Insert placeholder team - will be updated later if we get full info
            conn.execute(
                """
                INSERT OR IGNORE INTO teams (team_id, name, abbreviation, conference)
                VALUES (?, ?, ?, ?)
                """,
                (int(tid), f"Team {tid}", f"T{tid}", ""),
            )
        
        conn.commit()  # Commit teams before inserting players

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


def _get_synced_game_ids() -> Set[str]:
    """Get set of game IDs that have already been synced."""
    with get_conn() as conn:
        rows = conn.execute("SELECT game_id FROM synced_games").fetchall()
    return {row[0] for row in rows}


def _mark_game_synced(conn, game_id: str, game_date, home_team_id: int, away_team_id: int, stats_count: int) -> None:
    """Mark a game as synced in the cache."""
    from datetime import datetime
    conn.execute(
        """
        INSERT OR REPLACE INTO synced_games (game_id, game_date, home_team_id, away_team_id, synced_at, stats_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (game_id, game_date, home_team_id, away_team_id, datetime.utcnow().isoformat(), stats_count),
    )


def sync_player_stats_from_games(
    game_ids: List[str],
    league: str = DEFAULT_LEAGUE,
    progress_cb: Optional[Callable[[str], None]] = None,
    sleep_between: float = 0.5,
    skip_cached: bool = True,
    force_recent_days: int = 2,
) -> int:
    """
    Sync player stats by fetching box scores from completed games.
    
    This is the primary method for getting player stats in college basketball,
    since ESPN doesn't have a direct player game log endpoint.
    
    Args:
        game_ids: List of ESPN game IDs to sync
        league: League to sync
        progress_cb: Progress callback
        sleep_between: Delay between API requests
        skip_cached: If True, skip games already in synced_games table
        force_recent_days: Always re-fetch games from last N days (in case stats updated)
    
    Returns:
        Number of player stat records added
    """
    progress = progress_cb or (lambda _: None)
    total_added = 0
    
    # Get already-synced games for caching
    if skip_cached:
        already_synced = _get_synced_game_ids()
        original_count = len(game_ids)
        game_ids = [gid for gid in game_ids if gid not in already_synced]
        skipped = original_count - len(game_ids)
        if skipped > 0:
            progress(f"Skipping {skipped} already-synced games (cached)")
    
    if not game_ids:
        progress("All games already synced!")
        return 0
    
    progress(f"Syncing {len(game_ids)} games...")
    
    for idx, game_id in enumerate(game_ids, start=1):
        if idx % 5 == 0 or idx == 1:
            progress(f"Processing game {idx}/{len(game_ids)}...")
        
        try:
            stats_df = fetch_player_stats_from_game(game_id, league=league)
            if stats_df.empty:
                continue
            
            with get_conn() as conn:
                # First, ensure all teams from this game exist
                team_ids_in_game = set()
                if "team_id" in stats_df.columns:
                    team_ids_in_game.update(stats_df["team_id"].dropna().astype(int).unique())
                if "opponent_team_id" in stats_df.columns:
                    team_ids_in_game.update(stats_df["opponent_team_id"].dropna().astype(int).unique())
                
                for tid in team_ids_in_game:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO teams (team_id, name, abbreviation, conference)
                        VALUES (?, ?, ?, ?)
                        """,
                        (int(tid), f"Team {tid}", f"T{tid}", ""),
                    )
                
                # Then, ensure all players exist
                for _, row in stats_df.iterrows():
                    player_id = int(row["player_id"])
                    team_id = int(row["team_id"])
                    player_name = row.get("player_name", f"Player {player_id}")
                    
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO players (player_id, name, team_id, position, is_injured, injury_note)
                        VALUES (?, ?, ?, ?, 0, NULL)
                        """,
                        (player_id, player_name, team_id, ""),
                    )
                
                # Now insert the stats
                game_stats_added = 0
                game_date = None
                home_team_id = None
                away_team_id = None
                
                for _, row in stats_df.iterrows():
                    try:
                        # Parse shooting stats (format: "5-10" -> made=5, attempted=10)
                        def parse_shooting(val):
                            if not val or val == "0-0":
                                return 0, 0
                            try:
                                parts = str(val).split("-")
                                return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
                            except (ValueError, IndexError):
                                return 0, 0
                        
                        fg_made, fg_attempted = parse_shooting(row.get("fg", "0-0"))
                        fg3_made, fg3_attempted = parse_shooting(row.get("fg3", "0-0"))
                        ft_made, ft_attempted = parse_shooting(row.get("ft", "0-0"))
                        
                        conn.execute(
                            """
                            INSERT INTO player_stats
                                (player_id, opponent_team_id, is_home, game_date, game_id,
                                 points, rebounds, assists, minutes,
                                 steals, blocks, turnovers,
                                 fg_made, fg_attempted, fg3_made, fg3_attempted, ft_made, ft_attempted)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(player_id, opponent_team_id, game_date) DO UPDATE SET
                                game_id = excluded.game_id,
                                steals = excluded.steals,
                                blocks = excluded.blocks,
                                turnovers = excluded.turnovers,
                                fg_made = excluded.fg_made,
                                fg_attempted = excluded.fg_attempted,
                                fg3_made = excluded.fg3_made,
                                fg3_attempted = excluded.fg3_attempted,
                                ft_made = excluded.ft_made,
                                ft_attempted = excluded.ft_attempted
                            """,
                            (
                                int(row["player_id"]),
                                int(row["opponent_team_id"]),
                                int(row["is_home"]),
                                row["game_date"],
                                row.get("game_id", game_id),
                                float(row["points"]),
                                float(row["rebounds"]),
                                float(row["assists"]),
                                float(row["minutes"]),
                                float(row.get("steals", 0) or 0),
                                float(row.get("blocks", 0) or 0),
                                float(row.get("turnovers", 0) or 0),
                                fg_made, fg_attempted,
                                fg3_made, fg3_attempted,
                                ft_made, ft_attempted,
                            ),
                        )
                        game_stats_added += 1
                        total_added += 1
                        
                        # Track game info for cache
                        if game_date is None:
                            game_date = row["game_date"]
                        if row["is_home"]:
                            home_team_id = int(row["team_id"])
                            away_team_id = int(row["opponent_team_id"])
                        else:
                            away_team_id = int(row["team_id"])
                            home_team_id = int(row["opponent_team_id"])
                            
                    except Exception as e:
                        progress(f"Error inserting stat for player {row.get('player_id')}: {e}")
                
                # Mark this game as synced
                if game_date is not None:
                    _mark_game_synced(conn, game_id, game_date, home_team_id or 0, away_team_id or 0, game_stats_added)
                
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
    skip_cached: bool = False,
) -> int:
    """
    Sync player stats from recently completed games.
    
    For recent games (last 7 days), we don't use cache by default since
    stats might be updated.
    
    Args:
        days_back: Number of days to look back for completed games
        league: League to sync
        progress_cb: Progress callback
        skip_cached: If True, skip already-synced games
    
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
        
        if day_offset % 7 == 0:  # Progress every week
            progress(f"Checking games from {target_date}... ({day_offset}/{days_back} days)")
        
        try:
            games = fetch_scoreboard(league=league, dates=date_str)
            # Only include completed games
            completed = [g for g in games if g.get("status") == "post"]
            game_ids.extend([g["game_id"] for g in completed])
        except Exception as e:
            progress(f"Error fetching games for {target_date}: {e}")
            continue
    
    progress(f"Found {len(game_ids)} completed games to process")
    
    if not game_ids:
        return 0
    
    return sync_player_stats_from_games(
        game_ids, 
        league=league, 
        progress_cb=progress,
        skip_cached=skip_cached,
    )


def sync_season_stats(
    league: str = DEFAULT_LEAGUE,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Sync player stats for the entire current season with CACHING.
    
    - Old games (>3 days): Only fetched once, then cached
    - Recent games (last 3 days): Always re-fetched in case stats updated
    
    This makes subsequent syncs much faster after the first full sync.
    
    Returns:
        Number of stats added
    """
    progress = progress_cb or (lambda _: None)
    
    # Calculate days since season start (November 1)
    today = date.today()
    year = today.year if today.month >= 11 else today.year - 1
    season_start = date(year, 11, 1)  # Nov 1
    days_since_start = (today - season_start).days
    
    # Cap at reasonable number (full season is ~150 days max)
    days_back = min(days_since_start, 150)
    
    progress(f"Syncing full season: {days_back} days of games since {season_start}")
    progress("(Cached games will be skipped - only new games fetched)")
    
    # Collect all game IDs first, separating recent from older games
    recent_game_ids = []  # Last 3 days - always re-fetch
    older_game_ids = []   # Older than 3 days - use cache
    
    for day_offset in range(days_back):
        target_date = today - timedelta(days=day_offset)
        date_str = target_date.strftime("%Y%m%d")
        
        if day_offset % 14 == 0:  # Progress every 2 weeks
            progress(f"Scanning games from {target_date}... ({day_offset}/{days_back} days)")
        
        try:
            games = fetch_scoreboard(league=league, dates=date_str)
            completed = [g for g in games if g.get("status") == "post"]
            
            for g in completed:
                if day_offset <= 3:
                    recent_game_ids.append(g["game_id"])
                else:
                    older_game_ids.append(g["game_id"])
        except Exception as e:
            if day_offset % 7 == 0:
                progress(f"Error fetching games for {target_date}: {e}")
            continue
    
    total_added = 0
    
    # First, sync older games WITH caching (skip already synced)
    if older_game_ids:
        progress(f"Processing {len(older_game_ids)} older games (using cache)...")
        total_added += sync_player_stats_from_games(
            older_game_ids,
            league=league,
            progress_cb=progress,
            skip_cached=True,  # Use cache for older games
        )
    
    # Then, sync recent games WITHOUT caching (always re-fetch)
    if recent_game_ids:
        progress(f"Processing {len(recent_game_ids)} recent games (last 3 days, no cache)...")
        total_added += sync_player_stats_from_games(
            recent_game_ids,
            league=league,
            progress_cb=progress,
            skip_cached=False,  # Always fetch recent games
        )
    
    progress(f"Season sync complete: {total_added} total stats added")
    return total_added


def sync_injuries(progress_cb: Optional[Callable[[str], None]] = None) -> int:
    """
    Sync injury data from web sources.
    Returns number of injuries updated.
    """
    progress = progress_cb or (lambda _: None)
    
    # Fetch from multiple sources + manual entries
    data = get_all_injuries(progress_cb=progress)
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


def _sync_teams_from_schedule_data(
    schedule_df: pd.DataFrame,
    progress_cb: Optional[Callable[[str], None]] = None
) -> int:
    """
    Sync team names from schedule data to database.
    This ensures teams from upcoming games have proper names, not placeholders.
    
    Returns:
        Number of teams updated
    """
    progress = progress_cb or (lambda _msg: None)
    
    if schedule_df.empty:
        return 0
    
    updated = 0
    with get_conn() as conn:
        for _, r in schedule_df.iterrows():
            home_id = r.get("home_team_id")
            away_id = r.get("away_team_id")
            home_name = r.get("home_name") or ""
            away_name = r.get("away_name") or ""
            home_abbr = r.get("home_abbr") or ""
            away_abbr = r.get("away_abbr") or ""
            
            # Update home team if we have real name (not placeholder)
            if home_id and home_name and not home_name.startswith("Team "):
                cursor = conn.execute(
                    """
                    INSERT INTO teams (team_id, name, abbreviation, conference)
                    VALUES (?, ?, ?, '')
                    ON CONFLICT(team_id) DO UPDATE SET
                        name = excluded.name,
                        abbreviation = CASE 
                            WHEN teams.abbreviation LIKE 'T%' AND length(teams.abbreviation) > 3 
                            THEN excluded.abbreviation 
                            ELSE teams.abbreviation 
                        END
                    WHERE teams.name LIKE 'Team %' OR teams.name = ''
                    """,
                    (int(home_id), home_name, home_abbr or f"T{home_id}"),
                )
                updated += cursor.rowcount
            
            # Update away team if we have real name
            if away_id and away_name and not away_name.startswith("Team "):
                cursor = conn.execute(
                    """
                    INSERT INTO teams (team_id, name, abbreviation, conference)
                    VALUES (?, ?, ?, '')
                    ON CONFLICT(team_id) DO UPDATE SET
                        name = excluded.name,
                        abbreviation = CASE 
                            WHEN teams.abbreviation LIKE 'T%' AND length(teams.abbreviation) > 3 
                            THEN excluded.abbreviation 
                            ELSE teams.abbreviation 
                        END
                    WHERE teams.name LIKE 'Team %' OR teams.name = ''
                    """,
                    (int(away_id), away_name, away_abbr or f"T{away_id}"),
                )
                updated += cursor.rowcount
        
        conn.commit()
    
    if updated > 0:
        progress(f"Updated {updated} team names from schedule")
    
    return updated


def full_sync(
    season: Optional[str] = None,
    league: str = DEFAULT_LEAGUE,
    progress_cb: Optional[Callable[[str], None]] = None,
    days_back: int = 7,
    full_season: bool = False,
) -> None:
    """
    Full sync for college basketball.
    
    Day-ahead loading approach:
    1. Get upcoming schedule (14 days) and sync team names
    2. Sync teams and rosters for today's games
    3. Sync recent game stats (or full season if full_season=True)
    4. Sync current injuries
    
    Args:
        season: Season string (e.g., "2025-26")
        league: League to sync
        progress_cb: Progress callback
        days_back: Days of games to fetch (ignored if full_season=True)
        full_season: If True, fetch entire season's stats (~120 days)
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)
    
    # First, sync team names from upcoming schedule (14 days)
    # This ensures teams from future games have proper names
    progress("Sync: team names from upcoming schedule (14 days)...")
    try:
        schedule_df = fetch_schedule(league=league, include_future_days=14)
        _sync_teams_from_schedule_data(schedule_df, progress_cb=progress)
    except Exception as exc:
        progress(f"Schedule team sync skipped: {exc}")
    
    progress("Sync: reference data (teams + rosters from today's games)")
    players_df = sync_reference_data(progress_cb=progress, season=season, league=league)
    
    if not players_df.empty:
        if full_season:
            progress("Sync: FULL SEASON stats (this may take a while)...")
            sync_season_stats(league=league, progress_cb=progress)
        else:
            progress(f"Sync: recent game stats ({days_back} days)...")
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

    # Try to fetch full team info
    try:
        teams_df = fetch_teams(league=league)
        subset = teams_df[teams_df["id"].isin(missing)]
        
        with get_conn() as conn:
            # Insert teams we found info for
            if not subset.empty:
                for _, row in subset.iterrows():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO teams (team_id, name, abbreviation, conference)
                        VALUES (?, ?, ?, ?)
                        """,
                        (int(row["id"]), row["full_name"], row["abbreviation"], row.get("conference", "")),
                    )
            
            # Insert placeholder for any remaining teams we couldn't find
            still_missing = missing - set(subset["id"].astype(int)) if not subset.empty else missing
            for tid in still_missing:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO teams (team_id, name, abbreviation, conference)
                    VALUES (?, ?, ?, ?)
                    """,
                    (int(tid), f"Team {tid}", f"T{tid}", ""),
                )
            conn.commit()
    except Exception as e:
        # If fetch fails, just insert placeholders
        print(f"[sync] Warning: Could not fetch team info: {e}")
        with get_conn() as conn:
            for tid in missing:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO teams (team_id, name, abbreviation, conference)
                    VALUES (?, ?, ?, ?)
                    """,
                    (int(tid), f"Team {tid}", f"T{tid}", ""),
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
