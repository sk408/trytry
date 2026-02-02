from __future__ import annotations

import time
from typing import Callable, Iterable, List, Optional

import pandas as pd

from src.data.injury_scraper import fetch_injuries
from src.data.live_scores import fetch_live_games
from src.data.nba_fetcher import fetch_player_game_logs, fetch_players, fetch_schedule, fetch_teams, get_current_season
from src.database import migrations
from src.database.db import get_conn
from src.analytics.injury_history import build_injury_history


def sync_reference_data(
    progress_cb: Optional[Callable[[str], None]] = None, season: Optional[str] = None
) -> pd.DataFrame:
    season = season or get_current_season()
    """Sync teams and players to DB, returns players_df for reuse."""
    progress = progress_cb or (lambda _msg: None)
    migrations.init_db()
    migrations.ensure_columns()
    progress("Fetching teams...")
    teams_df = fetch_teams()
    progress("Fetching players...")
    players_df = fetch_players(season=season, progress_cb=progress_cb)
    players_df = players_df.dropna(subset=["team_id"])
    players_df["team_id"] = players_df["team_id"].astype(int)

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO teams (team_id, name, abbreviation, conference)
            VALUES (?, ?, ?, ?)
            """,
            teams_df[["id", "full_name", "abbreviation", "conference"]].itertuples(index=False),
        )

        conn.executemany(
            """
            INSERT INTO players (player_id, name, team_id, position, is_injured, injury_note)
            VALUES (?, ?, ?, ?, 0, NULL)
            ON CONFLICT(player_id) DO UPDATE SET
                name = excluded.name,
                team_id = excluded.team_id,
                position = excluded.position
            """,
            players_df[["id", "full_name", "team_id", "position"]].itertuples(index=False),
        )
        conn.commit()
    return players_df


def _team_abbr_lookup(conn) -> dict[str, int]:
    rows = conn.execute("SELECT abbreviation, team_id FROM teams").fetchall()
    return {abbr: tid for abbr, tid in rows}


def sync_player_logs(
    player_ids: Iterable[int],
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    sleep_between: float = 0.6,
    max_retries: int = 2,
) -> None:
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)
    player_ids = list(player_ids)
    total = len(player_ids)
    with get_conn() as conn:
        abbr_to_id = _team_abbr_lookup(conn)
        for idx, player_id in enumerate(player_ids, start=1):
            if idx == 1 or idx % 25 == 0 or idx == total:
                progress(f"Syncing game logs {idx}/{total}...")

            # Fetch with retries
            logs = None
            for attempt in range(max_retries + 1):
                try:
                    logs = fetch_player_game_logs(player_id, season)
                    break
                except Exception as exc:
                    if attempt < max_retries:
                        time.sleep(1.0 + attempt)  # backoff before retry
                    else:
                        progress(f"Player {player_id} logs failed after {max_retries+1} attempts: {exc}")

            if logs is None or logs.empty:
                time.sleep(sleep_between)
                continue

            # Map opponent abbreviation to team_id
            logs["opponent_team_id"] = logs["opponent_abbr"].map(abbr_to_id)
            logs = logs.dropna(subset=["opponent_team_id"])
            payload = [
                (
                    player_id,
                    int(row.opponent_team_id),
                    int(row.is_home),
                    row.game_date,
                    float(row.points),
                    float(row.rebounds),
                    float(row.assists),
                    float(row.minutes),
                )
                for row in logs.itertuples(index=False)
            ]
            conn.executemany(
                """
                INSERT INTO player_stats
                    (player_id, opponent_team_id, is_home, game_date, points, rebounds, assists, minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id, opponent_team_id, game_date) DO NOTHING
                """,
                payload,
            )
            # Rate limit between successful fetches
            time.sleep(sleep_between)
        conn.commit()


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
    include_future_days: int = 14,
) -> pd.DataFrame:
    season = season or get_current_season()
    return fetch_schedule(season=season, team_ids=team_ids, include_future_days=include_future_days)


def full_sync(
    active_only: bool = True, season: Optional[str] = None, progress_cb: Optional[Callable[[str], None]] = None
) -> None:
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)
    progress("Sync: reference data (teams + rosters)")
    players_df = sync_reference_data(progress_cb=progress_cb, season=season)
    progress(f"Sync: game logs for {len(players_df)} players (this may take a while)...")
    sync_player_logs(players_df["id"].tolist(), season=season, progress_cb=progress_cb)
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


def sync_live_scores(game_date: str | None = None) -> None:
    migrations.init_db()
    games = fetch_live_games(game_date)
    if not games:
        return

    # Ensure the referenced teams exist to avoid FK errors
    required_ids = {g["home_team_id"] for g in games} | {g["away_team_id"] for g in games}
    _ensure_teams_exist(required_ids)

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


def _ensure_teams_exist(team_ids: set[int]) -> None:
    """Insert missing teams (id + minimal fields) to satisfy FK constraints."""
    if not team_ids:
        return
    with get_conn() as conn:
        existing = {row[0] for row in conn.execute("SELECT team_id FROM teams").fetchall()}
    missing = team_ids - existing
    if not missing:
        return

    teams_df = fetch_teams()
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
