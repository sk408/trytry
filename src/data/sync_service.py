from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Callable, Iterable, List, Optional, Set

import pandas as pd

from src.data.injury_scraper import get_all_injuries
from src.data.live_scores import fetch_live_games
from src.data.nba_fetcher import (
    fetch_player_game_logs, fetch_players, fetch_schedule, fetch_teams,
    get_current_season,
    fetch_team_estimated_metrics, fetch_league_dash_team_stats,
    fetch_team_clutch_stats, fetch_team_hustle_stats,
    fetch_player_on_off, fetch_player_estimated_metrics,
    _safe_float, _safe_int,
)
from src.database import migrations
from src.database.db import get_conn
from src.analytics.injury_history import build_injury_history


# ============ CACHING HELPERS ============

def _get_cached_players(conn, max_age_hours: int = 24) -> Set[int]:
    """Get player IDs that were synced within the last max_age_hours."""
    cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
    rows = conn.execute(
        "SELECT player_id FROM player_sync_cache WHERE last_synced_at > ?",
        (cutoff,)
    ).fetchall()
    return {row[0] for row in rows}


def _get_players_with_recent_games(conn, days: int = 3) -> Set[int]:
    """Get player IDs who have games in the last N days (need fresh data)."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT DISTINCT player_id FROM player_stats 
        WHERE game_date >= ?
        """,
        (cutoff,)
    ).fetchall()
    return {row[0] for row in rows}


def _update_player_cache(conn, player_id: int, games_count: int, latest_date: Optional[date]) -> None:
    """Update the sync cache for a player."""
    conn.execute(
        """
        INSERT OR REPLACE INTO player_sync_cache 
            (player_id, last_synced_at, games_synced, latest_game_date)
        VALUES (?, ?, ?, ?)
        """,
        (player_id, datetime.now().isoformat(), games_count, latest_date)
    )


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

        # Ensure columns exist for roster context
        for col, default in [("height", ""), ("weight", ""), ("age", None), ("experience", None)]:
            if col not in players_df.columns:
                players_df[col] = default

        conn.executemany(
            """
            INSERT INTO players (player_id, name, team_id, position, is_injured, injury_note,
                                 height, weight, age, experience)
            VALUES (?, ?, ?, ?, 0, NULL, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                name = excluded.name,
                team_id = excluded.team_id,
                position = excluded.position,
                height = excluded.height,
                weight = excluded.weight,
                age = excluded.age,
                experience = excluded.experience
            """,
            players_df[["id", "full_name", "team_id", "position",
                        "height", "weight", "age", "experience"]].itertuples(index=False),
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
    skip_cached: bool = True,
    cache_max_age_hours: int = 24,
    force_recent_days: int = 3,
) -> None:
    """
    Sync all player game logs with comprehensive stats.
    
    Caching behavior:
    - Players synced within cache_max_age_hours are skipped (unless they have recent games)
    - Players with games in the last force_recent_days are always re-fetched
    - Set skip_cached=False to force full re-sync of all players
    
    Now stores extended stats: steals, blocks, turnovers, shooting stats,
    rebound breakdown, and plus/minus.
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)
    player_ids = list(player_ids)
    total = len(player_ids)
    
    with get_conn() as conn:
        abbr_to_id = _team_abbr_lookup(conn)
        
        # Determine which players to skip
        players_to_skip: Set[int] = set()
        if skip_cached:
            cached_players = _get_cached_players(conn, max_age_hours=cache_max_age_hours)
            recent_game_players = _get_players_with_recent_games(conn, days=force_recent_days)
            # Skip cached players UNLESS they have recent games
            players_to_skip = cached_players - recent_game_players
            
            if players_to_skip:
                progress(f"Skipping {len(players_to_skip)} cached players (no recent games)")
        
        players_synced = 0
        players_skipped = 0
        
        for idx, player_id in enumerate(player_ids, start=1):
            # Skip if cached
            if player_id in players_to_skip:
                players_skipped += 1
                continue
            
            players_synced += 1
            if players_synced == 1 or players_synced % 25 == 0:
                progress(f"Syncing game logs {players_synced}/{total - len(players_to_skip)}...")

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
            
            # Build payload with all stats (including WL and PF)
            payload = []
            for row in logs.itertuples(index=False):
                payload.append((
                    player_id,
                    int(row.opponent_team_id),
                    int(row.is_home),
                    row.game_date,
                    getattr(row, 'game_id', ''),
                    # Basic stats
                    float(row.points),
                    float(row.rebounds),
                    float(row.assists),
                    float(row.minutes),
                    # Defensive stats
                    float(getattr(row, 'steals', 0) or 0),
                    float(getattr(row, 'blocks', 0) or 0),
                    float(getattr(row, 'turnovers', 0) or 0),
                    # Shooting stats
                    int(getattr(row, 'fg_made', 0) or 0),
                    int(getattr(row, 'fg_attempted', 0) or 0),
                    int(getattr(row, 'fg3_made', 0) or 0),
                    int(getattr(row, 'fg3_attempted', 0) or 0),
                    int(getattr(row, 'ft_made', 0) or 0),
                    int(getattr(row, 'ft_attempted', 0) or 0),
                    # Rebound breakdown
                    float(getattr(row, 'oreb', 0) or 0),
                    float(getattr(row, 'dreb', 0) or 0),
                    # Impact
                    float(getattr(row, 'plus_minus', 0) or 0),
                    # Win/Loss and personal fouls (NEW)
                    str(getattr(row, 'win_loss', '') or '') or None,
                    float(getattr(row, 'personal_fouls', 0) or 0),
                ))
            
            conn.executemany(
                """
                INSERT INTO player_stats
                    (player_id, opponent_team_id, is_home, game_date, game_id,
                     points, rebounds, assists, minutes,
                     steals, blocks, turnovers,
                     fg_made, fg_attempted, fg3_made, fg3_attempted, ft_made, ft_attempted,
                     oreb, dreb, plus_minus,
                     win_loss, personal_fouls)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    ft_attempted = excluded.ft_attempted,
                    oreb = excluded.oreb,
                    dreb = excluded.dreb,
                    plus_minus = excluded.plus_minus,
                    win_loss = excluded.win_loss,
                    personal_fouls = excluded.personal_fouls
                """,
                payload,
            )
            
            # Update the sync cache for this player
            latest_game = max((row[3] for row in payload), default=None) if payload else None
            _update_player_cache(conn, player_id, len(payload), latest_game)
            
            # Rate limit between successful fetches
            time.sleep(sleep_between)
        
        conn.commit()
        
        if skip_cached:
            progress(f"Sync complete: {players_synced} synced, {players_skipped} cached (skipped)")


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
    include_future_days: int = 14,
) -> pd.DataFrame:
    season = season or get_current_season()
    return fetch_schedule(season=season, team_ids=team_ids, include_future_days=include_future_days)


def sync_team_metrics(
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Sync comprehensive team metrics from multiple NBA API endpoints into
    the consolidated team_metrics table.

    Endpoints called:
    - TeamEstimatedMetrics (off/def/net rating, pace)
    - LeagueDashTeamStats (Advanced)
    - LeagueDashTeamStats (Four Factors)
    - LeagueDashTeamStats (Opponent)
    - LeagueDashTeamStats (Base, Home)
    - LeagueDashTeamStats (Base, Road)
    - LeagueDashTeamClutch (Advanced)
    - LeagueHustleStatsTeam

    Returns number of teams updated.
    """
    import time as _time
    season = season or get_current_season()
    progress = progress_cb or (lambda _: None)
    migrations.init_db()

    # Accumulate per-team data in a dict keyed by team_id
    team_data: dict[int, dict] = {}

    def _ensure(tid: int) -> dict:
        if tid not in team_data:
            team_data[tid] = {"team_id": tid, "season": season}
        return team_data[tid]

    # 1. TeamEstimatedMetrics
    df = fetch_team_estimated_metrics(season=season, progress_cb=progress_cb)
    if not df.empty:
        for _, row in df.iterrows():
            tid = int(row.get("TEAM_ID", 0))
            if not tid:
                continue
            d = _ensure(tid)
            d["gp"] = _safe_int(row.get("GP"))
            d["w"] = _safe_int(row.get("W"))
            d["l"] = _safe_int(row.get("L"))
            d["w_pct"] = _safe_float(row.get("W_PCT"))
            d["e_off_rating"] = _safe_float(row.get("E_OFF_RATING"))
            d["e_def_rating"] = _safe_float(row.get("E_DEF_RATING"))
            d["e_net_rating"] = _safe_float(row.get("E_NET_RATING"))
            d["e_pace"] = _safe_float(row.get("E_PACE"))
            d["e_ast_ratio"] = _safe_float(row.get("E_AST_RATIO"))
            d["e_oreb_pct"] = _safe_float(row.get("E_OREB_PCT"))
            d["e_dreb_pct"] = _safe_float(row.get("E_DREB_PCT"))
            d["e_reb_pct"] = _safe_float(row.get("E_REB_PCT"))
            d["e_tm_tov_pct"] = _safe_float(row.get("E_TM_TOV_PCT"))
    _time.sleep(0.8)

    # 2. LeagueDashTeamStats (Advanced)
    df = fetch_league_dash_team_stats(season=season, measure_type="Advanced", progress_cb=progress_cb)
    if not df.empty:
        for _, row in df.iterrows():
            tid = int(row.get("TEAM_ID", 0))
            if not tid:
                continue
            d = _ensure(tid)
            d["off_rating"] = _safe_float(row.get("OFF_RATING"))
            d["def_rating"] = _safe_float(row.get("DEF_RATING"))
            d["net_rating"] = _safe_float(row.get("NET_RATING"))
            d["pace"] = _safe_float(row.get("PACE"))
            d["efg_pct"] = _safe_float(row.get("EFG_PCT"))
            d["ts_pct"] = _safe_float(row.get("TS_PCT"))
            d["ast_ratio"] = _safe_float(row.get("AST_RATIO"))
            d["ast_to"] = _safe_float(row.get("AST_TO"))
            d["oreb_pct"] = _safe_float(row.get("OREB_PCT"))
            d["dreb_pct"] = _safe_float(row.get("DREB_PCT"))
            d["reb_pct"] = _safe_float(row.get("REB_PCT"))
            d["tm_tov_pct"] = _safe_float(row.get("TM_TOV_PCT"))
            d["pie"] = _safe_float(row.get("PIE"))
    _time.sleep(0.8)

    # 3. Four Factors
    df = fetch_league_dash_team_stats(season=season, measure_type="Four Factors", progress_cb=progress_cb)
    if not df.empty:
        for _, row in df.iterrows():
            tid = int(row.get("TEAM_ID", 0))
            if not tid:
                continue
            d = _ensure(tid)
            d["ff_efg_pct"] = _safe_float(row.get("EFG_PCT"))
            d["ff_fta_rate"] = _safe_float(row.get("FTA_RATE"))
            d["ff_tm_tov_pct"] = _safe_float(row.get("TM_TOV_PCT"))
            d["ff_oreb_pct"] = _safe_float(row.get("OREB_PCT"))
            d["opp_efg_pct"] = _safe_float(row.get("OPP_EFG_PCT"))
            d["opp_fta_rate"] = _safe_float(row.get("OPP_FTA_RATE"))
            d["opp_tm_tov_pct"] = _safe_float(row.get("OPP_TM_TOV_PCT"))
            d["opp_oreb_pct"] = _safe_float(row.get("OPP_OREB_PCT"))
    _time.sleep(0.8)

    # 4. Opponent stats
    df = fetch_league_dash_team_stats(season=season, measure_type="Opponent", progress_cb=progress_cb)
    if not df.empty:
        for _, row in df.iterrows():
            tid = int(row.get("TEAM_ID", 0))
            if not tid:
                continue
            d = _ensure(tid)
            d["opp_pts"] = _safe_float(row.get("OPP_PTS"))
            d["opp_fg_pct"] = _safe_float(row.get("OPP_FG_PCT"))
            d["opp_fg3_pct"] = _safe_float(row.get("OPP_FG3_PCT"))
            d["opp_ft_pct"] = _safe_float(row.get("OPP_FT_PCT"))
    _time.sleep(0.8)

    # 5. Home splits
    df = fetch_league_dash_team_stats(season=season, measure_type="Base", location="Home", progress_cb=progress_cb)
    if not df.empty:
        for _, row in df.iterrows():
            tid = int(row.get("TEAM_ID", 0))
            if not tid:
                continue
            d = _ensure(tid)
            d["home_gp"] = _safe_int(row.get("GP"))
            d["home_w"] = _safe_int(row.get("W"))
            d["home_l"] = _safe_int(row.get("L"))
            d["home_pts"] = _safe_float(row.get("PTS"))
            # Compute opponent PPG at home from PLUS_MINUS: opp_pts = pts - plus_minus
            pm = _safe_float(row.get("PLUS_MINUS"), 0.0)
            pts = _safe_float(row.get("PTS"), 0.0)
            d["home_opp_pts"] = pts - pm if pts and pm is not None else None
    _time.sleep(0.8)

    # 6. Road splits
    df = fetch_league_dash_team_stats(season=season, measure_type="Base", location="Road", progress_cb=progress_cb)
    if not df.empty:
        for _, row in df.iterrows():
            tid = int(row.get("TEAM_ID", 0))
            if not tid:
                continue
            d = _ensure(tid)
            d["road_gp"] = _safe_int(row.get("GP"))
            d["road_w"] = _safe_int(row.get("W"))
            d["road_l"] = _safe_int(row.get("L"))
            d["road_pts"] = _safe_float(row.get("PTS"))
            pm = _safe_float(row.get("PLUS_MINUS"), 0.0)
            pts = _safe_float(row.get("PTS"), 0.0)
            d["road_opp_pts"] = pts - pm if pts and pm is not None else None
    _time.sleep(0.8)

    # 7. Clutch stats
    df = fetch_team_clutch_stats(season=season, progress_cb=progress_cb)
    if not df.empty:
        for _, row in df.iterrows():
            tid = int(row.get("TEAM_ID", 0))
            if not tid:
                continue
            d = _ensure(tid)
            d["clutch_gp"] = _safe_int(row.get("GP"))
            d["clutch_w"] = _safe_int(row.get("W"))
            d["clutch_l"] = _safe_int(row.get("L"))
            d["clutch_net_rating"] = _safe_float(row.get("NET_RATING"))
            d["clutch_efg_pct"] = _safe_float(row.get("EFG_PCT"))
            d["clutch_ts_pct"] = _safe_float(row.get("TS_PCT"))
    _time.sleep(0.8)

    # 8. Hustle stats
    df = fetch_team_hustle_stats(season=season, progress_cb=progress_cb)
    if not df.empty:
        for _, row in df.iterrows():
            tid = int(row.get("TEAM_ID", 0))
            if not tid:
                continue
            d = _ensure(tid)
            d["deflections"] = _safe_float(row.get("DEFLECTIONS"))
            d["loose_balls_recovered"] = _safe_float(row.get("LOOSE_BALLS_RECOVERED"))
            d["contested_shots"] = _safe_float(row.get("CONTESTED_SHOTS"))
            d["charges_drawn"] = _safe_float(row.get("CHARGES_DRAWN"))
            d["screen_assists"] = _safe_float(row.get("SCREEN_ASSISTS"))

    # Write all team_data to DB
    now_iso = datetime.now().isoformat()
    with get_conn() as conn:
        for d in team_data.values():
            d["last_synced_at"] = now_iso
            cols = list(d.keys())
            placeholders = ", ".join("?" for _ in cols)
            col_names = ", ".join(cols)
            # Build ON CONFLICT update for all columns except PK
            update_parts = [f"{c} = excluded.{c}" for c in cols if c not in ("team_id", "season")]
            update_clause = ", ".join(update_parts)
            conn.execute(
                f"INSERT INTO team_metrics ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT(team_id, season) DO UPDATE SET {update_clause}",
                [d[c] for c in cols],
            )
        conn.commit()

    progress(f"Team metrics synced for {len(team_data)} teams")
    return len(team_data)


def sync_player_impact(
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Sync player on/off impact and estimated metrics into player_impact table.

    - PlayerEstimatedMetrics: one call for all players (USG%, ratings)
    - TeamPlayerOnOffSummary: one call per team (on/off net rating differential)
    """
    import time as _time
    season = season or get_current_season()
    progress = progress_cb or (lambda _: None)
    migrations.init_db()

    player_data: dict[int, dict] = {}

    def _ensure(pid: int) -> dict:
        if pid not in player_data:
            player_data[pid] = {"player_id": pid, "season": season}
        return player_data[pid]

    # 1. Player Estimated Metrics (one API call for all players)
    df = fetch_player_estimated_metrics(season=season, progress_cb=progress_cb)
    if not df.empty:
        for _, row in df.iterrows():
            pid = int(row.get("PLAYER_ID", 0))
            if not pid:
                continue
            d = _ensure(pid)
            d["e_usg_pct"] = _safe_float(row.get("E_USG_PCT"))
            d["e_off_rating"] = _safe_float(row.get("E_OFF_RATING"))
            d["e_def_rating"] = _safe_float(row.get("E_DEF_RATING"))
            d["e_net_rating"] = _safe_float(row.get("E_NET_RATING"))
            d["e_pace"] = _safe_float(row.get("E_PACE"))
            d["e_ast_ratio"] = _safe_float(row.get("E_AST_RATIO"))
            d["e_oreb_pct"] = _safe_float(row.get("E_OREB_PCT"))
            d["e_dreb_pct"] = _safe_float(row.get("E_DREB_PCT"))
    _time.sleep(0.8)

    # 2. On/Off impact per team
    with get_conn() as conn:
        teams = conn.execute("SELECT team_id, abbreviation FROM teams ORDER BY abbreviation").fetchall()

    for idx, (tid, abbr) in enumerate(teams, start=1):
        progress(f"  Fetching on/off data {idx}/{len(teams)} ({abbr})...")
        on_df, off_df = fetch_player_on_off(tid, season=season, progress_cb=progress_cb)

        if not on_df.empty and not off_df.empty:
            # Build lookup: player_id -> on-court stats
            on_lookup = {}
            pid_col = "VS_PLAYER_ID" if "VS_PLAYER_ID" in on_df.columns else "PLAYER_ID"
            for _, row in on_df.iterrows():
                pid = int(row.get(pid_col, 0))
                if pid:
                    on_lookup[pid] = row

            off_pid_col = "VS_PLAYER_ID" if "VS_PLAYER_ID" in off_df.columns else "PLAYER_ID"
            for _, row in off_df.iterrows():
                pid = int(row.get(off_pid_col, 0))
                if pid and pid in on_lookup:
                    on_row = on_lookup[pid]
                    d = _ensure(pid)
                    d["team_id"] = tid
                    d["on_court_off_rating"] = _safe_float(on_row.get("OFF_RATING"))
                    d["on_court_def_rating"] = _safe_float(on_row.get("DEF_RATING"))
                    d["on_court_net_rating"] = _safe_float(on_row.get("NET_RATING"))
                    d["off_court_off_rating"] = _safe_float(row.get("OFF_RATING"))
                    d["off_court_def_rating"] = _safe_float(row.get("DEF_RATING"))
                    d["off_court_net_rating"] = _safe_float(row.get("NET_RATING"))
                    on_net = d["on_court_net_rating"]
                    off_net = d["off_court_net_rating"]
                    if on_net is not None and off_net is not None:
                        d["net_rating_diff"] = on_net - off_net
                    d["on_court_minutes"] = _safe_float(on_row.get("MIN"))
        _time.sleep(0.8)

    # Resolve team_id for players who only got estimated metrics (no on/off)
    with get_conn() as conn:
        pid_to_tid = dict(conn.execute("SELECT player_id, team_id FROM players").fetchall())

    now_iso = datetime.now().isoformat()
    with get_conn() as conn:
        for d in player_data.values():
            if "team_id" not in d:
                d["team_id"] = pid_to_tid.get(d["player_id"], 0)
            if not d.get("team_id"):
                continue  # skip if no team mapping
            d["last_synced_at"] = now_iso
            cols = list(d.keys())
            placeholders = ", ".join("?" for _ in cols)
            col_names = ", ".join(cols)
            update_parts = [f"{c} = excluded.{c}" for c in cols if c not in ("player_id", "season")]
            update_clause = ", ".join(update_parts)
            conn.execute(
                f"INSERT INTO player_impact ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT(player_id, season) DO UPDATE SET {update_clause}",
                [d[c] for c in cols],
            )
        conn.commit()

    progress(f"Player impact synced for {len(player_data)} players")
    return len(player_data)


def full_sync(
    active_only: bool = True, season: Optional[str] = None, progress_cb: Optional[Callable[[str], None]] = None
) -> None:
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)

    # 1. Core reference data (teams + rosters with height/weight/age/exp)
    progress("Sync: reference data (teams + rosters)")
    players_df = sync_reference_data(progress_cb=progress_cb, season=season)

    # 2. Player game logs (now includes WL and PF)
    progress(f"Sync: game logs for {len(players_df)} players (this may take a while)...")
    sync_player_logs(players_df["id"].tolist(), season=season, progress_cb=progress_cb)

    # 3. Current injuries
    progress("Sync: current injuries")
    try:
        sync_injuries(progress_cb=progress)
        progress("Current injuries synced")
    except Exception as exc:
        progress(f"Current injuries sync skipped: {exc}")

    # 4. Historical injury inference
    progress("Building historical injury data...")
    try:
        count = sync_injury_history(progress_cb=progress)
        progress(f"Historical injury data built: {count} records")
    except Exception as exc:
        progress(f"Historical injury build skipped: {exc}")

    # 5. Team advanced metrics (official NBA ratings, Four Factors, clutch, hustle, splits)
    progress("Sync: team advanced metrics (8 API endpoints)...")
    try:
        n = sync_team_metrics(season=season, progress_cb=progress_cb)
        progress(f"Team metrics synced: {n} teams")
    except Exception as exc:
        progress(f"Team metrics sync skipped: {exc}")

    # 6. Player impact (on/off + estimated metrics)
    progress("Sync: player impact data (on/off + estimated metrics)...")
    try:
        n = sync_player_impact(season=season, progress_cb=progress_cb)
        progress(f"Player impact synced: {n} players")
    except Exception as exc:
        progress(f"Player impact sync skipped: {exc}")

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
