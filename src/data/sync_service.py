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


class SyncCancelled(Exception):
    """Raised when a sync operation is cancelled by the user."""
    pass


def _check_cancel(cancel_check: Optional[Callable[[], bool]]) -> None:
    """Raise SyncCancelled if the cancel flag is set."""
    if cancel_check and cancel_check():
        raise SyncCancelled("Sync cancelled by user")


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


# ============ SYNC-META FRESHNESS TRACKING ============

def _get_db_game_snapshot(conn) -> tuple[int, str]:
    """Return (game_count, last_game_date) from player_stats.

    This is the authoritative measure of 'what completed games are in the DB'.
    When a game finishes and box scores are finalized, new player_stats rows
    appear, bumping the count and/or the max date.
    """
    row = conn.execute(
        "SELECT COUNT(DISTINCT game_id), MAX(game_date) FROM player_stats "
        "WHERE game_id IS NOT NULL AND game_id != ''"
    ).fetchone()
    return (row[0] or 0, row[1] or "")


def _get_sync_meta(conn, step_name: str) -> Optional[tuple[str, int, str]]:
    """Return (last_synced_at, game_count_at_sync, last_game_date_at_sync) or None."""
    row = conn.execute(
        "SELECT last_synced_at, game_count_at_sync, last_game_date_at_sync "
        "FROM sync_meta WHERE step_name = ?",
        (step_name,),
    ).fetchone()
    return row if row else None


def _set_sync_meta(conn, step_name: str, game_count: int, last_game_date: str) -> None:
    """Record that *step_name* was just completed."""
    conn.execute(
        """
        INSERT INTO sync_meta (step_name, last_synced_at, game_count_at_sync, last_game_date_at_sync)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(step_name) DO UPDATE SET
            last_synced_at = excluded.last_synced_at,
            game_count_at_sync = excluded.game_count_at_sync,
            last_game_date_at_sync = excluded.last_game_date_at_sync
        """,
        (step_name, datetime.now().isoformat(), game_count, last_game_date),
    )
    conn.commit()


def _is_step_fresh(conn, step_name: str, max_age_hours: float = 6.0) -> bool:
    """A step is fresh if:
    1. It was synced within *max_age_hours*, AND
    2. No new completed games have appeared since (game count + last date unchanged).
    """
    meta = _get_sync_meta(conn, step_name)
    if not meta:
        return False  # never ran
    last_synced_at, saved_count, saved_date = meta

    # Check age
    try:
        synced_dt = datetime.fromisoformat(last_synced_at)
        if (datetime.now() - synced_dt) > timedelta(hours=max_age_hours):
            return False  # too old
    except (ValueError, TypeError):
        return False

    # Check if completed games changed
    current_count, current_date = _get_db_game_snapshot(conn)
    if current_count != saved_count or current_date != saved_date:
        return False  # new games completed since last sync

    return True


def _get_teams_with_new_games(conn, step_name: str) -> Set[int]:
    """Return team IDs that have new completed games since the last sync of *step_name*.

    Used for targeted game-log syncing: only re-fetch players on teams
    that actually played since the last sync.
    """
    meta = _get_sync_meta(conn, step_name)
    if not meta:
        return set()  # never ran — caller should do full sync

    _, saved_count, saved_date = meta

    # Find games newer than what we had at last sync.
    # A game is "new" if its game_date is after saved_date, or if game count grew
    # (could be same date, late games).
    if saved_date:
        rows = conn.execute(
            """
            SELECT DISTINCT opponent_team_id FROM player_stats
            WHERE game_date >= ? AND game_id IS NOT NULL AND game_id != ''
            UNION
            SELECT DISTINCT ps2.opponent_team_id FROM player_stats ps2
            WHERE ps2.game_date = ? AND ps2.game_id IS NOT NULL AND ps2.game_id != ''
            """,
            (saved_date, saved_date),
        ).fetchall()
        # Also include the "home" side: player's own team
        player_rows = conn.execute(
            """
            SELECT DISTINCT p.team_id FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE ps.game_date >= ? AND ps.game_id IS NOT NULL AND ps.game_id != ''
            """,
            (saved_date,),
        ).fetchall()
        team_ids = {r[0] for r in rows} | {r[0] for r in player_rows}
        return team_ids
    return set()


def sync_reference_data(
    progress_cb: Optional[Callable[[str], None]] = None,
    season: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Sync teams and players to DB, returns players_df for reuse.

    Freshness: rosters change from trades/signings which are public well
    in advance.  Injury status is handled by ``sync_injuries``.  If teams +
    players already exist in the DB and were synced within the last 24 hours,
    skip the API calls.
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)
    migrations.init_db()
    migrations.ensure_columns()

    # ── Freshness check ──
    if not force:
        with get_conn() as conn:
            if _is_step_fresh(conn, "reference_data", max_age_hours=24.0):
                team_count = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
                player_count = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
                if team_count >= 30 and player_count > 100:
                    progress(f"Reference data fresh (synced <24 hrs ago, {player_count} players) — skipping API calls")
                    # Build players_df from DB
                    players_df = pd.read_sql(
                        "SELECT player_id AS id, name AS full_name, team_id, position, "
                        "height, weight, age, experience FROM players",
                        conn,
                    )
                    return players_df

    progress("Fetching teams...")
    teams_df = fetch_teams()
    progress("Fetching players...")
    players_df = fetch_players(season=season, progress_cb=progress_cb)
    players_df = players_df.dropna(subset=["team_id"])
    players_df["team_id"] = players_df["team_id"].astype(int)

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO teams (team_id, name, abbreviation, conference)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                name = excluded.name,
                abbreviation = excluded.abbreviation,
                conference = excluded.conference
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

        # Record freshness
        gc, gd = _get_db_game_snapshot(conn)
        _set_sync_meta(conn, "reference_data", gc, gd)
        conn.commit()

    progress(f"Reference data synced: {len(players_df)} players across {len(teams_df)} teams")
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
    force: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> None:
    """
    Sync all player game logs with comprehensive stats.
    
    Freshness / caching behaviour:
    - If no new completed games since last sync, skip entirely (fast path).
    - Otherwise, use the per-player sync cache: players synced within
      cache_max_age_hours are skipped UNLESS they have recent games or are
      on a team that just played.
    - Set force=True to bypass all freshness checks.
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)
    player_ids = list(player_ids)
    total = len(player_ids)
    
    with get_conn() as conn:
        abbr_to_id = _team_abbr_lookup(conn)

        # ── Fast-path freshness check ──
        if not force and _is_step_fresh(conn, "game_logs", max_age_hours=24.0):
            cached_count = conn.execute(
                "SELECT COUNT(*) FROM player_sync_cache"
            ).fetchone()[0]
            if cached_count > 100:
                progress(f"Game logs fresh (no new completed games since last sync, "
                         f"{cached_count} players cached) — skipping")
                return
        
        # Determine which players to skip
        players_to_skip: Set[int] = set()
        if skip_cached and not force:
            cached_players = _get_cached_players(conn, max_age_hours=cache_max_age_hours)
            recent_game_players = _get_players_with_recent_games(conn, days=force_recent_days)

            # Safety check: if the cache says players are synced but player_stats
            # is empty, the cache is stale (likely from a CASCADE delete).
            # Clear it so all players get re-fetched.
            if cached_players:
                stats_count = conn.execute("SELECT COUNT(*) FROM player_stats").fetchone()[0]
                if stats_count == 0:
                    progress("Cache is stale (player_stats is empty) — clearing cache and re-syncing all players")
                    conn.execute("DELETE FROM player_sync_cache")
                    conn.commit()
                    cached_players = set()

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
            
            _check_cancel(cancel_check)

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
        
        # Record freshness snapshot AFTER writing new data
        gc, gd = _get_db_game_snapshot(conn)
        _set_sync_meta(conn, "game_logs", gc, gd)
        conn.commit()
        
        if skip_cached or force:
            progress(f"Sync complete: {players_synced} synced, {players_skipped} cached (skipped)")


def _normalise_status_level(raw_status: str) -> str:
    """Map raw injury status text to a canonical level."""
    s = raw_status.strip().lower()
    if s in ("out", "o"):
        return "Out"
    if s in ("doubtful", "d"):
        return "Doubtful"
    if s in ("questionable", "q"):
        return "Questionable"
    if s in ("probable", "p"):
        return "Probable"
    if "day" in s and "day" in s:
        return "Day-To-Day"
    if "gtd" in s or "game time" in s:
        return "GTD"
    # Fallback: anything else treat as the raw text title-cased
    return raw_status.strip().title() or "Out"


def _extract_injury_keyword(injury_text: str) -> str:
    """Extract a normalised injury keyword for categorisation."""
    t = injury_text.lower().strip()
    # Ordered: check most specific first
    keywords = [
        ("rest", "rest"), ("load management", "rest"),
        ("personal", "personal"), ("suspension", "suspension"),
        ("illness", "illness"), ("flu", "illness"), ("sick", "illness"),
        ("concussion", "concussion"), ("head", "head"),
        ("hamstring", "hamstring"), ("quad", "quad"),
        ("calf", "calf"), ("groin", "groin"),
        ("ankle", "ankle"), ("foot", "foot"), ("toe", "toe"),
        ("knee", "knee"), ("acl", "knee"), ("mcl", "knee"), ("meniscus", "knee"),
        ("hip", "hip"), ("back", "back"), ("spine", "back"),
        ("shoulder", "shoulder"), ("elbow", "elbow"),
        ("wrist", "wrist"), ("hand", "hand"), ("finger", "finger"), ("thumb", "finger"),
        ("achilles", "achilles"),
        ("thigh", "thigh"), ("leg", "leg"),
        ("rib", "rib"), ("chest", "chest"), ("abdomen", "abdomen"),
        ("neck", "neck"),
        ("eye", "eye"),
        ("sprain", "sprain"), ("strain", "strain"), ("soreness", "soreness"),
        ("contusion", "contusion"), ("bruise", "contusion"),
        ("fracture", "fracture"), ("break", "fracture"),
        ("surgery", "surgery"), ("rehab", "surgery"),
    ]
    for needle, category in keywords:
        if needle in t:
            return category
    return "other"


def sync_injuries(progress_cb: Optional[Callable[[str], None]] = None) -> int:
    """
    Sync injury data from web sources.

    In addition to the legacy ``is_injured`` / ``injury_note`` flags on the
    ``players`` table, this now also **logs every observed status** into the
    ``injury_status_log`` table.  Over time this builds up the history needed
    for play-through rate analysis (see ``injury_intelligence.py``).

    Returns number of injuries updated.
    """
    progress = progress_cb or (lambda _: None)

    # Fetch from multiple sources + manual entries
    data = get_all_injuries(progress_cb=progress)
    if not data:
        progress("No injury data retrieved")
        return 0

    today_iso = date.today().isoformat()

    with get_conn() as conn:
        # First, clear all existing injury flags (they're refreshed each sync)
        progress("Clearing previous injury flags...")
        conn.execute("UPDATE players SET is_injured = 0, injury_note = NULL")

        updated = 0
        logged = 0
        for entry in data:
            player_name = entry["player"]
            raw_status = entry["status"]
            injury = entry["injury"]
            note = f"{raw_status}: {injury}".strip()
            if entry.get("update"):
                note += f" ({entry['update']})"

            status_level = _normalise_status_level(raw_status)
            injury_keyword = _extract_injury_keyword(injury)

            # ── Update legacy is_injured flag ──
            cursor = conn.execute(
                "UPDATE players SET is_injured = 1, injury_note = ? "
                "WHERE lower(name) = lower(?)",
                (note, player_name),
            )
            matched_pid: Optional[int] = None
            matched_tid: Optional[int] = None
            if cursor.rowcount > 0:
                updated += cursor.rowcount
                row = conn.execute(
                    "SELECT player_id, team_id FROM players WHERE lower(name) = lower(?)",
                    (player_name,),
                ).fetchone()
                if row:
                    matched_pid, matched_tid = row
            else:
                # Try partial match (last name)
                parts = player_name.split()
                if len(parts) >= 2:
                    last_name = parts[-1]
                    cursor = conn.execute(
                        "UPDATE players SET is_injured = 1, injury_note = ? "
                        "WHERE name LIKE ? AND is_injured = 0",
                        (note, f"%{last_name}%"),
                    )
                    updated += cursor.rowcount
                    if cursor.rowcount > 0:
                        row = conn.execute(
                            "SELECT player_id, team_id FROM players "
                            "WHERE name LIKE ? ORDER BY player_id LIMIT 1",
                            (f"%{last_name}%",),
                        ).fetchone()
                        if row:
                            matched_pid, matched_tid = row

            # ── Log to injury_status_log ──
            if matched_pid and matched_tid:
                conn.execute(
                    """
                    INSERT INTO injury_status_log
                        (player_id, team_id, log_date, status_level,
                         injury_keyword, injury_detail)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_id, log_date, status_level) DO UPDATE SET
                        injury_keyword = excluded.injury_keyword,
                        injury_detail  = excluded.injury_detail,
                        team_id        = excluded.team_id
                    """,
                    (matched_pid, matched_tid, today_iso,
                     status_level, injury_keyword, note),
                )
                logged += 1

        conn.commit()
        progress(f"Updated {updated} injury flags, logged {logged} status entries")
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
    force: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> int:
    """
    Sync comprehensive team metrics from multiple NBA API endpoints into
    the consolidated team_metrics table.

    Freshness: these are season aggregates that only change when games
    complete.  If no new completed games since the last sync, skip all
    8 API calls entirely.

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

    # ── Freshness check ──
    if not force:
        with get_conn() as conn:
            if _is_step_fresh(conn, "team_metrics", max_age_hours=168.0):  # up to 7 days if no new games
                existing = conn.execute("SELECT COUNT(*) FROM team_metrics").fetchone()[0]
                if existing >= 30:
                    progress(f"Team metrics fresh (no new completed games since last sync, "
                             f"{existing} teams cached) — skipping 8 API calls")
                    return existing

    # Accumulate per-team data in a dict keyed by team_id
    team_data: dict[int, dict] = {}

    def _ensure(tid: int) -> dict:
        if tid not in team_data:
            team_data[tid] = {"team_id": tid, "season": season}
        return team_data[tid]

    # 1. TeamEstimatedMetrics
    _check_cancel(cancel_check)
    progress("Team metrics [1/8] Estimated metrics...")
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
    _check_cancel(cancel_check)
    progress("Team metrics [2/8] Advanced stats...")
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
    _check_cancel(cancel_check)
    progress("Team metrics [3/8] Four factors...")
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
    _check_cancel(cancel_check)
    progress("Team metrics [4/8] Opponent stats...")
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
    _check_cancel(cancel_check)
    progress("Team metrics [5/8] Home splits...")
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
    _check_cancel(cancel_check)
    progress("Team metrics [6/8] Road splits...")
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
    _check_cancel(cancel_check)
    progress("Team metrics [7/8] Clutch stats...")
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
    _check_cancel(cancel_check)
    progress("Team metrics [8/8] Hustle stats...")
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

    # Record freshness
    with get_conn() as conn:
        gc, gd = _get_db_game_snapshot(conn)
        _set_sync_meta(conn, "team_metrics", gc, gd)

    progress(f"Team metrics synced for {len(team_data)} teams")
    return len(team_data)


def sync_player_impact(
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    force: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> int:
    """
    Sync player on/off impact and estimated metrics into player_impact table.

    Freshness: like team metrics, these only change after completed games.
    Skip the 31+ API calls if no new games.

    - PlayerEstimatedMetrics: one call for all players (USG%, ratings)
    - TeamPlayerOnOffSummary: one call per team (on/off net rating differential)
    """
    import time as _time
    season = season or get_current_season()
    progress = progress_cb or (lambda _: None)
    migrations.init_db()

    # ── Freshness check ──
    if not force:
        with get_conn() as conn:
            if _is_step_fresh(conn, "player_impact", max_age_hours=168.0):
                existing = conn.execute("SELECT COUNT(*) FROM player_impact").fetchone()[0]
                if existing > 100:
                    progress(f"Player impact fresh (no new completed games since last sync, "
                             f"{existing} players cached) — skipping API calls")
                    return existing

    player_data: dict[int, dict] = {}

    def _ensure(pid: int) -> dict:
        if pid not in player_data:
            player_data[pid] = {"player_id": pid, "season": season}
        return player_data[pid]

    # 1. Player Estimated Metrics (one API call for all players)
    _check_cancel(cancel_check)
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
        _check_cancel(cancel_check)
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

    # Record freshness
    with get_conn() as conn:
        gc, gd = _get_db_game_snapshot(conn)
        _set_sync_meta(conn, "player_impact", gc, gd)

    progress(f"Player impact synced for {len(player_data)} players")
    return len(player_data)


def full_sync(
    active_only: bool = True,
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    force: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Full data sync with smart freshness checks.

    Each step independently checks whether it needs to re-fetch from the
    NBA API.  The key principle: **if no new games have completed since the
    last sync of that step, skip the API calls**.

    Set *force=True* to bypass all freshness checks.
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)

    if force:
        progress("Force mode: all freshness checks bypassed")

    # 1. Core reference data (teams + rosters with height/weight/age/exp)
    _check_cancel(cancel_check)
    progress("Sync: reference data (teams + rosters)")
    players_df = sync_reference_data(progress_cb=progress_cb, season=season, force=force)

    # 2. Player game logs (now includes WL and PF)
    _check_cancel(cancel_check)
    progress(f"Sync: game logs for {len(players_df)} players...")
    sync_player_logs(
        players_df["id"].tolist(), season=season, progress_cb=progress_cb,
        force=force, cancel_check=cancel_check,
    )

    # 3. Current injuries (ALWAYS refresh — lightweight and can change any time)
    _check_cancel(cancel_check)
    progress("Sync: current injuries")
    try:
        sync_injuries(progress_cb=progress)
        progress("Current injuries synced")
    except Exception as exc:
        progress(f"Current injuries sync skipped: {exc}")

    # 3b. Backfill injury status outcomes (cross-reference log with game logs)
    _check_cancel(cancel_check)
    progress("Backfilling injury play outcomes...")
    try:
        from src.analytics.injury_intelligence import backfill_play_outcomes
        n = backfill_play_outcomes(progress_cb=progress)
        progress(f"Injury backfill: {n} outcomes resolved")
    except Exception as exc:
        progress(f"Injury backfill skipped: {exc}")

    # 4. Historical injury inference (skip if game log count unchanged)
    _check_cancel(cancel_check)
    progress("Building historical injury data...")
    try:
        _skip_history = False
        if not force:
            with get_conn() as conn:
                if _is_step_fresh(conn, "injury_history", max_age_hours=168.0):
                    existing = conn.execute("SELECT COUNT(*) FROM injury_history").fetchone()[0]
                    if existing > 0:
                        progress(f"Injury history fresh ({existing} records, no new games) — skipping")
                        _skip_history = True
        if not _skip_history:
            count = sync_injury_history(progress_cb=progress)
            progress(f"Historical injury data built: {count} records")
            with get_conn() as conn:
                gc, gd = _get_db_game_snapshot(conn)
                _set_sync_meta(conn, "injury_history", gc, gd)
    except Exception as exc:
        progress(f"Historical injury build skipped: {exc}")

    # 5. Team advanced metrics (official NBA ratings, Four Factors, clutch, hustle, splits)
    _check_cancel(cancel_check)
    progress("Sync: team advanced metrics (8 API endpoints)...")
    try:
        n = sync_team_metrics(season=season, progress_cb=progress_cb, force=force,
                              cancel_check=cancel_check)
        progress(f"Team metrics: {n} teams")
    except Exception as exc:
        progress(f"Team metrics sync skipped: {exc}")

    # 6. Player impact (on/off + estimated metrics)
    _check_cancel(cancel_check)
    progress("Sync: player impact data (on/off + estimated metrics)...")
    try:
        n = sync_player_impact(season=season, progress_cb=progress_cb, force=force,
                               cancel_check=cancel_check)
        progress(f"Player impact: {n} players")
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
            INSERT INTO teams (team_id, name, abbreviation, conference)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                name = excluded.name,
                abbreviation = excluded.abbreviation,
                conference = excluded.conference
            """,
            subset[["id", "full_name", "abbreviation", "conference"]].itertuples(index=False),
        )
        conn.commit()


def sync_quarter_scores(game_id: str) -> bool:
    """Opportunistically store quarter-by-quarter scores for a game.

    Called when viewing a game in the Gamecast tab.  Pulls linescores
    from the ESPN Summary API and upserts into ``game_quarter_scores``.

    Returns True if any rows were stored / updated.
    """
    from src.data.gamecast import get_quarter_scores

    migrations.init_db()
    qs = get_quarter_scores(game_id)
    if not qs:
        return False

    # We need our internal team_id (NBA API), not ESPN's team ID.
    # Map via abbreviation.
    with get_conn() as conn:
        abbr_to_id = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT abbreviation, team_id FROM teams"
            ).fetchall()
        }

    # ESPN uses shorter abbreviations for a few teams
    espn_to_nba = {
        "GS": "GSW", "SA": "SAS", "NY": "NYK",
        "NO": "NOP", "UTAH": "UTA", "WSH": "WAS",
    }

    rows = []
    for prefix in ("home", "away"):
        abbr = qs.get(f"{prefix}_abbr", "")
        abbr = espn_to_nba.get(abbr, abbr)  # normalise ESPN → NBA
        team_id = abbr_to_id.get(abbr)
        if not team_id:
            continue
        linescores = qs.get(f"{prefix}_linescores", [])
        q1 = linescores[0] if len(linescores) > 0 else None
        q2 = linescores[1] if len(linescores) > 1 else None
        q3 = linescores[2] if len(linescores) > 2 else None
        q4 = linescores[3] if len(linescores) > 3 else None
        ot = sum(linescores[4:]) if len(linescores) > 4 else 0
        final = qs.get(f"{prefix}_final", 0)
        game_date = qs.get("game_date", "")
        is_home = 1 if prefix == "home" else 0
        rows.append((
            game_id, team_id, q1, q2, q3, q4, ot, final, game_date, is_home,
        ))

    if not rows:
        return False

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO game_quarter_scores
                (game_id, team_id, q1, q2, q3, q4, ot, final_score, game_date, is_home)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id, team_id) DO UPDATE SET
                q1=excluded.q1, q2=excluded.q2, q3=excluded.q3, q4=excluded.q4,
                ot=excluded.ot, final_score=excluded.final_score,
                game_date=excluded.game_date, is_home=excluded.is_home
            """,
            rows,
        )
        conn.commit()
    return True
