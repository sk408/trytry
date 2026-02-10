"""
Historical injury inference from game logs.

This module analyzes player game logs to infer when players were likely
injured/out based on missing games. If a player who normally plays
significant minutes has no stats for a game where their team played,
they were likely out.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.database.db import get_conn


@dataclass
class PlayerGameStatus:
    """Status of a player for a specific game."""
    player_id: int
    player_name: str
    team_id: int
    position: str
    game_date: date
    was_out: bool
    avg_minutes: float  # Their typical minutes per game
    

def get_team_game_dates(team_id: int) -> List[Tuple[date, int, bool]]:
    """
    Get all game dates for a team with opponent and home/away info.
    Returns list of (game_date, opponent_id, is_home).
    """
    with get_conn() as conn:
        # Get games from player_stats - if any player on the team played, the team had a game
        df = pd.read_sql(
            """
            SELECT DISTINCT ps.game_date, ps.opponent_team_id, ps.is_home
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ?
            ORDER BY ps.game_date
            """,
            conn,
            params=[team_id],
        )
    
    return [
        (pd.to_datetime(row["game_date"]).date(), int(row["opponent_team_id"]), bool(row["is_home"]))
        for _, row in df.iterrows()
    ]


def get_player_season_average(player_id: int, before_date: Optional[date] = None) -> Dict[str, float]:
    """
    Get a player's season averages, optionally only from games before a date.
    This is used to know what their "normal" minutes were at that point in the season.
    """
    with get_conn() as conn:
        if before_date:
            df = pd.read_sql(
                """
                SELECT AVG(minutes) as avg_min, AVG(points) as avg_pts, COUNT(*) as games
                FROM player_stats
                WHERE player_id = ? AND game_date < ?
                """,
                conn,
                params=[player_id, str(before_date)],
            )
        else:
            df = pd.read_sql(
                """
                SELECT AVG(minutes) as avg_min, AVG(points) as avg_pts, COUNT(*) as games
                FROM player_stats
                WHERE player_id = ?
                """,
                conn,
                params=[player_id],
            )
    
    if df.empty or df.iloc[0]["games"] == 0:
        return {"avg_min": 0.0, "avg_pts": 0.0, "games": 0}
    
    return {
        "avg_min": float(df.iloc[0]["avg_min"] or 0),
        "avg_pts": float(df.iloc[0]["avg_pts"] or 0),
        "games": int(df.iloc[0]["games"]),
    }


def get_players_who_played_on_date(team_id: int, game_date: date) -> Set[int]:
    """Get set of player_ids who have stats for this team on this date."""
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
    return {r[0] for r in rows}


def infer_injuries_for_game(
    team_id: int,
    game_date: date,
    min_games_threshold: int = 3,
    min_minutes_threshold: float = 12.0,
) -> List[PlayerGameStatus]:
    """
    Infer which rotation players were out for a specific game.
    
    A player is considered "out" if:
    - They had played at least min_games_threshold games before this date
    - Their average minutes was >= min_minutes_threshold
    - They have no stats for this game date
    
    Returns list of PlayerGameStatus for players who were out.
    """
    # Get all players on this team (as of current roster - not perfect but workable)
    with get_conn() as conn:
        players_df = pd.read_sql(
            """
            SELECT player_id, name, position, team_id
            FROM players
            WHERE team_id = ?
            """,
            conn,
            params=[team_id],
        )
    
    if players_df.empty:
        return []
    
    # Get who actually played
    played = get_players_who_played_on_date(team_id, game_date)
    
    out_players = []
    for _, player in players_df.iterrows():
        player_id = int(player["player_id"])
        
        # Skip if they played
        if player_id in played:
            continue
        
        # Get their averages BEFORE this game date
        avgs = get_player_season_average(player_id, before_date=game_date)
        
        # Check if they were a rotation player at this point
        if avgs["games"] >= min_games_threshold and avgs["avg_min"] >= min_minutes_threshold:
            out_players.append(PlayerGameStatus(
                player_id=player_id,
                player_name=str(player["name"]),
                team_id=team_id,
                position=str(player["position"] or ""),
                game_date=game_date,
                was_out=True,
                avg_minutes=avgs["avg_min"],
            ))
    
    return out_players


def build_injury_history(
    progress_cb: Optional[Callable[[str], None]] = None,
    min_games_threshold: int = 3,
    min_minutes_threshold: float = 12.0,
) -> int:
    """
    Build injury history table by inferring injuries from game logs.

    Uses batch data loading (3 queries total) instead of per-player/per-game
    queries.  Includes diagnostic progress messages so silent-0 results can
    be debugged.

    Returns number of injury records created.
    """
    progress = progress_cb or (lambda _: None)

    # ------------------------------------------------------------------ #
    # 1.  Bulk-load everything we need in 3 queries
    # ------------------------------------------------------------------ #
    with get_conn() as conn:
        teams_df = pd.read_sql(
            "SELECT team_id, abbreviation FROM teams", conn
        )
        roster_df = pd.read_sql(
            "SELECT player_id, team_id, name, position FROM players", conn
        )
        stats_df = pd.read_sql(
            "SELECT player_id, game_date, minutes FROM player_stats "
            "ORDER BY game_date",
            conn,
        )
        # Clear old inferred records
        conn.execute("DELETE FROM injury_history WHERE reason = 'inferred'")
        conn.commit()

    progress(
        f"[Injury History] Loaded: {len(teams_df)} teams, "
        f"{len(roster_df)} roster players, {len(stats_df)} game-log rows"
    )

    # -- Early-exit diagnostics --
    if stats_df.empty:
        progress(
            "[Injury History] WARNING: player_stats table is EMPTY. "
            "Run a full data sync first."
        )
        return 0
    if roster_df.empty:
        progress("[Injury History] WARNING: players table is EMPTY.")
        return 0

    # Show sample dates for format verification
    sample_dates = stats_df["game_date"].dropna().unique()[:5].tolist()
    progress(f"[Injury History] Sample game_date values in DB: {sample_dates}")

    # ------------------------------------------------------------------ #
    # 2.  Build in-memory lookup structures
    # ------------------------------------------------------------------ #
    progress("[Injury History] Building lookup structures...")

    # player_id -> current team_id
    player_team: Dict[int, int] = dict(
        zip(roster_df["player_id"].astype(int), roster_df["team_id"].astype(int))
    )

    # player_id -> (name, position)
    player_info: Dict[int, Dict[str, str]] = {}
    for _, row in roster_df.iterrows():
        player_info[int(row["player_id"])] = {
            "name": str(row["name"]),
            "position": str(row["position"] or ""),
        }

    # Attach current team_id to every stats row so we can group by team
    stats_df["team_id"] = stats_df["player_id"].map(player_team)
    stats_df = stats_df.dropna(subset=["team_id"])
    stats_df["team_id"] = stats_df["team_id"].astype(int)
    stats_df["player_id"] = stats_df["player_id"].astype(int)
    # Ensure game_date is a plain string for consistent comparisons
    stats_df["game_date"] = stats_df["game_date"].astype(str)

    progress(
        f"[Injury History] After team mapping: {len(stats_df)} rows "
        f"({len(stats_df['player_id'].unique())} unique players)"
    )

    # team_id -> sorted list of unique game-date strings
    team_game_dates: Dict[int, List[str]] = {}
    for tid, grp in stats_df.groupby("team_id"):
        team_game_dates[int(tid)] = sorted(grp["game_date"].unique().tolist())

    # Fast "did this player play on this date?" set
    played_set: Set[Tuple[int, str]] = set(
        zip(stats_df["player_id"], stats_df["game_date"])
    )

    # Per-player cumulative stats for running-average lookups.
    # For each player we store parallel lists sorted by game_date:
    #   dates[i]    – the date of their (i+1)-th game
    #   cum_mins[i] – total minutes through their (i+1)-th game
    # To find stats *before* a query date gd  we binary-search dates and
    # use the cumulative value at the insertion point.
    progress("[Injury History] Building per-player cumulative stats...")
    player_cum: Dict[int, Tuple[List[str], List[float]]] = {}
    for pid, grp in stats_df.groupby("player_id"):
        pid = int(pid)
        sorted_rows = grp.sort_values("game_date")
        dates: List[str] = sorted_rows["game_date"].tolist()
        cum: List[float] = []
        total = 0.0
        for m in sorted_rows["minutes"].fillna(0).astype(float):
            total += m
            cum.append(total)
        player_cum[pid] = (dates, cum)

    def _stats_before(pid: int, gd: str) -> Tuple[int, float]:
        """Return (games_played, avg_minutes) for *pid* from games before *gd*."""
        if pid not in player_cum:
            return 0, 0.0
        dates, cum = player_cum[pid]
        n = bisect_left(dates, gd)  # games played strictly before gd
        if n == 0:
            return 0, 0.0
        return n, cum[n - 1] / n

    # ------------------------------------------------------------------ #
    # 3.  Scan every team × game-date for missing rotation players
    # ------------------------------------------------------------------ #
    total_injuries = 0
    total_checked = 0
    total_below_games = 0
    total_below_minutes = 0
    injury_records: List[Tuple[int, int, str, float]] = []

    n_teams = len(teams_df)
    progress(f"[Injury History] Scanning {n_teams} teams for missing players...")

    for t_idx, t_row in teams_df.iterrows():
        team_id = int(t_row["team_id"])
        team_abbr = str(t_row["abbreviation"])
        team_num = int(t_idx) + 1 if isinstance(t_idx, int) else t_idx

        team_roster = [pid for pid, tid in player_team.items() if tid == team_id]
        game_dates = team_game_dates.get(team_id, [])

        if not game_dates:
            progress(
                f"  [{team_num}/{n_teams}] {team_abbr}: 0 game dates – skipping"
            )
            continue

        team_out = 0
        for gd in game_dates:
            for pid in team_roster:
                if (pid, gd) in played_set:
                    continue  # player has stats for this game

                total_checked += 1
                n_games, avg_min = _stats_before(pid, gd)

                if n_games < min_games_threshold:
                    total_below_games += 1
                    continue
                if avg_min < min_minutes_threshold:
                    total_below_minutes += 1
                    continue

                injury_records.append((pid, team_id, gd, avg_min))
                team_out += 1

        total_injuries += team_out
        progress(
            f"  [{team_num}/{n_teams}] {team_abbr}: {len(game_dates)} games, "
            f"{len(team_roster)} roster, {team_out} inferred out"
        )

    # ------------------------------------------------------------------ #
    # 4.  Single batch insert
    # ------------------------------------------------------------------ #
    if injury_records:
        with get_conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO injury_history
                (player_id, team_id, game_date, was_out, avg_minutes, reason)
                VALUES (?, ?, ?, 1, ?, 'inferred')
                """,
                injury_records,
            )
            conn.commit()

    progress(
        f"[Injury History] Done – {total_injuries} injury/out records created"
    )
    progress(
        f"  {total_checked} absent player-game combos checked | "
        f"{total_below_games} below {min_games_threshold}-game threshold | "
        f"{total_below_minutes} below {min_minutes_threshold}-min threshold"
    )
    return total_injuries


def get_injuries_for_game(team_id: int, game_date: date) -> List[Dict]:
    """
    Get list of players who were out for a specific game.
    Returns list of dicts with player_id, name, position, avg_minutes.
    """
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ih.player_id, p.name, p.position, ih.avg_minutes
            FROM injury_history ih
            JOIN players p ON p.player_id = ih.player_id
            WHERE ih.team_id = ? AND ih.game_date = ? AND ih.was_out = 1
            ORDER BY ih.avg_minutes DESC
            """,
            conn,
            params=[team_id, str(game_date)],
        )
    
    return [
        {
            "player_id": int(row["player_id"]),
            "name": str(row["name"]),
            "position": str(row["position"] or ""),
            "avg_minutes": float(row["avg_minutes"]),
        }
        for _, row in df.iterrows()
    ]


def get_team_injuries_summary(team_id: int) -> pd.DataFrame:
    """Get summary of all inferred injuries for a team."""
    with get_conn() as conn:
        return pd.read_sql(
            """
            SELECT p.name, p.position, ih.game_date, ih.avg_minutes
            FROM injury_history ih
            JOIN players p ON p.player_id = ih.player_id
            WHERE ih.team_id = ? AND ih.was_out = 1
            ORDER BY ih.game_date DESC, ih.avg_minutes DESC
            """,
            conn,
            params=[team_id],
        )
