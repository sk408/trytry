"""
Historical injury inference from game logs.

This module analyzes player game logs to infer when players were likely
injured/out based on missing games. If a player who normally plays
significant minutes has no stats for a game where their team played,
they were likely out.
"""
from __future__ import annotations

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
    
    Returns number of injury records created.
    """
    progress = progress_cb or (lambda _: None)
    
    with get_conn() as conn:
        # Get all teams
        teams = conn.execute("SELECT team_id, abbreviation FROM teams").fetchall()
        
        # Clear existing inferred injuries
        progress("Clearing previous injury history...")
        conn.execute("DELETE FROM injury_history WHERE reason = 'inferred'")
        conn.commit()
    
    total_injuries = 0
    
    for idx, (team_id, team_abbr) in enumerate(teams, 1):
        progress(f"Analyzing injuries for {team_abbr} ({idx}/{len(teams)})...")
        
        # Get all game dates for this team
        game_dates = get_team_game_dates(team_id)
        
        team_injuries = []
        for game_date, opp_id, is_home in game_dates:
            out_players = infer_injuries_for_game(
                team_id, game_date,
                min_games_threshold=min_games_threshold,
                min_minutes_threshold=min_minutes_threshold,
            )
            team_injuries.extend(out_players)
        
        # Batch insert
        if team_injuries:
            with get_conn() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO injury_history 
                    (player_id, team_id, game_date, was_out, avg_minutes, reason)
                    VALUES (?, ?, ?, 1, ?, 'inferred')
                    """,
                    [
                        (p.player_id, p.team_id, str(p.game_date), p.avg_minutes)
                        for p in team_injuries
                    ],
                )
                conn.commit()
            total_injuries += len(team_injuries)
    
    progress(f"Inferred {total_injuries} injury/out records across all teams")
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
