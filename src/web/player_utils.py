"""Player utility functions for the web app (no PySide6 dependency)."""
from __future__ import annotations

import pandas as pd

from src.database.db import get_conn


def load_players_df() -> pd.DataFrame:
    """Load all players with team info."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT p.player_id, p.name, p.position, t.abbreviation AS team, 
                   p.is_injured, p.injury_note
            FROM players p
            LEFT JOIN teams t ON p.team_id = t.team_id
            ORDER BY t.abbreviation, p.name
            """,
            conn,
        )
    return df


def load_injured_with_stats() -> pd.DataFrame:
    """Load injured players with their stats for impact assessment."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT 
                p.player_id, 
                p.name, 
                p.position, 
                t.abbreviation AS team,
                p.injury_note,
                COALESCE(AVG(ps.points), 0) as ppg,
                COALESCE(AVG(ps.minutes), 0) as mpg,
                COALESCE(AVG(ps.rebounds), 0) as rpg,
                COALESCE(AVG(ps.assists), 0) as apg,
                COUNT(ps.game_date) as games
            FROM players p
            LEFT JOIN teams t ON p.team_id = t.team_id
            LEFT JOIN player_stats ps ON p.player_id = ps.player_id
            WHERE p.is_injured = 1
            GROUP BY p.player_id, p.name, p.position, t.abbreviation, p.injury_note
            ORDER BY t.abbreviation, COALESCE(AVG(ps.minutes), 0) DESC
            """,
            conn,
        )
    return df


def get_position_display(position: str) -> str:
    """Get readable position display."""
    pos = (position or "").upper().strip()
    if not pos:
        return "?"
    if pos in ("PG", "SG", "G"):
        return "G"
    if pos in ("SF", "PF", "F"):
        return "F"
    if pos == "C":
        return "C"
    if "-" in pos:
        return pos.split("-")[0]
    return pos[:2] if len(pos) > 2 else pos
