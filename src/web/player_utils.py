"""Player utility functions for web views."""

from typing import Dict, Any, List, Optional

from src.database import db


def get_player_headshot_url(player_id: int) -> str:
    """Get player headshot URL (NBA CDN)."""
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"


def get_team_logo_url(team_id: int) -> str:
    """Get team logo URL (NBA CDN)."""
    return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"


def format_player_stats(player: Dict) -> Dict[str, Any]:
    """Format player stats for display."""
    return {
        "player_id": player.get("player_id"),
        "player_name": player.get("player_name", "Unknown"),
        "position": player.get("position", ""),
        "ppg": round(player.get("ppg", 0) or 0, 1),
        "rpg": round(player.get("rpg", 0) or 0, 1),
        "apg": round(player.get("apg", 0) or 0, 1),
        "mpg": round(player.get("mpg", 0) or 0, 1),
        "headshot": get_player_headshot_url(player.get("player_id", 0)),
    }


def get_team_roster_with_injuries(team_id: int) -> List[Dict]:
    """Get team roster merged with injury status."""
    players = db.fetch_all("""
        SELECT p.*, i.status as injury_status, i.reason as injury_reason,
               COALESCE(
                   (SELECT AVG(minutes) FROM (
                       SELECT minutes FROM player_stats
                       WHERE player_id = p.player_id
                       ORDER BY game_date DESC LIMIT 15
                   )), 0
               ) AS mpg
        FROM players p
        LEFT JOIN injuries i ON p.player_id = i.player_id
        WHERE p.team_id = ?
        ORDER BY mpg DESC
    """, (team_id,))
    return [format_player_stats(dict(p)) | {
        "injury_status": p.get("injury_status"),
        "injury_reason": p.get("injury_reason"),
    } for p in players]


def categorize_injuries(injuries: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize injuries by impact level."""
    key_players = []    # 25+ MPG
    rotation = []       # 15+ MPG
    bench = []          # rest

    for inj in injuries:
        mpg = inj.get("mpg", 0) or 0
        if mpg >= 25:
            key_players.append(inj)
        elif mpg >= 15:
            rotation.append(inj)
        else:
            bench.append(inj)

    return {"key": key_players, "rotation": rotation, "bench": bench}
