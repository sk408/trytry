"""Live NBA scores via nba_api ScoreBoard."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def fetch_live_scores() -> List[Dict[str, Any]]:
    """Fetch current live scoreboard from nba_api."""
    try:
        from nba_api.live.nba.endpoints import ScoreBoard
        sb = ScoreBoard()
        data = sb.get_dict()
        games = []
        for game in data.get("scoreboard", {}).get("games", []):
            games.append({
                "game_id": game.get("gameId", ""),
                "status": game.get("gameStatusText", ""),
                "period": game.get("period", 0),
                "clock": game.get("gameClock", ""),
                "home_team_id": game.get("homeTeam", {}).get("teamId", 0),
                "away_team_id": game.get("awayTeam", {}).get("teamId", 0),
                "home_team": game.get("homeTeam", {}).get("teamTricode", ""),
                "away_team": game.get("awayTeam", {}).get("teamTricode", ""),
                "home_score": game.get("homeTeam", {}).get("score", 0),
                "away_score": game.get("awayTeam", {}).get("score", 0),
                "start_time": game.get("gameTimeUTC", ""),
            })
        return games
    except Exception as e:
        logger.error(f"Error fetching live scores: {e}")
        return []


def save_live_games(games: List[Dict[str, Any]]):
    """Upsert live game data to the database."""
    from src.database import db
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    for g in games:
        db.execute(
            """INSERT INTO live_games (game_id, home_team_id, away_team_id,
                start_time_utc, status, period, clock, home_score, away_score, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(game_id) DO UPDATE SET
                 status=excluded.status, period=excluded.period,
                 clock=excluded.clock, home_score=excluded.home_score,
                 away_score=excluded.away_score, last_updated=excluded.last_updated""",
            (g["game_id"], g.get("home_team_id", 0), g.get("away_team_id", 0),
             g.get("start_time", ""), g.get("status", ""),
             g.get("period", 0), g.get("clock", ""),
             g.get("home_score", 0), g.get("away_score", 0), now)
        )
