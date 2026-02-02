from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional


def _require_live():
    try:
        from nba_api.live.nba.endpoints import scoreboard  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive for missing dep/network
        raise RuntimeError(
            "nba_api live module is required. Ensure dependencies are installed and network is available."
        ) from exc
    return scoreboard


def fetch_live_games(game_date: Optional[str] = None) -> List[Dict]:
    """
    Fetch live (and recent) games. game_date format YYYY-MM-DD; defaults to today.
    Returns list of dicts with team ids, scores, status, period, clock.
    """
    scoreboard = _require_live()
    params = {"gameDate": game_date} if game_date else {}
    board = scoreboard.ScoreBoard(**params)
    data = board.get_dict()
    games = data.get("scoreboard", {}).get("games", [])
    now_iso = datetime.utcnow().isoformat()

    parsed: List[Dict] = []
    for g in games:
        parsed.append(
            {
                "game_id": g.get("gameId"),
                "home_team_id": int(g["homeTeam"]["teamId"]),
                "away_team_id": int(g["awayTeam"]["teamId"]),
                "start_time_utc": g.get("gameTimeUTC"),
                "status": g.get("gameStatusText"),
                "period": g.get("period"),
                "clock": g.get("gameClock"),
                "home_score": int(g["homeTeam"].get("score", 0)),
                "away_score": int(g["awayTeam"].get("score", 0)),
                "last_updated": now_iso,
            }
        )
    return parsed
