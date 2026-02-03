"""Live scores fetcher for college basketball using ESPN API.

Replaces nba_api.live with ESPN's scoreboard endpoint.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional

import requests

# ESPN API endpoint
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball"
DEFAULT_LEAGUE = "mens-college-basketball"


def fetch_live_games(
    game_date: Optional[str] = None,
    league: str = DEFAULT_LEAGUE,
) -> List[Dict]:
    """
    Fetch live (and recent) games. game_date format YYYYMMDD; defaults to today.
    Returns list of dicts with team ids, scores, status, period, clock.
    
    Args:
        game_date: Date in YYYYMMDD format (optional)
        league: 'mens-college-basketball' or 'womens-college-basketball'
    
    Returns:
        List of game dictionaries
    """
    url = f"{ESPN_BASE}/{league}/scoreboard"
    params = {}
    if game_date:
        # Convert from YYYY-MM-DD to YYYYMMDD if needed
        params["dates"] = game_date.replace("-", "")
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[live_scores] Error fetching scoreboard: {e}")
        return []
    
    now_iso = datetime.utcnow().isoformat()
    parsed: List[Dict] = []
    
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        
        if len(competitors) < 2:
            continue
        
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        
        home_team = home.get("team", {})
        away_team = away.get("team", {})
        
        status_obj = event.get("status", {})
        status_type = status_obj.get("type", {})
        
        # Map status states
        state = status_type.get("state", "pre")
        status_map = {
            "pre": "Scheduled",
            "in": "In Progress",
            "post": "Final",
        }
        status_text = status_type.get("shortDetail", status_map.get(state, state))
        
        parsed.append({
            "game_id": event.get("id", ""),
            "home_team_id": int(home_team.get("id", 0)),
            "away_team_id": int(away_team.get("id", 0)),
            "home_abbr": home_team.get("abbreviation", ""),
            "away_abbr": away_team.get("abbreviation", ""),
            "start_time_utc": event.get("date", ""),
            "status": status_text,
            "period": int(status_obj.get("period", 0) or 0),
            "clock": status_obj.get("displayClock", ""),
            "home_score": int(home.get("score", 0) or 0),
            "away_score": int(away.get("score", 0) or 0),
            "last_updated": now_iso,
            "neutral_site": competition.get("neutralSite", False),
            "venue": competition.get("venue", {}).get("fullName", ""),
        })
    
    return parsed


def fetch_game_details(game_id: str, league: str = DEFAULT_LEAGUE) -> Optional[Dict]:
    """
    Fetch detailed information for a specific game.
    
    Args:
        game_id: ESPN game ID
        league: League identifier
    
    Returns:
        Dictionary with game details including odds, box score preview
    """
    url = f"{ESPN_BASE}/{league}/summary"
    
    try:
        resp = requests.get(url, params={"event": game_id}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[live_scores] Error fetching game details for {game_id}: {e}")
        return None
