"""College basketball data fetcher using ESPN API and CBBpy.

Replaces nba_fetcher.py for college basketball data.
Supports both men's and women's college basketball across all divisions.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Callable, Optional, List, Dict, Any

import pandas as pd
import requests

# ESPN API base URLs
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball"
MENS_COLLEGE = "mens-college-basketball"
WOMENS_COLLEGE = "womens-college-basketball"

# Default to men's college basketball
DEFAULT_LEAGUE = MENS_COLLEGE


def _utc_to_pacific(dt) -> str:
    """Convert datetime to Pacific time string (PST/PDT)."""
    from datetime import timezone as tz, timedelta
    
    # Ensure dt has UTC timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz.utc)
    
    # Try zoneinfo first (Python 3.9+)
    try:
        from zoneinfo import ZoneInfo
        pacific = ZoneInfo("America/Los_Angeles")
        return dt.astimezone(pacific).strftime("%I:%M %p %Z").lstrip("0")
    except (ImportError, KeyError):
        pass  # zoneinfo not available or tzdata not installed
    
    # Try pytz as fallback
    try:
        from pytz import timezone
        pacific = timezone("America/Los_Angeles")
        return dt.astimezone(pacific).strftime("%I:%M %p %Z").lstrip("0")
    except ImportError:
        pass  # pytz not installed
    
    # Manual fallback: PST is UTC-8, PDT is UTC-7
    # Use simple heuristic: PDT from March to November
    utc_dt = dt.astimezone(tz.utc)
    month = utc_dt.month
    if 3 <= month <= 11:
        # Approximate PDT (not exact DST boundaries but close enough)
        offset = timedelta(hours=-7)
        tz_name = "PDT"
    else:
        offset = timedelta(hours=-8)
        tz_name = "PST"
    
    pacific_dt = utc_dt + offset
    return pacific_dt.strftime(f"%I:%M %p {tz_name}").lstrip("0")


def get_current_season() -> str:
    """
    Determine the current college basketball season string (e.g., '2025-26').
    College season runs Nov-March, so Jan-June is previous year's season.
    """
    today = date.today()
    year = today.year
    month = today.month
    # If we're in Jan-June, season started last year
    if month <= 6:
        return f"{year - 1}-{str(year)[2:]}"
    else:
        return f"{year}-{str(year + 1)[2:]}"


def _espn_request(endpoint: str, params: Optional[Dict] = None, timeout: int = 15) -> Dict[str, Any]:
    """Make a request to ESPN API."""
    url = f"{ESPN_BASE}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[college_fetcher] ESPN API error: {e}")
        return {}


def fetch_teams(
    league: str = DEFAULT_LEAGUE,
    groups: Optional[str] = None,  # Conference group ID
    limit: int = 500,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Fetch college basketball teams from ESPN.
    
    Args:
        league: 'mens-college-basketball' or 'womens-college-basketball'
        groups: Optional conference group ID to filter
        limit: Max teams to fetch (default 500 for D1)
        progress_cb: Progress callback
    
    Returns:
        DataFrame with id, full_name, abbreviation, conference, division
    """
    progress = progress_cb or (lambda _: None)
    progress(f"Fetching {league} teams...")
    
    params = {"limit": limit}
    if groups:
        params["groups"] = groups
    
    data = _espn_request(f"{league}/teams", params)
    
    teams = []
    for team_data in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team = team_data.get("team", {})
        teams.append({
            "id": int(team.get("id", 0)),
            "full_name": team.get("displayName", ""),
            "abbreviation": team.get("abbreviation", ""),
            "conference": team.get("groups", {}).get("name", ""),
            "division": "D1",  # ESPN API primarily returns D1
            "gender": "mens" if "mens" in league else "womens",
        })
    
    progress(f"Found {len(teams)} teams")
    return pd.DataFrame(teams)


def fetch_scoreboard(
    league: str = DEFAULT_LEAGUE,
    dates: Optional[str] = None,  # YYYYMMDD format
    groups: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch today's games (scoreboard) from ESPN.
    
    Args:
        league: 'mens-college-basketball' or 'womens-college-basketball'
        dates: Date in YYYYMMDD format (defaults to today)
        groups: Conference group ID to filter
        limit: Max games to return
    
    Returns:
        List of game dictionaries
    """
    params = {"limit": limit}
    if dates:
        params["dates"] = dates
    if groups:
        params["groups"] = groups
    
    data = _espn_request(f"{league}/scoreboard", params)
    
    games = []
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
        
        games.append({
            "game_id": event.get("id", ""),
            "home_team_id": int(home_team.get("id", 0)),
            "away_team_id": int(away_team.get("id", 0)),
            "home_team_name": home_team.get("displayName", ""),
            "away_team_name": away_team.get("displayName", ""),
            "home_abbr": home_team.get("abbreviation", ""),
            "away_abbr": away_team.get("abbreviation", ""),
            "home_score": int(home.get("score", 0) or 0),
            "away_score": int(away.get("score", 0) or 0),
            "status": status_type.get("state", "pre"),  # pre, in, post
            "status_detail": status_type.get("shortDetail", ""),
            "period": int(status_obj.get("period", 0) or 0),
            "clock": status_obj.get("displayClock", ""),
            "start_time": event.get("date", ""),
            "venue": competition.get("venue", {}).get("fullName", ""),
            "neutral_site": competition.get("neutralSite", False),
        })
    
    return games


def fetch_team_roster(
    team_id: int,
    league: str = DEFAULT_LEAGUE,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Fetch roster for a specific team.
    
    Returns:
        DataFrame with player_id, name, position, jersey_number, class_year
    """
    progress = progress_cb or (lambda _: None)
    progress(f"Fetching roster for team {team_id}...")
    
    data = _espn_request(f"{league}/teams/{team_id}/roster")
    
    players = []
    for athlete in data.get("athletes", []):
        players.append({
            "id": int(athlete.get("id", 0)),
            "full_name": athlete.get("displayName", ""),
            "position": athlete.get("position", {}).get("abbreviation", ""),
            "jersey_number": athlete.get("jersey", ""),
            "class_year": athlete.get("experience", {}).get("displayValue", ""),
            "height": athlete.get("displayHeight", ""),
            "weight": athlete.get("displayWeight", ""),
            "team_id": team_id,
        })
    
    progress(f"Found {len(players)} players")
    return pd.DataFrame(players)


def fetch_players(
    team_ids: Optional[List[int]] = None,
    league: str = DEFAULT_LEAGUE,
    progress_cb: Optional[Callable[[str], None]] = None,
    sleep_between: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch players for multiple teams.
    
    Args:
        team_ids: List of team IDs to fetch rosters for. If None, fetches for scheduled games.
        league: League to fetch from
        progress_cb: Progress callback
        sleep_between: Delay between requests
    
    Returns:
        DataFrame with all players
    """
    progress = progress_cb or (lambda _: None)
    
    if team_ids is None:
        # Get teams from today's games
        games = fetch_scoreboard(league=league)
        team_ids = list(set(
            [g["home_team_id"] for g in games] + [g["away_team_id"] for g in games]
        ))
        progress(f"Found {len(team_ids)} teams from today's schedule")
    
    all_players = []
    for idx, tid in enumerate(team_ids, start=1):
        if idx % 10 == 0:
            progress(f"Fetching roster {idx}/{len(team_ids)}...")
        
        try:
            roster_df = fetch_team_roster(tid, league=league)
            if not roster_df.empty:
                all_players.append(roster_df)
        except Exception as e:
            progress(f"Failed to fetch roster for team {tid}: {e}")
        
        if sleep_between > 0:
            time.sleep(sleep_between)
    
    if not all_players:
        return pd.DataFrame(columns=["id", "full_name", "team_id", "position"])
    
    result = pd.concat(all_players, ignore_index=True)
    progress(f"Total players: {len(result)}")
    return result


def fetch_game_summary(
    game_id: str,
    league: str = DEFAULT_LEAGUE,
) -> Dict[str, Any]:
    """
    Fetch detailed game summary including box scores, odds, etc.
    
    Returns:
        Dictionary with game details
    """
    data = _espn_request(f"{league}/summary", params={"event": game_id})
    return data


def fetch_player_stats_from_game(
    game_id: str,
    league: str = DEFAULT_LEAGUE,
) -> pd.DataFrame:
    """
    Fetch player stats from a specific game's box score.
    
    Returns:
        DataFrame with player stats from the game
    """
    data = fetch_game_summary(game_id, league)
    
    stats = []
    boxscore = data.get("boxscore", {})
    
    # Get game date from header
    header = data.get("header", {})
    comp = header.get("competitions", [{}])[0]
    game_date_str = comp.get("date", "")[:10]  # YYYY-MM-DD
    try:
        game_date = date.fromisoformat(game_date_str) if game_date_str else date.today()
    except ValueError:
        game_date = date.today()
    
    # Determine home/away team IDs
    home_team_id = None
    away_team_id = None
    for c in comp.get("competitors", []):
        team = c.get("team", {})
        tid = int(team.get("id", 0))
        if c.get("homeAway") == "home":
            home_team_id = tid
        else:
            away_team_id = tid
    
    for team_data in boxscore.get("players", []):
        team_info = team_data.get("team", {})
        team_id = int(team_info.get("id", 0))
        is_home = (team_id == home_team_id)
        opponent_id = away_team_id if is_home else home_team_id
        
        for stat_category in team_data.get("statistics", []):
            labels = stat_category.get("labels", [])
            
            for athlete in stat_category.get("athletes", []):
                player_info = athlete.get("athlete", {})
                player_stats = athlete.get("stats", [])
                
                # Map labels to stats
                stat_dict = {}
                for i, label in enumerate(labels):
                    if i < len(player_stats):
                        stat_dict[label.lower()] = player_stats[i]
                
                # Parse minutes (format: "32:15" or just "32")
                minutes_str = stat_dict.get("min", "0")
                try:
                    if ":" in str(minutes_str):
                        mins, secs = minutes_str.split(":")
                        minutes = float(mins) + float(secs) / 60
                    else:
                        minutes = float(minutes_str)
                except (ValueError, TypeError):
                    minutes = 0.0
                
                stats.append({
                    "player_id": int(player_info.get("id", 0)),
                    "player_name": player_info.get("displayName", ""),
                    "team_id": team_id,
                    "opponent_team_id": opponent_id,
                    "is_home": is_home,
                    "game_date": game_date,
                    "game_id": game_id,
                    "points": float(stat_dict.get("pts", 0) or 0),
                    "rebounds": float(stat_dict.get("reb", 0) or 0),
                    "assists": float(stat_dict.get("ast", 0) or 0),
                    "minutes": minutes,
                    "fg": stat_dict.get("fg", "0-0"),
                    "fg3": stat_dict.get("3pt", "0-0"),
                    "ft": stat_dict.get("ft", "0-0"),
                    "steals": float(stat_dict.get("stl", 0) or 0),
                    "blocks": float(stat_dict.get("blk", 0) or 0),
                    "turnovers": float(stat_dict.get("to", 0) or 0),
                })
    
    return pd.DataFrame(stats)


def fetch_schedule(
    league: str = DEFAULT_LEAGUE,
    team_ids: Optional[List[int]] = None,
    include_future_days: int = 1,  # Day-ahead loading by default
    include_past_days: int = 0,
) -> pd.DataFrame:
    """
    Fetch schedule for upcoming games.
    
    Args:
        league: League to fetch from
        team_ids: Optional list of team IDs to filter
        include_future_days: Days ahead to look (default 1 for day-ahead loading)
        include_past_days: Days back to look for recent games
    
    Returns:
        DataFrame with scheduled games
    """
    today = date.today()
    all_games = []
    
    # Fetch for each day in range
    for day_offset in range(-include_past_days, include_future_days + 1):
        target_date = today + timedelta(days=day_offset)
        date_str = target_date.strftime("%Y%m%d")
        
        games = fetch_scoreboard(league=league, dates=date_str)
        
        for g in games:
            # Filter by team_ids if specified
            if team_ids:
                if g["home_team_id"] not in team_ids and g["away_team_id"] not in team_ids:
                    continue
            
            # Parse start time (ISO format from ESPN: "2026-02-03T00:00Z")
            start_time_str = g.get("start_time", "")
            game_time = ""
            if start_time_str:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    # Convert to Pacific time (PST/PDT)
                    game_time = _utc_to_pacific(dt)
                except (ValueError, TypeError):
                    game_time = ""
            
            # Add single entry per game (not duplicated for each team)
            all_games.append({
                "game_id": g["game_id"],
                "game_date": target_date,
                "home_team_id": g["home_team_id"],
                "away_team_id": g["away_team_id"],
                "home_abbr": g["home_abbr"],
                "away_abbr": g["away_abbr"],
                "home_name": g.get("home_team_name", g["home_abbr"]),
                "away_name": g.get("away_team_name", g["away_abbr"]),
                "game_time": game_time,
                "status": g["status"],
                "venue": g.get("venue", ""),
                "neutral_site": g.get("neutral_site", False),
            })
    
    df = pd.DataFrame(all_games)
    if not df.empty:
        # Deduplicate by game_id (each game should appear once)
        df = df.drop_duplicates(subset=["game_id"])
        df = df.sort_values("game_date", ascending=False)
    
    return df


def fetch_player_game_logs(
    player_id: int,
    season: Optional[str] = None,
    league: str = DEFAULT_LEAGUE,
) -> pd.DataFrame:
    """
    Fetch game logs for a specific player.
    
    Note: ESPN's public API has limited historical player data.
    For comprehensive stats, consider using CBBpy as a supplement.
    
    Returns:
        DataFrame with player game logs
    """
    # ESPN doesn't have a direct player game log endpoint like NBA API
    # This would need to be built by fetching completed games
    # For now, return empty DataFrame - sync_service will need to
    # aggregate stats from game box scores instead
    print(f"[college_fetcher] Player game logs not available via ESPN API. Use game box scores instead.")
    return pd.DataFrame(columns=[
        "game_date", "opponent_abbr", "is_home", "points", "rebounds", "assists", "minutes"
    ])


def fetch_conferences(league: str = DEFAULT_LEAGUE) -> pd.DataFrame:
    """
    Fetch list of conferences.
    
    Returns:
        DataFrame with conference info
    """
    # ESPN groups endpoint
    data = _espn_request(f"{league}/groups")
    
    conferences = []
    for group in data.get("groups", []):
        conferences.append({
            "id": int(group.get("id", 0)),
            "name": group.get("name", ""),
            "abbreviation": group.get("abbreviation", ""),
            "parent_id": group.get("parent", {}).get("id"),
        })
    
    return pd.DataFrame(conferences)


# CBBpy integration for detailed stats
def _try_cbbpy_boxscore(game_id: str) -> Optional[pd.DataFrame]:
    """
    Try to get detailed box score from CBBpy if available.
    
    Returns:
        DataFrame with detailed stats, or None if CBBpy not available
    """
    try:
        from CBBpy import CBBpy
        cbb = CBBpy()
        # CBBpy uses different game ID format - this may need adjustment
        return cbb.get_game_boxscore(game_id)
    except ImportError:
        return None
    except Exception as e:
        print(f"[college_fetcher] CBBpy error: {e}")
        return None
