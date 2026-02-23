import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Callable
from src.database import db

logger = logging.getLogger(__name__)

def fetch_espn_odds(date_str: str) -> list:
    """Fetch games and odds from ESPN for a specific date (YYYYMMDD)."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("events", [])
    except Exception as e:
        logger.error(f"Error fetching ESPN odds for {date_str}: {e}")
        return []

def _map_espn_abbrev(espn_abbr: str) -> str:
    """Map ESPN abbreviations to standard NBA API ones if they differ."""
    mapping = {
        "GS": "GSW",
        "NO": "NOP",
        "NY": "NOP", # Wait, NY is NYK, ESPN might use NY
        "WSH": "WAS",
        "UTAH": "UTA",
        "SA": "UTA" # SA is SAS
    }
    if espn_abbr == "NY": return "NYK"
    if espn_abbr == "SA": return "SAS"
    return mapping.get(espn_abbr, espn_abbr)

def sync_odds_for_date(game_date: str, callback: Optional[Callable] = None) -> int:
    """Fetch and store odds for all games on a date (YYYY-MM-DD)."""
    espn_date = game_date.replace("-", "")
    events = fetch_espn_odds(espn_date)
    if not events:
        return 0

    # Get team mapping
    rows = db.fetch_all("SELECT team_id, abbreviation FROM teams")
    abbrev_to_id = {r["abbreviation"]: r["team_id"] for r in rows}
    
    saved_count = 0
    now = datetime.now().isoformat()

    for event in events:
        try:
            comps = event.get("competitions", [])
            if not comps: continue
            comp = comps[0]
            
            odds_list = comp.get("odds", [])
            if not odds_list: continue
            
            # Just take the first provider (usually ESPN BET or Consensus)
            odds = odds_list[0]
            
            home_id = None
            away_id = None
            
            for competitor in comp.get("competitors", []):
                abbr = competitor.get("team", {}).get("abbreviation", "")
                norm_abbr = _map_espn_abbrev(abbr)
                tid = abbrev_to_id.get(norm_abbr)
                
                if competitor.get("homeAway") == "home":
                    home_id = tid
                else:
                    away_id = tid
            
            if not home_id or not away_id:
                continue

            # ESPN details usually like "BOS -5.5" -> this is the favorite and spread.
            # But they also have "spread" field? Wait, usually ESPN odds dictionary might not have 'spread'.
            # We can extract it from "details" string or 'spread' if it exists.
            
            spread = odds.get("spread")
            # If spread is not directly available but details is:
            details = odds.get("details", "")
            if spread is None and details and details != "EVEN":
                parts = details.split()
                if len(parts) == 2:
                    # parts[0] is abbreviation, parts[1] is spread
                    try:
                        fav_abbr = _map_espn_abbrev(parts[0])
                        val = float(parts[1])
                        # If home is favorite, spread is negative (home perspective)
                        if abbrev_to_id.get(fav_abbr) == home_id:
                            spread = val
                        else:
                            spread = -val
                    except Exception:
                        pass
            
            if spread is None and details == "EVEN":
                spread = 0.0
                
            ou = odds.get("overUnder")
            
            home_ml = odds.get("homeTeamOdds", {}).get("moneyLine")
            away_ml = odds.get("awayTeamOdds", {}).get("moneyLine")
            
            if spread is None and ou is None:
                continue
                
            db.execute("""
                INSERT INTO game_odds (game_date, home_team_id, away_team_id, spread, over_under, home_moneyline, away_moneyline, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_date, home_team_id, away_team_id) DO UPDATE SET
                    spread=excluded.spread,
                    over_under=excluded.over_under,
                    home_moneyline=excluded.home_moneyline,
                    away_moneyline=excluded.away_moneyline,
                    fetched_at=excluded.fetched_at
            """, (game_date, home_id, away_id, spread, ou, home_ml, away_ml, now))
            
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to parse odds for event {event.get('id')}: {e}")
            
    if callback and saved_count > 0:
        callback(f"Saved odds for {saved_count} games on {game_date}")
        
    return saved_count

def backfill_odds(callback: Optional[Callable] = None) -> int:
    """Backfill odds for all historical games that have player_stats but no odds."""
    # Find dates with games but no odds
    rows = db.fetch_all("""
        SELECT DISTINCT game_date 
        FROM player_stats 
        WHERE game_date NOT IN (SELECT DISTINCT game_date FROM game_odds)
        ORDER BY game_date DESC
    """)
    
    dates = [r["game_date"] for r in rows]
    total_dates = len(dates)
    total_saved = 0
    
    if callback:
        callback(f"Found {total_dates} dates needing odds backfill. Starting...")
        
    for i, game_date in enumerate(dates):
        if not game_date or len(game_date) != 10:
            continue
        try:
            saved = sync_odds_for_date(game_date)
            total_saved += saved
        except Exception as e:
            logger.error(f"Error backfilling odds for {game_date}: {e}")
            
        if callback and (i + 1) % 10 == 0:
            callback(f"Odds backfill progress: {i + 1}/{total_dates} dates processed.")
            
    if callback:
        callback(f"Odds backfill complete! Saved odds for {total_saved} games.")
        
    return total_saved
