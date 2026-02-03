from __future__ import annotations

from typing import List, Dict, Optional, Callable

import requests
from bs4 import BeautifulSoup

# Primary source for college basketball injuries
ACTION_NETWORK_URL = "https://www.actionnetwork.com/ncaab/injury-report"

# Backup sources (less reliable for college)
CBS_INJURY_URL = "https://www.cbssports.com/college-basketball/injuries/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _fetch_action_network_injuries(timeout: int = 15, progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, str]]:
    """Scrape Action Network college basketball injury page."""
    log = progress or (lambda _: None)
    log("Trying Action Network injuries...")
    
    try:
        resp = requests.get(ACTION_NETWORK_URL, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        log(f"Action Network fetch failed: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, str]] = []

    # Action Network uses a table format with team headers
    # Look for the injury table
    table = soup.select_one("table")
    if not table:
        log("Action Network: no injury table found")
        return []
    
    current_team = "Unknown"
    rows = table.select("tr")
    
    for row in rows:
        # Check if this is a team header row
        team_cell = row.select_one("td[colspan]") or row.select_one("th[colspan]")
        if team_cell:
            team_text = team_cell.get_text(strip=True)
            if team_text and len(team_text) < 60:
                current_team = team_text
            continue
        
        cells = row.select("td")
        if len(cells) < 3:
            continue
        
        # Action Network format: Name, Pos, Status, Injury, Update, Updated
        player = cells[0].get_text(strip=True) if len(cells) > 0 else ""
        position = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        status = cells[2].get_text(strip=True) if len(cells) > 2 else "Out"
        injury = cells[3].get_text(strip=True) if len(cells) > 3 else ""
        update = cells[4].get_text(strip=True) if len(cells) > 4 else ""
        
        # Skip header rows
        if player.lower() in ["name", "player", ""]:
            continue
        
        if player:
            results.append({
                "team": current_team,
                "player": player,
                "position": position,
                "status": status,
                "injury": injury,
                "update": update,
            })
    
    if results:
        log(f"Action Network: found {len(results)} injuries")
    else:
        log("Action Network: no injuries found")
    return results


def _fetch_cbs_injuries(timeout: int = 15, progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, str]]:
    """Scrape CBS Sports injury page."""
    log = progress or (lambda _: None)
    log("Trying CBS Sports injuries...")
    
    try:
        resp = requests.get(CBS_INJURY_URL, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        log(f"CBS fetch failed: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, str]] = []

    # CBS uses table rows with team sections
    tables = soup.select("table.TableBase-table")
    if not tables:
        tables = soup.select("table")
    
    current_team = "Unknown"
    for table in tables:
        # Try to find team name in preceding header
        prev = table.find_previous(["h3", "h4", "div"])
        if prev:
            team_text = prev.get_text(strip=True)
            if team_text and len(team_text) < 50:
                current_team = team_text
        
        rows = table.select("tbody tr")
        for row in rows:
            cells = row.select("td")
            if len(cells) < 2:
                continue
            
            # CBS format: Player, Position, Injury, Status
            player_cell = cells[0]
            player = player_cell.get_text(strip=True)
            
            # Skip header rows
            if player.lower() in ["player", "name"]:
                continue
            
            injury = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            status = cells[3].get_text(strip=True) if len(cells) > 3 else "Out"
            
            if player:
                results.append({
                    "team": current_team,
                    "player": player,
                    "status": status,
                    "injury": injury,
                    "update": "",
                })
    
    if results:
        log(f"CBS: found {len(results)} injuries")
    else:
        log("CBS: no injuries found")
    return results




def fetch_injuries(
    timeout: int = 15,
    progress_cb: Optional[Callable[[str], None]] = None
) -> List[Dict[str, str]]:
    """
    Try multiple sources for injury data.
    Returns first successful result, or empty list if all fail.
    
    Note: College basketball injury data is limited compared to NBA.
    Action Network is the primary free source with actual data.
    """
    log = progress_cb or (lambda _: None)
    
    # Try sources in order of reliability
    sources = [
        ("Action Network", _fetch_action_network_injuries),
        ("CBS Sports", _fetch_cbs_injuries),
    ]
    
    for name, fetcher in sources:
        try:
            results = fetcher(timeout=timeout, progress=log)
            if results:
                log(f"Successfully loaded {len(results)} injuries from {name}")
                return results
        except Exception as exc:
            log(f"{name} error: {exc}")
            continue
    
    log("No injuries found - college basketball injury data is limited")
    return []


# ============================================================================
# Manual Injury Management
# ============================================================================

import json
import os
from pathlib import Path

# Store manual injuries in a JSON file in the data directory
def _get_manual_injuries_path() -> Path:
    """Get path to manual injuries JSON file."""
    # Look for data directory relative to this file
    src_dir = Path(__file__).parent.parent.parent
    data_dir = src_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "manual_injuries.json"


def load_manual_injuries() -> List[Dict[str, str]]:
    """Load manually added injuries from JSON file."""
    path = _get_manual_injuries_path()
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("injuries", [])
    except (json.JSONDecodeError, IOError):
        return []


def save_manual_injury(
    player: str,
    team: str,
    status: str = "Out",
    injury: str = "",
    position: str = "",
) -> bool:
    """
    Add a manual injury entry.
    
    Args:
        player: Player name
        team: Team name
        status: Injury status (Out, Doubtful, Questionable, Probable)
        injury: Injury description
        position: Player position
    
    Returns:
        True if saved successfully
    """
    path = _get_manual_injuries_path()
    
    # Load existing
    injuries = load_manual_injuries()
    
    # Check if player already exists, update if so
    updated = False
    for inj in injuries:
        if inj.get("player", "").lower() == player.lower() and \
           inj.get("team", "").lower() == team.lower():
            inj["status"] = status
            inj["injury"] = injury
            inj["position"] = position
            inj["manual"] = True
            updated = True
            break
    
    if not updated:
        injuries.append({
            "player": player,
            "team": team,
            "status": status,
            "injury": injury,
            "position": position,
            "update": "Manual entry",
            "manual": True,
        })
    
    try:
        with open(path, "w") as f:
            json.dump({"injuries": injuries}, f, indent=2)
        return True
    except IOError:
        return False


def remove_manual_injury(player: str, team: str) -> bool:
    """
    Remove a manual injury entry.
    
    Args:
        player: Player name
        team: Team name
    
    Returns:
        True if removed successfully
    """
    path = _get_manual_injuries_path()
    injuries = load_manual_injuries()
    
    original_len = len(injuries)
    injuries = [
        inj for inj in injuries
        if not (inj.get("player", "").lower() == player.lower() and 
                inj.get("team", "").lower() == team.lower())
    ]
    
    if len(injuries) == original_len:
        return False  # Not found
    
    try:
        with open(path, "w") as f:
            json.dump({"injuries": injuries}, f, indent=2)
        return True
    except IOError:
        return False


def get_all_injuries(
    timeout: int = 15,
    progress_cb: Optional[Callable[[str], None]] = None
) -> List[Dict[str, str]]:
    """
    Get all injuries: scraped + manual entries.
    Manual entries take precedence for same player/team.
    """
    log = progress_cb or (lambda _: None)
    
    # Get scraped injuries
    scraped = fetch_injuries(timeout=timeout, progress_cb=log)
    
    # Get manual injuries
    manual = load_manual_injuries()
    
    if manual:
        log(f"Adding {len(manual)} manual injury entries")
    
    # Combine: manual takes precedence
    manual_keys = {
        (inj.get("player", "").lower(), inj.get("team", "").lower())
        for inj in manual
    }
    
    # Filter out scraped entries that have manual overrides
    filtered_scraped = [
        inj for inj in scraped
        if (inj.get("player", "").lower(), inj.get("team", "").lower()) not in manual_keys
    ]
    
    return manual + filtered_scraped
