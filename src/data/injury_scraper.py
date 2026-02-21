from __future__ import annotations

import re as _re
from typing import List, Dict, Optional, Callable

import requests
from bs4 import BeautifulSoup

# Multiple sources to try
ESPN_INJURY_URL = "https://www.espn.com/nba/injuries"
CBS_INJURY_URL = "https://www.cbssports.com/nba/injuries/"
ROTOWIRE_INJURY_URL = "https://www.rotowire.com/basketball/injury-report.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _fetch_espn_injuries(timeout: int = 15, progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, str]]:
    """Scrape ESPN injury page."""
    log = progress or (lambda _: None)
    log("Trying ESPN injuries...")
    
    try:
        resp = requests.get(ESPN_INJURY_URL, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        log(f"ESPN fetch failed: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, str]] = []

    # Try multiple CSS selector patterns (ESPN changes their layout)
    sections = soup.select("div.InjuriesPage > div.ResponsiveTable")
    if not sections:
        sections = soup.select("div.ResponsiveTable")
    if not sections:
        sections = soup.select("section.Card")
    
    for section in sections:
        team_header = section.select_one("div.Table__Title") or section.select_one("h2")
        team_name = team_header.get_text(strip=True) if team_header else "Unknown"
        rows = section.select("tbody tr")
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.select("td")]
            if len(cells) < 3:
                continue
            # ESPN format: NAME | POS | EST. RETURN DATE | STATUS | COMMENT
            player = cells[0]
            status = cells[3] if len(cells) > 3 else "Out"
            comment = cells[4] if len(cells) > 4 else ""
            # Try to extract injury body-part from parentheses in comment
            m = _re.search(r"\(([^)]+)\)", comment)
            injury = m.group(1) if m else ""
            results.append({
                "team": team_name,
                "player": player,
                "status": status,
                "injury": injury,
                "update": comment,
            })
    
    if results:
        log(f"ESPN: found {len(results)} injuries")
    else:
        log("ESPN: no injuries found (page structure may have changed)")
    return results


def _normalise_cbs_status(desc: str) -> str:
    """Map CBS Sports verbose injury-status text to a canonical label."""
    d = desc.strip().lower()
    if not d:
        return "Out"
    if "out" in d or "not expected" in d or "will miss" in d or "unlikely" in d:
        return "Out"
    if "doubtful" in d:
        return "Doubtful"
    if "questionable" in d:
        return "Questionable"
    if "probable" in d or "expected to play" in d or "likely to play" in d:
        return "Probable"
    if "day-to-day" in d or "day to day" in d:
        return "Day-To-Day"
    if "game time" in d or "game-time" in d:
        return "GTD"
    return "Out"          # safe default for unknown descriptions


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
            
            # CBS format: Player | Position | Updated | Injury | Injury Status
            player_cell = cells[0]
            # CBS wraps names in two spans: CellPlayerName--short ("J. Smith")
            # and CellPlayerName--long ("John Smith").  get_text() concatenates
            # both.  Prefer the --long span for the full name.
            long_span = player_cell.select_one("span.CellPlayerName--long")
            if long_span:
                player = long_span.get_text(strip=True)
            else:
                player = player_cell.get_text(strip=True)
            
            # Skip header rows
            if player.lower() in ["player", "name"]:
                continue
            
            # cells[1]=Position (skip), cells[2]=Updated date,
            # cells[3]=Injury body-part, cells[4]=Injury Status description
            injury = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            status_desc = cells[4].get_text(strip=True) if len(cells) > 4 else "Out"
            updated = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            status = _normalise_cbs_status(status_desc)
            
            if player:
                results.append({
                    "team": current_team,
                    "player": player,
                    "status": status,
                    "injury": injury,
                    "update": status_desc if status_desc != status else "",
                })
    
    if results:
        log(f"CBS: found {len(results)} injuries")
    else:
        log("CBS: no injuries found")
    return results


def _fetch_rotowire_injuries(timeout: int = 15, progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, str]]:
    """Scrape RotoWire injury page."""
    log = progress or (lambda _: None)
    log("Trying RotoWire injuries...")
    
    try:
        resp = requests.get(ROTOWIRE_INJURY_URL, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        log(f"RotoWire fetch failed: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, str]] = []

    # RotoWire has injury tables grouped by team
    tables = soup.select("div.injury-report table")
    if not tables:
        tables = soup.select("table")
    
    for table in tables:
        # Find team name - usually in preceding header or table caption
        team_header = table.find_previous(["h3", "h4", "caption", "div.team-name"])
        team_name = team_header.get_text(strip=True) if team_header else "Unknown"
        
        # Clean up team name
        if len(team_name) > 40:
            team_name = "Unknown"
        
        rows = table.select("tr")
        for row in rows:
            cells = row.select("td")
            if len(cells) < 2:
                continue
            
            player = cells[0].get_text(strip=True)
            
            # Skip headers
            if player.lower() in ["player", "name", ""]:
                continue
            
            # Format varies - try to extract status and injury
            status = ""
            injury = ""
            for i, cell in enumerate(cells[1:], 1):
                text = cell.get_text(strip=True)
                if text.lower() in ["out", "doubtful", "questionable", "probable", "day-to-day", "gtd"]:
                    status = text
                elif text and not status:
                    injury = text
                elif text and status and not injury:
                    injury = text
            
            if player:
                results.append({
                    "team": team_name,
                    "player": player,
                    "status": status or "Out",
                    "injury": injury,
                    "update": "",
                })
    
    if results:
        log(f"RotoWire: found {len(results)} injuries")
    else:
        log("RotoWire: no injuries found")
    return results


def _normalise_player_key(name: str, team: str) -> str:
    """Create a consistent lookup key from player name and team."""
    return f"{name.strip().lower()}|{team.strip().lower()}"


def fetch_injuries(
    timeout: int = 15,
    progress_cb: Optional[Callable[[str], None]] = None
) -> List[Dict[str, str]]:
    """
    Fetch injury data from **all** available sources and merge by player.

    Previously used a first-hit-wins strategy which missed injuries only
    reported by one source.  Now merges all sources, preferring the most
    severe status when a player appears in multiple sources (e.g. ESPN
    says "Questionable" but RotoWire says "Out" → keep "Out").
    """
    log = progress_cb or (lambda _: None)

    severity_order = {
        "out": 0,
        "doubtful": 1,
        "questionable": 2,
        "day-to-day": 3,
        "probable": 4,
        "gtd": 2,  # game-time decision ≈ questionable
    }

    sources = [
        ("ESPN", _fetch_espn_injuries),
        ("CBS Sports", _fetch_cbs_injuries),
        ("RotoWire", _fetch_rotowire_injuries),
    ]

    merged: Dict[str, Dict[str, str]] = {}  # key -> best entry

    for name, fetcher in sources:
        try:
            results = fetcher(timeout=timeout, progress=log)
            if not results:
                continue
            for entry in results:
                key = _normalise_player_key(entry["player"], entry.get("team", ""))
                existing = merged.get(key)
                if existing is None:
                    merged[key] = dict(entry)
                else:
                    # Keep the more severe status
                    new_sev = severity_order.get(entry["status"].strip().lower(), 5)
                    old_sev = severity_order.get(existing["status"].strip().lower(), 5)
                    if new_sev < old_sev:
                        merged[key] = dict(entry)
                    # Fill in missing injury detail from new source
                    if not existing.get("injury") and entry.get("injury"):
                        merged[key]["injury"] = entry["injury"]
        except Exception as exc:
            log(f"{name} error: {exc}")
            continue

    if merged:
        log(f"Merged {len(merged)} unique injuries from all sources")
        return list(merged.values())

    log("All injury sources failed - injuries not updated")
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
