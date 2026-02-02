from __future__ import annotations

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
            player = cells[0]
            status = cells[1] if len(cells) > 1 else "Unknown"
            injury = cells[2] if len(cells) > 2 else ""
            update = cells[3] if len(cells) > 3 else ""
            results.append({
                "team": team_name,
                "player": player,
                "status": status,
                "injury": injury,
                "update": update,
            })
    
    if results:
        log(f"ESPN: found {len(results)} injuries")
    else:
        log("ESPN: no injuries found (page structure may have changed)")
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


def fetch_injuries(
    timeout: int = 15,
    progress_cb: Optional[Callable[[str], None]] = None
) -> List[Dict[str, str]]:
    """
    Try multiple sources for injury data.
    Returns first successful result, or empty list if all fail.
    """
    log = progress_cb or (lambda _: None)
    
    # Try sources in order of reliability
    sources = [
        ("ESPN", _fetch_espn_injuries),
        ("CBS Sports", _fetch_cbs_injuries),
        ("RotoWire", _fetch_rotowire_injuries),
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
    
    log("All injury sources failed - injuries not updated")
    return []
