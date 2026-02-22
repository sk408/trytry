"""ESPN integration: odds, play-by-play, box score, predictor, WebSocket."""

import logging
import json
from typing import Dict, Any, Optional, List

import requests

logger = logging.getLogger(__name__)

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
}

# ESPN uses shorter/different abbreviations than the NBA standard stored in our DB
ESPN_TO_NBA_ABBR: Dict[str, str] = {
    "GS":   "GSW",
    "SA":   "SAS",
    "NY":   "NYK",
    "NO":   "NOP",
    "WSH":  "WAS",
    "UTAH": "UTA",
    "PHO":  "PHX",
    "BK":   "BKN",
}

def normalize_espn_abbr(abbr: str) -> str:
    """Translate an ESPN abbreviation to the NBA/DB standard."""
    return ESPN_TO_NBA_ABBR.get(abbr.upper(), abbr)


def fetch_espn_scoreboard() -> List[Dict[str, Any]]:
    """Fetch today's ESPN scoreboard."""
    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        games = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})
            games.append({
                "espn_id": event.get("id", ""),
                "name": event.get("name", ""),
                "status": event.get("status", {}).get("type", {}).get("description", ""),
                "short_detail": event.get("status", {}).get("type", {}).get("shortDetail", ""),
                "period": event.get("status", {}).get("period", 0),
                "clock": event.get("status", {}).get("displayClock", ""),
                "state": event.get("status", {}).get("type", {}).get("state", ""),
                "home_team": normalize_espn_abbr(home.get("team", {}).get("abbreviation", "")),
                "away_team": normalize_espn_abbr(away.get("team", {}).get("abbreviation", "")),
                "home_score": int(home.get("score", 0) or 0),
                "away_score": int(away.get("score", 0) or 0),
            })
        return games
    except Exception as e:
        logger.error(f"ESPN scoreboard error: {e}")
        return []


def fetch_espn_game_summary(game_id: str) -> Dict[str, Any]:
    """Fetch full game summary from ESPN."""
    try:
        resp = requests.get(
            ESPN_SUMMARY_URL,
            params={"event": game_id},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"ESPN summary error for {game_id}: {e}")
        return {}


def get_espn_odds(game_id: str) -> Dict[str, Any]:
    """Extract odds from ESPN game summary (pickcenter section)."""
    summary = fetch_espn_game_summary(game_id)
    pickcenter = summary.get("pickcenter", [])
    if not pickcenter:
        return {}
    odds_data = pickcenter[0] if pickcenter else {}
    return {
        "spread": odds_data.get("details", ""),
        "over_under": odds_data.get("overUnder", None),
        "home_moneyline": odds_data.get("homeTeamOdds", {}).get("moneyLine", None),
        "away_moneyline": odds_data.get("awayTeamOdds", {}).get("moneyLine", None),
        "provider": odds_data.get("provider", {}).get("name", ""),
    }


def get_espn_predictor(game_id: str) -> Dict[str, float]:
    """Extract ESPN predictor win probabilities."""
    summary = fetch_espn_game_summary(game_id)
    predictor = summary.get("predictor", {})
    game_proj = predictor.get("gameProjection", {}) if predictor else {}
    home_pct = float(predictor.get("homeTeam", {}).get("gameProjection", 50.0))
    away_pct = float(predictor.get("awayTeam", {}).get("gameProjection", 50.0))
    return {"home_win_pct": home_pct, "away_win_pct": away_pct}


def get_espn_win_probability(game_id: str) -> List[Dict[str, Any]]:
    """Extract live win probability data."""
    summary = fetch_espn_game_summary(game_id)
    return summary.get("winprobability", [])


def get_espn_plays(game_id: str) -> List[Dict[str, Any]]:
    """Extract play-by-play from summary."""
    summary = fetch_espn_game_summary(game_id)
    return summary.get("plays", [])


def get_espn_boxscore(game_id: str) -> Dict[str, Any]:
    """Extract box score from summary."""
    summary = fetch_espn_game_summary(game_id)
    return summary.get("boxscore", {})


def get_espn_linescores(game_id: str) -> List[Dict[str, Any]]:
    """Extract quarter-by-quarter line scores."""
    summary = fetch_espn_game_summary(game_id)
    header = summary.get("header", {})
    competitions = header.get("competitions", [{}])
    if not competitions:
        return []
    competitors = competitions[0].get("competitors", [])
    scores = []
    for c in competitors:
        team = c.get("team", {})
        linescores = c.get("linescores", [])
        scores.append({
            "team": team.get("abbreviation", ""),
            "team_id": team.get("id", ""),
            "is_home": c.get("homeAway") == "home",
            "quarters": [int(q.get("displayValue", 0) or 0) for q in linescores],
            "score": int(c.get("score", 0) or 0),
        })
    return scores


class ESPNWebSocket:
    """Optional WebSocket connection for live ESPN data via linedrive."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.ws = None
        self._running = False

    def connect(self):
        try:
            import websocket
            self.ws = websocket.WebSocketApp(
                f"wss://linedrive.espn.com/v1/nba/game/{self.game_id}",
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._running = True
        except ImportError:
            logger.warning("websocket-client not available for ESPN WebSocket")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            logger.debug(f"ESPN WS message: {data.get('type', 'unknown')}")
        except json.JSONDecodeError:
            pass

    def _on_error(self, ws, error):
        logger.error(f"ESPN WS error: {error}")

    def _on_close(self, ws, code, reason):
        self._running = False

    def close(self):
        self._running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
