"""ESPN Gamecast integration for real-time NBA game data.

Uses the linedrive library for WebSocket play-by-play events and ESPN's
summary API for live odds and win probability.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional

import requests

# ESPN API endpoints
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


@dataclass
class GameInfo:
    """Basic game information."""
    game_id: str
    home_team: str
    away_team: str
    home_abbr: str
    away_abbr: str
    home_score: int = 0
    away_score: int = 0
    status: str = "scheduled"  # scheduled, in_progress, final
    period: int = 0
    clock: str = ""
    start_time: str = ""


@dataclass
class GameOdds:
    """Live betting odds from ESPN."""
    spread: float = 0.0
    spread_odds: str = ""
    over_under: float = 0.0
    over_odds: str = ""
    under_odds: str = ""
    home_ml: str = ""
    away_ml: str = ""
    home_win_pct: float = 0.0
    away_win_pct: float = 0.0
    home_ats_record: str = ""
    away_ats_record: str = ""


@dataclass
class PlayEvent:
    """A single play-by-play event."""
    event_id: str
    clock: str
    period: int
    text: str
    team: str = ""
    score_home: int = 0
    score_away: int = 0
    event_type: str = ""  # shot, foul, turnover, etc.
    timestamp: float = field(default_factory=time.time)


@dataclass 
class BoxScorePlayer:
    """Player stats in box score."""
    name: str
    position: str
    minutes: str
    points: int
    rebounds: int
    assists: int
    fg: str  # e.g. "5-10"
    fg3: str  # e.g. "2-4"
    ft: str  # e.g. "3-4"


@dataclass
class GameLeader:
    """A game leader in a statistical category."""
    name: str
    value: str
    team_abbr: str


@dataclass
class GameLeaders:
    """Game leaders for both teams."""
    home_points: Optional[GameLeader] = None
    home_rebounds: Optional[GameLeader] = None
    home_assists: Optional[GameLeader] = None
    away_points: Optional[GameLeader] = None
    away_rebounds: Optional[GameLeader] = None
    away_assists: Optional[GameLeader] = None


@dataclass
class BoxScore:
    """Live box score data."""
    home_players: List[BoxScorePlayer] = field(default_factory=list)
    away_players: List[BoxScorePlayer] = field(default_factory=list)
    home_totals: Dict[str, Any] = field(default_factory=dict)
    away_totals: Dict[str, Any] = field(default_factory=dict)


def get_live_games() -> List[GameInfo]:
    """Fetch today's NBA games from ESPN scoreboard."""
    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[gamecast] Error fetching scoreboard: {e}")
        return []

    games = []
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        
        if len(competitors) < 2:
            continue
            
        # ESPN lists home team first
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        
        home_team = home.get("team", {})
        away_team = away.get("team", {})
        
        status_obj = event.get("status", {})
        status_type = status_obj.get("type", {})
        status_state = status_type.get("state", "pre")  # pre, in, post
        
        status_map = {"pre": "scheduled", "in": "in_progress", "post": "final"}
        
        games.append(GameInfo(
            game_id=event.get("id", ""),
            home_team=home_team.get("displayName", "Home"),
            away_team=away_team.get("displayName", "Away"),
            home_abbr=home_team.get("abbreviation", "HOM"),
            away_abbr=away_team.get("abbreviation", "AWY"),
            home_score=int(home.get("score", 0) or 0),
            away_score=int(away.get("score", 0) or 0),
            status=status_map.get(status_state, "scheduled"),
            period=int(status_obj.get("period", 0) or 0),
            clock=status_obj.get("displayClock", ""),
            start_time=event.get("date", ""),
        ))
    
    return games


def get_game_odds(game_id: str) -> Optional[GameOdds]:
    """Fetch live odds and win probability from ESPN summary API."""
    try:
        resp = requests.get(
            ESPN_SUMMARY_URL,
            params={"event": game_id},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[gamecast] Error fetching odds for {game_id}: {e}")
        return None

    odds = GameOdds()
    
    # Parse pickcenter (betting odds)
    pickcenter = data.get("pickcenter", [])
    if pickcenter:
        pc = pickcenter[0]  # Usually first provider (DraftKings)
        
        # Spread - use numeric field directly
        odds.spread = float(pc.get("spread", 0) or 0)
        odds.over_under = float(pc.get("overUnder", 0) or 0)
        
        # Over/under odds
        over_odds = pc.get("overOdds")
        under_odds = pc.get("underOdds")
        odds.over_odds = f"{int(over_odds):+d}" if over_odds else ""
        odds.under_odds = f"{int(under_odds):+d}" if under_odds else ""
        
        # Money line and spread odds from team objects
        home_team_odds = pc.get("homeTeamOdds", {})
        away_team_odds = pc.get("awayTeamOdds", {})
        
        if home_team_odds:
            ml = home_team_odds.get("moneyLine")
            odds.home_ml = f"{int(ml):+d}" if ml else ""
            spread_odds = home_team_odds.get("spreadOdds")
            odds.spread_odds = f"{int(spread_odds):+d}" if spread_odds else ""
            # ATS record
            ats = home_team_odds.get("spreadRecord", {})
            if ats:
                odds.home_ats_record = f"{ats.get('wins', 0)}-{ats.get('losses', 0)}"
        
        if away_team_odds:
            ml = away_team_odds.get("moneyLine")
            odds.away_ml = f"{int(ml):+d}" if ml else ""
            # ATS record
            ats = away_team_odds.get("spreadRecord", {})
            if ats:
                odds.away_ats_record = f"{ats.get('wins', 0)}-{ats.get('losses', 0)}"
    
    # Parse win probability from winprobability array (NOT predictor)
    winprob = data.get("winprobability", [])
    if winprob:
        # Get the latest win probability (last entry in array)
        latest = winprob[-1]
        home_pct = float(latest.get("homeWinPercentage", 0) or 0)
        # Convert from decimal to percentage (0.06 -> 6.0)
        odds.home_win_pct = home_pct * 100
        odds.away_win_pct = (1 - home_pct) * 100
    
    return odds


def get_game_leaders(game_id: str) -> Optional[GameLeaders]:
    """Fetch game leaders (top scorers, rebounders, assisters) from ESPN."""
    try:
        resp = requests.get(
            ESPN_SUMMARY_URL,
            params={"event": game_id},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[gamecast] Error fetching leaders for {game_id}: {e}")
        return None

    leaders = GameLeaders()
    leaders_data = data.get("leaders", [])
    
    for team_data in leaders_data:
        team = team_data.get("team", {})
        team_abbr = team.get("abbreviation", "")
        
        # Determine if home or away by checking header
        header = data.get("header", {})
        comp = header.get("competitions", [{}])[0]
        is_home = False
        for c in comp.get("competitors", []):
            if c.get("team", {}).get("abbreviation") == team_abbr:
                is_home = c.get("homeAway") == "home"
                break
        
        # Parse each leader category
        for cat in team_data.get("leaders", []):
            cat_name = cat.get("name", "").lower()
            cat_leaders = cat.get("leaders", [])
            
            if cat_leaders:
                top = cat_leaders[0]
                athlete = top.get("athlete", {})
                leader = GameLeader(
                    name=athlete.get("displayName", "Unknown"),
                    value=str(top.get("displayValue", "0")),
                    team_abbr=team_abbr,
                )
                
                if cat_name == "points":
                    if is_home:
                        leaders.home_points = leader
                    else:
                        leaders.away_points = leader
                elif cat_name == "rebounds":
                    if is_home:
                        leaders.home_rebounds = leader
                    else:
                        leaders.away_rebounds = leader
                elif cat_name == "assists":
                    if is_home:
                        leaders.home_assists = leader
                    else:
                        leaders.away_assists = leader
    
    return leaders


def get_box_score(game_id: str) -> Optional[BoxScore]:
    """Fetch live box score from ESPN summary API."""
    try:
        resp = requests.get(
            ESPN_SUMMARY_URL,
            params={"event": game_id},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[gamecast] Error fetching box score for {game_id}: {e}")
        return None

    box = BoxScore()
    
    boxscore_data = data.get("boxscore", {})
    players_data = boxscore_data.get("players", [])
    
    for team_data in players_data:
        team_info = team_data.get("team", {})
        home_away = team_info.get("homeAway", "")
        
        stats_list = team_data.get("statistics", [])
        if not stats_list:
            continue
            
        # Find the main stats section
        stats = stats_list[0] if stats_list else {}
        athletes = stats.get("athletes", [])
        
        players = []
        for athlete in athletes:
            player_info = athlete.get("athlete", {})
            player_stats = athlete.get("stats", [])
            
            # ESPN returns stats in a specific order, but varies
            # Try to parse by label
            stat_dict = {}
            labels = stats.get("labels", [])
            for i, label in enumerate(labels):
                if i < len(player_stats):
                    stat_dict[label.lower()] = player_stats[i]
            
            players.append(BoxScorePlayer(
                name=player_info.get("displayName", "Unknown"),
                position=player_info.get("position", {}).get("abbreviation", ""),
                minutes=stat_dict.get("min", "0"),
                points=int(stat_dict.get("pts", 0) or 0),
                rebounds=int(stat_dict.get("reb", 0) or 0),
                assists=int(stat_dict.get("ast", 0) or 0),
                fg=stat_dict.get("fg", "0-0"),
                fg3=stat_dict.get("3pt", "0-0"),
                ft=stat_dict.get("ft", "0-0"),
            ))
        
        # Get totals
        totals = stats.get("totals", [])
        totals_dict = {}
        for i, label in enumerate(labels):
            if i < len(totals):
                totals_dict[label.lower()] = totals[i]
        
        if home_away == "home":
            box.home_players = players
            box.home_totals = totals_dict
        else:
            box.away_players = players
            box.away_totals = totals_dict
    
    return box


def get_play_by_play(game_id: str, last_event_id: str = "") -> List[PlayEvent]:
    """Fetch play-by-play events from ESPN summary API.
    
    Args:
        game_id: ESPN game ID
        last_event_id: Only return events after this ID (for incremental updates)
    
    Returns:
        List of play events, newest first
    """
    try:
        resp = requests.get(
            ESPN_SUMMARY_URL,
            params={"event": game_id},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[gamecast] Error fetching play-by-play for {game_id}: {e}")
        return []

    plays = []
    plays_data = data.get("plays", [])
    
    found_last = not last_event_id  # If no last_event_id, return all
    
    for play in reversed(plays_data):  # Newest first
        play_id = str(play.get("id", ""))
        
        if play_id == last_event_id:
            found_last = True
            continue
            
        if not found_last:
            continue
        
        # Parse team info
        team = play.get("team", {})
        team_abbr = team.get("abbreviation", "") if team else ""
        
        # Parse score
        score_value = play.get("scoreValue", 0)
        home_score = int(play.get("homeScore", 0) or 0)
        away_score = int(play.get("awayScore", 0) or 0)
        
        # Determine event type
        play_type = play.get("type", {})
        type_text = play_type.get("text", "") if play_type else ""
        
        plays.append(PlayEvent(
            event_id=play_id,
            clock=play.get("clock", {}).get("displayValue", ""),
            period=int(play.get("period", {}).get("number", 0) or 0),
            text=play.get("text", ""),
            team=team_abbr,
            score_home=home_score,
            score_away=away_score,
            event_type=type_text.lower() if type_text else "",
        ))
    
    return plays


class GamecastClient:
    """Client for streaming live game updates.
    
    Combines linedrive WebSocket events with periodic polling of the
    ESPN summary API for odds and box score updates.
    """
    
    def __init__(self, game_id: str):
        self.game_id = game_id
        self._stop_event = threading.Event()
        self._last_play_id = ""
        self._linedrive_available = False
        
        # Try to import linedrive
        try:
            import linedrive
            self._linedrive_available = True
        except ImportError:
            print("[gamecast] linedrive not available, using polling only")
    
    def stop(self) -> None:
        """Stop the streaming client."""
        self._stop_event.set()
    
    def stream_events(
        self,
        on_play: Optional[Callable[[PlayEvent], None]] = None,
        on_odds: Optional[Callable[[GameOdds], None]] = None,
        on_score: Optional[Callable[[GameInfo], None]] = None,
        poll_interval: float = 15.0,
        odds_interval: float = 30.0,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream game events.
        
        Yields dictionaries with event type and data:
        - {"type": "play", "data": PlayEvent}
        - {"type": "odds", "data": GameOdds}
        - {"type": "score", "data": GameInfo}
        - {"type": "error", "message": str}
        
        Args:
            on_play: Optional callback for play events
            on_odds: Optional callback for odds updates
            on_score: Optional callback for score updates
            poll_interval: Seconds between play-by-play polls
            odds_interval: Seconds between odds polls
        """
        last_odds_time = 0.0
        
        # Try linedrive first if available
        if self._linedrive_available:
            try:
                yield from self._stream_with_linedrive(
                    on_play, on_odds, on_score, odds_interval
                )
                return
            except Exception as e:
                yield {"type": "error", "message": f"WebSocket failed, falling back to polling: {e}"}
        
        # Fall back to polling
        yield from self._stream_with_polling(
            on_play, on_odds, on_score, poll_interval, odds_interval
        )
    
    def _stream_with_linedrive(
        self,
        on_play: Optional[Callable[[PlayEvent], None]],
        on_odds: Optional[Callable[[GameOdds], None]],
        on_score: Optional[Callable[[GameInfo], None]],
        odds_interval: float,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream using linedrive WebSocket."""
        import linedrive
        
        last_odds_time = 0.0
        
        # Note: linedrive API may vary - this is a best-effort implementation
        # based on the library's documented usage
        try:
            client = linedrive.Linedrive()
            
            for event in client.follow_event(self.game_id):
                if self._stop_event.is_set():
                    break
                
                # Parse linedrive event into our PlayEvent format
                play = PlayEvent(
                    event_id=str(event.get("id", time.time())),
                    clock=event.get("clock", ""),
                    period=int(event.get("period", 0)),
                    text=event.get("text", str(event)),
                    team=event.get("team", ""),
                    score_home=int(event.get("homeScore", 0)),
                    score_away=int(event.get("awayScore", 0)),
                    event_type=event.get("type", ""),
                )
                
                if on_play:
                    on_play(play)
                yield {"type": "play", "data": play}
                
                # Periodically fetch odds
                now = time.time()
                if now - last_odds_time >= odds_interval:
                    odds = get_game_odds(self.game_id)
                    if odds:
                        if on_odds:
                            on_odds(odds)
                        yield {"type": "odds", "data": odds}
                    last_odds_time = now
                    
        except Exception as e:
            raise RuntimeError(f"linedrive error: {e}")
    
    def _stream_with_polling(
        self,
        on_play: Optional[Callable[[PlayEvent], None]],
        on_odds: Optional[Callable[[GameOdds], None]],
        on_score: Optional[Callable[[GameInfo], None]],
        poll_interval: float,
        odds_interval: float,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream using polling fallback."""
        last_odds_time = 0.0
        
        while not self._stop_event.is_set():
            try:
                # Fetch new plays
                plays = get_play_by_play(self.game_id, self._last_play_id)
                
                for play in reversed(plays):  # Oldest first for chronological order
                    if on_play:
                        on_play(play)
                    yield {"type": "play", "data": play}
                    self._last_play_id = play.event_id
                
                # Update score from latest play
                if plays:
                    latest = plays[0]  # First is newest
                    # Get full game info
                    games = get_live_games()
                    game = next((g for g in games if g.game_id == self.game_id), None)
                    if game and on_score:
                        on_score(game)
                    if game:
                        yield {"type": "score", "data": game}
                
                # Periodically fetch odds
                now = time.time()
                if now - last_odds_time >= odds_interval:
                    odds = get_game_odds(self.game_id)
                    if odds:
                        if on_odds:
                            on_odds(odds)
                        yield {"type": "odds", "data": odds}
                    last_odds_time = now
                
            except Exception as e:
                yield {"type": "error", "message": str(e)}
            
            # Wait before next poll
            self._stop_event.wait(poll_interval)


def stream_game(
    game_id: str,
    poll_interval: float = 15.0,
    odds_interval: float = 30.0,
) -> Generator[Dict[str, Any], None, None]:
    """Convenience function to stream game events.
    
    Args:
        game_id: ESPN game ID
        poll_interval: Seconds between polls
        odds_interval: Seconds between odds updates
    
    Yields:
        Event dictionaries with type and data
    """
    client = GamecastClient(game_id)
    yield from client.stream_events(
        poll_interval=poll_interval,
        odds_interval=odds_interval,
    )
