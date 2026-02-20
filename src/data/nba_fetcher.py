from __future__ import annotations

import threading
import time
from datetime import date
from typing import Callable, Optional, List

import pandas as pd


# ====================================================================
#  Centralized rate limiter for NBA API requests
# ====================================================================

class _RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Ensures at most ``calls_per_second`` requests are made to the NBA
    API.  All ``nba_api`` calls should go through ``limiter.wait()``
    before executing.
    """

    def __init__(self, calls_per_second: float = 1.0) -> None:
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until it is safe to make the next API call."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


# Global rate limiter instance — 1 request per 0.6s (~1.67 req/s).
# This is conservative enough to avoid 429 responses from stats.nba.com.
_api_limiter = _RateLimiter(calls_per_second=1.67)


def _utc_to_pacific(dt) -> str:
    """Convert datetime to Pacific time string (PST/PDT)."""
    from datetime import timezone as tz, timedelta
    
    # Ensure dt has timezone
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
    Determine the current NBA season string (e.g., '2025-26').
    NBA season runs Oct-June, so Jan-June is previous year's season.
    """
    today = date.today()
    year = today.year
    month = today.month
    # If we're in Jan-June, season started last year
    if month <= 6:
        return f"{year - 1}-{str(year)[2:]}"
    else:
        return f"{year}-{str(year + 1)[2:]}"


def _require_nba_api():
    try:
        from nba_api.stats.static import teams as teams_static  # type: ignore
        from nba_api.stats.static import players as players_static  # type: ignore
        from nba_api.stats.endpoints import (
            playergamelog,
            leaguegamefinder,
            commonplayerinfo,
            commonteamroster,
        )  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "nba_api is required for fetching NBA data. Install dependencies and ensure network access."
        ) from exc
    return teams_static, players_static, playergamelog, leaguegamefinder, commonplayerinfo, commonteamroster


def fetch_teams() -> pd.DataFrame:
    teams_static, _, _, _, _, _ = _require_nba_api()
    teams = teams_static.get_teams()
    df = pd.DataFrame(teams)
    # Older payloads may not include conference; make it optional
    for col in ["conference"]:
        if col not in df.columns:
            df[col] = None
    return df[["id", "full_name", "abbreviation", "conference"]]


def fetch_players(
    active_only: bool = True,
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    sleep_between: float = 0.5,
) -> pd.DataFrame:
    season = season or get_current_season()
    progress = progress_cb or (lambda _msg: None)
    teams_static, players_static, _, _, _, commonteamroster = _require_nba_api()
    teams = teams_static.get_teams()
    team_ids = [t["id"] for t in teams]

    rows: List[dict] = []
    for idx, tid in enumerate(team_ids, start=1):
        progress(f"Fetching roster {idx}/{len(team_ids)} for team {tid}")
        roster_df = _fetch_team_roster(commonteamroster, tid, season, progress)
        if roster_df is None or roster_df.empty:
            continue
        for r in roster_df.itertuples(index=False):
            row = {
                "id": int(r.PLAYER_ID),
                "full_name": str(r.PLAYER),
                "team_id": int(tid),
                "position": str(r.POSITION or ""),
                "height": str(getattr(r, "HEIGHT", "") or ""),
                "weight": str(getattr(r, "WEIGHT", "") or ""),
                "age": int(getattr(r, "AGE", 0) or 0) or None,
                "experience": None,
            }
            exp_val = getattr(r, "EXP", None)
            if exp_val is not None and str(exp_val).strip().upper() != "R":
                try:
                    row["experience"] = int(exp_val)
                except (ValueError, TypeError):
                    row["experience"] = 0
            elif exp_val is not None:
                row["experience"] = 0  # Rookie
            rows.append(row)
        if sleep_between > 0:
            time.sleep(sleep_between)

    # If still empty, fallback to static players without team_id (will be dropped later)
    if not rows:
        progress("Roster fetch empty; falling back to static players list (no team mapping).")
        players = players_static.get_active_players() if active_only else players_static.get_players()
        return pd.DataFrame(players)[["id", "full_name"]].assign(team_id=None, position=None)

    progress(f"Rosters aggregated: {len(rows)} players")
    return pd.DataFrame(rows)[["id", "full_name", "team_id", "position", "height", "weight", "age", "experience"]]


def _fetch_team_roster(commonteamroster, team_id: int, season: str, progress: Callable[[str], None]):
    for attempt in range(3):
        try:
            timeout = 12 + attempt * 6  # backoff on timeout
            _api_limiter.wait()
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=timeout)
            df = roster.get_data_frames()[0]
            try:
                raw = roster.nba_response.get_response()  # type: ignore[attr-defined]
                progress(f"Team {team_id} roster bytes ~{len(str(raw).encode('utf-8'))}")
            except Exception:
                pass
            # Return extended roster columns: include height, weight, age, experience
            keep_cols = ["PLAYER_ID", "PLAYER", "POSITION"]
            for col in ["HEIGHT", "WEIGHT", "AGE", "EXP"]:
                if col in df.columns:
                    keep_cols.append(col)
            return df[keep_cols]
        except Exception as exc:
            progress(f"Team {team_id} roster attempt {attempt+1}/3 failed: {exc}")
            # brief backoff before next attempt
            time.sleep(1.5 + attempt)
            if attempt == 2:
                progress(f"Team {team_id} roster skipped after 3 attempts")
                return None


def fetch_player_game_logs(player_id: int, season: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch comprehensive player game logs with all available stats.
    
    Returns DataFrame with:
    - Basic: points, rebounds, assists, minutes
    - Defensive: steals, blocks, turnovers
    - Shooting: fg_made/attempted, fg3_made/attempted, ft_made/attempted
    - Rebounds: oreb, dreb
    - Impact: plus_minus
    """
    season = season or get_current_season()
    _, _, playergamelog, _, _, _ = _require_nba_api()
    _api_limiter.wait()
    logs = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    
    # Extract all available stats
    cols = [
        "Game_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        # Basic stats
        "PTS",
        "REB",
        "AST",
        "MIN",
        # Defensive stats
        "STL",
        "BLK",
        "TOV",
        # Shooting stats
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        # Rebound breakdown
        "OREB",
        "DREB",
        # Impact
        "PLUS_MINUS",
        # Fouls
        "PF",
    ]
    
    # Only use columns that exist in the response
    available_cols = [c for c in cols if c in logs.columns]
    df = logs[available_cols].copy()
    
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    df["IS_HOME"] = ~df["MATCHUP"].str.contains("@")
    df["OPP_TEAM_ABBR"] = df["MATCHUP"].str.split().str[-1]
    
    # Rename columns to snake_case
    rename_map = {
        "Game_ID": "game_id",
        "PTS": "points",
        "REB": "rebounds",
        "AST": "assists",
        "MIN": "minutes",
        "STL": "steals",
        "BLK": "blocks",
        "TOV": "turnovers",
        "FGM": "fg_made",
        "FGA": "fg_attempted",
        "FG3M": "fg3_made",
        "FG3A": "fg3_attempted",
        "FTM": "ft_made",
        "FTA": "ft_attempted",
        "OREB": "oreb",
        "DREB": "dreb",
        "PLUS_MINUS": "plus_minus",
        "GAME_DATE": "game_date",
        "IS_HOME": "is_home",
        "OPP_TEAM_ABBR": "opponent_abbr",
        "WL": "win_loss",
        "PF": "personal_fouls",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Ensure all expected columns exist with defaults
    for col, default in [
        ("steals", 0), ("blocks", 0), ("turnovers", 0),
        ("fg_made", 0), ("fg_attempted", 0),
        ("fg3_made", 0), ("fg3_attempted", 0),
        ("ft_made", 0), ("ft_attempted", 0),
        ("oreb", 0), ("dreb", 0), ("plus_minus", 0),
        ("game_id", ""),
        ("win_loss", None), ("personal_fouls", 0),
    ]:
        if col not in df.columns:
            df[col] = default
    
    return df


def fetch_schedule(
    season: Optional[str] = None,
    team_ids: Optional[list[int]] = None,
    include_future_days: int = 14,
) -> pd.DataFrame:
    """
    Fetch schedule:
    - Past/played games via LeagueGameFinder (season to date)
    - Future scheduled games from NBA's static schedule JSON (full season)
    """
    import requests
    from datetime import timedelta

    season = season or get_current_season()
    teams_static, _, _, leaguegamefinder, _, _ = _require_nba_api()
    teams_list = teams_static.get_teams()
    teams_map = {int(t["id"]): t["abbreviation"] for t in teams_list}
    # Also create reverse lookup: tricode -> team_id
    tricode_to_id = {t["abbreviation"]: int(t["id"]) for t in teams_list}

    # Played games from LeagueGameFinder
    _api_limiter.wait()
    finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = finder.get_data_frames()[0]
    cols = ["TEAM_ID", "GAME_DATE", "MATCHUP"]
    if team_ids:
        games = games[games["TEAM_ID"].isin(team_ids)]
    df_played = games[cols].copy()
    df_played["GAME_DATE"] = pd.to_datetime(df_played["GAME_DATE"]).dt.date
    df_played["IS_HOME"] = ~df_played["MATCHUP"].str.contains("@")
    df_played["OPPONENT_ABBR"] = df_played["MATCHUP"].str.split().str[-1]
    df_played = df_played.rename(
        columns={
            "TEAM_ID": "team_id",
            "GAME_DATE": "game_date",
            "IS_HOME": "is_home",
            "OPPONENT_ABBR": "opponent_abbr",
        }
    )

    # Future games from NBA's static schedule JSON
    future_rows: List[dict] = []
    today = date.today()
    cutoff_date = today + timedelta(days=include_future_days)

    try:
        # Fetch full season schedule from NBA CDN
        schedule_url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        resp = requests.get(schedule_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Parse game dates - structure: leagueSchedule -> gameDates -> [date] -> games
        game_dates = data.get("leagueSchedule", {}).get("gameDates", [])
        for gd in game_dates:
            games_on_date = gd.get("games", [])
            for g in games_on_date:
                # Parse date from gameDateEst (format: "2025-02-01T00:00:00Z" or similar)
                date_str = g.get("gameDateEst", "")[:10]  # Get YYYY-MM-DD
                if not date_str:
                    continue
                try:
                    game_dt = date.fromisoformat(date_str)
                except ValueError:
                    continue

                # Only include future games (from today onwards, up to cutoff)
                if game_dt < today or game_dt > cutoff_date:
                    continue

                home_team = g.get("homeTeam", {})
                away_team = g.get("awayTeam", {})
                home_tricode = home_team.get("teamTricode", "")
                away_tricode = away_team.get("teamTricode", "")
                home_id = tricode_to_id.get(home_tricode)
                away_id = tricode_to_id.get(away_tricode)

                if home_id and away_id:
                    # Get game time - NBA CDN uses Eastern time ("Est" suffix)
                    game_time_str = g.get("gameDateTimeEst", "") or g.get("gameTimeEst", "")
                    game_time = ""
                    if game_time_str:
                        try:
                            from datetime import datetime, timedelta, timezone as tz
                            if "T" in game_time_str:
                                # Parse as Eastern time and convert to Pacific
                                dt_str = game_time_str.replace("Z", "")
                                dt = datetime.fromisoformat(dt_str)
                                
                                # Try zoneinfo first
                                try:
                                    from zoneinfo import ZoneInfo
                                    eastern = ZoneInfo("America/New_York")
                                    pacific = ZoneInfo("America/Los_Angeles")
                                    dt_eastern = dt.replace(tzinfo=eastern)
                                    dt_pacific = dt_eastern.astimezone(pacific)
                                    game_time = dt_pacific.strftime("%I:%M %p %Z").lstrip("0")
                                except (ImportError, KeyError):
                                    # Try pytz as fallback
                                    try:
                                        from pytz import timezone
                                        eastern = timezone("America/New_York")
                                        pacific = timezone("America/Los_Angeles")
                                        dt_eastern = eastern.localize(dt)
                                        dt_pacific = dt_eastern.astimezone(pacific)
                                        game_time = dt_pacific.strftime("%I:%M %p %Z").lstrip("0")
                                    except ImportError:
                                        # Manual fallback: Eastern is UTC-5 (EST) or UTC-4 (EDT)
                                        # Pacific is UTC-8 (PST) or UTC-7 (PDT)
                                        # Difference is always 3 hours
                                        month = dt.month
                                        if 3 <= month <= 11:
                                            tz_name = "PDT"
                                        else:
                                            tz_name = "PST"
                                        # Eastern to Pacific = subtract 3 hours
                                        dt_pacific = dt - timedelta(hours=3)
                                        game_time = dt_pacific.strftime(f"%I:%M %p {tz_name}").lstrip("0")
                            else:
                                game_time = game_time_str
                        except (ValueError, TypeError, Exception):
                            game_time = ""
                    
                    # Get team names
                    home_name = home_team.get("teamName", "") or home_team.get("teamCity", "")
                    away_name = away_team.get("teamName", "") or away_team.get("teamCity", "")
                    
                    # Single entry per game (not duplicated)
                    future_rows.append({
                        "game_id": g.get("gameId", ""),
                        "game_date": game_dt,
                        "home_team_id": home_id,
                        "away_team_id": away_id,
                        "home_abbr": home_tricode,
                        "away_abbr": away_tricode,
                        "home_name": f"{home_team.get('teamCity', '')} {home_name}".strip() or home_tricode,
                        "away_name": f"{away_team.get('teamCity', '')} {away_name}".strip() or away_tricode,
                        "game_time": game_time,
                        "arena": g.get("arenaName", ""),
                    })
        print(f"[Schedule] Fetched {len(future_rows)} future games from NBA CDN")
    except Exception as e:
        print(f"[Schedule] Failed to fetch future games from NBA CDN: {e}")
        # If schedule fetch fails, we still have played games

    # Convert future games to DataFrame
    if future_rows:
        df_future = pd.DataFrame(future_rows)
    else:
        df_future = pd.DataFrame(columns=[
            "game_id", "game_date", "home_team_id", "away_team_id", 
            "home_abbr", "away_abbr", "home_name", "away_name", "game_time", "arena"
        ])
    
    # For played games, we need to deduplicate and convert to single-game format
    # Group played games by unique matchup
    if not df_played.empty:
        # Get home team entries only (each game appears once from home perspective)
        df_home = df_played[df_played["is_home"] == True].copy()
        df_home["home_team_id"] = df_home["team_id"]
        df_home["home_abbr"] = df_home["team_id"].map(teams_map)
        df_home["away_abbr"] = df_home["opponent_abbr"]
        df_home["away_team_id"] = df_home["opponent_abbr"].map(tricode_to_id)
        df_home["home_name"] = df_home["home_abbr"]  # Use abbr as fallback
        df_home["away_name"] = df_home["away_abbr"]
        df_home["game_time"] = ""  # Not available for past games
        df_home["arena"] = ""
        df_home["game_id"] = ""
        
        df_played_clean = df_home[[
            "game_id", "game_date", "home_team_id", "away_team_id",
            "home_abbr", "away_abbr", "home_name", "away_name", "game_time", "arena"
        ]].dropna(subset=["home_team_id", "away_team_id"])
    else:
        df_played_clean = pd.DataFrame(columns=[
            "game_id", "game_date", "home_team_id", "away_team_id",
            "home_abbr", "away_abbr", "home_name", "away_name", "game_time", "arena"
        ])

    # Combine future and played games
    combined = pd.concat([df_future, df_played_clean], ignore_index=True)
    
    if team_ids and not combined.empty:
        combined = combined[
            (combined["home_team_id"].isin(team_ids)) | 
            (combined["away_team_id"].isin(team_ids))
        ]

    # Drop duplicates (by date + teams)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["game_date", "home_team_id", "away_team_id"])
        combined = combined.sort_values("game_date", ascending=False)
    
    return combined


# ============ NEW ADVANCED METRIC FETCHERS ============


def _safe_float(val, default=None):
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if pd.notna(f) else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    """Safely convert a value to int, returning default on failure."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def fetch_team_estimated_metrics(
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Fetch NBA's official estimated team metrics (off/def rating, pace, etc.).
    Uses TeamEstimatedMetrics endpoint.  One API call for all teams.
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _: None)
    try:
        from nba_api.stats.endpoints import teamestimatedmetrics
        progress("Fetching TeamEstimatedMetrics...")
        _api_limiter.wait()
        resp = teamestimatedmetrics.TeamEstimatedMetrics(
            season=season, timeout=30
        )
        df = resp.get_data_frames()[0]
        progress(f"  Got estimated metrics for {len(df)} teams")
        return df
    except Exception as exc:
        progress(f"TeamEstimatedMetrics failed: {exc}")
        return pd.DataFrame()


def fetch_league_dash_team_stats(
    season: Optional[str] = None,
    measure_type: str = "Base",
    location: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Fetch team stats from LeagueDashTeamStats with flexible measure type and filters.

    measure_type: 'Base', 'Advanced', 'Four Factors', 'Opponent'
    location: None (all), 'Home', 'Road'
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _: None)
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        label = f"LeagueDashTeamStats({measure_type}"
        if location:
            label += f", {location}"
        label += ")"
        progress(f"Fetching {label}...")

        kwargs = dict(
            season=season,
            measure_type_detailed_defense=measure_type,
            per_mode_detailed="PerGame",
            timeout=30,
        )
        if location:
            kwargs["location_nullable"] = location

        _api_limiter.wait()
        resp = leaguedashteamstats.LeagueDashTeamStats(**kwargs)
        df = resp.get_data_frames()[0]
        progress(f"  Got {label}: {len(df)} teams")
        return df
    except Exception as exc:
        progress(f"LeagueDashTeamStats({measure_type}) failed: {exc}")
        return pd.DataFrame()


def fetch_team_clutch_stats(
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Fetch clutch team stats (last 5 min, score within 5) with advanced metrics.
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _: None)
    try:
        from nba_api.stats.endpoints import leaguedashteamclutch
        progress("Fetching LeagueDashTeamClutch (Advanced)...")
        _api_limiter.wait()
        resp = leaguedashteamclutch.LeagueDashTeamClutch(
            season=season,
            measure_type_detailed_defense="Advanced",
            clutch_time="Last 5 Minutes",
            ahead_behind="Ahead or Behind",
            point_diff=5,
            per_mode_detailed="PerGame",
            timeout=30,
        )
        df = resp.get_data_frames()[0]
        progress(f"  Got clutch stats for {len(df)} teams")
        return df
    except Exception as exc:
        progress(f"LeagueDashTeamClutch failed: {exc}")
        return pd.DataFrame()


def fetch_team_hustle_stats(
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Fetch hustle stats (deflections, contested shots, loose balls, etc.).
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _: None)
    try:
        from nba_api.stats.endpoints import leaguehustlestatsteam
        progress("Fetching LeagueHustleStatsTeam...")
        _api_limiter.wait()
        resp = leaguehustlestatsteam.LeagueHustleStatsTeam(
            season=season, per_mode_time="PerGame", timeout=30,
        )
        df = resp.get_data_frames()[0]
        progress(f"  Got hustle stats for {len(df)} teams")
        return df
    except Exception as exc:
        progress(f"LeagueHustleStatsTeam failed: {exc}")
        return pd.DataFrame()


def fetch_player_on_off(
    team_id: int,
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch player on/off court impact for a specific team.
    Returns (on_court_df, off_court_df).
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _: None)
    try:
        from nba_api.stats.endpoints import teamplayeronoffsummary
        _api_limiter.wait()
        resp = teamplayeronoffsummary.TeamPlayerOnOffSummary(
            team_id=team_id, season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=30,
        )
        frames = resp.get_data_frames()
        on_court = frames[0] if len(frames) > 0 else pd.DataFrame()
        off_court = frames[1] if len(frames) > 1 else pd.DataFrame()
        return on_court, off_court
    except Exception as exc:
        progress(f"TeamPlayerOnOffSummary(team={team_id}) failed: {exc}")
        return pd.DataFrame(), pd.DataFrame()


def fetch_player_estimated_metrics(
    season: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Fetch NBA's official estimated player metrics (USG%, off/def rating, etc.).
    One API call for all players.
    """
    season = season or get_current_season()
    progress = progress_cb or (lambda _: None)
    try:
        from nba_api.stats.endpoints import playerestimatedmetrics
        progress("Fetching PlayerEstimatedMetrics...")
        _api_limiter.wait()
        resp = playerestimatedmetrics.PlayerEstimatedMetrics(
            season=season, timeout=30
        )
        df = resp.get_data_frames()[0]
        progress(f"  Got estimated metrics for {len(df)} players")
        return df
    except Exception as exc:
        progress(f"PlayerEstimatedMetrics failed: {exc}")
        return pd.DataFrame()
