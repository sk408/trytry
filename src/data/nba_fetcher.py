from __future__ import annotations

import time
from datetime import date
from typing import Callable, Optional, List

import pandas as pd


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
        rows.extend(
            {
                "id": int(r.PLAYER_ID),
                "full_name": str(r.PLAYER),
                "team_id": int(tid),
                "position": str(r.POSITION or ""),
            }
            for r in roster_df.itertuples(index=False)
        )
        if sleep_between > 0:
            time.sleep(sleep_between)

    # If still empty, fallback to static players without team_id (will be dropped later)
    if not rows:
        progress("Roster fetch empty; falling back to static players list (no team mapping).")
        players = players_static.get_active_players() if active_only else players_static.get_players()
        return pd.DataFrame(players)[["id", "full_name"]].assign(team_id=None, position=None)

    progress(f"Rosters aggregated: {len(rows)} players")
    return pd.DataFrame(rows)[["id", "full_name", "team_id", "position"]]


def _fetch_team_roster(commonteamroster, team_id: int, season: str, progress: Callable[[str], None]):
    for attempt in range(3):
        try:
            timeout = 12 + attempt * 6  # backoff on timeout
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=timeout)
            df = roster.get_data_frames()[0]
            try:
                raw = roster.nba_response.get_response()  # type: ignore[attr-defined]
                progress(f"Team {team_id} roster bytes ~{len(str(raw).encode('utf-8'))}")
            except Exception:
                pass
            return df[["PLAYER_ID", "PLAYER", "POSITION"]]
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
    logs = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    
    # Extract all available stats
    cols = [
        "GAME_ID",
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
    ]
    
    # Only use columns that exist in the response
    available_cols = [c for c in cols if c in logs.columns]
    df = logs[available_cols].copy()
    
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    df["IS_HOME"] = ~df["MATCHUP"].str.contains("@")
    df["OPP_TEAM_ABBR"] = df["MATCHUP"].str.split().str[-1]
    
    # Rename columns to snake_case
    rename_map = {
        "GAME_ID": "game_id",
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
                    # Get game time
                    game_time_str = g.get("gameTimeEst", "") or g.get("gameDateTimeEst", "")
                    game_time = ""
                    if game_time_str:
                        try:
                            # Time is usually in format "7:30 pm ET" or embedded in datetime
                            if "T" in game_time_str:
                                from datetime import datetime
                                dt = datetime.fromisoformat(game_time_str.replace("Z", "+00:00"))
                                game_time = dt.strftime("%I:%M %p").lstrip("0")
                            else:
                                game_time = game_time_str
                        except (ValueError, TypeError):
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
