from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.database.db import get_conn
from src.analytics.injury_history import get_injuries_for_game


@dataclass
class GamePrediction:
    game_date: date
    home_team_id: int
    away_team_id: int
    home_abbr: str
    away_abbr: str
    # Predictions
    predicted_spread: float  # positive = home favored
    predicted_total: float
    predicted_home_score: float
    predicted_away_score: float
    # Actual results
    actual_home_score: float
    actual_away_score: float
    actual_spread: float  # home - away
    actual_total: float
    # Analysis
    spread_error: float  # predicted - actual (how far off)
    total_error: float  # predicted - actual
    home_score_error: float  # how close home prediction was
    away_score_error: float  # how close away prediction was
    predicted_winner: str  # "HOME" or "AWAY" or "PUSH"
    actual_winner: str
    winner_correct: bool  # did we pick the right team to win
    spread_within_5: bool  # spread prediction within 5 points
    total_within_10: bool  # total prediction within 10 points
    # Injury info
    home_injuries: List[str] = field(default_factory=list)  # Names of injured players
    away_injuries: List[str] = field(default_factory=list)
    home_injury_adj: float = 0.0  # Points adjustment from injuries
    away_injury_adj: float = 0.0


@dataclass
class TeamAccuracy:
    team_id: int
    team_abbr: str
    team_name: str
    games_analyzed: int = 0
    spread_correct: int = 0
    spread_accuracy: float = 0.0
    avg_spread_error: float = 0.0
    total_correct: int = 0
    total_accuracy: float = 0.0
    avg_total_error: float = 0.0
    # Record tracking
    wins: int = 0
    losses: int = 0


@dataclass
class FactorBreakdown:
    """Per-factor contribution tracking for a single prediction."""
    base_spread: float = 0.0
    home_court: float = 0.0
    turnover_adj: float = 0.0
    ts_adj: float = 0.0
    rebound_adj: float = 0.0
    assist_adj: float = 0.0
    net_rating_adj: float = 0.0
    sos_adj: float = 0.0
    injury_adj: float = 0.0
    pace_adj: float = 0.0      # how much pace scaled the total


@dataclass 
class BacktestResults:
    predictions: List[GamePrediction] = field(default_factory=list)
    team_accuracy: Dict[int, TeamAccuracy] = field(default_factory=dict)
    overall_spread_accuracy: float = 0.0
    overall_total_accuracy: float = 0.0
    total_games: int = 0
    # Per-factor average absolute contribution to spread error
    factor_error_breakdown: Dict[str, float] = field(default_factory=dict)


def get_team_record_before_date(team_id: int, before_date: date) -> Tuple[int, int]:
    """Get team's win-loss record before a specific date."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ps.game_date, ps.is_home, ps.points as team_pts,
                   (SELECT AVG(ps2.points) FROM player_stats ps2 
                    JOIN players p2 ON p2.player_id = ps2.player_id
                    WHERE ps2.opponent_team_id = p.team_id 
                    AND ps2.game_date = ps.game_date) as opp_pts
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ? AND ps.game_date < ?
            GROUP BY ps.game_date
            """,
            conn,
            params=[team_id, str(before_date)],
        )
    
    if df.empty:
        return 0, 0
    
    # This is approximate - we'd need actual game results
    # For now, return based on available data
    return 0, 0


def get_team_profile(team_id: int, before_date: date) -> Dict[str, float]:
    """Get team's statistical profile before a date (PPG, opponent PPG, etc.)."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT 
                AVG(ps.points) as ppg,
                AVG(ps.rebounds) as rpg,
                AVG(ps.assists) as apg,
                AVG(ps.minutes) as mpg,
                COUNT(DISTINCT ps.game_date) as games
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ? AND ps.game_date < ?
            """,
            conn,
            params=[team_id, str(before_date)],
        )
    
    if df.empty or df.iloc[0]["games"] == 0:
        return {"ppg": 0.0, "rpg": 0.0, "apg": 0.0, "games": 0}
    
    row = df.iloc[0]
    return {
        "ppg": float(row["ppg"] or 0),
        "rpg": float(row["rpg"] or 0),
        "apg": float(row["apg"] or 0),
        "games": int(row["games"] or 0),
    }


def find_similar_teams(team_id: int, before_date: date, threshold: float = 0.15) -> List[int]:
    """Find teams with similar profiles (within threshold of PPG)."""
    target_profile = get_team_profile(team_id, before_date)
    if target_profile["games"] < 3:
        return []
    
    target_ppg = target_profile["ppg"]
    
    with get_conn() as conn:
        teams = conn.execute("SELECT team_id FROM teams WHERE team_id != ?", (team_id,)).fetchall()
    
    similar = []
    for (tid,) in teams:
        profile = get_team_profile(tid, before_date)
        if profile["games"] < 3:
            continue
        # Check if PPG is within threshold (e.g., 15%)
        if target_ppg > 0:
            diff_pct = abs(profile["ppg"] - target_ppg) / target_ppg
            if diff_pct <= threshold:
                similar.append(tid)
    
    return similar


def _get_position_group(position: str) -> str:
    """Normalize positions into groups: Guard (G), Forward (F), Center (C)."""
    pos = position.upper().strip()
    if not pos:
        return "F"
    if pos in ("PG", "SG", "G") or "GUARD" in pos:
        return "G"
    if pos in ("SF", "PF", "F") or "FORWARD" in pos:
        return "F"
    if pos in ("C",) or "CENTER" in pos:
        return "C"
    if "-" in pos:
        return _get_position_group(pos.split("-")[0])
    return "F"


def calculate_injury_adjustment(
    team_id: int,
    game_date: date,
    base_projection: float,
) -> Tuple[float, List[Dict]]:
    """
    Calculate scoring adjustment based on injuries for a specific game.
    
    Returns: (adjustment_amount, list_of_injured_players)
    
    Logic:
    - Get players who were out for this game
    - For high-minute players, deduct ~60% of their PPG (replacement level is lower)
    - Partially offset by usage boost to remaining players
    """
    injured_players = get_injuries_for_game(team_id, game_date)
    
    if not injured_players:
        return 0.0, []
    
    # Group injuries by position
    injuries_by_pos: Dict[str, float] = {"G": 0.0, "F": 0.0, "C": 0.0}
    total_lost_mpg = 0.0
    
    for player in injured_players:
        avg_min = player.get("avg_minutes", 0)
        pos_group = _get_position_group(player.get("position", ""))
        
        # Estimate PPG from minutes (rough: ~0.5 PPG per minute is typical)
        est_ppg = avg_min * 0.5
        
        # Higher minute players have bigger impact
        if avg_min >= 25:
            # Key player - significant impact
            impact = est_ppg * 0.50  # Lose ~50% of their production
        elif avg_min >= 15:
            # Rotation player
            impact = est_ppg * 0.40  # Lose ~40%
        else:
            # Bench player - minimal impact
            impact = est_ppg * 0.25
        
        injuries_by_pos[pos_group] += impact
        total_lost_mpg += avg_min
    
    # Total estimated point loss from injuries
    total_deduction = sum(injuries_by_pos.values())
    
    # Some recovery from other players stepping up
    # (already somewhat baked into the historical averages)
    recovery_factor = 0.3  # Assume 30% of lost production recovered by others
    
    final_adjustment = -total_deduction * (1 - recovery_factor)
    
    return final_adjustment, injured_players


def _get_team_agg_stats_before(team_id: int, before_date) -> Dict[str, float]:
    """
    Aggregate per-game averages for rebounds, assists, steals, blocks,
    turnovers, and shooting stats for *team_id* using only games before
    *before_date*.  Returns a dict with per-game averages.
    """
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ps.game_date,
                   SUM(ps.rebounds) AS reb, SUM(ps.assists) AS ast,
                   SUM(ps.steals) AS stl, SUM(ps.blocks) AS blk,
                   SUM(ps.turnovers) AS tov, SUM(ps.points) AS pts,
                   SUM(ps.fg_attempted) AS fga, SUM(ps.fg3_attempted) AS fg3a,
                   SUM(ps.ft_attempted) AS fta, SUM(ps.fg_made) AS fgm
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ? AND ps.game_date < ?
            GROUP BY ps.game_date
            """,
            conn,
            params=[team_id, str(before_date)],
        )
    if df.empty:
        return {"reb": 0.0, "ast": 0.0, "stl": 0.0, "blk": 0.0,
                "tov": 0.0, "pts": 0.0, "fga": 0.0, "fg3a": 0.0,
                "fta": 0.0, "fgm": 0.0, "games": 0, "pace": 0.0,
                "ts_pct": 0.0}
    games = len(df)
    avg = {c: float(df[c].mean()) for c in df.columns if c != "game_date"}
    avg["games"] = games

    # Pace estimate
    oreb_est = avg["reb"] * 0.30
    avg["pace"] = avg["fga"] - oreb_est + avg["tov"] + 0.44 * avg["fta"]

    # TS%
    ts_denom = 2 * (avg["fga"] + 0.44 * avg["fta"])
    avg["ts_pct"] = (avg["pts"] / ts_denom * 100) if ts_denom > 0 else 0.0

    return avg


def _get_opponent_ppg_before(team_id: int, before_date) -> float:
    """Average points *allowed* by team_id (opponent scoring) before a date."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ps.game_date, SUM(ps.points) AS opp_pts
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE ps.opponent_team_id = ? AND ps.game_date < ?
            GROUP BY ps.game_date
            """,
            conn,
            params=[team_id, str(before_date)],
        )
    return float(df["opp_pts"].mean()) if not df.empty else 0.0


def predict_game_historical(
    home_team_id: int,
    away_team_id: int,
    game_date: date,
    home_court_adv: float = 2.5,
    use_injury_adjustment: bool = True,
) -> Tuple[float, float, float, float, FactorBreakdown]:
    """
    Predict spread and total using only data available before the game date.

    Returns: (spread, total, home_projected, away_projected, factor_breakdown)

    Now incorporates all the enhanced factors:
      - Rebounding differential
      - Assist differential
      - Net Rating differential
      - Strength of Schedule
      - Pace-adjusted total
    """
    from src.analytics.prediction import PREDICTION_CONFIG as cfg

    fb = FactorBreakdown()

    # Get actual game results which properly aggregates team scores
    all_games = get_actual_game_results()

    if all_games.empty:
        return 0.0, 210.0, 105.0, 105.0, fb

    all_games = all_games[all_games["game_date"] < game_date]
    if all_games.empty:
        return 0.0, 210.0, 105.0, 105.0, fb

    # ---- helper: team games view ----
    def get_team_games(team_id: int, as_home: Optional[bool] = None) -> pd.DataFrame:
        home_mask = all_games["home_team_id"] == team_id
        away_mask = all_games["away_team_id"] == team_id
        parts = []
        if as_home is None or as_home is True:
            hg = all_games[home_mask].copy()
            if not hg.empty:
                hg["team_score"] = hg["home_score"]
                hg["opp_score"] = hg["away_score"]
                hg["opp_id"] = hg["away_team_id"]
                hg["was_home"] = True
                parts.append(hg)
        if as_home is None or as_home is False:
            ag = all_games[away_mask].copy()
            if not ag.empty:
                ag["team_score"] = ag["away_score"]
                ag["opp_score"] = ag["home_score"]
                ag["opp_id"] = ag["home_team_id"]
                ag["was_home"] = False
                parts.append(ag)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    home_team_games = get_team_games(home_team_id)
    away_team_games = get_team_games(away_team_id)

    if home_team_games.empty or away_team_games.empty:
        return 0.0, 210.0, 105.0, 105.0, fb

    def calc_team_projection(games_df, opp_id, predicting_home, similar_teams):
        season_ppg = games_df["team_score"].mean()
        loc_df = games_df[games_df["was_home"] == predicting_home]
        location_ppg = loc_df["team_score"].mean() if not loc_df.empty else season_ppg
        h2h_df = games_df[games_df["opp_id"] == opp_id]
        if not h2h_df.empty:
            h2h_ppg = h2h_df["team_score"].mean()
        elif similar_teams:
            sim_df = games_df[games_df["opp_id"].isin(similar_teams)]
            h2h_ppg = sim_df["team_score"].mean() if not sim_df.empty else season_ppg
        else:
            h2h_ppg = season_ppg
        return season_ppg * 0.5 + location_ppg * 0.3 + h2h_ppg * 0.2

    similar_to_away = find_similar_teams(away_team_id, game_date)
    similar_to_home = find_similar_teams(home_team_id, game_date)

    home_proj = calc_team_projection(home_team_games, away_team_id, True, similar_to_away)
    away_proj = calc_team_projection(away_team_games, home_team_id, False, similar_to_home)

    # Injury adjustments
    if use_injury_adjustment:
        home_adj, _ = calculate_injury_adjustment(home_team_id, game_date, home_proj)
        away_adj, _ = calculate_injury_adjustment(away_team_id, game_date, away_proj)
        home_proj += home_adj
        away_proj += away_adj
        fb.injury_adj = home_adj - away_adj

    fb.base_spread = home_proj - away_proj
    fb.home_court = home_court_adv

    spread = fb.base_spread + fb.home_court

    # ---- Enhanced factors using pre-game aggregated stats ----
    home_agg = _get_team_agg_stats_before(home_team_id, game_date)
    away_agg = _get_team_agg_stats_before(away_team_id, game_date)

    if home_agg["games"] >= 3 and away_agg["games"] >= 3:
        # Turnover margin
        home_to_margin = home_agg["stl"] - home_agg["tov"]
        away_to_margin = away_agg["stl"] - away_agg["tov"]
        fb.turnover_adj = (home_to_margin - away_to_margin) * cfg["turnover_multiplier"]
        spread += fb.turnover_adj

        # TS% advantage
        ts_adv = home_agg["ts_pct"] - away_agg["ts_pct"]
        fb.ts_adj = ts_adv * cfg["ts_pct_multiplier"]
        spread += fb.ts_adj

        # Rebound differential
        reb_adv = home_agg["reb"] - away_agg["reb"]
        fb.rebound_adj = reb_adv * cfg["rebound_multiplier"]
        spread += fb.rebound_adj

        # Assist differential
        ast_adv = home_agg["ast"] - away_agg["ast"]
        fb.assist_adj = ast_adv * cfg["assist_multiplier"]
        spread += fb.assist_adj

        # Net Rating
        home_opp_ppg = _get_opponent_ppg_before(home_team_id, game_date)
        away_opp_ppg = _get_opponent_ppg_before(away_team_id, game_date)
        home_pace = home_agg["pace"]
        away_pace = away_agg["pace"]
        if home_pace > 0 and away_pace > 0:
            home_ortg = (home_agg["pts"] / home_pace) * 100
            home_drtg = (home_opp_ppg / home_pace) * 100 if home_opp_ppg else home_ortg
            away_ortg = (away_agg["pts"] / away_pace) * 100
            away_drtg = (away_opp_ppg / away_pace) * 100 if away_opp_ppg else away_ortg
            home_net = home_ortg - home_drtg
            away_net = away_ortg - away_drtg
            fb.net_rating_adj = (home_net - away_net) * cfg["net_rating_multiplier"]
            spread += fb.net_rating_adj

        # Strength of Schedule (simplified: average margin of opponents)
        home_opp_margin = home_agg["pts"] - home_opp_ppg if home_opp_ppg else 0.0
        away_opp_margin = away_agg["pts"] - away_opp_ppg if away_opp_ppg else 0.0
        # Reverse: team with lower own margin might have faced harder opponents
        # Use opponent average scoring as a rough SOS proxy
        if home_opp_ppg > 0 and away_opp_ppg > 0:
            # Normalize: higher opponent scoring = harder schedule
            avg_opp_ppg = (home_opp_ppg + away_opp_ppg) / 2
            home_sos = (home_opp_ppg - avg_opp_ppg) / 10.0  # scale to ~[-1, +1]
            away_sos = (away_opp_ppg - avg_opp_ppg) / 10.0
            fb.sos_adj = (home_sos - away_sos) * cfg["sos_multiplier"]
            spread += fb.sos_adj

    total = home_proj + away_proj

    # Pace-adjusted total
    if home_agg["pace"] > 0 and away_agg["pace"] > 0:
        combined_pace = (home_agg["pace"] + away_agg["pace"]) / 2
        pace_factor = combined_pace / cfg["college_avg_pace"]
        fb.pace_adj = total * (pace_factor - 1.0)
        total *= pace_factor

    return spread, total, home_proj, away_proj, fb


def get_actual_game_results() -> pd.DataFrame:
    """
    Get actual game results by aggregating player stats per game.
    Returns DataFrame with game_date, home_team_id, away_team_id, home_score, away_score.

    Uses players.team_id to assign each stat line to a team, then matches
    home/away sides by opponent_team_id and is_home. This allows filtering
    by home/away team reliably (even if trades add some noise).
    """
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT 
                ps.game_date,
                p.team_id AS team_id,
                ps.opponent_team_id,
                ps.is_home,
                SUM(ps.points) as team_score
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            GROUP BY ps.game_date, p.team_id, ps.opponent_team_id, ps.is_home
            ORDER BY ps.game_date DESC
            """,
            conn,
        )
        teams = pd.read_sql("SELECT team_id, abbreviation FROM teams", conn)
    
    if df.empty:
        return pd.DataFrame()
    
    team_lookup = {int(row["team_id"]): row["abbreviation"] for _, row in teams.iterrows()}
    
    games = []
    processed = set()
    
    # Build index for quick lookup: (date, team_id, opp_id, is_home) -> score
    index = {}
    for _, row in df.iterrows():
        key = (str(row["game_date"]), int(row["team_id"]), int(row["opponent_team_id"]), int(row["is_home"]))
        index[key] = float(row["team_score"])

    for (date_str, home_id, away_id, is_home), home_score in list(index.items()):
        if is_home != 1:
            continue

        # Find matching away entry (same date, opp/home swapped, is_home=0)
        opp_key = (date_str, away_id, home_id, 0)
        away_score = index.get(opp_key)
        if away_score is None:
            continue

        key = (date_str, min(home_id, away_id), max(home_id, away_id))
        if key in processed:
            continue

        # Basic sanity check
        if home_score < 20 or away_score < 20:
            continue

        games.append({
            "game_date": date_str,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_abbr": team_lookup.get(home_id, "???"),
            "away_abbr": team_lookup.get(away_id, "???"),
            "home_score": home_score,
            "away_score": away_score,
        })
        processed.add(key)
    
    return pd.DataFrame(games)


def run_backtest(
    min_games_before: int = 5,
    home_team_filter: Optional[int] = None,
    away_team_filter: Optional[int] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    use_injury_adjustment: bool = True,
) -> BacktestResults:
    """
    Run backtest on historical games.
    
    Args:
        min_games_before: Minimum games each team must have before we predict
        home_team_filter: If set, only analyze games where this team is HOME
        away_team_filter: If set, only analyze games where this team is AWAY
        progress_cb: Optional callback for progress updates
        use_injury_adjustment: If True, adjust predictions based on inferred injuries
    
    Filter combinations:
        - Both None: All games
        - Home only: Games where that team is home
        - Away only: Games where that team is away
        - Both set: Specific matchup (home vs away)
    """
    progress = progress_cb or (lambda msg: None)
    results = BacktestResults()
    
    progress("Loading game results...")
    games_df = get_actual_game_results()
    if games_df.empty:
        progress("No game data found")
        return results
    
    progress(f"Found {len(games_df)} total games in database")
    
    # Debug: show unique home teams
    if not games_df.empty:
        unique_home = games_df["home_team_id"].unique()
        progress(f"Unique home team IDs: {len(unique_home)} teams")
    
    # Apply filters
    if home_team_filter and away_team_filter:
        # Specific matchup
        games_df = games_df[
            (games_df["home_team_id"] == home_team_filter) & 
            (games_df["away_team_id"] == away_team_filter)
        ]
        progress(f"Filtered to {len(games_df)} games: home={home_team_filter} vs away={away_team_filter}")
    elif home_team_filter:
        # Only games where this team is home
        before_filter = len(games_df)
        games_df = games_df[games_df["home_team_id"] == home_team_filter]
        progress(f"Filtered for home_team={home_team_filter}: {before_filter} -> {len(games_df)} games")
        progress(f"Filtered to {len(games_df)} home games for team {home_team_filter}")
    elif away_team_filter:
        # Only games where this team is away
        before_filter = len(games_df)
        games_df = games_df[games_df["away_team_id"] == away_team_filter]
        progress(f"Filtered for away_team={away_team_filter}: {before_filter} -> {len(games_df)} games")
    
    # Sort by date ascending to process chronologically
    games_df = games_df.sort_values("game_date")
    
    # Get team info
    with get_conn() as conn:
        teams = pd.read_sql("SELECT team_id, abbreviation, name FROM teams", conn)
    team_info = {int(row["team_id"]): (row["abbreviation"], row["name"]) for _, row in teams.iterrows()}
    
    # Initialize team accuracy tracking
    for tid, (abbr, name) in team_info.items():
        results.team_accuracy[tid] = TeamAccuracy(
            team_id=tid, team_abbr=abbr, team_name=name
        )
    
    predictions = []
    factor_breakdowns: List[FactorBreakdown] = []
    total_to_process = len(games_df)

    for idx, (_, game) in enumerate(games_df.iterrows()):
        if idx % 20 == 0:
            progress(f"Analyzing game {idx + 1}/{total_to_process}...")
        
        game_date = game["game_date"]
        home_id = int(game["home_team_id"])
        away_id = int(game["away_team_id"])
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        
        # Check both teams have enough prior games
        home_profile = get_team_profile(home_id, game_date)
        away_profile = get_team_profile(away_id, game_date)
        
        if home_profile["games"] < min_games_before or away_profile["games"] < min_games_before:
            continue
        
        # Get injury information for this game
        home_adj, home_injured = calculate_injury_adjustment(home_id, game_date, 0)
        away_adj, away_injured = calculate_injury_adjustment(away_id, game_date, 0)
        
        home_injury_names = [p.get("name", "Unknown") for p in home_injured if p.get("avg_minutes", 0) >= 12]
        away_injury_names = [p.get("name", "Unknown") for p in away_injured if p.get("avg_minutes", 0) >= 12]
        
        # Make prediction using only data before this game
        pred_spread, pred_total, pred_home, pred_away, fb = predict_game_historical(
            home_id, away_id, game_date, use_injury_adjustment=use_injury_adjustment
        )
        factor_breakdowns.append(fb)

        actual_spread = home_score - away_score
        actual_total = home_score + away_score
        
        # Calculate errors
        spread_error = pred_spread - actual_spread
        total_error = pred_total - actual_total
        home_score_error = pred_home - home_score
        away_score_error = pred_away - away_score
        
        # Determine predicted and actual winners
        if pred_spread > 0.5:
            predicted_winner = "HOME"
        elif pred_spread < -0.5:
            predicted_winner = "AWAY"
        else:
            predicted_winner = "PUSH"
        
        if actual_spread > 0:
            actual_winner = "HOME"
        elif actual_spread < 0:
            actual_winner = "AWAY"
        else:
            actual_winner = "PUSH"
        
        winner_correct = (predicted_winner == actual_winner) or \
                        (predicted_winner == "PUSH" and abs(actual_spread) <= 3)
        
        pred = GamePrediction(
            game_date=game_date,
            home_team_id=home_id,
            away_team_id=away_id,
            home_abbr=game["home_abbr"],
            away_abbr=game["away_abbr"],
            predicted_spread=pred_spread,
            predicted_total=pred_total,
            predicted_home_score=pred_home,
            predicted_away_score=pred_away,
            actual_home_score=home_score,
            actual_away_score=away_score,
            actual_spread=actual_spread,
            actual_total=actual_total,
            spread_error=spread_error,
            total_error=total_error,
            home_score_error=home_score_error,
            away_score_error=away_score_error,
            predicted_winner=predicted_winner,
            actual_winner=actual_winner,
            winner_correct=winner_correct,
            spread_within_5=abs(spread_error) <= 5,
            total_within_10=abs(total_error) <= 10,
            home_injuries=home_injury_names,
            away_injuries=away_injury_names,
            home_injury_adj=home_adj,
            away_injury_adj=away_adj,
        )
        predictions.append(pred)
        
        # Update team accuracy
        for tid in [home_id, away_id]:
            if tid in results.team_accuracy:
                ta = results.team_accuracy[tid]
                ta.games_analyzed += 1
                if winner_correct:
                    ta.spread_correct += 1
                if pred.total_within_10:
                    ta.total_correct += 1
        
        # Track wins/losses
        if home_score > away_score:
            if home_id in results.team_accuracy:
                results.team_accuracy[home_id].wins += 1
            if away_id in results.team_accuracy:
                results.team_accuracy[away_id].losses += 1
        elif away_score > home_score:
            if away_id in results.team_accuracy:
                results.team_accuracy[away_id].wins += 1
            if home_id in results.team_accuracy:
                results.team_accuracy[home_id].losses += 1
    
    results.predictions = predictions
    results.total_games = len(predictions)

    progress("Calculating accuracy stats...")

    # Calculate final accuracy percentages
    if results.total_games > 0:
        total_winner_correct = sum(1 for p in predictions if p.winner_correct)
        total_total_correct = sum(1 for p in predictions if p.total_within_10)
        results.overall_spread_accuracy = total_winner_correct / results.total_games * 100
        results.overall_total_accuracy = total_total_correct / results.total_games * 100

    for ta in results.team_accuracy.values():
        if ta.games_analyzed > 0:
            ta.spread_accuracy = ta.spread_correct / ta.games_analyzed * 100
            ta.total_accuracy = ta.total_correct / ta.games_analyzed * 100
            errors = [abs(p.spread_error) for p in predictions
                     if p.home_team_id == ta.team_id or p.away_team_id == ta.team_id]
            ta.avg_spread_error = sum(errors) / len(errors) if errors else 0
            total_errors = [abs(p.total_error) for p in predictions
                          if p.home_team_id == ta.team_id or p.away_team_id == ta.team_id]
            ta.avg_total_error = sum(total_errors) / len(total_errors) if total_errors else 0

    # --- Per-factor average absolute contribution ---
    # This shows how much each factor contributed on average so you can
    # see which factors are most influential and potentially overtune.
    if factor_breakdowns:
        factor_names = ["base_spread", "home_court", "turnover_adj", "ts_adj",
                        "rebound_adj", "assist_adj", "net_rating_adj",
                        "sos_adj", "injury_adj", "pace_adj"]
        for fname in factor_names:
            vals = [abs(getattr(fb, fname)) for fb in factor_breakdowns]
            results.factor_error_breakdown[fname] = sum(vals) / len(vals) if vals else 0.0
        progress(
            "Factor breakdown (avg |contribution|): "
            + ", ".join(f"{k}={v:.2f}" for k, v in results.factor_error_breakdown.items())
        )

    progress(f"Backtest complete: {results.total_games} games analyzed")
    return results
