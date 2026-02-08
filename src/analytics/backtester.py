from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.database.db import get_conn
from src.analytics.injury_history import get_injuries_for_game
from src.analytics.stats_engine import get_team_metrics, detect_fatigue


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
    home_injuries: List[str] = field(default_factory=list)
    away_injuries: List[str] = field(default_factory=list)
    home_injury_adj: float = 0.0
    away_injury_adj: float = 0.0
    # Fatigue info
    home_fatigue_penalty: float = 0.0
    away_fatigue_penalty: float = 0.0


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
class BacktestResults:
    predictions: List[GamePrediction] = field(default_factory=list)
    team_accuracy: Dict[int, TeamAccuracy] = field(default_factory=dict)
    overall_spread_accuracy: float = 0.0
    overall_total_accuracy: float = 0.0
    total_games: int = 0


def get_team_record_before_date(team_id: int, before_date: date) -> Tuple[int, int]:
    """
    Get team's win-loss record before a specific date.
    Uses the win_loss column from player_stats for accurate results.
    """
    with get_conn() as conn:
        # Use win_loss column: get one row per game (any player from this team)
        rows = conn.execute(
            """
            SELECT DISTINCT ps.game_date, ps.win_loss
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ? AND ps.game_date < ? AND ps.win_loss IS NOT NULL
            GROUP BY ps.game_date
            """,
            (team_id, str(before_date)),
        ).fetchall()
    
    wins = sum(1 for _, wl in rows if wl == "W")
    losses = sum(1 for _, wl in rows if wl == "L")
    return wins, losses


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


# Cache for advanced profiles to avoid redundant DB queries during backtests
_adv_profile_cache: Dict[tuple, Dict[str, float]] = {}


def clear_advanced_profile_cache() -> None:
    """Clear the cached advanced profiles (call before each backtest run)."""
    _adv_profile_cache.clear()


def get_team_advanced_profile(team_id: int, before_date: date) -> Dict[str, float]:
    """
    Get team advanced stats before a date for backtesting.
    Includes rebounds, offensive/defensive rating, pace, HCA, and Four Factors.

    Checks official team_metrics first (if synced); then falls back to
    computing from player game logs.  Results are cached for performance.
    """
    cache_key = (team_id, str(before_date))
    if cache_key in _adv_profile_cache:
        return _adv_profile_cache[cache_key]
    
    result = {
        "rebounds_pg": 0.0,
        "off_rating": 110.0,
        "def_rating": 110.0,
        "pace": 98.0,
        "hca": 3.0,
        # Four Factors (default None = unavailable)
        "ff_efg_pct": None,
        "ff_fta_rate": None,
        "ff_tm_tov_pct": None,
        "ff_oreb_pct": None,
        "opp_efg_pct": None,
        "opp_fta_rate": None,
        "opp_tm_tov_pct": None,
        "opp_oreb_pct": None,
        # Clutch
        "clutch_net_rating": None,
    }

    # Try official metrics first (they represent the full season, which is
    # acceptable since we're backtesting within the same season)
    metrics = get_team_metrics(team_id)
    if metrics:
        result["off_rating"] = float(metrics.get("e_off_rating") or metrics.get("off_rating") or 110.0)
        result["def_rating"] = float(metrics.get("e_def_rating") or metrics.get("def_rating") or 110.0)
        result["pace"] = float(metrics.get("e_pace") or metrics.get("pace") or 98.0)
        # Four Factors
        for k in ["ff_efg_pct", "ff_fta_rate", "ff_tm_tov_pct", "ff_oreb_pct",
                   "opp_efg_pct", "opp_fta_rate", "opp_tm_tov_pct", "opp_oreb_pct"]:
            result[k] = metrics.get(k)
        result["clutch_net_rating"] = metrics.get("clutch_net_rating")
        # HCA from home/road splits
        home_pts = metrics.get("home_pts")
        road_pts = metrics.get("road_pts")
        if home_pts and road_pts:
            result["hca"] = max(1.5, min(5.0, float(home_pts) - float(road_pts)))

    # Always compute rebounds from game logs (per-date for accuracy)
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ps.game_date,
                   SUM(ps.points) as team_pts,
                   SUM(ps.rebounds) as team_reb,
                   SUM(ps.fg_attempted) as team_fga,
                   SUM(ps.ft_attempted) as team_fta,
                   SUM(ps.turnovers) as team_tov,
                   SUM(ps.oreb) as team_oreb
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ? AND ps.game_date < ?
            GROUP BY ps.game_date
            """,
            conn, params=[team_id, str(before_date)],
        )
    
    if not df.empty:
        result["rebounds_pg"] = float(df["team_reb"].mean())
        
        # If official metrics were NOT available, compute from logs
        if not metrics:
            avg_pts = float(df["team_pts"].mean())
            avg_fga = float(df["team_fga"].mean())
            avg_fta = float(df["team_fta"].mean())
            avg_tov = float(df["team_tov"].mean())
            avg_oreb = float(df["team_oreb"].mean())
            poss = avg_fga - avg_oreb + avg_tov + 0.44 * avg_fta
            if poss > 0:
                result["off_rating"] = (avg_pts / poss) * 100
                result["pace"] = poss

            # Defensive rating from opponent data
            opp_df = pd.read_sql(
                """
                SELECT ps.game_date,
                       SUM(ps.points) as opp_pts,
                       SUM(ps.fg_attempted) as opp_fga,
                       SUM(ps.ft_attempted) as opp_fta,
                       SUM(ps.turnovers) as opp_tov,
                       SUM(ps.oreb) as opp_oreb
                FROM player_stats ps
                WHERE ps.opponent_team_id = ? AND ps.game_date < ?
                GROUP BY ps.game_date
                """,
                conn, params=[team_id, str(before_date)],
            )
            if not opp_df.empty:
                opp_poss = float(opp_df["opp_fga"].mean()) - float(opp_df["opp_oreb"].mean()) + \
                           float(opp_df["opp_tov"].mean()) + 0.44 * float(opp_df["opp_fta"].mean())
                if opp_poss > 0:
                    result["def_rating"] = (float(opp_df["opp_pts"].mean()) / opp_poss) * 100

            # HCA from home/away scoring
            home_df = pd.read_sql(
                "SELECT ps.game_date, SUM(ps.points) as team_pts "
                "FROM player_stats ps JOIN players p ON p.player_id = ps.player_id "
                "WHERE p.team_id = ? AND ps.is_home = 1 AND ps.game_date < ? "
                "GROUP BY ps.game_date",
                conn, params=[team_id, str(before_date)],
            )
            away_df2 = pd.read_sql(
                "SELECT ps.game_date, SUM(ps.points) as team_pts "
                "FROM player_stats ps JOIN players p ON p.player_id = ps.player_id "
                "WHERE p.team_id = ? AND ps.is_home = 0 AND ps.game_date < ? "
                "GROUP BY ps.game_date",
                conn, params=[team_id, str(before_date)],
            )
            if not home_df.empty and not away_df2.empty:
                hca = float(home_df["team_pts"].mean()) - float(away_df2["team_pts"].mean())
                result["hca"] = max(1.5, min(5.0, hca))

    _adv_profile_cache[cache_key] = result
    return result


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


def predict_game_historical(
    home_team_id: int,
    away_team_id: int,
    game_date: date,
    home_court_adv: float = 3.0,
    use_injury_adjustment: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Predict spread and total using only data available before the game date.
    Returns: (spread, total, home_projected, away_projected)
    
    Uses actual completed game scores, not player-based aggregation.
    This avoids issues with player trades affecting team totals.
    
    Weights:
    - 50% season average (before this game)
    - 30% home/away splits
    - 20% head-to-head history (or similar opponents)
    
    If use_injury_adjustment=True, applies injury-based scoring adjustments.
    """
    # Get actual game results which properly aggregates team scores
    all_games = get_actual_game_results()
    
    if all_games.empty:
        return 0.0, 210.0, 105.0, 105.0
    
    # Filter to games before this date
    all_games = all_games[all_games["game_date"] < game_date]
    
    if all_games.empty:
        return 0.0, 210.0, 105.0, 105.0
    
    def get_team_games(team_id: int, as_home: Optional[bool] = None) -> pd.DataFrame:
        """Get all games for a team, optionally filtered by home/away."""
        # Games where this team was home
        home_mask = all_games["home_team_id"] == team_id
        # Games where this team was away
        away_mask = all_games["away_team_id"] == team_id
        
        results = []
        
        # Home games for this team
        if as_home is None or as_home is True:
            home_games = all_games[home_mask].copy()
            if not home_games.empty:
                home_games["team_score"] = home_games["home_score"]
                home_games["opp_score"] = home_games["away_score"]
                home_games["opp_id"] = home_games["away_team_id"]
                home_games["was_home"] = True
                results.append(home_games)
        
        # Away games for this team
        if as_home is None or as_home is False:
            away_games = all_games[away_mask].copy()
            if not away_games.empty:
                away_games["team_score"] = away_games["away_score"]
                away_games["opp_score"] = away_games["home_score"]
                away_games["opp_id"] = away_games["home_team_id"]
                away_games["was_home"] = False
                results.append(away_games)
        
        if not results:
            return pd.DataFrame()
        
        return pd.concat(results, ignore_index=True)
    
    # Get all games for each team
    home_team_games = get_team_games(home_team_id)
    away_team_games = get_team_games(away_team_id)
    
    if home_team_games.empty or away_team_games.empty:
        return 0.0, 210.0, 105.0, 105.0
    
    def calc_team_projection(games_df: pd.DataFrame, opp_id: int, predicting_home: bool, similar_teams: List[int]) -> float:
        # Season average PPG (50%)
        season_ppg = games_df["team_score"].mean()
        
        # Home/Away split PPG (30%)
        location_df = games_df[games_df["was_home"] == predicting_home]
        if not location_df.empty:
            location_ppg = location_df["team_score"].mean()
        else:
            location_ppg = season_ppg
        
        # Head-to-head PPG or similar opponents (20%)
        h2h_df = games_df[games_df["opp_id"] == opp_id]
        if not h2h_df.empty:
            h2h_ppg = h2h_df["team_score"].mean()
        elif similar_teams:
            similar_df = games_df[games_df["opp_id"].isin(similar_teams)]
            if not similar_df.empty:
                h2h_ppg = similar_df["team_score"].mean()
            else:
                h2h_ppg = season_ppg
        else:
            h2h_ppg = season_ppg
        
        return season_ppg * 0.5 + location_ppg * 0.3 + h2h_ppg * 0.2
    
    # Find similar teams for each
    similar_to_away = find_similar_teams(away_team_id, game_date)
    similar_to_home = find_similar_teams(home_team_id, game_date)
    
    home_proj = calc_team_projection(home_team_games, away_team_id, True, similar_to_away)
    away_proj = calc_team_projection(away_team_games, home_team_id, False, similar_to_home)
    
    # Apply injury adjustments if enabled
    if use_injury_adjustment:
        home_adj, _ = calculate_injury_adjustment(home_team_id, game_date, home_proj)
        away_adj, _ = calculate_injury_adjustment(away_team_id, game_date, away_proj)
        home_proj += home_adj
        away_proj += away_adj
    
    # ============ ADVANCED ADJUSTMENTS ============
    home_adv = get_team_advanced_profile(home_team_id, game_date)
    away_adv = get_team_advanced_profile(away_team_id, game_date)
    
    # Dynamic home court advantage
    hca = home_adv["hca"]
    
    # Rebound differential
    rebound_adj = (home_adv["rebounds_pg"] - away_adv["rebounds_pg"]) * 0.15
    
    # Off/Def rating comparison
    home_matchup_edge = home_adv["off_rating"] - away_adv["def_rating"]
    away_matchup_edge = away_adv["off_rating"] - home_adv["def_rating"]
    rating_adj = (home_matchup_edge - away_matchup_edge) * 0.12
    
    # Pace factor for total
    expected_pace = (home_adv["pace"] + away_adv["pace"]) / 2
    pace_factor = (expected_pace - 98.0) / 98.0
    
    # Four Factors adjustment (if available)
    four_factors_adj = 0.0
    h_efg = home_adv.get("ff_efg_pct")
    a_efg = away_adv.get("ff_efg_pct")
    if h_efg is not None and a_efg is not None:
        # Simplified Four Factors comparison for backtesting
        efg_edge = (h_efg or 0) - (a_efg or 0)
        tov_edge = (away_adv.get("ff_tm_tov_pct") or 0) - (home_adv.get("ff_tm_tov_pct") or 0)
        oreb_edge = (home_adv.get("ff_oreb_pct") or 0) - (away_adv.get("ff_oreb_pct") or 0)
        fta_edge = (home_adv.get("ff_fta_rate") or 0) - (away_adv.get("ff_fta_rate") or 0)
        four_factors_adj = (efg_edge * 0.40 + tov_edge * 0.25 +
                           oreb_edge * 0.20 + fta_edge * 0.15) * 0.3
    
    # Clutch adjustment for projected close games
    clutch_adj = 0.0
    base_spread = (home_proj - away_proj) + hca
    if abs(base_spread) < 6.0:
        h_clutch = home_adv.get("clutch_net_rating")
        a_clutch = away_adv.get("clutch_net_rating")
        if h_clutch is not None and a_clutch is not None:
            clutch_adj = max(-2.0, min(2.0, (h_clutch - a_clutch) * 0.05))

    # Fatigue detection
    home_fatigue = detect_fatigue(home_team_id, game_date)
    away_fatigue = detect_fatigue(away_team_id, game_date)
    fatigue_adj = home_fatigue["fatigue_penalty"] - away_fatigue["fatigue_penalty"]

    # Final spread and total
    spread = (home_proj - away_proj) + hca + rebound_adj + rating_adj + \
             four_factors_adj + clutch_adj - fatigue_adj
    total = (home_proj + away_proj) * (1 + pace_factor * 0.5)
    # Reduce total slightly for fatigued games
    combined_fatigue = home_fatigue["fatigue_penalty"] + away_fatigue["fatigue_penalty"]
    total -= combined_fatigue * 0.3
    
    return spread, total, home_proj, away_proj


def get_actual_game_results() -> pd.DataFrame:
    """
    Get actual game results by aggregating player stats per game.
    Returns DataFrame with game_date, home_team_id, away_team_id, home_score, away_score.

    Groups by game_id + is_home so that each side of the game is summed correctly,
    regardless of whether players were later traded to other teams.  The team
    identities come from opponent_team_id cross-referencing (the home side's
    opponent is the away team and vice-versa).
    """
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT
                ps.game_date,
                ps.game_id,
                ps.is_home,
                MIN(ps.opponent_team_id) AS opponent_team_id,
                SUM(ps.points)           AS team_score
            FROM player_stats ps
            WHERE ps.game_id IS NOT NULL AND ps.game_id != ''
            GROUP BY ps.game_date, ps.game_id, ps.is_home
            ORDER BY ps.game_date DESC
            """,
            conn,
        )
        teams = pd.read_sql("SELECT team_id, abbreviation FROM teams", conn)

    if df.empty:
        return pd.DataFrame()

    team_lookup = {int(row["team_id"]): row["abbreviation"] for _, row in teams.iterrows()}

    # Index: (game_id, is_home) -> {game_date, opponent_team_id, team_score}
    game_sides: dict[tuple, dict] = {}
    for _, row in df.iterrows():
        gid = str(row["game_id"])
        is_home = int(row["is_home"])
        game_sides[(gid, is_home)] = {
            "game_date": str(row["game_date"]),
            "opponent_team_id": int(row["opponent_team_id"]),
            "team_score": float(row["team_score"]),
        }

    games = []
    processed: set[str] = set()

    for (gid, is_home), data in game_sides.items():
        if is_home != 1 or gid in processed:
            continue

        away_data = game_sides.get((gid, 0))
        if away_data is None:
            continue

        home_score = data["team_score"]
        away_score = away_data["team_score"]

        # Home side's opponent = away team; away side's opponent = home team
        away_team_id = data["opponent_team_id"]
        home_team_id = away_data["opponent_team_id"]

        # Sanity check
        if home_score < 20 or away_score < 20:
            continue

        games.append({
            "game_date": data["game_date"],
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_abbr": team_lookup.get(home_team_id, "???"),
            "away_abbr": team_lookup.get(away_team_id, "???"),
            "home_score": home_score,
            "away_score": away_score,
        })
        processed.add(gid)

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
    
    # Clear cached advanced profiles from previous runs
    clear_advanced_profile_cache()
    
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

        # Fatigue detection
        home_fatigue_info = detect_fatigue(home_id, game_date)
        away_fatigue_info = detect_fatigue(away_id, game_date)
        
        # Make prediction using only data before this game
        pred_spread, pred_total, pred_home, pred_away = predict_game_historical(
            home_id, away_id, game_date, use_injury_adjustment=use_injury_adjustment
        )
        
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
            home_fatigue_penalty=home_fatigue_info["fatigue_penalty"],
            away_fatigue_penalty=away_fatigue_info["fatigue_penalty"],
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
    
    progress(f"Backtest complete: {results.total_games} games analyzed")
    return results
