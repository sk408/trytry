from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.database.db import get_conn


@dataclass
class PlayerStats:
    player_id: int
    name: str
    position: str
    is_injured: bool
    # Season averages - basic
    ppg: float  # points per game
    rpg: float  # rebounds per game
    apg: float  # assists per game
    mpg: float  # minutes per game
    games_played: int
    # Home/Away splits
    ppg_home: float
    ppg_away: float
    # Vs opponent (if available)
    ppg_vs_opp: float
    games_vs_opp: int
    # Defensive stats
    spg: float = 0.0  # steals per game
    bpg: float = 0.0  # blocks per game
    tpg: float = 0.0  # turnovers per game
    # Shooting efficiency
    fg_pct: float = 0.0   # field goal %
    fg3_pct: float = 0.0  # three-point %
    ft_pct: float = 0.0   # free throw %
    ts_pct: float = 0.0   # true shooting % = PTS / (2 * (FGA + 0.44 * FTA))
    efg_pct: float = 0.0  # effective FG% = (FGM + 0.5 * FG3M) / FGA
    # Shot attempts per game (for volume analysis)
    fg3_rate: float = 0.0  # 3PA / FGA (how often they shoot 3s)
    ft_rate: float = 0.0   # FTA / FGA (how often they get to the line)
    fta_pg: float = 0.0    # free throw attempts per game
    # Ratios
    ast_to_ratio: float = 0.0  # assist-to-turnover ratio


@dataclass
class TeamMatchupStats:
    team_id: int
    team_abbr: str
    team_name: str
    players: List[PlayerStats]
    # Basic totals
    total_ppg: float
    total_rpg: float
    total_apg: float
    projected_points: float  # weighted projection for this matchup
    # Defensive totals
    total_spg: float = 0.0  # team steals
    total_bpg: float = 0.0  # team blocks
    total_tpg: float = 0.0  # team turnovers
    # Team shooting efficiency (weighted by attempts)
    team_ts_pct: float = 0.0    # team true shooting %
    team_fg3_rate: float = 0.0  # team 3PT attempt rate
    team_ft_rate: float = 0.0   # team FT attempt rate
    # Net ratings
    turnover_margin: float = 0.0  # steals - turnovers (positive = good)
    # Pace-adjusted efficiency (per 100 possessions)
    offensive_rating: float = 0.0   # points scored per 100 possessions
    defensive_rating: float = 0.0   # points allowed per 100 possessions
    net_rating: float = 0.0         # ORtg - DRtg
    pace: float = 0.0               # estimated possessions per game
    # Strength of schedule
    sos: float = 0.0                # normalized SOS (0.0 = average)
    # Recent form
    recent_ppg: float = 0.0         # last 5 games team PPG (for trend display)


def _load_player_df(player_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT * FROM player_stats WHERE player_id = ? ORDER BY game_date DESC",
            conn,
            params=[player_id],
            parse_dates=["game_date"],
        )
    return df


def get_player_comprehensive_stats(
    player_id: int,
    player_name: str,
    position: str,
    is_injured: bool,
    opponent_team_id: Optional[int] = None,
) -> PlayerStats:
    """Get comprehensive stats for a single player including efficiency metrics."""
    df = _load_player_df(player_id)
    
    if df.empty:
        return PlayerStats(
            player_id=player_id,
            name=player_name,
            position=position,
            is_injured=is_injured,
            ppg=0.0, rpg=0.0, apg=0.0, mpg=0.0, games_played=0,
            ppg_home=0.0, ppg_away=0.0,
            ppg_vs_opp=0.0, games_vs_opp=0,
        )
    
    import math
    
    def safe_mean(series, default=0.0):
        """Safely compute mean, returning default for empty/NaN series."""
        if series.empty or series.isna().all():
            return default
        val = float(series.mean())
        return default if math.isnan(val) else val
    
    # Basic season averages
    ppg = safe_mean(df["points"])
    rpg = safe_mean(df["rebounds"])
    apg = safe_mean(df["assists"])
    mpg = safe_mean(df["minutes"])
    games_played = len(df)
    
    # Defensive stats (use 0 if column missing)
    spg = safe_mean(df["steals"]) if "steals" in df.columns else 0.0
    bpg = safe_mean(df["blocks"]) if "blocks" in df.columns else 0.0
    tpg = safe_mean(df["turnovers"]) if "turnovers" in df.columns else 0.0
    
    # Shooting stats - compute totals first for efficiency calculations
    total_fg_made = df["fg_made"].sum() if "fg_made" in df.columns else 0
    total_fg_attempted = df["fg_attempted"].sum() if "fg_attempted" in df.columns else 0
    total_fg3_made = df["fg3_made"].sum() if "fg3_made" in df.columns else 0
    total_fg3_attempted = df["fg3_attempted"].sum() if "fg3_attempted" in df.columns else 0
    total_ft_made = df["ft_made"].sum() if "ft_made" in df.columns else 0
    total_ft_attempted = df["ft_attempted"].sum() if "ft_attempted" in df.columns else 0
    total_points = df["points"].sum()
    
    # Shooting percentages (season totals for accuracy)
    fg_pct = (total_fg_made / total_fg_attempted * 100) if total_fg_attempted > 0 else 0.0
    fg3_pct = (total_fg3_made / total_fg3_attempted * 100) if total_fg3_attempted > 0 else 0.0
    ft_pct = (total_ft_made / total_ft_attempted * 100) if total_ft_attempted > 0 else 0.0
    
    # True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
    ts_denominator = 2 * (total_fg_attempted + 0.44 * total_ft_attempted)
    ts_pct = (total_points / ts_denominator * 100) if ts_denominator > 0 else 0.0
    
    # Effective FG% = (FGM + 0.5 * FG3M) / FGA
    efg_pct = ((total_fg_made + 0.5 * total_fg3_made) / total_fg_attempted * 100) if total_fg_attempted > 0 else 0.0
    
    # Shot selection rates
    fg3_rate = (total_fg3_attempted / total_fg_attempted * 100) if total_fg_attempted > 0 else 0.0
    ft_rate = (total_ft_attempted / total_fg_attempted * 100) if total_fg_attempted > 0 else 0.0
    
    # Free throw attempts per game (for injury impact calculation)
    fta_pg = total_ft_attempted / games_played if games_played > 0 else 0.0
    
    # Assist-to-turnover ratio
    total_assists = df["assists"].sum()
    total_turnovers = df["turnovers"].sum() if "turnovers" in df.columns else 0
    ast_to_ratio = (total_assists / total_turnovers) if total_turnovers > 0 else total_assists
    
    # Home/Away splits
    home_df = df[df["is_home"] == 1]
    away_df = df[df["is_home"] == 0]
    ppg_home = safe_mean(home_df["points"], ppg)
    ppg_away = safe_mean(away_df["points"], ppg)
    
    # Vs specific opponent
    ppg_vs_opp = 0.0
    games_vs_opp = 0
    if opponent_team_id is not None:
        opp_df = df[df["opponent_team_id"] == opponent_team_id]
        if not opp_df.empty:
            ppg_vs_opp = safe_mean(opp_df["points"], ppg)
            games_vs_opp = len(opp_df)
    
    return PlayerStats(
        player_id=player_id,
        name=player_name,
        position=position,
        is_injured=is_injured,
        ppg=ppg, rpg=rpg, apg=apg, mpg=mpg, games_played=games_played,
        ppg_home=ppg_home, ppg_away=ppg_away,
        ppg_vs_opp=ppg_vs_opp, games_vs_opp=games_vs_opp,
        # Extended stats
        spg=spg, bpg=bpg, tpg=tpg,
        fg_pct=fg_pct, fg3_pct=fg3_pct, ft_pct=ft_pct,
        ts_pct=ts_pct, efg_pct=efg_pct,
        fg3_rate=fg3_rate, ft_rate=ft_rate, fta_pg=fta_pg,
        ast_to_ratio=ast_to_ratio,
    )


def _get_position_group(position: str) -> str:
    """
    Normalize positions into groups: Guard (G), Forward (F), Center (C).
    Combo positions like G-F get grouped by primary position.
    """
    pos = position.upper().strip()
    if not pos:
        return "F"  # Default to forward if unknown
    
    # Handle common position formats
    if pos in ("PG", "SG", "G"):
        return "G"
    if pos in ("SF", "PF", "F"):
        return "F"
    if pos in ("C",):
        return "C"
    
    # Handle combo positions like "G-F", "F-C", "PG-SG"
    if "-" in pos:
        parts = pos.split("-")
        # Use first part as primary
        return _get_position_group(parts[0])
    
    # Handle formats like "Guard", "Forward", "Center"
    if "GUARD" in pos:
        return "G"
    if "FORWARD" in pos:
        return "F"
    if "CENTER" in pos:
        return "C"
    
    return "F"  # Default


def get_team_matchup_stats(
    team_id: int,
    opponent_team_id: Optional[int] = None,
    is_home: bool = True,
) -> TeamMatchupStats:
    """
    Get comprehensive team stats with player breakdowns for a matchup.
    
    Accounts for injuries by:
    1. Excluding injured players from projections
    2. Redistributing injured players' minutes to same-position players
    3. High scorers get usage boost when other high scorers are out
    """
    with get_conn() as conn:
        team_info = conn.execute(
            "SELECT team_id, abbreviation, name FROM teams WHERE team_id = ?",
            (team_id,)
        ).fetchone()
        if not team_info:
            return TeamMatchupStats(
                team_id=team_id, team_abbr="???", team_name="Unknown",
                players=[], total_ppg=0.0, total_rpg=0.0, total_apg=0.0,
                projected_points=0.0,
            )
        
        players_rows = conn.execute(
            "SELECT player_id, name, position, is_injured FROM players WHERE team_id = ?",
            (team_id,)
        ).fetchall()
    
    players: List[PlayerStats] = []
    for pid, pname, pos, injured in players_rows:
        pstats = get_player_comprehensive_stats(
            pid, pname, pos or "", bool(injured), opponent_team_id
        )
        players.append(pstats)
    
    # Sort by minutes played (most playing time first)
    players.sort(key=lambda p: p.mpg, reverse=True)
    
    # Separate active and injured players
    active = [p for p in players if not p.is_injured]
    injured = [p for p in players if p.is_injured]
    
    # College: A team plays 40 min * 5 players = 200 player-minutes per game
    # (NBA was 48 min * 5 = 240)
    TEAM_MINUTES_PER_GAME = 200.0
    
    # Calculate points-per-minute for redistribution estimation
    def get_ppm(p: PlayerStats) -> float:
        """Points per minute for a player."""
        return p.ppg / p.mpg if p.mpg > 0 else 0.0
    
    # Group injured players by position
    injured_by_pos: Dict[str, List[PlayerStats]] = {"G": [], "F": [], "C": []}
    for p in injured:
        pos_group = _get_position_group(p.position)
        injured_by_pos[pos_group].append(p)
    
    # Calculate injured minutes and points by position
    injured_minutes_by_pos = {pos: sum(p.mpg for p in plist) for pos, plist in injured_by_pos.items()}
    injured_ppg_by_pos = {pos: sum(p.ppg for p in plist) for pos, plist in injured_by_pos.items()}
    
    # High-scoring injured players (15+ PPG) create usage boost for remaining scorers
    high_scorer_injured_ppg = sum(p.ppg for p in injured if p.ppg >= 15)

    # ============ INJURED REBOUNDS / ASSISTS / DEFENSE LOSS ============
    # Track lost production beyond just points
    injured_rpg_by_pos = {pos: sum(p.rpg for p in plist) for pos, plist in injured_by_pos.items()}
    injured_apg_by_pos = {pos: sum(p.apg for p in plist) for pos, plist in injured_by_pos.items()}
    injured_spg_by_pos = {pos: sum(p.spg for p in plist) for pos, plist in injured_by_pos.items()}
    injured_bpg_by_pos = {pos: sum(p.bpg for p in plist) for pos, plist in injured_by_pos.items()}

    # ============ FT EFFICIENCY LOSS CALCULATION ============
    # When a high-FT% shooter is injured, their replacement may be worse at FTs
    # This matters especially in late-game foul situations
    ft_efficiency_penalty = 0.0
    
    # Group active players by position (need this first for FT calc)
    active_by_pos: Dict[str, List[PlayerStats]] = {"G": [], "F": [], "C": []}
    for p in active:
        pos_group = _get_position_group(p.position)
        active_by_pos[pos_group].append(p)
    
    for inj_player in injured:
        # Only consider players with meaningful FT volume (2+ FTA/game)
        if inj_player.fta_pg < 2.0 or inj_player.ft_pct <= 0:
            continue
        
        pos_group = _get_position_group(inj_player.position)
        
        # Find the likely replacement at the same position
        # Sort active players at this position by minutes (most likely to absorb minutes)
        position_replacements = sorted(
            active_by_pos.get(pos_group, []),
            key=lambda p: p.mpg,
            reverse=True
        )
        
        if not position_replacements:
            # No same-position replacement, look at adjacent positions
            adjacent = {"G": ["F"], "F": ["G", "C"], "C": ["F"]}
            for adj_pos in adjacent.get(pos_group, []):
                adj_players = active_by_pos.get(adj_pos, [])
                if adj_players:
                    position_replacements = sorted(adj_players, key=lambda p: p.mpg, reverse=True)
                    break
        
        if position_replacements:
            # Use top 1-2 replacements weighted by their likely minute share
            replacement_ft_pct = 0.0
            weight_sum = 0.0
            
            for i, repl in enumerate(position_replacements[:2]):
                # First replacement gets more weight
                weight = 1.0 if i == 0 else 0.5
                if repl.ft_pct > 0:
                    replacement_ft_pct += repl.ft_pct * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                replacement_ft_pct /= weight_sum
            else:
                # Fallback to college average FT% (~70%)
                replacement_ft_pct = 70.0
            
            # Calculate the FT% differential
            ft_pct_diff = inj_player.ft_pct - replacement_ft_pct
            
            # If injured player was better at FTs, we lose expected points
            # Loss = FTA_per_game * (FT%_injured - FT%_replacement) / 100
            if ft_pct_diff > 0:
                expected_ft_loss = inj_player.fta_pg * (ft_pct_diff / 100.0)
                ft_efficiency_penalty += expected_ft_loss
    
    # Calculate total active minutes by position for redistribution shares
    active_mpg_by_pos = {pos: sum(p.mpg for p in plist) for pos, plist in active_by_pos.items()}
    
    # Identify high scorers among active players (for usage boost)
    active_high_scorers = [p for p in active if p.ppg >= 12]
    total_high_scorer_ppg = sum(p.ppg for p in active_high_scorers)
    
    # Calculate base projection and redistribution boost
    total_ppg = 0.0
    total_rpg = 0.0
    total_apg = 0.0
    total_spg = 0.0
    total_bpg = 0.0
    total_tpg = 0.0
    projected = 0.0
    
    # For team-level efficiency, we'll weight by minutes played
    weighted_ts_sum = 0.0
    weighted_fg3_rate_sum = 0.0
    weighted_ft_rate_sum = 0.0
    total_minutes_weight = 0.0
    
    for p in active:
        pos_group = _get_position_group(p.position)
        pos_injured_minutes = injured_minutes_by_pos.get(pos_group, 0)
        pos_active_mpg = active_mpg_by_pos.get(pos_group, 0)

        extra_minutes = 0.0
        extra_points = 0.0
        extra_rebounds = 0.0
        extra_assists = 0.0

        # 1. Position-based minute redistribution
        if pos_active_mpg > 0 and pos_injured_minutes > 0:
            pos_share = p.mpg / pos_active_mpg
            extra_minutes = pos_injured_minutes * pos_share
            max_extra = max(0, 38 - p.mpg)
            extra_minutes = min(extra_minutes, max_extra)
            # Points from extra minutes (with fatigue discount)
            extra_points = extra_minutes * get_ppm(p) * 0.85

            # Rebounds redistribution: replacements recover ~70 % efficiency
            pos_lost_rpg = injured_rpg_by_pos.get(pos_group, 0)
            if pos_lost_rpg > 0:
                rpm = (p.rpg / p.mpg) if p.mpg > 0 else 0.0
                extra_rebounds = extra_minutes * rpm * 0.70

            # Assists redistribution: playmaking is harder to replace (~60 %)
            pos_lost_apg = injured_apg_by_pos.get(pos_group, 0)
            if pos_lost_apg > 0:
                apm = (p.apg / p.mpg) if p.mpg > 0 else 0.0
                extra_assists = extra_minutes * apm * 0.60

        # 2. Usage boost for high scorers when other high scorers are out
        if high_scorer_injured_ppg > 0 and p.ppg >= 12 and total_high_scorer_ppg > 0:
            usage_share = p.ppg / total_high_scorer_ppg
            usage_boost = high_scorer_injured_ppg * usage_share * 0.30
            extra_points += usage_boost

        # 3. Adjacent position spillover
        adjacent_positions = {"G": ["F"], "F": ["G", "C"], "C": ["F"]}
        for adj_pos in adjacent_positions.get(pos_group, []):
            adj_injured_minutes = injured_minutes_by_pos.get(adj_pos, 0)
            adj_active_mpg = active_mpg_by_pos.get(adj_pos, 0)

            if adj_injured_minutes > 5 and adj_active_mpg < 50:
                spillover_minutes = min(3, adj_injured_minutes * 0.15)
                spillover_points = spillover_minutes * get_ppm(p) * 0.75
                extra_points += spillover_points

        # Base stats + redistribution for rebounds/assists
        total_ppg += p.ppg
        total_rpg += p.rpg + extra_rebounds
        total_apg += p.apg + extra_assists
        total_spg += p.spg
        total_bpg += p.bpg
        total_tpg += p.tpg

        # Weight efficiency metrics by minutes
        if p.mpg > 0:
            weighted_ts_sum += p.ts_pct * p.mpg
            weighted_fg3_rate_sum += p.fg3_rate * p.mpg
            weighted_ft_rate_sum += p.ft_rate * p.mpg
            total_minutes_weight += p.mpg

        # For projection, weight by context
        base = p.ppg * 0.4
        loc_val = p.ppg_home if is_home else p.ppg_away
        location = (loc_val if loc_val > 0 else p.ppg) * 0.3
        vs_val = p.ppg_vs_opp if (p.games_vs_opp > 0 and p.ppg_vs_opp > 0) else p.ppg
        vs_opp = vs_val * 0.3
        player_proj = base + location + vs_opp + extra_points
        projected += player_proj
    
    # Scale projection to realistic team total if needed
    total_active_mpg = sum(p.mpg for p in active) + sum(injured_minutes_by_pos.values())
    if total_active_mpg > TEAM_MINUTES_PER_GAME:
        scale_factor = TEAM_MINUTES_PER_GAME / total_active_mpg
        projected *= scale_factor
        total_ppg *= scale_factor
        total_rpg *= scale_factor
        total_apg *= scale_factor
        total_spg *= scale_factor
        total_bpg *= scale_factor
        total_tpg *= scale_factor
    
    # Apply FT efficiency penalty from injuries
    # This accounts for losing good FT shooters and replacing with worse ones
    if ft_efficiency_penalty > 0:
        projected -= ft_efficiency_penalty
        total_ppg -= ft_efficiency_penalty
    
    # Calculate team-level efficiency metrics (weighted by minutes)
    team_ts_pct = (weighted_ts_sum / total_minutes_weight) if total_minutes_weight > 0 else 0.0
    team_fg3_rate = (weighted_fg3_rate_sum / total_minutes_weight) if total_minutes_weight > 0 else 0.0
    team_ft_rate = (weighted_ft_rate_sum / total_minutes_weight) if total_minutes_weight > 0 else 0.0
    
    # Turnover margin: positive means team creates more turnovers than commits
    turnover_margin = total_spg - total_tpg

    # ============ TEAM-LEVEL RATINGS, PACE, SOS, RECENT FORM ============
    ratings = calculate_team_ratings(team_id)
    sos = calculate_strength_of_schedule(team_id)
    recent_ppg = calculate_recent_form(team_id, last_n=5)

    return TeamMatchupStats(
        team_id=team_id,
        team_abbr=team_info[1],
        team_name=team_info[2],
        players=players,
        total_ppg=total_ppg,
        total_rpg=total_rpg,
        total_apg=total_apg,
        projected_points=projected,
        # Extended stats
        total_spg=total_spg,
        total_bpg=total_bpg,
        total_tpg=total_tpg,
        team_ts_pct=team_ts_pct,
        team_fg3_rate=team_fg3_rate,
        team_ft_rate=team_ft_rate,
        turnover_margin=turnover_margin,
        # Pace-adjusted efficiency
        offensive_rating=ratings["offensive_rating"],
        defensive_rating=ratings["defensive_rating"],
        net_rating=ratings["net_rating"],
        pace=ratings["pace"],
        # Strength of schedule & recent form
        sos=sos,
        recent_ppg=recent_ppg,
    )


def player_splits(
    player_id: int,
    opponent_team_id: Optional[int] = None,
    is_home: Optional[bool] = None,
    recent_games: Optional[int] = 10,
    use_recency_weights: bool = True,
) -> Dict[str, float]:
    """
    Get player stats splits with extended metrics and optional recency weighting.

    When *use_recency_weights* is True the average is computed as a tiered
    blend so that recent performance counts more than early-season games:
        - Last 5 games:  50 % weight
        - Games 6-10:    30 % weight
        - Games 11+:     20 % weight
    The shooting *totals* (fg_made, fg_attempted, ...) are still raw sums
    from the selected window so that aggregate efficiency calculations stay
    consistent.

    Returns comprehensive stats including shooting efficiency and defensive stats.
    """
    from src.analytics.prediction import PREDICTION_CONFIG

    df = _load_player_df(player_id)
    _empty: Dict[str, float] = {
        "points": 0.0, "rebounds": 0.0, "assists": 0.0, "minutes": 0.0,
        "steals": 0.0, "blocks": 0.0, "turnovers": 0.0,
        "fg_made": 0, "fg_attempted": 0, "fg3_made": 0, "fg3_attempted": 0,
        "ft_made": 0, "ft_attempted": 0,
        "ts_pct": 0.0, "fg3_rate": 0.0,
    }
    if df.empty:
        return _empty

    # Store original df for fallback
    original_df = df.copy()

    # Apply filters
    if opponent_team_id is not None:
        filtered = df[df["opponent_team_id"] == opponent_team_id]
        if not filtered.empty:
            df = filtered

    if is_home is not None:
        filtered = df[df["is_home"] == int(is_home)]
        if not filtered.empty:
            df = filtered
        elif not original_df.empty:
            df = original_df

    if recent_games:
        df = df.head(recent_games)

    # ---- helpers ----
    def safe_sum(col, default=0):
        if col not in df.columns or df.empty:
            return default
        return int(df[col].sum())

    stat_cols = ["points", "rebounds", "assists", "minutes",
                 "steals", "blocks", "turnovers"]

    # ---- recency-weighted means ----
    if use_recency_weights and len(df) > 5:
        w5 = PREDICTION_CONFIG["last5_weight"]    # 0.50
        w10 = PREDICTION_CONFIG["last10_weight"]   # 0.30
        ws = PREDICTION_CONFIG["season_weight"]     # 0.20

        last5 = df.head(5)
        mid = df.iloc[5:10]
        rest = df.iloc[10:]

        def _tier_mean(col: str, default: float = 0.0) -> float:
            if col not in df.columns:
                return default
            m5 = last5[col].mean() if not last5.empty else 0.0
            m10 = mid[col].mean() if not mid.empty else m5
            ms = rest[col].mean() if not rest.empty else m10
            m5 = m5 if pd.notna(m5) else 0.0
            m10 = m10 if pd.notna(m10) else m5
            ms = ms if pd.notna(ms) else m10
            return m5 * w5 + m10 * w10 + ms * ws

        means = {col: _tier_mean(col) for col in stat_cols}
    else:
        def _safe_mean(col: str, default: float = 0.0) -> float:
            if col not in df.columns or df.empty:
                return default
            val = df[col].mean()
            return val if pd.notna(val) else default

        means = {col: _safe_mean(col) for col in stat_cols}

    # ---- shooting totals (raw sums – not weighted) ----
    total_pts = safe_sum("points")
    total_fga = safe_sum("fg_attempted")
    total_fg3a = safe_sum("fg3_attempted")
    total_fta = safe_sum("ft_attempted")

    ts_denom = 2 * (total_fga + 0.44 * total_fta)
    ts_pct = (total_pts / ts_denom * 100) if ts_denom > 0 else 0.0
    fg3_rate = (total_fg3a / total_fga * 100) if total_fga > 0 else 0.0

    return {
        **means,
        # Shooting totals (for aggregate efficiency)
        "fg_made": safe_sum("fg_made"),
        "fg_attempted": safe_sum("fg_attempted"),
        "fg3_made": safe_sum("fg3_made"),
        "fg3_attempted": safe_sum("fg3_attempted"),
        "ft_made": safe_sum("ft_made"),
        "ft_attempted": safe_sum("ft_attempted"),
        # Efficiency metrics
        "ts_pct": ts_pct,
        "fg3_rate": fg3_rate,
    }


def team_strength(team_id: int, opponent_team_id: Optional[int] = None) -> Dict[str, float]:
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ps.*
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ?
            """,
            conn,
            params=[team_id],
        )
    if opponent_team_id:
        df = df[df["opponent_team_id"] == opponent_team_id]
    if df.empty:
        return {"offense": 0.0, "rebounds": 0.0, "assists": 0.0}
    return {
        "offense": df["points"].mean(),
        "rebounds": df["rebounds"].mean(),
        "assists": df["assists"].mean(),
    }


def calculate_team_pace(team_id: int) -> float:
    """
    Estimate a team's average possessions per game.

    Uses the standard possession estimate:
        Possessions ~ FGA - OREB + TO + 0.44 * FTA
    Offensive rebounds aren't tracked separately so we estimate them as
    ~30 % of total rebounds (college average).
    """
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT
                SUM(ps.fg_attempted)  AS fga,
                SUM(ps.rebounds)      AS reb,
                SUM(ps.turnovers)     AS tov,
                SUM(ps.ft_attempted)  AS fta,
                COUNT(DISTINCT ps.game_date) AS games
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ?
            """,
            conn,
            params=[team_id],
        )
    if df.empty or df.iloc[0]["games"] == 0:
        return 0.0

    row = df.iloc[0]
    games = int(row["games"])
    fga = float(row["fga"] or 0) / games
    reb = float(row["reb"] or 0) / games
    tov = float(row["tov"] or 0) / games
    fta = float(row["fta"] or 0) / games
    oreb_est = reb * 0.30  # ~30 % of total rebounds are offensive (college avg)
    pace = fga - oreb_est + tov + 0.44 * fta
    return pace


def calculate_team_ratings(team_id: int) -> Dict[str, float]:
    """
    Compute Offensive Rating, Defensive Rating and Net Rating for a team.

    ORtg = team points per 100 possessions
    DRtg = opponent points per 100 possessions (from games where
           the opponent's stats are linked via opponent_team_id)
    Net  = ORtg - DRtg
    """
    pace = calculate_team_pace(team_id)
    if pace <= 0:
        return {"offensive_rating": 0.0, "defensive_rating": 0.0, "net_rating": 0.0, "pace": 0.0}

    with get_conn() as conn:
        # Team's own per-game scoring
        own = pd.read_sql(
            """
            SELECT ps.game_date, SUM(ps.points) AS team_pts
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ?
            GROUP BY ps.game_date
            """,
            conn,
            params=[team_id],
        )
        # Opponent scoring in those same games
        opp = pd.read_sql(
            """
            SELECT ps.game_date, SUM(ps.points) AS opp_pts
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE ps.opponent_team_id = ?
            GROUP BY ps.game_date
            """,
            conn,
            params=[team_id],
        )
    if own.empty:
        return {"offensive_rating": 0.0, "defensive_rating": 0.0, "net_rating": 0.0, "pace": pace}

    avg_pts = own["team_pts"].mean()
    avg_opp = opp["opp_pts"].mean() if not opp.empty else avg_pts

    ortg = (avg_pts / pace) * 100 if pace > 0 else 0.0
    drtg = (avg_opp / pace) * 100 if pace > 0 else 0.0

    return {
        "offensive_rating": ortg,
        "defensive_rating": drtg,
        "net_rating": ortg - drtg,
        "pace": pace,
    }


def calculate_strength_of_schedule(team_id: int) -> float:
    """
    Compute a normalised Strength-of-Schedule metric for a team.

    Method: average scoring margin of all opponents the team has faced.
    The result is centred around 0.0 (league average).  A positive SOS
    means the team played opponents who score more than they allow on
    average (tougher schedule).

    Returns a float in roughly the range [-10, +10].
    """
    with get_conn() as conn:
        # Get distinct opponents this team has faced
        opp_ids = pd.read_sql(
            """
            SELECT DISTINCT ps.opponent_team_id AS opp_id
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ?
            """,
            conn,
            params=[team_id],
        )
    if opp_ids.empty:
        return 0.0

    margins: List[float] = []
    for _, row in opp_ids.iterrows():
        oid = int(row["opp_id"])
        with get_conn() as conn:
            # Opponent's own scoring per game
            own = pd.read_sql(
                """
                SELECT ps.game_date, SUM(ps.points) AS pts
                FROM player_stats ps
                JOIN players p ON p.player_id = ps.player_id
                WHERE p.team_id = ?
                GROUP BY ps.game_date
                """,
                conn,
                params=[oid],
            )
            # Points allowed by opponent per game
            allowed = pd.read_sql(
                """
                SELECT ps.game_date, SUM(ps.points) AS pts
                FROM player_stats ps
                JOIN players p ON p.player_id = ps.player_id
                WHERE ps.opponent_team_id = ?
                GROUP BY ps.game_date
                """,
                conn,
                params=[oid],
            )
        if own.empty or allowed.empty:
            continue
        margin = own["pts"].mean() - allowed["pts"].mean()
        margins.append(margin)

    return sum(margins) / len(margins) if margins else 0.0


def calculate_recent_form(team_id: int, last_n: int = 5) -> float:
    """
    Return the team's average PPG over their most recent *last_n* games.
    Useful for trend / momentum display on the UI.
    """
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ps.game_date, SUM(ps.points) AS team_pts
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ?
            GROUP BY ps.game_date
            ORDER BY ps.game_date DESC
            LIMIT ?
            """,
            conn,
            params=[team_id, last_n],
        )
    if df.empty:
        return 0.0
    return float(df["team_pts"].mean())


def aggregate_projection(
    player_ids: Iterable[int],
    opponent_team_id: Optional[int] = None,
    is_home: Optional[bool] = None,
    recent_games: int = 10,
) -> Dict[str, float]:
    """
    Aggregate player stats for team projection with extended metrics.

    Returns:
        Dict with team totals for all stats plus efficiency metrics.
    """
    totals = {
        "points": 0.0, "rebounds": 0.0, "assists": 0.0,
        "steals": 0.0, "blocks": 0.0, "turnovers": 0.0,
        "fg_made": 0, "fg_attempted": 0,
        "fg3_made": 0, "fg3_attempted": 0,
        "ft_made": 0, "ft_attempted": 0,
    }
    
    for pid in player_ids:
        splits = player_splits(
            pid,
            opponent_team_id=opponent_team_id,
            is_home=is_home,
            recent_games=recent_games,
        )
        for k in totals:
            totals[k] = totals[k] + splits.get(k, 0.0)
    
    # Calculate team-level efficiency from aggregated shooting stats
    total_fga = totals["fg_attempted"]
    total_fg3a = totals["fg3_attempted"]
    total_fta = totals["ft_attempted"]
    total_pts = totals["points"]
    
    # True Shooting % for the team
    ts_denom = 2 * (total_fga + 0.44 * total_fta)
    totals["ts_pct"] = (total_pts / ts_denom * 100) if ts_denom > 0 else 0.0
    
    # 3PT attempt rate
    totals["fg3_rate"] = (total_fg3a / total_fga * 100) if total_fga > 0 else 0.0
    
    # FT attempt rate  
    totals["ft_rate"] = (total_fta / total_fga * 100) if total_fga > 0 else 0.0
    
    # Turnover margin (positive = good, creates more than commits)
    totals["turnover_margin"] = totals["steals"] - totals["turnovers"]
    
    return totals


def usage_adjusted_projection(projection: Dict[str, float], usage_factor: float = 1.0) -> Dict[str, float]:
    """Scale projections based on expected usage changes (e.g., injuries)."""
    return {k: v * usage_factor for k, v in projection.items()}


def get_scheduled_games(include_future_days: int = 14) -> List[Dict]:
    """Get games from the schedule, including upcoming games."""
    from src.data.sync_service import sync_schedule

    try:
        df = sync_schedule(include_future_days=include_future_days)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return []
    
    # Get team lookup
    with get_conn() as conn:
        teams = pd.read_sql("SELECT team_id, abbreviation, name FROM teams", conn)
    team_lookup = {row["abbreviation"]: (int(row["team_id"]), row["name"]) for _, row in teams.iterrows()}
    
    games = []
    seen = set()
    
    for _, row in df.iterrows():
        game_date = row.get("game_date")
        team_id = row.get("team_id")
        opp_abbr = row.get("opponent_abbr")
        is_home = row.get("is_home", False)
        
        # Look up team info
        team_info = None
        for abbr, (tid, name) in team_lookup.items():
            if tid == team_id:
                team_info = (abbr, name)
                break
        
        if not team_info or opp_abbr not in team_lookup:
            continue
        
        opp_id, opp_name = team_lookup[opp_abbr]
        
        if is_home:
            home_id, home_abbr, home_name = team_id, team_info[0], team_info[1]
            away_id, away_abbr, away_name = opp_id, opp_abbr, opp_name
        else:
            away_id, away_abbr, away_name = team_id, team_info[0], team_info[1]
            home_id, home_abbr, home_name = opp_id, opp_abbr, opp_name
        
        # Deduplicate (each game appears twice - once for each team)
        key = tuple(sorted([home_id, away_id])) + (str(game_date),)
        if key in seen:
            continue
        seen.add(key)
        
        games.append({
            "game_date": game_date,
            "home_team_id": int(home_id),
            "home_abbr": home_abbr,
            "home_name": home_name,
            "away_team_id": int(away_id),
            "away_abbr": away_abbr,
            "away_name": away_name,
        })
    
    # Sort by date descending (most recent first)
    games.sort(key=lambda g: str(g.get("game_date", "")), reverse=True)
    return games[:100]
