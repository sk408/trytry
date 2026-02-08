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
    # Impact
    plus_minus_avg: float = 0.0  # average plus/minus
    # Rebound breakdown
    oreb_pg: float = 0.0  # offensive rebounds per game
    dreb_pg: float = 0.0  # defensive rebounds per game


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
    # Rebound breakdown
    total_oreb: float = 0.0  # team offensive rebounds per game
    total_dreb: float = 0.0  # team defensive rebounds per game
    # Team shooting efficiency (weighted by attempts)
    team_ts_pct: float = 0.0    # team true shooting %
    team_fg3_rate: float = 0.0  # team 3PT attempt rate
    team_ft_rate: float = 0.0   # team FT attempt rate
    # Net ratings
    turnover_margin: float = 0.0  # steals - turnovers (positive = good)
    # Advanced ratings
    off_rating: float = 0.0   # offensive rating (points per 100 possessions)
    def_rating: float = 0.0   # defensive rating (opponent pts per 100 possessions)


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
    
    # Plus/minus average
    plus_minus_avg = safe_mean(df["plus_minus"]) if "plus_minus" in df.columns else 0.0
    
    # Rebound breakdown per game
    oreb_pg = safe_mean(df["oreb"]) if "oreb" in df.columns else 0.0
    dreb_pg = safe_mean(df["dreb"]) if "dreb" in df.columns else 0.0
    
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
        plus_minus_avg=plus_minus_avg,
        oreb_pg=oreb_pg,
        dreb_pg=dreb_pg,
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
    
    # A team plays 48 min * 5 players = 240 player-minutes per game
    TEAM_MINUTES_PER_GAME = 240.0
    
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
    
    # ============ FT EFFICIENCY LOSS CALCULATION ============
    # When a high-FT% shooter is injured, their replacement may be worse at FTs
    # This matters especially in late-game foul situations
    ft_efficiency_penalty = 0.0
    
    for inj_player in injured:
        # Only consider players with meaningful FT volume (2+ FTA/game)
        if inj_player.fta_pg < 2.0 or inj_player.ft_pct <= 0:
            continue
        
        pos_group = _get_position_group(inj_player.position)
        
        # Find the likely replacement at the same position
        # Sort active players at this position by minutes (most likely to absorb minutes)
        position_replacements = sorted(
            [p for p in active if _get_position_group(p.position) == pos_group],
            key=lambda p: p.mpg,
            reverse=True
        )
        
        if not position_replacements:
            # No same-position replacement, look at adjacent positions
            adjacent = {"G": ["F"], "F": ["G", "C"], "C": ["F"]}
            for adj_pos in adjacent.get(pos_group, []):
                adj_players = [p for p in active if _get_position_group(p.position) == adj_pos]
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
                # Fallback to league average FT% (~78%)
                replacement_ft_pct = 78.0
            
            # Calculate the FT% differential
            ft_pct_diff = inj_player.ft_pct - replacement_ft_pct
            
            # If injured player was better at FTs, we lose expected points
            # Loss = FTA_per_game * (FT%_injured - FT%_replacement) / 100
            if ft_pct_diff > 0:
                expected_ft_loss = inj_player.fta_pg * (ft_pct_diff / 100.0)
                ft_efficiency_penalty += expected_ft_loss
    
    # Group active players by position
    active_by_pos: Dict[str, List[PlayerStats]] = {"G": [], "F": [], "C": []}
    for p in active:
        pos_group = _get_position_group(p.position)
        active_by_pos[pos_group].append(p)
    
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
    total_oreb = 0.0
    total_dreb = 0.0
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
        
        # 1. Position-based minute redistribution
        if pos_active_mpg > 0 and pos_injured_minutes > 0:
            # Player's share of position minutes
            pos_share = p.mpg / pos_active_mpg
            # Extra minutes from injured players at same position
            extra_minutes = pos_injured_minutes * pos_share
            # Cap at ~40 total minutes
            max_extra = max(0, 40 - p.mpg)
            extra_minutes = min(extra_minutes, max_extra)
            # Points from extra minutes (with fatigue discount)
            extra_points = extra_minutes * get_ppm(p) * 0.85
        
        # 2. Usage boost for high scorers when other high scorers are out
        # When a 20 PPG player is out, other scorers get more touches
        if high_scorer_injured_ppg > 0 and p.ppg >= 12 and total_high_scorer_ppg > 0:
            # This player's share of high-scorer production
            usage_share = p.ppg / total_high_scorer_ppg
            # Redistribute ~30% of lost scoring (rest goes to role players, bench, etc.)
            usage_boost = high_scorer_injured_ppg * usage_share * 0.30
            extra_points += usage_boost
        
        # 3. Adjacent position spillover
        # If no guards available, forwards/wings might play some guard minutes
        # And vice versa - this handles position flexibility
        adjacent_positions = {"G": ["F"], "F": ["G", "C"], "C": ["F"]}
        for adj_pos in adjacent_positions.get(pos_group, []):
            adj_injured_minutes = injured_minutes_by_pos.get(adj_pos, 0)
            adj_active_mpg = active_mpg_by_pos.get(adj_pos, 0)
            
            # If adjacent position has injuries and few active players there
            if adj_injured_minutes > 5 and adj_active_mpg < 50:
                # This player might pick up some of those minutes (spillover)
                spillover_minutes = min(3, adj_injured_minutes * 0.15)  # Small amount
                spillover_points = spillover_minutes * get_ppm(p) * 0.75  # Lower efficiency
                extra_points += spillover_points
        
        # Base stats (no change)
        total_ppg += p.ppg
        total_rpg += p.rpg
        total_apg += p.apg
        total_spg += p.spg
        total_bpg += p.bpg
        total_tpg += p.tpg
        total_oreb += p.oreb_pg
        total_dreb += p.dreb_pg
        
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
        total_oreb *= scale_factor
        total_dreb *= scale_factor
    
    # Apply FT efficiency penalty from injuries
    # This accounts for losing good FT shooters and replacing with worse ones
    if ft_efficiency_penalty > 0:
        projected -= ft_efficiency_penalty
        total_ppg -= ft_efficiency_penalty
    
    # ============ PLAYMAKER INJURY PENALTY ============
    # Losing a primary playmaker (6+ APG) hurts team offensive efficiency
    # beyond just their personal scoring -- assisted shots are higher %
    injured_assists = sum(p.apg for p in injured if p.apg >= 6.0)
    if injured_assists > 0:
        playmaker_penalty = injured_assists * 0.5
        projected -= playmaker_penalty
        total_ppg -= playmaker_penalty
    
    # ============ REBOUNDER INJURY PENALTY ============
    # Losing a dominant rebounder (8+ RPG) reduces second-chance points
    injured_rebounds = sum(p.rpg for p in injured if p.rpg >= 8.0)
    if injured_rebounds > 0:
        rebound_penalty = injured_rebounds * 0.2
        projected -= rebound_penalty
        total_ppg -= rebound_penalty
    
    # Calculate team-level efficiency metrics (weighted by minutes)
    team_ts_pct = (weighted_ts_sum / total_minutes_weight) if total_minutes_weight > 0 else 0.0
    team_fg3_rate = (weighted_fg3_rate_sum / total_minutes_weight) if total_minutes_weight > 0 else 0.0
    team_ft_rate = (weighted_ft_rate_sum / total_minutes_weight) if total_minutes_weight > 0 else 0.0
    
    # Turnover margin: positive means team creates more turnovers than commits
    turnover_margin = total_spg - total_tpg
    
    # Calculate offensive and defensive ratings from historical game data
    off_rating = get_offensive_rating(team_id)
    def_rating = get_defensive_rating(team_id)
    
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
        total_oreb=total_oreb,
        total_dreb=total_dreb,
        team_ts_pct=team_ts_pct,
        team_fg3_rate=team_fg3_rate,
        team_ft_rate=team_ft_rate,
        turnover_margin=turnover_margin,
        off_rating=off_rating,
        def_rating=def_rating,
    )


def player_splits(
    player_id: int,
    opponent_team_id: Optional[int] = None,
    is_home: Optional[bool] = None,
    recent_games: Optional[int] = 10,
) -> Dict[str, float]:
    """
    Get player stats splits with extended metrics.
    
    Returns comprehensive stats including shooting efficiency and defensive stats.
    """
    df = _load_player_df(player_id)
    if df.empty:
        return {
            "points": 0.0, "rebounds": 0.0, "assists": 0.0, "minutes": 0.0,
            "steals": 0.0, "blocks": 0.0, "turnovers": 0.0,
            "oreb": 0.0, "dreb": 0.0,
            "fg_made": 0, "fg_attempted": 0, "fg3_made": 0, "fg3_attempted": 0,
            "ft_made": 0, "ft_attempted": 0,
            "fga_pg": 0.0, "fta_pg": 0.0,
            "ts_pct": 0.0, "fg3_rate": 0.0,
        }

    # Store original df for fallback
    original_df = df.copy()
    
    # Apply filters
    if opponent_team_id is not None:
        filtered = df[df["opponent_team_id"] == opponent_team_id]
        # Only use filtered if we have data, otherwise keep original
        if not filtered.empty:
            df = filtered
    
    if is_home is not None:
        filtered = df[df["is_home"] == int(is_home)]
        # Only use filtered if we have data, otherwise fall back
        if not filtered.empty:
            df = filtered
        elif not original_df.empty:
            # Fall back to recent games if no home/away split data
            df = original_df
    
    if recent_games:
        df = df.head(recent_games)

    # Recency weighting: last 5 games get 60% weight, older games get 40%
    _use_recency = recent_games is not None and len(df) > 5

    def safe_mean(col, default=0.0):
        if col not in df.columns or df.empty:
            return default
        if _use_recency:
            recent_5 = df.head(5)
            older = df.iloc[5:]
            r_val = recent_5[col].mean() if not recent_5.empty else default
            o_val = older[col].mean() if not older.empty else default
            r_val = r_val if pd.notna(r_val) else default
            o_val = o_val if pd.notna(o_val) else default
            return r_val * 0.6 + o_val * 0.4
        val = df[col].mean()
        return val if pd.notna(val) else default
    
    def safe_sum(col, default=0):
        if col not in df.columns or df.empty:
            return default
        return int(df[col].sum())
    
    # Compute shooting totals for efficiency calculation
    total_pts = safe_sum("points")
    total_fga = safe_sum("fg_attempted")
    total_fg3a = safe_sum("fg3_attempted")
    total_fta = safe_sum("ft_attempted")
    
    # True Shooting %
    ts_denom = 2 * (total_fga + 0.44 * total_fta)
    ts_pct = (total_pts / ts_denom * 100) if ts_denom > 0 else 0.0
    
    # 3PT attempt rate
    fg3_rate = (total_fg3a / total_fga * 100) if total_fga > 0 else 0.0

    return {
        # Basic stats
        "points": safe_mean("points"),
        "rebounds": safe_mean("rebounds"),
        "assists": safe_mean("assists"),
        "minutes": safe_mean("minutes"),
        # Defensive stats
        "steals": safe_mean("steals"),
        "blocks": safe_mean("blocks"),
        "turnovers": safe_mean("turnovers"),
        # Rebound breakdown
        "oreb": safe_mean("oreb"),
        "dreb": safe_mean("dreb"),
        # Shooting totals (for aggregate efficiency)
        "fg_made": safe_sum("fg_made"),
        "fg_attempted": safe_sum("fg_attempted"),
        "fg3_made": safe_sum("fg3_made"),
        "fg3_attempted": safe_sum("fg3_attempted"),
        "ft_made": safe_sum("ft_made"),
        "ft_attempted": safe_sum("ft_attempted"),
        # Per-game shooting averages (for pace/possession calculation)
        "fga_pg": safe_mean("fg_attempted"),
        "fta_pg": safe_mean("ft_attempted"),
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
        "points": 0.0, "rebounds": 0.0, "assists": 0.0, "minutes": 0.0,
        "steals": 0.0, "blocks": 0.0, "turnovers": 0.0,
        "oreb": 0.0, "dreb": 0.0,
        "fg_made": 0, "fg_attempted": 0,
        "fg3_made": 0, "fg3_attempted": 0,
        "ft_made": 0, "ft_attempted": 0,
        "fga_pg": 0.0, "fta_pg": 0.0,
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
    
    # ============ POSSESSION & PACE ESTIMATION ============
    # Standard NBA possession formula: POSS = FGA - OREB + TOV + 0.44 * FTA
    # Using per-game averages for consistency
    team_fga_pg = totals["fga_pg"]
    team_oreb_pg = totals["oreb"]
    team_tov_pg = totals["turnovers"]
    team_fta_pg = totals["fta_pg"]
    possessions_pg = team_fga_pg - team_oreb_pg + team_tov_pg + 0.44 * team_fta_pg
    
    # Offensive Rating: points scored per 100 possessions
    totals["off_rating"] = (total_pts / possessions_pg * 100) if possessions_pg > 0 else 0.0
    
    # Pace: possessions per game (normalized to 240 player-minutes = 48 game-minutes)
    team_minutes = totals.get("minutes", 0)
    if team_minutes > 0 and possessions_pg > 0:
        totals["pace"] = possessions_pg * (240.0 / team_minutes)
    else:
        totals["pace"] = 96.0  # league average fallback
    
    return totals


def usage_adjusted_projection(projection: Dict[str, float], usage_factor: float = 1.0) -> Dict[str, float]:
    """Scale projections based on expected usage changes (e.g., injuries)."""
    return {k: v * usage_factor for k, v in projection.items()}


# ============ ADVANCED RATING FUNCTIONS ============


def get_offensive_rating(team_id: int) -> float:
    """
    Calculate team's offensive rating (points per 100 possessions).
    Higher = better offense.  Uses the last 20 games for stability.
    """
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT 
                ps.game_date,
                SUM(ps.points) as team_pts,
                SUM(ps.fg_attempted) as team_fga,
                SUM(ps.ft_attempted) as team_fta,
                SUM(ps.turnovers) as team_tov,
                SUM(ps.oreb) as team_oreb
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ?
            GROUP BY ps.game_date
            ORDER BY ps.game_date DESC
            LIMIT 20
            """,
            conn,
            params=[team_id],
        )
    
    if df.empty:
        return 110.0  # league average fallback
    
    avg_pts = float(df["team_pts"].mean())
    avg_fga = float(df["team_fga"].mean())
    avg_fta = float(df["team_fta"].mean())
    avg_tov = float(df["team_tov"].mean())
    avg_oreb = float(df["team_oreb"].mean())
    
    possessions = avg_fga - avg_oreb + avg_tov + 0.44 * avg_fta
    
    if possessions <= 0:
        return 110.0
    
    return (avg_pts / possessions) * 100


def get_defensive_rating(team_id: int) -> float:
    """
    Calculate team's defensive rating (opponent points per 100 possessions).
    Lower = better defense.  Uses opponent stats from the last 20 games.
    """
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT 
                ps.game_date,
                SUM(ps.points) as opp_pts,
                SUM(ps.fg_attempted) as opp_fga,
                SUM(ps.ft_attempted) as opp_fta,
                SUM(ps.turnovers) as opp_tov,
                SUM(ps.oreb) as opp_oreb
            FROM player_stats ps
            WHERE ps.opponent_team_id = ?
            GROUP BY ps.game_date
            ORDER BY ps.game_date DESC
            LIMIT 20
            """,
            conn,
            params=[team_id],
        )
    
    if df.empty:
        return 110.0  # league average fallback
    
    avg_pts = float(df["opp_pts"].mean())
    avg_fga = float(df["opp_fga"].mean())
    avg_fta = float(df["opp_fta"].mean())
    avg_tov = float(df["opp_tov"].mean())
    avg_oreb = float(df["opp_oreb"].mean())
    
    possessions = avg_fga - avg_oreb + avg_tov + 0.44 * avg_fta
    
    if possessions <= 0:
        return 110.0
    
    return (avg_pts / possessions) * 100


def get_opponent_defensive_factor(team_id: int) -> float:
    """
    Returns a scaling factor for how well opponents are held defensively.
    < 1.0 = good defense (opponents score less against this team)
    > 1.0 = bad defense (opponents score more against this team)
    Centered at 1.0 (league average).
    """
    with get_conn() as conn:
        # Average points scored against this team per game
        opp_row = conn.execute(
            """
            SELECT AVG(game_total) as avg_opp_pts FROM (
                SELECT game_date, SUM(points) as game_total
                FROM player_stats
                WHERE opponent_team_id = ?
                GROUP BY game_date
            )
            """,
            (team_id,),
        ).fetchone()
        
        # League-wide average points per team per game
        league_row = conn.execute(
            """
            SELECT AVG(game_total) as league_avg FROM (
                SELECT game_date, opponent_team_id, SUM(points) as game_total
                FROM player_stats
                GROUP BY game_date, opponent_team_id
            )
            """,
        ).fetchone()
    
    if not opp_row or not opp_row[0]:
        return 1.0
    
    avg_opp_pts = float(opp_row[0])
    league_avg = float(league_row[0]) if league_row and league_row[0] else 112.0
    
    if league_avg <= 0:
        return 1.0
    
    return avg_opp_pts / league_avg


def get_home_court_advantage(team_id: int) -> float:
    """
    Calculate team-specific home court advantage from historical scoring.
    Returns the expected point boost for playing at home (clamped to [1.5, 5.0]).
    """
    with get_conn() as conn:
        # Team scoring at home (game-level totals)
        home_df = pd.read_sql(
            """
            SELECT ps.game_date, SUM(ps.points) as team_pts
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ? AND ps.is_home = 1
            GROUP BY ps.game_date
            """,
            conn,
            params=[team_id],
        )
        # Team scoring away
        away_df = pd.read_sql(
            """
            SELECT ps.game_date, SUM(ps.points) as team_pts
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE p.team_id = ? AND ps.is_home = 0
            GROUP BY ps.game_date
            """,
            conn,
            params=[team_id],
        )
    
    if home_df.empty or away_df.empty:
        return 3.0  # league average fallback
    
    home_ppg = float(home_df["team_pts"].mean())
    away_ppg = float(away_df["team_pts"].mean())
    
    # HCA = how many more points team scores at home vs away
    hca = home_ppg - away_ppg
    
    # Clamp to reasonable range
    return max(1.5, min(5.0, hca))


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
