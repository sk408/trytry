from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.database.db import get_conn
from src.data.nba_fetcher import get_current_season


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
    # Fouls
    pf_pg: float = 0.0    # personal fouls per game
    # Injury intelligence
    play_probability: float = 1.0   # 0.0 (definitely out) … 1.0 (healthy)
    injury_keyword: str = ""        # normalised injury category
    injury_status: str = ""         # raw status level (Out/Questionable/etc.)


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
    
    # Personal fouls per game
    pf_pg = safe_mean(df["personal_fouls"]) if "personal_fouls" in df.columns else 0.0
    
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
        pf_pg=pf_pg,
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
            "SELECT player_id, name, position, is_injured, injury_note "
            "FROM players WHERE team_id = ?",
            (team_id,)
        ).fetchall()

    # ── Compute play probabilities via Injury Intelligence ──
    try:
        from src.analytics.injury_intelligence import compute_play_probability
        from src.data.sync_service import _normalise_status_level, _extract_injury_keyword
        _has_intel = True
    except Exception:
        _has_intel = False

    players: List[PlayerStats] = []
    for row in players_rows:
        pid, pname, pos, is_inj, inj_note = row
        pstats = get_player_comprehensive_stats(
            pid, pname, pos or "", bool(is_inj), opponent_team_id
        )

        # Assign play probability
        if not is_inj:
            pstats.play_probability = 1.0
        elif _has_intel and inj_note:
            status_raw = inj_note.split(":")[0].strip() if ":" in inj_note else inj_note
            injury_text = inj_note.split(":", 1)[1].strip() if ":" in inj_note else inj_note
            if "(" in injury_text:
                injury_text = injury_text[:injury_text.rfind("(")].strip()
            status_level = _normalise_status_level(status_raw)
            keyword = _extract_injury_keyword(injury_text)
            pstats.injury_status = status_level
            pstats.injury_keyword = keyword
            prob = compute_play_probability(pid, pname, status_level, keyword)
            pstats.play_probability = prob.composite_probability
        else:
            pstats.play_probability = 0.0  # injured with no note → assume out
            pstats.injury_status = "Out"

        players.append(pstats)

    # Sort by minutes played (most playing time first)
    players.sort(key=lambda p: p.mpg, reverse=True)

    # ── Probabilistic injury partitioning ──
    # Instead of a hard active/injured split, each player now has a
    # play_probability in [0,1].  The "absent fraction" of uncertain
    # players is what generates minute/point redistribution.
    #
    # For backward compat we still create active/injured lists, but
    # also track the fractional "absent weight" per injured player.
    active = [p for p in players if p.play_probability >= 1.0]
    injured = [p for p in players if p.play_probability < 1.0]
    # absent_frac: how much of this player's production is "missing"
    absent_frac = {p.player_id: (1.0 - p.play_probability) for p in injured}
    
    # A team plays 48 min * 5 players = 240 player-minutes per game
    TEAM_MINUTES_PER_GAME = 240.0
    
    # Calculate points-per-minute for redistribution estimation
    def get_ppm(p: PlayerStats) -> float:
        """Points per minute for a player."""
        return p.ppg / p.mpg if p.mpg > 0 else 0.0
    
    # Group injured players by position — weighted by absent fraction
    injured_by_pos: Dict[str, List[PlayerStats]] = {"G": [], "F": [], "C": []}
    for p in injured:
        pos_group = _get_position_group(p.position)
        injured_by_pos[pos_group].append(p)

    # Calculate injured minutes and points by position (scaled by absent fraction)
    injured_minutes_by_pos = {
        pos: sum(p.mpg * absent_frac.get(p.player_id, 1.0) for p in plist)
        for pos, plist in injured_by_pos.items()
    }
    injured_ppg_by_pos = {
        pos: sum(p.ppg * absent_frac.get(p.player_id, 1.0) for p in plist)
        for pos, plist in injured_by_pos.items()
    }

    # High-scoring injured players (15+ PPG) create usage boost (scaled)
    high_scorer_injured_ppg = sum(
        p.ppg * absent_frac.get(p.player_id, 1.0)
        for p in injured if p.ppg >= 15
    )
    
    # ============ FT EFFICIENCY LOSS CALCULATION ============
    # When a high-FT% shooter is injured, their replacement may be worse at FTs
    # This matters especially in late-game foul situations
    ft_efficiency_penalty = 0.0
    
    for inj_player in injured:
        af = absent_frac.get(inj_player.player_id, 1.0)
        if af <= 0:
            continue  # player is expected to play fully

        # Only consider players with meaningful FT volume (2+ FTA/game)
        if inj_player.fta_pg < 2.0 or inj_player.ft_pct <= 0:
            continue

        pos_group = _get_position_group(inj_player.position)

        # Find the likely replacement at the same position
        position_replacements = sorted(
            [p for p in active if _get_position_group(p.position) == pos_group],
            key=lambda p: p.mpg,
            reverse=True
        )

        if not position_replacements:
            adjacent = {"G": ["F"], "F": ["G", "C"], "C": ["F"]}
            for adj_pos in adjacent.get(pos_group, []):
                adj_players = [p for p in active if _get_position_group(p.position) == adj_pos]
                if adj_players:
                    position_replacements = sorted(adj_players, key=lambda p: p.mpg, reverse=True)
                    break

        if position_replacements:
            replacement_ft_pct = 0.0
            weight_sum = 0.0
            for i, repl in enumerate(position_replacements[:2]):
                weight = 1.0 if i == 0 else 0.5
                if repl.ft_pct > 0:
                    replacement_ft_pct += repl.ft_pct * weight
                    weight_sum += weight
            if weight_sum > 0:
                replacement_ft_pct /= weight_sum
            else:
                replacement_ft_pct = 78.0

            ft_pct_diff = inj_player.ft_pct - replacement_ft_pct
            if ft_pct_diff > 0:
                expected_ft_loss = inj_player.fta_pg * (ft_pct_diff / 100.0) * af
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

    # Helper: compute a single player's contribution
    def _player_contribution(p: PlayerStats, extra_points: float = 0.0) -> float:
        base = p.ppg * 0.4
        loc_val = p.ppg_home if is_home else p.ppg_away
        location = (loc_val if loc_val > 0 else p.ppg) * 0.3
        vs_val = p.ppg_vs_opp if (p.games_vs_opp > 0 and p.ppg_vs_opp > 0) else p.ppg
        vs_opp = vs_val * 0.3
        return base + location + vs_opp + extra_points

    # ── Active players (play_probability == 1.0): full contribution ──
    for p in active:
        pos_group = _get_position_group(p.position)
        pos_injured_minutes = injured_minutes_by_pos.get(pos_group, 0)
        pos_active_mpg = active_mpg_by_pos.get(pos_group, 0)

        extra_minutes = 0.0
        extra_points = 0.0

        # 1. Position-based minute redistribution
        if pos_active_mpg > 0 and pos_injured_minutes > 0:
            pos_share = p.mpg / pos_active_mpg
            extra_minutes = pos_injured_minutes * pos_share
            max_extra = max(0, 40 - p.mpg)
            extra_minutes = min(extra_minutes, max_extra)
            extra_points = extra_minutes * get_ppm(p) * 0.85

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

        # Accumulate stats
        total_ppg += p.ppg
        total_rpg += p.rpg
        total_apg += p.apg
        total_spg += p.spg
        total_bpg += p.bpg
        total_tpg += p.tpg
        total_oreb += p.oreb_pg
        total_dreb += p.dreb_pg

        if p.mpg > 0:
            weighted_ts_sum += p.ts_pct * p.mpg
            weighted_fg3_rate_sum += p.fg3_rate * p.mpg
            weighted_ft_rate_sum += p.ft_rate * p.mpg
            total_minutes_weight += p.mpg

        projected += _player_contribution(p, extra_points)

    # ── Uncertain players (0 < play_probability < 1): partial contribution ──
    # For "Questionable" or "Day-to-Day" players, include their expected
    # contribution proportional to their play probability.  This replaces
    # the binary in/out model.
    for p in injured:
        pp = p.play_probability
        if pp <= 0:
            continue  # definitely out — already handled by redistribution above

        # Add their proportional stats
        total_ppg += p.ppg * pp
        total_rpg += p.rpg * pp
        total_apg += p.apg * pp
        total_spg += p.spg * pp
        total_bpg += p.bpg * pp
        total_tpg += p.tpg * pp
        total_oreb += p.oreb_pg * pp
        total_dreb += p.dreb_pg * pp

        if p.mpg > 0:
            weighted_ts_sum += p.ts_pct * p.mpg * pp
            weighted_fg3_rate_sum += p.fg3_rate * p.mpg * pp
            weighted_ft_rate_sum += p.ft_rate * p.mpg * pp
            total_minutes_weight += p.mpg * pp

        projected += _player_contribution(p) * pp

    # Scale projection to realistic team total if needed
    total_active_mpg = sum(p.mpg for p in active) + sum(injured_minutes_by_pos.values())
    # Add back the partial minutes from uncertain players
    for p in injured:
        if p.play_probability > 0:
            total_active_mpg += p.mpg * p.play_probability
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
    if ft_efficiency_penalty > 0:
        projected -= ft_efficiency_penalty
        total_ppg -= ft_efficiency_penalty

    # ============ DATA-DRIVEN INJURY IMPACT (on/off metrics) ============
    # Use player on/off net rating differential to quantify true impact.
    # Scale penalty by absent_frac (so "Questionable" at 50% only gets 50% penalty).
    season = get_current_season()
    total_onoff_penalty = 0.0
    players_with_onoff = 0

    with get_conn() as conn:
        for inj_player in injured:
            af = absent_frac.get(inj_player.player_id, 1.0)
            if af <= 0:
                continue
            impact_row = conn.execute(
                "SELECT net_rating_diff, on_court_minutes, e_usg_pct FROM player_impact "
                "WHERE player_id = ? AND season = ?",
                (inj_player.player_id, season),
            ).fetchone()

            if impact_row and impact_row[0] is not None:
                net_diff = float(impact_row[0])
                on_minutes = float(impact_row[1] or inj_player.mpg)
                minute_fraction = on_minutes / 48.0
                point_impact = net_diff * minute_fraction * 0.5 * af
                total_onoff_penalty += max(0, point_impact)
                players_with_onoff += 1

    if players_with_onoff > 0:
        projected -= total_onoff_penalty
        total_ppg -= total_onoff_penalty
    else:
        # Fallback: heuristic penalties for playmakers and rebounders (scaled by absent frac)
        injured_assists = sum(
            p.apg * absent_frac.get(p.player_id, 1.0) for p in injured if p.apg >= 6.0
        )
        if injured_assists > 0:
            playmaker_penalty = injured_assists * 0.5
            projected -= playmaker_penalty
            total_ppg -= playmaker_penalty

        injured_rebounds = sum(
            p.rpg * absent_frac.get(p.player_id, 1.0) for p in injured if p.rpg >= 8.0
        )
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

    Uses a **blended** approach to avoid small-sample-size inflation:
    - Base: overall recent games           (50% weight)
    - Home/away split if available         (25% weight, else folded into base)
    - Vs-opponent split if available       (25% weight, scaled by sample size)

    Returns comprehensive stats including shooting efficiency and defensive stats.
    """
    empty = {
        "points": 0.0, "rebounds": 0.0, "assists": 0.0, "minutes": 0.0,
        "steals": 0.0, "blocks": 0.0, "turnovers": 0.0,
        "oreb": 0.0, "dreb": 0.0,
        "fg_made": 0, "fg_attempted": 0, "fg3_made": 0, "fg3_attempted": 0,
        "ft_made": 0, "ft_attempted": 0,
        "fga_pg": 0.0, "fta_pg": 0.0,
        "ts_pct": 0.0, "fg3_rate": 0.0,
    }

    df = _load_player_df(player_id)
    if df.empty:
        return empty

    # ---- helpers ----
    def _df_mean(subset: pd.DataFrame, col: str, default: float = 0.0) -> float:
        if col not in subset.columns or subset.empty:
            return default
        val = subset[col].mean()
        return float(val) if pd.notna(val) else default

    def _df_sum(subset: pd.DataFrame, col: str) -> int:
        if col not in subset.columns or subset.empty:
            return 0
        return int(subset[col].sum())

    # ---- slices ----
    base_df = df.head(recent_games) if recent_games else df

    loc_df = pd.DataFrame()
    if is_home is not None and not base_df.empty:
        loc_df = base_df[base_df["is_home"] == int(is_home)]

    opp_df = pd.DataFrame()
    if opponent_team_id is not None and not df.empty:
        opp_df = df[df["opponent_team_id"] == opponent_team_id]

    # ---- determine blend weights ----
    #  base  : always 50%
    #  loc   : 25% if we have >= 3 games, else 0% (folded into base)
    #  opp   : 25% if >= 3 games, 15% if 2 games, 10% if 1 game, else 0%
    w_base = 0.50
    w_loc = 0.25 if len(loc_df) >= 3 else 0.0
    w_opp = 0.0
    opp_n = len(opp_df)
    if opp_n >= 3:
        w_opp = 0.25
    elif opp_n == 2:
        w_opp = 0.15
    elif opp_n == 1:
        w_opp = 0.10

    # Redistribute unused weight to base
    w_base = 1.0 - w_loc - w_opp

    # ---- blend a stat column ----
    mean_cols = [
        "points", "rebounds", "assists", "minutes",
        "steals", "blocks", "turnovers", "oreb", "dreb",
        "fg_attempted", "ft_attempted",
    ]

    result: Dict[str, float] = {}
    for col in mean_cols:
        base_val = _df_mean(base_df, col)
        loc_val = _df_mean(loc_df, col) if w_loc > 0 else base_val
        opp_val = _df_mean(opp_df, col) if w_opp > 0 else base_val
        result[col] = w_base * base_val + w_loc * loc_val + w_opp * opp_val

    # Copy blended per-game averages for pace/possession estimation
    result["fga_pg"] = result["fg_attempted"]
    result["fta_pg"] = result["ft_attempted"]

    # ---- shooting totals from the base slice (used for aggregate efficiency) ----
    result["fg_made"] = _df_sum(base_df, "fg_made")
    result["fg_attempted"] = _df_sum(base_df, "fg_attempted")
    result["fg3_made"] = _df_sum(base_df, "fg3_made")
    result["fg3_attempted"] = _df_sum(base_df, "fg3_attempted")
    result["ft_made"] = _df_sum(base_df, "ft_made")
    result["ft_attempted"] = _df_sum(base_df, "ft_attempted")

    # ---- efficiency metrics (from base slice totals) ----
    total_pts = _df_sum(base_df, "points")
    total_fga = result["fg_attempted"]
    total_fg3a = result["fg3_attempted"]
    total_fta = result["ft_attempted"]

    ts_denom = 2 * (total_fga + 0.44 * total_fta)
    result["ts_pct"] = (total_pts / ts_denom * 100) if ts_denom > 0 else 0.0
    result["fg3_rate"] = (total_fg3a / total_fga * 100) if total_fga > 0 else 0.0

    return result


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

    # ---- MINUTE NORMALIZATION ----
    # A game has 240 player-minutes (5 players × 48 min).
    # Summing every roster player's per-game averages overcounts because
    # bench players' minutes overlap with starters' rest minutes.
    # Scale all counting stats down proportionally when total exceeds the budget.
    TEAM_MINUTES_PER_GAME = 240.0
    total_projected_minutes = totals.get("minutes", 0.0)
    if total_projected_minutes > TEAM_MINUTES_PER_GAME:
        scale = TEAM_MINUTES_PER_GAME / total_projected_minutes
        count_keys = [
            "points", "rebounds", "assists", "minutes",
            "steals", "blocks", "turnovers",
            "oreb", "dreb",
            "fg_made", "fg_attempted",
            "fg3_made", "fg3_attempted",
            "ft_made", "ft_attempted",
            "fga_pg", "fta_pg",
        ]
        for k in count_keys:
            if k in totals:
                totals[k] *= scale

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


# ============ TEAM METRICS (official NBA data with fallbacks) ============


def get_team_metrics(team_id: int, season: Optional[str] = None) -> Optional[Dict]:
    """
    Load the consolidated team_metrics row for a team.
    Returns dict of all columns, or None if not available.
    """
    season = season or get_current_season()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM team_metrics WHERE team_id = ? AND season = ?",
            (team_id, season),
        ).fetchone()
        if not row:
            return None
        cols = [desc[0] for desc in conn.execute("SELECT * FROM team_metrics LIMIT 0").description]
    return dict(zip(cols, row))


def get_player_impact_data(player_id: int, season: Optional[str] = None) -> Optional[Dict]:
    """Load player_impact row for a player."""
    season = season or get_current_season()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM player_impact WHERE player_id = ? AND season = ?",
            (player_id, season),
        ).fetchone()
        if not row:
            return None
        cols = [desc[0] for desc in conn.execute("SELECT * FROM player_impact LIMIT 0").description]
    return dict(zip(cols, row))


def get_offensive_rating(team_id: int) -> float:
    """
    Get team offensive rating.
    Prefers official NBA metrics from team_metrics table; falls back to
    hand-calculated from player game logs.
    """
    metrics = get_team_metrics(team_id)
    if metrics:
        # Prefer NBA estimated metric, then dashboard advanced
        for key in ("e_off_rating", "off_rating"):
            val = metrics.get(key)
            if val is not None:
                return float(val)

    # Fallback: hand-calculate from last 20 games
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ps.game_date,
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
            conn, params=[team_id],
        )
    if df.empty:
        return 110.0
    poss = float(df["team_fga"].mean()) - float(df["team_oreb"].mean()) + \
           float(df["team_tov"].mean()) + 0.44 * float(df["team_fta"].mean())
    return (float(df["team_pts"].mean()) / poss * 100) if poss > 0 else 110.0


def get_defensive_rating(team_id: int) -> float:
    """
    Get team defensive rating (opponent pts per 100 possessions).
    Lower = better defense.
    Prefers official metrics; falls back to hand-calculated.
    """
    metrics = get_team_metrics(team_id)
    if metrics:
        for key in ("e_def_rating", "def_rating"):
            val = metrics.get(key)
            if val is not None:
                return float(val)

    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT ps.game_date,
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
            conn, params=[team_id],
        )
    if df.empty:
        return 110.0
    poss = float(df["opp_fga"].mean()) - float(df["opp_oreb"].mean()) + \
           float(df["opp_tov"].mean()) + 0.44 * float(df["opp_fta"].mean())
    return (float(df["opp_pts"].mean()) / poss * 100) if poss > 0 else 110.0


def get_pace(team_id: int) -> float:
    """Get team pace (possessions per game). Prefers official metrics."""
    metrics = get_team_metrics(team_id)
    if metrics:
        for key in ("e_pace", "pace"):
            val = metrics.get(key)
            if val is not None:
                return float(val)
    return 98.0  # league average fallback


def get_opponent_defensive_factor(team_id: int) -> float:
    """
    Scaling factor for how well opponents defend.
    < 1.0 = good defense, > 1.0 = bad defense.
    Prefers official opponent stats from team_metrics.
    """
    metrics = get_team_metrics(team_id)
    if metrics:
        opp_pts = metrics.get("opp_pts")
        if opp_pts is not None:
            # Use league average of ~112-114 PPG as baseline
            league_avg = _get_league_avg_ppg()
            return float(opp_pts) / league_avg if league_avg > 0 else 1.0

    # Fallback: compute from game data
    with get_conn() as conn:
        opp_row = conn.execute(
            """
            SELECT AVG(game_total) as avg_opp_pts FROM (
                SELECT game_date, SUM(points) as game_total
                FROM player_stats WHERE opponent_team_id = ?
                GROUP BY game_date
            )
            """,
            (team_id,),
        ).fetchone()
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
    avg_opp = float(opp_row[0])
    league_avg = float(league_row[0]) if league_row and league_row[0] else 112.0
    return avg_opp / league_avg if league_avg > 0 else 1.0


def _get_league_avg_ppg() -> float:
    """Helper to compute current league average PPG."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT AVG(game_total) FROM (
                SELECT game_date, opponent_team_id, SUM(points) as game_total
                FROM player_stats
                GROUP BY game_date, opponent_team_id
            )
            """
        ).fetchone()
    return float(row[0]) if row and row[0] else 112.0


def get_home_court_advantage(team_id: int) -> float:
    """
    Team-specific home court advantage in points.
    Prefers official home/road splits from team_metrics.
    Clamped to [1.5, 5.0].
    """
    metrics = get_team_metrics(team_id)
    if metrics:
        home_pts = metrics.get("home_pts")
        road_pts = metrics.get("road_pts")
        if home_pts is not None and road_pts is not None:
            hca = float(home_pts) - float(road_pts)
            return max(1.5, min(5.0, hca))

    # Fallback: compute from player game logs
    with get_conn() as conn:
        home_df = pd.read_sql(
            "SELECT ps.game_date, SUM(ps.points) as team_pts "
            "FROM player_stats ps JOIN players p ON p.player_id = ps.player_id "
            "WHERE p.team_id = ? AND ps.is_home = 1 GROUP BY ps.game_date",
            conn, params=[team_id],
        )
        away_df = pd.read_sql(
            "SELECT ps.game_date, SUM(ps.points) as team_pts "
            "FROM player_stats ps JOIN players p ON p.player_id = ps.player_id "
            "WHERE p.team_id = ? AND ps.is_home = 0 GROUP BY ps.game_date",
            conn, params=[team_id],
        )
    if home_df.empty or away_df.empty:
        return 3.0
    hca = float(home_df["team_pts"].mean()) - float(away_df["team_pts"].mean())
    return max(1.5, min(5.0, hca))


def get_four_factors(team_id: int) -> Dict[str, Optional[float]]:
    """
    Get the Four Factors of basketball success for a team.
    Returns team's own factors AND opponent's (defensive forcing).
    """
    metrics = get_team_metrics(team_id)
    if metrics:
        return {
            "efg_pct": metrics.get("ff_efg_pct"),
            "fta_rate": metrics.get("ff_fta_rate"),
            "tm_tov_pct": metrics.get("ff_tm_tov_pct"),
            "oreb_pct": metrics.get("ff_oreb_pct"),
            "opp_efg_pct": metrics.get("opp_efg_pct"),
            "opp_fta_rate": metrics.get("opp_fta_rate"),
            "opp_tm_tov_pct": metrics.get("opp_tm_tov_pct"),
            "opp_oreb_pct": metrics.get("opp_oreb_pct"),
        }
    return {k: None for k in [
        "efg_pct", "fta_rate", "tm_tov_pct", "oreb_pct",
        "opp_efg_pct", "opp_fta_rate", "opp_tm_tov_pct", "opp_oreb_pct",
    ]}


def get_clutch_stats(team_id: int) -> Dict[str, Optional[float]]:
    """Get clutch performance metrics for a team."""
    metrics = get_team_metrics(team_id)
    if metrics:
        return {
            "clutch_gp": metrics.get("clutch_gp"),
            "clutch_w": metrics.get("clutch_w"),
            "clutch_l": metrics.get("clutch_l"),
            "clutch_net_rating": metrics.get("clutch_net_rating"),
            "clutch_efg_pct": metrics.get("clutch_efg_pct"),
            "clutch_ts_pct": metrics.get("clutch_ts_pct"),
        }
    return {k: None for k in [
        "clutch_gp", "clutch_w", "clutch_l",
        "clutch_net_rating", "clutch_efg_pct", "clutch_ts_pct",
    ]}


def get_hustle_stats(team_id: int) -> Dict[str, Optional[float]]:
    """Get hustle stats for a team."""
    metrics = get_team_metrics(team_id)
    if metrics:
        return {
            "deflections": metrics.get("deflections"),
            "loose_balls_recovered": metrics.get("loose_balls_recovered"),
            "contested_shots": metrics.get("contested_shots"),
            "charges_drawn": metrics.get("charges_drawn"),
            "screen_assists": metrics.get("screen_assists"),
        }
    return {k: None for k in [
        "deflections", "loose_balls_recovered", "contested_shots",
        "charges_drawn", "screen_assists",
    ]}


# ============ SCHEDULE INTELLIGENCE (fatigue / rest) ============


def get_team_schedule_dates(team_id: int, around_date: Optional[date] = None) -> List[date]:
    """
    Get a team's game dates from player_stats (past games) ordered ascending.
    Optionally filter to a window around a given date.
    """
    # Ensure around_date is a date object, not a string
    if isinstance(around_date, str):
        around_date = date.fromisoformat(around_date[:10])

    with get_conn() as conn:
        if around_date:
            window_start = around_date - timedelta(days=7)
            window_end = around_date + timedelta(days=1)
            rows = conn.execute(
                """
                SELECT DISTINCT ps.game_date FROM player_stats ps
                JOIN players p ON p.player_id = ps.player_id
                WHERE p.team_id = ? AND ps.game_date BETWEEN ? AND ?
                ORDER BY ps.game_date
                """,
                (team_id, str(window_start), str(window_end)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT DISTINCT ps.game_date FROM player_stats ps
                JOIN players p ON p.player_id = ps.player_id
                WHERE p.team_id = ?
                ORDER BY ps.game_date
                """,
                (team_id,),
            ).fetchall()
    
    result = []
    for (d,) in rows:
        if isinstance(d, str):
            try:
                result.append(date.fromisoformat(d))
            except ValueError:
                pass
        elif isinstance(d, date):
            result.append(d)
    return result


def detect_fatigue(team_id: int, game_date: date) -> Dict[str, object]:
    """
    Auto-detect fatigue factors for a team on a given date.

    Returns dict with:
    - is_back_to_back: bool
    - is_3_in_4: bool (3 games in 4 days including this one)
    - is_4_in_6: bool
    - rest_days: int (days since last game, 0 = B2B)
    - fatigue_penalty: float (total points penalty to apply)
    """
    # Convert game_date to date if needed (must happen before any date arithmetic)
    if isinstance(game_date, str):
        game_date = date.fromisoformat(game_date[:10])

    game_dates = get_team_schedule_dates(team_id, around_date=game_date)

    # Filter to games before this date
    prior = [d for d in game_dates if d < game_date]

    result = {
        "is_back_to_back": False,
        "is_3_in_4": False,
        "is_4_in_6": False,
        "rest_days": 99,  # default: well rested
        "fatigue_penalty": 0.0,
    }

    if not prior:
        return result

    last_game = prior[-1]
    rest_days = (game_date - last_game).days
    result["rest_days"] = rest_days

    # Back-to-back: played yesterday
    result["is_back_to_back"] = rest_days == 1

    # 3-in-4: including today, 3 games in a 4-day window
    window_4 = game_date - timedelta(days=3)
    games_in_4 = sum(1 for d in prior if d >= window_4) + 1  # +1 for today
    result["is_3_in_4"] = games_in_4 >= 3

    # 4-in-6: including today, 4 games in a 6-day window
    window_6 = game_date - timedelta(days=5)
    games_in_6 = sum(1 for d in prior if d >= window_6) + 1
    result["is_4_in_6"] = games_in_6 >= 4

    # Calculate graduated fatigue penalty using configurable weights
    from src.analytics.weight_config import get_weight_config
    w = get_weight_config()
    penalty = 0.0
    if result["is_back_to_back"]:
        penalty += w.fatigue_b2b  # B2B penalty (default 2.0)
    if result["is_3_in_4"]:
        penalty += w.fatigue_3in4  # Cumulative fatigue (default 1.0)
    if result["is_4_in_6"]:
        penalty += w.fatigue_4in6  # Heavy schedule (default 1.5)
    elif rest_days == 0:
        penalty += 3.0  # Same-day (rare, doubleheader scenario)

    result["fatigue_penalty"] = penalty
    return result


def get_scheduled_games(include_future_days: int = 14) -> List[Dict]:
    """
    Get games from the schedule, including upcoming games.
    fetch_schedule() returns per-game rows with:
      game_date, home_team_id, away_team_id, home_abbr, away_abbr,
      home_name, away_name, game_time, arena
    """
    from src.data.sync_service import sync_schedule

    try:
        df = sync_schedule(include_future_days=include_future_days)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return []

    # Get team name lookup for any rows missing names
    with get_conn() as conn:
        teams = pd.read_sql("SELECT team_id, abbreviation, name FROM teams", conn)
    id_to_info = {int(row["team_id"]): (row["abbreviation"], row["name"]) for _, row in teams.iterrows()}

    games = []
    seen = set()

    for _, row in df.iterrows():
        game_date = row.get("game_date")
        home_id = row.get("home_team_id")
        away_id = row.get("away_team_id")

        if not home_id or not away_id:
            continue

        home_id = int(home_id)
        away_id = int(away_id)

        # Deduplicate
        key = (min(home_id, away_id), max(home_id, away_id), str(game_date))
        if key in seen:
            continue
        seen.add(key)

        # Use names from the DataFrame, falling back to DB lookup
        home_abbr = str(row.get("home_abbr", ""))
        away_abbr = str(row.get("away_abbr", ""))
        home_name = str(row.get("home_name", ""))
        away_name = str(row.get("away_name", ""))

        if not home_abbr and home_id in id_to_info:
            home_abbr, home_name = id_to_info[home_id]
        if not away_abbr and away_id in id_to_info:
            away_abbr, away_name = id_to_info[away_id]

        games.append({
            "game_date": game_date,
            "home_team_id": home_id,
            "home_abbr": home_abbr,
            "home_name": home_name,
            "away_team_id": away_id,
            "away_abbr": away_abbr,
            "away_name": away_name,
            "game_time": str(row.get("game_time", "")),
            "arena": str(row.get("arena", "")),
        })

    # Sort by date descending (most recent first)
    games.sort(key=lambda g: str(g.get("game_date", "")), reverse=True)
    return games[:100]
