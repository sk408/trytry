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
    # Season averages
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


@dataclass
class TeamMatchupStats:
    team_id: int
    team_abbr: str
    team_name: str
    players: List[PlayerStats]
    total_ppg: float
    total_rpg: float
    total_apg: float
    projected_points: float  # weighted projection for this matchup


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
    """Get comprehensive stats for a single player."""
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
    
    # Season averages (handle NaN with fillna)
    ppg = float(df["points"].mean()) if not df["points"].isna().all() else 0.0
    rpg = float(df["rebounds"].mean()) if not df["rebounds"].isna().all() else 0.0
    apg = float(df["assists"].mean()) if not df["assists"].isna().all() else 0.0
    mpg = float(df["minutes"].mean()) if not df["minutes"].isna().all() else 0.0
    games_played = len(df)
    
    # Handle potential NaN values
    import math
    if math.isnan(ppg): ppg = 0.0
    if math.isnan(rpg): rpg = 0.0
    if math.isnan(apg): apg = 0.0
    if math.isnan(mpg): mpg = 0.0
    
    # Home/Away splits
    home_df = df[df["is_home"] == 1]
    away_df = df[df["is_home"] == 0]
    ppg_home = float(home_df["points"].mean()) if not home_df.empty else ppg
    ppg_away = float(away_df["points"].mean()) if not away_df.empty else ppg
    if math.isnan(ppg_home): ppg_home = ppg
    if math.isnan(ppg_away): ppg_away = ppg
    
    # Vs specific opponent
    ppg_vs_opp = 0.0
    games_vs_opp = 0
    if opponent_team_id is not None:
        opp_df = df[df["opponent_team_id"] == opponent_team_id]
        if not opp_df.empty:
            ppg_vs_opp = float(opp_df["points"].mean())
            if math.isnan(ppg_vs_opp): ppg_vs_opp = ppg
            games_vs_opp = len(opp_df)
    
    return PlayerStats(
        player_id=player_id,
        name=player_name,
        position=position,
        is_injured=is_injured,
        ppg=ppg, rpg=rpg, apg=apg, mpg=mpg, games_played=games_played,
        ppg_home=ppg_home, ppg_away=ppg_away,
        ppg_vs_opp=ppg_vs_opp, games_vs_opp=games_vs_opp,
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
    projected = 0.0
    
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
            # Cap at ~38 total minutes (college games are 40 min, players rarely play full game)
            max_extra = max(0, 38 - p.mpg)
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
    
    return TeamMatchupStats(
        team_id=team_id,
        team_abbr=team_info[1],
        team_name=team_info[2],
        players=players,
        total_ppg=total_ppg,
        total_rpg=total_rpg,
        total_apg=total_apg,
        projected_points=projected,
    )


def player_splits(
    player_id: int,
    opponent_team_id: Optional[int] = None,
    is_home: Optional[bool] = None,
    recent_games: Optional[int] = 10,
) -> Dict[str, float]:
    df = _load_player_df(player_id)
    if df.empty:
        return {"points": 0.0, "rebounds": 0.0, "assists": 0.0, "minutes": 0.0}

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

    return {
        "points": df["points"].mean() if not df.empty else 0.0,
        "rebounds": df["rebounds"].mean() if not df.empty else 0.0,
        "assists": df["assists"].mean() if not df.empty else 0.0,
        "minutes": df["minutes"].mean() if not df.empty else 0.0,
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
    totals = {"points": 0.0, "rebounds": 0.0, "assists": 0.0}
    for pid in player_ids:
        splits = player_splits(
            pid,
            opponent_team_id=opponent_team_id,
            is_home=is_home,
            recent_games=recent_games,
        )
        totals = {k: totals[k] + splits.get(k, 0.0) for k in totals}
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
