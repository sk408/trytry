"""Player splits, aggregate_projection(), 240-min budget, fatigue detection."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import numpy as np

from src.database import db
from src.analytics.memory_store import get_store

logger = logging.getLogger(__name__)

# ──────────────────── Constants ────────────────────
TEAM_MINUTES_PER_GAME = 240.0
_DECAY = 0.9
_HOME_COURT_FALLBACK = 2.0              # modern NBA HCA is ~2-3 pts
_HOME_COURT_CLAMP = (-1.0, 5.0)         # reduced from (-2,8) to fix +76 home bias
_PACE_FALLBACK = 98.0
_RATING_FALLBACK = 110.0
_LEAGUE_AVG_PPG = 112.0
_HIGH_SCORER_THRESHOLD = 15.0
_ACTIVE_HIGH_SCORER_THRESHOLD = 12.0
_PLAYMAKER_THRESHOLD = 6.0
_PLAYMAKER_PENALTY_MULT = 0.5
_REBOUNDER_THRESHOLD = 8.0
_REBOUNDER_PENALTY_MULT = 0.2
_FT_VOLUME_THRESHOLD = 2.0
_DEFAULT_REPLACEMENT_FT_PCT = 78.0
_RETURN_DISCOUNT_PER_GAME = 0.03
_RETURN_DISCOUNT_FLOOR = 0.85
_ROSTER_CHANGE_HIGH_IMPACT_MPG = 20.0

STAT_COLS = [
    "points", "rebounds", "assists", "minutes", "steals", "blocks",
    "turnovers", "oreb", "dreb", "fg_made", "fg_attempted",
    "fg3_made", "fg3_attempted", "ft_made", "ft_attempted",
    "plus_minus", "personal_fouls",
]


def _exponential_decay_weights(n: int, decay: float = _DECAY) -> np.ndarray:
    """Generate normalized exponential decay weights. Row 0 = most recent = weight 1.0."""
    if n <= 0:
        return np.array([])
    raw = np.array([decay ** i for i in range(n)])
    total = raw.sum()
    return raw / total if total > 0 else raw


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean, handling edge cases."""
    if len(values) == 0 or len(weights) == 0:
        return 0.0
    return float(np.dot(values, weights[:len(values)]))


def player_splits(player_id: int, opponent_team_id: int, is_home: int,
                  recent_games: int = 10, as_of_date: Optional[str] = None) -> Dict[str, float]:
    """Compute 50/25/25 blended stat projection for a player.

    Returns dict of stat averages.
    """
    store = get_store()
    if store.player_stats is not None and not store.player_stats.empty:
        ps = store.player_stats[store.player_stats["player_id"] == player_id].copy()
    else:
        rows = db.fetch_all(
            "SELECT * FROM player_stats WHERE player_id = ? ORDER BY game_date DESC",
            (player_id,)
        )
        if not rows:
            return {c: 0.0 for c in STAT_COLS}
        import pandas as pd
        ps = pd.DataFrame([dict(r) for r in rows])

    if ps.empty:
        return {c: 0.0 for c in STAT_COLS}

    # as_of_date filter (prevent lookahead)
    if as_of_date:
        ps = ps[ps["game_date"] < as_of_date]
    if ps.empty:
        return {c: 0.0 for c in STAT_COLS}

    ps = ps.sort_values("game_date", ascending=False)

    # Base: overall recent
    base = ps.head(recent_games)

    # Location split
    loc = ps[ps["is_home"] == is_home].head(recent_games)

    # vs-opponent split
    opp = ps[ps["opponent_team_id"] == opponent_team_id].head(recent_games)

    # Blend weights
    w_opp = 0.0
    n_opp = len(opp)
    if n_opp >= 3:
        w_opp = 0.25
    elif n_opp == 2:
        w_opp = 0.15
    elif n_opp == 1:
        w_opp = 0.10

    w_loc = 0.25 if len(loc) >= 3 else 0.0
    w_base = 1.0 - w_loc - w_opp

    result = {}
    for col in STAT_COLS:
        if col not in ps.columns:
            result[col] = 0.0
            continue

        base_vals = base[col].values.astype(float)
        base_weights = _exponential_decay_weights(len(base_vals))

        loc_vals = loc[col].values.astype(float) if w_loc > 0 else np.array([])
        loc_weights = _exponential_decay_weights(len(loc_vals)) if w_loc > 0 else np.array([])

        opp_vals = opp[col].values.astype(float) if w_opp > 0 else np.array([])
        opp_weights = _exponential_decay_weights(len(opp_vals)) if w_opp > 0 else np.array([])

        val = (w_base * _weighted_mean(base_vals, base_weights) +
               w_loc * _weighted_mean(loc_vals, loc_weights) +
               w_opp * _weighted_mean(opp_vals, opp_weights))
        result[col] = val

    return result


def get_games_missed_streak(player_id: int, as_of_date: Optional[str] = None) -> int:
    """Count consecutive games missed before as_of_date (or now)."""
    if as_of_date is None:
        as_of_date = datetime.now().strftime("%Y-%m-%d")

    rows = db.fetch_all(
        """SELECT game_date FROM injury_history
           WHERE player_id = ? AND was_out = 1 AND game_date <= ?
           ORDER BY game_date DESC LIMIT 20""",
        (player_id, as_of_date)
    )
    if not rows:
        return 0

    last_played = db.fetch_one(
        """SELECT MAX(game_date) as d FROM player_stats
           WHERE player_id = ? AND game_date <= ?""",
        (player_id, as_of_date)
    )
    if not last_played or not last_played["d"]:
        return len(rows)

    # Count missed games after last played
    count = 0
    for r in rows:
        if r["game_date"] > last_played["d"]:
            count += 1
        else:
            break
    return count


def aggregate_projection(team_id: int, opponent_team_id: int, is_home: int,
                         as_of_date: Optional[str] = None,
                         player_weights: Optional[Dict[int, float]] = None,
                         injured_players: Optional[Dict[int, float]] = None) -> Dict[str, float]:
    """Aggregate team projection from player splits with 240-min budget.

    Args:
        injured_players: {player_id: play_probability} for injured players
    """
    if player_weights is None:
        player_weights = {}
    if injured_players is None:
        injured_players = {}

    # Get active players
    rows = db.fetch_all(
        "SELECT player_id, name, position FROM players WHERE team_id = ?",
        (team_id,)
    )

    totals = {c: 0.0 for c in STAT_COLS}
    total_projected_minutes = 0.0
    active_players = []

    for p in rows:
        pid = p["player_id"]

        # Skip players who are definitely out
        play_prob = injured_players.get(pid, 1.0)
        if play_prob < 0.3:
            continue

        splits = player_splits(pid, opponent_team_id, is_home, recent_games=10,
                               as_of_date=as_of_date)
        if splits["minutes"] < 1.0:
            continue

        weight = player_weights.get(pid, 1.0) * play_prob

        # Return-from-injury discount
        missed = get_games_missed_streak(pid, as_of_date)
        if missed > 0:
            discount = max(_RETURN_DISCOUNT_FLOOR, 1.0 - _RETURN_DISCOUNT_PER_GAME * missed)
            weight *= discount

        for k in STAT_COLS:
            totals[k] += splits[k] * weight

        total_projected_minutes += splits["minutes"] * weight
        active_players.append({
            "player_id": pid,
            "name": p["name"],
            "position": p["position"],
            "splits": splits,
            "weight": weight,
        })

    # 240-minute budget normalization
    if total_projected_minutes > TEAM_MINUTES_PER_GAME:
        scale = TEAM_MINUTES_PER_GAME / total_projected_minutes
        scaling_cols = ["points", "rebounds", "assists", "minutes", "steals",
                        "blocks", "turnovers", "oreb", "dreb",
                        "fg_made", "fg_attempted", "fg3_made", "fg3_attempted",
                        "ft_made", "ft_attempted"]
        for c in scaling_cols:
            totals[c] *= scale

    totals["_active_players"] = active_players
    totals["_total_projected_minutes"] = min(total_projected_minutes, TEAM_MINUTES_PER_GAME)

    # Fallback: if projection produced zero points (e.g. first game of season
    # with no prior data), use league-average estimate so downstream calcs get
    # a reasonable baseline instead of 0.
    if totals["points"] < 1.0:
        totals["points"] = _LEAGUE_AVG_PPG
        logger.debug("aggregate_projection for team %d produced 0 pts — using league avg fallback", team_id)

    return totals


def get_home_court_advantage(team_id: int, season: Optional[str] = None) -> float:
    """Calculate home court advantage from home/road point differentials.

    NOTE: home_pts / road_pts are stored as **per-game** averages (from
    LeagueDashTeamStats with per_mode=PerGame), so we must NOT divide by GP.
    """
    from src.config import get_season
    if season is None:
        season = get_season()
    row = db.fetch_one(
        "SELECT home_pts, road_pts, home_gp, road_gp FROM team_metrics WHERE team_id = ? AND season = ?",
        (team_id, season)
    )
    if not row or not row["home_gp"] or not row["road_gp"]:
        return _HOME_COURT_FALLBACK

    home_ppg = row["home_pts"]   # already per-game
    road_ppg = row["road_pts"]   # already per-game
    hca = home_ppg - road_ppg
    return max(_HOME_COURT_CLAMP[0], min(_HOME_COURT_CLAMP[1], hca))


def compute_fatigue(team_id: int, game_date: str, w=None) -> Dict[str, Any]:
    """Detect back-to-back, 3-in-4, 4-in-6 fatigue situations.

    Args:
        w: Optional WeightConfig to read b2b/3in4/4in6 penalties from.
           Falls back to defaults if None.
    """
    try:
        gd = datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        return {"penalty": 0.0, "rest_days": 3, "b2b": False, "three_in_four": False, "four_in_six": False}

    # Find team's recent games
    rows = db.fetch_all(
        """SELECT DISTINCT game_date FROM player_stats ps
           JOIN players p ON ps.player_id = p.player_id
           WHERE p.team_id = ? AND ps.game_date < ?
           ORDER BY ps.game_date DESC LIMIT 10""",
        (team_id, game_date)
    )
    if not rows:
        return {"penalty": 0.0, "rest_days": 3, "b2b": False, "three_in_four": False, "four_in_six": False}

    dates = []
    for r in rows:
        try:
            dates.append(datetime.strptime(r["game_date"], "%Y-%m-%d"))
        except (ValueError, TypeError):
            pass
    if not dates:
        return {"penalty": 0.0, "rest_days": 3, "b2b": False, "three_in_four": False, "four_in_six": False}

    rest_days = (gd - dates[0]).days
    is_b2b = rest_days == 1
    is_same_day = rest_days == 0

    # 3-in-4 and 4-in-6
    window_4 = gd - timedelta(days=4)
    window_6 = gd - timedelta(days=6)
    games_in_4 = sum(1 for d in dates if d >= window_4) + 1  # +1 for current game
    games_in_6 = sum(1 for d in dates if d >= window_6) + 1

    is_3in4 = games_in_4 >= 3
    is_4in6 = games_in_6 >= 4

    # Read penalties from WeightConfig if provided, else use defaults
    b2b_pen = w.fatigue_b2b if w else 2.0
    pen_3in4 = w.fatigue_3in4 if w else 1.0
    pen_4in6 = w.fatigue_4in6 if w else 1.5

    penalty = 0.0
    if is_same_day:
        penalty += 3.0
    if is_b2b:
        penalty += b2b_pen
    if is_3in4:
        penalty += pen_3in4
    if is_4in6:
        penalty += pen_4in6

    # Rest bonus
    rest_bonus = 0.0
    if rest_days >= 4:
        rest_bonus = -1.5
    elif rest_days >= 3:
        rest_bonus = -1.0

    return {
        "penalty": penalty + rest_bonus,
        "rest_days": rest_days,
        "b2b": is_b2b,
        "three_in_four": is_3in4,
        "four_in_six": is_4in6,
    }


def compute_shooting_efficiency(proj: Dict[str, float]) -> Dict[str, float]:
    """Compute TS%, eFG%, FG3 rate, FT rate from projection."""
    fga = proj.get("fg_attempted", 0)
    fta = proj.get("ft_attempted", 0)
    fgm = proj.get("fg_made", 0)
    fg3m = proj.get("fg3_made", 0)
    pts = proj.get("points", 0)

    ts_pct = (pts / (2 * (fga + 0.44 * fta)) * 100) if (fga + 0.44 * fta) > 0 else 0
    efg_pct = ((fgm + 0.5 * fg3m) / fga * 100) if fga > 0 else 0
    fg3_rate = (proj.get("fg3_attempted", 0) / fga * 100) if fga > 0 else 0
    ft_rate = (fta / fga * 100) if fga > 0 else 0

    return {"ts_pct": ts_pct, "efg_pct": efg_pct, "fg3_rate": fg3_rate, "ft_rate": ft_rate}


def detect_roster_change(team_id: int) -> Dict[str, Any]:
    """Compare current roster vs players who appeared in last 5 game dates."""
    current = set()
    rows = db.fetch_all("SELECT player_id FROM players WHERE team_id = ?", (team_id,))
    for r in rows:
        current.add(r["player_id"])

    # Players who appeared in recent games
    recent = set()
    game_rows = db.fetch_all(
        """SELECT DISTINCT ps.player_id FROM player_stats ps
           JOIN players p ON ps.player_id = p.player_id
           WHERE p.team_id = ?
           AND ps.game_date IN (
             SELECT DISTINCT game_date FROM player_stats ps2
             JOIN players p2 ON ps2.player_id = p2.player_id
             WHERE p2.team_id = ?
             ORDER BY game_date DESC LIMIT 5
           )""",
        (team_id, team_id)
    )
    for r in game_rows:
        recent.add(r["player_id"])

    added = current - recent
    removed = recent - current

    # Check if any are high impact
    high_impact = False
    for pid in added | removed:
        avg_row = db.fetch_one(
            "SELECT AVG(minutes) as avg_min FROM player_stats WHERE player_id = ? ORDER BY game_date DESC LIMIT 10",
            (pid,)
        )
        if avg_row and avg_row["avg_min"] and avg_row["avg_min"] >= _ROSTER_CHANGE_HIGH_IMPACT_MPG:
            high_impact = True
            break

    return {
        "changed": bool(added or removed),
        "players_added": list(added),
        "players_removed": list(removed),
        "high_impact": high_impact,
    }


def get_team_matchup_stats(team_id: int, opponent_team_id: int,
                           is_home: int, as_of_date: Optional[str] = None) -> List[Dict]:
    """Get per-player contributions for matchup display.

    contribution = ppg * 0.4 + location_ppg * 0.3 + vs_opp_ppg * 0.3
    """
    rows = db.fetch_all(
        "SELECT player_id, name, position FROM players WHERE team_id = ?",
        (team_id,)
    )
    result = []
    for p in rows:
        pid = p["player_id"]
        splits = player_splits(pid, opponent_team_id, is_home, recent_games=10,
                               as_of_date=as_of_date)
        if splits["minutes"] < 1.0:
            continue

        # Location-specific ppg
        loc_rows = db.fetch_all(
            """SELECT AVG(points) as ppg FROM player_stats
               WHERE player_id = ? AND is_home = ?
               ORDER BY game_date DESC LIMIT 10""",
            (pid, is_home)
        )
        loc_ppg = loc_rows[0]["ppg"] if loc_rows and loc_rows[0]["ppg"] else splits["points"]

        # vs-opponent ppg
        vs_rows = db.fetch_all(
            """SELECT AVG(points) as ppg FROM player_stats
               WHERE player_id = ? AND opponent_team_id = ?
               ORDER BY game_date DESC LIMIT 10""",
            (pid, opponent_team_id)
        )
        vs_ppg = vs_rows[0]["ppg"] if vs_rows and vs_rows[0]["ppg"] else splits["points"]

        contribution = splits["points"] * 0.4 + loc_ppg * 0.3 + vs_ppg * 0.3

        result.append({
            "player_id": pid,
            "name": p["name"],
            "position": p["position"],
            "ppg": round(splits["points"], 1),
            "rpg": round(splits["rebounds"], 1),
            "apg": round(splits["assists"], 1),
            "mpg": round(splits["minutes"], 1),
            "contribution": round(contribution, 1),
        })

    result.sort(key=lambda x: x["contribution"], reverse=True)
    return result
