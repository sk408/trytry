"""Play-through rates, player tendencies, keyword modifiers — 3-layer blend."""

import logging
from typing import Dict, Any, Optional, List

from src.database import db

logger = logging.getLogger(__name__)

# Layer 1 defaults (league-wide fallback rates)
DEFAULT_RATES = {
    "Out": 0.00,
    "Doubtful": 0.10,
    "Questionable": 0.50,
    "GTD": 0.50,
    "Day-To-Day": 0.60,
    "Probable": 0.85,
    "Available": 1.00,
}

# Injury keywords (37 categories)
INJURY_KEYWORDS = [
    "rest", "personal", "suspension", "illness", "concussion",
    "hamstring", "quad", "calf", "groin", "ankle", "foot", "toe",
    "knee", "acl", "mcl", "meniscus", "hip", "back", "shoulder",
    "elbow", "wrist", "hand", "finger", "achilles", "thigh", "leg",
    "rib", "chest", "abdomen", "neck", "eye", "sprain", "strain",
    "soreness", "contusion", "fracture", "surgery",
]


def _classify_keyword(reason: str) -> str:
    """Extract injury keyword from reason string."""
    if not reason:
        return "other"
    reason_lower = reason.lower()
    for kw in INJURY_KEYWORDS:
        if kw in reason_lower:
            return kw
    return "other"


def get_league_play_rate(status: str) -> Dict[str, Any]:
    """Layer 1: League-wide play rate from injury_status_log."""
    rows = db.fetch_all("""
        SELECT did_play, COUNT(*) as cnt
        FROM injury_status_log
        WHERE status_level = ? AND did_play IS NOT NULL
        GROUP BY did_play
    """, (status,))

    total = sum(r["cnt"] for r in rows)
    played = sum(r["cnt"] for r in rows if r["did_play"])

    if total >= 5:
        rate = played / total
        if total >= 50:
            confidence = "high"
        elif total >= 15:
            confidence = "medium"
        else:
            confidence = "low"
        return {"rate": rate, "confidence": confidence, "observations": total, "source": "league"}

    # Fallback to defaults
    rate = DEFAULT_RATES.get(status, 0.50)
    return {"rate": rate, "confidence": "default", "observations": 0, "source": "default"}


def get_player_play_rate(player_id: int, status: str) -> Dict[str, Any]:
    """Layer 2: Player-specific tendency."""
    rows = db.fetch_all("""
        SELECT did_play, COUNT(*) as cnt
        FROM injury_status_log
        WHERE player_id = ? AND status_level = ? AND did_play IS NOT NULL
        GROUP BY did_play
    """, (player_id, status))

    total = sum(r["cnt"] for r in rows)
    played = sum(r["cnt"] for r in rows if r["did_play"])

    if total == 0:
        return {"rate": None, "observations": 0}

    rate = played / total if total > 0 else 0
    return {"rate": rate, "observations": total}


def get_keyword_modifier(keyword: str, status: str) -> float:
    """Layer 3: Keyword-specific rate modifier."""
    # Keyword-specific rate
    kw_rows = db.fetch_all("""
        SELECT did_play, COUNT(*) as cnt
        FROM injury_status_log
        WHERE injury_keyword = ? AND status_level = ? AND did_play IS NOT NULL
        GROUP BY did_play
    """, (keyword, status))

    kw_total = sum(r["cnt"] for r in kw_rows)
    kw_played = sum(r["cnt"] for r in kw_rows if r["did_play"])

    if kw_total < 3:
        return 1.0  # Not enough data, no modification

    kw_rate = kw_played / kw_total

    # Overall average rate for this status
    all_rows = db.fetch_all("""
        SELECT did_play, COUNT(*) as cnt
        FROM injury_status_log
        WHERE status_level = ? AND did_play IS NOT NULL
        GROUP BY did_play
    """, (status,))

    all_total = sum(r["cnt"] for r in all_rows)
    all_played = sum(r["cnt"] for r in all_rows if r["did_play"])

    if all_total == 0 or all_played == 0:
        return 1.0

    overall_rate = all_played / all_total

    if overall_rate == 0:
        return 1.0

    modifier = kw_rate / overall_rate
    # Dampened to ~0.5-1.0+
    modifier = 0.5 + modifier * 0.5
    return modifier


def compute_play_probability(player_id: int, status: str,
                              reason: str = "") -> Dict[str, Any]:
    """3-layer blend to compute play probability."""
    # Layer 1: League rate
    league = get_league_play_rate(status)
    league_rate = league["rate"]

    # Layer 2: Player tendency (70/30 blend)
    player = get_player_play_rate(player_id, status)
    player_rate = player["rate"]
    player_obs = player["observations"]

    if player_rate is not None and player_obs >= 5:
        player_weight = min(0.70, player_obs / 20.0)
        blended = player_rate * player_weight + league_rate * (1 - player_weight)
    elif player_rate is not None and player_obs > 0:
        nudge_weight = player_obs / 10.0
        blended = player_rate * nudge_weight + league_rate * (1 - nudge_weight)
    else:
        blended = league_rate

    # Layer 3: Keyword modifier
    keyword = _classify_keyword(reason)
    modifier = get_keyword_modifier(keyword, status)
    composite = max(0.0, min(1.0, blended * modifier))

    return {
        "play_probability": round(composite, 3),
        "league_rate": round(league_rate, 3),
        "player_rate": round(player_rate, 3) if player_rate is not None else None,
        "keyword": keyword,
        "keyword_modifier": round(modifier, 3),
        "confidence": league["confidence"],
        "player_observations": player_obs,
    }


def backfill_play_outcomes():
    """Cross-reference injury_status_log with player_stats to determine did_play."""
    # Get unresolved entries
    unresolved = db.fetch_all("""
        SELECT id, player_id, team_id, log_date
        FROM injury_status_log
        WHERE did_play IS NULL
        ORDER BY log_date
    """)

    if not unresolved:
        return 0

    # Build set of (player_id, game_date) that have stats
    played_set = set()
    rows = db.fetch_all("SELECT DISTINCT player_id, game_date FROM player_stats")
    for r in rows:
        played_set.add((r["player_id"], r["game_date"]))

    updated = 0
    for entry in unresolved:
        pid = entry["player_id"]
        tid = entry["team_id"]
        log_date = entry["log_date"]

        # Find team's first game after log_date
        # player_stats has no team_id; join through players table
        game = db.fetch_one("""
            SELECT DISTINCT ps.game_date FROM player_stats ps
            JOIN players p ON ps.player_id = p.player_id
            WHERE p.team_id = ? AND ps.game_date > ?
            ORDER BY ps.game_date ASC LIMIT 1
        """, (tid, log_date))

        if not game:
            continue

        game_date = game["game_date"]
        did_play = (pid, game_date) in played_set

        db.execute("""
            UPDATE injury_status_log SET did_play = ?, next_game_date = ?
            WHERE id = ?
        """, (did_play, game_date, entry["id"]))
        updated += 1

    logger.info(f"Backfilled {updated} injury status outcomes")
    return updated


def get_team_injury_impact(team_id: int, as_of_date: str = None) -> Dict[str, Any]:
    """Get current injury impact for a team."""
    query = """
        SELECT i.player_id, i.player_name, i.status, i.reason,
               COALESCE((SELECT AVG(ps.points) FROM player_stats ps
                         WHERE ps.player_id = i.player_id
                         ORDER BY ps.game_date DESC LIMIT 20), 0) as ppg,
               COALESCE((SELECT AVG(ps.rebounds) FROM player_stats ps
                         WHERE ps.player_id = i.player_id
                         ORDER BY ps.game_date DESC LIMIT 20), 0) as rpg,
               COALESCE((SELECT AVG(ps.assists) FROM player_stats ps
                         WHERE ps.player_id = i.player_id
                         ORDER BY ps.game_date DESC LIMIT 20), 0) as apg,
               COALESCE((SELECT AVG(ps.minutes) FROM player_stats ps
                         WHERE ps.player_id = i.player_id
                         ORDER BY ps.game_date DESC LIMIT 20), 0) as mpg,
               p.position
        FROM injuries i
        LEFT JOIN players p ON i.player_id = p.player_id
        WHERE i.team_id = ?
    """
    injuries = db.fetch_all(query, (team_id,))

    if not injuries:
        return {"injuries": [], "total_ppg_at_risk": 0, "total_mpg_at_risk": 0}

    result_injuries = []
    total_ppg_risk = 0
    total_mpg_risk = 0

    for inj in injuries:
        prob = compute_play_probability(
            inj["player_id"], inj["status"], inj.get("reason", "")
        )
        absent_fraction = 1.0 - prob["play_probability"]
        ppg = inj.get("ppg", 0) or 0
        mpg = inj.get("mpg", 0) or 0

        ppg_at_risk = ppg * absent_fraction
        mpg_at_risk = mpg * absent_fraction

        total_ppg_risk += ppg_at_risk
        total_mpg_risk += mpg_at_risk

        result_injuries.append({
            "player_id": inj["player_id"],
            "player_name": inj["player_name"],
            "status": inj["status"],
            "reason": inj.get("reason", ""),
            "play_probability": prob["play_probability"],
            "ppg": ppg,
            "mpg": mpg,
            "ppg_at_risk": round(ppg_at_risk, 1),
            "mpg_at_risk": round(mpg_at_risk, 1),
            "keyword": prob["keyword"],
            "confidence": prob["confidence"],
        })

    result_injuries.sort(key=lambda x: x["ppg_at_risk"], reverse=True)

    return {
        "injuries": result_injuries,
        "total_ppg_at_risk": round(total_ppg_risk, 1),
        "total_mpg_at_risk": round(total_mpg_risk, 1),
    }
