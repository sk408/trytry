"""Prediction quality, value-add metrics, and season progression tracking."""

import logging
from typing import List, Dict, Any
from datetime import datetime
from src.database import db

logger = logging.getLogger(__name__)

def compute_quality_metrics(per_game: List[Dict[str, Any]], per_team: Dict[str, Any]) -> Dict[str, Any]:
    """Compute value-add metrics from backtest per_game results."""
    if not per_game:
        return {}

    # 1. Fetch team records
    try:
        rows = db.fetch_all("SELECT team_id, w_pct, w, l FROM team_metrics")
        team_records = {r["team_id"]: {"w_pct": r.get("w_pct", 0.5), "w": r.get("w", 0), "l": r.get("l", 0)} for r in rows}
    except Exception as e:
        logger.warning(f"Could not fetch team_metrics for quality calculation: {e}")
        team_records = {}

    total_games = len(per_game)
    model_correct = sum(1 for g in per_game if g.get("winner_correct"))
    model_acc = (model_correct / total_games * 100) if total_games else 0

    # 2. Naive Baseline & Upset Detection
    naive_correct = 0
    upset_opportunities = 0
    upsets_detected = 0

    # 3. Confidence Calibration
    conf_buckets = {
        "toss_up": {"games": 0, "correct": 0, "label": "< 3 points"},
        "moderate": {"games": 0, "correct": 0, "label": "3-7 points"},
        "strong": {"games": 0, "correct": 0, "label": "7+ points"}
    }

    # 4. Tier Accuracy
    # Group teams into tiers based on w_pct
    sorted_teams = sorted(team_records.items(), key=lambda x: x[1]["w_pct"], reverse=True)
    elite_ids = {t[0] for t in sorted_teams[:10]}
    bottom_ids = {t[0] for t in sorted_teams[-10:]} if len(sorted_teams) >= 20 else set()
    middle_ids = {t[0] for t in sorted_teams if t[0] not in elite_ids and t[0] not in bottom_ids}

    def get_tier(tid):
        if tid in elite_ids: return "Elite"
        if tid in bottom_ids: return "Bottom"
        return "Middle"

    tier_combos = {}

    for g in per_game:
        htid = g["home_team_id"]
        atid = g["away_team_id"]
        hw = team_records.get(htid, {}).get("w", 0)
        aw = team_records.get(atid, {}).get("w", 0)
        hpct = team_records.get(htid, {}).get("w_pct", 0.5)
        apct = team_records.get(atid, {}).get("w_pct", 0.5)

        # Naive Baseline
        # "Always pick team with better win%"
        naive_winner = "HOME" if hpct > apct else ("AWAY" if apct > hpct else "HOME") # tie goes home
        actual_winner = g.get("winner", "PUSH")
        if naive_winner == actual_winner or (actual_winner == "PUSH"): # simplified
            naive_correct += 1

        # Upset Detection (records differ by > 5 games)
        # If the actual winner was the team with fewer wins, it's a true upset.
        # We only care if we predicted it correctly.
        if abs(hw - aw) > 5:
            underdog = "HOME" if hw < aw else "AWAY"
            if actual_winner == underdog:
                upset_opportunities += 1
                pred_winner = "HOME" if g.get("pred_spread", 0) > 0 else "AWAY"
                if pred_winner == underdog:
                    upsets_detected += 1

        # Confidence Calibration
        abs_spread = abs(g.get("pred_spread", 0))
        bucket = "toss_up" if abs_spread < 3 else ("moderate" if abs_spread <= 7 else "strong")
        conf_buckets[bucket]["games"] += 1
        if g.get("winner_correct"):
            conf_buckets[bucket]["correct"] += 1

        # Tiers
        htier = get_tier(htid)
        atier = get_tier(atid)
        combo = tuple(sorted([htier, atier])) # e.g. ('Bottom', 'Elite')
        combo_str = f"{combo[0]} vs {combo[1]}"
        if combo_str not in tier_combos:
            tier_combos[combo_str] = {"games": 0, "correct": 0}
        tier_combos[combo_str]["games"] += 1
        if g.get("winner_correct"):
            tier_combos[combo_str]["correct"] += 1

    baseline_acc = (naive_correct / total_games * 100) if total_games else 0
    skill_score = (model_acc - baseline_acc) / (100 - baseline_acc) if (100 - baseline_acc) > 0 else 0

    for v in conf_buckets.values():
        v["accuracy"] = round(v["correct"] / v["games"] * 100, 1) if v["games"] else 0

    for v in tier_combos.values():
        v["accuracy"] = round(v["correct"] / v["games"] * 100, 1) if v["games"] else 0

    upset_rate = round(upsets_detected / upset_opportunities * 100, 1) if upset_opportunities else 0

    return {
        "naive_baseline_acc": round(baseline_acc, 1),
        "skill_score": round(skill_score, 3),
        "upset_detection_rate": upset_rate,
        "upset_opportunities": upset_opportunities,
        "upsets_detected": upsets_detected,
        "confidence_buckets": conf_buckets,
        "tier_accuracy": tier_combos
    }

def compute_progression(per_game: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute monthly, rolling, and cumulative accuracy progression."""
    if not per_game:
        return {}

    # Assume per_game is sorted by game_date
    monthly_data = {}
    cumulative_curve = []
    
    total_games = 0
    total_correct = 0

    # For rolling 30
    rolling_window = []
    rolling_results = []

    for g in per_game:
        date_str = g.get("game_date", "")
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            month_key = dt.strftime("%Y-%m")
        except ValueError:
            month_key = "Unknown"

        if month_key not in monthly_data:
            monthly_data[month_key] = {"games": 0, "correct": 0, "spread_errors": []}
        
        monthly_data[month_key]["games"] += 1
        if g.get("winner_correct"):
            monthly_data[month_key]["correct"] += 1
        monthly_data[month_key]["spread_errors"].append(g.get("spread_error", 0))

        # Cumulative
        total_games += 1
        if g.get("winner_correct"):
            total_correct += 1
        
        if total_games % 10 == 0 or total_games == len(per_game): # Sample every 10 games to avoid huge arrays
            cumulative_curve.append({
                "game_num": total_games,
                "accuracy": round(total_correct / total_games * 100, 1)
            })

        # Rolling
        rolling_window.append(g)
        if len(rolling_window) > 30:
            rolling_window.pop(0)
        
        if total_games % 10 == 0 and len(rolling_window) >= 10:
            rcorrect = sum(1 for rw in rolling_window if rw.get("winner_correct"))
            rolling_results.append({
                "game_num": total_games,
                "accuracy": round(rcorrect / len(rolling_window) * 100, 1)
            })

    monthly_breakdown = []
    for m, d in sorted(monthly_data.items()):
        acc = round(d["correct"] / d["games"] * 100, 1) if d["games"] else 0
        mae = round(sum(d["spread_errors"]) / len(d["spread_errors"]), 2) if d["spread_errors"] else 0
        monthly_breakdown.append({
            "month": m,
            "games": d["games"],
            "accuracy": acc,
            "mae": mae
        })

    return {
        "monthly_breakdown": monthly_breakdown,
        "rolling_30": rolling_results,
        "cumulative_curve": cumulative_curve
    }

def compute_vegas_comparison(per_game: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare model predictions against stored Vegas odds, calculate ATS and betting value."""
    if not per_game:
        return {}

    # Fetch all odds for these games
    try:
        rows = db.fetch_all("SELECT game_date, home_team_id, away_team_id, spread, home_moneyline, away_moneyline FROM game_odds")
        odds_map = {(r["game_date"], r["home_team_id"], r["away_team_id"]): r for r in rows}
    except Exception as e:
        logger.warning(f"Could not fetch game_odds: {e}")
        odds_map = {}

    if not odds_map:
        return {"error": "No odds data available"}

    ats_covers = 0
    ats_pushes = 0
    ats_games = 0
    
    clv_wins = 0
    clv_games = 0

    edge_games = []
    edge_threshold = 3.0
    
    # Bankroll Simulation
    bankroll = 1000.0
    
    for g in per_game:
        key = (g.get("game_date"), g.get("home_team_id"), g.get("away_team_id"))
        odds = odds_map.get(key)
        if not odds or odds["spread"] is None:
            continue
            
        vegas_spread = odds["spread"] # Negative usually means home favored
        vegas_margin = -vegas_spread # expected home_score - away_score
        
        pred_spread = g.get("pred_spread", 0)
        actual_spread = g.get("actual_spread", 0)
        
        # ATS logic
        ats_games += 1
        pick_home = pred_spread > vegas_margin
        actual_home_cover = actual_spread > vegas_margin
        actual_push = actual_spread == vegas_margin
        
        if actual_push:
            ats_pushes += 1
        elif pick_home == actual_home_cover:
            ats_covers += 1

        # CLV (Model vs Vegas, who was closer to actual?)
        if abs(pred_spread - actual_spread) < abs(vegas_margin - actual_spread):
            clv_wins += 1
        clv_games += 1

        # Edge Detection
        edge_size = abs(pred_spread - vegas_margin)
        if edge_size >= edge_threshold:
            covered = (pick_home and actual_home_cover) or (not pick_home and not actual_home_cover and not actual_push)
            
            # Simple Kelly & Bankroll sim (assume -110 odds on spread, payout 0.909)
            p = 0.5 + min(edge_size * 0.02, 0.45) # Rough probability mapping
            b = 100 / 110 # decimal odds - 1
            kelly_fraction = max(0, p - (1 - p) / b)
            
            bet_size = bankroll * (kelly_fraction * 0.25) # Quarter Kelly
            if bet_size > 0 and not actual_push:
                if covered:
                    profit = bet_size * (100 / 110)
                else:
                    profit = -bet_size
                bankroll += profit

            edge_games.append({
                "game_id": g.get("game_id"),
                "date": g.get("game_date"),
                "home_id": g.get("home_team_id"),
                "away_id": g.get("away_team_id"),
                "vegas_spread": vegas_spread,
                "pred_spread": pred_spread,
                "actual_spread": actual_spread,
                "edge_size": round(edge_size, 2),
                "covered": covered,
                "kelly_rec": round(kelly_fraction, 3),
                "fatigue_adj": g.get("fatigue_adj", 0),
                "rating_matchup_adj": g.get("rating_matchup_adj", 0),
                "ml_blend_adj": g.get("ml_blend_adj", 0),
            })

    ats_rate = round(ats_covers / (ats_games - ats_pushes) * 100, 1) if ats_games - ats_pushes > 0 else 0
    clv_rate = round(clv_wins / clv_games * 100, 1) if clv_games > 0 else 0
    
    edge_hit_rate = round(sum(1 for e in edge_games if e["covered"]) / len(edge_games) * 100, 1) if edge_games else 0
    
    # Feature attribution (what drove the successful edges?)
    feature_attr = {}
    if edge_games:
        covered_edges = [e for e in edge_games if e["covered"]]
        if covered_edges:
            feature_attr = {
                "avg_fatigue_adj": round(sum(abs(e.get("fatigue_adj", 0)) for e in covered_edges) / len(covered_edges), 2),
                "avg_rating_adj": round(sum(abs(e.get("rating_matchup_adj", 0)) for e in covered_edges) / len(covered_edges), 2),
                "avg_ml_adj": round(sum(abs(e.get("ml_blend_adj", 0)) for e in covered_edges) / len(covered_edges), 2),
            }

    return {
        "ats_games": ats_games,
        "ats_rate": ats_rate,
        "clv_rate": clv_rate,
        "edge_hit_rate": edge_hit_rate,
        "total_edge_games": len(edge_games),
        "final_bankroll": round(bankroll, 2),
        "roi_pct": round((bankroll - 1000) / 1000 * 100, 1),
        "feature_attribution": feature_attr
    }
