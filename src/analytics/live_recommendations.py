"""Live betting recommendations based on live prediction vs market lines."""

import logging
from typing import Dict, Any, Optional, List

from src.analytics.live_prediction import live_predict
from src.analytics.odds_converter import american_to_probability, expected_value

logger = logging.getLogger(__name__)


def get_live_recommendations(home_team_id: int, away_team_id: int,
                              home_score: float, away_score: float,
                              minutes_elapsed: float,
                              quarter: int = 0,
                              home_cum_quarters: float = 0,
                              away_cum_quarters: float = 0,
                              market_spread: Optional[float] = None,
                              market_total: Optional[float] = None,
                              market_home_ml: Optional[int] = None,
                              market_away_ml: Optional[int] = None) -> Dict[str, Any]:
    """Generate live betting recommendations.
    
    Compares model predictions to market lines and identifies edges.
    """
    # Get live prediction
    pred = live_predict(
        home_team_id, away_team_id,
        home_score, away_score,
        minutes_elapsed,
        quarter=quarter,
        home_cum_quarters=home_cum_quarters,
        away_cum_quarters=away_cum_quarters,
    )

    recommendations = []
    edges = {}

    model_spread = pred["spread"]
    model_total = pred["total"]

    # Spread recommendation
    if market_spread is not None:
        spread_diff = model_spread - market_spread
        if abs(spread_diff) >= 2.0:
            if spread_diff > 0:
                rec = {
                    "type": "spread",
                    "side": "HOME",
                    "edge": round(spread_diff, 1),
                    "reasoning": f"Model: {model_spread:+.1f}, Market: {market_spread:+.1f}",
                    "confidence": _spread_confidence(spread_diff, minutes_elapsed),
                }
            else:
                rec = {
                    "type": "spread",
                    "side": "AWAY",
                    "edge": round(abs(spread_diff), 1),
                    "reasoning": f"Model: {model_spread:+.1f}, Market: {market_spread:+.1f}",
                    "confidence": _spread_confidence(abs(spread_diff), minutes_elapsed),
                }
            recommendations.append(rec)
        edges["spread"] = round(spread_diff, 1)

    # Total recommendation
    if market_total is not None:
        total_diff = model_total - market_total
        if abs(total_diff) >= 3.0:
            side = "OVER" if total_diff > 0 else "UNDER"
            rec = {
                "type": "total",
                "side": side,
                "edge": round(abs(total_diff), 1),
                "reasoning": f"Model: {model_total:.1f}, Market: {market_total:.1f}",
                "confidence": _total_confidence(abs(total_diff), minutes_elapsed),
            }
            recommendations.append(rec)
        edges["total"] = round(total_diff, 1)

    # Moneyline recommendation
    if market_home_ml is not None and market_away_ml is not None:
        # Model implied probability
        if model_spread > 0.5:
            model_home_prob = min(0.95, 0.5 + model_spread / 40.0)
        elif model_spread < -0.5:
            model_home_prob = max(0.05, 0.5 + model_spread / 40.0)
        else:
            model_home_prob = 0.5

        model_away_prob = 1.0 - model_home_prob

        market_home_prob = american_to_probability(market_home_ml)
        market_away_prob = american_to_probability(market_away_ml)

        home_ev = expected_value(model_home_prob, market_home_ml)
        away_ev = expected_value(model_away_prob, market_away_ml)

        if home_ev > 0.03:
            recommendations.append({
                "type": "moneyline",
                "side": "HOME",
                "edge": round(home_ev * 100, 1),
                "reasoning": f"Model: {model_home_prob:.0%}, Market: {market_home_prob:.0%}, EV: {home_ev:.1%}",
                "confidence": _ml_confidence(home_ev, minutes_elapsed),
            })

        if away_ev > 0.03:
            recommendations.append({
                "type": "moneyline",
                "side": "AWAY",
                "edge": round(away_ev * 100, 1),
                "reasoning": f"Model: {model_away_prob:.0%}, Market: {market_away_prob:.0%}, EV: {away_ev:.1%}",
                "confidence": _ml_confidence(away_ev, minutes_elapsed),
            })

        edges["home_ev"] = round(home_ev, 4)
        edges["away_ev"] = round(away_ev, 4)

    # Add advisories from prediction
    for adv in pred.get("advisories", []):
        recommendations.append({
            "type": "advisory",
            "side": adv["signal"],
            "edge": 0,
            "reasoning": adv["signal"],
            "confidence": "medium",
        })

    return {
        "prediction": pred,
        "recommendations": recommendations,
        "edges": edges,
        "minutes_elapsed": minutes_elapsed,
    }


def _spread_confidence(edge: float, minutes: float) -> str:
    """Confidence in spread recommendation."""
    # Higher confidence with more time elapsed and larger edge
    if minutes >= 36 and edge >= 4:
        return "high"
    if minutes >= 24 and edge >= 3:
        return "high"
    if minutes >= 12 and edge >= 5:
        return "medium"
    if edge >= 3:
        return "medium"
    return "low"


def _total_confidence(edge: float, minutes: float) -> str:
    """Confidence in total recommendation."""
    if minutes >= 36 and edge >= 6:
        return "high"
    if minutes >= 24 and edge >= 5:
        return "medium"
    if edge >= 4:
        return "medium"
    return "low"


def _ml_confidence(ev: float, minutes: float) -> str:
    """Confidence in moneyline recommendation."""
    if minutes >= 36 and ev >= 0.08:
        return "high"
    if minutes >= 24 and ev >= 0.06:
        return "medium"
    if ev >= 0.05:
        return "medium"
    return "low"
