"""American odds ↔ probability conversion utilities."""


def american_to_probability(odds: float) -> float:
    """Convert American odds to implied probability (0-1)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 0.5


def probability_to_american(prob: float) -> int:
    """Convert probability (0-1) to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob > 0.5:
        return int(-prob * 100.0 / (1.0 - prob))
    else:
        return int((1.0 - prob) * 100.0 / prob)


def expected_value(my_probability: float, market_odds: float) -> float:
    """Calculate expected value of a bet.
    
    Args:
        my_probability: Our estimated true probability (0-1)
        market_odds: American odds from the market
    """
    if market_odds < 0:
        payout = 100.0 / abs(market_odds)
    elif market_odds > 0:
        payout = market_odds / 100.0
    else:
        return 0.0
    return my_probability * payout - (1.0 - my_probability)


def spread_to_moneyline(spread: float) -> tuple:
    """Rough conversion from point spread to moneyline odds."""
    if abs(spread) < 0.5:
        return (-110, -110)
    # Approximate conversion
    if spread < 0:  # home favored
        home_prob = min(0.95, 0.5 + abs(spread) * 0.03)
    else:
        home_prob = max(0.05, 0.5 - abs(spread) * 0.03)
    away_prob = 1.0 - home_prob
    return (probability_to_american(home_prob), probability_to_american(away_prob))
