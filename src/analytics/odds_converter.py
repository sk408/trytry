from __future__ import annotations


def american_to_probability(odds: int) -> float:
    """Convert American odds (-110, +150) to implied probability (0-1)."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def probability_to_american(probability: float) -> int:
    """Convert probability (0-1) back to American odds."""
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1")
    if probability > 0.5:
        return int(-probability * 100 / (1 - probability))
    return int((1 - probability) * 100 / probability)


def to_robinhood_probability(american_odds: int) -> float:
    """
    Robinhood prediction markets operate on probability shares (0-1 scale).
    """
    return round(american_to_probability(american_odds), 4)


def expected_value(my_probability: float, market_odds: int) -> float:
    """
    Calculate expected value per unit stake using your win probability
    and market odds.
    """
    implied = american_to_probability(market_odds)
    payout = abs(100 / market_odds) if market_odds < 0 else market_odds / 100
    return my_probability * payout - (1 - my_probability)
