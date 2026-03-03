"""Shared test fixtures for NBA analytics tests."""

import pytest
from unittest.mock import MagicMock, patch
from src.analytics.prediction import PrecomputedGame
from src.analytics.weight_config import WeightConfig


@pytest.fixture
def default_weights():
    """Fresh WeightConfig with default values."""
    return WeightConfig()


@pytest.fixture
def sample_game():
    """A sample PrecomputedGame with realistic values."""
    return PrecomputedGame(
        game_date="2025-01-15",
        season="2024-25",
        home_team_id=1610612744,  # GSW
        away_team_id=1610612747,  # LAL
        actual_home_score=112.0,
        actual_away_score=108.0,
        home_proj={"points": 110.0, "rebounds": 44.0, "assists": 26.0,
                   "turnovers": 14.0, "steals": 7.0, "blocks": 5.0,
                   "oreb": 10.0, "minutes": 240.0,
                   "fg_made": 40.0, "fg_attempted": 88.0,
                   "fg3_made": 12.0, "fg3_attempted": 34.0,
                   "ft_made": 18.0, "ft_attempted": 22.0},
        away_proj={"points": 108.0, "rebounds": 42.0, "assists": 24.0,
                   "turnovers": 15.0, "steals": 8.0, "blocks": 4.0,
                   "oreb": 9.0, "minutes": 240.0,
                   "fg_made": 39.0, "fg_attempted": 86.0,
                   "fg3_made": 11.0, "fg3_attempted": 32.0,
                   "ft_made": 19.0, "ft_attempted": 24.0},
        home_court=3.0,
        away_def_factor_raw=1.02,
        home_def_factor_raw=0.98,
        home_fatigue_penalty=0.0,
        away_fatigue_penalty=0.0,
        home_rest_days=2,
        away_rest_days=1,
        home_b2b=False,
        away_b2b=True,
        home_off=112.0,
        away_off=110.0,
        home_def=108.0,
        away_def=109.0,
        home_pace=100.0,
        away_pace=99.0,
        home_ff={"efg": 0.54, "tov": 0.13, "oreb": 0.28, "fta": 0.25},
        away_ff={"efg": 0.52, "tov": 0.14, "oreb": 0.26, "fta": 0.24},
        home_clutch={"net_rating": 5.0},
        away_clutch={"net_rating": 2.0},
        home_hustle={"deflections": 15.0, "contested": 55.0},
        away_hustle={"deflections": 14.0, "contested": 52.0},
        vegas_spread=-3.5,
        vegas_home_ml=-160,
        vegas_away_ml=140,
    )


@pytest.fixture
def denver_home_game():
    """PrecomputedGame where away team is B2B at Denver (altitude penalty)."""
    return PrecomputedGame(
        game_date="2025-02-10",
        season="2024-25",
        home_team_id=1610612743,  # DEN
        away_team_id=1610612747,  # LAL
        actual_home_score=115.0,
        actual_away_score=105.0,
        home_proj={"points": 112.0, "rebounds": 45.0, "assists": 27.0,
                   "turnovers": 13.0, "steals": 7.0, "blocks": 5.0,
                   "oreb": 10.0, "minutes": 240.0},
        away_proj={"points": 106.0, "rebounds": 41.0, "assists": 23.0,
                   "turnovers": 15.0, "steals": 6.0, "blocks": 4.0,
                   "oreb": 9.0, "minutes": 240.0},
        home_court=3.5,
        away_def_factor_raw=1.01,
        home_def_factor_raw=0.99,
        home_fatigue_penalty=0.0,
        away_fatigue_penalty=1.5,
        home_rest_days=3,
        away_rest_days=1,
        home_b2b=False,
        away_b2b=True,
        home_off=114.0,
        away_off=110.0,
        home_def=106.0,
        away_def=110.0,
        home_pace=101.0,
        away_pace=99.0,
        home_ff={"efg": 0.55, "tov": 0.12, "oreb": 0.29, "fta": 0.26},
        away_ff={"efg": 0.52, "tov": 0.14, "oreb": 0.26, "fta": 0.24},
        home_clutch={"net_rating": 6.0},
        away_clutch={"net_rating": 1.0},
        home_hustle={"deflections": 16.0, "contested": 58.0},
        away_hustle={"deflections": 13.0, "contested": 50.0},
        vegas_spread=-7.0,
    )


@pytest.fixture
def mock_db():
    """Mock database that returns empty results."""
    with patch("src.database.db") as mock:
        mock.fetch_all.return_value = []
        mock.fetch_one.return_value = None
        mock.execute.return_value = None
        yield mock
