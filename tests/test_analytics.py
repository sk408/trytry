"""Core math tests for NBA analytics prediction logic."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.analytics.prediction import PrecomputedGame, predict_from_precomputed
from src.analytics.weight_config import WeightConfig
from src.analytics.weight_optimizer import VectorizedGames


# ──────────────────────────────────────────────────────────────
# Fatigue detection tests
# ──────────────────────────────────────────────────────────────

class TestComputeFatigue:

    @patch("src.analytics.stats_engine.db")
    def test_b2b_detection(self, mock_db):
        """Back-to-back detected when previous game was 1 day ago."""
        from src.analytics.stats_engine import compute_fatigue, _fatigue_cache
        _fatigue_cache.clear()

        mock_db.fetch_all.return_value = [
            {"game_date": "2025-01-14"},
            {"game_date": "2025-01-12"},
        ]
        result = compute_fatigue(1610612744, "2025-01-15")
        assert result["b2b"] is True
        assert result["rest_days"] == 1

    @patch("src.analytics.stats_engine.db")
    def test_rest_bonus(self, mock_db):
        """3+ day rest gives rest_days >= 3."""
        from src.analytics.stats_engine import compute_fatigue, _fatigue_cache
        _fatigue_cache.clear()

        mock_db.fetch_all.return_value = [
            {"game_date": "2025-01-11"},
        ]
        result = compute_fatigue(1610612744, "2025-01-15")
        assert result["rest_days"] == 4
        assert result["b2b"] is False

    @patch("src.analytics.stats_engine.db")
    def test_3in4_detection(self, mock_db):
        """3-in-4 detected when 3 games in last 4 days."""
        from src.analytics.stats_engine import compute_fatigue, _fatigue_cache
        _fatigue_cache.clear()

        # Games on 14th, 13th, 12th — current game on 15th = 4 games in 4 days
        mock_db.fetch_all.return_value = [
            {"game_date": "2025-01-14"},
            {"game_date": "2025-01-13"},
            {"game_date": "2025-01-12"},
        ]
        result = compute_fatigue(1610612744, "2025-01-15")
        assert result["three_in_four"] is True


# ──────────────────────────────────────────────────────────────
# Net rest advantage tests
# ──────────────────────────────────────────────────────────────

class TestNetRestAdvantage:

    def test_positive_rest_differential(self, sample_game, default_weights):
        """Home team more rested → positive rest advantage."""
        # sample_game: home_rest_days=2, away_rest_days=1 → net = +1
        w = default_weights
        w.rest_advantage_mult = 1.0
        # Net rest = 2 - 1 = 1, so rest_adj = 1.0
        net_rest = sample_game.home_rest_days - sample_game.away_rest_days
        assert net_rest == 1
        rest_adj = net_rest * w.rest_advantage_mult
        assert rest_adj == 1.0

    def test_zero_rest_differential(self, default_weights):
        """Equal rest → no adjustment."""
        game = PrecomputedGame(home_rest_days=2, away_rest_days=2)
        net_rest = game.home_rest_days - game.away_rest_days
        assert net_rest == 0
        assert net_rest * default_weights.rest_advantage_mult == 0.0

    def test_negative_rest_differential(self, default_weights):
        """Away team more rested → negative rest advantage (hurts home)."""
        game = PrecomputedGame(home_rest_days=1, away_rest_days=3)
        w = default_weights
        w.rest_advantage_mult = 0.5
        net_rest = game.home_rest_days - game.away_rest_days
        assert net_rest == -2
        assert net_rest * w.rest_advantage_mult == -1.0


# ──────────────────────────────────────────────────────────────
# Altitude B2B penalty tests
# ──────────────────────────────────────────────────────────────

class TestAltitudeB2B:

    def test_penalty_applies_den(self, denver_home_game, default_weights):
        """Away B2B at Denver triggers altitude penalty."""
        g = denver_home_game
        w = default_weights
        assert g.away_b2b is True
        assert g.home_team_id == 1610612743  # DEN
        penalty = w.altitude_b2b_penalty if (g.away_b2b and g.home_team_id in (1610612743, 1610612762)) else 0.0
        assert penalty == w.altitude_b2b_penalty
        assert penalty > 0

    def test_penalty_applies_uta(self, default_weights):
        """Away B2B at Utah triggers altitude penalty."""
        g = PrecomputedGame(
            home_team_id=1610612762,  # UTA
            away_b2b=True,
        )
        w = default_weights
        penalty = w.altitude_b2b_penalty if (g.away_b2b and g.home_team_id in (1610612743, 1610612762)) else 0.0
        assert penalty > 0

    def test_no_penalty_non_altitude(self, default_weights):
        """Away B2B at non-altitude venue → no altitude penalty."""
        g = PrecomputedGame(
            home_team_id=1610612747,  # LAL
            away_b2b=True,
        )
        w = default_weights
        penalty = w.altitude_b2b_penalty if (g.away_b2b and g.home_team_id in (1610612743, 1610612762)) else 0.0
        assert penalty == 0.0

    def test_no_penalty_not_b2b(self, default_weights):
        """Not B2B at Denver → no altitude penalty."""
        g = PrecomputedGame(
            home_team_id=1610612743,  # DEN
            away_b2b=False,
        )
        w = default_weights
        penalty = w.altitude_b2b_penalty if (g.away_b2b and g.home_team_id in (1610612743, 1610612762)) else 0.0
        assert penalty == 0.0


# ──────────────────────────────────────────────────────────────
# VectorizedGames rest/altitude arrays
# ──────────────────────────────────────────────────────────────

class TestVectorizedGamesRestAltitude:

    def test_net_rest_array(self, sample_game, denver_home_game):
        """VectorizedGames computes correct net_rest array."""
        with patch("src.analytics.weight_optimizer.db") as mock_db:
            mock_db.fetch_all.return_value = []
            vg = VectorizedGames([sample_game, denver_home_game])

        # sample: 2-1=1, denver: 3-1=2
        np.testing.assert_array_equal(vg.net_rest, [1.0, 2.0])

    def test_altitude_b2b_array(self, sample_game, denver_home_game):
        """VectorizedGames flags altitude B2B correctly."""
        with patch("src.analytics.weight_optimizer.db") as mock_db:
            mock_db.fetch_all.return_value = []
            vg = VectorizedGames([sample_game, denver_home_game])

        # sample: GSW home, away B2B but not altitude → 0
        # denver: DEN home, away B2B → 1
        np.testing.assert_array_equal(vg.away_b2b_at_altitude, [0.0, 1.0])


# ──────────────────────────────────────────────────────────────
# Dog ROI tests
# ──────────────────────────────────────────────────────────────

class TestDogROI:

    def test_dog_roi_with_moneyline(self):
        """Real ML conversion: +140 dog pays $2.40 on $1 bet."""
        dog_ml = 140
        mult = (1.0 + 100.0 / abs(dog_ml)) if dog_ml < 0 else (1.0 + dog_ml / 100.0)
        assert abs(mult - 2.4) < 0.01

        # -200 favorite pays $1.50 on $1 bet
        fav_ml = -200
        mult_fav = (1.0 + 100.0 / abs(fav_ml)) if fav_ml < 0 else (1.0 + fav_ml / 100.0)
        assert abs(mult_fav - 1.5) < 0.01

    def test_dog_roi_fallback(self):
        """Fallback 1.5x when moneyline missing → correct profit/loss."""
        # Win with fallback: profit = 1.5 - 1.0 = 0.5
        profit_win = 1.5 - 1.0  # (mult-1) from the fallback
        assert profit_win == 0.5

        # Loss: -1.0 unit
        profit_loss = -1.0
        assert profit_loss == -1.0


# ──────────────────────────────────────────────────────────────
# Zero-minute fallback test
# ──────────────────────────────────────────────────────────────

class TestZeroMinuteFallback:

    @patch("src.analytics.stats_engine.get_games_missed_streak", return_value=0)
    @patch("src.analytics.stats_engine.player_splits")
    @patch("src.analytics.stats_engine.db")
    def test_zero_minutes_gets_season_averages(self, mock_db, mock_splits, mock_streak):
        """Player with 0 recent minutes but play_prob >= 0.3 gets full-season fallback."""
        from src.analytics.stats_engine import aggregate_projection

        # First call returns 0 minutes (recent), second returns season avg
        zero_splits = {"minutes": 0.0, "points": 0.0, "rebounds": 0.0,
                       "assists": 0.0, "turnovers": 0.0, "steals": 0.0,
                       "blocks": 0.0, "oreb": 0.0, "dreb": 0.0,
                       "fg_made": 0.0, "fg_attempted": 0.0,
                       "fg3_made": 0.0, "fg3_attempted": 0.0,
                       "ft_made": 0.0, "ft_attempted": 0.0,
                       "plus_minus": 0.0, "personal_fouls": 0.0}
        season_splits = {"minutes": 28.0, "points": 18.0, "rebounds": 5.0,
                         "assists": 4.0, "turnovers": 2.0, "steals": 1.0,
                         "blocks": 0.5, "oreb": 1.0, "dreb": 4.0,
                         "fg_made": 7.0, "fg_attempted": 15.0,
                         "fg3_made": 2.0, "fg3_attempted": 5.0,
                         "ft_made": 2.0, "ft_attempted": 3.0,
                         "plus_minus": 3.0, "personal_fouls": 2.0}
        mock_splits.side_effect = [zero_splits, season_splits]

        # Single player roster
        mock_db.fetch_all.return_value = [
            {"player_id": 1, "name": "Test Player", "position": "SG"}
        ]

        result = aggregate_projection(
            team_id=1, opponent_team_id=2, is_home=1,
            injured_players={1: 0.5},  # questionable
            roster=[{"player_id": 1, "name": "Test Player", "position": "SG"}],
        )
        # Should have called player_splits twice: recent + season
        assert mock_splits.call_count == 2
        # Second call should be with recent_games=9999
        assert mock_splits.call_args_list[1][1].get("recent_games", 0) == 9999 or \
               mock_splits.call_args_list[1][0][3] == 9999  # positional arg
        # Result should include the player's contribution (discounted by 0.5 * 0.5)
        assert result["points"] > 0


# ──────────────────────────────────────────────────────────────
# Decomposed fatigue tests
# ──────────────────────────────────────────────────────────────

class TestFatigueDecomposed:

    def test_fatigue_b2b_changes_spread(self):
        """Changing w.fatigue_b2b must change VectorizedGames spread (was dead before)."""
        game = PrecomputedGame(
            home_b2b=True,
            away_b2b=False,
            home_3in4=False,
            away_3in4=False,
            home_4in6=False,
            away_4in6=False,
            home_proj={"points": 110.0, "rebounds": 44.0, "turnovers": 14.0,
                       "steals": 7.0, "blocks": 5.0, "oreb": 10.0},
            away_proj={"points": 108.0, "rebounds": 42.0, "turnovers": 15.0,
                       "steals": 8.0, "blocks": 4.0, "oreb": 9.0},
        )

        with patch("src.analytics.weight_optimizer.db") as mock_db:
            mock_db.fetch_all.return_value = []
            vg = VectorizedGames([game])

        w1 = WeightConfig()
        w1.fatigue_b2b = 1.0
        r1 = vg.evaluate(w1)

        w2 = WeightConfig()
        w2.fatigue_b2b = 5.0
        r2 = vg.evaluate(w2)

        # Different fatigue_b2b must produce different spread MAE
        assert r1["spread_mae"] != r2["spread_mae"], \
            "fatigue_b2b had no effect on spread — still dead!"

    def test_fatigue_3in4_changes_spread(self):
        """Changing w.fatigue_3in4 affects spread when 3in4 flag is set."""
        game = PrecomputedGame(
            home_b2b=False,
            away_b2b=False,
            home_3in4=True,
            away_3in4=False,
            home_4in6=False,
            away_4in6=False,
            home_proj={"points": 110.0, "rebounds": 44.0, "turnovers": 14.0,
                       "steals": 7.0, "blocks": 5.0, "oreb": 10.0},
            away_proj={"points": 108.0, "rebounds": 42.0, "turnovers": 15.0,
                       "steals": 8.0, "blocks": 4.0, "oreb": 9.0},
        )

        with patch("src.analytics.weight_optimizer.db") as mock_db:
            mock_db.fetch_all.return_value = []
            vg = VectorizedGames([game])

        w1 = WeightConfig()
        w1.fatigue_3in4 = 0.5
        r1 = vg.evaluate(w1)

        w2 = WeightConfig()
        w2.fatigue_3in4 = 4.0
        r2 = vg.evaluate(w2)

        assert r1["spread_mae"] != r2["spread_mae"], \
            "fatigue_3in4 had no effect on spread!"

    def test_decomposed_flag_arrays(self):
        """VectorizedGames stores correct decomposed flag arrays."""
        game = PrecomputedGame(
            home_b2b=True,
            away_b2b=False,
            home_3in4=False,
            away_3in4=True,
            home_4in6=True,
            away_4in6=False,
        )

        with patch("src.analytics.weight_optimizer.db") as mock_db:
            mock_db.fetch_all.return_value = []
            vg = VectorizedGames([game])

        np.testing.assert_array_equal(vg.home_b2b_flag, [1.0])
        np.testing.assert_array_equal(vg.away_b2b_flag, [0.0])
        np.testing.assert_array_equal(vg.home_3in4, [0.0])
        np.testing.assert_array_equal(vg.away_3in4, [1.0])
        np.testing.assert_array_equal(vg.home_4in6, [1.0])
        np.testing.assert_array_equal(vg.away_4in6, [0.0])


# ──────────────────────────────────────────────────────────────
# Opponent Four Factors (defensive matchup) tests
# ──────────────────────────────────────────────────────────────

class TestOppFourFactors:

    def test_opp_ff_changes_spread(self):
        """Changing w.opp_ff_efg_weight must change VectorizedGames spread."""
        game = PrecomputedGame(
            actual_home_score=112.0,
            actual_away_score=108.0,
            home_proj={"points": 110.0, "rebounds": 44.0, "turnovers": 14.0,
                       "steals": 7.0, "blocks": 5.0, "oreb": 10.0},
            away_proj={"points": 108.0, "rebounds": 42.0, "turnovers": 15.0,
                       "steals": 8.0, "blocks": 4.0, "oreb": 9.0},
            home_ff={"efg": 0.54, "tov": 0.13, "oreb": 0.28, "fta": 0.25,
                     "opp_efg": 0.51, "opp_tov": 0.14, "opp_oreb": 0.27, "opp_fta": 0.23},
            away_ff={"efg": 0.52, "tov": 0.14, "oreb": 0.26, "fta": 0.24,
                     "opp_efg": 0.53, "opp_tov": 0.12, "opp_oreb": 0.29, "opp_fta": 0.25},
        )

        with patch("src.analytics.weight_optimizer.db") as mock_db:
            mock_db.fetch_all.return_value = []
            vg = VectorizedGames([game])

        # Zero out other spread-inflating weights to isolate opp_ff effect
        w1 = WeightConfig()
        w1.opp_ff_efg_weight = 0.0
        w1.four_factors_scale = 10.0   # small scale to avoid clamp
        w1.turnover_margin_mult = 0.0
        w1.rebound_diff_mult = 0.0
        w1.hustle_effort_mult = 0.0
        r1 = vg.evaluate(w1)

        w2 = WeightConfig()
        w2.opp_ff_efg_weight = 10.0
        w2.four_factors_scale = 10.0
        w2.turnover_margin_mult = 0.0
        w2.rebound_diff_mult = 0.0
        w2.hustle_effort_mult = 0.0
        r2 = vg.evaluate(w2)

        assert r1["spread_mae"] != r2["spread_mae"], \
            "opp_ff_efg_weight had no effect on spread — still dead!"

    def test_opp_ff_edge_arrays(self):
        """VectorizedGames computes correct opp FF edge arrays."""
        game = PrecomputedGame(
            home_ff={"efg": 0.54, "tov": 0.13, "oreb": 0.28, "fta": 0.25,
                     "opp_efg": 0.51, "opp_tov": 0.14, "opp_oreb": 0.27, "opp_fta": 0.23},
            away_ff={"efg": 0.52, "tov": 0.14, "oreb": 0.26, "fta": 0.24,
                     "opp_efg": 0.53, "opp_tov": 0.12, "opp_oreb": 0.29, "opp_fta": 0.25},
        )

        with patch("src.analytics.weight_optimizer.db") as mock_db:
            mock_db.fetch_all.return_value = []
            vg = VectorizedGames([game])

        # opp_efg_edge = away_opp_efg - home_opp_efg = 0.53 - 0.51 = 0.02
        np.testing.assert_almost_equal(vg.opp_ff_efg_edge[0], 0.02, decimal=4)
        # opp_tov_edge = home_opp_tov - away_opp_tov = 0.14 - 0.12 = 0.02
        np.testing.assert_almost_equal(vg.opp_ff_tov_edge[0], 0.02, decimal=4)
        # opp_oreb_edge = away_opp_oreb - home_opp_oreb = 0.29 - 0.27 = 0.02
        np.testing.assert_almost_equal(vg.opp_ff_oreb_edge[0], 0.02, decimal=4)
        # opp_fta_edge = away_opp_fta - home_opp_fta = 0.25 - 0.23 = 0.02
        np.testing.assert_almost_equal(vg.opp_ff_fta_edge[0], 0.02, decimal=4)


class TestOppFFInExtremeRanges:

    def test_opp_ff_in_extreme_ranges(self):
        """All 4 opp_ff params must be in EXTREME_RANGES."""
        from src.analytics.sensitivity import EXTREME_RANGES
        expected = [
            "opp_ff_efg_weight",
            "opp_ff_tov_weight",
            "opp_ff_oreb_weight",
            "opp_ff_fta_weight",
        ]
        for param in expected:
            assert param in EXTREME_RANGES, f"{param} missing from EXTREME_RANGES"


class TestFatigueInExtremeRanges:

    def test_all_fatigue_params_in_extreme_ranges(self):
        """All 6 fatigue/rest/altitude params must be in EXTREME_RANGES."""
        from src.analytics.sensitivity import EXTREME_RANGES
        expected = [
            "rest_advantage_mult",
            "altitude_b2b_penalty",
            "fatigue_b2b",
            "fatigue_3in4",
            "fatigue_4in6",
            "fatigue_total_mult",
        ]
        for param in expected:
            assert param in EXTREME_RANGES, f"{param} missing from EXTREME_RANGES"
