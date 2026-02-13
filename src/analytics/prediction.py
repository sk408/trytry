from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Optional

from src.analytics.stats_engine import (
    aggregate_projection,
    get_defensive_rating,
    get_offensive_rating,
    get_opponent_defensive_factor,
    get_home_court_advantage,
    get_pace,
    get_four_factors,
    get_clutch_stats,
    get_hustle_stats,
    detect_fatigue,
    get_team_metrics,
)
from src.analytics.autotune import get_team_tuning
from src.analytics.weight_config import (
    WeightConfig,
    get_weight_config,
    load_team_weights,
)
from src.database.db import get_conn


def _get_team_injury_impact(team_id: int, as_of_date: Optional[date] = None) -> dict:
    """Estimate current injury impact for a team from recent player logs.

    Returns ``injured_count``, ``injury_ppg_lost``, and ``injury_minutes_lost``.
    When *as_of_date* is provided, only player stats before that date are used
    (avoids lookahead bias in backtesting).
    """
    date_clause = ""
    if as_of_date is not None:
        date_clause = "AND ps.game_date < ?"
        params = (str(as_of_date), team_id)
    else:
        params = (team_id,)

    with get_conn() as conn:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) as injured_count,
                COALESCE(SUM(COALESCE(s.ppg, 0)), 0) as ppg_lost,
                COALESCE(SUM(COALESCE(s.mpg, 0)), 0) as minutes_lost
            FROM players p
            LEFT JOIN (
                SELECT ps.player_id,
                       AVG(ps.points) as ppg,
                       AVG(ps.minutes) as mpg
                FROM player_stats ps
                WHERE 1 = 1 {date_clause}
                GROUP BY ps.player_id
            ) s ON s.player_id = p.player_id
            WHERE p.team_id = ? AND p.is_injured = 1
            """,
            params,
        ).fetchone()

    if not row:
        return {"injured_count": 0.0, "injury_ppg_lost": 0.0, "injury_minutes_lost": 0.0}
    return {
        "injured_count": float(row[0] or 0),
        "injury_ppg_lost": float(row[1] or 0.0),
        "injury_minutes_lost": float(row[2] or 0.0),
    }


@dataclass
class MatchupPrediction:
    home_team_id: int
    away_team_id: int
    game_date: date
    predicted_spread: float
    predicted_total: float
    predicted_home_score: float = 0.0
    predicted_away_score: float = 0.0
    # Diagnostic breakdown
    four_factors_adj: float = 0.0
    clutch_adj: float = 0.0
    hustle_adj: float = 0.0
    fatigue_adj: float = 0.0
    espn_blend_applied: bool = False
    # ML uncertainty (0 when ML not used)
    spread_std: float = 0.0
    total_std: float = 0.0
    spread_low: float = 0.0
    spread_high: float = 0.0
    total_low: float = 0.0
    total_high: float = 0.0


@dataclass
class DetailedPrediction:
    """Prediction bundled with the raw feature vector for ML analysis."""
    prediction: MatchupPrediction
    features: dict  # {feature_name: float_value}


def predict_matchup(
    home_team_id: int,
    away_team_id: int,
    home_players: Iterable[int],
    away_players: Iterable[int],
    game_date: Optional[date] = None,
    home_court: Optional[float] = None,
    # ESPN predictor data (if available from gamecast)
    espn_home_win_pct: float = 0.0,
    espn_away_win_pct: float = 0.0,
    _collect_features: dict | None = None,
) -> MatchupPrediction:
    """
    Predict game spread and total using comprehensive multi-factor model.

    Factors considered:
    - Player-level scoring projections with opponent defensive adjustment
    - Dynamic home court advantage (official home/road splits)
    - Home/away splits (30% weight) and head-to-head history (30% weight)
    - Official NBA off/def ratings and pace (from team_metrics)
    - Four Factors of basketball (eFG%, TOV%, OREB%, FT rate)
    - Turnover differential and True Shooting % advantage
    - Rebound differential
    - Clutch performance stats (for projected close games)
    - Hustle stats (deflections, contested shots = defensive intensity)
    - Auto-detected fatigue (B2B, 3-in-4, 4-in-6, rest days)
    - ESPN predictor ensemble blending (when available)
    - Per-team autotune corrections
    - Recent form weighting (last 5 games 60%, older 40%)
    """
    gd = game_date or date.today()

    # Check for per-team weight overrides (use home team's overrides if they
    # exist, otherwise away team's, otherwise global).  This lets the per-team
    # refinement influence predictions for that team's games.
    w = (
        load_team_weights(home_team_id)
        or load_team_weights(away_team_id)
        or get_weight_config()
    )

    # ============ 1. PLAYER-LEVEL PROJECTIONS ============
    home_proj = aggregate_projection(home_players, opponent_team_id=away_team_id, is_home=True)
    away_proj = aggregate_projection(away_players, opponent_team_id=home_team_id, is_home=False)

    # ============ 2. HOME COURT ADVANTAGE ============
    if home_court is None:
        home_court = get_home_court_advantage(home_team_id)

    # ============ 3. OPPONENT DEFENSIVE ADJUSTMENT ============
    away_def_factor_raw = get_opponent_defensive_factor(away_team_id)
    home_def_factor_raw = get_opponent_defensive_factor(home_team_id)
    away_def_factor = 1.0 + (away_def_factor_raw - 1.0) * w.def_factor_dampening
    home_def_factor = 1.0 + (home_def_factor_raw - 1.0) * w.def_factor_dampening

    home_base_pts = home_proj["points"] * away_def_factor
    away_base_pts = away_proj["points"] * home_def_factor

    # ============ 4. AUTOTUNE CORRECTIONS ============
    home_tuning = get_team_tuning(home_team_id)
    away_tuning = get_team_tuning(away_team_id)
    if home_tuning:
        home_base_pts += home_tuning["home_pts_correction"]
    if away_tuning:
        away_base_pts += away_tuning["away_pts_correction"]

    # ============ 5. FATIGUE (auto-detected) ============
    home_fatigue = detect_fatigue(home_team_id, gd)
    away_fatigue = detect_fatigue(away_team_id, gd)
    fatigue_adj = home_fatigue["fatigue_penalty"] - away_fatigue["fatigue_penalty"]

    # ============ SPREAD CALCULATION ============
    spread = (home_base_pts - away_base_pts) + home_court
    spread -= fatigue_adj

    # Turnover differential
    home_to_margin = home_proj.get("turnover_margin", 0)
    away_to_margin = away_proj.get("turnover_margin", 0)
    spread += (home_to_margin - away_to_margin) * w.turnover_margin_mult

    # Rebound differential
    home_reb = home_proj.get("rebounds", 0)
    away_reb = away_proj.get("rebounds", 0)
    spread += (home_reb - away_reb) * w.rebound_diff_mult

    # Off/Def rating matchup
    home_off = get_offensive_rating(home_team_id)
    away_off = get_offensive_rating(away_team_id)
    home_def = get_defensive_rating(home_team_id)
    away_def = get_defensive_rating(away_team_id)
    home_matchup_edge = home_off - away_def
    away_matchup_edge = away_off - home_def
    spread += (home_matchup_edge - away_matchup_edge) * w.rating_matchup_mult

    # Four Factors
    home_ff = get_four_factors(home_team_id)
    away_ff = get_four_factors(away_team_id)
    four_factors_adj = _compute_four_factors_spread(home_ff, away_ff, w)
    spread += four_factors_adj

    # Clutch (projected close games only, but always fetch for feature capture)
    home_clutch = get_clutch_stats(home_team_id)
    away_clutch = get_clutch_stats(away_team_id)
    clutch_adj = 0.0
    if abs(spread) < w.clutch_threshold:
        clutch_adj = _compute_clutch_adjustment(home_clutch, away_clutch, w)
        spread += clutch_adj

    # Hustle
    home_hustle = get_hustle_stats(home_team_id)
    away_hustle = get_hustle_stats(away_team_id)
    hustle_spread_adj = _compute_hustle_spread(home_hustle, away_hustle, w)
    spread += hustle_spread_adj

    # ============ TOTAL CALCULATION ============
    total = home_base_pts + away_base_pts

    # Pace
    home_pace = get_pace(home_team_id)
    away_pace = get_pace(away_team_id)
    expected_pace = (home_pace + away_pace) / 2
    pace_factor = (expected_pace - w.pace_baseline) / w.pace_baseline
    total *= (1 + pace_factor * w.pace_mult)

    # Defensive disruption
    combined_steals = home_proj.get("steals", 0) + away_proj.get("steals", 0)
    combined_blocks = home_proj.get("blocks", 0) + away_proj.get("blocks", 0)
    total -= (max(0, combined_steals - w.steals_threshold) * w.steals_penalty +
              max(0, combined_blocks - w.blocks_threshold) * w.blocks_penalty)

    # Offensive rebound impact
    combined_oreb = home_proj.get("oreb", 0) + away_proj.get("oreb", 0)
    total += (combined_oreb - w.oreb_baseline) * w.oreb_mult

    # Hustle total impact
    hustle_total_adj = _compute_hustle_total(home_hustle, away_hustle, w)
    total += hustle_total_adj

    # Fatigue total impact
    combined_fatigue = home_fatigue["fatigue_penalty"] + away_fatigue["fatigue_penalty"]
    total -= combined_fatigue * w.fatigue_total_mult

    # ============ 7. ESPN PREDICTOR ENSEMBLE ============
    espn_blend_applied = False
    if espn_home_win_pct > 0 and espn_away_win_pct > 0:
        spread, espn_blend_applied = _blend_with_espn(
            spread, espn_home_win_pct, espn_away_win_pct, w
        )

    # ============ 8. ML ENSEMBLE BLENDING ============
    ml_spread_std = 0.0
    ml_total_std = 0.0
    if w.ml_ensemble_weight > 0:
        try:
            from src.analytics.ml_model import predict_ml_with_uncertainty, is_ml_available
            if is_ml_available():
                # Build a PrecomputedGame-like feature dict for ML model
                _ml_feats = _build_ml_features_live(
                    home_proj, away_proj, home_off, away_off, home_def, away_def,
                    home_pace, away_pace, home_court,
                    home_fatigue["fatigue_penalty"], away_fatigue["fatigue_penalty"],
                    home_def_factor_raw, away_def_factor_raw,
                    home_ff, away_ff, home_clutch, away_clutch,
                    home_hustle, away_hustle,
                )
                ml_spread, ml_total, ml_conf, ml_spread_std, ml_total_std = (
                    predict_ml_with_uncertainty(_ml_feats)
                )
                if ml_conf > 0.3:
                    base_weight = 1.0 - w.ml_ensemble_weight
                    ml_wt = w.ml_ensemble_weight
                    # Dampen ML when it disagrees strongly with base
                    if abs(ml_spread - spread) > w.ml_disagree_threshold:
                        ml_wt *= w.ml_disagree_damp
                        base_weight = 1.0 - ml_wt
                    # Down-weight ML when model uncertainty is elevated
                    if ml_spread_std > 0 or ml_total_std > 0:
                        uncertainty_scale = max(
                            0.35,
                            min(1.0, 1.0 / (1.0 + (ml_spread_std / 12.0) + (ml_total_std / 20.0))),
                        )
                        ml_wt *= uncertainty_scale
                        base_weight = 1.0 - ml_wt
                    spread = base_weight * spread + ml_wt * ml_spread
                    total = base_weight * total + ml_wt * ml_total
        except Exception:
            pass  # Graceful fallback: use base model only

    # ============ SANITY CLAMPS ============
    spread = max(-w.spread_clamp, min(w.spread_clamp, spread))
    total = max(w.total_min, min(w.total_max, total))

    # ============ 9. RESIDUAL CALIBRATION ============
    spread = _apply_calibration(spread)

    # ============ DERIVE INDIVIDUAL SCORES ============
    pred_home_score = (total + spread) / 2
    pred_away_score = (total - spread) / 2

    # ============ FEATURE CAPTURE (for ML analysis) ============
    if _collect_features is not None:
        f = _collect_features  # shorthand

        # ── Original model-level features ──
        f["scoring_diff"] = home_base_pts - away_base_pts
        f["home_court_adv"] = home_court
        f["turnover_margin_diff"] = (
            home_proj.get("turnover_margin", 0) - away_proj.get("turnover_margin", 0)
        )
        f["rebound_diff"] = (
            home_proj.get("rebounds", 0) - away_proj.get("rebounds", 0)
        )
        f["rating_matchup_diff"] = home_matchup_edge - away_matchup_edge
        f["four_factors_adj"] = four_factors_adj
        f["clutch_adj"] = clutch_adj
        f["hustle_spread_adj"] = hustle_spread_adj
        f["pace_factor"] = pace_factor
        _s = home_proj.get("steals", 0) + away_proj.get("steals", 0)
        _b = home_proj.get("blocks", 0) + away_proj.get("blocks", 0)
        f["combined_steals_excess"] = max(0, _s - w.steals_threshold)
        f["combined_blocks_excess"] = max(0, _b - w.blocks_threshold)
        f["oreb_excess"] = (
            home_proj.get("oreb", 0) + away_proj.get("oreb", 0) - w.oreb_baseline
        )
        f["fatigue_diff"] = fatigue_adj
        f["home_fatigue_penalty"] = home_fatigue["fatigue_penalty"]
        f["away_fatigue_penalty"] = away_fatigue["fatigue_penalty"]

        # ── RAW TEAM PROJECTIONS (individual, not just diff) ──
        f["home_raw_points"] = home_proj.get("points", 0)
        f["away_raw_points"] = away_proj.get("points", 0)
        f["home_raw_rebounds"] = home_proj.get("rebounds", 0)
        f["away_raw_rebounds"] = away_proj.get("rebounds", 0)
        f["home_raw_assists"] = home_proj.get("assists", 0)
        f["away_raw_assists"] = away_proj.get("assists", 0)
        f["assists_diff"] = (
            home_proj.get("assists", 0) - away_proj.get("assists", 0)
        )
        f["home_raw_steals"] = home_proj.get("steals", 0)
        f["away_raw_steals"] = away_proj.get("steals", 0)
        f["home_raw_blocks"] = home_proj.get("blocks", 0)
        f["away_raw_blocks"] = away_proj.get("blocks", 0)
        f["home_raw_turnovers"] = home_proj.get("turnovers", 0)
        f["away_raw_turnovers"] = away_proj.get("turnovers", 0)
        f["home_raw_oreb"] = home_proj.get("oreb", 0)
        f["away_raw_oreb"] = away_proj.get("oreb", 0)

        # ── SHOOTING EFFICIENCY DIFFERENTIALS ──
        f["home_ts_pct"] = home_proj.get("ts_pct", 0)
        f["away_ts_pct"] = away_proj.get("ts_pct", 0)
        f["ts_pct_diff"] = (
            home_proj.get("ts_pct", 0) - away_proj.get("ts_pct", 0)
        )
        f["home_fg3_rate"] = home_proj.get("fg3_rate", 0)
        f["away_fg3_rate"] = away_proj.get("fg3_rate", 0)
        f["fg3_rate_diff"] = (
            home_proj.get("fg3_rate", 0) - away_proj.get("fg3_rate", 0)
        )
        f["home_ft_rate"] = home_proj.get("ft_rate", 0)
        f["away_ft_rate"] = away_proj.get("ft_rate", 0)
        f["ft_rate_diff"] = (
            home_proj.get("ft_rate", 0) - away_proj.get("ft_rate", 0)
        )

        # ── OFFENSIVE / DEFENSIVE RATINGS (raw) ──
        f["home_off_rating"] = home_off
        f["away_off_rating"] = away_off
        f["home_def_rating"] = home_def
        f["away_def_rating"] = away_def
        f["off_rating_diff"] = home_off - away_off
        f["def_rating_diff"] = home_def - away_def
        f["home_net_rating"] = home_off - home_def
        f["away_net_rating"] = away_off - away_def
        f["net_rating_diff"] = (home_off - home_def) - (away_off - away_def)

        # ── DEFENSIVE FACTORS (raw, before dampening) ──
        f["home_def_factor_raw"] = get_opponent_defensive_factor(home_team_id)
        f["away_def_factor_raw"] = get_opponent_defensive_factor(away_team_id)

        # ── PACE (raw values) ──
        f["home_pace"] = home_pace
        f["away_pace"] = away_pace
        f["pace_diff"] = home_pace - away_pace

        # ── FOUR FACTORS COMPONENTS (individual edges, not just blended) ──
        h_efg = home_ff.get("efg_pct") or 0
        a_efg = away_ff.get("efg_pct") or 0
        h_opp_efg = home_ff.get("opp_efg_pct") or 0
        a_opp_efg = away_ff.get("opp_efg_pct") or 0
        f["ff_efg_edge"] = (h_efg - a_opp_efg) - (a_efg - h_opp_efg)
        h_tov = home_ff.get("tm_tov_pct") or 0
        a_tov = away_ff.get("tm_tov_pct") or 0
        h_opp_tov = home_ff.get("opp_tm_tov_pct") or 0
        a_opp_tov = away_ff.get("opp_tm_tov_pct") or 0
        f["ff_tov_edge"] = (a_tov - h_opp_tov) - (h_tov - a_opp_tov)
        h_oreb_ff = home_ff.get("oreb_pct") or 0
        a_oreb_ff = away_ff.get("oreb_pct") or 0
        h_opp_oreb = home_ff.get("opp_oreb_pct") or 0
        a_opp_oreb = away_ff.get("opp_oreb_pct") or 0
        f["ff_oreb_edge"] = (h_oreb_ff - a_opp_oreb) - (a_oreb_ff - h_opp_oreb)
        h_fta = home_ff.get("fta_rate") or 0
        a_fta = away_ff.get("fta_rate") or 0
        h_opp_fta = home_ff.get("opp_fta_rate") or 0
        a_opp_fta = away_ff.get("opp_fta_rate") or 0
        f["ff_fta_edge"] = (h_fta - a_opp_fta) - (a_fta - h_opp_fta)

        # ── CLUTCH SUB-METRICS ──
        f["home_clutch_net"] = home_clutch.get("clutch_net_rating") or 0
        f["away_clutch_net"] = away_clutch.get("clutch_net_rating") or 0
        f["clutch_net_diff"] = f["home_clutch_net"] - f["away_clutch_net"]

        # ── HUSTLE SUB-METRICS ──
        f["home_deflections"] = home_hustle.get("deflections") or 0
        f["away_deflections"] = away_hustle.get("deflections") or 0
        f["home_contested"] = home_hustle.get("contested_shots") or 0
        f["away_contested"] = away_hustle.get("contested_shots") or 0
        f["deflection_diff"] = f["home_deflections"] - f["away_deflections"]

        # ── INJURY IMPACT ──
        # Full injury impact: count, PPG lost, minutes lost
        try:
            h_inj = _get_team_injury_impact(home_team_id, as_of_date=gd)
            a_inj = _get_team_injury_impact(away_team_id, as_of_date=gd)
            f["home_injured_count"] = h_inj["injured_count"]
            f["away_injured_count"] = a_inj["injured_count"]
            f["home_injury_ppg_lost"] = h_inj["injury_ppg_lost"]
            f["away_injury_ppg_lost"] = a_inj["injury_ppg_lost"]
            f["home_injury_minutes_lost"] = h_inj["injury_minutes_lost"]
            f["away_injury_minutes_lost"] = a_inj["injury_minutes_lost"]
        except Exception:
            f["home_injured_count"] = 0.0
            f["away_injured_count"] = 0.0
            f["home_injury_ppg_lost"] = 0.0
            f["away_injury_ppg_lost"] = 0.0
            f["home_injury_minutes_lost"] = 0.0
            f["away_injury_minutes_lost"] = 0.0
        f["injured_count_diff"] = f.get("home_injured_count", 0) - f.get("away_injured_count", 0)
        f["injury_ppg_lost_diff"] = f.get("home_injury_ppg_lost", 0.0) - f.get("away_injury_ppg_lost", 0.0)
        f["injury_minutes_lost_diff"] = f.get("home_injury_minutes_lost", 0.0) - f.get("away_injury_minutes_lost", 0.0)

    return MatchupPrediction(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=gd,
        predicted_spread=spread,
        predicted_total=total,
        predicted_home_score=pred_home_score,
        predicted_away_score=pred_away_score,
        four_factors_adj=four_factors_adj,
        clutch_adj=clutch_adj,
        hustle_adj=hustle_spread_adj,
        fatigue_adj=fatigue_adj,
        espn_blend_applied=espn_blend_applied,
        spread_std=ml_spread_std,
        total_std=ml_total_std,
        spread_low=spread - ml_spread_std,
        spread_high=spread + ml_spread_std,
        total_low=total - ml_total_std,
        total_high=total + ml_total_std,
    )


def predict_matchup_detailed(
    home_team_id: int,
    away_team_id: int,
    home_players: Iterable[int],
    away_players: Iterable[int],
    game_date: Optional[date] = None,
    home_court: Optional[float] = None,
    espn_home_win_pct: float = 0.0,
    espn_away_win_pct: float = 0.0,
) -> DetailedPrediction:
    """Predict game with raw feature vector for ML analysis.

    Calls the same engine as ``predict_matchup`` but also returns
    a ``features`` dict with the raw (pre-weight) factor values.
    """
    features: dict[str, float] = {}
    pred = predict_matchup(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_players=home_players,
        away_players=away_players,
        game_date=game_date,
        home_court=home_court,
        espn_home_win_pct=espn_home_win_pct,
        espn_away_win_pct=espn_away_win_pct,
        _collect_features=features,
    )
    return DetailedPrediction(prediction=pred, features=features)


# ============ HELPER FUNCTIONS ============


def _build_ml_features_live(
    home_proj: dict, away_proj: dict,
    home_off: float, away_off: float,
    home_def: float, away_def: float,
    home_pace: float, away_pace: float,
    home_court: float,
    home_fatigue: float, away_fatigue: float,
    home_def_factor_raw: float, away_def_factor_raw: float,
    home_ff: dict, away_ff: dict,
    home_clutch: dict, away_clutch: dict,
    home_hustle: dict, away_hustle: dict,
) -> dict:
    """Build a feature dict compatible with ml_model.extract_features
    from live prediction data.  This mirrors the feature extraction in
    ml_model.py so the same trained model can be used at inference time."""
    _s = lambda v, d=0.0: float(v) if v is not None else d
    f: dict[str, float] = {}
    hp, ap = home_proj, away_proj

    for key in ("points", "rebounds", "assists", "steals", "blocks",
                "turnovers", "oreb", "dreb"):
        f[f"home_{key}"] = _s(hp.get(key))
        f[f"away_{key}"] = _s(ap.get(key))
        f[f"diff_{key}"] = f[f"home_{key}"] - f[f"away_{key}"]

    for key in ("ts_pct", "fg3_rate", "ft_rate"):
        f[f"home_{key}"] = _s(hp.get(key))
        f[f"away_{key}"] = _s(ap.get(key))
        f[f"diff_{key}"] = f[f"home_{key}"] - f[f"away_{key}"]

    f["home_to_margin"] = _s(hp.get("turnover_margin"))
    f["away_to_margin"] = _s(ap.get("turnover_margin"))
    f["diff_to_margin"] = f["home_to_margin"] - f["away_to_margin"]

    f["home_off_rating"] = _s(home_off)
    f["away_off_rating"] = _s(away_off)
    f["home_def_rating"] = _s(home_def)
    f["away_def_rating"] = _s(away_def)
    f["home_net_rating"] = f["home_off_rating"] - f["home_def_rating"]
    f["away_net_rating"] = f["away_off_rating"] - f["away_def_rating"]
    f["diff_net_rating"] = f["home_net_rating"] - f["away_net_rating"]
    f["home_matchup_edge"] = f["home_off_rating"] - f["away_def_rating"]
    f["away_matchup_edge"] = f["away_off_rating"] - f["home_def_rating"]
    f["diff_matchup_edge"] = f["home_matchup_edge"] - f["away_matchup_edge"]

    f["home_def_factor_raw"] = _s(home_def_factor_raw, 1.0)
    f["away_def_factor_raw"] = _s(away_def_factor_raw, 1.0)

    f["home_pace"] = _s(home_pace, 98.0)
    f["away_pace"] = _s(away_pace, 98.0)
    f["avg_pace"] = (f["home_pace"] + f["away_pace"]) / 2
    f["diff_pace"] = f["home_pace"] - f["away_pace"]

    f["home_court"] = _s(home_court, 3.0)

    f["home_fatigue"] = _s(home_fatigue)
    f["away_fatigue"] = _s(away_fatigue)
    f["diff_fatigue"] = f["home_fatigue"] - f["away_fatigue"]
    f["combined_fatigue"] = f["home_fatigue"] + f["away_fatigue"]

    hff = home_ff or {}
    aff = away_ff or {}
    h_efg = _s(hff.get("efg_pct"))
    a_efg = _s(aff.get("efg_pct"))
    h_opp_efg = _s(hff.get("opp_efg_pct"))
    a_opp_efg = _s(aff.get("opp_efg_pct"))
    f["ff_efg_edge"] = (h_efg - a_opp_efg) - (a_efg - h_opp_efg)
    h_tov = _s(hff.get("tm_tov_pct"))
    a_tov = _s(aff.get("tm_tov_pct"))
    h_opp_tov = _s(hff.get("opp_tm_tov_pct"))
    a_opp_tov = _s(aff.get("opp_tm_tov_pct"))
    f["ff_tov_edge"] = (a_tov - h_opp_tov) - (h_tov - a_opp_tov)
    h_oreb_ff = _s(hff.get("oreb_pct"))
    a_oreb_ff = _s(aff.get("oreb_pct"))
    h_opp_oreb = _s(hff.get("opp_oreb_pct"))
    a_opp_oreb = _s(aff.get("opp_oreb_pct"))
    f["ff_oreb_edge"] = (h_oreb_ff - a_opp_oreb) - (a_oreb_ff - h_opp_oreb)
    h_fta = _s(hff.get("fta_rate"))
    a_fta = _s(aff.get("fta_rate"))
    h_opp_fta = _s(hff.get("opp_fta_rate"))
    a_opp_fta = _s(aff.get("opp_fta_rate"))
    f["ff_fta_edge"] = (h_fta - a_opp_fta) - (a_fta - h_opp_fta)

    hc = home_clutch or {}
    ac = away_clutch or {}
    f["home_clutch_net"] = _s(hc.get("clutch_net_rating"))
    f["away_clutch_net"] = _s(ac.get("clutch_net_rating"))
    f["diff_clutch_net"] = f["home_clutch_net"] - f["away_clutch_net"]
    f["home_clutch_efg"] = _s(hc.get("clutch_efg_pct"))
    f["away_clutch_efg"] = _s(ac.get("clutch_efg_pct"))

    hh = home_hustle or {}
    ah = away_hustle or {}
    f["home_deflections"] = _s(hh.get("deflections"))
    f["away_deflections"] = _s(ah.get("deflections"))
    f["diff_deflections"] = f["home_deflections"] - f["away_deflections"]
    f["home_contested"] = _s(hh.get("contested_shots"))
    f["away_contested"] = _s(ah.get("contested_shots"))
    f["home_loose_balls"] = _s(hh.get("loose_balls_recovered"))
    f["away_loose_balls"] = _s(ah.get("loose_balls_recovered"))

    return f


# Cache for residual calibration (loaded once, refreshed on demand)
_calibration_cache: list | None = None


def _load_calibration_bins() -> list:
    """Load residual calibration from DB (cached)."""
    global _calibration_cache
    if _calibration_cache is not None:
        return _calibration_cache
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT bin_low, bin_high, avg_residual, sample_count "
                "FROM residual_calibration ORDER BY bin_low"
            ).fetchall()
        _calibration_cache = [
            {"bin_low": r[0], "bin_high": r[1], "avg_residual": r[2], "sample_count": r[3]}
            for r in rows
        ]
    except Exception:
        _calibration_cache = []
    return _calibration_cache


def reload_calibration_cache() -> None:
    """Force a reload of calibration bins (call after rebuilding)."""
    global _calibration_cache
    _calibration_cache = None


def _apply_calibration(spread: float) -> float:
    """Adjust spread using saved residual calibration bins."""
    bins = _load_calibration_bins()
    for cal in bins:
        if cal["bin_low"] <= spread < cal["bin_high"] and cal["sample_count"] >= 5:
            return spread - cal["avg_residual"]
    return spread


def _compute_four_factors_spread(
    home_ff: dict, away_ff: dict, w: WeightConfig
) -> float:
    """Compare the Four Factors between two teams using configurable weights."""
    edges = []

    h_efg = home_ff.get("efg_pct")
    a_opp_efg = away_ff.get("opp_efg_pct")
    a_efg = away_ff.get("efg_pct")
    h_opp_efg = home_ff.get("opp_efg_pct")
    if all(v is not None for v in [h_efg, a_opp_efg, a_efg, h_opp_efg]):
        home_efg_edge = (h_efg - a_opp_efg) - (a_efg - h_opp_efg)
        edges.append(home_efg_edge * w.ff_efg_weight)

    h_tov = home_ff.get("tm_tov_pct")
    a_opp_tov = away_ff.get("opp_tm_tov_pct")
    a_tov = away_ff.get("tm_tov_pct")
    h_opp_tov = home_ff.get("opp_tm_tov_pct")
    if all(v is not None for v in [h_tov, a_opp_tov, a_tov, h_opp_tov]):
        home_tov_edge = (a_tov - h_opp_tov) - (h_tov - a_opp_tov)
        edges.append(home_tov_edge * w.ff_tov_weight)

    h_oreb = home_ff.get("oreb_pct")
    a_opp_oreb = away_ff.get("opp_oreb_pct")
    a_oreb = away_ff.get("oreb_pct")
    h_opp_oreb = home_ff.get("opp_oreb_pct")
    if all(v is not None for v in [h_oreb, a_opp_oreb, a_oreb, h_opp_oreb]):
        home_oreb_edge = (h_oreb - a_opp_oreb) - (a_oreb - h_opp_oreb)
        edges.append(home_oreb_edge * w.ff_oreb_weight)

    h_fta = home_ff.get("fta_rate")
    a_opp_fta = away_ff.get("opp_fta_rate")
    a_fta = away_ff.get("fta_rate")
    h_opp_fta = home_ff.get("opp_fta_rate")
    if all(v is not None for v in [h_fta, a_opp_fta, a_fta, h_opp_fta]):
        home_fta_edge = (h_fta - a_opp_fta) - (a_fta - h_opp_fta)
        edges.append(home_fta_edge * w.ff_fta_weight)

    if not edges:
        return 0.0
    return sum(edges) * w.four_factors_scale


def _compute_clutch_adjustment(
    home_clutch: dict, away_clutch: dict, w: WeightConfig
) -> float:
    """Clutch performance adjustment using configurable weights."""
    home_net = home_clutch.get("clutch_net_rating")
    away_net = away_clutch.get("clutch_net_rating")
    if home_net is None or away_net is None:
        return 0.0
    clutch_diff = (home_net - away_net) * w.clutch_scale
    return max(-w.clutch_cap, min(w.clutch_cap, clutch_diff))


def _compute_hustle_spread(
    home_hustle: dict, away_hustle: dict, w: WeightConfig
) -> float:
    """Hustle stats spread impact using configurable weights."""
    h_defl = home_hustle.get("deflections")
    a_defl = away_hustle.get("deflections")
    h_cont = home_hustle.get("contested_shots")
    a_cont = away_hustle.get("contested_shots")

    if h_defl is None or a_defl is None:
        return 0.0

    h_effort = (h_defl or 0) + (h_cont or 0) * w.hustle_contested_wt
    a_effort = (a_defl or 0) + (a_cont or 0) * w.hustle_contested_wt
    return (h_effort - a_effort) * w.hustle_effort_mult


def _compute_hustle_total(
    home_hustle: dict, away_hustle: dict, w: WeightConfig
) -> float:
    """Hustle impact on game total using configurable weights."""
    h_defl = home_hustle.get("deflections") or 0
    a_defl = away_hustle.get("deflections") or 0
    combined_defl = h_defl + a_defl

    if combined_defl > w.hustle_defl_baseline:
        excess = combined_defl - w.hustle_defl_baseline
        return -excess * w.hustle_defl_penalty
    return 0.0


def _blend_with_espn(
    model_spread: float,
    espn_home_pct: float,
    espn_away_pct: float,
    w: WeightConfig,
) -> tuple[float, bool]:
    """Blend model spread with ESPN predictor using configurable weights."""
    if espn_home_pct <= 0 and espn_away_pct <= 0:
        return model_spread, False

    espn_edge = espn_home_pct - 50.0
    espn_implied_spread = espn_edge * w.espn_spread_scale

    blended = model_spread * w.espn_model_weight + espn_implied_spread * w.espn_weight

    if (model_spread > 0.5 and espn_implied_spread < -0.5) or \
       (model_spread < -0.5 and espn_implied_spread > 0.5):
        blended *= w.espn_disagree_damp

    return blended, True


# ============ PRECOMPUTED FAST-PATH FOR OPTIMISER ============

@dataclass
class PrecomputedGame:
    """All raw inputs for a single game, extracted once from the DB.
    The optimiser can re-derive spread/total with different weights
    without touching the database."""
    # Identifiers
    game_date: date
    home_team_id: int
    away_team_id: int
    actual_home_score: float
    actual_away_score: float
    # Player-level projections (weight-independent aggregates)
    home_proj: dict  # from aggregate_projection
    away_proj: dict
    # Team-level raw data (weight-independent)
    home_court: float
    away_def_factor_raw: float  # before dampening
    home_def_factor_raw: float
    home_tuning_home_corr: float  # 0.0 if no tuning
    away_tuning_away_corr: float
    home_fatigue_penalty: float
    away_fatigue_penalty: float
    home_off: float
    away_off: float
    home_def: float
    away_def: float
    home_pace: float
    away_pace: float
    home_ff: dict  # four factors
    away_ff: dict
    home_clutch: dict
    away_clutch: dict
    home_hustle: dict
    away_hustle: dict
    # Injury context (team-level impact at game date)
    home_injured_count: float = 0.0
    away_injured_count: float = 0.0
    home_injury_ppg_lost: float = 0.0
    away_injury_ppg_lost: float = 0.0
    home_injury_minutes_lost: float = 0.0
    away_injury_minutes_lost: float = 0.0


def precompute_game_data(
    progress_cb: Optional[Callable] = None,
    use_cache: bool = True,
) -> list["PrecomputedGame"]:
    """Extract all raw game data from the DB once.

    Returns a list of ``PrecomputedGame`` that the optimiser can
    re-evaluate with different weights purely in memory.

    When *use_cache* is True (default), checks the in-memory store and
    disk pickle before hitting the database.
    """
    from src.analytics.backtester import (
        get_actual_game_results,
        get_team_profile,
        _get_roster_for_game,
    )
    from src.analytics.autotune import get_team_tuning
    from src.analytics.memory_store import get_memory_store
    from src.analytics.pipeline_cache import (
        load_pipeline_state,
        load_precomputed_games,
        save_precomputed_games,
        has_new_games,
    )

    progress = progress_cb or (lambda _: None)

    # ── Try in-memory cache first ──
    store = get_memory_store()
    if use_cache and store.precomputed_games:
        progress(f"Using {len(store.precomputed_games)} precomputed games from memory")
        return store.precomputed_games

    # ── Try disk pickle cache ──
    if use_cache:
        state = load_pipeline_state()
        if state.precomputed_games_hash and not has_new_games(state):
            cached = load_precomputed_games(state.precomputed_games_hash)
            if cached:
                progress(f"Loaded {len(cached)} precomputed games from disk cache")
                store.precomputed_games = cached
                return cached

    progress("Loading game results for precomputation...")

    games_df = get_actual_game_results()
    if games_df.empty:
        return []

    games_df = games_df.sort_values("game_date")
    precomputed: list[PrecomputedGame] = []
    total = len(games_df)

    for idx, (_, game) in enumerate(games_df.iterrows()):
        if idx % 40 == 0:
            progress(f"Precomputing game {idx + 1}/{total}...")

        gd_raw = game["game_date"]
        if isinstance(gd_raw, str):
            gd = date.fromisoformat(gd_raw[:10])
        else:
            gd = gd_raw

        home_id = int(game["home_team_id"])
        away_id = int(game["away_team_id"])

        # Skip if not enough history
        home_profile = get_team_profile(home_id, gd)
        away_profile = get_team_profile(away_id, gd)
        if home_profile["games"] < 5 or away_profile["games"] < 5:
            continue

        # Rosters
        home_pids = _get_roster_for_game(home_id, gd)
        away_pids = _get_roster_for_game(away_id, gd)
        if not home_pids or not away_pids:
            continue

        # Player projections
        home_proj = aggregate_projection(home_pids, opponent_team_id=away_id, is_home=True)
        away_proj = aggregate_projection(away_pids, opponent_team_id=home_id, is_home=False)

        # Team metrics
        hca = get_home_court_advantage(home_id)
        adf = get_opponent_defensive_factor(away_id)
        hdf = get_opponent_defensive_factor(home_id)
        home_fatigue_info = detect_fatigue(home_id, gd)
        away_fatigue_info = detect_fatigue(away_id, gd)
        home_injury = _get_team_injury_impact(home_id, as_of_date=gd)
        away_injury = _get_team_injury_impact(away_id, as_of_date=gd)

        # Tuning
        ht = get_team_tuning(home_id)
        at = get_team_tuning(away_id)

        precomputed.append(PrecomputedGame(
            game_date=gd,
            home_team_id=home_id,
            away_team_id=away_id,
            actual_home_score=float(game["home_score"]),
            actual_away_score=float(game["away_score"]),
            home_proj=dict(home_proj),
            away_proj=dict(away_proj),
            home_court=hca,
            away_def_factor_raw=adf,
            home_def_factor_raw=hdf,
            home_tuning_home_corr=ht["home_pts_correction"] if ht else 0.0,
            away_tuning_away_corr=at["away_pts_correction"] if at else 0.0,
            home_fatigue_penalty=home_fatigue_info["fatigue_penalty"],
            away_fatigue_penalty=away_fatigue_info["fatigue_penalty"],
            home_off=get_offensive_rating(home_id),
            away_off=get_offensive_rating(away_id),
            home_def=get_defensive_rating(home_id),
            away_def=get_defensive_rating(away_id),
            home_pace=get_pace(home_id),
            away_pace=get_pace(away_id),
            home_ff=get_four_factors(home_id),
            away_ff=get_four_factors(away_id),
            home_clutch=get_clutch_stats(home_id),
            away_clutch=get_clutch_stats(away_id),
            home_hustle=get_hustle_stats(home_id),
            away_hustle=get_hustle_stats(away_id),
            home_injured_count=home_injury["injured_count"],
            away_injured_count=away_injury["injured_count"],
            home_injury_ppg_lost=home_injury["injury_ppg_lost"],
            away_injury_ppg_lost=away_injury["injury_ppg_lost"],
            home_injury_minutes_lost=home_injury["injury_minutes_lost"],
            away_injury_minutes_lost=away_injury["injury_minutes_lost"],
        ))

    progress(f"Precomputed {len(precomputed)} games (no more DB I/O needed)")

    # ── Cache the results ──
    store.precomputed_games = precomputed
    try:
        h = save_precomputed_games(precomputed)
        state = load_pipeline_state()
        state.precomputed_games_hash = h
        from src.analytics.pipeline_cache import save_pipeline_state
        save_pipeline_state(state)
        progress("Precomputed games cached to memory + disk")
    except Exception:
        pass  # Non-fatal

    return precomputed


def predict_from_precomputed(g: "PrecomputedGame", w: "WeightConfig") -> tuple[float, float]:
    """Recompute spread and total from precomputed data + weights.

    Pure arithmetic — zero DB access.  Returns ``(spread, total)``.
    """
    # Defensive factor dampening
    adf = 1.0 + (g.away_def_factor_raw - 1.0) * w.def_factor_dampening
    hdf = 1.0 + (g.home_def_factor_raw - 1.0) * w.def_factor_dampening

    home_base_pts = g.home_proj["points"] * adf
    away_base_pts = g.away_proj["points"] * hdf

    # Autotune corrections
    home_base_pts += g.home_tuning_home_corr
    away_base_pts += g.away_tuning_away_corr

    # Fatigue
    fatigue_adj = g.home_fatigue_penalty - g.away_fatigue_penalty

    # Spread
    spread = (home_base_pts - away_base_pts) + g.home_court - fatigue_adj

    # Turnover differential
    h_to = g.home_proj.get("turnover_margin", 0)
    a_to = g.away_proj.get("turnover_margin", 0)
    spread += (h_to - a_to) * w.turnover_margin_mult

    # Rebound diff
    h_reb = g.home_proj.get("rebounds", 0)
    a_reb = g.away_proj.get("rebounds", 0)
    spread += (h_reb - a_reb) * w.rebound_diff_mult

    # Off/Def rating matchup
    home_edge = g.home_off - g.away_def
    away_edge = g.away_off - g.home_def
    spread += (home_edge - away_edge) * w.rating_matchup_mult

    # Four Factors
    ff_adj = _compute_four_factors_spread(g.home_ff, g.away_ff, w)
    spread += ff_adj

    # Clutch
    clutch_adj = 0.0
    if abs(spread) < w.clutch_threshold:
        clutch_adj = _compute_clutch_adjustment(g.home_clutch, g.away_clutch, w)
        spread += clutch_adj

    # Hustle
    hustle_adj = _compute_hustle_spread(g.home_hustle, g.away_hustle, w)
    spread += hustle_adj

    # Clamp spread
    spread = max(-w.spread_clamp, min(w.spread_clamp, spread))

    # Calibration (skip during optimisation — it's a post-hoc correction)
    # spread = _apply_calibration(spread)

    # ── Total ──
    total = home_base_pts + away_base_pts

    # Pace
    exp_pace = (g.home_pace + g.away_pace) / 2
    pace_factor = (exp_pace - w.pace_baseline) / w.pace_baseline
    total *= (1 + pace_factor * w.pace_mult)

    # Defensive disruption
    combined_steals = g.home_proj.get("steals", 0) + g.away_proj.get("steals", 0)
    combined_blocks = g.home_proj.get("blocks", 0) + g.away_proj.get("blocks", 0)
    total -= (max(0, combined_steals - w.steals_threshold) * w.steals_penalty +
              max(0, combined_blocks - w.blocks_threshold) * w.blocks_penalty)

    # OREB
    combined_oreb = g.home_proj.get("oreb", 0) + g.away_proj.get("oreb", 0)
    total += (combined_oreb - w.oreb_baseline) * w.oreb_mult

    # Hustle total
    hustle_total = _compute_hustle_total(g.home_hustle, g.away_hustle, w)
    total += hustle_total

    # Fatigue total
    combined_fatigue = g.home_fatigue_penalty + g.away_fatigue_penalty
    total -= combined_fatigue * w.fatigue_total_mult

    # ML Ensemble blending (in fast path)
    if w.ml_ensemble_weight > 0:
        try:
            from src.analytics.ml_model import (
                predict_ml_from_precomputed_with_uncertainty,
                is_ml_available,
            )
            if is_ml_available():
                ml_spread, ml_total, ml_conf, ml_s_std, ml_t_std = (
                    predict_ml_from_precomputed_with_uncertainty(g)
                )
                if ml_conf > 0.3:
                    base_weight = 1.0 - w.ml_ensemble_weight
                    ml_wt = w.ml_ensemble_weight
                    if abs(ml_spread - spread) > w.ml_disagree_threshold:
                        ml_wt *= w.ml_disagree_damp
                        base_weight = 1.0 - ml_wt
                    # Down-weight ML when uncertainty is elevated
                    if ml_s_std > 0 or ml_t_std > 0:
                        u_scale = max(
                            0.35,
                            min(1.0, 1.0 / (1.0 + (ml_s_std / 12.0) + (ml_t_std / 20.0))),
                        )
                        ml_wt *= u_scale
                        base_weight = 1.0 - ml_wt
                    spread = base_weight * spread + ml_wt * ml_spread
                    total = base_weight * total + ml_wt * ml_total
        except Exception:
            pass

    # Clamp total
    total = max(w.total_min, min(w.total_max, total))

    return spread, total


def save_prediction(prediction: MatchupPrediction) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (home_team_id, away_team_id, game_date, predicted_spread, predicted_total)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                prediction.home_team_id,
                prediction.away_team_id,
                prediction.game_date,
                prediction.predicted_spread,
                prediction.predicted_total,
            ),
        )
        conn.commit()
