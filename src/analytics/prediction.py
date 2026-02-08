from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

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
from src.database.db import get_conn


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

    # ============ 1. PLAYER-LEVEL PROJECTIONS ============
    home_proj = aggregate_projection(home_players, opponent_team_id=away_team_id, is_home=True)
    away_proj = aggregate_projection(away_players, opponent_team_id=home_team_id, is_home=False)

    # ============ 2. HOME COURT ADVANTAGE ============
    if home_court is None:
        home_court = get_home_court_advantage(home_team_id)

    # ============ 3. OPPONENT DEFENSIVE ADJUSTMENT ============
    away_def_factor = get_opponent_defensive_factor(away_team_id)
    home_def_factor = get_opponent_defensive_factor(home_team_id)
    # Dampen: 50% toward 1.0 to avoid overcorrection
    away_def_factor = 1.0 + (away_def_factor - 1.0) * 0.5
    home_def_factor = 1.0 + (home_def_factor - 1.0) * 0.5

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
    #
    # Player PPG is the base, but matchup-specific DIFFERENTIALS capture
    # nuances like rebound advantages, turnover forcing, and shooting style
    # mismatches that raw PPG can't express.  Four Factors covers OREB%,
    # TOV%, eFG%, and FT rate at a team level; the box-score differentials
    # below add the player-aggregated perspective at reduced weights to
    # avoid double-counting.

    # 5a. Base spread from defense-adjusted scoring differential
    spread = (home_base_pts - away_base_pts) + home_court

    # 5b. Fatigue penalty
    spread -= fatigue_adj

    # 5c. Turnover differential – net steals vs turnovers; partially covered
    #     by Four Factors TOV% so weight is reduced (~0.4 pts per net TO).
    home_to_margin = home_proj.get("turnover_margin", 0)
    away_to_margin = away_proj.get("turnover_margin", 0)
    spread += (home_to_margin - away_to_margin) * 0.4

    # 5d. Rebound differential – partially covered by Four Factors OREB%
    #     so weight is reduced (~0.08 pts per rebound).
    home_reb = home_proj.get("rebounds", 0)
    away_reb = away_proj.get("rebounds", 0)
    spread += (home_reb - away_reb) * 0.08

    # 5e. Official Off/Def rating matchup comparison
    home_off = get_offensive_rating(home_team_id)
    away_off = get_offensive_rating(away_team_id)
    home_def = get_defensive_rating(home_team_id)
    away_def = get_defensive_rating(away_team_id)
    home_matchup_edge = home_off - away_def
    away_matchup_edge = away_off - home_def
    rating_spread_adj = (home_matchup_edge - away_matchup_edge) * 0.08
    spread += rating_spread_adj

    # 5f. FOUR FACTORS COMPARISON (team-level quality signal – covers eFG%,
    #     TOV%, OREB%, FT rate and their opponent-forcing equivalents)
    home_ff = get_four_factors(home_team_id)
    away_ff = get_four_factors(away_team_id)
    four_factors_adj = _compute_four_factors_spread(home_ff, away_ff)
    spread += four_factors_adj

    # 5g. CLUTCH PERFORMANCE (for projected close games)
    clutch_adj = 0.0
    if abs(spread) < 6.0:
        home_clutch = get_clutch_stats(home_team_id)
        away_clutch = get_clutch_stats(away_team_id)
        clutch_adj = _compute_clutch_adjustment(home_clutch, away_clutch)
        spread += clutch_adj

    # 5h. HUSTLE STATS (defensive effort – not in box scores or Four Factors)
    home_hustle = get_hustle_stats(home_team_id)
    away_hustle = get_hustle_stats(away_team_id)
    hustle_spread_adj = _compute_hustle_spread(home_hustle, away_hustle)
    spread += hustle_spread_adj

    # ============ TOTAL CALCULATION ============
    #
    # Base total = summed player PPG (already embeds each team's individual
    # pace).  Adjustments below capture MATCHUP-SPECIFIC interactions that
    # differ from each team's season averages.

    # 6a. Base total from defense-adjusted scoring
    total = home_base_pts + away_base_pts

    # 6b. Pace matchup – player PPG embeds their own team's pace, but the
    #     game pace is the average of both teams.  A fast team vs a slow team
    #     lands in between.  Apply a moderate correction.
    home_pace = get_pace(home_team_id)
    away_pace = get_pace(away_team_id)
    expected_pace = (home_pace + away_pace) / 2
    pace_factor = (expected_pace - 98.0) / 98.0
    total *= (1 + pace_factor * 0.20)

    # 6c. Defensive disruption – combined steals + blocks suppress scoring
    #     beyond what individual PPG captures (forces bad shots / turnovers
    #     in this specific matchup).  Not covered by Four Factors.
    combined_steals = home_proj.get("steals", 0) + away_proj.get("steals", 0)
    combined_blocks = home_proj.get("blocks", 0) + away_proj.get("blocks", 0)
    total -= max(0, combined_steals - 14.0) * 0.15 + max(0, combined_blocks - 10.0) * 0.12

    # 6d. Offensive rebound impact – second-chance possessions raise scoring.
    #     Partially in Four Factors OREB% so weight is halved.
    combined_oreb = home_proj.get("oreb", 0) + away_proj.get("oreb", 0)
    total += (combined_oreb - 20.0) * 0.2

    # 6e. Hustle impact on total (high hustle = tighter defense = lower scoring)
    hustle_total_adj = _compute_hustle_total(home_hustle, away_hustle)
    total += hustle_total_adj

    # 6f. Fatigue impact on total
    combined_fatigue = home_fatigue["fatigue_penalty"] + away_fatigue["fatigue_penalty"]
    total -= combined_fatigue * 0.3

    # ============ 7. ESPN PREDICTOR ENSEMBLE ============
    espn_blend_applied = False
    if espn_home_win_pct > 0 and espn_away_win_pct > 0:
        spread, espn_blend_applied = _blend_with_espn(
            spread, espn_home_win_pct, espn_away_win_pct
        )

    # ============ SANITY CLAMPS ============
    # NBA scores are almost always 85-140 per team; totals 185-260.
    spread = max(-25.0, min(25.0, spread))
    total = max(185.0, min(260.0, total))

    # ============ DERIVE INDIVIDUAL SCORES ============
    pred_home_score = (total + spread) / 2
    pred_away_score = (total - spread) / 2

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
    )


# ============ HELPER FUNCTIONS ============


def _compute_four_factors_spread(
    home_ff: dict, away_ff: dict
) -> float:
    """
    Compare the Four Factors between two teams.
    Weights: eFG% 40%, TOV% 25%, OREB% 20%, FT rate 15%
    (based on Oliver's research on factors that determine winning)
    """
    # For each factor, compute the edge:
    #   team's offensive factor vs opponent's defensive forcing
    #   home_ff["efg_pct"] vs away_ff["opp_efg_pct"] (what away D forces)
    edges = []

    # eFG% edge: home shooting vs away defensive forcing
    h_efg = home_ff.get("efg_pct")
    a_opp_efg = away_ff.get("opp_efg_pct")
    a_efg = away_ff.get("efg_pct")
    h_opp_efg = home_ff.get("opp_efg_pct")
    if all(v is not None for v in [h_efg, a_opp_efg, a_efg, h_opp_efg]):
        # Positive = home advantage
        home_efg_edge = (h_efg - a_opp_efg) - (a_efg - h_opp_efg)
        edges.append(home_efg_edge * 0.40)

    # TOV% edge (lower is better for offense)
    h_tov = home_ff.get("tm_tov_pct")
    a_opp_tov = away_ff.get("opp_tm_tov_pct")
    a_tov = away_ff.get("tm_tov_pct")
    h_opp_tov = home_ff.get("opp_tm_tov_pct")
    if all(v is not None for v in [h_tov, a_opp_tov, a_tov, h_opp_tov]):
        # Negative = home advantage (home has lower TOV%)
        home_tov_edge = (a_tov - h_opp_tov) - (h_tov - a_opp_tov)
        edges.append(home_tov_edge * 0.25)

    # OREB% edge
    h_oreb = home_ff.get("oreb_pct")
    a_opp_oreb = away_ff.get("opp_oreb_pct")
    a_oreb = away_ff.get("oreb_pct")
    h_opp_oreb = home_ff.get("opp_oreb_pct")
    if all(v is not None for v in [h_oreb, a_opp_oreb, a_oreb, h_opp_oreb]):
        home_oreb_edge = (h_oreb - a_opp_oreb) - (a_oreb - h_opp_oreb)
        edges.append(home_oreb_edge * 0.20)

    # FT rate edge
    h_fta = home_ff.get("fta_rate")
    a_opp_fta = away_ff.get("opp_fta_rate")
    a_fta = away_ff.get("fta_rate")
    h_opp_fta = home_ff.get("opp_fta_rate")
    if all(v is not None for v in [h_fta, a_opp_fta, a_fta, h_opp_fta]):
        home_fta_edge = (h_fta - a_opp_fta) - (a_fta - h_opp_fta)
        edges.append(home_fta_edge * 0.15)

    if not edges:
        return 0.0

    # Scale: 1% combined Four Factors edge ≈ 0.3 points
    return sum(edges) * 0.3


def _compute_clutch_adjustment(
    home_clutch: dict, away_clutch: dict
) -> float:
    """
    For projected close games, adjust spread based on clutch performance.
    Teams that perform well in crunch time outperform in tight games.
    """
    home_net = home_clutch.get("clutch_net_rating")
    away_net = away_clutch.get("clutch_net_rating")
    if home_net is None or away_net is None:
        return 0.0

    # Net rating differential in clutch situations, dampened
    clutch_diff = (home_net - away_net) * 0.05  # ~0.05 pts per 1 net rating point
    # Cap the clutch adjustment to +-2.0 pts
    return max(-2.0, min(2.0, clutch_diff))


def _compute_hustle_spread(
    home_hustle: dict, away_hustle: dict
) -> float:
    """
    Hustle stats impact on spread: teams with more deflections, contested shots,
    and charges drawn tend to have stronger defensive effort.
    """
    h_defl = home_hustle.get("deflections")
    a_defl = away_hustle.get("deflections")
    h_cont = home_hustle.get("contested_shots")
    a_cont = away_hustle.get("contested_shots")

    if h_defl is None or a_defl is None:
        return 0.0

    # Combine deflections and contested shots into an "effort score"
    h_effort = (h_defl or 0) + (h_cont or 0) * 0.3
    a_effort = (a_defl or 0) + (a_cont or 0) * 0.3
    effort_diff = h_effort - a_effort

    # ~0.02 pts per unit of effort differential
    return effort_diff * 0.02


def _compute_hustle_total(
    home_hustle: dict, away_hustle: dict
) -> float:
    """
    High combined hustle = tighter defense overall = slightly lower totals.
    """
    h_defl = home_hustle.get("deflections") or 0
    a_defl = away_hustle.get("deflections") or 0
    combined_defl = h_defl + a_defl

    # League avg ~30 combined deflections per game (15 per team)
    if combined_defl > 30:
        excess = combined_defl - 30
        return -excess * 0.1  # ~0.1 pts reduction per excess deflection
    return 0.0


def _blend_with_espn(
    model_spread: float,
    espn_home_pct: float,
    espn_away_pct: float,
) -> tuple[float, bool]:
    """
    Blend our model's spread with ESPN's predictor.
    ESPN gives win probabilities; convert to implied spread.
    """
    if espn_home_pct <= 0 and espn_away_pct <= 0:
        return model_spread, False

    # Convert ESPN win probability to implied spread
    # Rough conversion: each 5% win probability ≈ 1.5 points of spread
    # Home at 60% → +3.0 pts; Home at 40% → -3.0 pts
    espn_edge = espn_home_pct - 50.0  # positive = home favored
    espn_implied_spread = espn_edge * 0.3  # scale factor

    # Blend: 80% our model, 20% ESPN
    blended = model_spread * 0.80 + espn_implied_spread * 0.20

    # Disagreement dampening: if our model and ESPN disagree on direction,
    # reduce confidence
    if (model_spread > 0.5 and espn_implied_spread < -0.5) or \
       (model_spread < -0.5 and espn_implied_spread > 0.5):
        blended *= 0.85  # 15% dampening for disagreement

    return blended, True


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
