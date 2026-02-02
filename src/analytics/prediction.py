from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from src.analytics.stats_engine import aggregate_projection
from src.database.db import get_conn


@dataclass
class MatchupPrediction:
    home_team_id: int
    away_team_id: int
    game_date: date
    predicted_spread: float
    predicted_total: float


def _rest_penalty(is_back_to_back: bool) -> float:
    return -1.0 if is_back_to_back else 0.0


def predict_matchup(
    home_team_id: int,
    away_team_id: int,
    home_players: Iterable[int],
    away_players: Iterable[int],
    game_date: Optional[date] = None,
    home_court: float = 3.0,
    is_back_to_back_home: bool = False,
    is_back_to_back_away: bool = False,
) -> MatchupPrediction:
    home_proj = aggregate_projection(home_players, opponent_team_id=away_team_id, is_home=True)
    away_proj = aggregate_projection(away_players, opponent_team_id=home_team_id, is_home=False)

    spread = (home_proj["points"] - away_proj["points"]) + home_court
    spread += _rest_penalty(is_back_to_back_home)
    spread -= _rest_penalty(is_back_to_back_away)

    total = home_proj["points"] + away_proj["points"]
    total += 0.1 * (home_proj["assists"] + away_proj["assists"])  # small pacing boost

    return MatchupPrediction(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=game_date or date.today(),
        predicted_spread=spread,
        predicted_total=total,
    )


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
