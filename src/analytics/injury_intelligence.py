"""
Injury Intelligence System
===========================

Tracks injury status history over time and computes:

1. **Play-through rates** — given a status level (Out / Doubtful / Questionable /
   Probable / Day-to-Day), what fraction of the time did the player actually
   play in their next game?

2. **Per-player tendencies** — some players are warriors (Questionable but almost
   always play), others are cautious.

3. **Injury-keyword modifiers** — rest/load-management games are very different
   from ankle sprains.

4. **Weighted play probability** — combines league-wide rate, player tendency,
   and injury-keyword modifier into a single probability that the player suits
   up.  This probability is used by the prediction engine instead of the
   binary "in/out" flag.

5. **Backfill** — after games are played, cross-reference game logs with the
   status log to fill in ``did_play``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.database.db import get_conn, connect
from src.database import migrations


# ====================================================================
#  Default play-through rates (used until enough data is collected)
# ====================================================================

DEFAULT_PLAY_RATES: Dict[str, float] = {
    "Out":          0.00,
    "Doubtful":     0.10,
    "Questionable": 0.50,
    "GTD":          0.50,
    "Day-To-Day":   0.60,
    "Probable":     0.85,
    "Available":    1.00,
}

# Minimum observations before we trust a computed rate
MIN_OBS_LEAGUE = 15      # league-wide per status level
MIN_OBS_PLAYER = 5       # per-player per status level
MIN_OBS_KEYWORD = 10     # per injury keyword


# ====================================================================
#  Data classes
# ====================================================================

@dataclass
class PlayThroughRate:
    """Computed play-through rate for a status level."""
    status_level: str
    played: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        return self.played / self.total if self.total > 0 else 0.0

    @property
    def has_enough_data(self) -> bool:
        return self.total >= MIN_OBS_LEAGUE


@dataclass
class PlayerTendency:
    """Per-player tendency for a given status level."""
    player_id: int
    player_name: str
    status_level: str
    played: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        return self.played / self.total if self.total > 0 else 0.0

    @property
    def has_enough_data(self) -> bool:
        return self.total >= MIN_OBS_PLAYER


@dataclass
class InjuryPlayProbability:
    """Final composite probability that a player plays their next game."""
    player_id: int
    player_name: str
    status_level: str
    injury_keyword: str
    league_rate: float       # league-wide for this status
    player_rate: Optional[float]   # player-specific (None if insufficient data)
    keyword_modifier: float  # multiplier from injury keyword
    composite_probability: float  # final blended probability 0.0 – 1.0
    confidence: str          # "high" | "medium" | "low"


# ====================================================================
#  Backfill:  did the player actually play?
# ====================================================================

def backfill_play_outcomes(
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """Backfill ``did_play`` for rows in ``injury_status_log`` where it's NULL.

    Logic:
        For each un-resolved row, find the team's *next* game date after
        ``log_date``, then check whether the player has stats on that date.
    """
    progress = progress_cb or (lambda _: None)
    migrations.init_db()

    with get_conn() as conn:
        # All un-resolved rows
        pending = conn.execute(
            """
            SELECT isl.id, isl.player_id, isl.team_id, isl.log_date
            FROM injury_status_log isl
            WHERE isl.did_play IS NULL
            ORDER BY isl.log_date
            """
        ).fetchall()

        if not pending:
            progress("Backfill: no pending rows")
            return 0

        progress(f"Backfill: {len(pending)} un-resolved injury status rows…")

        # Pre-load team game dates (player_stats has the canonical record)
        team_dates_df = pd.read_sql(
            """
            SELECT DISTINCT p.team_id, ps.game_date
            FROM player_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE ps.game_id IS NOT NULL AND ps.game_id != ''
            ORDER BY ps.game_date
            """,
            conn,
        )

        # team_id -> sorted list of game-date strings
        team_game_dates: Dict[int, List[str]] = {}
        for tid, grp in team_dates_df.groupby("team_id"):
            team_game_dates[int(tid)] = sorted(grp["game_date"].astype(str).unique().tolist())

        # Pre-load set of (player_id, game_date) who played
        played_set_df = pd.read_sql(
            "SELECT DISTINCT player_id, game_date FROM player_stats "
            "WHERE game_id IS NOT NULL AND game_id != ''",
            conn,
        )
        played_set = set(
            zip(played_set_df["player_id"].astype(int), played_set_df["game_date"].astype(str))
        )

        resolved = 0
        for row_id, pid, tid, log_date_str in pending:
            dates = team_game_dates.get(tid, [])
            if not dates:
                continue

            # Find team's first game date strictly *after* log_date
            next_game = None
            for gd in dates:
                if gd > log_date_str:
                    next_game = gd
                    break

            if next_game is None:
                continue  # no subsequent game yet (maybe season end / future)

            did_play = 1 if (pid, next_game) in played_set else 0
            conn.execute(
                "UPDATE injury_status_log SET did_play = ?, next_game_date = ? WHERE id = ?",
                (did_play, next_game, row_id),
            )
            resolved += 1

        conn.commit()
        progress(f"Backfill complete: {resolved}/{len(pending)} rows resolved")
        return resolved


# ====================================================================
#  League-wide play-through rates
# ====================================================================

def get_league_play_through_rates(conn=None) -> Dict[str, PlayThroughRate]:
    """Compute league-wide play-through rates per status level.

    Only uses rows where ``did_play`` is not NULL (i.e., already backfilled).
    """
    own_conn = conn is None
    if own_conn:
        conn = connect()

    try:
        rows = conn.execute(
            """
            SELECT status_level,
                   SUM(CASE WHEN did_play = 1 THEN 1 ELSE 0 END) AS played,
                   COUNT(*) AS total
            FROM injury_status_log
            WHERE did_play IS NOT NULL
            GROUP BY status_level
            """
        ).fetchall()

        rates: Dict[str, PlayThroughRate] = {}
        for status, played, total in rows:
            rates[status] = PlayThroughRate(
                status_level=status, played=played, total=total,
            )
        return rates
    finally:
        if own_conn:
            conn.close()


# ====================================================================
#  Per-player tendencies
# ====================================================================

def get_player_tendency(
    player_id: int, status_level: str, conn=None,
) -> Optional[PlayerTendency]:
    """Return the player's historical tendency for a given status level."""
    own_conn = conn is None
    if own_conn:
        conn = connect()

    try:
        row = conn.execute(
            """
            SELECT p.name,
                   SUM(CASE WHEN isl.did_play = 1 THEN 1 ELSE 0 END) AS played,
                   COUNT(*) AS total
            FROM injury_status_log isl
            JOIN players p ON p.player_id = isl.player_id
            WHERE isl.player_id = ? AND isl.status_level = ?
              AND isl.did_play IS NOT NULL
            """,
            (player_id, status_level),
        ).fetchone()

        if not row or row[2] == 0:
            return None

        return PlayerTendency(
            player_id=player_id,
            player_name=row[0] or "",
            status_level=status_level,
            played=row[1],
            total=row[2],
        )
    finally:
        if own_conn:
            conn.close()


# ====================================================================
#  Injury-keyword modifier
# ====================================================================

def get_keyword_play_rates(conn=None) -> Dict[str, float]:
    """Compute play-through rate per injury keyword (across all status levels).

    Returns a dict of keyword -> rate.  Only keywords with enough data.
    """
    own_conn = conn is None
    if own_conn:
        conn = connect()

    try:
        rows = conn.execute(
            """
            SELECT injury_keyword,
                   SUM(CASE WHEN did_play = 1 THEN 1 ELSE 0 END) AS played,
                   COUNT(*) AS total
            FROM injury_status_log
            WHERE did_play IS NOT NULL AND injury_keyword != ''
            GROUP BY injury_keyword
            HAVING COUNT(*) >= ?
            """,
            (MIN_OBS_KEYWORD,),
        ).fetchall()

        return {kw: played / total for kw, played, total in rows if total > 0}
    finally:
        if own_conn:
            conn.close()


# ====================================================================
#  Composite play probability
# ====================================================================

def compute_play_probability(
    player_id: int,
    player_name: str,
    status_level: str,
    injury_keyword: str = "",
    conn=None,
) -> InjuryPlayProbability:
    """Compute the composite probability that a player plays their next game.

    Blending strategy:
        1. Start with the league-wide rate for this ``status_level``.
           Falls back to ``DEFAULT_PLAY_RATES`` if insufficient data.
        2. If we have enough per-player data, blend it in (70% player / 30% league
           when data is strong, else less weight).
        3. Apply a keyword modifier based on how the injury keyword compares to the
           overall average play-through rate.

    The result is clamped to [0.0, 1.0].
    """
    own_conn = conn is None
    if own_conn:
        conn = connect()

    try:
        # 1. League rate
        league_rates = get_league_play_through_rates(conn)
        lr = league_rates.get(status_level)
        if lr and lr.has_enough_data:
            league_rate = lr.rate
            confidence = "high" if lr.total >= 50 else "medium"
        else:
            league_rate = DEFAULT_PLAY_RATES.get(status_level, 0.50)
            confidence = "low"

        # 2. Player-specific tendency
        pt = get_player_tendency(player_id, status_level, conn)
        player_rate: Optional[float] = None
        if pt and pt.has_enough_data:
            player_rate = pt.rate
            # Blend: more weight to player if they have lots of data
            player_weight = min(0.70, pt.total / 20.0)  # cap at 70%
            blended = player_rate * player_weight + league_rate * (1 - player_weight)
        else:
            blended = league_rate
            if pt and pt.total > 0:
                # Some data but not enough — small nudge
                player_rate = pt.rate
                nudge_weight = pt.total / (MIN_OBS_PLAYER * 2)
                blended = player_rate * nudge_weight + league_rate * (1 - nudge_weight)

        # 3. Keyword modifier
        keyword_modifier = 1.0
        if injury_keyword:
            kw_rates = get_keyword_play_rates(conn)
            if injury_keyword in kw_rates:
                kw_rate = kw_rates[injury_keyword]
                # Overall average across all keywords
                overall = sum(kw_rates.values()) / len(kw_rates) if kw_rates else 0.5
                if overall > 0:
                    # Ratio: >1 if this keyword plays through more, <1 if less
                    keyword_modifier = kw_rate / overall
                    # Dampen the modifier so it doesn't overwhelm
                    keyword_modifier = 0.5 + keyword_modifier * 0.5  # range ~0.5 – 1.0+

        composite = max(0.0, min(1.0, blended * keyword_modifier))

        return InjuryPlayProbability(
            player_id=player_id,
            player_name=player_name,
            status_level=status_level,
            injury_keyword=injury_keyword,
            league_rate=league_rate,
            player_rate=player_rate,
            keyword_modifier=keyword_modifier,
            composite_probability=composite,
            confidence=confidence,
        )
    finally:
        if own_conn:
            conn.close()


# ====================================================================
#  Batch helper:  get probabilities for all currently injured players
# ====================================================================

def get_team_injury_probabilities(
    team_id: int,
    conn=None,
) -> List[InjuryPlayProbability]:
    """Return play probabilities for every injured player on a team.

    Reads the ``players`` table for ``is_injured=1``, parses the
    ``injury_note`` to extract the status level and keyword, and computes
    the probability for each.
    """
    own_conn = conn is None
    if own_conn:
        conn = connect()

    try:
        rows = conn.execute(
            "SELECT player_id, name, injury_note FROM players "
            "WHERE team_id = ? AND is_injured = 1",
            (team_id,),
        ).fetchall()

        if not rows:
            return []

        from src.data.sync_service import _normalise_status_level, _extract_injury_keyword

        results: List[InjuryPlayProbability] = []
        for pid, pname, note in rows:
            note = note or ""
            # Parse status from note (format: "Status: Injury detail (update)")
            status_raw = note.split(":")[0].strip() if ":" in note else note
            injury_text = note.split(":", 1)[1].strip() if ":" in note else note
            # Remove trailing parenthetical update if present
            if "(" in injury_text:
                injury_text = injury_text[:injury_text.rfind("(")].strip()

            status_level = _normalise_status_level(status_raw)
            keyword = _extract_injury_keyword(injury_text)

            prob = compute_play_probability(pid, pname, status_level, keyword, conn)
            results.append(prob)

        return results
    finally:
        if own_conn:
            conn.close()


# ====================================================================
#  Summary / reporting
# ====================================================================

@dataclass
class InjuryIntelSummary:
    """High-level summary of injury intelligence data."""
    total_log_entries: int = 0
    resolved_entries: int = 0
    unresolved_entries: int = 0
    status_rates: Dict[str, Dict] = field(default_factory=dict)
    top_warriors: List[Dict] = field(default_factory=list)     # players who play through injuries
    top_cautious: List[Dict] = field(default_factory=list)     # players who sit more than expected


def get_injury_intel_summary() -> InjuryIntelSummary:
    """Generate a summary of the injury intelligence data for display."""
    summary = InjuryIntelSummary()

    with get_conn() as conn:
        # Counts
        row = conn.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN did_play IS NOT NULL THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN did_play IS NULL THEN 1 ELSE 0 END) "
            "FROM injury_status_log"
        ).fetchone()
        summary.total_log_entries = row[0] or 0
        summary.resolved_entries = row[1] or 0
        summary.unresolved_entries = row[2] or 0

        # League rates
        rates = get_league_play_through_rates(conn)
        for status, ptr in rates.items():
            summary.status_rates[status] = {
                "played": ptr.played,
                "total": ptr.total,
                "rate": round(ptr.rate * 100, 1),
            }

        # Warriors: highest play rate on "Questionable" / "Day-To-Day" with enough data
        warriors = conn.execute(
            """
            SELECT isl.player_id, p.name,
                   SUM(CASE WHEN isl.did_play = 1 THEN 1 ELSE 0 END) AS played,
                   COUNT(*) AS total
            FROM injury_status_log isl
            JOIN players p ON p.player_id = isl.player_id
            WHERE isl.status_level IN ('Questionable', 'Day-To-Day', 'GTD')
              AND isl.did_play IS NOT NULL
            GROUP BY isl.player_id
            HAVING COUNT(*) >= 3
            ORDER BY CAST(SUM(CASE WHEN isl.did_play = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) DESC
            LIMIT 10
            """,
        ).fetchall()
        for pid, name, played, total in warriors:
            summary.top_warriors.append({
                "player_id": pid, "name": name,
                "played": played, "total": total,
                "rate": round(played / total * 100, 1) if total else 0,
            })

        # Cautious: lowest play rate on same statuses
        cautious = conn.execute(
            """
            SELECT isl.player_id, p.name,
                   SUM(CASE WHEN isl.did_play = 1 THEN 1 ELSE 0 END) AS played,
                   COUNT(*) AS total
            FROM injury_status_log isl
            JOIN players p ON p.player_id = isl.player_id
            WHERE isl.status_level IN ('Questionable', 'Day-To-Day', 'GTD')
              AND isl.did_play IS NOT NULL
            GROUP BY isl.player_id
            HAVING COUNT(*) >= 3
            ORDER BY CAST(SUM(CASE WHEN isl.did_play = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) ASC
            LIMIT 10
            """,
        ).fetchall()
        for pid, name, played, total in cautious:
            summary.top_cautious.append({
                "player_id": pid, "name": name,
                "played": played, "total": total,
                "rate": round(played / total * 100, 1) if total else 0,
            })

    return summary
