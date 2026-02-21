"""In-memory data store for bulk-preloaded NBA data.

Loading all frequently-accessed tables into RAM once (at process start)
eliminates thousands of per-game SQLite round trips during backtesting.
With ~450 players × ~80 games and 30 teams, total memory usage is ~30-50 MB
— negligible on a 96 GB machine.

Usage::

    from src.analytics.data_store import preload_all, store

    preload_all()                        # once per process
    df = store.player_df(player_id)      # instant dict lookup
    m  = store.team_metrics(team_id)     # instant dict lookup
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from src.database.db import get_conn

_log = logging.getLogger(__name__)


class DataStore:
    """Centralised in-memory cache of all read-only reference data."""

    def __init__(self) -> None:
        self._loaded = False
        # player_stats partitioned by player_id → pre-sorted DESC by game_date
        self._player_dfs: Dict[int, pd.DataFrame] = {}
        # team_metrics keyed by (team_id, season)
        self._team_metrics: Dict[tuple, Dict] = {}
        # team_metrics keyed by team_id only (current season default)
        self._team_metrics_by_id: Dict[int, Dict] = {}
        # team_tuning keyed by team_id
        self._team_tuning: Dict[int, Dict] = {}
        # team_weight_overrides keyed by team_id → list of (key, value)
        self._team_weights: Dict[int, List[tuple]] = {}
        # model_weights → {key: value}
        self._model_weights: Dict[str, float] = {}
        # Sorted game dates per team (ascending)
        self._schedule_dates: Dict[int, List[date]] = {}
        # injury_history: (team_id, game_date_str) → list of injured player dicts
        self._injury_history: Dict[tuple, List[Dict]] = {}
        # players table: player_id → {team_id, name, position, is_injured}
        self._players: Dict[int, Dict] = {}
        # roster per game: (team_id, game_date_str) → [player_ids]
        self._rosters: Dict[tuple, List[int]] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Public accessors ────────────────────────────────────────────

    def player_df(self, player_id: int) -> Optional[pd.DataFrame]:
        """Return the preloaded DataFrame for a player (already sorted DESC by game_date)."""
        if not self._loaded:
            return None
        return self._player_dfs.get(player_id)

    def team_metrics(self, team_id: int, season: Optional[str] = None) -> Optional[Dict]:
        if not self._loaded:
            return None
        if season:
            return self._team_metrics.get((team_id, season))
        return self._team_metrics_by_id.get(team_id)

    def team_tuning(self, team_id: int) -> Optional[Dict]:
        if not self._loaded:
            return None
        # Return None to signal "not preloaded" vs "no tuning" (empty dict)
        return self._team_tuning.get(team_id, None)

    def team_weight_rows(self, team_id: int) -> Optional[List[tuple]]:
        if not self._loaded:
            return None
        return self._team_weights.get(team_id)

    def model_weight_rows(self) -> Optional[Dict[str, float]]:
        if not self._loaded:
            return None
        return self._model_weights

    def schedule_dates(self, team_id: int) -> Optional[List[date]]:
        if not self._loaded:
            return None
        return self._schedule_dates.get(team_id)

    def injuries_for_game(self, team_id: int, game_date_str: str) -> Optional[List[Dict]]:
        if not self._loaded:
            return None
        return self._injury_history.get((team_id, game_date_str))

    def player_info(self, player_id: int) -> Optional[Dict]:
        if not self._loaded:
            return None
        return self._players.get(player_id)

    def roster_for_game(self, team_id: int, game_date_str: str) -> Optional[List[int]]:
        if not self._loaded:
            return None
        return self._rosters.get((team_id, game_date_str))

    def all_player_ids_for_team(self, team_id: int) -> List[int]:
        """Return all non-injured player_ids on a team's current roster."""
        if not self._loaded:
            return []
        return [
            pid for pid, info in self._players.items()
            if info.get("team_id") == team_id and not info.get("is_injured", False)
        ]


# Module-level singleton
store = DataStore()


def preload_all() -> None:
    """Bulk-load all reference tables into the module-level ``store``.

    Takes ~1-2 seconds, saves minutes of per-game queries during backtesting.
    Safe to call multiple times (idempotent).
    """
    if store._loaded:
        return

    _log.info("[DataStore] Preloading all reference data into RAM...")

    with get_conn() as conn:
        # ── 1. player_stats  (the big one) ──────────────────────────
        all_stats = pd.read_sql(
            "SELECT * FROM player_stats ORDER BY player_id, game_date DESC",
            conn,
            parse_dates=["game_date"],
        )
        _log.info("[DataStore]   player_stats: %d rows", len(all_stats))

        store._player_dfs = {
            int(pid): group.reset_index(drop=True)
            for pid, group in all_stats.groupby("player_id")
        }

        # ── 1b. Build roster lookup from player_stats ───────────────
        # Each (team_id, game_date) → distinct player_ids who played
        roster_groups = (
            all_stats[["player_id", "game_date"]]
            .drop_duplicates()
        )
        # We need team_id — join via players table after loading it

        # ── 1c. Build schedule dates from player_stats ──────────────
        # Get team_id for each player so we can group by team
        player_team_rows = conn.execute(
            "SELECT player_id, team_id FROM players"
        ).fetchall()
        pid_to_team = {r[0]: r[1] for r in player_team_rows}

        # For each player_stats row, map to team_id then collect distinct dates
        dates_by_team: Dict[int, set] = defaultdict(set)
        for _, row in all_stats[["player_id", "game_date"]].drop_duplicates().iterrows():
            pid = int(row["player_id"])
            gd = row["game_date"]
            tid = pid_to_team.get(pid)
            if tid is not None:
                if isinstance(gd, pd.Timestamp):
                    dates_by_team[tid].add(gd.date())
                elif isinstance(gd, date):
                    dates_by_team[tid].add(gd)
                elif isinstance(gd, str):
                    dates_by_team[tid].add(date.fromisoformat(gd[:10]))

        store._schedule_dates = {
            tid: sorted(dates) for tid, dates in dates_by_team.items()
        }

        # Build roster lookup: (team_id, game_date_str) → [player_ids]
        rosters: Dict[tuple, List[int]] = defaultdict(list)
        for _, row in all_stats[["player_id", "game_date"]].drop_duplicates().iterrows():
            pid = int(row["player_id"])
            gd = row["game_date"]
            tid = pid_to_team.get(pid)
            if tid is not None:
                gd_str = str(gd.date()) if isinstance(gd, pd.Timestamp) else str(gd)[:10]
                rosters[(tid, gd_str)].append(pid)
        store._rosters = dict(rosters)

        # ── 2. team_metrics ─────────────────────────────────────────
        try:
            tm_rows = conn.execute("SELECT * FROM team_metrics").fetchall()
            tm_cols = [d[0] for d in conn.execute(
                "SELECT * FROM team_metrics LIMIT 0"
            ).description]
            for row in tm_rows:
                d = dict(zip(tm_cols, row))
                tid = int(d.get("team_id", 0))
                season = d.get("season", "")
                store._team_metrics[(tid, season)] = d
                store._team_metrics_by_id[tid] = d  # latest wins
            _log.info("[DataStore]   team_metrics: %d rows", len(tm_rows))
        except Exception:
            _log.info("[DataStore]   team_metrics: table not found or empty")

        # ── 3. team_tuning ──────────────────────────────────────────
        try:
            tt_rows = conn.execute("""
                SELECT team_id, home_pts_correction, away_pts_correction,
                       games_analyzed, avg_spread_error_before,
                       avg_total_error_before, last_tuned_at,
                       tuning_mode, tuning_version, tuning_sample_size
                FROM team_tuning
            """).fetchall()
            for row in tt_rows:
                store._team_tuning[int(row[0])] = {
                    "team_id": row[0],
                    "home_pts_correction": row[1],
                    "away_pts_correction": row[2],
                    "games_analyzed": row[3],
                    "avg_spread_error_before": row[4],
                    "avg_total_error_before": row[5],
                    "last_tuned_at": row[6],
                    "tuning_mode": row[7] if len(row) > 7 else "classic",
                    "tuning_version": row[8] if len(row) > 8 else "v1_classic",
                    "tuning_sample_size": row[9] if len(row) > 9 else row[3],
                }
            _log.info("[DataStore]   team_tuning: %d rows", len(tt_rows))
        except Exception:
            _log.info("[DataStore]   team_tuning: table not found or empty")

        # ── 4. team_weight_overrides ────────────────────────────────
        try:
            tw_rows = conn.execute(
                "SELECT team_id, key, value FROM team_weight_overrides"
            ).fetchall()
            tw_by_team: Dict[int, List[tuple]] = defaultdict(list)
            for tid, k, v in tw_rows:
                tw_by_team[int(tid)].append((k, v))
            store._team_weights = dict(tw_by_team)
            _log.info("[DataStore]   team_weight_overrides: %d rows", len(tw_rows))
        except Exception:
            _log.info("[DataStore]   team_weight_overrides: table not found or empty")

        # ── 5. model_weights ────────────────────────────────────────
        try:
            mw_rows = conn.execute(
                "SELECT key, value FROM model_weights"
            ).fetchall()
            store._model_weights = {k: float(v) for k, v in mw_rows}
            _log.info("[DataStore]   model_weights: %d rows", len(mw_rows))
        except Exception:
            _log.info("[DataStore]   model_weights: table not found or empty")

        # ── 6. injury_history ───────────────────────────────────────
        try:
            ih_df = pd.read_sql(
                """
                SELECT ih.team_id, ih.game_date, ih.player_id, ih.was_out,
                       ih.avg_minutes, p.name, p.position
                FROM injury_history ih
                JOIN players p ON p.player_id = ih.player_id
                WHERE ih.was_out = 1
                ORDER BY ih.team_id, ih.game_date, ih.avg_minutes DESC
                """,
                conn,
            )
            injuries: Dict[tuple, List[Dict]] = defaultdict(list)
            for _, row in ih_df.iterrows():
                key = (int(row["team_id"]), str(row["game_date"])[:10])
                injuries[key].append({
                    "player_id": int(row["player_id"]),
                    "name": str(row["name"]),
                    "position": str(row["position"] or ""),
                    "avg_minutes": float(row["avg_minutes"]),
                })
            store._injury_history = dict(injuries)
            _log.info("[DataStore]   injury_history: %d game-team entries", len(injuries))
        except Exception:
            _log.info("[DataStore]   injury_history: table not found or empty")

        # ── 7. players table ────────────────────────────────────────
        try:
            p_rows = conn.execute(
                "SELECT player_id, team_id, name, position, is_injured FROM players"
            ).fetchall()
            for row in p_rows:
                store._players[int(row[0])] = {
                    "team_id": int(row[1]) if row[1] else 0,
                    "name": str(row[2]),
                    "position": str(row[3] or ""),
                    "is_injured": bool(row[4]),
                }
            _log.info("[DataStore]   players: %d rows", len(p_rows))
        except Exception:
            _log.info("[DataStore]   players: table not found or empty")

    store._loaded = True
    _log.info("[DataStore] Preload complete — all data in RAM")


def clear_store() -> None:
    """Reset the store (e.g. after DB mutations during optimisation)."""
    store._loaded = False
    store._player_dfs.clear()
    store._team_metrics.clear()
    store._team_metrics_by_id.clear()
    store._team_tuning.clear()
    store._team_weights.clear()
    store._model_weights.clear()
    store._schedule_dates.clear()
    store._injury_history.clear()
    store._players.clear()
    store._rosters.clear()
    _log.info("[DataStore] Store cleared")
