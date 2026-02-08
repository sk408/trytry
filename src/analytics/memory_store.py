"""In-memory data store for high-performance analytics.

Loads entire DB tables into Pandas DataFrames so that backtesting,
optimisation, and precomputation can read from RAM instead of hitting
SQLite repeatedly.  Designed for machines with ample memory (16 GB+).

Usage::

    store = get_memory_store()
    if not store.is_loaded:
        store.load(progress_cb=print)
    # Now all analytics code transparently reads from memory.
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.database.db import get_conn

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_instance: Optional["InMemoryDataStore"] = None


def get_memory_store() -> "InMemoryDataStore":
    """Return the global singleton (thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = InMemoryDataStore()
    return _instance


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class InMemoryDataStore:
    """RAM-backed cache of every table used by the analytics pipeline."""

    def __init__(self) -> None:
        self._loaded = False
        self._lock = threading.Lock()

        # Core tables
        self.player_stats: pd.DataFrame = pd.DataFrame()
        self.teams: pd.DataFrame = pd.DataFrame()
        self.players: pd.DataFrame = pd.DataFrame()
        self.team_metrics: pd.DataFrame = pd.DataFrame()
        self.player_impact: pd.DataFrame = pd.DataFrame()
        self.team_tuning: pd.DataFrame = pd.DataFrame()
        self.injury_history: pd.DataFrame = pd.DataFrame()

        # Precomputed game data (set externally by the pipeline)
        self.precomputed_games: list | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Load / flush
    # ------------------------------------------------------------------

    def load(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        """Read all analytics tables into memory."""
        progress = progress_cb or (lambda _: None)

        with self._lock:
            progress("Loading tables into memory...")
            with get_conn() as conn:
                progress("  Loading player_stats...")
                self.player_stats = pd.read_sql("SELECT * FROM player_stats", conn)
                progress(f"  player_stats: {len(self.player_stats):,} rows")

                progress("  Loading teams...")
                self.teams = pd.read_sql("SELECT * FROM teams", conn)

                progress("  Loading players...")
                self.players = pd.read_sql("SELECT * FROM players", conn)

                progress("  Loading team_metrics...")
                try:
                    self.team_metrics = pd.read_sql("SELECT * FROM team_metrics", conn)
                except Exception:
                    self.team_metrics = pd.DataFrame()

                progress("  Loading player_impact...")
                try:
                    self.player_impact = pd.read_sql("SELECT * FROM player_impact", conn)
                except Exception:
                    self.player_impact = pd.DataFrame()

                progress("  Loading team_tuning...")
                try:
                    self.team_tuning = pd.read_sql("SELECT * FROM team_tuning", conn)
                except Exception:
                    self.team_tuning = pd.DataFrame()

                progress("  Loading injury_history...")
                try:
                    self.injury_history = pd.read_sql("SELECT * FROM injury_history", conn)
                except Exception:
                    self.injury_history = pd.DataFrame()

            total_rows = (
                len(self.player_stats) + len(self.teams) + len(self.players)
                + len(self.team_metrics) + len(self.player_impact)
                + len(self.team_tuning) + len(self.injury_history)
            )
            self._loaded = True
            progress(f"Memory store loaded: {total_rows:,} total rows across 7 tables")

    def reload(self, progress_cb: Optional[Callable[[str], None]] = None) -> None:
        """Flush and re-load all data (e.g. after a sync)."""
        self.flush()
        self.load(progress_cb=progress_cb)

    def flush(self) -> None:
        """Free all memory."""
        with self._lock:
            self.player_stats = pd.DataFrame()
            self.teams = pd.DataFrame()
            self.players = pd.DataFrame()
            self.team_metrics = pd.DataFrame()
            self.player_impact = pd.DataFrame()
            self.team_tuning = pd.DataFrame()
            self.injury_history = pd.DataFrame()
            self.precomputed_games = None
            self._loaded = False

    # ------------------------------------------------------------------
    # Convenience accessors (match common SQL queries)
    # ------------------------------------------------------------------

    def get_team_abbrs(self) -> Dict[int, str]:
        """Return {team_id: abbreviation} mapping."""
        if self.teams.empty:
            return {}
        return dict(zip(
            self.teams["team_id"].astype(int),
            self.teams["abbreviation"],
        ))

    def get_team_list(self) -> List[Dict]:
        """Return list of {team_id, abbreviation, name}."""
        if self.teams.empty:
            return []
        return self.teams[["team_id", "abbreviation", "name"]].to_dict("records")

    def get_player_stats_for_team(
        self, team_id: int, before_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Filtered player_stats rows for a team, optionally before a date."""
        if self.player_stats.empty:
            return pd.DataFrame()
        # Join with players to get team_id
        if "team_id" in self.player_stats.columns:
            mask = self.player_stats["team_id"] == team_id
        else:
            # player_stats doesn't have team_id directly; join via players
            pids = set(
                self.players.loc[
                    self.players["team_id"] == team_id, "player_id"
                ].tolist()
            )
            mask = self.player_stats["player_id"].isin(pids)

        df = self.player_stats[mask]
        if before_date and "game_date" in df.columns:
            df = df[df["game_date"] < before_date]
        return df

    def get_team_metrics_dict(self, team_id: int) -> Optional[Dict]:
        """Return team_metrics row as a dict, or None."""
        if self.team_metrics.empty:
            return None
        rows = self.team_metrics[self.team_metrics["team_id"] == team_id]
        if rows.empty:
            return None
        return rows.iloc[-1].to_dict()  # latest season

    def get_game_count_and_last_date(self) -> tuple[int, str]:
        """Return (total_unique_games, max_game_date) from player_stats."""
        if self.player_stats.empty:
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT game_date || '-' || opponent_team_id), "
                    "MAX(game_date) FROM player_stats"
                ).fetchone()
                return (row[0] or 0, row[1] or "")
        dates = self.player_stats["game_date"].unique()
        return len(dates), str(max(dates)) if len(dates) > 0 else ""
