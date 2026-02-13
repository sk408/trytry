"""Centralized in-memory caching for the NBA analytics app.

Provides:
- ``TeamCache`` -- a singleton for all team lookups (~30 rows, rarely changes).
- ``ttl_cache`` -- a decorator that memoises function results with a time-to-live.
- ``clear_all_caches()`` -- flush every registered cache (call after a sync).
- Session-scoped caches for batch workloads (backtests, injury history builds).

All TTL caches auto-expire and can be explicitly flushed at any time via
``clear_all_caches()``.
"""
from __future__ import annotations

import threading
import time as _time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.database.db import get_conn


# ====================================================================
#  Registry of clearable caches
# ====================================================================

_registered_caches: List[Callable[[], None]] = []


def _register(clear_fn: Callable[[], None]) -> None:
    _registered_caches.append(clear_fn)


def clear_all_caches() -> None:
    """Flush every registered cache.  Call after data syncs."""
    for fn in _registered_caches:
        try:
            fn()
        except Exception:
            pass


# ====================================================================
#  Generic TTL-cache decorator
# ====================================================================

def ttl_cache(seconds: float = 600.0, maxsize: int = 256):
    """Decorator: cache return values keyed by args for *seconds*.

    Usage::

        @ttl_cache(seconds=300)
        def expensive(team_id: int) -> dict: ...

    - Only positional/keyword args that are hashable are used as the key.
    - ``func.cache_clear()`` flushes the whole cache.
    - The cache is automatically registered with ``clear_all_caches()``.
    """
    def decorator(func: Callable) -> Callable:
        _store: Dict[tuple, Tuple[float, Any]] = {}
        _lock = threading.Lock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            now = _time.monotonic()
            with _lock:
                entry = _store.get(key)
                if entry is not None:
                    ts, val = entry
                    if now - ts < seconds:
                        return val
            result = func(*args, **kwargs)
            with _lock:
                if len(_store) >= maxsize:
                    # evict oldest
                    oldest_key = min(_store, key=lambda k: _store[k][0])
                    del _store[oldest_key]
                _store[key] = (now, result)
            return result

        def cache_clear() -> None:
            with _lock:
                _store.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        _register(cache_clear)
        return wrapper

    return decorator


# ====================================================================
#  TeamCache singleton
# ====================================================================

class _TeamCache:
    """Cached view of the ``teams`` table.

    All public methods return from RAM; DB is queried at most once per
    ``_TTL`` seconds.  Invalidate explicitly with ``clear()``.
    """

    _TTL = 600.0  # 10 minutes

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ts: float = 0.0
        # Canonical stores
        self._by_id: Dict[int, dict] = {}      # team_id -> {abbr, name}
        self._by_abbr: Dict[str, int] = {}      # abbreviation -> team_id

    # -- internal --

    def _maybe_refresh(self) -> None:
        now = _time.monotonic()
        if self._by_id and (now - self._ts) < self._TTL:
            return
        self._refresh()

    def _refresh(self) -> None:
        try:
            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT team_id, abbreviation, name FROM teams ORDER BY abbreviation"
                ).fetchall()
        except Exception:
            return  # DB may not exist yet; keep stale data
        by_id: Dict[int, dict] = {}
        by_abbr: Dict[str, int] = {}
        for tid, abbr, name in rows:
            by_id[int(tid)] = {"abbr": abbr, "name": name or ""}
            by_abbr[abbr] = int(tid)
        with self._lock:
            self._by_id = by_id
            self._by_abbr = by_abbr
            self._ts = _time.monotonic()

    # -- public API --

    def clear(self) -> None:
        with self._lock:
            self._ts = 0.0

    def id_to_abbr(self) -> Dict[int, str]:
        """Return ``{team_id: abbreviation}``."""
        self._maybe_refresh()
        return {tid: info["abbr"] for tid, info in self._by_id.items()}

    def abbr_to_id(self) -> Dict[str, int]:
        """Return ``{abbreviation: team_id}``."""
        self._maybe_refresh()
        return dict(self._by_abbr)

    def team_list(self) -> List[dict]:
        """Return ``[{id, abbr, name}, ...]`` sorted by abbreviation."""
        self._maybe_refresh()
        return [
            {"id": tid, "abbr": info["abbr"], "name": info["name"]}
            for tid, info in sorted(self._by_id.items(), key=lambda x: x[1]["abbr"])
        ]

    def get_abbr(self, team_id: int) -> Optional[str]:
        self._maybe_refresh()
        info = self._by_id.get(team_id)
        return info["abbr"] if info else None

    def get_id(self, abbr: str) -> Optional[int]:
        self._maybe_refresh()
        return self._by_abbr.get(abbr)

    def teams_df(self) -> pd.DataFrame:
        """Return a DataFrame with columns [team_id, abbreviation, name]."""
        self._maybe_refresh()
        rows = [
            {"team_id": tid, "abbreviation": info["abbr"], "name": info["name"]}
            for tid, info in self._by_id.items()
        ]
        if not rows:
            return pd.DataFrame(columns=["team_id", "abbreviation", "name"])
        return pd.DataFrame(rows).sort_values("abbreviation").reset_index(drop=True)

    def abbrs_for_ids(self, *team_ids: int) -> Dict[int, str]:
        """Return ``{team_id: abbreviation}`` for the given IDs."""
        self._maybe_refresh()
        return {
            tid: self._by_id[tid]["abbr"]
            for tid in team_ids
            if tid in self._by_id
        }


# Module-level singleton
team_cache = _TeamCache()
_register(team_cache.clear)


# ====================================================================
#  Session-scoped caches (for batch workloads)
# ====================================================================

class SessionCache:
    """A simple dict-based cache that can be enabled/disabled.

    When ``active`` the cache stores results keyed by arbitrary hashable
    keys.  When inactive, ``get`` always returns ``None`` and ``put`` is
    a no-op.  Call ``start()`` at the beginning of a batch job and
    ``stop()`` at the end.
    """

    def __init__(self) -> None:
        self._data: Dict[Any, Any] = {}
        self.active = False

    def start(self) -> None:
        self._data.clear()
        self.active = True

    def stop(self) -> None:
        self._data.clear()
        self.active = False

    def get(self, key) -> Any:
        if self.active:
            return self._data.get(key)
        return None

    def put(self, key, value) -> None:
        if self.active:
            self._data[key] = value

    def clear(self) -> None:
        self._data.clear()


# Pre-built session caches for backtest / injury-history batch jobs
backtest_cache = SessionCache()
injury_history_cache = SessionCache()
_register(backtest_cache.clear)
_register(injury_history_cache.clear)
