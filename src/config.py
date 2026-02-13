"""Application-wide settings, persisted to ``data/app_settings.json``.

Provides a thin get/set layer over a JSON file.  Settings survive
restarts and are shared across all UI and API entry points.

Usage::

    from src.config import get_setting, set_setting, get_default_workers

    workers = get_default_workers()          # respects saved setting
    set_setting("max_workers", 16)           # persists immediately
    get_setting("max_workers", fallback=8)   # with explicit fallback
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "app_settings.json"

# Detected CPU thread count (used as fallback when no setting is saved)
_CPU_COUNT = max(1, os.cpu_count() or 4)


def _load() -> dict:
    """Load the settings file, returning {} on any error."""
    try:
        if _SETTINGS_PATH.exists():
            return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save(data: dict) -> None:
    """Write settings to disk."""
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def get_setting(key: str, fallback: Any = None) -> Any:
    """Read a single setting.  Returns *fallback* if not set."""
    return _load().get(key, fallback)


def set_setting(key: str, value: Any) -> None:
    """Write a single setting (persists immediately)."""
    data = _load()
    data[key] = value
    _save(data)


def get_all_settings() -> dict:
    """Return a copy of all saved settings."""
    return _load()


# ── Convenience helpers for common settings ──


def get_default_workers() -> int:
    """Return the global max-workers setting.

    Falls back to ``os.cpu_count()`` when nothing has been saved yet.
    """
    val = get_setting("max_workers")
    if val is not None:
        try:
            return max(1, int(val))
        except (ValueError, TypeError):
            pass
    return _CPU_COUNT


def get_cpu_count() -> int:
    """Return the detected CPU thread count (not the saved setting)."""
    return _CPU_COUNT
