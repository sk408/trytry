"""Model weight configuration for the prediction engine.

Centralises every tunable constant used by ``predict_matchup`` into a
single ``WeightConfig`` dataclass.  Weights can be loaded from the DB
(populated by the weight optimizer) or fall back to sensible defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.database.db import get_conn


@dataclass
class WeightConfig:
    """All tunable weights used by ``predict_matchup``."""

    # --- Defensive adjustment dampening ---
    # 0.5 = dampen defensive factor 50 % toward 1.0
    def_factor_dampening: float = 0.5

    # --- Spread sub-adjustments ---
    turnover_margin_mult: float = 0.4
    rebound_diff_mult: float = 0.08
    rating_matchup_mult: float = 0.08
    four_factors_scale: float = 0.3
    clutch_scale: float = 0.05
    clutch_cap: float = 2.0
    clutch_threshold: float = 6.0       # only apply when |spread| < this
    hustle_effort_mult: float = 0.02
    hustle_contested_wt: float = 0.3    # weight of contested shots in effort

    # --- Four Factors sub-weights (must sum to ~1.0) ---
    ff_efg_weight: float = 0.40
    ff_tov_weight: float = 0.25
    ff_oreb_weight: float = 0.20
    ff_fta_weight: float = 0.15

    # --- Total sub-adjustments ---
    pace_baseline: float = 98.0
    pace_mult: float = 0.20
    steals_threshold: float = 14.0
    steals_penalty: float = 0.15
    blocks_threshold: float = 10.0
    blocks_penalty: float = 0.12
    oreb_baseline: float = 20.0
    oreb_mult: float = 0.2
    hustle_defl_baseline: float = 30.0
    hustle_defl_penalty: float = 0.1
    fatigue_total_mult: float = 0.3

    # --- ESPN blending ---
    espn_spread_scale: float = 0.3      # win-prob edge → implied spread
    espn_model_weight: float = 0.80
    espn_weight: float = 0.20
    espn_disagree_damp: float = 0.85

    # --- Fatigue penalties (points) ---
    fatigue_b2b: float = 2.0
    fatigue_3in4: float = 1.0
    fatigue_4in6: float = 1.5

    # --- Sanity clamps ---
    spread_clamp: float = 18.0
    total_min: float = 195.0
    total_max: float = 248.0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        """Return all weights as a flat ``{name: value}`` dict."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "WeightConfig":
        """Create a config from a dict, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# -----------------------------------------------------------------------
# DB persistence
# -----------------------------------------------------------------------

_DB_TABLE = "model_weights"


def _ensure_table() -> None:
    with get_conn() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_DB_TABLE} (
                key   TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
        """)
        conn.commit()


def save_weights(cfg: WeightConfig) -> None:
    """Persist every weight to the ``model_weights`` table."""
    _ensure_table()
    with get_conn() as conn:
        for k, v in cfg.to_dict().items():
            conn.execute(
                f"INSERT OR REPLACE INTO {_DB_TABLE} (key, value) VALUES (?, ?)",
                (k, float(v)),
            )
        conn.commit()


def load_weights() -> WeightConfig:
    """Load weights from the DB, falling back to defaults for missing keys."""
    _ensure_table()
    with get_conn() as conn:
        rows = conn.execute(f"SELECT key, value FROM {_DB_TABLE}").fetchall()
    if not rows:
        return WeightConfig()
    d = {k: v for k, v in rows}
    return WeightConfig.from_dict(d)


def clear_weights() -> None:
    """Delete all optimised weights so defaults are used."""
    _ensure_table()
    with get_conn() as conn:
        conn.execute(f"DELETE FROM {_DB_TABLE}")
        conn.commit()


# -----------------------------------------------------------------------
# Module-level cached instance (lazy-loaded once)
# -----------------------------------------------------------------------

_cached_config: Optional[WeightConfig] = None


def get_weight_config(force_reload: bool = False) -> WeightConfig:
    """Return the active ``WeightConfig``, loading from DB on first call."""
    global _cached_config
    if _cached_config is None or force_reload:
        _cached_config = load_weights()
    return _cached_config


def set_weight_config(cfg: WeightConfig) -> None:
    """Override the cached config (used during optimisation runs)."""
    global _cached_config
    _cached_config = cfg
