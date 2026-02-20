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

    # --- ML Ensemble blending ---
    ml_ensemble_weight: float = 0.4       # weight for ML model (0 = disabled)
    ml_disagree_damp: float = 0.7         # dampen ML when it disagrees with base by >8 pts
    ml_disagree_threshold: float = 8.0    # spread disagreement threshold (points)

    # --- Sanity clamps ---
    spread_clamp: float = 18.0
    total_min: float = 180.0       # was 195 — allow for rare low-scoring games
    total_max: float = 255.0       # was 248 — allow for rare high-scoring games

    # --- Player contribution blend weights ---
    # Used in aggregate_projection: overall/location/vs-opponent split
    player_base_weight: float = 0.40
    player_location_weight: float = 0.30
    player_vs_opp_weight: float = 0.30

    # --- Injury recovery ---
    injury_usage_boost: float = 0.30        # fraction of absent star's PPG redistributed
    injury_minute_efficiency: float = 0.85  # production efficiency for redistributed minutes
    injury_recovery_factor: float = 0.30    # fraction of lost production recovered by team
    injury_onoff_multiplier: float = 0.50   # scale for on/off net-rating impact

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

    def blend(self, other: "WeightConfig") -> "WeightConfig":
        """Return a new config whose fields are the average of *self* and *other*.

        Used when both teams in a matchup have per-team weight overrides —
        instead of arbitrarily picking one side's weights, we average them.
        """
        merged = {}
        for f in self.__dataclass_fields__:
            a = getattr(self, f)
            b = getattr(other, f)
            merged[f] = (a + b) / 2.0
        return WeightConfig(**merged)


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


# -----------------------------------------------------------------------
# Per-team weight overrides
# -----------------------------------------------------------------------

_TEAM_WEIGHTS_TABLE = "team_weight_overrides"


def _ensure_team_weights_table() -> None:
    with get_conn() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_TEAM_WEIGHTS_TABLE} (
                team_id  INTEGER NOT NULL,
                key      TEXT NOT NULL,
                value    REAL NOT NULL,
                PRIMARY KEY (team_id, key)
            )
        """)
        conn.commit()


def save_team_weights(team_id: int, cfg: WeightConfig) -> None:
    """Persist per-team weight overrides."""
    _ensure_team_weights_table()
    with get_conn() as conn:
        # Clear old overrides for this team
        conn.execute(f"DELETE FROM {_TEAM_WEIGHTS_TABLE} WHERE team_id = ?", (team_id,))
        for k, v in cfg.to_dict().items():
            conn.execute(
                f"INSERT INTO {_TEAM_WEIGHTS_TABLE} (team_id, key, value) VALUES (?, ?, ?)",
                (team_id, k, float(v)),
            )
        conn.commit()


def load_team_weights(team_id: int) -> Optional[WeightConfig]:
    """Load per-team weight overrides.  Returns None if none exist."""
    _ensure_team_weights_table()
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT key, value FROM {_TEAM_WEIGHTS_TABLE} WHERE team_id = ?",
            (team_id,),
        ).fetchall()
    if not rows:
        return None
    d = {k: v for k, v in rows}
    return WeightConfig.from_dict(d)


def clear_team_weights(team_id: Optional[int] = None) -> None:
    """Clear per-team overrides.  If team_id is None, clear ALL teams."""
    _ensure_team_weights_table()
    with get_conn() as conn:
        if team_id is not None:
            conn.execute(f"DELETE FROM {_TEAM_WEIGHTS_TABLE} WHERE team_id = ?", (team_id,))
        else:
            conn.execute(f"DELETE FROM {_TEAM_WEIGHTS_TABLE}")
        conn.commit()


def get_team_weight_config(team_id: int) -> WeightConfig:
    """Return team-specific weights if they exist, else global weights."""
    team_cfg = load_team_weights(team_id)
    if team_cfg is not None:
        return team_cfg
    return get_weight_config()


def get_team_refinement_summary() -> list[dict]:
    """Return a summary of which teams have overrides."""
    _ensure_team_weights_table()
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT tw.team_id, t.abbreviation, COUNT(*) as n_overrides
            FROM {_TEAM_WEIGHTS_TABLE} tw
            JOIN teams t ON t.team_id = tw.team_id
            GROUP BY tw.team_id
            ORDER BY t.abbreviation
        """).fetchall()
    return [{"team_id": r[0], "abbr": r[1], "n_overrides": r[2]} for r in rows]
