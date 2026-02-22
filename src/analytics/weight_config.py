"""WeightConfig dataclass — 30+ tunable prediction parameters."""

from dataclasses import dataclass, fields, asdict
from typing import Optional, Dict

from src.database import db


@dataclass
class WeightConfig:
    # Defensive adjustment
    def_factor_dampening: float = 0.90   # optimizer: 0.898 (keeps hitting ceiling)

    # Spread factors
    turnover_margin_mult: float = 0.50   # optimizer: 0.499 (was 0.26)
    rebound_diff_mult: float = 0.047     # optimizer: 0.047 re-enabled (was 0)
    rating_matchup_mult: float = 0.42    # optimizer: 0.419 (stable)

    # Four Factors
    four_factors_scale: float = 180.0    # optimizer: 180 (was 200)
    ff_efg_weight: float = 0.40
    ff_tov_weight: float = 0.25
    ff_oreb_weight: float = 0.20
    ff_fta_weight: float = 0.15

    # Clutch
    clutch_scale: float = 0.074          # optimizer: 0.074 (was 0.12)
    clutch_cap: float = 3.5
    clutch_threshold: float = 6.0

    # Hustle
    hustle_effort_mult: float = 0.007    # optimizer: 0.007 (nearly zero)
    hustle_contested_wt: float = 0.3

    # Pace / Total
    pace_baseline: float = 98.0
    pace_mult: float = 0.08             # optimizer: 0.081 (was 0.23)
    steals_threshold: float = 14.0
    steals_penalty: float = 0.15
    blocks_threshold: float = 10.0
    blocks_penalty: float = 0.12
    oreb_baseline: float = 20.0
    oreb_mult: float = 0.2
    hustle_defl_baseline: float = 30.0
    hustle_defl_penalty: float = 0.1

    # Fatigue
    fatigue_total_mult: float = 0.3
    fatigue_b2b: float = 2.0
    fatigue_3in4: float = 1.0
    fatigue_4in6: float = 1.5

    # ESPN blend
    espn_spread_scale: float = 0.3
    espn_model_weight: float = 0.80
    espn_weight: float = 0.20
    espn_disagree_damp: float = 0.85

    # ML ensemble
    ml_ensemble_weight: float = 0.33    # optimizer: 0.334 — ML now significant contributor
    ml_disagree_damp: float = 0.5
    ml_disagree_threshold: float = 6.0

    # Clamps
    spread_clamp: float = 25.0
    total_min: float = 140.0
    total_max: float = 280.0

    def blend(self, other: "WeightConfig") -> "WeightConfig":
        """Average all fields with another WeightConfig."""
        new_data = {}
        for f in fields(self):
            v1 = getattr(self, f.name)
            v2 = getattr(other, f.name)
            new_data[f.name] = (v1 + v2) / 2.0
        return WeightConfig(**new_data)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "WeightConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ──────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────

_cached_global: Optional[WeightConfig] = None


def get_weight_config() -> WeightConfig:
    """Lazy-loaded, cached singleton for global weights."""
    global _cached_global
    if _cached_global is not None:
        return _cached_global
    rows = db.fetch_all("SELECT key, value FROM model_weights")
    if rows:
        d = {r["key"]: r["value"] for r in rows}
        _cached_global = WeightConfig.from_dict(d)
    else:
        _cached_global = WeightConfig()
    return _cached_global


def save_weight_config(w: WeightConfig):
    """Save global weights to DB and refresh cache."""
    global _cached_global
    for k, v in w.to_dict().items():
        db.execute(
            "INSERT INTO model_weights (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (k, v)
        )
    _cached_global = w


def load_team_weights(team_id: int) -> Optional[WeightConfig]:
    """Load per-team weight overrides. Returns None if no overrides exist."""
    rows = db.fetch_all(
        "SELECT key, value FROM team_weight_overrides WHERE team_id = ?",
        (team_id,)
    )
    if not rows:
        return None
    base = get_weight_config().to_dict()
    for r in rows:
        if r["key"] in base:
            base[r["key"]] = r["value"]
    return WeightConfig.from_dict(base)


def save_team_weights(team_id: int, w: WeightConfig):
    """Save per-team weight overrides."""
    for k, v in w.to_dict().items():
        db.execute(
            "INSERT INTO team_weight_overrides (team_id, key, value) VALUES (?,?,?) "
            "ON CONFLICT(team_id, key) DO UPDATE SET value=excluded.value",
            (team_id, k, v)
        )


def clear_all_weights():
    """Clear global and per-team weights, reset cache."""
    global _cached_global
    db.execute("DELETE FROM model_weights")
    db.execute("DELETE FROM team_weight_overrides")
    _cached_global = None


def invalidate_weight_cache():
    """Force reload of cached weights."""
    global _cached_global
    _cached_global = None


# Optimizer ranges
OPTIMIZER_RANGES = {
    "def_factor_dampening": (0.25, 1.10),
    "turnover_margin_mult": (0.10, 0.65),
    "rebound_diff_mult": (0.0, 0.15),
    "rating_matchup_mult": (0.40, 1.20),
    "four_factors_scale": (80.0, 250.0),
    "clutch_scale": (0.02, 0.15),
    "hustle_effort_mult": (0.005, 0.05),
    "pace_mult": (0.08, 0.35),
    "fatigue_total_mult": (0.10, 0.60),
    "espn_model_weight": (0.60, 0.95),
    "ml_ensemble_weight": (0.0, 0.5),
    "ml_disagree_damp": (0.3, 1.0),
}
