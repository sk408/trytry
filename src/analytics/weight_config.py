"""WeightConfig dataclass — 30+ tunable prediction parameters."""

import json
import os
from dataclasses import dataclass, fields, asdict
from datetime import datetime
from typing import Optional, Dict, List

from src.database import db

_SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "snapshots")


@dataclass
class WeightConfig:
    # Defensive adjustment
    def_factor_dampening: float = 1.70   # sensitivity: optimal ~1.7

    # Spread factors
    turnover_margin_mult: float = 6.00   # sensitivity: optimal ~6.4
    rebound_diff_mult: float = 0.50      # sensitivity: optimal ~0.5
    rating_matchup_mult: float = 0.42    # sensitivity: already near optimal

    # Four Factors
    four_factors_scale: float = 1350.0   # sensitivity: optimal ~1378
    ff_efg_weight: float = 3.55          # sensitivity: optimal ~3.56
    ff_tov_weight: float = 3.40          # sensitivity: optimal ~3.38
    ff_oreb_weight: float = 2.80         # sensitivity: optimal ~2.81
    ff_fta_weight: float = 0.05          # sensitivity: optimal ~0.05

    # Clutch
    clutch_scale: float = 0.13           # sensitivity: optimal ~0.13
    clutch_cap: float = 3.5
    clutch_threshold: float = 6.0

    # Hustle
    hustle_effort_mult: float = 3.90     # sensitivity: optimal ~3.9
    hustle_contested_wt: float = 0.3

    # Pace / Total
    pace_baseline: float = 98.0
    pace_mult: float = 0.0              # sensitivity: dead weight, zeroed
    steals_threshold: float = 14.0
    steals_penalty: float = 0.15
    blocks_threshold: float = 10.0
    blocks_penalty: float = 0.12
    oreb_baseline: float = 20.0
    oreb_mult: float = 0.0              # sensitivity: dead weight, zeroed
    hustle_defl_baseline: float = 30.0
    hustle_defl_penalty: float = 0.1

    # Fatigue — sensitivity: all dead weight, zeroed
    fatigue_total_mult: float = 0.0
    fatigue_b2b: float = 0.0
    fatigue_3in4: float = 0.0
    fatigue_4in6: float = 0.0

    # ESPN blend — sensitivity: dead weight, zeroed
    espn_spread_scale: float = 0.3
    espn_model_weight: float = 0.0
    espn_weight: float = 0.0
    espn_disagree_damp: float = 0.85

    # ML ensemble — sensitivity: dead weight, zeroed
    ml_ensemble_weight: float = 0.0
    ml_disagree_damp: float = 0.0
    ml_disagree_threshold: float = 6.0

    # Sharp Money — edge = (money% - public%) / 100; typically ±0.05 to ±0.15
    sharp_money_weight: float = 1.5

    # Betting edge filter — minimum spread edge (in points) to qualify as a
    # "high-confidence" ATS pick for edge_rate / edge_roi evaluation.
    ats_edge_threshold: float = 3.0

    # Clamps
    spread_clamp: float = 30.0
    total_min: float = 140.0
    total_max: float = 280.0

    def blend(self, other: "WeightConfig",
              self_games: int = 0, other_games: int = 0) -> "WeightConfig":
        """Blend two WeightConfigs weighted by games_analyzed.

        If games counts are provided, the team with more analysed games
        gets proportionally higher influence. Falls back to simple average
        when no counts are provided (backward-compatible).
        """
        total = self_games + other_games
        if total > 0:
            w_self = self_games / total
            w_other = other_games / total
        else:
            w_self = 0.5
            w_other = 0.5

        new_data = {}
        for f in fields(self):
            v1 = getattr(self, f.name)
            v2 = getattr(other, f.name)
            new_data[f.name] = v1 * w_self + v2 * w_other
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


def _enforce_ranges(w: WeightConfig) -> WeightConfig:
    """Clamp all tunable parameters to their OPTIMIZER_RANGES bounds.

    Prevents degenerate configurations (negative weights, inverted signs, tiny clamps)
    from being persisted.
    """
    d = w.to_dict()
    for k, (lo, hi) in OPTIMIZER_RANGES.items():
        if k in d:
            old = d[k]
            d[k] = max(lo, min(hi, d[k]))
            if d[k] != old:
                import logging
                logging.getLogger(__name__).warning(
                    "Weight %s clamped from %.4f to %.4f (range %.4f–%.4f)",
                    k, old, d[k], lo, hi,
                )
    return WeightConfig.from_dict(d)


def save_weight_config(w: WeightConfig):
    """Save global weights to DB and refresh cache.

    All tunable parameters are clamped to OPTIMIZER_RANGES before saving
    to prevent degenerate configurations.
    """
    global _cached_global
    w = _enforce_ranges(w)
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
    """Clear global and per-team weights, reset cache and freshness."""
    global _cached_global
    db.execute("DELETE FROM model_weights")
    db.execute("DELETE FROM team_weight_overrides")
    _cached_global = None
    # Invalidate freshness so the pipeline re-runs weight-related steps
    for step in ("weight_optimize", "team_refine", "residual_cal"):
        db.execute("DELETE FROM sync_meta WHERE step_name = ?", (step,))


def invalidate_weight_cache():
    """Force reload of cached weights."""
    global _cached_global
    _cached_global = None


# Optimizer ranges — constrained to physically meaningful bounds.
# IMPORTANT: Signs must match the expected direction of each parameter.
# Parameters that should always be positive have floor >= 0.
# spread_clamp must stay large enough to distinguish blowouts from close games.
OPTIMIZER_RANGES = {
    # Signs are locked to match basketball logic — no sign flips allowed.
    # spread_clamp is NOT tunable; it's fixed at 30 to prevent compression cheating.
    # Ranges widened per sensitivity analysis 2025-02-28.
    "def_factor_dampening": (0.1, 6.0),      # sensitivity: optimal ~3.7, was capped at 3.0
    "turnover_margin_mult": (0.0, 10.0),     # sensitivity2: optimal ~6.4-6.6, was capped at 5.0
    "rebound_diff_mult": (0.0, 3.0),         # sensitivity: optimal ~0.54, small headroom
    "rating_matchup_mult": (0.0, 2.0),       # matchup edge (stable)
    "four_factors_scale": (50.0, 2000.0),    # sensitivity: optimal 1100-1500, widened ceiling
    "clutch_scale": (0.01, 2.0),             # sensitivity: optimal ~0.13 (in range)
    "hustle_effort_mult": (0.0, 8.0),        # sensitivity: optimal ~4.8, was capped at 5.0, widened
    "ff_efg_weight": (0.0, 6.0),             # sensitivity: optimal ~3.9, was capped at 3.0
    "ff_tov_weight": (0.0, 4.0),             # sensitivity: optimal ~2.6, was capped at 2.0
    "ff_oreb_weight": (0.0, 4.0),            # sensitivity: optimal ~2.0, headroom added
    "ff_fta_weight": (0.0, 4.0),             # sensitivity: optimal ~0.84-3.1, was capped at 2.0
    "blocks_penalty": (0.0, 4.0),            # sensitivity: optimal ~2.7, was capped at 2.0
    "steals_penalty": (0.0, 4.0),            # sensitivity2: optimal ~2.19, was capped at 2.0
    "sharp_money_weight": (0.0, 15.0),       # CD/sensitivity keep finding 12+, was capped at 10
    "ats_edge_threshold": (0.5, 6.0),        # sensitivity: optimal 0.5, lowered floor
}


# ──────────────────────────────────────────────────────────────
# Pipeline Snapshots
# ──────────────────────────────────────────────────────────────

def save_snapshot(name: str, notes: str = "", metrics: Optional[Dict] = None) -> str:
    """Save the current weight config + autotune + optimizer ranges to a named snapshot.

    Returns the path to the snapshot file.
    """
    os.makedirs(_SNAPSHOTS_DIR, exist_ok=True)
    w = get_weight_config()

    # Gather autotune corrections
    autotune = {}
    try:
        rows = db.fetch_all(
            "SELECT t.abbreviation, tt.home_pts_correction, tt.away_pts_correction, tt.games_analyzed "
            "FROM team_tuning tt JOIN teams t ON tt.team_id = t.team_id"
        )
        for r in rows:
            autotune[r["abbreviation"]] = {
                "home_corr": r["home_pts_correction"],
                "away_corr": r["away_pts_correction"],
                "games": r["games_analyzed"],
            }
    except Exception:
        pass

    snapshot = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "notes": notes,
        "weights": w.to_dict(),
        "optimizer_ranges": {k: list(v) for k, v in OPTIMIZER_RANGES.items()},
        "autotune": autotune,
        "metrics": metrics or {},
    }

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{safe_name}.json"
    path = os.path.join(_SNAPSHOTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return path


def load_snapshot(path: str) -> Dict:
    """Load a snapshot from a JSON file. Returns the parsed dict."""
    with open(path) as f:
        return json.load(f)


def restore_snapshot(path: str):
    """Restore weights and autotune corrections from a snapshot file."""
    snap = load_snapshot(path)

    # Restore global weights
    w = WeightConfig.from_dict(snap["weights"])
    save_weight_config(w)

    # Restore autotune corrections
    if snap.get("autotune"):
        db.execute("DELETE FROM team_tuning")
        for abbr, tune in snap["autotune"].items():
            team = db.fetch_one("SELECT team_id FROM teams WHERE abbreviation = ?", (abbr,))
            if team:
                db.execute(
                    "INSERT INTO team_tuning (team_id, home_pts_correction, away_pts_correction, "
                    "games_analyzed, last_tuned_at, tuning_mode) VALUES (?, ?, ?, ?, datetime('now'), 'restored')",
                    (team["team_id"], tune["home_corr"], tune["away_corr"], tune.get("games", 0))
                )

    invalidate_weight_cache()


def list_snapshots() -> List[Dict]:
    """List all available snapshots, newest first."""
    if not os.path.isdir(_SNAPSHOTS_DIR):
        return []
    snaps = []
    for f in sorted(os.listdir(_SNAPSHOTS_DIR), reverse=True):
        if f.endswith(".json"):
            try:
                path = os.path.join(_SNAPSHOTS_DIR, f)
                with open(path) as fh:
                    data = json.load(fh)
                snaps.append({
                    "filename": f,
                    "path": path,
                    "name": data.get("name", f),
                    "created_at": data.get("created_at", ""),
                    "notes": data.get("notes", ""),
                    "metrics": data.get("metrics", {}),
                })
            except Exception:
                continue
    return snaps
