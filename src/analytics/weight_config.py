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


# Optimizer ranges — expanded based on Round 9 sensitivity sweep findings
OPTIMIZER_RANGES = {
    "def_factor_dampening": (0.0, 4.0),      # sweep best-loss ~1.38; widen ceiling for interaction effects
    "turnover_margin_mult": (-1.0, 4.0),     # sweep best ~1.1; allow negative for flexibility
    "rebound_diff_mult": (-1.0, 2.0),        # sweep best-loss at -0.40; was clipped at -0.30
    "rating_matchup_mult": (-1.0, 2.0),      # sweep best ~0.40; keep wide
    "four_factors_scale": (50.0, 300.0),      # sweep best ~88; lower floor to reach it
    "clutch_scale": (-0.10, 1.0),            # sweep best-loss at 0.04; lower floor below old 0.02
    "hustle_effort_mult": (-0.10, 1.0),      # sweep best-loss at 0.04; allow slight negatives
    "pace_mult": (-0.50, 1.0),              # sweep best-loss at -0.10; was floored at 0.08
    "fatigue_total_mult": (-0.50, 2.0),      # sweep best-loss at 0.85; widen both ends
    "espn_model_weight": (0.0, 1.0),         # sweep best-loss at 0.0; was floored at 0.60
    "ml_ensemble_weight": (-2.0, 2.0),       # sweep best-loss at -2.0; was floored at 0.0
    "ml_disagree_damp": (0.0, 1.5),          # sweep best at 0.0; slight ceiling raise
    "spread_clamp": (3.0, 15.0),             # sweep best-loss at 6.9; tighten ceiling, lower floor
    "ff_efg_weight": (0.0, 3.0),             # NEW — sweep best-loss at 1.38
    "ff_tov_weight": (-0.5, 2.0),            # NEW — sweep best-loss at 0.22
    "ff_oreb_weight": (0.0, 3.0),            # NEW — sweep best-loss at 1.31
    "ff_fta_weight": (0.0, 2.0),             # NEW — sweep best-loss at 0.57
    "blocks_penalty": (-0.5, 2.0),           # NEW — sweep best-loss at 0.55
    "steals_penalty": (-0.5, 2.0),           # NEW — sweep best-loss at 0.25
    "oreb_mult": (-0.5, 2.0),               # NEW — sweep best-loss at 0.08
    "pace_baseline": (80.0, 115.0),           # NEW — sweep best-loss at 85.0
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
