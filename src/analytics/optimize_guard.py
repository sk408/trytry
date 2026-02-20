"""Optimization regression guard — auto-rollback, blacklist, and suggestions.

Provides:
- **Weight backup / restore** before and after optimisation runs.
- **Regression detection** using a composite score.
- **Auto-rollback** when the pipeline makes things worse.
- **Blacklist** of parameter fingerprints that previously caused regressions.
- **Full variable-diff logging** when a regression occurs.
- **Suggestions** for next-run parameter adjustments.

Persistence files (in ``data/``):
    optimize_history.json   – full run history with variable dumps
    optimize_blacklist.json – fingerprints of regressive configs
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_HISTORY_PATH = _DATA_DIR / "optimize_history.json"
_BLACKLIST_PATH = _DATA_DIR / "optimize_blacklist.json"
_ML_DIR = _DATA_DIR / "ml_models"
_ML_BACKUP_DIR = _DATA_DIR / "ml_models_backup"

# ---------------------------------------------------------------------------
#  Composite score
# ---------------------------------------------------------------------------

# Weights for the composite fitness score (higher → better).
# winner_pct is the most important (already a percentage),
# errors are inverted (lower is better → subtract them).
_W_WINNER = 1.0
_W_SPREAD_ERR = 2.0       # penalise spread error heavily
_W_TOTAL_IN_10 = 0.5
_W_TOTAL_ERR = 1.0

# Threshold: if composite drops by *this* much or more → regression.
REGRESSION_THRESHOLD = 0.0   # any drop counts


def composite_score(winner_pct: float, avg_spread_err: float,
                    total_in_10_pct: float, avg_total_err: float) -> float:
    """Compute a single fitness number (higher = better)."""
    return (winner_pct * _W_WINNER
            - avg_spread_err * _W_SPREAD_ERR
            + total_in_10_pct * _W_TOTAL_IN_10
            - avg_total_err * _W_TOTAL_ERR)


# ---------------------------------------------------------------------------
#  Run record
# ---------------------------------------------------------------------------

@dataclass
class OptimizeRunRecord:
    """Full record of a single optimization run."""
    timestamp: float = 0.0
    n_trials: int = 0
    team_trials: int = 0
    force_rerun: bool = False

    # Before / after metrics
    before_winner_pct: float = 0.0
    before_avg_spread_error: float = 0.0
    before_total_in_10_pct: float = 0.0
    before_avg_total_error: float = 0.0

    after_winner_pct: float = 0.0
    after_avg_spread_error: float = 0.0
    after_total_in_10_pct: float = 0.0
    after_avg_total_error: float = 0.0

    # Deltas
    delta_winner_pct: float = 0.0
    delta_avg_spread_error: float = 0.0
    delta_total_in_10_pct: float = 0.0
    delta_avg_total_error: float = 0.0

    # Scores
    score_before: float = 0.0
    score_after: float = 0.0
    score_delta: float = 0.0

    # Regression info
    was_regression: bool = False
    regression_reasons: List[str] = field(default_factory=list)
    was_rolled_back: bool = False

    # Full weight snapshots
    weights_before: Dict[str, float] = field(default_factory=dict)
    weights_after: Dict[str, float] = field(default_factory=dict)
    weight_diffs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Config fingerprint
    config_fingerprint: str = ""


# ---------------------------------------------------------------------------
#  Weight backup / restore
# ---------------------------------------------------------------------------

class WeightBackup:
    """Snapshot of all weights (global + per-team) and ML model files."""

    def __init__(self) -> None:
        self.global_weights: Dict[str, float] = {}
        self.team_overrides: Dict[int, Dict[str, float]] = {}
        self.timestamp: float = 0.0
        self._ml_backed_up = False

    def capture(self) -> None:
        """Snapshot current global weights, per-team overrides, and ML models."""
        from src.analytics.weight_config import (
            load_weights,
            get_team_refinement_summary,
            load_team_weights,
        )
        log.info("[OptGuard] Capturing weight backup...")
        self.timestamp = time.time()

        # Global
        cfg = load_weights()
        self.global_weights = cfg.to_dict()

        # Per-team
        self.team_overrides = {}
        try:
            teams = get_team_refinement_summary()
            for t in teams:
                tid = t["team_id"]
                tcfg = load_team_weights(tid)
                if tcfg is not None:
                    self.team_overrides[tid] = tcfg.to_dict()
        except Exception as exc:
            log.warning("[OptGuard] Could not load team overrides: %s", exc)

        # ML model files
        self._backup_ml_models()

        log.info("[OptGuard] Backup captured: %d global params, %d team overrides, ML=%s",
                 len(self.global_weights), len(self.team_overrides), self._ml_backed_up)

    def restore(self) -> None:
        """Restore all weights and ML models from backup."""
        from src.analytics.weight_config import (
            WeightConfig,
            save_weights,
            set_weight_config,
            save_team_weights,
            clear_team_weights,
        )
        if not self.global_weights:
            log.warning("[OptGuard] No backup to restore from!")
            return

        log.info("[OptGuard] ROLLING BACK → restoring weights from backup (ts=%.0f)", self.timestamp)

        # Restore global weights
        cfg = WeightConfig.from_dict(self.global_weights)
        save_weights(cfg)
        set_weight_config(cfg)

        # Restore per-team
        clear_team_weights()  # wipe all, then re-insert
        for tid, wdict in self.team_overrides.items():
            tcfg = WeightConfig.from_dict(wdict)
            save_team_weights(tid, tcfg)

        # Restore ML models
        self._restore_ml_models()

        log.info("[OptGuard] Rollback complete — weights and models restored")

    def _backup_ml_models(self) -> None:
        """Copy ML model artifacts to a backup dir."""
        try:
            if _ML_DIR.exists():
                if _ML_BACKUP_DIR.exists():
                    shutil.rmtree(_ML_BACKUP_DIR)
                shutil.copytree(_ML_DIR, _ML_BACKUP_DIR)
                self._ml_backed_up = True
                log.info("[OptGuard] ML models backed up → %s", _ML_BACKUP_DIR)
        except Exception as exc:
            log.warning("[OptGuard] ML backup failed: %s", exc)

    def _restore_ml_models(self) -> None:
        """Restore ML model artifacts from backup."""
        try:
            if self._ml_backed_up and _ML_BACKUP_DIR.exists():
                if _ML_DIR.exists():
                    shutil.rmtree(_ML_DIR)
                shutil.copytree(_ML_BACKUP_DIR, _ML_DIR)
                # Reload models in memory
                try:
                    from src.analytics.ml_model import reload_models
                    reload_models()
                except Exception:
                    pass
                log.info("[OptGuard] ML models restored from backup")
        except Exception as exc:
            log.warning("[OptGuard] ML model restore failed: %s", exc)

    def cleanup(self) -> None:
        """Remove temporary ML backup directory."""
        try:
            if _ML_BACKUP_DIR.exists():
                shutil.rmtree(_ML_BACKUP_DIR)
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  Regression detection
# ---------------------------------------------------------------------------

def detect_regression(
    before_winner: float, before_spread_err: float,
    before_total10: float, before_total_err: float,
    after_winner: float, after_spread_err: float,
    after_total10: float, after_total_err: float,
) -> Tuple[bool, List[str], float, float]:
    """Check if the after metrics are worse than before.

    Returns:
        (is_regression, reasons, score_before, score_after)
    """
    s_before = composite_score(before_winner, before_spread_err, before_total10, before_total_err)
    s_after = composite_score(after_winner, after_spread_err, after_total10, after_total_err)

    reasons: List[str] = []

    # Check individual metrics
    if after_winner < before_winner - 0.05:
        reasons.append(f"Winner % dropped: {before_winner:.1f}% → {after_winner:.1f}% "
                       f"(Δ{after_winner - before_winner:+.2f}%)")
    if after_spread_err > before_spread_err + 0.01:
        reasons.append(f"Spread error increased: {before_spread_err:.2f} → {after_spread_err:.2f} "
                       f"(Δ{after_spread_err - before_spread_err:+.2f})")
    if after_total10 < before_total10 - 0.05:
        reasons.append(f"Total-in-10 % dropped: {before_total10:.1f}% → {after_total10:.1f}% "
                       f"(Δ{after_total10 - before_total10:+.2f}%)")
    if after_total_err > before_total_err + 0.01:
        reasons.append(f"Total error increased: {before_total_err:.2f} → {after_total_err:.2f} "
                       f"(Δ{after_total_err - before_total_err:+.2f})")

    is_regression = s_after < s_before - REGRESSION_THRESHOLD
    return is_regression, reasons, s_before, s_after


# ---------------------------------------------------------------------------
#  Weight diff computation
# ---------------------------------------------------------------------------

def compute_weight_diffs(before: Dict[str, float],
                         after: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Compute per-parameter diffs between two weight snapshots.

    Returns dict of param → {"before": x, "after": y, "delta": y-x, "pct": %change}
    Only includes parameters that actually changed.
    """
    diffs: Dict[str, Dict[str, float]] = {}
    all_keys = set(before.keys()) | set(after.keys())
    for k in sorted(all_keys):
        bv = before.get(k, 0.0)
        av = after.get(k, 0.0)
        delta = av - bv
        if abs(delta) > 1e-9:
            pct = (delta / bv * 100) if abs(bv) > 1e-9 else 0.0
            diffs[k] = {"before": round(bv, 6), "after": round(av, 6),
                        "delta": round(delta, 6), "pct_change": round(pct, 2)}
    return diffs


# ---------------------------------------------------------------------------
#  Config fingerprinting
# ---------------------------------------------------------------------------

def _config_fingerprint(n_trials: int, team_trials: int,
                        weights: Dict[str, float]) -> str:
    """Create a fingerprint from settings + starting weights."""
    # Round weights to reduce noise
    rounded = {k: round(v, 3) for k, v in sorted(weights.items())}
    payload = json.dumps({"n": n_trials, "t": team_trials, "w": rounded},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
#  History persistence
# ---------------------------------------------------------------------------

def load_history() -> List[Dict]:
    """Load full optimization history."""
    if not _HISTORY_PATH.exists():
        return []
    try:
        with open(_HISTORY_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(records: List[Dict]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)


def append_history(record: OptimizeRunRecord) -> None:
    """Append a run record to history, keeping last 50 entries."""
    history = load_history()
    history.append(asdict(record))
    # Keep last 50
    if len(history) > 50:
        history = history[-50:]
    _save_history(history)


def get_regression_history() -> List[Dict]:
    """Return only records where was_regression is True."""
    return [r for r in load_history() if r.get("was_regression")]


# ---------------------------------------------------------------------------
#  Blacklist persistence
# ---------------------------------------------------------------------------

def load_blacklist() -> List[Dict]:
    """Load the blacklist of configs that caused regressions."""
    if not _BLACKLIST_PATH.exists():
        return []
    try:
        with open(_BLACKLIST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_blacklist(entries: List[Dict]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BLACKLIST_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def add_to_blacklist(fingerprint: str, record: OptimizeRunRecord) -> None:
    """Add a config fingerprint to the blacklist."""
    bl = load_blacklist()
    entry = {
        "fingerprint": fingerprint,
        "timestamp": record.timestamp,
        "n_trials": record.n_trials,
        "team_trials": record.team_trials,
        "score_delta": record.score_delta,
        "regression_reasons": record.regression_reasons,
        "top_weight_diffs": _top_diffs(record.weight_diffs, 10),
    }
    # Don't duplicate
    existing = {e["fingerprint"] for e in bl}
    if fingerprint not in existing:
        bl.append(entry)
        # Keep last 100
        if len(bl) > 100:
            bl = bl[-100:]
        _save_blacklist(bl)
        log.info("[OptGuard] Added fingerprint %s to blacklist (%d entries)",
                 fingerprint, len(bl))


def is_blacklisted(fingerprint: str) -> bool:
    """Check if a config fingerprint is in the blacklist."""
    bl = load_blacklist()
    return any(e["fingerprint"] == fingerprint for e in bl)


def clear_blacklist() -> None:
    """Clear the entire blacklist."""
    _save_blacklist([])
    log.info("[OptGuard] Blacklist cleared")


def _top_diffs(diffs: Dict[str, Dict], n: int) -> List[Dict]:
    """Return the N weight changes with the largest absolute delta."""
    items = [(k, v) for k, v in diffs.items()]
    items.sort(key=lambda x: abs(x[1].get("delta", 0)), reverse=True)
    return [{"param": k, **v} for k, v in items[:n]]


# ---------------------------------------------------------------------------
#  Suggestions engine
# ---------------------------------------------------------------------------

def generate_suggestions(record: OptimizeRunRecord,
                         history: Optional[List[Dict]] = None) -> List[str]:
    """Generate actionable suggestions based on regression analysis.

    Looks at which parameters deviated most and which directions were harmful.
    """
    suggestions: List[str] = []
    if history is None:
        history = load_history()

    # 1. Suggest different trial counts
    regressions = [r for r in history if r.get("was_regression")]
    trial_counts_that_failed = {r.get("n_trials") for r in regressions}

    if record.n_trials in trial_counts_that_failed:
        alt = record.n_trials
        # Try 50% more or 50% fewer
        candidates = [
            max(20, record.n_trials // 2),
            min(2000, int(record.n_trials * 1.5)),
            max(20, record.n_trials + 100),
        ]
        for c in candidates:
            if c not in trial_counts_that_failed:
                alt = c
                break
        suggestions.append(
            f"Try {alt} trials instead of {record.n_trials} "
            f"(n_trials={record.n_trials} has caused regressions before)"
        )

    # 2. Identify parameters that moved in bad directions
    if record.weight_diffs:
        bad_params = []
        for param, diff in record.weight_diffs.items():
            delta = diff.get("delta", 0)
            if abs(delta) > 1e-6:
                bad_params.append((param, diff))

        # Sort by magnitude
        bad_params.sort(key=lambda x: abs(x[1].get("delta", 0)), reverse=True)
        top_bad = bad_params[:5]

        if top_bad:
            suggestions.append("Parameters with largest harmful changes:")
            for param, diff in top_bad:
                bv = diff.get("before", 0)
                av = diff.get("after", 0)
                d = diff.get("delta", 0)
                pct = diff.get("pct_change", 0)
                direction = "↑" if d > 0 else "↓"
                suggestions.append(
                    f"  • {param}: {bv:.4f} → {av:.4f} ({direction}{abs(d):.4f}, {pct:+.1f}%)"
                )

        # 3. Suggest reverting specific params
        if top_bad:
            names = [p[0] for p in top_bad[:3]]
            suggestions.append(
                f"Consider locking these parameters: {', '.join(names)}"
            )

    # 4. If multiple recent regressions, suggest force_rerun
    recent_regressions = [r for r in regressions[-5:]]
    if len(recent_regressions) >= 2 and not record.force_rerun:
        suggestions.append(
            "Multiple recent regressions detected — try enabling 'Force re-run' "
            "to bypass caches and start fresh"
        )

    # 5. If no suggestions yet, generic advice
    if not suggestions:
        suggestions.append(
            "Regression detected but no clear pattern found. "
            "Try adjusting trial count or enabling force re-run."
        )

    return suggestions


def suggest_adjusted_params(record: OptimizeRunRecord) -> Dict[str, float]:
    """Suggest weight values that undo the most harmful changes.

    Returns a dict of param → suggested_value.  Each suggestion reverses
    the biggest deltas by moving them back halfway toward the 'before' value.
    """
    adjusted: Dict[str, float] = {}
    if not record.weight_diffs:
        return adjusted

    items = sorted(record.weight_diffs.items(),
                   key=lambda x: abs(x[1].get("delta", 0)), reverse=True)

    for param, diff in items[:10]:
        bv = diff.get("before", 0)
        av = diff.get("after", 0)
        # Suggest the midpoint between before and after (conservative adjustment)
        midpoint = (bv + av) / 2.0
        adjusted[param] = round(midpoint, 6)

    return adjusted


def apply_suggested_params(params: Dict[str, float]) -> None:
    """Apply a set of suggested parameter overrides to the current weights."""
    from src.analytics.weight_config import (
        WeightConfig,
        load_weights,
        save_weights,
        set_weight_config,
    )

    cfg = load_weights()
    d = cfg.to_dict()
    applied = []
    for k, v in params.items():
        if k in d:
            old = d[k]
            d[k] = v
            applied.append(f"{k}: {old:.4f} → {v:.4f}")

    new_cfg = WeightConfig.from_dict(d)
    save_weights(new_cfg)
    set_weight_config(new_cfg)
    log.info("[OptGuard] Applied %d suggested parameters: %s",
             len(applied), "; ".join(applied))


# ---------------------------------------------------------------------------
#  Full evaluation flow (called after pipeline finishes)
# ---------------------------------------------------------------------------

def evaluate_run(
    before_snap,  # Snapshot from optimize_view
    after_snap,   # Snapshot from optimize_view
    backup: WeightBackup,
    n_trials: int,
    team_trials: int,
    force_rerun: bool,
) -> OptimizeRunRecord:
    """Evaluate a completed optimization run for regression.

    If regression detected:
      - Restores weights from backup
      - Adds config to blacklist
      - Logs full variable dump

    Returns the full run record with regression info.
    """
    from src.analytics.weight_config import load_weights

    record = OptimizeRunRecord(
        timestamp=time.time(),
        n_trials=n_trials,
        team_trials=team_trials,
        force_rerun=force_rerun,
    )

    # Metrics from snapshots
    if before_snap:
        record.before_winner_pct = before_snap.winner_pct
        record.before_avg_spread_error = before_snap.avg_spread_error
        record.before_total_in_10_pct = before_snap.total_in_10_pct
        record.before_avg_total_error = before_snap.avg_total_error

    if after_snap:
        record.after_winner_pct = after_snap.winner_pct
        record.after_avg_spread_error = after_snap.avg_spread_error
        record.after_total_in_10_pct = after_snap.total_in_10_pct
        record.after_avg_total_error = after_snap.avg_total_error

    # Deltas
    record.delta_winner_pct = record.after_winner_pct - record.before_winner_pct
    record.delta_avg_spread_error = record.after_avg_spread_error - record.before_avg_spread_error
    record.delta_total_in_10_pct = record.after_total_in_10_pct - record.before_total_in_10_pct
    record.delta_avg_total_error = record.after_avg_total_error - record.before_avg_total_error

    # Weight snapshots
    record.weights_before = copy.deepcopy(backup.global_weights)
    after_cfg = load_weights()
    record.weights_after = after_cfg.to_dict()
    record.weight_diffs = compute_weight_diffs(record.weights_before, record.weights_after)

    # Fingerprint
    record.config_fingerprint = _config_fingerprint(n_trials, team_trials,
                                                     record.weights_before)

    # Regression detection
    is_reg, reasons, s_before, s_after = detect_regression(
        record.before_winner_pct, record.before_avg_spread_error,
        record.before_total_in_10_pct, record.before_avg_total_error,
        record.after_winner_pct, record.after_avg_spread_error,
        record.after_total_in_10_pct, record.after_avg_total_error,
    )
    record.score_before = round(s_before, 4)
    record.score_after = round(s_after, 4)
    record.score_delta = round(s_after - s_before, 4)
    record.was_regression = is_reg
    record.regression_reasons = reasons

    if is_reg:
        # ── REGRESSION DETECTED ──
        log.warning("=" * 70)
        log.warning("[OptGuard] ⚠️  REGRESSION DETECTED — ROLLING BACK")
        log.warning("=" * 70)
        log.warning("[OptGuard] Composite score: %.2f → %.2f (Δ%.2f)",
                    s_before, s_after, s_after - s_before)
        for reason in reasons:
            log.warning("[OptGuard]   • %s", reason)

        # Log full weight diff
        log.warning("[OptGuard] Weight changes that caused regression:")
        sorted_diffs = sorted(record.weight_diffs.items(),
                              key=lambda x: abs(x[1].get("delta", 0)), reverse=True)
        for param, diff in sorted_diffs:
            log.warning("[OptGuard]   %s: %.6f → %.6f (Δ%+.6f, %+.1f%%)",
                        param, diff["before"], diff["after"],
                        diff["delta"], diff.get("pct_change", 0))

        # Auto-rollback
        backup.restore()
        record.was_rolled_back = True

        # Add to blacklist
        add_to_blacklist(record.config_fingerprint, record)

        log.warning("[OptGuard] Rollback complete. Blacklisted fingerprint: %s",
                    record.config_fingerprint)
    else:
        # Successful run — clean up backup
        log.info("[OptGuard] ✓ Optimization improved results: score %.2f → %.2f (Δ%+.2f)",
                 s_before, s_after, s_after - s_before)
        backup.cleanup()

    # Save to history
    append_history(record)

    return record


# ---------------------------------------------------------------------------
#  Pre-run blacklist check
# ---------------------------------------------------------------------------

def check_blacklist_before_run(
    n_trials: int,
    team_trials: int,
) -> Tuple[bool, List[str], Optional[Dict[str, float]]]:
    """Check if the planned run matches a blacklisted configuration.

    Returns:
        (is_blacklisted, warnings, suggested_params_or_None)
    """
    from src.analytics.weight_config import load_weights

    current = load_weights().to_dict()
    fp = _config_fingerprint(n_trials, team_trials, current)

    if not is_blacklisted(fp):
        return False, [], None

    # This exact config caused a regression before
    warnings = [
        f"⚠️ This configuration (n_trials={n_trials}, team_trials={team_trials}) "
        f"with current weights previously caused a regression!",
        f"Fingerprint: {fp}",
    ]

    # Try to find what went wrong from history
    history = load_history()
    for rec in reversed(history):
        if rec.get("config_fingerprint") == fp and rec.get("was_regression"):
            if rec.get("regression_reasons"):
                warnings.append("Previous regression reasons:")
                for r in rec["regression_reasons"]:
                    warnings.append(f"  • {r}")
            break

    # Generate suggestions
    suggestions = _suggest_alternative_trials(n_trials, team_trials)
    if suggestions:
        warnings.extend(suggestions)

    return True, warnings, None


def _suggest_alternative_trials(n_trials: int,
                                team_trials: int) -> List[str]:
    """Suggest trial count alternatives that aren't blacklisted."""
    from src.analytics.weight_config import load_weights
    current = load_weights().to_dict()

    candidates = [
        (max(20, n_trials // 2), team_trials),
        (min(2000, int(n_trials * 1.5)), team_trials),
        (n_trials + 100, team_trials),
        (n_trials, max(20, team_trials // 2)),
        (n_trials, min(1000, int(team_trials * 1.5))),
    ]

    suggestions = []
    for nt, tt in candidates:
        fp = _config_fingerprint(nt, tt, current)
        if not is_blacklisted(fp):
            suggestions.append(
                f"  → Try n_trials={nt}, team_trials={tt} (not previously blacklisted)"
            )
            break  # just suggest one

    return suggestions


# ---------------------------------------------------------------------------
#  Utility: format a run record for display
# ---------------------------------------------------------------------------

def format_run_summary(record: OptimizeRunRecord) -> str:
    """Human-readable summary of a run for the log pane."""
    lines = []
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(record.timestamp))
    lines.append(f"Run @ {ts}  |  trials={record.n_trials}")
    lines.append(f"  Score: {record.score_before:.2f} → {record.score_after:.2f} "
                 f"(Δ{record.score_delta:+.2f})")
    lines.append(f"  Winner%: {record.before_winner_pct:.1f} → {record.after_winner_pct:.1f} "
                 f"(Δ{record.delta_winner_pct:+.1f})")
    lines.append(f"  Spread Err: {record.before_avg_spread_error:.2f} → "
                 f"{record.after_avg_spread_error:.2f} "
                 f"(Δ{record.delta_avg_spread_error:+.2f})")
    if record.was_regression:
        lines.append("  ⚠️  REGRESSION — rolled back: YES" if record.was_rolled_back
                      else "  ⚠️  REGRESSION — rolled back: NO")
        if record.regression_reasons:
            for r in record.regression_reasons:
                lines.append(f"    • {r}")
    else:
        lines.append("  ✓ Improvement (kept)")
    return "\n".join(lines)
