"""Prediction regression testing — snapshot baselines and comparison.

Usage:
    # Save a golden baseline:
    python -m src.analytics.regression_test save "before_ff_fix"

    # Compare current predictions against the baseline:
    python -m src.analytics.regression_test compare "before_ff_fix"

    # List all baselines:
    python -m src.analytics.regression_test list
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

BASELINE_DIR = os.path.join("data", "regression_baselines")


def _ensure_dir():
    os.makedirs(BASELINE_DIR, exist_ok=True)


def _baseline_path(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return os.path.join(BASELINE_DIR, f"{safe}.json")


def save_baseline(name: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Run a full backtest and save results as a named baseline.

    Returns the saved baseline dict.
    """
    from src.analytics.backtester import run_backtest

    _ensure_dir()

    if callback:
        callback(f"Running backtest for baseline '{name}'...")

    results = run_backtest(use_cache=False, callback=callback)

    if results.get("error"):
        if callback:
            callback(f"Backtest failed: {results['error']}")
        return results

    # Extract key metrics (summary only — not per-game for size)
    baseline = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "total_games": results["total_games"],
        "metrics": {
            "winner_pct": results["overall_spread_accuracy"],
            "spread_mae": results["avg_spread_error"],
            "total_mae": results["avg_total_error"],
            "spread_within_5_pct": results["spread_within_5_pct"],
            "total_within_10_pct": results["total_within_10_pct"],
        },
        "bias": results.get("bias", {}),
        "home_away": results.get("home_away", {}),
        "spread_ranges": results.get("spread_ranges", []),
        "total_ranges": results.get("total_ranges", []),
        "per_team": results.get("per_team", {}),
    }

    path = _baseline_path(name)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)

    if callback:
        callback(f"Baseline saved: {path}")
        callback(f"  Winner%: {baseline['metrics']['winner_pct']:.1f}%")
        callback(f"  Spread MAE: {baseline['metrics']['spread_mae']:.2f}")
        callback(f"  Total MAE: {baseline['metrics']['total_mae']:.2f}")

    return baseline


def load_baseline(name: str) -> Optional[Dict[str, Any]]:
    """Load a previously saved baseline."""
    path = _baseline_path(name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compare_to_baseline(name: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Run current backtest and compare against a saved baseline.

    Returns a comparison dict with deltas and regression flags.
    """
    from src.analytics.backtester import run_backtest

    baseline = load_baseline(name)
    if baseline is None:
        msg = f"No baseline found with name '{name}'"
        if callback:
            callback(msg)
        return {"error": msg}

    if callback:
        callback(f"Loaded baseline '{name}' from {baseline['created_at']}")
        callback("Running current backtest...")

    current = run_backtest(use_cache=False, callback=callback)
    if current.get("error"):
        return {"error": current["error"]}

    current_metrics = {
        "winner_pct": current["overall_spread_accuracy"],
        "spread_mae": current["avg_spread_error"],
        "total_mae": current["avg_total_error"],
        "spread_within_5_pct": current["spread_within_5_pct"],
        "total_within_10_pct": current["total_within_10_pct"],
    }

    base_metrics = baseline["metrics"]

    # Compute deltas (positive = improvement for winner%, within_5, within_10)
    # (negative = improvement for MAE)
    deltas = {}
    regressions = []
    improvements = []

    # Higher is better
    for key in ["winner_pct", "spread_within_5_pct", "total_within_10_pct"]:
        delta = current_metrics[key] - base_metrics[key]
        deltas[key] = round(delta, 2)
        if delta < -1.0:  # More than 1% worse
            regressions.append(f"{key}: {base_metrics[key]:.1f}% -> {current_metrics[key]:.1f}% ({delta:+.1f}%)")
        elif delta > 0.5:
            improvements.append(f"{key}: {base_metrics[key]:.1f}% -> {current_metrics[key]:.1f}% ({delta:+.1f}%)")

    # Lower is better
    for key in ["spread_mae", "total_mae"]:
        delta = current_metrics[key] - base_metrics[key]
        deltas[key] = round(delta, 2)
        if delta > 0.5:  # More than 0.5 worse
            regressions.append(f"{key}: {base_metrics[key]:.2f} -> {current_metrics[key]:.2f} ({delta:+.2f})")
        elif delta < -0.2:
            improvements.append(f"{key}: {base_metrics[key]:.2f} -> {current_metrics[key]:.2f} ({delta:+.2f})")

    # Per-team comparison
    team_regressions = []
    team_improvements = []
    base_teams = baseline.get("per_team", {})
    current_teams = current.get("per_team", {})
    for team in sorted(set(base_teams.keys()) & set(current_teams.keys())):
        bt = base_teams[team]
        ct = current_teams[team]
        w_delta = ct["winner_pct"] - bt["winner_pct"]
        s_delta = ct["avg_spread_error"] - bt["avg_spread_error"]
        if w_delta < -10.0 or s_delta > 2.0:
            team_regressions.append({
                "team": team,
                "winner_delta": round(w_delta, 1),
                "spread_mae_delta": round(s_delta, 2),
            })
        elif w_delta > 5.0 or s_delta < -1.0:
            team_improvements.append({
                "team": team,
                "winner_delta": round(w_delta, 1),
                "spread_mae_delta": round(s_delta, 2),
            })

    result = {
        "baseline_name": name,
        "baseline_date": baseline["created_at"],
        "baseline_games": baseline["total_games"],
        "current_games": current["total_games"],
        "baseline_metrics": base_metrics,
        "current_metrics": current_metrics,
        "deltas": deltas,
        "regressions": regressions,
        "improvements": improvements,
        "team_regressions": team_regressions,
        "team_improvements": team_improvements,
        "passed": len(regressions) == 0,
    }

    if callback:
        callback("")
        callback("=" * 60)
        callback(f"REGRESSION TEST: {'PASSED' if result['passed'] else 'FAILED'}")
        callback("=" * 60)
        callback(f"Baseline: {name} ({baseline['created_at'][:10]})")
        callback(f"Games: {baseline['total_games']} -> {current['total_games']}")
        callback("")
        callback("Metric Comparison:")
        for key in ["winner_pct", "spread_mae", "total_mae", "spread_within_5_pct", "total_within_10_pct"]:
            arrow = "+" if deltas[key] > 0 else ""
            callback(f"  {key:>25s}: {base_metrics[key]:>8.2f} -> {current_metrics[key]:>8.2f}  ({arrow}{deltas[key]:.2f})")
        if regressions:
            callback("")
            callback("REGRESSIONS:")
            for r in regressions:
                callback(f"  [!] {r}")
        if improvements:
            callback("")
            callback("IMPROVEMENTS:")
            for imp in improvements:
                callback(f"  [+] {imp}")
        if team_regressions:
            callback("")
            callback(f"TEAM REGRESSIONS ({len(team_regressions)}):")
            for tr in team_regressions[:10]:
                callback(f"  {tr['team']}: winner {tr['winner_delta']:+.1f}%, spread MAE {tr['spread_mae_delta']:+.2f}")
        if team_improvements:
            callback("")
            callback(f"TEAM IMPROVEMENTS ({len(team_improvements)}):")
            for ti in team_improvements[:10]:
                callback(f"  {ti['team']}: winner {ti['winner_delta']:+.1f}%, spread MAE {ti['spread_mae_delta']:+.2f}")

    return result


def list_baselines() -> list:
    """List all saved baselines."""
    if not os.path.isdir(BASELINE_DIR):
        return []
    results = []
    for f in sorted(os.listdir(BASELINE_DIR)):
        if f.endswith(".json"):
            try:
                path = os.path.join(BASELINE_DIR, f)
                with open(path) as fh:
                    data = json.load(fh)
                results.append({
                    "name": data.get("name", f),
                    "created_at": data.get("created_at", ""),
                    "total_games": data.get("total_games", 0),
                    "metrics": data.get("metrics", {}),
                })
            except Exception:
                continue
    return results


# Unit tests for feature extraction (no backtest needed)
def test_feature_extraction() -> Dict[str, Any]:
    """Verify ML feature extraction produces non-zero Four Factors and injury features.

    This is a fast sanity check that doesn't require running a full backtest.
    Returns dict with test results.
    """
    from src.analytics.prediction import PrecomputedGame
    from src.analytics.ml_model import _extract_features_from_precomputed

    results = {"tests": [], "passed": True}

    # Create a test game with known non-zero Four Factors
    game = PrecomputedGame(
        game_date="2025-12-01",
        home_team_id=1,
        away_team_id=2,
        actual_home_score=110,
        actual_away_score=105,
        home_proj={"points": 110, "rebounds": 44, "assists": 25, "steals": 8,
                   "blocks": 5, "turnovers": 13, "oreb": 10, "dreb": 34,
                   "fga": 88, "fgm": 42, "fg3a": 35, "fg3m": 13,
                   "fta": 22, "ftm": 18},
        away_proj={"points": 105, "rebounds": 42, "assists": 23, "steals": 7,
                   "blocks": 4, "turnovers": 14, "oreb": 9, "dreb": 33,
                   "fga": 86, "fgm": 40, "fg3a": 33, "fg3m": 11,
                   "fta": 20, "ftm": 16},
        home_court=3.0,
        home_off=112.0,
        away_off=108.0,
        home_def=108.0,
        away_def=110.0,
        home_pace=100.0,
        away_pace=98.0,
        home_ff={"efg": 0.55, "tov": 0.12, "oreb": 0.28, "fta": 0.25},
        away_ff={"efg": 0.52, "tov": 0.14, "oreb": 0.26, "fta": 0.23},
        home_clutch={"net_rating": 5.0, "efg_pct": 0.53},
        away_clutch={"net_rating": 2.0, "efg_pct": 0.50},
        home_hustle={"deflections": 4.5, "contested": 30.0, "loose_balls": 3.0},
        away_hustle={"deflections": 4.0, "contested": 28.0, "loose_balls": 2.5},
        home_injured_count=2,
        away_injured_count=1,
        home_injury_ppg_lost=15.5,
        away_injury_ppg_lost=8.2,
        home_injury_minutes_lost=30.0,
        away_injury_minutes_lost=20.0,
        home_games_played=40,
        away_games_played=38,
    )

    features = _extract_features_from_precomputed(game)

    if features is None:
        results["tests"].append({"name": "feature_extraction", "passed": False, "error": "returned None"})
        results["passed"] = False
        return results

    # Test 1: Four Factors edges should be non-zero
    ff_keys = ["ff_efg_edge", "ff_tov_edge", "ff_oreb_edge", "ff_fta_edge"]
    for key in ff_keys:
        val = features.get(key, 0)
        passed = val != 0
        results["tests"].append({
            "name": f"four_factors_{key}",
            "passed": passed,
            "expected": "non-zero",
            "actual": val,
        })
        if not passed:
            results["passed"] = False

    # Test 2: Injury features should be non-zero
    injury_keys = [
        "home_injured_count", "away_injured_count",
        "home_injury_ppg_lost", "away_injury_ppg_lost",
        "home_injury_minutes_lost", "away_injury_minutes_lost",
    ]
    for key in injury_keys:
        val = features.get(key, 0)
        passed = val != 0
        results["tests"].append({
            "name": f"injury_{key}",
            "passed": passed,
            "expected": "non-zero",
            "actual": val,
        })
        if not passed:
            results["passed"] = False

    # Test 3: Counting stats should be populated
    stat_keys = ["home_points", "away_points", "home_rebounds", "home_assists"]
    for key in stat_keys:
        val = features.get(key, 0)
        passed = val > 0
        results["tests"].append({
            "name": f"stats_{key}",
            "passed": passed,
            "expected": "> 0",
            "actual": val,
        })
        if not passed:
            results["passed"] = False

    # Test 4: Ratings should be non-default
    rating_keys = ["home_off_rating", "away_off_rating", "home_def_rating", "away_def_rating"]
    for key in rating_keys:
        val = features.get(key, 0)
        passed = val > 0
        results["tests"].append({
            "name": f"ratings_{key}",
            "passed": passed,
            "expected": "> 0",
            "actual": val,
        })
        if not passed:
            results["passed"] = False

    return results


# CLI interface
if __name__ == "__main__":
    from src.database.migrations import init_db
    init_db()

    def _print(msg):
        print(msg)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.analytics.regression_test save <name>")
        print("  python -m src.analytics.regression_test compare <name>")
        print("  python -m src.analytics.regression_test list")
        print("  python -m src.analytics.regression_test test-features")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "save":
        name = sys.argv[2] if len(sys.argv) > 2 else f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_baseline(name, callback=_print)

    elif cmd == "compare":
        if len(sys.argv) < 3:
            print("Usage: python -m src.analytics.regression_test compare <name>")
            sys.exit(1)
        name = sys.argv[2]
        result = compare_to_baseline(name, callback=_print)
        sys.exit(0 if result.get("passed") else 1)

    elif cmd == "list":
        baselines = list_baselines()
        if not baselines:
            print("No baselines saved yet.")
        else:
            print(f"{'Name':<30s} {'Date':<20s} {'Games':>6s} {'Winner%':>8s} {'Spread MAE':>11s}")
            print("-" * 80)
            for b in baselines:
                m = b.get("metrics", {})
                print(f"{b['name']:<30s} {b['created_at'][:19]:<20s} "
                      f"{b['total_games']:>6d} {m.get('winner_pct', 0):>7.1f}% "
                      f"{m.get('spread_mae', 0):>10.2f}")

    elif cmd == "test-features":
        result = test_feature_extraction()
        total = len(result["tests"])
        passed = sum(1 for t in result["tests"] if t["passed"])
        print(f"\nFeature Extraction Tests: {passed}/{total} passed")
        for t in result["tests"]:
            status = "PASS" if t["passed"] else "FAIL"
            print(f"  [{status}] {t['name']}: expected {t.get('expected', '?')}, got {t.get('actual', '?')}")
        sys.exit(0 if result["passed"] else 1)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
