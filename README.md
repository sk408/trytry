# NBA Game Prediction System

A PySide6 desktop application and analytics pipeline for predicting NBA game outcomes using a multi-factor weighted model with ML ensemble support, autotune corrections, and vectorized evaluation.

**Current metrics** (Round 10 — coordinate descent): Winner Accuracy **66.8%**, Spread MAE **11.95**

---

## Quick Start

1. **Install Python 3.8+** (recommended: 3.10+)
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   - If you are on Android/Termux and `xgboost` fails, you can comment it out in `requirements.txt` unless you want to use ML ensemble tuning.
3. **Run the web app:**
   ```sh
   python -m src.web.app
   ```
   - Or, for desktop: `python desktop.py`

---

## Mobile/Termux Notes
- XGBoost is only required for ML ensemble tuning and sensitivity sweeps. If you only want to run predictions, you can remove/comment `xgboost` from `requirements.txt`.
- To run on Termux:
   1. Install Python: `pkg install python`
   2. Install pip: `python -m ensurepip`
   3. Install dependencies: `pip install -r requirements.txt` (comment out `xgboost` if it fails)
   4. Run: `python -m src.web.app`

---

## Sensitivity Analysis (Parameter Sweeps)

The sensitivity tool sweeps individual prediction weights through extreme ranges to discover hidden value outside conventional optimizer bounds — like a dyno "power curve" for prediction tuning.

### Run a full sweep of all parameters

```bash
python -m src.analytics.sensitivity --all --steps 200 --chart
```

This sweeps every float weight in `WeightConfig` across its extreme range (defined in `EXTREME_RANGES`), producing:

- **CSV files** in `data/sensitivity/` — one per parameter with columns: `param_value`, `spread_mae`, `winner_pct`, `loss`
- **ASCII charts** printed to the terminal for each metric (MAE, Winner%, Loss)
- A **summary.txt** with the best values found for each parameter

### Sweep a single parameter

```bash
# Sweep with default extreme range
python -m src.analytics.sensitivity --param rating_matchup_mult --steps 200 --chart

# Sweep with custom range
python -m src.analytics.sensitivity --param pace_mult --min -10 --max 20 --steps 100 --chart
```

### CLI Reference

| Flag | Description |
|------|-------------|
| `--param NAME` | Sweep a single parameter by name |
| `--param2 NAME` | Second parameter for pairwise sweep |
| `--all` | Sweep all sweepable parameters |
| `--steps N` | Number of steps in the sweep (default: 200) |
| `--min X` | Override sweep minimum value |
| `--max X` | Override sweep maximum value |
| `--chart` | Print ASCII charts to terminal |
| `--show` | Show optimal values from existing CSVs (no re-run) |
| `--apply` | Apply optimal values to the pipeline (auto-saves snapshot) |
| `--metric M` | Metric for --show/--apply: `loss` (default), `spread_mae`, `winner_pct` |
| `--descent` | Run coordinate descent: iteratively sweep and lock in best values |
| `--descent-rounds N` | Max rounds for coordinate descent (default: 10) |
| `--pairwise` | Run pairwise interaction analysis on active parameters |

### Show optimal values from existing sweeps

```bash
python -m src.analytics.sensitivity --show --metric loss
```

### Apply optimal values to the pipeline

This reads the sweep CSVs, finds the best value for each active parameter, saves a snapshot of the current state (for easy rollback), and updates the weights in the database:

```bash
python -m src.analytics.sensitivity --apply --metric loss
```

To rollback after applying:
```python
from src.analytics.weight_config import restore_snapshot
restore_snapshot("data/snapshots/<snapshot_file>.json")
```

### List available parameters

```bash
python -m src.analytics.sensitivity
```

Prints all parameter names that can be swept.

### Output files

All output goes to `data/sensitivity/`:

```
data/sensitivity/
  rating_matchup_mult_sweep.csv
  pace_mult_sweep.csv
  ...
  full_sweep_summary.txt
```

### Interpreting results

- **spread_mae** — Mean Absolute Error of predicted spread vs actual. Lower is better.
- **winner_pct** — Percentage of games where the predicted winner was correct. Higher is better.
- **loss** — Combined loss metric used by the optimizer. Lower is better.
- If the optimal value falls **outside** the current `OPTIMIZER_RANGES`, the tool flags it with a warning. This means expanding the optimizer's search bounds may improve results.

---

## Coordinate Descent (Multi-Parameter Optimization)

Instead of sweeping each parameter independently, coordinate descent iteratively sweeps and locks in the best value for each parameter, then re-sweeps from the new baseline. This captures parameter interactions that single-parameter sweeps miss.

```bash
# Run 5 rounds of coordinate descent with 50 steps per param
python -m src.analytics.sensitivity --descent --steps 50 --descent-rounds 5

# Run descent and immediately apply the results
python -m src.analytics.sensitivity --descent --steps 50 --descent-rounds 5 --apply
```

The descent automatically excludes parameters not used by the vectorized evaluator (ML/ESPN/fatigue sub-penalties).

---

## Pairwise Interaction Analysis

Test two parameters simultaneously on a 2D grid to detect synergy or conflict between them.

```bash
# Analyze a single pair with ASCII heatmap
python -m src.analytics.sensitivity --pairwise --param rating_matchup_mult --param2 four_factors_scale --steps 40 --chart

# Analyze ALL pairs of active parameters (generates CSVs + summary)
python -m src.analytics.sensitivity --pairwise --steps 40
```

Output includes interaction scores: **SYNERGY** (joint optimum better than individual optima), **CONFLICT** (worse), or **NEUTRAL** (no interaction).

---

## Regression Testing

Save prediction baselines and compare after pipeline changes to detect regressions. This ensures changes to the prediction engine (weight tuning, bug fixes, ML retraining) don't silently degrade accuracy.

### Save a baseline before making changes

```bash
python -m src.analytics.regression_test save "before_ml_retrain"
```

This runs a full backtest and saves the results (metrics, per-team breakdowns, bias data) to `data/regression_baselines/before_ml_retrain.json`.

### Compare current predictions against a baseline

```bash
python -m src.analytics.regression_test compare "before_ml_retrain"
```

Runs a fresh backtest and compares against the saved baseline. Reports:

- **Metric deltas** — winner%, spread MAE, total MAE, spread-within-5, total-within-10
- **Regressions** — flagged when >1% worse (percentages) or >0.5 worse (MAE)
- **Improvements** — flagged when >0.5% better (percentages) or >0.2 better (MAE)
- **Per-team regressions/improvements** — teams with >10% winner change or >2.0 MAE change
- **Exit code** — 0 if passed (no regressions), 1 if failed

### List all saved baselines

```bash
python -m src.analytics.regression_test list
```

Output:

```
Name                           Date                 Games Winner%  Spread MAE
--------------------------------------------------------------------------------
before_ml_retrain              2026-02-20 14:30:00     412   66.8%      11.95
after_ff_fix                   2026-02-21 09:15:00     412   67.2%      11.82
```

### Run feature extraction sanity tests

```bash
python -m src.analytics.regression_test test-features
```

Fast check (no backtest needed) that verifies ML feature extraction produces non-zero values for:

- Four Factors edges (efg, tov, oreb, fta)
- Injury features (injured count, PPG lost, minutes lost)
- Counting stats (points, rebounds, assists)
- Ratings (offensive, defensive)

### Web API

The regression testing tools are also available via the web interface:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/regression/save?name=baseline` | SSE — save a baseline |
| GET | `/api/regression/compare?name=baseline` | SSE — compare against baseline |
| GET | `/api/regression/list` | JSON — list all baselines |
| GET | `/api/regression/test-features` | JSON — run feature extraction tests |

### Python API

```python
from src.analytics.regression_test import save_baseline, compare_to_baseline, list_baselines, test_feature_extraction

# Save baseline (with optional progress callback)
baseline = save_baseline("my_baseline", callback=print)

# Compare (returns dict with 'passed' bool, 'regressions', 'improvements')
result = compare_to_baseline("my_baseline", callback=print)
if not result["passed"]:
    print("Regressions detected!")

# Quick feature extraction check
result = test_feature_extraction()
print(f"{sum(1 for t in result['tests'] if t['passed'])}/{len(result['tests'])} passed")
```

### Recommended workflow

```bash
# 1. Save baseline BEFORE changes
python -m src.analytics.regression_test save "pre_change"

# 2. Make your changes (retrain ML, tune weights, fix bugs, etc.)

# 3. Compare
python -m src.analytics.regression_test compare "pre_change"

# 4. If regressions detected, revert and investigate
```

---

## Pipeline Snapshots

Save and restore the full prediction pipeline state (weights + autotune corrections + optimizer ranges) for safe experimentation.

### Python API

```python
from src.analytics.weight_config import save_snapshot, restore_snapshot, list_snapshots

# Save current state before experimenting
path = save_snapshot("before_experiment", notes="Baseline Round 9", metrics={"mae": 12.03, "winner_pct": 62.8})

# List all saved snapshots
for snap in list_snapshots():
    print(f"{snap['name']}  {snap['created_at']}  {snap['metrics']}")

# Restore a previous snapshot
restore_snapshot(path)
```

Snapshots are stored as JSON in `data/snapshots/`.

---

## Project Structure

```
src/
  analytics/
    prediction.py         # Core prediction pipeline (11-step)
    ml_model.py           # XGBoost spread/total models, SHAP, 88 features
    weight_config.py      # WeightConfig dataclass, persistence, snapshots
    weight_optimizer.py   # Vectorized optimizer with Optuna TPE
    sensitivity.py        # Sensitivity sweep tool (CLI)
    autotune.py           # Team-level autotune corrections
    backtester.py         # Historical game replay and metrics
    regression_test.py    # Prediction regression testing (baselines + comparison)
    live_prediction.py    # 3-signal in-game prediction blend
    stats_engine.py       # Player projections, 50/25/25 blend, 240-min budget
  data/
    nba_fetcher.py        # NBA API data fetcher
    injury_scraper.py     # ESPN/CBS/RotoWire scraping chain
    sync_service.py       # Full sync orchestrator (12 steps)
    gamecast.py           # ESPN live integration
    image_cache.py        # Player photo and team logo caching
  database/
    db.py                 # SQLite connection, WAL mode, RWLock
    migrations.py         # Schema creation and column migrations
  ui/
    views/                # PySide6 views (10 tabs)
  web/
    app.py                # FastAPI — 30+ routes, 18+ SSE endpoints
    player_utils.py       # Player stat utilities
    static/
      style.css           # Mobile-first dark theme
    templates/            # Jinja2 templates (10 pages)
data/
  regression_baselines/   # Saved prediction baselines (git-ignored)
  sensitivity/            # Sweep CSV output (git-ignored)
  snapshots/              # Pipeline snapshots (git-ignored)
  ml_models/              # Trained XGBoost models (git-ignored)
  cache/                  # Player photos and team logos
```
