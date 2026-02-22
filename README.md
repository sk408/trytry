# NBA Game Prediction System

A PySide6 desktop application and analytics pipeline for predicting NBA game outcomes using a multi-factor weighted model with ML ensemble support, autotune corrections, and vectorized evaluation.

**Current metrics** (Round 10 — coordinate descent): Winner Accuracy **66.8%**, Spread MAE **11.95**

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the desktop app
python desktop.py
```

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
    prediction.py         # Core prediction pipeline
    weight_config.py      # WeightConfig dataclass, persistence, snapshots
    weight_optimizer.py   # Vectorized optimizer with grid search
    sensitivity.py        # Sensitivity sweep tool (CLI)
    autotune.py           # Team-level autotune corrections
  database/
    db.py                 # SQLite connection, WAL mode, RWLock
    migrations.py         # Schema creation and column migrations
  ui/
    views/                # PySide6 views (matchup, standings, etc.)
  web/
    nba_fetcher.py        # NBA API data fetcher
    player_utils.py       # Player stat utilities
data/
  sensitivity/            # Sweep CSV output (git-ignored)
  snapshots/              # Pipeline snapshots (git-ignored)
```
