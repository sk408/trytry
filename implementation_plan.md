# Phase 1: Bug Fixes & Edge Cases

This phase focuses on fixing the specific logic bugs identified in the code review to stabilize the existing system before adding new features.

## Proposed Changes

### Configuration Utilities

#### [NEW] team_mapper.py
- Extract all ESPN and ActionNetwork abbreviation hard-codings into a single utility file `src/utils/team_mapper.py`.
- Introduce functions `normalize_espn_abbr(abbr)` and `normalize_action_abbr(abbr)`.

---

### Data Layer

#### gamecast.py
- Remove the inline `ESPN_TO_NBA_ABBR` dict and use `src/utils/team_mapper.py`.
- **WebSocket Reconnection Spam**: Update `FastcastWebSocket._run()` to implement exponential backoff instead of a static 5-second sleep when the connection fails.
    - Start at 2s sleep, double each retry, cap at 60s.

#### odds_sync.py
- Remove `_map_action_abbrev` and import from `src/utils/team_mapper.py`.

---

### Analytics & Prediction Layer

#### backtester.py
- **Inaccurate Dog ROI**: 
  - The current underdog ROI calculation incorrectly assumes a fixed `+150` payout (`1.5x`) for all underdog picks.
  - *Change*: Modify `compute_progression/dog_metrics` to calculate the actual moneyline implied payout.
  - Formula: If `vegas_ml > 0`, payout is `vegas_ml / 100`. If `vegas_ml < 0`, payout is `100 / abs(vegas_ml)`. Use this per-game to accumulate true historical underdog profit.

#### stats_engine.py
- **0-Minute Player Fallback**:
  - In `aggregate_projection()`, `if splits["minutes"] < 1.0: continue` drops players returning from long absences who have no recent minutes in their 10-game splits window.
  - *Change*: If a player is marked to play `play_prob >= 0.3` but has `< 1.0` minutes in their recent split, fall back to checking their *full season averages* from the DB. If they still have no stats, assign a replacement level baseline instead of dropping them entirely.

## Verification Plan

### Automated/Unit Tests
- **No new test suites** are added in Phase 1 (that's Phase 2), so we will run the backtester manually across a recent span to ensure the pipeline isn't broken.
- **Commands**: 
  - `python -c "from src.analytics.backtester import run_backtest; run_backtest(use_cache=False)"` (Check that dog ROI output looks realistic and doesn't crash).

### Manual Verification
- Verify `GamecastWebSocket` properly connects by opening the GUI `Gamecast` tab and inspecting logs.
- Trigger `odds_sync` via the FastAPI interface (or script) to ensure data fetching isn't broken by the abbreviation mapping extraction.

---

# Phase 2: Architecture & Refactoring

This phase focuses on improving the maintainability and reliability of the codebase without changing the underlying mathematical model or product features.

## Proposed Changes

### Refactoring `prediction.py` Monolith
The `src/analytics/prediction.py` file is currently ~1,500 lines long and handles everything from cache management to core prediction algorithms and ML feature engineering.

- **`src/models/prediction_models.py`**: Extract the `MatchupPrediction` and `PrecomputedGame` dataclasses here so they can be imported without introducing circular dependencies.
- **`src/analytics/precompute.py`**: Extract all caching and precomputation functions (`_load_pc_cache`, `_save_pc_cache`, `_build_precompute_context`, `precompute_game_data`, `_infer_historical_injuries`).
- **`src/analytics/ml_features.py`**: Extract the ML feature engineering inputs (`_build_injury_context`, `_build_ml_features`).
- **`src/analytics/prediction.py` (Engine)**: Retain the core pipeline functions (`predict_matchup`, `predict_from_precomputed`, and tuning utilities) in the stripped-down file.

### Adding API Retries (`tenacity`)
Network unreliability can cause the background sync jobs and manual data pulls to fail sporadically.
- Install `tenacity` via standard package practices.
- Apply `@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))` to functions making external HTTP requests in `src/data/gamecast.py`, `src/data/odds_sync.py`, `src/data/nba_fetcher.py`, and `src/data/injury_scraper.py`.

### Unit Testing Core Logic
The project lacks a unit test suite to verify math and preventing regressions.
- Create a `tests/` directory at the project root.
- Add `tests/test_analytics.py` using standard `pytest`/`unittest`.
- Write tests for `stats_engine.py` logic (e.g., `_exponential_decay_weights`, `compute_fatigue`, zero-minute player fallback).
- Write tests for `prediction.py` math (`_clamp`, pacing fallbacks).

## Verification Plan

### Automated Tests
- Execute `pytest tests/` to confirm that all newly added unit tests pass.
- Run `python -m src.analytics.backtester --use-cache` to ensure the exact same ROI metrics from Phase 1 are produced, proving the refactor did not change the mathematical output.

### Manual Verification
- Review terminal logs during an odds sync or gamecast execution to confirm that `tenacity` retry wrappers don't break the existing execution paths.
