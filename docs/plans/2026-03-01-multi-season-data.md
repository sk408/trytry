# Multi-Season Data for Optimizer Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 2023-24 and 2024-25 season data to the database so the weight optimizer trains on ~2,900 games instead of ~440, producing more robust weights.

**Architecture:** Add a `season` column to `player_stats`, add a `historical_seasons` config, teach the fetcher/sync to loop over prior seasons, and make precompute use the correct season's team metrics for each game. The optimizer, sensitivity, and backtester all consume `List[PrecomputedGame]` and need zero changes — they automatically benefit from the larger pool.

**Tech Stack:** Python 3, SQLite, nba_api, Optuna (unchanged)

---

### Task 1: Add `season` column to `player_stats` schema

**Files:**
- Modify: `src/database/migrations.py:31-59` (player_stats CREATE TABLE)
- Modify: `src/database/migrations.py` (add migration function for existing DBs)

**Step 1: Add `season` column to schema SQL**

In `migrations.py`, add `season TEXT NOT NULL DEFAULT '2025-26'` to the `player_stats` CREATE TABLE, right after the `game_id` column (line ~37). Also update the UNIQUE constraint to include season for safety.

```python
# In SCHEMA_SQL, the player_stats table becomes:
CREATE TABLE IF NOT EXISTS player_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    opponent_team_id INTEGER NOT NULL,
    is_home INTEGER NOT NULL,
    game_date DATE NOT NULL,
    game_id TEXT,
    season TEXT NOT NULL DEFAULT '2025-26',
    points REAL NOT NULL,
    rebounds REAL NOT NULL,
    assists REAL NOT NULL,
    minutes REAL NOT NULL,
    steals REAL DEFAULT 0,
    blocks REAL DEFAULT 0,
    turnovers REAL DEFAULT 0,
    fg_made INTEGER DEFAULT 0,
    fg_attempted INTEGER DEFAULT 0,
    fg3_made INTEGER DEFAULT 0,
    fg3_attempted INTEGER DEFAULT 0,
    ft_made INTEGER DEFAULT 0,
    ft_attempted INTEGER DEFAULT 0,
    oreb REAL DEFAULT 0,
    dreb REAL DEFAULT 0,
    plus_minus REAL DEFAULT 0,
    win_loss TEXT,
    personal_fouls REAL DEFAULT 0,
    UNIQUE(player_id, opponent_team_id, game_date, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (opponent_team_id) REFERENCES teams(team_id)
);
```

**Step 2: Add migration for existing databases**

Add a `_migrate_player_stats_season()` function that runs `ALTER TABLE player_stats ADD COLUMN season TEXT NOT NULL DEFAULT '2025-26'` inside a try/except (column may already exist). Call it from `init_db()`.

```python
def _migrate_player_stats_season():
    """Add season column to player_stats if missing (v2 migration)."""
    try:
        execute("ALTER TABLE player_stats ADD COLUMN season TEXT NOT NULL DEFAULT '2025-26'")
        _log.info("Added 'season' column to player_stats")
    except Exception:
        pass  # Column already exists
```

Add this call at the end of `init_db()`, after `execute_script(SCHEMA_SQL)`.

**Step 3: Run to verify migration works**

Run: `python -c "from src.database.migrations import init_db; init_db(); print('OK')"`
Expected: OK (no crash, column added or already exists)

**Step 4: Commit**

```bash
git add src/database/migrations.py
git commit -m "feat: add season column to player_stats schema"
```

---

### Task 2: Add `historical_seasons` to config

**Files:**
- Modify: `src/config.py:12-26` (_DEFAULTS dict)

**Step 1: Add the config key**

Add `"historical_seasons": ["2023-24", "2024-25"]` to the `_DEFAULTS` dict in `config.py`.

```python
_DEFAULTS: Dict[str, Any] = {
    "db_path": "data/nba_analytics.db",
    "season": "2025-26",
    "season_year": "2025",
    "historical_seasons": ["2023-24", "2024-25"],
    # ... rest unchanged
}
```

**Step 2: Add a getter**

```python
def get_historical_seasons() -> list:
    return get("historical_seasons", [])
```

**Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add historical_seasons config setting"
```

---

### Task 3: Add season parameter to fetcher functions

**Files:**
- Modify: `src/data/nba_fetcher.py`

The bulk game log fetcher already accepts `season` param. We need to:

1. Add `season` param to `fetch_team_estimated_metrics()` (line ~238)
2. Add `season` param to `fetch_league_dash_team_stats()` (line ~279)
3. Add `season` param to `fetch_team_clutch_stats()` (line ~303)
4. Add `season` param to `fetch_team_hustle_stats()` (line ~328)
5. Update `save_game_logs()` to include the `season` column (line ~470)

**Step 1: Add season param to metric fetchers**

For each of the 4 functions above, change `season = get_season()` to:
```python
if season is None:
    season = get_season()
```
And add `season: Optional[str] = None` to the function signature.

Example for `fetch_team_estimated_metrics`:
```python
def fetch_team_estimated_metrics(season: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch TeamEstimatedMetrics."""
    try:
        from nba_api.stats.endpoints import TeamEstimatedMetrics
        if season is None:
            season = get_season()
        # ... rest unchanged
```

Apply the same pattern to `fetch_league_dash_team_stats`, `fetch_team_clutch_stats`, `fetch_team_hustle_stats`.

**Step 2: Update `save_game_logs` to include season column**

Modify `save_game_logs()` to accept and store the season:

```python
def save_game_logs(logs: List[Dict[str, Any]], season: Optional[str] = None):
    """Insert game logs into player_stats with conflict ignore (batched)."""
    if season is None:
        season = get_season()
    batch = []
    for log in logs:
        opp_id = log.get("opponent_team_id", 0)
        if opp_id == 0:
            opp_id = resolve_opponent_team_id(log.get("opponent_abbr", ""))
        if opp_id == 0:
            continue
        batch.append((
            log["player_id"], opp_id, log["is_home"], log["game_date"],
            log.get("game_id", ""), season,
            log["points"], log["rebounds"], log["assists"], log["minutes"],
            log.get("steals", 0), log.get("blocks", 0), log.get("turnovers", 0),
            log.get("fg_made", 0), log.get("fg_attempted", 0),
            log.get("fg3_made", 0), log.get("fg3_attempted", 0),
            log.get("ft_made", 0), log.get("ft_attempted", 0),
            log.get("oreb", 0), log.get("dreb", 0),
            log.get("plus_minus", 0), log.get("win_loss", ""),
            log.get("personal_fouls", 0),
        ))
    if batch:
        db.execute_many(
            """INSERT OR IGNORE INTO player_stats
               (player_id, opponent_team_id, is_home, game_date, game_id, season,
                points, rebounds, assists, minutes, steals, blocks, turnovers,
                fg_made, fg_attempted, fg3_made, fg3_attempted, ft_made, ft_attempted,
                oreb, dreb, plus_minus, win_loss, personal_fouls)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            batch,
        )
```

**Step 3: Commit**

```bash
git add src/data/nba_fetcher.py
git commit -m "feat: add season param to fetcher functions and save_game_logs"
```

---

### Task 4: Add `sync_historical_seasons()` to sync service

**Files:**
- Modify: `src/data/sync_service.py`

**Step 1: Add the new sync function**

Add this function before `full_sync()`:

```python
def sync_historical_seasons(callback: Optional[Callable] = None, force: bool = False):
    """Sync game logs and team metrics for historical seasons (one-time fetch)."""
    from src.config import get_historical_seasons

    historical = get_historical_seasons()
    if not historical:
        if callback:
            callback("No historical seasons configured, skipping...")
        return

    for hist_season in historical:
        step_key = f"historical_{hist_season}"

        # Historical data is immutable — only fetch once unless forced
        if not force and _is_fresh(step_key, hours=8760):  # 1 year freshness
            if callback:
                callback(f"Historical season {hist_season} already synced, skipping...")
            continue

        if callback:
            callback(f"Fetching game logs for {hist_season}...")

        # 1. Bulk game logs
        logs = nba_fetcher.fetch_bulk_game_logs(season=hist_season)
        if logs:
            nba_fetcher.save_game_logs(logs, season=hist_season)
            if callback:
                callback(f"  Saved {len(logs)} game logs for {hist_season}")
        else:
            if callback:
                callback(f"  No game logs found for {hist_season}")

        # 2. Team metrics for this season
        if callback:
            callback(f"Fetching team metrics for {hist_season}...")

        season = hist_season
        now = datetime.now().isoformat()

        # Estimated metrics — insert rows so UPDATE calls below have rows to update
        est_metrics = nba_fetcher.fetch_team_estimated_metrics(season=season)
        for m in est_metrics:
            tid = m["team_id"]
            db.execute(
                """INSERT INTO team_metrics (team_id, season, gp, w, l, w_pct,
                     e_off_rating, e_def_rating, e_net_rating, e_pace,
                     e_ast_ratio, e_oreb_pct, e_dreb_pct, e_reb_pct, e_tm_tov_pct, last_synced_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(team_id, season) DO UPDATE SET
                     gp=excluded.gp, w=excluded.w, l=excluded.l, w_pct=excluded.w_pct,
                     e_off_rating=excluded.e_off_rating, e_def_rating=excluded.e_def_rating,
                     e_net_rating=excluded.e_net_rating, e_pace=excluded.e_pace,
                     e_ast_ratio=excluded.e_ast_ratio, e_oreb_pct=excluded.e_oreb_pct,
                     e_dreb_pct=excluded.e_dreb_pct, e_reb_pct=excluded.e_reb_pct,
                     e_tm_tov_pct=excluded.e_tm_tov_pct, last_synced_at=excluded.last_synced_at""",
                (tid, season, m["gp"], m["w"], m["l"], m["w_pct"],
                 m["e_off_rating"], m["e_def_rating"], m["e_net_rating"], m["e_pace"],
                 m["e_ast_ratio"], m["e_oreb_pct"], m["e_dreb_pct"], m["e_reb_pct"],
                 m["e_tm_tov_pct"], now)
            )

        # Advanced, Four Factors, Opponent, Home/Road, Clutch, Hustle
        _update_team_dash_stats(season, "Advanced", "", now, callback)
        _update_team_dash_stats(season, "Four Factors", "", now, callback)
        _update_opponent_stats(season, now, callback)
        _update_home_road_stats(season, "Home", now, callback)
        _update_home_road_stats(season, "Road", now, callback)
        _update_clutch_stats(season, now, callback)
        _update_hustle_stats(season, now, callback)

        _set_sync_meta(step_key, _get_game_count(), _get_last_game_date())
        if callback:
            callback(f"Historical season {hist_season} sync complete")
```

**Step 2: Update `full_sync()` to include historical step**

Add the historical sync as a new step in the `full_sync()` steps list, right after "2/7 Player game logs":

```python
def full_sync(callback: Optional[Callable] = None, force: bool = False):
    """Full 8-step data sync."""
    if force:
        if callback:
            callback("Force mode: clearing sync caches...")
        clear_sync_cache()

    steps = [
        ("1/8 Reference data", sync_reference_data),
        ("2/8 Player game logs", sync_player_game_logs),
        ("3/8 Historical seasons", sync_historical_seasons),
        ("4/8 Injuries", sync_injuries_step),
        ("5/8 Injury history", sync_injury_history),
        ("6/8 Team metrics", sync_team_metrics),
        ("7/8 Player impact", sync_player_impact),
        ("8/8 Vegas odds", sync_historical_odds),
    ]
    for label, func in steps:
        if callback:
            callback(f"=== {label} ===")
        try:
            func(callback=callback, force=force)
        except Exception as e:
            logger.error(f"Error in {label}: {e}")
            if callback:
                callback(f"ERROR in {label}: {e}")
    if callback:
        callback("Full data sync complete!")
```

**Step 3: Update the helper functions to accept season as param**

The `_update_team_dash_stats`, `_update_opponent_stats`, `_update_home_road_stats`, `_update_clutch_stats`, and `_update_hustle_stats` helper functions all call `nba_fetcher.fetch_*` without passing `season`. They already receive `season` as their first argument but don't pass it through to the fetcher. Fix each:

- `_update_team_dash_stats`: change `nba_fetcher.fetch_league_dash_team_stats(measure_type=measure, location=location)` → add `season=season`
- `_update_opponent_stats`: change `nba_fetcher.fetch_league_dash_team_stats(measure_type="Opponent")` → add `season=season`
- `_update_home_road_stats`: change `nba_fetcher.fetch_league_dash_team_stats(measure_type="Base", location=location)` → add `season=season`
- `_update_clutch_stats`: change `nba_fetcher.fetch_team_clutch_stats()` → `nba_fetcher.fetch_team_clutch_stats(season=season)`
- `_update_hustle_stats`: change `nba_fetcher.fetch_team_hustle_stats()` → `nba_fetcher.fetch_team_hustle_stats(season=season)`

**Step 4: Commit**

```bash
git add src/data/sync_service.py
git commit -m "feat: add sync_historical_seasons step to sync pipeline"
```

---

### Task 5: Make precompute use correct season's team metrics

**Files:**
- Modify: `src/analytics/prediction.py:396-426` (`_get_team_metrics`)
- Modify: `src/analytics/prediction.py` (`_precompute_one` inside `precompute_game_data`)

**Step 1: Add season parameter to `_get_team_metrics`**

```python
def _get_team_metrics(team_id: int, season: Optional[str] = None) -> Dict[str, float]:
    """Fetch team metrics as a flat dict, using cache + memory store."""
    if season is None:
        season = get_season()

    cache_key = f"metrics_{season}"
    cached = team_cache.get(team_id, cache_key)
    if cached is not None:
        return cached

    try:
        from src.analytics.memory_store import get_store
        store = get_store()
        if store.team_metrics is not None and not store.team_metrics.empty:
            rows = store.team_metrics[
                (store.team_metrics["team_id"] == team_id) &
                (store.team_metrics["season"] == season)
            ]
            if not rows.empty:
                result = {str(k): (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in rows.iloc[0].to_dict().items()}
                team_cache.set(team_id, cache_key, result)
                return result
    except Exception:
        pass

    row = db.fetch_one(
        "SELECT * FROM team_metrics WHERE team_id = ? AND season = ?",
        (team_id, season)
    )
    result = dict(row) if row else {}
    team_cache.set(team_id, cache_key, result)
    return result
```

**Step 2: Add `_game_date_to_season()` helper**

Add a utility to map a game date to its NBA season string:

```python
def _game_date_to_season(game_date: str) -> str:
    """Map a game date (YYYY-MM-DD) to NBA season string (e.g. '2024-25').

    NBA regular season runs Oct–Apr. Games before July belong to the
    season that started the prior calendar year.
    """
    from src.config import get_season
    try:
        year, month = int(game_date[:4]), int(game_date[5:7])
        if month >= 7:  # Oct-Dec: season starts this year
            return f"{year}-{str(year + 1)[2:]}"
        else:  # Jan-Jun: season started last year
            return f"{year - 1}-{str(year)[2:]}"
    except (ValueError, IndexError):
        return get_season()
```

**Step 3: Use it in `_precompute_one`**

In `precompute_game_data()`, inside the `_precompute_one(g)` closure, change:

```python
# Before:
hm = _get_team_metrics(htid)
am = _get_team_metrics(atid)

# After:
game_season = _game_date_to_season(gdate)
hm = _get_team_metrics(htid, season=game_season)
am = _get_team_metrics(atid, season=game_season)
```

This ensures that a game from 2023-24 uses 2023-24 team metrics, not 2025-26 metrics.

**Step 4: Add `season` field to `PrecomputedGame` dataclass**

Add `season: str = ""` to the `PrecomputedGame` dataclass so the optimizer/backtester can see which season a game belongs to. Set it in `_precompute_one`:

```python
# In the PrecomputedGame construction:
season=game_season,
```

This bumps the schema hash automatically (field names changed), invalidating old precompute cache.

**Step 5: Commit**

```bash
git add src/analytics/prediction.py
git commit -m "feat: season-aware team metrics lookup for precompute"
```

---

### Task 6: Invalidate caches and test end-to-end

**Files:**
- No new files — this is a verification task

**Step 1: Delete old precompute cache**

The schema hash auto-invalidates, but to be safe:

```bash
rm -f data/cache/precomputed_games.pkl
```

**Step 2: Run a force sync to fetch historical data**

This is the big test — it will:
1. Fetch ~2,460 game logs for 2023-24 and 2024-25 (2 API calls, ~2 seconds)
2. Fetch team metrics for both seasons (~12 API calls, ~10 seconds)
3. Store everything in SQLite with correct season tags
4. Then the normal current-season sync runs as usual

```bash
python -c "
from src.database.migrations import init_db
init_db()
from src.data.sync_service import sync_historical_seasons
sync_historical_seasons(callback=print, force=True)
"
```

Expected: Should print progress messages for each season and complete without errors.

**Step 3: Verify data is in DB**

```bash
python -c "
from src.database import db
from src.database.migrations import init_db
init_db()

# Game logs per season
rows = db.fetch_all('SELECT season, COUNT(*) as cnt FROM player_stats GROUP BY season')
for r in rows:
    print(f'{r[\"season\"]}: {r[\"cnt\"]} game logs')

# Team metrics per season
rows = db.fetch_all('SELECT season, COUNT(*) as cnt FROM team_metrics GROUP BY season')
for r in rows:
    print(f'Team metrics {r[\"season\"]}: {r[\"cnt\"]} teams')
"
```

Expected: Should show ~30,000+ game logs for each historical season (each game has ~10-13 player rows per side) and 30 teams per season for team_metrics.

**Step 4: Test precompute with historical games**

```bash
python -c "
from src.database.migrations import init_db
init_db()
from src.analytics.prediction import precompute_game_data
games = precompute_game_data(callback=print)
print(f'Total precomputed games: {len(games)}')
seasons = {}
for g in games:
    s = g.season if hasattr(g, 'season') and g.season else 'unknown'
    seasons[s] = seasons.get(s, 0) + 1
for s, c in sorted(seasons.items()):
    print(f'  {s}: {c} games')
"
```

Expected: ~2,900 total games across 3 seasons.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: multi-season data integration complete"
```

---

## Summary of Changes

| File | Lines Changed | What |
|------|:---:|--------|
| `src/config.py` | ~3 | Add `historical_seasons` default + getter |
| `src/database/migrations.py` | ~10 | Add `season` column to `player_stats` |
| `src/data/nba_fetcher.py` | ~20 | Season params on 4 fetchers + `save_game_logs` |
| `src/data/sync_service.py` | ~60 | New `sync_historical_seasons()` + updated `full_sync()` |
| `src/analytics/prediction.py` | ~25 | Season-aware `_get_team_metrics`, `_game_date_to_season`, `PrecomputedGame.season` |

**Total: ~120 lines changed across 5 files. No breaking changes to optimizer, sensitivity, backtester, or any downstream consumer.**
