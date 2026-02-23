# NBA Game Prediction System — Full Reproduction Specification

## TL;DR

This is a Python-based NBA analytics platform that predicts game outcomes (point spreads and totals) by combining player-level statistical projections, team advanced metrics, injury intelligence, an XGBoost ML ensemble, and ESPN predictor blending. The core philosophy: train the model against **historical game results** from the current season, optimize prediction weights via Bayesian search until past-game accuracy is maximized, then apply those calibrated weights to **future matchups**. It runs as both a FastAPI web app and a PySide6 desktop GUI, backed by SQLite.

---

## 1. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13+ |
| Database | SQLite (WAL mode, `busy_timeout=30s`, `RLock` concurrency) |
| Web | FastAPI 0.99.1, Uvicorn, Jinja2 templates, SSE streaming |
| Desktop | PySide6 (Qt6), QSS dark theme (`#0f1923` bg) |
| ML | XGBoost (spread + total models), SHAP TreeExplainer |
| Optimization | Optuna (TPE sampler, Bayesian) |
| Data Sources | `nba_api` (NBA Stats), ESPN APIs (scoreboard, summary, pickcenter, winprobability, predictor), CBS Sports, RotoWire (injury scraping) |
| Key Libraries | pandas, numpy, matplotlib, requests, beautifulsoup4, linedrive (WebSocket), websocket-client |

### Dependencies (requirements.txt — web)
```
fastapi==0.99.1
uvicorn==0.30.6
jinja2==3.1.4
python-multipart==0.0.9
pydantic>=1.10,<2.0
nba_api==1.11.3
requests==2.32.5
beautifulsoup4==4.14.3
linedrive
websocket-client
tzdata
optuna
xgboost
shap
```

### Dependencies (requirements-desktop.txt — desktop GUI)
```
pandas, numpy, matplotlib
nba_api, requests, beautifulsoup4
PySide6
fastapi, uvicorn, jinja2, python-multipart, pydantic>=1.10,<2.0
linedrive, websocket-client
optuna, xgboost, shap
```

---

## 2. Project Structure

```
main.py                      # Web entry — runs FastAPI via uvicorn on port 8000
desktop.py                   # Desktop entry — launches PySide6 GUI
src/
  __init__.py                # Package init — calls patch_nba_api_headers()
  config.py                  # JSON-backed settings at data/app_settings.json
  analytics/
    prediction.py            # Core: predict_matchup(), MatchupPrediction, PrecomputedGame
    stats_engine.py           # Player splits, aggregate_projection(), 240-min budget
    weight_config.py          # WeightConfig dataclass (30 tunable parameters)
    weight_optimizer.py       # VectorizedGames, Optuna TPE, per-team refinement, residual calibration, feature importance
    backtester.py             # Historical game replay, caching, metrics
    autotune.py               # Per-team scoring corrections via grid search
    ml_model.py               # XGBoost train/predict, SHAP, model persistence
    regression_test.py        # Prediction regression testing (baselines + comparison)
    pipeline.py               # 12-step orchestrator with smart caching
    live_prediction.py        # 3-signal blend for in-game predictions
    live_recommendations.py   # Live betting recommendations
    injury_intelligence.py    # Play-through rates, player tendencies, keyword modifiers
    injury_history.py         # Infer injuries from missing games in logs
    memory_store.py           # Singleton InMemoryDataStore (7 pandas DataFrames)
    pipeline_cache.py         # Pickle cache + SHA-256 invalidation
    cache.py                  # TTL decorator, TeamCache, SessionCache
    odds_converter.py         # American odds ↔ probability conversions
  data/
    nba_fetcher.py            # nba_api wrappers (teams, rosters, logs, metrics)
    live_scores.py            # nba_api live ScoreBoard
    gamecast.py               # ESPN integration (odds, plays, box, WebSocket)
    injury_scraper.py         # ESPN/CBS/RotoWire scraping chain
    sync_service.py           # Central orchestrator: full_sync() (6 steps)
    image_cache.py            # Disk cache for player/team images
    _http_headers.py          # Browser-like headers, monkey-patches nba_api
  database/
    db.py                     # SQLite WAL, RLock, busy_timeout=30s, FIFO queue
    migrations.py             # 16 tables schema, indexes, init_db()
    models.py                 # Dataclasses: Team, Player, PlayerStat, etc.
  notifications/
    __init__.py
    service.py                # DB persistence, listener pattern, webhook/ntfy/toast push
    models.py                 # Notification dataclass
    injury_monitor.py         # Background diff engine, HIGH_IMPACT_MPG=20.0
  ui/
    main_window.py            # MainWindow — 10 tabs, dark QSS, notification bell
    dashboard.py              # 4 stat cards, 6 sync buttons, activity log
    live_view.py              # Auto-refresh 30s, color-coded rows
    gamecast_view.py          # Live game detail, box score, play-by-play, odds
    matchup_view.py           # Game selector, prediction cards, injury labels, H2H
    schedule_view.py          # Schedule table, lazy loading
    players_view.py           # All players + injured by impact tier
    accuracy_view.py          # 12 QObject workers, summary cards, team/prediction tables
    autotune_view.py          # Team corrections with before/after error
    allstar_view.py           # All-Star Weekend (MVP, 3PT, Rising Stars, Game Winner)
    admin_view.py             # DB path/size, reset
    notification_widget.py    # Bell badge + popup panel
    workers.py                # 6 QObject workers for sync operations
    widgets/
      __init__.py
  web/
    __init__.py
    app.py                    # FastAPI — 30+ routes, 18+ SSE endpoints
    player_utils.py           # Web utilities
    static/
      style.css               # Mobile-first dark theme (968 lines)
    templates/
      base.html               # Jinja2 base — 9 nav links
      dashboard.html
      live.html
      gamecast.html
      players.html
      matchups.html
      schedule.html
      accuracy.html
      autotune.html
      admin.html
data/
  nba_analytics.db            # SQLite database (auto-created)
  pipeline_state.json         # Pipeline step timestamps + game snapshot
  app_settings.json           # User settings (auto-created)
  manual_injuries.json        # Manual injury overrides
  regression_baselines/       # Saved prediction baselines for regression testing
  backtest_cache/             # Cached backtest results (bt_{hash}.json)
  cache/
    player_photos/            # Cached headshot images
    team_logos/                # Cached team logo images
  ml_models/
    spread_model.json         # Serialized XGBoost spread model
    total_model.json          # Serialized XGBoost total model
    model_meta.json           # Training metadata (features, counts, date)
    feature_columns.json      # Ordered list of 73 ML feature names
```

---

## 3. Data Sources & APIs

### NBA Stats (via `nba_api` library)

| Function | Endpoint | Data Fetched |
|---|---|---|
| `fetch_teams()` | `nba_api.stats.static.teams.get_teams()` | id, full_name, abbreviation, conference |
| `fetch_players()` | `CommonTeamRoster` (per team, 30 calls) | PLAYER_ID, PLAYER, POSITION, HEIGHT, WEIGHT, AGE, EXP |
| `fetch_player_game_logs()` | `PlayerGameLog` | Game_ID, GAME_DATE, MATCHUP, WL, PTS, REB, AST, MIN, STL, BLK, TOV, FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, PLUS_MINUS, PF |
| `fetch_schedule()` (played) | `LeagueGameFinder` | TEAM_ID, GAME_DATE, MATCHUP |
| `fetch_team_estimated_metrics()` | `TeamEstimatedMetrics` | E_OFF_RATING, E_DEF_RATING, E_PACE, etc. |
| `fetch_league_dash_team_stats()` | `LeagueDashTeamStats` | Flexible: Base/Advanced/Four Factors/Opponent, Home/Road |
| `fetch_team_clutch_stats()` | `LeagueDashTeamClutch` | NET_RATING, EFG_PCT, TS_PCT (last 5 min, within 5 pts) |
| `fetch_team_hustle_stats()` | `LeagueHustleStatsTeam` | DEFLECTIONS, LOOSE_BALLS_RECOVERED, CONTESTED_SHOTS, CHARGES_DRAWN, SCREEN_ASSISTS |
| `fetch_player_on_off()` | `TeamPlayerOnOffSummary` | On/off court OFF_RATING, DEF_RATING, NET_RATING, MIN |
| `fetch_player_estimated_metrics()` | `PlayerEstimatedMetrics` | E_USG_PCT, E_OFF_RATING, E_DEF_RATING, E_NET_RATING, E_PACE, E_AST_RATIO, E_OREB_PCT, E_DREB_PCT |

All NBA API calls use browser-like User-Agent headers (monkey-patched into `nba_api` globals) and 0.8s sleep between calls.

### NBA CDN (Schedule)
- URL: `https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json`
- Parses `leagueSchedule.gameDates[].games[]` for future games
- Extracts: `gameDateEst`, `homeTeam.teamTricode`, `awayTeam.teamTricode`, `gameDateTimeEst`, `arenaName`
- Converts Eastern → Pacific time

### ESPN APIs

| Constant | URL |
|---|---|
| Scoreboard | `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard` |
| Game Summary | `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}` |

Summary API sections used:
- `pickcenter` — odds/spread/over-under/moneyline/ATS records
- `winprobability` — live win percentage
- `predictor` — ESPN ML-based pre-game prediction (`gameProjection` home/away %)
- `plays` — play-by-play events
- `boxscore` — player box scores
- `leaders` — game stat leaders
- `header` — linescores per quarter

Optional: Linedrive WebSocket for live streaming (with polling fallback at 15s play / 30s odds intervals).

### Injury Scraping (waterfall — first success wins)

| Priority | Source | URL |
|---|---|---|
| 1 | ESPN | `https://www.espn.com/nba/injuries` |
| 2 | CBS Sports | `https://www.cbssports.com/nba/injuries/` |
| 3 | RotoWire | `https://www.rotowire.com/basketball/injury-report.php` |

Plus `data/manual_injuries.json` for manual overrides (take precedence over scraped data).

User-Agent: `Mozilla/5.0 (Windows NT 10.0; Win64; x64) ... Chrome/120.0.0.0`

---

## 4. Database Schema (16 Tables, SQLite)

### Core Tables

#### `teams`
| Column | Type | Constraints |
|---|---|---|
| `team_id` | INTEGER | PRIMARY KEY |
| `name` | TEXT | NOT NULL |
| `abbreviation` | TEXT | NOT NULL UNIQUE |
| `conference` | TEXT | |

#### `players`
| Column | Type | Constraints |
|---|---|---|
| `player_id` | INTEGER | PRIMARY KEY |
| `name` | TEXT | NOT NULL |
| `team_id` | INTEGER | NOT NULL, FK → teams |
| `position` | TEXT | |
| `is_injured` | INTEGER | NOT NULL DEFAULT 0 |
| `injury_note` | TEXT | |
| `height` | TEXT | |
| `weight` | TEXT | |
| `age` | INTEGER | |
| `experience` | INTEGER | |

#### `player_stats`
| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PK AUTOINCREMENT |
| `player_id` | INTEGER | NOT NULL, FK → players |
| `opponent_team_id` | INTEGER | NOT NULL, FK → teams |
| `is_home` | INTEGER | NOT NULL |
| `game_date` | DATE | NOT NULL |
| `game_id` | TEXT | |
| `points` | REAL | NOT NULL |
| `rebounds` | REAL | NOT NULL |
| `assists` | REAL | NOT NULL |
| `minutes` | REAL | NOT NULL |
| `steals` | REAL | DEFAULT 0 |
| `blocks` | REAL | DEFAULT 0 |
| `turnovers` | REAL | DEFAULT 0 |
| `fg_made` | INTEGER | DEFAULT 0 |
| `fg_attempted` | INTEGER | DEFAULT 0 |
| `fg3_made` | INTEGER | DEFAULT 0 |
| `fg3_attempted` | INTEGER | DEFAULT 0 |
| `ft_made` | INTEGER | DEFAULT 0 |
| `ft_attempted` | INTEGER | DEFAULT 0 |
| `oreb` | REAL | DEFAULT 0 |
| `dreb` | REAL | DEFAULT 0 |
| `plus_minus` | REAL | DEFAULT 0 |
| `win_loss` | TEXT | |
| `personal_fouls` | REAL | DEFAULT 0 |
| | | UNIQUE(player_id, opponent_team_id, game_date) |

#### `predictions`
| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PK AUTOINCREMENT |
| `home_team_id` | INTEGER | NOT NULL, FK → teams |
| `away_team_id` | INTEGER | NOT NULL, FK → teams |
| `game_date` | DATE | NOT NULL |
| `predicted_spread` | REAL | NOT NULL |
| `predicted_total` | REAL | NOT NULL |
| `actual_spread` | REAL | |
| `actual_total` | REAL | |

#### `live_games`
| Column | Type | Constraints |
|---|---|---|
| `game_id` | TEXT | PRIMARY KEY |
| `home_team_id` | INTEGER | NOT NULL, FK → teams |
| `away_team_id` | INTEGER | NOT NULL, FK → teams |
| `start_time_utc` | TEXT | |
| `status` | TEXT | |
| `period` | INTEGER | |
| `clock` | TEXT | |
| `home_score` | INTEGER | |
| `away_score` | INTEGER | |
| `last_updated` | TEXT | |

### Advanced Metrics Tables

#### `team_metrics` — PK(`team_id`, `season`), ~55 columns
- Basic: `gp`, `w`, `l`, `w_pct`
- Estimated: `e_off_rating`, `e_def_rating`, `e_net_rating`, `e_pace`, `e_ast_ratio`, `e_oreb_pct`, `e_dreb_pct`, `e_reb_pct`, `e_tm_tov_pct`
- Official: `off_rating`, `def_rating`, `net_rating`, `pace`, `efg_pct`, `ts_pct`, `ast_ratio`, `ast_to`, `oreb_pct`, `dreb_pct`, `reb_pct`, `tm_tov_pct`, `pie`
- Four Factors (own): `ff_efg_pct`, `ff_fta_rate`, `ff_tm_tov_pct`, `ff_oreb_pct`
- Four Factors (opponent): `opp_efg_pct`, `opp_fta_rate`, `opp_tm_tov_pct`, `opp_oreb_pct`
- Opponent: `opp_pts`, `opp_fg_pct`, `opp_fg3_pct`, `opp_ft_pct`
- Clutch: `clutch_gp`, `clutch_w`, `clutch_l`, `clutch_net_rating`, `clutch_efg_pct`, `clutch_ts_pct`
- Hustle: `deflections`, `loose_balls_recovered`, `contested_shots`, `charges_drawn`, `screen_assists`
- Home/Road splits: `home_gp`, `home_w`, `home_l`, `home_pts`, `home_opp_pts`, `road_gp`, `road_w`, `road_l`, `road_pts`, `road_opp_pts`
- `last_synced_at` TEXT

#### `player_impact` — PK(`player_id`, `season`)
| Column | Type |
|---|---|
| `player_id` | INTEGER NOT NULL, FK → players |
| `team_id` | INTEGER NOT NULL, FK → teams |
| `season` | TEXT NOT NULL |
| `on_court_off_rating` | REAL |
| `on_court_def_rating` | REAL |
| `on_court_net_rating` | REAL |
| `off_court_off_rating` | REAL |
| `off_court_def_rating` | REAL |
| `off_court_net_rating` | REAL |
| `net_rating_diff` | REAL (on_net - off_net) |
| `on_court_minutes` | REAL |
| `e_usg_pct` | REAL |
| `e_off_rating` | REAL |
| `e_def_rating` | REAL |
| `e_net_rating` | REAL |
| `e_pace` | REAL |
| `e_ast_ratio` | REAL |
| `e_oreb_pct` | REAL |
| `e_dreb_pct` | REAL |
| `last_synced_at` | TEXT |

### Injury Tracking Tables

#### `injury_history`
| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PK AUTOINCREMENT |
| `player_id` | INTEGER | NOT NULL, FK → players |
| `team_id` | INTEGER | NOT NULL, FK → teams |
| `game_date` | DATE | NOT NULL |
| `was_out` | INTEGER | NOT NULL DEFAULT 1 |
| `avg_minutes` | REAL | |
| `reason` | TEXT | |
| | | UNIQUE(player_id, game_date) |

#### `injury_status_log`
| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PK AUTOINCREMENT |
| `player_id` | INTEGER | NOT NULL, FK → players |
| `team_id` | INTEGER | NOT NULL, FK → teams |
| `log_date` | TEXT | NOT NULL (ISO) |
| `status_level` | TEXT | NOT NULL (Out/Doubtful/Questionable/Probable/Day-To-Day/GTD/Available) |
| `injury_keyword` | TEXT | DEFAULT '' |
| `injury_detail` | TEXT | DEFAULT '' |
| `next_game_date` | TEXT | |
| `did_play` | INTEGER | (1/0/NULL) |
| | | UNIQUE(player_id, log_date, status_level) |

### Tuning / Calibration Tables

#### `team_tuning`
| Column | Type | Constraints |
|---|---|---|
| `team_id` | INTEGER | PRIMARY KEY, FK → teams |
| `home_pts_correction` | REAL | DEFAULT 0.0 |
| `away_pts_correction` | REAL | DEFAULT 0.0 |
| `games_analyzed` | INTEGER | DEFAULT 0 |
| `avg_spread_error_before` | REAL | DEFAULT 0.0 |
| `avg_total_error_before` | REAL | DEFAULT 0.0 |
| `last_tuned_at` | TEXT | |
| `tuning_mode` | TEXT | DEFAULT 'classic' |
| `tuning_version` | TEXT | DEFAULT 'v1_classic' |
| `tuning_sample_size` | INTEGER | DEFAULT 0 |

#### `model_weights`
| Column | Type | Constraints |
|---|---|---|
| `key` | TEXT | PRIMARY KEY |
| `value` | REAL | |

#### `team_weight_overrides`
| Column | Type | Constraints |
|---|---|---|
| `team_id` | INTEGER | |
| `key` | TEXT | |
| `value` | REAL | |
| | | PK(team_id, key) |

#### `residual_calibration` / `residual_calibration_total` (created dynamically)
Bin-based correction tables storing `avg_residual` per prediction range bin.

### Other Tables

#### `game_quarter_scores`
| Column | Type | Constraints |
|---|---|---|
| `game_id` | TEXT | NOT NULL |
| `team_id` | INTEGER | NOT NULL |
| `q1` | INTEGER | |
| `q2` | INTEGER | |
| `q3` | INTEGER | |
| `q4` | INTEGER | |
| `ot` | INTEGER | DEFAULT 0 |
| `final_score` | INTEGER | |
| `game_date` | TEXT | |
| `is_home` | INTEGER | DEFAULT 0 |
| | | PK(game_id, team_id) |

#### `player_sync_cache`
| Column | Type | Constraints |
|---|---|---|
| `player_id` | INTEGER | PRIMARY KEY |
| `last_synced_at` | TEXT | NOT NULL |
| `games_synced` | INTEGER | DEFAULT 0 |
| `latest_game_date` | DATE | |

#### `sync_meta`
| Column | Type | Constraints |
|---|---|---|
| `step_name` | TEXT | PRIMARY KEY |
| `last_synced_at` | TEXT | NOT NULL |
| `game_count_at_sync` | INTEGER | DEFAULT 0 |
| `last_game_date_at_sync` | TEXT | DEFAULT '' |
| `extra` | TEXT | DEFAULT '' |

#### `notifications`
| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PK AUTOINCREMENT |
| `category` | TEXT | NOT NULL ('injury', 'matchup', 'insight') |
| `severity` | TEXT | NOT NULL ('info', 'warning', 'critical') |
| `title` | TEXT | NOT NULL |
| `body` | TEXT | NOT NULL |
| `data` | TEXT | DEFAULT '{}' (JSON) |
| `created_at` | TEXT | NOT NULL |
| `read` | INTEGER | NOT NULL DEFAULT 0 |

### All Indexes

| Index | Table | Columns |
|---|---|---|
| `idx_player_stats_player_date` | player_stats | (player_id, game_date DESC) |
| `idx_player_stats_matchup` | player_stats | (opponent_team_id, game_date DESC) |
| `idx_player_stats_game_id` | player_stats | (game_id, is_home) |
| `idx_predictions_matchup` | predictions | (home_team_id, away_team_id, game_date) |
| `idx_injury_history_team_date` | injury_history | (team_id, game_date) |
| `idx_injury_history_player` | injury_history | (player_id, game_date) |
| `idx_injury_status_log_player` | injury_status_log | (player_id, log_date) |
| `idx_injury_status_log_status` | injury_status_log | (status_level, did_play) |
| `idx_injury_status_log_team` | injury_status_log | (team_id, log_date) |
| `idx_player_sync_cache_date` | player_sync_cache | (last_synced_at) |
| `idx_player_impact_team` | player_impact | (team_id, season) |
| `idx_quarter_scores_team_date` | game_quarter_scores | (team_id, game_date) |
| `idx_notifications_unread` | notifications | (read, id DESC) |

---

## 5. The Prediction Engine — Core Algorithm

The prediction flows through `predict_matchup()` in `src/analytics/prediction.py`. Here is every step:

### Step 0: Weight Resolution
```python
home_w = load_team_weights(home_team_id)   # per-team overrides or None
away_w = load_team_weights(away_team_id)
if home_w and away_w:
    w = home_w.blend(away_w)    # average all fields
elif home_w:
    w = home_w
elif away_w:
    w = away_w
else:
    w = get_weight_config()     # global defaults
```

### Step 1: Player-Level Projections
For each team, `aggregate_projection()` runs `player_splits()` for every active player.

#### The 50/25/25 Blended Stat Projection with Exponential Decay

```python
_DECAY = 0.9  # per-game exponential decay
# row 0 (most recent) weight = 1.0
# row 1 = 0.9, row 2 = 0.81, row 3 = 0.729 ...
# For 10 games: ~60% weight on last 5

# Blend weights:
w_base = 0.50   # overall recent games (last 10)
w_loc  = 0.25   # home/away split (if >= 3 games, else 0 → folded into base)
w_opp  = 0.25   # vs-opponent (if >= 3 games)
       = 0.15   # if 2 games vs opp
       = 0.10   # if 1 game vs opp
       = 0.0    # if 0 games vs opp

# Unused weight folds into base:
w_base = 1.0 - w_loc - w_opp

# Each slice uses exponential decay (0.9^n, normalized) as weights
result[col] = w_base * weighted_mean(base) + w_loc * weighted_mean(loc) + w_opp * weighted_mean(opp)
```

#### 240-Minute Budget Algorithm
```python
for each player:
    splits = player_splits(pid, opponent_team_id, is_home, recent_games=10, as_of_date)
    weight = player_weights.get(pid, 1.0)

    # Return-from-injury discount:
    missed = get_games_missed_streak(pid, as_of_date)
    if missed > 0:
        weight *= max(0.85, 1.0 - 0.03 * missed)   # 3% per game, capped at 15%

    totals[k] += splits[k] * weight

# If total projected minutes > 240:
if total_projected_minutes > 240.0:
    scale = 240.0 / total_projected_minutes
    # Scale ALL counting stats (points, rebounds, assists, minutes, steals,
    # blocks, turnovers, oreb, dreb, shot attempts) × scale
```

#### `as_of_date` Filtering
When provided (backtesting/ML training), only games **strictly before** that date are used — prevents lookahead bias.

### Step 2: Home Court Advantage
```python
home_court = home_pts_avg - road_pts_avg  # from team_metrics table
# Clamped to [1.5, 5.0], change clamp to [0.5,10]
# Default fallback: 3.0
```

### Step 3: Opponent Defensive Adjustment
```python
away_def_factor_raw = opponent_pts_allowed / league_avg_ppg  # < 1.0 = good defense
home_def_factor_raw = opponent_pts_allowed / league_avg_ppg

# Dampen toward 1.0:
away_def_factor = 1.0 + (away_def_factor_raw - 1.0) * w.def_factor_dampening  # default 0.5
home_def_factor = 1.0 + (home_def_factor_raw - 1.0) * w.def_factor_dampening

home_base_pts = home_proj["points"] * away_def_factor
away_base_pts = away_proj["points"] * home_def_factor
```

### Step 4: Autotune Corrections
```python
home_base_pts += home_tuning["home_pts_correction"]  # from team_tuning table
away_base_pts += away_tuning["away_pts_correction"]
```

### Step 5: Fatigue Detection
```python
rest_days = (game_date - last_game).days
is_back_to_back = (rest_days == 1)
is_3_in_4 = (games in 4-day window >= 3)
is_4_in_6 = (games in 6-day window >= 4)

penalty = 0.0
if B2B:        penalty += 2.0      # w.fatigue_b2b
if 3-in-4:    penalty += 1.0      # w.fatigue_3in4
if 4-in-6:    penalty += 1.5      # w.fatigue_4in6
if rest == 0:  penalty += 3.0      # same-day doubleheader

rest_bonus = 0.0
if rest_days >= 4: rest_bonus = -1.5
elif rest_days >= 3: rest_bonus = -1.0

fatigue_penalty = penalty + rest_bonus   # can be negative (= advantage)
fatigue_adj = home_fatigue_penalty - away_fatigue_penalty
```

### Step 6: Spread Calculation (cumulative)
```python
spread = (home_base_pts - away_base_pts) + home_court
spread -= fatigue_adj

# Turnover differential
spread += (home_to_margin - away_to_margin) * w.turnover_margin_mult     # default 0.4

# Rebound differential
spread += (home_reb - away_reb) * w.rebound_diff_mult                    # default 0.08

# Off/Def rating matchup
home_matchup_edge = home_off - away_def
away_matchup_edge = away_off - home_def
spread += (home_matchup_edge - away_matchup_edge) * w.rating_matchup_mult  # default 0.08

# Four Factors (see formula below)
spread += four_factors_adj

# Clutch (only if |spread| < w.clutch_threshold = 6.0)
if abs(spread) < 6.0:
    spread += clutch_adj

# Hustle
spread += hustle_spread_adj
```

#### Four Factors Adjustment
```python
efg_edge  = (home_efg - away_opp_efg) - (away_efg - home_opp_efg)
tov_edge  = (away_tov - home_opp_tov) - (home_tov - away_opp_tov)
oreb_edge = (home_oreb - away_opp_oreb) - (away_oreb - home_opp_oreb)
fta_edge  = (home_fta - away_opp_fta) - (away_fta - home_opp_fta)

four_factors_adj = (efg_edge * 0.40 + tov_edge * 0.25 +
                    oreb_edge * 0.20 + fta_edge * 0.15) * 0.3
```

#### Clutch Adjustment
```python
clutch_diff = (home_clutch_net_rating - away_clutch_net_rating) * 0.05
clutch_adj = clamp(-10.0, clutch_diff, 10.0)
```

#### Hustle Spread Adjustment
```python
home_effort = home_deflections + home_contested_shots * 0.3
away_effort = away_deflections + away_contested_shots * 0.3
hustle_spread_adj = (home_effort - away_effort) * 0.02
```

### Step 7: Total Calculation
```python
total = home_base_pts + away_base_pts

# Injury impact on total (base model only when ml_ensemble_weight == 0)
total -= (home_ppg_lost + away_ppg_lost)

# Pace adjustment
expected_pace = (home_pace + away_pace) / 2
pace_factor = (expected_pace - 98.0) / 98.0        # w.pace_baseline = 98.0
total *= (1 + pace_factor * 0.20)                   # w.pace_mult = 0.20

# Defensive disruption
total -= (max(0, combined_steals - 14.0) * 0.15 +   # steals threshold/penalty
          max(0, combined_blocks - 10.0) * 0.12)     # blocks threshold/penalty

# Offensive rebound boost
total += (combined_oreb - 20.0) * 0.2                # oreb baseline/mult

# Hustle total (deflections above baseline reduce total)
if combined_deflections > 30.0:
    total -= (combined_deflections - 30.0) * 0.1

# Fatigue total impact
total -= combined_fatigue * 0.3                       # w.fatigue_total_mult
```

### Step 8: ESPN Predictor Blend (80/20)
```python
espn_edge = espn_home_win_pct - 50.0
espn_implied_spread = espn_edge * 0.3                     # w.espn_spread_scale
blended = model_spread * 0.80 + espn_implied_spread * 0.20  # w.espn_model_weight / espn_weight

# If model and ESPN disagree in sign (one > 0.5, other < -0.5):
blended *= 0.85                                           # w.espn_disagree_damp
```

### Step 9: ML Ensemble Blend
Only when `w.ml_ensemble_weight > 0` (default 0.4), ML model loaded, and `ml_confidence > 0.3`:

```python
ml_wt = 0.4  # w.ml_ensemble_weight

# Early-season dampening: if min(home_gp, away_gp) < 15 games
ml_wt *= (min_gp / 15.0)    # linear ramp 0→1

# Disagreement dampening: if |ml_spread - spread| > 8.0
ml_wt *= 0.7                 # w.ml_disagree_damp

# Uncertainty dampening:
uncertainty_scale = max(0.35, min(1.0, 1.0 / (1.0 + (ml_spread_std / 12.0) + (ml_total_std / 20.0))))
ml_wt *= uncertainty_scale

base_weight = 1.0 - ml_wt
spread = base_weight * spread + ml_wt * ml_spread
total  = base_weight * total  + ml_wt * ml_total
```

### Step 10: Sanity Clamps
```python
spread = clamp(-30.0, spread, 30.0)     # w.spread_clamp
total  = clamp(185.0, total, 268.0)      # w.total_min / w.total_max
```

### Step 11: Residual Calibration
Lookup predicted spread in bin table (9 bins) and total in bin table (8 bins). If bin has ≥5 samples, subtract the average residual (`predicted - actual`) for that bin.

**Spread bins (9):**

| Label | Low | High |
|---|---|---|
| `big_away` | -30 | -18 |
| `med_away` | -18 | -12 |
| `small_away` | -12 | -8 |
| `slight_away` | -8 | -4 |
| `toss_up` | -4 | 4 |
| `slight_home` | 4 | 8 |
| `small_home` | 8 | 12 |
| `med_home` | 12 | 18 |
| `big_home` | 18 | 30 |

**Total bins (8):**

| Label | Low | High |
|---|---|---|
| `very_low` | 180 | 200 |
| `low` | 200 | 210 |
| `below_avg` | 210 | 215 |
| `avg_low` | 215 | 220 |
| `avg_high` | 220 | 225 |
| `above_avg` | 225 | 230 |
| `high` | 230 | 240 |
| `very_high` | 240 | 270 |

### Final: Derive Individual Scores
```python
predicted_home_score = (total + spread) / 2
predicted_away_score = (total - spread) / 2
```

---

## 6. WeightConfig — All 30+ Tunable Parameters

| Parameter | Default | Optimizer Range | Purpose |
|---|---|---|---|
| `def_factor_dampening` | 0.5 | [0.25, 0.75] | Dampen defensive factor toward 1.0 |
| `turnover_margin_mult` | 0.4 | [0.15, 0.65] | TO margin → spread points |
| `rebound_diff_mult` | 0.08 | [0.02, 0.15] | Rebound diff → spread |
| `rating_matchup_mult` | 0.08 | [0.02, 0.15] | Off/Def matchup → spread |
| `four_factors_scale` | 0.3 | [0.10, 0.60] | Overall Four Factors scaling |
| `ff_efg_weight` | 0.40 | — | eFG% edge sub-weight |
| `ff_tov_weight` | 0.25 | — | TOV% edge sub-weight |
| `ff_oreb_weight` | 0.20 | — | OREB% edge sub-weight |
| `ff_fta_weight` | 0.15 | — | FT rate edge sub-weight |
| `clutch_scale` | 0.05 | [0.02, 0.10] | Clutch net rating multiplier |
| `clutch_cap` | 2.0 | — | Max clutch adjustment ± |
| `clutch_threshold` | 6.0 | — | Only apply clutch when |spread| < this |
| `hustle_effort_mult` | 0.02 | [0.005, 0.05] | Hustle effort diff → spread |
| `hustle_contested_wt` | 0.3 | — | Contested shots weight in effort |
| `pace_baseline` | 98.0 | — | League average pace |
| `pace_mult` | 0.20 | [0.08, 0.35] | Pace deviation → total scaling |
| `steals_threshold` | 14.0 | — | Combined steals baseline |
| `steals_penalty` | 0.15 | — | Points per excess steal |
| `blocks_threshold` | 10.0 | — | Combined blocks baseline |
| `blocks_penalty` | 0.12 | — | Points per excess block |
| `oreb_baseline` | 20.0 | — | Combined OREB baseline |
| `oreb_mult` | 0.2 | — | Points per excess OREB |
| `hustle_defl_baseline` | 30.0 | — | Combined deflections baseline |
| `hustle_defl_penalty` | 0.1 | — | Points per excess deflection |
| `fatigue_total_mult` | 0.3 | [0.10, 0.60] | Combined fatigue → total reduction |
| `fatigue_b2b` | 2.0 | — | Back-to-back penalty |
| `fatigue_3in4` | 1.0 | — | 3-in-4-days penalty |
| `fatigue_4in6` | 1.5 | — | 4-in-6-days penalty |
| `espn_spread_scale` | 0.3 | — | Win-prob edge → implied spread |
| `espn_model_weight` | 0.80 | [0.60, 0.95] | Model weight in ESPN blend |
| `espn_weight` | 0.20 | — | = 1 - espn_model_weight |
| `espn_disagree_damp` | 0.85 | — | Dampening when they disagree |
| `ml_ensemble_weight` | 0.4 | [0.0, 0.6] | ML model weight |
| `ml_disagree_damp` | 0.7 | [0.3, 1.0] | ML dampening on disagreement |
| `ml_disagree_threshold` | 8.0 | — | Spread disagreement threshold |
| `spread_clamp` | 40.0 | — | Max absolute spread |
| `total_min` | 185.0 | — | Minimum total |
| `total_max` | 268.0 | — | Maximum total |

### Persistence
- Global weights: `model_weights` table (`key TEXT PK, value REAL`)
- Per-team overrides: `team_weight_overrides` table (`team_id INTEGER, key TEXT, value REAL`)
- `get_weight_config()` — lazy-loaded, cached singleton
- `load_team_weights(team_id)` — returns `None` if no overrides
- `blend(other)` — averages all fields

---

## 7. XGBoost ML Model

### Hyperparameters (both spread and total models)
```python
n_estimators=500
max_depth=3
learning_rate=0.05
subsample=0.7
colsample_bytree=0.6
min_child_weight=5
reg_alpha=0.5          # L1 regularization
reg_lambda=2.0         # L2 regularization
early_stopping_rounds=20
```

### ~73-88 Features (extracted from `PrecomputedGame`)

**Counting stats** (8 × 3 = 24): `{home|away|diff}_{points|rebounds|assists|steals|blocks|turnovers|oreb|dreb}`

**Shooting efficiency** (3 × 3 = 9): `{home|away|diff}_{ts_pct|fg3_rate|ft_rate}`

**Turnover margin** (3): `{home|away|diff}_to_margin`

**Ratings** (11): `{home|away}_{off_rating|def_rating}`, `{home|away|diff}_net_rating`, `{home|away|diff}_matchup_edge`, `{home|away}_def_factor_raw`

**Pace** (4): `{home|away|avg|diff}_pace`

**Home court** (1): `home_court`

**Fatigue** (4): `{home|away|diff|combined}_fatigue`

**Four Factors edges** (4): `ff_{efg|tov|oreb|fta}_edge`

**Clutch** (5): `{home|away|diff}_clutch_net`, `{home|away}_clutch_efg`

**Hustle** (7): `{home|away|diff}_deflections`, `{home|away}_contested`, `{home|away}_loose_balls`

**Injury context** (9): `{home|away|diff}_{injured_count|injury_ppg_lost|injury_minutes_lost}`

**Season phase** (4): `{home|away|min}_games_played`, `games_played_diff`

**Roster change** (2): `{home|away}_roster_changed`

### Train/Val Split
Time-based (prevents future leakage): `n_val = max(10, int(len(X) * 0.2))`. Minimum: 30 total games.

### SHAP Integration
After training: `shap.TreeExplainer(model)`, `shap_values = explainer.shap_values(X_val)`, stores top 10 features by `mean(|SHAP|)` per model.

### Persistence
- `data/ml_models/spread_model.json` — XGBoost `save_model()`
- `data/ml_models/total_model.json`
- `data/ml_models/model_meta.json` — training metadata (n_train, n_val, n_features, MAEs, feature_cols, trained_at)
- `data/ml_models/feature_columns.json` — ordered feature names

### Inference
- `predict_ml(features)` → `(ml_spread, ml_total, confidence)` where confidence = `n_present / (n_features * 0.6)`
- `predict_ml_with_uncertainty(features)` → adds `spread_std, total_std` from per-tree prediction variance
- Module-level model cache, loaded once via `_ensure_models_loaded()`

---

## 8. Weight Optimization System

### VectorizedGames
Converts `List[PrecomputedGame]` into 34 flat NumPy arrays for ~50-100× faster loss evaluation. The entire prediction formula is replicated in vectorized form (no Python loop).

Arrays: `n`, `home_team_ids`, `away_team_ids`, `actual_spread`, `actual_total`, `home_pts_raw`, `away_pts_raw`, `home_def_factor_raw`, `away_def_factor_raw`, `home_tuning_corr`, `away_tuning_corr`, `home_fatigue`, `away_fatigue`, `home_court`, `to_diff`, `reb_diff`, `home_off`, `away_off`, `home_def`, `away_def`, `ff_efg_edge`, `ff_tov_edge`, `ff_oreb_edge`, `ff_fta_edge`, `clutch_diff`, `hustle_effort_diff`, `combined_deflections`, `avg_pace`, `combined_steals`, `combined_blocks`, `combined_oreb`, `combined_fatigue`, `injury_count_diff`, `injury_ppg_lost_diff`, `injury_minutes_lost_diff`.

### Loss Function
```
loss = spread_MAE × 1.0 + total_MAE × 0.3 + (100 - winner_pct) × 0.1
```
Where `winner_pct` = % of games where predicted winner matches actual. Pushes (±0.5 threshold) count as correct if actual spread ≤ 3.

### Optuna Optimization
- Sampler: TPESampler (Bayesian)
- Direction: minimize
- 12 weights tuned within defined ranges (see WeightConfig table)
- Constraint: `espn_weight = 1 - espn_model_weight`
- Saves to DB only if `best_loss < baseline_loss`
- Callback logs every 25 trials or on new best
- Fallback: random sampling if Optuna not installed

### Per-Team Refinement
For each of 30 teams:
1. Group all games involving team (as home or away)
2. Sort by date, split into train (older) and holdout (last 5 games)
3. Run 100 random trials perturbing 4 weights by ±20%:
   - `def_factor_dampening` [0.25, 0.75]
   - `turnover_margin_mult` [0.15, 0.65]
   - `four_factors_scale` [0.10, 0.60]
   - `pace_mult` [0.08, 0.35]
4. Regressive validation on holdout
5. Keep per-team weights ONLY if `team_loss_holdout < global_loss_holdout` AND not >30% worse
6. Save via `save_team_weights()` or keep global

### Residual Calibration
After optimization: for each bin with ≥5 samples, compute `avg_residual = mean(predicted - actual)`. At inference, subtract this correction. Stored in DB tables.

### Feature Importance Methods

1. **Individual** — Disables each of 14 features one at a time, measures loss delta
2. **Grouped** — Disables 10 feature groups to capture interaction effects
3. **ML/SHAP** — XGBoost (n=100, depth=4, lr=0.1) + `shap.TreeExplainer` mean |SHAP|
4. **FFT Error Analysis** — `np.fft.rfft` on chronological errors, identifies spectral peaks > 2× mean magnitude (weekly/monthly cycles)

---

## 9. Backtesting System

### How Historical Games Are Replayed
1. `get_actual_game_results()` — aggregates `player_stats` by `(game_id, is_home)` to reconstruct game scores. Sanity filter: skip games where either score < 20.
2. Filter by team (optional), sort ascending by date.
3. Winner classification: home spread > 0.5 = "HOME", < -0.5 = "AWAY", else "PUSH".

### Metrics
- **Per-game**: `spread_error`, `total_error`, `home_score_error`, `away_score_error`, `winner_correct`, `spread_within_5`, `total_within_10`
- **Per-team**: `spread_accuracy`, `total_accuracy`, `avg_spread_error`, `avg_total_error`, `wins`, `losses`
- **Overall**: `overall_spread_accuracy` (winner pick %), `overall_total_accuracy` (within-10 %), `total_games`

### Configuration
- `ThreadPoolExecutor(max_workers=4)`, progress every 20 games
- Thread-safe via `_progress_lock`
- Session caches started/stopped for each run

### Cache
- SHA-256 hash of `(model_weights, team_tuning, ml_models/model_meta.json)` → first 16 hex chars
- Files at `data/backtest_cache/bt_{hash}.json`
- TTL: 60 minutes

---

## 10. Autotune System

### Per-Team Scoring Corrections via Grid Search

#### Modes
- **`classic`** — uses ALL historical games for each team
- **`walk_forward`** — uses only the most recent 20 games

#### Algorithm
1. Get all actual game results, filter to team's games
2. Skip first 5 games (insufficient data)
3. For each game: get actual players who played, compute `aggregate_projection()` with dampened defensive factor
4. Compute spread errors (predicted - actual) for home and away separately

#### Grid Search
- Step size: 0.25
- Range: ±10.0 (configurable)
- Objective: minimize composite score

#### Composite Score
```
score = 3.0 × wrong_rate + 1.0 × MAE + 0.25 × P90
```
Where `wrong_rate` = fraction picking wrong winner, `MAE` = mean absolute error, `P90` = 90th percentile error.

#### Correction Formula
```python
confidence = min(1.0, n / 15.0)
home_shift *= strength * confidence
away_shift *= strength * confidence
home_correction = clamp(home_shift, -max_abs_correction, +max_abs_correction)
away_correction = clamp(-away_shift, -max_abs_correction, +max_abs_correction)
```

#### Guardrails
- Acceptance: reject if post-correction composite score ≥ pre-correction
- Min threshold: if |correction| < 1.5, set to 0
- Optional global rollback: if overall backtest accuracy worsens, revert all tuning

Parameters: `strength=0.75`, `min_threshold=1.5`, `max_abs_correction=10.0`

---

## 11. Injury Intelligence

### Play Probability Computation (3-layer blend)

#### Layer 1: League-Wide Rate
From `injury_status_log` where `did_play IS NOT NULL`, grouped by `status_level`.

Default rates (fallback):

| Status | Rate |
|---|---|
| Out | 0.00 |
| Doubtful | 0.10 |
| Questionable | 0.50 |
| GTD | 0.50 |
| Day-To-Day | 0.60 |
| Probable | 0.85 |
| Available | 1.00 |

Confidence: "high" if ≥50 obs, "medium" if ≥15, "low" otherwise.

#### Layer 2: Player-Specific Tendency (70/30 blend)
```python
if player has ≥ 5 observations:
    player_weight = min(0.70, total / 20.0)
    blended = player_rate * player_weight + league_rate * (1 - player_weight)
elif player has some data:
    nudge_weight = total / 10.0
    blended = player_rate * nudge_weight + league_rate * (1 - nudge_weight)
else:
    blended = league_rate
```

#### Layer 3: Injury Keyword Modifier
```python
keyword_modifier = keyword_specific_rate / overall_average_rate
keyword_modifier = 0.5 + keyword_modifier * 0.5  # dampened to ~0.5–1.0+
composite = clamp(blended * keyword_modifier, 0.0, 1.0)
```

### Injury Keywords (37 categories)
rest, personal, suspension, illness, concussion, hamstring, quad, calf, groin, ankle, foot, toe, knee (incl. acl/mcl/meniscus), hip, back, shoulder, elbow, wrist, hand, finger, achilles, thigh, leg, rib, chest, abdomen, neck, eye, sprain, strain, soreness, contusion, fracture, surgery, etc. Default: "other".

### Status Normalization
| Raw Status | Canonical |
|---|---|
| "out", "o" | "Out" |
| "doubtful", "d" | "Doubtful" |
| "questionable", "q" | "Questionable" |
| "probable", "p" | "Probable" |
| contains "day" | "Day-To-Day" |
| "gtd", "game time" | "GTD" |
| anything else | Title-cased, defaults to "Out" |

### Backfill
Cross-references `injury_status_log` (where `did_play IS NULL`) with `player_stats` to determine if a player actually played after being listed. Finds team's first game date strictly after `log_date`, checks if `(player_id, game_date)` exists in played set.

### Injury Impact on Team Projections (in stats_engine.py)
- **Position-based minute redistribution**: `extra_minutes = injured_pos_minutes * (player_mpg / pos_active_mpg)`, capped at 40 - player_mpg. Extra points = `extra_minutes * PPM * 0.85` (85% efficiency)
- **Usage boost**: for high scorers (12+ PPG) when other high scorers (15+ PPG) are out: `usage_boost = injured_ppg * (player_ppg / total_active_high_scorer_ppg) * 0.30`
- **Adjacent position spillover**: `spillover = min(3, injured_minutes * 0.15)`, extra points = `spillover * PPM * 0.75`
- **On/off net rating impact**: `point_impact = net_rating_diff * (on_court_min / 48) * 0.5 * absent_fraction`
- **FT efficiency loss**: `fta_pg * (injured_ft_pct - replacement_ft_pct) / 100 * absent_fraction`
- **Fallback penalties**: `playmaker_penalty = injured_apg * 0.5` (if 6+ APG), `rebounder_penalty = injured_rpg * 0.2` (if 8+ RPG)

---

## 12. Live Prediction (In-Game)

### 3-Signal Blend with Time-Varying Weights

| Game Minute | Pregame Model | Pace Extrapolation | Quarter History |
|---|---|---|---|
| 0 (tip-off) | 0.90 | 0.10 | 0.00 |
| 12 (end Q1) | 0.60 | 0.25 | 0.15 |
| 24 (halftime) | 0.35 | 0.45 | 0.20 |
| 36 (end Q3) | 0.15 | 0.65 | 0.20 |
| 43 (~5 min left) | 0.05 | 0.85 | 0.10 |
| 48 (end reg) | 0.02 | 0.95 | 0.03 |
| >48 (OT) | 0.02 | 0.95 | 0.03 |

Weights linearly interpolated between anchors. If quarter history unavailable, its weight redistributed to pace.

### Signal 1: Pre-game Model
Calls `predict_matchup()` with play-probability-filtered rosters (threshold ≥ 0.3). Cached via `@lru_cache(maxsize=64)`.

### Signal 2: Pace Extrapolation
```python
home_ppm = home_score / minutes_elapsed
away_ppm = away_score / minutes_elapsed
pace_home = home_score + home_ppm * remaining_minutes
pace_away = away_score + away_ppm * remaining_minutes

# Early dampening (first 4 minutes):
alpha = minutes / 4.0
pace_home = alpha * pace_home + (1 - alpha) * historical_ppm * 48
```
Historical PPM fallback: league average `112.0 / 48.0`.

### Signal 3: Quarter-History Lookup
Queries `game_quarter_scores` for games where cumulative score through completed quarters is within ±4 points. Requires ≥5 matching games. Returns `AVG(final_score)`.

### Advisory Signals
- **Over/Under**: "OVER likely" if `blended_total > pregame_total + 4`, "UNDER likely" if `< pregame_total - 4`
- **Spread** (after 6 min): "Home covering" if `margin > pregame_spread + 4`, "Away covering" if `< spread - 4`

---

## 13. Data Sync Pipeline (12 Steps)

The full pipeline orchestrates everything:

| # | Step | Skip Condition |
|---|---|---|
| 1 | Check pipeline state | Never skipped |
| 2 | Load data into memory (7 pandas DataFrames) | Never skipped |
| 3 | Full data sync (6 sub-steps below) | API-level caching |
| 4 | Reload memory if new data | If unchanged |
| 5 | Build injury history (infer from game logs) | If fresh + no new data |
| 6 | Injury intelligence backfill + roster change detection | Never (lightweight) |
| 7 | Autotune all teams (strength=0.75, classic mode) | If fresh + no new data |
| 8 | Train ML models (XGBoost) | If fresh + no new data |
| 9 | Global weight optimization (200 Optuna trials) | If fresh + no new data |
| 10 | Per-team refinement (100 trials per team) | If fresh + no new data |
| 11 | Build residual calibration | If fresh + no new data |
| 12 | Validation backtest | Always runs |

### Full Sync Sub-Steps

| # | Step | Details | Freshness |
|---|---|---|---|
| 1 | Reference data | Teams + players via `nba_api` | Skip if <24hrs + ≥30 teams + >100 players |
| 2 | Player game logs | Per-player `PlayerGameLog` | Per-player 24hr TTL, force-refresh if recent (3 days) |
| 3 | Current injuries | Scrape ESPN/CBS/RotoWire + manual | Always runs |
| 3b | Injury backfill | Cross-ref injury log with game logs | Always (lightweight) |
| 4 | Injury history | Infer from missing games | Skip if same game count + <168hrs |
| 5 | Team metrics | 8 API calls per season | Skip if no new games + <168hrs |
| 6 | Player impact | Estimated metrics + on/off per team | Same as #5 |

### Team Metrics — 8 API Calls

| # | Endpoint | Data |
|---|---|---|
| 1 | `TeamEstimatedMetrics` | GP, W, L, W_PCT, E_OFF/DEF/NET_RATING, E_PACE, E_AST_RATIO, etc. |
| 2 | `LeagueDashTeamStats` (Advanced) | OFF/DEF/NET_RATING, PACE, EFG_PCT, TS_PCT, AST_RATIO, etc. |
| 3 | `LeagueDashTeamStats` (Four Factors) | FF_EFG_PCT, FF_FTA_RATE, FF_TM_TOV_PCT, FF_OREB_PCT + opponent |
| 4 | `LeagueDashTeamStats` (Opponent) | OPP_PTS, OPP_FG_PCT, OPP_FG3_PCT, OPP_FT_PCT |
| 5 | `LeagueDashTeamStats` (Home) | HOME_GP, HOME_W, HOME_L, HOME_PTS, HOME_OPP_PTS |
| 6 | `LeagueDashTeamStats` (Road) | ROAD_GP, ROAD_W, ROAD_L, ROAD_PTS, ROAD_OPP_PTS |
| 7 | `LeagueDashTeamClutch` (Advanced) | CLUTCH_GP/W/L, CLUTCH_NET_RATING, CLUTCH_EFG/TS_PCT |
| 8 | `LeagueHustleStatsTeam` | DEFLECTIONS, LOOSE_BALLS, CONTESTED_SHOTS, CHARGES, SCREEN_ASSISTS |

0.8s sleep between calls. Written to `team_metrics` with `ON CONFLICT DO UPDATE`.

### Pipeline State
Persisted to `data/pipeline_state.json`. Freshness tracked in `sync_meta` table: `(step_name, last_synced_at, game_count_at_sync, last_game_date_at_sync)`.

---

## 14. Stats Engine Constants

| Constant | Value | Purpose |
|---|---|---|
| `TEAM_MINUTES_PER_GAME` | 240.0 | 5 players × 48 min |
| `_DECAY` | 0.9 | Exponential decay per-game |
| Home court fallback | 3.0 | Default HCA |
| HCA clamp | [1.5, 5.0] | |
| Pace fallback | 98.0 | When no metrics |
| Off/Def rating fallback | 110.0 | When no data |
| League avg PPG fallback | 112.0 | |
| Rest bonus (4+ days) | -1.5 | Negative = advantage |
| Rest bonus (3 days) | -1.0 | |
| Same-day penalty | 3.0 | Doubleheader |
| High scorer threshold | 15+ PPG | For injury usage boost |
| Active high scorer threshold | 12+ PPG | For usage redistribution |
| Playmaker threshold | 6+ APG | Fallback injury penalty |
| Playmaker penalty mult | 0.5 | × injured APG |
| Rebounder threshold | 8+ RPG | Fallback injury penalty |
| Rebounder penalty mult | 0.2 | × injured RPG |
| FT volume threshold | 2+ FTA/game | |
| Default replacement FT% | 78.0 | |
| Return-from-injury discount | 3%/game, max 15% (floor 0.85) | |
| Roster change high-impact | 20+ MPG | For `detect_roster_change()` |

### Shooting Efficiency Formulas
```python
ts_pct = PTS / (2 * (FGA + 0.44 * FTA)) * 100
efg_pct = (FGM + 0.5 * FG3M) / FGA * 100
fg3_rate = FG3A / FGA * 100
ft_rate = FTA / FGA * 100
poss = FGA - OREB + TOV + 0.44 * FTA
off_rating = (team_pts / poss) * 100
pace = possessions_pg * (240.0 / team_minutes)
```

### Player Contribution Formula (for `get_team_matchup_stats`)
```python
contribution = ppg * 0.4 + location_ppg * 0.3 + vs_opp_ppg * 0.3
```

### Roster Change Detection
Compares current `players` table vs players who appeared in last 5 game dates. Returns `changed`, `players_added`, `players_removed`, `high_impact` (any player with 20+ MPG).

---

## 15. Web Interface (FastAPI)

### HTML Page Routes

| Method | Path | Description |
|---|---|---|
| GET | `/` | Redirect to `/dashboard` (307) |
| GET | `/dashboard` | DB counts (teams, players, game_logs, injured), 6 sync buttons |
| POST | `/dashboard/sync` | Trigger `full_sync()` |
| POST | `/dashboard/injuries` | Trigger `sync_injuries()` |
| POST | `/dashboard/injury-history` | Trigger `sync_injury_history()` |
| POST | `/dashboard/team-metrics` | Trigger `sync_team_metrics()` |
| POST | `/dashboard/player-impact` | Trigger `sync_player_impact()` |
| GET | `/live` | Live games with recommendations |
| GET | `/players` | All players + injured list + manual injuries |
| POST | `/players/injury/add` | Add manual injury entry |
| POST | `/players/injury/remove` | Remove manual injury |
| GET | `/schedule` | NBA schedule (today + 14 days) |
| GET | `/matchups` | Full prediction: Four Factors/clutch/fatigue/injury, player projections |
| GET | `/accuracy` | Backtest accuracy page |
| GET | `/autotune` | Autotune page |
| POST | `/autotune/clear` | Clear corrections |
| GET | `/admin` | DB path/size |
| POST | `/admin/reset` | Delete & reinitialize DB |
| GET | `/gamecast` | Live game detail |

### SSE Streaming Endpoints

| Path | What it streams |
|---|---|
| `/api/sync/data` | `full_sync()` progress |
| `/api/sync/injuries` | Injury sync progress |
| `/api/sync/injury-history` | Injury history progress |
| `/api/sync/team-metrics` | Team metrics progress |
| `/api/sync/player-impact` | Player impact progress |
| `/api/sync/images` | Image preload progress |
| `/api/backtest` | Backtest progress + `[RESULTS_JSON]` payload |
| `/api/optimize` | Weight optimization progress |
| `/api/calibrate` | Residual calibration progress |
| `/api/feature-importance` | Individual feature importance |
| `/api/grouped-feature-importance` | Grouped feature importance |
| `/api/ml-feature-importance` | ML/SHAP feature importance |
| `/api/fft-analysis` | FFT error pattern analysis |
| `/api/ml-train` | ML training progress + SHAP/gain results |
| `/api/team-refinement` | Per-team refinement progress |
| `/api/continuous-optimize` | Continuous optimization (loops until cancelled) |
| `/api/optimize-all` | Combo optimization (global + per-team) |
| `/api/full-pipeline` | Full 12-step pipeline |
| `/api/gamecast/stream/{game_id}` | Live play-by-play + score + odds (10s poll) |
| `/api/regression/save?name=X` | Save regression baseline |
| `/api/regression/compare?name=X` | Compare against regression baseline |

### REST API Endpoints

| Method | Path | Returns |
|---|---|---|
| POST | `/api/sync/cancel` | `{"status": "cancel_requested"}` |
| POST | `/api/continuous-optimize/cancel` | `{"status": "cancel_requested"}` |
| POST | `/api/pipeline/cancel` | `{"status": "ok"}` |
| POST | `/api/weights/clear` | Clears global + per-team weights |
| GET | `/api/weights` | Current weight config dict |
| GET | `/api/calibration` | Saved residual calibration bins |
| GET | `/api/backtest-cache-age` | `{"age_minutes": float|null}` |
| GET | `/api/gamecast/games` | Today's games with scores/status |
| GET | `/api/gamecast/odds/{game_id}` | Odds: spread, O/U, moneyline, win%, ATS |
| GET | `/api/gamecast/boxscore/{game_id}` | Box score: players + totals |
| GET | `/api/regression/list` | List all saved regression baselines |
| GET | `/api/regression/test-features` | Run ML feature extraction sanity tests |

### Theme
Mobile-first dark theme: CSS vars `--bg-dark: #0f172a`, `--accent: #3b82f6`. Responsive hamburger nav, card/table/form/badge styles. 968 lines.

### 10 Templates
`base.html` (Jinja2 base with 9 nav links), `dashboard.html`, `live.html`, `gamecast.html`, `players.html`, `matchups.html`, `schedule.html`, `accuracy.html`, `autotune.html`, `admin.html`

---

## 16. Desktop Interface (PySide6)

### 10 Tabs in MainWindow
1. **Dashboard** — 4 stat cards, 6 sync buttons + Stop, activity log with color HTML
2. **Live Games** — Auto-refresh 30s, color-coded rows (green=live, purple=final)
3. **Gamecast** — Live game detail, parallel pre-fetch, score/prediction/odds/box/play-by-play, bonus status, 20s/120s polling
4. **Players** — Split view: all players + injured by impact (25+ MPG = KEY, 15+ = ROTATION)
5. **Matchups** — Game selector (today+14d), prediction cards, injury labels with play probabilities, H2H table, player tables with headshots
6. **Schedule** — Schedule table, `game_selected` signal, lazy loading
7. **Accuracy** — 12 QObject workers, summary cards, team accuracy table, predictions table, results table
8. **Autotune** — Team selector, strength/mode/max-correction controls, corrections table
9. **All-Star** — 4 sub-tabs (MVP, 3PT Contest, Rising Stars, Game Winner), scoring models, `BettingTable`, 2026 prefill data
10. **Admin** — DB path/size, delete + reinitialize

### Theme
`GLOBAL_STYLESHEET` (~300 lines QSS, bg=`#0f1923`), dark theme throughout.

### Workers
6 QObject workers: Sync, Injury, InjuryHistory, TeamMetrics, PlayerImpact, Images — each with `start_*_worker()` factory.

12 Accuracy workers: Backtest, Optimizer, Calibration, FeatureImportance, MLFeature, Grouped, FFT, TeamRefine, Combo, Continuous, FullPipeline, MLTrain.

### Notifications
`NotificationBell` (custom painted badge) + `NotificationPanel` (popup, 30 recent, severity colors, mark-all-read). 5-minute injury monitoring loop.

---

## 17. Notification System

- DB-persisted in `notifications` table
- Categories: injury, matchup, insight
- Severities: info, warning, critical
- Listener pattern for real-time UI updates
- Push channels: webhook URL, ntfy topic, BurntToast (Windows toast), plyer (cross-platform)
- Injury monitor: background diff engine polls every 5 minutes, `HIGH_IMPACT_MPG = 20.0` threshold for "critical" severity

---

## 18. Caching Strategy (Multi-Level)

| Level | Mechanism | TTL |
|---|---|---|
| Function | `@functools.lru_cache` (e.g., pregame predictions) | Session |
| Session | `TeamCache`, `SessionCache` in `cache.py` | Configurable |
| Memory Store | 7 pandas DataFrames in singleton | Until `reload()` |
| Precomputed | `PrecomputedGame` list cached in memory + disk pickle | SHA-256 invalidation |
| Backtest | `data/backtest_cache/bt_{hash}.json` | 60 minutes |
| Player Sync | `player_sync_cache` table per player | 24 hours |
| Pipeline State | `data/pipeline_state.json` | Per-step freshness |
| Images | `data/cache/player_photos/`, `team_logos/` | Disk permanent, 0.4s rate limit |

---

## 19. Odds Converter Utilities

```python
american_to_probability(odds):
    if odds > 0: return 100 / (odds + 100)
    if odds <= 0: return |odds| / (|odds| + 100)

probability_to_american(prob):
    if prob > 0.5: return int(-prob * 100 / (1 - prob))
    if prob <= 0.5: return int((1 - prob) * 100 / prob)

expected_value(my_probability, market_odds):
    implied = american_to_probability(market_odds)
    payout = |100 / odds| if negative else odds / 100
    return my_probability * payout - (1 - my_probability)
```

---

## 20. Memory Store (Singleton)

Double-checked locking singleton `InMemoryDataStore`. Loads 7 tables as pandas DataFrames:

| Attribute | Source |
|---|---|
| `player_stats` | `SELECT * FROM player_stats` |
| `teams` | `SELECT * FROM teams` |
| `players` | `SELECT * FROM players` |
| `team_metrics` | `SELECT * FROM team_metrics` |
| `player_impact` | `SELECT * FROM player_impact` |
| `team_tuning` | `SELECT * FROM team_tuning` |
| `injury_history` | `SELECT * FROM injury_history` |

Plus `precomputed_games: list | None` set externally by the pipeline.

Convenience methods: `get_team_abbrs()`, `get_team_list()`, `get_player_stats_for_team()`, `get_team_metrics_dict()`, `get_game_count_and_last_date()`.

---

## 21. PrecomputedGame Dataclass

Zero-DB-access replay structure for optimization. All fields:

`game_date`, `home_team_id`, `away_team_id`, `actual_home_score`, `actual_away_score`, `home_proj` (dict), `away_proj` (dict), `home_court`, `away_def_factor_raw`, `home_def_factor_raw`, `home_tuning_home_corr`, `away_tuning_away_corr`, `home_fatigue_penalty`, `away_fatigue_penalty`, `home_off`, `away_off`, `home_def`, `away_def`, `home_pace`, `away_pace`, `home_ff` (dict), `away_ff` (dict), `home_clutch` (dict), `away_clutch` (dict), `home_hustle` (dict), `away_hustle` (dict), `home_injured_count` (0.0), `away_injured_count` (0.0), `home_injury_ppg_lost` (0.0), `away_injury_ppg_lost` (0.0), `home_injury_minutes_lost` (0.0), `away_injury_minutes_lost` (0.0), `home_return_discount` (1.0), `away_return_discount` (1.0), `home_games_played` (0), `away_games_played` (0), `home_roster_changed` (False), `away_roster_changed` (False).

`precompute_game_data()` builds this list: loads game results, sorts by date, skips teams with < 5 games, fetches rosters/projections/metrics with `as_of_date` to prevent lookahead, caches to memory + disk pickle.

`predict_from_precomputed(g, w)` — identical logic to `predict_matchup()` but operates on this dataclass. Skips residual calibration during optimization.

---

## 22. Example End-to-End Flow

**Scenario**: Lakers (home) vs Celtics (away), today

1. **Roster fetch**: Get current Lakers and Celtics players from `players` table
2. **Injury check**: Scrape ESPN → get "LeBron James: Questionable (knee)". Compute play probability: league 50%, player history 65% (8 obs), keyword "knee" modifier 0.92 → composite ~60%
3. **Player projections**: For each active player (play_prob ≥ 0.3), compute 50/25/25 blended stats over last 10 games with 0.9^n decay. LeBron at 60% weight contributes proportionally.
4. **240-min normalization**: If total projected minutes = 260, scale all stats by 240/260
5. **Injury redistribution**: LeBron's 40% absent fraction → redistribute ~16 minutes to position mates at 85% efficiency, usage boost to AD
6. **Defensive adjustment**: Multiply projected points by opponent's dampened defensive factor
7. **Autotune**: Add per-team corrections (e.g., Lakers +1.2 home, Celtics -0.8 away)
8. **Spread**: Layer home court (+3.5), fatigue (B2B: +2.0), turnovers, rebounds, matchup ratings, Four Factors, clutch (if close), hustle
9. **Total**: Sum bases, pace adjustment, defensive disruption, OREB boost, fatigue penalty
10. **ESPN blend**: 80% our model + 20% ESPN implied spread
11. **ML blend**: 60% model + 40% XGBoost (dampened for early season/disagreement/uncertainty)
12. **Clamp**: spread ±30, total 185–268
13. **Residual calibration**: Subtract historical bias for prediction's bin
14. **Output**: predicted spread -4.2 (Lakers favored), total 223.5, Lakers 113.8, Celtics 109.7, all adjustment breakdowns

---

## 23. Key Design Patterns

- **Singleton** memory store with double-checked locking
- **PrecomputedGame** for zero-DB optimization loops (~50-100× speedup)
- **Vectorized NumPy** loss evaluation (no Python loops in optimizer)
- **TTL caching** at every layer (function/session/disk/DB)
- **Graceful degradation** everywhere (missing data → fallbacks, missing libraries → alternatives)
- **Lookahead prevention** via `as_of_date` throughout stats engine
- **Time-based ML splits** only (never random shuffle)
- **SSE streaming** for all long-running operations
- **Thread-safe** DB access via RLock + FIFO queue
- **Browser-like headers** to avoid API blocks
- **Waterfall scraping** for injury data (ESPN → CBS → RotoWire)
