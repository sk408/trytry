# API Data Reference

Complete reference of all data sources used in (and available to) this project, with suggestions for improving prediction accuracy.

---

## 1. NBA Official API (`nba_api` library)

Python package: `nba_api` v1.11.3 | Source: `stats.nba.com`

### Currently Used Endpoints

#### `PlayerGameLog` (used in `nba_fetcher.py`)
Per-game stats for a single player across a season.

| Field | Stored? | Column in DB |
|-------|---------|-------------|
| GAME_ID | Yes | `game_id` |
| GAME_DATE | Yes | `game_date` |
| MATCHUP | Parsed | `is_home`, `opponent_abbr` |
| WL (Win/Loss) | **No** | - |
| MIN (minutes) | Yes | `minutes` |
| PTS | Yes | `points` |
| REB | Yes | `rebounds` |
| AST | Yes | `assists` |
| STL | Yes | `steals` |
| BLK | Yes | `blocks` |
| TOV | Yes | `turnovers` |
| FGM / FGA | Yes | `fg_made` / `fg_attempted` |
| FG3M / FG3A | Yes | `fg3_made` / `fg3_attempted` |
| FTM / FTA | Yes | `ft_made` / `ft_attempted` |
| OREB / DREB | Yes | `oreb` / `dreb` |
| PLUS_MINUS | Yes | `plus_minus` |
| FG_PCT | **No** (computable) | - |
| FG3_PCT | **No** (computable) | - |
| FT_PCT | **No** (computable) | - |
| PF (personal fouls) | **No** | - |
| VIDEO_AVAILABLE | **No** (not useful) | - |

#### `CommonTeamRoster` (used in `nba_fetcher.py`)
Current roster for a team.

| Field | Stored? |
|-------|---------|
| PLAYER_ID | Yes |
| PLAYER (name) | Yes |
| POSITION | Yes |
| NUM (jersey) | No |
| HEIGHT | **No** |
| WEIGHT | **No** |
| BIRTH_DATE | **No** |
| AGE | **No** |
| EXP (years experience) | **No** |
| SCHOOL | No |
| HOW_ACQUIRED | **No** |

#### `LeagueGameFinder` (used in `nba_fetcher.py`)
Finds all games matching filters. Used to get played games for schedule.

| Field | Used? |
|-------|-------|
| TEAM_ID | Yes |
| GAME_DATE | Yes |
| MATCHUP | Yes (parsed for home/away) |
| WL | No |
| PTS | No |
| FG_PCT, FG3_PCT, FT_PCT | No |
| REB, AST, STL, BLK, TOV | No |
| PLUS_MINUS | No |

#### `ScoreBoard` (Live, used in `live_scores.py`)
Real-time game scores via `nba_api.live.nba.endpoints.scoreboard`.

| Field | Stored? |
|-------|---------|
| gameId | Yes |
| homeTeam.teamId / awayTeam.teamId | Yes |
| homeTeam.score / awayTeam.score | Yes |
| gameTimeUTC | Yes |
| gameStatusText | Yes |
| period | Yes |
| gameClock | Yes |

### Available But NOT Used Endpoints

#### `TeamEstimatedMetrics` -- **HIGH VALUE**
NBA's own estimated advanced team metrics for the current season.

| Field | Description | Impact |
|-------|-------------|--------|
| E_OFF_RATING | Estimated offensive rating | Direct replacement for hand-calculated off_rating |
| E_DEF_RATING | Estimated defensive rating | Direct replacement for hand-calculated def_rating |
| E_NET_RATING | Estimated net rating (off - def) | Overall team strength metric |
| E_PACE | Estimated pace | Direct replacement for hand-calculated pace |
| E_AST_RATIO | Estimated assist ratio | Ball movement / team play indicator |
| E_OREB_PCT | Estimated offensive rebound % | Second-chance opportunity rate |
| E_DREB_PCT | Estimated defensive rebound % | Opponent second-chance denial |
| E_REB_PCT | Estimated total rebound % | Overall rebounding dominance |
| E_TM_TOV_PCT | Estimated team turnover % | Ball security metric |
| W, L, W_PCT | Win/loss record | Overall team strength |

#### `PlayerEstimatedMetrics` -- **HIGH VALUE**
NBA's own estimated advanced player metrics.

| Field | Description | Impact |
|-------|-------------|--------|
| E_OFF_RATING | Player offensive rating | Individual offensive efficiency |
| E_DEF_RATING | Player defensive rating | Individual defensive impact |
| E_NET_RATING | Player net rating | Overall player value |
| E_AST_RATIO | Player assist ratio | Playmaking contribution |
| E_OREB_PCT | Player offensive rebound % | Individual rebounding impact |
| E_DREB_PCT | Player defensive rebound % | Individual defensive rebounding |
| E_USG_PCT | Player usage percentage | How much of the offense runs through them |
| E_PACE | Player estimated pace | Game speed when player is on floor |

#### `LeagueDashTeamStats` -- **HIGH VALUE**
Comprehensive team stats with flexible filtering.

| Field | Description | Impact |
|-------|-------------|--------|
| GP, W, L, W_PCT | Record | Team strength baseline |
| FGM/FGA/FG_PCT | Team shooting | Scoring efficiency |
| FG3M/FG3A/FG3_PCT | Team 3PT shooting | 3PT volume and accuracy |
| FTM/FTA/FT_PCT | Team free throws | FT efficiency |
| OREB/DREB/REB | Team rebounds | Board control |
| AST | Team assists | Ball movement |
| TOV/STL | Turnovers and steals | Possession differential |
| BLK/BLKA | Blocks given/received | Rim protection metric |
| PF/PFD | Fouls committed/drawn | Foul trouble + FT generation |
| PTS | Points per game | Baseline scoring |
| PLUS_MINUS | Plus/minus | Net performance |
| *MeasureType=Advanced* | OFF_RATING, DEF_RATING, NET_RATING, PACE, PIE, AST_PCT, AST_TO, AST_RATIO, OREB_PCT, DREB_PCT, REB_PCT, TM_TOV_PCT, EFG_PCT, TS_PCT | Full advanced metrics suite |
| *MeasureType=Opponent* | Same fields but for opponents | Defensive context |
| *MeasureType=Four Factors* | EFG_PCT, FTA_RATE, TM_TOV_PCT, OREB_PCT | "Four Factors" of winning |
| **Filterable by**: | Location (Home/Road), Outcome (W/L), LastNGames, DateRange, OpponentTeamID, Conference, Division | Situational stats |

#### `BoxScoreAdvancedV3` -- **MEDIUM VALUE**
Per-game advanced box score for a specific game.

| Field | Description | Impact |
|-------|-------------|--------|
| offensiveRating / defensiveRating | Player off/def rating for that game | Per-game efficiency context |
| netRating | Player net rating | Per-game value |
| usagePercentage | How much of offense through player | Workload distribution |
| trueShootingPercentage | True shooting % | Shooting efficiency |
| effectiveFieldGoalPercentage | eFG% | Shooting efficiency (incl. 3PT value) |
| assistPercentage | AST% | Playmaking share |
| reboundPercentage | REB% | Rebounding share |
| turnoverRatio | TOV ratio | Ball security |
| pace / possessions | Game pace and total possessions | Tempo context |
| PIE | Player Impact Estimate | Overall game impact |

#### `LeagueDashLineups` -- **MEDIUM VALUE**
Lineup-level performance stats.

| Field | Description | Impact |
|-------|-------------|--------|
| GROUP_NAME (5 players) | Specific 5-man lineup | Lineup synergy data |
| GP, W, L, MIN | Lineup usage | How often lineup plays |
| All standard stats (PTS, REB, etc.) | Per-lineup performance | Which combos work |
| PLUS_MINUS | Lineup net impact | Key lineups identification |

#### `TeamPlayerOnOffDetails` / `TeamPlayerOnOffSummary` -- **HIGH VALUE**
Player on/off court impact.

| Field | Description | Impact |
|-------|-------------|--------|
| On-court vs off-court splits | Team performance with/without each player | True player value beyond box score |
| Net rating differential | How much better/worse team is with player | Critical for injury impact modeling |

#### `PlayerDashboardByLastNGames` -- **MEDIUM VALUE**
Player stats split by recency.

| Field | Description | Impact |
|-------|-------------|--------|
| Last 5 / Last 10 / Last 15 / Last 20 games | Pre-calculated recency splits | Current form without manual weighting |

#### `LeagueStandings` / `LeagueStandingsV3` -- **LOW-MEDIUM VALUE**
Full conference/division standings.

| Field | Description | Impact |
|-------|-------------|--------|
| W, L, W_PCT | Overall record | Team strength indicator |
| HOME_RECORD / ROAD_RECORD | Home/away splits | HCA calculation |
| L10 (last 10 games) | Recent form | Momentum indicator |
| STREAK | Current streak | Momentum indicator |
| OPP_PTS_PG | Points allowed per game | Defensive strength |
| PTS_PG | Points scored per game | Offensive strength |

#### Other Noteworthy Endpoints

| Endpoint | What It Provides | Value |
|----------|-----------------|-------|
| `SynergyPlayTypes` | Play type efficiency (ISO, PnR, transition, etc.) | Could reveal matchup-specific advantages |
| `LeagueDashPtTeamDefend` | Defensive tracking data (contested shots, etc.) | Defensive quality beyond blocks/steals |
| `LeagueDashPlayerClutch` | Clutch performance stats (last 5 min, score within 5) | Late-game prediction accuracy |
| `GameRotation` | Player rotation/substitution patterns | Minute distribution patterns |
| `PlayerVsPlayer` | Head-to-head individual matchup data | Direct matchup impact |
| `BoxScoreHustleV2` | Hustle stats (loose balls, deflections, charges) | Effort/intangible metrics |
| `LeagueHustleStatsPlayer/Team` | Season hustle stats | Consistent effort measurement |
| `WinProbabilityPBP` | Play-by-play win probability | Game flow modeling |

---

## 2. ESPN API (used in `gamecast.py`)

Base URL: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/`

### Currently Used

#### Scoreboard (`/scoreboard`)
| Field | Used? |
|-------|-------|
| game_id, teams, scores | Yes |
| status (pre/in/post) | Yes |
| period, clock | Yes |
| start_time | Yes |
| **venue** | **No** |
| **attendance** | **No** |
| **broadcasts** | **No** |
| **odds (inline)** | **No** (fetched separately) |

#### Summary (`/summary?event={id}`)
| Field | Used? |
|-------|-------|
| pickcenter (odds) | Yes -- spread, O/U, moneyline, ATS record |
| winprobability | Yes -- live win % |
| plays (play-by-play) | Yes |
| boxscore | Yes -- player stats |
| leaders | Yes -- game leaders |
| **predictor** | **No** -- ESPN's own pre-game predictions |
| **standings** | **No** -- team standings context |
| **seasonseries** | **No** -- head-to-head record this season |
| **againstTheSpread** | **No** -- team ATS performance data |
| **injuries** (per-game) | **No** -- game-specific injury report |
| **news** | **No** (not useful for predictions) |
| **videos** | **No** (not useful) |

### Available But NOT Used ESPN Endpoints

| Endpoint | URL Pattern | Data Available |
|----------|-------------|---------------|
| **Standings** | `/standings` | Full conference standings, records, streaks |
| **Teams** | `/teams` | Team details, logos, records |
| **Team Stats** | `/teams/{id}/statistics` | Seasonal team stats |
| **Team Schedule** | `/teams/{id}/schedule` | Full team schedule with results |
| **Rankings** | `/rankings` | Power rankings |

---

## 3. NBA CDN (used in `nba_fetcher.py`)

#### Schedule JSON
URL: `https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json`

| Field | Used? |
|-------|-------|
| gameDateEst / gameDateTimeEst | Yes |
| homeTeam / awayTeam (tricode, name, city) | Yes |
| gameId | Yes |
| arenaName | Yes |
| **arenaCity / arenaState** | **No** |
| **broadcasters** | **No** |
| **seriesText** | **No** (playoff context) |
| **ifNecessary** | **No** |

---

## 4. Injury Scraping (used in `injury_scraper.py`)

| Source | Data Retrieved |
|--------|---------------|
| ESPN (`/nba/injuries`) | Player name, team, status, injury type, update |
| CBS Sports (`/nba/injuries/`) | Player name, team, position, injury, status |
| RotoWire (`/basketball/injury-report.php`) | Player name, team, status, injury |

**Not captured**: Expected return date, severity grade, game-time decisions.

---

## 5. Summary: What's Currently Stored in DB

| Table | Key Columns | Source |
|-------|-------------|--------|
| `teams` | team_id, name, abbreviation, conference | nba_api static |
| `players` | player_id, name, team_id, position, is_injured, injury_note | nba_api roster + injury scraping |
| `player_stats` | All per-game stats (21 fields per game) | nba_api PlayerGameLog |
| `predictions` | predicted/actual spread and total | Internal |
| `live_games` | Live scores and status | nba_api live ScoreBoard |
| `injury_history` | Inferred missed games | Internal analysis |
| `team_tuning` | Autotune corrections | Internal autotune |
| `player_sync_cache` | Sync timestamps | Internal caching |

---

## 6. Recommendations: Data to Incorporate

### Priority 1 -- High Impact, Low Effort

| Data | Source | Why It Helps | Implementation |
|------|--------|--------------|----------------|
| **NBA Team Estimated Metrics** (Off/Def Rating, Pace, Net Rating) | `TeamEstimatedMetrics` | Replace hand-calculated ratings with NBA's own, which account for lineup data and possession-level tracking. Much more accurate than aggregating player game logs. | One API call per season. Store in a new `team_advanced_stats` table. Use in `predict_matchup()` instead of `get_offensive_rating()` / `get_defensive_rating()`. |
| **Team W/L record and home/road splits** | `LeagueDashTeamStats` with `Location=Home` / `Location=Road` | Currently the project reconstructs records from player stats (which caused the Chicago 1-25 bug). Official records are authoritative. | One API call. Use for matchup display and as a team strength factor. |
| **Back-to-back detection** | `scheduleLeagueV2.json` (already fetched) | Currently `is_back_to_back` must be passed manually. Auto-detect from schedule data. | Parse schedule to check if team played yesterday. |
| **Personal fouls (PF) from PlayerGameLog** | `PlayerGameLog` (WL, PF already available) | PF data enables foul trouble prediction. Players in foul trouble play fewer minutes. Free throw rate differential is a significant scoring factor. | Add `pf` column to `player_stats`, fetch with existing game log call (no new API call needed). |

### Priority 2 -- High Impact, Medium Effort

| Data | Source | Why It Helps | Implementation |
|------|--------|--------------|----------------|
| **Player on/off court impact** | `TeamPlayerOnOffSummary` | Tells you how much a team's performance changes with a specific player on vs off the court. Critical for accurate injury modeling -- current injury penalties are fixed heuristics. | Sync once per day per team. Store net rating differential. Replace fixed injury penalties with data-driven values. |
| **Clutch stats** | `LeagueDashPlayerClutch` or `LeagueDashTeamClutch` | Close games are decided in crunch time. Teams/players with strong clutch stats outperform spread in tight games. | Sync once daily. Factor into spread predictions for projected close games. |
| **ESPN pre-game predictor** | ESPN Summary API `predictor` field | ESPN's own ML-based pre-game prediction. Can be used as a consensus/ensemble signal or sanity check. | Already fetching Summary API -- just parse the additional field. |
| **Win/Loss from PlayerGameLog** | `WL` field (already in API response, not stored) | Direct W/L per player-game. Enables accurate team record calculations without aggregation bugs. | Add `wl` column to `player_stats`, capture from existing data. |

### Priority 3 -- Medium Impact, Higher Effort

| Data | Source | Why It Helps | Implementation |
|------|--------|--------------|----------------|
| **Lineup data** | `LeagueDashLineups` | Identifies which 5-man combos are effective. When key players are injured, helps predict which backup lineups will be used and how they perform. | Sync periodically (data changes slowly). New table for lineup performance. Cross-reference with available players to estimate lineup quality. |
| **Player Usage Rate (USG%)** | `PlayerEstimatedMetrics` or `BoxScoreAdvancedV3` | Shows how much of the offense flows through each player. When a high-usage player is out, remaining players' scoring shares change non-linearly. Current injury model uses simple minute redistribution. | Sync with player metrics. Use USG% to model scoring redistribution when players are injured. |
| **Four Factors** | `LeagueDashTeamStats` with `MeasureType=Four Factors` | The "Four Factors of Basketball Success" (eFG%, turnover %, offensive rebound %, FT rate) are the strongest predictors of winning. Currently approximated from player stats. | One API call. Official team-level Four Factors are more reliable than player-aggregated estimates. |
| **Opponent team stats** | `LeagueDashTeamStats` with `MeasureType=Opponent` | Shows what opponents shoot/score against each team. Better than computing `get_opponent_defensive_factor()` from raw game data. | One API call. Direct measure of defensive impact. |
| **Rest days / schedule density** | Computed from `scheduleLeagueV2.json` | Not just back-to-backs: 3 games in 4 nights, long road trips, timezone travel, altitude (Denver). All affect performance. | Parse schedule for game density patterns. Weight fatigue accordingly. |
| **Hustle stats** | `BoxScoreHustleV2` / `LeagueHustleStatsPlayer` | Loose balls, deflections, contested shots, charges drawn. These "effort" metrics correlate with defensive performance and predict underdog upsets. | New sync job. Factor into defensive efficiency calculations. |

### Priority 4 -- Lower Impact or Experimental

| Data | Source | Why It Helps |
|------|--------|--------------|
| Height/weight from roster | `CommonTeamRoster` | Size matchup advantages (e.g., small-ball vs. big lineups) |
| Player age/experience | `CommonTeamRoster` | Young teams are less consistent; veteran teams perform better in high-pressure situations |
| ESPN ATS records | ESPN Summary API `againstTheSpread` | Historical ATS performance reveals market inefficiencies |
| Shot chart data | `ShotChartDetail` | Shot zone efficiency for matchup-specific adjustments |
| Play type synergy | `SynergyPlayTypes` | Which play types each team excels at and which they struggle to defend |

---

## 7. Quick Wins (no new API calls needed)

These improvements use data that's **already being fetched** but not stored or used:

1. **Store `WL` (win/loss) from PlayerGameLog** -- already in API response, just not captured. Enables accurate record computation.
2. **Store `PF` (personal fouls) from PlayerGameLog** -- already in API response. Enables foul trouble analysis and free throw rate modeling.
3. **Auto-detect back-to-backs from schedule** -- schedule data is already fetched. Just check if a team played yesterday.
4. **Use ESPN `predictor` from Summary API** -- already calling this endpoint for odds. Just parse the additional field for a consensus signal.
5. **Use ESPN `seasonseries` from Summary API** -- already calling this endpoint. Provides authoritative H2H record.
