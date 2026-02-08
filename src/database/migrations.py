from __future__ import annotations

import sqlite3

from .db import get_conn


SCHEMA = """
-- ============ CORE TABLES ============

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    abbreviation TEXT NOT NULL UNIQUE,
    conference TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    position TEXT,
    is_injured INTEGER NOT NULL DEFAULT 0,
    injury_note TEXT,
    -- Roster context (from CommonTeamRoster)
    height TEXT,
    weight TEXT,
    age INTEGER,
    experience INTEGER,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS player_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    opponent_team_id INTEGER NOT NULL,
    is_home INTEGER NOT NULL,
    game_date DATE NOT NULL,
    game_id TEXT,
    -- Basic stats
    points REAL NOT NULL,
    rebounds REAL NOT NULL,
    assists REAL NOT NULL,
    minutes REAL NOT NULL,
    -- Defensive stats
    steals REAL DEFAULT 0,
    blocks REAL DEFAULT 0,
    turnovers REAL DEFAULT 0,
    -- Shooting stats (made/attempted)
    fg_made INTEGER DEFAULT 0,
    fg_attempted INTEGER DEFAULT 0,
    fg3_made INTEGER DEFAULT 0,
    fg3_attempted INTEGER DEFAULT 0,
    ft_made INTEGER DEFAULT 0,
    ft_attempted INTEGER DEFAULT 0,
    -- Rebound breakdown
    oreb REAL DEFAULT 0,
    dreb REAL DEFAULT 0,
    -- Impact rating
    plus_minus REAL DEFAULT 0,
    -- NEW: Win/Loss result and personal fouls (from PlayerGameLog)
    win_loss TEXT,           -- 'W' or 'L'
    personal_fouls REAL DEFAULT 0,
    UNIQUE(player_id, opponent_team_id, game_date),
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    FOREIGN KEY (opponent_team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    predicted_spread REAL NOT NULL,
    predicted_total REAL NOT NULL,
    actual_spread REAL,
    actual_total REAL,
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS live_games (
    game_id TEXT PRIMARY KEY,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    start_time_utc TEXT,
    status TEXT,
    period INTEGER,
    clock TEXT,
    home_score INTEGER,
    away_score INTEGER,
    last_updated TEXT,
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

-- ============ INDEXES (core tables) ============

CREATE INDEX IF NOT EXISTS idx_player_stats_player_date
    ON player_stats(player_id, game_date DESC);

CREATE INDEX IF NOT EXISTS idx_player_stats_matchup
    ON player_stats(opponent_team_id, game_date DESC);

CREATE INDEX IF NOT EXISTS idx_player_stats_game_id
    ON player_stats(game_id, is_home);

CREATE INDEX IF NOT EXISTS idx_predictions_matchup
    ON predictions(home_team_id, away_team_id, game_date);

-- ============ INJURY TRACKING ============

CREATE TABLE IF NOT EXISTS injury_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    was_out INTEGER NOT NULL DEFAULT 1,
    avg_minutes REAL,
    reason TEXT,
    UNIQUE(player_id, game_date),
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_injury_history_team_date
    ON injury_history(team_id, game_date);

CREATE INDEX IF NOT EXISTS idx_injury_history_player
    ON injury_history(player_id, game_date);

-- ============ SYNC CACHE ============

CREATE TABLE IF NOT EXISTS player_sync_cache (
    player_id INTEGER PRIMARY KEY,
    last_synced_at TEXT NOT NULL,
    games_synced INTEGER DEFAULT 0,
    latest_game_date DATE
);

CREATE INDEX IF NOT EXISTS idx_player_sync_cache_date
    ON player_sync_cache(last_synced_at);

-- ============ AUTOTUNE ============

CREATE TABLE IF NOT EXISTS team_tuning (
    team_id INTEGER PRIMARY KEY,
    home_pts_correction REAL DEFAULT 0.0,
    away_pts_correction REAL DEFAULT 0.0,
    games_analyzed INTEGER DEFAULT 0,
    avg_spread_error_before REAL DEFAULT 0.0,
    avg_total_error_before REAL DEFAULT 0.0,
    last_tuned_at TEXT,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

-- ============ TEAM ADVANCED METRICS (new) ============
-- Consolidated team-level metrics from multiple NBA API endpoints.
-- One row per team per season.  Populated by sync_team_metrics().

CREATE TABLE IF NOT EXISTS team_metrics (
    team_id INTEGER NOT NULL,
    season TEXT NOT NULL,

    -- Record
    gp INTEGER DEFAULT 0,
    w INTEGER DEFAULT 0,
    l INTEGER DEFAULT 0,
    w_pct REAL DEFAULT 0.0,

    -- From TeamEstimatedMetrics (NBA tracking-based)
    e_off_rating REAL,
    e_def_rating REAL,
    e_net_rating REAL,
    e_pace REAL,
    e_ast_ratio REAL,
    e_oreb_pct REAL,
    e_dreb_pct REAL,
    e_reb_pct REAL,
    e_tm_tov_pct REAL,

    -- From LeagueDashTeamStats (Advanced)
    off_rating REAL,
    def_rating REAL,
    net_rating REAL,
    pace REAL,
    efg_pct REAL,
    ts_pct REAL,
    ast_ratio REAL,
    ast_to REAL,
    oreb_pct REAL,
    dreb_pct REAL,
    reb_pct REAL,
    tm_tov_pct REAL,
    pie REAL,

    -- Four Factors (team offense)
    ff_efg_pct REAL,
    ff_fta_rate REAL,
    ff_tm_tov_pct REAL,
    ff_oreb_pct REAL,

    -- Four Factors (opponent / defensive forcing)
    opp_efg_pct REAL,
    opp_fta_rate REAL,
    opp_tm_tov_pct REAL,
    opp_oreb_pct REAL,

    -- Opponent stats (what opponents score/shoot against us)
    opp_pts REAL,
    opp_fg_pct REAL,
    opp_fg3_pct REAL,
    opp_ft_pct REAL,

    -- Clutch stats (last 5 min, score within 5)
    clutch_gp INTEGER DEFAULT 0,
    clutch_w INTEGER DEFAULT 0,
    clutch_l INTEGER DEFAULT 0,
    clutch_net_rating REAL,
    clutch_efg_pct REAL,
    clutch_ts_pct REAL,

    -- Hustle stats
    deflections REAL,
    loose_balls_recovered REAL,
    contested_shots REAL,
    charges_drawn REAL,
    screen_assists REAL,

    -- Home / Road splits
    home_gp INTEGER DEFAULT 0,
    home_w INTEGER DEFAULT 0,
    home_l INTEGER DEFAULT 0,
    home_pts REAL,
    home_opp_pts REAL,
    road_gp INTEGER DEFAULT 0,
    road_w INTEGER DEFAULT 0,
    road_l INTEGER DEFAULT 0,
    road_pts REAL,
    road_opp_pts REAL,

    last_synced_at TEXT,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

-- ============ PLAYER IMPACT METRICS (new) ============
-- On/off court impact + player estimated metrics.
-- One row per player per season.  Populated by sync_player_impact().

CREATE TABLE IF NOT EXISTS player_impact (
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    season TEXT NOT NULL,

    -- On/Off court splits (from TeamPlayerOnOffSummary)
    on_court_off_rating REAL,
    on_court_def_rating REAL,
    on_court_net_rating REAL,
    off_court_off_rating REAL,
    off_court_def_rating REAL,
    off_court_net_rating REAL,
    net_rating_diff REAL,          -- on_net - off_net (positive = team better WITH player)
    on_court_minutes REAL,

    -- Player Estimated Metrics (from PlayerEstimatedMetrics)
    e_usg_pct REAL,
    e_off_rating REAL,
    e_def_rating REAL,
    e_net_rating REAL,
    e_pace REAL,
    e_ast_ratio REAL,
    e_oreb_pct REAL,
    e_dreb_pct REAL,

    last_synced_at TEXT,
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_player_impact_team
    ON player_impact(team_id, season);

-- ============ QUARTER SCORES (for live prediction history) ============
-- Stores per-quarter scoring from ESPN linescores.
-- One row per team per game.  Populated opportunistically when viewing gamecast.

CREATE TABLE IF NOT EXISTS game_quarter_scores (
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    q1 INTEGER,
    q2 INTEGER,
    q3 INTEGER,
    q4 INTEGER,
    ot INTEGER DEFAULT 0,
    final_score INTEGER,
    game_date TEXT,
    is_home INTEGER DEFAULT 0,
    PRIMARY KEY (game_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_quarter_scores_team_date
    ON game_quarter_scores(team_id, game_date);
"""


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _ensure_columns(conn)
        conn.commit()


def clear_all(conn: sqlite3.Connection) -> None:
    """Dangerous: wipe all tables; useful for full refreshes."""
    conn.executescript(
        """
        DELETE FROM team_tuning;
        DELETE FROM player_sync_cache;
        DELETE FROM injury_history;
        DELETE FROM predictions;
        DELETE FROM player_stats;
        DELETE FROM players;
        DELETE FROM teams;
        DELETE FROM live_games;
        """
    )
    conn.commit()


def ensure_columns() -> None:
    """Public helper to ensure new columns exist on older DBs."""
    with get_conn() as conn:
        _ensure_columns(conn)
        conn.commit()


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Backfill columns that may be missing from older DBs."""
    def has_column(table: str, column: str) -> bool:
        cur = conn.execute(f"PRAGMA table_info({table});")
        return any(row[1] == column for row in cur.fetchall())

    def has_table(table: str) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (table,),
        )
        return cur.fetchone() is not None

    # Player columns
    if not has_column("players", "is_injured"):
        conn.execute("ALTER TABLE players ADD COLUMN is_injured INTEGER NOT NULL DEFAULT 0;")
    if not has_column("players", "injury_note"):
        conn.execute("ALTER TABLE players ADD COLUMN injury_note TEXT;")
    if not has_column("players", "height"):
        conn.execute("ALTER TABLE players ADD COLUMN height TEXT;")
    if not has_column("players", "weight"):
        conn.execute("ALTER TABLE players ADD COLUMN weight TEXT;")
    if not has_column("players", "age"):
        conn.execute("ALTER TABLE players ADD COLUMN age INTEGER;")
    if not has_column("players", "experience"):
        conn.execute("ALTER TABLE players ADD COLUMN experience INTEGER;")

    # Extended player_stats columns
    if not has_column("player_stats", "game_id"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN game_id TEXT;")
    if not has_column("player_stats", "steals"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN steals REAL DEFAULT 0;")
    if not has_column("player_stats", "blocks"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN blocks REAL DEFAULT 0;")
    if not has_column("player_stats", "turnovers"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN turnovers REAL DEFAULT 0;")
    if not has_column("player_stats", "fg_made"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN fg_made INTEGER DEFAULT 0;")
    if not has_column("player_stats", "fg_attempted"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN fg_attempted INTEGER DEFAULT 0;")
    if not has_column("player_stats", "fg3_made"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN fg3_made INTEGER DEFAULT 0;")
    if not has_column("player_stats", "fg3_attempted"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN fg3_attempted INTEGER DEFAULT 0;")
    if not has_column("player_stats", "ft_made"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN ft_made INTEGER DEFAULT 0;")
    if not has_column("player_stats", "ft_attempted"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN ft_attempted INTEGER DEFAULT 0;")
    if not has_column("player_stats", "oreb"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN oreb REAL DEFAULT 0;")
    if not has_column("player_stats", "dreb"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN dreb REAL DEFAULT 0;")
    if not has_column("player_stats", "plus_minus"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN plus_minus REAL DEFAULT 0;")
    if not has_column("player_stats", "win_loss"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN win_loss TEXT;")
    if not has_column("player_stats", "personal_fouls"):
        conn.execute("ALTER TABLE player_stats ADD COLUMN personal_fouls REAL DEFAULT 0;")
