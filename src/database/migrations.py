from __future__ import annotations

import sqlite3

from .db import get_conn


SCHEMA = """
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
    -- Rebound breakdown (NBA specific)
    oreb REAL DEFAULT 0,
    dreb REAL DEFAULT 0,
    -- Impact rating
    plus_minus REAL DEFAULT 0,
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

CREATE INDEX IF NOT EXISTS idx_player_stats_player_date
    ON player_stats(player_id, game_date DESC);

CREATE INDEX IF NOT EXISTS idx_player_stats_matchup
    ON player_stats(opponent_team_id, game_date DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_matchup
    ON predictions(home_team_id, away_team_id, game_date);

-- Historical injury tracking (inferred from game logs)
CREATE TABLE IF NOT EXISTS injury_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    was_out INTEGER NOT NULL DEFAULT 1,
    avg_minutes REAL,  -- Player's avg minutes before this game
    reason TEXT,       -- 'inferred' or 'reported'
    UNIQUE(player_id, game_date),
    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_injury_history_team_date
    ON injury_history(team_id, game_date);

CREATE INDEX IF NOT EXISTS idx_injury_history_player
    ON injury_history(player_id, game_date);

-- Track when player logs were last synced (for caching)
CREATE TABLE IF NOT EXISTS player_sync_cache (
    player_id INTEGER PRIMARY KEY,
    last_synced_at TEXT NOT NULL,
    games_synced INTEGER DEFAULT 0,
    latest_game_date DATE
);

CREATE INDEX IF NOT EXISTS idx_player_sync_cache_date
    ON player_sync_cache(last_synced_at);
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

    # Player columns
    if not has_column("players", "is_injured"):
        conn.execute("ALTER TABLE players ADD COLUMN is_injured INTEGER NOT NULL DEFAULT 0;")
    if not has_column("players", "injury_note"):
        conn.execute("ALTER TABLE players ADD COLUMN injury_note TEXT;")
    
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
