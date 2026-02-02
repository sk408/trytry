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
    points REAL NOT NULL,
    rebounds REAL NOT NULL,
    assists REAL NOT NULL,
    minutes REAL NOT NULL,
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
        DELETE FROM predictions;
        DELETE FROM player_stats;
        DELETE FROM players;
        DELETE FROM teams;
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

    if not has_column("players", "is_injured"):
        conn.execute("ALTER TABLE players ADD COLUMN is_injured INTEGER NOT NULL DEFAULT 0;")
    if not has_column("players", "injury_note"):
        conn.execute("ALTER TABLE players ADD COLUMN injury_note TEXT;")
