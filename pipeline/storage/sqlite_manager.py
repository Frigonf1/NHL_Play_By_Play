import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "sqlite_schema.sql"

def get_connection(db_path: Path) -> sqlite3.Connection:
    """Opens a connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def initialize_db(db_path: Path):
    """Creates tables from schema if they don't exist."""
    conn = get_connection(db_path)
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.close()

def insert_games(games_df, db_path: Path):
    conn = get_connection(db_path)
    rows = list(games_df.itertuples(index=False, name=None))
    conn.executemany("""
        INSERT OR IGNORE INTO games (game_id, season, home_team_id, away_team_id, date)
        VALUES (?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()

def insert_teams(teams_df, db_path: Path):
    conn = get_connection(db_path)
    rows = list(teams_df.itertuples(index=False, name=None))
    conn.executemany("""
        INSERT OR IGNORE INTO teams (team_id, team_abbr)
        VALUES (?, ?)
    """, rows)
    conn.commit()
    conn.close()

def insert_players(players_df, db_path: Path):
    conn = get_connection(db_path)
    rows = list(players_df.itertuples(index=False, name=None))
    conn.executemany("""
        INSERT OR IGNORE INTO players (player_id, player_name)
        VALUES (?, ?)
    """, rows)
    conn.commit()
    conn.close()

def insert_shots(shots_df, db_path: Path):
    conn = get_connection(db_path)
    rows = list(shots_df.itertuples(index=False, name=None))
    conn.executemany("""
        INSERT OR IGNORE INTO shots (
            game_id, event_id, period, period_time,
            team_id, team_abbr, shot_type_desc, is_goal,
            x_coord, y_coord, player_id, player_name,
            goalie_id, goalie_name, shot_type, is_empty_net,
            strength, season
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shots_game_id ON shots (game_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shots_player_id ON shots (player_id)")
    conn.commit()
    conn.close()