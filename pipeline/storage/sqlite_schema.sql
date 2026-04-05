CREATE TABLE IF NOT EXISTS games (
    game_id     INTEGER PRIMARY KEY,
    season      TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    date        TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    team_id     INTEGER PRIMARY KEY,
    team_abbr   TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,
    player_name TEXT
);

CREATE TABLE IF NOT EXISTS shots (
    game_id         INTEGER,
    event_id        INTEGER,
    period          INTEGER,
    period_time     TEXT,
    team_id         INTEGER,
    team_abbr       TEXT,
    shot_type_desc  TEXT,
    is_goal         INTEGER,
    x_coord         REAL,
    y_coord         REAL,
    player_id       INTEGER,
    player_name     TEXT,
    goalie_id       INTEGER,
    goalie_name     TEXT,
    shot_type       TEXT,
    is_empty_net    INTEGER,
    strength        TEXT,
    season          TEXT,
    PRIMARY KEY (game_id, event_id),
    FOREIGN KEY (game_id)   REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id)   REFERENCES teams(team_id)
);