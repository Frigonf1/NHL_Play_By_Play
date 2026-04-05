from pipeline.ingestion.download_data import download_data, create_shots_dataframe
from pipeline.storage.sqlite_manager import initialize_db, insert_games, insert_teams, insert_players, insert_shots
from nhlpy import NHLClient
from pathlib import Path
import json
import time
import pandas as pd
import sqlite3

if __name__ == "__main__":

    # Extract team abbreviations if not already done, otherwise read from a file or database
    db_path = Path("data/processed/sqlite/nhl_analytics.db")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT team_abbr FROM teams")
        rows = cursor.fetchall()
        if rows:
            teams_abbr = [row[0] for row in rows]
            print(teams_abbr)

        else:
            nhl_client = NHLClient()
            teams = nhl_client.teams.teams()
            teams_abbr = [team["abbr"] for team in teams]
            print(teams_abbr)

    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    shots_list = []

    unique_games = set()
    unique_players = set()
    unique_teams = set()

    for year in years:
        for team_abbr in teams_abbr:
            save_path = Path(f"data/raw/{year}_{year+1}/nhl_play_by_play_{team_abbr}_{year}_{year+1}.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Processing team: {team_abbr}")
            download_data(team_abbr, year, save_path, db_path)
            # time.sleep(1)  # Add a delay to avoid hitting API rate limits

            with open(save_path, "r", encoding="utf-8") as f:
                all_game_data = json.load(f)

            shots_df = create_shots_dataframe(all_game_data, team_abbr, year)
            shots_list.append(shots_df)

        
            for game in all_game_data:

                # Store unique game info
                unique_games.add((
                    game.get("id"),
                    f"{str(game.get('season', ''))[:4]}-{str(game.get('season', ''))[4:]}",
                    game.get("homeTeam", {}).get("id"),
                    game.get("awayTeam", {}).get("id"),
                    game.get("gameDate", "")
                ))

                # Store unique teams
                for side in ["homeTeam", "awayTeam"]:
                    t = game.get(side, {})
                    unique_teams.add((t.get("id"), t.get("abbrev")))

                # Store unique players from rosters
                for spot in game.get("rosterSpots", []):
                    pid = spot.get("playerId")
                    first = (spot.get("firstName", {}) or {}).get("default", "")
                    last = (spot.get("lastName", {}) or {}).get("default", "")
                    if pid:
                        unique_players.add((pid, f"{first} {last}".strip()))

    # Convert sets to DataFrames and save (Set automatically handled duplicates)
    games_df = pd.DataFrame(list(unique_games), columns=["game_id", "season", "home_team_id", "away_team_id", "date"]) 
    teams_df = pd.DataFrame(list(unique_teams), columns=["team_id", "team_abbr"]) 
    players_df = pd.DataFrame(list(unique_players), columns=["player_id", "player_name"]) 

    insert_games(games_df, db_path)
    insert_teams(teams_df, db_path)
    insert_players(players_df, db_path)
    
    all_shots_df = pd.concat(shots_list, ignore_index=True)
    all_shots_df = all_shots_df.drop_duplicates(subset=["game_id", "event_id"], keep="first")
    insert_shots(all_shots_df, db_path)
