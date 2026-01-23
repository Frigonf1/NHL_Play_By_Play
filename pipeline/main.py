from download_data import download_data, create_shots_dataframe 
from nhlpy import NHLClient
from pathlib import Path
import json
import time

if __name__ == "__main__":

    client = NHLClient()
                                                                                                               
    teams = client.teams.teams()

    # Extract team abbreviations
    teams_abbr = [team['abbr'] for team in teams]
    print(teams_abbr)

    years = [2023, 2024]

    for year in years:
        for team_abbr in teams_abbr:
            save_path = Path(f"data/{year}_{year+1}/nhl_play_by_play_{team_abbr}_{year}_{year+1}.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Processing team: {team_abbr}")
            download_data(team_abbr, year, save_path)
            time.sleep(8)  # Add a delay to avoid hitting API rate limits

            with open(save_path, "r", encoding="utf-8") as f:
                all_game_data = json.load(f)

            shots_df = create_shots_dataframe(all_game_data, team_abbr, year)
        
        time.sleep(10)  # Additional delay between years to be respectful to the API
