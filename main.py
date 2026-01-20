from pipeline.download_data import download_data, create_shots_dataframe
from data_viz.data_plots import plot_shot_totals, plot_shot_heatmap 
from nhlpy import NHLClient
from pathlib import Path
import json

if __name__ == "__main__":

    client = NHLClient()

    teams = client.teams.teams()

    # Extract team abbreviations
    teams_abbr = [team['abbr'] for team in teams]
    print(teams_abbr)

    year = 2024 

    for team_abbr in teams_abbr:
        save_path = Path(f"data/nhl_play_by_play_{team_abbr}_{year}_{year+1}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        download_data(team_abbr, year, save_path)

        with open(save_path, "r", encoding="utf-8") as f:
            all_game_data = json.load(f)

        shots_df = create_shots_dataframe(all_game_data, team_abbr, year)
    
        # plot_shot_totals(shots_df, team_abbr, year)
    
    # plot_shot_heatmap(shots_df, team_abbr, year)