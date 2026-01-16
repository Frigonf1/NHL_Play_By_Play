from pipeline.download_data import download_data, create_shots_dataframe
from data_viz.data_plots import plot_shot_totals, plot_shot_heatmap 
from pathlib import Path
import json

if __name__ == "__main__":
    team_abbr = "MTL" 
    years = [2022, 2023, 2024] 

    for year in years:
        save_path = Path(f"data/nhl_play_by_play_{team_abbr}_{year}_{year+1}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        download_data(team_abbr, year, save_path)

        with open(save_path, "r", encoding="utf-8") as f:
            all_game_data = json.load(f)

        shots_df = create_shots_dataframe(all_game_data, team_abbr, year)
        print(shots_df.head())
    
        # plot_shot_totals(shots_df, team_abbr, year)
        plot_shot_heatmap(shots_df, team_abbr, year)