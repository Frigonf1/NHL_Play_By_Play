from download_data import download_data, create_shots_dataframe
from pathlib import Path
import json

if __name__ == "__main__":
    team_abbr = "MTL"  # Montreal Canadiens
    year = 2022
    save_path = Path(f"data/nhl_play_by_play_{team_abbr}_{year}_{year+1}.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    download_data(team_abbr, year, save_path)

    with open(save_path, "r", encoding="utf-8") as f:
        all_game_data = json.load(f)

    shots_df = create_shots_dataframe(all_game_data, team_abbr, year)
    print(shots_df.head())
