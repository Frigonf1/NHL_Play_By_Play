from download_data import extract_game_ids
from transform_data import display_games_data
from download_data import download_data
from pathlib import Path

if __name__ == "__main__":
    team_abbr = "MTL"  # Montreal Canadiens
    year = 2022
    save_path = Path(f"data/nhl_play_by_play_{team_abbr}_{year}_{year+1}")
    download_data(team_abbr, year, save_path)
    display_games_data(extract_game_ids(team_abbr, year))
