import json
import tqdm
import requests
import os
from pathlib import Path

def extract_game_ids(team_abbr: str, year: int) -> list:
    """
    Extracts all game IDs for a specific team and year from the NHL API schedule endpoint.

    Args:
        team_abbr (str): The abbreviation of the team (e.g., 'MTL' for Montreal Canadiens).
        year (int): The first year of the season to retrieve, i.e. for the 2016-17
            season you'd put in 2016
    Returns:
        list: A list of game IDs for the specified team and year.
    """

    schedule_url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_abbr}/{year}{year+1}"

    response = requests.get(schedule_url)

    # Raise an error if the request was unsuccessful
    response.raise_for_status()

    # Parse the JSON response to extract game IDs
    schedule_data = response.json()
    game_ids = []

    for game in schedule_data.get("games", []):
        game_ids.append(game.get("id"))

    return game_ids

def download_data(team_abbr: str, year: int, save_path: Path):
    """
    Downloads play-by-play data from the NHL API for a specific team and for a specific year, and saves it locally.

    Args:
        year (int): The first year of the season to retrieve, i.e. for the 2016-17
            season you'd put in 2016
        save_path (Path): The local path where the file should be saved.
    """

    # Ensure the save directory exists
    os.makedirs(save_path.parent, exist_ok=True)

    print(f"Downloading play-by-play data for the {year}-{year+1} season...")

    # Verify if the data for the year already exists
    if save_path.exists():
        print(f"Data for the {year}-{year+1} season already exists at {save_path}. Skipping download.")
        return

    # Access the schedule endpoint to get all game IDs for the season
    pbp_endpoint = "/v1/gamecenter/{game-id}/play-by-play"
    base_url = "https://api-web.nhle.com/"

    game_ids = extract_game_ids(team_abbr, year)
    all_game_data = []
    for game_id in tqdm.tqdm(game_ids, desc="Downloading game data"):
        url = base_url + pbp_endpoint.replace("{game-id}", str(game_id))
        response = requests.get(url)
        response.raise_for_status()
        game_data = response.json()
        all_game_data.append(game_data)

    # Save the aggregated data to the specified path
    with open(save_path, 'w') as f:
        json.dump(all_game_data, f, indent=2)
