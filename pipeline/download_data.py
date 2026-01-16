import json
import tqdm
import requests
import os
from pathlib import Path
import pandas as pd

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

        # Only include regular season and playoff games
        if game.get("gameType") in [2, 3]:
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

# Function that extracts plays from a game in the downloaded JSON file that are shots on goal, missed shots, or goals
def extract_shots_from_game(game_data: dict) -> list:
    """
    Extracts plays that are shots on goal, missed shots, or goals
    from a single game's play-by-play data.
    """
    shot_types = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
    shots = []

    for play in game_data.get("plays", []):
        if play.get("typeDescKey") in shot_types:
            shots.append(play)

    return shots

# Function that builds a lookup dictionary mapping player IDs to player names from the game data
def build_player_lookup(game_data: dict) -> dict[int, str]:
    lookup = {}
    for spot in game_data.get("rosterSpots", []):
        pid = spot.get("playerId")
        if pid is None:
            continue
        first = (spot.get("firstName", {}) or {}).get("default", "")
        last  = (spot.get("lastName", {}) or {}).get("default", "")
        name = (first + " " + last).strip()
        if name:
            lookup[int(pid)] = name
    return lookup


# Function that uses the game ids to iterate through the downloaded JSON file and create a pandas dataframe including : 
# time and period of play, event ID, which team made the shot, whether it was a goal or not, the coordinates on the ice, the player and goalie ids 
def create_shots_dataframe(all_game_data: list, team_abbr: str, year: int) -> 'pd.DataFrame':
    """
    Creates a pandas DataFrame containing shot data from all games.

    Args:
        all_game_data (list): List of game data dictionaries.
        team_abbr (str): The abbreviation of the team (e.g., 'MTL' for Montreal Canadiens).
        year (int): The first year of the season to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing shot data.
    """
    shots_list = []

    situation_codes = {
        "1551": "even-strength",
        "1541": "power-play",
        "1451": "short-handed",
        "1531": "five-on-three power-play",
        "1351": "three-on-five short-handed",
        "1441": "four-on-four",
        "1431": "four-on-three power-play",
        "1341": "three-on-four short-handed",
        "1331": "three-on-three",
        "0651": "six-on-five empty-net",
        "1560": "five-on-six empty-net",
        "0641": "six-on-four empty-net",
        "1460": "four-on-six empty-net",
        
    }

    for game_data in all_game_data:
        player_lookup = build_player_lookup(game_data)

        team_lookup = {
            game_data["homeTeam"]["id"]: game_data["homeTeam"]["abbrev"],
            game_data["awayTeam"]["id"]: game_data["awayTeam"]["abbrev"],
        }

        game_id = game_data.get("id")
        shots = extract_shots_from_game(game_data)

        for shot in shots:
            player_id = shot.get("details", {}).get("shootingPlayerId") or shot.get("details", {}).get("scoringPlayerId")
            goalie_id = shot.get("details", {}).get("goalieInNetId")
            shot_info = {
                "game_id": game_id,
                "event_id": shot.get("eventId"),
                "period": shot.get("periodDescriptor", {}).get("number", ""),
                "period_time": shot.get("timeInPeriod", ""),
                "team_id": shot.get("details", {}).get("eventOwnerTeamId", ""),
                "team_abbr": team_lookup.get(shot.get("details", {}).get("eventOwnerTeamId", ""), ""),
                "shot_type_desc": shot.get("typeDescKey", ""),
                "is_goal": shot.get("typeDescKey") == "goal",
                "x_coord": shot.get("details", {}).get("xCoord"),
                "y_coord": shot.get("details", {}).get("yCoord"),
                "player_id": shot.get("details", {}).get("shootingPlayerId") if shot.get("typeDescKey") != "goal" else shot.get("details", {}).get("scoringPlayerId"),
                "player_name": player_lookup.get(player_id, ""),
                "goalie_id": shot.get("details", {}).get("goalieInNetId"),
                "goalie_name": player_lookup.get(goalie_id, ""),
                "shot_type": shot.get("details", {}).get("shotType", ""),
                "is_empty_net": shot.get("details", {}).get("goalieInNetId") is None,
                "strength": situation_codes.get(shot.get("situationCode", ""), "unknown"),
            }
            shots_list.append(shot_info)

    shots_df = pd.DataFrame(shots_list)

    # Saving the dataframe to a CSV file for further analysis
    shots_df.to_csv(f"data/shots_data_{team_abbr}_{year}_{year+1}.csv", index=False)

    return shots_df