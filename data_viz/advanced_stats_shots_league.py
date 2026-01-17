from __future__ import annotations
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import plotly.express as px
from nhlpy import NHLClient
from pipeline.download_data import extract_game_ids

# On extrait les abbréviations des équipes de la ligue
def get_league_team_abbrs(client):
    teams = client.teams.teams()
    teams_abbr = [team['abbr'] for team in teams]
    return teams_abbr

# Fonction pour transformer les coordonnées des tirs
def transform_coordinates(shots_df):
    x0 = shots_df["x_coord"].to_numpy(dtype=float)
    y0 = shots_df["y_coord"].to_numpy(dtype=float)

    mask_right = x0 > 0 

    shots_df["x_coord"] = np.where(mask_right, -x0, x0)
    shots_df["y_coord"] = np.where(mask_right, -y0, y0)

    x_temp = shots_df["x_coord"].copy()
    shots_df["x_coord"] = -shots_df["y_coord"]
    shots_df["y_coord"] = x_temp

    shots_df["y_coord"] = shots_df["y_coord"] - 2.5  # ajuster verticalement pour centrer sur le but

    shots_df["x_coord"] = shots_df["x_coord"].clip(-42.5, 42.5)
    shots_df["y_coord"] = shots_df["y_coord"].clip(-100, 0)

# On crée un DataFrame des tirs pour toute la ligue
def create_league_shots_dataframe(year):
    client = NHLClient()
    teams_abbr = get_league_team_abbrs(client)
    league_shots_df = pd.DataFrame()

    for team in teams_abbr:
        data_path = Path(f"../data/shots_data_{team}_{year}_{year+1}.csv")
        team_shots_df = pd.read_csv(data_path)
        transform_coordinates(team_shots_df)
        league_shots_df = pd.concat([league_shots_df, team_shots_df], ignore_index=True)
        print(f"Added shots for team: {team}")
    return league_shots_df

# Fonction pour calculer le temps total joué à 5v5
def compute_team_5v5_seconds(
    team_games: list[dict],
    *,
    situation_5v5_code: str = "1551",
    reg_period_len: int = 20 * 60,
    period_len_overrides: dict[int, int] | None = None,  # ex: {4: 5*60}
) -> int:
    """
    Calcule le temps total joué à 5v5 (en secondes) pour une équipe,
    à partir d'un JSON "par équipe" (liste de matchs).

    Hypothèses sur le JSON:
      - team_games est une liste de dicts (un par match)
      - chaque match a une clé "plays" (liste d'événements)
      - chaque play a:
          - play["periodDescriptor"]["number"]
          - play["timeRemaining"] (format "MM:SS")
          - play["situationCode"] (ex: "1551" pour 5v5)

    La logique:
      - On trie les plays par (période, temps écoulé)
      - Si le play i est en 5v5, on ajoute le temps jusqu'au play suivant
        (ou jusqu'à la fin de la période si c'est le dernier play de la période)
    """
    if period_len_overrides is None:
        period_len_overrides = {}

    def period_len(p: int) -> int:
        return int(period_len_overrides.get(int(p), reg_period_len))

    def mmss_to_seconds_remaining(s: str) -> int:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)

    total = 0

    for game in team_games:
        plays = game.get("plays", [])
        if not plays:
            continue

        enriched = []
        for pl in plays:
            per = pl.get("periodDescriptor", {}).get("number")
            tr = pl.get("timeRemaining")
            sc = pl.get("situationCode")
            if per is None or tr is None or sc is None:
                continue

            try:
                per = int(per)
                rem = mmss_to_seconds_remaining(str(tr))
                elapsed = period_len(per) - rem
            except Exception:
                continue

            enriched.append((per, elapsed, str(sc)))

        enriched.sort(key=lambda x: (x[0], x[1]))
        if not enriched:
            continue

        for i, (per, elapsed, sc) in enumerate(enriched):
            # delta vers le prochain play dans la même période, sinon fin de période
            if i + 1 < len(enriched) and enriched[i + 1][0] == per:
                delta = enriched[i + 1][1] - elapsed
            else:
                delta = period_len(per) - elapsed

            if delta <= 0:
                continue

            if sc == situation_5v5_code:
                total += int(delta)

    return total


# Fonction pour calculer la densité de tirs par 60 minutes à l'aide de KDE
def kde_surface_shots_per_60(
    df,
    seconds_5v5,
    *,
    x_col="x_coord",
    y_col="y_coord",
    x_range=(-42.5, 42.5),
    y_range=(-100, 0),
    grid_size=180,
    bw=0.20,
    max_samples=8000, 
    seed=0,
):
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    n = x.size
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        x = x[idx]
        y = y[idx]

    xg = np.linspace(x_range[0], x_range[1], grid_size)
    yg = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(xg, yg)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(grid_size, grid_size)

    Z_60 = Z * (3600.0 / float(seconds_5v5))
    return X, Y, Z_60

# Main
if __name__ == "__main__":
    year = 2024
    teams = get_league_team_abbrs(NHLClient())

    league_shots_df = create_league_shots_dataframe(year)
    print(f"Total shots in league for {year}-{year+1}: {len(league_shots_df)}")

    # Dictionnaires pour enregistrer les données par équipe
    team_5v5_seconds_dict = {}
    X_team_dict = {}
    Y_team_dict = {}
    Z_team_60_dict = {}

    for team in teams:
        team_shots = league_shots_df[league_shots_df["team_abbr"] == team]
        print(f"Team {team} has {len(team_shots)} shots.")

        # Calculer le temps passé à 5v5 en secondes pour l'équipe
        team_json_path = Path(f"../data/nhl_play_by_play_{team}_{year}_{year+1}.json")
        
        if not team_json_path.exists():
            print(f"Skip {team}: missing JSON {team_json_path}")
            continue

        with open(team_json_path, "r", encoding="utf-8") as f:
            team_games_json = json.load(f)
        
        team_5v5_seconds = compute_team_5v5_seconds(team_games_json, 
                                                    situation_5v5_code="1551", 
                                                    period_len_overrides={4: 5*60},
                                                    )

        if team_5v5_seconds <= 0:
            print(f"Skip {team}: 5v5 seconds={team_5v5_seconds}")
            continue

        # Calculer la densité de tirs par 60 minutes pour l'équipe
        X_team, Y_team, Z_team_60 = kde_surface_shots_per_60(
            team_shots,
            team_5v5_seconds,
            x_col="x_coord",
            y_col="y_coord",
            x_range=(-42.5, 42.5),
            y_range=(-100, 0),
            grid_size=120,
            bw=0.20,
            max_samples=8000,
            seed=0,
        )

        # Sauvegarde
        team_5v5_seconds_dict[team] = team_5v5_seconds
        X_team_dict[team] = X_team
        Y_team_dict[team] = Y_team
        Z_team_60_dict[team] = Z_team_60

    # Ligue: moyenne par équipe (pas somme) 
    valid_teams = list(Z_team_60_dict.keys())
    if len(valid_teams) < 2:
        raise RuntimeError("Not enough valid teams to compute league average/differences.")

    # Toutes les équipes partagent la même grille -> on peut prendre celle d'une équipe
    ref_team = valid_teams[0]
    X_ref = X_team_dict[ref_team]
    Y_ref = Y_team_dict[ref_team]

    # Somme de toutes les surfaces
    Z_sum = np.zeros_like(Z_team_60_dict[ref_team])
    for t in valid_teams:
        Z_sum += Z_team_60_dict[t]

    # Moyenne ligue
    Z_league_mean_60 = Z_sum / len(valid_teams)

    # Différences: team - mean(others) 
    Z_diff_60_dict = {}
    for t in valid_teams:
        Z_mean_others = (Z_sum - Z_team_60_dict[t]) / (len(valid_teams) - 1)
        Z_diff_60_dict[t] = Z_team_60_dict[t] - Z_mean_others

    print(f"Computed Z_diff surfaces for {len(Z_diff_60_dict)} teams.")