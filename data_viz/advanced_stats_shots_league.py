from __future__ import annotations
import pickle
import json
from pathlib import Path
import base64
from io import BytesIO
import matplotlib.image as mpimg
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nhlpy import NHLClient

# On extrait les abbréviations des équipes de la ligue
def get_league_team_abbrs(client):
    teams = client.teams.teams()
    teams_abbr = [team['abbr'] for team in teams]
    return teams_abbr

# Fonction pour transformer les coordonnées des tirs
def transform_coordinates(shots_df):

    df = shots_df.copy()

    x0 = df["x_coord"].to_numpy(dtype=float)
    y0 = df["y_coord"].to_numpy(dtype=float)

    mask_right = x0 > 0 

    df["x_coord"] = np.where(mask_right, -x0, x0)
    df["y_coord"] = np.where(mask_right, -y0, y0)

    x_temp = df["x_coord"].copy()
    df["x_coord"] = -df["y_coord"]
    df["y_coord"] = x_temp

    df["y_coord"] = df["y_coord"] - 2.5  # ajuster verticalement pour centrer sur le but

    df["x_coord"] = df["x_coord"].clip(-42.5, 42.5)
    df["y_coord"] = df["y_coord"].clip(-90, 0) # couper au niveau de la ligne de but 

    return df

# On crée un DataFrame des tirs pour toute la ligue
def create_league_shots_dataframe(year):
    client = NHLClient()
    teams_abbr = get_league_team_abbrs(client)
    league_shots_df = pd.DataFrame()

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    for team in teams_abbr:
        data_path = DATA_DIR/f"shots_data_{team}_{year}_{year+1}.csv"
        team_shots_df = pd.read_csv(data_path)
        team_shots_df = transform_coordinates(team_shots_df)
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
    y_range=(-90, 0),
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

def compute_league_kde(year):
    teams = get_league_team_abbrs(NHLClient())
    league_shots_df = create_league_shots_dataframe(year)

    team_5v5_seconds_dict = {}
    X_team_dict = {}
    Y_team_dict = {}
    Z_team_60_dict = {}
    Z_diff_60_dict = {}

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    for team in teams:
        team_shots = league_shots_df.loc[league_shots_df["team_abbr"] == team].copy()

        team_json_path = DATA_DIR / f"nhl_play_by_play_{team}_{year}_{year+1}.json"
        if not team_json_path.exists():
            continue

        with open(team_json_path, "r", encoding="utf-8") as f:
            team_games_json = json.load(f)

        team_5v5_seconds = compute_team_5v5_seconds(
            team_games_json,
            situation_5v5_code="1551",
            period_len_overrides={4: 5 * 60},
        )

        if team_5v5_seconds <= 0 or len(team_shots) == 0:
            continue

        X, Y, Z_60 = kde_surface_shots_per_60(
            team_shots,
            team_5v5_seconds,
            grid_size=120,
            bw=0.20,
        )

        team_5v5_seconds_dict[team] = team_5v5_seconds
        X_team_dict[team] = X
        Y_team_dict[team] = Y
        Z_team_60_dict[team] = Z_60

    valid_teams = list(Z_team_60_dict.keys())
    if len(valid_teams) < 2:
        raise RuntimeError("Not enough valid teams to compute league average.")

    Z_sum = np.zeros_like(Z_team_60_dict[valid_teams[0]])
    for t in valid_teams:
        Z_sum += Z_team_60_dict[t]

    for t in valid_teams:
        Z_diff_60_dict[t] = Z_team_60_dict[t] - (Z_sum - Z_team_60_dict[t]) / (len(valid_teams) - 1)

    return {
        "valid_teams": valid_teams,
        "X_team": X_team_dict,
        "Y_team": Y_team_dict,
        "Z_diff": Z_diff_60_dict,
    }

def load_or_compute_league_kde(year, force=False):
    BASE_DIR = Path(__file__).resolve().parent.parent
    cache_dir = BASE_DIR / "data" / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"league_kde_{year}.pkl"

    if cache_path.exists() and not force:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    results = compute_league_kde(year)

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)

    return results


# Main
if __name__ == "__main__":
    year = 2024

    results = load_or_compute_league_kde(year)

    valid_teams = results["valid_teams"]
    X_team_dict = results["X_team"]
    Y_team_dict = results["Y_team"]
    Z_diff_60_dict = results["Z_diff"]

    ref_team = valid_teams[0]
    X_ref = X_team_dict[ref_team]
    Y_ref = Y_team_dict[ref_team]

# Visualisation Plotly 

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Installe Pillow: pip install pillow") from e


    def image_to_base64_uri(path: Path) -> str:
        """Encode une image locale en data URI (pratique pour fig.write_html)."""
        img = Image.open(path).convert("RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return "data:image/png;base64," + b64


    def mask_below_threshold(Z: np.ndarray, threshold: float) -> np.ndarray:
        """Plotly: pour 'masquer', on met None (NaN) sous le seuil."""
        Zm = Z.copy().astype(float)
        Zm[np.abs(Zm) < threshold] = np.nan
        return Zm


    # Choix de l'équipe par défaut
    default_team = valid_teams[0]

    # Grille plotly
    x = X_ref[0, :].tolist()
    y = Y_ref[:, 0].tolist()

    # vlim = percentile 99 des valeurs absolues (symétrique autour de 0)
    Z0 = Z_diff_60_dict[default_team]
    vlim = float(np.percentile(np.abs(Z0), 99))

    # Niveaux: 33 niveaux comme ton exemple (=> 32 intervalles)
    n_levels = 33
    step = (2 * vlim) / (n_levels - 1)

    # Seuil: comme ton exemple (percentile 60 des |Z|)
    threshold = float(np.percentile(np.abs(Z0), 60))

    # Appliquer le masquage
    Z0_masked = mask_below_threshold(Z0, threshold).tolist()

    # Fond glace 
    BASE_DIR = Path(__file__).resolve().parent.parent
    RINK_PATH = BASE_DIR/"assets"/"half_rink.png" 

    rink_uri = None
    if RINK_PATH.exists():
        rink_uri = image_to_base64_uri(RINK_PATH)
    else:
        print(f"[Plotly] Fond glace introuvable: {RINK_PATH} (je continue sans).")

    fig = go.Figure()

    # Surface remplie 
    fig.add_trace(
        go.Contour(
            x=x, y=y, z=Z0_masked,
            contours=dict(
                start=-vlim,
                end=vlim,
                size=step,
                coloring="fill",
                showlines=False,
            ),
            colorscale="RdBu",      # palette divergente
            zmin=-vlim, zmax=vlim,  # symétrique autour de 0
            colorbar=dict(title="Unblocked shots / 60 (Team − Rest of League)"),
            connectgaps=False,
            opacity=0.80,
            hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>Δ=%{z:.4f}<extra></extra>",
        )
    )

    # Contours lignes par-dessus 
    fig.add_trace(
        go.Contour(
            x=x, y=y, z=Z0_masked,
            contours=dict(
                start=-vlim,
                end=vlim,
                size=step,
                coloring="none",
                showlines=True,
            ),
            line=dict(color="black", width=0.6),
            showscale=False,
            connectgaps=False,
            opacity=0.35,
            hoverinfo="skip",
        )
    )

    # Background rink
    if rink_uri is not None:
        fig.add_layout_image(
            dict(
                source=rink_uri,
                xref="x", yref="y",
                x=-42.5, y=-100,
                xanchor="left", yanchor="top", 
                sizex=85.0, sizey=100.0,
                sizing="stretch",
                opacity=1.0,
                layer="below",
            )
        )

    # Axes / layout 
    fig.update_layout(
        title=f"{default_team} 5v5 Unblocked Shot Rate Differential<br><sup>(KDE, Shots/60 vs Rest of League)</sup>",
        xaxis_title="X (ft)",
        yaxis_title="Y (ft)",
        xaxis=dict(range=[-42.5, 42.5], zeroline=False),
        yaxis=dict(range=[0, -100], zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(r=140),
    )

    # Dropdown: on met à jour Z + on recalcule vlim/threshold par équipe 
    buttons = []
    for t in valid_teams:
        Zt = Z_diff_60_dict[t]
        vlim_t = float(np.percentile(np.abs(Zt), 99))
        step_t = (2 * vlim_t) / (n_levels - 1)
        thr_t = float(np.percentile(np.abs(Zt), 60))
        Zt_masked = mask_below_threshold(Zt, thr_t).tolist()

        buttons.append(
            dict(
                label=t,
                method="update",
                args=[
                    # Met à jour z pour les 2 traces (fill + lines)
                    {"z": [Zt_masked, Zt_masked]},
                    # Met à jour layout + échelle symétrique + pas des niveaux
                    {
                        "title": f"{t} 5v5 Unblocked Shot Rate Differential<br><sup>(KDE, Shots/60 vs Rest of League)</sup>",
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=1.16, y=0.90,
                xanchor="left", yanchor="top",
                showactive=True,
            )
        ],
    )

    fig.show()
    # fig.write_html("shot_diff_contours_dropdown.html", auto_open=True)

