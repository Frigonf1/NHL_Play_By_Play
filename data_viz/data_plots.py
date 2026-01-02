from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RINK_IMG_PATH = ASSETS_DIR / "hockey_rink.jpg"

# Function to plot totals for each shot type, and distinguish between goals and misses
def plot_shot_totals(shots_df: 'pd.DataFrame', team_abbr: str, year: int) -> None:
    """
    Plots the total number of shots for each shot type, distinguishing between goals and misses.

    Args:
        shots_df (pd.DataFrame): DataFrame containing shot data.
        team_abbr (str): Team abbreviation.
        year (int): Season year.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(data=shots_df, x='shot_type', hue='is_goal', palette='Set2')
    plt.title(f'Shot Totals by Type for {team_abbr} in {year}-{year+1} Season')
    plt.xlabel('Shot Type')
    plt.ylabel('Total Shots')
    plt.xticks(rotation=45)
    plt.legend(title='Is Goal', labels=['Miss', 'Goal'])
    plt.tight_layout()
    plt.show()

def plot_shot_heatmap(shots_df: 'pd.DataFrame', team_abbr: str, year: int) -> None:
    """
    Plots a heatmap of shot locations on the ice.

    Args:
        shots_df (pd.DataFrame): DataFrame containing shot data.
        team_abbr (str): Team abbreviation.
        year (int): Season year.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1) Heatmap first (lower zorder)
    sns.kdeplot(
        x=shots_df['x_coord'],
        y=shots_df['y_coord'],
        fill=True,
        cmap='Reds',
        bw_adjust=0.5,
        thresh=0.05,
        levels=100,
        ax=ax,
        zorder=1,
        alpha=0.9
    )

    # 2) Load rink image (.jpg) and create transparency for "ice"
    rink = plt.imread(RINK_IMG_PATH)

    # Convert to float in [0,1] if needed
    if rink.dtype != np.float32 and rink.dtype != np.float64:
        rink = rink / 255.0

    # Add alpha channel (RGBA)
    alpha = np.ones((rink.shape[0], rink.shape[1], 1), dtype=rink.dtype)
    rink_rgba = np.concatenate([rink, alpha], axis=2)

    # Make near-white pixels transparent (tweak threshold if needed)
    white_thresh = 0.95
    mask = (
        (rink_rgba[..., 0] > white_thresh) &
        (rink_rgba[..., 1] > white_thresh) &
        (rink_rgba[..., 2] > white_thresh)
    )
    rink_rgba[..., 3] = np.where(mask, 0.0, 1.0)

    # 3) Draw rink overlay LAST (higher zorder) so lines are on top
    ax.imshow(
        rink_rgba,
        extent=[-100, 100, -42.5, 42.5],
        origin="upper",   # change to "lower" if your rink is flipped vertically
        zorder=10
    )

    # Formatting
    ax.set_title(f'Shot Location Heatmap for {team_abbr} in {year}-{year+1} Season')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-42.5, 42.5)

    ax.axvline(0, color='black', linestyle='--', zorder=20)
    ax.axhline(0, color='black', linestyle='--', zorder=20)

    fig.tight_layout()
    plt.show()
