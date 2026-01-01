import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
