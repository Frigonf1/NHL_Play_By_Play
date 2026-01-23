import wandb
from pathlib import Path
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import os

def clean_data(df):

    # Keeping only relevant columns and dropping rows with missing values 
    model_df = df[['x_coord', 'y_coord', 'is_goal', 'is_empty_net']].dropna().copy()

    # Using symmetry to obtain only positive x-coordinates
    model_df['x_coord'] = model_df['x_coord'].abs()

    # Adding a column for shot distance, assuming rink length is 200 feet and goal coordinates are (89, 0)
    model_df['shot_distance'] = ((model_df['x_coord'] - 89)**2 + model_df['y_coord']**2)**0.5

    # Adding a column for shot angle
    model_df['shot_angle'] = np.arctan2(model_df['y_coord'], 89 - model_df['x_coord']) * (180 / np.pi)

    return model_df

def shot_distance_histogram(df):

    # Separate empty net shots
    non_empty_net_shots = df[df['is_empty_net'] == False]
    empty_net_shots = df[df['is_empty_net'] == True]

    dataframes = [non_empty_net_shots, empty_net_shots]

    for dataframe in dataframes:
        plt.figure(figsize=(10, 6))
        plt.hist(dataframe['shot_distance'], bins=30, edgecolor='black')
        plt.title('Shot Distance Distribution for ' + ('Non-Empty Net Shots' if dataframe is non_empty_net_shots else 'Empty Net Shots'))
        plt.xlabel('Distance (feet)')
        plt.ylabel('Frequency')
        plt.show()

def shot_angle_histogram(df):

    # Separate empty net shots
    non_empty_net_shots = df[df['is_empty_net'] == False]
    empty_net_shots = df[df['is_empty_net'] == True]

    dataframes = [non_empty_net_shots, empty_net_shots]

    for dataframe in dataframes:
        plt.figure(figsize=(10, 6))
        plt.hist(dataframe['shot_angle'], bins=30, edgecolor='black')
        plt.title('Shot Angle Distribution for ' + ('Non-Empty Net Shots' if dataframe is non_empty_net_shots else 'Empty Net Shots'))
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Frequency')
        plt.show()

# Figures pour les taux de buts en fonction de la distance et de l'angle. Une figure jointe pour les buts en filet désert, puis une seconde pour les autres tirs.
def plot_goal_rates(df):

    # Separate empty net shots
    non_empty_net_shots = df[df['is_empty_net'] == False]
    empty_net_shots = df[df['is_empty_net'] == True]

    dataframes = [non_empty_net_shots, empty_net_shots]
    titles = ['Non-Empty Net Shots', 'Empty Net Shots']

    # Goal rate by distance (joint figure) 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, dataframe in enumerate(dataframes):
        distance_bins = np.arange(0, 201, 10)
        distance_goal_rates = []

        for j in range(len(distance_bins) - 1):
            bin_shots = dataframe[
                (dataframe['shot_distance'] >= distance_bins[j]) &
                (dataframe['shot_distance'] < distance_bins[j+1])
            ]
            goal_rate = bin_shots['is_goal'].mean() if len(bin_shots) > 0 else 0
            distance_goal_rates.append(goal_rate)

        axes[i].bar(
            distance_bins[:-1],
            distance_goal_rates,
            width=10,
            edgecolor='black',
            align='edge'
        )
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Shot Distance (feet)')
        axes[i].set_ylabel('Goal Rate')
        axes[i].set_ylim(0, 1)

    fig.suptitle('Goal Rate by Shot Distance')
    plt.tight_layout()
    plt.show()

    # Goal rate by angle (joint figure)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, dataframe in enumerate(dataframes):
        angle_bins = np.arange(-90, 91, 10)
        angle_goal_rates = []

        for j in range(len(angle_bins) - 1):
            bin_shots = dataframe[
                (dataframe['shot_angle'] >= angle_bins[j]) &
                (dataframe['shot_angle'] < angle_bins[j+1])
            ]
            goal_rate = bin_shots['is_goal'].mean() if len(bin_shots) > 0 else 0
            angle_goal_rates.append(goal_rate)

        axes[i].bar(
            angle_bins[:-1],
            angle_goal_rates,
            width=10,
            edgecolor='black',
            align='edge'
        )
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Shot Angle (degrees)')
        axes[i].set_ylabel('Goal Rate')
        axes[i].set_ylim(0, 1)

    fig.suptitle('Goal Rate by Shot Angle')
    plt.tight_layout()
    plt.show()

# Histogramme des buts en fonction de la distance
def goal_distance_histogram(df):

    goals = df[df['is_goal'] == True].copy()

    plt.figure(figsize=(10, 6))
    plt.hist(goals['shot_distance'], bins=30, edgecolor='black')
    plt.title('Goal Distance Distribution')
    plt.xlabel('Distance (feet)')
    plt.ylabel('Frequency')
    plt.show()

def load_season(season_start: int, base_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all CSV files for a given season.
    Example: season_start=2018 -> folder '2018_2019'
    """
    season_dir = base_dir / f"{season_start}_{season_start + 1}"
    csv_files = list(season_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {season_dir}")

    dfs = [pd.read_csv(csv) for csv in csv_files]
    return pd.concat(dfs, ignore_index=True)


def train_test_split_by_season(base_dir: Path):
    train_seasons = [2018, 2019, 2020, 2021, 2022, 2023]
    test_season = 2024

    # Load training data 
    train_dfs = []
    for season in train_seasons:
        season_df = load_season(season, base_dir)
        season_df["season"] = season
        train_dfs.append(season_df)

    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = clean_data(train_df)
    train_df = train_df[train_df["is_empty_net"] == False].copy()

    X_train = train_df.drop(columns=["is_goal"])
    y_train = train_df["is_goal"]

    # Load test data 
    test_df = load_season(test_season, base_dir)
    test_df["season"] = test_season
    test_df = clean_data(test_df)
    test_df = test_df[test_df["is_empty_net"] == False].copy()

    X_test = test_df.drop(columns=["is_goal"])
    y_test = test_df["is_goal"]

    return X_train, y_train, X_test, y_test


def logistic_regression_model(X_train, y_train, X_test, y_test):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    wandb.log({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    })

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    return model

if __name__ == "__main__":

    load_dotenv()
    wandb.login()

    base_data_dir = Path("../NHL_Play_By_Play/data")
    print(base_data_dir.resolve())
    print((base_data_dir / "2018_2019").exists())

    FEATURES = ['shot_distance', 'shot_angle']
    X_train, y_train, X_test, y_test = train_test_split_by_season(base_data_dir)
    X_train = X_train[FEATURES]
    X_test = X_test[FEATURES]

    run = wandb.init(
    entity="francois-frigon-universite-de-montreal",
    project="NHL_shot_quality_logistic_regression_model",
    config={
        "model": "logistic_regression",
        "max_iter": 1000,
        "train_seasons": [2018, 2019, 2020, 2021, 2022, 2023],
        "test_season": 2024,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test)
    }
    )

    clt = logistic_regression_model(X_train, y_train, X_test, y_test)

    wandb.finish()
