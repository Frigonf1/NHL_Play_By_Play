import sys
import wandb
import sqlite3
from pathlib import Path
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import os

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data_viz.data_plots import plot_shot_totals, plot_shot_heatmap 

"""
This module contains functions for loading, cleaning, and analyzing NHL shot data, 
as well as training a logistic regression model to predict shot quality. 

By setting run_model = False, you can generate exploratory visualizations of shot distance 
and angle distributions, as well as goal rates by distance and angle, without training the model.

By setting run_model = True, the module will load training and testing data by season, 
train a logistic regression model, and log the results to Weights & Biases.
"""

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

def load_season(season_start: int, db_path: Path) -> pd.DataFrame:

    season = f"{season_start}-{season_start + 1}"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT * FROM shots WHERE season = ?
    """, conn, params=(season,))
    conn.close()

    return df


def train_test_split_by_season(db_path: Path):
    train_seasons = [2018, 2019, 2020, 2021, 2022, 2023]
    test_season = 2024

    train_dfs = []
    for season in train_seasons:
        train_dfs.append(load_season(season, db_path))

    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = clean_data(train_df)
    train_df = train_df[train_df["is_empty_net"] == False].copy()

    X_train = train_df.drop(columns=["is_goal"])
    y_train = train_df["is_goal"]

    test_df = load_season(test_season, db_path)
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

    db_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "sqlite" / "nhl_analytics.db"

    run_model = False

    # Use functions from data_plots.py instead 
    if not run_model: 
        df_2023 = load_season(2023, db_path)
        df_2023_cleaned = clean_data(df_2023)

        # plot_shot_totals(df_2023, "MTL", 2023)
        plot_shot_heatmap(df_2023, "MTL", 2023)

        raise SystemExit

    load_dotenv()
    wandb.login()

    FEATURES = ['shot_distance', 'shot_angle']
    X_train, y_train, X_test, y_test = train_test_split_by_season(db_path)
    X_train = X_train[FEATURES]
    X_test = X_test[FEATURES]

    run = wandb.init(
    entity=os.getenv("WANDB_ENTITY"),
    project=os.getenv("WANDB_PROJECT"),
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
