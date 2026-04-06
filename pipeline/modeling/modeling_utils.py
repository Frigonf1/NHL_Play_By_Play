import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

"""
This module contains utility functions for data cleaning, feature engineering, and loading shot data for modeling.
"""

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and engineers features from raw shot data.
    """
    # Columns to keep for modeling 
    model_df = df[['x_coord', 'y_coord', 'is_goal', 'is_empty_net',
                   'shot_type', 'strength', 'period']].dropna().copy()

    # Using symmetry to normalize all shots to the same end of the ice. 
    # The y-coordinate is flipped for shots from the left side to maintain the correct angle.
    negative_x = model_df['x_coord'] < 0
    model_df.loc[negative_x, 'x_coord'] = model_df.loc[negative_x, 'x_coord'].abs()
    model_df.loc[negative_x, 'y_coord'] = -model_df.loc[negative_x, 'y_coord']

    # Shot distance from the goal (goal is at x=89, y=0 on a 200ft rink)
    model_df['shot_distance'] = ((model_df['x_coord'] - 89)**2 + model_df['y_coord']**2)**0.5

    # Shot angle relative to the goal center 
    model_df['shot_angle'] = np.abs(np.arctan2(model_df['y_coord'], 89 - model_df['x_coord']) * (180 / np.pi))

    # Flagging overtime shots, different dymanics in OT because of 3-on-3 format 
    model_df['is_overtime'] = (model_df['period'] >= 4).astype(int)

    # One-hot encode shot_type and strength to convert categorical variables into binary features 
    model_df = pd.get_dummies(model_df, columns=['shot_type', 'strength'], drop_first=True)

    # Drop raw columns now that features have been derived from them 
    model_df = model_df.drop(columns=['x_coord', 'y_coord', 'period'])

    return model_df


def load_season(season_start: int, db_path: Path) -> pd.DataFrame:
    """Loads all shot data for a given season from the SQLite database."""

    season = f"{season_start}-{season_start + 1}"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM shots WHERE season = ?", conn, params=(season,))
    conn.close()

    return df


def train_test_split_by_season(db_path: Path):
    """
    Loads and splits data into training (2018-2023) and testing (2024) sets.

    - Empty net shots are filtered here after clean_data, keeping clean_data reusable
      for visualization purposes (where you may want to keep empty net shots)
    - is_empty_net is dropped from X after filtering since it's no longer informative
    """

    train_seasons = [2018, 2019, 2020, 2021, 2022, 2023]
    test_season = 2024

    train_dfs = [load_season(s, db_path) for s in train_seasons]
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = clean_data(train_df)

    # Filter empty net shots after cleaning so clean_data stays reusable
    # Empty net shots are excluded because they are not representative of shot quality —
    # the goalie is not present, making them trivially easy to score
    train_df = train_df[train_df["is_empty_net"] == False].drop(columns=["is_empty_net"])

    # Separate features (X) from target variable (y) 
    X_train = train_df.drop(columns=["is_goal"])
    y_train = train_df["is_goal"]

    test_df = load_season(test_season, db_path)
    test_df = clean_data(test_df)

    # Filter empty net shots from test set as well to maintain consistency with training data 
    test_df = test_df[test_df["is_empty_net"] == False].drop(columns=["is_empty_net"])

    # Separate features (X) from target variable (y) for test set 
    X_test = test_df.drop(columns=["is_goal"])
    y_test = test_df["is_goal"]

    return X_train, y_train, X_test, y_test