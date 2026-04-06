import wandb
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss 
from dotenv import load_dotenv
import os

"""
This module contains a logistic regression model for predicting shot quality (xG).
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


def logistic_regression_model(X_train, y_train, X_test, y_test, config):
    """
    Trains a logistic regression model and logs results to WandB.

    - Accepts a config dict for hyperparameters (C, penalty, solver) instead of hardcoded values
      This allows WandB sweeps to vary these parameters automatically across runs
    - Evaluation includes AUC-ROC, log loss, and confusion matrix in addition to accuracy
    """

    # C: regularization strength (smaller = stronger regularization, simpler model)
    # penalty: l1 can zero out irrelevant features, l2 shrinks all coefficients
    # solver: saga supports both l1 and l2 and scales well to large datasets
    model = LogisticRegression(
        C=config["C"],
        penalty=config["penalty"],
        solver=config["solver"],
        tol=0.1,
        max_iter=500
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)

    # predict_proba returns probabilities for each class; [:, 1] gives the probability of a goal
    # This is needed for AUC-ROC and log loss, which work on probabilities, not binary predictions
    proba = model.predict_proba(X_test_scaled)[:, 1]
    preds = model.predict(X_test_scaled)

    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)

    # AUC-ROC: measures how well the model separates goals from non-goals
    # 0.5 = random guessing, 1.0 = perfect. Standard metric for xG models.
    test_auc = roc_auc_score(y_test, proba)

    # Log loss: penalizes confident wrong predictions
    # Evaluates the quality of probability estimates, not just rankings. Lower is better.
    test_log_loss = log_loss(y_test, proba)

    # Log all metrics to WandB including confusion matrix
    # The confusion matrix shows true/false positives and negatives 
    wandb.log({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "test_log_loss": test_log_loss,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(),
            preds=preds.tolist(),
            class_names=["No Goal", "Goal"]
        )
    })

    print(f"Training Accuracy : {train_accuracy:.4f}")
    print(f"Testing Accuracy  : {test_accuracy:.4f}")
    print(f"AUC-ROC           : {test_auc:.4f}")
    print(f"Log Loss          : {test_log_loss:.4f}")

    return model

if __name__ == "__main__":

    load_dotenv()

    db_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "sqlite" / "nhl_analytics.db"

    api_key = os.getenv("WANDB_API_KEY")
    use_wandb = api_key is not None and api_key.strip() != ""

    configs = [
        {"C": 1,   "penalty": "l2", "solver": "saga"},
        {"C": 1,   "penalty": "l1", "solver": "saga"},
        {"C": 10,  "penalty": "l2", "solver": "saga"},
    ]

    if use_wandb:
        print("WandB API Key found. Running sweep with WandB logging enabled.")
        wandb.login(key=api_key)

        X_train, y_train, X_test, y_test = train_test_split_by_season(db_path)
        
        for config in configs:
            run_name = f"logreg_C{config['C']}_{config['penalty']}"
            with wandb.init(
                project=os.getenv("WANDB_PROJECT"),
                entity=os.getenv("WANDB_ENTITY"),
                name=run_name,
                config={**config, "model": "logistic_regression", "train_seasons": [2018, 2019, 2020, 2021, 2022, 2023], "test_season": 2024},
                settings=wandb.Settings(init_timeout=300)
            ):
                logistic_regression_model(X_train, y_train, X_test, y_test, config)
    else:
        print("No WandB API Key found. Running model without WandB logging.")
        X_train, y_train, X_test, y_test = train_test_split_by_season(db_path)
        for config in configs:
            print(f"Running config: C={config['C']}, penalty={config['penalty']}")
            logistic_regression_model(X_train, y_train, X_test, y_test, config)
