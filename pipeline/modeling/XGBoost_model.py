import wandb
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from dotenv import load_dotenv
import os
from modeling_utils import train_test_split_by_season

"""
This module contains an XGBoost model for predicting shot quality (xG).
"""

def xgboost_model(X_train, y_train, X_test, y_test, config):
    """
    Trains an XGBoost model and logs results to WandB.

    - Accepts a config dict for hyperparameters (n_estimators, max_depth, learning_rate) instead of hardcoded values
      This allows WandB sweeps to vary these parameters automatically across runs
    """

    model = xgb.XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    proba = model.predict_proba(X_test_scaled)[:, 1]
    preds = model.predict(X_test_scaled)

    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)

    test_auc = roc_auc_score(y_test, proba)
    test_log_loss = log_loss(y_test, proba)

    if wandb.run is not None:
        wandb.log({
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "test_auc": test_auc,
            "test_log_loss": test_log_loss,
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_test.tolist(), 
                preds=preds.tolist(), 
                class_names=["No Goal", "Goal"])
        })

    print(f"XGBoost - Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:           {test_accuracy:.4f}")
    print(f"Test AUC:                {test_auc:.4f}") 
    print(f"Test Log Loss:           {test_log_loss:.4f}")

    return model

if __name__ == "__main__":

    load_dotenv()
    db_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "sqlite" / "nhl_analytics.db"

    api_key = os.getenv("WANDB_API_KEY")
    use_wandb = api_key is not None and api_key.strip() != ""

    configs = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.01},
    ]

    if use_wandb:
        print("WandB API Key found. Running sweep with WandB logging enabled.")
        wandb.login(key=api_key)

        X_train, y_train, X_test, y_test = train_test_split_by_season(db_path)

        for config in configs:
            run_name = f"XGB_n{config['n_estimators']}_d{config['max_depth']}_lr{config['learning_rate']}"
            with wandb.init(
                project=os.getenv("WANDB_PROJECT"),
                entity=os.getenv("WANDB_ENTITY"),
                name=run_name,
                config={**config, "model": "xgboost", "train_seasons": [2018, 2019, 2020, 2021, 2022, 2023], "test_season": 2024},
                settings=wandb.Settings(init_timeout=300)
            ):
                
                # Sample part of the training data for faster runs during development
                # Comment out the next two lines to use the full training set
                X_train_sampled = X_train.sample(frac=0.1, random_state=42)
                y_train_sampled = y_train.loc[X_train_sampled.index]
                
                xgboost_model(X_train_sampled, y_train_sampled, X_test, y_test, config)
            
    else:
        print("No WandB API Key found. Running model without WandB logging.")
        X_train, y_train, X_test, y_test = train_test_split_by_season(db_path)

        for config in configs:
            
            print(f"Running config: n_estimators={config['n_estimators']}, max_depth={config['max_depth']}, learning_rate={config['learning_rate']}")

            # Sample part of the training data for faster runs during development
            # Comment out the next two lines to use the full training set
            X_train_sampled = X_train.sample(frac=0.1, random_state=42)
            y_train_sampled = y_train.loc[X_train_sampled.index]
            
            xgboost_model(X_train_sampled, y_train_sampled, X_test, y_test, config)
        