import wandb
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss 
from dotenv import load_dotenv
import os
from modeling_utils import train_test_split_by_season

"""
This module contains a logistic regression model for predicting shot quality (xG).
"""

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
    if wandb.run is not None:
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
                config={**config, "model": "logistic_regression", "train_seasons": [2020, 2021, 2022, 2023], "test_season": 2024},
                settings=wandb.Settings(init_timeout=300)
            ):
                logistic_regression_model(X_train, y_train, X_test, y_test, config)
    else:
        print("No WandB API Key found. Running model without WandB logging.")
        X_train, y_train, X_test, y_test = train_test_split_by_season(db_path)

        for config in configs:
            
            print(f"Running config: C={config['C']}, penalty={config['penalty']}")
            logistic_regression_model(X_train, y_train, X_test, y_test, config)
