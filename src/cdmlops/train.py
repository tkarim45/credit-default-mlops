"""Train the credit-default classifier, log everything to MLflow, register the model,
and persist a serving artifact.

Stage outputs (DVC-tracked): artifacts/model.pkl, artifacts/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import (
    ARTIFACTS, FEATURES, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, METRICS_PATH,
    MODEL_PATH, REF_PARQUET, REGISTERED_MODEL, TARGET,
)
from .evaluate import evaluate_predictions
from .features import build_preprocessor


def train(seed: int = 7, register: bool = True) -> dict:
    df = pd.read_parquet(REF_PARQUET)
    # time-honest split: hold out the last 25% as the evaluation set
    n = len(df)
    cut = int(n * 0.75)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    params = dict(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, eval_metric="logloss",
        random_state=seed,
    )
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", XGBClassifier(**params)),
    ])

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        pipe.fit(train_df[FEATURES], train_df[TARGET])
        proba = pipe.predict_proba(test_df[FEATURES])[:, 1]
        metrics = evaluate_predictions(test_df[TARGET].values, proba)

        mlflow.log_params(params)
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_test", len(test_df))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipe, name="model",
            registered_model_name=REGISTERED_MODEL if register else None,
            # cloudpickle: skops (the 3.x default) rejects XGBoost as an "untrusted type"
            serialization_format="cloudpickle",
        )
        run_id = run.info.run_id

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipe, "features": FEATURES, "run_id": run_id}, f)
    METRICS_PATH.write_text(json.dumps({**metrics, "run_id": run_id}, indent=2))

    print(f"[train] run={run_id}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"[train] model -> {MODEL_PATH}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="train + log to MLflow + register")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    train(seed=args.seed, register=not args.no_register)


if __name__ == "__main__":
    main()
