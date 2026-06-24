"""Central paths, schema, and the quality gate thresholds CI enforces."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("CDM_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

# dataset files (DVC-tracked outputs of the prepare stage)
REF_PARQUET = DATA_DIR / "reference.parquet"      # training distribution
CUR_PARQUET = DATA_DIR / "current.parquet"        # later "production" slice (drifted)

MODEL_PATH = ARTIFACTS / "model.pkl"              # serialized pipeline for serving
METRICS_PATH = ARTIFACTS / "metrics.json"
DRIFT_PATH = REPORTS / "drift.json"

# MLflow 3.x deprecated the file store; use a local sqlite backend by default.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "credit-default")
REGISTERED_MODEL = "credit-default-classifier"

TARGET = "default"
NUMERIC = [
    "age", "income", "loan_amount", "employment_length", "debt_to_income",
    "credit_history_length", "num_delinquencies", "utilization", "num_credit_lines",
    "interest_rate",
]
CATEGORICAL = ["home_ownership", "purpose"]
FEATURES = NUMERIC + CATEGORICAL

# --- Quality gate: CI fails the build if a freshly trained model is below these. ---
GATE = {
    "roc_auc": 0.70,      # discrimination floor
    "pr_auc": 0.45,       # precision-recall (class-imbalanced)
    "brier_max": 0.20,    # calibration ceiling (lower is better)
}

# Drift alarm: share-of-drifted-columns above this flags the dataset.
DRIFT_SHARE_THRESHOLD = 0.25
