"""Prometheus instruments for the serving layer. Imported by the API; scraped at /metrics."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS = Counter(
    "cdm_predictions_total", "Total prediction requests", ["outcome"],  # approve / decline
)
PREDICT_LATENCY = Histogram(
    "cdm_predict_latency_seconds", "Prediction latency (s)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
SCORE_HIST = Histogram(
    "cdm_predicted_default_probability", "Distribution of predicted default probability",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
# Online drift gauges — updated by a monitoring job comparing live traffic to the
# training reference; surfaced here so Grafana can alert on them.
DRIFT_SHARE = Gauge("cdm_drift_share", "Share of features drifted vs. training reference")
DATASET_DRIFT = Gauge("cdm_dataset_drift", "1 if dataset-level drift detected, else 0")
MODEL_INFO = Gauge("cdm_model_info", "Deployed model run id (label)", ["run_id"])
