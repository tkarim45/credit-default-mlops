"""FastAPI serving layer for the credit-default model.

  POST /predict   score an applicant -> {probability, decision, threshold}
  GET  /metrics   Prometheus exposition (latency, throughput, score distribution, drift)
  GET  /drift     latest drift report (if the drift stage has run)
  GET  /health    liveness + which model run is deployed
"""
from __future__ import annotations

import json
import os
import pickle
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from cdmlops.config import DRIFT_PATH, FEATURES, MODEL_PATH
from cdmlops.monitoring import (
    DRIFT_SHARE, DATASET_DRIFT, MODEL_INFO, PREDICT_LATENCY, PREDICTIONS, SCORE_HIST,
)

THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
app = FastAPI(title="Credit Default MLOps", version="0.1.0")
_model = None


def _load():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"model not found at {MODEL_PATH}; run training first")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        MODEL_INFO.labels(run_id=_model.get("run_id", "unknown")).set(1)
        _refresh_drift_gauges()
    return _model


def _refresh_drift_gauges():
    if DRIFT_PATH.exists():
        d = json.loads(DRIFT_PATH.read_text())
        DRIFT_SHARE.set(d.get("share_drifted", 0.0))
        DATASET_DRIFT.set(1.0 if d.get("dataset_drift") else 0.0)


class Applicant(BaseModel):
    age: float
    income: float
    loan_amount: float
    employment_length: float
    debt_to_income: float
    credit_history_length: float
    num_delinquencies: float
    utilization: float = Field(..., ge=0, le=1)
    num_credit_lines: float
    interest_rate: float
    home_ownership: str
    purpose: str


class Prediction(BaseModel):
    default_probability: float
    decision: str
    threshold: float
    model_run_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None,
            "run_id": _model.get("run_id") if _model else None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/drift")
def drift():
    if not DRIFT_PATH.exists():
        raise HTTPException(404, "no drift report yet; run the drift stage")
    return json.loads(DRIFT_PATH.read_text())


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    model = _load()
    t0 = time.perf_counter()
    row = pd.DataFrame([applicant.model_dump()])[FEATURES]
    proba = float(model["pipeline"].predict_proba(row)[0, 1])
    PREDICT_LATENCY.observe(time.perf_counter() - t0)

    decision = "decline" if proba >= THRESHOLD else "approve"
    PREDICTIONS.labels(outcome=decision).inc()
    SCORE_HIST.observe(proba)
    return Prediction(default_probability=round(proba, 5), decision=decision,
                      threshold=THRESHOLD, model_run_id=model.get("run_id", "unknown"))
