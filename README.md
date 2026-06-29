# 🏦 Credit-Default MLOps — train, gate, monitor, serve

> **A production MLOps pipeline around a credit-default classifier.** Versioned data
> (DVC) → tracked + registered model (MLflow) → an **automated quality gate** that blocks
> bad models in CI → **data-drift detection** (PSI + Evidently) → a **FastAPI** service
> instrumented for **Prometheus + Grafana**. The whole thing is reproducible (`dvc repro`)
> and self-contained (synthetic data — runs offline, no dataset download, no API keys).

The gap most ML portfolios never cross is "notebook → operated system." This repo is the
operations layer: experiment tracking, a model registry, a CI gate, drift monitoring with
alerting, and a metricised serving endpoint — the things a model needs to live in
production rather than die in a notebook.

---

## What it demonstrates

| Concern | How |
|---|---|
| **Data versioning + reproducibility** | `dvc.yaml` pipeline (prepare → train → evaluate → drift); `dvc repro` reruns only what changed |
| **Experiment tracking + registry** | every train run logs params/metrics/model to **MLflow** and registers `credit-default-classifier` |
| **Automated quality gate** | `evaluate.py` exits non-zero if ROC-AUC / PR-AUC / Brier miss thresholds → **CI fails the build** |
| **Drift detection** | per-feature **PSI** (numpy, deterministic) + an **Evidently** HTML report; dataset-drift alarm |
| **Serving + observability** | **FastAPI** `/predict` + `/metrics`; **Prometheus** scrape + **Grafana** dashboard (latency, throughput, score dist, drift) |
| **CI/CD** | GitHub Actions: test → `dvc repro` (gate) → publish artifacts |
| **Containerized stack** | `docker compose up` → API + Prometheus + Grafana |

---

## Architecture

```
            DVC pipeline (dvc repro)                         observability stack
  ┌──────────────────────────────────────────┐      ┌──────────────────────────────┐
  prepare ─► train ─► evaluate(GATE) ─► drift  │      │  FastAPI /predict /metrics   │
   data.py   MLflow    exit≠0 fails   PSI +    │      │       │ Prometheus scrape    │
             track +   the build      Evidently│      │       ▼                      │
             register                          │      │   Grafana dashboard          │
  └─────────────┬───────────────┬──────────────┘      │   (latency · drift · score)  │
                │ model.pkl      │ drift.json  ───────►│                              │
                └────────────────┴─────────────────────┘                              │
                                                       └──────────────────────────────┘
```

---

## Quickstart

> Uses the conda **`personal`** env (per environment conventions — never `base`).

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY -m pip install -e ".[all]"

# --- run the pipeline (DVC, reproducible) ------------------------------------
$PY -m dvc repro                 # prepare -> train -> GATE -> drift
#   or without DVC:  make pipeline   (generate -> train -> gate -> drift)

$PY -m mlflow ui --backend-store-uri mlruns --port 5000   # browse runs + registry

# --- serve + observe ----------------------------------------------------------
$PY -m uvicorn api.main:app --port 8000      # /predict, /metrics, /drift, /health
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "age":35,"income":42000,"loan_amount":18000,"employment_length":4,
  "debt_to_income":0.43,"credit_history_length":9,"num_delinquencies":2,
  "utilization":0.72,"num_credit_lines":6,"interest_rate":17.5,
  "home_ownership":"rent","purpose":"debt_consolidation"}'

# full stack with dashboards:
docker compose up --build        # API :8000 · Prometheus :9090 · Grafana :3000
```

---

## The quality gate (the MLOps heart)

`src/cdmlops/config.py` declares thresholds; `evaluate.py` enforces them and **exits
non-zero on failure**, so the `evaluate` DVC stage — and therefore CI — fails on a bad model:

```python
GATE = {"roc_auc": 0.70, "pr_auc": 0.45, "brier_max": 0.20}
```

This is what stops a regression from ever being registered or shipped. Tighten the floors
and the build tells you immediately.

---

## Drift detection

The `drift` stage compares the training **reference** distribution against a later
**current** ("production") slice. The synthetic generator injects a deliberate macro shock
into the current slice (incomes down, utilization/delinquencies/rates up) so the monitor
has real drift to catch:

- **PSI per feature** (numpy, deterministic) — the source of truth feeding `reports/drift.json`
  and the Prometheus `cdm_drift_share` / `cdm_dataset_drift` gauges.
- **Evidently** HTML report (`reports/drift.html`) for the rich visual — best-effort, so
  Evidently's fast-moving API can never break the pipeline.

PSI guide: <0.1 stable · 0.1–0.2 moderate · >0.2 drifted. Dataset-drift alarm fires when
the share of drifted features exceeds 0.25.

---

## Serving metrics (Prometheus)

`/metrics` exposes: `cdm_predictions_total{outcome}` (throughput, approve/decline),
`cdm_predict_latency_seconds` (histogram → p50/p95), `cdm_predicted_default_probability`
(score distribution — catches prediction drift), `cdm_drift_share` / `cdm_dataset_drift`
(data drift), `cdm_model_info{run_id}` (deployed model lineage). The bundled Grafana
dashboard (`monitoring/grafana/`) renders all of them.

---

## Repo layout

```
credit-default-mlops/
├── src/cdmlops/
│   ├── data.py        synthetic credit dataset (reference + drifted current slice)
│   ├── features.py    sklearn preprocessing (shipped inside the model)
│   ├── train.py       XGBoost pipeline -> MLflow track + register -> model.pkl
│   ├── evaluate.py    metrics (ROC-AUC, PR-AUC, Brier, KS) + the CI quality gate
│   ├── drift.py       PSI drift + Evidently report
│   ├── monitoring.py  Prometheus instruments
│   └── config.py      paths, schema, gate thresholds
├── api/main.py        FastAPI /predict · /metrics · /drift · /health
├── dvc.yaml           reproducible pipeline (prepare → train → evaluate → drift)
├── monitoring/        prometheus.yml + Grafana datasource & dashboard
├── docker-compose.yml app + Prometheus + Grafana
├── tests/             model-free unit tests (gate · KS · drift)
└── .github/workflows/ci.yml   test → dvc repro (gate) → upload artifacts
```

---

## Résumé framing

> *Built a production MLOps pipeline for a credit-default classifier — DVC-versioned data,
> MLflow tracking + model registry, an automated CI quality gate (ROC-AUC/PR-AUC/Brier),
> PSI + Evidently drift monitoring, and a Prometheus-instrumented FastAPI service with a
> Grafana dashboard; fully reproducible via `dvc repro` and containerized with Docker Compose.*

## License
MIT (`LICENSE`).
