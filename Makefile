PY  ?= ~/miniconda3/envs/personal/bin/python
PIP ?= ~/miniconda3/envs/personal/bin/pip

.PHONY: install pipeline generate train gate drift repro serve mlflow-ui test stack

install:
	$(PIP) install -e ".[all]"

# full pipeline without DVC (handy for a quick local run)
pipeline: generate train gate drift

generate:
	$(PY) -m cdmlops.data

train:
	$(PY) -m cdmlops.train

gate:
	$(PY) -m cdmlops.evaluate

drift:
	$(PY) -m cdmlops.drift

repro:           # reproducible DVC run: prepare -> train -> gate -> drift
	$(PY) -m dvc repro

serve:
	$(PY) -m uvicorn api.main:app --reload --port 8000

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri mlruns --port 5000

test:
	$(PY) -m pytest -q

stack:           # API + Prometheus + Grafana
	docker compose up --build
