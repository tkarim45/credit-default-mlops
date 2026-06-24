"""credit-default-mlops — a production MLOps pipeline around a credit-default classifier.

Pipeline: data (DVC) -> train (MLflow) -> evaluate (quality gate) -> drift (Evidently)
-> serve (FastAPI + Prometheus). See README for the full architecture.
"""

__version__ = "0.1.0"
