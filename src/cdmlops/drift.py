"""Data-drift detection: reference vs. current ("production") distribution.

Primary signal is a self-contained **Population Stability Index (PSI)** per feature
(numpy, deterministic) — it feeds the JSON report and the Prometheus gauge and never
depends on a third-party API. We *also* emit a rich **Evidently** HTML report for the
visual dashboard (best-effort, tolerant of Evidently's fast-moving API).

PSI rule of thumb: <0.1 stable · 0.1–0.2 moderate shift · >0.2 significant drift.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL, CUR_PARQUET, DRIFT_PATH, DRIFT_SHARE_THRESHOLD, NUMERIC,
    REF_PARQUET, REPORTS,
)

PSI_DRIFT = 0.2  # per-feature PSI above this counts the column as drifted


def _psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).values
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def compute_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    psi: dict[str, float] = {}
    for col in NUMERIC:
        psi[col] = _psi_numeric(ref[col].values, cur[col].values)
    for col in CATEGORICAL:
        psi[col] = _psi_categorical(ref[col], cur[col])
    drifted = [c for c, v in psi.items() if v > PSI_DRIFT]
    share = len(drifted) / len(psi)
    return {
        "psi": {k: round(v, 4) for k, v in sorted(psi.items(), key=lambda kv: -kv[1])},
        "n_features": len(psi),
        "n_drifted": len(drifted),
        "share_drifted": round(share, 4),
        "drifted_features": drifted,
        "dataset_drift": share > DRIFT_SHARE_THRESHOLD,
    }


def evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, path) -> bool:
    """Best-effort Evidently report — the numpy PSI above is the source of truth."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=ref, current_data=cur)
        result.save_html(str(path))
        return True
    except Exception as e:  # noqa: BLE001 — Evidently API churn shouldn't break the pipeline
        print(f"[drift] Evidently HTML skipped: {type(e).__name__}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="reference-vs-current drift report")
    ap.add_argument("--reference", default=str(REF_PARQUET))
    ap.add_argument("--current", default=str(CUR_PARQUET))
    ap.add_argument("--fail-on-drift", action="store_true", help="exit 1 if dataset drift detected")
    args = ap.parse_args()

    ref = pd.read_parquet(args.reference)
    cur = pd.read_parquet(args.current)
    report = compute_drift(ref, cur)

    REPORTS.mkdir(parents=True, exist_ok=True)
    DRIFT_PATH.write_text(json.dumps(report, indent=2))
    report["evidently_html"] = evidently_html(ref, cur, REPORTS / "drift.html")

    top = list(report["psi"].items())[:5]
    print(f"[drift] {report['n_drifted']}/{report['n_features']} features drifted "
          f"(share={report['share_drifted']}, dataset_drift={report['dataset_drift']})")
    print("[drift] top PSI: " + ", ".join(f"{k}={v}" for k, v in top))
    print(f"[drift] report -> {DRIFT_PATH}")

    if args.fail_on_drift and report["dataset_drift"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
