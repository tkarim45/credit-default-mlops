"""Metrics + the quality gate. `check_gate` exits non-zero so CI blocks a bad model."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import GATE, METRICS_PATH


def ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Kolmogorov–Smirnov separation between default/non-default score distributions."""
    order = np.argsort(proba)
    y = np.asarray(y_true)[order]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return 0.0
    tpr = np.cumsum(y) / pos
    fpr = np.cumsum(1 - y) / neg
    return float(np.max(np.abs(tpr - fpr)))


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "ks": ks_statistic(y_true, proba),
    }


def check_gate(metrics: dict) -> tuple[bool, list[str]]:
    """Return (passed, failures). Brier is a ceiling; the rest are floors."""
    fails = []
    if metrics["roc_auc"] < GATE["roc_auc"]:
        fails.append(f"roc_auc {metrics['roc_auc']:.4f} < {GATE['roc_auc']}")
    if metrics["pr_auc"] < GATE["pr_auc"]:
        fails.append(f"pr_auc {metrics['pr_auc']:.4f} < {GATE['pr_auc']}")
    if metrics["brier"] > GATE["brier_max"]:
        fails.append(f"brier {metrics['brier']:.4f} > {GATE['brier_max']}")
    return (not fails), fails


def main() -> None:
    ap = argparse.ArgumentParser(description="enforce the quality gate on artifacts/metrics.json")
    ap.add_argument("--metrics", default=str(METRICS_PATH))
    args = ap.parse_args()

    metrics = json.loads(open(args.metrics).read())
    passed, fails = check_gate(metrics)
    print("[gate] metrics: " + "  ".join(f"{k}={metrics[k]:.4f}" for k in ("roc_auc", "pr_auc", "brier", "ks")))
    if passed:
        print("[gate] PASS ✅")
        sys.exit(0)
    print("[gate] FAIL ❌ — " + "; ".join(fails))
    sys.exit(1)


if __name__ == "__main__":
    main()
