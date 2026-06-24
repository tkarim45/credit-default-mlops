"""Model-free unit tests: metrics, the quality gate, and drift detection."""
import numpy as np

from cdmlops.data import generate
from cdmlops.drift import compute_drift
from cdmlops.evaluate import check_gate, evaluate_predictions, ks_statistic


def test_perfect_scores_pass_gate():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    proba = y * 0.99 + (1 - y) * 0.01  # near-perfect separation
    m = evaluate_predictions(y, proba)
    assert m["roc_auc"] == 1.0
    passed, fails = check_gate(m)
    assert passed and not fails


def test_random_scores_fail_gate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = rng.random(500)  # no signal -> AUC ~ 0.5 -> below floor
    passed, fails = check_gate(evaluate_predictions(y, proba))
    assert not passed and any("roc_auc" in f for f in fails)


def test_ks_in_unit_interval():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    assert 0.0 <= ks_statistic(y, rng.random(200)) <= 1.0


def test_no_drift_between_reference_halves():
    ref, _ = generate(n_ref=4000, n_cur=10, seed=3)
    a, b = ref.iloc[:2000], ref.iloc[2000:]
    report = compute_drift(a, b)
    assert report["share_drifted"] < 0.3 and not report["dataset_drift"]


def test_macro_shock_triggers_dataset_drift():
    ref, cur = generate(n_ref=4000, n_cur=2000, seed=3)
    report = compute_drift(ref, cur)
    # the current slice applies an income/utilization/rate shock -> drift must fire
    assert report["dataset_drift"]
    assert "utilization" in report["drifted_features"] or "income" in report["drifted_features"]
