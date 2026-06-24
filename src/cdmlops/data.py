"""Synthetic credit-default dataset generator (deterministic, no external download).

Produces two slices so the drift stage has something real to detect:
  - reference.parquet : the training distribution
  - current.parquet   : a later "production" slice with deliberate covariate +
                        prediction drift (macro shock: incomes down, utilization and
                        delinquencies up, rates up) — exactly what a monitor must catch.

The target is a logistic function of the features + noise, so the model has genuine
signal to learn and the gate thresholds are meetable.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import CATEGORICAL, CUR_PARQUET, DATA_DIR, NUMERIC, REF_PARQUET, TARGET

HOME = ["rent", "own", "mortgage"]
PURPOSE = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"]


def _sample(n: int, rng: np.random.Generator, drift: bool) -> pd.DataFrame:
    # macro shock applied to the "current" slice — a realistic multi-feature downturn
    # (incomes down, leverage + delinquencies + rates up, tenures shorter)
    income_mult = 0.72 if drift else 1.0
    util_shift = 0.16 if drift else 0.0
    delinq_mult = 2.2 if drift else 1.0
    rate_shift = 4.5 if drift else 0.0
    emp_mult = 0.7 if drift else 1.0

    age = (rng.normal(40, 12, n) - (4 if drift else 0)).clip(18, 85)
    income = (rng.lognormal(10.9, 0.5, n) * income_mult).clip(8_000, 500_000)
    loan_amount = (income * rng.uniform(0.05, 0.6, n)).clip(1_000, 100_000)
    employment_length = (rng.gamma(3, 2, n) * emp_mult).clip(0, 40)
    dti = (loan_amount / income * rng.uniform(0.8, 1.5, n)).clip(0.01, 1.5)
    credit_history_length = rng.gamma(4, 3, n).clip(0, 50)
    num_delinquencies = rng.poisson(0.4 * delinq_mult, n).clip(0, 12)
    utilization = (rng.beta(2, 5, n) + util_shift).clip(0, 1)
    num_credit_lines = rng.poisson(6, n).clip(0, 30)
    interest_rate = (rng.normal(13, 4, n) + rate_shift).clip(3, 35)

    df = pd.DataFrame({
        "age": age, "income": income, "loan_amount": loan_amount,
        "employment_length": employment_length, "debt_to_income": dti,
        "credit_history_length": credit_history_length,
        "num_delinquencies": num_delinquencies, "utilization": utilization,
        "num_credit_lines": num_credit_lines, "interest_rate": interest_rate,
        "home_ownership": rng.choice(HOME, n, p=[0.4, 0.25, 0.35]),
        "purpose": rng.choice(PURPOSE, n, p=[0.35, 0.25, 0.15, 0.1, 0.15]),
    })

    # default propensity: higher DTI / utilization / delinquencies / rate -> more default
    z = (
        -3.6
        + 3.6 * df.debt_to_income
        + 3.6 * df.utilization
        + 0.55 * df.num_delinquencies
        + 0.11 * df.interest_rate
        - 0.000008 * df.income
        - 0.06 * df.employment_length
        - 0.03 * df.credit_history_length
        + (df.home_ownership == "rent").astype(float) * 0.6
    )
    p = 1 / (1 + np.exp(-z))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df


def generate(n_ref: int = 12_000, n_cur: int = 4_000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ref = _sample(n_ref, rng, drift=False)
    cur = _sample(n_cur, np.random.default_rng(seed + 1), drift=True)
    return ref, cur


def main() -> None:
    ap = argparse.ArgumentParser(description="generate the credit-default dataset")
    ap.add_argument("--n-ref", type=int, default=12_000)
    ap.add_argument("--n-cur", type=int, default=4_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ref, cur = generate(args.n_ref, args.n_cur, args.seed)
    ref.to_parquet(REF_PARQUET, index=False)
    cur.to_parquet(CUR_PARQUET, index=False)
    print(f"reference: {len(ref):,} rows (default rate {ref[TARGET].mean():.3f}) -> {REF_PARQUET}")
    print(f"current  : {len(cur):,} rows (default rate {cur[TARGET].mean():.3f}) -> {CUR_PARQUET}")
    assert set(NUMERIC + CATEGORICAL).issubset(ref.columns)


if __name__ == "__main__":
    main()
