"""
S3: Comparability violation (validates importance of Assumption 2).

The calibration mapping mu differs between P=0 and P=1:
    P=0: Y = b0 + b1*Y* + b2*X + eta
    P=1: Y_true = b0 + (b1 + delta)*Y* + b2*X + eta

The calibrated ATE (trained on P=0) misses the delta contribution,
producing bias that grows with delta.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dgp import ComparabilityViolationDGP
from estimators import fit_calibration, calibrated_ate, raw_llm_ate
from monte_carlo import run_mc, summarize


def one_rep(seed, delta=0.0, n=2000, sigma_eps=0.3, method="ols"):
    dgp = ComparabilityViolationDGP(delta=delta, n=n, sigma_eps=sigma_eps,
                                     sigma_eta=0.2, seed=seed)

    # P=0 uses the base mapping (b1 coefficient)
    p0 = dgp.generate_sample(population=0)

    # P=1: generate Y* (same LLM DGP) and also "true" Y under P=1 mapping
    p1 = dgp.generate_sample(population=1)
    Y_true_p1 = dgp.generate_outcome_p1(p1.Ystar, p1.X)

    # Train calibration on P=0
    model = fit_calibration(p0.Ystar, p0.X, p0.Y, method=method)
    tau_cal = calibrated_ate(model, p1.Ystar, p1.X, p1.W)
    tau_raw = raw_llm_ate(p1.Ystar, p1.W)

    # "True" ATE under P=1 mapping
    tau_true_p1 = Y_true_p1[p1.W == 1].mean() - Y_true_p1[p1.W == 0].mean()

    return {
        "tau_calibrated": tau_cal,
        "tau_raw_llm": tau_raw,
        "tau_true_p1": tau_true_p1,
        "tau_true_p0": dgp.true_ate(),
        "tau_true_p1_analytical": dgp.true_ate_p1(),
        "delta": delta,
    }


def run_s3(n_reps=1000, seed=42):
    delta_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = {}

    for delta in delta_values:
        df = run_mc(one_rep, n_reps=n_reps, seed=seed, delta=delta)
        results[f"delta={delta}"] = df
        dgp = ComparabilityViolationDGP(delta=delta)
        true_p1 = dgp.true_ate_p1()
        s = summarize(df, true_p1)
        print(f"S3 delta={delta:.2f}: bias={s['bias']:.4f}  "
              f"rmse={s['rmse']:.4f}  true_ate_p1={true_p1:.4f}")

    return results


if __name__ == "__main__":
    run_s3(n_reps=100)
