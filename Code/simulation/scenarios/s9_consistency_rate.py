"""
S9: Consistency-rate verification (validates Theorem 3.1 / sqrt(n)-rate).

Under Assumptions 1-2, Theorem 3.1 establishes consistency of the calibrated
ATE: hat_tau_K -p-> tau as n -> infty. For parametric calibration estimated
on n iid units, the rate is sqrt(n). This scenario varies the per-population
sample size over a wide grid and verifies that

    RMSE(hat_tau_K, n) ~ C / sqrt(n),   <=>   RMSE * sqrt(n) -> constant.

DGP: LinearDGP with default parameters (where surrogacy and comparability hold
by construction). Calibration: OLS of Y on (Y*, X) on P=0; calibrated ATE
applied to P=1.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from dgp import LinearDGP
from estimators import fit_calibration, calibrated_ate, raw_llm_ate
from monte_carlo import run_mc, summarize


def one_rep(seed, n=2000, sigma_eps=0.3, method="ols"):
    dgp = LinearDGP(n=n, sigma_eps=sigma_eps, sigma_eta=0.2, seed=seed)
    p0 = dgp.generate_sample(population=0)
    p1 = dgp.generate_sample(population=1)

    model = fit_calibration(p0.Ystar, p0.X, p0.Y, method=method)
    tau_cal = calibrated_ate(model, p1.Ystar, p1.X, p1.W)
    tau_raw = raw_llm_ate(p1.Ystar, p1.W)

    return {
        "tau_calibrated": tau_cal,
        "tau_raw_llm": tau_raw,
        "tau_true": dgp.true_ate(),
        "n": n,
        "sigma_eps": sigma_eps,
    }


def run_s9(n_reps=500, seed=42):
    n_values = [250, 500, 1000, 2000, 5000, 10000, 20000]
    results = {}
    true_ate = LinearDGP(sigma_eps=0.3).true_ate()

    for n in n_values:
        df = run_mc(one_rep, n_reps=n_reps, seed=seed, n=n)
        results[f"n={n}"] = df
        s = summarize(df, true_ate)
        rmse_root_n = s["rmse"] * np.sqrt(n)
        print(f"S9 n={n:6d}: bias={s['bias']:+.5f}  rmse={s['rmse']:.5f}  "
              f"rmse*sqrt(n)={rmse_root_n:.4f}  true={true_ate:.4f}")

    return results


if __name__ == "__main__":
    run_s9(n_reps=200)
