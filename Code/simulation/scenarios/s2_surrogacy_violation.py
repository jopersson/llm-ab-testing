"""
S2: Surrogacy violation (validates importance of Assumption 1).

Adds a direct effect of W on Y that bypasses Y*:
    Y = mu(Y*, X) + gamma * W + eta

Varies gamma from 0 to 0.5. Shows calibrated ATE bias grows with gamma.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dgp import SurrogacyViolationDGP
from estimators import fit_calibration, calibrated_ate, raw_llm_ate
from monte_carlo import run_mc, summarize


def one_rep(seed, gamma=0.0, n=2000, sigma_eps=0.3, method="ols"):
    dgp = SurrogacyViolationDGP(gamma=gamma, n=n, sigma_eps=sigma_eps,
                                 sigma_eta=0.2, seed=seed)
    p0 = dgp.generate_sample(population=0)
    p1 = dgp.generate_sample(population=1)

    # Train calibration on P=0 (regression of Y on Y*, X -- omitting W)
    model = fit_calibration(p0.Ystar, p0.X, p0.Y, method=method)
    tau_cal = calibrated_ate(model, p1.Ystar, p1.X, p1.W)
    tau_raw = raw_llm_ate(p1.Ystar, p1.W)

    return {
        "tau_calibrated": tau_cal,
        "tau_raw_llm": tau_raw,
        "tau_true": dgp.true_ate(),
        "gamma": gamma,
    }


def run_s2(n_reps=1000, seed=42):
    gamma_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = {}

    for gamma in gamma_values:
        df = run_mc(one_rep, n_reps=n_reps, seed=seed, gamma=gamma)
        results[f"gamma={gamma}"] = df
        true_ate = SurrogacyViolationDGP(gamma=gamma).true_ate()
        s = summarize(df, true_ate)
        print(f"S2 gamma={gamma:.2f}: bias={s['bias']:.4f}  "
              f"rmse={s['rmse']:.4f}  true_ate={true_ate:.4f}")

    return results


if __name__ == "__main__":
    run_s2(n_reps=100)
