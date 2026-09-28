"""
S1: Identification with stochastic surrogates (validates Theorem 1).

Both assumptions hold. Stochastic Y* is the default.
Shows that raw LLM ATE is biased but calibrated ATE recovers true tau.
Runs both linear and nonlinear DGP variants.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dgp import LinearDGP, NonlinearDGP
from estimators import fit_calibration, calibrated_ate, raw_llm_ate
from monte_carlo import run_mc, summarize


def one_rep_linear(seed, n=2000, sigma_eps=0.3, method="ols"):
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
        "sigma_eps": sigma_eps,
        "variant": "linear",
        "method": method,
    }


def one_rep_nonlinear(seed, n=2000, sigma_eps=0.2, method="rf"):
    dgp = NonlinearDGP(n=n, sigma_eps=sigma_eps, sigma_eta=0.15, seed=seed)
    p0 = dgp.generate_sample(population=0)
    p1 = dgp.generate_sample(population=1)

    model = fit_calibration(p0.Ystar, p0.X, p0.Y, method=method)
    tau_cal = calibrated_ate(model, p1.Ystar, p1.X, p1.W)
    tau_raw = raw_llm_ate(p1.Ystar, p1.W)

    return {
        "tau_calibrated": tau_cal,
        "tau_raw_llm": tau_raw,
        "tau_true": dgp.true_ate(),
        "sigma_eps": sigma_eps,
        "variant": "nonlinear",
        "method": method,
    }


def run_s1(n_reps=1000, seed=42):
    results = {}

    # Linear: stochastic
    df_lin = run_mc(one_rep_linear, n_reps=n_reps, seed=seed)
    results["linear_stochastic"] = df_lin
    true_ate_lin = LinearDGP().true_ate()
    print(f"S1 Linear (stochastic): {summarize(df_lin, true_ate_lin)}")

    # Linear: deterministic
    df_lin_det = run_mc(one_rep_linear, n_reps=n_reps, seed=seed,
                        sigma_eps=0.0)
    results["linear_deterministic"] = df_lin_det
    print(f"S1 Linear (deterministic): {summarize(df_lin_det, true_ate_lin)}")

    # Nonlinear: stochastic
    df_nl = run_mc(one_rep_nonlinear, n_reps=n_reps, seed=seed)
    results["nonlinear_stochastic"] = df_nl
    true_ate_nl = NonlinearDGP().true_ate()
    print(f"S1 Nonlinear (stochastic): {summarize(df_nl, true_ate_nl)}")

    # Nonlinear: deterministic
    df_nl_det = run_mc(one_rep_nonlinear, n_reps=n_reps, seed=seed,
                       sigma_eps=0.0)
    results["nonlinear_deterministic"] = df_nl_det
    print(f"S1 Nonlinear (deterministic): {summarize(df_nl_det, true_ate_nl)}")

    return results


if __name__ == "__main__":
    run_s1(n_reps=100)
