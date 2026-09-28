"""
S7: K-draw estimation consequences (validates Proposition 1 + Remark on attenuation).

Uses the MultiDrawSurrogacyDGP (Y depends on theta, not Y*) with moderate
noise (sigma_eps=0.5). Varies K from 1 to 50. Shows:
  - RMSE decreases with K (variance reduction from Proposition 1)
  - Mean calibrated ATE de-attenuates toward true tau (Remark on attenuation)

The attenuation arises because estimating mu by regressing Y on (Y_bar, X)
treats the noise in Y_bar as classical measurement error. Larger K reduces
this noise, improving both precision and bias.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dgp import MultiDrawSurrogacyDGP
from estimators import fit_calibration, calibrated_ate, raw_llm_ate
from monte_carlo import run_mc, summarize


def one_rep(seed, K=1, n=2000, sigma_eps=0.5, method="ols"):
    dgp = MultiDrawSurrogacyDGP(n=n, sigma_eps=sigma_eps, sigma_eta=0.2,
                                 seed=seed)
    p0 = dgp.generate_sample(population=0, K=K)
    p1 = dgp.generate_sample(population=1, K=K)

    model = fit_calibration(p0.Ystar, p0.X, p0.Y, method=method)
    tau_cal = calibrated_ate(model, p1.Ystar, p1.X, p1.W)
    tau_raw = raw_llm_ate(p1.Ystar, p1.W)

    return {
        "tau_calibrated": tau_cal,
        "tau_raw_llm": tau_raw,
        "tau_true": dgp.true_ate(),
        "K": K,
        "sigma_eps": sigma_eps,
        "var_theta_given_X": _var_theta_given_X(dgp),
    }


def _var_theta_given_X(dgp):
    """
    Closed-form E_X[Var(theta | X)] for MultiDrawSurrogacyDGP.

    theta(W, X) = a0 + a1*W + a2*X + a3*W*X. Conditional on X, the only
    randomness is W ~ Bern(0.5), so Var(theta | X) = (a1 + a3*X)^2 / 4.
    Averaging over X ~ U(0, 1):
        E_X[Var(theta | X)] = (1/4) * E_X[(a1 + a3*X)^2]
                            = (1/4) * (a1^2 + a1*a3 + a3^2/3).
    """
    a1, a3 = dgp.a1, dgp.a3
    return 0.25 * (a1**2 + a1 * a3 + a3**2 / 3.0)


def reliability_ratio(K, var_theta_given_X, sigma_eps):
    """R_K = Var(theta|X) / [Var(theta|X) + sigma_eps^2 / K]."""
    return var_theta_given_X / (var_theta_given_X + sigma_eps**2 / K)


def run_s7(n_reps=1000, seed=42):
    K_values = [1, 2, 3, 5, 10, 20, 30, 50]
    results = {}
    base_dgp = MultiDrawSurrogacyDGP(sigma_eps=0.5)
    true_ate = base_dgp.true_ate()
    var_theta_given_X = _var_theta_given_X(base_dgp)

    for K in K_values:
        df = run_mc(one_rep, n_reps=n_reps, seed=seed, K=K)
        results[f"K={K}"] = df
        s = summarize(df, true_ate)
        R_K = reliability_ratio(K, var_theta_given_X, base_dgp.sigma_eps)
        print(f"S7 K={K:3d}: bias={s['bias']:.4f}  rmse={s['rmse']:.4f}  "
              f"mean={s['mean_estimate']:.4f}  true={true_ate:.4f}  "
              f"R_K={R_K:.3f}  predicted_mean={R_K*true_ate:.4f}")

    return results


if __name__ == "__main__":
    run_s7(n_reps=100)
