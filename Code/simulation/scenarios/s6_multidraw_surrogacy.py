"""
S6: Multi-draw surrogacy relaxation (validates Remark on multi-draw surrogacy).

The headline stochastic scenario. Y depends on theta (the latent surrogate),
NOT on a single noisy draw Y*. Single-draw surrogacy approximately FAILS
when noise is large, but multi-draw surrogacy recovers as K grows because
the averaged Y-bar converges to theta.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dgp import MultiDrawSurrogacyDGP
from estimators import fit_calibration, calibrated_ate, raw_llm_ate
from monte_carlo import run_mc, summarize


def one_rep(seed, K=1, n=2000, sigma_eps=1.5, method="ols"):
    dgp = MultiDrawSurrogacyDGP(n=n, sigma_eps=sigma_eps, sigma_eta=0.2,
                                 seed=seed)

    # P=0: observe (W, X, Y*, Y) -- use K draws for surrogate
    p0 = dgp.generate_sample(population=0, K=K)

    # P=1: observe (W, X, Y*) only -- use same K
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
    }


def run_s6(n_reps=1000, seed=42):
    K_values = [1, 2, 5, 10, 20, 50, 100, 200]
    results = {}
    true_ate = MultiDrawSurrogacyDGP().true_ate()

    for K in K_values:
        df = run_mc(one_rep, n_reps=n_reps, seed=seed, K=K)
        results[f"K={K}"] = df
        s = summarize(df, true_ate)
        print(f"S6 K={K:3d}: bias={s['bias']:.4f}  rmse={s['rmse']:.4f}  "
              f"mean={s['mean_estimate']:.4f}  true={true_ate:.4f}")

    return results


if __name__ == "__main__":
    run_s6(n_reps=100)
