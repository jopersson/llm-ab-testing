"""
S5: Back-testing necessary but not sufficient (validates Section 5.1).

Multi-treatment setup: W in {0, 1, 2}.
Historical treatment (W=1 vs W=0): calibrated ATE is correct.
Novel treatment (W=2 vs W=0): calibrated ATE fails because the novel
treatment operates through a channel not captured by Y*.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dgp import BackTestingDGP
from estimators import fit_calibration, predict_calibrated
from monte_carlo import run_mc


def one_rep(seed, n=3000, sigma_eps=0.3, gamma_novel=0.3):
    dgp = BackTestingDGP(n=n, sigma_eps=sigma_eps, sigma_eta=0.2,
                          gamma_novel=gamma_novel, seed=seed)

    p0 = dgp.generate_sample(population=0,
                              treatment_probs=[1/3, 1/3, 1/3])

    # Train calibration on P=0 using (Y*, X) -> Y, ignoring W
    from sklearn.linear_model import LinearRegression
    features = np.column_stack([p0.Ystar, p0.X])
    model = LinearRegression().fit(features, p0.Y)

    # Generate P=1 with all three treatment arms
    p1 = dgp.generate_sample(population=1,
                              treatment_probs=[1/3, 1/3, 1/3])
    Y_hat = predict_calibrated(model, p1.Ystar, p1.X)

    # Calibrated ATEs
    cal_ate_hist = (Y_hat[p1.W == 1].mean() - Y_hat[p1.W == 0].mean())
    cal_ate_novel = (Y_hat[p1.W == 2].mean() - Y_hat[p1.W == 0].mean())

    return {
        "cal_ate_historical": cal_ate_hist,
        "cal_ate_novel": cal_ate_novel,
        "true_ate_historical": dgp.true_ate_historical(),
        "true_ate_novel": dgp.true_ate_novel(),
        "bias_historical": cal_ate_hist - dgp.true_ate_historical(),
        "bias_novel": cal_ate_novel - dgp.true_ate_novel(),
    }


def run_s5(n_reps=1000, seed=42):
    df = run_mc(one_rep, n_reps=n_reps, seed=seed)

    true_hist = BackTestingDGP().true_ate_historical()
    true_novel = BackTestingDGP().true_ate_novel()

    print(f"S5 Historical (W=1 vs W=0):")
    print(f"  True ATE:       {true_hist:.4f}")
    print(f"  Mean calibrated: {df['cal_ate_historical'].mean():.4f}")
    print(f"  Mean bias:       {df['bias_historical'].mean():.4f}")
    print(f"S5 Novel (W=2 vs W=0):")
    print(f"  True ATE:       {true_novel:.4f}")
    print(f"  Mean calibrated: {df['cal_ate_novel'].mean():.4f}")
    print(f"  Mean bias:       {df['bias_novel'].mean():.4f}")

    return {"backtesting": df}


if __name__ == "__main__":
    run_s5(n_reps=100)
