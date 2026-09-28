"""
S4: Sensitivity bound validation (validates Proposition 2).

Shifts the distribution of Y* in P=1 relative to P=0.
Shows that the actual bias stays below the theoretical bound
  2B * [(1 - OVL_1) + (1 - OVL_0)].
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dgp import SensitivityBoundDGP
from estimators import fit_calibration, calibrated_ate
from overlap import estimate_ovl, sensitivity_bound
from monte_carlo import run_mc


def one_rep(seed, shift_star=0.0, n=2000, sigma_eps=0.3, B=3.0):
    dgp = SensitivityBoundDGP(shift_star=shift_star, n=n,
                               sigma_eps=sigma_eps, sigma_eta=0.2,
                               seed=seed)
    p0 = dgp.generate_sample(population=0)
    p1 = dgp.generate_sample(population=1)

    # Clip Y to [-B, B] so Assumption 3 holds exactly
    p0.Y = np.clip(p0.Y, -B, B)

    model = fit_calibration(p0.Ystar, p0.X, p0.Y, method="ols")

    from estimators import predict_calibrated
    Y_hat_p1 = predict_calibrated(model, p1.Ystar, p1.X)
    Y_hat_p0 = predict_calibrated(model, p0.Ystar, p0.X)

    Y_hat_p1 = np.clip(Y_hat_p1, -B, B)
    Y_hat_p0 = np.clip(Y_hat_p0, -B, B)

    tau_cal_p1 = Y_hat_p1[p1.W == 1].mean() - Y_hat_p1[p1.W == 0].mean()
    tau_cal_p0 = Y_hat_p0[p0.W == 1].mean() - Y_hat_p0[p0.W == 0].mean()
    actual_discrepancy = abs(tau_cal_p1 - tau_cal_p0)

    # Estimate overlap per arm
    Z_p0_w1 = np.column_stack([p0.X[p0.W == 1], p0.Ystar[p0.W == 1]])
    Z_p1_w1 = np.column_stack([p1.X[p1.W == 1], p1.Ystar[p1.W == 1]])
    Z_p0_w0 = np.column_stack([p0.X[p0.W == 0], p0.Ystar[p0.W == 0]])
    Z_p1_w0 = np.column_stack([p1.X[p1.W == 0], p1.Ystar[p1.W == 0]])

    ovl_1 = estimate_ovl(Z_p0_w1, Z_p1_w1, seed=seed)
    ovl_0 = estimate_ovl(Z_p0_w0, Z_p1_w0, seed=seed)
    bound = sensitivity_bound(B, ovl_1, ovl_0)

    return {
        "tau_cal_p1": tau_cal_p1,
        "tau_cal_p0": tau_cal_p0,
        "actual_discrepancy": actual_discrepancy,
        "ovl_1": ovl_1,
        "ovl_0": ovl_0,
        "bound": bound,
        "bound_holds": int(actual_discrepancy <= bound + 1e-10),
        "shift_star": shift_star,
    }


def run_s4(n_reps=200, seed=42):
    """Fewer reps than other scenarios because OVL estimation is expensive."""
    shift_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    results = {}

    for shift in shift_values:
        df = run_mc(one_rep, n_reps=n_reps, seed=seed, shift_star=shift)
        results[f"shift={shift}"] = df
        mean_disc = df["actual_discrepancy"].mean()
        mean_bound = df["bound"].mean()
        frac_holds = df["bound_holds"].mean()
        print(f"S4 shift={shift:.1f}: mean_disc={mean_disc:.4f}  "
              f"mean_bound={mean_bound:.4f}  "
              f"bound_holds={frac_holds:.2%}")

    return results


if __name__ == "__main__":
    run_s4(n_reps=50)
