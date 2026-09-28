"""
C1.§6.2 / A4 — Quantitative bias-slope analysis for failure-mode simulations.

For both surrogacy violations (S2, varying gamma) and comparability violations
(S3, varying delta), the theory predicts an exact linear relationship between
the violation magnitude and the bias of the calibrated ATE.

Comparability (delta).  When the calibration mapping in P=1 is
    Y_true_p1 = b0 + (b1 + delta)*Y* + b2*X + eta,
but mu_hat is fit on P=0 (using b1), the calibrated ATE on P=1 misses the
delta*Y* contribution exactly:
    bias = -(a1 + 0.5*a3) * delta = -0.75 * delta,
giving slope = -0.75, intercept = 0.

Surrogacy (gamma).  When Y on P=0 is
    Y = b0 + b1*Y* + b2*X + gamma*W + eta,
and mu_hat is fit by OLS of Y on (Y*, X) (omitting W), the OLS coefficient
on Y* (after partialling X linearly) absorbs part of the omitted gamma*W
through Y*'s correlation with W.  In population:
    beta_OLS = b1 + gamma * Cov(M_X W, M_X Y*) / Var(M_X Y*),
where M_X denotes residualisation against (1, X).  For W ~ Bern(0.5),
X ~ U(0,1), Y* = a0 + a1*W + a2*X + a3*W*X + eps:
    Var(M_X Y*) = (1/4)*(a1^2 + a1*a3 + a3^2/3) + sigma_eps^2
    Cov(M_X W, M_X Y*) = (1/4)*(a1 + 0.5*a3)
The calibrated ATE on P=1 is then beta_OLS * (a1 + 0.5*a3), and
    bias = gamma * [ Cov(M_X W, M_X Y*) / Var(M_X Y*) * (a1 + 0.5*a3) - 1 ].
This reduces to the textbook one-for-one (slope = -1) only when sigma_eps = 0
and there is no measurement-error attenuation.  With sigma_eps > 0 the slope is
attenuated toward zero in absolute value: surrogacy violations partly survive
into the calibrated estimator through the omitted-variable channel, but only
attenuated by the reliability of Y* as a proxy for W.

This script:
1. Loads the existing s2.pkl / s3.pkl pickles produced by `run_all.py`.
2. Computes mean bias at each grid point and the across-replication SE.
3. OLS-regresses mean bias on the violation magnitude (gamma or delta),
   weighting each point by its precision.
4. Compares the empirical slope to (a) the textbook prediction (slope -1 for
   gamma, -0.75 for delta) and (b) the OLS-attenuation-corrected prediction
   for gamma.

Output (printed and JSON-saved): all three slopes (textbook, attenuation-aware
theory, empirical), t-test against the attenuation-aware theory, R^2 of the
linear fit. Saved to `Code/simulation/output/c1_slope_results.json` for later
citation in the paper.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _load(name: str):
    with open(OUTDIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def _grid_summary(results: dict, param_name: str, true_col: str):
    """For each grid value, return (param, mean_bias, se_bias, n_reps)."""
    rows = []
    for _, df in sorted(results.items(),
                        key=lambda x: float(x[1][param_name].iloc[0])):
        x = float(df[param_name].iloc[0])
        true = float(df[true_col].iloc[0])
        bias = df["tau_calibrated"].values - true
        rows.append({
            "x": x,
            "mean_bias": float(np.mean(bias)),
            "se_mean_bias": float(np.std(bias, ddof=1) / np.sqrt(len(bias))),
            "n_reps": int(len(bias)),
        })
    return rows


def _wls_fit(x, y, se_y):
    """Weighted OLS of y on x (with intercept) using inverse-variance weights.

    Returns (slope, slope_se, intercept, intercept_se, r2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = 1.0 / np.asarray(se_y, dtype=float) ** 2
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    coef = np.linalg.solve(XtWX, XtWy)
    intercept, slope = coef
    yhat = X @ coef
    resid = y - yhat
    dof = len(x) - 2
    if dof > 0:
        sigma2_hat = (resid @ (W @ resid)) / dof
    else:
        sigma2_hat = 0.0
    cov = sigma2_hat * np.linalg.inv(XtWX)
    intercept_se = float(np.sqrt(max(cov[0, 0], 0.0)))
    slope_se = float(np.sqrt(max(cov[1, 1], 0.0)))
    ss_res = float(np.sum(w * resid ** 2))
    ybar_w = float(np.sum(w * y) / np.sum(w))
    ss_tot = float(np.sum(w * (y - ybar_w) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), slope_se, float(intercept), intercept_se, float(r2)


def gamma_attenuated_slope(a1=0.6, a3=0.3, sigma_eps=0.3):
    """Closed-form attenuation-aware theoretical slope of bias on gamma
    under the LinearDGP and OLS calibration of Y on (Y*, X).

    See module docstring for derivation. Returns the slope; intercept is 0.
    """
    var_MY = 0.25 * (a1**2 + a1 * a3 + a3**2 / 3.0) + sigma_eps**2
    cov_MW_MY = 0.25 * (a1 + 0.5 * a3)
    R_W = cov_MW_MY / var_MY
    return R_W * (a1 + 0.5 * a3) - 1.0


def delta_theory_slope(a1=0.6, a3=0.3):
    """Closed-form theoretical slope of bias on delta (no attenuation issue,
    since the violation enters Y on P=1 only and not the calibration fit).
    """
    return -(a1 + 0.5 * a3)


def main(verbose=True):
    s2 = _load("s2")
    s3 = _load("s3")

    s2_rows = _grid_summary(s2, "gamma", "tau_true")
    s3_rows = _grid_summary(s3, "delta", "tau_true_p1_analytical")

    g = np.array([r["x"] for r in s2_rows])
    g_bias = np.array([r["mean_bias"] for r in s2_rows])
    g_se = np.array([r["se_mean_bias"] for r in s2_rows])
    g_slope, g_slope_se, g_int, g_int_se, g_r2 = _wls_fit(g, g_bias, g_se)
    g_theory_slope_textbook = -1.0
    g_theory_slope_atten = gamma_attenuated_slope()

    d = np.array([r["x"] for r in s3_rows])
    d_bias = np.array([r["mean_bias"] for r in s3_rows])
    d_se = np.array([r["se_mean_bias"] for r in s3_rows])
    d_slope, d_slope_se, d_int, d_int_se, d_r2 = _wls_fit(d, d_bias, d_se)
    d_theory_slope = delta_theory_slope()

    out = {
        "surrogacy_violation": {
            "param": "gamma",
            "grid_rows": s2_rows,
            "fit": {
                "slope": g_slope,
                "slope_se": g_slope_se,
                "intercept": g_int,
                "intercept_se": g_int_se,
                "r2": g_r2,
            },
            "theory_slope_textbook": g_theory_slope_textbook,
            "theory_slope_attenuation": g_theory_slope_atten,
            "theory_intercept": 0.0,
            "slope_t_vs_textbook": (g_slope - g_theory_slope_textbook) / g_slope_se if g_slope_se > 0 else float("nan"),
            "slope_t_vs_attenuation": (g_slope - g_theory_slope_atten) / g_slope_se if g_slope_se > 0 else float("nan"),
        },
        "comparability_violation": {
            "param": "delta",
            "grid_rows": s3_rows,
            "fit": {
                "slope": d_slope,
                "slope_se": d_slope_se,
                "intercept": d_int,
                "intercept_se": d_int_se,
                "r2": d_r2,
            },
            "theory_slope": d_theory_slope,
            "theory_intercept": 0.0,
            "slope_t_vs_theory": (d_slope - d_theory_slope) / d_slope_se if d_slope_se > 0 else float("nan"),
        },
    }

    out_path = OUTDIR / "c1_slope_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print("=" * 72)
        print("C1.§6.2 / A4 — bias-on-violation slope regressions")
        print("=" * 72)

        print("\nSurrogacy violation (S2): bias_calibrated_ATE on gamma")
        print("-" * 60)
        print(f"  Textbook prediction (no attenuation): slope = {g_theory_slope_textbook:+.4f}")
        print(f"  Attenuation-aware prediction:         slope = {g_theory_slope_atten:+.4f}")
        print(f"  Empirical:                            slope = {g_slope:+.4f} (SE {g_slope_se:.4f})")
        print(f"  Empirical intercept: {g_int:+.5f} (SE {g_int_se:.5f})")
        print(f"  R^2 of linear fit:   {g_r2:.4f}")
        print(f"  t(empirical vs textbook):   {(g_slope - g_theory_slope_textbook) / g_slope_se:+.2f}")
        print(f"  t(empirical vs attenuation): {(g_slope - g_theory_slope_atten) / g_slope_se:+.2f}")
        print()
        print("  Grid:")
        for r in s2_rows:
            print(f"    gamma={r['x']:.2f}: bias={r['mean_bias']:+.5f} "
                  f"(SE {r['se_mean_bias']:.5f}, n={r['n_reps']})")

        print("\nComparability violation (S3): bias_calibrated_ATE on delta")
        print("-" * 60)
        print(f"  Theory:    slope = {d_theory_slope:+.4f} (= -(a1 + 0.5*a3)), "
              f"intercept = 0")
        print(f"  Empirical: slope = {d_slope:+.4f} (SE {d_slope_se:.4f}), "
              f"intercept = {d_int:+.5f} (SE {d_int_se:.5f})")
        print(f"  R^2 of linear fit: {d_r2:.4f}")
        print(f"  t(slope vs theory): {(d_slope - d_theory_slope) / d_slope_se:+.2f}")
        print()
        print("  Grid:")
        for r in s3_rows:
            print(f"    delta={r['x']:.2f}: bias={r['mean_bias']:+.5f} "
                  f"(SE {r['se_mean_bias']:.5f}, n={r['n_reps']})")

        print(f"\nSaved: {out_path}")

    return out


if __name__ == "__main__":
    main()
