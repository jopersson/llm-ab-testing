"""
Robustness of the calibrated ATE estimator to the choice of calibration model.

Replicates scenario S1 (surrogacy + comparability both hold) under three
calibration methods: OLS, random forest, and gradient-boosted trees.
Runs on both the linear and nonlinear DGPs.

Produces:
  - output/s8_calibration_robustness.pkl (raw results)
  - output/tables/calibration_robustness.tex (LaTeX booktabs table)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pickle
import numpy as np
import pandas as pd

from dgp import LinearDGP, NonlinearDGP
from estimators import fit_calibration, calibrated_ate, raw_llm_ate
from monte_carlo import run_mc, summarize


OUTDIR = Path(__file__).resolve().parent / "output"
TABDIR = OUTDIR / "tables"
OUTDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)


def one_rep_linear(seed, n=2000, sigma_eps=0.3, method="ols"):
    dgp = LinearDGP(n=n, sigma_eps=sigma_eps, sigma_eta=0.2, seed=seed)
    p0 = dgp.generate_sample(population=0)
    p1 = dgp.generate_sample(population=1)
    model = fit_calibration(p0.Ystar, p0.X, p0.Y, method=method)
    return {
        "tau_calibrated": calibrated_ate(model, p1.Ystar, p1.X, p1.W),
        "tau_raw_llm": raw_llm_ate(p1.Ystar, p1.W),
        "tau_true": dgp.true_ate(),
        "variant": "linear",
        "method": method,
    }


def one_rep_nonlinear(seed, n=2000, sigma_eps=0.2, method="ols"):
    dgp = NonlinearDGP(n=n, sigma_eps=sigma_eps, sigma_eta=0.15, seed=seed)
    p0 = dgp.generate_sample(population=0)
    p1 = dgp.generate_sample(population=1)
    model = fit_calibration(p0.Ystar, p0.X, p0.Y, method=method)
    return {
        "tau_calibrated": calibrated_ate(model, p1.Ystar, p1.X, p1.W),
        "tau_raw_llm": raw_llm_ate(p1.Ystar, p1.W),
        "tau_true": dgp.true_ate(),
        "variant": "nonlinear",
        "method": method,
    }


METHOD_LABEL = {"ols": "OLS", "rf": "Random forest", "gbt": "Gradient boosting"}


def run(n_reps=1000, seed=42):
    results = {}
    true_ate_lin = LinearDGP().true_ate()
    true_ate_nl = NonlinearDGP().true_ate()

    for method in ["ols", "rf", "gbt"]:
        print(f"Linear DGP, method = {method}")
        df = run_mc(one_rep_linear, n_reps=n_reps, seed=seed, method=method)
        results[("linear", method)] = df
        s = summarize(df, true_ate_lin)
        print(f"  bias={s['bias']:+.4f}  rmse={s['rmse']:.4f}  "
              f"mean={s['mean_estimate']:.4f}  true={true_ate_lin:.4f}")

        print(f"Nonlinear DGP, method = {method}")
        df = run_mc(one_rep_nonlinear, n_reps=n_reps, seed=seed, method=method)
        results[("nonlinear", method)] = df
        s = summarize(df, true_ate_nl)
        print(f"  bias={s['bias']:+.4f}  rmse={s['rmse']:.4f}  "
              f"mean={s['mean_estimate']:.4f}  true={true_ate_nl:.4f}")

    with open(OUTDIR / "s8_calibration_robustness.pkl", "wb") as f:
        pickle.dump(results, f)

    write_latex_table(results, true_ate_lin, true_ate_nl)
    return results


def write_latex_table(results, true_ate_lin, true_ate_nl):
    rows = []
    for variant, true_ate in [("linear", true_ate_lin),
                              ("nonlinear", true_ate_nl)]:
        for method in ["ols", "rf", "gbt"]:
            df = results[(variant, method)]
            s = summarize(df, true_ate)
            rows.append({
                "variant": variant,
                "method": METHOD_LABEL[method],
                "mean": s["mean_estimate"],
                "bias": s["bias"],
                "rmse": s["rmse"],
                "std": s["std"],
                "true_ate": true_ate,
            })
    tbl = pd.DataFrame(rows)

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"DGP & Calibration model & Mean $\hat\tau$ & Bias & RMSE & SD \\",
        r"\midrule",
    ]
    current_variant = None
    for r in tbl.itertuples():
        variant_label = ""
        if r.variant != current_variant:
            variant_label = (
                f"{r.variant.capitalize()} "
                f"(true $\\tau = {r.true_ate:.3f}$)"
            )
            current_variant = r.variant
            if lines[-1] != r"\midrule":
                lines.append(r"\midrule")
        lines.append(
            f"{variant_label} & {r.method} "
            f"& {r.mean:.4f} & {r.bias:+.4f} "
            f"& {r.rmse:.4f} & {r.std:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    (TABDIR / "calibration_robustness.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote LaTeX table to {TABDIR / 'calibration_robustness.tex'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    n_reps = 100 if args.pilot else 1000
    run(n_reps=n_reps)
