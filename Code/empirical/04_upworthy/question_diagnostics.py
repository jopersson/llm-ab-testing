"""
Diagnostics on the SAME sample as the headline question-treatment result.

Motivation
----------
The headline Upworthy result (`question_treatment.py`) is the within-test
question-vs-non-question ATE on the 417 distinct-headline, question-discordant
tests. The original `analyze.py` diagnostics (back-test moment, overlap / Prop.2
bound, positive control) were instead computed on the full 3,603-test archive
with the placebo lexicographic-orientation W. That diagnoses the calibration
machinery in general, not the 417-test estimand.

This script re-runs the three sample-dependent diagnostics on the 417-test
question subsample, using an early/late temporal split WITHIN those tests, so the
diagnosed object matches the headline object. The memorization probe is
sample-independent and is left to `contamination_probe.py`.

Outputs
-------
    output/tables/uw_question_diagnostics.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EMP = HERE.parent
sys.path.insert(0, str(EMP))

from common.pipeline import (  # noqa: E402
    run_full_pipeline, ystar_columns, calibrated_ate_with_se,
)

DATA_DIR = EMP / "output" / "data"
_OUT = Path(__import__("os").environ.get("UW_OUT_DIR", str(EMP / "output")))
TAB_DIR = _OUT / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

X_COLS_BASE = ["headline_len", "test_week", "n_variants_in_test"]
K_DRAWS = 10
N_BOOT = 500
SEED = 2026


def select_question_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Replicate question_treatment.py's selection: distinct-headline AND
    question-discordant AND complete top-2 variant tests; set W = question."""
    df = df.copy()
    df["q"] = df["headline"].astype(str).str.contains(r"\?", regex=True).astype(int)
    df["W"] = df["q"]
    g = df.groupby("test_id")
    distinct = g["headline"].nunique() == 2
    discordant_q = g["q"].nunique() == 2
    complete = g.size() == 2
    keep = distinct.index[(distinct & discordant_q & complete).to_numpy()]
    return df[df["test_id"].isin(keep)].copy(), np.asarray(keep)


def temporal_split(df: pd.DataFrame, frac_train: float = 0.8
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Early (P=0) / late (P=1) split by test creation time, as in analyze.py."""
    test_time = df.groupby("test_id")["created_at"].min().sort_values()
    n_train = int(len(test_time) * frac_train)
    train_tests = set(test_time.index[:n_train])
    p0 = df[df["test_id"].isin(train_tests)].copy()
    p1 = df[~df["test_id"].isin(train_tests)].copy()
    return p0, p1


def inject_signal_and_surrogate(df, tau_dagger, sigma_nu, noise):
    out = df.copy()
    y = out["Y"].to_numpy(dtype=float)
    w = out["W"].to_numpy(dtype=float)
    y_tilde = y + tau_dagger * w
    out["Y"] = y_tilde
    for k in range(K_DRAWS):
        out[f"Ystar_{k}"] = y_tilde + sigma_nu * noise[:, k]
    return out


def reliability_ratio(sigma_y, sigma_nu, K):
    var_signal = sigma_y ** 2
    return var_signal / (var_signal + (sigma_nu ** 2) / K)


def main():
    df = pd.read_parquet(DATA_DIR / "uw_with_ystar.parquet")
    df = df.dropna(subset=["test_week"]).copy()
    df = df.dropna(subset=ystar_columns(df), how="all").copy()
    df = df.dropna(subset=X_COLS_BASE).copy()

    sub, keep = select_question_tests(df)
    print(f"[qdiag] question subsample: {len(sub)} variants across "
          f"{sub['test_id'].nunique()} tests")

    p0, p1 = temporal_split(sub)
    print(f"[qdiag] P=0 (early): {len(p0)} variants / {p0['test_id'].nunique()} tests; "
          f"P=1 (late): {len(p1)} variants / {p1['test_id'].nunique()} tests")

    ystar_cols = ystar_columns(sub)
    pipe_cols = ["unit_id", "test_id", "W", "Y"] + X_COLS_BASE + ystar_cols
    p0_pipe = p0[pipe_cols].copy()
    p1_pipe = p1[pipe_cols].copy()

    # --- (1) back-test moment + (2) overlap / Prop.2 bound ---
    pipe = run_full_pipeline(
        x_cols=X_COLS_BASE, B=1.0,
        splits=(p0_pipe, p1_pipe),
        cluster_col="test_id",
    )

    ovls = pipe["ovl_by_arm"]
    tv = {arm: (1.0 - ovls[arm]) if np.isfinite(ovls.get(arm, np.nan)) else float("nan")
          for arm in (0, 1)}

    out = {
        "sample": "417-test question subsample, temporal 80/20 split",
        "n_p0": len(p0), "n_p1": len(p1),
        "n_tests_p0": int(p0["test_id"].nunique()),
        "n_tests_p1": int(p1["test_id"].nunique()),
        "backtest": {m: pipe[f"backtest_{m}"] for m in ("ols", "rf", "gbt")},
        "ovl_by_arm": ovls,
        "tv_by_arm": tv,
        "prop2_bound": pipe["prop2_bound"],
        "B": pipe["B"],
        "ate_human_hold": pipe["ate_human_hold"],
        "ate_calib": {m: pipe[f"ate_calib_{m}"] for m in ("ols", "rf", "gbt")},
    }

    # --- (3) positive control on the question subsample ---
    sigma_y = float(sub["Y"].std(ddof=1))
    p0_inj_base = p0[pipe_cols].copy().reset_index(drop=True)
    p1_inj_base = p1[pipe_cols].copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    noise_p0 = rng.standard_normal((len(p0_inj_base), K_DRAWS))
    noise_p1 = rng.standard_normal((len(p1_inj_base), K_DRAWS))

    tau_grid = [0.0, 0.001, 0.005, 0.01, 0.02]
    sigma_grid = [sigma_y, 3 * sigma_y, 10 * sigma_y]
    methods = ["ols", "rf", "gbt"]

    cells = []
    t0 = time.time()
    for sigma_nu in sigma_grid:
        for tau_dagger in tau_grid:
            p0_inj = inject_signal_and_surrogate(p0_inj_base, tau_dagger, sigma_nu, noise_p0)
            p1_inj = inject_signal_and_surrogate(p1_inj_base, tau_dagger, sigma_nu, noise_p1)
            for method in methods:
                res = calibrated_ate_with_se(
                    p0_inj, p1_inj, X_COLS_BASE, method=method, K=None,
                    n_boot=N_BOOT, seed=SEED, cluster_col="test_id",
                )
                cells.append({
                    "tau_dagger": tau_dagger,
                    "sigma_nu": sigma_nu,
                    "sigma_nu_over_sigma_y": sigma_nu / sigma_y,
                    "reliability_K": reliability_ratio(sigma_y, sigma_nu, K_DRAWS),
                    "method": method,
                    "tau_hat": res.value, "se": res.se,
                    "bias": res.value - tau_dagger,
                })
    pc_meta = {
        "sigma_y": sigma_y,
        "K_draws": K_DRAWS,
        "n_boot": N_BOOT,
        "seed": SEED,
        "n_p0": int(len(p0)), "n_p1": int(len(p1)),
        "n_tests_p0": int(p0["test_id"].nunique()),
        "n_tests_p1": int(p1["test_id"].nunique()),
        "tau_grid": tau_grid,
        "sigma_grid": sigma_grid,
        "sigma_grid_in_sigma_y_units": [round(s / sigma_y, 4) for s in sigma_grid],
        "reliability_at_K": [reliability_ratio(sigma_y, s, K_DRAWS) for s in sigma_grid],
        "x_cols": X_COLS_BASE,
        "cluster_col": "test_id",
        "sample": "417-test question subsample, temporal 80/20 split",
    }
    out["positive_control"] = {"meta": pc_meta, "cells": cells}
    print(f"[qdiag] positive control: {len(cells)} cells in {time.time()-t0:.1f}s")

    out_path = TAB_DIR / "uw_question_diagnostics.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[qdiag] wrote {out_path}")

    # Emit the positive control in the schema replot_paper_figs.fig_signal_recovery
    # expects (meta + cells keyed by absolute sigma_nu), so the styled figure
    # regenerates from the 417-test sample.
    sig_path = TAB_DIR / "uw_synthetic_signal.json"
    sig_path.write_text(json.dumps({"meta": pc_meta, "cells": cells},
                                   indent=2, default=float))
    print(f"[qdiag] wrote {sig_path}")

    # --- terminal summary ---
    print("\n=== Back-test moment (late hold-out, cluster boot at test_id) ===")
    for m in ("ols", "rf", "gbt"):
        b = out["backtest"][m]
        z = b["moment"] / b["se_boot"] if b["se_boot"] else float("nan")
        print(f"  {m:4s}: moment={b['moment']:+.3e}  se_boot={b['se_boot']:.3e}  "
              f"z={z:+.1f}  (n_hold={b['n_hold']})")
    print("\n=== Overlap / Prop.2 bound ===")
    print(f"  OVL_0={ovls.get(0):.3f}  OVL_1={ovls.get(1):.3f}  "
          f"TV_0={tv[0]:.3f}  TV_1={tv[1]:.3f}")
    print(f"  bound 2B(TV0+TV1) = {out['prop2_bound']:.3f}  (B={out['B']})")
    print(f"  human hold ATE = {out['ate_human_hold']['value']:+.5f} "
          f"(se {out['ate_human_hold']['se']:.5f})")
    for m in ("ols", "rf", "gbt"):
        r = out["ate_calib"][m]
        err = abs(r["value"] - out["ate_human_hold"]["value"])
        print(f"  calib {m:4s} hold ATE = {r['value']:+.5f}  |err vs human|={err:.3e}")
    print("\n=== Positive control (selected cells) ===")
    for c in cells:
        if c["method"] == "ols" and c["tau_dagger"] in (0.0, 0.01):
            print(f"  ols tau_dag={c['tau_dagger']:.3f} "
                  f"R_K={c['reliability_K']:.2f}: tau_hat={c['tau_hat']:+.5f} "
                  f"(recover {c['tau_hat']/c['tau_dagger']:.2f})"
                  if c["tau_dagger"] else
                  f"  ols tau_dag=0.000 R_K={c['reliability_K']:.2f}: "
                  f"tau_hat={c['tau_hat']:+.2e}")


if __name__ == "__main__":
    main()
