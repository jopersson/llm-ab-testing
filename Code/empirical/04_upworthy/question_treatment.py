"""
Primary Upworthy analysis: an EXOGENOUS, LLM-independent headline treatment.

Motivation
----------
Upworthy "clickability tests" randomize users across several packages that vary
the headline and/or the image. The only randomized contrast is "which variant a
user saw"; an arbitrary label of it (e.g. alphabetical order) is a placebo. To
obtain a *real* treatment effect that is still independent of the LLM, we use an
observable, headline-encoded feature as the treatment:

    W = 1  if the variant's headline is phrased as a QUESTION (contains '?'),
    W = 0  otherwise.

We restrict to tests whose two highest-impression variants (i) have DISTINCT
headlines (so the LLM input actually differs) and (ii) are DISCORDANT on the
question feature (exactly one variant is a question). Within each such test the
user is randomized across the two variants, so W is a genuinely randomized,
LLM-independent contrast. The estimand is the average human CTR difference
between the question-framed and non-question variant.

Caveat (interpretation, not validity): the question mark is bundled with other
wording differences between the two distinct headlines, so this is "the effect
of the question-framed variant", not a ceteris-paribus question-mark effect.

What we report
--------------
For the selected tests, the within-test paired ATE (one difference per test) for
    * human Y,
    * raw LLM surrogate  bar Y* (K draws),
    * calibrated  mu_hat(bar Y*, X), via 5-fold CROSS-FITTING over tests
      (calibration fit on out-of-fold variants only -> no leakage, full power),
for OLS / RF / GBT calibrations. We also sweep K = 1..10 to show how averaging
draws affects the raw surrogate ATE and its precision (Prop. attenuation /
multidraw). Inference is clustered at the test level by construction (each test
contributes a single paired difference).

Outputs
-------
    output/tables/uw_question.json
    output/figures/uw_question_ate.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "lines.linewidth": 1.0, "lines.markersize": 3.5,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EMP = HERE.parent
for p in (str(EMP), str(EMP / "common"), str(EMP.parent / "simulation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from estimators import fit_calibration, predict_calibrated  # noqa: E402

DATA_DIR = EMP / "output" / "data"
_OUT = Path(__import__("os").environ.get("UW_OUT_DIR", str(EMP / "output")))
FIG_DIR = _OUT / "figures"
TAB_DIR = _OUT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

X_COLS = ["headline_len", "test_week", "n_variants_in_test"]
N_FOLDS = 5
SEED = 2026


def ystar_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("Ystar_")]
    cols.sort(key=lambda c: int(c.split("_")[1]))
    return cols


def paired_contrast(sub: pd.DataFrame, col: str) -> pd.Series:
    """Per-test within-test difference value(W=1) - value(W=0), indexed by test_id."""
    a = sub[sub["W"] == 1].groupby("test_id")[col].first()
    b = sub[sub["W"] == 0].groupby("test_id")[col].first()
    return (a - b).dropna()


def paired_ate(sub: pd.DataFrame, col: str) -> dict:
    """Within-test paired ATE = mean_t [ value(W=1) - value(W=0) ]."""
    d = paired_contrast(sub, col)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    return {"ate": mean, "se": se, "t": mean / se if se > 0 else float("nan"),
            "n_tests": int(n)}


def recovery_gap(sub: pd.DataFrame, pred_col: str, human: pd.Series) -> dict:
    """Paired test of whether an estimator recovers the human ATE.

    gap_t = [pred(W=1)-pred(W=0)]_t - [Y(W=1)-Y(W=0)]_t, computed test-by-test so
    the human and estimator contrasts are differenced on the SAME tests. A mean
    gap indistinguishable from zero (|t| < 2) means the estimator recovers the
    human ATE; a significant gap means residual attenuation/bias.
    """
    d = paired_contrast(sub, pred_col)
    gap = (d - human).dropna()
    n = len(gap)
    mean = float(gap.mean())
    se = float(gap.std(ddof=1) / np.sqrt(n))
    return {"gap": mean, "se": se, "t": mean / se if se > 0 else float("nan"),
            "recovers": bool(abs(mean / se) < 2) if se > 0 else False,
            "n_tests": int(n)}


def crossfit_yhat(df: pd.DataFrame, eval_tests: np.ndarray, method: str,
                  K: int | None = None) -> pd.Series:
    """5-fold cross-fitted calibrated prediction over `eval_tests`.

    For each fold, mu_hat is fit on ALL variants whose test is NOT in the fold
    (the full sample, not just the selected tests) and applied to the held-out
    fold's variants. Returns a Series of yhat indexed like df (NaN off-sample).
    """
    cols = ystar_cols(df)
    if K is not None:
        cols = cols[:K]
    ybar = df[cols].mean(axis=1).to_numpy()
    Xv = df[X_COLS].to_numpy(float)
    yhat = pd.Series(np.nan, index=df.index)

    rng = np.random.default_rng(SEED)
    tests = np.array(sorted(eval_tests))
    rng.shuffle(tests)
    folds = np.array_split(tests, N_FOLDS)
    for fold in folds:
        held = set(fold.tolist())
        tr = ~df["test_id"].isin(held)
        model = fit_calibration(ybar[tr.to_numpy()], Xv[tr.to_numpy()],
                                df.loc[tr, "Y"].to_numpy(float), method=method)
        sel = df["test_id"].isin(held)
        yhat.loc[sel] = predict_calibrated(model, ybar[sel.to_numpy()],
                                           Xv[sel.to_numpy()])
    return yhat


def main():
    df = pd.read_parquet(DATA_DIR / "uw_with_ystar.parquet")
    df = df.dropna(subset=X_COLS).copy()
    yc = ystar_cols(df)
    df = df.dropna(subset=yc, how="all").copy()
    df["ybar"] = df[yc].mean(axis=1)
    df["q"] = df["headline"].astype(str).str.contains(r"\?", regex=True).astype(int)
    # The treatment for this analysis IS the question feature, not the
    # alphabetical orientation from prepare.py. Redefine W = q so the paired
    # ATE contrasts the question-framed vs non-question variant.
    df["W"] = df["q"]

    # Select tests: distinct headlines AND discordant on the question feature.
    g = df.groupby("test_id")
    distinct = g["headline"].nunique() == 2
    discordant_q = g["q"].nunique() == 2
    complete = g.size() == 2
    keep = distinct.index[(distinct & discordant_q & complete).to_numpy()]
    sub = df[df["test_id"].isin(keep)].copy()

    # Cross-fitted calibrated predictions (full K) for each method.
    for m in ("ols", "rf", "gbt"):
        df[f"yhat_{m}"] = crossfit_yhat(df, np.asarray(keep), m, K=None)
    sub = df[df["test_id"].isin(keep)].copy()

    out: dict = {
        "treatment": "question_headline (W=1 if headline contains '?')",
        "selection": "distinct-headline AND question-discordant top-2 variant tests",
        "n_tests": int(len(keep)),
        "n_question_lower_ctr_by_design": None,
        "ate": {
            "human": paired_ate(sub, "Y"),
            "raw_Ystar": paired_ate(sub, "ybar"),
            "calib_ols": paired_ate(sub, "yhat_ols"),
            "calib_rf": paired_ate(sub, "yhat_rf"),
            "calib_gbt": paired_ate(sub, "yhat_gbt"),
        },
    }

    # Paired recovery-gap test: does each estimator recover the human ATE?
    human_d = paired_contrast(sub, "Y")
    out["recovery_gap"] = {
        "raw_Ystar": recovery_gap(sub, "ybar", human_d),
        "calib_ols": recovery_gap(sub, "yhat_ols", human_d),
        "calib_rf": recovery_gap(sub, "yhat_rf", human_d),
        "calib_gbt": recovery_gap(sub, "yhat_gbt", human_d),
    }

    # K-sweep on the raw surrogate ATE (averaging draws): show ATE and SE vs K.
    K_GRID = (1, 2, 3, 5, 10)
    ksweep = {}
    for K in K_GRID:
        cols = yc[:K]
        sub[f"ybar_K{K}"] = sub[cols].mean(axis=1)
        ksweep[K] = paired_ate(sub, f"ybar_K{K}")
    out["k_sweep_raw"] = ksweep

    # K-sweep for the calibrated estimators: does averaging draws also
    # de-attenuate the deployed estimator? Re-cross-fit mu_hat on the first K
    # draws at each K so the calibration sees exactly the K-averaged surrogate.
    ksweep_calib = {m: {} for m in ("ols", "rf", "gbt")}
    for K in K_GRID:
        for m in ("ols", "rf", "gbt"):
            df[f"yhat_{m}_K{K}"] = crossfit_yhat(df, np.asarray(keep), m, K=K)
        subK = df[df["test_id"].isin(keep)].copy()
        for m in ("ols", "rf", "gbt"):
            ksweep_calib[m][K] = paired_ate(subK, f"yhat_{m}_K{K}")
    out["k_sweep_calib"] = ksweep_calib

    # Scale / level context.
    out["scale"] = {
        "mean_Ystar": float(df["ybar"].mean()),
        "mean_Y": float(df["Y"].mean()),
        "ratio_Ystar_over_Y": float(df["ybar"].mean() / df["Y"].mean()),
    }

    out_path = TAB_DIR / "uw_question.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[uwq] wrote {out_path}")

    # --- terminal summary ---
    print(f"\n=== Question-headline treatment: {len(keep)} randomized tests ===")
    a = out["ate"]
    for lab, key in [("human", "human"), ("raw Y*", "raw_Ystar"),
                     ("calib OLS", "calib_ols"), ("calib RF", "calib_rf"),
                     ("calib GBT", "calib_gbt")]:
        r = a[key]
        print(f"  {lab:10s} ATE={r['ate']:+.5f}  SE={r['se']:.5f}  "
              f"t={r['t']:+.1f}  (n={r['n_tests']})")
    print("\n  Paired recovery gap vs human (|t|<2 => recovers):")
    for lab, key in [("raw Y*", "raw_Ystar"), ("calib OLS", "calib_ols"),
                     ("calib RF", "calib_rf"), ("calib GBT", "calib_gbt")]:
        r = out["recovery_gap"][key]
        print(f"    {lab:10s} gap={r['gap']:+.5f}  SE={r['se']:.5f}  "
              f"t={r['t']:+.1f}  -> {'RECOVERS' if r['recovers'] else 'BIASED'}")
    print("\n  K-sweep ATE (raw / OLS / RF / GBT):")
    for K in K_GRID:
        print(f"    K={K:2d}: raw={ksweep[K]['ate']:+.5f}  "
              f"ols={ksweep_calib['ols'][K]['ate']:+.5f}  "
              f"rf={ksweep_calib['rf'][K]['ate']:+.5f}  "
              f"gbt={ksweep_calib['gbt'][K]['ate']:+.5f}")

    # --- figure: ATE comparison vs human band ---
    labels = ["raw $Y^*$", "calib OLS", "calib RF", "calib GBT"]
    keys = ["raw_Ystar", "calib_ols", "calib_rf", "calib_gbt"]
    vals = [a[k]["ate"] for k in keys]
    ses = [a[k]["se"] for k in keys]
    hum = a["human"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    xs = np.arange(len(labels))
    ax.errorbar(xs, vals, yerr=[1.96 * s for s in ses], fmt="o",
                color="#3366aa", capsize=3, label="estimate (95% CI)")
    ax.axhline(hum["ate"], color="#cc3333", lw=1.4,
               label=f"human ATE = {hum['ate']:.4f}")
    ax.axhspan(hum["ate"] - 1.96 * hum["se"], hum["ate"] + 1.96 * hum["se"],
               color="#cc3333", alpha=0.12)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"ATE of question headline on CTR")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uw_question_ate.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[uwq] wrote {FIG_DIR / 'uw_question_ate.pdf'}")

    # --- figure: paired recovery gap (estimator - human), centered on zero ---
    # gap CI excluding 0 => biased; gap CI crossing 0 => recovers the human ATE.
    g = out["recovery_gap"]
    gkeys = ["raw_Ystar", "calib_ols", "calib_rf", "calib_gbt"]
    gvals = [g[k]["gap"] for k in gkeys]
    gses = [g[k]["se"] for k in gkeys]
    recovers = [g[k]["recovers"] for k in gkeys]
    colors = ["#3a9e57" if r else "#cc3333" for r in recovers]
    figg, axg = plt.subplots(figsize=(5.4, 3.6))
    ys = np.arange(len(labels))[::-1]
    for y, v, s, c in zip(ys, gvals, gses, colors):
        axg.errorbar(v, y, xerr=1.96 * s, fmt="o", color=c, capsize=3, lw=1.6)
    axg.axvline(0, color="k", lw=1.2)
    axg.set_ylim(-1.05, len(labels) - 1 + 1.25)
    axg.annotate("human ATE", xy=(0, 1), xycoords=("data", "axes fraction"),
                 xytext=(0, 2), textcoords="offset points",
                 fontsize=8, ha="center", va="bottom", color="k", clip_on=False)
    axg.set_yticks(ys)
    axg.set_yticklabels(labels)
    axg.set_xlabel(r"estimate $-$ human ATE  (per-test paired gap, 95% CI)")
    from matplotlib.lines import Line2D
    axg.legend(handles=[
        Line2D([0], [0], marker="o", color="#cc3333", lw=0, label="biased (CI excludes 0)"),
        Line2D([0], [0], marker="o", color="#3a9e57", lw=0, label="recovers (CI crosses 0)"),
    ], frameon=False, fontsize=8, loc="lower right")
    figg.tight_layout()
    figg.savefig(FIG_DIR / "uw_question_recovery_gap.pdf", bbox_inches="tight")
    plt.close(figg)
    print(f"[uwq] wrote {FIG_DIR / 'uw_question_recovery_gap.pdf'}")

    # --- figure: multi-draw de-attenuation across estimators vs K ---
    Ks = sorted(ksweep.keys())
    series = [
        (r"raw $\bar Y^*$", ksweep, "#3366aa", "o"),
        ("calib OLS", ksweep_calib["ols"], "#8c564b", "s"),
        ("calib RF", ksweep_calib["rf"], "#3a9e57", "^"),
        ("calib GBT", ksweep_calib["gbt"], "#d9882b", "D"),
    ]
    fig2, ax2 = plt.subplots(figsize=(3.3, 2.4))
    xpos = np.arange(len(Ks), dtype=float)
    offs = np.linspace(-0.20, 0.20, len(series))
    for (lab, sw, color, mk), dx in zip(series, offs):
        vals_k = [sw[K]["ate"] for K in Ks]
        se_k = [sw[K]["se"] for K in Ks]
        ax2.errorbar(xpos + dx, vals_k, yerr=[1.96 * s for s in se_k],
                     fmt=mk + "-", color=color, capsize=2, elinewidth=1.0,
                     lw=1.1, ms=3.5, label=lab)
    ax2.axhline(hum["ate"], color="#cc3333", lw=1.4,
                label=f"human ATE = {hum['ate']:.4f}")
    ax2.axhspan(hum["ate"] - 1.96 * hum["se"], hum["ate"] + 1.96 * hum["se"],
                color="#cc3333", alpha=0.12)
    ax2.set_xlabel(r"Number of LLM draws $K$")
    ax2.set_ylabel("ATE")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([str(K) for K in Ks])
    ax2.set_xlim(-0.5, len(Ks) - 0.5)
    ax2.legend(frameon=False, ncol=2, loc="upper right")
    fig2.tight_layout()
    # No bbox_inches="tight": keep the page width exactly the 3.3in column width
    # so this figure matches the other single-column Upworthy figures.
    fig2.savefig(FIG_DIR / "uw_question_ksweep.pdf")
    fig2.savefig(FIG_DIR / "uw_question_ksweep.png", dpi=180)
    plt.close(fig2)
    print(f"[uwq] wrote {FIG_DIR / 'uw_question_ksweep.pdf'}")


if __name__ == "__main__":
    main()
