"""
Shared analysis pipeline for each empirical dataset.

Given a long-format DataFrame with columns
    unit_id, W (0/1), Y (human outcome), plus one or more covariate columns X_*,
    and a wide matrix of K LLM draws columns Ystar_0, Ystar_1, ..., Ystar_{K-1},
the pipeline produces:

    * raw-LLM ATE (mean-of-draws, K=1, K=K)
    * calibrated ATE via OLS / RF / GBT, fit on a calibration sample
    * back-test moment (Prop. 3): E[ Y - mu_hat(Y*, X) ]
    * overlap coefficient OVL per arm (Prop. 2 input)
    * sensitivity bound  (Prop. 2)  given B

We intentionally re-use the simulation code in ../simulation/ (estimators.py,
overlap.py) so the empirical story matches the theoretical one verbatim.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent.parent / "simulation"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from estimators import fit_calibration, predict_calibrated  # noqa: E402
from overlap import estimate_ovl, sensitivity_bound  # noqa: E402


# ---------------------------------------------------------------------------
# Data layout helpers
# ---------------------------------------------------------------------------

def ystar_columns(df: pd.DataFrame) -> list[str]:
    """All Ystar_k draw columns, sorted by k."""
    cols = [c for c in df.columns if c.startswith("Ystar_")]
    cols.sort(key=lambda c: int(c.split("_")[1]))
    return cols


def mean_ystar(df: pd.DataFrame, K: Optional[int] = None) -> np.ndarray:
    """Mean LLM outcome across the first K draws (default: all available)."""
    cols = ystar_columns(df)
    if K is not None:
        cols = cols[:K]
    if not cols:
        raise ValueError("No Ystar_* columns present in the dataframe.")
    return df[cols].mean(axis=1).to_numpy()


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def backtest_split(df: pd.DataFrame, calib_frac: float = 0.5,
                   seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split units into a calibration half (where Y is observed) and a held-out
    half (where we pretend Y is unseen to stress-test the calibrated ATE).
    The split is by unit_id so the same unit is never in both halves.
    """
    rng = np.random.default_rng(seed)
    uids = np.array(df["unit_id"].drop_duplicates().to_list())
    rng.shuffle(uids)
    n_cal = int(round(calib_frac * len(uids)))
    cal_ids = set(uids[:n_cal])
    hold_ids = set(uids[n_cal:])
    cal = df[df["unit_id"].isin(cal_ids)].copy()
    hold = df[df["unit_id"].isin(hold_ids)].copy()
    return cal, hold


# ---------------------------------------------------------------------------
# Estimators built on sim code
# ---------------------------------------------------------------------------

def _feature_matrix(df: pd.DataFrame, x_cols: list[str]) -> np.ndarray:
    if not x_cols:
        return np.zeros((len(df), 1))
    return df[x_cols].to_numpy().astype(float)


def _cluster_index_map(values: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (unique_clusters, {cluster_id: positional_indices_array})."""
    unique = pd.unique(values)
    idx_map = {c: np.where(values == c)[0] for c in unique}
    return unique, idx_map


def _bootstrap_indices(
    n_rows: int,
    cluster_values: Optional[np.ndarray],
    rng: np.random.Generator,
    cluster_cache: Optional[tuple[np.ndarray, dict]] = None,
) -> np.ndarray:
    """One bootstrap resample of positional indices.
    If cluster_values is None: i.i.d. row resample. Otherwise: cluster
    resample (sample unique clusters with replacement, concat their rows).
    `cluster_cache` is the precomputed `_cluster_index_map(cluster_values)`;
    pass it from outside the bootstrap loop to avoid rebuilding it on every
    iteration (otherwise the clustered bootstrap is O(n_boot * n_clusters)).
    """
    if cluster_values is None:
        return rng.choice(n_rows, size=n_rows, replace=True)
    unique, idx_map = (cluster_cache if cluster_cache is not None
                       else _cluster_index_map(cluster_values))
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([idx_map[c] for c in sampled])


@dataclass
class ATEResult:
    estimator: str
    value: float
    se: Optional[float] = None
    extra: dict = field(default_factory=dict)


def human_ate_with_se(
    df: pd.DataFrame,
    cluster_col: Optional[str] = None,
    n_boot: int = 500,
    seed: int = 2026,
) -> ATEResult:
    y = df["Y"].to_numpy(dtype=float)
    w = df["W"].to_numpy()
    tau = y[w == 1].mean() - y[w == 0].mean()
    if cluster_col is None:
        y1 = y[w == 1]; y0 = y[w == 0]
        se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    else:
        rng = np.random.default_rng(seed)
        cluster_values = df[cluster_col].to_numpy()
        cluster_cache = _cluster_index_map(cluster_values)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = _bootstrap_indices(len(df), cluster_values, rng, cluster_cache)
            yb = y[idx]; wb = w[idx]
            boots[b] = yb[wb == 1].mean() - yb[wb == 0].mean()
        se = float(boots.std(ddof=1))
    return ATEResult("human", float(tau), se)


def raw_llm_ate_with_se(
    df: pd.DataFrame,
    K: Optional[int] = None,
    cluster_col: Optional[str] = None,
    n_boot: int = 500,
    seed: int = 2026,
) -> ATEResult:
    ystar = mean_ystar(df, K=K)
    w = df["W"].to_numpy()
    tau = ystar[w == 1].mean() - ystar[w == 0].mean()
    if cluster_col is None:
        y1 = ystar[w == 1]; y0 = ystar[w == 0]
        se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    else:
        rng = np.random.default_rng(seed)
        cluster_values = df[cluster_col].to_numpy()
        cluster_cache = _cluster_index_map(cluster_values)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = _bootstrap_indices(len(df), cluster_values, rng, cluster_cache)
            yb = ystar[idx]; wb = w[idx]
            boots[b] = yb[wb == 1].mean() - yb[wb == 0].mean()
        se = float(boots.std(ddof=1))
    return ATEResult(f"raw_llm_K={len(ystar_columns(df)) if K is None else K}",
                     float(tau), se)


def calibrated_ate_with_se(
    calib_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    x_cols: list[str],
    method: str = "ols",
    K: Optional[int] = None,
    n_boot: int = 500,
    seed: int = 2026,
    cluster_col: Optional[str] = None,
) -> ATEResult:
    """
    Fit mu_hat: (Ystar_bar, X) -> Y on `calib_df`, apply to `eval_df`, report
    tau_hat = E_eval[mu_hat(Ystar_bar, X) | W=1] - E_eval[...| W=0], with
    nonparametric bootstrap SE over eval units.
    If `cluster_col` is set, the bootstrap resamples unique values of that
    column in `eval_df` (cluster bootstrap); otherwise rows are resampled
    i.i.d.
    """
    x_cal = _feature_matrix(calib_df, x_cols)
    x_ev = _feature_matrix(eval_df, x_cols)
    ystar_cal = mean_ystar(calib_df, K=K)
    ystar_ev = mean_ystar(eval_df, K=K)
    y_cal = calib_df["Y"].to_numpy(dtype=float)
    w_ev = eval_df["W"].to_numpy()

    model = fit_calibration(ystar_cal, x_cal, y_cal, method=method)
    yhat = predict_calibrated(model, ystar_ev, x_ev)
    tau = yhat[w_ev == 1].mean() - yhat[w_ev == 0].mean()

    rng = np.random.default_rng(seed)
    n_ev = len(eval_df)
    cluster_values = (eval_df[cluster_col].to_numpy()
                      if cluster_col is not None else None)
    cluster_cache = (_cluster_index_map(cluster_values)
                     if cluster_values is not None else None)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _bootstrap_indices(n_ev, cluster_values, rng, cluster_cache)
        boots[b] = (yhat[idx][w_ev[idx] == 1].mean()
                    - yhat[idx][w_ev[idx] == 0].mean())
    se = float(boots.std(ddof=1))
    extra = {"n_calib": len(calib_df), "n_eval": n_ev}
    if cluster_col is not None:
        extra["cluster_col"] = cluster_col
        extra["n_clusters"] = int(len(np.unique(cluster_values)))
    return ATEResult(f"calib_{method}", float(tau), se, extra=extra)


# ---------------------------------------------------------------------------
# Back-test moment (Prop. 3) & overlap
# ---------------------------------------------------------------------------

def backtest_moment(calib_df, hold_df, x_cols, method="ols", K=None,
                     seed=2026, cluster_col: Optional[str] = None,
                     n_boot: int = 500):
    """
    E[Y - mu_hat(Y*,X)] evaluated on the held-out sample, with bootstrap SE.
    Under correct surrogacy + overlap this should be 0.
    If `cluster_col` is set, the bootstrap clusters at that column in
    `hold_df`; otherwise rows are resampled i.i.d.
    """
    x_cal = _feature_matrix(calib_df, x_cols)
    x_ho = _feature_matrix(hold_df, x_cols)
    ystar_cal = mean_ystar(calib_df, K=K)
    ystar_ho = mean_ystar(hold_df, K=K)
    y_cal = calib_df["Y"].to_numpy(dtype=float)
    y_ho = hold_df["Y"].to_numpy(dtype=float)

    model = fit_calibration(ystar_cal, x_cal, y_cal, method=method)
    yhat_ho = predict_calibrated(model, ystar_ho, x_ho)
    resid = y_ho - yhat_ho
    m = float(resid.mean())
    se = float(resid.std(ddof=1) / np.sqrt(len(resid)))

    rng = np.random.default_rng(seed)
    cluster_values = (hold_df[cluster_col].to_numpy()
                      if cluster_col is not None else None)
    cluster_cache = (_cluster_index_map(cluster_values)
                     if cluster_values is not None else None)
    boots = np.empty(n_boot)
    n_ho = len(resid)
    for b in range(n_boot):
        idx = _bootstrap_indices(n_ho, cluster_values, rng, cluster_cache)
        boots[b] = resid[idx].mean()
    se_boot = float(boots.std(ddof=1))
    out = {"moment": m, "se_analytic": se, "se_boot": se_boot,
           "n_hold": len(hold_df)}
    if cluster_col is not None:
        out["cluster_col"] = cluster_col
        out["n_clusters"] = int(len(np.unique(cluster_values)))
    return out


def arm_overlap(df_cal: pd.DataFrame, df_hold: pd.DataFrame,
                x_cols: list[str], K: Optional[int] = None,
                n_mc: int = 20000, seed: int = 0) -> dict:
    """Estimate OVL of joint (X, Ystar_bar) between calibration and hold-out
    populations, per treatment arm. These feed the Prop. 2 bound."""
    ystar_cal = mean_ystar(df_cal, K=K)
    ystar_ho = mean_ystar(df_hold, K=K)
    x_cal = _feature_matrix(df_cal, x_cols)
    x_ho = _feature_matrix(df_hold, x_cols)

    ovls = {}
    for arm in (0, 1):
        z0 = np.column_stack([x_cal[df_cal["W"].to_numpy() == arm],
                              ystar_cal[df_cal["W"].to_numpy() == arm]])
        z1 = np.column_stack([x_ho[df_hold["W"].to_numpy() == arm],
                              ystar_ho[df_hold["W"].to_numpy() == arm]])
        if len(z0) < 5 or len(z1) < 5:
            ovls[arm] = float("nan")
            continue
        ovls[arm] = estimate_ovl(z0, z1, n_mc=n_mc, seed=seed + arm)
    return ovls


def prop2_bound(B: float, ovls: dict) -> float:
    if not np.isfinite(ovls.get(0, np.nan)) or not np.isfinite(ovls.get(1, np.nan)):
        return float("nan")
    return sensitivity_bound(B, ovls[1], ovls[0])


# ---------------------------------------------------------------------------
# One-shot driver
# ---------------------------------------------------------------------------

def run_full_pipeline(
    df: Optional[pd.DataFrame] = None,
    x_cols: Optional[list[str]] = None,
    B: float = 1.0,
    calib_frac: float = 0.5,
    seed: int = 2026,
    splits: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
    cluster_col: Optional[str] = None,
) -> dict:
    """
    One-call summary used by each dataset's analyze.py.
    `df` (when provided) must contain unit_id, W, Y, Ystar_* and the x_cols.
    `B` is a bound on |Y| for the Prop. 2 sensitivity bound; pass the sample
    range or a priori max.

    Optional arguments
    ------------------
    splits : (calib_df, hold_df), optional
        Pre-computed split. When provided, `df` may be None — the pipeline
        builds the full sample as the row-wise concatenation of the two
        splits. Useful for temporal or stratified splits.
    cluster_col : str, optional
        Column to cluster the bootstrap on (e.g. "test_id" for Upworthy).
        When None (default), rows are resampled i.i.d.
    """
    if x_cols is None:
        raise ValueError("x_cols is required.")
    if splits is not None:
        calib, hold = splits
        full_df = pd.concat([calib, hold], ignore_index=True)
    else:
        if df is None:
            raise ValueError(
                "Either `df` or `splits=(calib_df, hold_df)` must be provided.")
        calib, hold = backtest_split(df, calib_frac=calib_frac, seed=seed)
        full_df = df

    result: dict = {
        "n_total": len(full_df), "n_calib": len(calib), "n_hold": len(hold),
        "K": len(ystar_columns(full_df)),
        "cluster_col": cluster_col,
        "splits_provided": splits is not None,
    }

    # Ground-truth ATEs
    result["ate_human_full"] = human_ate_with_se(
        full_df, cluster_col=cluster_col, seed=seed).__dict__
    result["ate_human_hold"] = human_ate_with_se(
        hold, cluster_col=cluster_col, seed=seed).__dict__

    # Raw LLM, on the SAME hold-out sample as the human/calibrated rows so the
    # Table 1 comparison is apples-to-apples (SEs clustered if cluster_col set).
    result["ate_raw_K1"] = raw_llm_ate_with_se(
        hold, K=1, cluster_col=cluster_col, seed=seed).__dict__
    result["ate_raw_KK"] = raw_llm_ate_with_se(
        hold, K=None, cluster_col=cluster_col, seed=seed).__dict__

    # Calibrated on calib-half, evaluated on hold-half
    for method in ("ols", "rf", "gbt"):
        result[f"ate_calib_{method}"] = calibrated_ate_with_se(
            calib, hold, x_cols, method=method, seed=seed,
            cluster_col=cluster_col,
        ).__dict__

    # Back-test moment
    for method in ("ols", "rf", "gbt"):
        result[f"backtest_{method}"] = backtest_moment(
            calib, hold, x_cols, method=method, seed=seed,
            cluster_col=cluster_col,
        )

    # Overlap (primary spec: OLS-equivalent feature set, mean of K draws)
    ovls = arm_overlap(calib, hold, x_cols, seed=seed)
    result["ovl_by_arm"] = ovls
    result["prop2_bound"] = prop2_bound(B, ovls)
    result["B"] = B

    return result
