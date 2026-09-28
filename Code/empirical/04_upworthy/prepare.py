"""
Load the Upworthy exploratory release, apply the Matias et al. (2024)
randomisation-bug filter, pair top-2 variants per test, and write:

    uw_experiments.parquet  -- one row per retained (test, variant).

Columns emitted:
    unit_id, test_id, variant_rank, W, Y, headline,
    impressions, clicks, headline_len, test_week, log_impressions, created_at

Usage:
    python3 prepare.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EMP = HERE.parent
DATA_DIR = EMP / "output" / "data"
CSV_PATH = DATA_DIR / "upworthy_exploratory.csv"

BUG_START = pd.Timestamp("2013-06-25")
BUG_END = pd.Timestamp("2014-01-10")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} missing. Download with:\n"
            f"  curl -sL -o {CSV_PATH} https://osf.io/download/3vqmp/"
        )

    df = pd.read_csv(CSV_PATH)
    n0 = len(df)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["created_at"] = df["created_at"].dt.tz_convert(None)
    df = df.dropna(subset=["headline", "impressions", "clicks",
                           "clickability_test_id", "created_at"])
    df = df[(df["impressions"] >= 100)].copy()

    buggy_mask = df["created_at"].between(BUG_START, BUG_END)
    buggy_tests = df.loc[buggy_mask, "clickability_test_id"].unique()
    df = df[~df["clickability_test_id"].isin(buggy_tests)].copy()
    n1 = len(df)

    tests_per = df.groupby("clickability_test_id").size()
    keep_tests = tests_per[tests_per >= 2].index
    df = df[df["clickability_test_id"].isin(keep_tests)].copy()

    n_variants_in_test = (df.groupby("clickability_test_id").size()
                            .rename("n_variants_in_test"))

    df = df.sort_values(["clickability_test_id", "impressions"],
                        ascending=[True, False])
    df["rank_within_test"] = df.groupby("clickability_test_id").cumcount()
    pairs = df[df["rank_within_test"] < 2].copy()

    pairs = pairs.sort_values(["clickability_test_id", "headline"])
    pairs["W"] = pairs.groupby("clickability_test_id").cumcount()  # 0 or 1

    pairs["Y"] = pairs["clicks"] / pairs["impressions"]
    pairs["headline_len"] = pairs["headline"].str.len().astype(float)
    pairs["log_impressions"] = np.log(pairs["impressions"].astype(float))
    pairs["test_id"] = pairs["clickability_test_id"].astype(str)
    pairs["unit_id"] = pairs["test_id"] + "__W" + pairs["W"].astype(str)
    pairs["test_week_num"] = pd.to_numeric(pairs["test_week"],
                                           errors="coerce").astype(float)
    pairs = pairs.merge(n_variants_in_test,
                        left_on="clickability_test_id", right_index=True)
    pairs["n_variants_in_test"] = pairs["n_variants_in_test"].astype(float)

    test_sizes = pairs.groupby("test_id").size()
    complete = test_sizes[test_sizes == 2].index
    pairs = pairs[pairs["test_id"].isin(complete)].copy()

    out = pairs[[
        "unit_id", "test_id", "rank_within_test", "W", "Y",
        "headline", "impressions", "clicks", "headline_len",
        "test_week_num", "log_impressions", "n_variants_in_test",
        "created_at",
    ]].rename(columns={"rank_within_test": "variant_rank",
                       "test_week_num": "test_week"})

    out_path = DATA_DIR / "uw_experiments.parquet"
    out.to_parquet(out_path)
    n_tests = out["test_id"].nunique()
    print(f"[uw] read {n0} rows -> {n1} after bug+impression filter "
          f"-> {len(out)} rows across {n_tests} complete A/B test pairs")
    print(f"[uw] wrote {out_path}")
    print(f"[uw] summary:")
    print(f"     date range: {out['created_at'].min()} .. {out['created_at'].max()}")
    print(f"     CTR by W:   W=0 mean {out.loc[out['W']==0, 'Y'].mean():.4f} | "
          f"W=1 mean {out.loc[out['W']==1, 'Y'].mean():.4f}")
    raw_ate = out.loc[out["W"] == 1, "Y"].mean() - out.loc[out["W"] == 0, "Y"].mean()
    print(f"     pooled ATE of W=1 vs W=0: {raw_ate:+.5f}")


if __name__ == "__main__":
    main()
