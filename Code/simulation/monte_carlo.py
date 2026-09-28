"""
Generic Monte Carlo simulation runner.
"""

import numpy as np
import pandas as pd
from typing import Callable


def run_mc(scenario_fn: Callable, n_reps: int = 1000,
           seed: int = 42, **scenario_kwargs) -> pd.DataFrame:
    """
    Run a scenario function n_reps times and collect results.

    Parameters
    ----------
    scenario_fn : callable
        Must accept (seed=int, **kwargs) and return a dict of scalar results.
    n_reps : int
        Number of Monte Carlo repetitions.
    seed : int
        Master seed; per-rep seeds are derived deterministically.
    scenario_kwargs : dict
        Passed through to scenario_fn.

    Returns
    -------
    results : pd.DataFrame with one row per repetition.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_reps)

    rows = []
    for r in range(n_reps):
        row = scenario_fn(seed=int(seeds[r]), **scenario_kwargs)
        row["rep"] = r
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, true_ate: float,
              ate_col: str = "tau_calibrated") -> dict:
    """Compute summary statistics for a set of MC results."""
    estimates = df[ate_col].values
    bias = estimates.mean() - true_ate
    rmse = np.sqrt(((estimates - true_ate) ** 2).mean())
    std = estimates.std()
    return {
        "mean_estimate": estimates.mean(),
        "bias": bias,
        "rmse": rmse,
        "std": std,
        "true_ate": true_ate,
        "n_reps": len(df),
    }
