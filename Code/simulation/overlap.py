"""
Overlap coefficient estimation for the sensitivity bound (Proposition 2).
"""

import numpy as np
from scipy.stats import gaussian_kde


def estimate_ovl(Z_p0, Z_p1, n_mc=50_000, seed=0):
    """
    Estimate the overlap coefficient OVL = int min(q0, q1) dz
    using kernel density estimation and Monte Carlo integration.

    Parameters
    ----------
    Z_p0 : array of shape (n0, d)
        Joint (X, Y*) for P=0 within a treatment arm.
    Z_p1 : array of shape (n1, d)
        Joint (X, Y*) for P=1 within a treatment arm.
    n_mc : int
        Number of MC draws for numerical integration.

    Returns
    -------
    ovl : float in [0, 1]
    """
    rng = np.random.default_rng(seed)

    kde0 = gaussian_kde(Z_p0.T)
    kde1 = gaussian_kde(Z_p1.T)

    # Draw evaluation points from a mixture of both distributions
    n_half = n_mc // 2
    idx0 = rng.choice(len(Z_p0), size=n_half, replace=True)
    idx1 = rng.choice(len(Z_p1), size=n_mc - n_half, replace=True)
    Z_eval = np.vstack([Z_p0[idx0], Z_p1[idx1]])

    q0 = kde0(Z_eval.T)
    q1 = kde1(Z_eval.T)
    q_mix = 0.5 * (q0 + q1)

    # importance-weighted estimate: OVL = E_{q_mix}[min(q0, q1) / q_mix]
    ratios = np.minimum(q0, q1) / np.maximum(q_mix, 1e-300)
    ovl = float(np.clip(ratios.mean(), 0.0, 1.0))
    return ovl


def sensitivity_bound(B, ovl_1, ovl_0):
    """
    Compute the sensitivity bound: 2B * [(1 - OVL_1) + (1 - OVL_0)].
    """
    return 2.0 * B * ((1.0 - ovl_1) + (1.0 - ovl_0))
