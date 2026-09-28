"""
Data-generating processes for the surrogacy simulation study.

All DGPs generate data structurally: X, W -> Y* -> Y, ensuring surrogacy
holds by construction (unless explicitly violated). Stochastic Y* is the
default; deterministic arises as sigma_eps=0.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SampleData:
    """Container for a single population sample."""
    W: np.ndarray
    X: np.ndarray
    Ystar: np.ndarray
    Y: Optional[np.ndarray] = None
    Ystar_draws: Optional[np.ndarray] = None


class LinearDGP:
    """
    Linear DGP where surrogacy holds by construction.

    Structural model:
        theta(W, X) = a0 + a1*W + a2*X + a3*W*X
        Y*          = theta + eps,      eps ~ N(0, sigma_eps^2)
        Y           = b0 + b1*Y* + b2*X + eta,  eta ~ N(0, sigma_eta^2)

    True ATE = b1 * E[theta(1,X) - theta(0,X)] = b1 * (a1 + a3 * E[X]).
    """

    def __init__(self, n=2000, sigma_eps=0.3, sigma_eta=0.2, seed=None,
                 a0=0.2, a1=0.6, a2=0.8, a3=0.3,
                 b0=0.5, b1=0.4, b2=0.3):
        self.n = n
        self.sigma_eps = sigma_eps
        self.sigma_eta = sigma_eta
        self.rng = np.random.default_rng(seed)
        self.a0, self.a1, self.a2, self.a3 = a0, a1, a2, a3
        self.b0, self.b1, self.b2 = b0, b1, b2

    def surrogate_signal(self, W, X):
        return self.a0 + self.a1 * W + self.a2 * X + self.a3 * W * X

    def calibration_true(self, Ystar, X):
        return self.b0 + self.b1 * Ystar + self.b2 * X

    def generate_surrogate(self, W, X, K=1):
        theta = self.surrogate_signal(W, X)
        n = len(W)
        if K == 1:
            eps = self.rng.normal(0, self.sigma_eps, size=n)
            return theta + eps, None
        eps = self.rng.normal(0, self.sigma_eps, size=(n, K))
        draws = theta[:, None] + eps
        return draws.mean(axis=1), draws

    def generate_outcome(self, Ystar, X, W=None):
        eta = self.rng.normal(0, self.sigma_eta, size=len(X))
        return self.calibration_true(Ystar, X) + eta

    def true_ate(self):
        # b1 * (a1 + a3 * E[X]), E[X] = 0.5 for Uniform(0,1)
        return self.b1 * (self.a1 + self.a3 * 0.5)

    def raw_llm_ate(self):
        # a1 + a3 * E[X]
        return self.a1 + self.a3 * 0.5

    def generate_sample(self, population, K=1, n=None):
        n = n or self.n
        X = self.rng.uniform(0, 1, size=n)
        W = self.rng.binomial(1, 0.5, size=n)
        Ystar, Ystar_draws = self.generate_surrogate(W, X, K=K)
        Y = None
        if population == 0:
            Y = self.generate_outcome(Ystar, X, W)
        return SampleData(W=W, X=X, Ystar=Ystar, Y=Y, Ystar_draws=Ystar_draws)


class NonlinearDGP:
    """
    Nonlinear DGP where surrogacy holds by construction.

    Structural model:
        theta(W, X) = sin(pi*X) + 0.8*W*exp(-X)
        Y*          = theta + eps
        Y           = 0.3*(Y*)^2 + 0.5*cos(pi*X) + eta
    """

    def __init__(self, n=2000, sigma_eps=0.2, sigma_eta=0.15, seed=None):
        self.n = n
        self.sigma_eps = sigma_eps
        self.sigma_eta = sigma_eta
        self.rng = np.random.default_rng(seed)

    def surrogate_signal(self, W, X):
        return np.sin(np.pi * X) + 0.8 * W * np.exp(-X)

    def calibration_true(self, Ystar, X):
        return 0.3 * Ystar**2 + 0.5 * np.cos(np.pi * X)

    def generate_surrogate(self, W, X, K=1):
        theta = self.surrogate_signal(W, X)
        n = len(W)
        if K == 1:
            eps = self.rng.normal(0, self.sigma_eps, size=n)
            return theta + eps, None
        eps = self.rng.normal(0, self.sigma_eps, size=(n, K))
        draws = theta[:, None] + eps
        return draws.mean(axis=1), draws

    def generate_outcome(self, Ystar, X, W=None):
        eta = self.rng.normal(0, self.sigma_eta, size=len(X))
        return self.calibration_true(Ystar, X) + eta

    def true_ate(self, n_mc=500_000):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=n_mc)
        eps = rng.normal(0, self.sigma_eps, size=n_mc)
        Ystar_1 = self.surrogate_signal(np.ones(n_mc), X) + eps
        Ystar_0 = self.surrogate_signal(np.zeros(n_mc), X) + eps
        return (self.calibration_true(Ystar_1, X).mean()
                - self.calibration_true(Ystar_0, X).mean())

    def generate_sample(self, population, K=1, n=None):
        n = n or self.n
        X = self.rng.uniform(0, 1, size=n)
        W = self.rng.binomial(1, 0.5, size=n)
        Ystar, Ystar_draws = self.generate_surrogate(W, X, K=K)
        Y = None
        if population == 0:
            Y = self.generate_outcome(Ystar, X, W)
        return SampleData(W=W, X=X, Ystar=Ystar, Y=Y, Ystar_draws=Ystar_draws)


class SurrogacyViolationDGP(LinearDGP):
    """
    Linear DGP with a direct effect of W on Y that bypasses Y*.

    Y = b0 + b1*Y* + b2*X + gamma*W + eta

    When gamma=0, reduces to LinearDGP (surrogacy holds).
    """

    def __init__(self, gamma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def generate_outcome(self, Ystar, X, W=None):
        eta = self.rng.normal(0, self.sigma_eta, size=len(X))
        base = self.calibration_true(Ystar, X) + eta
        if W is not None and self.gamma != 0:
            base = base + self.gamma * W
        return base

    def true_ate(self):
        return super().true_ate() + self.gamma


class ComparabilityViolationDGP(LinearDGP):
    """
    The calibration function differs between P=0 and P=1.

    P=0: Y = b0 + b1*Y* + b2*X + eta
    P=1: Y_true = b0 + (b1 + delta)*Y* + b2*X + eta

    The calibrated ATE uses the P=0 mapping, so it misses the
    delta contribution.
    """

    def __init__(self, delta=0.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def true_ate_p1(self):
        """ATE under the P=1 mapping (what calibration should recover)."""
        return (self.b1 + self.delta) * (self.a1 + self.a3 * 0.5)

    def generate_outcome_p1(self, Ystar, X):
        """Generate Y under the P=1 mapping (for ground-truth comparison)."""
        eta = self.rng.normal(0, self.sigma_eta, size=len(X))
        return self.b0 + (self.b1 + self.delta) * Ystar + self.b2 * X + eta


class SensitivityBoundDGP(LinearDGP):
    """
    Linear DGP with an asymmetric location shift in Y* for P=1.

    P=0: Y* = theta(W, X) + eps
    P=1: Y* = theta(W, X) + eps + shift_star * I(W=1)

    Only the W=1 arm of P=1 is shifted relative to P=0. This produces
    a *genuine* discrepancy between the P=0 and P=1 calibrated ATEs
    (the shift does not cancel across arms) while simultaneously
    degrading overlap only in the W=1 arm. The structural calibration
    mu is unchanged.

    With mu linear in Y*:
        Actual discrepancy = |b1 * shift_star|
        Bound              = 2B * (1 - OVL_1)   (plus noise on OVL_0)
    Both grow with shift_star; the bound remains valid (Proposition 2)
    and becomes informative rather than trivially vacuous.
    """

    def __init__(self, shift_star=0.0, shift_arms=(1,), **kwargs):
        super().__init__(**kwargs)
        self.shift_star = shift_star
        self.shift_arms = tuple(shift_arms)

    def generate_sample(self, population, K=1, n=None):
        sample = super().generate_sample(population, K=K, n=n)
        if population == 1 and self.shift_star != 0:
            mask = np.isin(sample.W, self.shift_arms)
            sample.Ystar = sample.Ystar + self.shift_star * mask
            if sample.Ystar_draws is not None:
                sample.Ystar_draws = (sample.Ystar_draws
                                      + self.shift_star * mask[:, None])
        return sample


class MultiDrawSurrogacyDGP:
    """
    DGP where Y depends on theta (the latent surrogate), NOT on Y*.

    theta(W, X) = a0 + a1*W + a2*X + a3*W*X
    Y*_k        = theta + eps_k
    Y           = b0 + b1*theta + b2*X + eta    <-- theta, not Y*

    Single-draw surrogacy (Y perp W | X, Y*) approximately FAILS
    when sigma_eps is large because Y* is a noisy proxy for theta.
    Multi-draw surrogacy recovers as K -> inf because Y_bar -> theta.
    """

    def __init__(self, n=2000, sigma_eps=1.5, sigma_eta=0.2, seed=None,
                 a0=0.2, a1=0.6, a2=0.8, a3=0.3,
                 b0=0.5, b1=0.4, b2=0.3):
        self.n = n
        self.sigma_eps = sigma_eps
        self.sigma_eta = sigma_eta
        self.rng = np.random.default_rng(seed)
        self.a0, self.a1, self.a2, self.a3 = a0, a1, a2, a3
        self.b0, self.b1, self.b2 = b0, b1, b2

    def surrogate_signal(self, W, X):
        return self.a0 + self.a1 * W + self.a2 * X + self.a3 * W * X

    def generate_surrogate(self, W, X, K=1):
        theta = self.surrogate_signal(W, X)
        n = len(W)
        if K == 1:
            eps = self.rng.normal(0, self.sigma_eps, size=n)
            return theta + eps, None
        eps = self.rng.normal(0, self.sigma_eps, size=(n, K))
        draws = theta[:, None] + eps
        return draws.mean(axis=1), draws

    def generate_outcome(self, Ystar, X, W):
        """Y depends on theta(W, X), not on Y*."""
        theta = self.surrogate_signal(W, X)
        eta = self.rng.normal(0, self.sigma_eta, size=len(X))
        return self.b0 + self.b1 * theta + self.b2 * X + eta

    def true_ate(self):
        return self.b1 * (self.a1 + self.a3 * 0.5)

    def generate_sample(self, population, K=1, n=None):
        n = n or self.n
        X = self.rng.uniform(0, 1, size=n)
        W = self.rng.binomial(1, 0.5, size=n)
        Ystar, Ystar_draws = self.generate_surrogate(W, X, K=K)
        Y = None
        if population == 0:
            Y = self.generate_outcome(Ystar, X, W)
        return SampleData(W=W, X=X, Ystar=Ystar, Y=Y, Ystar_draws=Ystar_draws)


class BackTestingDGP:
    """
    Multi-treatment DGP: W in {0, 1, 2}.

    Historical treatments (W=0, W=1): surrogacy holds.
    Novel treatment (W=2): surrogacy fails due to a direct effect.

    Y*(w) = c0 + c2*X + treatment_shift(w) + eps
    Y     = b0 + b1*Y* + b2*X + gamma_novel * I(W=2) + eta
    """

    def __init__(self, n=2000, sigma_eps=0.3, sigma_eta=0.2,
                 gamma_novel=0.3, seed=None):
        self.n = n
        self.sigma_eps = sigma_eps
        self.sigma_eta = sigma_eta
        self.gamma_novel = gamma_novel
        self.rng = np.random.default_rng(seed)

    def surrogate_signal(self, W, X):
        theta = 1.0 + 0.8 * X
        theta = np.where(W == 1, theta + 0.6 + 0.3 * X, theta)
        theta = np.where(W == 2, theta + 0.8 + 0.5 * X, theta)
        return theta

    def generate_surrogate(self, W, X, K=1):
        theta = self.surrogate_signal(W, X)
        n = len(W)
        if K == 1:
            eps = self.rng.normal(0, self.sigma_eps, size=n)
            return theta + eps, None
        eps = self.rng.normal(0, self.sigma_eps, size=(n, K))
        draws = theta[:, None] + eps
        return draws.mean(axis=1), draws

    def generate_outcome(self, Ystar, X, W):
        eta = self.rng.normal(0, self.sigma_eta, size=len(X))
        Y = 0.5 + 0.4 * Ystar + 0.3 * X + eta
        Y = Y + self.gamma_novel * (W == 2).astype(float)
        return Y

    def true_ate_historical(self):
        """ATE for W=1 vs W=0 (captured by surrogate)."""
        return 0.4 * (0.6 + 0.3 * 0.5)

    def true_ate_novel(self):
        """ATE for W=2 vs W=0 (includes direct effect)."""
        return 0.4 * (0.8 + 0.5 * 0.5) + self.gamma_novel

    def generate_sample(self, population, treatment_probs=None, K=1, n=None):
        n = n or self.n
        X = self.rng.uniform(0, 1, size=n)
        if treatment_probs is None:
            treatment_probs = [1/3, 1/3, 1/3]
        W = self.rng.choice([0, 1, 2], size=n, p=treatment_probs)
        Ystar, Ystar_draws = self.generate_surrogate(W, X, K=K)
        Y = None
        if population == 0:
            Y = self.generate_outcome(Ystar, X, W)
        return SampleData(W=W, X=X, Ystar=Ystar, Y=Y, Ystar_draws=Ystar_draws)
