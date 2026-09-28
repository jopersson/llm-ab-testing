"""
Publication-quality figures for the surrogacy simulation study.

Figure 1: Identification works (S1)
Figure 2: Failure modes (S2 + S3)
Figure 3: Multi-draw surrogacy relaxation (S6)
Figure 4: Estimation quality vs K (S7)
Figure 5: Sensitivity bound (S4)

All figures share the editorial style defined in `Code/figstyle.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figstyle import apply_style, COLORS, SIZES, style_axes, panel_label  # noqa: E402

apply_style()

FIGDIR = HERE / "output" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png")
    plt.close(fig)
    print(f"Saved {FIGDIR / f'{name}.pdf'}")


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    """Monte Carlo mean and SE of the mean of `x`."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def _rmse_se(x: np.ndarray, truth: float) -> tuple[float, float]:
    """RMSE and its MC SE via delta method.

    RMSE = sqrt(mean(sq)) where sq = (x - truth)^2.  Var(MSE) = Var(sq) / N.
    By the delta method, SE(RMSE) = SE(MSE) / (2 * RMSE).
    """
    sq = (np.asarray(x, dtype=float) - truth) ** 2
    mse = sq.mean()
    rmse = float(np.sqrt(mse))
    se_mse = float(sq.std(ddof=1) / np.sqrt(len(sq)))
    se_rmse = se_mse / (2 * rmse) if rmse > 0 else 0.0
    return rmse, se_rmse


def _identification_panel(ax, df, title: str, *, xlabel: bool = True,
                          legend_loc: str = "upper right",
                          alpha_raw: float = 0.55,
                          alpha_cal: float = 0.85,
                          bar_lw: float = 0.4,
                          edgecolor: str = "white") -> None:
    """One calibrated-vs-raw ATE density panel."""
    true_ate = float(df["tau_true"].iloc[0])
    ax.hist(df["tau_raw_llm"], bins=40, density=True,
            color=COLORS["accent"], alpha=alpha_raw,
            edgecolor=edgecolor, linewidth=bar_lw,
            label="Raw LLM ATE")
    ax.hist(df["tau_calibrated"], bins=40, density=True,
            color=COLORS["primary"], alpha=alpha_cal,
            edgecolor=edgecolor, linewidth=bar_lw,
            label="Calibrated ATE")
    ax.axvline(true_ate, color=COLORS["spine"],
               linestyle="--", linewidth=1.0,
               label=fr"True $\tau = {true_ate:.2f}$")
    if xlabel:
        ax.set_xlabel("ATE estimate")
    ax.set_ylabel("Density")
    ax.set_title(title)
    if legend_loc == "upper center":
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.85)
        ax.legend(loc="upper center", ncol=3, fontsize=6.0,
                  handlelength=1.15, columnspacing=0.65,
                  handletextpad=0.3)
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.62)
        ax.legend(loc=legend_loc)
    style_axes(ax)


def figure1_identification(results_s1) -> None:
    """Sampling distribution of calibrated vs raw LLM ATE under Assumptions 1--2."""
    fig, axes = plt.subplots(1, 2, figsize=SIZES["single_col_2panel"])

    cases = [
        ("linear_stochastic", "Linear DGP"),
        ("nonlinear_stochastic", "Nonlinear DGP"),
    ]

    for ax, (key, label) in zip(axes, cases):
        _identification_panel(ax, results_s1[key], label,
                              alpha_raw=0.90, alpha_cal=0.95,
                              bar_lw=0.0, edgecolor="none")

    fig.tight_layout()
    _save(fig, "fig1_identification")


def figure1_identification_stacked(results_s1) -> None:
    """Vertical stack of the two DGP densities for the AIMS two-panel figure.

    Authored at roughly the on-page width of a 0.48\\linewidth workshop
    subfigure, with aspect close to ``wide_half`` (Figure 1 panel b) so the
    stacked densities match that panel's height.
    """
    fig, axes = plt.subplots(2, 1, figsize=(3.50, 2.70))

    cases = [
        ("linear_stochastic", "Linear DGP"),
        ("nonlinear_stochastic", "Nonlinear DGP"),
    ]

    for i, (ax, (key, label)) in enumerate(zip(axes, cases)):
        _identification_panel(ax, results_s1[key], label, xlabel=(i == 1),
                              legend_loc="upper center",
                              alpha_raw=0.90, alpha_cal=0.95,
                              bar_lw=0.0, edgecolor="none")

    fig.tight_layout(h_pad=0.35)
    _save(fig, "fig1_identification_stacked")


def figure2_failure_modes(results_s2, results_s3) -> None:
    r"""Bias of calibrated ATE as each identifying assumption is violated.

    Overlays the closed-form theoretical bias slope on each panel.

    Panel (a) -- Surrogacy violation: with the LinearDGP and OLS calibration
    of $Y$ on $(Y^*, X)$ omitting $W$, the omitted-variable channel partly
    survives into $\hat\mu$ via $Y^*$'s correlation with $W$, and the
    measurement error in $Y^*$ attenuates the effective gamma. The
    attenuation-aware slope is $R_W (a_1 + 0.5 a_3) - 1$ where
    $R_W = \mathrm{Cov}(M_X W, M_X Y^*) / \mathrm{Var}(M_X Y^*)$. For the
    paper's parameters this is approximately -0.395, not -1. The textbook
    one-for-one prediction (slope = -1) holds only in the deterministic
    limit sigma_eps -> 0.

    Panel (b) -- Comparability violation: bias is exactly
    -(a_1 + 0.5 a_3) * delta = -0.75 * delta (no attenuation issue, since
    delta enters Y on P=1 only and the calibration fit is unaffected).
    """
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.05))

    def _collect(results, param_name, true_col):
        xs, biases, ses = [], [], []
        for _, df in sorted(results.items(),
                            key=lambda x: float(x[1][param_name].iloc[0])):
            xs.append(float(df[param_name].iloc[0]))
            true = float(df[true_col].iloc[0])
            mu = float(df["tau_calibrated"].mean())
            biases.append(mu - true)
            ses.append(float(df["tau_calibrated"].std() / np.sqrt(len(df))))
        return np.array(xs), np.array(biases), np.array(ses)

    a1, a3, sigma_eps = 0.6, 0.3, 0.3
    var_MY = 0.25 * (a1**2 + a1 * a3 + a3**2 / 3.0) + sigma_eps**2
    cov_MW_MY = 0.25 * (a1 + 0.5 * a3)
    R_W = cov_MW_MY / var_MY
    gamma_theory_slope = R_W * (a1 + 0.5 * a3) - 1.0
    delta_theory_slope = -(a1 + 0.5 * a3)

    g, b, se = _collect(results_s2, "gamma", "tau_true")
    print(f"  [fig2a] max MC SE on bias (gamma) = {se.max():.2e}")
    ax = axes[0]
    g_theory_grid = np.linspace(g.min(), g.max(), 200)
    ax.plot(g_theory_grid, gamma_theory_slope * g_theory_grid, "--",
            color=COLORS["spine"], lw=1.3, zorder=4, dashes=(4, 2.4),
            label=fr"Predicted slope $\approx {gamma_theory_slope:.3f}$")
    ax.errorbar(g, b, yerr=se, fmt="o-", color=COLORS["accent"],
                markersize=4.5, markeredgewidth=1.2, zorder=2,
                capsize=2.0, elinewidth=0.9,
                label="Empirical bias $\\pm$ MC SE")
    ax.axhline(0, color=COLORS["muted"], lw=0.7, linestyle=":")
    ax.set_xlabel(r"$\gamma$ (direct effect of $W$ on $Y$)")
    ax.set_ylabel("Bias of calibrated ATE")
    panel_label(ax, "(a) Surrogacy violation")
    ax.legend(loc="lower left")
    style_axes(ax)

    d, b, se = _collect(results_s3, "delta", "tau_true_p1_analytical")
    print(f"  [fig2b] max MC SE on bias (delta) = {se.max():.2e}")
    ax = axes[1]
    d_theory_grid = np.linspace(d.min(), d.max(), 200)
    ax.plot(d_theory_grid, delta_theory_slope * d_theory_grid, "--",
            color=COLORS["spine"], lw=1.3, zorder=4, dashes=(4, 2.4),
            label=fr"Predicted slope $= {delta_theory_slope:.2f}$")
    ax.errorbar(d, b, yerr=se, fmt="s-", color=COLORS["primary"],
                markersize=4.5, markeredgewidth=1.2, zorder=2,
                capsize=2.0, elinewidth=0.9,
                label="Empirical bias $\\pm$ MC SE")
    ax.axhline(0, color=COLORS["muted"], lw=0.7, linestyle=":")
    ax.set_xlabel(r"$\delta$ (shift in calibration coefficient)")
    ax.set_ylabel("Bias of calibrated ATE")
    panel_label(ax, "(b) Comparability violation")
    ax.legend(loc="lower left")
    style_axes(ax)

    fig.tight_layout()
    _save(fig, "fig2_failure_modes")


def figure3_multidraw_surrogacy(results_s6) -> None:
    """Calibrated ATE vs K with 2.5/97.5 percentile band and MC SE bars."""
    fig, ax = plt.subplots(figsize=(3.60, 2.30))

    Ks, means, ses, lo, hi = [], [], [], [], []
    true_ate = None
    for _, df in sorted(results_s6.items(),
                        key=lambda x: float(x[1]["K"].iloc[0])):
        Ks.append(float(df["K"].iloc[0]))
        v = df["tau_calibrated"].values
        m, se = _mean_se(v)
        means.append(m); ses.append(se)
        lo.append(np.percentile(v, 2.5))
        hi.append(np.percentile(v, 97.5))
        true_ate = float(df["tau_true"].iloc[0])

    Ks = np.array(Ks); means = np.array(means); ses = np.array(ses)
    lo = np.array(lo); hi = np.array(hi)
    print(f"  [fig3] max MC SE on mean ATE = {ses.max():.2e}")

    ax.fill_between(Ks, lo, hi, color=COLORS["primary"],
                    alpha=0.25, lw=0, label="95% MC interval")
    ax.errorbar(Ks, means, yerr=ses, fmt="o-", color=COLORS["primary"],
                markersize=4.5, markeredgewidth=1.2,
                capsize=2.0, elinewidth=0.9,
                label="Mean calibrated ATE $\\pm$ MC SE")
    ax.axhline(true_ate, color=COLORS["spine"],
               linestyle="--", linewidth=1.0,
               label=fr"True $\tau = {true_ate:.2f}$")

    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([f"{int(k)}" for k in Ks])
    ax.set_xlabel(r"Replication count $K$")
    ax.set_ylabel("Mean calibrated ATE")
    ax.legend(loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig3_multidraw_surrogacy")


def figure4_estimation_quality(results_s7) -> None:
    """Two-panel: RMSE vs K and de-attenuation of mean calibrated ATE vs K.

    Overlays the closed-form theoretical prediction from Proposition~1(b)
    and the Remark on attenuation: the calibrated ATE under classical
    measurement error in $Y^*$ is multiplicatively biased by the
    reliability ratio
        R_K = Var(theta | X) / [Var(theta | X) + sigma_eps^2 / K],
    so E[hat_tau] -> R_K * tau and the asymptotic-bias contribution to
    RMSE is |R_K - 1| * |tau|. The DGP-implied Var(theta | X) is read
    from each scenario row (stored by `s7_kdraw_estimation.one_rep`).
    """
    fig, axes = plt.subplots(1, 2, figsize=(5.00, 2.14))

    Ks, rmses, rmse_ses, mean_ates, mean_ses = [], [], [], [], []
    true_ate = None
    var_theta = None
    sigma_eps = None
    for _, df in sorted(results_s7.items(),
                        key=lambda x: float(x[1]["K"].iloc[0])):
        K = float(df["K"].iloc[0])
        true_ate = float(df["tau_true"].iloc[0])
        if "var_theta_given_X" in df.columns:
            var_theta = float(df["var_theta_given_X"].iloc[0])
        if "sigma_eps" in df.columns:
            sigma_eps = float(df["sigma_eps"].iloc[0])
        v = df["tau_calibrated"].values
        Ks.append(K)
        rmse, se_rmse = _rmse_se(v, true_ate)
        m, se_m = _mean_se(v)
        rmses.append(rmse); rmse_ses.append(se_rmse)
        mean_ates.append(m); mean_ses.append(se_m)

    Ks = np.array(Ks); rmses = np.array(rmses); rmse_ses = np.array(rmse_ses)
    mean_ates = np.array(mean_ates); mean_ses = np.array(mean_ses)
    print(f"  [fig4] max MC SE on RMSE = {rmse_ses.max():.2e}, "
          f"on mean ATE = {mean_ses.max():.2e}")

    K_grid = np.logspace(np.log10(Ks.min()), np.log10(Ks.max()), 200)
    if var_theta is not None and sigma_eps is not None:
        R_K_grid = var_theta / (var_theta + sigma_eps**2 / K_grid)
        predicted_mean_grid = R_K_grid * true_ate
        predicted_bias_grid = np.abs(R_K_grid - 1.0) * np.abs(true_ate)
    else:
        R_K_grid = None
        predicted_mean_grid = None
        predicted_bias_grid = None

    ax = axes[0]
    ax.errorbar(Ks, rmses, yerr=rmse_ses, fmt="o-", color=COLORS["secondary"],
                markersize=4.5, markeredgewidth=1.2,
                capsize=2.0, elinewidth=0.9,
                label="Empirical RMSE $\\pm$ MC SE")
    if predicted_bias_grid is not None:
        ax.plot(K_grid, predicted_bias_grid, "--",
                color=COLORS["accent"], lw=1.1,
                label=r"Predicted $|R_K - 1|\,|\tau|$")
    rmse_one_over_sqrtK = rmses[0] * np.sqrt(Ks[0] / Ks)
    ax.plot(Ks, rmse_one_over_sqrtK, ":", color=COLORS["muted"],
            lw=1.0, label=r"$\propto K^{-1/2}$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([f"{int(k)}" for k in Ks])
    ax.set_xlabel(r"Replication count $K$")
    ax.set_ylabel("RMSE")
    panel_label(ax, "(a) Estimation precision")
    ax.legend(loc="upper right")
    style_axes(ax)

    ax = axes[1]
    ax.errorbar(Ks, mean_ates, yerr=mean_ses, fmt="s-", color=COLORS["tertiary"],
                markersize=4.5, markeredgewidth=1.2,
                capsize=2.0, elinewidth=0.9,
                label="Mean calibrated ATE $\\pm$ MC SE")
    if predicted_mean_grid is not None:
        ax.plot(K_grid, predicted_mean_grid, "--",
                color=COLORS["accent"], lw=1.1,
                label=r"Predicted $R_K\,\tau$")
    ax.axhline(true_ate, color=COLORS["spine"],
               linestyle="--", linewidth=1.0,
               label=fr"True $\tau = {true_ate:.2f}$")
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([f"{int(k)}" for k in Ks])
    ax.set_xlabel(r"Replication count $K$")
    ax.set_ylabel("Mean calibrated ATE")
    panel_label(ax, "(b) De-attenuation")
    ax.legend(loc="lower right")
    style_axes(ax)

    fig.tight_layout()
    _save(fig, "fig4_estimation_quality")


def figure6_consistency_rate(results_s9) -> None:
    """Two-panel: RMSE vs n on log-log axes (with $n^{-1/2}$ reference) and
    RMSE * sqrt(n) vs n on a linear-log axis.

    Empirical verification of Theorem~3.1's sqrt(n)-consistency under the
    LinearDGP. The right panel should be approximately flat (modulo Monte-
    Carlo noise) if the sqrt(n) rate holds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(5.00, 2.14))

    ns, rmses, rmse_ses = [], [], []
    true_ate = None
    for _, df in sorted(results_s9.items(),
                        key=lambda x: float(x[1]["n"].iloc[0])):
        n = float(df["n"].iloc[0])
        true_ate = float(df["tau_true"].iloc[0])
        v = df["tau_calibrated"].values
        rmse, se_rmse = _rmse_se(v, true_ate)
        ns.append(n)
        rmses.append(rmse); rmse_ses.append(se_rmse)

    ns = np.array(ns); rmses = np.array(rmses); rmse_ses = np.array(rmse_ses)
    print(f"  [fig6] max MC SE on RMSE = {rmse_ses.max():.2e}")

    ax = axes[0]
    ax.errorbar(ns, rmses, yerr=rmse_ses, fmt="o-", color=COLORS["secondary"],
                markersize=4.5, markeredgewidth=1.2,
                capsize=2.0, elinewidth=0.9,
                label="Empirical RMSE $\\pm$ MC SE")
    rmse_ref = rmses[0] * np.sqrt(ns[0] / ns)
    ax.plot(ns, rmse_ref, "--", color=COLORS["accent"], lw=1.0,
            label=r"$\propto n^{-1/2}$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(k)}" for k in ns], rotation=0)
    ax.set_xlabel(r"Per-population sample size $n$")
    ax.set_ylabel("RMSE")
    panel_label(ax, "(a) Convergence")
    ax.legend(loc="upper right")
    style_axes(ax)

    ax = axes[1]
    flat = rmses * np.sqrt(ns)
    flat_ses = rmse_ses * np.sqrt(ns)
    ax.errorbar(ns, flat, yerr=flat_ses, fmt="s-", color=COLORS["tertiary"],
                markersize=4.5, markeredgewidth=1.2,
                capsize=2.0, elinewidth=0.9,
                label=r"$\mathrm{RMSE} \cdot \sqrt{n}\;\pm$ MC SE")
    ax.axhline(np.mean(flat), color=COLORS["spine"], lw=0.9,
               linestyle="--",
               label=fr"Mean $= {np.mean(flat):.3f}$")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(k)}" for k in ns], rotation=0)
    ax.set_xlabel(r"Per-population sample size $n$")
    ax.set_ylabel(r"$\mathrm{RMSE} \cdot \sqrt{n}$")
    panel_label(ax, r"(b) $\sqrt{n}$-rate verification")
    ax.legend(loc="upper right")
    style_axes(ax)

    fig.tight_layout()
    _save(fig, "fig6_consistency_rate")


def figure5_sensitivity_bound(results_s4) -> None:
    """Actual ATE discrepancy vs theoretical bound as overlap degrades."""
    fig, ax = plt.subplots(figsize=(3.60, 2.35))

    shifts, disc, disc_se, bound, bound_se = [], [], [], [], []
    for _, df in sorted(results_s4.items(),
                        key=lambda x: float(x[1]["shift_star"].iloc[0])):
        shifts.append(float(df["shift_star"].iloc[0]))
        d_mean, d_se = _mean_se(df["actual_discrepancy"].values)
        b_mean, b_se = _mean_se(df["bound"].values)
        disc.append(d_mean); disc_se.append(d_se)
        bound.append(b_mean); bound_se.append(b_se)
    shifts = np.array(shifts)
    disc = np.array(disc); disc_se = np.array(disc_se)
    bound = np.array(bound); bound_se = np.array(bound_se)
    print(f"  [fig5] max MC SE on discrepancy = {disc_se.max():.2e}, "
          f"on bound = {bound_se.max():.2e}")

    ax.fill_between(shifts, disc, bound, color=COLORS["accent"],
                    alpha=0.20, lw=0, label="Slack")
    ax.errorbar(shifts, bound, yerr=bound_se, fmt="s--", color=COLORS["accent"],
                markersize=4.5, markeredgewidth=1.2,
                capsize=2.0, elinewidth=0.9,
                label="Theoretical bound $\\pm$ MC SE")
    ax.errorbar(shifts, disc, yerr=disc_se, fmt="o-", color=COLORS["primary"],
                markersize=4.5, markeredgewidth=1.2,
                capsize=2.0, elinewidth=0.9,
                label="Actual discrepancy $\\pm$ MC SE")

    ax.set_xlabel(r"Shift $s$ in $Y^*$ for treated arm of $P=1$")
    ax.set_ylabel("ATE discrepancy")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.34)
    ax.legend(loc="upper left", fontsize=8.0)
    style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig5_sensitivity_bound")
