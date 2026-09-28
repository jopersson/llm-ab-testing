"""Replot the three Upworthy paper figures on the shared editorial figstyle.

Reads cached results only (no LLM calls, no re-estimation):
    output/tables/uw_question.json                  (k-sweep + human ATE)
    output/tables/uw_synthetic_signal.json          (positive control cells)
    output/tables/uw_contamination_probe.json       (token-F1 summary)
    output/tables/uw_contamination_probe_rows.parquet (token-F1 rows)

Writes (matching the names used in main_extended_stochastic.tex):
    output/figures/uw_question_ksweep.{pdf,png}
    output/figures/uw_synthetic_signal_recovery.{pdf,png}
    output/figures/uw_contamination_probe.{pdf,png}

These figures are authored ~1.25x their on-page width so that the LaTeX
downscale lands the type near the same ~6pt size as the simulation figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
EMP = HERE.parent
ROOT = EMP.parent  # Code/
sys.path.insert(0, str(ROOT))
from figstyle import apply_style, COLORS, style_axes  # noqa: E402

apply_style()

TAB = EMP / "output" / "tables"
FIG = EMP / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig_ksweep() -> None:
    d = json.loads((TAB / "uw_question.json").read_text())
    hum = d["ate"]["human"]
    raw = d["k_sweep_raw"]
    cal = d["k_sweep_calib"]
    Ks = sorted((int(k) for k in raw))
    Kstr = [str(k) for k in Ks]

    series = [
        (r"Raw $\bar Y^*$", raw, COLORS["primary"], "o"),
        ("Calib. OLS", cal["ols"], COLORS["tertiary"], "s"),
        ("Calib. RF", cal["rf"], COLORS["secondary"], "^"),
        ("Calib. GBT", cal["gbt"], COLORS["neutral"], "D"),
    ]

    fig, ax = plt.subplots(figsize=(3.35, 2.32))
    xpos = np.arange(len(Ks), dtype=float)
    offs = np.linspace(-0.16, 0.16, len(series))

    ax.axhline(hum["ate"], color=COLORS["accent"], lw=1.2,
               label=fr"Human ATE $= {hum['ate']:.4f}$", zorder=1)
    ax.axhspan(hum["ate"] - 1.96 * hum["se"], hum["ate"] + 1.96 * hum["se"],
               color=COLORS["accent"], alpha=0.22, lw=0, zorder=0)

    for (lab, sw, color, mk), dx in zip(series, offs):
        vals = [sw[k]["ate"] for k in Kstr]
        ses = [1.96 * sw[k]["se"] for k in Kstr]
        ax.errorbar(xpos + dx, vals, yerr=ses, fmt=mk + "-", color=color,
                     capsize=2.2, elinewidth=1.0,
                    label=lab, zorder=3)

    ax.set_xticks(xpos)
    ax.set_xticklabels(Kstr)
    ax.set_xlim(-0.5, len(Ks) - 0.5)
    ax.set_xlabel(r"Number of LLM draws $K$")
    ax.set_ylabel("Estimated ATE")
    # Five entries will not fit inside these axes without covering the human
    # band and the GBT series, so the legend sits above the plot area.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, fontsize=7.6, handlelength=1.5,
              columnspacing=1.1, handletextpad=0.4)
    style_axes(ax)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "uw_question_ksweep.pdf")
    fig.savefig(FIG / "uw_question_ksweep.png", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG / 'uw_question_ksweep.pdf'}")


def fig_signal_recovery() -> None:
    d = json.loads((TAB / "uw_synthetic_signal.json").read_text())
    cells = pd.DataFrame(d["cells"])
    meta = d["meta"]
    sigma_grid = meta["sigma_grid"]
    tau_grid = meta["tau_grid"]
    methods = [("ols", "OLS", COLORS["primary"], "o"),
               ("rf", "Random forest", COLORS["tertiary"], "s"),
               ("gbt", "Gradient boost", COLORS["secondary"], "^")]

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5), sharex=True, sharey=True)
    tau_lo = min(tau_grid) - 0.002
    tau_hi = max(tau_grid) + 0.002
    # Small horizontal dodge so the three near-coincident method curves are
    # individually visible instead of stacking on one another.
    dodge = {"ols": -4.5e-4, "rf": 0.0, "gbt": 4.5e-4}

    for ax, sigma_nu in zip(axes, sigma_grid):
        sub = cells[np.isclose(cells["sigma_nu"], sigma_nu)]
        ax.plot([tau_lo, tau_hi], [tau_lo, tau_hi], "--",
                color=COLORS["spine"], lw=0.9, alpha=0.7, zorder=1,
                label=r"$45^\circ$")
        ax.axhline(0, color=COLORS["muted"], lw=0.4, alpha=0.7, zorder=0)
        ax.axvline(0, color=COLORS["muted"], lw=0.4, alpha=0.7, zorder=0)
        for key, lab, color, mk in methods:
            s = sub[sub["method"] == key].sort_values("tau_dagger")
            ax.errorbar(s["tau_dagger"] + dodge[key], s["tau_hat"], yerr=s["se"],
                        fmt=mk + "-", color=color, 
                        lw=1.6, markersize=5.0, markeredgewidth=1.2,
                        capsize=3.0, elinewidth=1.3, capthick=1.3,
                        label=lab, zorder=3)
        ratio = float(sub["sigma_nu_over_sigma_y"].iloc[0])
        R_K = float(sub["reliability_K"].iloc[0])
        ax.text(0.035, 0.95,
                rf"$\sigma_\nu = {ratio:.0f}\,\sigma_Y$,  $R_{{K=10}} = {R_K:.2f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.6,
                color=COLORS["spine"])
        ax.set_xlim(tau_lo, tau_hi)

    fig.supxlabel(r"Known effect $\tau^\dagger$")
    fig.supylabel(r"$\hat\tau_{\rm calib}$")
    axes[0].legend(loc="lower right", ncol=1, framealpha=1.0,
                   facecolor="white", edgecolor="0.8", fontsize=7.2)
    for ax in axes:
        style_axes(ax)
    fig.tight_layout(pad=0.2, w_pad=0.9)
    fig.savefig(FIG / "uw_synthetic_signal_recovery.pdf")
    fig.savefig(FIG / "uw_synthetic_signal_recovery.png", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG / 'uw_synthetic_signal_recovery.pdf'}")


def fig_contamination() -> None:
    sc = pd.read_parquet(TAB / "uw_contamination_probe_rows.parquet")
    summary = json.loads((TAB / "uw_contamination_probe.json").read_text())
    mean_f1 = summary["mean_token_f1"]

    fig, ax = plt.subplots(figsize=(3.40, 2.10))
    ax.axvspan(0.1, 0.3, facecolor=COLORS["muted"], alpha=0.28,
               edgecolor=COLORS["muted"], linewidth=0.7)
    ax.hist(sc["token_f1"], bins=25, color=COLORS["primary"], alpha=1.0,
            edgecolor=COLORS["primary"], linewidth=0.4, zorder=2)
    ax.axvline(mean_f1, color=COLORS["accent"], lw=1.3, zorder=3)
    ax.set_xlim(0.0, max(0.5, float(sc["token_f1"].max()) * 1.05))
    ax.margins(y=0.18)
    ax.set_xlabel(r"Token $F_1$ (LLM continuation vs. true suffix)")
    ax.set_ylabel("Count")
    handles = [
        Patch(facecolor=COLORS["muted"], alpha=0.28, edgecolor=COLORS["muted"],
              linewidth=0.7, label="Carlini et al. (2023)"),
        Line2D([0], [0], color=COLORS["accent"], lw=1.3,
               label=fr"mean $= {mean_f1:.3f}$"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=1.0,
              facecolor="white", edgecolor="0.8")
    style_axes(ax)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "uw_contamination_probe.pdf")
    fig.savefig(FIG / "uw_contamination_probe.png", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG / 'uw_contamination_probe.pdf'}")


if __name__ == "__main__":
    fig_ksweep()
    fig_signal_recovery()
    fig_contamination()
    print("done")
