"""Figures 1 and 2 of the paper.

  figures/fig1_convergence.pdf   ||v_t - V_M||_inf of every run (thin) and their mean (thick), from results/,
                                 and the range of the final distance ||v_T - V_TD||_inf (printed)
  figures/fig2_bias.pdf          worst-case and mean bias ||V_TD - V_M||_inf over the emission matrices and the
                                 bound B_rho of Theorem 2 (computed exactly, needs no runs)

Usage: python plot_figures.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

import config as cfg
import model

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42,
                     "font.family": "serif", "mathtext.fontset": "cm"})
COLORS = ["C0", "C1", "C2"]            # one colour per gamma in Fig. 2


def load(prefix):
    """results/{prefix}_g*.npz, sorted by decreasing gamma."""
    Ds = [dict(np.load(f)) for f in glob.glob(os.path.join(cfg.RESULTS, f"{prefix}_g*.npz"))]
    return sorted(Ds, key=lambda D: -float(D["gamma"]))


def plain_log_ticks(ax, xs=(0.01, 0.1, 1)):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    os.makedirs(cfg.FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(cfg.FIGURES, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(cfg.FIGURES, name + ".pdf"))


def fig1_convergence():
    conv = load("conv")
    if not conv:
        return print("no results/conv_g*.npz: run run_convergence.py first")
    C = len(cfg.RHO_MIN)
    fig, axs = plt.subplots(C, len(conv), figsize=(7.16, 1.9 * C), sharex=True, sharey=True, squeeze=False,
                            constrained_layout=True)
    for j, D in enumerate(conv):
        for c, rs in enumerate(cfg.RHO_MIN):
            ax = axs[c, j]
            for ie in range(len(cfg.EPS_LIST)):              # every run, then the mean on top
                ax.plot(D["ck"], D["err"][c, ie].T, color=f"C{ie}", lw=0.25, alpha=0.01, rasterized=True)
            for ie in range(len(cfg.EPS_LIST)):
                ax.plot(D["ck"], D["err"][c, ie].astype(float).mean(0), color=f"C{ie}", lw=1.4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 2.0)
            ax.set_title(f"({'abcdef'[c * len(conv) + j]}) " + rf"$\rho_{{\min}}={rs:g}$, "
                         rf"$\gamma={float(D['gamma']):g}$")
            if c == C - 1:
                ax.set_xlabel(r"$t$")
    for c in range(C):
        axs[c, 0].set_ylabel(r"$\|v_t-V^{\varphi}_{\mathcal{M}}\|_\infty$")
    hs = [Line2D([], [], color=f"C{ie}", lw=1.4, label=rf"$\varepsilon={e:g}$") for ie, e in enumerate(cfg.EPS_LIST)]
    half = (len(hs) + 1) // 2                                  # legend split over panels (a) and (d)
    leg = dict(loc="lower left", fontsize=6, handlelength=1.6, labelspacing=0.25, borderpad=0.35, framealpha=0.9)
    axs[0, 0].legend(handles=hs[:half], **leg)
    axs[-1, 0].legend(handles=hs[half:], **leg)
    fig.suptitle("Convergence of TD on observations to the value under full observation", fontsize=9)
    save(fig, "fig1_convergence")
    final = [D["err_td"][..., -1].astype(float).mean(-1) for D in conv]   # mean over the runs at the last update
    print(f"mean ||v_T - V_TD||_inf over the runs: between {min(f.min() for f in final):.2e} "
          f"and {max(f.max() for f in final):.2e}")


def fig2_bias():
    """One panel per rho_min, one colour per gamma. Solid with markers: worst-case bias over the emission matrices,
    dotted: mean bias, dashed: bound B_rho of Theorem 2, drawn up to eps_bar where it reaches 2 V_max (triangle)."""
    e = np.array(cfg.EPS_LIST[1:])
    fig, axs = plt.subplots(1, len(cfg.RHO_MIN), figsize=(3.5, 2.7), sharey=True, constrained_layout=True)
    for c, rs in enumerate(cfg.RHO_MIN):
        ax = axs[c]
        for j, g in enumerate(cfg.GAMMAS):
            ex = model.setting(g, cfg.K)
            ebar = model.eps_bar(rs, g)
            eb = np.geomspace(e.min(), min(ebar, e.max()), 100)
            ax.plot(eb, model.bias_bound(eb, rs, ex["RMAX"], g), color=COLORS[j], ls="--", lw=0.9)
            if ebar <= e.max():
                ax.plot([ebar], [2 * ex["RMAX"] / (1 - g)], marker="v", ms=4, color=COLORS[j], ls="none")
            ax.plot(e, ex["bias"][c, 1:].mean(1), color=COLORS[j], lw=1.0, ls=":")
            ax.plot(e, ex["bias"][c, 1:].max(1), color=COLORS[j], lw=1.3, marker="o", ms=2.8)
        plain_log_ticks(ax)
        ax.set_title(f"({'ab'[c]}) " + rf"$\rho_{{\min}}={rs:g}$")
        ax.set_xlabel(r"$\varepsilon$")
    axs[0].set_ylabel(r"$\|V_{\mathrm{TD}}-V^{\varphi}_{\mathcal{M}}\|_\infty$")
    gam = [Line2D([], [], color=COLORS[j], lw=1.6, label=rf"$\gamma={g:g}$") for j, g in enumerate(cfg.GAMMAS)]
    kind = [Line2D([], [], color="0.3", lw=1.3, marker="o", ms=2.8, label="worst-case bias"),
            Line2D([], [], color="0.3", lw=1.0, ls=":", label="mean bias"),
            Line2D([], [], color="0.3", lw=0.9, ls="--", label=r"bound $\mathfrak{B}_\rho$"),
            Line2D([], [], color="0.3", ls="none", marker="v", ms=4, label=r"$\mathfrak{B}_\rho=2V_{\max}$")]
    axs[0].set_ylim(3e-4, 40)                                  # room below the curves for the legends
    leg = dict(loc="lower right", fontsize=6, handlelength=1.6, labelspacing=0.25, borderpad=0.35)
    axs[0].legend(handles=gam, **leg)
    axs[1].legend(handles=kind, **leg)
    fig.suptitle("Bias of TD on observations and the bound of Theorem 2", fontsize=8)
    save(fig, "fig2_bias")


if __name__ == "__main__":
    fig1_convergence()
    fig2_bias()
