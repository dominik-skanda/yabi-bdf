#!/usr/bin/env python3
"""
YABI benchmark gallery — one-command reproduction.

    python3 pyplot/plot_all.py

Steps:
  1. Configure + build the library and the trajectory dumper (CMake).
  2. Run the dumper to write trajectories/*.csv for every benchmark.
  3. Render one polished figure per benchmark plus a combined overview grid
     (plots/benchmarks_overview.png).

The plots show YABI's own trajectories across the canonical stiff test set
(Hairer–Wanner §IV, STIFF DETEST): time series on linear/log axes, phase
portraits for the oscillators, and a spatial snapshot for the reaction–diffusion
problem. Reference values (scipy Radau / testbench) are marked with stars.
"""
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

# keep console output ordered when piped (subprocess writes are unbuffered)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(ROOT, "trajectories")
OUT = os.path.join(ROOT, "plots")
os.makedirs(OUT, exist_ok=True)

PALETTE = ["#0d3b66", "#ee6c4d", "#3d5a80", "#98c1d9", "#e0a96d", "#2a9d8f",
           "#e76f51", "#264653", "#e9c46a", "#6d597a", "#b56576", "#588157"]
plt.rcParams.update({
    "figure.dpi": 110,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def run(cmd):
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}")


def build_and_dump():
    print("== 1/3 build ==")
    run(["cmake", "-S", ".", "-B", "build"])
    run(["cmake", "--build", "build", "-j"])
    print("== 2/3 dump trajectories ==")
    run([os.path.join("build", "dump_trajectories")])


# --- data --------------------------------------------------------------------
def load(name):
    path = os.path.join(TRAJ, name + ".csv")
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def species_colors(n):
    return [colormaps["plasma"](i / max(n - 1, 1)) for i in range(n)]


# --- panels ------------------------------------------------------------------
def panel_time(ax, name, title, ycols, logx=False, logy=False, labels=None, ref=None):
    d = load(name)
    if d is None:
        ax.set_title(title); ax.text(0.5, 0.5, "no data", ha="center", va="center"); return
    t = d[:, 0]
    for k, c in enumerate(ycols):
        ax.plot(t, d[:, c], lw=1.2, label=labels[k] if labels else f"$y_{c}$")
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    if ref:
        for (tt, yy, lab) in ref:
            ax.scatter([tt], [yy], marker="*", s=90, zorder=6, edgecolor="black",
                       linewidth=0.4, label=lab)
    ax.set_xlabel("t"); ax.set_title(title)
    if labels: ax.legend(loc="best")


def panel_phase(ax, name, title, xlab="$y_1$", ylab="$y_2$"):
    d = load(name)
    if d is None:
        ax.set_title(title); ax.text(0.5, 0.5, "no data", ha="center", va="center"); return
    ax.plot(d[:, 1], d[:, 2], lw=1.2, color=PALETTE[0])
    ax.scatter([d[0, 1]], [d[0, 2]], color=PALETTE[0], s=40, zorder=5, label="start")
    ax.scatter([d[-1, 1]], [d[-1, 2]], color=PALETTE[1], s=40, zorder=5, marker="s", label="end")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title); ax.legend(loc="best")


def panel_species(ax, name, title, n, logx=False, logy=False):
    d = load(name)
    if d is None:
        ax.set_title(title); ax.text(0.5, 0.5, "no data", ha="center", va="center"); return
    t = d[:, 0]
    cols = species_colors(n)
    for i in range(n):
        ax.plot(t, d[:, 1 + i], lw=1.1, color=cols[i], label=f"$y_{i}$")
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel("t"); ax.set_title(title)


def panel_spatial(ax, name, title):
    d = load(name)
    if d is None:
        ax.set_title(title); ax.text(0.5, 0.5, "no data", ha="center", va="center"); return
    n = d.shape[1] - 1
    node = np.arange(n)
    times = [d[0, 0], d[len(d) // 2, 0], d[-1, 0]]
    cols = [PALETTE[0], PALETTE[4], PALETTE[1]]
    for tt, c in zip(times, cols):
        row = d[np.argmin(np.abs(d[:, 0] - tt)), 1:]
        ax.plot(node, row, lw=1.4, color=c, label=f"t={tt:.2g}")
    ax.set_xlabel("node i"); ax.set_title(title); ax.legend(loc="best")


# --- figures -----------------------------------------------------------------
def make_overview():
    fig, axes = plt.subplots(4, 3, figsize=(16, 11))
    fig.suptitle("YABI — BDF integrator benchmark gallery", fontsize=16, fontweight="bold", y=0.98)

    panel_time(axes[0, 0], "brusselator", "Brusselator (limit cycle, stiff)", [1, 2],
               labels=["$y_1$", "$y_2$"],
               ref=[(5, 0.938299, "ref y1(5)"), (5, 4.52728, "ref y2(5)")])
    panel_time(axes[0, 1], "robertson", "Robertson (stiff, t→10⁴)", [1, 2, 3],
               labels=["$y_1$", "$y_2$", "$y_3$"], logx=True,
               ref=[(1e4, 0.1073004, "ref y1"), (1e4, 0.8926991, "ref y3")])
    panel_phase(axes[0, 2], "vanderpol", "Van der Pol μ=1000 (relaxation limit cycle)")
    panel_species(axes[1, 0], "hires", "HIRES (8-species photochemistry)", 8)
    panel_time(axes[1, 1], "prothero_robinson", "Prothero–Robinson", [1, 2], labels=["$y_1$", "$y_2$"])
    panel_time(axes[1, 2], "oregonator", "Oregonator (Belousov–Zhabotinsky)", [1, 2, 3],
               labels=["$y_1$", "$y_2$", "$y_3$"], logy=True)
    panel_phase(axes[2, 0], "pendulum", "Damped pendulum", xlab="$\\theta$", ylab="$\\omega$")
    panel_spatial(axes[2, 1], "polanyi_wigner", "Polanyi–Wigner (reaction–diffusion, n=20)")
    panel_species(axes[2, 2], "network", "Reaction network (6 species)", 6, logy=True)
    panel_time(axes[3, 0], "e5", "E5 (stiff, t→10⁶)", [1, 2, 3], labels=["$y_1$", "$y_2$", "$y_3$"],
               logx=True, logy=True)
    panel_phase(axes[3, 1], "oscillator", "Harmonic oscillator", xlab="$y_1$", ylab="$y_2$")

    ax = axes[3, 2]
    ax.axis("off")
    summary = (
        "YABI — variable-order, variable-step BDF integrator\n"
        "(orders 1–6) with forward / adjoint / simultaneous\n"
        "sensitivity via CppAD.\n\n"
        "Panels show YABI trajectories across the canonical\n"
        "stiff test set (Hairer–Wanner §IV, STIFF DETEST).\n"
        "★ marks the independently verified reference value\n"
        "(scipy Radau / testbench).\n\n"
        "Phase portraits (Van der Pol, pendulum, oscillator)\n"
        "project the state; Robertson and E5 use log axes to\n"
        "span 4–7 decades."
    )
    ax.text(0.0, 0.98, summary, va="top", ha="left", fontsize=9, linespacing=1.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "benchmarks_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/benchmarks_overview.png")


def make_individual():
    specs = [
        ("brusselator", "time", "Brusselator (limit cycle, stiff)", [1, 2]),
        ("robertson", "time", "Robertson (stiff, t→10⁴)", [1, 2, 3]),
        ("vanderpol", "phase", "Van der Pol μ=1000 (relaxation limit cycle)", None),
        ("hires", "species", "HIRES (8-species photochemistry)", 8),
        ("prothero_robinson", "time", "Prothero–Robinson", [1, 2]),
        ("oregonator", "time", "Oregonator (Belousov–Zhabotinsky)", [1, 2, 3]),
        ("pendulum", "phase", "Damped pendulum", None),
        ("polanyi_wigner", "spatial", "Polanyi–Wigner (reaction–diffusion, n=20)", None),
        ("network", "species", "Reaction network (6 species)", 6),
        ("e5", "time", "E5 (stiff, t→10⁶)", [1, 2, 3]),
        ("oscillator", "phase", "Harmonic oscillator", None),
    ]
    for name, kind, title, arg in specs:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        if kind == "time":
            panel_time(ax, name, title, arg, labels=[f"$y_{c}$" for c in arg])
        elif kind == "phase":
            panel_phase(ax, name, title)
        elif kind == "species":
            panel_species(ax, name, title, arg)
        elif kind == "spatial":
            panel_spatial(ax, name, title)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("wrote individual plots to plots/")


if __name__ == "__main__":
    build_and_dump()
    print("== 3/3 plot ==")
    make_individual()
    make_overview()
    print("Done — see plots/benchmarks_overview.png")
