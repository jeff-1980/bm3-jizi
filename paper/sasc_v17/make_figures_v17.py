#!/usr/bin/env python3
"""Generate the independent-noise figures of the SASC version (v17).

Data: the 210 independent-noise cells released in bm3-jizi (tree 3837c9d),
  recheck/stage8/cells_stage8_A_xjtu.jsonl       (forward XJTU-SY, 5 arms)
  recheck/stage8/cells_stage8_B_pu_shared.jsonl  (Paderborn shared bearings, 3 arms)
  recheck/stage8/cells_stage8_C_xjtu_clti.jsonl  (forward XJTU-SY, fully frozen gateless/additive)
  recheck/stage9/cells_stage9_D_xjtu_rev.jsonl   (reverse XJTU-SY, 4 arms)
Run from the repository root (default P9_RECHECK_DIR=recheck).

  fig_effects: paired final-epoch differences (mean, 95% paired t-interval, seeds)
  fig_curves:  per-epoch evaluation macro-F1 and training loss (5-seed mean, min-max band)
  printed:     the contrasts of the figure, the noise-depth table and the training-loss table
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RECHECK = os.environ.get("P9_RECHECK_DIR", "recheck")
OUT = os.environ.get("P9_FIG_OUT", os.path.dirname(os.path.abspath(__file__)))

BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
AQUA = "#1baf7a"
RED = "#e34948"
YELLOW = "#eda100"
GRID_HAIRLINE = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID_HAIRLINE,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

T975_DF4 = 2.7764451051977987
SNRS = [0.0, -2.0, -6.0]
FILES = {
    "fwd": ["stage8/cells_stage8_A_xjtu.jsonl", "stage8/cells_stage8_C_xjtu_clti.jsonl"],
    "pu": ["stage8/cells_stage8_B_pu_shared.jsonl"],
    "rev": ["stage9/cells_stage9_D_xjtu_rev.jsonl"],
}


def load():
    cells = {}
    for setting, rels in FILES.items():
        for rel in rels:
            for line in open(os.path.join(RECHECK, rel)):
                r = json.loads(line)
                cells[(setting, r["arm"], float(r["snr_db"]), int(r["seed"]))] = r
    return cells


def paired(cells, setting, a, b, snr):
    d = np.array([(cells[(setting, a, snr, s)]["final"] - cells[(setting, b, snr, s)]["final"]) * 100
                  for s in range(5)])
    h = T975_DF4 * d.std(ddof=1) / np.sqrt(5)
    return d, d.mean(), h


def style_axes(ax):
    ax.grid(True, axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


PANELS = [
    ("(a) Frozen host $-$ S4D", [
        ("fwd", "frozen_clti", "s4d", "forward, fully frozen", VIOLET, "o"),
        ("fwd", "bm3_frozen", "s4d", "forward, partially frozen", BLUE, "s"),
        ("rev", "frozen_clti", "s4d", "reverse, fully frozen", ORANGE, "o"),
        ("rev", "bm3_frozen", "s4d", "reverse, partially frozen", YELLOW, "s"),
    ]),
    ("(b) Gate-removal cost (host $-$ gateless)", [
        ("fwd", "frozen_clti", "frozen_clti_nogate", "forward, fully frozen", VIOLET, "o"),
        ("fwd", "bm3_frozen", "frozen_nogate", "forward, partially frozen", BLUE, "s"),
        ("rev", "bm3_frozen", "frozen_nogate", "reverse, partially frozen", YELLOW, "s"),
        ("pu", "bm3_frozen", "frozen_nogate", "PU shared, partially frozen", AQUA, "D"),
    ]),
    ("(c) Multiplicative $-$ additive", [
        ("fwd", "frozen_clti", "frozen_clti_add", "forward, fully frozen", VIOLET, "o"),
        ("fwd", "bm3_frozen", "bm3_frozen_add", "forward, partially frozen", BLUE, "s"),
    ]),
    ("(d) Gateless host $-$ S4D", [
        ("fwd", "frozen_clti_nogate", "s4d", "forward, fully frozen", VIOLET, "o"),
        ("fwd", "frozen_nogate", "s4d", "forward, partially frozen", BLUE, "s"),
        ("rev", "frozen_nogate", "s4d", "reverse, partially frozen", YELLOW, "s"),
    ]),
]


def fig_effects(cells):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for ax, (title, series) in zip(axes.flat, PANELS):
        n = len(series)
        offs = np.linspace(-0.28, 0.28, n) if n > 1 else [0.0]
        for (setting, a, b, label, color, marker), off in zip(series, offs):
            for i, snr in enumerate(SNRS):
                d, m, h = paired(cells, setting, a, b, snr)
                x = i + off
                ax.scatter(np.full(5, x), d, s=9, color=color, alpha=0.5, linewidths=0, zorder=2)
                ax.errorbar([x], [m], yerr=[[h], [h]], fmt=marker, color=color, markersize=5,
                            capsize=2.5, elinewidth=1.1, zorder=3, label=label if i == 0 else None)
        ax.axhline(0, color=INK, linewidth=0.8, zorder=1)
        style_axes(ax)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["0 dB", "$-$2 dB", "$-$6 dB"])
        ax.set_title(title, loc="left")
        ax.set_ylabel("Difference (pp)")
    handles, labels = {}, {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig_effects")


CURVE_PANELS = [
    ("fwd", -6.0, "Forward, $-$6 dB", [
        ("frozen_clti", "fully frozen", VIOLET, "-"),
        ("frozen_clti_add", "fully frozen, additive", AQUA, "-."),
        ("frozen_clti_nogate", "fully frozen, gateless", RED, "--"),
        ("bm3_frozen", "partially frozen", BLUE, "-"),
        ("s4d", "S4D", MUTED, ":"),
    ]),
    ("rev", 0.0, "Reverse, 0 dB", [
        ("frozen_clti", "fully frozen", VIOLET, "-"),
        ("bm3_frozen", "partially frozen", BLUE, "-"),
        ("frozen_nogate", "partially frozen, gateless", RED, "--"),
        ("s4d", "S4D", MUTED, ":"),
    ]),
]


def fig_curves(cells):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex=True)
    ep = np.arange(1, 51)
    for col, (setting, snr, title, arms) in enumerate(CURVE_PANELS):
        for arm, label, color, ls in arms:
            f1 = np.array([cells[(setting, arm, snr, s)]["hist"] for s in range(5)]) * 100
            tl = np.array([cells[(setting, arm, snr, s)]["train_loss"] for s in range(5)])
            for row, y in enumerate([f1, tl]):
                ax = axes[row, col]
                ax.plot(ep, y.mean(0), color=color, linestyle=ls, linewidth=1.6, label=label, zorder=3)
                ax.fill_between(ep, y.min(0), y.max(0), color=color, alpha=0.12, linewidth=0, zorder=2)
        axes[0, col].set_title(title, loc="left")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Epoch")
    if True:
        # trivial floors of the reverse evaluation set (Section 4.2)
        axes[0, 1].axhline(46.8, color=INK, linewidth=0.7, linestyle=":", zorder=1)
        axes[0, 1].text(1, 42.5, "majority class", fontsize=7, color=SECONDARY_INK, ha="left")
    axes[0, 0].set_ylabel("Evaluation macro-F1 (%)")
    axes[1, 0].set_ylabel("Training loss")
    for ax in axes.flat:
        style_axes(ax)
    axes[0, 0].legend(loc="lower right", fontsize=6.5)
    axes[0, 1].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_curves")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    c = load()
    assert len(c) == 210, len(c)
    fig_effects(c)
    fig_curves(c)
    # print the numbers the figure shows, for cross-checking against the tables
    for title, series in PANELS:
        for setting, a, b, label, *_ in series:
            vals = [paired(c, setting, a, b, s) for s in SNRS]
            print(f"{title[:4]} {label:30s} " + "  ".join(
                f"{m:+.1f} [{m - h:.1f},{m + h:.1f}] {int((d > 0).sum())}/5" for d, m, h in vals))
    # Table (noise depth): change of the gate-removal cost relative to 0 dB, seeds paired
    for setting, host, gl, name in [("fwd", "frozen_clti", "frozen_clti_nogate", "Forward, fully frozen"),
                                    ("fwd", "bm3_frozen", "frozen_nogate", "Forward, partially frozen"),
                                    ("rev", "bm3_frozen", "frozen_nogate", "Reverse, partially frozen"),
                                    ("pu", "bm3_frozen", "frozen_nogate", "PU shared, partially frozen")]:
        cost = {snr: paired(c, setting, host, gl, snr)[0] for snr in SNRS}
        out = []
        for snr in (-2.0, -6.0):
            d = cost[snr] - cost[0.0]
            h = T975_DF4 * d.std(ddof=1) / np.sqrt(5)
            out.append(f"{d.mean():+.1f} [{d.mean() - h:.1f},{d.mean() + h:.1f}] {int((d > 0).sum())}/5")
        print(f"depth {name:28s} " + "  ".join(out))
    # Table (training loss): final-epoch training cross-entropy, mean [min, max] over seeds
    for (setting, arm, snr) in sorted({(k[0], k[1], k[2]) for k in c}):
        v = np.array([c[(setting, arm, snr, s)]["train_loss"][-1] for s in range(5)])
        print(f"loss {setting:4s} {arm:20s} {snr:+.0f} dB  {v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]")
    print("done")
