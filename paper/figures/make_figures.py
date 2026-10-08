#!/usr/bin/env python3
"""Generate Fig.2-5 for the Neurocomputing mechanism paper from results/ data.

Data sources (see scratchpad/data_inventory.md for full provenance):
  Fig.2: results/frozensel_curvefix_20260703-0748/frozensel_curve_table.csv
  Fig.3, Fig.4: final-epoch re-runs in recheck/ (stage1/cells_recheck.jsonl,
         stage5/cells_stage5_exp2.jsonl); override the directory with P9_RECHECK_DIR
  Fig.5: results/perclass_analysis_20260708-1316/per_class_report.csv
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.environ.get("P9_RESULTS_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.environ.get("P9_FIG_OUT", os.path.dirname(os.path.abspath(__file__)))

# ---- validated CVD-safe categorical palette (fixed slot order, dataviz skill) ----
BLUE = "#2a78d6"     # BM3 / bm3_kin
ORANGE = "#eb6834"   # BM3-frozen
VIOLET = "#4a3aa7"   # S4D
AQUA = "#1baf7a"      # S4D+gate
RED = "#e34948"       # frozen-gate (nogate) -- primary finding
YELLOW = "#eda100"    # frozen-conv (noconv)
MAGENTA = "#e87ba4"   # frozen-A (As4d)

GRID_HAIRLINE = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
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

SNR_ORDER = ["clean", "+10dB", "+6dB", "0dB", "-2dB", "-6dB", "-10dB"]
SNR_X = [15, 10, 6, 0, -2, -6, -10]  # matches frozensel_curve_table.csv snr_db_for_plot convention
ADJ_BAND = {0, -2, -6}


def style_axes(ax):
    ax.grid(True, axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def shade_adjudication_band(ax):
    ax.axvspan(-6.5, 0.5, color=GRID_HAIRLINE, alpha=0.5, zorder=0, lw=0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig.2 -- section 4a: BM3 / BM3-frozen / S4D across full SNR grid
# ---------------------------------------------------------------------------
def fig2():
    path = os.path.join(ROOT, "results", "frozensel_curvefix_20260703-0748", "frozensel_curve_table.csv")
    rows = list(csv.DictReader(open(path)))
    arms = {"bm3_kin": ("BM3", BLUE, "o", "-"),
            "bm3_frozen": ("BM3-frozen", ORANGE, "s", "--"),
            "s4d": ("S4D", VIOLET, "^", ":")}
    data = {a: {} for a in arms}
    for r in rows:
        a = r["arm"]
        if a in data:
            data[a][float(r["snr_db_for_plot"])] = (float(r["mean_macro_f1"]), float(r["std_macro_f1"]))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    shade_adjudication_band(ax)
    for a, (label, color, marker, ls) in arms.items():
        xs = sorted(data[a])
        ys = [data[a][x][0] * 100 for x in xs]
        es = [data[a][x][1] * 100 for x in xs]
        ax.errorbar(xs, ys, yerr=es, label=label, color=color, marker=marker, linestyle=ls,
                     linewidth=2, markersize=6, capsize=3, zorder=3)
    style_axes(ax)
    ax.set_xlabel("SNR (dB); clean plotted at 15")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Freezing Δ,A,B,C projections")
    ax.text(-3, 45, "adjudication\nband", fontsize=8, color=MUTED, ha="center")
    ax.legend(loc="upper right")
    save(fig, "fig2_snr_curves")


# ---------------------------------------------------------------------------
# Fig.3 / Fig.4 -- final-epoch re-runs (primary rule), adjudication band only
# ---------------------------------------------------------------------------
RECHECK = os.environ.get("P9_RECHECK_DIR", os.path.join(ROOT, "recheck"))
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
BAND_LABELS = ["0 dB", "-2 dB", "-6 dB"]
T975_DF4 = 2.7764451051977987


def _cells():
    rows = []
    for rel in ["stage1/cells_recheck.jsonl", "stage5/cells_stage5_exp2.jsonl"]:
        rows += [json.loads(l) for l in open(os.path.join(RECHECK, rel))]
    return rows


def _vals(rows, arm, cond, rule):
    d = {r["seed"]: r[rule] * 100 for r in rows if r["arm"] == arm and r["condition"] == cond}
    assert sorted(d) == [0, 1, 2, 3, 4], (arm, cond, sorted(d))
    return [d[k] for k in range(5)]


def _paired(rows, a, b, cond, rule):
    d = [x - y for x, y in zip(_vals(rows, a, cond, rule), _vals(rows, b, cond, rule))]
    m = sum(d) / 5
    sd = (sum((x - m) ** 2 for x in d) / 4) ** 0.5
    return m, T975_DF4 * sd / 5 ** 0.5


def fig3():
    rows = _cells()
    comp = {"frozen_nogate": ("SiLU gate", RED), "frozen_noconv": ("conv stem", YELLOW),
            "frozen_As4d": ("heavy-tailed A", MAGENTA)}
    x = list(range(len(BAND)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0, color=AXIS, linewidth=1, zorder=2)
    ax.axhline(8, color=MUTED, linewidth=0.8, linestyle=":", zorder=2, label="pre-specified 8 pp threshold")
    for i, (arm, (label, color)) in enumerate(comp.items()):
        fin = [_paired(rows, "bm3_frozen", arm, c, "final") for c in BAND]
        best = [_paired(rows, "bm3_frozen", arm, c, "oracle")[0] for c in BAND]
        xs = [xi + (i - 1) * width for xi in x]
        ax.bar(xs, [m for m, _ in fin], width=width, color=color, label=label, zorder=3,
               yerr=[h for _, h in fin], error_kw=dict(ecolor=INK, elinewidth=0.9, capsize=2.5))
        ax.scatter(xs, best, marker="_", s=140, color=INK, linewidths=1.6, zorder=4,
                   label="best epoch (re-run)" if i == 0 else None)
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_xlabel("Condition (adjudication band)")
    ax.set_ylabel(r"$\Delta$(frozen $-$ arm), pp")
    ax.set_title("Single-component removal, final epoch")
    ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(-27, 46)
    save(fig, "fig3_ablation_bars")


def fig4():
    rows = _cells()
    arms = {"s4d_plus_gate": ("S4D+gate", AQUA, "D", "-."), "s4d": ("S4D", VIOLET, "^", ":"),
            "bm3_frozen": ("BM3-frozen", ORANGE, "s", "--")}
    xs = [0, -2, -6]
    m = {a: [sum(_vals(rows, a, c, "final")) / 5 for c in BAND] for a in arms}
    R = [(g - s) / (f - s) for g, s, f in zip(m["s4d_plus_gate"], m["s4d"], m["bm3_frozen"])]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for a, (label, color, marker, ls) in arms.items():
        ax.plot(xs, m[a], label=label, color=color, marker=marker, linestyle=ls, linewidth=2, markersize=6, zorder=3)
    style_axes(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0", "-2", "-6"])
    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB), adjudication band")
    ax.set_ylabel("Macro-F1 (%), final epoch")
    ax.set_title("Narrow gate graft on S4D (width-matched: see text)")
    for xv, rv, yv in zip(xs, R, m["s4d_plus_gate"]):
        ax.annotate(f"R={rv:.3f}", xy=(xv, yv), xytext=(xv, yv + 4.0), fontsize=8, color=SECONDARY_INK,
                    ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.text(0.5, 0.05, f"mean R = {sum(R) / 3:.3f} (< 0.3 'insufficient' bin; best epoch 0.174)",
            transform=ax.transAxes, fontsize=8.5, color=SECONDARY_INK, ha="center")
    ax.set_ylim(55, 95)
    ax.legend(loc="upper right")
    save(fig, "fig4_graft")


# ---------------------------------------------------------------------------
# Fig.5 -- section 4d: per-class IR/OR recall x gate presence x noise
# ---------------------------------------------------------------------------
def fig5():
    path = os.path.join(ROOT, "results", "perclass_analysis_20260708-1316", "per_class_report.csv")
    rows = list(csv.DictReader(open(path)))
    arm_style = {
        "bm3_kin": ("BM3", BLUE, "-"),
        "bm3_frozen": ("BM3-frozen", ORANGE, "-"),
        "s4d_plus_gate": ("S4D+gate", AQUA, "-"),
        "frozen_nogate": ("frozen$-$gate", RED, "--"),
        "s4d": ("S4D", VIOLET, "--"),
    }
    cond_order = ["clean", "awgn@+0dB", "awgn@-6dB"]
    cond_x = {"clean": 15, "awgn@+0dB": 0, "awgn@-6dB": -6}
    data = {a: {"OR": {}, "IR": {}} for a in arm_style}
    for r in rows:
        a, c, cls = r["arm"], r["condition"], r["class"]
        if a in data:
            data[a][cls][cond_x[c]] = float(r["recall"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, cls, title in zip(axes, ["IR", "OR"], ["Inner-race (hard class)", "Outer-race"]):
        shade_adjudication_band(ax)
        for a, (label, color, ls) in arm_style.items():
            xs = sorted(data[a][cls])
            ys = [data[a][cls][x] for x in xs]
            marker = "o" if ls == "-" else "x"
            ax.plot(xs, ys, label=label, color=color, linestyle=ls, marker=marker,
                     linewidth=2, markersize=6, zorder=3)
        style_axes(ax)
        ax.set_xlabel("SNR (dB); clean plotted at 15")
        ax.set_title(title)
        ax.set_ylim(0.30, 1.03)
    axes[0].set_ylabel("Recall")
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Gate-related differences concentrate on the minority class",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig5_perclass")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig2()
    fig3()
    fig4()
    fig5()
    print("done")
