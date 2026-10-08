#!/usr/bin/env python3
"""Fig.1 -- schematic of the three-step preregistered falsification chain
(freeze -> remove -> graft). Pure vector diagram, no data plotted."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = os.environ.get("P9_RESULTS_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.environ.get("P9_FIG_OUT", os.path.dirname(os.path.abspath(__file__)))

BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
AQUA = "#1baf7a"
RED = "#e34948"
YELLOW = "#eda100"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
HAIRLINE = "#e1e0d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "pdf.fonttype": 42,
})

COMP_COLORS = {
    "conv": YELLOW,
    "scan": BLUE,
    "gate": RED,
    "proj": MUTED,
}
COMP_LABELS = {"conv": "conv\nstem", "scan": "scan\n(Δ,A,B,C)", "gate": "SiLU\ngate", "proj": "proj"}


def block(ax, x0, y0, w, h, components, dim=None, removed=None, added=None, hatch_removed=False):
    """Draw a 4-slot block diagram; dim=set of slot names to grey out; removed=slot to cross out."""
    n = len(components)
    slot_w = w / n
    for i, comp in enumerate(components):
        cx = x0 + i * slot_w
        color = COMP_COLORS[comp]
        is_dim = dim and comp in dim
        is_removed = removed and comp == removed
        is_added = added and comp == added
        face = color if not is_dim else "#e8e7e2"
        edge = INK if not is_removed else CRITICAL
        rect = Rectangle((cx + 0.03, y0), slot_w - 0.06, h, facecolor=face, edgecolor=edge,
                          linewidth=1.6 if (is_removed or is_added) else 1.0, zorder=3,
                          linestyle="--" if is_removed else ("-." if is_added else "-"))
        ax.add_patch(rect)
        label = COMP_LABELS[comp]
        txtcolor = "white" if (not is_dim and comp in ("scan", "gate")) else INK
        if is_dim:
            txtcolor = MUTED
        ax.text(cx + slot_w / 2, y0 + h / 2, label, ha="center", va="center",
                 fontsize=7.3, color=txtcolor, zorder=4, linespacing=1.0)
        if is_removed:
            ax.plot([cx + 0.06, cx + slot_w - 0.06], [y0 + 0.03, y0 + h - 0.03],
                     color=CRITICAL, lw=2.2, zorder=5)
            ax.plot([cx + 0.06, cx + slot_w - 0.06], [y0 + h - 0.03, y0 + 0.03],
                     color=CRITICAL, lw=2.2, zorder=5)
        if is_added:
            ax.text(cx + slot_w / 2, y0 + h + 0.16, "+ added", ha="center", va="bottom",
                     fontsize=7, color=GOOD, fontweight="bold", zorder=4)
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=6))


def verdict_box(ax, x, y, w, text, color):
    box = FancyBboxPatch((x, y), w, 0.52, boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=color, edgecolor="none", alpha=0.14, zorder=2)
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch((x, y), w, 0.52, boxstyle="round,pad=0.02,rounding_size=0.05",
                                  facecolor="none", edgecolor=color, linewidth=1.3, zorder=3))
    ax.text(x + w / 2, y + 0.26, text, ha="center", va="center", fontsize=8.4,
             color=INK, zorder=4, linespacing=1.25, wrap=True)


def main():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    bw, bh = 2.6, 0.85
    y_block = 3.15
    stage_x = [0.35, 4.25, 8.15]
    titles = ["Step 1 -- Freeze the scan\n(necessity test)", "Step 2 -- Remove gate\n(localise)", "Step 3 -- Add gate to S4D\n(sufficiency test)"]

    # Step 1: freeze selectivity (scan slot dimmed = constants instead of input-dependent)
    block(ax, stage_x[0], y_block, bw, bh, ["conv", "scan", "gate", "proj"], dim={"scan"})
    ax.text(stage_x[0] + bw / 2, y_block + bh + 0.32, "BM3-frozen", ha="center", fontsize=8.5,
             color=SECONDARY_INK, style="italic")

    # Step 2: remove gate (crossed out) on frozen base
    block(ax, stage_x[1], y_block, bw, bh, ["conv", "scan", "gate", "proj"], dim={"scan"}, removed="gate")
    ax.text(stage_x[1] + bw / 2, y_block + bh + 0.32, "frozen − gate", ha="center", fontsize=8.5,
             color=SECONDARY_INK, style="italic")

    # Step 3: graft gate onto plain S4D
    block(ax, stage_x[2], y_block, bw, bh, ["scan", "gate"], added="gate")
    ax.text(stage_x[2] + bw / 2, y_block + bh + 0.32, "S4D + gate", ha="center", fontsize=8.5,
             color=SECONDARY_INK, style="italic")

    # arrows between stages
    for i in range(2):
        x_from = stage_x[i] + bw + 0.08
        x_to = stage_x[i + 1] - 0.08
        arr = FancyArrowPatch((x_from, y_block + bh / 2), (x_to, y_block + bh / 2),
                                arrowstyle="-|>", mutation_scale=16, color=MUTED, linewidth=1.6, zorder=2)
        ax.add_patch(arr)

    step_titles_y = y_block + bh + 0.85
    for x, t in zip(stage_x, titles):
        ax.text(x + bw / 2, step_titles_y, t, ha="center", va="bottom", fontsize=10.5,
                 fontweight="bold", color=INK, linespacing=1.3)

    # verdict boxes
    vy = 1.55
    verdict_box(ax, stage_x[0] - 0.15, vy, bw + 0.3,
                 "Δ(BM3−frozen) ≤ 5pp (CI incl. 0);\nvs S4D: +14–20pp, full freeze +21–24pp\n→ no scan term is primary", CRITICAL)
    verdict_box(ax, stage_x[1] - 0.15, vy, bw + 0.3,
                 "gate removal: −9 to −24pp,\nto ≈ S4D level or below\n→ gate is necessary here", GOOD)
    verdict_box(ax, stage_x[2] - 0.15, vy, bw + 0.3,
                 "S4D: gate 0.30 ≤ additive 0.41\nBM3: additive rescues ρ = 0.02\n→ gate-specific only in native host", CRITICAL)

    # bottom takeaway
    ax.text(5.75, 0.55, "Conditional on this block, testbed and budget: the gate's benefit depends on the host.",
             ha="center", va="center", fontsize=9.8, color=INK, style="italic")

    # legend for component colors
    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=10,
                        markerfacecolor=COMP_COLORS[c], markeredgecolor=INK, label=COMP_LABELS[c].replace("\n", " "))
               for c in ["conv", "scan", "gate", "proj"]]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.06), ncol=4,
               frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.2)

    fig.savefig(os.path.join(OUT, "fig1_chain.pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(OUT, "fig1_chain.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
    print("done")
