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
COMP_LABELS = {"conv": "conv\nstem", "scan": "scan", "gate": "SiLU\ngate", "proj": "proj"}


def block(ax, x0, y0, w, h, components, dim=None, removed=None, added=None, hatch_removed=False, labels=None, added_text="+ added"):
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
        label = (labels or {}).get(comp, COMP_LABELS[comp])
        txtcolor = "white" if (not is_dim and comp in ("scan", "gate")) else INK
        if is_dim:
            txtcolor = MUTED
        ax.text(cx + slot_w / 2, y0 + h / 2, label, ha="center", va="center",
                 fontsize=6.4, color=txtcolor, zorder=4, linespacing=1.0)
        if is_removed:
            ax.plot([cx + 0.06, cx + slot_w - 0.06], [y0 + 0.03, y0 + h - 0.03],
                     color=INK, lw=1.4, zorder=5)
            ax.plot([cx + 0.06, cx + slot_w - 0.06], [y0 + h - 0.03, y0 + 0.03],
                     color=INK, lw=1.4, zorder=5)
        if is_added:
            ax.text(cx + slot_w / 2, y0 + h + 0.16, added_text, ha="center", va="bottom",
                     fontsize=6.8, color=GOOD, fontweight="bold", zorder=4)
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=6))


def verdict_box(ax, x, y, w, text, color):
    box = FancyBboxPatch((x, y), w, 0.62, boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=color, edgecolor="none", alpha=0.14, zorder=2)
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch((x, y), w, 0.62, boxstyle="round,pad=0.02,rounding_size=0.05",
                                  facecolor="none", edgecolor=color, linewidth=1.3, zorder=3))
    ax.text(x + w / 2, y + 0.31, text, ha="center", va="center", fontsize=8.6,
             color=INK, zorder=4, linespacing=1.25, wrap=True)


def main():
    import textwrap
    W, H = 7.6, 4.35                      # 1 data unit = 1 inch; the figure is shown at about 6.3 in
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    bw, bh = 2.15, 0.72
    y_block = 2.3
    stage_x = [0.2, 2.75, 5.3]
    titles = ["Step 1: freeze the scan\n(partial and full)", "Step 2: remove the gate\n(in the frozen host)",
              "Step 3: additive branch\nreplaces the gate"]
    block(ax, stage_x[0], y_block, bw, bh, ["conv", "scan", "gate", "proj"], dim={"scan"})
    block(ax, stage_x[1], y_block, bw, bh, ["conv", "scan", "gate", "proj"], dim={"scan"}, removed="gate")
    block(ax, stage_x[2], y_block, bw, bh, ["conv", "scan", "gate", "proj"], dim={"scan"}, added="gate",
          labels={"gate": "additive\nbranch"}, added_text="")
    for x, t in zip(stage_x, ["BM3-frozen / frozen$-$all", "frozen $-$ gate", "frozen + additive branch"]):
        ax.text(x + bw / 2, y_block + bh + 0.1, t, ha="center", va="bottom", fontsize=8, color=SECONDARY_INK, style="italic")
    for i in range(2):
        ax.add_patch(FancyArrowPatch((stage_x[i] + bw + 0.06, y_block + bh / 2), (stage_x[i + 1] - 0.06, y_block + bh / 2),
                                     arrowstyle="-|>", mutation_scale=14, color=SECONDARY_INK, linewidth=1.6, zorder=2))
    for x, t in zip(stage_x, titles):
        ax.text(x + bw / 2, y_block + bh + 0.5, t, ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK, linespacing=1.25)
    vy, vh, vw = 0.95, 0.95, 2.3
    boxes = [("partial / full freeze vs S4D:\n+12\u201318 / +20\u201323 pp (forward)\n\u2192 input-independent\ncoefficients suffice", CRITICAL),
             ("partial host, forward:\n+9 / +16 / +27 pp\n(resolved at \u22126 dB only);\nreverse: resolved at every level", GOOD),
             ("mult. \u2212 add. at \u22126 dB:\n+23 (full) / +27 (partial) pp;\nadditive vs gateless:\nnot resolved", CRITICAL)]
    for x, (t, c) in zip(stage_x, boxes):
        xb = x - 0.05
        ax.add_patch(FancyBboxPatch((xb, vy), vw, vh, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=c, edgecolor="none", alpha=0.14, zorder=2))
        ax.add_patch(FancyBboxPatch((xb, vy), vw, vh, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="none", edgecolor=c, linewidth=1.2, zorder=3))
        ax.text(xb + vw / 2, vy + vh / 2, t, ha="center", va="center", fontsize=7.9, color=INK, zorder=4, linespacing=1.3)
    msg = "Conditional on this block, testbed and budget: input-independent scan coefficients suffice; gate cost grows with noise only in the forward fully frozen block."
    ax.text(W / 2, 0.38, "\n".join(textwrap.wrap(msg, 88)), ha="center", va="center", fontsize=8.2, color=INK, style="italic")
    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=8, markerfacecolor=COMP_COLORS[c], markeredgecolor=INK,
                      label=("scan (Δ,A,B,C)" if c == "scan" else COMP_LABELS[c].replace("\n", " "))) for c in ["conv", "scan", "gate", "proj"]]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=4, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.0)
    fig.savefig(os.path.join(OUT, "fig1_chain.pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(OUT, "fig1_chain.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
    print("done")
