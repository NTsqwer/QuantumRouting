"""Figure: the four-pass compilation pipeline and the objective mismatch.

Initial mapping -> Routing -> Gate cancellation -> Scheduling, drawn as a
left-to-right flow. The routing pass (the one we modify) is accented; its
optimisation target, SWAP count, is shown above it, and the pipeline's true
output metric, makespan, is shown above the scheduling pass. A "not equal"
marker between the two targets carries the paper's core observation: the proxy
the router optimises is not the cost the pipeline produces.

Colour is used sparingly and with intent: one accent for the pass we change,
one for the output metric, neutral grey for the rest. Matches the house style
of figures/make_fig_swap_example.py. Output: figures/pipeline.{pdf,png}.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Palette: restrained, print-friendly.
INK = "#1a1a1a"
GREY_FILL = "#f4f4f4"
GREY_EDGE = "#9a9a9a"
ROUTE_FILL = "#dde8f5"   # soft blue: the pass we modify
ROUTE_EDGE = "#2f6db0"
METRIC = "#b03030"        # red: the output metric / mismatch

BOX_W, BOX_H = 2.35, 0.95
GAP = 0.95
Y = 0.0

PASSES = [
    ("Initial\nmapping", "grey"),
    ("Routing\n(SABRE)", "route"),
    ("Gate\ncancellation", "grey"),
    ("Scheduling", "grey"),
]


def box(ax, x, label, kind):
    if kind == "route":
        fc, ec, lw = ROUTE_FILL, ROUTE_EDGE, 1.8
    else:
        fc, ec, lw = GREY_FILL, GREY_EDGE, 1.2
    p = FancyBboxPatch((x, Y - BOX_H / 2), BOX_W, BOX_H,
                       boxstyle="round,pad=0.02,rounding_size=0.12",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + BOX_W / 2, Y, label, ha="center", va="center",
            fontsize=10, color=INK, zorder=3)
    return x + BOX_W / 2  # centre x


def main():
    fig, ax = plt.subplots(figsize=(7.4, 2.55))

    centres = []
    x = 0.0
    for label, kind in PASSES:
        cx = box(ax, x, label, kind)
        centres.append(cx)
        x += BOX_W + GAP

    # Flow arrows between consecutive passes.
    for i in range(len(PASSES) - 1):
        x0 = centres[i] + BOX_W / 2
        x1 = centres[i + 1] - BOX_W / 2
        ax.annotate("", xy=(x1, Y), xytext=(x0, Y),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))

    # Objective labels above routing and scheduling.
    y_obj = BOX_H / 2 + 1.05
    cx_rt, cx_sc = centres[1], centres[3]
    ax.text(cx_rt, y_obj, "optimises\nSWAP count", ha="center", va="center",
            fontsize=9, color=ROUTE_EDGE, style="italic", zorder=3)
    ax.text(cx_sc, y_obj, "output metric\nmakespan", ha="center", va="center",
            fontsize=9, color=METRIC, style="italic", zorder=3)
    # Connectors from label down to its pass.
    for cx, col in [(cx_rt, ROUTE_EDGE), (cx_sc, METRIC)]:
        ax.annotate("", xy=(cx, BOX_H / 2 + 0.06), xytext=(cx, y_obj - 0.32),
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.1,
                                    linestyle=(0, (2, 2))))

    # "not equal" between the two objectives.
    xmid = (cx_rt + cx_sc) / 2
    ax.annotate("", xy=(cx_sc - 0.55, y_obj), xytext=(cx_rt + 0.55, y_obj),
                arrowprops=dict(arrowstyle="-", color=METRIC, lw=1.3))
    ax.text(xmid, y_obj + 0.02, r"$\neq$", ha="center", va="center",
            fontsize=15, color=METRIC, zorder=4,
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=METRIC, lw=1.2))

    ax.set_xlim(-0.4, x - GAP + 0.4)
    ax.set_ylim(-BOX_H / 2 - 0.4, y_obj + 0.7)
    ax.set_aspect("equal")
    ax.set_axis_off()

    plt.tight_layout(pad=0.3)
    out = "figures/pipeline.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.05, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
