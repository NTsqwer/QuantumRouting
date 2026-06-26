"""Figure: the lambda frontier --- tunable makespan vs SWAP-count trade-off.

Reads results/lambda_frontier.json. Plots, averaged over the routing-heavy cells,
the mean makespan reduction (%) against the mean number of EXTRA two-qubit gates
SABRE-MS inserts, as lambda sweeps the grid. lambda=0 is production SABRE (origin:
0 extra gates, 0% reduction). Each point is annotated with its lambda. The curve
shows that small lambda already captures most of the makespan reduction (and even
removes gates), while large lambda buys little extra makespan at a rising gate
cost --- so lambda is a knob the practitioner sets to the device.

Output: figures/lambda_frontier.{pdf,png}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1a1a1a"
BLUE = "#2f6db0"
GREY = "#9a9a9a"


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = json.load(open(os.path.join(here, "results", "lambda_frontier.json")))
    cells = [r for r in d.values() if r and "sweep" in r]
    lambdas = [row["lam"] for row in cells[0]["sweep"]]

    mk = [np.mean([next(x for x in r["sweep"] if x["lam"] == lam)["makespan_red_pct"]
                   for r in cells]) for lam in lambdas]
    dq = [np.mean([next(x for x in r["sweep"] if x["lam"] == lam)["n2q_delta"]
                   for r in cells]) for lam in lambdas]

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(dq, mk, "-", color=BLUE, lw=1.4, zorder=1)
    ax.scatter(dq, mk, s=28, color=BLUE, edgecolor=INK, linewidth=0.6, zorder=2)

    for lam, x, y in zip(lambdas, dq, mk):
        tag = "$\\lambda=0$ (SABRE)" if lam == 0 else f"{lam:g}"
        dy = -10 if lam == 0 else 6
        ax.annotate(tag, (x, y), textcoords="offset points", xytext=(4, dy),
                    fontsize=8, color=(GREY if lam == 0 else INK))

    ax.axhline(0, color=GREY, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.axvline(0, color=GREY, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_xlabel("extra two-qubit gates vs SABRE  (mean per circuit)")
    ax.set_ylabel("makespan reduction vs SABRE")
    ax.set_title("Tunable makespan / SWAP-count trade-off ($\\lambda$ sweep)",
                 fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(here, "figures", "lambda_frontier.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print("Saved", out)


if __name__ == "__main__":
    main()
