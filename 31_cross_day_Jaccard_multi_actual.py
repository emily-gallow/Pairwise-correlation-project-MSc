"""Cross-day Jaccard overlap vs density, signal and noise panels.

Inputs:
  <root>/cross_day_jaccard_multi.csv

Outputs:
  <root>/cross_day_jaccard_multi_actual.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIR_COLOR = {"A-B": "#1f77b4", "A-C": "#ff7f0e", "B-C": "#2ca02c"}
PAIR_ORDER = ["A-B", "A-C", "B-C"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
args = parser.parse_args()
root = Path(args.root)

jac = pd.read_csv(root / "cross_day_jaccard_multi.csv")
means = jac.groupby(["density", "type", "pair"])["jaccard"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

for ax, tp, title in [(axes[0], "signal", "Signal correlation"),
                      (axes[1], "noise", "Noise correlation")]:
    sub = jac[jac["type"] == tp]

    for pair in PAIR_ORDER:
        for _, g in sub[sub["pair"] == pair].groupby("container_id"):
            g = g.sort_values("density")
            ax.plot(g["density"] * 100, g["jaccard"], "-",
                    color=PAIR_COLOR[pair], lw=0.8, alpha=0.25, zorder=1)

    for pair in PAIR_ORDER:
        ps = means[(means["type"] == tp) & (means["pair"] == pair)].sort_values("density")
        if ps.empty:
            continue
        ax.plot(ps["density"] * 100, ps["jaccard"], "-",
                color=PAIR_COLOR[pair], lw=2.8, label=pair, zorder=3)

    d = np.linspace(sub["density"].min(), sub["density"].max(), 60)
    ax.plot(d * 100, d / (2 - d), "--", color="grey", lw=1.5,
            label="Chance", zorder=2)

    ax.set_xlabel("Density threshold (top-X %)", fontsize=12)
    ax.set_ylabel("Cross-day Jaccard overlap", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

fig.suptitle("Cross-day correlation network overlap vs density", fontsize=13, y=1.00)
plt.tight_layout()
out = root / "cross_day_jaccard_multi_actual.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
