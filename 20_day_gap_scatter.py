"""Per-mouse cross-day RSA against nominal day gap, signal and noise panels.

Writes the long-format table that script 21 aggregates across mice.

Inputs:
  <root>/cross_session_rsa.csv

Outputs:
  <root>/day_gap_scatter.csv
  <root>/day_gap_scatter[_fulldata].png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]
PAIR_GAP = {("A", "B"): 1, ("B", "C"): 1, ("A", "C"): 2}
PAIR_COLOR = {("A", "B"): "#a05a8a", ("B", "C"): "#d63384", ("A", "C"): "#ffd60a"}
# A-B and B-C share a nominal gap of 1 day; jitter separates them visually only.
JITTER = {("A", "B"): -0.05, ("B", "C"): +0.05, ("A", "C"): 0.0}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--metric", choices=["matched", "fulldata"], default="matched")
args = parser.parse_args()

root = Path(args.root)
df = pd.read_csv(root / "cross_session_rsa.csv")

if args.metric == "matched":
    col_map = {("A", "B"): "AB", ("B", "C"): "BC", ("A", "C"): "AC"}
    metric_label = "matched (5-rep, mean of 4 half-pairings)"
else:
    col_map = {("A", "B"): "AB_fulldata", ("B", "C"): "BC_fulldata",
               ("A", "C"): "AC_fulldata"}
    metric_label = "full data (10-rep cross-day)"

rows = {}
for tp in ("signal", "noise"):
    sub = df[df["type"] == tp]
    if sub.empty:
        raise ValueError(f"No '{tp}' row in {root / 'cross_session_rsa.csv'}")
    rows[tp] = sub.iloc[0]

long_df = pd.DataFrame([
    dict(pair=f"{a}-{b}", days_apart=PAIR_GAP[(a, b)], type=tp,
         metric=args.metric, r=float(rows[tp][col_map[(a, b)]]))
    for tp in ("signal", "noise") for a, b in PAIRS
])
long_df.to_csv(root / "day_gap_scatter.csv", index=False)
print(long_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

for ax, tp, panel_title in [(axes[0], "signal", "Signal correlation"),
                            (axes[1], "noise", "Noise correlation")]:
    pts = [(PAIR_GAP[pair] + JITTER[pair], float(rows[tp][col_map[pair]]), pair)
           for pair in PAIRS]

    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color="grey", lw=1.5, alpha=0.7, zorder=1)
    for x, y, pair in pts:
        ax.scatter(x, y, s=180, color=PAIR_COLOR[pair], edgecolors="black",
                   linewidth=1.2, zorder=2, label=f"{pair[0]}-{pair[1]}")

    ax.set_xlabel("Days apart", fontsize=12)
    ax.set_ylabel("Cross-day similarity (RSA Pearson r)", fontsize=12)
    ax.set_title(panel_title, fontsize=13)
    ax.set_xticks([1, 2])
    ax.set_xlim(0.6, 2.4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="best", frameon=True)

cid_label = root.name if root.name.isdigit() else "661437138"
try:
    cells_str = f"{int(np.load(root / 'A' / 'signal_corr.npy').shape[0])} matched cells"
except Exception:
    cells_str = "matched cells"

fig.suptitle(f"Cross-day correlation structure stability vs day gap - container "
             f"{cid_label} (VISp)\n{cells_str} - natural_movie_one - RSA: "
             f"{metric_label}", fontsize=12, y=1.02)
plt.tight_layout()

suffix = "" if args.metric == "matched" else f"_{args.metric}"
out = root / f"day_gap_scatter{suffix}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
print(f"Saved {root / 'day_gap_scatter.csv'}")
