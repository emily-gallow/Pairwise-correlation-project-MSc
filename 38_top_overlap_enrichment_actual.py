"""Top-X overlap and fold enrichment, with session pairs re-labelled by
within-mouse acquisition-date interval rank (shortest / intermediate / longest).

Reuses the values in top_overlap_enrichment.csv; the asymmetric overlap
statistic is unchanged by the relabelling, only the grouping differs.

Outputs:
  <root>/top_overlap_enrichment_actual.csv
  <root>/top_overlap_enrichment_actual.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsa_common import load_gaps

RANK_COLOR = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--date-table", default="outputs/movie1/session_date_table.csv")
parser.add_argument("--ylim-overlap", type=float, nargs=2, default=None,
                    help="Shared y-limits for the overlap panel, e.g. 0 0.55")
parser.add_argument("--ylim-fold", type=float, nargs=2, default=None,
                    help="Shared y-limits for the fold panel, e.g. 0 48")
args = parser.parse_args()

root = Path(args.root)
gaps = load_gaps(args.date_table)

enr = pd.read_csv(root / "top_overlap_enrichment.csv")
enr["container_id"] = enr["container_id"].astype(str)
enr = enr[enr["container_id"].isin(gaps)].copy()
enr["actual_gap_days"] = enr.apply(
    lambda r: gaps.get(r["container_id"], {}).get(r["pair"], np.nan), axis=1)
enr["gap_rank"] = (enr.groupby(["container_id", "percentile"])["actual_gap_days"]
                      .rank(method="first").astype(int))

agg = enr.groupby(["gap_rank", "percentile"]).agg(
    overlap=("overlap", "mean"),
    chance=("chance_overlap", "mean"),
    fold=("fold_enrichment", "mean"),
    n=("container_id", "count"),
).reset_index()

rank_medians = (enr.drop_duplicates(subset=["container_id", "gap_rank"])
                   .groupby("gap_rank")["actual_gap_days"].median())
agg["median_gap_days"] = agg["gap_rank"].map(rank_medians.to_dict())
agg.to_csv(root / "top_overlap_enrichment_actual.csv", index=False)

labels = {1: f"Shortest interval (median {rank_medians[1]:.2f} d)",
          2: f"Intermediate interval (median {rank_medians[2]:.2f} d)",
          3: f"Longest interval (median {rank_medians[3]:.2f} d)"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
percentiles = sorted(agg["percentile"].unique(), reverse=True)

for ax, col, ylabel, title in [
    (axes[0], "overlap", "Top-X overlap fraction", "Observed top-X overlap vs chance"),
    (axes[1], "fold", "Fold enrichment (observed / chance)", "Fold enrichment over chance"),
]:
    per_mouse_col = "overlap" if col == "overlap" else "fold_enrichment"
    for rank in (1, 2, 3):
        for _, g in enr[enr["gap_rank"] == rank].groupby("container_id"):
            g = g.sort_values("percentile", ascending=False)
            ax.plot(g["percentile"], g[per_mouse_col], "-", color=RANK_COLOR[rank],
                    lw=0.8, alpha=0.25, zorder=1)

    for rank in (1, 2, 3):
        sub = agg[agg["gap_rank"] == rank].sort_values("percentile", ascending=False)
        ax.plot(sub["percentile"], sub[col], "-o", color=RANK_COLOR[rank],
                lw=2.8, ms=9, label=labels[rank], zorder=3)

    if col == "overlap":
        ax.plot(percentiles, np.array(percentiles) / 100, "--", color="red",
                lw=1.5, label="Chance (X/100)", zorder=2)
        loc = "upper right"
    else:
        ax.axhline(1, ls="--", color="#1f77b4", lw=1.5, label="Chance (1x)", zorder=2)
        loc = "upper left"

    ylim = args.ylim_overlap if col == "overlap" else args.ylim_fold
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Threshold X% (pairs ranked by |r|)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.invert_xaxis()
    ax.set_xticks(percentiles)
    ax.set_xticklabels([f"{p}%" for p in percentiles])
    ax.grid(alpha=0.3)
    ax.legend(loc=loc, fontsize=10, frameon=True)

plt.tight_layout()
out = root / "top_overlap_enrichment_actual.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
print(f"Saved {root / 'top_overlap_enrichment_actual.csv'}")
