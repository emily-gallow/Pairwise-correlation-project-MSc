"""Matched (5-rep) cross-day RSA against acquisition-date interval.

Per-mouse OLS, exact sign-flip permutation and mouse-level bootstrap.
Drift magnitude D = -slope, so positive D means RSA falls as the gap grows.

Outputs:
  <root>/day_gap_scatter_multi_<metric>_actual.csv
  <root>/per_mouse_slope_<metric>_actual.csv
  <root>/day_gap_scatter_multi_<metric>_actual.png
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

from rsa_common import PAIR_COLOR, load_gaps, per_mouse_fits, summarise_drift

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--metric", choices=["matched", "fulldata"], default="matched")
parser.add_argument("--date-table", default="outputs/movie1/session_date_table.csv")
parser.add_argument("--boot", type=int, default=20000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

root = Path(args.root)
gaps = load_gaps(args.date_table)

found = sorted(p for p in root.glob("*/day_gap_scatter.csv") if p.parent.name.isdigit())
if not found:
    raise SystemExit(f"No day_gap_scatter.csv found under {root}/*/")

frames = []
for path in found:
    df = pd.read_csv(path)
    df["container_id"] = path.parent.name
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_df = all_df[all_df["metric"] == args.metric].copy()
all_df["actual_gap_days"] = all_df.apply(
    lambda r: gaps.get(str(r["container_id"]), {}).get(r["pair"], np.nan), axis=1)
all_df.to_csv(root / f"day_gap_scatter_multi_{args.metric}_actual.csv", index=False)

per_mouse, summary = [], {}
for tp in ("signal", "noise"):
    pm = per_mouse_fits(all_df[all_df["type"] == tp])
    pm["type"] = tp
    per_mouse.append(pm)

    s = summarise_drift(pm["drift_per_day"].values, boot=args.boot, seed=args.seed)
    s["slope_mean"] = -s["mean_drift"]
    s["intercept_mean"] = float(pm["ols_intercept"].mean())
    summary[tp] = s

    print(f"{tp} (n = {s['n']} mice): D = {s['mean_drift']:+.4f} r/day, "
          f"p = {s['p_exact']:.4f} ({s['n_perms']:,} perms), "
          f"95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}], |d_z| = {abs(s['dz']):.3f}")

pd.concat(per_mouse, ignore_index=True).to_csv(
    root / f"per_mouse_slope_{args.metric}_actual.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
gap_min, gap_max = 0.5, np.ceil(all_df["actual_gap_days"].max() + 0.5)

for ax, tp, title in [
    (axes[0], "signal", "Signal correlation: cross-day RSA vs actual interval"),
    (axes[1], "noise", "Noise correlation: cross-day RSA vs actual interval"),
]:
    sub = all_df[all_df["type"] == tp].dropna(subset=["actual_gap_days", "r"])
    for pair in ("A-B", "A-C", "B-C"):
        ps = sub[sub["pair"] == pair]
        ax.scatter(ps["actual_gap_days"], ps["r"], s=60, color=PAIR_COLOR[pair],
                   alpha=0.85, edgecolors="none", zorder=2, label=pair)

    s = summary[tp]
    xr = np.linspace(gap_min, gap_max, 60)
    ax.plot(xr, s["intercept_mean"] + s["slope_mean"] * xr, "-",
            color="#1f77b4", lw=2.5, zorder=3,
            label=f"Within-mouse slope = {s['slope_mean']:+.3f} RSA/day")

    ax.set_xlabel("Actual interval between sessions (days)", fontsize=12)
    ax.set_ylabel("Cross-day RSA (Pearson r)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(gap_min, gap_max)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

plt.tight_layout()
out = root / f"day_gap_scatter_multi_{args.metric}_actual.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
