"""Full-data (10-rep) cross-day RSA against acquisition-date interval.

Per-mouse OLS with exact sign-flip permutation and mouse-level bootstrap for
the reported statistics; the figure shows a random-intercept mixed-effects fit
with its 95% confidence band.

Outputs:
  <root>/day_gap_scatter_multi_fulldata_actual.csv
  <root>/per_mouse_slope_fulldata_actual.csv
  <root>/day_gap_scatter_multi_fulldata_actual.png
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsa_common import load_gaps, per_mouse_fits, summarise_drift

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--date-table", default="outputs/movie1/session_date_table.csv")
parser.add_argument("--exclude", nargs="*", default=[])
parser.add_argument("--boot", type=int, default=20000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

root = Path(args.root)
excluded = [str(x) for x in args.exclude]
suffix = f"_excl_{'_'.join(excluded)}" if excluded else ""
gaps = load_gaps(args.date_table)

rows = []
for f in sorted(root.glob("*/cross_session_rsa.csv")):
    cid = f.parent.name
    if not cid.isdigit() or cid not in gaps or cid in excluded:
        continue
    for _, r in pd.read_csv(f).iterrows():
        for pair, col in (("A-B", "AB_fulldata"), ("B-C", "BC_fulldata"),
                          ("A-C", "AC_fulldata")):
            rows.append(dict(container_id=cid, type=r["type"], pair=pair,
                             actual_gap_days=gaps[cid][pair],
                             metric="fulldata", r=float(r[col])))

all_df = pd.DataFrame(rows)
all_df.to_csv(root / f"day_gap_scatter_multi_fulldata_actual{suffix}.csv", index=False)
print(f"Cohort: n = {all_df['container_id'].nunique()} mice")

per_mouse, summary = [], {}
for tp in ("signal", "noise"):
    pm = per_mouse_fits(all_df[all_df["type"] == tp])
    pm["type"] = tp
    per_mouse.append(pm)

    s = summarise_drift(pm["drift_per_day"].values, boot=args.boot, seed=args.seed)
    summary[tp] = s
    print(f"{tp} (n = {s['n']} mice): D = {s['mean_drift']:+.4f} r/day, "
          f"p = {s['p_exact']:.4f} ({s['n_perms']:,} perms), "
          f"95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}], |d_z| = {abs(s['dz']):.3f}")

pd.concat(per_mouse, ignore_index=True).to_csv(
    root / f"per_mouse_slope_fulldata_actual{suffix}.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
gap_min, gap_max = 0.5, np.ceil(all_df["actual_gap_days"].max() + 0.5)

for ax, tp, title in [
    (axes[0], "signal", "Signal correlation: cross-day RSA vs interval"),
    (axes[1], "noise", "Noise correlation: cross-day RSA vs interval"),
]:
    sub = all_df[all_df["type"] == tp].dropna(subset=["actual_gap_days", "r"])

    for _, g in sub.groupby("container_id"):
        g = g.sort_values("actual_gap_days")
        ax.plot(g["actual_gap_days"], g["r"], "-", color="grey",
                lw=0.6, alpha=0.5, zorder=1)

    ax.scatter(sub["actual_gap_days"], sub["r"], s=60, color="#1f77b4",
               alpha=0.85, edgecolors="none", zorder=2,
               label="Cross-day RSA (per session pair)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf = smf.mixedlm("r ~ actual_gap_days", sub,
                         groups=sub["container_id"]).fit(reml=True, method="lbfgs",
                                                         disp=False)
    b0 = float(mf.fe_params["Intercept"])
    b1 = float(mf.fe_params["actual_gap_days"])
    cov = mf.cov_params().loc[["Intercept", "actual_gap_days"],
                              ["Intercept", "actual_gap_days"]].values

    xr = np.linspace(gap_min, gap_max, 120)
    yr = b0 + b1 * xr
    se = np.sqrt(np.maximum(cov[0, 0] + xr ** 2 * cov[1, 1] + 2 * xr * cov[0, 1], 0.0))
    ax.fill_between(xr, yr - 1.96 * se, yr + 1.96 * se, color="#e91e63",
                    alpha=0.18, zorder=3, label="Mixed-effects 95% CI")
    ax.plot(xr, yr, "-", color="#e91e63", lw=2.5, zorder=4,
            label=f"MixedLM slope = {b1:+.3f} RSA/day")

    ax.set_xlabel("Interval between sessions (days)", fontsize=12)
    ax.set_ylabel("Cross-day RSA (Pearson r)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(gap_min, gap_max)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

plt.tight_layout()
out = root / f"day_gap_scatter_multi_fulldata_actual{suffix}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
