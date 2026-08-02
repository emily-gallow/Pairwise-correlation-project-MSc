"""Timescale sensitivity sweep on cross-day RSA.

Refits the per-mouse RSA-on-interval regression at several event-integration
bin widths to check the drift result is not specific to the frame rate.
Drift magnitude D = -slope.

Outputs:
  <root>/timescale_sweep_rsa_actual.csv
  <root>/timescale_sweep_stats_actual.csv
  <root>/timescale_sweep_actual.png
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsa_common import PAIR_COLOR, PAIRS, load_gaps, per_mouse_fits, summarise_drift

SESSIONS = ["A", "B", "C"]
FPS = 30.0
EPS = 1e-10

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--bin-ms", type=int, nargs="+", default=[33, 100, 250, 500])
parser.add_argument("--boot", type=int, default=20000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--date-table", default="outputs/movie1/session_date_table.csv")
args = parser.parse_args()

root = Path(args.root)
gaps = load_gaps(args.date_table)


def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def bin_epoch(epoch, bf):
    L, n = epoch.shape
    nb = L // bf
    return epoch[:nb * bf].reshape(nb, bf, n).sum(axis=1)


def session_corrs(events, frames, bf, cell_idx):
    ev = events[cell_idx, :].T
    L = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    T = np.stack([bin_epoch(ev[int(sf): int(sf) + L, :], bf) for sf, _ in frames])
    R, nb, n = T.shape

    psth = T.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = np.corrcoef(psth, rowvar=False)
    dead = psth.var(axis=0) <= EPS
    S[dead, :] = np.nan
    S[:, dead] = np.nan

    sd = T.std(axis=0)
    nz = sd > EPS
    z = ((T - psth[None]) / np.where(nz, sd, 1.0)[None]) * nz[None]
    Z = z.reshape(R * nb, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        N = np.corrcoef(Z, rowvar=False)
    dead = Z.var(axis=0) <= EPS
    N[dead, :] = np.nan
    N[:, dead] = np.nan

    return {"signal": S, "noise": N}


def upper_tri_corr(A, B):
    iu = np.triu_indices(A.shape[0], k=1)
    a, b = A[iu], B[iu]
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[good], b[good])[0, 1])


cid_dirs = [p for p in sorted(root.iterdir())
            if p.is_dir() and p.name.isdigit()
            and all((p / s / "events_matrix.npy").exists() for s in SESSIONS)]
print(f"Containers with all 3 sessions: {len(cid_dirs)}")

rows = []
for cid_dir in cid_dirs:
    cid = cid_dir.name
    if cid not in gaps:
        continue
    ev = {s: np.load(cid_dir / s / "events_matrix.npy") for s in SESSIONS}
    fr = {s: np.load(cid_dir / s / "movie_repeat_frames.npy") for s in SESSIONS}
    matched = np.where(present_mask(ev["A"]) & present_mask(ev["B"])
                       & present_mask(ev["C"]))[0]
    if len(matched) < 5:
        continue

    for bin_ms in args.bin_ms:
        bf = max(1, int(round(bin_ms / 1000.0 * FPS)))
        mats = {s: session_corrs(ev[s], fr[s], bf, matched) for s in SESSIONS}
        for a, b in PAIRS:
            pair = f"{a}-{b}"
            for tp in ("signal", "noise"):
                rows.append(dict(container_id=cid, bin_ms=bin_ms, bin_frames=bf,
                                 pair=pair, actual_gap_days=gaps[cid][pair], type=tp,
                                 r=upper_tri_corr(mats[a][tp], mats[b][tp]),
                                 n_matched_cells=int(len(matched))))
    print(f"  {cid}: {len(matched)} matched cells")

raw = pd.DataFrame(rows)
raw.to_csv(root / "timescale_sweep_rsa_actual.csv", index=False)

stat_rows, fits = [], {}
for bin_ms in args.bin_ms:
    for tp in ("signal", "noise"):
        pm = per_mouse_fits(raw[(raw["bin_ms"] == bin_ms) & (raw["type"] == tp)])
        s = summarise_drift(pm["drift_per_day"].values, boot=args.boot, seed=args.seed)
        slope = -s["mean_drift"]
        intercept = float(pm["ols_intercept"].mean())
        fits[(bin_ms, tp)] = (slope, intercept)
        stat_rows.append(dict(bin_ms=bin_ms, type=tp, n_animals=s["n"],
                              per_day_drift=s["mean_drift"], perm_p=s["p_exact"],
                              n_perms=s["n_perms"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                              cohens_dz=s["dz"], mean_slope=slope,
                              mean_intercept=intercept))

stats = pd.DataFrame(stat_rows)
stats.to_csv(root / "timescale_sweep_stats_actual.csv", index=False)
print(stats.round(4).to_string(index=False))

n_bins = len(args.bin_ms)
fig, axes = plt.subplots(2, n_bins, figsize=(3.6 * n_bins + 1, 8),
                         sharex=True, sharey="row")
gap_min, gap_max = 0.5, np.ceil(raw["actual_gap_days"].max() + 0.5)
lookup = {(int(r["bin_ms"]), r["type"]): r for _, r in stats.iterrows()}

for row_i, tp in enumerate(("signal", "noise")):
    for col_i, bin_ms in enumerate(args.bin_ms):
        ax = axes[row_i, col_i]
        sub = raw[(raw["bin_ms"] == bin_ms) & (raw["type"] == tp)].dropna(
            subset=["actual_gap_days", "r"])

        for pair in ("A-B", "A-C", "B-C"):
            ps = sub[sub["pair"] == pair]
            ax.scatter(ps["actual_gap_days"], ps["r"], s=35, color=PAIR_COLOR[pair],
                       alpha=0.75, edgecolors="none", zorder=2)

        slope, intercept = fits[(bin_ms, tp)]
        xr = np.linspace(gap_min, gap_max, 60)
        ax.plot(xr, intercept + slope * xr, "-", color="#1f77b4", lw=2.0, zorder=3)

        st = lookup[(bin_ms, tp)]
        stars = ("***" if st["perm_p"] < 0.001 else
                 "**" if st["perm_p"] < 0.01 else
                 "*" if st["perm_p"] < 0.05 else "")
        ax.text(0.97, 0.97,
                f"slope: {slope:+.3f} RSA/day\n"
                f"p = {st['perm_p']:.3f}{(' ' + stars) if stars else ''}   "
                f"|d_z| = {abs(st['cohens_dz']):.2f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey",
                          alpha=0.9, lw=0.5))

        ax.set_xlim(gap_min, gap_max)
        ax.grid(alpha=0.3)
        if row_i == 0:
            ax.set_title(f"{bin_ms} ms bin", fontsize=11)
        if row_i == 1:
            ax.set_xlabel("Actual interval (days)", fontsize=10)
        if col_i == 0:
            ax.set_ylabel(f"{tp.capitalize()}\nCross-day RSA", fontsize=11)

handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PAIR_COLOR[p],
                      markersize=8, label=p) for p in ("A-B", "A-C", "B-C")]
handles.append(plt.Line2D([0], [0], color="#1f77b4", lw=2.5,
                          label="Within-mouse mean slope"))
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=True)
fig.suptitle("Timescale sweep: cross-day RSA vs actual interval at each bin width\n"
             "(exact sign-flip p; * p<0.05, ** p<0.01, *** p<0.001)",
             fontsize=11, y=1.01)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
out = root / "timescale_sweep_actual.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
