"""Asymmetric top-X pair persistence across days, with fold enrichment.

Cell pairs are ranked by |signal r| independently on the earlier and later
session; persistence at threshold X is

    P(X) = |top_X_early & top_X_late| / |top_X_early|

Outputs:
  <root>/top_overlap_enrichment.csv
  <root>/top_overlap_enrichment_stats.csv
  <root>/top_overlap_enrichment.png
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSIONS = ["A", "B", "C"]
PAIRS = [("A", "B", 1, "#1f77b4"), ("B", "C", 1, "#2ca02c"), ("A", "C", 2, "#ff7f0e")]
PERCENTILES = [25, 20, 15, 10, 5, 1]
FPS = 30.0
EPS = 1e-10

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--perm", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

root = Path(args.root)
rng = np.random.default_rng(args.seed)


def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def session_signal_corr(events, frames, cell_idx):
    ev = events[cell_idx, :].T
    L = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    T = np.stack([ev[int(sf): int(sf) + L, :] for sf, _ in frames], axis=0)
    psth = T.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = np.corrcoef(psth, rowvar=False)
    dead = psth.var(axis=0) <= EPS
    S[dead, :] = np.nan
    S[:, dead] = np.nan
    return S


def boot_mean_ci(values, B, alpha, rng):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    boot = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return (float(np.mean(x)),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def sign_flip_p(d, B, rng):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    obs = float(np.mean(d))
    null = (rng.choice([-1.0, 1.0], size=(B, len(d))) * d).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (B + 1))


cid_dirs = [p for p in sorted(root.iterdir())
            if p.is_dir() and p.name.isdigit()
            and all((p / s / "events_matrix.npy").exists() for s in SESSIONS)]
print(f"Containers found: {len(cid_dirs)}")

rows = []
for cid_dir in cid_dirs:
    cid = cid_dir.name
    ev = {s: np.load(cid_dir / s / "events_matrix.npy") for s in SESSIONS}
    fr = {s: np.load(cid_dir / s / "movie_repeat_frames.npy") for s in SESSIONS}
    matched = np.where(present_mask(ev["A"]) & present_mask(ev["B"])
                       & present_mask(ev["C"]))[0]
    n_cells = len(matched)
    if n_cells < 10:
        print(f"  skip {cid}: only {n_cells} matched cells")
        continue

    mats = {s: session_signal_corr(ev[s], fr[s], matched) for s in SESSIONS}
    iu = np.triu_indices(n_cells, k=1)

    for earlier, later, gap, _ in PAIRS:
        r_e, r_l = mats[earlier][iu], mats[later][iu]
        mask = np.isfinite(r_e) & np.isfinite(r_l)
        if mask.sum() < 50:
            continue
        abs_e, abs_l = np.abs(r_e[mask]), np.abs(r_l[mask])
        n_pairs = len(abs_e)
        order_e, order_l = np.argsort(abs_e), np.argsort(abs_l)

        for X in PERCENTILES:
            n_keep = max(3, int(round(n_pairs * X / 100.0)))
            overlap = len(set(order_e[-n_keep:]) & set(order_l[-n_keep:])) / n_keep
            chance = n_keep / n_pairs
            rows.append(dict(container_id=cid, n_cells=n_cells, n_pairs=n_pairs,
                             pair=f"{earlier}-{later}", days_apart=gap,
                             percentile=X, n_keep=n_keep, chance_overlap=chance,
                             overlap=overlap, fold_enrichment=overlap / chance))
    print(f"  {cid}: {n_cells} cells, {n_pairs} pairs")

df = pd.DataFrame(rows)
df.to_csv(root / "top_overlap_enrichment.csv", index=False)
print(f"Saved per-mouse table ({len(df)} rows, {df['container_id'].nunique()} mice)")

stat_rows = []
for earlier, later, gap, _ in PAIRS:
    for X in PERCENTILES:
        sub = df[(df["pair"] == f"{earlier}-{later}") & (df["percentile"] == X)]
        if sub.empty:
            continue
        m_ov, lo_ov, hi_ov = boot_mean_ci(sub["overlap"].values, args.boot, 0.05, rng)
        m_fo, lo_fo, hi_fo = boot_mean_ci(sub["fold_enrichment"].values,
                                          args.boot, 0.05, rng)
        excess = sub["overlap"].values - sub["chance_overlap"].values
        stat_rows.append(dict(
            pair=f"{earlier}-{later}", days_apart=gap, percentile=X, n_mice=len(sub),
            chance_overlap=float(sub["chance_overlap"].mean()),
            overlap_mean=m_ov, overlap_lo=lo_ov, overlap_hi=hi_ov,
            fold_mean=m_fo, fold_lo=lo_fo, fold_hi=hi_fo,
            above_chance_diff=float(excess.mean()),
            above_chance_p=sign_flip_p(excess, args.perm, rng)))

stats = pd.DataFrame(stat_rows)

paired_rows = []
for X in PERCENTILES:
    piv = (df[df["percentile"] == X]
           .pivot_table(index="container_id", columns="pair",
                        values="fold_enrichment", aggfunc="first")
           .dropna())
    if {"A-B", "B-C", "A-C"}.issubset(piv.columns):
        d = 0.5 * (piv["A-B"].values + piv["B-C"].values) - piv["A-C"].values
        mean_d = float(np.mean(d))
        if len(d) > 1:
            p = sign_flip_p(d, args.perm, rng)
            lo, hi = boot_mean_ci(d, args.boot, 0.05, rng)[1:]
        else:
            p = lo = hi = np.nan
        n = len(d)
    else:
        n = 0
        mean_d = p = lo = hi = np.nan
    paired_rows.append(dict(percentile=X, n_mice=n, mean_fold_d=mean_d,
                            ci_lo=lo, ci_hi=hi, perm_p=p))

paired = pd.DataFrame(paired_rows)
stats["row_type"] = "per_pair"
paired["row_type"] = "paired_1d_vs_2d"
pd.concat([stats, paired], ignore_index=True, sort=False).to_csv(
    root / "top_overlap_enrichment_stats.csv", index=False)

for earlier, later, gap, _ in PAIRS:
    sub = stats[stats["pair"] == f"{earlier}-{later}"]
    print(f"\n{earlier}-{later} ({gap} day{'s' if gap > 1 else ''}):")
    for _, r in sub.iterrows():
        print(f"  {int(r['percentile']):>3}%  overlap {r['overlap_mean']:.3f} "
              f"[{r['overlap_lo']:.3f}, {r['overlap_hi']:.3f}]  "
              f"chance {r['chance_overlap']:.3f}  "
              f"fold {r['fold_mean']:5.2f}x [{r['fold_lo']:5.2f}, {r['fold_hi']:5.2f}]  "
              f"p = {r['above_chance_p']:.4f}")

print("\nPaired 1d-vs-2d fold enrichment "
      "(d = mean(fold_AB, fold_BC) - fold_AC):")
for _, r in paired.iterrows():
    print(f"  {int(r['percentile']):>3}%  n = {int(r['n_mice'])}  "
          f"d = {r['mean_fold_d']:+.2f}  "
          f"95% CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  p = {r['perm_p']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

for earlier, later, gap, color in PAIRS:
    for _, g in df[df["pair"] == f"{earlier}-{later}"].groupby("container_id"):
        g = g.sort_values("percentile", ascending=False)
        axes[0].plot(g["percentile"], g["overlap"], color=color, lw=0.5,
                     alpha=0.18, zorder=1)
        axes[1].plot(g["percentile"], g["fold_enrichment"], color=color, lw=0.5,
                     alpha=0.18, zorder=1)

for earlier, later, gap, color in PAIRS:
    sub = stats[stats["pair"] == f"{earlier}-{later}"].sort_values(
        "percentile", ascending=False)
    xs = sub["percentile"].values
    label = f"{earlier}-{later} ({gap} day{'s' if gap > 1 else ''})"
    for ax, mean_col, lo_col, hi_col in [(axes[0], "overlap_mean", "overlap_lo", "overlap_hi"),
                                          (axes[1], "fold_mean", "fold_lo", "fold_hi")]:
        ax.plot(xs, sub[mean_col].values, color=color, lw=2.6, marker="o", ms=7,
                mew=0, label=label, zorder=4)
        ax.fill_between(xs, sub[lo_col].values, sub[hi_col].values, color=color,
                        alpha=0.22, zorder=2)

xs_all = np.array(PERCENTILES, dtype=float)
axes[0].plot(xs_all, xs_all / 100.0, color="black", ls="--", lw=1.2,
             label="Chance (= X/100)", zorder=3)
axes[1].axhline(1.0, color="black", ls="--", lw=1.2, label="Chance (= 1.0x)", zorder=3)

for ax in axes:
    ax.set_xticks(PERCENTILES)
    ax.set_xticklabels([f"{t}%" for t in PERCENTILES])
    ax.invert_xaxis()
    ax.set_xlabel("Threshold X% (rank pairs by |r| on the earlier day)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

axes[0].set_ylabel("Top-X overlap fraction", fontsize=11)
axes[0].set_title("Observed top-X overlap vs chance", fontsize=12)
axes[1].set_ylabel("Fold enrichment (observed / chance)", fontsize=11)
axes[1].set_title("Fold enrichment over chance", fontsize=12)
fig.suptitle("Strongest pairs over days", fontsize=12, fontweight="bold", y=1.02)

plt.tight_layout()
out = root / "top_overlap_enrichment.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out}")
