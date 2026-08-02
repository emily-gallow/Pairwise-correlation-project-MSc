"""Cohort-level cross-day Jaccard sweep across graph densities.

For each mouse, builds signal and noise correlation matrices per session on the
matched cell set at frame rate, binarises each at density d, and computes the
Jaccard overlap between session pairs. Analytic chance is d/(2-d).

Inputs:
  <root>/<cid>/{A,B,C}/events_matrix.npy
  <root>/<cid>/{A,B,C}/movie_repeat_frames.npy

Outputs:
  <root>/cross_day_jaccard_multi.csv
  <root>/cross_day_jaccard_multi.png
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 30.0
EPS = 1e-10
SESSIONS = ["A", "B", "C"]
PAIRS = [("A", "B", 1, "#a05a8a"), ("B", "C", 1, "#d63384"), ("A", "C", 2, "#ffd60a")]
PAIR_LABELS = ["A-B", "B-C", "A-C"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--n-densities", type=int, default=50)
parser.add_argument("--d-min", type=float, default=0.01)
parser.add_argument("--d-max", type=float, default=0.25)
args = parser.parse_args()

root = Path(args.root)
densities = np.linspace(args.d_min, args.d_max, args.n_densities)


def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def session_corrs(events, frames, cell_idx):
    ev = events[cell_idx, :].T
    L = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    T = np.stack([ev[int(sf): int(sf) + L, :] for sf, _ in frames], axis=0)
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

    return dict(signal=S, noise=N)


def binary_at_density(C, d):
    """Symmetric boolean matrix marking the top-d fraction of |r| pairs.
    NaN entries map to -1 so they are never selected."""
    n = C.shape[0]
    vals = np.abs(C[np.triu_indices(n, k=1)])
    valid = np.isfinite(vals)
    if not valid.any():
        return np.zeros((n, n), dtype=bool)
    thr = np.quantile(vals[valid], 1.0 - d)
    B = np.where(np.isfinite(C), np.abs(C), -1.0) >= thr
    np.fill_diagonal(B, False)
    return B


def jaccard(A_bool, B_bool):
    iu = np.triu_indices(A_bool.shape[0], k=1)
    a, b = A_bool[iu], B_bool[iu]
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else np.nan


cid_dirs = [p for p in sorted(root.iterdir())
            if p.is_dir() and p.name.isdigit()
            and all((p / s / "events_matrix.npy").exists() for s in SESSIONS)]
print(f"Containers with all 3 sessions: {len(cid_dirs)}")

rows = []
for cid_dir in cid_dirs:
    cid = cid_dir.name
    ev = {s: np.load(cid_dir / s / "events_matrix.npy") for s in SESSIONS}
    fr = {s: np.load(cid_dir / s / "movie_repeat_frames.npy") for s in SESSIONS}
    matched = np.where(present_mask(ev["A"]) & present_mask(ev["B"])
                       & present_mask(ev["C"]))[0]
    if len(matched) < 5:
        continue

    mats = {s: session_corrs(ev[s], fr[s], matched) for s in SESSIONS}

    for d in densities:
        for tp in ("signal", "noise"):
            bx = {s: binary_at_density(mats[s][tp], d) for s in SESSIONS}
            for (a, b, gap, _), pair_label in zip(PAIRS, PAIR_LABELS):
                rows.append(dict(container_id=cid, density=float(d), type=tp,
                                 pair=pair_label, days_apart=gap,
                                 jaccard=jaccard(bx[a], bx[b]),
                                 n_matched=int(len(matched))))
    print(f"  {cid}: {len(matched)} matched cells")

df = pd.DataFrame(rows)
out_csv = root / "cross_day_jaccard_multi.csv"
df.to_csv(out_csv, index=False)
print(f"Saved {out_csv} ({len(df)} rows, {df['container_id'].nunique()} containers)")

n_animals = df["container_id"].nunique()
chance = densities / (2 - densities)

# x-axis is expressed as the percentage of weakest pairs zeroed, so the strict
# end of the sweep sits on the right.
def density_to_threshold(d):
    return 100.0 * (1.0 - d)


fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

for ax, tp, title in [(axes[0], "signal", "Signal correlation"),
                      (axes[1], "noise", "Noise correlation")]:
    sub = df[df["type"] == tp]

    for (a, b, gap, color), pair_label in zip(PAIRS, PAIR_LABELS):
        for _, g in sub[sub["pair"] == pair_label].groupby("container_id"):
            g = g.sort_values("density")
            ax.plot(density_to_threshold(g["density"]), g["jaccard"],
                    color=color, lw=0.6, alpha=0.18, zorder=1)

    for (a, b, gap, color), pair_label in zip(PAIRS, PAIR_LABELS):
        per = sub[sub["pair"] == pair_label]
        agg = per.groupby("density")["jaccard"].agg(
            ["mean", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0])
        agg.columns = ["mean", "sem"]
        x_vals = density_to_threshold(agg.index.values)
        ax.plot(x_vals, agg["mean"], color=color, lw=2.6, zorder=4,
                label=f"{pair_label} ({gap} day{'s' if gap > 1 else ''})")
        ax.fill_between(x_vals, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                        color=color, alpha=0.22, zorder=3)

    ax.plot(density_to_threshold(densities), chance, color="grey", ls="--",
            lw=1.2, zorder=2, label="chance d / (2 - d)")
    ax.set_xlabel("Threshold % (% of weakest |r| pairs zeroed)", fontsize=12)
    ax.set_ylabel("Cross-day Jaccard", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(density_to_threshold(args.d_max), density_to_threshold(args.d_min))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

fig.suptitle(f"Cross-day correlation-network overlap vs density - {n_animals} mice\n"
             "Per-animal traces (light) and cohort mean +/- SEM (bold)",
             fontsize=12, y=1.02)
plt.tight_layout(rect=[0, 0.02, 1, 1])

out_png = root / "cross_day_jaccard_multi.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")

unique_densities = sorted(df["density"].unique())
for d_target in (0.05, 0.10, 0.20):
    d_actual = min(unique_densities, key=lambda x: abs(x - d_target))
    if abs(d_actual - d_target) > 0.01:
        continue
    ch = d_actual / (2 - d_actual)
    print(f"\nd = {d_actual * 100:.2f}% (chance = {ch:.3f})")
    for tp in ("signal", "noise"):
        for pair_label in PAIR_LABELS:
            sub = df[(df["type"] == tp) & (df["pair"] == pair_label)
                     & (df["density"] == d_actual)]
            if not len(sub):
                continue
            m = sub["jaccard"].mean()
            s = sub["jaccard"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0
            print(f"  {tp:6} {pair_label:>4}  {m:.3f} +/- {s:.3f}  "
                  f"(n = {len(sub)}, {m / ch:.2f}x chance)")
