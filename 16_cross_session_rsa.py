"""Second-order RSA: is pairwise correlation structure preserved across days?

session, cells tracked across all three session, compute
signal and noise correlation matrices, and correlates their upper-triangle
vectors between sessions.


Inputs:
  <root>/{S}/events_matrix.npy
  <root>/{S}/movie_repeat_frames.npy

Outputs:
  <root>/cross_session_rsa.csv
  <root>/cross_session_rsa.png
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
PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]
FPS = 30.0
EPS = 1e-10

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--bin-ms", type=float, default=33.0)
parser.add_argument("--metric", choices=["pearson", "spearman"], default="pearson")
args = parser.parse_args()

root = Path(args.root)
bin_frames = max(1, int(round(args.bin_ms / 1000.0 * FPS)))


def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def bin_epoch(epoch, bf):
    L, n = epoch.shape
    nb = L // bf
    return epoch[: nb * bf].reshape(nb, bf, n).sum(axis=1)


def build_tensor(ev_cols, frames, bf):
    L = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    return np.stack([bin_epoch(ev_cols[int(sf): int(sf) + L, :], bf)
                     for sf, _ in frames], axis=0)


def corr_dead(samples):
    dead = samples.var(axis=0) <= EPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        C = np.corrcoef(samples, rowvar=False)
    C[dead, :] = np.nan
    C[:, dead] = np.nan
    return C


def signal_corr(T):
    return corr_dead(T.mean(axis=0))


def noise_corr(T):
    psth = T.mean(axis=0)
    sd = T.std(axis=0)
    nz = sd > EPS
    z = (T - psth[None]) / np.where(nz, sd, 1.0)[None]
    z = z * nz[None]
    return corr_dead(z.reshape(T.shape[0] * T.shape[1], T.shape[2]))


def offdiag(C):
    return C[np.triu_indices(C.shape[0], k=1)]


def similarity(vi, vj):
    good = ~np.isnan(vi) & ~np.isnan(vj)
    if good.sum() < 3:
        return np.nan
    a, b = vi[good], vj[good]
    if args.metric == "spearman":
        a, b = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(a, b)[0, 1])


events, frames, present = {}, {}, {}
for s in SESSIONS:
    ev = np.load(root / s / "events_matrix.npy")
    flat = ev[~np.isnan(ev)]
    if np.mean(flat < 0) > 0.01:
        raise ValueError(f"Session {s}: events_matrix is "
                         f"{100 * np.mean(flat < 0):.1f}% negative — dF/F, not L0 events")
    events[s] = ev
    frames[s] = np.load(root / s / "movie_repeat_frames.npy")
    present[s] = present_mask(ev)

shared = np.where(present["A"] & present["B"] & present["C"])[0]
print(f"Matched cells across all 3 sessions: {len(shared)}")

sig, noi, sig_half, noi_half = {}, {}, {}, {}
for s in SESSIONS:
    T = build_tensor(events[s][shared, :].T, frames[s], bin_frames)
    h = T.shape[0] // 2
    sig[s] = signal_corr(T)
    noi[s] = noise_corr(T)
    sig_half[s] = (signal_corr(T[:h]), signal_corr(T[h:]))
    noi_half[s] = (noise_corr(T[:h]), noise_corr(T[h:]))
    print(f"Session {s}: {T.shape[0]} repeats x {T.shape[1]} bins")


def rsa_matched(halves):
    M = np.full((3, 3), np.nan)
    for i, si in enumerate(SESSIONS):
        h1i, h2i = halves[si]
        M[i, i] = similarity(offdiag(h1i), offdiag(h2i))
        for j, sj in enumerate(SESSIONS):
            if j <= i:
                continue
            h1j, h2j = halves[sj]
            combos = [similarity(offdiag(a), offdiag(b))
                      for a in (h1i, h2i) for b in (h1j, h2j)]
            M[i, j] = M[j, i] = float(np.nanmean(combos))
    return M


def rsa_fulldata(corrs):
    M = np.eye(3)
    for i, si in enumerate(SESSIONS):
        for j, sj in enumerate(SESSIONS):
            if j > i:
                M[i, j] = M[j, i] = similarity(offdiag(corrs[si]), offdiag(corrs[sj]))
    return M


rsa_sig, rsa_noi = rsa_matched(sig_half), rsa_matched(noi_half)
full_sig, full_noi = rsa_fulldata(sig), rsa_fulldata(noi)
IDX = {s: k for k, s in enumerate(SESSIONS)}

rows = []
for label, M, F in [("signal", rsa_sig, full_sig), ("noise", rsa_noi, full_noi)]:
    cross = [M[0, 1], M[1, 2], M[0, 2]]
    rel = [M[0, 0], M[1, 1], M[2, 2]]
    full_cross = [F[0, 1], F[1, 2], F[0, 2]]
    ratio = np.nanmean(cross) / np.nanmean(rel) if np.nanmean(rel) else np.nan
    print(f"{label}: cross-day {np.nanmean(cross):+.3f}, ceiling {np.nanmean(rel):+.3f}, "
          f"ratio {ratio:.2f}, full-data cross-day {np.nanmean(full_cross):+.3f}")
    rows.append(dict(type=label, AB=M[0, 1], BC=M[1, 2], AC=M[0, 2],
                     rel_A=M[0, 0], rel_B=M[1, 1], rel_C=M[2, 2],
                     mean_cross_day_matched=np.nanmean(cross),
                     mean_ceiling=np.nanmean(rel), cross_over_ceiling=ratio,
                     AB_fulldata=F[0, 1], BC_fulldata=F[1, 2], AC_fulldata=F[0, 2],
                     mean_cross_day_fulldata=np.nanmean(full_cross)))

pd.DataFrame(rows).to_csv(root / "cross_session_rsa.csv", index=False)
print(f"Saved {root / 'cross_session_rsa.csv'}")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for ri, (label, halves, M, color) in enumerate(
        [("signal", sig_half, rsa_sig, "#1f77b4"),
         ("noise", noi_half, rsa_noi, "#d62728")]):

    for ci, (a, b) in enumerate(PAIRS):
        ax = axes[ri, ci]
        vi, vj = offdiag(halves[a][0]), offdiag(halves[b][0])
        good = ~np.isnan(vi) & ~np.isnan(vj)
        ax.scatter(vi[good], vj[good], s=6, alpha=0.25, color=color, edgecolors="none")
        lim = [min(vi[good].min(), vj[good].min()), max(vi[good].max(), vj[good].max())]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="grey", lw=0.4)
        ax.axvline(0, color="grey", lw=0.4)
        ax.set_title(f"{label}: {a} vs {b}   r = {M[IDX[a], IDX[b]]:+.2f}", fontsize=9)
        ax.set_xlabel(f"{label} weight — {a} h1", fontsize=8)
        ax.set_ylabel(f"{label} weight — {b} h1", fontsize=8)
        ax.grid(alpha=0.2)

    ax = axes[ri, 3]
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(SESSIONS)
    ax.set_yticklabels(SESSIONS)
    cross = np.nanmean([M[0, 1], M[1, 2], M[0, 2]])
    rel = np.nanmean([M[0, 0], M[1, 1], M[2, 2]])
    ax.set_title(f"{label} RSA (5-rep)\ndiag = ceiling, cross/ceiling = {cross / rel:.2f}",
                 fontsize=9)
    for i in range(3):
        for j in range(3):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                        color="black" if abs(M[i, j]) < 0.6 else "white",
                        fontweight="bold" if i == j else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

fig.suptitle(f"Cross-day stability of correlation structure — {len(shared)} matched "
             f"cells, bin {args.bin_ms:.0f} ms", fontsize=12, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = root / "cross_session_rsa.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
