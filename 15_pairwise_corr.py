"""Per-session signal and noise correlation matrices for natural movie 1.

The default bin width is one acquisition frame (~33 ms at 30 Hz). Natural
movie 1 changes continuously, so each frame is effectively its own stimulus
condition; 

All matrices are computed on the cells present in all three sessions, so a
given row refers to the same neuron in every figure.

Inputs:
  <root>/{S}/events_matrix.npy
  <root>/{S}/movie_repeat_frames.npy

Outputs:
  <root>/{S}/signal_corr.npy
  <root>/{S}/noise_corr.npy
  <root>/{S}/corr_cell_index.npy
  <root>/pairwise_corr_summary.csv
  <root>/pairwise_corr_matrices.png
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
FPS = 30.0
EPS = 1e-10

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--bin-ms", type=float, default=33.0)
parser.add_argument("--sweep", action="store_true",
                    help="Report noise-correlation structure across bin widths")
parser.add_argument("--clim", type=float, default=None)
parser.add_argument("--noise-clim", type=float, default=None)
parser.add_argument("--clim-pct", type=float, default=99.0)
args = parser.parse_args()

root = Path(args.root)
bin_frames = max(1, int(round(args.bin_ms / 1000.0 * FPS)))


def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def bin_epoch(epoch, bf):
    L, n_cells = epoch.shape
    nb = L // bf
    return epoch[:nb * bf].reshape(nb, bf, n_cells).sum(axis=1)


def build_tensor(ev_cols, frames, bf):
    L = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    nb = L // bf
    if nb < 2:
        raise ValueError(f"Only {nb} bins per repeat at this bin width")
    T = np.stack([bin_epoch(ev_cols[int(sf): int(sf) + L, :], bf)
                  for sf, _ in frames], axis=0)
    return T, nb, L


def corr_with_dead_mask(samples):
    dead = samples.var(axis=0) <= EPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        C = np.corrcoef(samples, rowvar=False)
    C[dead, :] = np.nan
    C[:, dead] = np.nan
    return C, dead


def offdiag(C):
    v = C[np.triu_indices(C.shape[0], k=1)]
    return v[~np.isnan(v)]


def eig_order(C):
    """Order cells by the leading eigenvector of |C| so co-correlated cells
    sit together in the heatmaps."""
    M = np.nan_to_num(C, nan=0.0).copy()
    np.fill_diagonal(M, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, V = np.linalg.eigh(M)
    return np.argsort(V[:, np.argmax(w)])


def session_corrs(events, frames, bf, cell_idx=None):
    idx = np.where(present_mask(events))[0] if cell_idx is None else np.asarray(cell_idx)
    ev_cols = events[idx, :].T
    n_cells = ev_cols.shape[1]

    T, n_bins, L = build_tensor(ev_cols, frames, bf)
    R = T.shape[0]

    psth = T.mean(axis=0)
    signal, _ = corr_with_dead_mask(psth)

    sd = T.std(axis=0)
    nz = sd > EPS
    z = ((T - psth[None]) / np.where(nz, sd, 1.0)[None]) * nz[None]
    noise, dead_noise = corr_with_dead_mask(z.reshape(R * n_bins, n_cells))

    return dict(signal=signal, noise=noise, idx=idx, n_p=n_cells, R=R,
                n_bins=n_bins, L_common=L,
                frac_zero_var=float(np.mean(~nz)),
                frac_dead=float(np.mean(dead_noise)),
                mean_event=float(np.nanmean(T)))


def auto_clim(matrices, percentile):
    """Percentile of |off-diagonal r| pooled across matrices. Rounded to 3
    decimals below 0.1 so small noise limits keep useful precision."""
    pooled = np.concatenate([offdiag(np.abs(C)) for C in matrices])
    val = float(np.percentile(pooled, percentile))
    return float(np.round(val, 3 if val < 0.1 else 2))


present_per = {s: present_mask(np.load(root / s / "events_matrix.npy"))
               for s in SESSIONS}
shared_idx = np.where(present_per["A"] & present_per["B"] & present_per["C"])[0]
print(f"Matched cells across all 3 sessions: {len(shared_idx)}")

results, summary, loaded = {}, [], {}
for s in SESSIONS:
    sd = root / s
    events = np.load(sd / "events_matrix.npy")
    frames = np.load(sd / "movie_repeat_frames.npy")
    loaded[s] = (events, frames)

    res = session_corrs(events, frames, bin_frames, cell_idx=shared_idx)
    signal, noise = res["signal"], res["noise"]

    np.save(sd / "signal_corr.npy", signal)
    np.save(sd / "noise_corr.npy", noise)
    np.save(sd / "corr_cell_index.npy", res["idx"])

    iu = np.triu_indices(signal.shape[0], k=1)
    sflat, nflat = signal[iu], noise[iu]
    good = ~np.isnan(sflat) & ~np.isnan(nflat)
    sn_r = (float(np.corrcoef(sflat[good], nflat[good])[0, 1])
            if good.sum() > 2 else np.nan)

    sig_od, noi_od = offdiag(signal), offdiag(noise)
    summary.append(dict(
        session=s, n_present_cells=int(res["n_p"]), n_repeats=int(res["R"]),
        n_bins=int(res["n_bins"]), bin_ms=args.bin_ms, bin_frames=int(bin_frames),
        repeat_len_frames=int(res["L_common"]), n_valid_pairs=int(good.sum()),
        mean_signal_corr=float(np.mean(sig_od)) if len(sig_od) else np.nan,
        mean_noise_corr=float(np.mean(noi_od)) if len(noi_od) else np.nan,
        median_noise_corr=float(np.median(noi_od)) if len(noi_od) else np.nan,
        std_noise_corr=float(np.std(noi_od)) if len(noi_od) else np.nan,
        frac_noise_pos=float(np.mean(noi_od > 0)) if len(noi_od) else np.nan,
        signal_noise_pair_r=sn_r,
        mean_event_per_bin=res["mean_event"],
        frac_zero_var_bins=res["frac_zero_var"],
        frac_dead_cells=res["frac_dead"]))

    results[s] = dict(signal=signal, noise=noise,
                      sflat=sflat[good], nflat=nflat[good])

    print(f"Session {s}: {res['n_p']} cells, {res['R']} repeats x {res['n_bins']} bins, "
          f"mean signal r = {np.mean(sig_od):+.3f}, mean noise r = {np.mean(noi_od):+.3f}, "
          f"signal-vs-noise r = {sn_r:+.3f}, "
          f"zero-variance (bin,cell) = {100 * res['frac_zero_var']:.1f}%")

df = pd.DataFrame(summary)
df.to_csv(root / "pairwise_corr_summary.csv", index=False)
print(f"Saved {root / 'pairwise_corr_summary.csv'}")

if args.sweep:
    sweep_frames = sorted({max(1, int(round(m / 1000.0 * FPS)))
                           for m in (33.0, 100.0, 250.0, 500.0)})
    print("\nTimescale sweep (struct r = correlation of the noise-correlation "
          "vector against frame resolution)")
    for s in SESSIONS:
        events, frames = loaded[s]
        ref_vec, iu = None, None
        for bf in sweep_frames:
            res = session_corrs(events, frames, bf)
            C = res["noise"]
            if iu is None:
                iu = np.triu_indices(C.shape[0], k=1)
            vec = C[iu]
            if ref_vec is None:
                ref_vec = vec
            both = ~np.isnan(vec) & ~np.isnan(ref_vec)
            struct_r = (float(np.corrcoef(vec[both], ref_vec[both])[0, 1])
                        if both.sum() > 2 else np.nan)
            print(f"  {s}  {bf / FPS * 1000:>4.0f} ms  "
                  f"mean noise r {np.nanmean(vec):+.3f}  "
                  f"zero-var {100 * res['frac_zero_var']:>5.1f}%  "
                  f"struct r {struct_r:.3f}")

sig_mats = [results[s]["signal"] for s in SESSIONS]
noi_mats = [results[s]["noise"] for s in SESSIONS]
clim = args.clim if args.clim is not None else auto_clim(sig_mats, args.clim_pct)
noise_clim = (args.noise_clim if args.noise_clim is not None
              else auto_clim(noi_mats, args.clim_pct))
print(f"Colour limits ({args.clim_pct:g}th percentile of |r|): "
      f"signal +/-{clim:g}, noise +/-{noise_clim:g}")

all_vals = np.concatenate([results[s][k] for s in SESSIONS for k in ("sflat", "nflat")])
scatter_lim = float(np.nanmax(np.abs(all_vals))) * 1.05

fig, axes = plt.subplots(3, 3, figsize=(15, 14))

for i, s in enumerate(SESSIONS):
    order = eig_order(results[s]["signal"])

    for j, (C, name, cl) in enumerate([(results[s]["signal"], "signal", clim),
                                       (results[s]["noise"], "noise", noise_clim)]):
        ax = axes[i, j]
        im = ax.imshow(C[np.ix_(order, order)], vmin=-cl, vmax=cl, cmap="RdBu_r",
                       interpolation="nearest", aspect="equal")
        ax.set_title(f"Session {s} — {name} correlation (±{cl:g})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=10)

    ax = axes[i, 2]
    ax.scatter(results[s]["sflat"], results[s]["nflat"], s=4, alpha=0.25,
               color="steelblue", edgecolors="none")
    r = df.loc[df["session"] == s, "signal_noise_pair_r"].iloc[0]
    ax.set_title(f"Session {s} — noise vs signal corr\n(pair r = {r:+.3f})", fontsize=11)
    ax.set_xlabel("signal correlation", fontsize=10)
    ax.set_ylabel("noise correlation", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(-scatter_lim, scatter_lim)
    ax.set_ylim(-scatter_lim, scatter_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.plot([-scatter_lim, scatter_lim], [-scatter_lim, scatter_lim],
            color="black", ls="--", lw=0.7, alpha=0.4)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.grid(alpha=0.2)

for j, letter in enumerate(("A.", "B.", "C.")):
    axes[0, j].text(-0.12, 1.18, letter, transform=axes[0, j].transAxes,
                    fontsize=16, fontweight="bold", ha="left", va="bottom")

plt.tight_layout()
out = root / "pairwise_corr_matrices.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
