"""Cross-day noise RSA against trial-level running and pupil mismatch.

Behavioural mismatch is computed as in script 41: the mean over the 10 x 10
trial pairs of the absolute difference in running fraction, or in pupil area
z-scored across all 30 trials of a mouse so that the three sessions share a
common scale.

Inputs:
  <root>/<cid>/{A,B,C}/events_matrix.npy
  <root>/<cid>/{A,B,C}/movie_repeat_frames.npy
  <root>/<cid>/{A,B,C}/running_speed.npy
  <root>/<cid>/{A,B,C}/pupil_size.npy

Outputs:
  <root>/noise_rsa_trial_mismatch.csv
  <root>/noise_rsa_trial_mismatch_stats.csv
  <root>/noise_rsa_trial_mismatch.png
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-10
ROOT = Path("outputs/movie1")
SESSIONS = ["A", "B", "C"]
PAIRS = [("A", "B", 1, "#a05a8a"),
         ("B", "C", 1, "#d63384"),
         ("A", "C", 2, "#ffd60a")]
RUN_THRESH = 1.0
NEEDED = ["events_matrix.npy", "movie_repeat_frames.npy", "running_speed.npy",
          "pupil_size.npy"]
PROXIES = [("running", "mean_trial_pair_running_mismatch"),
           ("pupil", "mean_trial_pair_pupil_mismatch")]
SUBSETS = ["all (1d + 2d)", "1-day only", "2-day only"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--perm", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)


def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def session_noise_corr(events, frames, cell_idx):
    ev = events[cell_idx, :].T
    L = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    T = np.stack([ev[int(sf): int(sf) + L, :] for sf, _ in frames], axis=0)
    R, nb, n = T.shape
    psth = T.mean(axis=0)
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
    return N


def per_trial_running_frac(running, sf, ef, threshold=RUN_THRESH):
    s = running[0, sf: ef + 1]
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.nan
    return float(np.mean(np.abs(s) > threshold))


def per_trial_pupil_mean(pupil, sf, ef):
    p = pupil[1, sf: ef + 1]
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return np.nan
    return float(np.mean(p))


def z_score(values):
    a = np.asarray(values, float)
    mask = np.isfinite(a)
    if mask.sum() < 2:
        return np.full_like(a, np.nan)
    s = a[mask].std(ddof=1)
    if s < EPS:
        return np.where(mask, 0.0, np.nan)
    z = np.full_like(a, np.nan)
    z[mask] = (a[mask] - a[mask].mean()) / s
    return z


def matrix_rsa(C1, C2):
    iu = np.triu_indices(C1.shape[0], k=1)
    a, b = C1[iu], C2[iu]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    a, b = a[m], b[m]
    if np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def mean_trial_pair_mismatch(x, y):
    diffs = [abs(xi - yj) for xi in x for yj in y
             if np.isfinite(xi) and np.isfinite(yj)]
    return float(np.mean(diffs)) if diffs else np.nan


def rankdata_avg(x):
    x = np.asarray(x, float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(n, float)
    r[order] = np.arange(1, n + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def spearman_rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan
    ra, rb = rankdata_avg(a[m]), rankdata_avg(b[m])
    if np.std(ra) < EPS or np.std(rb) < EPS:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_rho(sub, x_col, y_col, B, rng):
    """Resample mice with replacement, taking all rows of each chosen mouse."""
    obs = spearman_rho(sub[x_col].values, sub[y_col].values)
    mice = sub["container_id"].unique()
    rhos = np.empty(B, float)
    for b in range(B):
        sel = rng.choice(mice, size=len(mice), replace=True)
        boot = pd.concat([sub[sub["container_id"] == m] for m in sel],
                         ignore_index=True)
        rhos[b] = spearman_rho(boot[x_col].values, boot[y_col].values)
    rhos = rhos[np.isfinite(rhos)]
    if len(rhos) == 0:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def within_mouse_perm_p(sub, x_col, y_col, B, rng):
    """Shuffle y within mouse, preserving mouse-level structure."""
    obs = spearman_rho(sub[x_col].values, sub[y_col].values)
    if not np.isfinite(obs):
        return np.nan
    groups = [g[y_col].values for _, g in sub.groupby("container_id", sort=False)]
    x = sub[x_col].values
    nulls = np.array([spearman_rho(x, np.concatenate(
        [rng.permutation(g) for g in groups])) for _ in range(B)])
    nulls = nulls[np.isfinite(nulls)]
    return float((np.sum(np.abs(nulls) >= abs(obs)) + 1) / (len(nulls) + 1))


rows = []
for cid_dir in sorted(ROOT.iterdir()):
    if not (cid_dir.is_dir() and cid_dir.name.isdigit()):
        continue
    if not all((cid_dir / s / f).exists() for s in SESSIONS for f in NEEDED):
        continue

    ev = {s: np.load(cid_dir / s / "events_matrix.npy") for s in SESSIONS}
    fr = {s: np.load(cid_dir / s / "movie_repeat_frames.npy") for s in SESSIONS}
    rn = {s: np.load(cid_dir / s / "running_speed.npy") for s in SESSIONS}
    pu = {s: np.load(cid_dir / s / "pupil_size.npy") for s in SESSIONS}

    matched = np.where(present_mask(ev["A"]) & present_mask(ev["B"])
                       & present_mask(ev["C"]))[0]
    if len(matched) < 10:
        continue
    if any(len(fr[s]) != 10 for s in SESSIONS):
        continue

    mats = {s: session_noise_corr(ev[s], fr[s], matched) for s in SESSIONS}
    trial_runs = {s: [per_trial_running_frac(rn[s], int(sf), int(ef))
                      for sf, ef in fr[s]] for s in SESSIONS}
    trial_pupil = {s: [per_trial_pupil_mean(pu[s], int(sf), int(ef))
                       for sf, ef in fr[s]] for s in SESSIONS}

    # Pupil area has no absolute scale, so z-score across the mouse's 30 trials.
    z_all = z_score(trial_pupil["A"] + trial_pupil["B"] + trial_pupil["C"])
    trial_pupil_z = {"A": list(z_all[:10]), "B": list(z_all[10:20]),
                     "C": list(z_all[20:30])}

    for s1, s2, gap, _ in PAIRS:
        rows.append(dict(
            container_id=cid_dir.name, pair=f"{s1}-{s2}", days_apart=gap,
            noise_rsa=matrix_rsa(mats[s1], mats[s2]),
            mean_trial_pair_running_mismatch=mean_trial_pair_mismatch(
                trial_runs[s1], trial_runs[s2]),
            mean_trial_pair_pupil_mismatch=mean_trial_pair_mismatch(
                trial_pupil_z[s1], trial_pupil_z[s2])))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "noise_rsa_trial_mismatch.csv", index=False)
ana = df.dropna(subset=["noise_rsa", "mean_trial_pair_running_mismatch",
                        "mean_trial_pair_pupil_mismatch"]).copy()
print(f"Saved noise_rsa_trial_mismatch.csv "
      f"({len(ana)} rows, {ana['container_id'].nunique()} mice)")

stat_rows = []
for proxy_label, proxy_col in PROXIES:
    for label in SUBSETS:
        sub = ana if label == SUBSETS[0] else ana[
            ana["days_apart"] == (1 if label == SUBSETS[1] else 2)]
        obs, lo, hi = bootstrap_rho(sub, proxy_col, "noise_rsa", args.boot, rng)
        p = within_mouse_perm_p(sub, proxy_col, "noise_rsa", args.perm, rng)
        print(f"{proxy_label:>8} {label:>16}: n = {len(sub):>2}  "
              f"rho = {obs:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  perm p = {p:.4f}")
        stat_rows.append(dict(proxy=proxy_label, subset=label, n=len(sub),
                              rho=obs, ci_lo=lo, ci_hi=hi, perm_p=p))

pd.DataFrame(stat_rows).to_csv(ROOT / "noise_rsa_trial_mismatch_stats.csv",
                               index=False)


def fmt_p(p):
    if not np.isfinite(p):
        return "n.s."
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))

panels = [
    ("mean_trial_pair_running_mismatch", "Running mismatch",
     "Mean trial-pair running mismatch\n(|fraction time running > 1 cm/s, "
     "day X - day Y|, mean over 100 trial-pairs)", axes[0], stat_rows[0:3]),
    ("mean_trial_pair_pupil_mismatch", "Pupil mismatch",
     "Mean trial-pair pupil mismatch\n(|z-scored mean pupil area, "
     "day X - day Y|, mean over 100 trial-pairs)", axes[1], stat_rows[3:6]),
]

for x_col, panel_title, x_label, ax, triplet in panels:
    for s1, s2, gap, color in PAIRS:
        sub = df[df["pair"] == f"{s1}-{s2}"].dropna(subset=["noise_rsa", x_col])
        if sub.empty:
            continue
        x, y = sub[x_col].values, sub["noise_rsa"].values
        ax.scatter(x, y, s=80, color=color, alpha=0.85, edgecolors="black",
                   linewidth=0.6, zorder=3,
                   label=f"{s1}-{s2}  ({gap} day{'s' if gap > 1 else ''})")
        if len(x) >= 3 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, slope * xx + intercept, color=color, lw=1.5, alpha=0.65,
                    ls="--", zorder=2)

    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.set_xlabel(x_label, fontsize=10.5)
    ax.set_ylabel("Session-level cross-day NOISE RSA\n(10-trial PSTH-based)",
                  fontsize=10.5)
    ax.set_title(panel_title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    ann = "\n".join(
        ["Per-mouse-resampled Spearman rho:"] +
        [f"  {name:<12} rho = {s['rho']:+.3f}  "
         f"CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]  {fmt_p(s['perm_p'])}"
         for name, s in zip(["All pairs:", "1-day only:", "2-day only:"], triplet)])
    ax.text(0.98, 0.02, ann, transform=ax.transAxes, fontsize=8.5, va="bottom",
            ha="right", family="monospace",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.92,
                      boxstyle="round,pad=0.4"))

fig.suptitle(f"Cross-day NOISE RSA vs trial-level behavioural mismatch - running "
             f"(left) and pupil (right)  ({ana['container_id'].nunique()} mice, "
             f"{len(ana)} session-pair points per panel)\n10-trial PSTH-based "
             f"z-scored trial-residual noise correlation matrix; same x-axis and "
             f"recipe as the signal RSA companion.", fontsize=11, y=1.02)

plt.tight_layout()
fig.savefig(ROOT / "noise_rsa_trial_mismatch.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved noise_rsa_trial_mismatch.png")
