"""Session-level cross-day signal RSA against trial-level running mismatch.

Uses the 10-trial PSTH signal correlation matrices (the same estimator as the
headline cross-day RSA) rather than single-trial matrices, which are too noisy
at ~900 frames per trial to support a trial-pair regression.

Per mouse and session pair:
  y = Pearson r between the upper-triangle vectors of the two session-level
      signal correlation matrices on the matched cell set
  x = mean over the 10 x 10 trial pairs of |running fraction day X trial i
      - running fraction day Y trial j|, where running fraction is the
      proportion of trial frames with |speed| > 1 cm/s

Inputs:
  <root>/<cid>/{A,B,C}/events_matrix.npy
  <root>/<cid>/{A,B,C}/movie_repeat_frames.npy
  <root>/<cid>/{A,B,C}/running_speed.npy

Outputs:
  <root>/session_rsa_trial_mismatch.csv
  <root>/session_rsa_trial_mismatch_stats.csv
  <root>/session_rsa_trial_mismatch.png
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
NEEDED = ["events_matrix.npy", "movie_repeat_frames.npy", "running_speed.npy"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--perm", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
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


def per_trial_running_frac(running, sf, ef, threshold=RUN_THRESH):
    s = running[0, sf: ef + 1]
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.nan
    return float(np.mean(np.abs(s) > threshold))


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


def mean_trial_pair_mismatch(run_X, run_Y):
    diffs = [abs(ri - rj) for ri in run_X for rj in run_Y
             if np.isfinite(ri) and np.isfinite(rj)]
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


def bootstrap_rho(sub_df, x_col, y_col, mouse_col, B, rng):
    """Resample mice with replacement, taking all rows of each chosen mouse."""
    obs = spearman_rho(sub_df[x_col].values, sub_df[y_col].values)
    mice = sub_df[mouse_col].unique()
    rhos = np.empty(B, float)
    for b in range(B):
        sel = rng.choice(mice, size=len(mice), replace=True)
        boot = pd.concat([sub_df[sub_df[mouse_col] == m] for m in sel],
                         ignore_index=True)
        rhos[b] = spearman_rho(boot[x_col].values, boot[y_col].values)
    rhos = rhos[np.isfinite(rhos)]
    if len(rhos) == 0:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def within_mouse_perm_p(sub_df, x_col, y_col, mouse_col, B, rng):
    """Shuffle y within mouse, preserving mouse-level structure."""
    obs = spearman_rho(sub_df[x_col].values, sub_df[y_col].values)
    if not np.isfinite(obs):
        return np.nan
    groups = [g[y_col].values for _, g in sub_df.groupby(mouse_col, sort=False)]
    x = sub_df[x_col].values
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

    matched = np.where(present_mask(ev["A"]) & present_mask(ev["B"])
                       & present_mask(ev["C"]))[0]
    if len(matched) < 10:
        continue

    mats = {s: session_signal_corr(ev[s], fr[s], matched) for s in SESSIONS}

    if any(len(fr[s]) != 10 for s in SESSIONS):
        continue
    trial_runs = {s: [per_trial_running_frac(rn[s], int(sf), int(ef))
                      for sf, ef in fr[s]] for s in SESSIONS}

    for s1, s2, gap, _ in PAIRS:
        rows.append(dict(
            container_id=cid_dir.name, pair=f"{s1}-{s2}", days_apart=gap,
            session_rsa=matrix_rsa(mats[s1], mats[s2]),
            mean_trial_pair_running_mismatch=mean_trial_pair_mismatch(
                trial_runs[s1], trial_runs[s2])))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "session_rsa_trial_mismatch.csv", index=False)
print(f"Saved session_rsa_trial_mismatch.csv "
      f"({len(df)} rows, {df['container_id'].nunique()} mice)")

ana = df.dropna(subset=["session_rsa", "mean_trial_pair_running_mismatch"]).copy()

stat_rows = []
for label, sub in [("all (1d + 2d)", ana),
                   ("1-day only", ana[ana["days_apart"] == 1]),
                   ("2-day only", ana[ana["days_apart"] == 2])]:
    obs, lo, hi = bootstrap_rho(sub, "mean_trial_pair_running_mismatch",
                                "session_rsa", "container_id", args.boot, rng)
    p = within_mouse_perm_p(sub, "mean_trial_pair_running_mismatch",
                            "session_rsa", "container_id", args.perm, rng)
    print(f"{label:>16}: n = {len(sub):>2}  rho = {obs:+.3f}  "
          f"CI [{lo:+.3f}, {hi:+.3f}]  perm p = {p:.4f}")
    stat_rows.append(dict(subset=label, n=len(sub), rho=obs, ci_lo=lo,
                          ci_hi=hi, perm_p=p))

pd.DataFrame(stat_rows).to_csv(ROOT / "session_rsa_trial_mismatch_stats.csv",
                               index=False)
s_all, s_1d, s_2d = stat_rows

fig, ax = plt.subplots(figsize=(10.0, 6.2))

for s1, s2, gap, color in PAIRS:
    sub = df[df["pair"] == f"{s1}-{s2}"].dropna(
        subset=["session_rsa", "mean_trial_pair_running_mismatch"])
    if sub.empty:
        continue
    x = sub["mean_trial_pair_running_mismatch"].values
    y = sub["session_rsa"].values
    ax.scatter(x, y, s=85, color=color, alpha=0.85, edgecolors="black",
               linewidth=0.7, zorder=3,
               label=f"{s1}-{s2}  ({gap} day{'s' if gap > 1 else ''})")
    if len(x) >= 3 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, color=color, lw=1.6, alpha=0.65,
                ls="--", zorder=2)

ax.axhline(0, color="black", lw=0.5, zorder=1)
ax.set_xlabel("Mean trial-pair running mismatch\n(mean over 100 trial-pairs of "
              "|fraction time running > 1 cm/s, day X - day Y|)", fontsize=11)
ax.set_ylabel("Session-level cross-day RSA\n(10-trial PSTH-based)", fontsize=11)
ax.set_title(f"Session-level cross-day RSA vs trial-level running mismatch "
             f"({df['container_id'].nunique()} mice, {len(ana)} session-pair "
             f"points)\n10-trial PSTH matrices; x summarises the 10x10 "
             f"trial-pair running-mismatch matrix per session pair.", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=10, frameon=True)


def fmt_p(p):
    if not np.isfinite(p):
        return "n.s."
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


ann = "\n".join(
    ["Per-mouse-resampled Spearman rho:"] +
    [f"  {name:<20} rho = {s['rho']:+.3f}  "
     f"CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]  {fmt_p(s['perm_p'])}"
     for name, s in [("All pairs (1d + 2d):", s_all), ("1-day only:", s_1d),
                     ("2-day only:", s_2d)]])
ax.text(0.98, 0.02, ann, transform=ax.transAxes, fontsize=9, va="bottom",
        ha="right", family="monospace",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.92,
                  boxstyle="round,pad=0.4"))

plt.tight_layout()
fig.savefig(ROOT / "session_rsa_trial_mismatch.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved session_rsa_trial_mismatch.png")
