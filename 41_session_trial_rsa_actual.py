"""Behavioural mismatch against cross-day SIGNAL RSA, adjusted for the actual
inter-session interval.

    signal_RSA_ij = b0 + b1 * mismatch_ij + b2 * actual_gap_ij + u_i + e_ij

b1 is reported with its MixedLM Wald CI and p-value, a mouse-level cluster
bootstrap CI, and a within-mouse permutation p-value; see behaviour_common.

Inputs:
  <root>/session_rsa_trial_mismatch.csv

Outputs:
  <root>/session_rsa_trial_mismatch_actual_stats.csv
  <root>/session_rsa_trial_mismatch_actual.png
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from behaviour_common import (PAIR_COLOR, PAIR_ORDER, PROXIES, centred_name,
                              load_with_actual_gaps, analyse_proxy)

RSA_COL = "signal_rsa"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--date-table", default="outputs/movie1/session_date_table.csv")
parser.add_argument("--boot", type=int, default=20000)
parser.add_argument("--perm", type=int, default=20000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

root = Path(args.root)
df = load_with_actual_gaps(root / "session_rsa_trial_mismatch.csv",
                           args.date_table, RSA_COL)
print(f"Loaded {len(df)} session pairs from {df['container_id'].nunique()} mice")

results, lines = [], {}
for proxy_col, proxy_label, _ in PROXIES:
    print(f"\n{proxy_label.upper()}")
    sub = df.dropna(subset=[proxy_col]).copy()
    row, line = analyse_proxy(sub, RSA_COL, proxy_col, proxy_label,
                              args.boot, args.perm, args.seed)
    results.append(row)
    lines[proxy_label] = line

pd.DataFrame(results).to_csv(
    root / "session_rsa_trial_mismatch_actual_stats.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
for ax, (proxy_col, proxy_label, panel_title) in zip(axes, PROXIES):
    sub = df.dropna(subset=[proxy_col])
    row = next(r for r in results if r["proxy"] == proxy_label)
    x_range, y_line, y_lo, y_hi = lines[proxy_label]

    for pair in PAIR_ORDER:
        ps = sub[sub["pair"] == pair]
        ax.scatter(ps[centred_name(proxy_col)], ps[RSA_COL], s=60,
                   color=PAIR_COLOR[pair], alpha=0.85, edgecolors="none",
                   zorder=2, label=pair)

    ax.fill_between(x_range, y_lo, y_hi, color="black", alpha=0.18, zorder=3,
                    label="95 % CI band (cluster bootstrap)")
    ax.plot(x_range, y_line, "-", color="black", lw=2.5, zorder=4,
            label=f"Adjusted slope β₁ = {row['beta1_beh']:+.4f}")

    ax.set_xlabel(f"Within-mouse centred {proxy_label} mismatch", fontsize=12)
    ax.set_ylabel("Cross-day signal RSA (Pearson r)", fontsize=12)
    ax.set_title(panel_title, fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

fig.suptitle("Cross-day signal RSA vs behavioural mismatch", fontsize=13, y=1.02)
plt.tight_layout()
out = root / "session_rsa_trial_mismatch_actual.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out}")
