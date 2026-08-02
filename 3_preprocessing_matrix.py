"""Clean and z-score the per-repeat response matrix for a session.

Active neurons are cells with non-NaN data and mean event rate above
MIN_MEAN_EVENT_RATE, determined from the full-session events matrix.

Inputs:
  <root>/{S}/X_trials_neurons.npy
  <root>/{S}/trial_metadata.csv
  <root>/{S}/events_matrix.npy

Outputs:
  <root>/{S}/X_clean.npy
  <root>/{S}/X_clean_active.npy
  <root>/{S}/active_neuron_indices.npy
  <root>/{S}/trial_metadata_clean.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MIN_MEAN_EVENT_RATE = 1e-4
SESSIONS_ALL = ["A", "B", "C"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--session", required=True, choices=["A", "B", "C", "all"])
parser.add_argument("--root", default="outputs/movie1")
args = parser.parse_args()

root = Path(args.root)
sessions = SESSIONS_ALL if args.session == "all" else [args.session]


def preprocess(session):
    d = root / session
    X = np.load(d / "X_trials_neurons.npy")
    meta = pd.read_csv(d / "trial_metadata.csv")
    events = np.load(d / "events_matrix.npy")
    n_repeats, n_neurons = X.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_rates = np.nanmean(events, axis=1)
    present = ~np.isnan(mean_rates)
    active_idx = np.where(present & (mean_rates > MIN_MEAN_EVENT_RATE))[0]

    X_z = np.full_like(X, np.nan, dtype=np.float32)
    for col in np.where(present)[0]:
        vals = X[:, col]
        sigma = np.nanstd(vals)
        X_z[:, col] = (vals - np.nanmean(vals)) / sigma if sigma > 1e-10 else 0.0

    np.save(d / "X_clean.npy", X_z)
    np.save(d / "X_clean_active.npy", X_z[:, active_idx])
    np.save(d / "active_neuron_indices.npy", active_idx)
    meta.to_csv(d / "trial_metadata_clean.csv", index=False)

    print(f"Session {session}: {n_repeats} repeats, {n_neurons} neurons, "
          f"{present.sum()} present, {len(active_idx)} active")


for s in sessions:
    preprocess(s)
