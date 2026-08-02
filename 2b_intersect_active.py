"""Intersect active cells across sessions A, B and C.

Inputs:
  <root>/{A,B,C}/active_neuron_indices.npy

Outputs:
  <root>/active_neuron_indices_intersect.npy
  <root>/active_summary.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SESSIONS = ["A", "B", "C"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1")
args = parser.parse_args()

root = Path(args.root)

active = {}
for s in SESSIONS:
    path = root / s / "active_neuron_indices.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    active[s] = set(int(i) for i in np.load(path))
    print(f"Session {s}: {len(active[s])} active cells")

intersect = sorted(set.intersection(*active.values()))
union = sorted(set.union(*active.values()))
print(f"Intersection: {len(intersect)} cells")
print(f"Union: {len(union)} cells")

table = pd.DataFrame({"cell_row": union})
for s in SESSIONS:
    table[f"active_in_{s}"] = table["cell_row"].isin(active[s])
table["active_in_all"] = table[[f"active_in_{s}" for s in SESSIONS]].all(axis=1)

np.save(root / "active_neuron_indices_intersect.npy",
        np.array(intersect, dtype=np.int64))
table.to_csv(root / "active_summary.csv", index=False)

print(f"Saved {root / 'active_neuron_indices_intersect.npy'}")
print(f"Saved {root / 'active_summary.csv'}")
