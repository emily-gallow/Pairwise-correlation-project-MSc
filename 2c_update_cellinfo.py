"""Regenerate cell_info.csv for each session with natural-movie-1 per-cell stats.

Queries Allen's cell specimens table; writes cell_info.csv in place.

Outputs:
  outputs/movie1/{A,B,C}/cell_info.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

CONTAINER_ID = 661437138
SESSIONS = {"A": 661437140, "B": 662351346, "C": 662358233}
CACHE_DIR = Path.home() / "allen_cache" / "visual_coding"
BASE_OUTPUT = Path("outputs") / "movie1"

KEEP_COLS = ["cell_specimen_id", "area", "imaging_depth",
             "rf_center_on_x_lsn", "rf_center_on_y_lsn",
             "reliability_nm1", "peak_dff_nm1"]

boc = BrainObservatoryCache(manifest_file=str(CACHE_DIR / "manifest.json"))

cell_table = pd.DataFrame(
    boc.get_cell_specimens(experiment_container_ids=[CONTAINER_ID]))
container_ids = list(cell_table["cell_specimen_id"])
print(f"Container cells: {len(container_ids)}")

missing = [c for c in KEEP_COLS if c not in cell_table.columns]
if missing:
    print(f"Missing from Allen table, omitted: {missing}")
keep = [c for c in KEEP_COLS if c in cell_table.columns]

cell_info_base = (cell_table[keep]
                  .set_index("cell_specimen_id")
                  .reindex(container_ids)
                  .reset_index()
                  .rename(columns={"rf_center_on_x_lsn": "x",
                                   "rf_center_on_y_lsn": "y"}))

for label, exp_id in SESSIONS.items():
    session_dir = BASE_OUTPUT / label
    if not session_dir.exists():
        print(f"Session {label}: {session_dir} not found, skipping")
        continue

    session_ids = set(boc.get_ophys_experiment_data(exp_id).get_cell_specimen_ids())
    present = np.array([cid in session_ids for cid in container_ids])

    ci = cell_info_base.copy()
    ci[f"present_in_session_{label.lower()}"] = present
    ci.to_csv(session_dir / "cell_info.csv", index=False)
    print(f"Session {label}: {present.sum()}/{len(container_ids)} cells present, "
          f"wrote {session_dir / 'cell_info.csv'}")
