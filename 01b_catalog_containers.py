"""Catalog every Visual Coding 2P container with a complete three-session set.

Natural Movie 1 is presented in all three session types, so any complete
container provides the common stimulus across days. Cells are pre-matched
within a container by cell_specimen_id.

Metadata only, apart from one cached download of the cell-specimen table used
for the matched-cell count (skipped with --no-cell-count).

Outputs:
  outputs/movie1/container_catalog.csv
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

CACHE_DIR = Path.home() / "allen_cache" / "visual_coding"
BASE_OUTPUT = Path("outputs") / "movie1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

ORIGINAL_CONTAINER = 661437138
SESSION_A_TYPES = {"three_session_a"}
SESSION_B_TYPES = {"three_session_b"}
SESSION_C_TYPES = {"three_session_c", "three_session_c2"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--areas", nargs="*", default=None)
parser.add_argument("--cre", nargs="*", default=None)
parser.add_argument("--no-cell-count", action="store_true")
parser.add_argument("--require-pupil", action="store_true")
parser.add_argument("--min-matched-cells", type=int, default=0)
args = parser.parse_args()

boc = BrainObservatoryCache(manifest_file=str(CACHE_DIR / "manifest.json"))

cont_kwargs = {}
if args.areas:
    cont_kwargs["targeted_structures"] = args.areas
if args.cre:
    cont_kwargs["cre_lines"] = args.cre

containers = boc.get_experiment_containers(**cont_kwargs)
cont_meta = {c["id"]: c for c in containers}
valid_cids = set(cont_meta)
print(f"Containers returned: {len(valid_cids)}")

by_cont = defaultdict(dict)
eye_fail = {}
for e in boc.get_ophys_experiments():
    cid = e.get("experiment_container_id")
    if cid not in valid_cids:
        continue
    stype = str(e.get("session_type", "") or "").lower().strip()
    if stype:
        by_cont[cid][stype] = e["id"]
    # Missing fail_eye_tracking is treated as unknown, i.e. permissive.
    eye_fail[e["id"]] = bool(e.get("fail_eye_tracking", False))


def resolve_sessions(stype_map):
    a = next((eid for st, eid in stype_map.items() if st in SESSION_A_TYPES), None)
    b = next((eid for st, eid in stype_map.items() if st in SESSION_B_TYPES), None)
    c = c_type = None
    for st, eid in stype_map.items():
        if st in SESSION_C_TYPES:
            c, c_type = eid, st
            break
    return a, b, c, c_type


def all_have_pupil(*eids):
    return all(eid is not None and not eye_fail.get(eid, False) for eid in eids)


matched_per_cont = {}
if not args.no_cell_count:
    cells_df = pd.DataFrame(boc.get_cell_specimens())
    if "experiment_container_id" in cells_df.columns:
        matched_per_cont = (cells_df.groupby("experiment_container_id")
                            ["cell_specimen_id"].nunique().to_dict())
        print(f"Matched-cell counts for {len(matched_per_cont)} containers")
    else:
        print("WARNING: experiment_container_id absent from cell-specimen table")

rows = []
for cid in sorted(valid_cids):
    a, b, c, c_type = resolve_sessions(by_cont.get(cid, {}))
    complete = all(x is not None for x in (a, b, c))
    meta = cont_meta[cid]
    rows.append(dict(
        container_id=cid,
        cre_line=meta.get("cre_line", "unknown"),
        area=meta.get("targeted_structure", "unknown"),
        imaging_depth=meta.get("imaging_depth", -1),
        reporter_line=meta.get("reporter_line", ""),
        session_A_id=a, session_B_id=b, session_C_id=c, session_C_type=c_type,
        n_matched_cells=int(matched_per_cont.get(cid, -1)),
        has_pupil_tracking=bool(complete and all_have_pupil(a, b, c)),
        is_original=(cid == ORIGINAL_CONTAINER),
        complete=complete,
    ))

df = pd.DataFrame(rows)
complete_df = df[df["complete"]].copy()

if args.require_pupil:
    complete_df = complete_df[complete_df["has_pupil_tracking"]].copy()
    print(f"--require-pupil: {len(complete_df)} containers remaining")
if args.min_matched_cells > 0:
    complete_df = complete_df[
        complete_df["n_matched_cells"] >= args.min_matched_cells].copy()
    print(f"--min-matched-cells {args.min_matched_cells}: "
          f"{len(complete_df)} containers remaining")

complete_df = complete_df.sort_values(["n_matched_cells", "container_id"],
                                      ascending=[False, True])

out_path = BASE_OUTPUT / "container_catalog.csv"
pd.concat([complete_df, df[~df["complete"]].sort_values("container_id")],
          ignore_index=True).to_csv(out_path, index=False)

print(f"\nExamined {len(df)} containers, {len(complete_df)} complete")
if len(complete_df) and not args.no_cell_count:
    mc = complete_df.loc[complete_df["n_matched_cells"] >= 0, "n_matched_cells"]
    if len(mc):
        print(f"Matched cells: min {mc.min()}, median {int(mc.median())}, "
              f"max {mc.max()}; {int((mc >= 80).sum())} containers with >=80")
print(f"Saved {out_path}")
