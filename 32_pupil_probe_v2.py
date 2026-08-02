"""Probe which containers have usable eye-tracking / pupil data.

Two steps: query Allen for VISp Slc17a7 experiments flagged as eye-tracked and
find those complete across all three sessions, then test actual retrieval on
the cohort catalog via get_eye_tracking and get_ophys_pupil_data.

Outputs:
  <out-root>/pupil_probe_v2_candidate_containers.csv
  <out-root>/pupil_probe_v2_existing_cohort.csv
  <out-root>/pupil_probe_v2_report.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

SESS = {"A": {"three_session_a"},
        "B": {"three_session_b"},
        "C": {"three_session_c", "three_session_c2"}}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache-dir",
                    default=str(Path.home() / "allen_cache" / "visual_coding"))
parser.add_argument("--catalog", default="outputs/movie1/container_catalog.csv")
parser.add_argument("--out-root", default="outputs/movie1")
parser.add_argument("--skip-existing-cohort", action="store_true")
args = parser.parse_args()

out_root = Path(args.out_root)
boc = BrainObservatoryCache(
    manifest_file=str(Path(args.cache_dir) / "manifest.json"))

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


try:
    eye_tracked = boc.get_ophys_experiments(require_eye_tracking=True,
                                            targeted_structures=["VISp"],
                                            cre_lines=["Slc17a7-IRES2-Cre"],
                                            simple=True)
    log(f"VISp/Slc17a7 experiments flagged eye-tracked: {len(eye_tracked)}")
except Exception as exc:
    log(f"require_eye_tracking query failed: {exc}")
    eye_tracked = []

container_eye_tracking = {}
for e in eye_tracked:
    cid = e.get("experiment_container_id")
    if cid is None:
        continue
    stype = str(e.get("session_type", "") or "").lower()
    letter = next((k for k, names in SESS.items() if stype in names), None)
    container_eye_tracking.setdefault(cid, {})[letter or stype] = e.get("id")

complete_eye_containers = [cid for cid, s in container_eye_tracking.items()
                           if all(x in s for x in "ABC")]
log(f"Containers eye-tracked on all three sessions: {len(complete_eye_containers)}")

existing_cohort = set()
if Path(args.catalog).exists():
    cat = pd.read_csv(args.catalog)
    existing_cohort = {int(x) for x in
                       cat[cat["complete"].astype(bool)]["container_id"]}
log(f"Cohort containers: {len(existing_cohort)}")

new_eye_containers = sorted(set(complete_eye_containers) - existing_cohort)
overlap = sorted(set(complete_eye_containers) & existing_cohort)
log(f"Overlap with cohort: {len(overlap)}; new candidates: {len(new_eye_containers)}")

pd.DataFrame([
    dict(container_id=cid, in_current_cohort=(cid in existing_cohort),
         session_A_id=container_eye_tracking[cid].get("A"),
         session_B_id=container_eye_tracking[cid].get("B"),
         session_C_id=container_eye_tracking[cid].get("C"))
    for cid in sorted(complete_eye_containers)
]).to_csv(out_root / "pupil_probe_v2_candidate_containers.csv", index=False)
log(f"Saved {out_root / 'pupil_probe_v2_candidate_containers.csv'}")

if args.skip_existing_cohort:
    log("Skipping per-container retrieval probe")
else:
    cat = pd.read_csv(args.catalog)
    cat = cat[cat["complete"].astype(bool)]
    log(f"\nProbing retrieval on {len(cat)} cohort containers")

    cohort_rows = []
    for _, row in cat.iterrows():
        cid = int(row["container_id"])
        exps = boc.get_ophys_experiments(experiment_container_ids=[cid])

        for s in "ABC":
            eid = next((e["id"] for e in exps
                        if str(e.get("session_type", "") or "").lower() in SESS[s]),
                       None)
            if eid is None:
                log(f"{cid}/{s}: no experiment id")
                continue

            row_out = dict(container_id=cid, session=s, experiment_id=eid)

            try:
                arr = boc.get_eye_tracking(ophys_experiment_id=eid)
                n_finite = (int(np.isfinite(arr).all(axis=1).sum()) if arr.ndim == 2
                            else int(np.isfinite(arr).sum()))
                row_out.update(get_eye_tracking_status="OK",
                               get_eye_tracking_shape=str(arr.shape),
                               get_eye_tracking_dtype=str(arr.dtype),
                               get_eye_tracking_n_finite_rows=n_finite)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    row_out["get_eye_tracking_col0_range"] = (
                        f"[{np.nanmin(arr[:, 0]):.2f}, {np.nanmax(arr[:, 0]):.2f}]")
                    if arr.shape[1] >= 5:
                        row_out["get_eye_tracking_col4_range"] = (
                            f"[{np.nanmin(arr[:, 4]):.4f}, {np.nanmax(arr[:, 4]):.4f}]")
                log(f"{cid}/{s}: get_eye_tracking OK, shape {arr.shape}, "
                    f"{n_finite} finite rows")
            except Exception as exc:
                row_out["get_eye_tracking_status"] = f"FAIL: {type(exc).__name__}: {exc}"
                log(f"{cid}/{s}: get_eye_tracking FAIL {type(exc).__name__}")

            try:
                df_pup = boc.get_ophys_pupil_data(ophys_experiment_id=eid,
                                                  suppress_pupil_data=False)
                row_out.update(
                    get_ophys_pupil_data_status="OK" if len(df_pup) else "EMPTY",
                    get_ophys_pupil_data_rows=len(df_pup),
                    get_ophys_pupil_data_columns="|".join(df_pup.columns))
                log(f"{cid}/{s}: get_ophys_pupil_data "
                    f"{'OK' if len(df_pup) else 'EMPTY'}, {len(df_pup)} rows")
            except Exception as exc:
                row_out["get_ophys_pupil_data_status"] = (
                    f"FAIL: {type(exc).__name__}: {exc}")
                log(f"{cid}/{s}: get_ophys_pupil_data FAIL {type(exc).__name__}")

            cohort_rows.append(row_out)

    pd.DataFrame(cohort_rows).to_csv(
        out_root / "pupil_probe_v2_existing_cohort.csv", index=False)
    log(f"Saved {out_root / 'pupil_probe_v2_existing_cohort.csv'}")

if new_eye_containers:
    log(f"\nNew eye-tracking containers not in cohort ({len(new_eye_containers)}):")
    for cid in new_eye_containers[:30]:
        log(f"  {cid}")
    if len(new_eye_containers) > 30:
        log(f"  ... {len(new_eye_containers) - 30} more")

(out_root / "pupil_probe_v2_report.txt").write_text("\n".join(log_lines))
