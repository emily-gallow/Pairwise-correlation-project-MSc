"""Search Allen Brain Observatory Visual Coding 2P for VISp excitatory
containers with a complete three-session set and a usable matched-cell count.

Metadata only; no NWB downloads.

Outputs:
  outputs/container_summary.csv
"""
from pathlib import Path

import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

CACHE_DIR = Path.home() / "allen_cache" / "visual_coding"
OUTPUT_DIR = Path("outputs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

EXCITATORY_CRES = ["Slc17a7-IRES2-Cre", "Emx1-IRES-Cre", "CamKII-tTA"]
PREFERRED_INDICATOR = "GCaMP6f"
TARGET_MIN, TARGET_MAX, MIN_USABLE = 100, 200, 80
N_CONTAINERS = 100

boc = BrainObservatoryCache(manifest_file=str(CACHE_DIR / "manifest.json"))

raw_containers = []
for cre in EXCITATORY_CRES:
    try:
        batch = boc.get_experiment_containers(targeted_structures=["VISp"],
                                              cre_lines=[cre])
        print(f"{cre}: {len(batch)} containers")
        raw_containers.extend(batch)
    except Exception as exc:
        print(f"{cre}: fetch failed - {exc}")

seen = set()
containers = []
for c in raw_containers:
    if c["id"] not in seen:
        seen.add(c["id"])
        containers.append(c)
print(f"Unique VISp excitatory containers: {len(containers)}")


def classify_sessions(experiments):
    """Map a container's experiments to session A/B/C ids by session_type,
    falling back to stimulus heuristics where the field is missing."""
    a = b = c = None
    for exp in experiments:
        eid = exp.get("id")
        if eid is None:
            continue
        stype = str(exp.get("session_type", "") or "").lower()

        if "three_session_a" in stype or stype.endswith("_a"):
            a = eid
        elif "three_session_b" in stype or stype.endswith("_b"):
            b = eid
        elif "three_session_c" in stype or stype.endswith("_c"):
            c = eid
        elif "scene" in stype and a is None:
            a = eid
        elif "movie" in stype:
            blob = " ".join(str(exp.get(k, "") or "") for k in
                            ["session_type", "stimulus_name", "stimulus_type",
                             "stimulus"]).lower()
            if "sparse" in blob and c is None:
                c = eid
            elif ("drifting" in blob or "grating" in blob) and b is None:
                b = eid
    return a, b, c


records = []
for container in containers[:N_CONTAINERS]:
    cid = container["id"]
    cre = container.get("cre_line", "unknown")
    depth = container.get("imaging_depth", "?")
    reporter = container.get("reporter_line", "")

    try:
        experiments = boc.get_ophys_experiments(experiment_container_ids=[cid])
        sid_a, sid_b, sid_c = classify_sessions(experiments)
        has_all = all(s is not None for s in (sid_a, sid_b, sid_c))
        exp_by_id = {e["id"]: e for e in experiments}

        def n_cells(sid):
            if sid is None:
                return None
            e = exp_by_id.get(sid, {})
            for key in ("num_cells", "cell_count", "number_of_cells"):
                if e.get(key):
                    return int(e[key])
            return 0

        n_a, n_b, n_c = n_cells(sid_a), n_cells(sid_b), n_cells(sid_c)

        # The container-level cell-specimen table is the authoritative matched
        # count; per-session metadata fields are often empty in newer manifests.
        if has_all:
            matched = len(boc.get_cell_specimens(experiment_container_ids=[cid]))
            if n_a == 0 and n_b == 0 and n_c == 0:
                n_a = n_b = n_c = matched
        else:
            matched = 0

        if not has_all:
            status = "missing-sessions"
        elif TARGET_MIN <= matched <= TARGET_MAX:
            status = ("IDEAL+GCaMP6f"
                      if PREFERRED_INDICATOR.lower() in str(reporter).lower()
                      else "IDEAL")
        elif matched >= MIN_USABLE:
            status = "usable"
        else:
            status = "too-few"

        print(f"{cid} {cre} {depth}um  nA={n_a} nB={n_b} nC={n_c}  "
              f"matched={matched}  {status}")

        records.append(dict(container_id=cid, cre_line=cre, imaging_depth=depth,
                            reporter_line=reporter, n_cells_A=n_a, n_cells_B=n_b,
                            n_cells_C=n_c, n_matched=matched,
                            has_all_sessions=has_all, session_A_id=sid_a,
                            session_B_id=sid_b, session_C_id=sid_c,
                            status=status))

    except Exception as exc:
        print(f"{cid} {cre} - ERROR: {exc}")
        records.append(dict(container_id=cid, cre_line=cre, imaging_depth=depth,
                            n_matched=0, has_all_sessions=False, status="error"))

df = pd.DataFrame(records)
priority = {"IDEAL+GCaMP6f": 0, "IDEAL": 1, "usable": 2,
            "too-few": 3, "missing-sessions": 4, "error": 5}
df["_rank"] = df["status"].map(priority).fillna(9).astype(int)
df = df.sort_values(["_rank", "n_matched"], ascending=[True, False]).drop(columns="_rank")

out_path = OUTPUT_DIR / "container_summary.csv"
df.to_csv(out_path, index=False)

ideal = df[df["status"].str.startswith("IDEAL", na=False)]
usable = df[df["status"] == "usable"]
print(f"\nChecked {len(df)} containers, {df['has_all_sessions'].sum()} complete, "
      f"{len(ideal)} ideal ({TARGET_MIN}-{TARGET_MAX} cells), {len(usable)} usable")
print(f"Saved {out_path}")
