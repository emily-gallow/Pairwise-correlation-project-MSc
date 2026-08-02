"""Fetch raw pupil-area traces and resample them onto the imaging timebase.

Recovers pupil data for containers where the NWB pupil getters used in script
02 failed but eye-gaze mappings are available through
get_ophys_pupil_data(suppress_pupil_data=False).

Blinks and dropouts are non-finite samples and are filled by linear
interpolation; duplicate timestamps are averaged; the cleaned trace is then
linearly interpolated onto the two-photon frame timestamps.

Outputs:
  <out-root>/<container>/<session>/pupil_size.npy
  <out-root>/pupil_size_fetch_status.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SES = {"A": {"three_session_a"},
       "B": {"three_session_b"},
       "C": {"three_session_c", "three_session_c2"}}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",
                        default=str(Path.home() / "allen_cache" / "visual_coding"))
    parser.add_argument("--catalog", default="outputs/movie1/container_catalog.csv")
    parser.add_argument("--out-root", default="outputs/movie1")
    parser.add_argument("--container", type=int, nargs="*")
    parser.add_argument("--sessions", nargs="*", choices=["A", "B", "C"],
                        default=["A", "B", "C"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_series(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("expected 1D sequence")
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.full_like(x, np.nan)
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def uniq_sorted(t, y):
    """Sort by timestamp and average any duplicated timestamps."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) != len(y):
        raise ValueError("timestamps and values must be same length")
    order = np.argsort(t)
    t, y = t[order], y[order]
    if len(t) == 0:
        return t, y
    mask = np.ones(len(t), dtype=bool)
    mask[1:] = t[1:] != t[:-1]
    if mask.all():
        return t, y

    uniq_t, uniq_y, i = [], [], 0
    while i < len(t):
        j = i + 1
        while j < len(t) and t[j] == t[i]:
            j += 1
        uniq_t.append(t[i])
        uniq_y.append(np.nanmean(y[i:j]))
        i = j
    return np.asarray(uniq_t), np.asarray(uniq_y)


def resample_pupil(event_ts, pupil_ts, pupil_area):
    if len(pupil_ts) == 0 or len(pupil_area) == 0:
        return np.full(len(event_ts), np.nan, dtype=float)
    pupil_area = clean_series(pupil_area)
    pupil_ts, pupil_area = uniq_sorted(pupil_ts, pupil_area)
    if len(pupil_ts) == 0:
        return np.full(len(event_ts), np.nan, dtype=float)
    if not np.all(np.diff(pupil_ts) > 0):
        pupil_ts, pupil_area = uniq_sorted(pupil_ts, pupil_area)
    if len(pupil_ts) == 1:
        return np.full(len(event_ts), pupil_area[0], dtype=float)
    return np.interp(event_ts, pupil_ts, pupil_area)


def resolve_session_ids(boc, container_id):
    session_ids = {}
    for exp in boc.get_ophys_experiments(experiment_container_ids=[container_id]):
        st = str(exp.get("session_type", "") or "").lower().strip()
        for label, names in SES.items():
            if st in names:
                session_ids[label] = exp["id"]
                break
    return session_ids


def process_session(boc, container_id, session_label, out_root, overwrite):
    base = dict(container_id=container_id, session=session_label)
    session_dir = Path(out_root) / str(container_id) / session_label

    if not session_dir.exists():
        return {**base, "status": "missing_session_dir"}
    events_path = session_dir / "events_matrix.npy"
    if not events_path.exists():
        return {**base, "status": "missing_events_matrix"}
    pupil_path = session_dir / "pupil_size.npy"
    if pupil_path.exists() and not overwrite:
        return {**base, "status": "skipped_exists"}

    eid = resolve_session_ids(boc, container_id).get(session_label)
    if eid is None:
        return {**base, "status": "missing_experiment_id"}
    base["experiment_id"] = eid

    try:
        ds = boc.get_ophys_experiment_data(eid)
    except Exception as exc:
        return {**base, "status": "failed_load_dataset", "notes": str(exc)}

    event_ts, _ = ds.get_dff_traces()
    T = np.load(events_path).shape[1]
    if len(event_ts) != T:
        old_len = len(event_ts)
        event_ts = event_ts[:T]
        if len(event_ts) != T:
            return {**base, "status": "timestamp_length_mismatch",
                    "notes": f"dff timestamps {old_len} vs events {T}"}

    try:
        df = boc.get_ophys_pupil_data(ophys_experiment_id=eid,
                                      suppress_pupil_data=False)
    except Exception as exc:
        return {**base, "status": "failed_get_ophys_pupil_data", "notes": str(exc)}

    if "raw_pupil_area" not in df.columns:
        return {**base, "status": "missing_raw_pupil_area",
                "notes": f"columns={list(df.columns)}"}

    pupil_ts = np.asarray(df.index, dtype=float)
    pupil_area = np.asarray(df["raw_pupil_area"].values, dtype=float)
    if pupil_ts.ndim != 1 or pupil_area.ndim != 1:
        return {**base, "status": "bad_pupil_shape",
                "notes": f"ts {pupil_ts.shape}, area {pupil_area.shape}"}

    np.save(pupil_path,
            np.vstack([event_ts, resample_pupil(event_ts, pupil_ts, pupil_area)]))
    flag_path = session_dir / "pupil_unavailable.flag"
    if flag_path.exists():
        flag_path.unlink()

    return {**base, "status": "saved", "n_pupil_rows": len(pupil_area),
            "n_event_frames": T}


def main():
    args = parse_args()
    out_root = Path(args.out_root)

    if not Path(args.catalog).exists():
        raise SystemExit(f"Catalog not found: {args.catalog}")
    catalog = pd.read_csv(args.catalog)
    if "complete" in catalog.columns:
        catalog = catalog[catalog["complete"].astype(bool)]
    if "has_pupil_tracking" in catalog.columns:
        catalog = catalog[catalog["has_pupil_tracking"].astype(bool)]
    container_ids = sorted(catalog["container_id"].astype(int).unique())

    if args.container:
        selected = [int(c) for c in args.container]
        missing = set(selected) - set(container_ids)
        if missing:
            print(f"Not in viable catalog set: {sorted(missing)}")
        container_ids = [c for c in selected if c in container_ids]

    if not container_ids:
        raise SystemExit("No containers selected.")

    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    boc = BrainObservatoryCache(
        manifest_file=str(Path(args.cache_dir) / "manifest.json"))
    print(f"Processing {len(container_ids)} containers: {container_ids}")

    rows = []
    for container_id in container_ids:
        for session_label in args.sessions:
            row = process_session(boc, container_id, session_label, out_root,
                                  args.overwrite)
            rows.append(row)
            notes = f" ({row['notes']})" if row.get("notes") else ""
            print(f"{container_id}/{session_label}: {row['status']}{notes}")

    out_csv = out_root / "pupil_size_fetch_status.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
