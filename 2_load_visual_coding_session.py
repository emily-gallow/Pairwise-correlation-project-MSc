"""Load one Allen Visual Coding 2P container and extract the Natural Movie 1
data needed downstream.

Uses the published L0-deconvolved events rather than dF/F: the GCaMP6f decay
kernel autocorrelates dF/F at roughly 0.95 per frame, which would masquerade as
fast coupling in the frame-rate noise correlations. A sanity gate rejects any
signal that is substantially negative, since that indicates dF/F was loaded.

Cells are aligned to the container-level cell-specimen ordering, so a given row
refers to the same neuron in every session; cells absent from a session are NaN.

Outputs (per session):
  <root>/<container>/<S>/events_matrix.npy
  <root>/<container>/<S>/X_trials_neurons.npy
  <root>/<container>/<S>/movie_repeat_frames.npy
  <root>/<container>/<S>/trial_metadata.csv
  <root>/<container>/<S>/cell_info.csv
  <root>/<container>/<S>/running_speed.npy
  <root>/<container>/<S>/pupil_size.npy or pupil_unavailable.flag
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

DEFAULT_CONTAINER = 661437138
CACHE_DIR = Path.home() / "allen_cache" / "visual_coding"
BASE_OUTPUT = Path("outputs") / "movie1"
STIMULUS_NAME = "natural_movie_one"

SESSION_A_TYPES = {"three_session_a"}
SESSION_B_TYPES = {"three_session_b"}
SESSION_C_TYPES = {"three_session_c", "three_session_c2"}

KEEP_COLS = ["cell_specimen_id", "area", "imaging_depth",
             "rf_center_on_x_lsn", "rf_center_on_y_lsn",
             "reliability_nm1", "peak_dff_nm1"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--session", required=True, choices=["A", "B", "C", "all"])
parser.add_argument("--container", type=int, default=DEFAULT_CONTAINER)
args = parser.parse_args()

CONTAINER_ID = args.container
sessions_to_run = ["A", "B", "C"] if args.session == "all" else [args.session]
CONTAINER_OUTPUT = BASE_OUTPUT / str(CONTAINER_ID)
CONTAINER_OUTPUT.mkdir(parents=True, exist_ok=True)

boc = BrainObservatoryCache(manifest_file=str(CACHE_DIR / "manifest.json"))

container_exps = boc.get_ophys_experiments(experiment_container_ids=[CONTAINER_ID])
SESSIONS = {}
for e in container_exps:
    stype = str(e.get("session_type", "") or "").lower().strip()
    if stype in SESSION_A_TYPES:
        SESSIONS["A"] = e["id"]
    elif stype in SESSION_B_TYPES:
        SESSIONS["B"] = e["id"]
    elif stype in SESSION_C_TYPES:
        SESSIONS["C"] = e["id"]

missing = [k for k in ("A", "B", "C") if k not in SESSIONS]
if missing:
    raise RuntimeError(f"Container {CONTAINER_ID}: missing sessions {missing}; "
                       f"found {[str(e.get('session_type')) for e in container_exps]}")
print(f"Container {CONTAINER_ID}: A={SESSIONS['A']} B={SESSIONS['B']} C={SESSIONS['C']}")


def load_one_session(session_label):
    experiment_id = SESSIONS[session_label]
    out_dir = CONTAINER_OUTPUT / session_label
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = boc.get_ophys_experiment_data(experiment_id)

    # Timestamps and cell ordering come from the dF/F API; the events array
    # shares that time base and cell-specimen ordering.
    timestamps, _ = dataset.get_dff_traces()
    events = boc.get_ophys_experiment_events(experiment_id)
    if events.shape[1] != timestamps.shape[0]:
        n = min(events.shape[1], timestamps.shape[0])
        events, timestamps = events[:, :n], timestamps[:n]

    neg_frac = float(np.mean(events < 0))
    if neg_frac > 0.01:
        raise ValueError(f"Experiment {experiment_id}: {100 * neg_frac:.1f}% "
                         f"negative values - this is dF/F, not L0 events")

    stim_table = dataset.get_stimulus_table(STIMULUS_NAME)
    if "repeat" not in stim_table.columns:
        raise ValueError(f"Expected 'repeat' column; got {list(stim_table.columns)}")

    per_repeat = (stim_table.groupby("repeat")
                  .agg(start_frame=("start", "min"), end_frame=("end", "max"))
                  .reset_index().sort_values("repeat").reset_index(drop=True))
    per_repeat["start_time"] = [float(timestamps[int(f)])
                                for f in per_repeat["start_frame"]]
    per_repeat["end_time"] = [float(timestamps[min(int(f), len(timestamps) - 1)])
                              for f in per_repeat["end_frame"]]
    per_repeat["duration_s"] = per_repeat["end_time"] - per_repeat["start_time"]
    n_repeats = len(per_repeat)

    cell_table = pd.DataFrame(
        boc.get_cell_specimens(experiment_container_ids=[CONTAINER_ID]))
    container_ids = list(cell_table["cell_specimen_id"])
    n_matched = len(container_ids)
    session_ids = list(dataset.get_cell_specimen_ids())
    id_to_pos = {cid: i for i, cid in enumerate(session_ids)}
    present_mask = np.array([cid in id_to_pos for cid in container_ids])
    session_row = np.array([id_to_pos.get(cid, -1) for cid in container_ids])

    def align_to_container(session_matrix):
        out = np.full((n_matched,) + session_matrix.shape[1:], np.nan,
                      dtype=np.float32)
        for c_pos, s_row in enumerate(session_row):
            if s_row >= 0:
                out[c_pos] = session_matrix[s_row]
        return out

    run_speed, run_timestamps = dataset.get_running_speed()

    def mean_run_speed(t0, t1):
        m = (run_timestamps >= t0) & (run_timestamps < t1)
        return float(np.mean(run_speed[m])) if m.sum() else np.nan

    X_session = np.full((n_repeats, len(session_ids)), np.nan, dtype=np.float32)
    movie_repeat_frames = np.zeros((n_repeats, 2), dtype=np.int64)
    drop_trial = np.zeros(n_repeats, dtype=bool)
    meta_rows = []

    for i, row in per_repeat.iterrows():
        sf, ef = int(row["start_frame"]), int(row["end_frame"])
        movie_repeat_frames[i] = [sf, ef]
        # end_frame is inclusive, hence the +1 in both the count and the slice.
        n_frames_in = ef - sf + 1
        if n_frames_in <= 0:
            drop_trial[i] = True
        else:
            X_session[i, :] = events[:, sf:ef + 1].mean(axis=1).astype(np.float32)

        meta_rows.append({
            "trial_id": i,
            "repeat_index": int(row["repeat"]),
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "duration_s": float(row["duration_s"]),
            "n_frames": n_frames_in,
            "mean_running_speed": mean_run_speed(float(row["start_time"]),
                                                 float(row["end_time"])),
        })

    if drop_trial.sum():
        X_session = X_session[~drop_trial]
        meta_rows = [r for r, d in zip(meta_rows, drop_trial) if not d]
        movie_repeat_frames = movie_repeat_frames[~drop_trial]

    X = align_to_container(X_session.T).T
    np.save(out_dir / "X_trials_neurons.npy", X)
    np.save(out_dir / "movie_repeat_frames.npy", movie_repeat_frames)
    pd.DataFrame(meta_rows).to_csv(out_dir / "trial_metadata.csv", index=False)

   
    keep = [c for c in KEEP_COLS if c in cell_table.columns]
    cell_info = (cell_table[keep].set_index("cell_specimen_id")
                 .reindex(container_ids).reset_index()
                 .rename(columns={"rf_center_on_x_lsn": "x",
                                  "rf_center_on_y_lsn": "y"}))
    cell_info[f"present_in_session_{session_label.lower()}"] = present_mask
    cell_info.to_csv(out_dir / "cell_info.csv", index=False)

    events_aligned = align_to_container(events)
    np.save(out_dir / "events_matrix.npy", events_aligned)
    np.save(out_dir / "running_speed.npy",
            np.stack([run_speed, run_timestamps], axis=0))

    # Pupil tracking is only present where eye tracking did not fail; script 32b
    # recovers it for sessions that land here without it.
    pupil_saved = False
    for getter_name in ("get_pupil_size", "get_pupil_location"):
        if hasattr(dataset, getter_name):
            try:
                ts_p, val_p = getattr(dataset, getter_name)()
                arr = (np.stack([ts_p, val_p], axis=0) if val_p.ndim == 1
                       else np.concatenate([ts_p[None, :], val_p], axis=0))
                np.save(out_dir / f"{getter_name.replace('get_', '')}.npy", arr)
                pupil_saved = True
            except Exception as exc:
                print(f"  {getter_name} unavailable: {exc}")
    if not pupil_saved:
        (out_dir / "pupil_unavailable.flag").touch()

    n_present = int(present_mask.sum())
    print(f"Session {session_label}: {len(meta_rows)} repeats, {n_present} present, "
          f"{n_matched - n_present} absent, {events.shape[1]} timepoints")

    return dict(session=session_label, n_repeats=len(meta_rows),
                n_present=n_present, n_absent=n_matched - n_present,
                n_timepoints=events.shape[1])


for s in sessions_to_run:
    load_one_session(s)

print(f"Outputs in {CONTAINER_OUTPUT.resolve()}")
