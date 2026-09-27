"""Expert per-condition population means (3 stimuli), overall + projection-resolved.

Expert sessions have three fixed-probability stimuli ('100%','50%','0%'). This
computes, per session, the trial-averaged baseline-subtracted dF/F per stimulus
(cond_avg), averages over neurons (all SNR-filtered, and split by striatum
projection status), and pools across sessions.

Preprocessing: SNR-filtered cells (cellID where filtered_cells==1, matching the
TDR/population convention used for the reversal population); per-trial pre-stim
baseline subtraction over [-1, 0) s; average over trials.

Outputs (results/transfer/data/population/):
  population_means_expert.csv                 stimulus, frame, time_s, pop_mean_pooled,
                                              sem_over_sessions, n_neurons_total, n_sessions
  population_means_expert_by_projection.csv   + projection_status column
  population_means_expert_meta.json

Per-session cond_avg cached under $TRANSFER_DECODE_CACHE/expert_pop (resumable).
"""

from __future__ import annotations

import gc
import glob
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sources import LOADER_ROOT, loader_pkl_path, snr_session_dir, DATA_ROOT, require  # noqa: E402
from _loader_pkl import load_loader  # noqa: E402

OUT = DATA_ROOT / "population"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("TRANSFER_DECODE_CACHE", "/tmp/transfer_decode_cache")) / "expert_pop"
CACHE.mkdir(parents=True, exist_ok=True)
_REPO_ROOT = Path(__file__).resolve().parents[3]

STIMS = ["100%", "50%", "0%"]
STIM_LABEL = {"100%": "100", "50%": "50", "0%": "0"}
FS = 15.0
T_AX = (np.arange(90) - 45) / FS            # -3.0 .. 2.933, onset frame 45 = 0
BASE_MASK = (T_AX >= -1.0) & (T_AX < 0.0)   # pre-stim baseline window
GROUPS = ["projecting", "nonprojecting"]


def _proj_group(row):
    if row.get("is_projecting", 0) == 1:
        return "projecting"
    if row.get("is_nonprojecting", 0) == 1:
        return "nonprojecting"
    return "unclassified"


def compute_session(name):
    """Return dict: cond_avg per stim (n_cells,90), and projection group per cell."""
    cf = CACHE / f"{name}.pkl"
    if cf.exists():
        return pickle.load(open(cf, "rb"))
    csvs = glob.glob(f"{snr_session_dir(name)}/*_snr_filtered_cells.csv")
    if not csvs:
        return None
    proj = pd.read_csv(csvs[0])
    snr_ids = proj.loc[proj["filtered_cells"] == 1.0, "cellID"].astype(int).values  # 0-based
    try:
        L = load_loader(loader_pkl_path(name))
    except FileNotFoundError:
        return None
    dff = L.dff_stim_aligned
    if any(s not in dff for s in STIMS):
        return None
    out = {"cond_avg": {}, "n_cells": len(snr_ids)}
    for s in STIMS:
        arr = np.asarray(dff[s])[snr_ids]                # (n_snr, 90, n_trials)
        base = arr[:, BASE_MASK, :].mean(axis=1, keepdims=True)
        out["cond_avg"][s] = (arr - base).mean(axis=2)   # (n_snr, 90)
    del L, dff
    gc.collect()
    groups = proj.set_index("cellID").loc[snr_ids].apply(_proj_group, axis=1).values
    out["groups"] = np.asarray(groups)
    pickle.dump(out, open(cf, "wb"))
    return out


def _aggregate(session_results, cell_filter=None):
    """Pool over sessions. cell_filter(groups)->bool mask, or None for all cells."""
    sum_pooled = {s: np.zeros(90) for s in STIMS}
    n_cells = {s: 0 for s in STIMS}
    sess_means = {s: [] for s in STIMS}
    for res in session_results:
        if res is None:
            continue
        mask = np.ones(res["n_cells"], bool) if cell_filter is None else cell_filter(res["groups"])
        if mask.sum() == 0:
            continue
        for s in STIMS:
            ca = res["cond_avg"][s][mask]          # (n_sel, 90)
            sum_pooled[s] += ca.sum(axis=0)
            n_cells[s] += ca.shape[0]
            sess_means[s].append(ca.mean(axis=0))
    rows = []
    for s in STIMS:
        if n_cells[s] == 0:
            continue
        pooled = sum_pooled[s] / n_cells[s]
        stack = np.stack(sess_means[s], axis=0)
        sem = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else np.zeros(90)
        rows.append((s, pooled, sem, n_cells[s], stack.shape[0]))
    return rows


def main():
    require(LOADER_ROOT, "Loader pkl root")
    names = json.loads((_REPO_ROOT / "metadata" / "SAT_good_expert_sessions.json").read_text())["sessions"]
    results = []
    for name in names:
        try:
            r = compute_session(name)
        except Exception as e:
            print(f"  ! {name}: {e}"); r = None
        results.append(r)
        print(f"  {'✓' if r else '!'} {name}", flush=True)

    # overall
    rows = []
    for s, pooled, sem, n, ns in _aggregate(results):
        for ti in range(90):
            rows.append((STIM_LABEL[s], ti, float(T_AX[ti]), float(pooled[ti]),
                         float(sem[ti]), n, ns))
    pd.DataFrame(rows, columns=["stimulus", "frame", "time_s", "pop_mean_pooled",
                                "sem_over_sessions", "n_neurons_total", "n_sessions"]
                 ).to_csv(OUT / "population_means_expert.csv", index=False)

    # by projection
    prows = []
    for g in GROUPS:
        for s, pooled, sem, n, ns in _aggregate(results, lambda gr, g=g: gr == g):
            for ti in range(90):
                prows.append((STIM_LABEL[s], g, ti, float(T_AX[ti]), float(pooled[ti]),
                              float(sem[ti]), n, ns))
    pd.DataFrame(prows, columns=["stimulus", "projection_status", "frame", "time_s",
                                 "pop_mean_pooled", "sem_over_sessions",
                                 "n_neurons_total", "n_sessions"]
                 ).to_csv(OUT / "population_means_expert_by_projection.csv", index=False)

    used = [n for n, r in zip(names, results) if r]
    meta = {
        "analysis": "Expert per-condition population means (3 stimuli), + projection-resolved",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stimuli": STIM_LABEL,
        "preprocessing": "SNR-filtered cells (filtered_cells==1); per-trial baseline "
                         "subtract over [-1,0)s; average over trials",
        "n_sessions": len(used),
        "sessions": used,
    }
    (OUT / "population_means_expert_meta.json").write_text(json.dumps(meta, indent=2))
    print("Expert population extraction complete →", OUT)


if __name__ == "__main__":
    main()
