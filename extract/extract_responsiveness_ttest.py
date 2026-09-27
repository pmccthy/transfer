"""Significant-responder extraction via the temporal t-test criterion.

Uses the repo module ``analysis/responsiveness_ttest.py`` (ported from
``analysis/26_08_26_responsiveness_t_test.ipynb``): for each neuron and each
stimulus, an independent t-test at every timepoint compares the stimulus window
(0-2 s) against the pre-stimulus baseline (-2-0 s); the neuron is a *significant
responder* to that stimulus when its longest tolerated-contiguous run of
significant timepoints exceeds ``n_timepoints / 3`` (gaps up to 3 frames
tolerated). This is computed *per stimulus* (a neuron can respond to more than
one), so the outputs are long/tidy.

Data source is each session's loader ``dff_stim_aligned`` (per condition,
(n_cells, n_time, n_trials)) — the same source the selectivity/heatmap/population
extractors use — restricted to the SNR-filtered cells so neuron ids line up with
the rest of the transfer bundle (reversal: TDR ``filtered_cell_indices``;
expert: ``filtered_cells == 1``). Reversal uses the TDR per-session ``t_ax``;
expert reconstructs the time axis from ``fs_image`` with a 3 s pre-stim pad.

Definition knobs (change here, re-run; every Subgroups figure reads these CSVs):
    ALPHA, MAX_INT, FRAC, EPOCH windows, SCALE.

Outputs (results/transfer/data/subgroups/):
  responsiveness_ttest_reversal_long.csv  session, neuron_id, phase, stimulus, sig, effect, resp
  responsiveness_ttest_expert_long.csv    session, neuron_id, stimulus, sig, effect, resp
  responsiveness_ttest_meta.json
    sig    = significant responder (temporal t-test criterion)
    effect = z-scored stim-baseline effect (per-neuron scaled; used for sig)
    resp   = avg stimulus response in raw dF/F (what the heatmaps display)
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
from _sources import (TDR_DIR, LOADER_ROOT, loader_pkl_path, snr_session_dir,  # noqa: E402
                      DATA_ROOT, require)
from _loader_pkl import load_loader  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "analysis"))
import responsiveness_ttest as RT  # noqa: E402

OUT = DATA_ROOT / "subgroups"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("TRANSFER_DECODE_CACHE", "/tmp/transfer_decode_cache")) / "responsiveness_ttest_v2"
CACHE.mkdir(parents=True, exist_ok=True)

# --- responsiveness definition (see notebook) --------------------------------
ALPHA = 0.05
MAX_INT = 3            # tolerate gaps up to 3 frames within a "contiguous" run
FRAC = 1.0 / 3.0       # run must exceed n_timepoints * FRAC
SCALE = True           # z-score each neuron across the trial window (comparable effect sizes)
MULTIPLE_TEST = "fdr_bh"   # Benjamini-Hochberg correction of the per-timepoint p-values,
                            # WITHIN each neuron, across its n_timepoints simultaneous tests
                            # (t_test_temporal's own multiple_test arg -- previously left at
                            # its default None/uncorrected here). None reproduces the old
                            # uncorrected behaviour. NOTE: this corrects across TIMEPOINTS
                            # per neuron; it does not by itself correct across NEURONS (a
                            # separate axis -- see model_responders.py / the chat for the
                            # analogous model-side fix, which does correct across units).
BASE_WINDOW_S = (-2.0, 0.0)   # pre-stimulus baseline
STIM_WINDOW_S = (0.0, 2.0)    # stimulus epoch
PRE_STIM_S_EXPERT = 3.0       # loader pad when no TDR t_ax is available (expert)

REV_STIMS = ["100%-->0%", "50%", "0%-->100%"]
REV_LABELS = ["100_to_0", "50", "0_to_100"]
EXP_STIMS = ["100%", "50%", "0%"]
EXP_LABELS = ["100", "50", "0"]


def _win(t_ax, lo, hi):
    return (t_ax >= lo) & (t_ax < hi)


def _responders(arr, t_ax, snr_idx):
    """arr: (n_cells, n_time, n_trials) for one condition. Returns
    (sig, effect, resp) over the SNR-filtered cells (length len(snr_idx)):
      sig     significant responder (temporal t-test criterion)
      effect  z-scored stim-baseline effect (per-neuron scaled; comparable)
      resp    avg stimulus response = trial-averaged mean(stim) - mean(baseline)
              in raw dF/F (NOT z-scored) -- what the heatmaps display."""
    bm, sm = _win(t_ax, *BASE_WINDOW_S), _win(t_ax, *STIM_WINDOW_S)
    sub = np.asarray(arr)[snr_idx]                       # (n_snr, n_time, n_trials)
    # avg stimulus response in raw dF/F (mean over stim window & trials, baseline-subtracted)
    resp = (sub[:, sm, :].mean(axis=1) - sub[:, bm, :].mean(axis=1)).mean(axis=1)  # (n_snr,)
    # temporal t-test responder detection (per-neuron scaled internally)
    baseline = np.transpose(sub[:, bm, :], (2, 1, 0))    # -> (trials, time, neurons)
    stim = np.transpose(sub[:, sm, :], (2, 1, 0))
    res = RT.responders_from_windows(baseline, stim, alpha=ALPHA, max_int=MAX_INT,
                                     frac=FRAC, scale=SCALE, multiple_test=MULTIPLE_TEST)
    return res["sig"], res["effect"], np.asarray(resp)


def _reversal_session(name):
    cf = CACHE / f"rev_{name}.pkl"
    if cf.exists():
        return pickle.load(open(cf, "rb"))
    tdr = pickle.load(open(TDR_DIR / f"{name}_tdr_v2.pkl", "rb"))
    t_ax = np.asarray(tdr["t_ax"])
    snr = np.asarray(tdr["filtered_cell_indices"]).astype(int)
    L = load_loader(loader_pkl_path(name))
    dff = L.dff_stim_aligned
    rows = []  # (neuron_id, phase, stim_label, sig, effect)
    for stim, lab in zip(REV_STIMS, REV_LABELS):
        for phase, key in [("pre", f"{stim} (pre-Rev)"), ("post", f"{stim} (post-Rev)")]:
            if key not in dff:
                del L, dff
                gc.collect()
                return None
            arr = np.asarray(dff[key])
            if arr.ndim != 3 or arr.shape[2] == 0:
                continue
            sig, eff, resp = _responders(arr, t_ax, snr)
            for nid in range(len(snr)):
                rows.append((nid, phase, lab, bool(sig[nid]), float(eff[nid]), float(resp[nid])))
    del L, dff
    gc.collect()
    out = {"n": int(len(snr)), "rows": rows}
    pickle.dump(out, open(cf, "wb"))
    return out


def _expert_session(name):
    cf = CACHE / f"exp_{name}.pkl"
    if cf.exists():
        return pickle.load(open(cf, "rb"))
    csvs = glob.glob(f"{snr_session_dir(name)}/*_snr_filtered_cells.csv")
    if not csvs:
        return None
    proj = pd.read_csv(csvs[0])
    snr = proj.loc[proj["filtered_cells"] == 1.0, "cellID"].astype(int).values
    try:
        L = load_loader(loader_pkl_path(name))
    except FileNotFoundError:
        return None
    dff = L.dff_stim_aligned
    fs = float(L.fs_image)
    if any(s not in dff for s in EXP_STIMS):
        del L, dff
        gc.collect()
        return None
    n_time = np.asarray(dff[EXP_STIMS[0]]).shape[1]
    pf = int(np.ceil(PRE_STIM_S_EXPERT * fs))
    t_ax = (np.arange(n_time) - pf) / fs
    rows = []
    for stim, lab in zip(EXP_STIMS, EXP_LABELS):
        arr = np.asarray(dff[stim])
        if arr.ndim != 3 or arr.shape[2] == 0:
            continue
        sig, eff, resp = _responders(arr, t_ax, snr)
        for nid in range(len(snr)):
            rows.append((nid, lab, bool(sig[nid]), float(eff[nid]), float(resp[nid])))
    del L, dff
    gc.collect()
    out = {"n": int(len(snr)), "rows": rows}
    pickle.dump(out, open(cf, "wb"))
    return out


def main():
    require(TDR_DIR, "TDR run"); require(LOADER_ROOT, "loader root")
    rev_names = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())["session_names"]
    exp_names = json.loads((_REPO_ROOT / "metadata" / "SAT_good_expert_sessions.json").read_text())["sessions"]

    rev_rows = []
    for name in rev_names:
        r = _reversal_session(name)
        print(f"  [rev] {'OK' if r else '!!'} {name}", flush=True)
        if r is None:
            continue
        for row in r["rows"]:
            rev_rows.append((name,) + tuple(row))
    pd.DataFrame(rev_rows, columns=["session", "neuron_id", "phase", "stimulus", "sig", "effect", "resp"]
                 ).to_csv(OUT / "responsiveness_ttest_reversal_long.csv", index=False)

    exp_rows = []
    for name in exp_names:
        try:
            r = _expert_session(name)
        except Exception as e:
            print(f"  [exp] !! {name}: {e}"); r = None
        print(f"  [exp] {'OK' if r else '!!'} {name}", flush=True)
        if r is None:
            continue
        for row in r["rows"]:
            exp_rows.append((name,) + tuple(row))
    pd.DataFrame(exp_rows, columns=["session", "neuron_id", "stimulus", "sig", "effect", "resp"]
                 ).to_csv(OUT / "responsiveness_ttest_expert_long.csv", index=False)

    meta = {
        "analysis": "Significant responders via temporal t-test (contiguous-significant criterion)",
        "method_module": "analysis/responsiveness_ttest.py (ported from analysis/26_08_26_responsiveness_t_test.ipynb)",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "epoch": "stim (0-2 s) vs pre-stim baseline (-2-0 s)",
        "criterion": "longest contiguous run of p<alpha timepoints > n_timepoints/3 (gaps up to MAX_INT tolerated)",
        "alpha": ALPHA, "max_int": MAX_INT, "frac": FRAC, "scale_per_neuron": SCALE,
        "base_window_s": list(BASE_WINDOW_S), "stim_window_s": list(STIM_WINDOW_S),
        "per_stimulus": "significance computed per stimulus; a neuron may respond to more than one",
        "cells": "SNR-filtered (reversal: TDR filtered_cell_indices; expert: filtered_cells==1)",
        "neuron_id": "0-based index within the session's SNR-filtered set (matches heatmap/Sankey)",
        "effect": "mean(stim) - mean(baseline), trial-averaged, on per-neuron z-scored activity (used for sig scaling)",
        "resp": "avg stimulus response = trial-averaged mean(stim 0-2s) - mean(baseline -2-0s) in raw dF/F; this is what the heatmaps display",
        "stim_labels_reversal": dict(zip(REV_STIMS, REV_LABELS)),
        "stim_labels_expert": dict(zip(EXP_STIMS, EXP_LABELS)),
        "n_reversal_sessions": len({r[0] for r in rev_rows}),
        "n_expert_sessions": len({r[0] for r in exp_rows}),
    }
    (OUT / "responsiveness_ttest_meta.json").write_text(json.dumps(meta, indent=2))
    print("t-test responsiveness extraction complete ->", OUT)


if __name__ == "__main__":
    main()
