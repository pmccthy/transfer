"""Extract single-trial neural responses (trial resolution) for the neuron x
condition heatmaps.

For each of the 13 canonical reversal sessions we read the loader pkl's
``dff_stim_aligned`` (per condition, shape (n_cells, n_time, n_trials)) and, for
every trial, compute the mean dF/F in a response window and a pre-stim baseline
window per neuron. Keeping these at *trial resolution* means the neuron x
condition heatmaps can later be recomputed over any subset of trials (see
``figures/plot_heatmaps.py::condition_means_from_trials``).

SNR-filtered cell indices and the condition list/time axis are taken from the
canonical TDR per-session pkls, so cells and conditions match the rest of the
transfer bundle exactly.

Outputs (results/transfer/data/single_trial/):
  <session>_single_trial.npz   per session:
        resp_mean   (n_cells, n_trials)   mean dF/F in response window
        base_mean   (n_cells, n_trials)   mean dF/F in baseline window
        cell_index  (n_cells,)            0-based cell id within the session
        snr_mask    (n_cells,) bool       True = passes SNR filter
        trial_condition        (n_trials,)  condition label per trial
        trial_session_number   (n_trials,)  trial number within the session (beh_df)
        t_ax        (n_time,)             for reference
  single_trial_metadata.csv    session, trial_col_index, condition,
                               session_trial_number, stimulus_type, reward_prob_rev
  heatmap_condition_means.csv  session, cell_index, snr_filtered, condition,
                               mean_evoked  (evoked = resp-base, averaged over ALL
                               trials in the condition; the default full-session heatmap)
  single_trial_meta.json       provenance / window definitions
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sources import TDR_DIR, LOADER_ROOT, loader_pkl_path, DATA_ROOT, require  # noqa: E402
from _loader_pkl import load_loader  # noqa: E402
import pickle  # noqa: E402

OUT = DATA_ROOT / "single_trial"
OUT.mkdir(parents=True, exist_ok=True)
HEATMAP_OUT = DATA_ROOT / "heatmap"
HEATMAP_OUT.mkdir(parents=True, exist_ok=True)

RESP_WINDOW_S = (0.0, 2.0)     # stimulus response window
BASE_WINDOW_S = (-1.0, 0.0)    # pre-stim baseline window


def _win_mask(t_ax, lo, hi):
    return (t_ax >= lo) & (t_ax < hi)


def main() -> None:
    require(TDR_DIR, "Canonical TDR run (for cell filter + condition list)")
    require(LOADER_ROOT, "Loader pkl root (SAT_dff_from_flu_processed)")
    summary = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())
    session_names = summary["session_names"]

    meta_rows, heatmap_rows = [], []

    for name in session_names:
        tdr = pickle.load(open(TDR_DIR / f"{name}_tdr_v2.pkl", "rb"))
        conds = list(tdr["trial_type_list"])
        t_ax = np.asarray(tdr["t_ax"])
        snr_idx = np.asarray(tdr["filtered_cell_indices"]).astype(int)

        L = load_loader(loader_pkl_path(name))
        dff = L.dff_stim_aligned
        trial_idxs = L.trial_idxs
        beh = L.beh_df

        n_cells = np.asarray(dff[conds[0]]).shape[0]
        resp_m = _win_mask(t_ax, *RESP_WINDOW_S)
        base_m = _win_mask(t_ax, *BASE_WINDOW_S)

        resp_cols, base_cols = [], []
        trial_condition, trial_session_number = [], []

        for cond in conds:
            arr = np.asarray(dff[cond])          # (n_cells, n_time, n_trials)
            if arr.ndim != 3 or arr.shape[2] == 0:
                continue
            resp = arr[:, resp_m, :].mean(axis=1)   # (n_cells, n_trials)
            base = arr[:, base_m, :].mean(axis=1)
            resp_cols.append(resp)
            base_cols.append(base)
            sess_nums = list(trial_idxs.get(cond, []))
            for k in range(arr.shape[2]):
                trial_condition.append(cond)
                trial_session_number.append(int(sess_nums[k]) if k < len(sess_nums) else -1)

        resp_mean = np.concatenate(resp_cols, axis=1)   # (n_cells, n_trials_total)
        base_mean = np.concatenate(base_cols, axis=1)
        trial_condition = np.array(trial_condition, dtype=object)
        trial_session_number = np.array(trial_session_number, dtype=int)

        # capture the small behaviour columns, then release the heavy loader object
        stim_type = np.asarray(beh["stimulusType"].values)
        rprev = np.asarray(beh["rewardProbRev"].values)
        del L, dff, trial_idxs, beh, resp_cols, base_cols
        gc.collect()

        snr_mask = np.zeros(n_cells, dtype=bool)
        snr_mask[snr_idx] = True

        np.savez_compressed(
            OUT / f"{name}_single_trial.npz",
            resp_mean=resp_mean.astype(np.float32),
            base_mean=base_mean.astype(np.float32),
            cell_index=np.arange(n_cells, dtype=int),
            snr_mask=snr_mask,
            trial_condition=trial_condition.astype(str),
            trial_session_number=trial_session_number,
            t_ax=t_ax.astype(np.float32),
            resp_window_s=np.array(RESP_WINDOW_S),
            base_window_s=np.array(BASE_WINDOW_S),
        )

        # trial metadata rows
        for j, (cond, sn) in enumerate(zip(trial_condition, trial_session_number)):
            meta_rows.append((name, j, cond, sn,
                              stim_type[sn] if 0 <= sn < len(stim_type) else "",
                              rprev[sn] if 0 <= sn < len(rprev) else ""))

        # default full-session heatmap: evoked, averaged over ALL trials per condition,
        # SNR-filtered cells only
        evoked = resp_mean - base_mean
        for cond in conds:
            col = trial_condition == cond
            if col.sum() == 0:
                continue
            cond_mean = evoked[:, col].mean(axis=1)     # (n_cells,)
            for ci in snr_idx:
                heatmap_rows.append((name, int(ci), True, cond, float(cond_mean[ci])))

        print(f"  ✓ {name}: {resp_mean.shape[0]} cells x {resp_mean.shape[1]} trials "
              f"({int(snr_mask.sum())} SNR-filtered)", flush=True)
        del resp_mean, base_mean, evoked, tdr
        gc.collect()

    pd.DataFrame(meta_rows,
                 columns=["session", "trial_col_index", "condition",
                          "session_trial_number", "stimulus_type", "reward_prob_rev"]
                 ).to_csv(OUT / "single_trial_metadata.csv", index=False)
    pd.DataFrame(heatmap_rows,
                 columns=["session", "cell_index", "snr_filtered", "condition", "mean_evoked"]
                 ).to_csv(HEATMAP_OUT / "heatmap_condition_means.csv", index=False)

    meta = {
        "analysis": "Single-trial neural responses + neuron x condition heatmap data",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "loader_root": str(LOADER_ROOT),
        "cell_filter_and_conditions_from": str(TDR_DIR),
        "sessions": session_names,
        "response_window_s": list(RESP_WINDOW_S),
        "baseline_window_s": list(BASE_WINDOW_S),
        "evoked_definition": "mean(response window) - mean(baseline window), per neuron per trial",
        "units": "dF/F",
        "trial_resolution": (
            "resp_mean/base_mean are (n_cells, n_trials); recompute condition means "
            "over any trial subset for future figures — see "
            "figures/plot_heatmaps.py::condition_means_from_trials."
        ),
        "conditions": ["100%-->0%", "50%", "0%-->100%"],
        "heatmap_default": "SNR-filtered cells, evoked, averaged over all trials in each condition",
    }
    (OUT / "single_trial_meta.json").write_text(json.dumps(meta, indent=2))
    print("Single-trial extraction complete →", OUT, "and", HEATMAP_OUT)


if __name__ == "__main__":
    main()
