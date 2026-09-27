"""Extract per-condition population-mean activity to tidy CSV.

Population mean = trial-averaged dF/F averaged across neurons, per condition and
time point. Computed from the condition-averaged arrays (``cond_avg``, shape
(n_cond, n_neurons, n_time)) stored in the canonical TDR per-session pickles, so
it uses exactly the same neurons, sessions and preprocessing as the TDR result.

Two aggregations:
  * per session  — mean over that session's neurons,
  * combined     — mean over all neurons pooled across sessions (neuron-weighted),
                   with SEM computed across the 13 session means.

Outputs (results/transfer/data/population/):
  population_means_per_session.csv  session, condition, frame, time_s, pop_mean, n_neurons
  population_means_combined.csv     condition, frame, time_s, pop_mean_pooled,
                                    mean_of_session_means, sem_over_sessions, n_sessions
  population_means_meta.json        provenance
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sources import TDR_DIR, DATA_ROOT, require  # noqa: E402

OUT = DATA_ROOT / "population"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    require(TDR_DIR, "Canonical TDR run directory (source of cond_avg)")
    summary = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())
    session_names = summary["session_names"]

    print(f"Loading {len(session_names)} per-session TDR pickles for cond_avg ...")

    per_rows = []
    trial_type_list = None
    t_ax_ref = None
    # accumulate for pooled + session-mean stats
    sum_pooled = None            # (n_cond, n_time)  running sum over neurons
    total_neurons = 0
    session_means = []           # list of (n_cond, n_time)

    for name in session_names:
        d = pickle.load(open(TDR_DIR / f"{name}_tdr_v2.pkl", "rb"))
        cond_avg = np.asarray(d["cond_avg"])         # (n_cond, n_neurons, n_time)
        t_ax = np.asarray(d["t_ax"])
        ttl = list(d["trial_type_list"])
        if trial_type_list is None:
            trial_type_list = ttl
            t_ax_ref = t_ax
            n_cond, _, n_time = cond_avg.shape
            sum_pooled = np.zeros((n_cond, n_time))

        pop_mean = cond_avg.mean(axis=1)             # (n_cond, n_time)
        session_means.append(pop_mean)
        sum_pooled += cond_avg.sum(axis=1)
        total_neurons += cond_avg.shape[1]

        for ci, cond in enumerate(ttl):
            for ti in range(cond_avg.shape[2]):
                per_rows.append((name, cond, ti, float(t_ax[ti]),
                                 float(pop_mean[ci, ti]), int(cond_avg.shape[1])))

    pd.DataFrame(per_rows,
                 columns=["session", "condition", "frame", "time_s",
                          "pop_mean", "n_neurons"]
                 ).to_csv(OUT / "population_means_per_session.csv", index=False)

    pooled = sum_pooled / total_neurons              # (n_cond, n_time)
    sess_stack = np.stack(session_means, axis=0)     # (n_sess, n_cond, n_time)
    mean_of_means = sess_stack.mean(axis=0)
    sem_over_sess = sess_stack.std(axis=0, ddof=1) / np.sqrt(sess_stack.shape[0])

    comb_rows = []
    for ci, cond in enumerate(trial_type_list):
        for ti in range(pooled.shape[1]):
            comb_rows.append((cond, ti, float(t_ax_ref[ti]),
                              float(pooled[ci, ti]),
                              float(mean_of_means[ci, ti]),
                              float(sem_over_sess[ci, ti]),
                              sess_stack.shape[0]))
    pd.DataFrame(comb_rows,
                 columns=["condition", "frame", "time_s", "pop_mean_pooled",
                          "mean_of_session_means", "sem_over_sessions", "n_sessions"]
                 ).to_csv(OUT / "population_means_combined.csv", index=False)

    meta = {
        "analysis": "Per-condition population mean activity (trial-averaged, neuron-averaged)",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(TDR_DIR),
        "source_field": "cond_avg (n_cond, n_neurons, n_time) from *_tdr_v2.pkl",
        "trial_type_list": trial_type_list,
        "n_sessions": len(session_names),
        "sessions": session_names,
        "n_neurons_total": int(total_neurons),
        "fs_image_hz": 15.0,
        "stim_onset_time_s": 0.0,
        "stim_duration_s": summary["stim_duration_s"],
        "normalisation": summary["normalisation"],
        "units": "dF/F (baseline-subtracted per trial), averaged over neurons",
        "notes": (
            "pop_mean_pooled averages over all neurons pooled across sessions "
            "(neuron-weighted). sem_over_sessions is computed across the 13 "
            "per-session population means (session as the unit)."
        ),
    }
    (OUT / "population_means_meta.json").write_text(json.dumps(meta, indent=2))

    print("Population-means extraction complete →", OUT)
    for f in sorted(OUT.glob("*")):
        print("   ", f.name)


if __name__ == "__main__":
    main()
