"""Extract canonical TDR figure data to tidy CSV + provenance JSON.

Source run (see ``analysis/CANONICAL_RESULTS.md``):
    08_04_26_reversal_dff_from_flu_processed_baseline_sub_snr_val_peak_2_3s_tdr_v2

What it does:
  * loads the 13 per-session ``*_tdr_v2.pkl`` files,
  * rebuilds the combined (cross-session) projection with the *exact* source
    function ``run_tdr_v2.combine_trials`` — so the combined arrays are
    bit-identical to the blessed ``combined_tdr_v2_results.pkl`` without having
    to load that 1.8 GB file,
  * writes long-format CSVs that any plotting tool can read.

Outputs (results/transfer/data/tdr/):
  tdr_combined_projections.csv   proj_type, condition, predictor, frame, time_s, projection
  tdr_per_session_projections.csv session, condition, predictor, frame, time_s, projection
  tdr_design_matrix.csv          condition, predictor, value
  tdr_peak_times.csv             predictor, peak_frame, peak_time_s   (combined axes)
  tdr_variance_explained.csv     scope, session, var_explained
  tdr_time_axis.csv              axis, frame, time_s
  tdr_meta.json                  provenance / arguments
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

# Make the source analysis package importable so we can reuse combine_trials.
_REPO_ROOT = Path(__file__).resolve().parents[3]      # .../neuronal-representations
sys.path.insert(0, str(_REPO_ROOT / "analysis"))
import run_tdr_v2 as R                                # noqa: E402

OUT = DATA_ROOT / "tdr"
OUT.mkdir(parents=True, exist_ok=True)

N_PCS = 100  # matches analysis_summary_v2.json (n_pcs)


def main() -> None:
    require(TDR_DIR, "Canonical TDR run directory")
    summary = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())
    session_names = summary["session_names"]

    print(f"Loading {len(session_names)} per-session TDR pickles from\n  {TDR_DIR}")
    session_data_list = []
    for name in session_names:
        with open(TDR_DIR / f"{name}_tdr_v2.pkl", "rb") as f:
            session_data_list.append(pickle.load(f))

    ref = session_data_list[0]
    predictor_names = list(ref["predictor_names"])
    trial_type_list = list(ref["trial_type_list"])
    t_ax = np.asarray(ref["t_ax"])
    t_ax_fit = np.asarray(ref["t_ax_fit"])

    # --- rebuild the combined projection with the exact source routine --------
    print("Rebuilding combined projection via run_tdr_v2.combine_trials ...")
    combined = R.combine_trials(session_data_list, N_PCS)

    # ---- combined projections (peak + pooled) --------------------------------
    rows = []
    for proj_type, key in [("peak", "proj_peak_combined"),
                           ("pooled", "proj_pooled_combined")]:
        proj = np.asarray(combined[key])          # (n_cond, n_pred, n_time)
        for ci, cond in enumerate(trial_type_list):
            for pi, pred in enumerate(predictor_names):
                for ti in range(proj.shape[2]):
                    rows.append((proj_type, cond, pred, ti,
                                 float(t_ax[ti]), float(proj[ci, pi, ti])))
    pd.DataFrame(rows, columns=["proj_type", "condition", "predictor",
                                "frame", "time_s", "projection"]
                 ).to_csv(OUT / "tdr_combined_projections.csv", index=False)

    # ---- per-session projections (peak axes) ---------------------------------
    rows = []
    for sd in session_data_list:
        proj = np.asarray(sd["proj_peak"])        # (n_cond, n_pred, n_time)
        t = np.asarray(sd["t_ax"])
        for ci, cond in enumerate(trial_type_list):
            for pi, pred in enumerate(predictor_names):
                for ti in range(proj.shape[2]):
                    rows.append((sd["session_name"], cond, pred, ti,
                                 float(t[ti]), float(proj[ci, pi, ti])))
    pd.DataFrame(rows, columns=["session", "condition", "predictor",
                                "frame", "time_s", "projection"]
                 ).to_csv(OUT / "tdr_per_session_projections.csv", index=False)

    # ---- design matrix -------------------------------------------------------
    X = np.asarray(ref["X_design"])               # (n_cond, n_pred)
    rows = [(trial_type_list[ci], predictor_names[pi], float(X[ci, pi]))
            for ci in range(X.shape[0]) for pi in range(X.shape[1])]
    pd.DataFrame(rows, columns=["condition", "predictor", "value"]
                 ).to_csv(OUT / "tdr_design_matrix.csv", index=False)

    # ---- combined peak times -------------------------------------------------
    peak_frames = np.asarray(combined["peak_times"])
    peak_s = np.asarray(combined["peak_times_s"])
    pd.DataFrame({"predictor": predictor_names,
                  "peak_frame": peak_frames.astype(int),
                  "peak_time_s": peak_s.astype(float)}
                 ).to_csv(OUT / "tdr_peak_times.csv", index=False)

    # ---- variance explained --------------------------------------------------
    rows = [("combined", "ALL", float(combined["var_explained"]))]
    for sd in session_data_list:
        rows.append(("session", sd["session_name"], float(sd["var_explained"])))
    pd.DataFrame(rows, columns=["scope", "session", "var_explained"]
                 ).to_csv(OUT / "tdr_variance_explained.csv", index=False)

    # ---- time axes -----------------------------------------------------------
    rows = [("t_ax", i, float(v)) for i, v in enumerate(t_ax)]
    rows += [("t_ax_fit", i, float(v)) for i, v in enumerate(t_ax_fit)]
    pd.DataFrame(rows, columns=["axis", "frame", "time_s"]
                 ).to_csv(OUT / "tdr_time_axis.csv", index=False)

    # ---- provenance ----------------------------------------------------------
    meta = {
        "analysis": "TDR (targeted dimensionality reduction), individual-trial betas",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(TDR_DIR),
        "source_script": "analysis/run_tdr_v2.py",
        "combined_via": "run_tdr_v2.combine_trials(session_data_list, n_pcs=%d)" % N_PCS,
        "canonical_record": "analysis/CANONICAL_RESULTS.md (TDR)",
        "design": summary["design"],
        "predictor_names": predictor_names,
        "trial_type_list": trial_type_list,
        "n_sessions": summary["n_sessions_processed"],
        "session_names": session_names,
        "n_neurons_total": summary["n_neurons_total"],
        "n_neurons_per_session": summary["n_neurons_per_session"],
        "fs_image_hz": float(ref["fs_image"]),
        "stim_onset_time_s": 0.0,
        "stim_duration_s": summary["stim_duration_s"],
        "normalisation": summary["normalisation"],
        "n_pcs": summary["n_pcs"],
        "value_peak_window_s": [2.0, 3.0],
        "proj_axis_key": "peak (value axis at 2-3 s peak window); pooled provided for comparison",
        "notes": (
            "proj_type='peak' is the canonical headline projection. Conditions are "
            "3 visual stimuli x 2 contexts (pre/post reversal). Predictors: stim "
            "(identity), context (pre/post), value (current reward probability = "
            "stim x context interaction)."
        ),
    }
    (OUT / "tdr_meta.json").write_text(json.dumps(meta, indent=2))

    print("TDR extraction complete →", OUT)
    for f in sorted(OUT.glob("*")):
        print("   ", f.name)


if __name__ == "__main__":
    main()
