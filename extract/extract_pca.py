"""Extract PCA state-space figure data to tidy CSV + provenance JSON.

Source run (PCA companion of the canonical TDR run):
    08_04_26_reversal_dff_from_flu_processed_baseline_sub_snr_pca

Each per-session ``*_pca.pkl`` holds two PCA variants:
  * ``pca_time_avg``  — PCA fit on the time-averaged stimulus response,
  * ``pca_all_time``  — PCA fit on the full time course,
each with per-condition projections of shape (n_components, n_time).

We keep the top ``N_PC_KEEP`` PCs (enough for PC1-vs-PC2 trajectories and PC
time courses) and write them long-format.

Outputs (results/transfer/data/pca/):
  pca_projections.csv       session, variant, condition, pc, frame, time_s, projection
  pca_variance_explained.csv session, variant, n_components, var_explained
  pca_time_axis.csv         frame, time_s
  pca_meta.json             provenance
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
from _sources import PCA_DIR, DATA_ROOT, require  # noqa: E402

OUT = DATA_ROOT / "pca"
OUT.mkdir(parents=True, exist_ok=True)

N_PC_KEEP = 6          # top PCs to export
VARIANTS = ["pca_time_avg", "pca_all_time"]


def main() -> None:
    require(PCA_DIR, "PCA run directory")
    pkls = sorted(PCA_DIR.glob("*_pca.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No *_pca.pkl files in {PCA_DIR}")
    print(f"Loading {len(pkls)} per-session PCA pickles from\n  {PCA_DIR}")

    proj_rows, var_rows = [], []
    t_ax_ref = None
    trial_type_list = None

    for p in pkls:
        d = pickle.load(open(p, "rb"))
        session = d["session_name"]
        t_ax = np.asarray(d["t_ax"])
        if t_ax_ref is None:
            t_ax_ref = t_ax
            trial_type_list = list(d["trial_type_list"])

        for variant in VARIANTS:
            g = d[variant]
            var_rows.append((session, variant, int(g["n_components"]),
                             float(g["var_explained"])))
            projections = g["projections"]        # {cond: (n_comp, n_time)}
            for cond, arr in projections.items():
                arr = np.asarray(arr)
                n_keep = min(N_PC_KEEP, arr.shape[0])
                for pc in range(n_keep):
                    for ti in range(arr.shape[1]):
                        proj_rows.append((session, variant, cond, pc + 1, ti,
                                          float(t_ax[ti]), float(arr[pc, ti])))

    pd.DataFrame(proj_rows,
                 columns=["session", "variant", "condition", "pc",
                          "frame", "time_s", "projection"]
                 ).to_csv(OUT / "pca_projections.csv", index=False)
    pd.DataFrame(var_rows,
                 columns=["session", "variant", "n_components", "var_explained"]
                 ).to_csv(OUT / "pca_variance_explained.csv", index=False)
    pd.DataFrame({"frame": range(len(t_ax_ref)),
                  "time_s": [float(v) for v in t_ax_ref]}
                 ).to_csv(OUT / "pca_time_axis.csv", index=False)

    d0 = pickle.load(open(pkls[0], "rb"))
    meta = {
        "analysis": "PCA state-space (per session)",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(PCA_DIR),
        "companion_of": "canonical TDR run (same 13 sessions, same preprocessing)",
        "variants": {
            "pca_time_avg": "PCA fit on time-averaged stimulus response",
            "pca_all_time": "PCA fit on full time course",
        },
        "n_pcs_exported": N_PC_KEEP,
        "trial_type_list": trial_type_list,
        "sessions": [pickle.load(open(p, "rb"))["session_name"] for p in pkls],
        "fs_image_hz": float(d0["fs_image"]),
        "normalisation": d0.get("normalisation"),
        "response_window_s": list(d0.get("response_window_s", [])),
        "pre_stim_s": d0.get("pre_stim_s"),
        "post_stim_s": d0.get("post_stim_s"),
        "stim_onset_time_s": 0.0,
        "stim_duration_s": d0.get("stim_duration_s"),
        "random_state": d0.get("random_state"),
        "notes": "projection[pc] is the trial-averaged projection onto PC (pc) over time.",
    }
    (OUT / "pca_meta.json").write_text(json.dumps(meta, indent=2))

    print("PCA extraction complete →", OUT)
    for f in sorted(OUT.glob("*")):
        print("   ", f.name)


if __name__ == "__main__":
    main()
