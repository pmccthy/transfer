"""Dimensionality of stimulus responses via PCA (neuron-dimension reduction).

For each session PCA is run on the condition-averaged stimulus responses (samples
= stimulus x time, features = neurons) to reduce the neuron dimension. From the
eigenspectrum we record: explained-variance spectrum, number of PCs to reach 90%
variance, participation ratio ((Σλ)² / Σλ²), and the trajectory of each stimulus
projected onto the top PCs.

Reversal: PCA computed separately for pre- and post-reversal (3 stimuli each),
from the TDR ``cond_avg`` (baseline-subtracted dF/F). Expert: 3 stimuli, from the
expert cond_avg (reuses extract_expert_population).

Outputs (results/transfer/data/dimensionality/):
  dimensionality_eigenspectrum.csv   scope, session, phase, pc_index, evr, cum_evr
  dimensionality_summary.csv         scope, session, phase, n_pcs_90, participation_ratio, n_neurons
  dimensionality_pca_projections.csv scope, session, phase, stimulus, pc, time_s, projection
  dimensionality_meta.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sources import TDR_DIR, DATA_ROOT, require  # noqa: E402
import extract_expert_population as EP  # noqa: E402

OUT = DATA_ROOT / "dimensionality"
OUT.mkdir(parents=True, exist_ok=True)
_REPO_ROOT = Path(__file__).resolve().parents[3]

N_KEEP_PC = 3      # PCs kept for projection trajectories
FS = 15.0
T_AX = (np.arange(90) - 45) / FS
REV_PRE = ["100%-->0% (pre-Rev)", "50% (pre-Rev)", "0%-->100% (pre-Rev)"]
REV_POST = ["100%-->0% (post-Rev)", "50% (post-Rev)", "0%-->100% (post-Rev)"]
REV_LAB = ["100_to_0", "50", "0_to_100"]
EXP_STIMS = ["100%", "50%", "0%"]
EXP_LAB = ["100", "50", "0"]


def _pca_on(stim_arrays, labels, t_ax):
    """stim_arrays: list of (n_neurons, n_time). Returns spectrum, summary, projections."""
    n_time = stim_arrays[0].shape[1]
    X = np.concatenate([a.T for a in stim_arrays], axis=0)   # (n_stim*n_time, n_neurons)
    X = X - X.mean(axis=0, keepdims=True)
    n_comp = min(X.shape)
    pca = PCA(n_components=n_comp)
    proj = pca.fit_transform(X)                              # (n_samples, n_comp)
    evr = pca.explained_variance_ratio_
    lam = pca.explained_variance_
    pr = float((lam.sum() ** 2) / (np.square(lam).sum())) if lam.sum() > 0 else np.nan
    n90 = int(np.argmax(np.cumsum(evr) >= 0.90) + 1)
    proj = proj.reshape(len(stim_arrays), n_time, n_comp)
    return evr, np.cumsum(evr), pr, n90, proj


def _emit(scope, session, phase, arrays, labels, spec_rows, sum_rows, proj_rows):
    evr, cum, pr, n90, proj = _pca_on(arrays, labels, T_AX)
    for i in range(len(evr)):
        spec_rows.append((scope, session, phase, i + 1, float(evr[i]), float(cum[i])))
    sum_rows.append((scope, session, phase, n90, pr, arrays[0].shape[0]))
    for si, lab in enumerate(labels):
        for pc in range(min(N_KEEP_PC, proj.shape[2])):
            for ti in range(proj.shape[1]):
                proj_rows.append((scope, session, phase, lab, pc + 1,
                                  float(T_AX[ti]), float(proj[si, ti, pc])))


def main():
    require(TDR_DIR, "TDR run")
    rev_names = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())["session_names"]
    exp_names = json.loads((_REPO_ROOT / "metadata" / "SAT_good_expert_sessions.json").read_text())["sessions"]

    spec_rows, sum_rows, proj_rows = [], [], []

    for name in rev_names:
        tdr = pickle.load(open(TDR_DIR / f"{name}_tdr_v2.pkl", "rb"))
        cond_avg = np.asarray(tdr["cond_avg"])       # (6, n_filt, 90)
        ttl = list(tdr["trial_type_list"])
        for phase, keys in [("pre", REV_PRE), ("post", REV_POST)]:
            arrays = [cond_avg[ttl.index(k)] for k in keys]
            _emit("reversal", name, phase, arrays, REV_LAB, spec_rows, sum_rows, proj_rows)
        print(f"  [rev] ✓ {name}", flush=True)

    for name in exp_names:
        try:
            r = EP.compute_session(name)             # cached expert cond_avg
        except Exception:
            r = None
        if r is None:
            print(f"  [exp] ! {name}"); continue
        arrays = [r["cond_avg"][s] for s in EXP_STIMS]
        _emit("expert", name, "expert", arrays, EXP_LAB, spec_rows, sum_rows, proj_rows)
        print(f"  [exp] ✓ {name}", flush=True)

    pd.DataFrame(spec_rows, columns=["scope", "session", "phase", "pc_index", "evr", "cum_evr"]
                 ).to_csv(OUT / "dimensionality_eigenspectrum.csv", index=False)
    pd.DataFrame(sum_rows, columns=["scope", "session", "phase", "n_pcs_90",
                                    "participation_ratio", "n_neurons"]
                 ).to_csv(OUT / "dimensionality_summary.csv", index=False)
    pd.DataFrame(proj_rows, columns=["scope", "session", "phase", "stimulus", "pc",
                                     "time_s", "projection"]
                 ).to_csv(OUT / "dimensionality_pca_projections.csv", index=False)

    meta = {
        "analysis": "Dimensionality of stimulus responses (PCA on neurons)",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "method": "PCA on (stimulus x time) samples, neuron features; baseline-subtracted cond_avg",
        "reversal": "pre & post reversal separately (3 stimuli each, TDR cond_avg)",
        "expert": "3 stimuli (expert cond_avg)",
        "metrics": {"n_pcs_90": "PCs for 90% variance",
                    "participation_ratio": "(Σλ)²/Σλ² over eigenvalues",
                    "eigenspectrum": "explained_variance_ratio per PC",
                    "projections": f"top {N_KEEP_PC} PCs, per-stimulus trajectory over time"},
    }
    (OUT / "dimensionality_meta.json").write_text(json.dumps(meta, indent=2))
    print("Dimensionality extraction complete →", OUT)


if __name__ == "__main__":
    main()
