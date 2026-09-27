"""Recompute ALL reversal decoders self-contained, from the bundle's own code.

Rather than reading an older run's stored results, this recomputes every reversal
decoder with the canonical pipeline code (``run_time_resolved_decoding``) driven
by the transfer bundle's memory-safe data loader — so the whole decoding result
is reproducible from code + data in the repo and internally consistent, and it
includes the new ``stim_identity`` decoder on equal footing with the rest.

Per session: load dF/F (safe loader) → SNR-filter with the *decoding* filter
(cellID-1) → per-trial pre-stim baseline subtract → ``process_session`` (all
decoders) → aggregate across sessions with the pipeline's ``aggregate_results``.

Decoders (see run_time_resolved_decoding.py):
  within : stim_pre/post_{0v1,0v50,1v50}, context_{0to100,100to0,pooled},
           value_xor, stim_identity, phase_{50,100to0,0to100}_pre_vs_post
  cross  : stim_cross_{0v1,0v50,1v50}_{pre2post,post2pre}

Outputs (results/transfer/data/decoding/):
  decoding_aggregated.csv     decoder, time_s, mean_accuracy, sem, n_sessions
  decoding_per_session.csv    decoder, session, session_index, time_s, accuracy
  decoding_cross_phase.csv    decoder, split, time_s, mean_accuracy, sem, n_sessions
  decoding_meta.json          provenance / settings

Per-session results are cached under ``$TRANSFER_DECODE_CACHE`` (default a temp
dir) so the run is resumable if interrupted.
"""

from __future__ import annotations

import gc
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
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analysis"))
import run_time_resolved_decoding as P  # noqa: E402
from sat_snr_cell_filter import load_snr_filtered_neuron_indices  # noqa: E402
import _neuron_cap as NCAP  # noqa: E402

OUT = DATA_ROOT / "decoding"
OUT.mkdir(parents=True, exist_ok=True)
# cache keyed by neuron-cap so all-neuron and capped runs never collide
_CTAG = f"cap{NCAP.NUM_NEURONS}x{NCAP.N_NEURON_PERM}" if NCAP.cap_enabled() else "allneurons"
CACHE = Path(os.environ.get("TRANSFER_DECODE_CACHE", "/tmp/transfer_decode_cache")) / _CTAG
CACHE.mkdir(parents=True, exist_ok=True)

PRE_STIM_S = 3.0


def compute_session(name):
    """process_session result for one session (cached).

    Decodes from NCAP.NUM_NEURONS neurons subsampled from the session's
    SNR-filtered set, averaged over NCAP.N_NEURON_PERM random permutations
    (matches the canonical SAT pipeline). All-neuron behaviour is recovered by
    setting the cap to "all" (see _neuron_cap.py)."""
    cache_f = CACHE / f"{name}.pkl"
    if cache_f.exists():
        with open(cache_f, "rb") as f:
            return pickle.load(f)

    snr_idx = load_snr_filtered_neuron_indices(snr_session_dir(name))
    if snr_idx is None:
        return None
    L = load_loader(loader_pkl_path(name))
    dff = {k: np.asarray(v) for k, v in L.dff_stim_aligned.items() if hasattr(v, "shape")}
    fs = float(L.fs_image)
    del L
    gc.collect()
    for tt in P.ALL_REQUIRED:
        if tt not in dff:
            return None
        dff[tt] = dff[tt][snr_idx]
    dff = P.per_trial_prestim_baseline_subtract_dff_dict(dff, fs, pre_stim_s=PRE_STIM_S)

    n_neurons = dff[P.ALL_REQUIRED[0]].shape[0]
    subsets = NCAP.neuron_subsets(n_neurons)
    perm_results = [P.process_session({tt: arr[idx] for tt, arr in dff.items()},
                                      fs, name, window_size=1) for idx in subsets]
    result = NCAP.average_time_resolved(perm_results)
    del dff
    gc.collect()
    with open(cache_f, "wb") as f:
        pickle.dump(result, f)
    return result


def main():
    require(TDR_DIR, "Canonical TDR run (session list)")
    require(LOADER_ROOT, "Loader pkl root")
    sessions = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())["session_names"]

    all_results = {}
    for name in sessions:
        res = compute_session(name)
        all_results[name] = res
        print(f"  {'✓' if res else '!'} {name}", flush=True)

    aggregated = P.aggregate_results(all_results, sessions)

    std_rows, per_rows, cross_rows = [], [], []
    for dec, e in aggregated.items():
        t = np.asarray(e["time_ax"], dtype=float) / 1000.0   # ms -> s
        if "mean" in e:
            mean, se = np.asarray(e["mean"]), np.asarray(e["sem"])
            n = int(e["n_sessions"])
            for ti in range(len(t)):
                std_rows.append((dec, float(t[ti]), float(mean[ti]), float(se[ti]), n))
            ps = np.asarray(e["per_session"])
            for si in range(ps.shape[0]):
                for ti in range(ps.shape[1]):
                    per_rows.append((dec, sessions[si], si, float(t[ti]), float(ps[si, ti])))
        if "train_mean" in e:
            n = int(e["n_sessions"])
            for split in ("train", "test"):
                mean, se = np.asarray(e[f"{split}_mean"]), np.asarray(e[f"{split}_sem"])
                for ti in range(len(t)):
                    cross_rows.append((dec, split, float(t[ti]), float(mean[ti]),
                                       float(se[ti]), n))

    pd.DataFrame(std_rows, columns=["decoder", "time_s", "mean_accuracy", "sem",
                                    "n_sessions"]).to_csv(OUT / "decoding_aggregated.csv", index=False)
    pd.DataFrame(per_rows, columns=["decoder", "session", "session_index", "time_s",
                                    "accuracy"]).to_csv(OUT / "decoding_per_session.csv", index=False)
    pd.DataFrame(cross_rows, columns=["decoder", "split", "time_s", "mean_accuracy",
                                      "sem", "n_sessions"]).to_csv(OUT / "decoding_cross_phase.csv", index=False)

    meta = {
        "analysis": "Time-resolved decoding — recomputed self-contained from bundle code",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_code": "analysis/run_time_resolved_decoding.py (process_session + aggregate_results)",
        "decoder_engine": "LinearDecoder (linear SVM), 5-fold CV, per-timepoint StandardScaler",
        "preprocessing": "SNR-filtered cells (decoding filter, cellID-1); per-trial "
                         f"pre-stim baseline subtract (pre_stim={PRE_STIM_S}s); balanced min_n trials",
        **NCAP.meta(),
        "n_sessions": len([s for s in sessions if all_results.get(s)]),
        "sessions": sessions,
        "chance_level": 0.5,
        "decoders_within": [k for k, v in aggregated.items() if "mean" in v],
        "decoders_cross": [k for k, v in aggregated.items() if "train_mean" in v],
        "note": "Recomputed from code+data in this bundle; supersedes the earlier "
                "blessed-run extraction so all decoders (incl. stim_identity) are "
                "internally consistent.",
    }
    (OUT / "decoding_meta.json").write_text(json.dumps(meta, indent=2))
    print("Decoding recompute complete →", OUT)


if __name__ == "__main__":
    main()
