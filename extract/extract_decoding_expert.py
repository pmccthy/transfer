"""Recompute expert stim-pair decoding self-contained, from the bundle's code.

Expert (non-reversal) sessions have a single stable contingency, so only three
pairwise stimulus decoders apply (stim_0v1 / stim_0v50 / stim_1v50). This mirrors
extract_decoding.py but uses ``process_session_expert`` + ``EXPERT_DECODER_CONFIGS``
over the expert session list in metadata/.

Session list source (first that exists):
  metadata/SAT_expert_sessions.json  or  metadata/expert_sessions.json
Override with $TRANSFER_EXPERT_SESSIONS (path to a JSON list of session names).

Outputs (results/transfer/data/decoding/):
  decoding_expert.csv             decoder, time_s, mean_accuracy, sem, n_sessions
  decoding_expert_per_session.csv decoder, session, session_index, time_s, accuracy
  decoding_expert_meta.json       provenance

Per-session results cached under $TRANSFER_DECODE_CACHE/expert (resumable).
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
from _sources import (LOADER_ROOT, loader_pkl_path, snr_session_dir,  # noqa: E402
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
_CTAG = f"cap{NCAP.NUM_NEURONS}x{NCAP.N_NEURON_PERM}" if NCAP.cap_enabled() else "allneurons"
CACHE = Path(os.environ.get("TRANSFER_DECODE_CACHE", "/tmp/transfer_decode_cache")) / f"expert_{_CTAG}"
CACHE.mkdir(parents=True, exist_ok=True)
PRE_STIM_S = 3.0


def expert_sessions():
    env = os.environ.get("TRANSFER_EXPERT_SESSIONS")
    if env and Path(env).exists():
        return json.loads(Path(env).read_text())
    # SAT_good_expert_sessions.json (44 sessions) is the intended expert set
    # (matches the blessed expert decoding run exactly).
    for cand in ["SAT_good_expert_sessions.json", "SAT_expert_sessions.json",
                 "expert_sessions.json"]:
        p = _REPO_ROOT / "metadata" / cand
        if p.exists():
            data = json.loads(p.read_text())
            return data if isinstance(data, list) else data.get("sessions", list(data))
    raise FileNotFoundError("No expert session list found in metadata/")


def compute_session(name):
    cache_f = CACHE / f"{name}.pkl"
    if cache_f.exists():
        with open(cache_f, "rb") as f:
            return pickle.load(f)
    snr_idx = load_snr_filtered_neuron_indices(snr_session_dir(name))
    if snr_idx is None:
        return None
    try:
        L = load_loader(loader_pkl_path(name))
    except FileNotFoundError:
        return None
    dff = {k: np.asarray(v) for k, v in L.dff_stim_aligned.items() if hasattr(v, "shape")}
    fs = float(L.fs_image)
    del L
    gc.collect()
    if any(tt not in dff for tt in P.EXPERT_REQUIRED):
        return None
    n_cells = dff[P.EXPERT_REQUIRED[0]].shape[0]
    snr_idx = snr_idx[snr_idx < n_cells]
    for tt in P.EXPERT_REQUIRED:
        dff[tt] = dff[tt][snr_idx]
    dff = P.per_trial_prestim_baseline_subtract_dff_dict(dff, fs, pre_stim_s=PRE_STIM_S)
    n_neurons = dff[P.EXPERT_REQUIRED[0]].shape[0]
    perm_results = [P.process_session_expert({tt: arr[idx] for tt, arr in dff.items()},
                                             fs, name, window_size=1)
                    for idx in NCAP.neuron_subsets(n_neurons)]
    result = NCAP.average_time_resolved(perm_results)
    del dff
    gc.collect()
    with open(cache_f, "wb") as f:
        pickle.dump(result, f)
    return result


def main():
    require(LOADER_ROOT, "Loader pkl root")
    sessions = expert_sessions()
    print(f"Expert sessions: {len(sessions)}")
    all_results = {}
    for name in sessions:
        try:
            res = compute_session(name)
        except Exception as e:
            print(f"  ! {name}: {e}")
            res = None
        all_results[name] = res
        print(f"  {'✓' if res else '!'} {name}", flush=True)

    aggregated = P.aggregate_results(all_results, sessions,
                                     decoder_configs=P.EXPERT_DECODER_CONFIGS)
    std_rows, per_rows = [], []
    used = [s for s in sessions if all_results.get(s)]
    for dec, e in aggregated.items():
        t = np.asarray(e["time_ax"], dtype=float) / 1000.0
        mean, se = np.asarray(e["mean"]), np.asarray(e["sem"])
        n = int(e["n_sessions"])
        for ti in range(len(t)):
            std_rows.append((dec, float(t[ti]), float(mean[ti]), float(se[ti]), n))
        ps = np.asarray(e["per_session"])
        # map per_session rows to the sessions that contributed (in order)
        contrib = [s for s in sessions if all_results.get(s) and dec in all_results[s]]
        for si in range(ps.shape[0]):
            sname = contrib[si] if si < len(contrib) else f"session_{si}"
            for ti in range(ps.shape[1]):
                per_rows.append((dec, sname, si, float(t[ti]), float(ps[si, ti])))

    pd.DataFrame(std_rows, columns=["decoder", "time_s", "mean_accuracy", "sem",
                                    "n_sessions"]).to_csv(OUT / "decoding_expert.csv", index=False)
    pd.DataFrame(per_rows, columns=["decoder", "session", "session_index", "time_s",
                                    "accuracy"]).to_csv(OUT / "decoding_expert_per_session.csv", index=False)
    meta = {
        "analysis": "Expert stim-pair decoding — recomputed self-contained",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_code": "analysis/run_time_resolved_decoding.py (process_session_expert)",
        "decoders": list(aggregated.keys()),
        **NCAP.meta(),
        "n_sessions": len(used),
        "sessions": used,
        "chance_level": 0.5,
        "note": "Expert sessions have no reversal, so only pairwise stimulus decoders.",
    }
    (OUT / "decoding_expert_meta.json").write_text(json.dumps(meta, indent=2))
    print("Expert decoding recompute complete →", OUT)


if __name__ == "__main__":
    main()
