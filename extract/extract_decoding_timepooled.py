"""Time-pooled decoding (canonical LinearDecoder.time_pooled_decoder).

Decodes from activity averaged over the [0, 2000] ms response window (one
accuracy per session per decoder) using the canonical time-pooled decoder, with
the SAME decoder definitions and preprocessing as the time-resolved recompute
(SNR cells cellID-1, per-trial pre-stim baseline subtract) so the time-pooled
bars are directly comparable to the time-resolved traces. Includes the new
``stim_identity`` decoder (also added to analysis/run_time_pooled_no_baseline.py).

Reversal: all within + cross decoders. Expert: the 3 stimulus pairs.

Outputs (results/transfer/data/decoding/):
  decoding_timepooled.csv             scope, decoder, mean_accuracy, sem, n_sessions
  decoding_timepooled_per_session.csv scope, decoder, session, accuracy
  decoding_timepooled_cross.csv       decoder, split, mean_accuracy, sem, n_sessions
  decoding_timepooled_meta.json

Per-session results cached under $TRANSFER_DECODE_CACHE/timepooled (resumable).
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
from scipy.stats import sem as scipy_sem

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
_CTAG = f"cap{NCAP.NUM_NEURONS}x{NCAP.N_NEURON_PERM}" if NCAP.cap_enabled() else "allneurons"
CACHE = Path(os.environ.get("TRANSFER_DECODE_CACHE", "/tmp/transfer_decode_cache")) / f"timepooled_{_CTAG}"
CACHE.mkdir(parents=True, exist_ok=True)

WINDOW_MS = [0, 2000]
PRE_STIM_S = 3.0


def _decode_once(dff, fs, t_ax, min_n, configs):
    """One neuron subset: {decoder: ('within',acc) | ('cross',train,test)}."""
    out = {}
    for dec, cfg in configs.items():
        if cfg["type"] == "within":
            dm, lab = P.build_within_data(dff, cfg["class0_types"], cfg["class1_types"], min_n)
            nd = P.NeuralData(data=dm, fs=fs, time_ax=t_ax, labels=lab)
            dobj = P.LinearDecoder(neural_data=nd, targets=lab, model="svm")
            fold_scores = dobj.time_pooled_decoder(
                time_window=WINDOW_MS, cross_val_folds=P.N_CV_FOLDS,
                normalise=True, random_state=P.RANDOM_STATE)[0]
            out[dec] = ("within", float(np.mean(fold_scores)))
        else:
            trm, trl, tem, tel = P.build_cross_data(
                dff, cfg["train_class0_types"], cfg["train_class1_types"],
                cfg["test_class0_types"], cfg["test_class1_types"], min_n)
            nd = P.NeuralData(data=trm, fs=fs, time_ax=t_ax, labels=trl)
            dobj = P.LinearDecoder(neural_data=nd, targets=trl, model="svm")
            res = dobj.time_pooled_decoder_cross_testing(
                time_window=WINDOW_MS, train_data=trm, train_targets=trl,
                test_data=tem, test_targets=tel, cross_val_folds=P.N_CV_FOLDS,
                normalise=True, random_state=P.RANDOM_STATE)
            out[dec] = ("cross", float(np.mean(res[0])), float(np.mean(res[1])))
    return out


def _average_out(perm_outs):
    """Average per-decoder accuracies across neuron permutations."""
    if len(perm_outs) == 1:
        return perm_outs[0]
    merged = {}
    for dec in perm_outs[0]:
        kind = perm_outs[0][dec][0]
        if kind == "within":
            merged[dec] = ("within", float(np.mean([o[dec][1] for o in perm_outs])))
        else:
            merged[dec] = ("cross",
                           float(np.mean([o[dec][1] for o in perm_outs])),
                           float(np.mean([o[dec][2] for o in perm_outs])))
    return merged


def _decode_session(name, configs, required):
    """Return {decoder: ('within',acc) | ('cross',train,test)} for one session.

    Decodes from NCAP.NUM_NEURONS neurons subsampled from the SNR-filtered set,
    averaged over NCAP.N_NEURON_PERM random permutations (matches the canonical
    SAT pipeline). Cached per session, keyed by the neuron-cap tag."""
    cf = CACHE / f"{name}.pkl"
    if cf.exists():
        return pickle.load(open(cf, "rb"))
    snr = load_snr_filtered_neuron_indices(snr_session_dir(name))
    if snr is None:
        return None
    try:
        L = load_loader(loader_pkl_path(name))
    except FileNotFoundError:
        return None
    dff = {k: np.asarray(v) for k, v in L.dff_stim_aligned.items() if hasattr(v, "shape")}
    fs = float(L.fs_image)
    del L
    gc.collect()
    if any(tt not in dff for tt in required):
        return None
    n_cells = dff[required[0]].shape[0]
    snr = snr[snr < n_cells]
    for tt in required:
        dff[tt] = dff[tt][snr]
    dff = P.per_trial_prestim_baseline_subtract_dff_dict(dff, fs, pre_stim_s=PRE_STIM_S)
    min_n = min(dff[tt].shape[-1] for tt in required)
    if min_n < P.N_CV_FOLDS:
        return None
    t_ax = P.make_time_axis(dff, fs)

    n_neurons = dff[required[0]].shape[0]
    perm_outs = [_decode_once({tt: arr[idx] for tt, arr in dff.items()}, fs, t_ax, min_n, configs)
                 for idx in NCAP.neuron_subsets(n_neurons)]
    out = _average_out(perm_outs)
    del dff
    gc.collect()
    pickle.dump(out, open(cf, "wb"))
    return out


def _agg(results, sessions):
    within = {}   # decoder -> list of acc
    cross = {}    # decoder -> {'train':[], 'test':[]}
    persess = []
    for name, r in zip(sessions, results):
        if r is None:
            continue
        for dec, val in r.items():
            if val[0] == "within":
                within.setdefault(dec, []).append(val[1])
                persess.append((dec, name, val[1]))
            else:
                cross.setdefault(dec, {"train": [], "test": []})
                cross[dec]["train"].append(val[1])
                cross[dec]["test"].append(val[2])
    return within, cross, persess


def _run(scope, sessions, configs, required):
    results = []
    for name in sessions:
        try:
            r = _decode_session(name, configs, required)
        except Exception as e:
            print(f"  ! {name}: {e}"); r = None
        results.append(r)
        print(f"  [{scope}] {'✓' if r else '!'} {name}", flush=True)
    return _agg(results, sessions)


def main():
    require(TDR_DIR, "TDR run (reversal session list)")
    rev_sessions = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())["session_names"]
    exp_sessions = json.loads((_REPO_ROOT / "metadata" / "SAT_good_expert_sessions.json").read_text())["sessions"]

    rev_within, rev_cross, rev_ps = _run("reversal", rev_sessions,
                                         P.DECODER_CONFIGS, P.ALL_REQUIRED)
    exp_within, _, exp_ps = _run("expert", exp_sessions,
                                 P.EXPERT_DECODER_CONFIGS, P.EXPERT_REQUIRED)

    std_rows, ps_rows, cross_rows = [], [], []
    for scope, within, ps in [("reversal", rev_within, rev_ps), ("expert", exp_within, exp_ps)]:
        for dec, vals in within.items():
            v = np.asarray(vals, float)
            std_rows.append((scope, dec, float(v.mean()),
                             float(scipy_sem(v)) if len(v) > 1 else 0.0, len(v)))
        for dec, name, acc in ps:
            ps_rows.append((scope, dec, name, acc))
    for dec, d in rev_cross.items():
        for split in ("train", "test"):
            v = np.asarray(d[split], float)
            cross_rows.append((dec, split, float(v.mean()),
                               float(scipy_sem(v)) if len(v) > 1 else 0.0, len(v)))

    pd.DataFrame(std_rows, columns=["scope", "decoder", "mean_accuracy", "sem", "n_sessions"]
                 ).to_csv(OUT / "decoding_timepooled.csv", index=False)
    pd.DataFrame(ps_rows, columns=["scope", "decoder", "session", "accuracy"]
                 ).to_csv(OUT / "decoding_timepooled_per_session.csv", index=False)
    pd.DataFrame(cross_rows, columns=["decoder", "split", "mean_accuracy", "sem", "n_sessions"]
                 ).to_csv(OUT / "decoding_timepooled_cross.csv", index=False)

    meta = {
        "analysis": "Time-pooled decoding (canonical time_pooled_decoder over [0,2000] ms)",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "response_window_ms": WINDOW_MS,
        "decoder_engine": "LinearDecoder.time_pooled_decoder / _cross_testing (linear SVM, 5-fold CV)",
        "preprocessing": "SNR cells (cellID-1); per-trial pre-stim baseline subtract "
                         f"(pre_stim={PRE_STIM_S}s) — matches the time-resolved recompute",
        **NCAP.meta(),
        "reversal_decoders": list(rev_within.keys()),
        "reversal_cross_decoders": list(rev_cross.keys()),
        "expert_decoders": list(exp_within.keys()),
        "chance_level": 0.5,
        "also_added_to": "analysis/run_time_pooled_no_baseline.py (stim_identity)",
    }
    (OUT / "decoding_timepooled_meta.json").write_text(json.dumps(meta, indent=2))
    print("Time-pooled decoding complete →", OUT)


if __name__ == "__main__":
    main()
