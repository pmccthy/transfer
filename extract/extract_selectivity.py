"""Significant-responder / selectivity extraction (reuses repo selectivity code).

Uses the canonical ``run_selectivity_index_analysis.compute_selectivity`` (variant
``trial_mean_abs3``: trial-mean baseline z-score, 3-way absolute SI, permutation
p-values over the 0-2 s window). For each neuron it gives the preferred stimulus,
a z-scored response to each stimulus (responsiveness), and a permutation p-value;
``significant responder`` = p < ALPHA.

Responses are built directly from the correctly-split (pre-Rev)/(post-Rev) dF/F
keys (the loader beh_df lacks reversalNumber, so build_responses' auto-split is
avoided). SNR-filtered cells (reversal: TDR filtered_cell_indices; expert:
filtered_cells==1), matching the transfer heatmaps/population.

Flexibility: the whole responsiveness definition is the repo variant + ALPHA
below. Change VARIANT / ALPHA / N_PERM and re-run; every Subgroups figure reads
these CSVs, so nothing else needs editing.

Outputs (results/transfer/data/subgroups/):
  selectivity_reversal_responses.csv  session, neuron_id, phase, stimulus, R
  selectivity_reversal_groups.csv     session, neuron_id, pref_pre, pval_pre, sig_pre,
                                      pref_post, pval_post, sig_post
  selectivity_expert_responses.csv    session, neuron_id, stimulus, R
  selectivity_expert_groups.csv       session, neuron_id, pref, pval, sig
  selectivity_meta.json
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
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analysis"))
import run_selectivity_index_analysis as S  # noqa: E402

OUT = DATA_ROOT / "subgroups"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("TRANSFER_DECODE_CACHE", "/tmp/transfer_decode_cache")) / "selectivity"
CACHE.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
N_PERM = 1000
ZFUNC = S.baseline_zscore_trial_mean_across_trial_std  # 'trial_mean' variant
ABS3 = True                                            # 'abs3' variant
PRE_STIM_S = 3.0

REV_STIMS = ["100%-->0%", "50%", "0%-->100%"]
REV_LABELS = ["100_to_0", "50", "0_to_100"]
EXP_STIMS = ["100%", "50%", "0%"]
EXP_LABELS = ["100", "50", "0"]


def _t_axis(n_tp, fs):
    pf = int(np.ceil(PRE_STIM_S * fs))
    return (np.arange(n_tp) - pf) / fs, pf


def _sel(responses, phase, stim_order, fs):
    n_tp = next(v[phase] for v in responses.values() if v[phase] is not None).shape[1]
    t, pf = _t_axis(n_tp, fs)
    return S.compute_selectivity(responses, phase, stim_order, t, S.SEL_WINDOW, pf,
                                 ZFUNC, use_abs3_si=ABS3, n_perm=N_PERM,
                                 rng=np.random.default_rng(0))


def _reversal_session(name):
    cf = CACHE / f"rev_{name}.pkl"
    if cf.exists():
        return pickle.load(open(cf, "rb"))
    tdr = pickle.load(open(TDR_DIR / f"{name}_tdr_v2.pkl", "rb"))
    snr = np.asarray(tdr["filtered_cell_indices"]).astype(int)
    L = load_loader(loader_pkl_path(name))
    dff = L.dff_stim_aligned
    fs = float(L.fs_image)
    responses = {}
    for stim in REV_STIMS:
        pre_k, post_k = f"{stim} (pre-Rev)", f"{stim} (post-Rev)"
        if pre_k not in dff or post_k not in dff:
            return None
        responses[stim] = {"pre": np.asarray(dff[pre_k])[snr],
                           "post": np.asarray(dff[post_k])[snr]}
    del L
    gc.collect()
    sel_pre = _sel(responses, "pre", REV_STIMS, fs)
    sel_post = _sel(responses, "post", REV_STIMS, fs)
    out = {"n": len(snr), "pre": sel_pre, "post": sel_post}
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
        return None
    responses = {s: {"pre": np.asarray(dff[s])[snr], "post": None} for s in EXP_STIMS}
    del L
    gc.collect()
    sel = _sel(responses, "pre", EXP_STIMS, fs)
    out = {"n": len(snr), "sel": sel}
    pickle.dump(out, open(cf, "wb"))
    return out


def main():
    require(TDR_DIR, "TDR run"); require(LOADER_ROOT, "loader root")
    rev_names = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())["session_names"]
    exp_names = json.loads((_REPO_ROOT / "metadata" / "SAT_good_expert_sessions.json").read_text())["sessions"]

    rev_resp, rev_grp = [], []
    for name in rev_names:
        r = _reversal_session(name)
        print(f"  [rev] {'✓' if r else '!'} {name}", flush=True)
        if r is None:
            continue
        for nid in range(r["n"]):
            for phase, sel, labels in [("pre", r["pre"], REV_LABELS), ("post", r["post"], REV_LABELS)]:
                for si, lab in enumerate(labels):
                    rev_resp.append((name, nid, phase, lab, float(sel["response_matrix"][nid, si])))
            rev_grp.append((name, nid,
                            REV_LABELS[r["pre"]["pref_stim_idx"][nid]], float(r["pre"]["pvalues"][nid]),
                            bool(r["pre"]["pvalues"][nid] < ALPHA),
                            REV_LABELS[r["post"]["pref_stim_idx"][nid]], float(r["post"]["pvalues"][nid]),
                            bool(r["post"]["pvalues"][nid] < ALPHA)))
    pd.DataFrame(rev_resp, columns=["session", "neuron_id", "phase", "stimulus", "R"]
                 ).to_csv(OUT / "selectivity_reversal_responses.csv", index=False)
    pd.DataFrame(rev_grp, columns=["session", "neuron_id", "pref_pre", "pval_pre", "sig_pre",
                                   "pref_post", "pval_post", "sig_post"]
                 ).to_csv(OUT / "selectivity_reversal_groups.csv", index=False)

    exp_resp, exp_grp = [], []
    for name in exp_names:
        try:
            r = _expert_session(name)
        except Exception as e:
            print(f"  [exp] ! {name}: {e}"); r = None
        print(f"  [exp] {'✓' if r else '!'} {name}", flush=True)
        if r is None:
            continue
        sel = r["sel"]
        for nid in range(r["n"]):
            for si, lab in enumerate(EXP_LABELS):
                exp_resp.append((name, nid, lab, float(sel["response_matrix"][nid, si])))
            exp_grp.append((name, nid, EXP_LABELS[sel["pref_stim_idx"][nid]],
                            float(sel["pvalues"][nid]), bool(sel["pvalues"][nid] < ALPHA)))
    pd.DataFrame(exp_resp, columns=["session", "neuron_id", "stimulus", "R"]
                 ).to_csv(OUT / "selectivity_expert_responses.csv", index=False)
    pd.DataFrame(exp_grp, columns=["session", "neuron_id", "pref", "pval", "sig"]
                 ).to_csv(OUT / "selectivity_expert_groups.csv", index=False)

    meta = {
        "analysis": "Significant responders / selectivity (repo compute_selectivity)",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "variant": "trial_mean_abs3", "alpha": ALPHA, "n_perm": N_PERM,
        "window_s": list(S.SEL_WINDOW),
        "significant_responder": f"permutation p < {ALPHA} for the neuron's max selectivity",
        "responsiveness_metric": "max over stimuli of the z-scored response R (response_matrix)",
        "stim_labels_reversal": dict(zip(REV_STIMS, REV_LABELS)),
        "stim_labels_expert": dict(zip(EXP_STIMS, EXP_LABELS)),
    }
    (OUT / "selectivity_meta.json").write_text(json.dumps(meta, indent=2))
    print("Selectivity extraction complete →", OUT)


if __name__ == "__main__":
    main()
