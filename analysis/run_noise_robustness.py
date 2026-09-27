#!/usr/bin/env python3
"""Noise-robustness of stimulus decoding, per model type: does adding noise
to the (test-set) hidden activations degrade stimulus decoding accuracy, and
does it do so in a stimulus-specific way -- i.e. do particular stimuli get
systematically misclassified as particular OTHER stimuli once corrupted, or
does noise just push everything toward chance uniformly? Requested as a way
to probe whether some representational geometry (rather than others) makes
a model's stimulus code more robust to noise.

Method, per model_type, per (pre-reversal, "expert") seed:
  1. Load that seed's time_resolved npz (unified_figures/transfer/
     figure_data/time_resolved/<model_type>_seed<N>.npz -- the SAME
     per-trial hidden-state data used by run_decoding.py's pairwise_decode,
     via act_dict_adapter), average over the stim-presentation window
     ("average" pooling, matching the existing decode convention) to get
     one (n_trials, n_units) feature matrix and one (n_trials,) stimulus
     label array (3-way: 0%/50%/100%).
  2. ONE stratified train/test split (70/30, random_state=42) -- NOT
     k-fold, since here we need a single fixed held-out "test set" whose
     activations we then corrupt with noise (the analysis this was
     requested for), not a cross-validated accuracy estimate.
  3. Fit StandardScaler + LinearSVC (same classifier family as cxval.
     analysis.pairwise_decode) on the CLEAN train split.
  4. For each noise magnitude sigma_mult in NOISE_LEVELS: add iid Gaussian
     noise to the TEST split only, PER-UNIT SCALED by that unit's own
     train-set activation std (so sigma_mult is a unit-free "how many
     typical-activity-units of noise", comparable across units with very
     different native scales) -- i.e. noisy_X = X_test + sigma_mult *
     train_std * N(0, 1), independently per unit per trial. Averaged over
     N_REPEATS independent noise draws per level (cheap -- no retraining,
     just re-predicting) to smooth the curve.
  5. Record, per level: overall accuracy, per-true-stimulus accuracy
     (recall -- "of trials whose real stimulus was X, what fraction did
     the decoder still call X"), and a row-normalized (by true class)
     confusion matrix.

Aggregation across seeds: per-seed values (overall acc, per-stim acc,
row-normalized confusion matrices) are each averaged UNWEIGHTED across
seeds (one seed, one vote -- same convention used throughout this codebase,
e.g. vigour_value/coding_angle_comparison), not pooled trial-by-trial.

Output: figs/noise_robustness/noise_robustness_summary.json -- small
(aggregated only, no per-trial data), read by panels/noise_robustness.py.

Needs sklearn (not available in the sandbox this was drafted in). Run in
your cxval env:

    cd context-value-RNNs
    python3 unified_figures/analysis/run_noise_robustness.py \
        --figure-data unified_figures/transfer/figure_data \
        --out unified_figures/figs/noise_robustness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import act_dict_adapter as AD  # noqa: E402

MODEL_TYPES = ["rl_only", "classif_rl", "classif_rl_readout_only"]
STIM_LABELS = ["0%", "50%", "100%"]
NOISE_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0]
N_REPEATS = 8
TEST_SIZE = 0.3
RANDOM_STATE = 42
# Shared noise magnitude (sigma_mult, same units as NOISE_LEVELS) used for
# the single "representative" confusion matrix shown per model in the main
# 6-panel figure -- chosen post-hoc (after inspecting the accuracy-vs-noise
# curves) as a level where every model type shows clearly visible but not
# yet ceiling/floor-saturated degradation. See panels/noise_robustness.py.
CONFUSION_LEVEL = 8.0


def _one_seed(npz_path, rng):
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    act = AD.act_dict_from_npz(npz_path)
    X = act["stim_hidden"].mean(axis=1)          # (n_trials, n_units) -- "average" pooling
    y = act["stimulus"].astype(int)               # (n_trials,) in {0,1,2}
    n_stim = int(y.max()) + 1

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    sc = StandardScaler().fit(X_tr)
    clf = LinearSVC(max_iter=5000, dual="auto").fit(sc.transform(X_tr), y_tr)

    train_std = X_tr.std(axis=0)                   # (n_units,) per-unit noise scale
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)

    overall = []
    per_stim = {s: [] for s in range(n_stim)}
    confusions = {}   # level -> (n_stim, n_stim) row-normalized, averaged over repeats
    for level in NOISE_LEVELS:
        accs = []
        per_stim_accs = {s: [] for s in range(n_stim)}
        conf_sum = np.zeros((n_stim, n_stim), dtype=float)
        conf_row_n = np.zeros(n_stim, dtype=float)
        for _ in range(N_REPEATS):
            if level == 0.0:
                X_noisy = X_te
            else:
                noise = rng.normal(0.0, 1.0, size=X_te.shape) * (level * train_std)
                X_noisy = X_te + noise
            y_pred = clf.predict(sc.transform(X_noisy))
            accs.append(float((y_pred == y_te).mean()))
            for s in range(n_stim):
                m = y_te == s
                if m.sum() == 0:
                    continue
                per_stim_accs[s].append(float((y_pred[m] == s).mean()))
                for p in range(n_stim):
                    conf_sum[s, p] += float((y_pred[m] == p).sum())
                conf_row_n[s] += m.sum()
        overall.append(float(np.mean(accs)))
        for s in range(n_stim):
            per_stim[s].append(float(np.mean(per_stim_accs[s])) if per_stim_accs[s] else float("nan"))
        row_n = np.where(conf_row_n == 0, 1.0, conf_row_n)[:, None]
        confusions[level] = (conf_sum / row_n).tolist()

    return {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "n_stim": n_stim,
        "overall_acc": overall, "per_stim_acc": {str(s): v for s, v in per_stim.items()},
        "confusion": {str(k): v for k, v in confusions.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure-data", default=str(_HERE.parent / "transfer" / "figure_data"))
    ap.add_argument("--out", default=str(_HERE.parent / "figs" / "noise_robustness"))
    ap.add_argument("--model-types", nargs="*", default=MODEL_TYPES)
    args = ap.parse_args()

    td = Path(args.figure_data) / "time_resolved"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"noise_levels": NOISE_LEVELS, "confusion_level": CONFUSION_LEVEL,
               "stim_labels": STIM_LABELS, "models": {}}
    for mt in args.model_types:
        files = AD.seed_files(td, mt)
        print(f"{mt}: {len(files)} seeds")
        seed_results = {}
        for seed, path in sorted(files.items()):
            rng = np.random.default_rng(RANDOM_STATE + seed)
            try:
                seed_results[str(seed)] = _one_seed(path, rng)
                print(f"  seed {seed}: n_test={seed_results[str(seed)]['n_test']}  "
                      f"clean_acc={seed_results[str(seed)]['overall_acc'][0]:.3f}")
            except Exception as e:
                print(f"  seed {seed}: SKIPPED ({e})")
        results["models"][mt] = seed_results

    out_path = out_dir / "noise_robustness_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
