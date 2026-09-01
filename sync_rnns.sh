#!/usr/bin/env bash
# Sync + reorganize the RNN-side pipelines from context-value-RNNs into
# data/rnns/, merging transfer_final/ (pre-reversal) and reversal_study/
# (reversal-learning: baseline + intervention studies) into one flat layout.
#
# Why merged rather than mirrored 1:1: checked by MD5, transfer_final/'s own
# model_runs_reversal(_5k)/figure_data_reversal(_5k)/figures_reversal(_5k)
# are byte-identical duplicates of reversal_study/results/reversal_2500 and
# reversal_5000; going the other way, reversal_study/results/model_runs and
# figure_data are byte-identical duplicates of transfer_final's copies. Each
# side's non-duplicate content is synced once, into data/rnns/ (pre-reversal)
# and data/rnns/reversal/ (everything reversal-related, including the 8
# intervention/pilot studies and RSA/population-similarity analyses that
# only exist under reversal_study/).
#
# reversal_study/reproducibility/ is skipped entirely: its training/ is
# empty, and its cxval/, inference/ are stale duplicates of transfer_final's
# copies. The scripts that actually produced every run under
# data/rnns/reversal/ live at the repo root, scripts/16_06_26_*.py -- pulled
# in separately below, renamed into data/rnns/reversal/reproducibility/.
#
# Re-run any time the source repo changes; pass --dry-run (or -n) to preview.
set -euo pipefail

SRC="${CXVAL_REPO:-$HOME/Documents/context-value-RNNs}"
DST="$(cd "$(dirname "$0")" && pwd)/data/rnns"

EXCLUDES=(--exclude='__pycache__/' --exclude='.DS_Store' --exclude='*.pyc')

mkdir -p "$DST" "$DST/reversal/reproducibility"

echo "== transfer_final/ (pre-reversal side) -> data/rnns/ =="
for d in code reproducibility model_runs figure_data figures; do
  mkdir -p "$DST/$d"
  rsync -avh "${EXCLUDES[@]}" "$@" "$SRC/transfer_final/$d/" "$DST/$d/"
done
rsync -avh "$@" "$SRC/transfer_final/README.md" "$SRC/transfer_final/environment.yml" "$SRC/transfer_final/requirements.txt" "$DST/"

echo
echo "== reversal_study/ (reversal-learning side) -> data/rnns/reversal/ =="
mkdir -p "$DST/reversal/code"
rsync -avh "${EXCLUDES[@]}" "$@" "$SRC/reversal_study/code/" "$DST/reversal/code/"
rsync -avh "$@" "$SRC/reversal_study/README.md" "$DST/reversal/"
for d in reversal_2500 reversal_5000 population_similarity rsa \
         action_std_0p15_full action_std_pilot min_vigour_0p1_full \
         squash_1p3_full reward_scale_intervention terminal_rpe \
         terminal_value_minvig vigour_floor_pilot; do
  mkdir -p "$DST/reversal/$d"
  rsync -avh "${EXCLUDES[@]}" "$@" "$SRC/reversal_study/results/$d/" "$DST/reversal/$d/"
done

echo
echo "== reversal training/inference scripts (repo-root scripts/, not reversal_study/) -> data/rnns/reversal/reproducibility/ =="
for f in train_model train_reversal run_inference; do
  rsync -avh "$@" "$SRC/scripts/16_06_26_${f}.py" "$DST/reversal/reproducibility/${f}.py"
done
