#!/usr/bin/env bash
# Sync the RNN-side pipeline from context-value-RNNs into data/rnns/.
#
# As of the "unify transfer results" commit (context-value-RNNs 639c5db,
# 1 Sep 2026), the pre-reversal pipeline, the reversal-learning pipeline, and
# the combined RNN+experiment analysis all live together under one
# results/transfer/ tree -- they used to be three separate top-level
# bundles (transfer_final/, reversal_study/, unified_figures/), which is
# what this script originally targeted and results/transfer_final/ still
# sitting on disk, abandoned/stale since Jun 26, is a leftover of. This
# script now pulls results/transfer/ itself (the pre-reversal side, minus
# its combined/ and reversal/ subfolders -- see sync_combined.sh and below)
# and results/transfer/reversal/ (the reversal-learning side) each as one
# whole directory, rather than enumerating every result subfolder by name --
# the old enumerated list had already drifted out of date (missing e.g.
# reproducibility/, reversal_2500_ckpt/, reversal_5000_ckpt/) and a fresh
# intervention/sweep study now only needs this script re-run, not edited.
#
# Known scratch/junk dirs (_to_delete, _smoketest, _mock_test,
# _stub_placeholder_not_real, _stale, _exploration -- flagged during the
# Aug/Sep cleanup pass, not real content) are excluded.
#
# Re-run any time the source repo changes; pass --dry-run (or -n) to preview.
set -euo pipefail

SRC="${CXVAL_REPO:-$HOME/Documents/context-value-RNNs}/results/transfer"
DST="$(cd "$(dirname "$0")" && pwd)/data/rnns"

TOP_EXCLUDES=(
  --exclude='__pycache__/' --exclude='.DS_Store' --exclude='*.pyc'
  --exclude='_to_delete/' --exclude='_smoketest/' --exclude='_mock_test/'
  --exclude='_stub_placeholder_not_real/' --exclude='_stale/' --exclude='_exploration/'
  --exclude='smoketest/'
  --exclude='/combined/'   # synced separately, see sync_combined.sh
  --exclude='/reversal/'   # synced separately, below
  --exclude='/code/'       # CODE, not data -- synced into top-level extract/analysis/fig_gen/
                            # instead, see sync_code.py (called from sync_all.sh)
)
REV_EXCLUDES=(
  --exclude='__pycache__/' --exclude='.DS_Store' --exclude='*.pyc'
  --exclude='_to_delete/' --exclude='_smoketest/' --exclude='_mock_test/'
  --exclude='_stub_placeholder_not_real/' --exclude='_stale/' --exclude='_exploration/'
  --exclude='smoketest/'
  # reversal/code/'s analysis+fig-gen .py/.mplstyle files -- CODE, not data --
  # synced into top-level analysis/fig_gen/ instead, see sync_code.py. The
  # run_*_full.sh / run_reward_scale_*.sh sweep launchers in that same
  # directory are reproducibility/training material and stay put.
  --exclude='/code/*.py'
  --exclude='/code/*.mplstyle'
)

mkdir -p "$DST" "$DST/reversal"

echo "== results/transfer/ (pre-reversal side, minus combined/ + reversal/) -> data/rnns/ =="
rsync -avh "${TOP_EXCLUDES[@]}" "$@" "$SRC/" "$DST/"

echo
echo "== results/transfer/reversal/ (reversal-learning side, whole) -> data/rnns/reversal/ =="
rsync -avh "${REV_EXCLUDES[@]}" "$@" "$SRC/reversal/" "$DST/reversal/"
