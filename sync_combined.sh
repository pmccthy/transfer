#!/usr/bin/env bash
# Sync the cross-model-vs-experiment ("combined") figure-generation code from
# context-value-RNNs/results/transfer/combined/ into data/combined/.
#
# This is for figures that combine analysis of the experimental and RNN data
# together (e.g. the chi-squared responder-group fit against real data) --
# as opposed to data/rnns/ and data/experimental/, which are each one side
# alone.
#
# One thing is excluded because it's a shim into content synced elsewhere,
# not unique to this side:
#   combined/transfer/   relative symlinks -> ../../{code,figure_data,
#                         model_runs_ckpt,reversal/...}  (now data/rnns/...
#                         via sync_rnns.sh)
# The experimental side isn't symlinked in at all -- combined/
# cross_model_vs_experiment/*.py reads it via an absolute NEURONAL_REPO path
# (default ~/Documents/neuronal-representations/results/transfer), i.e.
# data/experimental/ once synced by sync_experimental.sh, not a path inside
# this repo.
#
# Paths inside compose.py/make_panels.py that currently resolve relative to
# results/transfer/combined/transfer/ (the symlink shim above) or via
# NEURONAL_REPO will need repointing at ../rnns/... and ../experimental
# once this lands -- that's fig_gen/combined/ work, not part of this raw
# sync.
#
# Known scratch/junk dirs under figs/ (_to_delete, _smoketest, _mock_test,
# _stub_placeholder_not_real -- flagged during the Aug/Sep cleanup pass, not
# real content) are excluded.
set -euo pipefail

SRC="${CXVAL_REPO:-$HOME/Documents/context-value-RNNs}/results/transfer/combined"
DST="$(cd "$(dirname "$0")" && pwd)/data/combined"

EXCLUDES=(
  --exclude='__pycache__/' --exclude='.DS_Store' --exclude='*.pyc'
  --exclude='_to_delete/' --exclude='_smoketest/' --exclude='_mock_test/'
  --exclude='_stub_placeholder_not_real/' --exclude='smoketest/'
  --exclude='/transfer/'
  # CODE, not data -- synced into top-level analysis/fig_gen/ instead, see
  # sync_code.py (called from sync_all.sh). cross_model_vs_experiment/'s
  # group_counts/ and README.md are DATA and deliberately kept here.
  --exclude='/analysis/'
  --exclude='/cross_model_vs_experiment/*.py'
  --exclude='/panels/'
  --exclude='/style/'
  --exclude='/compose.py'
  --exclude='/make_panels.py'
)

mkdir -p "$DST"

echo "== results/transfer/combined/ (minus transfer/ shim) -> data/combined/ =="
rsync -avh "${EXCLUDES[@]}" "$@" "$SRC/" "$DST/"
