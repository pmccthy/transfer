#!/usr/bin/env bash
# Sync the cross-model-vs-experiment ("combined") figure-generation code from
# context-value-RNNs/unified_figures/ into data/combined/.
#
# This is for figures that combine analysis of the experimental and RNN data
# together (e.g. the chi-squared responder-group fit against real data) --
# as opposed to data/rnns/ and data/experimental/, which are each one side
# alone.
#
# Three things are excluded because they're shims into content synced
# elsewhere, not unique to this side:
#   unified_figures/transfer/        symlinks -> transfer_final/           (now data/rnns/transfer_final/)
#   unified_figures/reversal/        symlinks -> reversal_study/results/   (now data/rnns/reversal_study/)
#   unified_figures/experiment_data  symlink  -> neuronal-representations/results/transfer  (now data/experimental/)
# Paths inside compose.py/make_panels.py that currently resolve relative to
# unified_figures/ (or via that absolute experiment_data symlink) will need
# repointing at ../rnns/transfer_final, ../rnns/reversal_study, and
# ../experimental once this lands -- that's fig_gen/combined/ work, not part
# of this raw sync.
set -euo pipefail

SRC="${CXVAL_REPO:-$HOME/Documents/context-value-RNNs}/unified_figures"
DST="$(cd "$(dirname "$0")" && pwd)/data/combined"

EXCLUDES=(
  --exclude='__pycache__/' --exclude='.DS_Store' --exclude='*.pyc'
  --exclude='/transfer/'
  --exclude='/reversal/'
  --exclude='/experiment_data'
)

mkdir -p "$DST"

echo "== unified_figures/ (minus transfer/, reversal/, experiment_data shims) -> data/combined/ =="
rsync -avh "${EXCLUDES[@]}" "$@" "$SRC/" "$DST/"
