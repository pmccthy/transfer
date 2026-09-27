#!/usr/bin/env bash
# Sync the real-data self-contained figure pipeline (results/transfer/) from
# neuronal-representations into data/experimental/.
#
# results/transfer/ is already self-contained: extract/ (code that reads the
# blessed source runs and writes tidy data/), figures/ (code that draws the
# figures from that tidy data), data/ (the tidy outputs themselves), plus
# README/provenance. No raw imaging data is touched by this sync.
#
# Known scratch/junk dirs (_to_delete, _stale, _exploration, _smoketest --
# flagged during the Aug/Sep cleanup pass, not real content) are excluded.
set -euo pipefail

SRC="${NEURONAL_REPO:-$HOME/Documents/neuronal-representations}"
DST="$(cd "$(dirname "$0")" && pwd)/data/experimental"

EXCLUDES=(
  --exclude='__pycache__/' --exclude='.DS_Store' --exclude='*.pyc'
  --exclude='_to_delete/' --exclude='_stale/' --exclude='_exploration/' --exclude='_smoketest/'
  --exclude='.fuse_hidden*'
  # CODE, not data -- synced into top-level extract/fig_gen/ instead, see
  # sync_code.py (called from sync_all.sh). figures/figs/ (actual rendered
  # output) is deliberately NOT excluded -- that's data.
  --exclude='/extract/'
  --exclude='/figures/panels/'
  --exclude='/figures/_common.py'
  --exclude='/figures/compose.py'
  --exclude='/figures/make_all_figures.py'
  --exclude='/figures/make_panels.py'
  --exclude='/figures/plot_*.py'
  --exclude='/figures/sat_plot_colours.py'
)

mkdir -p "$DST"

echo "== results/transfer/ -> data/experimental/ =="
rsync -avh "${EXCLUDES[@]}" "$@" "$SRC/results/transfer/" "$DST/"
