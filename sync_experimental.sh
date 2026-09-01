#!/usr/bin/env bash
# Sync the real-data self-contained figure pipeline (results/transfer/) from
# neuronal-representations into data/experimental/.
#
# results/transfer/ is already self-contained: extract/ (code that reads the
# blessed source runs and writes tidy data/), figures/ (code that draws the
# figures from that tidy data), data/ (the tidy outputs themselves), plus
# README/provenance. No raw imaging data is touched by this sync.
set -euo pipefail

SRC="${NEURONAL_REPO:-$HOME/Documents/neuronal-representations}"
DST="$(cd "$(dirname "$0")" && pwd)/data/experimental"

EXCLUDES=(--exclude='__pycache__/' --exclude='.DS_Store' --exclude='*.pyc')

mkdir -p "$DST"

echo "== results/transfer/ -> data/experimental/ =="
rsync -avh "${EXCLUDES[@]}" "$@" "$SRC/results/transfer/" "$DST/"
