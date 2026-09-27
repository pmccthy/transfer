#!/usr/bin/env bash
# Run all three side syncs, then refresh figs/special_subset/ from the fresh
# copies. Pass --dry-run (or -n) to preview the three syncs only -- special_subset
# isn't dry-run-aware (it's a plain copy, not rsync), so it's skipped in that case.
set -euo pipefail
cd "$(dirname "$0")"
bash sync_rnns.sh "$@"
echo
bash sync_experimental.sh "$@"
echo
bash sync_combined.sh "$@"

echo
echo "== code (extract/, analysis/, fig_gen/) =="
python3 sync_code.py "$@"

DRY=0
for a in "$@"; do [ "$a" = "--dry-run" ] || [ "$a" = "-n" ] && DRY=1; done
if [ "$DRY" = 0 ]; then
  echo
  echo "== figs/special_subset/ =="
  python3 fig_gen/sync_special_subset.py
fi
