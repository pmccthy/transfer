#!/usr/bin/env bash
# Run all three side syncs. Pass --dry-run (or -n) to preview only.
set -euo pipefail
cd "$(dirname "$0")"
bash sync_rnns.sh "$@"
echo
bash sync_experimental.sh "$@"
echo
bash sync_combined.sh "$@"
