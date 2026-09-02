#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${1:-${REMOTE_HOST:-autodl}}"
REMOTE_DIR="${2:-${REMOTE_DIR:-~/BC_Tactile_Lab}}"

paths=(
  logs
  model
  models
  outputs
  output
  integrate/output
  scripts/force_map/official_replay/output
  tacmap/output
)

for path in "${paths[@]}"; do
  if ssh "$REMOTE_HOST" "test -e $REMOTE_DIR/$path"; then
    mkdir -p "$ROOT/$(dirname "$path")"
    rsync -az --human-readable --info=progress2 \
      "$REMOTE_HOST:$REMOTE_DIR/$path" "$ROOT/$(dirname "$path")/"
  fi
done
