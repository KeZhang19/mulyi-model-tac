#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${1:-${REMOTE_HOST:-autodl}}"
REMOTE_DIR="${2:-${REMOTE_DIR:-~/BC_Tactile_Lab}}"

ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

rsync -az --human-readable --info=progress2 \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".pytest_cache/" \
  --exclude ".mypy_cache/" \
  --exclude ".ruff_cache/" \
  --exclude ".venv/" \
  --exclude "venv/" \
  --exclude "env/" \
  --exclude "env_*/" \
  --exclude "revolab/" \
  --exclude "logs/" \
  --exclude "model/" \
  --exclude "models/" \
  --exclude "outputs/" \
  --exclude "output/" \
  --exclude "integrate/output*/" \
  --exclude "scripts/force_map/official_replay/output*/" \
  --exclude "tacmap/output*/" \
  "$ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"
