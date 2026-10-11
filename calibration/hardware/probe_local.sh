#!/usr/bin/env bash
# Local SDK environment; the Python probe performs bounded, read-only acquisition.
set -euo pipefail
calibration_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sdk_python="${REVO3_PYTHON:-/home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python}"
if [[ ! -x "$sdk_python" ]]; then
  echo "SDK Python not found: $sdk_python; set REVO3_PYTHON to a compatible interpreter." >&2
  exit 1
fi
exec "$sdk_python" "$calibration_dir/probe_revo3_readonly.py" "$@"
