#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./calib.sh <finger> <segment>

Arguments:
  finger   one of: middle index ring pinky thumb
  segment  one of: mcp pip

Example:
  ./calib.sh index mcp
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

finger="$1"
segment="$2"
shift 2

case "$finger" in
  middle|index|ring|pinky|thumb) ;;
  *) echo "Invalid finger: $finger" >&2; usage; exit 2 ;;
esac

case "$segment" in
  mcp|pip) ;;
  *) echo "Invalid segment: $segment" >&2; usage; exit 2 ;;
esac

run_id="${finger}_${segment}"
python_bin="${PYTHON:-python}"

exec "$python_bin" scripts/rsl_rl/run_rl_ball_probe_pressure_calib.py \
 --task BrainCo-Dexsuite-Revo3-Right-Lift-v0 \
 --num_envs 1 \
 --focus-finger "$finger" \
 --pressure-pad-segment "$segment" \
 --press-start-offset 0.025 \
 --press-contact-search-distance 0.02 \
 --press-indent-depth 0.0005 \
 --pressure-pad-press-steps 26 \
 --presser-axis-max-speed 0.01 \
 --enable-presser-collision \
 --lock-viewport-camera \
 --pressure-pad-contact-offset 0.0005 \
 --pressure-pad-rest-offset -0.0017 \
 --pressure-pad-taxel-surface-offset 0.0 \
 --save-pressure-trace \
 --pressure-trace-dir "outputs/rl_ball_probe_pressure_calib/${run_id}" \
 --pressure-trace-run-id "$run_id" \
 --show-pressure-window \
 "$@"
