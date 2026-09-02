#!/usr/bin/env bash
set -euo pipefail

DISPLAY="${DISPLAY_OVERRIDE:-:1}"
XAUTHORITY="${XAUTHORITY_OVERRIDE:-/run/user/1000/gdm/Xauthority}"
OUTPUT="${PRIME_OUTPUT:-HDMI-1-0}"

env DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
  xrandr --output "$OUTPUT" --set "PRIME Synchronization" 1
