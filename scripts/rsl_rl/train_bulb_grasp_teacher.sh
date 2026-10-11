#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${BULB_TEACHER_PYTHON:-/home/liuxinyu/miniconda3/envs/brainco/bin/python}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/liuxinyu/docker_workplace/IsaacLab}"
export PYTHONPATH="$ROOT/source/BrainCo_DexHand${PYTHONPATH:+:$PYTHONPATH}"
for package in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks; do
    export PYTHONPATH="$ISAACLAB_ROOT/source/$package:$PYTHONPATH"
done
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$ROOT"
exec "$PYTHON" scripts/rsl_rl/train_bulb_grasp_teacher.py "$@"
