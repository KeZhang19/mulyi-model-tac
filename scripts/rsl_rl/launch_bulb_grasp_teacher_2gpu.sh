#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${BULB_TEACHER_PYTHON:-/home/admin/workspace/xinyu/brainco_isaaclab_2_3_2_env/.pixi/envs/default/bin/python}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/admin/workspace/xinyu/IsaacLab-d94504bc}"
export PYTHONPATH="$ROOT/source/BrainCo_DexHand${PYTHONPATH:+:$PYTHONPATH}"
# Task-owned, pinned V2 teacher RSL-RL fork; no global environment replacement.
export PYTHONPATH="$ROOT/runtime/bulb_grasp_teacher:$PYTHONPATH"
for package in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks; do
    export PYTHONPATH="$ISAACLAB_ROOT/source/$package:$PYTHONPATH"
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TERM=xterm
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export BULB_TEACHER_PORTABLE_ROOT="$(mktemp -d /tmp/bulb-grasp-teacher-XXXXXX)"
export BULB_TEACHER_TRAIN_SCRIPT="$ROOT/scripts/rsl_rl/train_bulb_grasp_teacher.py"
export BULB_TEACHER_RANK_STAGGER_SECONDS=20
cd "$ROOT"
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/rsl_rl/bulb_grasp_teacher_split_gpu.py \
    --distributed --expected_world_size 2 --num_envs 4096 --max_iterations "${ITERATIONS:-30000}" \
    --seed 42 --device cuda:0 --headless \
    --log_dir "${RUN_DIR:-$ROOT/logs/rsl_rl/bulb_grasp_teacher_v2test/$(date +%Y-%m-%d_%H-%M-%S)_scratch_2x4096}" "$@"
