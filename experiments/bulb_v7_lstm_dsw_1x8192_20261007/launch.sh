#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/admin/workspace/xinyu/mulyi-model-tac-v7-lstm-20261007
PYTHON=/home/admin/workspace/xinyu/brainco_isaaclab_2_3_2_env/.pixi/envs/default/bin/python
ISAACLAB_ROOT=/home/admin/workspace/xinyu/IsaacLab-d94504bc
export CUDA_VISIBLE_DEVICES=0
export OMNI_KIT_ALLOW_ROOT=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TERM=xterm
export OMP_NUM_THREADS=8
export PYTHONPATH="$ROOT/runtime/bulb_grasp_teacher:$ROOT/source/BrainCo_DexHand"
for package in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks; do
    export PYTHONPATH="$ISAACLAB_ROOT/source/$package:$PYTHONPATH"
done
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
export BULB_TEACHER_PORTABLE_ROOT="$(mktemp -d /tmp/bulb-v7-lstm-XXXXXXXX)"
export BULB_TEACHER_TRAIN_SCRIPT="$ROOT/scripts/rsl_rl/train_bulb_grasp_teacher.py"
export BULB_TEACHER_RANK_STAGGER_SECONDS=0
cd "$ROOT"
exec "$PYTHON" -u scripts/rsl_rl/bulb_grasp_teacher_split_gpu.py \
    --lstm --expected_world_size 1 --num_envs 8192 --max_iterations 30000 \
    --seed 42 --device cuda:0 --headless \
    --physx_buffer_scale 2 --physx_collision_stack_size 4026531840 \
    --log_dir "$ROOT/logs/rsl_rl/bulb_grasp_teacher_v7_lstm/2026-10-07_scratch_1x8192"
