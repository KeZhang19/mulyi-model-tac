#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/admin/workspace/xinyu/mulyi-model-tac-v5-20261006
export CUDA_VISIBLE_DEVICES=0,1
export OMNI_KIT_ALLOW_ROOT=1
export OMP_NUM_THREADS=8
export ITERATIONS=30000
export RUN_DIR="$PWD/logs/rsl_rl/bulb_grasp_teacher_v5/2026-10-06_scratch_2x8192"
exec bash scripts/rsl_rl/launch_bulb_grasp_teacher_2gpu.sh \
    --num_envs 8192 \
    --physx_buffer_scale 2 \
    --physx_collision_stack_size 4026531840
