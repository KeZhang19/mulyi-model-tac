#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/admin/workspace/xinyu/mulyi-model-tac-v6-20261007
export CUDA_VISIBLE_DEVICES=0,1
export OMNI_KIT_ALLOW_ROOT=1
export OMP_NUM_THREADS=8
export ITERATIONS=20500
export RUN_DIR="$PWD/logs/rsl_rl/bulb_grasp_teacher_v6/2026-10-07_resume9500_2x8192"
exec bash scripts/rsl_rl/launch_bulb_grasp_teacher_2gpu.sh \
    --checkpoint /home/admin/workspace/xinyu/mulyi-model-tac-v5-20261006/logs/rsl_rl/bulb_grasp_teacher_v5/2026-10-06_scratch_2x8192/rank0/model_9500.pt \
    --num_envs 8192 \
    --initial_adr_difficulty 0 \
    --physx_buffer_scale 2 \
    --physx_collision_stack_size 4026531840
