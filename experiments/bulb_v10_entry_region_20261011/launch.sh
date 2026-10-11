#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007
export CUDA_VISIBLE_DEVICES=0,1 OMNI_KIT_ALLOW_ROOT=1 OMP_NUM_THREADS=8
export ITERATIONS=28900
export RUN_DIR=/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007/logs/rsl_rl/bulb_grasp_teacher_v10_mlp/2026-10-11_resume_model1100_2x8192
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to reuse existing run: $RUN_DIR" >&2
    exit 1
fi
exec bash scripts/rsl_rl/launch_bulb_grasp_teacher_2gpu.sh --num_envs 8192 --initial_adr_difficulty 0 --physx_buffer_scale 2 --physx_collision_stack_size 4026531840 --checkpoint /home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007/logs/rsl_rl/bulb_grasp_teacher_v9_mlp/2026-10-11_resume_model900_2x8192/rank0/model_1100.pt --migrate-v9-entry-window
