#!/usr/bin/env bash
# Two independent Isaac processes, 512 environments per GPU, shared PPO updates.
set -euo pipefail
d_peg_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$d_peg_repo"
d_peg_python="${D_PEG_PYTHON:-/home/admin/workspace/xinyu/IsaacLab/_isaac_sim/python.sh}"
d_peg_log_dir="${1:-logs/rsl_rl/dexsuite_revo3_insert_d_peg_v3/dual_gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
export PYTHONPATH="$d_peg_repo/source/BrainCo_DexHand:$d_peg_repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "$d_peg_python" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0 \
  --distributed --expected_world_size 2 --num_envs "${D_PEG_ENVS_PER_GPU:-512}" --headless \
  --logger tensorboard --max_iterations "${D_PEG_MAX_ITERATIONS:-15000}" --seed 7 \
  --log_dir "$d_peg_log_dir"
