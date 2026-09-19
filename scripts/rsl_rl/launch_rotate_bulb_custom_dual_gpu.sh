#!/usr/bin/env bash
# Custom bulb task: 4096 environments per GPU and a 2 GiB collision stack.
set -euo pipefail
bulb_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$bulb_repo"
bulb_python="${BULB_PYTHON:-/home/admin/workspace/xinyu/IsaacLab/_isaac_sim/python.sh}"
# Preserve the RSL-RL checkout selected by this server's original launcher.
bulb_rsl_rl="${BULB_RSL_RL_PATH:-/home/admin/workspace/xinyu/HCL_DexHand_arm/HCL_DexHand/rsl_rl}"
bulb_log_dir="${1:-logs/rsl_rl/dexsuite_revo3_rotate_bulb_custom/$(date -u +%Y%m%dT%H%M%SZ)_nomodal_4096_stack2g}"
if (( $# > 0 )); then shift; fi
export PYTHONPATH="$bulb_repo/source/BrainCo_DexHand:$bulb_repo:$bulb_rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
bulb_shared_mesh_args=()
for bulb_finger in middip indexdip ringdip pinkydip thumbdip; do
  for bulb_surface in tacmap_surface hydroshear_marker_surface; do
    bulb_shared_mesh_args+=("env.scene.right_${bulb_finger}_roll_rubber_link_${bulb_surface}_s.mesh_prim_paths.0.is_shared=true")
  done
done
exec "$bulb_python" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 --max-restarts=0 \
  scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0 \
  --distributed --headless --num_envs 4096 --expected_world_size 2 \
  --max_iterations 15000 --seed 42 --logger tensorboard \
  --run_name bulb_nomodal_4096_seed42_dual_gpu_stack2g --log_dir "$bulb_log_dir" \
  agent.save_interval=25 agent.resume=false \
  env.rewards.action_l2.weight=-0.0001 env.rewards.action_rate_l2.weight=-0.0002 \
  env.tactile_policy_enabled=false env.tactile_reconstruction_diagnostics=false \
  env.rewards.any_finger_contact.weight=0.1 env.rewards.any_finger_contact.params.force_scale=0.5 \
  "${bulb_shared_mesh_args[@]}" env.sim.physx.gpu_collision_stack_size=2147483648 "$@"
