#!/usr/bin/env bash
set -euo pipefail
cd /home/brainco1/workspace/xinyu/tacsl_insertion_lab
job_dir=${1:?supply a new job directory}
mkdir "$job_dir"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python_bin=/home/brainco1/workspace/xinyu/assembly_play_sim6/bin/python
trap 'code=$?; printf "{\"exit_code\":%d,\"finished_at\":\"%s\"}\n" "$code" "$(TZ=Asia/Shanghai date --iso-8601=seconds)" > "$job_dir/exit.json"' EXIT
mkdir "$job_dir/code"
cp -a scripts tacsl_insertion_lab provenance.json "$job_dir/code/"
"$python_bin" scripts/guard_run.py --logfile "$job_dir/train.log" -- \
  "$python_bin" scripts/run.py --mode train --num_envs 4096 --iterations 1500 \
  --horizon 128 --minibatches 32 --save_interval 25 --physics_buffer_scale 4 \
  --collision_stack_size 4293918720 \
  --without_tactile --compact_logs --seed 42 --headless --output "$job_dir/train"
"$python_bin" scripts/guard_run.py --logfile "$job_dir/eval.log" -- \
  "$python_bin" scripts/run.py --mode eval --num_envs 256 --steps 1020 --seed 1042 \
  --physics_buffer_scale 4 --collision_stack_size 4293918720 \
  --without_tactile --headless --checkpoint "$PWD/$job_dir/train/final.pt" \
  --output "$job_dir/eval"
