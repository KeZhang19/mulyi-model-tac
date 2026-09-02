# Revo2 TacMap tactile environment

This folder is a standalone migration of the Revo2 TacMap tactile press demo.

## Main entry points

- `run_revo2_tactile.py`: run the Revo2 tactile press test.
- `run_revo2_hand.py`: load the Revo2 hand and preview tactile maps.
- `revo2_tactile_env.py`: Isaac Lab direct environment.
- `revo2_tactile_env_cfg.py`: robot, object, contact sensor, and TacMap sensor config.
- `tacmap_sensor/`: TacMap sensor implementation.

## Assets

- `assets/revo2_system/urdf/`: local Revo2 USD/URDF files.
- `assets/tactilesensor_map/`: tactile point and normal maps.
- `assets/presser/`: test presser USDs.
- `assets/test_case/`: press test JSON/trajectory cases.

## Example

Run from the RevoLab root:

```bash
python tacmap/run_revo2_tactile.py --num_envs 1 --finger middle
```

## Parallel tactile accuracy benchmark

`benchmark_parallel_tactile.py` implements the raw TacMap multi-env verification workflow from `multi-env verification.md`. The default scenario is the primary B0/B3 case: `finger=middle`, `presser=square_4`, `ray-mode=link_surface`, press offset `0.025 -> 0.018`, then a `0.004 m` slide along `+y` over `300` steps. The default link-surface map is `240x240`.

Create a single-env reference with the Isaac Lab environment:

```bash
/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py \
  --num-envs 1 \
  --label b0_middle_square4_link_surface
```

The command prints the trace path. Use that `.npz` as a single-env reference/baseline for parallel runs with matching trajectory and recording cadence. This is a TacMap reproducibility reference, not real measured tactile ground truth:

```bash
/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py \
  --num-envs 4 \
  --reference outputs/tactile_parallel_benchmark/<baseline_trace>.npz \
  --label b3_envs_4 \
  --fail-on-threshold

/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py \
  --num-envs 16 \
  --reference outputs/tactile_parallel_benchmark/<baseline_trace>.npz \
  --label b3_envs_16 \
  --fail-on-threshold

/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py \
  --num-envs 64 \
  --reference outputs/tactile_parallel_benchmark/<baseline_trace>.npz \
  --label b3_envs_64 \
  --fail-on-threshold
```

For larger env counts, reduce the save cadence and make the reference with the same `--record-every`, because comparison requires identical recorded step indices:

```bash
/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py --num-envs 1 --record-every 10 --label b0_record10
/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py --num-envs 256 --record-every 10 --reference <record10_reference.npz> --label b3_envs_256 --fail-on-threshold
/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py --num-envs 1024 --record-every 10 --reference <record10_reference.npz> --label b3_envs_1024 --fail-on-threshold
```

Useful localization sweeps:

```bash
/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py --num-envs 64 --env-spacing 0.25 --reference <baseline_trace>.npz --label spacing_025
/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py --num-envs 64 --env-spacing 1.50 --reference <baseline_trace>.npz --label spacing_150
/home/wangziyi/env_isaaclab/bin/python tacmap/benchmark_parallel_tactile.py --num-envs 1 --ray-mode surface_normal --label b5_surface_normal
```

Each trace stores `tacmap_raw`, `tacmap_surface_raw`, `tacmap_object_raw`, quantized `tacmap`/`vbts_deform`, contact forces, local contact points, step timings, recorded step indices, and metadata. With `--reference`, the script writes sibling `__metrics.json` and `.csv` files. Raw-depth metrics are primary: active IoU, active pixel count, centroid, bbox, principal-axis angle, onset/offset frame error, RMSE/MAE/p95/max depth error, indentation volume, monotonicity, slide stability, precontact leakage, invalid depth counts, link-surface formula/baseline checks, and env drift by index/origin.
