#!/usr/bin/env python3
"""Benchmark the official Sharpa-TacMap multi-environment depth example.

This adapter intentionally leaves ``/home/abc/sharpa-tacmap/run.py`` unchanged.
It instantiates the same environment/configuration, disables display-only debug
geometry, performs complete environment steps, and measures a fixed amount of
wall-clock time after reset and warm-up have completed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import warnings

import numpy as np

from isaaclab.app import AppLauncher


DEFAULT_SHARPA_ROOT = Path("/home/abc/sharpa-tacmap")


parser = argparse.ArgumentParser(description="Timed Sharpa-TacMap multi-environment depth benchmark.")
parser.add_argument("--sharpa-root", type=Path, default=DEFAULT_SHARPA_ROOT)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--warmup-steps", type=int, default=1)
parser.add_argument("--measure-seconds", type=float, default=3.0)
parser.add_argument(
    "--press-info",
    type=Path,
    default=Path("assets/test_case/cylinder_D4_left_145_20250904193821_40_60.json"),
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--sync-cuda",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Synchronize CUDA around every timed environment step.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

sharpa_root = args_cli.sharpa_root.expanduser().resolve()
if not (sharpa_root / "run.py").is_file():
    parser.error(f"Sharpa-TacMap repository is missing run.py: {sharpa_root}")
if str(sharpa_root) not in sys.path:
    sys.path.insert(0, str(sharpa_root))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

from env import SharpaWaveInhandRotateTactileAlignEnv  # noqa: E402
from env_cfg import SharpaWaveEnvCfg  # noqa: E402


class NvmlMemorySampler:
    """Track this child process and whole-device VRAM through NVML."""

    def __init__(self, device: str) -> None:
        self.available = False
        self.error: str | None = None
        self.process_start_bytes: int | None = None
        self.process_peak_bytes: int | None = None
        self.process_end_bytes: int | None = None
        self.device_start_bytes: int | None = None
        self.device_peak_bytes: int | None = None
        self.device_end_bytes: int | None = None
        self._nvml = None
        self._handle = None
        if not str(device).startswith("cuda"):
            self.error = "NVML sampling requires a CUDA device"
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml

            pynvml.nvmlInit()
            device_spec = torch.device(device)
            index = torch.cuda.current_device() if device_spec.index is None else int(device_spec.index)
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _running_processes(self) -> list[Any]:
        if not self.available or self._nvml is None or self._handle is None:
            return []
        results: list[Any] = []
        for prefix in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            for suffix in ("_v3", "_v2", ""):
                function = getattr(self._nvml, prefix + suffix, None)
                if function is None:
                    continue
                try:
                    results.extend(function(self._handle))
                    break
                except Exception:
                    continue
        return results

    def sample(self) -> None:
        if not self.available or self._nvml is None or self._handle is None:
            return
        try:
            device_used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            process_used_by_pid: dict[int, int] = {}
            for process in self._running_processes():
                pid = int(getattr(process, "pid", -1))
                used = int(getattr(process, "usedGpuMemory", -1))
                if pid >= 0 and 0 <= used < 2**63:
                    process_used_by_pid[pid] = max(process_used_by_pid.get(pid, 0), used)
            process_used = process_used_by_pid.get(os.getpid())
            if self.device_start_bytes is None:
                self.device_start_bytes = device_used
            self.device_end_bytes = device_used
            self.device_peak_bytes = max(self.device_peak_bytes or 0, device_used)
            if process_used is not None:
                if self.process_start_bytes is None:
                    self.process_start_bytes = process_used
                self.process_end_bytes = process_used
                self.process_peak_bytes = max(self.process_peak_bytes or 0, process_used)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def result(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "error": self.error,
            "process_current_start_bytes": self.process_start_bytes,
            "process_current_peak_bytes": self.process_peak_bytes,
            "process_current_end_bytes": self.process_end_bytes,
            "device_used_start_bytes": self.device_start_bytes,
            "device_used_peak_bytes": self.device_peak_bytes,
            "device_used_end_bytes": self.device_end_bytes,
            "note": "process_current_* isolates this benchmark PID; device_used_* includes all GPU processes",
        }


def validate_args() -> None:
    if int(args_cli.num_envs) <= 0:
        raise ValueError("--num-envs must be positive")
    if int(args_cli.warmup_steps) < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if not math.isfinite(float(args_cli.measure_seconds)) or float(args_cli.measure_seconds) <= 0.0:
        raise ValueError("--measure-seconds must be finite and positive")


def resolve_under_sharpa(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (sharpa_root / expanded).resolve()


def configure_environment() -> SharpaWaveEnvCfg:
    """Apply the same default press case used by Sharpa-TacMap's run.py."""

    cfg = SharpaWaveEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.seed = int(args_cli.seed)
    cfg.sim.device = str(args_cli.device)
    cfg.enable_deform = False
    cfg.enable_deform_vis = True
    cfg.debug_show_axes = False
    for sensor_cfg in cfg.vbts_sensor:
        sensor_cfg.debug_viz = False
        sensor_cfg.debug_viz_surface = False
        sensor_cfg.debug_viz_normals = False

    press_info_path = resolve_under_sharpa(args_cli.press_info)
    press_info = json.loads(press_info_path.read_text(encoding="utf-8"))
    presser_name = press_info.get("presser_name")
    if presser_name:
        cfg.object_cfg.spawn.usd_path = cfg.object_cfg.spawn.usd_path.replace(cfg.presser_name, presser_name)
        cfg.presser_name = str(presser_name)
    if press_info.get("presser_direction") is not None:
        cfg.press_direction = list(press_info["presser_direction"])
    cfg.press_info = press_info_path.stem
    if press_info.get("presser_init_pos") is not None:
        cfg.presser_init_pos[:3] = list(press_info["presser_init_pos"])
    if press_info.get("presser_init_rot") is not None:
        cfg.presser_init_pos[3:] = list(press_info["presser_init_rot"])

    direction = np.asarray(cfg.press_direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm > 1.0e-6:
        cfg.presser_init_pos[:3] = (
            np.asarray(cfg.presser_init_pos[:3], dtype=np.float64) - 0.004 * direction / norm
        ).tolist()

    trajectory_path = Path(str(press_info["action_target_pos_file"]))
    if not trajectory_path.is_absolute():
        trajectory_path = (press_info_path.parent / trajectory_path).resolve()
    cfg.action_target_pos = np.load(trajectory_path, allow_pickle=False)
    return cfg


def synchronize(device: str) -> None:
    if bool(args_cli.sync_cuda) and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def tensor_shape_from_observation(observation: dict[str, Any]) -> list[int] | None:
    value = observation.get("vbts_deform") if isinstance(observation, dict) else None
    return [int(size) for size in value.shape] if isinstance(value, torch.Tensor) else None


def run_benchmark() -> dict[str, Any]:
    cfg = configure_environment()
    env = SharpaWaveInhandRotateTactileAlignEnv(cfg)
    observation, _extras = env.reset()
    actions = torch.zeros((cfg.scene.num_envs, cfg.action_space), device=env.device)

    for _ in range(int(args_cli.warmup_steps)):
        observation, *_ = env.step(actions)
    synchronize(str(env.device))

    if torch.cuda.is_available() and str(env.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(env.device)
    sampler = NvmlMemorySampler(str(env.device))
    sampler.sample()

    step_times: list[float] = []
    elapsed = 0.0
    while elapsed < float(args_cli.measure_seconds):
        synchronize(str(env.device))
        start = time.perf_counter()
        observation, *_ = env.step(actions)
        synchronize(str(env.device))
        duration = time.perf_counter() - start
        step_times.append(duration)
        elapsed += duration
        sampler.sample()

    sampler.sample()
    times = np.asarray(step_times, dtype=np.float64)
    mean_step_s = float(np.mean(times))
    sim_hz = 1.0 / mean_step_s if mean_step_s > 0.0 else 0.0
    gpu: dict[str, Any] = {
        "device": str(env.device),
        "name": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "nvml": sampler.result(),
    }
    if torch.cuda.is_available() and str(env.device).startswith("cuda"):
        properties = torch.cuda.get_device_properties(env.device)
        gpu.update(
            {
                "name": properties.name,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(env.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(env.device)),
            }
        )

    return {
        "schema_version": 1,
        "benchmark": "sharpa_tacmap_official_multienv_depth",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": {
            "repository": str(sharpa_root),
            "official_environment": "SharpaWaveInhandRotateTactileAlignEnv",
            "direct_rl_environment": True,
            "policy_inference": False,
            "headless": True,
        },
        "configuration": {
            "num_envs": int(env.num_envs),
            "sensor_count": len(cfg.vbts_sensor),
            "depth_rows": 240 // int(cfg.resolution_step),
            "depth_cols": 240 // int(cfg.resolution_step),
            "warmup_steps": int(args_cli.warmup_steps),
            "requested_measure_seconds": float(args_cli.measure_seconds),
            "measured_steps": len(step_times),
            "physics_dt_s": float(env.physics_dt),
            "decimation": int(cfg.decimation),
            "control_dt_s": float(env.step_dt),
            "sync_cuda_per_step": bool(args_cli.sync_cuda),
            "press_info": str(resolve_under_sharpa(args_cli.press_info)),
        },
        "metrics": {
            "total_time_s": float(np.sum(times)),
            "mean_step_s": mean_step_s,
            "median_step_s": float(np.median(times)),
            "p95_step_s": float(np.percentile(times, 95)),
            "p99_step_s": float(np.percentile(times, 99)),
            "min_step_s": float(np.min(times)),
            "max_step_s": float(np.max(times)),
            "sim_hz": sim_hz,
            "env_steps_per_s": float(env.num_envs) * sim_hz,
            "real_time_factor": sim_hz * float(env.step_dt),
        },
        "gpu": gpu,
        "observations": {
            "depth": {
                "shape": tensor_shape_from_observation(observation),
                "dtype": str(observation["vbts_deform"].dtype),
            }
        },
    }


def main() -> None:
    validate_args()
    result = run_benchmark()
    output = args_cli.output.expanduser()
    output = output.resolve() if output.is_absolute() else (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"[done] Sharpa-TacMap depth benchmark: {output}", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
