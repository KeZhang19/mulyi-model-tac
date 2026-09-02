#!/usr/bin/env python3
"""Benchmark the official IsaacLab Contrib TacSL VisuoTactileSensor."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Timed official IsaacLab TacSL sensor benchmark.")
parser.add_argument("--mode", choices=("force_field", "rgb", "all"), default="force_field")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--measure-seconds", type=float, default=3.0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--sync-cuda",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Synchronize CUDA around each measured step.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = str(args_cli.mode) in ("rgb", "all")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # noqa: E402
from isaaclab_assets.sensors import GELSIGHT_R15_CFG  # noqa: E402
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg  # noqa: E402


def validate_args() -> None:
    if int(args_cli.num_envs) <= 0:
        raise ValueError("--num-envs must be positive")
    if int(args_cli.warmup_steps) < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if not math.isfinite(float(args_cli.measure_seconds)) or float(args_cli.measure_seconds) <= 0.0:
        raise ValueError("--measure-seconds must be finite and positive")


class NvmlMemorySampler:
    """Track peak VRAM for the current benchmark process."""

    def __init__(self, device: str) -> None:
        self.available = False
        self.error: str | None = None
        self.process_peak_bytes: int | None = None
        self.device_peak_bytes: int | None = None
        self._nvml = None
        self._handle = None
        if not str(device).startswith("cuda"):
            self.error = "NVML requires a CUDA device"
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml

            pynvml.nvmlInit()
            index = torch.device(device).index
            if index is None:
                index = torch.cuda.current_device()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def sample(self) -> None:
        if not self.available or self._nvml is None or self._handle is None:
            return
        try:
            self.device_peak_bytes = max(
                self.device_peak_bytes or 0,
                int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used),
            )
            process_values: list[int] = []
            for prefix in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                for suffix in ("_v3", "_v2", ""):
                    function = getattr(self._nvml, prefix + suffix, None)
                    if function is None:
                        continue
                    try:
                        for process in function(self._handle):
                            if int(getattr(process, "pid", -1)) == os.getpid():
                                used = int(getattr(process, "usedGpuMemory", -1))
                                if 0 <= used < 2**63:
                                    process_values.append(used)
                        break
                    except Exception:
                        continue
            if process_values:
                self.process_peak_bytes = max(self.process_peak_bytes or 0, max(process_values))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def result(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "process_current_peak_bytes": self.process_peak_bytes,
            "device_used_peak_bytes": self.device_peak_bytes,
        }


@configclass
class TacSLBenchmarkSceneCfg(InteractiveSceneCfg):
    """Official GelSight R15 TacSL scene, matching the IsaacLab demo."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            physics_material_prim_path="elastomer",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.0005),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )
    tactile_sensor = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=str(args_cli.mode) in ("rgb", "all"),
        enable_force_field=str(args_cli.mode) in ("force_field", "all"),
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )
    contact_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=180.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.06776, 0.498),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def synchronize() -> None:
    if bool(args_cli.sync_cuda) and torch.cuda.is_available() and str(args_cli.device).startswith("cuda"):
        torch.cuda.synchronize(args_cli.device)


def step_scene(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    force: torch.Tensor,
    torque: torch.Tensor,
    count: int,
):
    if count > 20:
        indices = torch.arange(scene.num_envs, device=sim.device)
        torque[indices % 2 == 1, 0, 2] = 10.0
        torque[indices % 2 == 0, 0, 2] = -10.0
        scene["contact_object"].permanent_wrench_composer.set_forces_and_torques(force, torque)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    return scene["tactile_sensor"].data


def tensor_shapes(data) -> dict[str, list[int]]:
    names = ("tactile_depth_image", "tactile_rgb_image", "tactile_normal_force", "tactile_shear_force")
    return {
        name: [int(value) for value in tensor.shape]
        for name in names
        if isinstance((tensor := getattr(data, name, None)), torch.Tensor)
    }


def run_benchmark() -> dict[str, Any]:
    validate_args()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.005,
            device=args_cli.device,
            physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2**30),
        )
    )
    scene = InteractiveScene(TacSLBenchmarkSceneCfg(num_envs=int(args_cli.num_envs), env_spacing=0.2))
    sim.reset()
    if str(args_cli.mode) in ("rgb", "all"):
        scene["tactile_sensor"].get_initial_render()

    force = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque = torch.zeros_like(force)
    force[:, 0, 2] = -1.0
    count = 0
    data = None
    for _ in range(int(args_cli.warmup_steps)):
        data = step_scene(sim, scene, force, torque, count)
        count += 1
    synchronize()
    if torch.cuda.is_available() and str(args_cli.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args_cli.device)
    sampler = NvmlMemorySampler(str(args_cli.device))
    sampler.sample()

    step_times: list[float] = []
    measured_start = time.perf_counter()
    while time.perf_counter() - measured_start < float(args_cli.measure_seconds):
        synchronize()
        start = time.perf_counter()
        data = step_scene(sim, scene, force, torque, count)
        synchronize()
        step_times.append(time.perf_counter() - start)
        sampler.sample()
        count += 1

    times = np.asarray(step_times, dtype=np.float64)
    if times.size == 0:
        raise RuntimeError("TacSL benchmark produced no measured steps")
    total_time = float(times.sum())
    batch_hz = float(times.size) / total_time
    gpu: dict[str, Any] = {"device": str(args_cli.device), "nvml": sampler.result()}
    if torch.cuda.is_available() and str(args_cli.device).startswith("cuda"):
        properties = torch.cuda.get_device_properties(args_cli.device)
        gpu.update(
            {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args_cli.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(args_cli.device)),
            }
        )
    return {
        "schema_version": 1,
        "benchmark": "isaaclab_contrib_official_tacsl_sensor",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "sensor_class": "isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor",
            "demo": "/home/abc/IsaacLab/scripts/demos/sensors/tacsl_sensor.py",
        },
        "configuration": {
            "mode": str(args_cli.mode),
            "num_envs": int(scene.num_envs),
            "sensor_count": int(scene.num_envs),
            "tactile_array_size": [20, 25],
            "rgb_resolution": [int(GELSIGHT_R15_CFG.image_height), int(GELSIGHT_R15_CFG.image_width)],
            "physics_dt_s": float(sim.get_physics_dt()),
            "warmup_steps": int(args_cli.warmup_steps),
            "requested_measure_seconds": float(args_cli.measure_seconds),
            "measured_steps": int(times.size),
            "sync_cuda_per_step": bool(args_cli.sync_cuda),
            "headless": True,
        },
        "metrics": {
            "total_time_s": total_time,
            "mean_step_s": float(times.mean()),
            "median_step_s": float(np.median(times)),
            "p95_step_s": float(np.percentile(times, 95)),
            "p99_step_s": float(np.percentile(times, 99)),
            "batch_hz": batch_hz,
            "env_sensor_frames_per_s": batch_hz * float(scene.num_envs),
        },
        "observations": tensor_shapes(data),
        "gpu": gpu,
        "runtime": {
            "scene": "isaaclab.scene.InteractiveScene",
            "rl_environment": False,
            "policy_inference": False,
            "official_sensor_class": True,
        },
    }


def main() -> None:
    output = args_cli.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_benchmark()
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        print(f"[done] official IsaacLab TacSL benchmark: {output}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
