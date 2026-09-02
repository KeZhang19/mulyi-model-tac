#!/usr/bin/env python3
"""Benchmark vectorized Revo3 tactile simulation without an RL environment.

This entrypoint creates one :class:`isaaclab.scene.InteractiveScene` containing
multiple cloned Revo3 hands and square pressers.  It advances PhysX directly
and calls the selected TacMap, FOTS, TacSL, HydroShear, or OURS GPU kernels
without Gym, an observation manager, an action manager, rewards, or a policy.
The named OURS modes isolate each visual-tactile output or execute all three.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any
import warnings

import numpy as np
import torch

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
PRESSER_USD = REPO_ROOT / "tacmap" / "assets" / "presser" / "square_4.usd"
FINGER_CHOICES = ("middle", "index", "ring", "pinky", "thumb")
MODALITY_CHOICES = ("pressure", "depth", "rgb", "marker")
IMPLEMENTATION_CHOICES = ("ours", "tacmap", "fots", "tacsl", "hydroshear")
BENCHMARK_MODE_MODALITIES: dict[str, tuple[str, ...]] = {
    "depth": ("depth",),
    "rgb": ("rgb",),
    "marker": ("marker",),
    "all": ("depth", "rgb", "marker"),
}

for path in (REPO_ROOT, SOURCE_ROOT, REPO_ROOT / "integrate"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


parser = argparse.ArgumentParser(
    description="Benchmark vectorized Revo3 raw tactile simulation in InteractiveScene (no RL runtime)."
)
parser.add_argument("--num-envs", type=int, default=16, help="Parallel environments in one Isaac Sim process.")
parser.add_argument("--env-spacing", type=float, default=1.5)
parser.add_argument("--focus-finger", choices=FINGER_CHOICES, default="index")
parser.add_argument(
    "--implementation",
    choices=IMPLEMENTATION_CHOICES,
    default="ours",
    help=(
        "GPU tactile implementation. 'ours' retains the depth/rgb/marker modes; "
        "the other choices run one published baseline implementation each."
    ),
)
modality_selection = parser.add_mutually_exclusive_group()
modality_selection.add_argument(
    "--mode",
    choices=tuple(BENCHMARK_MODE_MODALITIES),
    default="all",
    help="GPU benchmark preset: depth only, RGB only, marker only, or all three.",
)
modality_selection.add_argument(
    "--modalities",
    nargs="+",
    choices=MODALITY_CHOICES,
    default=None,
    help="Advanced manual override; named --mode presets are preferred.",
)
parser.add_argument("--warmup-steps", type=int, default=200)
parser.add_argument("--measure-steps", type=int, default=1000)
parser.add_argument(
    "--measure-seconds",
    type=float,
    default=None,
    help="Stop after this much measured wall-clock time; when set, it overrides --measure-steps.",
)
parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
parser.add_argument("--decimation", type=int, default=2)
parser.add_argument("--press-start-offset", type=float, default=0.035)
parser.add_argument("--press-end-offset", type=float, default=0.025)
parser.add_argument("--press-steps", type=int, default=120)
parser.add_argument("--hold-steps", type=int, default=60)
parser.add_argument("--release-steps", type=int, default=120)
parser.add_argument(
    "--phase-stagger",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Distribute parallel environments across approach/hold/release phases.",
)
parser.add_argument(
    "--presser-collision",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable PhysX collision for each kinematic square presser.",
)
parser.add_argument(
    "--sync-cuda",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Synchronize CUDA around each measured step for honest end-to-end timing.",
)
parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Show env_0 tactile panes for the selected mode; window updates are outside the timed region.",
)
parser.add_argument(
    "--visualize-update-interval",
    type=int,
    default=2,
    help="Refresh the optional tactile panel every N control steps.",
)
parser.add_argument(
    "--depth-display-max-mm",
    type=float,
    default=1.0,
    help="Depth value mapped to the top of the visualization color scale.",
)
parser.add_argument("--output", type=Path, default=None, help="Optional benchmark JSON output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The default remains a strict headless throughput benchmark.  ``--visualize``
# is a display-only diagnostic that opens Kit and updates an observation panel.
if bool(args_cli.visualize) and bool(args_cli.headless):
    parser.error("--visualize cannot be combined with --headless")
args_cli.headless = not bool(args_cli.visualize)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402

from BrainCo_DexHand.tasks.manager_based.dexsuite import mdp  # noqa: E402
from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3 import (  # noqa: E402
    dexsuite_revo3_env_cfg_grasp as revo_cfg,
)


def selected_modalities() -> tuple[str, ...]:
    """Resolve the named visual-tactile mode or advanced manual override."""

    if args_cli.modalities is not None:
        return tuple(dict.fromkeys(str(name) for name in args_cli.modalities))
    return BENCHMARK_MODE_MODALITIES[str(args_cli.mode)]


def selected_mode_name() -> str:
    """Return a stable label for result files and benchmark metadata."""

    if str(args_cli.implementation) != "ours":
        return str(args_cli.implementation)
    return "custom" if args_cli.modalities is not None else str(args_cli.mode)


def selected_ours_terms() -> tuple[str, ...]:
    """Map the standalone selection to the smallest required OURS sensor set."""

    mapping = {
        "pressure": "pressure",
        "depth": "tacmap_policy",
        "rgb": "taxim_rgb",
        "marker": "hydroshear",
    }
    return tuple(mapping[name] for name in selected_modalities())


class StandaloneTactileRuntime:
    """Minimal context required by the reusable tactile GPU kernels."""

    def __init__(self, sim, scene: InteractiveScene, source_cfg, *, decimation: int) -> None:
        self.sim = sim
        self.scene = scene
        self.cfg = source_cfg
        self.num_envs = int(scene.num_envs)
        self.device = str(sim.device)
        self.physics_dt = float(source_cfg.sim.dt)
        self.step_dt = self.physics_dt * int(decimation)
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self._sim_step_counter = 0

    @property
    def unwrapped(self) -> "StandaloneTactileRuntime":
        return self


class HardRobotHold:
    """Keep every cloned robot at its initialized root and joint state."""

    def __init__(self, runtime: StandaloneTactileRuntime) -> None:
        self.runtime = runtime
        self.robot: Articulation = runtime.scene["robot"]
        self.root_pose = torch.cat(
            (self.robot.data.root_pos_w.clone(), self.robot.data.root_quat_w.clone()),
            dim=-1,
        )
        self.root_velocity = torch.zeros(
            (runtime.num_envs, 6),
            device=runtime.device,
            dtype=torch.float32,
        )
        self.joint_position = self.robot.data.joint_pos.clone()
        self.joint_velocity = torch.zeros_like(self.robot.data.joint_vel)

    def write(self) -> None:
        self.robot.write_root_pose_to_sim(self.root_pose)
        self.robot.write_root_velocity_to_sim(self.root_velocity)
        self.robot.write_joint_state_to_sim(self.joint_position, self.joint_velocity)
        self.robot.set_joint_position_target(self.joint_position)
        self.robot.update(0.0)
        self.robot.write_data_to_sim()


def validate_args() -> None:
    if int(args_cli.num_envs) <= 0:
        raise ValueError("--num-envs must be positive")
    if int(args_cli.warmup_steps) < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if int(args_cli.measure_steps) <= 0:
        raise ValueError("--measure-steps must be positive")
    if args_cli.measure_seconds is not None and (
        not np.isfinite(float(args_cli.measure_seconds)) or float(args_cli.measure_seconds) <= 0.0
    ):
        raise ValueError("--measure-seconds must be finite and positive")
    if not np.isfinite(float(args_cli.physics_dt)) or float(args_cli.physics_dt) <= 0.0:
        raise ValueError("--physics-dt must be finite and positive")
    if int(args_cli.decimation) <= 0:
        raise ValueError("--decimation must be positive")
    for name in ("press_steps", "release_steps"):
        if int(getattr(args_cli, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args_cli.hold_steps) < 0:
        raise ValueError("--hold-steps must be non-negative")
    if int(args_cli.visualize_update_interval) <= 0:
        raise ValueError("--visualize-update-interval must be positive")
    if not np.isfinite(float(args_cli.depth_display_max_mm)) or float(args_cli.depth_display_max_mm) <= 0.0:
        raise ValueError("--depth-display-max-mm must be finite and positive")
    if str(args_cli.implementation) != "ours" and args_cli.modalities is not None:
        raise ValueError("--modalities is only valid with --implementation ours")
    if not PRESSER_USD.is_file():
        raise FileNotFoundError(f"square_4 presser USD is missing: {PRESSER_USD}")


def kernel_params_from(source_cfg) -> dict[str, Any]:
    """Copy selected observation parameters without creating any manager."""

    implementation = str(args_cli.implementation)
    if implementation == "ours":
        for term_name in (
            "rl_ours_pressure",
            "rl_ours_tacmap_policy",
            "rl_ours_hydroshear",
            "rl_ours_taxim_rgb",
        ):
            term = getattr(source_cfg.observations.proprio, term_name, None)
            params = None if term is None else getattr(term, "params", None)
            if isinstance(params, dict) and isinstance(params.get("component_cfg"), dict):
                result = dict(params["component_cfg"])
                result.update(
                    {
                        "cache_pressure_output": True,
                        "hydroshear_cache_debug_output": False,
                        "hydroshear_render_debug_marker_images": False,
                        "hydroshear_cache_marker_depth_output": False,
                        "tacmap_policy_full_depth_image": True,
                    }
                )
                return result
        raise RuntimeError("Revo3 OURS component configuration was not found")

    term_name = {
        "fots": "rl_fots_baseline",
        "tacsl": "rl_tacsl_baseline",
        "tacmap": "rl_tacmap",
        "hydroshear": "rl_hydroshear_baseline",
    }[implementation]
    term = getattr(source_cfg.observations.proprio, term_name, None)
    params = None if term is None else getattr(term, "params", None)
    if not isinstance(params, dict):
        raise RuntimeError(f"Revo3 {implementation} observation parameters were not found")
    return dict(params)


def instantiate_source_cfg():
    """Instantiate the task config with only the requested tactile sensors."""

    switch_names = (
        "ENABLE_RL_OURS_OBS",
        "ENABLE_RL_TACSL_BASELINE_OBS",
        "ENABLE_RL_TACMAP_BASELINE_OBS",
        "ENABLE_RL_HYDROSHEAR_BASELINE_OBS",
        "ENABLE_RL_FOTS_BASELINE_OBS",
        "RL_OURS_OBSERVATION_TERMS",
    )
    previous = {name: getattr(revo_cfg, name) for name in switch_names}
    implementation = str(args_cli.implementation)
    try:
        revo_cfg.ENABLE_RL_OURS_OBS = implementation == "ours"
        revo_cfg.ENABLE_RL_TACSL_BASELINE_OBS = implementation == "tacsl"
        revo_cfg.ENABLE_RL_TACMAP_BASELINE_OBS = implementation == "tacmap"
        revo_cfg.ENABLE_RL_HYDROSHEAR_BASELINE_OBS = implementation == "hydroshear"
        revo_cfg.ENABLE_RL_FOTS_BASELINE_OBS = implementation == "fots"
        if implementation == "ours":
            revo_cfg.RL_OURS_OBSERVATION_TERMS = selected_ours_terms()
        return revo_cfg.DexsuiteRevo3LiftEnvCfg()
    finally:
        for name, value in previous.items():
            setattr(revo_cfg, name, value)


def configure_scene():
    """Build only the assets and sensors needed by the standalone benchmark."""

    source_cfg = instantiate_source_cfg()
    source_cfg.curriculum = None
    source_cfg.sim.device = str(args_cli.device)
    source_cfg.sim.dt = float(args_cli.physics_dt)
    source_cfg.sim.render_interval = int(args_cli.decimation)
    source_cfg.decimation = int(args_cli.decimation)
    source_cfg.scene.num_envs = int(args_cli.num_envs)
    source_cfg.scene.env_spacing = float(args_cli.env_spacing)

    # Vectorization is provided by InteractiveScene and batched Tensor APIs.
    # Per-environment physics replication stays disabled because the custom
    # target meshes and compliant material bindings are environment-specific.
    source_cfg.scene.replicate_physics = False

    # These assets and reward-only contact sensors are unrelated to raw tactile
    # generation and therefore do not belong in this standalone scene.
    source_cfg.scene.table = None
    source_cfg.scene.marker_helper = None
    source_cfg.scene.plane = None
    source_cfg.scene.sky_light = None
    for link_name in revo_cfg.TIANJI_HAND_DIP_BODIES:
        setattr(source_cfg.scene, f"{link_name}_object_s", None)

    source_cfg.scene.object.spawn = sim_utils.UsdFileCfg(
        usd_path=str(PRESSER_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=bool(args_cli.presser_collision),
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    )
    source_cfg.scene.object.init_state.pos = (0.0, 0.0, 2.0)
    source_cfg.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    return source_cfg, kernel_params_from(source_cfg)


def reset_scene(runtime: StandaloneTactileRuntime) -> None:
    robot: Articulation = runtime.scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += runtime.scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:13])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    runtime.scene.reset()
    runtime.scene.write_data_to_sim()
    runtime.sim.forward()
    runtime.scene.update(runtime.physics_dt)
    mdp.invalidate_ours_tactile_cache_on_reset(runtime, None)


def apply_touch_physics(runtime: StandaloneTactileRuntime) -> None:
    """Apply the same tactile collision properties used by the main simulator."""

    revo_cfg.apply_revo3_touch_compliant_materials(
        runtime,
        None,
        link_names=tuple(revo_cfg.TIANJI_TACMAP_LINK_ORDER),
        stiffness=revo_cfg.TIANJI_TOUCH_COMPLIANT_STIFFNESS,
        damping=revo_cfg.TIANJI_TOUCH_COMPLIANT_DAMPING,
        contact_offset=revo_cfg.TIANJI_TOUCH_CONTACT_OFFSET,
        rest_offset=revo_cfg.TIANJI_TOUCH_REST_OFFSET,
    )
    revo_cfg.apply_revo3_link_collision_properties(
        runtime,
        None,
        link_names=tuple(revo_cfg.TIANJI_PRESSURE_PAD_COLLISION_LINK_ORDER),
        contact_offset=revo_cfg.TIANJI_PRESSURE_PAD_CONTACT_OFFSET,
        rest_offset=revo_cfg.TIANJI_PRESSURE_PAD_REST_OFFSET,
        label="pressure pad",
    )


def body_index(robot: Articulation, link_name: str) -> int:
    matches = [index for index, name in enumerate(robot.body_names) if str(name) == str(link_name)]
    if not matches:
        raise RuntimeError(f"Body {link_name!r} is missing from the Revo3 articulation")
    return int(matches[0])


def axis_vector(axis: str) -> np.ndarray:
    value = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }.get(str(axis).lower())
    if value is None:
        raise ValueError(f"Unsupported axis: {axis}")
    return np.asarray(value, dtype=np.float32)


def press_target(finger: str) -> tuple[str, np.ndarray, np.ndarray]:
    finger_index = FINGER_CHOICES.index(str(finger))
    link_name = str(revo_cfg.TIANJI_TACMAP_LINK_ORDER[finger_index])
    params = revo_cfg.TIANJI_TACMAP_LINK_SURFACE_DEFAULTS[link_name]
    _points_path, normals_path = revo_cfg._tacmap_asset_paths(link_name)
    mean_normal = revo_cfg._mean_tacmap_normal_direction(str(normals_path))
    normal = axis_vector(str(params["ray_axis"])) if mean_normal is None else np.asarray(mean_normal, dtype=np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
    center = np.asarray(params["grid_center"], dtype=np.float32)
    return link_name, center, normal


def square_presser_quaternion(device: str, count: int) -> torch.Tensor:
    half_angle = 0.5 * math.radians(-110.425)
    local = torch.tensor(
        (math.cos(half_angle), 0.0, math.sin(half_angle), 0.0),
        device=device,
        dtype=torch.float32,
    )
    return local.unsqueeze(0).expand(count, -1)


def phase_offsets(runtime: StandaloneTactileRuntime, control_step: int) -> torch.Tensor:
    approach = int(args_cli.press_steps)
    hold = int(args_cli.hold_steps)
    release = int(args_cli.release_steps)
    cycle = approach + hold + release
    env_ids = torch.arange(runtime.num_envs, device=runtime.device, dtype=torch.long)
    if bool(args_cli.phase_stagger) and runtime.num_envs > 1:
        shifts = torch.div(env_ids * cycle, runtime.num_envs, rounding_mode="floor")
    else:
        shifts = torch.zeros_like(env_ids)
    phase = torch.remainder(shifts + int(control_step), cycle)

    alpha = torch.ones(runtime.num_envs, device=runtime.device, dtype=torch.float32)
    approaching = phase < approach
    alpha[approaching] = phase[approaching].to(torch.float32) / float(max(1, approach - 1))
    releasing = phase >= approach + hold
    release_phase = phase[releasing] - (approach + hold)
    alpha[releasing] = 1.0 - release_phase.to(torch.float32) / float(max(1, release - 1))
    alpha.clamp_(0.0, 1.0)
    return float(args_cli.press_start_offset) + alpha * (
        float(args_cli.press_end_offset) - float(args_cli.press_start_offset)
    )


def write_presser_pose(
    runtime: StandaloneTactileRuntime,
    *,
    link_name: str,
    center_l_np: np.ndarray,
    normal_l_np: np.ndarray,
    offsets_m: torch.Tensor,
) -> None:
    robot: Articulation = runtime.scene["robot"]
    presser: RigidObject = runtime.scene["object"]
    link_state = robot.data.body_link_state_w[:, body_index(robot, link_name), :7]
    link_pos_w = link_state[:, :3]
    link_quat_w = link_state[:, 3:7]
    center_l = torch.as_tensor(center_l_np, device=runtime.device, dtype=torch.float32)
    normal_l = torch.as_tensor(normal_l_np, device=runtime.device, dtype=torch.float32)
    presser_pos_l = center_l.unsqueeze(0) + normal_l.unsqueeze(0) * offsets_m.unsqueeze(-1)
    presser_pos_w = link_pos_w + quat_apply(link_quat_w, presser_pos_l)
    presser_quat_w = quat_mul(
        link_quat_w,
        square_presser_quaternion(runtime.device, runtime.num_envs),
    )
    presser.write_root_pose_to_sim(torch.cat((presser_pos_w, presser_quat_w), dim=-1))
    presser.write_root_velocity_to_sim(
        torch.zeros((runtime.num_envs, 6), device=runtime.device, dtype=torch.float32)
    )


def compute_raw_tactile(
    runtime: StandaloneTactileRuntime,
    kernel_params: dict[str, Any],
) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    implementation = str(args_cli.implementation)
    if implementation == "tacmap":
        depth = mdp.tacmap_rl_obs(runtime, **kernel_params)
        rows = max(1, int(kernel_params["tacmap_rows"]))
        cols = max(1, int(kernel_params["tacmap_cols"]))
        outputs["depth_m"] = depth.reshape(runtime.num_envs, len(FINGER_CHOICES), rows, cols)
        return outputs
    if implementation == "fots":
        output = mdp.fots_baseline_rl_obs(runtime, **kernel_params)
        finger_count = max(0, int(kernel_params.get("finger_count", len(FINGER_CHOICES))))
        marker_count = max(1, int(kernel_params.get("marker_rows", 16))) * max(
            1, int(kernel_params.get("marker_cols", 8))
        )
        outputs["marker_motion_px"] = output.reshape(runtime.num_envs, finger_count, marker_count, 2)
        return outputs
    if implementation == "tacsl":
        output = mdp.tacsl_baseline_rl_obs(runtime, **kernel_params)
        finger_count = len(tuple(kernel_params.get("finger_link_names", FINGER_CHOICES)))
        rows = max(1, int(kernel_params.get("output_rows", 16)))
        cols = max(1, int(kernel_params.get("output_cols", 8)))
        outputs["force_field"] = output.reshape(runtime.num_envs, finger_count, rows, cols, 3)
        return outputs
    if implementation == "hydroshear":
        output = mdp.hydroshear_baseline_rl_obs(runtime, **kernel_params)
        finger_count = max(0, int(kernel_params.get("finger_count", len(FINGER_CHOICES))))
        marker_count = max(1, int(kernel_params.get("marker_rows", 16))) * max(
            1, int(kernel_params.get("marker_cols", 8))
        )
        outputs["marker_displacement_m"] = output.reshape(
            runtime.num_envs, finger_count, marker_count, 3
        )
        return outputs

    selected = set(selected_modalities())
    if "pressure" in selected:
        outputs["pressure"] = mdp.ours_rl_pressure_obs(runtime, kernel_params)
    if "depth" in selected:
        depth = mdp.ours_rl_tacmap_policy_obs(runtime, kernel_params)
        outputs["depth_m"] = depth.reshape(runtime.num_envs, len(FINGER_CHOICES), 240, 320)
    if "marker" in selected:
        marker = mdp.ours_rl_hydroshear_obs(runtime, kernel_params)
        outputs["marker_motion"] = marker.reshape(runtime.num_envs, len(FINGER_CHOICES), 100, 3)
    if "rgb" in selected:
        rgb = mdp.ours_rl_taxim_rgb_obs(runtime, kernel_params)
        outputs["rgb"] = rgb.reshape(runtime.num_envs, len(FINGER_CHOICES), 3, 240, 320)
    return outputs


def control_step(
    runtime: StandaloneTactileRuntime,
    hold: HardRobotHold,
    kernel_params: dict[str, Any],
    *,
    link_name: str,
    center_l: np.ndarray,
    normal_l: np.ndarray,
    control_step_index: int,
) -> dict[str, torch.Tensor]:
    offsets = phase_offsets(runtime, control_step_index)
    for _ in range(int(args_cli.decimation)):
        runtime._sim_step_counter += 1
        hold.write()
        write_presser_pose(
            runtime,
            link_name=link_name,
            center_l_np=center_l,
            normal_l_np=normal_l,
            offsets_m=offsets,
        )
        runtime.scene.write_data_to_sim()
        runtime.sim.step(render=False)
        runtime.scene.update(runtime.physics_dt)
        hold.write()
    runtime.episode_length_buf += 1
    runtime.common_step_counter += 1
    return compute_raw_tactile(runtime, kernel_params)


def synchronize(runtime: StandaloneTactileRuntime) -> None:
    if not bool(args_cli.sync_cuda):
        return
    if not torch.cuda.is_available() or not runtime.device.startswith("cuda"):
        return
    torch.cuda.synchronize(device=runtime.device)


class NvmlMemorySampler:
    """Sample whole-process and whole-device VRAM through NVIDIA NVML."""

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
            torch_device = torch.device(device)
            device_index = torch.cuda.current_device() if torch_device.index is None else int(torch_device.index)
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
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
        process_growth = (
            None
            if self.process_start_bytes is None or self.process_peak_bytes is None
            else max(0, self.process_peak_bytes - self.process_start_bytes)
        )
        device_growth = (
            None
            if self.device_start_bytes is None or self.device_peak_bytes is None
            else max(0, self.device_peak_bytes - self.device_start_bytes)
        )
        return {
            "available": bool(self.available),
            "error": self.error,
            "process_current_start_bytes": self.process_start_bytes,
            "process_current_peak_bytes": self.process_peak_bytes,
            "process_current_end_bytes": self.process_end_bytes,
            "process_growth_peak_bytes": process_growth,
            "device_used_start_bytes": self.device_start_bytes,
            "device_used_peak_bytes": self.device_peak_bytes,
            "device_used_end_bytes": self.device_end_bytes,
            "device_growth_peak_bytes": device_growth,
            "note": (
                "process_current_* isolates this benchmark PID; device_used_* also includes other GPU processes"
            ),
        }


def tensor_summary(tensor: torch.Tensor) -> dict[str, float | int | list[int]]:
    finite = torch.isfinite(tensor)
    safe = torch.where(finite, tensor, torch.zeros_like(tensor))
    return {
        "shape": [int(value) for value in tensor.shape],
        "finite_fraction": float(finite.to(torch.float32).mean().detach().cpu()),
        "nonzero_fraction": float((safe != 0).to(torch.float32).mean().detach().cpu()),
        "min": float(torch.amin(safe).detach().cpu()),
        "max": float(torch.amax(safe).detach().cpu()),
        "mean": float(torch.mean(safe.to(torch.float32)).detach().cpu()),
    }


def _resize_visual_pane(image: np.ndarray, *, width: int = 320, height: int = 240) -> np.ndarray:
    """Convert a display pane to contiguous RGB at the common native size."""

    import cv2

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or int(array.shape[-1]) < 3:
        raise ValueError(f"Expected an RGB-like image, got {tuple(array.shape)}")
    array = np.ascontiguousarray(array[..., :3], dtype=np.uint8)
    if tuple(array.shape[:2]) != (int(height), int(width)):
        array = cv2.resize(array, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(array)


def _label_visual_pane(image: np.ndarray, label: str) -> np.ndarray:
    import cv2

    pane = _resize_visual_pane(image).copy()
    cv2.rectangle(pane, (0, 0), (max(72, 12 + 12 * len(label)), 28), (20, 20, 20), thickness=-1)
    cv2.putText(
        pane,
        str(label),
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return pane


def _depth_visual_pane(outputs: dict[str, torch.Tensor], focus_index: int) -> np.ndarray | None:
    depth = outputs.get("depth_m")
    if not isinstance(depth, torch.Tensor) or depth.ndim != 4 or int(depth.shape[0]) < 1:
        return None
    import cv2

    index = max(0, min(int(focus_index), int(depth.shape[1]) - 1))
    depth_np = torch.nan_to_num(depth[0, index].detach()).to(dtype=torch.float32).cpu().numpy()
    display_max_m = max(float(args_cli.depth_display_max_mm) * 1.0e-3, 1.0e-8)
    normalized = np.clip(depth_np / display_max_m, 0.0, 1.0)
    colored_bgr = cv2.applyColorMap(np.asarray(np.round(normalized * 255.0), dtype=np.uint8), cv2.COLORMAP_TURBO)
    return _label_visual_pane(colored_bgr[..., ::-1], "DEPTH")


def _rgb_visual_pane(outputs: dict[str, torch.Tensor], focus_index: int) -> np.ndarray | None:
    rgb = outputs.get("rgb")
    if not isinstance(rgb, torch.Tensor) or rgb.ndim != 5 or int(rgb.shape[0]) < 1:
        return None
    index = max(0, min(int(focus_index), int(rgb.shape[1]) - 1))
    image = torch.nan_to_num(rgb[0, index].detach()).to(dtype=torch.float32).permute(1, 2, 0).cpu().numpy()
    image = np.asarray(np.round(np.clip(image, 0.0, 1.0) * 255.0), dtype=np.uint8)
    return _label_visual_pane(image, "RGB")


def _marker_magnitude_fallback(marker: torch.Tensor, focus_index: int) -> np.ndarray:
    """Provide a robust fallback when the calibrated vector renderer is unavailable."""

    import cv2

    index = max(0, min(int(focus_index), int(marker.shape[1]) - 1))
    vectors = torch.nan_to_num(marker[0, index].detach()).to(dtype=torch.float32)
    magnitude = torch.linalg.norm(vectors, dim=-1).cpu().numpy()
    side = max(1, int(round(math.sqrt(float(magnitude.size)))))
    usable = min(int(magnitude.size), side * side)
    grid = np.zeros((side, side), dtype=np.float32)
    grid.reshape(-1)[:usable] = magnitude.reshape(-1)[:usable]
    maximum = max(float(np.max(grid)), 1.0e-12)
    normalized = np.asarray(np.round(np.clip(grid / maximum, 0.0, 1.0) * 255.0), dtype=np.uint8)
    colored_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    return _label_visual_pane(colored_bgr[..., ::-1], "MARKER MAG")


def _marker_visual_pane(
    runtime: StandaloneTactileRuntime,
    outputs: dict[str, torch.Tensor],
    focus_index: int,
) -> np.ndarray | None:
    marker = outputs.get("marker_motion")
    if not isinstance(marker, torch.Tensor) or marker.ndim != 4 or int(marker.shape[0]) < 1:
        return None
    try:
        adapter = getattr(runtime, "_brainco_rl_hydroshear_adapter")
        depth = getattr(runtime, "_rl_tacmap_penetration_m")
        surface_points = getattr(runtime, "_rl_tacmap_surface_points_w")
        surface_valid = getattr(runtime, "_rl_tacmap_surface_valid")
        sensor_count = int(marker.shape[1])
        if not (
            isinstance(depth, torch.Tensor)
            and isinstance(surface_points, torch.Tensor)
            and isinstance(surface_valid, torch.Tensor)
            and depth.ndim == 4
            and surface_points.ndim == 5
            and surface_valid.ndim == 4
        ):
            raise RuntimeError("HydroShear render inputs are unavailable")
        rendered = adapter.render_displacement_output(
            marker.reshape(runtime.num_envs * sensor_count, marker.shape[2], 3),
            depth[:, :sensor_count].reshape(runtime.num_envs * sensor_count, *depth.shape[2:]),
            surface_points[:, :sensor_count].reshape(
                runtime.num_envs * sensor_count,
                *surface_points.shape[2:],
            ),
            surface_valid[:, :sensor_count].reshape(
                runtime.num_envs * sensor_count,
                *surface_valid.shape[2:],
            ),
            render_marker_images=True,
            debug_batch_index=max(0, min(int(focus_index), sensor_count - 1)),
        )
        images = np.asarray(rendered.marker_images)
        index = max(0, min(int(focus_index), int(images.shape[0]) - 1))
        return _label_visual_pane(images[index], "MARKER")
    except Exception as exc:
        if not bool(getattr(runtime, "_standalone_marker_visual_warning", False)):
            print(f"[WARN] Calibrated marker pane unavailable; showing magnitude heatmap: {exc}", flush=True)
            runtime._standalone_marker_visual_warning = True
        return _marker_magnitude_fallback(marker, focus_index)


def build_visualization_image(
    runtime: StandaloneTactileRuntime,
    outputs: dict[str, torch.Tensor],
) -> np.ndarray:
    """Build an env_0 panel matching the selected benchmark mode."""

    focus_index = FINGER_CHOICES.index(str(args_cli.focus_finger))
    panes = [
        pane
        for pane in (
            _depth_visual_pane(outputs, focus_index),
            _rgb_visual_pane(outputs, focus_index),
            _marker_visual_pane(runtime, outputs, focus_index),
        )
        if pane is not None
    ]
    if not panes:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    gap = np.full((240, 8, 3), 32, dtype=np.uint8)
    image = panes[0]
    for pane in panes[1:]:
        image = np.concatenate((image, gap, pane), axis=1)
    return np.ascontiguousarray(image)


class StandaloneObservationPanel:
    """Small Kit panel for display-only inspection of env_0 tactile outputs."""

    def __init__(self, image: np.ndarray) -> None:
        import omni.ui as ui

        self._height, self._width = (int(value) for value in image.shape[:2])
        self._window = ui.Window(
            f"Standalone Tactile — {selected_mode_name()}",
            width=self._width + 24,
            height=self._height + 86,
        )
        with self._window.frame:
            with ui.VStack(spacing=6):
                self._stats = ui.Label("waiting")
                self._provider = ui.ByteImageProvider()
                ui.ImageWithProvider(self._provider, width=self._width, height=self._height)
        self.update(image, step=0)

    def update(self, image: np.ndarray, *, step: int) -> None:
        rgb = np.ascontiguousarray(image, dtype=np.uint8)
        alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=-1))
        self._provider.set_bytes_data(memoryview(rgba.reshape(-1)), (self._width, self._height))
        self._stats.text = (
            f"mode={selected_mode_name()} env_0/{int(args_cli.num_envs)} "
            f"focus={args_cli.focus_finger} step={int(step)}"
        )


def git_metadata() -> dict[str, str | bool | None]:
    def run(*git_args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *git_args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        value = completed.stdout.strip()
        return value if completed.returncode == 0 and value else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "short_commit": run("rev-parse", "--short", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def result_path() -> Path:
    if args_cli.output is not None:
        path = Path(args_cli.output).expanduser()
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        REPO_ROOT
        / "outputs"
        / "revo3_tactile_standalone_benchmark"
        / f"{timestamp}__mode-{selected_mode_name()}__envs-{int(args_cli.num_envs):04d}.json"
    )


def make_result(
    runtime: StandaloneTactileRuntime,
    step_times_s: list[float],
    outputs: dict[str, torch.Tensor],
    nvml_sampler: NvmlMemorySampler,
) -> dict[str, Any]:
    times = np.asarray(step_times_s, dtype=np.float64)
    mean_s = float(np.mean(times))
    sim_hz = 1.0 / mean_s if mean_s > 0.0 else 0.0
    target_hz = 1.0 / runtime.step_dt
    gpu: dict[str, Any] = {
        "device": runtime.device,
        "name": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    if torch.cuda.is_available() and runtime.device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(runtime.device)
        gpu.update(
            {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(runtime.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime.device)),
            }
        )
    gpu["nvml"] = nvml_sampler.result()
    return {
        "schema_version": 2,
        "benchmark": "revo3_standalone_raw_tactile",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_metadata(),
        "runtime": {
            "scene": "isaaclab.scene.InteractiveScene",
            "gym_environment": False,
            "rl_environment": False,
            "observation_manager": False,
            "action_manager": False,
            "reward_manager": False,
            "policy_inference": False,
            "raw_tactile_only": True,
        },
        "configuration": {
            "num_envs": runtime.num_envs,
            "env_spacing_m": float(args_cli.env_spacing),
            "replicate_physics": False,
            "warmup_steps": int(args_cli.warmup_steps),
            "measurement_mode": "seconds" if args_cli.measure_seconds is not None else "steps",
            "requested_measure_steps": (
                None if args_cli.measure_seconds is not None else int(args_cli.measure_steps)
            ),
            "requested_measure_seconds": (
                None if args_cli.measure_seconds is None else float(args_cli.measure_seconds)
            ),
            "measured_steps": len(step_times_s),
            "physics_dt_s": runtime.physics_dt,
            "decimation": int(args_cli.decimation),
            "control_dt_s": runtime.step_dt,
            "target_sensor_hz": target_hz,
            "sync_cuda_per_step": bool(args_cli.sync_cuda),
            "implementation": str(args_cli.implementation),
            "mode": selected_mode_name(),
            "modalities": list(selected_modalities()) if str(args_cli.implementation) == "ours" else [],
            "focus_finger": str(args_cli.focus_finger),
            "presser": "square_4",
            "presser_body": "kinematic",
            "presser_collision": bool(args_cli.presser_collision),
            "press_start_offset_m": float(args_cli.press_start_offset),
            "press_end_offset_m": float(args_cli.press_end_offset),
            "press_steps": int(args_cli.press_steps),
            "hold_steps": int(args_cli.hold_steps),
            "release_steps": int(args_cli.release_steps),
            "phase_stagger": bool(args_cli.phase_stagger),
            "headless": bool(args_cli.headless),
            "visualization_enabled": bool(args_cli.visualize),
            "visualized_environment": 0 if bool(args_cli.visualize) else None,
            "visualization_update_interval": int(args_cli.visualize_update_interval),
            "visualization_in_timed_region": False,
        },
        "metrics": {
            "total_time_s": float(np.sum(times)),
            "mean_step_s": mean_s,
            "median_step_s": float(np.median(times)),
            "std_step_s": float(np.std(times)),
            "min_step_s": float(np.min(times)),
            "p95_step_s": float(np.percentile(times, 95)),
            "p99_step_s": float(np.percentile(times, 99)),
            "max_step_s": float(np.max(times)),
            "sim_hz": sim_hz,
            "env_steps_per_s": float(runtime.num_envs) * sim_hz,
            "physics_steps_per_s": float(args_cli.decimation) * sim_hz,
            "real_time_factor": sim_hz / target_hz,
        },
        "gpu": gpu,
        "observations": {name: tensor_summary(value) for name, value in outputs.items()},
    }


def main() -> None:
    validate_args()
    source_cfg, kernel_params = configure_scene()
    sim = sim_utils.SimulationContext(source_cfg.sim)
    scene = InteractiveScene(source_cfg.scene)
    sim.reset()

    runtime = StandaloneTactileRuntime(
        sim,
        scene,
        source_cfg,
        decimation=int(args_cli.decimation),
    )
    apply_touch_physics(runtime)
    reset_scene(runtime)
    hold = HardRobotHold(runtime)
    link_name, center_l, normal_l = press_target(str(args_cli.focus_finger))
    nvml_sampler = NvmlMemorySampler(runtime.device)
    nvml_sampler.sample()

    measured_times: list[float] = []
    measured_elapsed_s = 0.0
    measurement_target_s = (
        None if args_cli.measure_seconds is None else float(args_cli.measure_seconds)
    )
    outputs: dict[str, torch.Tensor] = {}
    panel: StandaloneObservationPanel | None = None
    control_step_index = 0
    while True:
        measuring = control_step_index >= int(args_cli.warmup_steps)
        if measuring:
            if control_step_index == int(args_cli.warmup_steps):
                if torch.cuda.is_available() and runtime.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(runtime.device)
            synchronize(runtime)
            start = time.perf_counter()
        outputs = control_step(
            runtime,
            hold,
            kernel_params,
            link_name=link_name,
            center_l=center_l,
            normal_l=normal_l,
            control_step_index=control_step_index,
        )
        if measuring:
            synchronize(runtime)
            measured_step_s = time.perf_counter() - start
            measured_times.append(measured_step_s)
            measured_elapsed_s += measured_step_s
            nvml_sampler.sample()
        if bool(args_cli.visualize) and control_step_index % int(args_cli.visualize_update_interval) == 0:
            image = build_visualization_image(runtime, outputs)
            if panel is None:
                panel = StandaloneObservationPanel(image)
            else:
                panel.update(image, step=control_step_index)
            if runtime.sim.has_gui() or runtime.sim.has_rtx_sensors():
                runtime.sim.render()
            else:
                simulation_app.update()
        control_step_index += 1
        if measuring:
            if measurement_target_s is not None and measured_elapsed_s >= measurement_target_s:
                break
            if measurement_target_s is None and len(measured_times) >= int(args_cli.measure_steps):
                break

    if torch.cuda.is_available() and runtime.device.startswith("cuda"):
        # Peak memory should describe steady execution, not lazy initialization.
        # A second short measured sample is not needed; warm-up already populated
        # the allocator and the peak remains useful as an upper bound.
        synchronize(runtime)

    nvml_sampler.sample()
    result = make_result(runtime, measured_times, outputs, nvml_sampler)
    output_path = result_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"[done] standalone tactile benchmark: {output_path}", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:  # keep a usable traceback even when Kit teardown is unhealthy
        traceback.print_exc()
        exit_code = 1
    finally:
        # Every invocation owns a fresh process and has no Replicator writers.
        # On this custom-sensor scene Kit can block indefinitely while
        # tearing down the USD stage. Results are already closed by write_text;
        # flushing the streams and exiting the owned process lets the OS release
        # CUDA/USD resources reliably between benchmark runs.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
