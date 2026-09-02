#!/usr/bin/env python3
"""Collect synchronized RGB/Depth/Marker data for the index fingertip.

This is a separate entrypoint.  It reuses the proven press-control helpers in
``visualize_rl_tactile_obs.py`` without modifying that script.

Example:
    python scripts/tactile_representation/collect_index_cross_modal.py \
        --sampling-mode sweep --output datasets/revo3_index_cross_modal_v1 --headless
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
VISUALIZER_PATH = REPO_ROOT / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
COLLECTION_UTILS_PATH = (
    SOURCE_ROOT / "BrainCo_DexHand" / "tactile_representation" / "collection.py"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _load_collection_utils():
    """Load the Isaac-independent utilities without importing the torch package."""

    spec = importlib.util.spec_from_file_location(
        "_tactile_collection_utils",
        COLLECTION_UTILS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load collection utilities from {COLLECTION_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_collection = _load_collection_utils()
EpisodePlan = _collection.EpisodePlan
NpzShardWriter = _collection.NpzShardWriter
PHASE_CODES = _collection.PHASE_CODES
SWEEP_STAGE_CODES = _collection.SWEEP_STAGE_CODES
make_cross_modal_sample = _collection.make_cross_modal_sample
make_episode_plans = _collection.make_episode_plans
make_sweep_episode_plans = _collection.make_sweep_episode_plans
batch_episode_plans = _collection.batch_episode_plans
merge_part_datasets = _collection.merge_part_datasets


# ``ball_probe`` is a WarpSDF calibration asset and is not a PhysX rigid body
# under the visualizer's collision-aware dynamic-force mode. Keep this
# collector on the two pressers verified for dynamic-axis force collection.
PRESSER_CHOICES = ("square_4", "cylinder_D4")
FINGER_NAME = "index"


@dataclass
class PressRuntime:
    gym_env: Any
    env: Any
    env_cfg: Any
    component_cfg: dict[str, Any]

    def close(self) -> None:
        self.gym_env.close()


@dataclass(frozen=True)
class BatchGeometry:
    robot_hold: Any
    tac_link: str
    tac_center_l: np.ndarray
    tac_normal_l: np.ndarray
    tac_slide_axis_l: np.ndarray
    tac_obj_quat_l: np.ndarray
    slide_distance_m: np.ndarray
    active_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect synchronized index-finger RGB, metric Depth, and 2D Marker Motion."
    )
    parser.add_argument("--task", default="BrainCo-Dexsuite-Revo3-Right-Lift-v0")
    parser.add_argument("--output", type=Path, default=Path("datasets/revo3_index_cross_modal_v1"))
    parser.add_argument(
        "--sampling-mode",
        choices=("sweep", "random"),
        default="sweep",
        help="Use the deterministic staircase protocol (default) or legacy random plans.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Limit/repeat the sweep length; omitted means the full sweep (30 for random mode).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Simulator seed; also controls episode generation in random mode.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel Isaac environments used by each presser worker.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print episode plans without starting Isaac Sim.",
    )
    parser.add_argument("--worker-config", type=Path, default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--pressers",
        nargs="+",
        choices=PRESSER_CHOICES,
        default=list(PRESSER_CHOICES),
    )
    parser.add_argument(
        "--force-levels-n",
        type=float,
        nargs="+",
        default=(5.0, 10.0, 20.0),
    )
    parser.add_argument("--offset-u-range-mm", type=float, default=3.0)
    parser.add_argument("--offset-v-range-mm", type=float, default=3.0)
    parser.add_argument("--offset-step-mm", type=float, default=1.0)
    parser.add_argument("--tilt-max-deg", type=float, default=15.0)
    parser.add_argument("--tilt-step-deg", type=float, default=5.0)
    parser.add_argument("--tilt-probability", type=float, default=0.8)
    parser.add_argument("--slide-max-mm", type=float, default=4.0)
    parser.add_argument("--slide-step-mm", type=float, default=1.0)
    parser.add_argument("--slide-probability", type=float, default=0.75)
    parser.add_argument("--sweep-reference-force-n", type=float, default=10.0)

    parser.add_argument("--baseline-steps", type=int, default=12)
    parser.add_argument("--press-steps", type=int, default=120)
    parser.add_argument("--slide-steps", type=int, default=40)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-marker-3d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-marker-count", type=int, default=100)
    parser.add_argument("--contact-threshold-mm", type=float, default=0.001)

    parser.add_argument("--press-start-offset", type=float, default=0.035)
    parser.add_argument("--press-end-offset", type=float, default=0.017)
    parser.add_argument("--presser-axis-max-travel", type=float, default=0.5)
    parser.add_argument("--presser-axis-max-speed", type=float, default=0.08)
    parser.add_argument(
        "--robot-hold-mode",
        choices=("target", "sharpa", "root", "free_focus", "hard", "none"),
        default="hard",
    )
    parser.add_argument(
        "--taxim-rgb-background",
        choices=("marker", "markerless", "taxim"),
        default="marker",
    )
    parser.add_argument("--taxim-rgb-with-shadow", action="store_true", default=False)
    parser.add_argument("--tacmap-contact-shell", type=float, default=0.0)
    parser.add_argument("--tacmap-local-roi-margin-mm", type=float, default=1.0)
    args = parser.parse_args()
    if args.worker_config is not None:
        with args.worker_config.expanduser().resolve().open("r", encoding="utf-8") as file_obj:
            worker_payload = json.load(file_obj)
        for name, value in worker_payload["args"].items():
            setattr(args, name, Path(value) if name == "output" else value)
        args._worker_plans = worker_payload["plans"]
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes is not None and int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive")
    if int(args.num_envs) <= 0:
        raise ValueError("--num-envs must be positive")
    if int(args.save_every) <= 0 or int(args.shard_size) <= 0:
        raise ValueError("--save-every and --shard-size must be positive")
    if int(args.expected_marker_count) <= 0:
        raise ValueError("--expected-marker-count must be positive")
    if float(args.contact_threshold_mm) < 0.0:
        raise ValueError("--contact-threshold-mm must be non-negative")


def build_plans(args: argparse.Namespace) -> list[EpisodePlan]:
    common = {
        "pressers": args.pressers,
        "force_levels_n": args.force_levels_n,
        "offset_u_range_m": float(args.offset_u_range_mm) * 1.0e-3,
        "offset_v_range_m": float(args.offset_v_range_mm) * 1.0e-3,
        "tilt_max_deg": float(args.tilt_max_deg),
        "slide_max_m": float(args.slide_max_mm) * 1.0e-3,
        "baseline_steps": int(args.baseline_steps),
        "press_steps": int(args.press_steps),
        "slide_steps": int(args.slide_steps),
        "hold_steps": int(args.hold_steps),
    }
    if str(args.sampling_mode) == "sweep":
        return make_sweep_episode_plans(
            episode_count=None if args.episodes is None else int(args.episodes),
            reference_force_n=float(args.sweep_reference_force_n),
            offset_step_m=float(args.offset_step_mm) * 1.0e-3,
            tilt_step_deg=float(args.tilt_step_deg),
            slide_step_m=float(args.slide_step_mm) * 1.0e-3,
            **common,
        )
    return make_episode_plans(
        episode_count=30 if args.episodes is None else int(args.episodes),
        seed=int(args.seed),
        tilt_probability=float(args.tilt_probability),
        slide_probability=float(args.slide_probability),
        **common,
    )


def visualizer_argv(args: argparse.Namespace, first_plan: EpisodePlan) -> list[str]:
    argv = [
        str(VISUALIZER_PATH),
        "--task",
        str(args.task),
        "--num_envs",
        str(args.num_envs),
        "--focus-finger",
        FINGER_NAME,
        "--target",
        "tacmap",
        "--presser",
        first_plan.presser,
        "--press-motion-actor",
        "object",
        "--presser-body-mode",
        "dynamic_axis",
        "--presser-force-n",
        str(first_plan.target_force_n),
        "--presser-force-ramp-steps",
        "0",
        "--presser-axis-max-travel",
        str(args.presser_axis_max_travel),
        "--presser-axis-max-speed",
        str(args.presser_axis_max_speed),
        "--enable-presser-collision",
        "--robot-hold-mode",
        str(args.robot_hold_mode),
        "--press-start-offset",
        str(args.press_start_offset),
        "--press-end-offset",
        str(args.press_end_offset),
        "--press-steps",
        str(args.press_steps),
        "--press-slide-steps",
        str(args.slide_steps),
        "--tacmap-contact-shell",
        str(args.tacmap_contact_shell),
        "--tacmap-resize-mode",
        "surface",
        "--tacmap-local-roi-margin-mm",
        str(args.tacmap_local_roi_margin_mm),
        "--taxim-rgb-background",
        str(args.taxim_rgb_background),
        "--show-taxim-rgb",
        "--show-hydroshear-marker",
        "--hide-surface-debug",
        "--hide-hydroshear-roi",
        "--no-local-ui",
        "--print-every",
        "0",
        "--device",
        str(args.device),
    ]
    if bool(args.taxim_rgb_with_shadow):
        argv.append("--taxim-rgb-with-shadow")
    if bool(args.headless):
        argv.append("--headless")
    return argv


def load_press_visualizer(args: argparse.Namespace, first_plan: EpisodePlan):
    """Load the existing visualizer as an unmodified press-runtime module."""

    if not VISUALIZER_PATH.is_file():
        raise FileNotFoundError(VISUALIZER_PATH)
    previous_argv = sys.argv
    sys.argv = visualizer_argv(args, first_plan)
    try:
        spec = importlib.util.spec_from_file_location("_index_cross_modal_press_runtime", VISUALIZER_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load press runtime from {VISUALIZER_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = previous_argv


def apply_plan_to_runtime_args(viz, args: argparse.Namespace, plan: EpisodePlan) -> None:
    viz.args_cli.presser = plan.presser
    viz.args_cli.presser_force_n = float(plan.target_force_n)
    viz.args_cli.presser_force_ramp_steps = 0
    viz.args_cli.press_steps = int(plan.press_steps)
    viz.args_cli.press_slide_steps = int(plan.slide_steps)
    viz.args_cli.press_slide_axis = str(plan.slide_axis)
    viz.args_cli.press_slide_distance = float(plan.slide_distance_m)
    viz.args_cli.tacmap_press_tilt_axis = str(plan.tilt_axis)
    viz.args_cli.tacmap_press_tilt_deg = float(plan.tilt_deg)
    viz.args_cli.press_start_offset = float(args.press_start_offset)
    viz.args_cli.press_end_offset = float(args.press_end_offset)


def create_runtime(viz, args: argparse.Namespace) -> PressRuntime:
    env_cfg = viz.configure_env()
    env_cfg.seed = int(args.seed)
    # HydroShear materializes the shared local TacMap dependency. Keep that
    # runtime term and read raw RGB/Depth directly, so the collector does not
    # spend time running the policy's frozen ResNet encoders.
    proprio = env_cfg.observations.proprio
    proprio.rl_ours_pressure = None
    proprio.rl_ours_tacmap_policy = None
    proprio.rl_ours_taxim_rgb = None
    env_cfg.observations.perception.object_point_cloud.params["visualize"] = False
    gym_env = viz.gym.make(viz.args_cli.task, cfg=env_cfg, render_mode=None)
    env = gym_env.unwrapped
    object_pose_term = env.command_manager.get_term("object_pose")
    pose_visualizer = getattr(object_pose_term, "curr_visualizer", None)
    if pose_visualizer is not None:
        pose_visualizer.set_visibility(False)
    env.reset()
    viz.initialize_visual_local_tacmap_refinement(env)
    component_cfg = viz.ours_rl_component_params(env_cfg)
    required = {
        "local_tacmap_enabled",
        "taxim_rgb_render_rows",
        "taxim_rgb_render_cols",
        "hydroshear_marker_layout_path",
    }
    missing = sorted(name for name in required if name not in component_cfg)
    if missing:
        gym_env.close()
        raise RuntimeError(f"RL tactile component configuration is missing: {missing}")
    if not bool(component_cfg["local_tacmap_enabled"]):
        gym_env.close()
        raise RuntimeError("Cross-modal collection requires local_tacmap_enabled=True")
    return PressRuntime(gym_env=gym_env, env=env, env_cfg=env_cfg, component_cfg=component_cfg)


def _tangent_axis(viz, link_name: str, axis_name: str, normal_l: np.ndarray) -> np.ndarray:
    params = viz.TIANJI_TACMAP_LINK_SURFACE_DEFAULTS[link_name]
    axis = np.asarray(
        viz.axis_to_vector(
            axis_name,
            grid_u_axis=str(params["grid_u_axis"]),
            grid_v_axis=str(params["grid_v_axis"]),
            ray_axis=str(params["ray_axis"]),
            ray_direction=tuple(float(value) for value in normal_l),
        ),
        dtype=np.float32,
    )
    axis = axis - normal_l * float(np.dot(axis, normal_l))
    return viz.normalize_np(axis, np.asarray((0.0, 1.0, 0.0), dtype=np.float32))


def prepare_batch(
    viz,
    runtime: PressRuntime,
    args: argparse.Namespace,
    plans: list[EpisodePlan],
) -> BatchGeometry:
    """Reset all environments and assign one compatible plan to each slot."""

    env = runtime.env
    if not plans or len(plans) > int(env.num_envs):
        raise ValueError(f"Batch must contain between 1 and {env.num_envs} plans")
    env.reset()
    padded_plans = plans + [plans[-1]] * (int(env.num_envs) - len(plans))
    robot_hold = viz.RobotHold(
        env,
        focus_finger=FINGER_NAME,
        press_joint_names=viz.default_press_joint_names(FINGER_NAME, "tacmap"),
        sharpa_finger_target_gain=float(viz.args_cli.sharpa_finger_target_gain),
    )

    links: list[str] = []
    centers: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    slide_axes: list[np.ndarray] = []
    object_quats: list[np.ndarray] = []
    slide_distances: list[float] = []
    for plan in padded_plans:
        apply_plan_to_runtime_args(viz, args, plan)
        tac_link, center_l, normal_l, slide_axis_l, tilt_quat_l = viz.tacmap_target(FINGER_NAME)
        normal_l = np.asarray(normal_l, dtype=np.float32)
        u_axis = _tangent_axis(viz, tac_link, "+u", normal_l)
        v_axis = _tangent_axis(viz, tac_link, "+v", normal_l)
        center_l = (
            np.asarray(center_l, dtype=np.float32)
            + u_axis * float(plan.offset_u_m)
            + v_axis * float(plan.offset_v_m)
        )
        object_quat_l = np.asarray(
            viz.quat_mul_tuple(
                tuple(float(value) for value in tilt_quat_l),
                tuple(float(value) for value in viz.presser_quat_l()),
            ),
            dtype=np.float32,
        )
        links.append(str(tac_link))
        centers.append(center_l)
        normals.append(normal_l)
        slide_axes.append(np.asarray(slide_axis_l, dtype=np.float32))
        object_quats.append(object_quat_l)
        slide_distances.append(abs(float(plan.slide_distance_m)))
    if len(set(links)) != 1:
        raise RuntimeError(f"A batch must target one tactile link, got {sorted(set(links))}")

    geometry = BatchGeometry(
        robot_hold=robot_hold,
        tac_link=links[0],
        tac_center_l=np.stack(centers),
        tac_normal_l=np.stack(normals),
        tac_slide_axis_l=np.stack(slide_axes),
        tac_obj_quat_l=np.stack(object_quats),
        slide_distance_m=np.asarray(slide_distances, dtype=np.float32),
        active_count=len(plans),
    )
    env._rl_tactile_viz_tacmap_link = geometry.tac_link
    clear_presser_wrench(viz, env)
    initialize_batch_presser(viz, env, args, geometry)
    return geometry


def clear_presser_wrench(viz, env) -> None:
    obj = env.scene["object"]
    composer = getattr(obj, "permanent_wrench_composer", None)
    if composer is not None and hasattr(composer, "reset"):
        composer.reset()
    elif hasattr(obj, "set_external_force_and_torque"):
        body_count = max(1, int(getattr(obj, "num_bodies", 1)))
        zeros = viz.torch.zeros((env.num_envs, body_count, 3), device=env.device, dtype=viz.torch.float32)
        obj.set_external_force_and_torque(forces=zeros, torques=zeros, is_global=True)


def phase_schedule(plan: EpisodePlan):
    for step in range(plan.baseline_steps):
        yield "baseline", min(step, max(0, plan.press_steps - 1)), 0.0, 0.0
    for step in range(plan.press_steps):
        alpha = 1.0 if plan.press_steps == 1 else float(step) / float(plan.press_steps - 1)
        force_n = plan.target_force_n * float(step + 1) / float(plan.press_steps)
        yield "loading", step, alpha, force_n
    for step in range(plan.slide_steps):
        yield "sliding", plan.press_steps + step, 1.0, plan.target_force_n
    final_local_step = max(0, plan.press_steps + plan.slide_steps - 1)
    for _step in range(plan.hold_steps):
        yield "holding", final_local_step, 1.0, plan.target_force_n


def initialize_batch_presser(
    viz,
    env,
    args: argparse.Namespace,
    geometry: BatchGeometry,
) -> None:
    """Place and constrain a different dynamic presser pose in each environment."""

    robot = env.scene["robot"]
    obj = env.scene["object"]
    link_idx = viz.body_index(robot, geometry.tac_link)
    link_state = robot.data.body_link_state_w[:, link_idx, :7]
    link_pos_w = link_state[:, :3]
    link_quat_w = link_state[:, 3:7]
    centers_l = viz.torch.as_tensor(geometry.tac_center_l, device=env.device, dtype=viz.torch.float32)
    normals_l = viz.torch.as_tensor(geometry.tac_normal_l, device=env.device, dtype=viz.torch.float32)
    slide_axes_l = viz.torch.as_tensor(
        geometry.tac_slide_axis_l,
        device=env.device,
        dtype=viz.torch.float32,
    )
    object_quats_l = viz.torch.as_tensor(
        geometry.tac_obj_quat_l,
        device=env.device,
        dtype=viz.torch.float32,
    )
    start_offset = float(args.press_start_offset)
    object_pos_l = centers_l + normals_l * start_offset
    object_pos_w = link_pos_w + viz.quat_apply(link_quat_w, object_pos_l)
    object_quat_w = viz.quat_mul(link_quat_w, object_quats_l)
    obj.write_root_pose_to_sim(viz.torch.cat((object_pos_w, object_quat_w), dim=-1))
    obj.write_root_velocity_to_sim(
        viz.torch.zeros((env.num_envs, 6), device=env.device, dtype=viz.torch.float32)
    )
    obj.update(0.0)

    axis_w = viz.quat_apply(link_quat_w, normals_l)
    axis_w = axis_w / viz.torch.clamp(viz.torch.linalg.norm(axis_w, dim=-1, keepdim=True), min=1.0e-8)
    slide_axis_w = viz.quat_apply(link_quat_w, slide_axes_l)
    slide_axis_w = slide_axis_w - axis_w * viz.torch.sum(slide_axis_w * axis_w, dim=-1, keepdim=True)
    slide_axis_w = slide_axis_w / viz.torch.clamp(
        viz.torch.linalg.norm(slide_axis_w, dim=-1, keepdim=True),
        min=1.0e-8,
    )
    travel = max(0.0, float(args.press_start_offset) - float(args.press_end_offset))
    env._rl_dynamic_axis_anchor_w = obj.data.root_pos_w.detach().clone()
    env._rl_dynamic_axis_axis_w = axis_w.detach().clone()
    env._rl_dynamic_axis_drive_axis_w = (-axis_w).detach().clone()
    env._rl_dynamic_axis_slide_axis_w = slide_axis_w.detach().clone()
    env._rl_dynamic_axis_quat_w = obj.data.root_quat_w.detach().clone()
    env._rl_dynamic_axis_max_travel_m = max(travel, float(args.presser_axis_max_travel))
    env._rl_dynamic_axis_max_slide_m = float(np.max(geometry.slide_distance_m, initial=0.0))
    env._rl_dynamic_axis_target_slide_m = viz.torch.zeros(
        (env.num_envs, 1),
        device=env.device,
        dtype=viz.torch.float32,
    )
    env._rl_dynamic_axis_disp_m = viz.torch.zeros(
        (env.num_envs,),
        device=env.device,
        dtype=viz.torch.float32,
    )
    env._rl_dynamic_axis_slide_disp_m = viz.torch.zeros_like(env._rl_dynamic_axis_disp_m)


def drive_batch_presser(viz, env, target_slide_m: np.ndarray) -> None:
    obj = env.scene["object"]
    obj.update(0.0)
    drive_axis_w = env._rl_dynamic_axis_drive_axis_w
    slide_axis_w = env._rl_dynamic_axis_slide_axis_w
    pos_w = obj.data.root_pos_w
    velocity_w = obj.data.root_vel_w[:, :3]
    axial_speed = viz.torch.sum(velocity_w * drive_axis_w, dim=-1, keepdim=True)
    linear_velocity_w = drive_axis_w * axial_speed
    slide_disp = viz.torch.sum(
        (pos_w - env._rl_dynamic_axis_anchor_w) * slide_axis_w,
        dim=-1,
        keepdim=True,
    )
    slide_target = viz.torch.as_tensor(
        target_slide_m,
        device=env.device,
        dtype=viz.torch.float32,
    ).reshape(env.num_envs, 1)
    slide_target = viz.torch.clamp(
        slide_target,
        min=0.0,
        max=float(env._rl_dynamic_axis_max_slide_m),
    )
    env._rl_dynamic_axis_target_slide_m = slide_target.detach().clone()
    dt = max(float(getattr(env, "physics_dt", 1.0 / 120.0)), 1.0e-6)
    slide_speed = (slide_target - slide_disp) / dt
    max_speed = max(0.0, float(viz.args_cli.presser_axis_max_speed))
    if max_speed > 0.0:
        slide_speed = viz.torch.clamp(slide_speed, min=-max_speed, max=max_speed)
    linear_velocity_w = linear_velocity_w + slide_axis_w * slide_speed
    root_velocity_w = viz.torch.cat(
        (
            linear_velocity_w,
            viz.torch.zeros((env.num_envs, 3), device=env.device, dtype=viz.torch.float32),
        ),
        dim=-1,
    )
    obj.write_root_velocity_to_sim(root_velocity_w)
    obj.update(0.0)


def apply_batch_forces(viz, env, force_n: np.ndarray) -> None:
    obj = env.scene["object"]
    forces = viz.torch.as_tensor(force_n, device=env.device, dtype=viz.torch.float32).reshape(
        env.num_envs,
        1,
        1,
    )
    body_count = max(1, int(getattr(obj, "num_bodies", 1)))
    forces_w = env._rl_dynamic_axis_drive_axis_w[:, None, :].expand(
        env.num_envs,
        body_count,
        3,
    ) * forces
    torques_w = viz.torch.zeros_like(forces_w)
    composer = getattr(obj, "permanent_wrench_composer", None)
    if composer is not None and hasattr(composer, "set_forces_and_torques"):
        if hasattr(composer, "reset"):
            composer.reset()
        composer.set_forces_and_torques(forces=forces_w, torques=torques_w, is_global=True)
    elif hasattr(obj, "set_external_force_and_torque"):
        obj.set_external_force_and_torque(forces=forces_w, torques=torques_w, is_global=True)
    env._rl_dynamic_axis_force_n = forces.reshape(-1).detach().clone()


def write_batch_state(
    viz,
    env,
    geometry: BatchGeometry,
    force_n: np.ndarray,
    target_slide_m: np.ndarray,
) -> None:
    geometry.robot_hold.write(str(viz.args_cli.robot_hold_mode))
    drive_batch_presser(viz, env, target_slide_m)
    apply_batch_forces(viz, env, force_n)


def step_batch(
    viz,
    runtime: PressRuntime,
    geometry: BatchGeometry,
    *,
    phase: str,
    force_n: np.ndarray,
    target_slide_m: np.ndarray,
    global_step: int,
) -> None:
    env = runtime.env
    env._rl_tactile_viz_phase = "tacmap_hold" if phase == "holding" else "tacmap"
    env._rl_tactile_viz_global_step = int(global_step)
    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    render_interval = max(1, int(getattr(env.cfg.sim, "render_interval", 1)))
    decimation = max(1, int(getattr(env.cfg, "decimation", 1)))
    for _ in range(decimation):
        env._sim_step_counter += 1
        write_batch_state(viz, env, geometry, force_n, target_slide_m)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        write_batch_state(viz, env, geometry, force_n, target_slide_m)
        viz.project_dynamic_axis_presser(env)
        env.scene.write_data_to_sim()
        if env._sim_step_counter % render_interval == 0 and is_rendering:
            env.sim.render()
    env.episode_length_buf += 1
    env.common_step_counter += 1
    if hasattr(env, "observation_manager"):
        env.obs_buf = env.observation_manager.compute(update_history=True)


def _to_numpy(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def batch_normal_forces(viz, env, geometry: BatchGeometry) -> np.ndarray:
    sensor = env.scene.sensors.get(f"{geometry.tac_link}_object_s")
    force_matrix = None if sensor is None else getattr(getattr(sensor, "data", None), "force_matrix_w", None)
    if not isinstance(force_matrix, viz.torch.Tensor) or force_matrix.numel() == 0:
        return np.zeros((env.num_envs,), dtype=np.float32)
    force_w = viz.torch.nan_to_num(force_matrix.reshape(env.num_envs, -1, 3)).sum(dim=1)
    normal_force = viz.torch.abs(
        viz.torch.sum(force_w * env._rl_dynamic_axis_axis_w, dim=-1)
    )
    return np.asarray(normal_force.detach().cpu().numpy(), dtype=np.float32)


def extract_batch_samples(
    viz,
    runtime: PressRuntime,
    args: argparse.Namespace,
    plans: list[EpisodePlan],
    geometry: BatchGeometry,
    *,
    phase: str,
    episode_step: int,
    global_step: int,
    alpha: float,
    commanded_force_n: np.ndarray,
    wave_id: int,
) -> list[dict[str, np.ndarray]]:
    env = runtime.env
    render_rows = int(runtime.component_cfg["taxim_rgb_render_rows"])
    render_cols = int(runtime.component_cfg["taxim_rgb_render_cols"])
    finger_count = len(viz.FINGER_CHOICES)
    finger_index = viz.FINGER_CHOICES.index(FINGER_NAME)

    rgb_flat = viz.ours_rl_taxim_rgb_obs(env, runtime.component_cfg)
    rgb_batch = _to_numpy(rgb_flat).reshape(
        env.num_envs,
        finger_count,
        3,
        render_rows,
        render_cols,
    )[:, finger_index]
    dense_depth = getattr(env, "_rl_ours_dense_tacmap_depth_m", None)
    if dense_depth is None:
        raise RuntimeError("Dense metric Depth cache was not produced")
    depth_batch = _to_numpy(dense_depth)[:, finger_index]

    output = getattr(env, "_rl_hydroshear_output", None)
    if output is None:
        raise RuntimeError("HydroShear debug output was not cached")
    marker_flow_batch = _to_numpy(output.marker_flow)
    marker_displacement_batch = _to_numpy(output.displacement_m)
    if marker_flow_batch.ndim != 4 or marker_flow_batch.shape[1] != 2:
        raise RuntimeError(f"Unexpected marker_flow shape: {marker_flow_batch.shape}")
    expected_flat_slots = int(env.num_envs) * finger_count
    if int(marker_flow_batch.shape[0]) != expected_flat_slots:
        raise RuntimeError(
            f"Marker flow has {marker_flow_batch.shape[0]} slots, expected {expected_flat_slots}"
        )

    layout = getattr(env, "_brainco_rl_hydroshear_marker_layout", None)
    if not isinstance(layout, tuple) or len(layout) < 4:
        raise RuntimeError("Calibrated HydroShear marker validity layout is unavailable")
    marker_valid = np.asarray(layout[3], dtype=bool)[finger_index]
    measured_forces_n = batch_normal_forces(viz, env, geometry)
    samples: list[dict[str, np.ndarray]] = []
    for env_id, plan in enumerate(plans):
        flat_slot = env_id * finger_count + finger_index
        depth = depth_batch[env_id]
        max_depth_m = float(np.max(np.nan_to_num(depth, nan=0.0)))
        contact = max_depth_m > float(args.contact_threshold_mm) * 1.0e-3
        marker_displacement = (
            marker_displacement_batch[flat_slot]
            if bool(args.save_marker_3d)
            else None
        )
        scalar_fields = {
            "episode_id": np.int32(plan.episode_id),
            "episode_step": np.int32(episode_step),
            "global_step": np.int64(global_step),
            "wave_id": np.int32(wave_id),
            "env_id": np.int16(env_id),
            "phase": np.int8(PHASE_CODES[phase]),
            "contact": np.bool_(contact),
            "press_alpha": np.float32(alpha),
            "commanded_force_n": np.float32(commanded_force_n[env_id]),
            "target_force_n": np.float32(plan.target_force_n),
            "measured_normal_force_n": np.float32(measured_forces_n[env_id]),
            "max_depth_m": np.float32(max_depth_m),
            "offset_u_m": np.float32(plan.offset_u_m),
            "offset_v_m": np.float32(plan.offset_v_m),
            "tilt_deg": np.float32(plan.tilt_deg),
            "slide_distance_m": np.float32(plan.slide_distance_m),
            "presser_id": np.int8(PRESSER_CHOICES.index(plan.presser)),
            "sweep_stage": np.int8(SWEEP_STAGE_CODES[plan.sweep_stage]),
        }
        samples.append(
            make_cross_modal_sample(
                rgb_chw=rgb_batch[env_id],
                depth_m=depth,
                marker_flow=marker_flow_batch[flat_slot],
                marker_valid=marker_valid,
                marker_displacement_3d_m=marker_displacement,
                scalar_fields=scalar_fields,
                expected_marker_count=int(args.expected_marker_count),
            )
        )
    return samples


def metadata(args: argparse.Namespace, plans: list[EpisodePlan]) -> dict[str, Any]:
    return {
        "dataset": "revo3_index_cross_modal",
        "finger": FINGER_NAME,
        "modalities": {
            "rgb": {"layout": "CHW", "dtype": "uint8", "range": [0, 255]},
            "depth_m": {"layout": "CHW", "dtype": "float32", "unit": "m"},
            "marker_2d": {
                "layout": "NF",
                "dtype": "float32",
                "columns": ["x0_px", "y0_px", "dx_px", "dy_px", "valid"],
            },
            "marker_displacement_3d_m": {
                "enabled": bool(args.save_marker_3d),
                "layout": "N3",
                "dtype": "float32",
                "unit": "m",
            },
        },
        "phase_codes": PHASE_CODES,
        "sweep_stage_codes": SWEEP_STAGE_CODES,
        "presser_codes": {name: index for index, name in enumerate(PRESSER_CHOICES)},
        "task": str(args.task),
        "sampling_mode": str(args.sampling_mode),
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "save_every": int(args.save_every),
        "expected_marker_count": int(args.expected_marker_count),
        "contact_threshold_m": float(args.contact_threshold_mm) * 1.0e-3,
        "episode_plans": [plan.to_json() for plan in plans],
        "press_runtime": str(VISUALIZER_PATH.relative_to(REPO_ROOT)),
        "old_scripts_modified": False,
    }


def collect(args: argparse.Namespace, plans: list[EpisodePlan]) -> None:
    selected_pressers = {plan.presser for plan in plans}
    if len(selected_pressers) != 1:
        raise ValueError("One Isaac worker can collect exactly one presser type")
    viz = load_press_visualizer(args, plans[0])
    writer = NpzShardWriter(
        args.output,
        shard_size=int(args.shard_size),
        compressed=bool(args.compressed),
        metadata=metadata(args, plans),
    )
    runtime: PressRuntime | None = None
    global_step = 0
    try:
        apply_plan_to_runtime_args(viz, args, plans[0])
        runtime = create_runtime(viz, args)
        waves = batch_episode_plans(
            sorted(plans, key=lambda item: item.episode_id),
            int(args.num_envs),
        )
        for wave_id, wave_plans in enumerate(waves):
            geometry = prepare_batch(viz, runtime, args, wave_plans)
            reference_plan = wave_plans[0]
            target_forces_n = np.zeros((runtime.env.num_envs,), dtype=np.float32)
            target_forces_n[: len(wave_plans)] = [plan.target_force_n for plan in wave_plans]
            episode_step = 0
            saved_per_plan = 0
            for phase, local_step, alpha, _reference_force_n in phase_schedule(reference_plan):
                if not viz.simulation_app.is_running():
                    raise RuntimeError("Isaac Sim stopped before collection completed")
                if phase == "baseline":
                    force_scale = 0.0
                elif phase == "loading":
                    force_scale = float(local_step + 1) / float(reference_plan.press_steps)
                else:
                    force_scale = 1.0
                commanded_forces_n = target_forces_n * force_scale
                if phase == "sliding":
                    slide_progress = float(local_step - reference_plan.press_steps + 1) / float(
                        max(1, reference_plan.slide_steps)
                    )
                elif phase == "holding":
                    slide_progress = 1.0
                else:
                    slide_progress = 0.0
                target_slide_m = geometry.slide_distance_m * np.float32(slide_progress)
                target_slide_m[len(wave_plans) :] = 0.0
                step_batch(
                    viz,
                    runtime,
                    geometry,
                    phase=phase,
                    force_n=commanded_forces_n,
                    target_slide_m=target_slide_m,
                    global_step=global_step,
                )
                if episode_step % int(args.save_every) == 0:
                    samples = extract_batch_samples(
                        viz,
                        runtime,
                        args,
                        wave_plans,
                        geometry,
                        phase=phase,
                        episode_step=episode_step,
                        global_step=global_step,
                        alpha=alpha,
                        commanded_force_n=commanded_forces_n,
                        wave_id=wave_id,
                    )
                    for sample in samples:
                        writer.append(sample)
                    saved_per_plan += 1
                episode_step += 1
                global_step += 1
            clear_presser_wrench(viz, runtime.env)
            episode_ids = ",".join(str(plan.episode_id) for plan in wave_plans)
            print(
                f"[wave {wave_id:03d}] presser={reference_plan.presser} "
                f"envs={len(wave_plans)}/{runtime.env.num_envs} episodes=[{episode_ids}] "
                f"saved_per_episode={saved_per_plan}",
                flush=True,
            )
        writer.close(status="complete")
    except BaseException:
        writer.close(status="interrupted")
        raise
    finally:
        if runtime is not None:
            runtime.close()
        viz.simulation_app.close()

    print(
        f"[done] samples={writer.sample_count} shards={writer.shard_count} "
        f"manifest={writer.manifest_path}",
        flush=True,
    )


def _json_ready_args(args: argparse.Namespace, *, output: Path, presser: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, value in vars(args).items():
        if name.startswith("_") or name in {"worker_config", "plan_only"}:
            continue
        values[name] = str(value) if isinstance(value, Path) else value
    values["output"] = str(output)
    values["pressers"] = [presser]
    values["plan_only"] = False
    return values


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def collect_multiple_pressers(args: argparse.Namespace, plans: list[EpisodePlan]) -> None:
    output_root = args.output.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use a new directory to avoid overwriting data."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    grouped = {
        presser: [plan for plan in plans if plan.presser == presser]
        for presser in PRESSER_CHOICES
        if any(plan.presser == presser for plan in plans)
    }
    part_dirs: list[Path] = []
    try:
        for presser, presser_plans in grouped.items():
            part_dir = output_root / "parts" / presser
            config_path = output_root / "worker_configs" / f"{presser}.json"
            payload = {
                "args": _json_ready_args(args, output=part_dir, presser=presser),
                "plans": [plan.to_json() for plan in presser_plans],
            }
            _write_json_atomic(config_path, payload)
            print(
                f"[parent] launching presser={presser} episodes={len(presser_plans)}",
                flush=True,
            )
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker-config", str(config_path)],
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Presser worker {presser!r} exited with code {result.returncode}")
            part_dirs.append(part_dir)
        manifest_path = merge_part_datasets(
            output_root,
            part_dirs,
            metadata=metadata(args, plans),
        )
    except BaseException:
        _write_json_atomic(
            output_root / "manifest.json",
            {
                "schema_version": 1,
                "status": "interrupted",
                "metadata": metadata(args, plans),
            },
        )
        raise
    print(f"[done] merged multi-presser dataset: {manifest_path}", flush=True)


def main() -> None:
    args = parse_args()
    plans = (
        [EpisodePlan(**value) for value in args._worker_plans]
        if hasattr(args, "_worker_plans")
        else build_plans(args)
    )
    if bool(args.plan_only):
        print(json.dumps([plan.to_json() for plan in plans], ensure_ascii=False, indent=2))
        return
    if args.worker_config is None and len({plan.presser for plan in plans}) > 1:
        collect_multiple_pressers(args, plans)
    else:
        collect(args, plans)


if __name__ == "__main__":
    main()
