# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RL-side ball-probe pressure-pad calibration.

This runner keeps the RL Revo3 pressure-pad WarpSDF sensors, replaces the task
object with the real ball_probe mesh, and scripts the probe along its local -Y
axis for small post-contact indentation sweeps.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

FINGER_CHOICES = ("middle", "index", "ring", "pinky", "thumb")
PRESSURE_PAD_SEGMENTS = ("mcp", "pip")
PRESSURE_PAD_FINGER_STEMS = {
    "middle": "mid",
    "index": "index",
    "ring": "ring",
    "pinky": "pinky",
    "thumb": "thumb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="BrainCo-Dexsuite-Revo3-Right-Lift-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--focus-finger", choices=FINGER_CHOICES, default="index")
    parser.add_argument("--pressure-pad-segment", choices=PRESSURE_PAD_SEGMENTS, default="mcp")
    parser.add_argument("--press-start-offset", type=float, default=0.025)
    parser.add_argument("--press-contact-search-distance", type=float, default=0.02)
    parser.add_argument("--press-indent-depth", type=float, default=0.0005)
    parser.add_argument("--pressure-pad-press-steps", type=int, default=26)
    parser.add_argument("--press-indent-settle-steps", type=int, default=10)
    parser.add_argument("--press-contact-threshold", type=float, default=1.0e-7)
    parser.add_argument("--presser-scale", type=float, default=1.0)
    parser.add_argument("--presser-mass", type=float, default=0.2)
    parser.add_argument("--presser-axis-max-speed", type=float, default=0.005)
    parser.add_argument("--enable-presser-collision", action="store_true", default=False)
    parser.add_argument("--pressure-pad-contact-offset", type=float, default=0.0005)
    parser.add_argument("--pressure-pad-rest-offset", type=float, default=-0.0017)
    parser.add_argument("--pressure-pad-taxel-surface-offset", type=float, default=0.0)
    parser.add_argument("--save-pressure-trace", action="store_true", default=False)
    parser.add_argument(
        "--pressure-trace-dir",
        default=str(REPO_ROOT / "outputs" / "rl_ball_probe_pressure_calib"),
    )
    parser.add_argument("--pressure-trace-run-id", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--show-pressure-window", action="store_true", default=False)
    parser.add_argument("--pressure-window-scale", type=int, default=20)
    parser.add_argument("--lock-viewport-camera", action="store_true", default=False)
    parser.add_argument("--viewport-side-distance", type=float, default=0.12)
    parser.add_argument("--viewport-side-height", type=float, default=0.03)
    parser.add_argument("--viewport-target-outward-offset", type=float, default=0.004)
    parser.add_argument("--viewport-side-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402

from BrainCo_DexHand.force_map import load_pressure_pad_specs_from_urdf  # noqa: E402
from BrainCo_DexHand.force_map.rl_ball_probe_pressure_calib import (  # noqa: E402
    BallProbePressureCalibState,
    RlPressureTraceRecorder,
    validate_press_indent_depth,
)
from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.dexsuite_revo3_env_cfg_grasp import (  # noqa: E402
    TIANJI_PRESSURE_PAD_LINK_ORDER,
    TIANJI_PRESSURE_SENSOR_NAMES,
    TIANJI_PRESSUREPAD_URDF,
)


BALL_PROBE_USD = REPO_ROOT / "tacmap" / "assets" / "presser" / "ball_probe.usd"
BALL_PROBE_PHYSICS_USD = REPO_ROOT / "outputs" / "rl_ball_probe_pressure_calib" / "assets" / "ball_probe.physics.usd"
BALL_PROBE_PRESS_AXIS_L = torch.tensor((0.0, -1.0, 0.0), dtype=torch.float32)


def pressure_pad_link_name(finger: str, segment: str) -> str:
    return f"right_{PRESSURE_PAD_FINGER_STEMS[str(finger)]}{segment}_roll_touch_link"


def body_index(robot: Articulation, link_name: str) -> int:
    for i, name in enumerate(robot.body_names):
        if str(name) == str(link_name) or str(name).endswith(str(link_name)):
            return int(i)
    raise RuntimeError(f"Body {link_name!r} not found in robot body names.")


def normalize_np(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1.0e-8:
        vec = np.asarray(fallback, dtype=np.float32)
        norm = max(float(np.linalg.norm(vec)), 1.0e-8)
    return vec / norm


def quat_from_two_vectors_np(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = normalize_np(src, np.asarray((1.0, 0.0, 0.0), dtype=np.float32)).astype(np.float64)
    dst = normalize_np(dst, np.asarray((1.0, 0.0, 0.0), dtype=np.float32)).astype(np.float64)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1.0e-8:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    if dot < -1.0 + 1.0e-8:
        axis = np.cross(src, np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
        if np.linalg.norm(axis) < 1.0e-8:
            axis = np.cross(src, np.asarray((0.0, 1.0, 0.0), dtype=np.float64))
        axis = axis / max(float(np.linalg.norm(axis)), 1.0e-8)
        return np.asarray((0.0, axis[0], axis[1], axis[2]), dtype=np.float32)
    axis = np.cross(src, dst)
    quat = np.asarray((1.0 + dot, axis[0], axis[1], axis[2]), dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-8)
    return quat.astype(np.float32)


def ensure_ball_probe_physics_usd() -> Path:
    if not BALL_PROBE_USD.is_file():
        raise FileNotFoundError(f"ball_probe USD not found: {BALL_PROBE_USD}")
    BALL_PROBE_PHYSICS_USD.parent.mkdir(parents=True, exist_ok=True)
    if BALL_PROBE_PHYSICS_USD.is_file() and BALL_PROBE_PHYSICS_USD.stat().st_mtime >= BALL_PROBE_USD.stat().st_mtime:
        return BALL_PROBE_PHYSICS_USD

    import shutil
    from pxr import Usd, UsdGeom, UsdPhysics

    shutil.copyfile(BALL_PROBE_USD, BALL_PROBE_PHYSICS_USD)
    stage = Usd.Stage.Open(str(BALL_PROBE_PHYSICS_USD))
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPrimAtPath("/ball_probe")
    if not root or not root.IsValid():
        raise RuntimeError(f"Could not find default prim in {BALL_PROBE_PHYSICS_USD}")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(args_cli.presser_mass))
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim)
    stage.GetRootLayer().Save()
    return BALL_PROBE_PHYSICS_USD


def configure_env():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=int(args_cli.num_envs),
        use_fabric=not bool(args_cli.disable_fabric),
    )
    env_cfg.scene.object.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ensure_ball_probe_physics_usd()),
        scale=(float(args_cli.presser_scale),) * 3,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=bool(args_cli.enable_presser_collision),
            contact_offset=0.0001,
            rest_offset=-0.0005,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=float(args_cli.presser_mass)),
    )
    env_cfg.scene.object.init_state.pos = (0.0, 0.0, 2.0)
    env_cfg.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    if hasattr(env_cfg.scene, "table"):
        env_cfg.scene.table.spawn.visible = False
        env_cfg.scene.table.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    if hasattr(env_cfg.commands, "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False
    env_cfg.curriculum = None
    for event_name in ("randomize_object_scale", "object_scale_mass", "reset_table", "reset_object", "variable_gravity"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    event = getattr(env_cfg.events, "zz_revo3_pressure_pad_collision_properties", None)
    if event is not None:
        event.params["contact_offset"] = float(args_cli.pressure_pad_contact_offset)
        event.params["rest_offset"] = float(args_cli.pressure_pad_rest_offset)
    pressure_specs_by_link = {
        str(spec.link_name): spec for spec in load_pressure_pad_specs_from_urdf(TIANJI_PRESSUREPAD_URDF, require_files=False)
    }
    for link_name, sensor_name in zip(TIANJI_PRESSURE_PAD_LINK_ORDER, TIANJI_PRESSURE_SENSOR_NAMES, strict=True):
        cfg = getattr(env_cfg.scene, str(sensor_name), None)
        if cfg is not None:
            spec = pressure_specs_by_link.get(str(link_name))
            if spec is None:
                raise RuntimeError(f"Pressure pad link {link_name!r} is not declared in {TIANJI_PRESSUREPAD_URDF}.")
            cfg.store_debug_fields = True
            cfg.target_mesh_prim_path = "/World/envs/env_0/Object"
            cfg.normal_offset = float(spec.normal_offset) + float(spec.normal_sign) * float(
                args_cli.pressure_pad_taxel_surface_offset
            )
    env_cfg.observations.proprio.contact = None
    env_cfg.observations.proprio.rl_tacmap = None
    env_cfg.observations.perception = None
    return env_cfg


def pressure_pad_target(link_name: str) -> tuple[str, np.ndarray, np.ndarray]:
    specs = load_pressure_pad_specs_from_urdf(TIANJI_PRESSUREPAD_URDF, require_files=False)
    spec = next((item for item in specs if str(item.link_name) == str(link_name)), None)
    if spec is None:
        raise RuntimeError(f"Pressure pad link {link_name!r} is not declared in {TIANJI_PRESSUREPAD_URDF}.")
    taxel_map = spec.to_taxel_map()
    center_l = np.asarray(taxel_map.points_l, dtype=np.float32).mean(axis=0)
    fallback = np.zeros(3, dtype=np.float32)
    fallback[int(spec.normal_axis)] = 1.0 if float(spec.normal_sign) >= 0.0 else -1.0
    normal_l = normalize_np(np.asarray(taxel_map.normals_l, dtype=np.float32).mean(axis=0), fallback)
    return str(spec.link_name), center_l.astype(np.float32), normal_l.astype(np.float32)


def pressure_trace_layout_arrays(link_name: str) -> tuple[dict[str, np.ndarray], int, int]:
    specs = load_pressure_pad_specs_from_urdf(TIANJI_PRESSUREPAD_URDF, require_files=False)
    spec = next((item for item in specs if str(item.link_name) == str(link_name)), None)
    if spec is None:
        raise RuntimeError(f"Pressure pad link {link_name!r} is not declared in {TIANJI_PRESSUREPAD_URDF}.")
    taxel_map = spec.to_taxel_map()
    rows, cols = (int(v) for v in taxel_map.image_shape)
    return {
        "pressure_taxel_points_l_m": np.asarray(taxel_map.points_l, dtype=np.float32).reshape(1, rows, cols, 3),
        "pressure_taxel_normals_l": np.asarray(taxel_map.normals_l, dtype=np.float32).reshape(1, rows, cols, 3),
        "pressure_taxel_layout_valid": np.ones((1,), dtype=np.uint8),
        "pressure_taxel_link_name": np.asarray([str(link_name)], dtype="<U256"),
    }, rows, cols


def pressure_display_layouts() -> list[tuple[str, np.ndarray]]:
    specs = load_pressure_pad_specs_from_urdf(TIANJI_PRESSUREPAD_URDF, require_files=False)
    specs_by_link = {str(spec.link_name): spec for spec in specs}
    return [
        (str(link_name), np.asarray(specs_by_link[str(link_name)].to_taxel_map().points_l, dtype=np.float32))
        for link_name in TIANJI_PRESSURE_PAD_LINK_ORDER
    ]


def sync_pressure_sensors_to_object(env: ManagerBasedRLEnv) -> None:
    obj: RigidObject = env.scene["object"]
    for name in TIANJI_PRESSURE_SENSOR_NAMES:
        sensor = env.scene.sensors.get(str(name))
        if sensor is None:
            continue
        if getattr(sensor, "_target_mesh_prim_path", None) is not None and hasattr(sensor, "set_target_pose"):
            sensor.set_target_pose(obj.data.root_pos_w, obj.data.root_quat_w)
        outdated = getattr(sensor, "_is_outdated", None)
        if isinstance(outdated, torch.Tensor):
            outdated[:] = True
        sensor.update(0.0, force_recompute=True)


def pressure_tensor_map_from_env(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    attr_name: str,
    rows: int,
    cols: int,
) -> torch.Tensor:
    sensor = env.scene.sensors.get(str(sensor_name))
    data = None if sensor is None else getattr(sensor, "data", None)
    value = None if data is None else getattr(data, str(attr_name), None)
    out = torch.zeros((env.num_envs, rows, cols), device=env.device, dtype=torch.float32)
    if value is not None:
        source = value.detach()
        if source.ndim == 4:
            source = source[:, 0]
        elif source.ndim == 2:
            source = source.unsqueeze(0)
        elif source.ndim == 3 and source.shape[0] != env.num_envs:
            source = source[0].unsqueeze(0)
        source = source[: env.num_envs, :rows, :cols].to(device=env.device, dtype=torch.float32)
        out[:, : source.shape[-2], : source.shape[-1]] = torch.nan_to_num(source)
    return out.unsqueeze(1)


def pressure_trace_frame_from_env(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    rows: int,
    cols: int,
) -> dict[str, np.ndarray]:
    sync_pressure_sensors_to_object(env)
    tensors = {
        "pressure_norm": pressure_tensor_map_from_env(env, sensor_name, "pressure_force_map", rows, cols),
        "pressure_raw_n": pressure_tensor_map_from_env(env, sensor_name, "pressure_force_map_raw", rows, cols),
        "penetration_m": pressure_tensor_map_from_env(env, sensor_name, "penetration_map", rows, cols),
        "signed_distance_m": pressure_tensor_map_from_env(env, sensor_name, "signed_distance_map", rows, cols),
    }
    return {key: value[0].detach().cpu().numpy().astype(np.float32) for key, value in tensors.items()}


def pressure_map_values_from_env(env: ManagerBasedRLEnv, sensor_name: str, count: int) -> np.ndarray:
    sensor = env.scene.sensors.get(str(sensor_name))
    data = None if sensor is None else getattr(sensor, "data", None)
    value = None if data is None else getattr(data, "pressure_force_map", None)
    out = np.zeros((int(count),), dtype=np.float32)
    if value is None:
        return out
    source = value.detach()
    if source.ndim == 4:
        source = source[0, 0]
    elif source.ndim >= 2:
        source = source[0]
    flat = torch.nan_to_num(source.reshape(-1).to(dtype=torch.float32)).detach().cpu().numpy()
    size = min(out.size, flat.size)
    out[:size] = flat[:size]
    return out


def pressure_layout_pixel_centers(points_l: np.ndarray, spacing_px: int) -> tuple[np.ndarray, int, int]:
    yz = np.asarray(points_l, dtype=np.float32)[:, 1:3]
    distances = np.linalg.norm(yz[:, None] - yz[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    positive = nearest[np.isfinite(nearest) & (nearest > 1.0e-9)]
    physical_spacing = float(np.median(positive)) if positive.size else 1.0
    spacing = max(2, int(spacing_px))
    pixels_per_unit = float(spacing) / physical_spacing
    padding = max(1, spacing // 2)
    rows = np.rint((float(np.max(yz[:, 1])) - yz[:, 1]) * pixels_per_unit + padding).astype(np.int64)
    cols = np.rint((yz[:, 0] - float(np.min(yz[:, 0]))) * pixels_per_unit + padding).astype(np.int64)
    return np.column_stack((rows, cols)), int(np.max(rows) + padding + 1), int(np.max(cols) + padding + 1)


def pressure_layout_pane(values: np.ndarray, points_l: np.ndarray, spacing_px: int) -> np.ndarray:
    centers, height, width = pressure_layout_pixel_centers(points_l, spacing_px)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    vmax = float(np.max(flat)) if flat.size else 0.0
    norm = flat / (vmax + 1.0e-8) if vmax > 0.0 else np.zeros_like(flat)
    colors = np.stack(
        (
            np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0),
        ),
        axis=-1,
    )
    colors = np.asarray(colors * 255.0, dtype=np.uint8)
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    radius = max(1, int(spacing_px) // 4)
    for (row, col), color in zip(centers, colors, strict=True):
        image[
            max(0, int(row) - radius) : min(height, int(row) + radius + 1),
            max(0, int(col) - radius) : min(width, int(col) + radius + 1),
        ] = color
    return image


def tile_pressure_panes(panes: list[np.ndarray], cols: int = 6, gap: int = 4) -> np.ndarray:
    rows = []
    for start in range(0, len(panes), int(cols)):
        items = panes[start : start + int(cols)]
        height = max(int(item.shape[0]) for item in items)
        padded = [
            np.pad(item, ((0, height - int(item.shape[0])), (0, 0), (0, 0)), constant_values=32)
            for item in items
        ]
        separator = np.full((height, int(gap), 3), 32, dtype=np.uint8)
        row = padded[0]
        for item in padded[1:]:
            row = np.concatenate((row, separator, item), axis=1)
        rows.append(row)
    width = max(int(row.shape[1]) for row in rows)
    rows = [np.pad(row, ((0, 0), (0, width - int(row.shape[1])), (0, 0)), constant_values=32) for row in rows]
    separator = np.full((int(gap), width, 3), 32, dtype=np.uint8)
    return np.concatenate((rows[0], separator, rows[1]), axis=0)


def pressure_heatmap_image(
    env: ManagerBasedRLEnv,
    layouts: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    spacing = max(2, int(args_cli.pressure_window_scale))
    panes = [
        pressure_layout_pane(
            pressure_map_values_from_env(env, f"{link_name}_warpsdf_s", len(points_l)),
            points_l,
            spacing,
        )
        for link_name, points_l in layouts
    ]
    return tile_pressure_panes(panes)


class PressureHeatmapPanel:
    def __init__(self, image: np.ndarray):
        import omni.ui as ui

        self._height, self._width = (int(v) for v in image.shape[:2])
        self._window = ui.Window(
            "RL Ball Probe Pressure",
            width=max(360, self._width + 24),
            height=self._height + 104,
        )
        with self._window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Rows: middle M/P, index M/P, ring M/P | pinky M/P, thumb M/P, palm")
                self._stats = ui.Label("waiting")
                self._provider = ui.ByteImageProvider()
                ui.ImageWithProvider(self._provider, width=self._width, height=self._height)

    def update(self, image: np.ndarray, stats: str) -> None:
        alpha = np.full((self._height, self._width, 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate((image, alpha), axis=-1), dtype=np.uint8)
        self._provider.set_bytes_data(memoryview(rgba.reshape(-1)), (self._width, self._height))
        self._stats.text = str(stats)


def write_robot_hold(env: ManagerBasedRLEnv) -> None:
    robot: Articulation = env.scene["robot"]
    if not hasattr(env, "_ball_probe_calib_robot_root"):
        env._ball_probe_calib_robot_root = torch.cat((robot.data.root_pos_w.clone(), robot.data.root_quat_w.clone()), dim=-1)
        env._ball_probe_calib_joint_pos = robot.data.default_joint_pos.clone()
        env._ball_probe_calib_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_root_pose_to_sim(env._ball_probe_calib_robot_root)
    robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32))
    robot.write_joint_state_to_sim(env._ball_probe_calib_joint_pos, env._ball_probe_calib_joint_vel)
    robot.update(0.0)
    robot.set_joint_position_target(env._ball_probe_calib_joint_pos)
    robot.write_data_to_sim()


def pressure_pad_frame_w(
    env: ManagerBasedRLEnv,
    *,
    pressure_link: str,
    center_l_np: np.ndarray,
    normal_l_np: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene["robot"]
    link_idx = body_index(robot, pressure_link)
    link_state = robot.data.body_link_state_w[:, link_idx, :7]
    link_pos_w = link_state[:, :3]
    link_quat_w = link_state[:, 3:7]
    center_l = torch.as_tensor(center_l_np, device=env.device, dtype=torch.float32)
    normal_l = torch.as_tensor(normal_l_np, device=env.device, dtype=torch.float32)
    center_w = link_pos_w + quat_apply(link_quat_w, center_l.unsqueeze(0).expand(env.num_envs, -1))
    normal_w = quat_apply(link_quat_w, normal_l.unsqueeze(0).expand(env.num_envs, -1))
    normal_w = normal_w / torch.clamp(torch.linalg.norm(normal_w, dim=-1, keepdim=True), min=1.0e-8)
    return center_w, normal_w


def update_viewport_camera(
    env: ManagerBasedRLEnv,
    *,
    pressure_link: str,
    center_l_np: np.ndarray,
    normal_l_np: np.ndarray,
) -> None:
    if not bool(args_cli.lock_viewport_camera):
        return
    if not env.sim.has_gui():
        if not bool(getattr(env, "_ball_probe_viewport_camera_warned", False)):
            print("[WARN] --lock-viewport-camera ignored because Isaac GUI is not active.", flush=True)
            env._ball_probe_viewport_camera_warned = True
        return
    center_w, normal_w = pressure_pad_frame_w(
        env,
        pressure_link=pressure_link,
        center_l_np=center_l_np,
        normal_l_np=normal_l_np,
    )
    target = center_w[0] + normal_w[0] * float(args_cli.viewport_target_outward_offset)
    press_axis = -normal_w[0]
    up = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=torch.float32)
    side = torch.cross(press_axis, up, dim=0)
    if float(torch.linalg.norm(side)) < 1.0e-6:
        side = torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=torch.float32)
    side = side / torch.clamp(torch.linalg.norm(side), min=1.0e-8)
    side = side * float(args_cli.viewport_side_sign)
    eye = target + side * float(args_cli.viewport_side_distance) + up * float(args_cli.viewport_side_height)
    env.sim.set_camera_view(eye=eye.detach().cpu().tolist(), target=target.detach().cpu().tolist())


def write_probe_pose(
    env: ManagerBasedRLEnv,
    *,
    pressure_link: str,
    center_l_np: np.ndarray,
    normal_l_np: np.ndarray,
    base_offset_m: float,
    travel_m: float,
) -> tuple[float, float]:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    link_idx = body_index(robot, pressure_link)
    link_state = robot.data.body_link_state_w[:, link_idx, :7]
    link_pos_w = link_state[:, :3]
    link_quat_w = link_state[:, 3:7]
    center_l = torch.as_tensor(center_l_np, device=env.device, dtype=torch.float32)
    normal_l = torch.as_tensor(normal_l_np, device=env.device, dtype=torch.float32)
    normal_w = quat_apply(link_quat_w, normal_l.unsqueeze(0).expand(env.num_envs, -1))
    normal_w = normal_w / torch.clamp(torch.linalg.norm(normal_w, dim=-1, keepdim=True), min=1.0e-8)
    center_w = link_pos_w + quat_apply(link_quat_w, center_l.unsqueeze(0).expand(env.num_envs, -1))
    axis_w = -normal_w
    q_align_l = torch.as_tensor(
        quat_from_two_vectors_np(np.asarray((0.0, -1.0, 0.0), dtype=np.float32), (-normal_l_np).astype(np.float32)),
        device=env.device,
        dtype=torch.float32,
    )
    quat_w = quat_mul(link_quat_w, q_align_l.unsqueeze(0).expand(env.num_envs, -1))
    pos_w = center_w + normal_w * float(base_offset_m) + axis_w * float(travel_m)
    prev_pos = getattr(env, "_ball_probe_calib_prev_pos_w", None)
    root_vel = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
    if isinstance(prev_pos, torch.Tensor):
        dt = max(float(getattr(env, "physics_dt", 0.0)), 1.0e-8)
        root_vel[:, :3] = (pos_w - prev_pos) / dt
    env._ball_probe_calib_prev_pos_w = pos_w.detach().clone()
    obj.write_root_pose_to_sim(torch.cat((pos_w, quat_w), dim=-1))
    obj.write_root_velocity_to_sim(root_vel)
    obj.update(0.0)
    axis_disp_m = float(travel_m)
    root_clearance_m = float(base_offset_m - travel_m)
    return axis_disp_m, root_clearance_m


def step_physics(env: ManagerBasedRLEnv, *, pressure_link: str, center_l: np.ndarray, normal_l: np.ndarray, travel_m: float) -> float:
    speed = max(0.0, float(args_cli.presser_axis_max_speed))
    current = float(getattr(env, "_ball_probe_calib_travel_m", 0.0))
    if speed > 0.0:
        max_delta = speed * max(float(getattr(env, "physics_dt", 1.0 / 120.0)), 1.0e-8)
        target = min(float(travel_m), current + max_delta)
    else:
        target = float(travel_m)
    env._ball_probe_calib_travel_m = target
    for _ in range(max(1, int(getattr(env.cfg, "decimation", 1)))):
        write_robot_hold(env)
        write_probe_pose(
            env,
            pressure_link=pressure_link,
            center_l_np=center_l,
            normal_l_np=normal_l,
            base_offset_m=float(args_cli.press_start_offset),
            travel_m=target,
        )
        env.scene.write_data_to_sim()
        env.sim.step(render=env.sim.has_gui() or env.sim.has_rtx_sensors())
        env.scene.update(dt=env.physics_dt)
    return target


def main() -> None:
    validate_press_indent_depth(float(args_cli.press_indent_depth))
    if int(args_cli.num_envs) != 1:
        raise ValueError("RL ball-probe pressure calibration currently supports --num_envs 1.")

    env_cfg = configure_env()
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env: ManagerBasedRLEnv = gym_env.unwrapped
    env.reset()

    pressure_link, center_l, normal_l = pressure_pad_target(
        pressure_pad_link_name(str(args_cli.focus_finger), str(args_cli.pressure_pad_segment))
    )
    pressure_sensor_name = TIANJI_PRESSURE_SENSOR_NAMES[TIANJI_PRESSURE_PAD_LINK_ORDER.index(pressure_link)]
    layout_arrays, rows, cols = pressure_trace_layout_arrays(pressure_link)
    display_layouts = pressure_display_layouts()
    calib_state = BallProbePressureCalibState(
        search_distance_m=float(args_cli.press_contact_search_distance),
        indent_depth_m=float(args_cli.press_indent_depth),
        indent_steps=int(args_cli.pressure_pad_press_steps),
        settle_steps=int(args_cli.press_indent_settle_steps),
        contact_penetration_threshold_m=float(args_cli.press_contact_threshold),
        contact_force_threshold_n=float("inf"),
    )
    recorder = None
    if bool(args_cli.save_pressure_trace):
        run_id = str(args_cli.pressure_trace_run_id).strip() or (
            f"{time.strftime('%Y%m%d_%H%M%S')}_{args_cli.focus_finger}_{args_cli.pressure_pad_segment}"
        )
        recorder = RlPressureTraceRecorder(
            args_cli.pressure_trace_dir,
            {
                "run_id": run_id,
                "pressure_backend_id": "rl_warpsdf_ball_probe_mesh",
                "pressure_layout_id": f"rl_revo3:{pressure_link}:{rows}x{cols}",
                "pressure_calibration_id": "rl_ball_probe_stl_pressurepad",
                "source": "run_rl_ball_probe_pressure_calib",
                "task": str(args_cli.task),
                "ball_probe_usd": str(BALL_PROBE_USD),
                "focus_finger": str(args_cli.focus_finger),
                "pressure_pad_segment": str(args_cli.pressure_pad_segment),
                "press_axis": "ball_probe_local_-Y",
                "press_indent_depth_m": float(args_cli.press_indent_depth),
                "press_contact_search_distance_m": float(args_cli.press_contact_search_distance),
                "pressure_pad_taxel_surface_offset_m": float(args_cli.pressure_pad_taxel_surface_offset),
            },
            layout_arrays=layout_arrays,
        )

    print(
        f"[INFO] RL ball-probe mesh pressure calib: link={pressure_link} "
        f"search={float(args_cli.press_contact_search_distance) * 1000:.3f}mm "
        f"indent={float(args_cli.press_indent_depth) * 1000:.3f}mm steps={int(args_cli.pressure_pad_press_steps)} "
        f"taxel_surface_offset={float(args_cli.pressure_pad_taxel_surface_offset) * 1000:.3f}mm "
        f"mesh={BALL_PROBE_USD}",
        flush=True,
    )

    update_viewport_camera(env, pressure_link=pressure_link, center_l_np=center_l, normal_l_np=normal_l)
    step = 0
    pressure_panel = None
    max_steps = int(args_cli.max_steps) if int(args_cli.max_steps) > 0 else 100000
    while simulation_app.is_running() and step < max_steps:
        desired = calib_state.desired_travel_m
        actual_travel = step_physics(env, pressure_link=pressure_link, center_l=center_l, normal_l=normal_l, travel_m=desired)
        update_viewport_camera(env, pressure_link=pressure_link, center_l_np=center_l, normal_l_np=normal_l)
        frame = pressure_trace_frame_from_env(env, pressure_sensor_name, rows, cols)
        penetration_max = float(np.max(frame["penetration_m"])) if frame["penetration_m"].size else 0.0
        contact_force_n = 0.0
        contact_normal_force_n = 0.0
        if bool(args_cli.show_pressure_window) and env.sim.has_gui():
            heatmap = pressure_heatmap_image(env, display_layouts)
            if pressure_panel is None:
                pressure_panel = PressureHeatmapPanel(heatmap)
            pressure_panel.update(
                heatmap,
                f"step={step} {calib_state.phase} indent={calib_state.commanded_indent_m * 1000.0:.3f}mm "
                f"max={float(np.max(frame['pressure_raw_n'])):.3f} pen={penetration_max * 1000.0:.4f}mm",
            )
        should_record = calib_state.observe(
            step=step,
            penetration_max_m=penetration_max,
            normal_force_n=contact_normal_force_n,
            actual_axis_disp_m=actual_travel,
        )
        if should_record:
            actual_indent = calib_state.actual_post_contact_indent_m(actual_travel)
            if recorder is not None:
                recorder.record(
                    step=step,
                    phase=calib_state.phase,
                    commanded_indent_m=calib_state.commanded_indent_m,
                    actual_axis_disp_m=actual_travel,
                    actual_post_contact_indent_m=actual_indent,
                    contact_force_n=contact_force_n,
                    contact_normal_force_n=contact_normal_force_n,
                    contact_found=calib_state.contact_found,
                    **frame,
                )
            print(
                f"[CALIB_SAMPLE] step={step} idx={calib_state.indent_index} "
                f"cmd_indent={calib_state.commanded_indent_m * 1000.0:.4f}mm "
                f"actual_indent={actual_indent * 1000.0:.4f}mm pen_max={penetration_max * 1000.0:.5f}mm",
                flush=True,
            )
            calib_state.mark_sample_recorded()
        if int(args_cli.print_every) > 0 and step % int(args_cli.print_every) == 0:
            print(
                f"[step {step:06d}] phase={calib_state.phase} travel={actual_travel * 1000.0:.3f}mm "
                f"cmd_indent={calib_state.commanded_indent_m * 1000.0:.4f}mm "
                f"pressure_max={float(np.max(frame['pressure_raw_n'])):.4f} pen_max={penetration_max * 1000.0:.5f}mm",
                flush=True,
            )
        if calib_state.done:
            break
        step += 1

    if recorder is not None:
        if getattr(recorder, "_rows", None):
            npz_path, metadata_path = recorder.close()
            print(f"[INFO] Wrote RL pressure trace: {npz_path} metadata={metadata_path}", flush=True)
        else:
            print("[WARN] No pressure trace samples recorded; skipping trace write.", flush=True)
    gym_env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
