# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone ALOHA tactile force-map demo.

This script spawns the migrated ALOHA tactile robot asset, a target cuboid, and
four Warp SDF tactile sensors. It prints per-pad force maxima and optionally
shows the force maps with OpenCV.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn ALOHA and display Warp SDF tactile force maps.")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum simulation steps. Use -1 to run forever.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps before reporting force maps.")
parser.add_argument("--show_cv", action="store_true", help="Show force maps in an OpenCV window.")
parser.add_argument(
    "--save_force_maps",
    action="store_true",
    help="Save force-map images to outputs/force_map when OpenCV window display is unavailable.",
)
parser.add_argument("--save_every", type=int, default=30, help="Save one force-map image every N steps.")
parser.add_argument("--tactile_scale", type=int, default=8, help="Nearest-neighbor display scale for force maps.")
parser.add_argument("--target_pos", type=float, nargs=3, default=(0.0, 0.0, 0.05), help="Target cuboid xyz.")
parser.add_argument("--target_size", type=float, nargs=3, default=(0.06, 0.06, 0.06), help="Target cuboid size.")
parser.add_argument(
    "--sdf_backend",
    choices=("box", "mesh"),
    default="box",
    help="Use the built-in analytic box SDF for a cuboid demo, or query the spawned USD mesh.",
)
parser.add_argument(
    "--no_auto_contact",
    action="store_true",
    help="Do not move the target box to a tactile pad. Use --target_pos exactly as provided.",
)
parser.add_argument(
    "--contact_slot",
    type=int,
    default=0,
    choices=(0, 1, 2, 3),
    help="Tactile slot used by auto-contact mode: 0=L-L, 1=L-R, 2=R-L, 3=R-R.",
)
parser.add_argument("--num_rows", type=int, default=12, help="Taxel rows per pad.")
parser.add_argument("--num_cols", type=int, default=32, help="Taxel columns per pad.")
parser.add_argument("--point_distance", type=float, default=0.002, help="Taxel spacing in meters.")
parser.add_argument("--normal_axis", type=int, default=0, choices=(0, 1, 2), help="Patch-local normal axis.")
parser.add_argument("--normal_offset", type=float, default=0.0036, help="Patch-local normal offset in meters.")
parser.add_argument("--stiffness", type=float, default=5000.0, help="Penalty stiffness for force-map computation.")
parser.add_argument("--max_force", type=float, default=10.0, help="Force clamp before normalization.")
parser.add_argument("--pressure_gain", type=float, default=1.0, help="Calibrated pressure gain applied after stiffness.")
parser.add_argument("--pressure_bias", type=float, default=0.0, help="Calibrated pressure bias before clamping.")
parser.add_argument("--pressure_gamma", type=float, default=1.0, help="Calibrated pressure gamma after normalization.")
parser.add_argument("--pressure_threshold", type=float, default=0.0, help="Raw pressure threshold before normalization.")
parser.add_argument("--taxel_area", type=float, default=1.0, help="Taxel area/weight used by calibrated pressure maps.")
parser.add_argument("--debug_vis", action="store_true", help="Show contact taxel markers in the Isaac viewport.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim.schemas import activate_contact_sensors  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

from BrainCo_DexHand.assets import ALOHA_TACTILE_CFG, ALOHA_TACTILE_URDF_PATH  # noqa: E402
from BrainCo_DexHand.force_map import WarpSdfTactileSensor, WarpSdfTactileSensorCfg  # noqa: E402


ROBOT_PRIM_PATH = "/World/Robot"
TARGET_PRIM_PATH = "/World/TargetBox"
SENSOR_LABELS = ["L-L", "L-R", "R-L", "R-R"]
PATCH_OFFSET_QUAT = (0.7071068, 0.0, 0.0, -0.7071068)


def _parse_elastomer_origins(urdf_path: str | Path) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    out = {}
    root = ET.parse(os.path.expanduser(str(urdf_path))).getroot()
    for joint in root.findall("joint"):
        name = joint.get("name", "").lower()
        if "elastomer_joint_left" in name:
            key = "left"
        elif "elastomer_joint_right" in name:
            key = "right"
        else:
            continue

        origin = joint.find("origin")
        if origin is None:
            continue
        xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
        rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
        if len(xyz) == 3 and len(rpy) == 3:
            out[key] = (xyz, rpy)
    return out


def _infer_arm(path: str) -> str | None:
    text = path.lower()
    if "left_arm_" in text or "/left/" in text:
        return "left"
    if "right_arm_" in text or "/right/" in text:
        return "right"
    return None


def _infer_finger(path: str) -> str | None:
    text = path.lower()
    if "elastomer_left" in text or "_left_finger_link" in text:
        return "left_finger"
    if "elastomer_right" in text or "_right_finger_link" in text:
        return "right_finger"
    return None


def _sensor_slot(path: str) -> int:
    arm = _infer_arm(path)
    finger = _infer_finger(path)
    if arm is None or finger is None:
        return 0
    return (0 if arm == "left" else 2) + (0 if finger == "left_finger" else 1)


def _sort_elastomer_links(paths: list[str]) -> list[str]:
    def sort_key(path: str):
        arm_order = 0 if _infer_arm(path) == "left" else 1
        finger_order = 0 if _infer_finger(path) == "left_finger" else 1
        return (arm_order, finger_order, path)

    return sorted(paths, key=sort_key)


def _find_elastomer_links(sim_utils_module) -> list[str]:
    bodies = sim_utils_module.get_all_matching_child_prims(
        ROBOT_PRIM_PATH,
        predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and prim.HasAPI(PhysxSchema.PhysxContactReportAPI),
        traverse_instance_prims=False,
    )
    elastomers = [prim.GetPath().pathString for prim in bodies if "elastomer" in prim.GetPath().pathString.lower()]
    if not elastomers:
        raise RuntimeError("No elastomer links found. Check that ALOHA contact sensors are active.")
    return _sort_elastomer_links(elastomers)[:4]


def _resolve_mesh_prim(root_path: str) -> str:
    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Invalid target prim: {root_path}")
    if root.IsA(UsdGeom.Mesh):
        return root_path
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            return prim.GetPath().pathString
    raise RuntimeError(f"No UsdGeom.Mesh found under {root_path}")


def _compute_patch_transform(
    link_path: str,
    origins: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    lower = link_path.lower()
    if "_left_finger_link" in lower:
        side = "left"
    elif "_right_finger_link" in lower:
        side = "right"
    else:
        return (0.0, 0.0, 0.0), PATCH_OFFSET_QUAT

    if side not in origins:
        return (0.0, 0.0, 0.0), PATCH_OFFSET_QUAT

    base_xyz, base_rpy = origins[side]
    base_xyz_t = torch.tensor(base_xyz, dtype=torch.float32).unsqueeze(0)
    base_rpy_t = torch.tensor(base_rpy, dtype=torch.float32).unsqueeze(0)
    user_quat_t = torch.tensor(PATCH_OFFSET_QUAT, dtype=torch.float32).unsqueeze(0)
    q_be = math_utils.quat_from_euler_xyz(base_rpy_t[:, 0], base_rpy_t[:, 1], base_rpy_t[:, 2])
    quat_bp = math_utils.quat_mul(q_be, user_quat_t).squeeze(0)
    pos_bp = base_xyz_t.squeeze(0)
    return tuple(float(v) for v in pos_bp), tuple(float(v) for v in quat_bp)


def _spawn_world_and_target():
    sim_utils.spawn_ground_plane("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light(
        prim_path="/World/Light/DomeLight",
        cfg=sim_utils.DomeLightCfg(intensity=2000),
        translation=(-4.5, 3.5, 10.0),
    )
    sim_utils.spawn_mesh_cuboid(
        prim_path=TARGET_PRIM_PATH,
        cfg=sim_utils.MeshCuboidCfg(
            size=tuple(float(v) for v in args.target_size),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.8)),
        ),
        translation=tuple(float(v) for v in args.target_pos),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _make_force_strip(tactile: np.ndarray, scale: int) -> np.ndarray:
    images = []
    for i, grid in enumerate(tactile):
        vmax = float(grid.max())
        norm = grid / (vmax + 1.0e-8) if vmax > 0 else np.zeros_like(grid)
        rgb = _jet_colormap(norm)
        if scale > 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        images.append(rgb)
    return np.concatenate(images, axis=1)


def _maybe_show_cv(image_rgb: np.ndarray):
    if not args.show_cv:
        return
    try:
        import cv2
    except ImportError:
        print("[WARN] OpenCV is not installed. Re-run without --show_cv or install opencv-python.", flush=True)
        args.show_cv = False
        return
    try:
        cv2.imshow("ALOHA tactile force maps", image_rgb[:, :, ::-1])
        cv2.waitKey(1)
    except cv2.error as exc:
        print(f"[WARN] OpenCV window display is unavailable, disabling --show_cv: {exc}", flush=True)
        args.show_cv = False


def _maybe_save_force_map(image_rgb: np.ndarray, step: int):
    if not args.save_force_maps:
        return
    save_every = max(1, int(args.save_every))
    if step % save_every != 0:
        return
    out_dir = REPO_ROOT / "outputs" / "force_map"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(image_rgb).save(out_dir / f"force_map_{step:06d}.png")
    except ImportError:
        np.save(out_dir / f"force_map_{step:06d}.npy", image_rgb)


def _set_target_visual_position(pos_w: np.ndarray):
    """Best-effort visual alignment for auto-contact mode."""
    try:
        prim = sim_utils.get_current_stage().GetPrimAtPath(TARGET_PRIM_PATH)
        if prim.IsValid():
            UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in pos_w))
    except Exception as exc:
        print(f"[WARN] Could not move target visual prim: {exc}", flush=True)


def _set_sensor_target_pose(
    sensor: WarpSdfTactileSensor,
    pos_w: tuple[float, float, float] | np.ndarray | torch.Tensor,
    quat_w: tuple[float, float, float, float] | np.ndarray | torch.Tensor,
):
    if args.sdf_backend == "mesh":
        sensor.set_target_pose(pos_w, quat_w)
    else:
        sensor.set_box_pose(pos_w, quat_w)


def _make_auto_contact_pose(
    sensors: list[WarpSdfTactileSensor],
    slot_order: list[int],
    target_quat: tuple[float, float, float, float],
    dt: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    if args.no_auto_contact:
        return None

    sensor_idx = None
    for idx, slot in enumerate(slot_order):
        if slot == int(args.contact_slot):
            sensor_idx = idx
            break
    if sensor_idx is None:
        print(f"[WARN] Could not find tactile slot {args.contact_slot}; auto-contact disabled.", flush=True)
        return None

    # Force one update so tactile point world positions are valid.
    sensors[sensor_idx].update(dt=dt)
    data = sensors[sensor_idx].data.tactile_points_w
    if data is None:
        print("[WARN] Tactile data is not initialized; auto-contact disabled.", flush=True)
        return None

    points_w = data[0, :, :3]
    center_w = points_w.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    _set_target_visual_position(center_w)
    print(
        f"[INFO] Auto-contact enabled: target centered on slot {args.contact_slot} "
        f"at ({center_w[0]:.4f}, {center_w[1]:.4f}, {center_w[2]:.4f}).",
        flush=True,
    )
    return center_w, target_quat


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
        device=str(getattr(args, "device", "cuda:0") or "cuda:0"),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    _spawn_world_and_target()

    robot_cfg = ALOHA_TACTILE_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    robot = Articulation(robot_cfg)
    activate_contact_sensors(ROBOT_PRIM_PATH, threshold=0.0)

    target_mesh_path = _resolve_mesh_prim(TARGET_PRIM_PATH)
    elastomer_paths = _find_elastomer_links(sim_utils)
    origins = _parse_elastomer_origins(ALOHA_TACTILE_URDF_PATH)

    sensors = []
    slot_order = []
    for link_path in elastomer_paths:
        patch_pos, patch_quat = _compute_patch_transform(link_path, origins)
        sensor_cfg = WarpSdfTactileSensorCfg(
            prim_path=ROBOT_PRIM_PATH,
            elastomer_prim_paths=[link_path],
            num_rows=args.num_rows,
            num_cols=args.num_cols,
            point_distance=args.point_distance,
            normal_axis=args.normal_axis,
            normal_offset=args.normal_offset,
            patch_offset_pos_b=patch_pos,
            patch_offset_quat_b=patch_quat,
            box_pos_w=tuple(float(v) for v in args.target_pos),
            box_quat_w=(1.0, 0.0, 0.0, 0.0),
            box_half_extents=tuple(float(v) * 0.5 for v in args.target_size),
            target_mesh_prim_path=target_mesh_path if args.sdf_backend == "mesh" else None,
            mesh_max_dist=0.20,
            mesh_use_signed_distance=False,
            mesh_signed_distance_method="winding",
            mesh_smooth_normals=True,
            mesh_shell_thickness=0.001,
            mesh_unsigned_shell_as_contact=True,
            mesh_unsigned_contact_mode="normal_ray",
            stiffness=args.stiffness,
            max_force=args.max_force,
            pressure_gain=args.pressure_gain,
            pressure_bias=args.pressure_bias,
            pressure_gamma=args.pressure_gamma,
            pressure_threshold=args.pressure_threshold,
            taxel_area=args.taxel_area,
            normalize_forces=True,
            debug_vis=args.debug_vis,
        )
        sensors.append(WarpSdfTactileSensor(sensor_cfg))
        slot_order.append(_sensor_slot(link_path))
        print(f"[INFO] tactile slot={slot_order[-1]} elastomer={link_path} target={target_mesh_path}", flush=True)

    sim.reset()
    robot.update(1.0 / 120.0)

    init_joint_pos = robot.data.default_joint_pos.clone()
    init_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel)
    robot.reset()

    target_prim = sim_utils.get_current_stage().GetPrimAtPath(target_mesh_path)
    dt = 1.0 / 120.0
    target_pos, target_quat = sim_utils.resolve_prim_pose(target_prim)
    for sensor in sensors:
        _set_sensor_target_pose(sensor, target_pos, target_quat)
    auto_contact_pose = _make_auto_contact_pose(sensors, slot_order, target_quat, dt)

    step = 0
    max_steps = int(args.max_steps)

    while simulation_app.is_running() and (max_steps < 0 or step < max_steps):
        robot.set_joint_position_target(init_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=not args.headless or args.show_cv or args.debug_vis)
        robot.update(dt)

        if auto_contact_pose is None:
            target_pos, target_quat = sim_utils.resolve_prim_pose(target_prim)
        else:
            target_pos, target_quat = auto_contact_pose
        tactile = np.zeros((4, args.num_rows, args.num_cols), dtype=np.float32)
        for sensor, slot in zip(sensors, slot_order, strict=True):
            _set_sensor_target_pose(sensor, target_pos, target_quat)
            sensor.update(dt=dt)
            data = sensor.data.tactile_points_w
            if data is not None:
                force = data[0, :, 3].detach().cpu().numpy().astype(np.float32)
                tactile[slot] = force.reshape(args.num_rows, args.num_cols)

        if step >= args.warmup_steps:
            if step % 30 == 0:
                maxes = ", ".join(f"{SENSOR_LABELS[i]}={float(tactile[i].max()):.4f}" for i in range(4))
                print(f"[step {step:06d}] {maxes}", flush=True)
            force_strip = _make_force_strip(tactile, args.tactile_scale)
            _maybe_show_cv(force_strip)
            _maybe_save_force_map(force_strip, step)

        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
