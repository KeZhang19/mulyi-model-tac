# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rotate-Bulb-Custom robot, actions, contact and tactile sensor configuration."""

from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import torch
import isaaclab.sim as sim_utils
# from pxr import Sdf, Usd, UsdPhysics
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from .robot_asset_cfg import TIANJI_REVO3_RIGHT_CFG
from .robot_contract import ASSET_ROOT, HAND_POINT_NAMES, TACTILE_LINK_MAP
from BrainCo_DexHand.force_map import WarpSdfTactileSensorCfg, load_pressure_pad_specs_from_urdf

from . import base_env_cfg as dexsuite
from ... import mdp


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "tacmap").is_dir() and (parent / "source").is_dir():
            return parent
    return start.resolve().parents[8]


REPO_ROOT = _find_repo_root(Path(__file__))
TACMAP_ROOT = REPO_ROOT / "tacmap"
if str(TACMAP_ROOT) not in sys.path:
    sys.path.append(str(TACMAP_ROOT))

from tacmap_sensor.sharpa_tacmap_link_surface import SharpaTacmapLinkSurface, SharpaTacmapLinkSurfaceCfg  # noqa: E402


# Task-owned HydroShear sampling defaults.
RL_HYDROSHEAR_OBJECT_SAMPLE_COUNT = 32768
RL_HYDROSHEAR_OBJECT_SAMPLE_SEED = 17
RL_HYDROSHEAR_POISSON_RADIUS = 0.00075
RL_HYDROSHEAR_POISSON_INITIAL_COUNT = 5000
RL_HYDROSHEAR_OBJECT_SAMPLE_REFERENCE_COUNT = 4096
RL_HYDROSHEAR_OBJECT_SAMPLE_ROI_COUNT = 168

ISAACLAB_ENV_REGEX_NS = "/World/envs/env_.*"
TIANJI_PALM_BODY_NAME = "palm"
# Keep logical sensor keys stable for the existing tactile collector/rewards;
# TACTILE_LINK_MAP translates them to the new asset's physical body names.
TIANJI_HAND_DIP_BODIES = [
    "right_pinkydip_roll_rubber_link",
    "right_ringdip_roll_rubber_link",
    "right_middip_roll_rubber_link",
    "right_indexdip_roll_rubber_link",
    "right_thumbdip_roll_rubber_link",
]
TIANJI_HAND_TIP_BODIES = list(HAND_POINT_NAMES)
TIANJI_RL_FINGER_ORDER = ("middle", "index", "ring", "pinky", "thumb")
TIANJI_PRESSURE_PAD_FINGER_STEMS = {
    "middle": "mid",
    "index": "index",
    "ring": "ring",
    "pinky": "pinky",
    "thumb": "thumb",
}
TIANJI_PRESSURE_PAD_SEGMENTS = ("mcp", "pip")
TIANJI_PRESSURE_PAD_LINK_ORDER = tuple(
    f"right_{TIANJI_PRESSURE_PAD_FINGER_STEMS[finger]}{segment}_roll_touch_link"
    for finger in TIANJI_RL_FINGER_ORDER
    for segment in TIANJI_PRESSURE_PAD_SEGMENTS
) + ("right_hand_rubber_link",)
TIANJI_PRESSURE_PAD_COLLISION_LINK_ORDER = tuple(
    TACTILE_LINK_MAP[f"right_{TIANJI_PRESSURE_PAD_FINGER_STEMS[finger]}{segment}_roll_touch_link"]
    for finger in TIANJI_RL_FINGER_ORDER
    for segment in TIANJI_PRESSURE_PAD_SEGMENTS
)
TIANJI_PRESSURE_SENSOR_NAMES = tuple(f"{link_name}_warpsdf_s" for link_name in TIANJI_PRESSURE_PAD_LINK_ORDER)
TIANJI_PRESSUREPAD_URDF = ASSET_ROOT / "tactile/pressure/dv2_pressure_taxel_layout.urdf"
TIANJI_TACMAP_LINK_ORDER = (
    "right_middip_roll_rubber_link",
    "right_indexdip_roll_rubber_link",
    "right_ringdip_roll_rubber_link",
    "right_pinkydip_roll_rubber_link",
    "right_thumbdip_roll_rubber_link",
)
TIANJI_TOUCH_COMPLIANT_STIFFNESS = 280.0
TIANJI_TOUCH_COMPLIANT_DAMPING = 20.0
TIANJI_TOUCH_CONTACT_OFFSET = 0.0002
TIANJI_TOUCH_REST_OFFSET = -0.0001
TIANJI_PRESSURE_PAD_CONTACT_OFFSET = 0.0002
TIANJI_PRESSURE_PAD_REST_OFFSET = -0.0017
TIANJI_TACMAP_SURFACE_SENSOR_NAMES = tuple(f"{link_name}_tacmap_surface_s" for link_name in TIANJI_TACMAP_LINK_ORDER)
TIANJI_TACMAP_OBJECT_SENSOR_NAMES = tuple(f"{link_name}_tacmap_object_s" for link_name in TIANJI_TACMAP_LINK_ORDER)
TIANJI_HYDROSHEAR_MARKER_SURFACE_SENSOR_NAMES = tuple(
    f"{link_name}_hydroshear_marker_surface_s" for link_name in TIANJI_TACMAP_LINK_ORDER
)
TIANJI_HYDROSHEAR_MARKER_OBJECT_SENSOR_NAMES = tuple(
    f"{link_name}_hydroshear_marker_object_s" for link_name in TIANJI_TACMAP_LINK_ORDER
)
TIANJI_TACMAP_DIR = ASSET_ROOT / "tactile/tacmap"
TIANJI_TACMAP_LINK_SURFACE_DEFAULTS = json.loads((TIANJI_TACMAP_DIR / "grid_defaults.json").read_text())


@sim_utils.clone
def spawn_revo3_robot_for_touch(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Expand collider instances before PhysX creates its tensor views.

    The existing startup material callbacks then edit stable shapes without
    changing their order relative to the task's material randomization.
    """
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    sim_utils.make_uninstanceable(prim.GetPath().pathString, stage=prim.GetStage())
    # spawn_from_usd cannot author overrides inside instance proxies. Apply
    # the configured offsets to the now-editable colliders before simulation.
    sim_utils.modify_collision_properties(prim.GetPath().pathString, cfg.collision_props, stage=prim.GetStage())
    return prim


def _make_revo3_robot_cfg_with_touch_compliance() -> ArticulationCfg:
    robot_cfg = TIANJI_REVO3_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    spawn = robot_cfg.spawn
    robot_cfg.spawn = sim_utils.UsdFileCfg(
        func=spawn_revo3_robot_for_touch,
        usd_path=str(spawn.usd_path),
        variants=getattr(spawn, "variants", None),
        scale=getattr(spawn, "scale", None),
        articulation_props=getattr(spawn, "articulation_props", None),
        fixed_tendons_props=getattr(spawn, "fixed_tendons_props", None),
        spatial_tendons_props=getattr(spawn, "spatial_tendons_props", None),
        joint_drive_props=getattr(spawn, "joint_drive_props", None),
        visual_material_path=getattr(spawn, "visual_material_path", "material"),
        visual_material=getattr(spawn, "visual_material", None),
        mass_props=getattr(spawn, "mass_props", None),
        rigid_props=getattr(spawn, "rigid_props", None),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=TIANJI_TOUCH_CONTACT_OFFSET,
            rest_offset=TIANJI_TOUCH_REST_OFFSET,
        ),
        activate_contact_sensors=getattr(spawn, "activate_contact_sensors", False),
        visible=getattr(spawn, "visible", True),
        semantic_tags=getattr(spawn, "semantic_tags", None),
        copy_from_source=getattr(spawn, "copy_from_source", True),
    )
    return robot_cfg


def _collision_prim_paths_under(stage, root_path: str) -> list[str]:
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []

    collision_paths: list[str] = []
    prims = [root]
    while prims:
        prim = prims.pop(0)
        if prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim):
            collision_paths.append(prim.GetPath().pathString)
        prims.extend(prim.GetChildren())
    return collision_paths


def apply_revo3_touch_compliant_materials(
    env: ManagerBasedRLEnv,
    env_ids,
    *,
    link_names: tuple[str, ...],
    stiffness: float,
    damping: float,
    contact_offset: float,
    rest_offset: float,
) -> None:
    stage = sim_utils.get_current_stage()
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        compliant_contact_stiffness=float(stiffness),
        compliant_contact_damping=float(damping),
    )
    collision_cfg = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=float(contact_offset),
        rest_offset=float(rest_offset),
    )
    env_range = range(env.num_envs) if env_ids is None else [int(env_id) for env_id in env_ids]
    success_count = 0

    for env_id in env_range:
        robot_path = f"/World/envs/env_{env_id}/Robot"
        try:
            sim_utils.make_uninstanceable(robot_path, stage=stage)
        except Exception as exc:
            print(f"[WARN] Could not make {robot_path} uninstanceable before touch material binding: {exc}", flush=True)

        for link_name in link_names:
            link_path = f"{robot_path}/{link_name}"
            collision_paths = _collision_prim_paths_under(stage, link_path)
            if not collision_paths:
                print(f"[WARN] No CollisionAPI prims found under {link_path}; touch compliant material skipped.", flush=True)
                continue

            for collision_path in collision_paths:
                material_path = f"{collision_path}/compliant_material"
                try:
                    material_cfg.func(material_path, material_cfg)
                    sim_utils.modify_collision_properties(collision_path, collision_cfg, stage=stage)
                    sim_utils.bind_physics_material(collision_path, material_path, stage=stage)
                    success_count += 1
                    if env_id == 0:
                        print(
                            "[INFO] Revo3 touch compliant material applied: "
                            f"{collision_path}, stiffness={float(stiffness):g}, damping={float(damping):g}, "
                            f"contact_offset={float(contact_offset):g}, rest_offset={float(rest_offset):g}",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"[WARN] Could not bind touch compliant material on {collision_path}: {exc}", flush=True)

    if success_count == 0:
        print("[WARN] Revo3 touch compliant material binding did not affect any collision prims.", flush=True)


def apply_revo3_link_collision_properties(
    env: ManagerBasedRLEnv,
    env_ids,
    *,
    link_names: tuple[str, ...],
    contact_offset: float,
    rest_offset: float,
    label: str = "links",
) -> None:
    stage = sim_utils.get_current_stage()
    collision_cfg = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=float(contact_offset),
        rest_offset=float(rest_offset),
    )
    env_range = range(env.num_envs) if env_ids is None else [int(env_id) for env_id in env_ids]
    success_count = 0

    for env_id in env_range:
        robot_path = f"/World/envs/env_{env_id}/Robot"
        try:
            sim_utils.make_uninstanceable(robot_path, stage=stage)
        except Exception as exc:
            print(f"[WARN] Could not make {robot_path} uninstanceable before collision property update: {exc}", flush=True)

        for link_name in link_names:
            link_path = f"{robot_path}/{link_name}"
            collision_paths = _collision_prim_paths_under(stage, link_path)
            if not collision_paths:
                print(f"[WARN] No CollisionAPI prims found under {link_path}; {label} collision properties skipped.", flush=True)
                continue

            for collision_path in collision_paths:
                try:
                    sim_utils.modify_collision_properties(collision_path, collision_cfg, stage=stage)
                    success_count += 1
                    if env_id == 0:
                        print(
                            f"[INFO] Revo3 {label} collision properties applied: "
                            f"{collision_path}, contact_offset={float(contact_offset):g}, rest_offset={float(rest_offset):g}",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"[WARN] Could not modify {label} collision properties on {collision_path}: {exc}", flush=True)

    if success_count == 0:
        print(f"[WARN] Revo3 {label} collision property update did not affect any collision prims.", flush=True)


# Select exactly one tactile implementation for the RL policy input.
ENABLE_RL_OURS_OBS = True
# RL_OURS_OBSERVATION_TERMS = ("pressure", "hydroshear")
RL_OURS_OBSERVATION_TERMS = ("pressure", "tacmap_policy", "hydroshear", "taxim_rgb")
ENABLE_RL_TACSL_BASELINE_OBS = False
ENABLE_RL_TACMAP_BASELINE_OBS = False
ENABLE_RL_HYDROSHEAR_BASELINE_OBS = False
ENABLE_RL_FOTS_BASELINE_OBS = False
TIANJI_RL_PRESSURE_DIFFUSION_ENABLED = True
TIANJI_RL_PRESSURE_DIFFUSION_SIGMA_M = 0.003
TIANJI_RL_PRESSURE_DIFFUSION_BLEND = 0.7
TIANJI_RL_PRESSURE_DIFFUSION_RADIUS_SIGMA = 3.0
TIANJI_RL_PRESSURE_DIFFUSION_NORMAL_POWER = 1.0
TIANJI_TACMAP_RL_ROWS = 32
TIANJI_TACMAP_RL_COLS = 24
TIANJI_TACMAP_CAMERA_PLANE_MAX_DISTANCE_M = 0.05
TIANJI_LOCAL_TACMAP_ENABLED = True
TIANJI_LOCAL_TACMAP_REFERENCE_ROWS = 240
TIANJI_LOCAL_TACMAP_REFERENCE_COLS = 320
TIANJI_LOCAL_TACMAP_ROWS = 25
TIANJI_LOCAL_TACMAP_COLS = 40
TIANJI_LOCAL_TACMAP_CONTACT_THRESHOLD_M = 0.02e-3
TIANJI_LOCAL_TACMAP_ROI_MARGIN_M = 1.0e-3
TIANJI_LOCAL_TACMAP_PENETRATION_DEADBAND_M = 1.0e-6
TIANJI_TAXIM_RGB_RENDER_ROWS = 240
TIANJI_TAXIM_RGB_RENDER_COLS = 320
TIANJI_TAXIM_RGB_RENDER_CHUNK_SIZE = 128
TIANJI_TACTILE_RESNET_OUTPUT_DIM = 128
TIANJI_TACTILE_RESNET_INPUT_ROWS = 224
TIANJI_TACTILE_RESNET_INPUT_COLS = 224
TIANJI_TACTILE_RESNET_DEPTH_MAX_M = 0.002
TIANJI_TACTILE_RESNET_CHUNK_SIZE = 128
TIANJI_TAXIM_RGB_BACKGROUND = (
    REPO_ROOT / "vitai_4Fingers-320*240" / "marker_annotations" / "reference_median.png"
)
TIANJI_TACSL_BASELINE_RAY_ROWS = 320
TIANJI_TACSL_BASELINE_RAY_COLS = 240
TIANJI_TACSL_BASELINE_OUTPUT_ROWS = 16
TIANJI_TACSL_BASELINE_OUTPUT_COLS = 8
TIANJI_TACSL_NORMAL_CONTACT_STIFFNESS = 1.0
TIANJI_TACSL_TANGENTIAL_STIFFNESS = 0.1
TIANJI_TACSL_FRICTION_COEFFICIENT = 2.0
TIANJI_TACMAP_BASELINE_RAY_ROWS = 240
TIANJI_TACMAP_BASELINE_RAY_COLS = 240
TIANJI_TACMAP_BASELINE_SURFACE_REFERENCE_NPY = (
    TIANJI_TACMAP_DIR / "tacmap_surface_reference_240x240.npy"
)
TIANJI_HYDROSHEAR_RENDER_ROWS = 320
TIANJI_HYDROSHEAR_RENDER_COLS = 240
TIANJI_HYDROSHEAR_MARKER_MARGIN_X = 15.0
TIANJI_HYDROSHEAR_MARKER_MARGIN_Y = 26.0 * TIANJI_HYDROSHEAR_RENDER_ROWS / 240.0
TIANJI_VITAI_MARKER_LAYOUT = ASSET_ROOT / "tactile/markers/marker_positions.npz"
TIANJI_VITAI_RENDER_ROWS = 240
TIANJI_VITAI_RENDER_COLS = 320
TIANJI_VITAI_MARKER_ROWS = 10
TIANJI_VITAI_MARKER_COLS = 10
TIANJI_HYDROSHEAR_BASELINE_RAY_ROWS = 32
TIANJI_HYDROSHEAR_BASELINE_RAY_COLS = 24
TIANJI_HYDROSHEAR_BASELINE_RENDER_ROWS = TIANJI_HYDROSHEAR_RENDER_ROWS
TIANJI_HYDROSHEAR_BASELINE_RENDER_COLS = TIANJI_HYDROSHEAR_RENDER_COLS
TIANJI_HYDROSHEAR_BASELINE_MARKER_ROWS = 16
TIANJI_HYDROSHEAR_BASELINE_MARKER_COLS = 8
TIANJI_FOTS_BASELINE_RAY_ROWS = 32
TIANJI_FOTS_BASELINE_RAY_COLS = 24
TIANJI_FOTS_BASELINE_RENDER_ROWS = 320
TIANJI_FOTS_BASELINE_RENDER_COLS = 240
TIANJI_FOTS_BASELINE_MARKER_ROWS = 16
TIANJI_FOTS_BASELINE_MARKER_COLS = 8
TIANJI_FOTS_BASELINE_MARKER_MARGIN_X = 15.0
TIANJI_FOTS_BASELINE_MARKER_MARGIN_Y = 26.0 * TIANJI_FOTS_BASELINE_RENDER_ROWS / 240.0
TIANJI_FOTS_BASELINE_MM2PIX = 19.58
TIANJI_FOTS_BASELINE_LAMB = (0.00125, 0.00021, 0.00038)
TIANJI_FOTS_BASELINE_CONTACT_THRESHOLD_MM = 0.02


def _selected_rl_tactile_implementation() -> str:
    enabled = [
        name
        for name, is_enabled in (
            ("ours", ENABLE_RL_OURS_OBS),
            ("tacsl_baseline", ENABLE_RL_TACSL_BASELINE_OBS),
            ("tacmap_baseline", ENABLE_RL_TACMAP_BASELINE_OBS),
            ("hydroshear_baseline", ENABLE_RL_HYDROSHEAR_BASELINE_OBS),
            ("fots_baseline", ENABLE_RL_FOTS_BASELINE_OBS),
        )
        if is_enabled
    ]
    if len(enabled) != 1:
        raise ValueError(
            "Exactly one RL tactile implementation must be enabled; "
            f"selected={enabled or ['none']}."
        )
    return enabled[0]


def _pressure_pad_specs_by_link():
    specs = load_pressure_pad_specs_from_urdf(TIANJI_PRESSUREPAD_URDF, require_files=False)
    return {spec.link_name: spec for spec in specs}


def _quat_wxyz_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    quat = np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float64,
    )
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    return tuple(float(value) for value in quat)


def _tacmap_asset_paths(link_name: str) -> tuple[Path, Path]:
    stem = link_name.removesuffix("_link")
    points = TIANJI_TACMAP_DIR / f"{stem}_point.npy"
    normals = TIANJI_TACMAP_DIR / f"{stem}_normal.npy"
    if not points.is_file() or not normals.is_file():
        raise RuntimeError(f"Missing TacMap npy assets for {link_name!r}: {points}, {normals}")
    return points, normals


@lru_cache(maxsize=None)
def _mean_tacmap_normal_direction(normals_npy: str) -> tuple[float, float, float] | None:
    normals = np.load(normals_npy).astype(np.float32).reshape(-1, 3)
    valid = np.linalg.norm(normals, axis=-1) > 1.0e-8
    if not np.any(valid):
        return None

    normals = normals[valid]
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1.0e-12)
    mean_normal = np.mean(normals, axis=0)
    norm = float(np.linalg.norm(mean_normal))
    if norm < 1.0e-8:
        return None

    mean_normal = mean_normal / norm
    return (float(mean_normal[0]), float(mean_normal[1]), float(mean_normal[2]))


def _axis_vector_tuple(axis_name: str) -> tuple[float, float, float]:
    sign = -1.0 if str(axis_name).startswith("-") else 1.0
    axis = str(axis_name)[-1:].lower()
    values = {
        "x": (sign, 0.0, 0.0),
        "y": (0.0, sign, 0.0),
        "z": (0.0, 0.0, sign),
    }.get(axis)
    if values is None:
        raise ValueError(f"Unsupported tactile axis: {axis_name!r}")
    return values


def _make_tacmap_link_surface_cfg(
    link_name: str,
    *,
    sensor_kind: str,
    image_rows: int = TIANJI_TACMAP_RL_ROWS,
    image_cols: int = TIANJI_TACMAP_RL_COLS,
    ray_layout_prefix: str | None = None,
) -> SharpaTacmapLinkSurfaceCfg:
    points_npy, normals_npy = _tacmap_asset_paths(link_name)
    params = TIANJI_TACMAP_LINK_SURFACE_DEFAULTS[link_name]
    body_name = TACTILE_LINK_MAP[link_name]
    target_expr = "{ENV_REGEX_NS}/Object" if sensor_kind == "object" else f"{{ENV_REGEX_NS}}/Robot/{body_name}"
    ray_hit_index = 1 if sensor_kind == "object" else 2
    ray_direction = _mean_tacmap_normal_direction(str(normals_npy))
    return SharpaTacmapLinkSurfaceCfg(
        class_type=SharpaTacmapLinkSurface,
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        mesh_prim_paths=[
            SharpaTacmapLinkSurfaceCfg.RaycastTargetCfg(
                prim_expr=target_expr,
                track_mesh_transforms=True,
            )
        ],
        update_period=0.0,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
        offset=SharpaTacmapLinkSurfaceCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["distance_along_normal", "distance_along_normal_raw"],
        points_npy=str(points_npy),
        normals_npy=str(normals_npy),
        resolution_step=1,
        max_distance=(
            TIANJI_TACMAP_CAMERA_PLANE_MAX_DISTANCE_M
            if ray_layout_prefix is not None
            else float(params["max_distance"])
        ),
        cpd_max_dist=0.5,
        correction_scale=1.0e-3,
        pts_offsets=0.0,
        image_width=max(1, int(image_cols)),
        image_height=max(1, int(image_rows)),
        ray_axis=str(params["ray_axis"]),
        ray_direction=ray_direction,
        grid_u_axis=str(params["grid_u_axis"]),
        grid_v_axis=str(params["grid_v_axis"]),
        grid_u_size=float(params["grid_u_size"]),
        grid_v_size=float(params["grid_v_size"]),
        grid_center=tuple(float(v) for v in params["grid_center"]),
        ray_layout_npz=(
            str(TIANJI_VITAI_MARKER_LAYOUT)
            if ray_layout_prefix is not None
            else None
        ),
        ray_layout_prefix="" if ray_layout_prefix is None else ray_layout_prefix,
        ray_hit_index=ray_hit_index,
        use_ray_hit_index_layout=(
            sensor_kind == "surface" and ray_layout_prefix is not None
        ),
        use_first_hit_fallback=sensor_kind == "surface",
        debug_viz_env_id=0,
        debug_viz_max_points=15000,
        debug_viz_link_surfaces=False,
        debug_viz_hits=False,
        debug_viz_hits_external_mask=sensor_kind == "object",
        debug_viz_rays=False,
    )


# def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | list[int] | slice | None) -> list[int]:
#     if env_ids is None or env_ids == slice(None):
#         return list(range(env.num_envs))
#     if isinstance(env_ids, torch.Tensor):
#         return env_ids.cpu().tolist()
#     return list(env_ids)
#
#
# def _collect_collision_prims(stage: Usd.Stage, link_prim_path: str) -> list[Sdf.Path]:
#     """Collect collision prims under a link, falling back to the link prim."""
#     link_prim = stage.GetPrimAtPath(link_prim_path)
#     if not link_prim.IsValid():
#         return []
#     collision_prims: list[Sdf.Path] = []
#     for prim in Usd.PrimRange(link_prim):
#         if prim.HasAPI(UsdPhysics.CollisionAPI):
#             collision_prims.append(prim.GetPath())
#     return collision_prims
#
#
# def _get_filtered_pairs_rel(prim: Usd.Prim) -> Usd.Relationship:
#     """Return a relationship that authors filtered collision pairs."""
#     if hasattr(UsdPhysics, "FilteredPairsAPI"):
#         api = UsdPhysics.FilteredPairsAPI.Apply(prim)
#         if hasattr(api, "GetFilteredPairsRel"):
#             rel = api.GetFilteredPairsRel()
#             if not rel:
#                 rel = api.CreateFilteredPairsRel()
#             return rel
#     rel = prim.GetRelationship("filteredPairs")
#     if not rel:
#         rel = prim.CreateRelationship("filteredPairs")
#     return rel
#
#
# def _build_link_pairs(
#     group_links: dict[str, list[str]],
#     filtered_group_pairs: list[tuple[str, str]] | None,
#     self_filtered_groups: list[str] | None,
# ) -> list[tuple[str, str]]:
#     filtered_group_pairs = filtered_group_pairs or []
#     self_filtered_groups = self_filtered_groups or []
#     pairs: set[tuple[str, str]] = set()
#
#     for group_name in self_filtered_groups:
#         links = group_links.get(group_name, [])
#         for i in range(len(links)):
#             for j in range(i + 1, len(links)):
#                 a, b = links[i], links[j]
#                 pairs.add((a, b) if a < b else (b, a))
#
#     for group_a, group_b in filtered_group_pairs:
#         for a in group_links.get(group_a, []):
#             for b in group_links.get(group_b, []):
#                 if a == b:
#                     continue
#                 pairs.add((a, b) if a < b else (b, a))
#
#     return sorted(pairs)
#
#
# def disable_collision_pairs_filtered(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor | list[int] | slice | None,
#     group_links: dict[str, list[str]],
#     filtered_group_pairs: list[tuple[str, str]] | None = None,
#     self_filtered_groups: list[str] | None = None,
#     asset_root: str = "Robot",
# ) -> None:
#     """Disable collisions using USD FilteredPairsAPI based on link groups."""
#     stage = sim_utils.get_current_stage()
#     if stage is None:
#         return
#
#     link_pairs = _build_link_pairs(group_links, filtered_group_pairs, self_filtered_groups)
#     if not link_pairs:
#         return
#
#     for env_id in _resolve_env_ids(env, env_ids):
#         env_path = f"{env.scene.env_ns}/env_{env_id}"
#         robot_path = f"{env_path}/{asset_root}"
#         for link_a, link_b in link_pairs:
#             link_a_path = f"{robot_path}/{link_a}"
#             link_b_path = f"{robot_path}/{link_b}"
#             prims_a = _collect_collision_prims(stage, link_a_path) or [Sdf.Path(link_a_path)]
#             prims_b = _collect_collision_prims(stage, link_b_path) or [Sdf.Path(link_b_path)]
#
#             for prim_a_path in prims_a:
#                 prim_a = stage.GetPrimAtPath(prim_a_path)
#                 if not prim_a.IsValid():
#                     continue
#                 rel_a = _get_filtered_pairs_rel(prim_a)
#                 for prim_b_path in prims_b:
#                     prim_b = stage.GetPrimAtPath(prim_b_path)
#                     if not prim_b.IsValid():
#                         continue
#                     rel_a.AddTarget(prim_b_path)
#                     rel_b = _get_filtered_pairs_rel(prim_b)
#                     rel_b.AddTarget(prim_a_path)


@configclass
class Revo3RelJointPosActionCfg:
    """Relative joint position control for the public Revo3 arm-hand robot."""

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.1,
    )


def tianji_hand_contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Thumb plus at least one other fingertip contact for the Tianji hand."""
    thumb_contact_sensor: ContactSensor = env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["right_indexdip_roll_rubber_link_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["right_middip_roll_rubber_link_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["right_ringdip_roll_rubber_link_object_s"]
    little_contact_sensor: ContactSensor = env.scene.sensors["right_pinkydip_roll_rubber_link_object_s"]

    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    little_contact = little_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    little_contact_mag = torch.norm(little_contact, dim=-1)

    return (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold)
        | (middle_contact_mag > threshold)
        | (ring_contact_mag > threshold)
        | (little_contact_mag > threshold)
    )


def tianji_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Position tracking reward gated by Tianji hand contact."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import combine_frame_transforms

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(obj.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * tianji_hand_contacts(env, 1.0).float()


def tianji_orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Orientation tracking reward gated by Tianji hand contact."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import quat_error_magnitude, quat_mul

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    quat_error = quat_error_magnitude(obj.data.root_quat_w, des_quat_w)
    return (1 - torch.tanh(quat_error / std)) * tianji_hand_contacts(env, 1.0).float()


@configclass
class Revo3ReorientRewardCfg(dexsuite.RewardsCfg):
    """Reward configuration dedicated to Revo3 lift/reorient tasks."""

    # any_finger_contact = RewTerm(
    #     func=mdp.any_finger_contact,
    #     weight=0.7,
    #     params={"threshold": 1.0},
    # )

    good_finger_contact = RewTerm(
        func=tianji_hand_contacts,
        weight=1.0,
        params={"threshold": 1.0},
    )

    position_tracking = RewTerm(
        func=tianji_position_command_error_tanh,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    orientation_tracking = RewTerm(
        func=tianji_orientation_command_error_tanh,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class Revo3MixinCfg:
    """Mixin that attaches the public Revo3 robot asset."""

    rewards: Revo3ReorientRewardCfg = Revo3ReorientRewardCfg()
    actions: Revo3RelJointPosActionCfg = Revo3RelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()

        self.commands.object_pose.body_name = TIANJI_PALM_BODY_NAME
        self.commands.object_pose.debug_vis = True

        self.scene.robot = _make_revo3_robot_cfg_with_touch_compliance()

        # The task-owned asset config includes the donor's final initial pose.

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]", "right_.*_joint"]),
                "position_range": [0.0, 0.0],
                "velocity_range": [0.0, 0.0],
            },
        )

        self.events.reset_robot_wrist_joint = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="joint7"),
                "position_range": [0.0, 0.0],
                "velocity_range": [0.0, 0.0],
            },
        )

        for link_name in TIANJI_HAND_DIP_BODIES:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + TACTILE_LINK_MAP[link_name],
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in TIANJI_HAND_DIP_BODIES]},
            clip=(-20.0, 20.0),
        )

        tactile_implementation = _selected_rl_tactile_implementation()
        self.observations.proprio.rl_ours = None
        self.observations.proprio.rl_ours_pressure = None
        self.observations.proprio.rl_ours_tacmap_policy = None
        self.observations.proprio.rl_ours_hydroshear = None
        self.observations.proprio.rl_ours_taxim_rgb = None
        self.observations.proprio.rl_tacsl_baseline = None
        self.observations.proprio.rl_hydroshear_baseline = None
        self.observations.proprio.rl_fots_baseline = None
        self.observations.proprio.rl_warpsdf_pressure = None
        self.observations.proprio.rl_tacmap = None
        self.observations.proprio.rl_hydroshear = None
        if tactile_implementation not in (
            "ours",
            "tacsl_baseline",
            "tacmap_baseline",
            "hydroshear_baseline",
            "fots_baseline",
        ):
            raise NotImplementedError(
                f"RL tactile implementation {tactile_implementation!r} is selected but has not been connected yet."
            )
        self.events.zz_ours_tactile_cache_reset = (
            EventTerm(
                func=mdp.invalidate_ours_tactile_cache_on_reset,
                mode="reset",
            )
            if tactile_implementation == "ours"
            else None
        )

        selected_ours_terms: tuple[str, ...] = ()
        if tactile_implementation == "ours":
            selected_ours_terms = tuple(str(name) for name in RL_OURS_OBSERVATION_TERMS)
            valid_ours_terms = {"pressure", "tacmap_policy", "hydroshear", "taxim_rgb"}
            unknown_ours_terms = set(selected_ours_terms) - valid_ours_terms
            if unknown_ours_terms or not selected_ours_terms:
                raise ValueError(
                    "RL_OURS_OBSERVATION_TERMS must select pressure, tacmap_policy, hydroshear, and/or taxim_rgb; "
                    f"selected={selected_ours_terms}, unknown={sorted(unknown_ours_terms)}"
                )
        ours_pressure_enabled = "pressure" in selected_ours_terms
        ours_hydroshear_enabled = "hydroshear" in selected_ours_terms
        ours_taxim_rgb_enabled = "taxim_rgb" in selected_ours_terms
        ours_tacmap_needed = (
            "tacmap_policy" in selected_ours_terms
            or ours_hydroshear_enabled
            or ours_taxim_rgb_enabled
        )

        pressure_taxel_counts: list[int] = []
        if tactile_implementation == "ours" and ours_pressure_enabled:
            pressure_specs = _pressure_pad_specs_by_link()
            for link_name, sensor_name in zip(
                TIANJI_PRESSURE_PAD_LINK_ORDER, TIANJI_PRESSURE_SENSOR_NAMES, strict=True
            ):
                spec = pressure_specs.get(link_name)
                if spec is None:
                    raise RuntimeError(f"Missing pressure_pad metadata for {link_name!r} in {TIANJI_PRESSUREPAD_URDF}")
                taxel_map = spec.to_taxel_map()
                rows, cols = (int(value) for value in taxel_map.image_shape)
                points_l = np.asarray(taxel_map.points_l, dtype=np.float32)
                normals_l = np.asarray(taxel_map.normals_l, dtype=np.float32)
                points_l = points_l + normals_l * 0.001
                pressure_taxel_counts.append(rows * cols)

                setattr(
                    self.scene,
                    sensor_name,
                    WarpSdfTactileSensorCfg(
                        prim_path="{ENV_REGEX_NS}/Robot",
                        elastomer_prim_paths=[f"{ISAACLAB_ENV_REGEX_NS}/Robot/{TACTILE_LINK_MAP[link_name]}"],
                        num_rows=rows,
                        num_cols=cols,
                        normal_axis=int(spec.normal_axis),
                        normal_offset=0.0,
                        normal_sign=float(spec.normal_sign),
                        taxel_points_l=[tuple(float(value) for value in point) for point in points_l],
                        taxel_normals_l=[tuple(float(value) for value in normal) for normal in normals_l],
                        box_half_extents=(0.03, 0.03, 0.03),
                        target_mesh_prim_path=f"{ISAACLAB_ENV_REGEX_NS}/Object",
                        mesh_max_dist=0.20,
                        mesh_use_signed_distance=True,
                        mesh_signed_distance_method="winding",
                        mesh_smooth_normals=True,
                        stiffness=spec.calibration.stiffness,
                        damping=spec.calibration.damping,
                        max_force=spec.calibration.max_force,
                        pressure_gain=spec.calibration.gain,
                        pressure_bias=spec.calibration.bias,
                        pressure_gamma=spec.calibration.gamma,
                        pressure_threshold=spec.calibration.threshold,
                        taxel_area=spec.calibration.area,
                        normalize_forces=True,
                        store_debug_fields=False,
                        debug_vis=False,
                    ),
                )

        if tactile_implementation == "tacsl_baseline":
            tacmap_rows = TIANJI_TACSL_BASELINE_RAY_ROWS
            tacmap_cols = TIANJI_TACSL_BASELINE_RAY_COLS
        elif tactile_implementation == "tacmap_baseline":
            tacmap_rows = TIANJI_TACMAP_BASELINE_RAY_ROWS
            tacmap_cols = TIANJI_TACMAP_BASELINE_RAY_COLS
        else:
            tacmap_rows = TIANJI_TACMAP_RL_ROWS
            tacmap_cols = TIANJI_TACMAP_RL_COLS
        for finger, link_name, surface_name, object_name in zip(
            TIANJI_RL_FINGER_ORDER,
            TIANJI_TACMAP_LINK_ORDER,
            TIANJI_TACMAP_SURFACE_SENSOR_NAMES,
            TIANJI_TACMAP_OBJECT_SENSOR_NAMES,
            strict=True,
        ):
            if tactile_implementation == "ours" and not ours_tacmap_needed:
                continue
            if tactile_implementation != "tacmap_baseline":
                setattr(
                    self.scene,
                    surface_name,
                    _make_tacmap_link_surface_cfg(
                        link_name,
                        sensor_kind="surface",
                        image_rows=tacmap_rows,
                        image_cols=tacmap_cols,
                        ray_layout_prefix=(
                            finger if tactile_implementation == "ours" else None
                        ),
                    ),
                )
            setattr(
                self.scene,
                object_name,
                _make_tacmap_link_surface_cfg(
                    link_name,
                    sensor_kind="object",
                    image_rows=tacmap_rows,
                    image_cols=tacmap_cols,
                    ray_layout_prefix=(
                        finger if tactile_implementation == "ours" else None
                    ),
                ),
            )
            if tactile_implementation == "ours" and ours_hydroshear_enabled:
                marker_prefix = f"{finger}_marker"
                setattr(
                    self.scene,
                    f"{link_name}_hydroshear_marker_surface_s",
                    _make_tacmap_link_surface_cfg(
                        link_name,
                        sensor_kind="surface",
                        image_rows=TIANJI_VITAI_MARKER_ROWS,
                        image_cols=TIANJI_VITAI_MARKER_COLS,
                        ray_layout_prefix=marker_prefix,
                    ),
                )
                setattr(
                    self.scene,
                    f"{link_name}_hydroshear_marker_object_s",
                    _make_tacmap_link_surface_cfg(
                        link_name,
                        sensor_kind="object",
                        image_rows=TIANJI_VITAI_MARKER_ROWS,
                        image_cols=TIANJI_VITAI_MARKER_COLS,
                        ray_layout_prefix=marker_prefix,
                    ),
                )

        if tactile_implementation == "ours":
            ours_component_cfg = {
                "pressure_sensor_names": list(TIANJI_PRESSURE_SENSOR_NAMES),
                "pressure_taxel_counts": pressure_taxel_counts,
                "pressure_diffusion_enabled": TIANJI_RL_PRESSURE_DIFFUSION_ENABLED,
                "pressure_diffusion_sigma_m": TIANJI_RL_PRESSURE_DIFFUSION_SIGMA_M,
                "pressure_diffusion_blend": TIANJI_RL_PRESSURE_DIFFUSION_BLEND,
                "pressure_diffusion_radius_sigma": TIANJI_RL_PRESSURE_DIFFUSION_RADIUS_SIGMA,
                "pressure_diffusion_normal_power": TIANJI_RL_PRESSURE_DIFFUSION_NORMAL_POWER,
                "tacmap_sensor_names": list(TIANJI_TACMAP_OBJECT_SENSOR_NAMES),
                "tacmap_surface_sensor_names": list(TIANJI_TACMAP_SURFACE_SENSOR_NAMES),
                "hydroshear_marker_sensor_names": list(TIANJI_HYDROSHEAR_MARKER_OBJECT_SENSOR_NAMES),
                "hydroshear_marker_surface_sensor_names": list(
                    TIANJI_HYDROSHEAR_MARKER_SURFACE_SENSOR_NAMES
                ),
                "tacmap_rows": TIANJI_TACMAP_RL_ROWS,
                "tacmap_cols": TIANJI_TACMAP_RL_COLS,
                "local_tacmap_enabled": TIANJI_LOCAL_TACMAP_ENABLED,
                "local_tacmap_reference_rows": TIANJI_LOCAL_TACMAP_REFERENCE_ROWS,
                "local_tacmap_reference_cols": TIANJI_LOCAL_TACMAP_REFERENCE_COLS,
                "local_tacmap_rows": TIANJI_LOCAL_TACMAP_ROWS,
                "local_tacmap_cols": TIANJI_LOCAL_TACMAP_COLS,
                "local_tacmap_contact_threshold_m": TIANJI_LOCAL_TACMAP_CONTACT_THRESHOLD_M,
                "local_tacmap_roi_margin_m": TIANJI_LOCAL_TACMAP_ROI_MARGIN_M,
                "local_tacmap_penetration_deadband_m": TIANJI_LOCAL_TACMAP_PENETRATION_DEADBAND_M,
                "local_tacmap_max_distance_m": TIANJI_TACMAP_CAMERA_PLANE_MAX_DISTANCE_M,
                "tacmap_policy_full_depth_image": True,
                "taxim_rgb_render_rows": TIANJI_TAXIM_RGB_RENDER_ROWS,
                "taxim_rgb_render_cols": TIANJI_TAXIM_RGB_RENDER_COLS,
                "taxim_rgb_render_chunk_size": TIANJI_TAXIM_RGB_RENDER_CHUNK_SIZE,
                "taxim_rgb_background_path": str(TIANJI_TAXIM_RGB_BACKGROUND),
                "tactile_resnet_output_dim": TIANJI_TACTILE_RESNET_OUTPUT_DIM,
                "tactile_resnet_input_rows": TIANJI_TACTILE_RESNET_INPUT_ROWS,
                "tactile_resnet_input_cols": TIANJI_TACTILE_RESNET_INPUT_COLS,
                "tactile_resnet_depth_max_m": TIANJI_TACTILE_RESNET_DEPTH_MAX_M,
                "tactile_resnet_chunk_size": TIANJI_TACTILE_RESNET_CHUNK_SIZE,
                "hydroshear_render_rows": TIANJI_VITAI_RENDER_ROWS,
                "hydroshear_render_cols": TIANJI_VITAI_RENDER_COLS,
                "hydroshear_marker_rows": TIANJI_VITAI_MARKER_ROWS,
                "hydroshear_marker_cols": TIANJI_VITAI_MARKER_COLS,
                "hydroshear_marker_margin_x": TIANJI_HYDROSHEAR_MARKER_MARGIN_X,
                "hydroshear_marker_margin_y": TIANJI_HYDROSHEAR_MARKER_MARGIN_Y,
                "hydroshear_marker_layout_path": str(TIANJI_VITAI_MARKER_LAYOUT),
                "hydroshear_marker_finger_link_names": [TACTILE_LINK_MAP[name] for name in TIANJI_TACMAP_LINK_ORDER],
                "hydroshear_object_sample_mode": "poisson",
                "hydroshear_object_sample_count": RL_HYDROSHEAR_OBJECT_SAMPLE_COUNT,
                "hydroshear_object_sample_seed": RL_HYDROSHEAR_OBJECT_SAMPLE_SEED,
                "hydroshear_object_poisson_radius": RL_HYDROSHEAR_POISSON_RADIUS,
                "hydroshear_object_poisson_initial_count": RL_HYDROSHEAR_POISSON_INITIAL_COUNT,
                "hydroshear_object_sample_reference_count": RL_HYDROSHEAR_OBJECT_SAMPLE_REFERENCE_COUNT,
                "hydroshear_object_sample_roi_count": RL_HYDROSHEAR_OBJECT_SAMPLE_ROI_COUNT,
                "hydroshear_use_object_surface_samples": True,
            }
            if "pressure" in selected_ours_terms:
                self.observations.proprio.rl_ours_pressure = ObsTerm(
                    func=mdp.ours_rl_pressure_obs,
                    params={"component_cfg": dict(ours_component_cfg)},
                )
            if "tacmap_policy" in selected_ours_terms:
                self.observations.proprio.rl_ours_tacmap_policy = ObsTerm(
                    func=mdp.ours_rl_tacmap_resnet_obs,
                    params={"component_cfg": dict(ours_component_cfg)},
                )
            if "hydroshear" in selected_ours_terms:
                self.observations.proprio.rl_ours_hydroshear = ObsTerm(
                    func=mdp.ours_rl_hydroshear_obs,
                    params={"component_cfg": dict(ours_component_cfg)},
                )
            if "taxim_rgb" in selected_ours_terms:
                self.observations.proprio.rl_ours_taxim_rgb = ObsTerm(
                    func=mdp.ours_rl_taxim_resnet_obs,
                    params={"component_cfg": dict(ours_component_cfg)},
                )
        elif tactile_implementation == "tacsl_baseline":
            ray_direction_axes = []
            row_axis_names = []
            col_axis_names = []
            for link_name in TIANJI_TACMAP_LINK_ORDER:
                _, normals_npy = _tacmap_asset_paths(link_name)
                params = TIANJI_TACMAP_LINK_SURFACE_DEFAULTS[link_name]
                ray_direction_axes.append(
                    _mean_tacmap_normal_direction(str(normals_npy))
                    or _axis_vector_tuple(str(params["ray_axis"]))
                )
                row_axis_names.append(str(params["grid_v_axis"]))
                col_axis_names.append(str(params["grid_u_axis"]))
            self.observations.proprio.rl_tacsl_baseline = ObsTerm(
                func=mdp.tacsl_baseline_rl_obs,
                params={
                    "tacmap_sensor_names": list(TIANJI_TACMAP_OBJECT_SENSOR_NAMES),
                    "tacmap_surface_sensor_names": list(TIANJI_TACMAP_SURFACE_SENSOR_NAMES),
                    "finger_link_names": [TACTILE_LINK_MAP[name] for name in TIANJI_TACMAP_LINK_ORDER],
                    "ray_direction_axes": ray_direction_axes,
                    "row_axis_names": row_axis_names,
                    "col_axis_names": col_axis_names,
                    "ray_rows": TIANJI_TACSL_BASELINE_RAY_ROWS,
                    "ray_cols": TIANJI_TACSL_BASELINE_RAY_COLS,
                    "output_rows": TIANJI_TACSL_BASELINE_OUTPUT_ROWS,
                    "output_cols": TIANJI_TACSL_BASELINE_OUTPUT_COLS,
                    "normal_contact_stiffness": TIANJI_TACSL_NORMAL_CONTACT_STIFFNESS,
                    "tangential_stiffness": TIANJI_TACSL_TANGENTIAL_STIFFNESS,
                    "friction_coefficient": TIANJI_TACSL_FRICTION_COEFFICIENT,
                },
            )
        elif tactile_implementation == "tacmap_baseline":
            self.observations.proprio.rl_tacmap = ObsTerm(
                func=mdp.tacmap_rl_obs,
                params={
                    "tacmap_sensor_names": list(TIANJI_TACMAP_OBJECT_SENSOR_NAMES),
                    "tacmap_surface_sensor_names": [],
                    "tacmap_surface_reference_npy": str(TIANJI_TACMAP_BASELINE_SURFACE_REFERENCE_NPY),
                    "tacmap_rows": TIANJI_TACMAP_BASELINE_RAY_ROWS,
                    "tacmap_cols": TIANJI_TACMAP_BASELINE_RAY_COLS,
                    "cache_aux_fields": False,
                    "contact_shell_m": 0.0,
                },
            )
        elif tactile_implementation == "hydroshear_baseline":
            self.observations.proprio.rl_hydroshear_baseline = ObsTerm(
                func=mdp.hydroshear_baseline_rl_obs,
                params={
                    "tacmap_sensor_names": list(TIANJI_TACMAP_OBJECT_SENSOR_NAMES),
                    "tacmap_surface_sensor_names": list(TIANJI_TACMAP_SURFACE_SENSOR_NAMES),
                    "tacmap_rows": TIANJI_HYDROSHEAR_BASELINE_RAY_ROWS,
                    "tacmap_cols": TIANJI_HYDROSHEAR_BASELINE_RAY_COLS,
                    "render_rows": TIANJI_HYDROSHEAR_BASELINE_RENDER_ROWS,
                    "render_cols": TIANJI_HYDROSHEAR_BASELINE_RENDER_COLS,
                    "marker_rows": TIANJI_HYDROSHEAR_BASELINE_MARKER_ROWS,
                    "marker_cols": TIANJI_HYDROSHEAR_BASELINE_MARKER_COLS,
                    "marker_margin_x": TIANJI_HYDROSHEAR_MARKER_MARGIN_X,
                    "marker_margin_y": TIANJI_HYDROSHEAR_MARKER_MARGIN_Y,
                    "object_sample_mode": "poisson",
                    "object_sample_count": RL_HYDROSHEAR_OBJECT_SAMPLE_COUNT,
                    "object_sample_seed": RL_HYDROSHEAR_OBJECT_SAMPLE_SEED,
                    "object_poisson_radius": RL_HYDROSHEAR_POISSON_RADIUS,
                    "object_poisson_initial_count": RL_HYDROSHEAR_POISSON_INITIAL_COUNT,
                    "object_sample_reference_count": RL_HYDROSHEAR_OBJECT_SAMPLE_REFERENCE_COUNT,
                },
            )
        elif tactile_implementation == "fots_baseline":
            ray_direction_axes = []
            for link_name in TIANJI_TACMAP_LINK_ORDER:
                _, normals_npy = _tacmap_asset_paths(link_name)
                params = TIANJI_TACMAP_LINK_SURFACE_DEFAULTS[link_name]
                ray_direction_axes.append(
                    _mean_tacmap_normal_direction(str(normals_npy))
                    or _axis_vector_tuple(str(params["ray_axis"]))
                )
            self.observations.proprio.rl_fots_baseline = ObsTerm(
                func=mdp.fots_baseline_rl_obs,
                params={
                    "tacmap_sensor_names": list(TIANJI_TACMAP_OBJECT_SENSOR_NAMES),
                    "tacmap_surface_sensor_names": list(TIANJI_TACMAP_SURFACE_SENSOR_NAMES),
                    "finger_link_names": [TACTILE_LINK_MAP[name] for name in TIANJI_TACMAP_LINK_ORDER],
                    "ray_direction_axes": ray_direction_axes,
                    "ray_rows": TIANJI_FOTS_BASELINE_RAY_ROWS,
                    "ray_cols": TIANJI_FOTS_BASELINE_RAY_COLS,
                    "render_rows": TIANJI_FOTS_BASELINE_RENDER_ROWS,
                    "render_cols": TIANJI_FOTS_BASELINE_RENDER_COLS,
                    "marker_rows": TIANJI_FOTS_BASELINE_MARKER_ROWS,
                    "marker_cols": TIANJI_FOTS_BASELINE_MARKER_COLS,
                    "marker_margin_x": TIANJI_FOTS_BASELINE_MARKER_MARGIN_X,
                    "marker_margin_y": TIANJI_FOTS_BASELINE_MARKER_MARGIN_Y,
                    "mm2pix": TIANJI_FOTS_BASELINE_MM2PIX,
                    "lamb_dilate": TIANJI_FOTS_BASELINE_LAMB[0],
                    "lamb_shear": TIANJI_FOTS_BASELINE_LAMB[1],
                    "lamb_twist": TIANJI_FOTS_BASELINE_LAMB[2],
                    "contact_threshold_mm": TIANJI_FOTS_BASELINE_CONTACT_THRESHOLD_MM,
                    "track_contact_center": True,
                },
            )

        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = TIANJI_HAND_TIP_BODIES

        if hasattr(self.rewards, "fingers_to_object"):
            self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                body_names=TIANJI_HAND_TIP_BODIES,
            )
        if hasattr(self.rewards, "fingers_to_object_delta"):
            self.rewards.fingers_to_object_delta.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                body_names=TIANJI_HAND_TIP_BODIES,
            )

        self.events.zz_revo3_touch_compliant_materials = EventTerm(
            func=apply_revo3_touch_compliant_materials,
            mode="startup",
            params={
                "link_names": tuple(TACTILE_LINK_MAP[name] for name in TIANJI_TACMAP_LINK_ORDER),
                "stiffness": TIANJI_TOUCH_COMPLIANT_STIFFNESS,
                "damping": TIANJI_TOUCH_COMPLIANT_DAMPING,
                "contact_offset": TIANJI_TOUCH_CONTACT_OFFSET,
                "rest_offset": TIANJI_TOUCH_REST_OFFSET,
            },
        )
        self.events.zz_revo3_pressure_pad_collision_properties = EventTerm(
            func=apply_revo3_link_collision_properties,
            mode="startup",
            params={
                "link_names": tuple(TIANJI_PRESSURE_PAD_COLLISION_LINK_ORDER),
                "contact_offset": TIANJI_PRESSURE_PAD_CONTACT_OFFSET,
                "rest_offset": TIANJI_PRESSURE_PAD_REST_OFFSET,
                "label": "pressure pad",
            },
        )

        # self.events.disable_collision_pairs_filtered = EventTerm(
        #     func=disable_collision_pairs_filtered,
        #     mode="prestartup",
        #     params={
        #         "asset_root": "Robot",
        #         "group_links": {
        #             "palm": ["base_link"],
        #             "mcp": [
        #                 "right_thumbmcp_roll_link",
        #                 "right_indexmcp_roll_link",
        #                 "right_midmcp_roll_link",
        #                 "right_ringmcp_roll_link",
        #                 "right_pinkymcp_roll_link",
        #             ],
        #         },
        #         "filtered_group_pairs": [("palm", "mcp")],
        #         "self_filtered_groups": [],
        #     },
        # )


@configclass
class DexsuiteRevo3LiftEnvCfg(Revo3MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Configuration for Revo3 lift environment (training)."""

    pass


@configclass
class DexsuiteRevo3LiftEnvCfg_PLAY(Revo3MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Configuration for Revo3 lift environment (evaluation/play)."""

    pass
