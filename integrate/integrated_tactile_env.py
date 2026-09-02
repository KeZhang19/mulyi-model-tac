"""Integrated WarpSDF + TacMap tactile environment.

This module keeps the already-working official_replay scene as the canonical
scene template, then attaches SharpaTacmap sensors to the same Revo touch link
and target mesh. The two tactile outputs are updated in the same Isaac
simulation step.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPLAY_ROOT = REPO_ROOT / "scripts" / "force_map" / "official_replay"
TACMAP_ROOT = REPO_ROOT / "tacmap"
REVO21_TACMAP_DIR = TACMAP_ROOT / "assets" / "tactilesensor_map" / "revo21_dv2"
REVO21_URDF = (
    REPO_ROOT
    / "assets"
    / "revo21_right_touch"
    / "urdf"
    / "revo21_dv2_urdf_right-touch.SLDASM.urdf"
)
REVO21_USD = REVO21_URDF.with_suffix(".usd")

for path in (OFFICIAL_REPLAY_ROOT, TACMAP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aloha_tactile_cfg import AlohaTactileEnvCfg  # noqa: E402
from aloha_tactile_env import AlohaTactileEnv, pressure_sensor_labels_from_env  # noqa: E402


TACMAP_FILES = {
    "right_indexdip_roll_rubber_link": (
        REVO21_TACMAP_DIR / "right_indexdip_roll_rubber_point.npy",
        REVO21_TACMAP_DIR / "right_indexdip_roll_rubber_normal.npy",
    ),
    "right_middip_roll_rubber_link": (
        REVO21_TACMAP_DIR / "right_middip_roll_rubber_point.npy",
        REVO21_TACMAP_DIR / "right_middip_roll_rubber_normal.npy",
    ),
    "right_pinkydip_roll_rubber_link": (
        REVO21_TACMAP_DIR / "right_pinkydip_roll_rubber_point.npy",
        REVO21_TACMAP_DIR / "right_pinkydip_roll_rubber_normal.npy",
    ),
    "right_ringdip_roll_rubber_link": (
        REVO21_TACMAP_DIR / "right_ringdip_roll_rubber_point.npy",
        REVO21_TACMAP_DIR / "right_ringdip_roll_rubber_normal.npy",
    ),
    "right_thumbdip_roll_rubber_link": (
        REVO21_TACMAP_DIR / "right_thumbdip_roll_rubber_point.npy",
        REVO21_TACMAP_DIR / "right_thumbdip_roll_rubber_normal.npy",
    ),
    "right_middle_touch_link": (
        TACMAP_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_point.npy",
        TACMAP_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_normal.npy",
    ),
}

RL_FINGER_ORDER = ("middle", "index", "ring", "pinky", "thumb")
RL_PRESSURE_PAD_FINGER_STEMS = {
    "middle": "mid",
    "index": "index",
    "ring": "ring",
    "pinky": "pinky",
    "thumb": "thumb",
}
RL_PRESSURE_PAD_SEGMENTS = ("mcp", "pip")
RL_PRESSURE_PAD_LINK_ORDER = tuple(
    f"right_{RL_PRESSURE_PAD_FINGER_STEMS[finger]}{segment}_roll_touch_link"
    for finger in RL_FINGER_ORDER
    for segment in RL_PRESSURE_PAD_SEGMENTS
)
RL_TACMAP_LABEL_HINTS = {
    "middle": ("middip", "middle"),
    "index": ("indexdip", "index"),
    "ring": ("ringdip", "ring"),
    "pinky": ("pinkydip", "pinky"),
    "thumb": ("thumbdip", "thumb"),
}
RL_HYDROSHEAR_MARKER_ROWS = 9
RL_HYDROSHEAR_MARKER_COLS = 11

TACMAP_LINK_SURFACE_DEFAULTS = {
    "right_indexdip_roll_rubber_link": {
        "max_distance": 0.015,
        "ray_axis": "+x",
        "grid_u_axis": "+y",
        "grid_v_axis": "+z",
        "grid_u_size": 0.02068358,
        "grid_v_size": 0.01846460,
        "grid_center": (0.00105538, 0.00114480, 0.01765266),
    },
    "right_middip_roll_rubber_link": {
        "max_distance": 0.015,
        "ray_axis": "+x",
        "grid_u_axis": "+y",
        "grid_v_axis": "+z",
        "grid_u_size": 0.0202,
        "grid_v_size": 0.0182,
        "grid_center": (0.00159, 0.00103, 0.04757),
    },
    "right_pinkydip_roll_rubber_link": {
        "max_distance": 0.015,
        "ray_axis": "+x",
        "grid_u_axis": "+y",
        "grid_v_axis": "+z",
        "grid_u_size": 0.02068569,
        "grid_v_size": 0.01843646,
        "grid_center": (0.00100531, -0.00114829, 0.01764967),
    },
    "right_ringdip_roll_rubber_link": {
        "max_distance": 0.015,
        "ray_axis": "+x",
        "grid_u_axis": "+y",
        "grid_v_axis": "+z",
        "grid_u_size": 0.02056884,
        "grid_v_size": 0.01818433,
        "grid_center": (0.00104103, -0.00038099, 0.01768645),
    },
    "right_thumbdip_roll_rubber_link": {
        "max_distance": 0.015,
        "ray_axis": "+y",
        "grid_u_axis": "+x",
        "grid_v_axis": "+z",
        "grid_u_size": 0.01343826,
        "grid_v_size": 0.02319049,
        "grid_center": (-0.00072565, 0.02421566, -0.00022462),
    },
}

def _is_revo21_pressure_pad_touch_link(link_path: str) -> bool:
    name = Path(str(link_path)).name.lower()
    return name.endswith("_roll_touch_link") and (
        "mcp_roll_touch_link" in name or "pip_roll_touch_link" in name
    )


_LEGACY_LINK_SURFACE_DEFAULTS = {
    "max_distance": 0.008,
    "grid_u_size": 0.014,
    "grid_v_size": 0.020,
    "grid_center": (-0.008, 0.0, 0.0012),
}


@dataclass
class IntegratedTactileEnvCfg(AlohaTactileEnvCfg):
    """Aloha tactile config plus TacMap/VBTS options."""

    urdf_path: str = str(REVO21_URDF)
    robot_usd_path: str = ""
    robot_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.5)
    robot_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    usd_output_dir: str | None = str(REPO_ROOT / "integrate" / "output" / "revo21_urdf")
    force_urdf_conversion: bool = True
    merge_fixed_joints: bool = False
    tactile_link_keywords: tuple[str, ...] = ("rubber_link",)
    touch_collision_paths: tuple[str, ...] = ("right_middip_roll_rubber_link/collisions",)
    press_touch_link: str = "right_middip_roll_rubber_link"
    press_points_npy: str = str(REVO21_TACMAP_DIR / "right_middip_roll_rubber_point.npy")
    press_normals_npy: str = str(REVO21_TACMAP_DIR / "right_middip_roll_rubber_normal.npy")
    press_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dataset_joint_order: tuple[str, ...] = (
        "right_thumbcmp_roll_joint",
        "right_thumbcmr_roll_joint",
        "right_thumbmcp_roll_joint",
        "right_thumbpip_roll_joint",
        "right_thumbdip_roll_joint",
        "right_indexmcp_yaw_joint",
        "right_indexmcp_roll_joint",
        "right_indexpip_roll_joint",
        "right_indexdip_roll_joint",
        "right_midmcp_yaw_joint",
        "right_midmcp_roll_joint",
        "right_midpip_roll_joint",
        "right_middip_roll_joint",
        "right_ringmcp_yaw_joint",
        "right_ringmcp_roll_joint",
        "right_ringpip_roll_joint",
        "right_ringdip_roll_joint",
        "right_pinkymcp_yaw_joint",
        "right_pinkymcp_roll_joint",
        "right_pinkypip_roll_joint",
        "right_pinkydip_roll_joint",
    )
    enable_tacmap: bool = True
    tacmap_resolution_step: int = 1
    tacmap_max_distance: float = 0.015
    tacmap_cpd_max_dist: float = 0.5
    tacmap_correction_scale: float = 1.0e-3
    tacmap_points_offset: float = 0.0
    tacmap_invert_normals_file: bool = False
    tacmap_debug_vis: bool = False
    tacmap_debug_hit_stats: bool = False
    tacmap_debug_hit_stats_every: int = 50
    tacmap_debug_viz_max_points: int = 15000
    tacmap_debug_viz_normals: bool = False
    tacmap_debug_viz_active_points: bool = False
    tacmap_debug_viz_active_threshold: float = 0.0
    tacmap_ray_mode: str = "surface_normal"
    tacmap_link_surface_width: int = 240
    tacmap_link_surface_height: int = 240
    tacmap_link_surface_ray_axis: str = "+x"
    tacmap_link_surface_ray_direction: tuple[float, float, float] | None = None
    tacmap_link_surface_use_mean_normal: bool = True
    tacmap_link_surface_grid_u_axis: str = "+y"
    tacmap_link_surface_grid_v_axis: str = "+z"
    tacmap_link_surface_grid_u_size: float = 0.0202
    tacmap_link_surface_grid_v_size: float = 0.0182
    tacmap_link_surface_grid_center: tuple[float, float, float] = (0.00159, 0.00103, 0.04757)
    tacmap_link_surface_debug_surfaces: bool = False
    tacmap_link_surface_debug_rays: bool = False
    tacmap_link_surface_debug_ray_length: float = 0.008
    tacmap_link_surface_debug_ray_width: float = 0.00006
    enable_tacsl_force_field: bool = False
    tacsl_sdf_gradient_eps: float = 1.0e-4
    tacsl_normal_contact_stiffness: float = 1.0
    tacsl_tangential_stiffness: float = 0.1
    tacsl_friction_coefficient: float = 2.0


class IntegratedTactileEnv(AlohaTactileEnv):
    """Aloha scene with both WarpSDF force maps and SharpaTacmap images."""

    def __init__(self, cfg: IntegratedTactileEnvCfg, simulation_app=None):
        self._tacmap_sensors: list = []
        self._tacmap_surface_sensors: list = []
        self._tacmap_surface_raw_cache: list[np.ndarray | None] = []
        self._tacmap_surface_local_cache: list[dict[str, torch.Tensor] | None] = []
        self._tacmap_penetration_cache: list[tuple[np.ndarray, np.ndarray] | None] = []
        self._tacmap_slot_order: list[int] = []
        self._tacmap_rows = 0
        self._tacmap_cols = 0
        self._tacsl_prev_tactile_points_w: list[torch.Tensor | None] = []
        self._tacsl_prev_closest_points_w: list[torch.Tensor | None] = []
        self._tacsl_prev_valid_masks: list[torch.Tensor | None] = []
        self._tacsl_prev_contact_pos_l: list[torch.Tensor | None] = []
        self._tacsl_prev_contact_quat_l: list[torch.Tensor | None] = []
        self._tacsl_physics_sim_view = None
        self._tacsl_sdf_views: list[object | None] = []
        self._tacsl_body_views: list[object | None] = []
        self._tacsl_elastomer_body_views: list[object | None] = []
        self._tacsl_contact_com_b: list[torch.Tensor | None] = []
        self._tacsl_elastomer_com_b: list[torch.Tensor | None] = []
        self._tacsl_sdf_view_query_counts: list[int | None] = []
        self._tacsl_warned_sdf_views: set[int] = set()
        super().__init__(cfg, simulation_app=simulation_app)

    def reset(self, *args, **kwargs):
        self._tacmap_surface_raw_cache = [None] * len(self._tacmap_surface_sensors)
        self._tacmap_surface_local_cache = [None] * len(self._tacmap_surface_sensors)
        self._tacmap_penetration_cache = [None] * len(self._tacmap_sensors)
        tacsl_count = max(len(self._tactile_sensors), len(self._tacmap_surface_sensors))
        self._tacsl_prev_tactile_points_w = [None] * tacsl_count
        self._tacsl_prev_closest_points_w = [None] * tacsl_count
        self._tacsl_prev_valid_masks = [None] * tacsl_count
        self._tacsl_prev_contact_pos_l = [None] * tacsl_count
        self._tacsl_prev_contact_quat_l = [None] * tacsl_count
        return super().reset(*args, **kwargs)

    def _post_spawn_init(self, cfg: IntegratedTactileEnvCfg, sim_utils, target_query_paths: list[str]) -> None:
        self._prepare_tacsl_sdf_collision_meshes(cfg, sim_utils, target_query_paths)
        return super()._post_spawn_init(cfg, sim_utils, target_query_paths)

    def _prepare_tacsl_sdf_collision_meshes(
        self,
        cfg: IntegratedTactileEnvCfg,
        sim_utils,
        target_query_paths: list[str],
    ) -> None:
        if not bool(getattr(cfg, "enable_tacsl_force_field", False)):
            return

        try:
            from pxr import PhysxSchema, UsdPhysics
        except Exception as exc:
            print(f"[WARN] TacSL SDF collision setup skipped: {exc}", flush=True)
            return

        stage = sim_utils.get_current_stage()
        for query_path in sorted({str(path) for path in target_query_paths if path}):
            prim = stage.GetPrimAtPath(query_path)
            if not prim or not prim.IsValid():
                print(f"[WARN] TacSL SDF collision setup skipped invalid mesh: {query_path}", flush=True)
                continue
            try:
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision_api.GetApproximationAttr().Set("sdf")
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                print(f"[INFO] TacSL SDF collision mesh enabled: {query_path}", flush=True)
            except Exception as exc:
                print(f"[WARN] Could not enable TacSL SDF collision mesh on {query_path}: {exc}", flush=True)

    def _create_tactile_sensors(
        self,
        cfg: IntegratedTactileEnvCfg,
        selected_links: list[str],
        target_root_paths: list[str],
        target_query_paths: list[str],
        WarpSdfTactileSensor,
        WarpSdfTactileSensorCfg,
        math_utils,
    ):
        sensors, slot_order = super()._create_tactile_sensors(
            cfg,
            selected_links,
            target_root_paths,
            target_query_paths,
            WarpSdfTactileSensor,
            WarpSdfTactileSensorCfg,
            math_utils,
        )

        self._tacmap_sensors = []
        self._tacmap_surface_sensors = []
        self._tacmap_surface_raw_cache = []
        self._tacmap_surface_local_cache = []
        self._tacmap_penetration_cache = []
        self._tacmap_slot_order = []
        if str(getattr(cfg, "tacmap_ray_mode", "surface_normal")) == "link_surface":
            self._tacmap_rows = max(1, int(cfg.tacmap_link_surface_height))
            self._tacmap_cols = max(1, int(cfg.tacmap_link_surface_width))
        else:
            self._tacmap_rows = 240 // max(1, int(cfg.tacmap_resolution_step))
            self._tacmap_cols = self._tacmap_rows

        if not bool(getattr(cfg, "enable_tacmap", True)):
            return sensors, slot_order

        # Deferred imports require SimulationApp.
        from isaaclab.sensors.ray_caster import patterns
        from tacmap_sensor.sharpa_tacmap_cfg import SharpaTacmapCfg
        from tacmap_sensor.sharpa_tacmap_vbts import SharpaTacmap
        from tacmap_sensor.sharpa_tacmap_link_surface import SharpaTacmapLinkSurface, SharpaTacmapLinkSurfaceCfg

        for i, link_path in enumerate(selected_links):
            points_npy, normals_npy = self._resolve_tacmap_files(cfg, link_path)
            if points_npy is None or normals_npy is None:
                print(f"[WARN] TacMap skipped for {link_path}: no points/normals npy mapping.", flush=True)
                continue
            normals_npy = self._prepare_tacmap_normals(cfg, normals_npy)

            ray_mode = str(getattr(cfg, "tacmap_ray_mode", "surface_normal"))
            if ray_mode == "link_surface":
                surface_target_path = str(link_path)
                target_path = str(target_query_paths[i])
                ray_direction = self._resolve_link_surface_ray_direction(cfg, normals_npy)
                link_surface_params = self._resolve_link_surface_params(cfg, link_path)

                common_kwargs = dict(
                    prim_path=str(link_path),
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
                    max_distance=float(link_surface_params["max_distance"]),
                    cpd_max_dist=float(cfg.tacmap_cpd_max_dist),
                    correction_scale=float(cfg.tacmap_correction_scale),
                    pts_offsets=float(cfg.tacmap_points_offset),
                    image_width=max(1, int(cfg.tacmap_link_surface_width)),
                    image_height=max(1, int(cfg.tacmap_link_surface_height)),
                    ray_axis=str(link_surface_params["ray_axis"]),
                    ray_direction=ray_direction,
                    grid_u_axis=str(link_surface_params["grid_u_axis"]),
                    grid_v_axis=str(link_surface_params["grid_v_axis"]),
                    grid_u_size=float(link_surface_params["grid_u_size"]),
                    grid_v_size=float(link_surface_params["grid_v_size"]),
                    grid_center=tuple(float(v) for v in link_surface_params["grid_center"]),
                    debug_viz_env_id=0,
                    debug_viz_max_points=int(cfg.tacmap_debug_viz_max_points),
                )
                surface_cfg = SharpaTacmapLinkSurfaceCfg(
                    **common_kwargs,
                    ray_hit_index=2,
                    mesh_prim_paths=[
                        SharpaTacmapLinkSurfaceCfg.RaycastTargetCfg(
                            prim_expr=surface_target_path,
                            track_mesh_transforms=True,
                        )
                    ],
                    debug_viz_hits=False,
                    debug_viz_link_surfaces=bool(cfg.tacmap_link_surface_debug_surfaces),
                    debug_viz_rays=False,
                    debug_viz_ray_length=float(cfg.tacmap_link_surface_debug_ray_length),
                    debug_viz_ray_width=float(cfg.tacmap_link_surface_debug_ray_width),
                )
                object_cfg = SharpaTacmapLinkSurfaceCfg(
                    **common_kwargs,
                    ray_hit_index=1,
                    mesh_prim_paths=[
                        SharpaTacmapLinkSurfaceCfg.RaycastTargetCfg(
                            prim_expr=target_path,
                            track_mesh_transforms=True,
                        )
                    ],
                    debug_viz_hits=bool(cfg.tacmap_debug_viz_active_points),
                    debug_viz_hits_external_mask=True,
                    debug_viz_rays=bool(cfg.tacmap_link_surface_debug_rays),
                    debug_viz_ray_length=float(cfg.tacmap_link_surface_debug_ray_length),
                    debug_viz_ray_width=float(cfg.tacmap_link_surface_debug_ray_width),
                )
                surface_sensor = SharpaTacmapLinkSurface(surface_cfg)
                sensor = SharpaTacmapLinkSurface(object_cfg)
            else:
                target_path = str(target_query_paths[i])
                sensor_cfg = SharpaTacmapCfg(
                    prim_path=str(link_path),
                    mesh_prim_paths=[
                        SharpaTacmapCfg.RaycastTargetCfg(
                            prim_expr=target_path,
                            track_mesh_transforms=True,
                        )
                    ],
                    update_period=0.0,
                    pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
                    offset=SharpaTacmapCfg.OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        convention="world",
                    ),
                    data_types=["distance_along_normal", "distance_along_normal_raw"],
                    points_npy=str(points_npy),
                    normals_npy=str(normals_npy),
                    resolution_step=max(1, int(cfg.tacmap_resolution_step)),
                    max_distance=float(cfg.tacmap_max_distance),
                    cpd_max_dist=float(cfg.tacmap_cpd_max_dist),
                    correction_scale=float(cfg.tacmap_correction_scale),
                    pts_offsets=float(cfg.tacmap_points_offset),
                    debug_hit_stats=bool(cfg.tacmap_debug_hit_stats),
                    debug_hit_stats_every=int(cfg.tacmap_debug_hit_stats_every),
                    debug_viz=bool(cfg.tacmap_debug_vis),
                    debug_viz_max_points=int(cfg.tacmap_debug_viz_max_points),
                    debug_viz_normals=bool(cfg.tacmap_debug_viz_normals),
                    debug_viz_active_points=bool(cfg.tacmap_debug_viz_active_points),
                    debug_viz_active_threshold=float(cfg.tacmap_debug_viz_active_threshold),
                )
                sensor = SharpaTacmap(sensor_cfg)
                surface_sensor = None
            self._tacmap_sensors.append(sensor)
            self._tacmap_surface_sensors.append(surface_sensor)
            self._tacmap_surface_raw_cache.append(None)
            self._tacmap_surface_local_cache.append(None)
            self._tacmap_penetration_cache.append(None)
            self._tacmap_slot_order.append(slot_order[i] if i < len(slot_order) else len(self._tacmap_slot_order))
            detail = (
                f"surface={surface_target_path} -> query={target_path}"
                if ray_mode == "link_surface"
                else f"link={link_path} -> query={target_path}"
            )
            print(
                f"  [TacMap {len(self._tacmap_sensors) - 1}] slot={self._tacmap_slot_order[-1]} "
                f"mode={ray_mode} {detail}",
                flush=True,
            )

        return sensors, slot_order

    @staticmethod
    def _find_tacmap_link_key(link_path: str, table: dict) -> str | None:
        path_text = str(link_path)
        for key in table:
            if key in path_text:
                return key
        return None

    def _resolve_tacmap_files(self, cfg: IntegratedTactileEnvCfg, link_path: str):
        link_key = self._find_tacmap_link_key(link_path, TACMAP_FILES)
        if link_key is not None:
            points, normals = TACMAP_FILES[link_key]
            if points.is_file() and normals.is_file():
                return points, normals

        if _is_revo21_pressure_pad_touch_link(link_path):
            return None, None

        if cfg.enable_press_motion or cfg.enable_sample_point_view:
            if str(cfg.press_touch_link).lower() in str(link_path).lower() and not bool(
                getattr(cfg, "press_tacmap_files_enabled", True)
            ):
                return None, None
            points = Path(os.path.expanduser(str(cfg.press_points_npy)))
            normals = Path(os.path.expanduser(str(cfg.press_normals_npy)))
            if points.is_file() and normals.is_file():
                return points, normals

        for key, files in TACMAP_FILES.items():
            if key in str(link_path):
                points, normals = files
                if points.is_file() and normals.is_file():
                    return points, normals
        return None, None

    def _resolve_link_surface_params(self, cfg: IntegratedTactileEnvCfg, link_path: str) -> dict:
        params = {
            "max_distance": float(cfg.tacmap_max_distance),
            "ray_axis": str(cfg.tacmap_link_surface_ray_axis),
            "grid_u_axis": str(cfg.tacmap_link_surface_grid_u_axis),
            "grid_v_axis": str(cfg.tacmap_link_surface_grid_v_axis),
            "grid_u_size": float(cfg.tacmap_link_surface_grid_u_size),
            "grid_v_size": float(cfg.tacmap_link_surface_grid_v_size),
            "grid_center": tuple(float(v) for v in cfg.tacmap_link_surface_grid_center),
        }

        link_key = self._find_tacmap_link_key(link_path, TACMAP_LINK_SURFACE_DEFAULTS)
        if link_key is None:
            return params

        defaults = TACMAP_LINK_SURFACE_DEFAULTS[link_key]
        for name, old_value in _LEGACY_LINK_SURFACE_DEFAULTS.items():
            current_value = params[name]
            if self._same_link_surface_value(current_value, old_value):
                params[name] = defaults[name]
        for name in ("ray_axis", "grid_u_axis", "grid_v_axis"):
            params[name] = defaults.get(name, params[name])
        return params

    @staticmethod
    def _same_link_surface_value(lhs, rhs, eps: float = 1.0e-9) -> bool:
        lhs_arr = np.asarray(lhs, dtype=np.float64).reshape(-1)
        rhs_arr = np.asarray(rhs, dtype=np.float64).reshape(-1)
        return lhs_arr.shape == rhs_arr.shape and bool(np.all(np.abs(lhs_arr - rhs_arr) <= eps))

    def _prepare_tacmap_normals(self, cfg: IntegratedTactileEnvCfg, normals_npy: Path) -> Path:
        """Return normals file for SharpaTacmap.

        SharpaTacmap already flips normals internally in the same way as the
        original tacmap demo. The normal-file inversion option is kept only as
        a debugging switch for assets whose tactile map uses the opposite sign.
        """
        if not bool(getattr(cfg, "tacmap_invert_normals_file", False)):
            return normals_npy

        out_dir = REPO_ROOT / "integrate" / "output" / "normal_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{normals_npy.stem}_neg.npy"
        src_mtime = normals_npy.stat().st_mtime
        if out_path.is_file() and out_path.stat().st_mtime >= src_mtime:
            return out_path

        normals = np.load(normals_npy).astype(np.float32)
        np.save(out_path, -normals)
        return out_path

    def _resolve_link_surface_ray_direction(
        self,
        cfg: IntegratedTactileEnvCfg,
        normals_npy: Path,
    ) -> tuple[float, float, float] | None:
        direction = getattr(cfg, "tacmap_link_surface_ray_direction", None)
        if direction is not None:
            vec = np.asarray(direction, dtype=np.float32)
        elif bool(getattr(cfg, "tacmap_link_surface_use_mean_normal", True)):
            normals = np.load(normals_npy).astype(np.float32).reshape(-1, 3)
            valid = np.linalg.norm(normals, axis=-1) > 1.0e-8
            if not np.any(valid):
                return None
            normals = normals[valid]
            normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1.0e-12)
            vec = np.mean(normals, axis=0)
        else:
            return None

        norm = float(np.linalg.norm(vec))
        if norm < 1.0e-8:
            return None
        vec = vec / norm
        return (float(vec[0]), float(vec[1]), float(vec[2]))

    def _get_obs(self) -> dict:
        obs = super()._get_obs()
        obs["tacmap"] = self._get_tacmap_obs()
        obs["tacmap_raw"] = self._get_tacmap_raw_obs(update=False)
        obs["tacmap_object_raw"] = self._get_tacmap_object_raw_obs(update=False)
        obs["tacmap_surface_raw"] = self._get_tacmap_surface_raw_obs(update=False)
        (
            obs["tacmap_surface_points_w"],
            obs["tacmap_surface_normals_w"],
            obs["tacmap_surface_valid"],
            obs["tacmap_object_points_w"],
            obs["tacmap_object_valid"],
        ) = self._get_tacmap_link_surface_geometry_obs(update=False)
        obs["fots_theta"] = self._get_fots_theta_obs()
        tacsl_depth, tacsl_normal, tacsl_shear = self._get_tacsl_force_field_obs(update=False)
        obs["tacsl_penetration_depth"] = tacsl_depth
        obs["tacsl_normal_force"] = tacsl_normal
        obs["tacsl_shear_force"] = tacsl_shear
        obs.update(self._build_rl_tactile_obs())
        return obs

    def update_rl_hydroshear_obs(
        self,
        obs: dict,
        hydroshear_displacement_m: np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor:
        """Patch the current obs dict with the modified-HydroShear RL channel."""

        obs.update(self._build_rl_tactile_obs(hydroshear_displacement_m=hydroshear_displacement_m))
        return obs["rl_tactile_obs"]

    def _build_rl_tactile_obs(
        self,
        *,
        hydroshear_displacement_m: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        warpsdf = self._rl_warpsdf_pressure_obs()
        tacmap = self._rl_tacmap_obs()
        hydroshear = self._rl_hydroshear_obs(hydroshear_displacement_m)
        return {
            "rl_warpsdf_pressure_obs": warpsdf,
            "rl_tacmap_obs": tacmap,
            "rl_hydroshear_obs": hydroshear,
            "rl_tactile_obs": torch.cat((warpsdf, tacmap, hydroshear), dim=0),
        }

    def _rl_device(self) -> torch.device:
        return torch.device(getattr(self, "_device", "cpu"))

    def _rl_empty(self, *shape: int) -> torch.Tensor:
        return torch.zeros(tuple(max(0, int(value)) for value in shape), device=self._rl_device(), dtype=torch.float32)

    def _rl_warpsdf_pressure_obs(self) -> torch.Tensor:
        rows = max(1, int(self._cfg.num_rows))
        cols = max(1, int(self._cfg.num_cols))
        out = self._rl_empty(len(RL_PRESSURE_PAD_LINK_ORDER), rows, cols)
        slot_count = max(len(self._tactile_sensors), max(self._sensor_slot_order) + 1 if self._sensor_slot_order else 0)
        labels = pressure_sensor_labels_from_env(self, slot_count, fallback_label="")
        slot_to_sensor = {
            int(slot): sensor_index
            for sensor_index, slot in enumerate(self._sensor_slot_order)
            if 0 <= int(slot) < slot_count
        }
        for out_index, link_name in enumerate(RL_PRESSURE_PAD_LINK_ORDER):
            source_index = self._matching_label_index(labels, (link_name,))
            if source_index is None and slot_count == len(RL_PRESSURE_PAD_LINK_ORDER):
                source_index = out_index
            sensor_index = None if source_index is None else slot_to_sensor.get(int(source_index))
            if sensor_index is None or sensor_index >= len(self._tactile_sensors):
                continue
            source = self._pressure_force_map_tensor(self._tactile_sensors[sensor_index], rows, cols)
            if source is None:
                continue
            copy_rows = min(rows, source.shape[0])
            copy_cols = min(cols, source.shape[1])
            out[out_index, :copy_rows, :copy_cols] = source[:copy_rows, :copy_cols]
        return out.reshape(-1)

    def _pressure_force_map_tensor(self, sensor, rows: int, cols: int) -> torch.Tensor | None:
        sensor_data = getattr(sensor, "data", None)
        if sensor_data is None:
            return None

        force_map = getattr(sensor_data, "pressure_force_map", None)
        if force_map is not None:
            source = force_map.detach()
            if source.ndim >= 4:
                source = source[0, 0]
            elif source.ndim == 3:
                source = source[0]
            elif source.ndim == 1 and source.numel() == rows * cols:
                source = source.reshape(rows, cols)
            if source.ndim >= 2:
                return self._finite_tensor(source[:rows, :cols])

        tactile_points = getattr(sensor_data, "tactile_points_w", None)
        if tactile_points is not None:
            values = tactile_points[0, :, 3].detach()
            if values.numel() == rows * cols:
                return self._finite_tensor(values.reshape(rows, cols))
        return None

    def _rl_tacmap_obs(self) -> torch.Tensor:
        rows, cols = self._tacmap_rows, self._tacmap_cols
        out = self._rl_empty(len(RL_FINGER_ORDER), rows, cols)
        if rows <= 0 or cols <= 0:
            return out.reshape(-1)

        slot_count = self._tacmap_output_shape()[0]
        labels = self._tacmap_slot_labels(slot_count)
        slot_to_sensor = {
            int(slot): sensor_index
            for sensor_index, slot in enumerate(self._tacmap_slot_order)
            if 0 <= int(slot) < slot_count
        }
        for out_index, finger in enumerate(RL_FINGER_ORDER):
            source_index = self._matching_label_index(labels, RL_TACMAP_LABEL_HINTS[finger])
            if source_index is None and slot_count == len(RL_FINGER_ORDER):
                source_index = out_index
            sensor_index = None if source_index is None else slot_to_sensor.get(int(source_index))
            if sensor_index is None or sensor_index >= len(self._tacmap_sensors):
                continue
            source = self._tacmap_raw_tensor(sensor_index, rows, cols)
            if source is None:
                continue
            copy_rows = min(rows, source.shape[0])
            copy_cols = min(cols, source.shape[1])
            out[out_index, :copy_rows, :copy_cols] = source[:copy_rows, :copy_cols]
        return out.reshape(-1)

    def _tacmap_raw_tensor(self, sensor_index: int, rows: int, cols: int) -> torch.Tensor | None:
        penetration = self._link_surface_penetration_tensor(sensor_index, rows, cols)
        if penetration is not None:
            return penetration
        return self._raw_sensor_tensor(self._tacmap_sensors[sensor_index], rows, cols)

    def _link_surface_penetration_tensor(self, sensor_index: int, rows: int, cols: int) -> torch.Tensor | None:
        if sensor_index >= len(self._tacmap_sensors) or sensor_index >= len(self._tacmap_surface_sensors):
            return None
        surface_dist = self._surface_raw_tensor(sensor_index, rows, cols)
        object_dist = self._raw_sensor_tensor(self._tacmap_sensors[sensor_index], rows, cols)
        if surface_dist is None or object_dist is None:
            return None
        valid = (surface_dist > 0.0) & (object_dist > 0.0) & (object_dist <= surface_dist)
        return torch.where(valid, surface_dist - object_dist, torch.zeros_like(surface_dist))

    def _surface_raw_tensor(self, sensor_index: int, rows: int, cols: int) -> torch.Tensor | None:
        surface_sensor = self._update_tacmap_surface_sensor(sensor_index)
        if surface_sensor is None and sensor_index < len(self._tacmap_surface_sensors):
            surface_sensor = self._tacmap_surface_sensors[sensor_index]
        if surface_sensor is None:
            return None
        return self._raw_sensor_tensor(surface_sensor, rows, cols)

    def _raw_sensor_tensor(self, sensor, rows: int, cols: int) -> torch.Tensor | None:
        data = sensor.data.output.get("distance_along_normal_raw")
        if data is None:
            return None
        source = data[0].detach()
        if source.numel() != rows * cols:
            return None
        source = source.reshape(rows, cols)
        return self._finite_tensor(source)

    def _finite_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(device=self._rl_device(), dtype=torch.float32)
        return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))

    def _rl_hydroshear_obs(self, hydroshear_displacement_m: np.ndarray | torch.Tensor | None) -> torch.Tensor:
        marker_count = RL_HYDROSHEAR_MARKER_ROWS * RL_HYDROSHEAR_MARKER_COLS
        out = self._rl_empty(len(RL_FINGER_ORDER), marker_count, 3)
        if hydroshear_displacement_m is None:
            return out.reshape(-1)

        if isinstance(hydroshear_displacement_m, torch.Tensor):
            disp = hydroshear_displacement_m.detach().to(device=self._rl_device(), dtype=torch.float32)
        else:
            disp = torch.as_tensor(hydroshear_displacement_m, dtype=torch.float32).to(device=self._rl_device())
        disp = torch.where(torch.isfinite(disp), disp, torch.zeros_like(disp))
        if disp.ndim == 4 and disp.shape[-1] == 3:
            disp = disp.reshape(disp.shape[0], -1, 3)
        if disp.ndim != 3 or disp.shape[-1] != 3:
            return out.reshape(-1)
        labels = self._tacmap_slot_labels(disp.shape[0])
        for out_index, finger in enumerate(RL_FINGER_ORDER):
            source_index = self._matching_label_index(labels, RL_TACMAP_LABEL_HINTS[finger])
            if source_index is None and disp.shape[0] == len(RL_FINGER_ORDER):
                source_index = out_index
            if source_index is None or source_index >= disp.shape[0]:
                continue
            copy_count = min(marker_count, disp.shape[1])
            out[out_index, :copy_count, :] = disp[source_index, :copy_count, :]
        return out.reshape(-1)

    def _tacmap_slot_labels(self, count: int) -> list[str]:
        labels = [f"S{i}" for i in range(max(0, int(count)))]
        for sensor_index, sensor in enumerate(self._tacmap_sensors):
            try:
                slot = int(self._tacmap_slot_order[sensor_index]) if sensor_index < len(self._tacmap_slot_order) else sensor_index
            except (TypeError, ValueError):
                slot = sensor_index
            if not (0 <= slot < len(labels)):
                continue
            cfg = getattr(sensor, "cfg", None)
            prim_path = str(getattr(cfg, "prim_path", ""))
            labels[slot] = Path(prim_path).name or prim_path or labels[slot]
        return labels

    @staticmethod
    def _matching_label_index(labels: list[str], hints: tuple[str, ...]) -> int | None:
        normalized_hints = tuple(str(hint).lower() for hint in hints if str(hint))
        for index, label in enumerate(labels):
            text = str(label).lower()
            if any(text == hint or hint in text or text in hint for hint in normalized_hints):
                return index
        return None

    def _update_tacmap_sensors(self) -> None:
        self._tacmap_penetration_cache = [None] * len(self._tacmap_sensors)
        for i, sensor in enumerate(self._tacmap_sensors):
            self._update_tacmap_surface_sensor(i)
            sensor.update(dt=float(self._cfg.physics_dt), force_recompute=True)

    def _tacmap_output_shape(self) -> tuple[int, int, int]:
        count = max(len(self._tacmap_sensors), max(self._tacmap_slot_order) + 1 if self._tacmap_slot_order else 0)
        return count, self._tacmap_rows, self._tacmap_cols

    def _raw_sensor_array(self, sensor) -> np.ndarray | None:
        data = sensor.data.output.get("distance_along_normal_raw")
        if data is None:
            return None
        arr = data[0].detach().reshape(self._tacmap_rows, self._tacmap_cols).cpu().numpy()
        return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _update_tacmap_surface_sensor(self, sensor_index: int):
        if sensor_index >= len(self._tacmap_surface_sensors):
            return None
        surface_sensor = self._tacmap_surface_sensors[sensor_index]
        if surface_sensor is None:
            return None
        cache = self._ensure_tacmap_surface_local_cache(sensor_index)
        if cache is None:
            return None
        self._refresh_tacmap_surface_world_geometry(surface_sensor, cache)
        return surface_sensor

    def _ensure_tacmap_surface_local_cache(self, sensor_index: int) -> dict[str, torch.Tensor] | None:
        if sensor_index >= len(self._tacmap_surface_sensors):
            return None
        surface_sensor = self._tacmap_surface_sensors[sensor_index]
        if surface_sensor is None:
            return None
        while sensor_index >= len(self._tacmap_surface_local_cache):
            self._tacmap_surface_local_cache.append(None)
        cached = self._tacmap_surface_local_cache[sensor_index]
        if cached is not None:
            return cached

        surface_sensor.update(dt=float(self._cfg.physics_dt), force_recompute=True)
        data = surface_sensor.data
        hits_w = getattr(surface_sensor, "second_ray_hits_w", None)
        normals_w = getattr(surface_sensor, "second_ray_normals_w", None)
        valid = getattr(surface_sensor, "second_ray_hit_valid", None)
        first_hits_w = getattr(surface_sensor, "first_ray_hits_w", None)
        first_normals_w = getattr(surface_sensor, "first_ray_normals_w", None)
        first_valid = getattr(surface_sensor, "first_ray_hit_valid", None)
        if hits_w is None or normals_w is None or valid is None:
            return None

        pos_w = surface_sensor._data.pos_w.detach()
        quat_w = surface_sensor._data.quat_w.detach()
        quat_inv = self._math_utils.quat_inv(quat_w)
        quat_inv_expanded = quat_inv.unsqueeze(1).expand(-1, hits_w.shape[1], 4)
        points_l = self._math_utils.quat_apply(quat_inv_expanded, hits_w.detach() - pos_w.unsqueeze(1))
        normals_l = self._math_utils.quat_apply(quat_inv_expanded, normals_w.detach())

        cache: dict[str, torch.Tensor] = {
            "points_l": points_l.detach().clone(),
            "normals_l": normals_l.detach().clone(),
            "valid": valid.detach().clone(),
        }
        if first_hits_w is not None and first_normals_w is not None and first_valid is not None:
            first_points_l = self._math_utils.quat_apply(
                quat_inv_expanded,
                first_hits_w.detach() - pos_w.unsqueeze(1),
            )
            first_normals_l = self._math_utils.quat_apply(quat_inv_expanded, first_normals_w.detach())
            cache["first_points_l"] = first_points_l.detach().clone()
            cache["first_normals_l"] = first_normals_l.detach().clone()
            cache["first_valid"] = first_valid.detach().clone()

        raw = data.output.get("distance_along_normal_raw")
        if raw is not None:
            cache["raw"] = raw.detach().clone()
            arr = self._raw_sensor_array(surface_sensor)
            if arr is not None:
                while sensor_index >= len(self._tacmap_surface_raw_cache):
                    self._tacmap_surface_raw_cache.append(None)
                self._tacmap_surface_raw_cache[sensor_index] = arr.copy()
        quantized = data.output.get("distance_along_normal")
        if quantized is not None:
            cache["quantized"] = quantized.detach().clone()

        self._tacmap_surface_local_cache[sensor_index] = cache
        return cache

    def _refresh_tacmap_surface_world_geometry(self, surface_sensor, cache: dict[str, torch.Tensor]) -> None:
        env_ids = getattr(surface_sensor, "_ALL_INDICES", None)
        if env_ids is None:
            view = getattr(surface_sensor, "_view", None)
            count = int(getattr(view, "count", 1))
            env_ids = torch.arange(count, device=self._device, dtype=torch.long)

        surface_sensor._update_ray_infos(env_ids)
        pos_w = surface_sensor._data.pos_w
        quat_w = surface_sensor._data.quat_w
        points_l = cache["points_l"].to(device=pos_w.device, dtype=pos_w.dtype)
        normals_l = cache["normals_l"].to(device=pos_w.device, dtype=pos_w.dtype)
        valid = cache["valid"].to(device=pos_w.device, dtype=torch.bool)
        quat_expanded = quat_w.unsqueeze(1).expand(-1, points_l.shape[1], 4)

        points_w = self._math_utils.quat_apply(quat_expanded, points_l) + pos_w.unsqueeze(1)
        normals_w = self._math_utils.quat_apply(quat_expanded, normals_l)
        zero_points = torch.zeros_like(points_w)
        zero_normals = torch.zeros_like(normals_w)
        surface_sensor.second_ray_hits_w[:] = torch.where(valid.unsqueeze(-1), points_w, zero_points)
        surface_sensor.second_ray_normals_w[:] = torch.where(valid.unsqueeze(-1), normals_w, zero_normals)
        surface_sensor.second_ray_hit_valid[:] = valid
        surface_sensor.ray_hits_w[:] = surface_sensor.second_ray_hits_w

        if "first_points_l" in cache and hasattr(surface_sensor, "first_ray_hits_w"):
            first_points_l = cache["first_points_l"].to(device=pos_w.device, dtype=pos_w.dtype)
            first_normals_l = cache["first_normals_l"].to(device=pos_w.device, dtype=pos_w.dtype)
            first_valid = cache["first_valid"].to(device=pos_w.device, dtype=torch.bool)
            first_points_w = self._math_utils.quat_apply(quat_expanded, first_points_l) + pos_w.unsqueeze(1)
            first_normals_w = self._math_utils.quat_apply(quat_expanded, first_normals_l)
            surface_sensor.first_ray_hits_w[:] = torch.where(
                first_valid.unsqueeze(-1), first_points_w, torch.zeros_like(first_points_w)
            )
            surface_sensor.first_ray_normals_w[:] = torch.where(
                first_valid.unsqueeze(-1), first_normals_w, torch.zeros_like(first_normals_w)
            )
            surface_sensor.first_ray_hit_valid[:] = first_valid

        raw = cache.get("raw")
        if raw is not None and "distance_along_normal_raw" in surface_sensor._data.output:
            surface_sensor._data.output["distance_along_normal_raw"][:] = raw.to(
                device=pos_w.device,
                dtype=surface_sensor._data.output["distance_along_normal_raw"].dtype,
            )
        quantized = cache.get("quantized")
        if quantized is not None and "distance_along_normal" in surface_sensor._data.output:
            surface_sensor._data.output["distance_along_normal"][:] = quantized.to(
                device=pos_w.device,
                dtype=surface_sensor._data.output["distance_along_normal"].dtype,
            )

        if bool(getattr(surface_sensor.cfg, "debug_viz_link_surfaces", False)) and hasattr(
            surface_sensor, "_update_link_surfaces_viz_usd"
        ):
            surface_sensor._update_link_surfaces_viz_usd(
                env_ids,
                surface_sensor.first_ray_hits_w[env_ids],
                surface_sensor.first_ray_hit_valid[env_ids],
                surface_sensor.second_ray_hits_w[env_ids],
                surface_sensor.second_ray_hit_valid[env_ids],
            )
        if hasattr(surface_sensor, "_is_outdated"):
            surface_sensor._is_outdated[:] = False
        if hasattr(surface_sensor, "_timestamp_last_update") and hasattr(surface_sensor, "_timestamp"):
            surface_sensor._timestamp_last_update[:] = surface_sensor._timestamp

    def _ensure_tacmap_surface_cache(self, sensor_index: int, *, update_if_needed: bool = True) -> np.ndarray | None:
        if sensor_index >= len(self._tacmap_surface_sensors):
            return None
        surface_sensor = self._tacmap_surface_sensors[sensor_index]
        if surface_sensor is None:
            return None
        while sensor_index >= len(self._tacmap_surface_raw_cache):
            self._tacmap_surface_raw_cache.append(None)
        cached = self._tacmap_surface_raw_cache[sensor_index]
        if cached is not None:
            return cached

        if update_if_needed:
            self._update_tacmap_surface_sensor(sensor_index)
        local_cache = self._ensure_tacmap_surface_local_cache(sensor_index)
        if local_cache is not None and "raw" in local_cache:
            raw = local_cache["raw"].detach().reshape(surface_sensor._view.count, self._tacmap_rows, self._tacmap_cols, 1)
            arr = raw[0, :, :, 0].cpu().numpy().astype(np.float32)
        else:
            arr = self._raw_sensor_array(surface_sensor)
        if arr is not None:
            arr = arr.copy()
            self._tacmap_surface_raw_cache[sensor_index] = arr
        return arr

    def _link_surface_penetration_entry(self, sensor_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        if sensor_index >= len(self._tacmap_sensors) or sensor_index >= len(self._tacmap_surface_sensors):
            return None
        while sensor_index >= len(self._tacmap_penetration_cache):
            self._tacmap_penetration_cache.append(None)
        cached = self._tacmap_penetration_cache[sensor_index]
        if cached is not None:
            return cached

        surface_dist = self._ensure_tacmap_surface_cache(sensor_index)
        if surface_dist is None:
            return None

        object_dist = self._raw_sensor_array(self._tacmap_sensors[sensor_index])
        if object_dist is None:
            return None

        valid = (surface_dist > 0.0) & (object_dist > 0.0) & (object_dist <= surface_dist)
        penetration = np.where(valid, surface_dist - object_dist, 0.0)
        active = valid & (penetration > float(getattr(self._cfg, "tacmap_debug_viz_active_threshold", 0.0)))
        self._update_link_surface_valid_viz(sensor_index, active)
        entry = (penetration.astype(np.float32), valid.astype(bool))
        self._tacmap_penetration_cache[sensor_index] = entry
        return entry

    def _link_surface_penetration(self, sensor_index: int) -> np.ndarray | None:
        entry = self._link_surface_penetration_entry(sensor_index)
        return None if entry is None else entry[0]

    def _update_link_surface_valid_viz(self, sensor_index: int, valid_mask: np.ndarray) -> None:
        if sensor_index >= len(self._tacmap_sensors):
            return
        update_fn = getattr(self._tacmap_sensors[sensor_index], "update_hit_points_viz_mask", None)
        if callable(update_fn):
            update_fn(valid_mask, env_id=0)

    @staticmethod
    def _quantize_tacmap_depth(depth_m: np.ndarray) -> np.ndarray:
        depth_mm = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
        out = np.zeros_like(depth_mm, dtype=np.float32)
        mask = depth_mm < 0.5
        out[mask] = depth_mm[mask] / 5.0e-3
        out[~mask] = (depth_mm[~mask] - 0.5) / 3.0e-2 + 100.0
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def _get_tacmap_obs(self, update: bool = True) -> np.ndarray:
        if not self._tacmap_sensors:
            return np.zeros((0, self._tacmap_rows, self._tacmap_cols), dtype=np.uint8)

        if update:
            self._update_tacmap_sensors()

        out = np.zeros(self._tacmap_output_shape(), dtype=np.uint8)
        for i, sensor in enumerate(self._tacmap_sensors):
            penetration = self._link_surface_penetration(i)
            if penetration is not None:
                arr = self._quantize_tacmap_depth(penetration)
            else:
                data = sensor.data.output.get("distance_along_normal")
                if data is None:
                    continue
                arr = data[0].detach().reshape(self._tacmap_rows, self._tacmap_cols).cpu().numpy().astype(np.uint8)
            slot = self._tacmap_slot_order[i] if i < len(self._tacmap_slot_order) else i
            if 0 <= slot < out.shape[0]:
                out[slot] = arr
        return out

    def _get_tacmap_raw_obs(self, update: bool = True) -> np.ndarray:
        if not self._tacmap_sensors:
            return np.zeros((0, self._tacmap_rows, self._tacmap_cols), dtype=np.float32)

        if update:
            self._update_tacmap_sensors()

        out = np.zeros(self._tacmap_output_shape(), dtype=np.float32)
        for i, sensor in enumerate(self._tacmap_sensors):
            penetration = self._link_surface_penetration(i)
            if penetration is not None:
                arr = penetration
            else:
                data = sensor.data.output.get("distance_along_normal_raw")
                if data is None:
                    continue
                arr = data[0].detach().reshape(self._tacmap_rows, self._tacmap_cols).cpu().numpy().astype(np.float32)
            slot = self._tacmap_slot_order[i] if i < len(self._tacmap_slot_order) else i
            if 0 <= slot < out.shape[0]:
                out[slot] = arr
        return out

    def _get_tacmap_surface_raw_obs(self, update: bool = True) -> np.ndarray:
        if not self._tacmap_sensors:
            return np.zeros((0, self._tacmap_rows, self._tacmap_cols), dtype=np.float32)

        if update:
            self._update_tacmap_sensors()

        out = np.zeros(self._tacmap_output_shape(), dtype=np.float32)
        for i, _surface_sensor in enumerate(self._tacmap_surface_sensors):
            arr = self._ensure_tacmap_surface_cache(i)
            if arr is None:
                continue
            slot = self._tacmap_slot_order[i] if i < len(self._tacmap_slot_order) else i
            if 0 <= slot < out.shape[0]:
                out[slot] = arr
        return out

    def _get_tacmap_object_raw_obs(self, update: bool = True) -> np.ndarray:
        if not self._tacmap_sensors:
            return np.zeros((0, self._tacmap_rows, self._tacmap_cols), dtype=np.float32)

        if update:
            self._update_tacmap_sensors()

        out = np.zeros(self._tacmap_output_shape(), dtype=np.float32)
        for i, sensor in enumerate(self._tacmap_sensors):
            arr = self._raw_sensor_array(sensor)
            if arr is None:
                continue
            slot = self._tacmap_slot_order[i] if i < len(self._tacmap_slot_order) else i
            if 0 <= slot < out.shape[0]:
                out[slot] = arr
        return out

    def _get_tacmap_link_surface_geometry_obs(
        self,
        update: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = self._tacmap_output_shape()
        surface_points = np.zeros((*shape, 3), dtype=np.float32)
        surface_normals = np.zeros((*shape, 3), dtype=np.float32)
        object_points = np.zeros((*shape, 3), dtype=np.float32)
        surface_valid = np.zeros(shape, dtype=np.uint8)
        object_valid = np.zeros(shape, dtype=np.uint8)

        if not self._tacmap_sensors:
            return surface_points, surface_normals, surface_valid, object_points, object_valid

        if update:
            self._update_tacmap_sensors()

        rows, cols = self._tacmap_rows, self._tacmap_cols
        num_points = rows * cols
        for i, sensor in enumerate(self._tacmap_sensors):
            slot = self._tacmap_slot_order[i] if i < len(self._tacmap_slot_order) else i
            if not (0 <= slot < shape[0]):
                continue

            if i < len(self._tacmap_surface_sensors):
                surface_sensor = self._tacmap_surface_sensors[i]
            else:
                surface_sensor = None

            if surface_sensor is not None:
                hits = getattr(surface_sensor, "second_ray_hits_w", None)
                normals = getattr(surface_sensor, "second_ray_normals_w", None)
                valid = getattr(surface_sensor, "second_ray_hit_valid", None)
                if hits is not None and valid is not None and hits.shape[1] == num_points:
                    surface_points[slot] = hits[0].detach().reshape(rows, cols, 3).cpu().numpy().astype(np.float32)
                    surface_valid[slot] = valid[0].detach().reshape(rows, cols).cpu().numpy().astype(np.uint8)
                if normals is not None and normals.shape[1] == num_points:
                    surface_normals[slot] = normals[0].detach().reshape(rows, cols, 3).cpu().numpy().astype(np.float32)

            hits = getattr(sensor, "ray_hits_w", None)
            raw = self._raw_sensor_array(sensor)
            if hits is not None and hits.shape[1] == num_points:
                object_points[slot] = hits[0].detach().reshape(rows, cols, 3).cpu().numpy().astype(np.float32)
                if raw is not None:
                    object_valid[slot] = (np.asarray(raw, dtype=np.float32).reshape(rows, cols) > 0.0).astype(np.uint8)

        return surface_points, surface_normals, surface_valid, object_points, object_valid

    def _get_fots_theta_obs(self) -> np.ndarray:
        """Return target twist in each tactile source frame.

        TacEx feeds FOTS the target orientation expressed in the tactile source
        frame. For RevoLab the tactile image plane is defined by the configured
        grid u/v axes, so theta is the signed twist around that plane normal.
        """
        out = np.full((self._tacmap_output_shape()[0],), np.nan, dtype=np.float32)
        if self._plug_obj is None or self._press_touch_body_idx is None:
            return out

        root_quat_w = getattr(self._plug_obj.data, "root_quat_w", None)
        if root_quat_w is None:
            return out

        try:
            touch_quat_w = self._robot.data.body_link_state_w[0, self._press_touch_body_idx, 3:7]
            object_quat_w = root_quat_w[0]
            raycaster_quat_l = self._fots_raycaster_frame_quat_l()
            raycaster_quat_w = self._math_utils.quat_mul(
                touch_quat_w.unsqueeze(0),
                raycaster_quat_l.unsqueeze(0),
            )
            rel_quat = self._math_utils.quat_mul(
                self._math_utils.quat_inv(raycaster_quat_w),
                object_quat_w.unsqueeze(0),
            )
            theta = float(
                self._signed_twist_angle(rel_quat[0], self._local_axis_tensor("+x")).detach().cpu().item()
            )
        except Exception:
            return out

        for i, _sensor in enumerate(self._tacmap_sensors):
            slot = self._tacmap_slot_order[i] if i < len(self._tacmap_slot_order) else i
            if 0 <= slot < out.shape[0]:
                out[slot] = theta
        return out

    def _fots_twist_axis_l(self) -> torch.Tensor:
        return self._fots_raycaster_axes_l()[0]

    def _fots_raycaster_frame_quat_l(self) -> torch.Tensor:
        ray_axis_l, u_axis_l, v_axis_l = self._fots_raycaster_axes_l()
        rot_l = torch.stack((ray_axis_l, u_axis_l, v_axis_l), dim=-1)
        if float(torch.linalg.det(rot_l).detach().cpu().item()) < 0.0:
            v_axis_l = -v_axis_l
            rot_l = torch.stack((ray_axis_l, u_axis_l, v_axis_l), dim=-1)
        quat = self._math_utils.quat_from_matrix(rot_l.unsqueeze(0))[0]
        return quat / torch.linalg.norm(quat).clamp_min(1.0e-8)

    def _fots_raycaster_axes_l(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        press_normal_l = getattr(self, "_press_touch_normal_l", None)
        if (
            press_normal_l is not None
            and str(getattr(self._cfg, "press_motion_frame", "touch")) == "link_surface"
        ):
            norm = torch.linalg.norm(press_normal_l)
            if float(norm.detach().cpu().item()) >= 1.0e-8:
                ray_axis = press_normal_l / norm.clamp_min(1.0e-8)
            else:
                ray_axis = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_ray_axis", "+x")))
        else:
            direction = getattr(self._cfg, "tacmap_link_surface_ray_direction", None)
            if direction is not None:
                axis = torch.tensor(direction, dtype=torch.float32, device=self._device)
                norm = torch.linalg.norm(axis)
                if float(norm.detach().cpu().item()) >= 1.0e-8:
                    ray_axis = axis / norm.clamp_min(1.0e-8)
                else:
                    ray_axis = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_ray_axis", "+x")))
            else:
                ray_axis = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_ray_axis", "+x")))

        u_axis = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_grid_u_axis", "+y")))
        v_axis = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_grid_v_axis", "+z")))
        return self._orthonormalize_raycaster_axes_l(ray_axis, u_axis, v_axis)

    def _orthonormalize_raycaster_axes_l(
        self,
        ray_axis: torch.Tensor,
        u_axis: torch.Tensor,
        v_axis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ray_axis = ray_axis / torch.linalg.norm(ray_axis).clamp_min(1.0e-8)

        u_axis = u_axis - torch.dot(u_axis, ray_axis) * ray_axis
        u_norm = torch.linalg.norm(u_axis)
        if float(u_norm.detach().cpu().item()) < 1.0e-8:
            for candidate in (
                torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=self._device),
                torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=self._device),
                torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=self._device),
            ):
                projected = candidate - torch.dot(candidate, ray_axis) * ray_axis
                if float(torch.linalg.norm(projected).detach().cpu().item()) >= 1.0e-8:
                    u_axis = projected
                    u_norm = torch.linalg.norm(u_axis)
                    break
        u_axis = u_axis / u_norm.clamp_min(1.0e-8)

        v_axis = v_axis - torch.dot(v_axis, ray_axis) * ray_axis - torch.dot(v_axis, u_axis) * u_axis
        v_norm = torch.linalg.norm(v_axis)
        if float(v_norm.detach().cpu().item()) < 1.0e-8:
            v_axis = torch.linalg.cross(ray_axis, u_axis)
            v_norm = torch.linalg.norm(v_axis)
        v_axis = v_axis / v_norm.clamp_min(1.0e-8)
        return ray_axis, u_axis, v_axis

    def _local_axis_tensor(self, axis_name: str) -> torch.Tensor:
        sign = -1.0 if axis_name.startswith("-") else 1.0
        axis = axis_name[-1].lower()
        values = {
            "x": (sign, 0.0, 0.0),
            "y": (0.0, sign, 0.0),
            "z": (0.0, 0.0, sign),
        }.get(axis, (1.0, 0.0, 0.0))
        return torch.tensor(values, dtype=torch.float32, device=self._device)

    @staticmethod
    def _signed_twist_angle(quat_wxyz: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        quat = quat_wxyz / torch.linalg.norm(quat_wxyz).clamp_min(1.0e-8)
        axis = axis / torch.linalg.norm(axis).clamp_min(1.0e-8)
        projected = torch.sum(quat[1:4] * axis)
        angle = 2.0 * torch.atan2(projected, quat[0])
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _tactile_force_output_shape(self) -> tuple[int, int, int]:
        count = max(len(self._tactile_sensors), max(self._sensor_slot_order) + 1 if self._sensor_slot_order else 0)
        return count, int(self._cfg.num_rows), int(self._cfg.num_cols)

    def _tacsl_force_output_shape(self) -> tuple[int, int, int]:
        if (
            bool(getattr(self._cfg, "enable_tacsl_force_field", False))
            and str(getattr(self._cfg, "tacmap_ray_mode", "surface_normal")) == "link_surface"
            and self._tacmap_surface_sensors
        ):
            return self._tacmap_output_shape()
        return self._tactile_force_output_shape()

    def _get_tacsl_force_field_obs(self, update: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = self._tacsl_force_output_shape()
        depth_out = torch.zeros(shape, dtype=torch.float32, device=self._device)
        normal_out = torch.zeros(shape, dtype=torch.float32, device=self._device)
        shear_out = torch.zeros((*shape, 2), dtype=torch.float32, device=self._device)

        if not bool(getattr(self._cfg, "enable_tacsl_force_field", False)):
            return depth_out, normal_out, shear_out

        use_link_surface = (
            str(getattr(self._cfg, "tacmap_ray_mode", "surface_normal")) == "link_surface"
            and self._tacmap_surface_sensors
        )
        if use_link_surface:
            if update:
                self._update_tacmap_sensors()

            self._ensure_tacsl_prev_buffers(len(self._tacmap_surface_sensors))
            for i, surface_sensor in enumerate(self._tacmap_surface_sensors):
                if surface_sensor is None:
                    continue
                result = self._compute_tacsl_force_field_from_link_surface(i, surface_sensor)
                if result is None:
                    continue
                depth, normal_force, shear_force = result
                slot = self._tacmap_slot_order[i] if i < len(self._tacmap_slot_order) else i
                if 0 <= slot < depth_out.shape[0]:
                    depth_out[slot] = depth.to(device=self._device, dtype=torch.float32)
                    normal_out[slot] = normal_force.to(device=self._device, dtype=torch.float32)
                    shear_out[slot] = shear_force.to(device=self._device, dtype=torch.float32)
            return depth_out, normal_out, shear_out

        if update:
            for sensor in self._tactile_sensors:
                sensor.update(dt=float(self._cfg.physics_dt))

        self._ensure_tacsl_prev_buffers(len(self._tactile_sensors))

        for i, sensor in enumerate(self._tactile_sensors):
            result = self._compute_tacsl_force_field_for_sensor(i, sensor)
            if result is None:
                continue
            depth, normal_force, shear_force = result
            slot = self._sensor_slot_order[i] if i < len(self._sensor_slot_order) else i
            if 0 <= slot < depth_out.shape[0]:
                    depth_out[slot] = depth.to(device=self._device, dtype=torch.float32)
                    normal_out[slot] = normal_force.to(device=self._device, dtype=torch.float32)
                    shear_out[slot] = shear_force.to(device=self._device, dtype=torch.float32)

        return depth_out, normal_out, shear_out

    def _ensure_tacsl_prev_buffers(self, count: int) -> None:
        while len(self._tacsl_prev_tactile_points_w) < count:
            self._tacsl_prev_tactile_points_w.append(None)
        while len(self._tacsl_prev_closest_points_w) < count:
            self._tacsl_prev_closest_points_w.append(None)
        while len(self._tacsl_prev_valid_masks) < count:
            self._tacsl_prev_valid_masks.append(None)
        while len(self._tacsl_prev_contact_pos_l) < count:
            self._tacsl_prev_contact_pos_l.append(None)
        while len(self._tacsl_prev_contact_quat_l) < count:
            self._tacsl_prev_contact_quat_l.append(None)

    def _compute_tacsl_force_field_from_link_surface(
        self,
        sensor_index: int,
        surface_sensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        rows = int(getattr(self._cfg, "tacmap_link_surface_height", self._tacmap_rows))
        cols = int(getattr(self._cfg, "tacmap_link_surface_width", self._tacmap_cols))
        num_points = rows * cols
        if num_points <= 0:
            return None

        # Keep the baseline surface points current in world frame without re-raycasting
        # the rigid touch-link surface.
        self._update_tacmap_surface_sensor(sensor_index)
        if not hasattr(surface_sensor, "second_ray_hits_w") or not hasattr(surface_sensor, "second_ray_hit_valid"):
            return None

        surface_points_w = surface_sensor.second_ray_hits_w[0].detach()
        valid_mask = surface_sensor.second_ray_hit_valid[0].detach()
        if surface_points_w.shape[0] != num_points or valid_mask.shape[0] != num_points:
            return None

        views = self._ensure_tacsl_sdf_views(
            sensor_index,
            num_query_points=num_points,
            elastomer_body_path=str(getattr(surface_sensor.cfg, "prim_path", "")),
        )
        if views is None:
            return None
        sdf_view, body_view, elastomer_body_view, contact_com_b, elastomer_com_b = views

        contact_object_pos_w, contact_object_quat_xyzw = body_view.get_transforms().to(self._device).split([3, 4], dim=-1)
        contact_object_quat_w = self._math_utils.convert_quat(contact_object_quat_xyzw, to="wxyz")
        elastomer_pos_w, elastomer_quat_xyzw = elastomer_body_view.get_transforms().to(self._device).split([3, 4], dim=-1)
        elastomer_quat_w = self._math_utils.convert_quat(elastomer_quat_xyzw, to="wxyz")

        points_w = surface_points_w.unsqueeze(0)
        valid = valid_mask.to(device=points_w.device, dtype=torch.bool).reshape(-1)
        points_contact_object_local = self._points_world_to_contact_object_local(
            points_w,
            contact_object_pos_w,
            contact_object_quat_w,
        )
        ray_depth, ray_closest_points_w, ray_valid = self._link_surface_ray_penetration_tensors(
            sensor_index,
            points_w.device,
        )

        eps = max(float(getattr(self._cfg, "tacsl_sdf_gradient_eps", 1.0e-4)), 1.0e-8)
        try:
            sdf_values, sdf_gradients = self._query_sdf_values_and_finite_difference_gradients(
                sdf_view,
                points_contact_object_local,
                eps,
            )
        except Exception as exc:
            if sensor_index not in self._tacsl_warned_sdf_views:
                print(f"[WARN] TacSL SDF query failed for sensor {sensor_index}: {exc}", flush=True)
                self._tacsl_warned_sdf_views.add(sensor_index)
            return None

        sdf_values = torch.nan_to_num(sdf_values[0], nan=float("inf"), posinf=float("inf"), neginf=-float("inf"))
        sdf_gradients = torch.nan_to_num(sdf_gradients[0], nan=0.0, posinf=0.0, neginf=0.0)
        normals_local = torch.nn.functional.normalize(sdf_gradients, dim=-1, eps=1.0e-8)

        quat_expanded = contact_object_quat_w[:1].unsqueeze(1).expand(-1, num_points, 4)
        surface_normal_w = self._link_surface_normal_w(elastomer_quat_w)
        normals_w = surface_normal_w.unsqueeze(0).expand(num_points, 3)

        sdf_depth = (-sdf_values).clamp_min(0.0)
        depth = torch.maximum(sdf_depth, ray_depth)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        sdf_for_closest = torch.where(torch.isfinite(sdf_values), sdf_values, torch.zeros_like(sdf_values))
        closest_points_local = points_contact_object_local[0] - sdf_for_closest.unsqueeze(-1) * normals_local
        closest_points_w = self._math_utils.quat_apply(quat_expanded, closest_points_local.unsqueeze(0)).squeeze(0)
        closest_points_w = closest_points_w + contact_object_pos_w[:1]
        use_ray_closest = ray_valid & (depth > 0.0)
        closest_points_w = torch.where(use_ray_closest.unsqueeze(-1), ray_closest_points_w, closest_points_w)

        relative_velocity_w = self._compute_tacsl_link_frame_relative_velocity(
            sensor_index=sensor_index,
            closest_points_w=closest_points_w,
            contact_object_pos_w=contact_object_pos_w,
            contact_object_quat_w=contact_object_quat_w,
            elastomer_pos_w=elastomer_pos_w,
            elastomer_quat_w=elastomer_quat_w,
        )
        vt_w = relative_velocity_w - normals_w * torch.sum(normals_w * relative_velocity_w, dim=-1, keepdim=True)
        vt_w = torch.where(valid.unsqueeze(-1), vt_w, torch.zeros_like(vt_w))
        vt_norm = torch.linalg.norm(vt_w, dim=-1)

        fc_norm = float(self._cfg.tacsl_normal_contact_stiffness) * depth
        fc_world = fc_norm.unsqueeze(-1) * normals_w
        ft_static_norm = float(self._cfg.tacsl_tangential_stiffness) * vt_norm
        ft_dynamic_norm = float(self._cfg.tacsl_friction_coefficient) * fc_norm
        ft_norm = torch.minimum(ft_static_norm, ft_dynamic_norm)
        ft_world = -ft_norm.unsqueeze(-1) * vt_w / vt_norm.unsqueeze(-1).clamp_min(1.0e-9)

        contact_mask = (depth > 0.0) & valid
        normal_force = torch.where(contact_mask, fc_norm, torch.zeros_like(fc_norm))
        shear_force = self._project_tacsl_shear_to_link_grid(
            ft_world,
            elastomer_quat_w,
            contact_mask,
        )

        return (
            depth.reshape(rows, cols).detach().to(device=self._device, dtype=torch.float32),
            normal_force.reshape(rows, cols).detach().to(device=self._device, dtype=torch.float32),
            shear_force.reshape(rows, cols, 2).detach().to(device=self._device, dtype=torch.float32),
        )

    def _query_sdf_values_and_finite_difference_gradients(
        self,
        sdf_view,
        points_local: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query scalar SDF and compute gradients with central differences.

        SdfShapeView exposes values and gradients in one API, but for the TacSL
        link-surface path we only trust the scalar SDF value and build the
        gradient explicitly from SDF(p + eps) - SDF(p - eps).
        """
        points_local = points_local.to(device=self._device, dtype=torch.float32)
        sdf_values = self._query_sdf_values_from_view(sdf_view, points_local)
        basis = torch.eye(3, device=points_local.device, dtype=points_local.dtype)
        gradients = []
        with torch.no_grad():
            for axis in range(3):
                offset = basis[axis].view(1, 1, 3) * float(eps)
                sdf_plus = self._query_sdf_values_from_view(sdf_view, points_local + offset)
                sdf_minus = self._query_sdf_values_from_view(sdf_view, points_local - offset)
                gradients.append((sdf_plus - sdf_minus) / (2.0 * float(eps)))
        return sdf_values, torch.stack(gradients, dim=-1)

    def _query_sdf_values_from_view(self, sdf_view, points_local: torch.Tensor) -> torch.Tensor:
        sdf_values_and_gradients = sdf_view.get_sdf_and_gradients(points_local)
        sdf_values_and_gradients = sdf_values_and_gradients.to(self._device)
        return sdf_values_and_gradients[..., -1]

    def _compute_tacsl_link_frame_relative_velocity(
        self,
        *,
        sensor_index: int,
        closest_points_w: torch.Tensor,
        contact_object_pos_w: torch.Tensor,
        contact_object_quat_w: torch.Tensor,
        elastomer_pos_w: torch.Tensor,
        elastomer_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        dt = max(float(getattr(self._cfg, "physics_dt", 0.0)), 1.0e-8)
        quat_inv = self._math_utils.quat_inv(elastomer_quat_w[:1].to(self._device))
        contact_pos_l = self._math_utils.quat_apply(
            quat_inv,
            contact_object_pos_w[:1].to(self._device) - elastomer_pos_w[:1].to(self._device),
        ).squeeze(0)
        contact_quat_l = self._math_utils.quat_mul(quat_inv, contact_object_quat_w[:1].to(self._device)).squeeze(0)
        contact_quat_l = contact_quat_l / torch.linalg.norm(contact_quat_l).clamp_min(1.0e-8)

        quat_expanded = quat_inv.expand(closest_points_w.shape[0], 4)
        closest_points_l = self._math_utils.quat_apply(
            quat_expanded,
            closest_points_w.to(self._device) - elastomer_pos_w[:1].to(self._device),
        )

        prev_pos_l = self._tacsl_prev_contact_pos_l[sensor_index]
        prev_quat_l = self._tacsl_prev_contact_quat_l[sensor_index]
        if prev_pos_l is None or prev_quat_l is None or prev_pos_l.shape != contact_pos_l.shape:
            relative_velocity_l = torch.zeros_like(closest_points_l)
        else:
            contact_linvel_l = (contact_pos_l - prev_pos_l.to(self._device)) / dt
            contact_angvel_l = self._angular_velocity_from_quat_delta(
                prev_quat_l.to(self._device),
                contact_quat_l,
                dt,
            )
            closest_relative_l = closest_points_l - contact_pos_l.unsqueeze(0)
            closest_velocity_l = contact_linvel_l.unsqueeze(0) + torch.linalg.cross(
                contact_angvel_l.unsqueeze(0).expand_as(closest_relative_l),
                closest_relative_l,
                dim=-1,
            )
            relative_velocity_l = -closest_velocity_l

        self._tacsl_prev_contact_pos_l[sensor_index] = contact_pos_l.detach().clone()
        self._tacsl_prev_contact_quat_l[sensor_index] = contact_quat_l.detach().clone()
        return self._math_utils.quat_apply(
            elastomer_quat_w[:1].to(self._device).expand(relative_velocity_l.shape[0], 4),
            relative_velocity_l,
        )

    def _compute_tacsl_rigid_body_point_velocities(
        self,
        *,
        surface_points_w: torch.Tensor,
        closest_points_w: torch.Tensor,
        contact_body_view,
        contact_object_pos_w: torch.Tensor,
        contact_object_quat_w: torch.Tensor,
        contact_com_b: torch.Tensor,
        elastomer_body_view,
        elastomer_pos_w: torch.Tensor,
        elastomer_quat_w: torch.Tensor,
        elastomer_com_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute point velocities from rigid body linear/angular velocities, like TacSL."""
        contact_velocities = contact_body_view.get_velocities().to(self._device)
        contact_linvel_w_com = contact_velocities[:1, :3]
        contact_angvel_w = contact_velocities[:1, 3:]

        elastomer_velocities = elastomer_body_view.get_velocities().to(self._device)
        elastomer_linvel_w_com = elastomer_velocities[:1, :3]
        elastomer_angvel_w = elastomer_velocities[:1, 3:]

        contact_com_w_offset = self._math_utils.quat_apply(contact_object_quat_w[:1], contact_com_b[:1])
        contact_linvel_w = contact_linvel_w_com - torch.linalg.cross(
            contact_angvel_w,
            contact_com_w_offset,
            dim=-1,
        )

        elastomer_com_w_offset = self._math_utils.quat_apply(elastomer_quat_w[:1], elastomer_com_b[:1])
        elastomer_linvel_w = elastomer_linvel_w_com - torch.linalg.cross(
            elastomer_angvel_w,
            elastomer_com_w_offset,
            dim=-1,
        )

        tactile_relative_w = surface_points_w - elastomer_pos_w[:1]
        tactile_velocity_w = (
            torch.linalg.cross(
                elastomer_angvel_w.expand_as(tactile_relative_w),
                tactile_relative_w,
                dim=-1,
            )
            + elastomer_linvel_w
        )

        closest_relative_w = closest_points_w - contact_object_pos_w[:1]
        closest_velocity_w = (
            torch.linalg.cross(
                contact_angvel_w.expand_as(closest_relative_w),
                closest_relative_w,
                dim=-1,
            )
            + contact_linvel_w
        )
        return tactile_velocity_w, closest_velocity_w

    def _link_surface_ray_penetration_tensors(
        self,
        sensor_index: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = int(getattr(self._cfg, "tacmap_link_surface_height", self._tacmap_rows))
        cols = int(getattr(self._cfg, "tacmap_link_surface_width", self._tacmap_cols))
        num_points = rows * cols
        zeros = torch.zeros(num_points, device=device, dtype=torch.float32)
        zero_points = torch.zeros(num_points, 3, device=device, dtype=torch.float32)
        false_mask = torch.zeros(num_points, device=device, dtype=torch.bool)

        if sensor_index >= len(self._tacmap_sensors):
            return zeros, zero_points, false_mask

        penetration_entry = self._link_surface_penetration_entry(sensor_index)
        if penetration_entry is None:
            return zeros, zero_points, false_mask

        depth_np, valid_np = penetration_entry
        depth_np = np.asarray(depth_np, dtype=np.float32).reshape(-1)
        valid_np = np.asarray(valid_np, dtype=bool).reshape(-1)
        if depth_np.size != num_points or valid_np.size != num_points:
            return zeros, zero_points, false_mask

        depth = torch.as_tensor(depth_np, device=device, dtype=torch.float32)
        valid = torch.as_tensor(valid_np, device=device, dtype=torch.bool)

        object_sensor = self._tacmap_sensors[sensor_index]
        hits = getattr(object_sensor, "ray_hits_w", None)
        if hits is None or hits.shape[1] != num_points:
            return depth, zero_points, valid
        closest_points = hits[0].detach().to(device=device, dtype=torch.float32)
        closest_points = torch.where(valid.unsqueeze(-1), closest_points, zero_points)
        return depth, closest_points, valid

    def _ensure_tacsl_sdf_views(
        self,
        sensor_index: int,
        num_query_points: int,
        elastomer_body_path: str | None = None,
    ):
        while len(self._tacsl_sdf_views) <= sensor_index:
            self._tacsl_sdf_views.append(None)
            self._tacsl_body_views.append(None)
            self._tacsl_elastomer_body_views.append(None)
            self._tacsl_contact_com_b.append(None)
            self._tacsl_elastomer_com_b.append(None)
            self._tacsl_sdf_view_query_counts.append(None)

        if (
            self._tacsl_sdf_views[sensor_index] is not None
            and self._tacsl_body_views[sensor_index] is not None
            and self._tacsl_elastomer_body_views[sensor_index] is not None
            and self._tacsl_contact_com_b[sensor_index] is not None
            and self._tacsl_elastomer_com_b[sensor_index] is not None
            and self._tacsl_sdf_view_query_counts[sensor_index] == int(num_query_points)
        ):
            return (
                self._tacsl_sdf_views[sensor_index],
                self._tacsl_body_views[sensor_index],
                self._tacsl_elastomer_body_views[sensor_index],
                self._tacsl_contact_com_b[sensor_index],
                self._tacsl_elastomer_com_b[sensor_index],
            )

        query_paths = getattr(self, "_per_sensor_target_query_paths", [])
        if sensor_index >= len(query_paths):
            return None
        mesh_path = str(query_paths[sensor_index])
        body_path = self._target_body_path_for_query(mesh_path)
        if not mesh_path or not body_path or not elastomer_body_path:
            return None

        try:
            from isaacsim.core.simulation_manager import SimulationManager

            if self._tacsl_physics_sim_view is None:
                self._tacsl_physics_sim_view = SimulationManager.get_physics_sim_view()
            mesh_path_pattern = mesh_path.replace("env_0", "env_*")
            body_path_pattern = body_path.replace("env_0", "env_*")
            elastomer_body_path_pattern = str(elastomer_body_path).replace("env_0", "env_*")
            sdf_view = self._tacsl_physics_sim_view.create_sdf_shape_view(mesh_path_pattern, int(num_query_points))
            body_view = self._tacsl_physics_sim_view.create_rigid_body_view([body_path_pattern])
            elastomer_body_view = self._tacsl_physics_sim_view.create_rigid_body_view([elastomer_body_path_pattern])
            contact_com_b = body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]
            elastomer_com_b = elastomer_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]
        except Exception as exc:
            if sensor_index not in self._tacsl_warned_sdf_views:
                print(
                    f"[WARN] TacSL SDF view init failed for sensor {sensor_index}: mesh={mesh_path}, "
                    f"body={body_path}, error={exc}",
                    flush=True,
                )
                self._tacsl_warned_sdf_views.add(sensor_index)
            return None

        self._tacsl_sdf_views[sensor_index] = sdf_view
        self._tacsl_body_views[sensor_index] = body_view
        self._tacsl_elastomer_body_views[sensor_index] = elastomer_body_view
        self._tacsl_contact_com_b[sensor_index] = contact_com_b
        self._tacsl_elastomer_com_b[sensor_index] = elastomer_com_b
        self._tacsl_sdf_view_query_counts[sensor_index] = int(num_query_points)
        return sdf_view, body_view, elastomer_body_view, contact_com_b, elastomer_com_b

    def _target_body_path_for_query(self, query_path: str) -> str | None:
        stage = getattr(self, "_stage", None)
        prim = stage.GetPrimAtPath(query_path) if stage is not None else None
        if prim and prim.IsValid():
            curr = prim
            while curr.IsValid() and not curr.IsPseudoRoot():
                try:
                    if curr.HasAPI(self._UsdPhysics.RigidBodyAPI) or curr.HasAPI(self._UsdPhysics.MassAPI):
                        return curr.GetPath().pathString
                except Exception:
                    break
                curr = curr.GetParent()

        for candidate in ("/World/Plug", "/World/Socket", str(getattr(self._cfg, "robot_prim_path", "/World/Robot"))):
            if str(query_path).startswith(candidate):
                return candidate
        parts = str(query_path).split("/")
        if len(parts) >= 3:
            return "/" + "/".join(parts[1:3])
        return None

    def _points_world_to_contact_object_local(
        self,
        points_w: torch.Tensor,
        contact_object_pos_w: torch.Tensor,
        contact_object_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        num_points = points_w.shape[1]
        quat_inv = self._math_utils.quat_inv(contact_object_quat_w)
        quat_expanded = quat_inv.unsqueeze(1).expand(-1, num_points, 4)
        rel_points = points_w - contact_object_pos_w.unsqueeze(1)
        return self._math_utils.quat_apply(quat_expanded, rel_points)

    def _compute_tacsl_force_field_for_sensor(self, sensor_index: int, sensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        data = sensor.data.tactile_points_w_per_sensor
        sdf_all = getattr(sensor, "_sdf_out", None)
        if data is None or sdf_all is None:
            return None
        if data.shape[1] < 1 or sdf_all.shape[1] < 1:
            return None

        points_w = data[0, 0, :, :3].detach()
        sdf = sdf_all[0, 0, :].detach()
        if points_w.numel() == 0 or sdf.numel() == 0:
            return None

        rows = int(self._cfg.num_rows)
        cols = int(self._cfg.num_cols)
        if points_w.shape[0] != rows * cols:
            return None

        gradients_w = self._query_sdf_gradient_world(sensor, points_w)
        if gradients_w is None:
            return None
        normals_w = torch.nn.functional.normalize(gradients_w, dim=-1, eps=1.0e-8)

        use_mesh = getattr(sensor, "_target_mesh_prim_path", None) is not None
        if use_mesh and not bool(getattr(sensor.cfg, "mesh_use_signed_distance", False)):
            shell = float(getattr(sensor.cfg, "mesh_shell_thickness", 0.001))
            depth = (shell - sdf).clamp_min(0.0)
        else:
            depth = (-sdf).clamp_min(0.0)

        closest_points_w = points_w - sdf.unsqueeze(-1) * normals_w
        dt = max(float(getattr(self._cfg, "physics_dt", 0.0)), 1.0e-8)
        prev_points = self._tacsl_prev_tactile_points_w[sensor_index]
        prev_closest = self._tacsl_prev_closest_points_w[sensor_index]
        if prev_points is None or prev_closest is None or prev_points.shape != points_w.shape:
            tactile_velocity_w = torch.zeros_like(points_w)
            closest_velocity_w = torch.zeros_like(points_w)
        else:
            tactile_velocity_w = (points_w - prev_points.to(points_w.device)) / dt
            closest_velocity_w = (closest_points_w - prev_closest.to(points_w.device)) / dt

        self._tacsl_prev_tactile_points_w[sensor_index] = points_w.detach().clone()
        self._tacsl_prev_closest_points_w[sensor_index] = closest_points_w.detach().clone()

        relative_velocity_w = tactile_velocity_w - closest_velocity_w
        vt_w = relative_velocity_w - normals_w * torch.sum(normals_w * relative_velocity_w, dim=-1, keepdim=True)
        vt_norm = torch.linalg.norm(vt_w, dim=-1)

        fc_norm = float(self._cfg.tacsl_normal_contact_stiffness) * depth
        fc_world = fc_norm.unsqueeze(-1) * normals_w
        ft_static_norm = float(self._cfg.tacsl_tangential_stiffness) * vt_norm
        ft_dynamic_norm = float(self._cfg.tacsl_friction_coefficient) * fc_norm
        ft_norm = torch.minimum(ft_static_norm, ft_dynamic_norm)
        ft_world = -ft_norm.unsqueeze(-1) * vt_w / vt_norm.unsqueeze(-1).clamp_min(1.0e-9)

        row_axis_w, col_axis_w = self._estimate_tactile_grid_axes(points_w, rows, cols)
        contact_mask = depth > 0.0
        normal_force, shear_force = self._project_tacsl_force_to_grid(
            fc_world,
            ft_world,
            row_axis_w,
            col_axis_w,
            contact_mask,
        )
        normal_force = torch.where(contact_mask, fc_norm, torch.zeros_like(fc_norm))

        return (
            depth.reshape(rows, cols).detach().to(device=self._device, dtype=torch.float32),
            normal_force.reshape(rows, cols).detach().to(device=self._device, dtype=torch.float32),
            shear_force.reshape(rows, cols, 2).detach().to(device=self._device, dtype=torch.float32),
        )

    def _query_sdf_gradient_world(self, sensor, points_w: torch.Tensor) -> torch.Tensor | None:
        query_fn = getattr(sensor, "query_sdf_world", None)
        if not callable(query_fn):
            return None

        eps = max(float(getattr(self._cfg, "tacsl_sdf_gradient_eps", 1.0e-4)), 1.0e-8)
        basis = torch.eye(3, device=points_w.device, dtype=points_w.dtype)
        grads = []
        with torch.no_grad():
            for axis in range(3):
                offset = basis[axis].unsqueeze(0) * eps
                sdf_plus = query_fn((points_w + offset).unsqueeze(0))[0]
                sdf_minus = query_fn((points_w - offset).unsqueeze(0))[0]
                grads.append((sdf_plus - sdf_minus) / (2.0 * eps))
        return torch.stack(grads, dim=-1)

    @staticmethod
    def _estimate_tactile_grid_axes(points_w: torch.Tensor, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        grid = points_w.reshape(rows, cols, 3)
        if rows > 1:
            row_axis = torch.mean(grid[1:, :, :] - grid[:-1, :, :], dim=(0, 1))
        else:
            row_axis = torch.tensor((1.0, 0.0, 0.0), device=points_w.device, dtype=points_w.dtype)
        if cols > 1:
            col_axis = torch.mean(grid[:, 1:, :] - grid[:, :-1, :], dim=(0, 1))
        else:
            col_axis = torch.tensor((0.0, 1.0, 0.0), device=points_w.device, dtype=points_w.dtype)

        row_axis = row_axis / torch.linalg.norm(row_axis).clamp_min(1.0e-8)
        col_axis = col_axis - row_axis * torch.sum(row_axis * col_axis)
        col_axis = col_axis / torch.linalg.norm(col_axis).clamp_min(1.0e-8)
        return row_axis, col_axis

    def _link_surface_tactile_grid_axes_w(self, elastomer_quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the display row/column axes from the link-surface grid frame."""
        row_axis_l = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_grid_v_axis", "+z")))
        col_axis_l = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_grid_u_axis", "+y")))
        quat_w = elastomer_quat_w[:1].to(self._device)
        row_axis_w = self._math_utils.quat_apply(quat_w, row_axis_l.unsqueeze(0)).squeeze(0)
        col_axis_w = self._math_utils.quat_apply(quat_w, col_axis_l.unsqueeze(0)).squeeze(0)
        row_axis_w = row_axis_w / torch.linalg.norm(row_axis_w).clamp_min(1.0e-8)
        col_axis_w = col_axis_w - row_axis_w * torch.sum(row_axis_w * col_axis_w)
        col_axis_w = col_axis_w / torch.linalg.norm(col_axis_w).clamp_min(1.0e-8)
        return row_axis_w, col_axis_w

    def _link_surface_normal_w(self, elastomer_quat_w: torch.Tensor) -> torch.Tensor:
        direction = getattr(self._cfg, "tacmap_link_surface_ray_direction", None)
        if direction is None:
            normal_l = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_ray_axis", "+x")))
        else:
            normal_l = torch.tensor(tuple(float(v) for v in direction), dtype=torch.float32, device=self._device)
        normal_l = normal_l / torch.linalg.norm(normal_l).clamp_min(1.0e-8)
        normal_w = self._math_utils.quat_apply(elastomer_quat_w[:1].to(self._device), normal_l.unsqueeze(0)).squeeze(0)
        return normal_w / torch.linalg.norm(normal_w).clamp_min(1.0e-8)

    def _project_tacsl_shear_to_link_grid(
        self,
        shear_force_w: torch.Tensor,
        elastomer_quat_w: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> torch.Tensor:
        quat_inv = self._math_utils.quat_inv(elastomer_quat_w[:1].to(self._device))
        quat_expanded = quat_inv.expand(shear_force_w.shape[0], 4)
        shear_force_l = self._math_utils.quat_apply(quat_expanded, shear_force_w)
        row_axis_l = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_grid_v_axis", "+z")))
        col_axis_l = self._local_axis_tensor(str(getattr(self._cfg, "tacmap_link_surface_grid_u_axis", "+y")))
        shear_row = torch.sum(shear_force_l * row_axis_l.unsqueeze(0), dim=-1)
        shear_col = torch.sum(shear_force_l * col_axis_l.unsqueeze(0), dim=-1)
        return torch.stack(
            (
                torch.where(contact_mask, shear_row, torch.zeros_like(shear_row)),
                torch.where(contact_mask, shear_col, torch.zeros_like(shear_col)),
            ),
            dim=-1,
        )

    @staticmethod
    def _project_tacsl_force_to_local_grid(
        normal_force_w: torch.Tensor,
        shear_force_w: torch.Tensor,
        row_axes_w: torch.Tensor,
        col_axes_w: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normal_axes_w = torch.linalg.cross(col_axes_w, row_axes_w, dim=-1)
        normal_axes_w = normal_axes_w / torch.linalg.norm(normal_axes_w, dim=-1, keepdim=True).clamp_min(1.0e-8)

        if torch.any(contact_mask):
            normal_sign = torch.mean(torch.sum(normal_force_w[contact_mask] * normal_axes_w[contact_mask], dim=-1))
            if bool((normal_sign < 0.0).detach().cpu().item()):
                normal_axes_w = -normal_axes_w

        normal_component = torch.linalg.norm(normal_force_w, dim=-1)
        shear_row = torch.sum(shear_force_w * row_axes_w, dim=-1)
        shear_col = torch.sum(shear_force_w * col_axes_w, dim=-1)

        normal_force = torch.where(contact_mask, normal_component, torch.zeros_like(normal_component))
        shear_force = torch.stack(
            (
                torch.where(contact_mask, shear_row, torch.zeros_like(shear_row)),
                torch.where(contact_mask, shear_col, torch.zeros_like(shear_col)),
            ),
            dim=-1,
        )
        return normal_force, shear_force

    @staticmethod
    def _project_tacsl_force_to_grid(
        normal_force_w: torch.Tensor,
        shear_force_w: torch.Tensor,
        row_axis_w: torch.Tensor,
        col_axis_w: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_axes_w = row_axis_w.unsqueeze(0).expand_as(normal_force_w)
        col_axes_w = col_axis_w.unsqueeze(0).expand_as(normal_force_w)
        return IntegratedTactileEnv._project_tacsl_force_to_local_grid(
            normal_force_w,
            shear_force_w,
            row_axes_w,
            col_axes_w,
            contact_mask,
        )

    def _build_spaces(self, cfg: IntegratedTactileEnvCfg) -> None:
        super()._build_spaces(cfg)
        pressure_dim = len(RL_PRESSURE_PAD_LINK_ORDER) * max(1, int(cfg.num_rows)) * max(1, int(cfg.num_cols))
        tacmap_dim = len(RL_FINGER_ORDER) * max(0, int(self._tacmap_rows)) * max(0, int(self._tacmap_cols))
        hydroshear_dim = len(RL_FINGER_ORDER) * RL_HYDROSHEAR_MARKER_ROWS * RL_HYDROSHEAR_MARKER_COLS * 3

        def flat_box(dim: int) -> gymnasium.spaces.Box:
            return gymnasium.spaces.Box(-np.inf, np.inf, shape=(max(0, int(dim)),), dtype=np.float32)

        self.observation_space.spaces["tacmap"] = gymnasium.spaces.Box(
            0,
            255,
            shape=self._tacmap_output_shape(),
            dtype=np.uint8,
        )
        self.observation_space.spaces["tacmap_raw"] = gymnasium.spaces.Box(
            0.0,
            float(cfg.tacmap_max_distance),
            shape=self._tacmap_output_shape(),
            dtype=np.float32,
        )
        self.observation_space.spaces["tacmap_object_raw"] = gymnasium.spaces.Box(
            0.0,
            float(cfg.tacmap_max_distance),
            shape=self._tacmap_output_shape(),
            dtype=np.float32,
        )
        self.observation_space.spaces["tacmap_surface_raw"] = gymnasium.spaces.Box(
            0.0,
            float(cfg.tacmap_max_distance),
            shape=self._tacmap_output_shape(),
            dtype=np.float32,
        )
        self.observation_space.spaces["tacmap_surface_points_w"] = gymnasium.spaces.Box(
            -np.inf,
            np.inf,
            shape=(*self._tacmap_output_shape(), 3),
            dtype=np.float32,
        )
        self.observation_space.spaces["tacmap_surface_normals_w"] = gymnasium.spaces.Box(
            -1.0,
            1.0,
            shape=(*self._tacmap_output_shape(), 3),
            dtype=np.float32,
        )
        self.observation_space.spaces["tacmap_surface_valid"] = gymnasium.spaces.Box(
            0,
            1,
            shape=self._tacmap_output_shape(),
            dtype=np.uint8,
        )
        self.observation_space.spaces["tacmap_object_points_w"] = gymnasium.spaces.Box(
            -np.inf,
            np.inf,
            shape=(*self._tacmap_output_shape(), 3),
            dtype=np.float32,
        )
        self.observation_space.spaces["tacmap_object_valid"] = gymnasium.spaces.Box(
            0,
            1,
            shape=self._tacmap_output_shape(),
            dtype=np.uint8,
        )
        self.observation_space.spaces["fots_theta"] = gymnasium.spaces.Box(
            -np.inf,
            np.inf,
            shape=(self._tacmap_output_shape()[0],),
            dtype=np.float32,
        )
        self.observation_space.spaces["tacsl_penetration_depth"] = gymnasium.spaces.Box(
            0.0,
            np.inf,
            shape=self._tacsl_force_output_shape(),
            dtype=np.float32,
        )
        self.observation_space.spaces["tacsl_normal_force"] = gymnasium.spaces.Box(
            0.0,
            np.inf,
            shape=self._tacsl_force_output_shape(),
            dtype=np.float32,
        )
        self.observation_space.spaces["tacsl_shear_force"] = gymnasium.spaces.Box(
            -np.inf,
            np.inf,
            shape=(*self._tacsl_force_output_shape(), 2),
            dtype=np.float32,
        )
        self.observation_space.spaces["rl_warpsdf_pressure_obs"] = flat_box(pressure_dim)
        self.observation_space.spaces["rl_tacmap_obs"] = flat_box(tacmap_dim)
        self.observation_space.spaces["rl_hydroshear_obs"] = flat_box(hydroshear_dim)
        self.observation_space.spaces["rl_tactile_obs"] = flat_box(pressure_dim + tacmap_dim + hydroshear_dim)

    def close(self):
        self._tacmap_sensors.clear()
        self._tacmap_surface_sensors.clear()
        self._tacmap_surface_raw_cache.clear()
        self._tacmap_penetration_cache.clear()
        self._tacsl_prev_tactile_points_w.clear()
        self._tacsl_prev_closest_points_w.clear()
        self._tacsl_prev_valid_masks.clear()
        self._tacsl_prev_contact_pos_l.clear()
        self._tacsl_prev_contact_quat_l.clear()
        self._tacsl_sdf_views.clear()
        self._tacsl_body_views.clear()
        self._tacsl_elastomer_body_views.clear()
        self._tacsl_contact_com_b.clear()
        self._tacsl_elastomer_com_b.clear()
        self._tacsl_sdf_view_query_counts.clear()
        self._tacsl_warned_sdf_views.clear()
        self._tacsl_physics_sim_view = None
        return super().close()
