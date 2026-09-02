"""Standalone gymnasium.Env for robot replay with WarpSdf tactile sensors.

Accepts configured joint position targets, returns tactile force grids,
joint states, object poses, and RGB renders.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium
import numpy as np
import torch
from utils.utlis_misc import _parse_elastomer_origins, _infer_arm, _infer_finger, _sensor_slot
from utils.utils import _xyzw_to_wxyz, _look_at_quat, _resolve_joint_ids
from aloha_tactile_cfg import AlohaTactileEnvCfg, DATASET_JOINT_ORDER, TrackInfo


REPO_ROOT = Path(__file__).resolve().parents[3]
EXT_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))


def _short_link_label(link_path: str) -> str:
    label = Path(str(link_path)).name
    return label or str(link_path)


def pressure_sensor_labels_from_env(env, sensor_count: int, fallback_label: str = "sensor") -> list[str]:
    """Return display labels tied to actual tactile links, not legacy CLI fingers."""

    count = max(0, int(sensor_count))
    labels = [f"S{i}" for i in range(count)]
    links = list(getattr(env, "_selected_links", []) or [])
    slots = list(getattr(env, "_sensor_slot_order", []) or [])
    if not links:
        if labels:
            labels[0] = str(fallback_label)
        return labels

    for sensor_idx, link in enumerate(links):
        try:
            slot = int(slots[sensor_idx]) if sensor_idx < len(slots) else int(sensor_idx)
        except (TypeError, ValueError):
            slot = int(sensor_idx)
        target_idx = slot if 0 <= slot < count else sensor_idx
        if 0 <= target_idx < count:
            labels[target_idx] = _short_link_label(str(link))
    return labels


def _pose_xyz_np(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


def _unit_vec3_np(value) -> np.ndarray | None:
    vec = _pose_xyz_np(value)
    if vec is None:
        return None
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-9:
        return None
    return vec / norm


def _press_axis_w_np(env) -> tuple[np.ndarray | None, str]:
    axis = _unit_vec3_np(getattr(env, "_press_debug_axis_w", None))
    if axis is not None:
        return axis, str(getattr(env, "_press_debug_axis_source", "debug"))
    axis_fn = getattr(env, "_press_object_axis_w", None)
    robot = getattr(env, "_robot", None)
    body_idx = getattr(env, "_press_touch_body_idx", None)
    if callable(axis_fn) and robot is not None and body_idx is not None:
        try:
            quat_w = robot.data.body_link_state_w[0, int(body_idx), 3:7]
            axis = _unit_vec3_np(axis_fn(quat_w))
        except Exception:
            axis = None
        if axis is not None:
            return axis, "press_axis"
    return None, "delta_norm"


def _axis_depth(delta: np.ndarray, axis_w: np.ndarray | None) -> tuple[float, float]:
    if axis_w is None:
        return float(np.linalg.norm(delta)), 0.0
    signed_depth = float(np.dot(delta, axis_w))
    lateral = delta - axis_w * signed_depth
    return signed_depth, float(np.linalg.norm(lateral))


def press_depth_summary_from_env(env, penetration_map=None, plug_pose=None) -> dict[str, object]:
    """Summarize commanded press depth and measured WarpSDF penetration."""

    penetration_arr = np.asarray(penetration_map, dtype=np.float32) if penetration_map is not None else np.asarray([])
    summary: dict[str, object] = {
        "phase": "unknown",
        "contact_detected": bool(getattr(env, "_press_contact_detected", False)),
        "sdf_penetration_max_m": float(np.nanmax(penetration_arr)) if penetration_arr.size else 0.0,
    }

    cfg = getattr(env, "_cfg", None)
    target_indent = getattr(cfg, "press_indent_depth", None) if cfg is not None else None
    if target_indent is not None:
        summary["target_indent_m"] = float(target_indent)

    current = _pose_xyz_np(getattr(env, "_press_object_pose_w", None))
    if current is None:
        current = _pose_xyz_np(plug_pose)
    contact = _pose_xyz_np(getattr(env, "_press_contact_pose_w", None))
    base = _pose_xyz_np(getattr(env, "_press_object_manual_base_pose_w", None))
    axis_w, axis_source = _press_axis_w_np(env)
    summary["axis_source"] = axis_source
    if axis_w is not None:
        summary["axis_w"] = [float(v) for v in axis_w]

    if summary["contact_detected"] and current is not None and contact is not None:
        delta = current - contact
        depth, lateral = _axis_depth(delta, axis_w)
        summary["phase"] = "indent"
        summary["cmd_indent_m"] = depth
        summary["indent_depth_m"] = depth
        summary["lateral_error_m"] = lateral
        summary["indent_delta_w_m"] = [float(v) for v in delta]
    elif current is not None and base is not None:
        delta = current - base
        depth, lateral = _axis_depth(delta, axis_w)
        summary["phase"] = "search"
        summary["search_depth_m"] = depth
        summary["lateral_error_m"] = lateral
        summary["search_delta_w_m"] = [float(v) for v in delta]
    return summary


_PRESSURE_PAD_LINK_ELEMENT_CACHE: dict[
    str,
    tuple[set[str], tuple[tuple[str, ET.Element], ...]],
] = {}


def _pressure_pad_link_elements_from_urdf(
    urdf_path: str | os.PathLike[str],
) -> tuple[set[str], tuple[tuple[str, ET.Element], ...]]:
    key = os.path.expanduser(str(urdf_path))
    cached = _PRESSURE_PAD_LINK_ELEMENT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        root = ET.parse(key).getroot()
    except Exception:
        result: tuple[set[str], tuple[tuple[str, ET.Element], ...]] = (set(), ())
        _PRESSURE_PAD_LINK_ELEMENT_CACHE[key] = result
        return result

    available: set[str] = set()
    pressure_links: list[tuple[str, ET.Element]] = []
    for link in root.findall("link"):
        link_name = str(link.attrib.get("name", "")).strip()
        if link_name:
            available.add(link_name)
        pad = link.find("pressure_pad")
        if pad is not None and link_name:
            pressure_links.append((link_name, pad))
    result = (available, tuple(pressure_links))
    _PRESSURE_PAD_LINK_ELEMENT_CACHE[key] = result
    return result


def _pressure_pad_visual_links_from_urdf(urdf_path: str | os.PathLike[str]) -> list[str]:
    """Return pressure-pad links plus matching outer rubber/tubber visual links."""

    available, pressure_pads = _pressure_pad_link_elements_from_urdf(urdf_path)
    pressure_links = [link_name for link_name, _pad in pressure_pads]

    visual_links: list[str] = []
    seen: set[str] = set()
    for link_name in sorted(set(pressure_links)):
        candidates = [link_name]
        if link_name.endswith("_touch_link"):
            prefix = link_name[: -len("_touch_link")]
            candidates.extend([f"{prefix}_rubber_link", f"{prefix}_tubber_link"])
        for candidate in candidates:
            if candidate in available and candidate not in seen:
                visual_links.append(candidate)
                seen.add(candidate)
    return visual_links


def _pressure_pad_center_links_from_urdf(urdf_path: str | os.PathLike[str]) -> list[str]:
    """Return pressure-pad links whose centers should be marked in the GUI."""

    links: list[str] = []
    _available, pressure_pads = _pressure_pad_link_elements_from_urdf(urdf_path)
    for link_name, _pad in pressure_pads:
        lowered = link_name.lower()
        if "palm" in lowered or lowered == "base_link":
            continue
        links.append(link_name)
    return sorted(set(links))


def _pressure_pad_surface_axis_by_link_from_urdf(urdf_path: str | os.PathLike[str]) -> dict[str, tuple[int, float]]:
    axis_by_link: dict[str, tuple[int, float]] = {}
    _available, pressure_pads = _pressure_pad_link_elements_from_urdf(urdf_path)
    for link_name, pad in pressure_pads:
        try:
            normal_axis = int(float(pad.attrib.get("normal_axis", 0)))
        except (TypeError, ValueError):
            normal_axis = 0
        try:
            normal_sign = float(pad.attrib.get("normal_sign", 1.0))
        except (TypeError, ValueError):
            normal_sign = 1.0
        axis_by_link[link_name] = (
            normal_axis if normal_axis in (0, 1, 2) else 0,
            1.0 if normal_sign >= 0.0 else -1.0,
        )
    return axis_by_link


def _pressure_pad_origin_by_link_from_urdf(urdf_path: str | os.PathLike[str]) -> dict[str, tuple[float, float, float]]:
    origin_by_link: dict[str, tuple[float, float, float]] = {}
    _available, pressure_pads = _pressure_pad_link_elements_from_urdf(urdf_path)
    for link_name, pad in pressure_pads:
        origin = pad.find("origin")
        if origin is None:
            continue
        values = str(origin.attrib.get("xyz", "")).split()
        if len(values) != 3:
            continue
        try:
            origin_by_link[link_name] = tuple(float(value) for value in values)
        except ValueError:
            continue
    return origin_by_link


def _pressure_pad_taxel_points_by_link_from_urdf(
    urdf_path: str | os.PathLike[str],
) -> dict[str, list[tuple[float, float, float]]]:
    """Return link-local taxel points for grid or NPY pressure-pad layouts."""

    try:
        from BrainCo_DexHand.force_map.urdf_pressure_layout import load_pressure_taxel_maps_from_urdf

        taxel_maps = load_pressure_taxel_maps_from_urdf(urdf_path)
    except (OSError, ValueError, ET.ParseError):
        return {}
    return {
        taxel_map.link_name: [tuple(float(value) for value in point) for point in np.asarray(taxel_map.points_l)]
        for taxel_map in taxel_maps
    }


def _bbox_surface_center(min_v, max_v, *, normal_axis: int, normal_sign: float) -> tuple[float, float, float]:
    center = [0.5 * (float(min_v[i]) + float(max_v[i])) for i in range(3)]
    axis = int(normal_axis) if int(normal_axis) in (0, 1, 2) else 0
    center[axis] = float(max_v[axis]) if float(normal_sign) >= 0.0 else float(min_v[axis])
    return (center[0], center[1], center[2])


def _press_touch_link_names(cfg: AlohaTactileEnvCfg) -> list[str]:
    links: list[str] = []
    for link_name in getattr(cfg, "press_touch_links", ()) or ():
        text = str(link_name).strip()
        if text and text not in links:
            links.append(text)
    if links:
        return links

    text = str(getattr(cfg, "press_touch_link", "")).strip()
    return [text] if text else []


def _select_press_elastomer_links(selected_links: list[str], press_touch_links) -> list[str]:
    """Pick requested press/tactile rigid-body prims before falling back to loose path matches."""

    if isinstance(press_touch_links, str):
        requests = [press_touch_links]
    else:
        requests = [str(value) for value in press_touch_links or ()]
    requests = [value.strip() for value in requests if str(value).strip()]
    if not requests:
        return []

    def basename(path: str) -> str:
        return str(path).rstrip("/").rsplit("/", 1)[-1]

    out: list[str] = []
    for requested in requests:
        exact = [path for path in selected_links if basename(path) == requested]
        basename_matches = [path for path in selected_links if requested in basename(path)]
        path_matches = [path for path in selected_links if requested in str(path)]
        matches = exact or basename_matches or path_matches
        if matches and matches[0] not in out:
            out.append(matches[0])
    return out



class AlohaTactileEnv(gymnasium.Env):
    """Robot tactile replay environment.

    Observations:
        tactile:     (num_sensors, num_rows, num_cols) force grids
        joint_pos:   joint positions in dataset order
        joint_vel:   joint velocities in dataset order
        plug_pose:   (7,) pos + quat_wxyz (zeros if disabled)
        socket_pose: (7,) pos + quat_wxyz (zeros if disabled)
        rgb:         (H, W, 3) uint8 (if enable_camera)

    Actions:
        Raw joint position targets in DATASET_JOINT_ORDER.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: AlohaTactileEnvCfg, simulation_app=None):
        super().__init__()
        self._cfg = cfg
        self._simulation_app = simulation_app
        self._step_count = 0
        self._warned_action_size = False
        self._dataset_joint_ids = []
        self._press_hold_joint_ids = []
        self._press_hold_joint_pos = None
        self._press_hold_joint_vel = None
        self._press_finger_joint_ids = []
        self._press_finger_joint_names = []
        self._action_dim = len(DATASET_JOINT_ORDER)
        self._tactile_sensor_count = 0
        self._press_counter = 0
        self._press_touch_body_idx = None
        self._press_touch_center_l = None
        self._press_touch_center_base_l = None
        self._press_touch_normal_l = None
        self._press_touch_normal_base_l = None
        self._press_object_rot_l = None
        self._press_object_flip_l = None
        self._press_slide_axis_l = None
        self._press_hand_axis_body_idx = None
        self._press_hand_axis_l = None
        self._manual_press_offset_m = None
        self._manual_press_slide_offset_l = None
        self._press_object_pose_w = None
        self._press_object_manual_base_pose_w = None
        self._pressure_pad_presser_obj = None
        self._pressure_pad_presser_body_idx = None
        self._pressure_pad_presser_center_l = None
        self._pressure_pad_presser_normal_l = None
        self._pressure_pad_presser_offset_l = None
        self._pressure_pad_presser_start_offset_l = None
        self._pressure_pad_presser_end_offset_l = None
        self._pressure_pad_presser_rot_l = None
        self._pressure_pad_presser_flip_l = None
        self._pressure_pad_presser_pose_w = None
        self._pressure_pad_presser_prev_pos_w = None
        self._pressure_pad_presser_prev_quat_w = None
        self._press_contact_pose_w = None
        self._press_contact_counter = None
        self._press_contact_detected = False
        self._press_debug_axis_w = None
        self._press_debug_axis_source = ""
        self._press_robot_base_pose_w = None
        self._press_prev_robot_root_pos_w = None
        self._press_manual_drag_baseline_pose_w = None
        self._warned_target_pose_update = False
        self._press_prev_object_pos_w = None
        self._press_prev_object_quat_w = None
        self._locked_press_joint_ids = None
        self._locked_press_joint_pos = None
        self._locked_press_joint_vel = None

        # Deferred imports (require SimulationApp)
        import isaacsim.core.utils.prims as prim_utils
        from isaacsim.core.api.simulation_context import SimulationContext
        from isaacsim.core.utils.extensions import enable_extension

        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, RigidObject
        from isaaclab.assets.articulation import ArticulationCfg
        from isaaclab.assets.rigid_object import RigidObjectCfg
        from BrainCo_DexHand.force_map import PressureContactEventBatch, WarpSdfTactileSensor, WarpSdfTactileSensorCfg
        from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
        from isaaclab.sim.converters import UrdfConverterCfg
        from isaaclab.sim.schemas import activate_contact_sensors

        self._prim_utils = prim_utils
        self._sim_utils = sim_utils
        self._math_utils = math_utils
        self._PressureContactEventBatch = PressureContactEventBatch
        self._ContactSensor = ContactSensor
        self._ContactSensorCfg = ContactSensorCfg
        self._physx_contact_pressure_sensors = []

        enable_extension("isaacsim.asset.importer.urdf")

        self._init_sim(SimulationContext, cfg)
        self._spawn_basic_world(sim_utils)

        # Camera
        self._setup_camera(cfg, sim_utils)

        # Parse URDF elastomer origins
        urdf_path = os.path.expanduser(cfg.urdf_path)
        self._urdf_origins = _parse_elastomer_origins(urdf_path)

        self._plug_obj, self._socket_obj = self._spawn_plug_socket(cfg, sim_utils, RigidObject, RigidObjectCfg)
        
        # Spawn robot
        out_dir = cfg.usd_output_dir or os.path.join(os.path.dirname(__file__), "output", "aloha_urdf")
        os.makedirs(out_dir, exist_ok=True)

        self._robot = self._spawn_robot(
            cfg,
            urdf_path,
            sim_utils,
            Articulation,
            ArticulationCfg,
            ImplicitActuatorCfg,
            UrdfConverterCfg,
        )
        activate_contact_sensors(cfg.robot_prim_path, threshold=0.0)
        self._apply_pressure_pad_visual_materials(cfg, urdf_path, sim_utils)
        self._apply_touch_contact_materials(cfg, sim_utils)

        # Find and sort tactile surface links.
        from pxr import PhysxSchema, UsdPhysics
        self._UsdPhysics = UsdPhysics

        elastomers = self._find_elastomer_links(cfg, sim_utils, UsdPhysics, PhysxSchema)
        selected = self._sort_elastomer_links(elastomers)
        if cfg.enable_press_motion or cfg.enable_sample_point_view:
            press_links = _press_touch_link_names(cfg)
            selected = _select_press_elastomer_links(selected, press_links)
            if not selected:
                raise RuntimeError(f"Press touch link(s) {press_links!r} were not found in tactile links.")
        self._selected_links = selected

        # Resolve per-sensor target mesh prims
        target_root_paths, target_query_paths = self._build_target_query_paths(
            cfg, selected, prim_utils, sim_utils
        )
        self._per_sensor_target_query_paths = target_query_paths

        for i, (link, root, query) in enumerate(zip(selected, target_root_paths, target_query_paths)):
            print(
                f"  [{i}] slot={_sensor_slot(link)} arm={_infer_arm(link) or '?':>5s}"
                f" elastomer={link} -> query={query}",
                flush=True,
            )


        # Compute patch offsets per elastomer and create tactile sensors
        self._tactile_sensors, self._sensor_slot_order = self._create_tactile_sensors(
            cfg,
            selected,
            target_root_paths,
            target_query_paths,
            WarpSdfTactileSensor,
            WarpSdfTactileSensorCfg,
            math_utils,
        )

        # Reset simulation (triggers sensor PLAY callbacks)
        self._post_spawn_init(cfg, sim_utils, target_query_paths)
        self._build_spaces(cfg)

    # -------------------------------------------------------------------
    # Gym interface
    # -------------------------------------------------------------------

    def sync_press_robot_hold_pose(self) -> None:
        """Keep press-only alternate URDF joints fixed during GUI setup and stepping."""

        if len(self._press_hold_joint_ids) == 0 or self._press_hold_joint_pos is None:
            return
        self._write_press_robot_joint_state(self._press_hold_joint_pos)

    def sync_press_robot_root_pose(self) -> None:
        """Keep the movable hand root fixed while the GUI setup phase is waiting."""

        if not self._cfg.enable_press_motion:
            return
        if not (self._press_motion_actor() == "hand" or bool(getattr(self._cfg, "press_setup_edit_robot_pose", False))):
            return
        if bool(getattr(self._cfg, "press_setup_edit_robot_pose", False)):
            pose = self._read_robot_usd_pose()
            if pose is not None and pose.shape == (7,) and float(np.linalg.norm(pose[3:7])) > 0.5:
                pose_t = torch.tensor(pose, dtype=torch.float32, device=self._device)
                pose_t[3:7] = pose_t[3:7] / torch.linalg.norm(pose_t[3:7]).clamp_min(1.0e-8)
                self._press_robot_base_pose_w = pose_t.detach().clone()
        if self._press_robot_base_pose_w is None:
            root_state = self._robot.data.root_state_w[0, :7].detach().clone()
            root_state[3:7] = root_state[3:7] / torch.linalg.norm(root_state[3:7]).clamp_min(1.0e-8)
            self._press_robot_base_pose_w = root_state
        self._write_press_robot_root_state(self._press_robot_base_pose_w[:3], self._press_robot_base_pose_w[3:7])

    def _write_press_robot_joint_state(self, joint_pos: torch.Tensor) -> None:
        joint_vel = (
            self._press_hold_joint_vel
            if self._press_hold_joint_vel is not None
            else torch.zeros_like(self._robot.data.default_joint_vel)
        )
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._robot.update(0.0)
        if len(self._press_hold_joint_ids) > 0:
            self._robot.set_joint_position_target(joint_pos, joint_ids=self._press_hold_joint_ids)
        else:
            self._robot.set_joint_position_target(joint_pos)
        self._robot.write_data_to_sim()

    def _press_motion_actor(self) -> str:
        actor = str(getattr(self._cfg, "press_motion_actor", "object")).lower()
        return actor if actor in {"object", "finger", "hand"} else "object"

    def _write_press_robot_root_state(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> None:
        root_state = self._robot.data.default_root_state.clone().to(device=self._device)
        if root_state.ndim == 1:
            root_state = root_state.unsqueeze(0)
        root_state[:, :3] = root_pos_w.unsqueeze(0)
        root_state[:, 3:7] = root_quat_w.unsqueeze(0)
        root_state[:, 7:] = 0.0
        if self._press_prev_robot_root_pos_w is not None:
            dt = max(float(getattr(self._cfg, "physics_dt", 0.0)), 1.0e-8)
            root_state[:, 7:10] = ((root_pos_w - self._press_prev_robot_root_pos_w) / dt).unsqueeze(0)
        self._press_prev_robot_root_pos_w = root_pos_w.detach().clone()

        if hasattr(self._robot, "write_root_pose_to_sim") and hasattr(self._robot, "write_root_velocity_to_sim"):
            self._robot.write_root_pose_to_sim(root_state[:, :7])
            self._robot.write_root_velocity_to_sim(root_state[:, 7:])
        else:
            self._robot.write_root_state_to_sim(root_state)
        self._robot.update(0.0)

    def _reset_press_robot_root_pose(self) -> None:
        root_state = self._robot.data.default_root_state.clone().to(device=self._device)
        if root_state.ndim == 2:
            root_state = root_state[0]
        quat = root_state[3:7]
        quat = quat / torch.linalg.norm(quat).clamp_min(1.0e-8)
        self._press_prev_robot_root_pos_w = None
        self._write_press_robot_root_state(root_state[:3], quat)
        self._press_robot_base_pose_w = torch.cat([root_state[:3], quat], dim=0).detach().clone()

    def _write_press_hand_pose(self) -> None:
        if self._press_touch_body_idx is None:
            return
        if self._press_touch_normal_l is None:
            return
        if self._press_robot_base_pose_w is None:
            root_state = self._robot.data.root_state_w[0, :7].detach().clone()
            root_state[3:7] = root_state[3:7] / torch.linalg.norm(root_state[3:7]).clamp_min(1.0e-8)
            self._press_robot_base_pose_w = root_state

        if str(getattr(self._cfg, "press_hand_axis_link", "")).strip().lower() == "world":
            axis_w = self._press_hand_axis_l
            if axis_w is None:
                return
        elif self._press_hand_axis_body_idx is None:
            axis_body_idx = self._press_touch_body_idx
            axis_l = self._press_touch_normal_l
            if axis_l is None:
                return
            axis_quat_w = self._robot.data.body_link_state_w[0, int(axis_body_idx), 3:7]
            axis_w = self._math_utils.quat_apply(axis_quat_w.unsqueeze(0), axis_l.unsqueeze(0)).squeeze(0)
        else:
            axis_body_idx = self._press_hand_axis_body_idx
            axis_l = self._press_hand_axis_l
            if axis_l is None:
                return
            axis_quat_w = self._robot.data.body_link_state_w[0, int(axis_body_idx), 3:7]
            axis_w = self._math_utils.quat_apply(axis_quat_w.unsqueeze(0), axis_l.unsqueeze(0)).squeeze(0)
        axis_w = axis_w / torch.linalg.norm(axis_w).clamp_min(1.0e-8)

        denom = max(1, int(self._cfg.press_steps) - 1)
        alpha = min(max(float(self._press_counter) / float(denom), 0.0), 1.0)
        offset = (
            float(self._manual_press_offset_m)
            if self._manual_press_offset_m is not None
            else float(self._cfg.press_start_offset)
            + alpha * (float(self._cfg.press_end_offset) - float(self._cfg.press_start_offset))
        )
        # ponytail: rigid translation is enough for press-rig calibration;
        # add orientation control only if the rig needs it.
        root_delta_w = -axis_w * (offset - float(self._cfg.press_start_offset))
        root_pos_w = self._press_robot_base_pose_w[:3] + root_delta_w
        root_quat_w = self._press_robot_base_pose_w[3:7]
        self._write_press_robot_root_state(root_pos_w, root_quat_w)

    def _write_press_finger_pose(self) -> None:
        if self._press_hold_joint_pos is None or len(self._press_finger_joint_ids) == 0:
            self.sync_press_robot_hold_pose()
            return

        denom = max(1, int(self._cfg.press_steps) - 1)
        alpha = min(max(float(self._press_counter) / float(denom), 0.0), 1.0)
        start = float(getattr(self._cfg, "press_finger_start_rad", 0.0))
        end = float(getattr(self._cfg, "press_finger_end_rad", 0.0))
        target = start + alpha * (end - start)

        joint_pos = self._press_hold_joint_pos.clone()
        for joint_id in self._press_finger_joint_ids:
            joint_pos[:, int(joint_id)] = target
        self._write_press_robot_joint_state(joint_pos)

    def _hold_press_object_pose(self) -> None:
        if self._plug_obj is None:
            return
        pose = self._press_object_manual_base_pose_w
        if pose is None:
            pose = self._press_object_pose_w
        if pose is None:
            current = _obj_pose_numpy(self._plug_obj)
            if current is None or current.shape != (7,) or float(np.linalg.norm(current[3:7])) < 0.5:
                return
            pose = torch.tensor(current, dtype=torch.float32, device=self._device)
            pose[3:7] = pose[3:7] / torch.linalg.norm(pose[3:7]).clamp_min(1.0e-8)
            self._press_object_pose_w = pose.detach().clone()
        self._write_press_object_state(pose[:3], pose[3:7])

    def _sync_manual_gui_press_object_pose_from_stage(self) -> None:
        if not (self._cfg.enable_press_motion and self._press_object_control_mode() == "manual_gui"):
            return
        pose = self._read_press_object_usd_pose()
        if pose is None or pose.shape != (7,) or float(np.linalg.norm(pose[3:7])) < 0.5:
            return
        pose_t = torch.tensor(pose, dtype=torch.float32, device=self._device)
        pose_t[3:7] = pose_t[3:7] / torch.linalg.norm(pose_t[3:7]).clamp_min(1.0e-8)
        self._press_object_manual_base_pose_w = pose_t.detach().clone()
        self._press_object_pose_w = pose_t.detach().clone()
        self._write_press_object_state(pose_t[:3], pose_t[3:7])
        if self._plug_obj:
            self._plug_obj.update(0.0)

    def step(self, action: np.ndarray):
        action = self._coerce_action(action)
        action_t = torch.tensor(action, dtype=torch.float32, device=self._device)
        self._sync_manual_gui_press_object_pose_from_stage()
        if (
            self._cfg.enable_press_motion
            and bool(getattr(self._cfg, "press_setup_edit_robot_pose", False))
            and self._press_motion_actor() != "hand"
        ):
            self.sync_press_robot_root_pose()

        if self._cfg.enable_press_motion and self._press_motion_actor() == "hand":
            self._press_counter = min(self._press_counter + 1, max(0, self._press_total_steps() - 1))
            self.sync_press_robot_hold_pose()
            self._write_press_hand_pose()
            self._hold_press_object_pose()
        elif self._cfg.enable_press_motion and self._press_motion_actor() == "finger":
            self._press_counter = min(self._press_counter + 1, max(0, self._press_total_steps() - 1))
            self._write_press_finger_pose()
            self._hold_press_object_pose()
        elif len(self._dataset_joint_ids) > 0:
            self._robot.set_joint_position_target(action_t, joint_ids=self._dataset_joint_ids)
            self._set_locked_press_finger_joint_targets()
            self._robot.write_data_to_sim()
        else:
            self.sync_press_robot_hold_pose()
        if (
            self._cfg.enable_press_motion
            and self._press_motion_actor() == "object"
            and self._press_object_control_mode() == "scripted"
        ):
            self._press_counter = min(self._press_counter + 1, max(0, self._press_total_steps() - 1))
            self._write_press_object_pose()
        if self._cfg.enable_press_motion and bool(getattr(self._cfg, "enable_pressure_pad_presser", False)):
            self._write_pressure_pad_presser_pose()

        render = not self._cfg.headless or self._camera is not None
        if render:
            self._refresh_tactile_debug_visualization()
        self._sim.step(render=render)
        self._write_locked_press_finger_joint_state()

        dt = self._cfg.physics_dt
        self._robot.update(dt)
        if self._plug_obj:
            self._plug_obj.update(dt)
        if self._socket_obj:
            self._socket_obj.update(dt)
        if self._pressure_pad_presser_obj:
            self._pressure_pad_presser_obj.update(dt)

        if (
            self._cfg.enable_press_motion
            and self._press_motion_actor() == "finger"
            and bool(getattr(self._cfg, "press_finger_kinematic_sensor_pose", True))
        ):
            # WarpSDF is evaluated from the commanded geometric press pose, not
            # from the post-solver rebound pose after PhysX contact resolution.
            self._write_press_finger_pose()
            self._hold_press_object_pose()
            self._robot.update(0.0)
            if self._plug_obj:
                self._plug_obj.update(0.0)
            if self._pressure_pad_presser_obj:
                self._write_pressure_pad_presser_pose()
                self._pressure_pad_presser_obj.update(0.0)
        if self._cfg.enable_press_motion and self._press_motion_actor() == "hand":
            self.sync_press_robot_hold_pose()
            self._write_press_hand_pose()
            self._hold_press_object_pose()
            if self._plug_obj:
                self._plug_obj.update(0.0)
            if self._pressure_pad_presser_obj:
                self._write_pressure_pad_presser_pose()
                self._pressure_pad_presser_obj.update(0.0)

        self._update_target_poses()
        for sensor in self._tactile_sensors:
            sensor.update(dt=dt)
        for sensor in self._physx_contact_pressure_sensors:
            sensor.update(dt=dt, force_recompute=True)
        if self._camera:
            self._camera.update(dt=dt)

        obs = self._get_obs()
        self._update_press_contact_zero(obs)

        if self._render_output_dir and self._camera:
            self._save_render(obs.get("rgb"), self._step_count)

        self._step_count += 1
        return obs, 0.0, False, False, {}

    def _refresh_tactile_debug_visualization(self) -> None:
        if not bool(getattr(self._cfg, "debug_vis", False)):
            return
        for sensor in self._tactile_sensors:
            refresh = getattr(sensor, "refresh_debug_visualization", None)
            if callable(refresh):
                refresh()

    def _update_press_contact_zero(self, obs: dict) -> None:
        if getattr(self._cfg, "press_indent_depth", None) is None:
            return
        if self._press_contact_detected:
            return
        if not (self._cfg.enable_press_motion and self._press_motion_actor() == "object"):
            return
        if self._press_object_control_mode() != "scripted":
            return
        penetration = obs.get("pressure_penetration_map")
        if penetration is None:
            return
        max_penetration = float(np.max(np.asarray(penetration, dtype=np.float32))) if np.size(penetration) else 0.0
        if max_penetration < float(getattr(self._cfg, "press_contact_threshold", 1.0e-7)):
            return
        pose = self._press_object_pose_w
        if pose is None:
            pose_np = _obj_pose_numpy(self._plug_obj)
            if pose_np is None:
                return
            pose = torch.tensor(pose_np, dtype=torch.float32, device=self._device)
        self._press_contact_pose_w = pose.detach().clone()
        self._press_contact_counter = int(self._press_counter)
        self._press_contact_detected = True
        pos = self._press_contact_pose_w[:3].detach().cpu().numpy()
        print(
            f"[PRESS_CONTACT_ZERO] step={self._step_count} press_counter={self._press_counter} "
            f"penetration={max_penetration:.9g}m "
            f"pose=({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})",
            flush=True,
        )

    def set_press_motion_control(
        self,
        *,
        center_l=None,
        center_offset_l=None,
        normal_l=None,
        offset_m=None,
        slide_offset_l=None,
    ) -> None:
        """Override scripted press target in the press link local frame."""

        if self._press_touch_center_base_l is None or self._press_touch_normal_base_l is None:
            return

        center = self._press_touch_center_base_l
        if center_l is not None:
            center = torch.tensor(center_l, dtype=torch.float32, device=self._device)
        elif center_offset_l is not None:
            center = center + torch.tensor(center_offset_l, dtype=torch.float32, device=self._device)
        self._press_touch_center_l = center

        normal = self._press_touch_normal_base_l
        if normal_l is not None:
            normal = torch.tensor(normal_l, dtype=torch.float32, device=self._device)
            normal_norm = torch.linalg.norm(normal)
            if float(normal_norm.detach().cpu().item()) > 1.0e-8:
                normal = normal / normal_norm
        self._press_touch_normal_l = normal

        self._manual_press_offset_m = None if offset_m is None else float(offset_m)
        self._manual_press_slide_offset_l = (
            None
            if slide_offset_l is None
            else torch.tensor(slide_offset_l, dtype=torch.float32, device=self._device)
        )

    def capture_press_object_pose_as_motion_start(self) -> np.ndarray | None:
        """Use the current /World/Plug world pose as the scripted press trajectory origin."""

        if self._plug_obj is None:
            return None
        pose = self._read_press_object_usd_pose()
        if pose is None or pose.shape != (7,) or float(np.linalg.norm(pose[3:7])) < 0.5:
            pose = _obj_pose_numpy(self._plug_obj)
        if pose is None or pose.shape != (7,) or float(np.linalg.norm(pose[3:7])) < 0.5:
            return None
        pose_t = torch.tensor(pose, dtype=torch.float32, device=self._device)
        pose_t[3:7] = pose_t[3:7] / torch.linalg.norm(pose_t[3:7]).clamp_min(1.0e-8)
        self._press_object_manual_base_pose_w = pose_t.detach().clone()
        self._press_object_pose_w = pose_t.detach().clone()
        self._press_manual_drag_baseline_pose_w = np.asarray(pose, dtype=np.float32).copy()
        self._press_contact_pose_w = None
        self._press_contact_counter = None
        self._press_contact_detected = False
        self._press_counter = -1
        self._write_press_object_state(pose_t[:3], pose_t[3:7])
        if self._plug_obj:
            self._plug_obj.update(0.0)
        return pose_t.detach().cpu().numpy().astype(np.float32)

    def set_press_object_world_pose(self, pos_w, quat_wxyz=None) -> None:
        if self._plug_obj is None:
            return
        pos = torch.as_tensor(pos_w, dtype=torch.float32, device=self._device)
        quat = (
            torch.as_tensor(quat_wxyz, dtype=torch.float32, device=self._device)
            if quat_wxyz is not None
            else torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self._device)
        )
        quat = quat / torch.linalg.norm(quat).clamp_min(1.0e-8)
        self._press_object_manual_base_pose_w = torch.cat([pos, quat], dim=0).detach().clone()
        self._press_object_pose_w = self._press_object_manual_base_pose_w.detach().clone()
        self._press_contact_pose_w = None
        self._press_contact_counter = None
        self._press_contact_detected = False
        self._write_press_object_state(pos, quat)
        self._sync_press_object_usd_pose(pos, quat)
        self._plug_obj.update(0.0)

    def prepare_press_object_manual_setup(self) -> np.ndarray | None:
        """Expose the scripted initial presser pose on the USD stage before GUI editing."""

        if self._press_object_pose_w is None:
            return None
        pose_t = self._press_object_pose_w.detach().clone()
        if (
            str(getattr(self._cfg, "press_touch_link", "")) == "right_thumbmcp_roll_touch_link"
            and "ball_probe" in str(getattr(self._cfg, "press_object_usd_path", "")).lower()
        ):
            # ponytail: measured GUI alignment for the current thumb MCP pressure-pad calibration setup.
            pose_t[1] = 0.042644
        self._write_press_object_state(pose_t[:3], pose_t[3:7])
        self._sync_press_object_usd_pose(pose_t[:3], pose_t[3:7])
        if self._plug_obj:
            self._plug_obj.update(0.0)
        self._press_manual_drag_baseline_pose_w = pose_t.detach().cpu().numpy().astype(np.float32)
        return self._press_manual_drag_baseline_pose_w.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        reset_joint_pos = (
            self._press_hold_joint_pos.clone()
            if self._press_hold_joint_pos is not None
            else self._robot.data.default_joint_pos.clone()
        )
        reset_joint_vel = (
            self._press_hold_joint_vel.clone()
            if self._press_hold_joint_vel is not None
            else self._robot.data.default_joint_vel.clone()
        )
        self._robot.write_joint_state_to_sim(reset_joint_pos, reset_joint_vel)
        self._robot.reset()
        if self._cfg.enable_press_motion and self._press_motion_actor() == "hand":
            self._reset_press_robot_root_pose()
        self._set_locked_press_finger_joint_targets()
        self._write_locked_press_finger_joint_state()

        for obj in (self._plug_obj, self._socket_obj, self._pressure_pad_presser_obj):
            if obj:
                obj.write_root_state_to_sim(obj.data.default_root_state)
                obj.reset()

        render = not self._cfg.headless or self._camera is not None
        self._sim.step(render=render)
        dt = self._cfg.physics_dt
        self._robot.update(dt)
        for obj in (self._plug_obj, self._socket_obj, self._pressure_pad_presser_obj):
            if obj:
                obj.update(dt)
        if self._cfg.enable_press_motion and self._press_motion_actor() == "hand":
            self._reset_press_robot_root_pose()
            self._robot.update(0.0)

        if self._cfg.enable_press_motion:
            self._press_counter = 0
            self._press_object_manual_base_pose_w = None
            self._press_contact_pose_w = None
            self._press_contact_counter = None
            self._press_contact_detected = False
            self._press_prev_object_pos_w = None
            self._press_prev_object_quat_w = None
            self._pressure_pad_presser_prev_pos_w = None
            self._pressure_pad_presser_prev_quat_w = None
            self._press_prev_robot_root_pos_w = None
            if self._press_motion_actor() != "hand":
                self._press_robot_base_pose_w = None
            if self._press_object_control_mode() in ("scripted", "initial_only"):
                self._write_press_object_pose()
                if self._plug_obj:
                    self._plug_obj.update(dt)
            else:
                self._press_object_pose_w = None
            if self._socket_obj:
                self._socket_obj.update(dt)
            if bool(getattr(self._cfg, "enable_pressure_pad_presser", False)):
                self._write_pressure_pad_presser_pose()
                if self._pressure_pad_presser_obj:
                    self._pressure_pad_presser_obj.update(dt)

        for sensor in self._tactile_sensors:
            sensor.reset()
        for sensor in self._physx_contact_pressure_sensors:
            sensor.reset()
            sensor.update(dt=dt, force_recompute=True)
        if self._camera:
            self._camera.reset()
            self._camera.update(dt=dt)

        self._step_count = 0
        return self._get_obs(), {}

    def close(self):
        self._tactile_sensors.clear()
        self._physx_contact_pressure_sensors.clear()

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _get_obs(self) -> dict:
        cfg = self._cfg

        tactile = np.zeros((self._tactile_sensor_count, cfg.num_rows, cfg.num_cols), dtype=np.float32)
        pressure_force_map = np.zeros_like(tactile)
        pressure_force_map_raw = np.zeros_like(tactile)
        pressure_signed_distance_map = np.zeros_like(tactile)
        pressure_penetration_map = np.zeros_like(tactile)
        pressure_penetration_velocity_map = np.zeros_like(tactile)
        for i, sensor in enumerate(self._tactile_sensors):
            sensor_data = sensor.data
            data = sensor_data.tactile_points_w
            slot = self._sensor_slot_order[i]
            if data is not None:
                forces = data[0, :, 3].detach().cpu().numpy().astype(np.float32)
                if 0 <= slot < tactile.shape[0]:
                    tactile[slot] = forces.reshape(cfg.num_rows, cfg.num_cols)

            force_map = getattr(sensor_data, "pressure_force_map", None)
            if force_map is not None:
                forces = force_map[0, 0].detach().cpu().numpy().astype(np.float32)
                if 0 <= slot < pressure_force_map.shape[0]:
                    pressure_force_map[slot] = forces

            force_map_raw = getattr(sensor_data, "pressure_force_map_raw", None)
            if force_map_raw is not None:
                forces = force_map_raw[0, 0].detach().cpu().numpy().astype(np.float32)
                if 0 <= slot < pressure_force_map_raw.shape[0]:
                    pressure_force_map_raw[slot] = forces

            signed_distance_map = getattr(sensor_data, "signed_distance_map", None)
            if signed_distance_map is not None:
                values = signed_distance_map[0, 0].detach().cpu().numpy().astype(np.float32)
                if 0 <= slot < pressure_signed_distance_map.shape[0]:
                    pressure_signed_distance_map[slot] = values

            penetration_map = getattr(sensor_data, "penetration_map", None)
            if penetration_map is not None:
                values = penetration_map[0, 0].detach().cpu().numpy().astype(np.float32)
                if 0 <= slot < pressure_penetration_map.shape[0]:
                    pressure_penetration_map[slot] = values

            penetration_velocity_map = getattr(sensor_data, "penetration_velocity_map", None)
            if penetration_velocity_map is not None:
                values = penetration_velocity_map[0, 0].detach().cpu().numpy().astype(np.float32)
                if 0 <= slot < pressure_penetration_velocity_map.shape[0]:
                    pressure_penetration_velocity_map[slot] = values

        physx_contact_force_map, physx_contact_force_map_raw, physx_contact_counts = (
            self._get_physx_contact_force_map_obs()
        )
        if (
            bool(getattr(cfg, "gate_warpsdf_with_physx_contact", False))
            and bool(getattr(cfg, "enable_physx_contact_force_map", False))
            and len(self._physx_contact_pressure_sensors) > 0
        ):
            # Experimental debug gate only: PhysX contacts are sparse solver
            # events, not a dense pressure-map ground-truth source.
            min_contacts = max(1, int(getattr(cfg, "warpsdf_contact_gate_min_contacts", 1)))
            active = (physx_contact_counts >= float(min_contacts)).reshape(-1, 1, 1)
            tactile = np.where(active, tactile, 0.0).astype(np.float32, copy=False)
            pressure_force_map = np.where(active, pressure_force_map, 0.0).astype(np.float32, copy=False)
            pressure_force_map_raw = np.where(active, pressure_force_map_raw, 0.0).astype(np.float32, copy=False)
            pressure_penetration_map = np.where(active, pressure_penetration_map, 0.0).astype(np.float32, copy=False)
            pressure_penetration_velocity_map = np.where(active, pressure_penetration_velocity_map, 0.0).astype(
                np.float32, copy=False
            )

        ids = self._dataset_joint_ids
        joint_pos = self._robot.data.joint_pos[0, ids].detach().cpu().numpy().astype(np.float32)
        joint_vel = self._robot.data.joint_vel[0, ids].detach().cpu().numpy().astype(np.float32)

        obs = {
            "tactile": tactile,
            "pressure_force_map": pressure_force_map,
            "pressure_force_map_raw": pressure_force_map_raw,
            "pressure_signed_distance_map": pressure_signed_distance_map,
            "pressure_penetration_map": pressure_penetration_map,
            "pressure_penetration_velocity_map": pressure_penetration_velocity_map,
            "physx_contact_force_map": physx_contact_force_map,
            "physx_contact_force_map_raw": physx_contact_force_map_raw,
            "physx_contact_count": physx_contact_counts,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "plug_pose": self.press_object_pose_w_numpy(),
            "socket_pose": _obj_pose_numpy(self._socket_obj),
            "press_touch_pose": self._press_touch_pose_numpy(),
        }

        if self._camera:
            try:
                rgb = self._camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
                obs["rgb"] = rgb
            except Exception:
                obs["rgb"] = np.zeros((cfg.camera_height, cfg.camera_width, 3), dtype=np.uint8)

        return obs

    def press_object_pose_w_numpy(self) -> np.ndarray:
        pose = None
        if self._cfg.enable_press_motion and self._press_object_control_mode() in ("initial_only", "manual_gui"):
            pose = self._read_press_object_usd_pose()
        if pose is None:
            pose = _obj_pose_numpy(self._plug_obj)
        return np.asarray(pose, dtype=np.float32).reshape(7)

    def _press_touch_pose_numpy(self) -> np.ndarray:
        pose = np.zeros(7, dtype=np.float32)
        if self._press_touch_body_idx is None:
            return pose
        try:
            state = self._robot.data.body_link_state_w[0, self._press_touch_body_idx, :7]
        except Exception:
            return pose
        pose[:] = to_numpy(state, dtype=np.float32, shape=(7,))
        return pose

    def _get_physx_contact_force_map_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self._cfg
        force_map = np.zeros((self._tactile_sensor_count, cfg.num_rows, cfg.num_cols), dtype=np.float32)
        force_map_raw = np.zeros_like(force_map)
        contact_counts = np.zeros((self._tactile_sensor_count,), dtype=np.float32)
        if not getattr(cfg, "enable_physx_contact_force_map", False):
            return force_map, force_map_raw, contact_counts

        for i, contact_sensor in enumerate(self._physx_contact_pressure_sensors):
            if i >= len(self._tactile_sensors):
                continue
            slot = self._sensor_slot_order[i] if i < len(self._sensor_slot_order) else i
            if slot < 0 or slot >= force_map.shape[0]:
                continue

            output, contact_count = self._physx_contact_output_for_sensor(i, contact_sensor)
            if output is None:
                continue
            force_map[slot] = output.force_map.detach().cpu().numpy().astype(np.float32)
            force_map_raw[slot] = output.raw_force_map.detach().cpu().numpy().astype(np.float32)
            contact_counts[slot] = float(contact_count)
        return force_map, force_map_raw, contact_counts

    def _press_object_axis_w(self, touch_quat_w: torch.Tensor) -> torch.Tensor | None:
        axis_link = str(getattr(self._cfg, "press_hand_axis_link", "")).strip().lower()
        if not axis_link:
            axis_w = self._ball_probe_local_minus_y_axis_w()
            if axis_w is not None:
                return axis_w
        if axis_link == "touch_nearest_world_z":
            axes_l = torch.tensor(
                (
                    (1.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                ),
                dtype=torch.float32,
                device=self._device,
            )
            axes_w = self._math_utils.quat_apply(touch_quat_w.unsqueeze(0).repeat(6, 1), axes_l)
            world_down = torch.tensor((0.0, 0.0, -1.0), dtype=torch.float32, device=self._device)
            axis_w = axes_w[int(torch.argmax(axes_w @ world_down).detach().cpu().item())]
        elif axis_link == "world" and self._press_hand_axis_l is not None:
            axis_w = self._press_hand_axis_l
        elif self._press_hand_axis_body_idx is not None and self._press_hand_axis_l is not None:
            axis_quat_w = self._robot.data.body_link_state_w[0, int(self._press_hand_axis_body_idx), 3:7]
            axis_w = self._math_utils.quat_apply(
                axis_quat_w.unsqueeze(0), self._press_hand_axis_l.unsqueeze(0)
            ).squeeze(0)
        elif self._press_touch_normal_l is not None:
            normal_w = self._math_utils.quat_apply(
                touch_quat_w.unsqueeze(0), self._press_touch_normal_l.unsqueeze(0)
            ).squeeze(0)
            axis_w = -normal_w
        else:
            return None
        axis_w = axis_w / torch.linalg.norm(axis_w).clamp_min(1.0e-8)
        self._press_debug_axis_w = axis_w.detach().clone()
        self._press_debug_axis_source = axis_link or "sensor_normal"
        return axis_w

    def _ball_probe_local_minus_y_axis_w(self) -> torch.Tensor | None:
        if "ball_probe" not in str(getattr(self._cfg, "press_object_usd_path", "")).lower():
            return None
        pose = self._press_object_manual_base_pose_w
        if pose is None:
            pose = self._press_object_pose_w
        if pose is None:
            return None
        quat_w = pose[3:7]
        probe_axis_l = torch.tensor((0.0, -1.0, 0.0), dtype=torch.float32, device=self._device)
        axis_w = self._math_utils.quat_apply(quat_w.unsqueeze(0), probe_axis_l.unsqueeze(0)).squeeze(0)
        axis_w = axis_w / torch.linalg.norm(axis_w).clamp_min(1.0e-8)
        self._press_debug_axis_w = axis_w.detach().clone()
        self._press_debug_axis_source = "ball_probe_local_-y"
        return axis_w

    def _physx_contact_output_for_sensor(self, sensor_index: int, contact_sensor):
        if not contact_sensor.is_initialized:
            return None, 0
        tactile_sensor = self._tactile_sensors[sensor_index]
        taxel_maps = getattr(tactile_sensor, "pressure_taxel_maps", ())
        if not taxel_maps:
            return None, 0
        taxel_map = taxel_maps[0]

        event_batch, contact_count = self._physx_contact_events_for_sensor(contact_sensor, taxel_map)
        if event_batch is None:
            return None, 0
        output = taxel_map.force_from_contacts(
            event_batch,
            kernel_sigma=getattr(self._cfg, "physx_contact_kernel_sigma", None),
            kernel_radius=getattr(self._cfg, "physx_contact_kernel_radius", None),
            conserve_total_force=True,
            normalize=True,
        )
        return output, contact_count

    def _physx_contact_events_for_sensor(self, contact_sensor, taxel_map):
        data = contact_sensor.data
        if data.pos_w is None or data.quat_w is None:
            return None, 0

        contact_forces, contact_points_w, contact_normals_w, _distances, buffer_count, buffer_start_indices = (
            contact_sensor.contact_physx_view.get_contact_data(dt=float(self._cfg.physics_dt))
        )
        if buffer_count.numel() == 0:
            return self._PressureContactEventBatch(
                source_link=taxel_map.link_name,
                contact_points_l=taxel_map.points_l.new_zeros((0, 3)),
                contact_normals_l=taxel_map.points_l.new_zeros((0, 3)),
                normal_forces=taxel_map.points_l.new_zeros((0,)),
            ), 0

        counts = buffer_count.reshape(-1, buffer_count.shape[-1])
        starts = buffer_start_indices.reshape(-1, buffer_start_indices.shape[-1])
        row = 0
        indices = []
        for filter_i in range(counts.shape[1]):
            count = int(counts[row, filter_i].detach().cpu().item())
            if count <= 0:
                continue
            start = int(starts[row, filter_i].detach().cpu().item())
            indices.append(torch.arange(start, start + count, device=contact_points_w.device, dtype=torch.long))
        if not indices:
            return self._PressureContactEventBatch(
                source_link=taxel_map.link_name,
                contact_points_l=taxel_map.points_l.new_zeros((0, 3)),
                contact_normals_l=taxel_map.points_l.new_zeros((0, 3)),
                normal_forces=taxel_map.points_l.new_zeros((0,)),
            ), 0

        idx = torch.cat(indices, dim=0)
        points_w = contact_points_w.index_select(0, idx).to(device=taxel_map.points_l.device, dtype=torch.float32)
        normals_w = contact_normals_w.index_select(0, idx).to(device=taxel_map.points_l.device, dtype=torch.float32)
        forces = contact_forces.index_select(0, idx).to(device=taxel_map.points_l.device, dtype=torch.float32)
        if forces.ndim >= 2 and forces.shape[-1] == 1:
            normal_forces = forces.reshape(-1).abs()
        elif forces.ndim >= 2 and forces.shape[-1] == 3:
            normal_forces = torch.linalg.norm(forces.reshape(-1, 3), dim=-1)
        else:
            normal_forces = forces.reshape(-1).abs()

        finite = (
            torch.isfinite(points_w).all(dim=-1)
            & torch.isfinite(normals_w).all(dim=-1)
            & torch.isfinite(normal_forces)
            & (normal_forces > 0.0)
        )
        if not torch.any(finite):
            return self._PressureContactEventBatch(
                source_link=taxel_map.link_name,
                contact_points_l=taxel_map.points_l.new_zeros((0, 3)),
                contact_normals_l=taxel_map.points_l.new_zeros((0, 3)),
                normal_forces=taxel_map.points_l.new_zeros((0,)),
            ), 0

        points_w = points_w[finite]
        normals_w = normals_w[finite]
        normal_forces = normal_forces[finite]

        pos_w = data.pos_w[0, 0].to(device=taxel_map.points_l.device, dtype=torch.float32)
        quat_w = data.quat_w[0, 0].to(device=taxel_map.points_l.device, dtype=torch.float32)
        quat = quat_w.unsqueeze(0).expand(points_w.shape[0], -1)
        points_l = self._math_utils.quat_apply_inverse(quat, points_w - pos_w.unsqueeze(0))
        normals_l = self._math_utils.quat_apply_inverse(quat, normals_w)
        normals_l = normals_l / torch.linalg.norm(normals_l, dim=-1, keepdim=True).clamp_min(1.0e-8)

        mean_normal = torch.mean(taxel_map.normals_l, dim=0)
        mean_normal = mean_normal / torch.linalg.norm(mean_normal).clamp_min(1.0e-8)
        alignment = torch.sum(normals_l * mean_normal.unsqueeze(0), dim=-1, keepdim=True)
        normals_l = torch.where(alignment < 0.0, -normals_l, normals_l)

        event_batch = self._PressureContactEventBatch(
            source_link=taxel_map.link_name,
            contact_points_l=points_l,
            contact_normals_l=normals_l,
            normal_forces=normal_forces,
        )
        return event_batch, int(points_l.shape[0])

    def _coerce_action(self, action: np.ndarray) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        action_dim = len(self._dataset_joint_ids)
        if raw.size == action_dim:
            return raw

        current = self._robot.data.joint_pos[0, self._dataset_joint_ids].detach().cpu().numpy().astype(np.float32)
        n = min(raw.size, action_dim)
        current[:n] = raw[:n]

        if not self._warned_action_size:
            print(
                f"[WARN] Action size {raw.size} does not match robot action_dim {action_dim}; "
                f"using the first {n} values and keeping the rest at current joint positions.",
                flush=True,
            )
            self._warned_action_size = True
        return current

    @staticmethod
    def _load_press_center_and_normal(cfg: AlohaTactileEnvCfg) -> tuple[np.ndarray, np.ndarray]:
        if getattr(cfg, "press_center_l", None) is not None and getattr(cfg, "press_normal_l", None) is not None:
            center = np.asarray(cfg.press_center_l, dtype=np.float64)
            normal = np.asarray(cfg.press_normal_l, dtype=np.float64)
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            return center.astype(np.float32), normal.astype(np.float32)

        points = np.load(os.path.expanduser(cfg.press_points_npy)).astype(np.float64)
        normals = np.load(os.path.expanduser(cfg.press_normals_npy)).astype(np.float64)
        points *= float(cfg.press_map_correction_scale)
        valid = (
            np.isfinite(points).all(axis=-1)
            & np.isfinite(normals).all(axis=-1)
            & (np.linalg.norm(points, axis=-1) > 1e-12)
            & (np.linalg.norm(normals, axis=-1) > 0.5)
        )
        if not np.any(valid):
            raise RuntimeError(f"No valid tactile samples found in {cfg.press_points_npy}")

        pts = points[valid]
        nrms = normals[valid]
        nrms = nrms / (np.linalg.norm(nrms, axis=-1, keepdims=True) + 1e-12)
        center = np.mean(pts, axis=0)
        normal = np.mean(nrms, axis=0)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        return center.astype(np.float32), normal.astype(np.float32)

    @staticmethod
    def _local_axis_vector(axis_name: str) -> np.ndarray:
        axis_name = str(axis_name).strip().lower()
        sign = -1.0 if axis_name.startswith("-") else 1.0
        axis = axis_name[-1]
        values = {
            "x": (sign, 0.0, 0.0),
            "y": (0.0, sign, 0.0),
            "z": (0.0, 0.0, sign),
        }.get(axis, (1.0, 0.0, 0.0))
        return np.asarray(values, dtype=np.float64)

    @classmethod
    def _press_center_and_normal_l(cls, cfg: AlohaTactileEnvCfg) -> tuple[np.ndarray, np.ndarray]:
        if str(getattr(cfg, "press_motion_frame", "touch")).lower() != "link_surface":
            return cls._load_press_center_and_normal(cfg)

        center = np.asarray(
            getattr(cfg, "tacmap_link_surface_grid_center", (-0.008, 0.0, 0.0012)),
            dtype=np.float64,
        )
        direction = getattr(cfg, "tacmap_link_surface_ray_direction", None)
        if direction is not None:
            normal = np.asarray(tuple(float(v) for v in direction), dtype=np.float64)
        elif bool(getattr(cfg, "tacmap_link_surface_use_mean_normal", True)):
            _, normal = cls._load_press_center_and_normal(cfg)
            normal = np.asarray(normal, dtype=np.float64)
        else:
            normal = cls._local_axis_vector(getattr(cfg, "tacmap_link_surface_ray_axis", "+x"))

        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal_norm) or normal_norm < 1.0e-8:
            normal = cls._local_axis_vector(getattr(cfg, "tacmap_link_surface_ray_axis", "+x"))
            normal_norm = float(np.linalg.norm(normal))
        normal = normal / max(normal_norm, 1.0e-12)
        return center.astype(np.float32), normal.astype(np.float32)

    @staticmethod
    def _quat_from_vectors(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, float, float]:
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        src = src / (np.linalg.norm(src) + 1e-12)
        dst = dst / (np.linalg.norm(dst) + 1e-12)
        dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
        if dot < -0.999999:
            axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            return (0.0, float(axis[0]), float(axis[1]), float(axis[2]))

        axis = np.cross(src, dst)
        quat = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=np.float64)
        quat = quat / (np.linalg.norm(quat) + 1e-12)
        return tuple(float(v) for v in quat)

    def _angular_velocity_from_quat_delta(
        self,
        prev_quat_w: torch.Tensor,
        current_quat_w: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        prev_quat = prev_quat_w.to(device=self._device, dtype=torch.float32)
        current_quat = current_quat_w.to(device=self._device, dtype=torch.float32)
        prev_quat = prev_quat / torch.linalg.norm(prev_quat).clamp_min(1.0e-8)
        current_quat = current_quat / torch.linalg.norm(current_quat).clamp_min(1.0e-8)
        if bool((torch.dot(prev_quat, current_quat) < 0.0).detach().cpu().item()):
            current_quat = -current_quat

        delta_quat = self._math_utils.quat_mul(
            current_quat.unsqueeze(0),
            self._math_utils.quat_inv(prev_quat.unsqueeze(0)),
        ).squeeze(0)
        if bool((delta_quat[0] < 0.0).detach().cpu().item()):
            delta_quat = -delta_quat

        axis_scaled = delta_quat[1:4]
        sin_half_angle = torch.linalg.norm(axis_scaled)
        if bool((sin_half_angle < 1.0e-8).detach().cpu().item()):
            return torch.zeros(3, device=self._device, dtype=torch.float32)
        angle = 2.0 * torch.atan2(sin_half_angle, delta_quat[0])
        return axis_scaled * (angle / (sin_half_angle * max(float(dt), 1.0e-8)))

    def _setup_press_motion(self, cfg: AlohaTactileEnvCfg) -> None:
        if not cfg.enable_press_motion:
            return

        body_names = [str(name) for name in self._robot.body_names]

        def resolve_body_idx(name: str) -> int:
            if name in body_names:
                return body_names.index(name)
            matches = [i for i, body_name in enumerate(body_names) if body_name.endswith(name) or name in body_name]
            if not matches:
                raise RuntimeError(f"Press body link {name!r} not found in robot body names: {body_names}")
            return matches[0]

        touch_link = str(cfg.press_touch_link)
        self._press_touch_body_idx = resolve_body_idx(touch_link)
        hand_axis_link = str(getattr(cfg, "press_hand_axis_link", "")).strip()
        world_axis = hand_axis_link.lower() == "world"
        touch_nearest_world_z = hand_axis_link.lower() == "touch_nearest_world_z"
        self._press_hand_axis_body_idx = (
            None if world_axis or touch_nearest_world_z or not hand_axis_link else resolve_body_idx(hand_axis_link)
        )
        hand_axis_l = torch.tensor(
            getattr(cfg, "press_hand_axis_l", (1.0, 0.0, 0.0)),
            dtype=torch.float32,
            device=self._device,
        )
        self._press_hand_axis_l = hand_axis_l / torch.linalg.norm(hand_axis_l).clamp_min(1.0e-8)

        center_l, normal_l = self._press_center_and_normal_l(cfg)
        self._press_touch_center_base_l = torch.tensor(center_l, dtype=torch.float32, device=self._device)
        self._press_touch_normal_base_l = torch.tensor(normal_l, dtype=torch.float32, device=self._device)
        self._press_touch_center_l = self._press_touch_center_base_l.clone()
        self._press_touch_normal_l = self._press_touch_normal_base_l.clone()
        self._press_object_rot_l = torch.tensor(
            cfg.press_object_rot_in_touch_frame, dtype=torch.float32, device=self._device
        )
        self._press_object_flip_l = torch.tensor(
            cfg.press_object_flip_quat_in_touch_frame, dtype=torch.float32, device=self._device
        )
        if (
            "ball_probe" in str(getattr(cfg, "press_object_usd_path", "")).lower()
            and not str(getattr(cfg, "press_hand_axis_link", "")).strip()
            and not bool(getattr(cfg, "press_tacmap_files_enabled", True))
        ):
            align_quat = self._quat_from_vectors((0.0, -1.0, 0.0), -np.asarray(normal_l, dtype=np.float64))
            self._press_object_rot_l = torch.tensor(align_quat, dtype=torch.float32, device=self._device)
            self._press_object_flip_l = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self._device)
        slide_axis = torch.tensor(cfg.press_slide_axis_l, dtype=torch.float32, device=self._device)
        slide_axis_norm = torch.linalg.norm(slide_axis)
        if float(slide_axis_norm.detach().cpu().item()) < 1.0e-8:
            slide_axis = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=self._device)
            slide_axis_norm = torch.linalg.norm(slide_axis)
        self._press_slide_axis_l = slide_axis / slide_axis_norm.clamp_min(1.0e-8)
        hand_axis_text = (
            f"{'world' if world_axis else hand_axis_link} {tuple(float(v) for v in self._press_hand_axis_l.detach().cpu())}"
            if hand_axis_link
            else "<sensor_normal>"
        )
        print(
            f"[INFO] Press motion: link={body_names[self._press_touch_body_idx]}, "
            f"frame={getattr(cfg, 'press_motion_frame', 'touch')}, "
            f"center_l={center_l}, normal_l={normal_l}, "
            f"offset={cfg.press_start_offset}->{cfg.press_end_offset} over {cfg.press_steps} steps, "
            f"slide={cfg.press_slide_distance}m over {cfg.press_slide_steps} steps axis_l={tuple(cfg.press_slide_axis_l)}, "
            f"hand_axis={hand_axis_text}, "
            f"actor={self._press_motion_actor()}, "
            f"object_control={self._press_object_control_mode()}",
            flush=True,
        )

    def _setup_pressure_pad_presser_motion(self, cfg: AlohaTactileEnvCfg) -> None:
        if not (cfg.enable_press_motion and bool(getattr(cfg, "enable_pressure_pad_presser", False))):
            return
        if self._pressure_pad_presser_obj is None:
            print("[WARN] pressure-pad presser is enabled, but the presser object was not spawned.", flush=True)
            return

        link_name = str(getattr(cfg, "pressure_pad_presser_link", "")).strip()
        if not link_name:
            print("[WARN] pressure-pad presser is enabled, but pressure_pad_presser_link is empty.", flush=True)
            return

        body_names = [str(name) for name in self._robot.body_names]
        matches = [i for i, body_name in enumerate(body_names) if body_name == link_name]
        if not matches:
            matches = [i for i, body_name in enumerate(body_names) if body_name.endswith(link_name) or link_name in body_name]
        if not matches:
            raise RuntimeError(f"Pressure-pad presser link {link_name!r} not found in robot body names: {body_names}")
        self._pressure_pad_presser_body_idx = int(matches[0])

        patch_spec = self._press_link_patch_spec(link_name, cfg)
        if patch_spec is None:
            center_l = np.zeros(3, dtype=np.float32)
            normal_l = np.zeros(3, dtype=np.float32)
            axis = int(getattr(cfg, "normal_axis", 0))
            normal_l[axis if axis in (0, 1, 2) else 0] = 1.0 if float(getattr(cfg, "normal_sign", 1.0)) >= 0.0 else -1.0
        else:
            center_l, normal_l, _is_primary = patch_spec
            center_l = np.asarray(center_l, dtype=np.float32)
            normal_l = np.asarray(normal_l, dtype=np.float32)
        normal_norm = np.linalg.norm(normal_l)
        if center_l.shape != (3,) or normal_l.shape != (3,) or normal_norm < 1.0e-8:
            raise RuntimeError(f"Invalid pressure-pad presser patch for link {link_name!r}")

        normal_unit_l = normal_l / normal_norm
        self._pressure_pad_presser_center_l = torch.tensor(center_l, dtype=torch.float32, device=self._device)
        self._pressure_pad_presser_normal_l = torch.tensor(
            normal_unit_l,
            dtype=torch.float32,
            device=self._device,
        )
        offset_l = np.asarray(
            getattr(cfg, "pressure_pad_presser_offset_l", (0.0, 0.0, 0.0)),
            dtype=np.float32,
        )
        if offset_l.shape != (3,) or not np.all(np.isfinite(offset_l)):
            raise RuntimeError(f"Invalid pressure-pad presser offset_l for link {link_name!r}: {offset_l}")
        self._pressure_pad_presser_offset_l = torch.tensor(offset_l, dtype=torch.float32, device=self._device)

        def _trajectory_offset_l(name: str, values, default_scalar: float) -> np.ndarray:
            if values is None:
                return np.asarray(normal_unit_l * float(default_scalar), dtype=np.float32)
            offset_vec_l = np.asarray(values, dtype=np.float32)
            if offset_vec_l.shape != (3,) or not np.all(np.isfinite(offset_vec_l)):
                raise RuntimeError(f"Invalid pressure-pad presser {name} for link {link_name!r}: {offset_vec_l}")
            return offset_vec_l

        start_offset_l = _trajectory_offset_l(
            "start_offset_l",
            getattr(cfg, "pressure_pad_presser_start_offset_l", None),
            float(getattr(cfg, "pressure_pad_presser_start_offset", 0.012)),
        )
        end_offset_l = _trajectory_offset_l(
            "end_offset_l",
            getattr(cfg, "pressure_pad_presser_end_offset_l", None),
            float(getattr(cfg, "pressure_pad_presser_end_offset", -0.001)),
        )
        self._pressure_pad_presser_start_offset_l = torch.tensor(
            start_offset_l,
            dtype=torch.float32,
            device=self._device,
        )
        self._pressure_pad_presser_end_offset_l = torch.tensor(
            end_offset_l,
            dtype=torch.float32,
            device=self._device,
        )
        self._pressure_pad_presser_rot_l = torch.tensor(
            cfg.press_object_rot_in_touch_frame,
            dtype=torch.float32,
            device=self._device,
        )
        self._pressure_pad_presser_flip_l = torch.tensor(
            cfg.press_object_flip_quat_in_touch_frame,
            dtype=torch.float32,
            device=self._device,
        )
        print(
            "[INFO] pressure-pad presser motion: "
            f"link={body_names[self._pressure_pad_presser_body_idx]}, "
            f"center_l={tuple(float(v) for v in center_l)}, "
            f"normal_l={tuple(float(v) for v in normal_unit_l)}, "
            f"offset={float(getattr(cfg, 'pressure_pad_presser_start_offset', 0.012)):g}"
            f"->{float(getattr(cfg, 'pressure_pad_presser_end_offset', -0.001)):g}, "
            f"steps={int(getattr(cfg, 'pressure_pad_presser_steps', 0) or cfg.press_steps)}, "
            f"offset_l={tuple(float(v) for v in offset_l)}, "
            f"start_offset_l={tuple(float(v) for v in start_offset_l)}, "
            f"end_offset_l={tuple(float(v) for v in end_offset_l)}",
            flush=True,
        )

    def _apply_press_initial_joint_degrees(self, cfg: AlohaTactileEnvCfg) -> None:
        if self._press_hold_joint_pos is None:
            return

        overrides = getattr(cfg, "press_initial_joint_degrees", ()) or ()
        if not overrides:
            return
        joint_names = [str(name) for name in self._robot.joint_names]
        applied_overrides = []
        for joint_name, degrees in overrides:
            matches = [i for i, name in enumerate(joint_names) if name == str(joint_name)]
            if not matches:
                matches = [i for i, name in enumerate(joint_names) if name.endswith(str(joint_name))]
            if not matches:
                raise RuntimeError(f"Initial press joint {joint_name!r} not found in robot joints: {joint_names}")
            joint_id = int(matches[0])
            radians = float(degrees) * float(np.pi) / 180.0
            if self._press_hold_joint_pos.ndim == 1:
                self._press_hold_joint_pos[joint_id] = radians
            else:
                self._press_hold_joint_pos[:, joint_id] = radians
            applied_overrides.append((joint_names[joint_id], radians, float(degrees)))
        text = ", ".join(f"{name}={degrees:g}deg/{radians:g}rad" for name, radians, degrees in applied_overrides)
        print(f"[INFO] Press initial joint overrides: {text}", flush=True)

    def _setup_press_finger_motion(self, cfg: AlohaTactileEnvCfg) -> None:
        actor = self._press_motion_actor()
        if not cfg.enable_press_motion:
            return
        if actor not in {"finger", "hand"}:
            return

        joint_names = [str(name) for name in self._robot.joint_names]
        if self._press_hold_joint_pos is None:
            hold_pose = str(getattr(cfg, "press_hold_joint_pose", "zero")).lower()
            self._press_hold_joint_pos = self._robot.data.default_joint_pos.clone()
            if hold_pose == "zero":
                self._press_hold_joint_pos.zero_()
            elif hold_pose != "default":
                raise ValueError(f"Unsupported press_hold_joint_pose={hold_pose!r}; expected 'zero' or 'default'")
            self._apply_press_initial_joint_degrees(cfg)
            self._press_hold_joint_vel = torch.zeros_like(self._robot.data.default_joint_vel)
        if len(self._press_hold_joint_ids) == 0:
            self._press_hold_joint_ids = list(range(len(self._robot.joint_names)))

        if actor == "hand":
            self._write_press_robot_joint_state(self._press_hold_joint_pos)
            print(
                "[INFO] Press hand actor: "
                f"moving hand root along configured axis; holding {len(self._press_hold_joint_ids)} joints.",
                flush=True,
            )
            return

        requested = tuple(str(name).strip() for name in getattr(cfg, "press_finger_joints", ()) if str(name).strip())
        if not requested:
            raise RuntimeError("--press-motion-actor finger requires at least one --press-finger-joint.")

        ids: list[int] = []
        resolved_names: list[str] = []
        for requested_name in requested:
            matches = [i for i, name in enumerate(joint_names) if name == requested_name]
            if not matches:
                matches = [i for i, name in enumerate(joint_names) if name.endswith(requested_name)]
            if not matches:
                raise RuntimeError(
                    f"Press finger joint {requested_name!r} not found in robot joints: {joint_names}"
                )
            joint_id = int(matches[0])
            if joint_id not in ids:
                ids.append(joint_id)
                resolved_names.append(joint_names[joint_id])

        self._press_finger_joint_ids = ids
        self._press_finger_joint_names = resolved_names
        print(
            "[INFO] Press finger actor: "
            f"joints={resolved_names}, "
            f"target={float(getattr(cfg, 'press_finger_start_rad', 0.0)):g}"
            f"->{float(getattr(cfg, 'press_finger_end_rad', 0.0)):g} rad over {cfg.press_steps} steps; "
            "Plug pose is held fixed during the run.",
            flush=True,
        )

    def _press_total_steps(self) -> int:
        return max(1, int(self._cfg.press_steps) + max(0, int(self._cfg.press_slide_steps)))

    def _press_object_control_mode(self) -> str:
        mode = str(getattr(self._cfg, "press_object_control", "scripted"))
        return mode if mode in {"scripted", "initial_only", "manual_gui"} else "scripted"

    def _setup_locked_press_finger_joints(self, cfg: AlohaTactileEnvCfg) -> None:
        self._locked_press_joint_ids = None
        self._locked_press_joint_pos = None
        self._locked_press_joint_vel = None
        if not bool(getattr(cfg, "lock_press_finger_joints", False)):
            return

        touch_link = str(cfg.press_touch_link)
        finger_prefix = touch_link.removesuffix("_touch_link")
        joint_names = [str(name) for name in self._robot.joint_names]
        joint_ids = [i for i, name in enumerate(joint_names) if name.startswith(f"{finger_prefix}_")]
        if not joint_ids:
            print(
                f"[WARN] Press finger joint lock requested, but no joints matched prefix {finger_prefix!r}.",
                flush=True,
            )
            return

        self._locked_press_joint_ids = joint_ids
        self._locked_press_joint_pos = self._robot.data.joint_pos[0, joint_ids].detach().clone()
        self._locked_press_joint_vel = torch.zeros_like(self._locked_press_joint_pos)
        locked_names = [joint_names[i] for i in joint_ids]
        print(
            f"[INFO] Locked press finger joints at initial positions: {locked_names}",
            flush=True,
        )

    def _set_locked_press_finger_joint_targets(self) -> None:
        if self._locked_press_joint_ids is None or self._locked_press_joint_pos is None:
            return
        self._robot.set_joint_position_target(
            self._locked_press_joint_pos,
            joint_ids=self._locked_press_joint_ids,
        )

    def _write_locked_press_finger_joint_state(self) -> None:
        if (
            self._locked_press_joint_ids is None
            or self._locked_press_joint_pos is None
            or self._locked_press_joint_vel is None
        ):
            return
        self._robot.write_joint_state_to_sim(
            self._locked_press_joint_pos.unsqueeze(0),
            self._locked_press_joint_vel.unsqueeze(0),
            joint_ids=self._locked_press_joint_ids,
        )

    def _write_press_object_pose(self) -> None:
        if self._plug_obj is None or self._press_touch_body_idx is None:
            return
        if self._press_touch_center_l is None or self._press_touch_normal_l is None:
            return

        touch_state = self._robot.data.body_link_state_w[0, self._press_touch_body_idx, :7]
        touch_pos_w = touch_state[:3]
        touch_quat_w = touch_state[3:7]

        denom = max(1, int(self._cfg.press_steps) - 1)
        alpha = min(max(float(self._press_counter) / float(denom), 0.0), 1.0)
        offset = (
            float(self._manual_press_offset_m)
            if self._manual_press_offset_m is not None
            else float(self._cfg.press_start_offset)
            + alpha * (float(self._cfg.press_end_offset) - float(self._cfg.press_start_offset))
        )
        object_delta_l = self._press_touch_normal_l * (offset - float(self._cfg.press_start_offset))
        object_pos_l = self._press_touch_center_l + self._press_touch_normal_l * offset
        slide_steps = max(0, int(self._cfg.press_slide_steps))
        slide_distance = float(self._cfg.press_slide_distance)
        if slide_steps > 0 and abs(slide_distance) > 0.0 and self._press_slide_axis_l is not None:
            slide_counter = max(0, int(self._press_counter) - (max(1, int(self._cfg.press_steps)) - 1))
            slide_alpha = min(max(float(slide_counter) / float(max(1, slide_steps)), 0.0), 1.0)
            slide_delta_l = self._press_slide_axis_l * (slide_distance * slide_alpha)
            object_pos_l = object_pos_l + slide_delta_l
            object_delta_l = object_delta_l + slide_delta_l
        if self._manual_press_slide_offset_l is not None:
            object_pos_l = object_pos_l + self._manual_press_slide_offset_l
            object_delta_l = object_delta_l + self._manual_press_slide_offset_l

        if self._press_object_manual_base_pose_w is not None:
            base_pose = self._press_object_manual_base_pose_w
            if getattr(self._cfg, "press_indent_depth", None) is not None:
                axis_w = self._press_object_axis_w(touch_quat_w)
                if axis_w is None:
                    axis_w = self._ball_probe_local_minus_y_axis_w()
                if axis_w is None:
                    return
                if self._press_contact_detected and self._press_contact_pose_w is not None:
                    contact_counter = int(self._press_contact_counter or self._press_counter)
                    remaining = max(1, int(self._cfg.press_steps) - contact_counter - 1)
                    post_alpha = min(max(float(self._press_counter - contact_counter) / float(remaining), 0.0), 1.0)
                    object_pos_w = self._press_contact_pose_w[:3] + axis_w * (
                        float(self._cfg.press_indent_depth) * post_alpha
                    )
                    object_quat_w = self._press_contact_pose_w[3:7]
                else:
                    search_distance = getattr(self._cfg, "press_contact_search_distance", None)
                    if search_distance is None:
                        search_distance = abs(float(self._cfg.press_start_offset) - float(self._cfg.press_end_offset))
                    object_pos_w = base_pose[:3] + axis_w * (float(search_distance) * alpha)
                    object_quat_w = base_pose[3:7]
            elif str(getattr(self._cfg, "press_hand_axis_link", "")).strip().lower() == "touch_nearest_world_z":
                axis_w = self._press_object_axis_w(touch_quat_w)
                if axis_w is None:
                    return
                object_pos_w = base_pose[:3] + axis_w * (float(self._cfg.press_start_offset) - offset)
            elif (
                str(getattr(self._cfg, "press_hand_axis_link", "")).strip().lower() == "world"
                and self._press_hand_axis_l is not None
            ):
                axis_w = self._press_hand_axis_l
                axis_w = axis_w / torch.linalg.norm(axis_w).clamp_min(1.0e-8)
                object_pos_w = base_pose[:3] + axis_w * (float(self._cfg.press_start_offset) - offset)
            elif self._press_hand_axis_body_idx is not None and self._press_hand_axis_l is not None:
                axis_quat_w = self._robot.data.body_link_state_w[0, int(self._press_hand_axis_body_idx), 3:7]
                axis_w = self._math_utils.quat_apply(
                    axis_quat_w.unsqueeze(0), self._press_hand_axis_l.unsqueeze(0)
                ).squeeze(0)
                axis_w = axis_w / torch.linalg.norm(axis_w).clamp_min(1.0e-8)
                object_pos_w = base_pose[:3] + axis_w * (float(self._cfg.press_start_offset) - offset)
            else:
                axis_w = self._ball_probe_local_minus_y_axis_w()
                if axis_w is None:
                    object_pos_w = base_pose[:3] + self._math_utils.quat_apply(
                        touch_quat_w.unsqueeze(0), object_delta_l.unsqueeze(0)
                    ).squeeze(0)
                else:
                    object_pos_w = base_pose[:3] + axis_w * (float(self._cfg.press_start_offset) - offset)
                object_quat_w = base_pose[3:7]
            if getattr(self._cfg, "press_indent_depth", None) is None:
                object_quat_w = base_pose[3:7]
        else:
            object_pos_w = self._math_utils.quat_apply(touch_quat_w.unsqueeze(0), object_pos_l.unsqueeze(0)).squeeze(0)
            object_pos_w = object_pos_w + touch_pos_w
            object_quat_l = self._math_utils.quat_mul(
                self._press_object_rot_l.unsqueeze(0),
                self._press_object_flip_l.unsqueeze(0),
            ).squeeze(0)
            object_quat_w = self._math_utils.quat_mul(touch_quat_w.unsqueeze(0), object_quat_l.unsqueeze(0)).squeeze(0)
        self._press_object_pose_w = torch.cat([object_pos_w, object_quat_w], dim=0).detach().clone()
        self._write_press_object_state(object_pos_w, object_quat_w)
        if self._press_object_control_mode() == "initial_only":
            self._sync_press_object_usd_pose(object_pos_w, object_quat_w)

    def _write_press_object_state(self, object_pos_w, object_quat_w) -> None:
        if self._plug_obj is None:
            return
        object_state = self._plug_obj.data.default_root_state.clone()
        if object_state.ndim == 1:
            object_state = object_state.unsqueeze(0)
        object_state[:, :3] = object_pos_w.unsqueeze(0)
        object_state[:, 3:7] = object_quat_w.unsqueeze(0)
        object_state[:, 7:] = 0.0
        dt = max(float(getattr(self._cfg, "physics_dt", 0.0)), 1.0e-8)
        if self._press_prev_object_pos_w is not None:
            object_state[:, 7:10] = ((object_pos_w - self._press_prev_object_pos_w) / dt).unsqueeze(0)
        prev_object_quat_w = getattr(self, "_press_prev_object_quat_w", None)
        if prev_object_quat_w is not None:
            object_state[:, 10:13] = self._angular_velocity_from_quat_delta(
                prev_object_quat_w,
                object_quat_w,
                dt,
            ).unsqueeze(0)
        self._press_prev_object_pos_w = object_pos_w.detach().clone()
        self._press_prev_object_quat_w = object_quat_w.detach().clone()

        if hasattr(self._plug_obj, "write_root_pose_to_sim") and hasattr(self._plug_obj, "write_root_velocity_to_sim"):
            self._plug_obj.write_root_pose_to_sim(object_state[:, :7])
            self._plug_obj.write_root_velocity_to_sim(object_state[:, 7:])
        else:
            self._plug_obj.write_root_state_to_sim(object_state)

    def _write_pressure_pad_presser_pose(self) -> None:
        if self._pressure_pad_presser_obj is None or self._pressure_pad_presser_body_idx is None:
            return
        if self._pressure_pad_presser_center_l is None or self._pressure_pad_presser_normal_l is None:
            return
        if self._pressure_pad_presser_offset_l is None:
            return
        if self._pressure_pad_presser_start_offset_l is None or self._pressure_pad_presser_end_offset_l is None:
            return
        if self._pressure_pad_presser_rot_l is None or self._pressure_pad_presser_flip_l is None:
            return

        touch_state = self._robot.data.body_link_state_w[0, int(self._pressure_pad_presser_body_idx), :7]
        touch_pos_w = touch_state[:3]
        touch_quat_w = touch_state[3:7]

        steps = int(getattr(self._cfg, "pressure_pad_presser_steps", 0) or self._cfg.press_steps)
        denom = max(1, steps - 1)
        alpha = min(max(float(self._press_counter) / float(denom), 0.0), 1.0)
        trajectory_offset_l = self._pressure_pad_presser_start_offset_l + alpha * (
            self._pressure_pad_presser_end_offset_l - self._pressure_pad_presser_start_offset_l
        )

        object_pos_l = (
            self._pressure_pad_presser_center_l
            + self._pressure_pad_presser_offset_l
            + trajectory_offset_l
        )
        object_pos_w = self._math_utils.quat_apply(touch_quat_w.unsqueeze(0), object_pos_l.unsqueeze(0)).squeeze(0)
        object_pos_w = object_pos_w + touch_pos_w
        object_quat_l = self._math_utils.quat_mul(
            self._pressure_pad_presser_rot_l.unsqueeze(0),
            self._pressure_pad_presser_flip_l.unsqueeze(0),
        ).squeeze(0)
        object_quat_w = self._math_utils.quat_mul(touch_quat_w.unsqueeze(0), object_quat_l.unsqueeze(0)).squeeze(0)
        self._pressure_pad_presser_pose_w = torch.cat([object_pos_w, object_quat_w], dim=0).detach().clone()
        self._write_pressure_pad_presser_state(object_pos_w, object_quat_w)

    def _write_pressure_pad_presser_state(self, object_pos_w, object_quat_w) -> None:
        if self._pressure_pad_presser_obj is None:
            return
        object_state = self._pressure_pad_presser_obj.data.default_root_state.clone()
        if object_state.ndim == 1:
            object_state = object_state.unsqueeze(0)
        object_state[:, :3] = object_pos_w.unsqueeze(0)
        object_state[:, 3:7] = object_quat_w.unsqueeze(0)
        object_state[:, 7:] = 0.0
        dt = max(float(getattr(self._cfg, "physics_dt", 0.0)), 1.0e-8)
        if self._pressure_pad_presser_prev_pos_w is not None:
            object_state[:, 7:10] = ((object_pos_w - self._pressure_pad_presser_prev_pos_w) / dt).unsqueeze(0)
        if self._pressure_pad_presser_prev_quat_w is not None:
            object_state[:, 10:13] = self._angular_velocity_from_quat_delta(
                self._pressure_pad_presser_prev_quat_w,
                object_quat_w,
                dt,
            ).unsqueeze(0)
        self._pressure_pad_presser_prev_pos_w = object_pos_w.detach().clone()
        self._pressure_pad_presser_prev_quat_w = object_quat_w.detach().clone()

        if hasattr(self._pressure_pad_presser_obj, "write_root_pose_to_sim") and hasattr(
            self._pressure_pad_presser_obj,
            "write_root_velocity_to_sim",
        ):
            self._pressure_pad_presser_obj.write_root_pose_to_sim(object_state[:, :7])
            self._pressure_pad_presser_obj.write_root_velocity_to_sim(object_state[:, 7:])
        else:
            self._pressure_pad_presser_obj.write_root_state_to_sim(object_state)

    def _sync_press_object_usd_pose(self, pos_w, quat_w) -> None:
        try:
            from pxr import Gf, UsdGeom

            pos = to_numpy(pos_w, dtype=np.float64, shape=(3,))
            quat = to_numpy(quat_w, dtype=np.float64, shape=(4,))
            prim = self._sim_utils.get_current_stage().GetPrimAtPath("/World/Plug")
            if not prim or not prim.IsValid():
                return
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
            orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            orient_op.Set(Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))))
        except Exception as exc:
            print(f"[WARN] Could not sync /World/Plug USD pose for manual placement: {exc}", flush=True)

    def _read_usd_pose(self, prim_path: str) -> np.ndarray | None:
        try:
            root_prim = self._sim_utils.get_current_stage().GetPrimAtPath(str(prim_path))
            if not root_prim or not root_prim.IsValid():
                return None
            pos, quat = self._sim_utils.resolve_prim_pose(root_prim)
            return np.concatenate(
                [
                    np.asarray(pos, dtype=np.float32).reshape(3),
                    np.asarray(quat, dtype=np.float32).reshape(4),
                ]
            )
        except Exception:
            return None

    def _read_press_object_usd_pose(self) -> np.ndarray | None:
        return self._read_usd_pose("/World/Plug")

    def _read_robot_usd_pose(self) -> np.ndarray | None:
        return self._read_usd_pose(str(self._cfg.robot_prim_path))

    def _press_object_usd_pose_changed(self, pose_w: np.ndarray | None) -> bool:
        if pose_w is None or self._press_manual_drag_baseline_pose_w is None:
            return False
        current = np.asarray(pose_w, dtype=np.float32).reshape(7)
        baseline = np.asarray(self._press_manual_drag_baseline_pose_w, dtype=np.float32).reshape(7)
        return bool(np.max(np.abs(current - baseline)) > 1.0e-6)

    def _update_target_poses(self):
        for sensor, tgt_prim in zip(self._tactile_sensors, self._per_sensor_target_prims):
            if not tgt_prim or not tgt_prim.IsValid():
                continue

            path = tgt_prim.GetPath().pathString
            info = self._dynamic_track_map.get(path)

            if info is not None:
                try:
                    if (
                        self._cfg.enable_press_motion
                        and self._press_object_control_mode() == "scripted"
                        and str(info.rb_path).startswith("/World/Plug")
                        and self._press_object_pose_w is not None
                    ):
                        root_pose = self._press_object_pose_w.detach().cpu().numpy().astype(np.float32)
                        pos, quat = root_pose[:3], root_pose[3:7]
                    elif (
                        self._cfg.enable_press_motion
                        and str(info.rb_path).startswith("/World/Plug")
                        and self._press_object_control_mode() in ("initial_only", "manual_gui")
                    ):
                        root_pose = self._read_press_object_usd_pose()
                        if self._press_manual_drag_baseline_pose_w is None and root_pose is not None:
                            self._press_manual_drag_baseline_pose_w = np.asarray(root_pose, dtype=np.float32).copy()
                        if (
                            self._press_object_control_mode() == "initial_only"
                            and self._press_object_pose_w is not None
                            and not self._press_object_usd_pose_changed(root_pose)
                        ):
                            root_pose = self._press_object_pose_w.detach().cpu().numpy().astype(np.float32)
                        if root_pose is None:
                            raise ValueError("Could not read /World/Plug pose")
                        pos, quat = root_pose[:3], root_pose[3:7]
                    elif (
                        self._cfg.enable_press_motion
                        and bool(getattr(self._cfg, "enable_pressure_pad_presser", False))
                        and str(info.rb_path).startswith(str(getattr(self._cfg, "pressure_pad_presser_prim_path", "")))
                        and self._pressure_pad_presser_pose_w is not None
                    ):
                        root_pose = self._pressure_pad_presser_pose_w.detach().cpu().numpy().astype(np.float32)
                        pos, quat = root_pose[:3], root_pose[3:7]
                    else:
                        pos, quat = self._tracked_target_pose(info)
                except Exception as exc:
                    if not self._warned_target_pose_update:
                        print(f"[WARN] Target pose update fell back to USD pose for {path}: {exc}", flush=True)
                        self._warned_target_pose_update = True
                    pos, quat = self._sim_utils.resolve_prim_pose(tgt_prim)
            else:
                pos, quat = self._sim_utils.resolve_prim_pose(tgt_prim)

            sensor.set_target_pose(pos, quat)

    def _setup_dynamic_tracking(self):
        """Build map: target mesh prim path -> TrackInfo (best-effort)."""
        math_utils = self._math_utils
        sim_utils = self._sim_utils
        UsdPhysics = self._UsdPhysics

        # Try to import RigidPrim (API name varies)
        RigidPrim = None
        for mod in ("omni.isaac.core.prims", "isaacsim.core.prims"):
            try:
                RigidPrim = __import__(mod, fromlist=["RigidPrim"]).RigidPrim
                break
            except ImportError:
                continue

        self._dynamic_track_map = {}
        if RigidPrim is None:
            return

        def make_rigid_prim(path: str):
            """Try common ctor signatures; return constructed RigidPrim or raise last error."""
            candidates = (
                ((), {}),                         # RigidPrim(path) handled outside
                ((), {"prim_path": path}),
                ((), {"path": path}),             # occasionally seen, harmless if wrong
                ((), {"name": path.replace("/", "_")}),
            )

            # First: many versions accept positional path
            try:
                return RigidPrim(path)
            except TypeError:
                pass

            last_err = None
            for _, kwargs in candidates:
                try:
                    # Some candidates above don't include prim_path; add it when needed
                    if "prim_path" not in kwargs and "path" not in kwargs:
                        continue
                    return RigidPrim(**kwargs)
                except TypeError as e:
                    last_err = e
                    continue

            # Fallback (most common keyword)
            if last_err is not None:
                return RigidPrim(prim_path=path)
            return RigidPrim(prim_path=path)

        def to_t(x):
            return torch.tensor(x, device=self._device, dtype=torch.float32)

        for query_path in self._per_sensor_target_query_paths:
            if not query_path:
                continue

            prim = self._stage.GetPrimAtPath(query_path)
            if not prim.IsValid():
                continue

            # Find parent rigid body
            curr = prim
            rb_prim = None
            while curr.IsValid() and not curr.IsPseudoRoot():
                if curr.HasAPI(UsdPhysics.RigidBodyAPI) or curr.HasAPI(UsdPhysics.MassAPI):
                    rb_prim = curr
                    break
                curr = curr.GetParent()
            if rb_prim is None:
                continue

            rb_path = rb_prim.GetPath().pathString

            # Single best-effort try/except per query_path
            try:
                rp = make_rigid_prim(rb_path)
                if hasattr(rp, "initialize"):
                    rp.initialize()

                p_m, q_m = sim_utils.resolve_prim_pose(prim)
                p_b, q_b = sim_utils.resolve_prim_pose(rb_prim)

                q_b_inv = math_utils.quat_inv(to_t(q_b))
                p_rel = math_utils.quat_apply(q_b_inv, to_t(p_m) - to_t(p_b))
                q_rel = math_utils.quat_mul(q_b_inv, to_t(q_m))

                self._dynamic_track_map[query_path] = TrackInfo(
                    rp=rp, p_rel=p_rel, q_rel=q_rel, rb_path=rb_path
                )

            except Exception as e:
                print(f"[WARN] Dynamic tracking init failed for {rb_path}: {e}", flush=True)

        if self._dynamic_track_map:
            tracked = ", ".join(f"{path}->{info.rb_path}" for path, info in self._dynamic_track_map.items())
            print(f"[INFO] Dynamic target tracking: {tracked}", flush=True)


    def _tracked_target_pose(self, info: TrackInfo):
        """Return (pos, quat) numpy for the tracked target pose. Raises on failure."""
        pos_b, quat_b = _rigid_prim_world_pose(info.rp)
        return self._tracked_target_pose_from_root(info, np.concatenate([pos_b, quat_b]).astype(np.float32))

    def _tracked_target_pose_from_root(self, info: TrackInfo, root_pose: np.ndarray):
        """Return target mesh pose from a rigid-body root pose and cached mesh offset."""
        root_pose = np.asarray(root_pose, dtype=np.float32)
        if root_pose.shape != (7,) or float(np.linalg.norm(root_pose[3:7])) < 0.5:
            raise ValueError("Invalid rigid body root pose.")
        pos_b_t = torch.tensor(root_pose[:3], device=self._device, dtype=torch.float32)
        quat_b_t = torch.tensor(root_pose[3:7], device=self._device, dtype=torch.float32)

        pos_t = pos_b_t + self._math_utils.quat_apply(quat_b_t, info.p_rel)
        quat_t = self._math_utils.quat_mul(quat_b_t, info.q_rel)

        return pos_t.detach().cpu().numpy(), quat_t.detach().cpu().numpy()


    def _save_render(self, rgb: np.ndarray | None, step: int):
        if rgb is None or not self._render_output_dir:
            return
        try:
            from PIL import Image
            Image.fromarray(rgb).save(os.path.join(self._render_output_dir, f"frame_{step:06d}.png"))
        except ImportError:
            np.save(os.path.join(self._render_output_dir, f"frame_{step:06d}.npy"), rgb)

    def _init_sim(self, SimulationContext, cfg: AlohaTactileEnvCfg) -> None:
        """Create simulation context and cache device."""
        try:
            self._sim = SimulationContext(
                physics_dt=cfg.physics_dt,
                rendering_dt=cfg.physics_dt,
                backend="torch",
                device=cfg.device,
            )
        except TypeError:
            sim_cfg = self._sim_utils.SimulationCfg(dt=cfg.physics_dt, render_interval=1, device=cfg.device)
            self._sim = self._sim_utils.SimulationContext(sim_cfg)
        self._device = cfg.device


    def _spawn_basic_world(self, sim_utils) -> None:
        """Spawn ground plane and dome light."""
        # Ground plane
        sim_utils.spawn_mesh_cuboid(
            prim_path="/World/defaultGroundPlane",
            cfg=sim_utils.MeshCuboidCfg(
                size=(10.0, 10.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.004, rest_offset=0.0
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True
                ),
            ),
            translation=(0.0, 0.0, -0.05),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # Dome light
        sim_utils.spawn_light(
            prim_path="/World/Light/DomeLight",
            cfg=sim_utils.DomeLightCfg(intensity=2000),
            translation=(-4.5, 3.5, 10.0),
        )

    def _setup_camera(self, cfg: AlohaTactileEnvCfg, sim_utils):
        """Create camera (optional) and set render output dir (optional)."""
        self._camera = None
        self._render_output_dir = None

        should_set_viewport = bool(getattr(cfg, "set_viewport_camera", False)) and not bool(cfg.headless)
        if should_set_viewport or cfg.enable_camera:
            from isaacsim.core.utils.viewports import set_camera_view

            eye = getattr(cfg, "viewport_camera_eye", cfg.camera_eye) if should_set_viewport else cfg.camera_eye
            target = getattr(cfg, "viewport_camera_target", cfg.camera_target) if should_set_viewport else cfg.camera_target
            set_camera_view(list(eye), list(target))

        if not cfg.enable_camera:
            return

        # Deferred (camera modules rely on Isaac)
        from isaaclab.sensors.camera import Camera, CameraCfg

        cam_rot = _look_at_quat(cfg.camera_eye, cfg.camera_target)

        self._camera = Camera(CameraCfg(
            prim_path=cfg.camera_prim_path,
            update_period=0.0,
            height=cfg.camera_height,
            width=cfg.camera_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=cfg.camera_eye,
                rot=cam_rot,
                convention="world",
            ),
        ))

        if cfg.save_renders:
            self._render_output_dir = cfg.render_output_dir or os.path.join(
                os.path.dirname(__file__), "output", "renders"
            )
            os.makedirs(self._render_output_dir, exist_ok=True)


    def _ensure_press_usd_physics_schemas(
        self,
        prim_path: str,
        cfg: AlohaTactileEnvCfg,
        sim_utils,
        *,
        collision_enabled: bool | None = None,
    ) -> None:
        """Make a lightweight press USD usable as an IsaacLab RigidObject."""

        from isaaclab.sim.schemas import (
            define_collision_properties,
            define_mass_properties,
            define_rigid_body_properties,
        )
        from pxr import UsdGeom, UsdPhysics

        stage = sim_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(str(prim_path))
        if not root_prim or not root_prim.IsValid():
            raise RuntimeError(f"Press object prim was not spawned: {prim_path}")

        rigid_cfg = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=bool(cfg.enable_press_motion),
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
        )
        define_rigid_body_properties(str(prim_path), rigid_cfg, stage=stage)
        define_mass_properties(
            str(prim_path),
            sim_utils.MassPropertiesCfg(mass=float(cfg.press_object_mass)),
            stage=stage,
        )

        collision_cfg = sim_utils.CollisionPropertiesCfg(
            collision_enabled=(
                bool(getattr(cfg, "press_object_collision_enabled", True))
                if collision_enabled is None
                else bool(collision_enabled)
            ),
            contact_offset=float(cfg.press_object_contact_offset),
            rest_offset=float(cfg.press_object_rest_offset),
        )
        mesh_prims = []
        if root_prim.IsA(UsdGeom.Mesh):
            mesh_prims.append(root_prim)
        mesh_prims.extend(sim_utils.get_all_matching_child_prims(
            str(prim_path),
            predicate=lambda p: p.IsA(UsdGeom.Mesh),
            traverse_instance_prims=True,
        ))
        seen_mesh_paths = set()
        for mesh_prim in mesh_prims:
            mesh_path = mesh_prim.GetPath().pathString
            if mesh_path in seen_mesh_paths:
                continue
            seen_mesh_paths.add(mesh_path)
            define_collision_properties(mesh_path, collision_cfg, stage=stage)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)

    def _spawn_plug_socket(self, cfg: AlohaTactileEnvCfg, sim_utils, RigidObject, RigidObjectCfg):
        """Optionally spawn plug and/or socket RigidObjects. Returns (plug, socket)."""
        plug_obj = None
        socket_obj = None

        # Always ensure output dir exists (used by USD conversion)
        objs_out_dir = os.path.join(os.path.dirname(__file__), "output", "automate_scaled_urdf")
        os.makedirs(objs_out_dir, exist_ok=True)

        if not (cfg.enable_plug or cfg.enable_socket or bool(getattr(cfg, "enable_pressure_pad_presser", False))):
            return plug_obj, socket_obj

        automate_dir = os.path.join(os.path.expanduser(cfg.asset_root), "automate_scaled", "urdf")
        plug_urdf = os.path.join(automate_dir, f"{cfg.automate_asset_id}_plug.urdf")
        socket_urdf = os.path.join(automate_dir, f"{cfg.automate_asset_id}_socket.urdf")

        def spawn_press_usd(
            prim_path: str,
            usd_path: str,
            pose,
            scale: float,
            *,
            collision_enabled: bool | None = None,
        ):
            press_usd = os.path.expanduser(str(usd_path))
            if not os.path.isfile(press_usd):
                raise FileNotFoundError(f"Press object USD not found: {press_usd}")
            pos = tuple(float(v) for v in pose[:3])
            rot = _xyzw_to_wxyz(pose[3:7])
            obj = RigidObject(RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=press_usd,
                    scale=(scale, scale, scale),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            ))
            self._ensure_press_usd_physics_schemas(
                prim_path,
                cfg,
                sim_utils,
                collision_enabled=collision_enabled,
            )
            return obj

        def spawn_one(urdf_path: str, pose, prim_path: str, scale: float, fix_base: bool, collider_type: str):
            press_usd = os.path.expanduser(str(getattr(cfg, "press_object_usd_path", "")))
            if prim_path == "/World/Plug" and cfg.enable_press_motion and press_usd:
                return spawn_press_usd(prim_path, press_usd, pose, scale)

            if not os.path.isfile(urdf_path):
                return None
            pos = tuple(float(v) for v in pose[:3])
            rot = _xyzw_to_wxyz(pose[3:7])
            return RigidObject(RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=urdf_path,
                    scale=(scale,) * 3 if scale != 1.0 else None,
                    fix_base=fix_base,
                    joint_drive=None,
                    link_density=1000.0,
                    usd_dir=objs_out_dir,
                    force_usd_conversion=cfg.force_objects_urdf_conversion,
                    collider_type=collider_type,
                    activate_contact_sensors=False,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            ))

        if cfg.enable_plug:
            plug_obj = spawn_one(
                plug_urdf, cfg.plug_default_pose, "/World/Plug",
                cfg.plug_scale, cfg.plug_fix_base, cfg.plug_collider_type,
            )

        if cfg.enable_socket:
            socket_obj = spawn_one(
                socket_urdf, cfg.socket_default_pose, "/World/Socket",
                cfg.socket_scale, cfg.socket_fix_base, cfg.socket_collider_type,
            )

        if bool(getattr(cfg, "enable_pressure_pad_presser", False)):
            prim_path = str(getattr(cfg, "pressure_pad_presser_prim_path", "/World/PressurePadPresser"))
            press_usd = str(getattr(cfg, "pressure_pad_presser_usd_path", "") or getattr(cfg, "press_object_usd_path", ""))
            scale = float(getattr(cfg, "pressure_pad_presser_scale", getattr(cfg, "plug_scale", 1.0)))
            spawn_pose = tuple(float(value) for value in cfg.plug_default_pose)
            spawn_pose = (spawn_pose[0], spawn_pose[1], spawn_pose[2] + 10.0, *spawn_pose[3:7])
            self._pressure_pad_presser_obj = spawn_press_usd(
                prim_path,
                press_usd,
                spawn_pose,
                scale,
                collision_enabled=False,
            )
            print(
                f"[INFO] pressure-pad presser spawned: prim={prim_path}, usd={press_usd}, "
                f"scale={scale:g}, collision=off, safe_spawn_pose={spawn_pose}",
                flush=True,
            )

        return plug_obj, socket_obj


    def _spawn_robot(
        self,
        cfg: AlohaTactileEnvCfg,
        urdf_path: str,
        sim_utils,
        Articulation,
        ArticulationCfg,
        ImplicitActuatorCfg,
        UrdfConverterCfg,
    ):
        """Spawn the ALOHA articulation from URDF and return the Articulation."""
        out_dir = cfg.usd_output_dir or os.path.join(os.path.dirname(__file__), "output", "aloha_urdf")
        os.makedirs(out_dir, exist_ok=True)

        robot = Articulation(ArticulationCfg(
            prim_path=cfg.robot_prim_path,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=urdf_path,
                fix_base=cfg.fix_base,
                merge_fixed_joints=cfg.merge_fixed_joints,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=cfg.urdf_drive_stiffness,
                        damping=cfg.urdf_drive_damping,
                    )
                ),
                usd_dir=out_dir,
                force_usd_conversion=cfg.force_urdf_conversion,
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(float(v) for v in cfg.robot_init_pos),
                rot=tuple(float(v) for v in cfg.robot_init_rot_wxyz),
            ),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=cfg.urdf_drive_stiffness,
                    damping=cfg.urdf_drive_damping,
                )
            },
        ))
        return robot

    def _apply_pressure_pad_visual_materials(self, cfg: AlohaTactileEnvCfg, urdf_path: str, sim_utils) -> None:
        """Color pressure-pad visuals and matching outer skin visuals in the live stage."""

        taxel_markers_enabled = bool(getattr(cfg, "show_pressure_pad_taxel_points", False))
        visual_links = _pressure_pad_visual_links_from_urdf(urdf_path)
        if not visual_links and not taxel_markers_enabled:
            return

        try:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
        except Exception as exc:
            print(f"[WARN] Could not import USD material APIs for pressure-pad visuals: {exc}", flush=True)
            return

        stage = sim_utils.get_current_stage()
        robot_path = str(cfg.robot_prim_path).rstrip("/")
        try:
            sim_utils.make_uninstanceable(robot_path)
        except Exception as exc:
            print(f"[WARN] Could not make {robot_path} uninstanceable before pressure-pad visual binding: {exc}", flush=True)

        def _iter_descendants(root_prim):
            stack = list(root_prim.GetChildren())
            while stack:
                prim = stack.pop()
                yield prim
                stack.extend(list(prim.GetChildren()))

        def _bind_mesh_visuals(root_prim, material, color: Gf.Vec3f) -> int:
            mesh_count = 0
            for prim in _iter_descendants(root_prim):
                if prim.GetTypeName() != "Mesh":
                    continue
                try:
                    prim.CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
                    prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([color])
                    prim.CreateAttribute("primvars:displayColor:interpolation", Sdf.ValueTypeNames.Token).Set("constant")
                    prim.CreateAttribute("primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray).Set([1.0])
                    prim.CreateAttribute("primvars:displayOpacity:interpolation", Sdf.ValueTypeNames.Token).Set("constant")
                    UsdGeom.Imageable(prim).MakeVisible()
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    )
                    mesh_count += 1
                except Exception as exc:
                    print(f"[WARN] Could not force visual mesh render attrs on {prim.GetPath()}: {exc}", flush=True)
            return mesh_count

        robot_gray_path = Sdf.Path("/World/Looks/revo21_robot_visual_gray")
        robot_gray = UsdShade.Material.Define(stage, robot_gray_path)
        robot_gray_shader = UsdShade.Shader.Define(stage, robot_gray_path.AppendPath("PreviewSurface"))
        robot_gray_shader.CreateIdAttr("UsdPreviewSurface")
        robot_gray_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.82, 0.84, 0.86))
        robot_gray_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.08, 0.08, 0.08))
        robot_gray_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        robot_gray_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        robot_gray_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        robot_gray.CreateSurfaceOutput().ConnectToSource(robot_gray_shader.ConnectableAPI(), "surface")

        material_path = Sdf.Path("/World/Looks/pressure_pad_cyan")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendPath("PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.85, 1.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.08, 0.10))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        forced_robot_meshes = 0
        robot_prim = stage.GetPrimAtPath(robot_path)
        if robot_prim.IsValid():
            for prim in _iter_descendants(robot_prim):
                path = prim.GetPath().pathString
                if prim.GetTypeName() == "Mesh" and "/visuals/" in path:
                    try:
                        prim.CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
                        prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
                            [Gf.Vec3f(0.82, 0.84, 0.86)]
                        )
                        prim.CreateAttribute("primvars:displayColor:interpolation", Sdf.ValueTypeNames.Token).Set(
                            "constant"
                        )
                        prim.CreateAttribute("primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray).Set([1.0])
                        prim.CreateAttribute("primvars:displayOpacity:interpolation", Sdf.ValueTypeNames.Token).Set(
                            "constant"
                        )
                        UsdGeom.Imageable(prim).MakeVisible()
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                            robot_gray,
                            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                        )
                        forced_robot_meshes += 1
                    except Exception as exc:
                        print(f"[WARN] Could not force robot visual mesh render attrs on {path}: {exc}", flush=True)
        print(
            f"[INFO] robot visual mesh render attrs forced: {forced_robot_meshes} meshes material={robot_gray_path}",
            flush=True,
        )

        bound: list[str] = []
        missing: list[str] = []
        mesh_bound = 0
        for link_name in visual_links:
            visual_path = f"{robot_path}/{link_name}/visuals"
            visual_prim = stage.GetPrimAtPath(visual_path)
            if not visual_prim.IsValid():
                visual_path = f"{robot_path}/{link_name}"
                visual_prim = stage.GetPrimAtPath(visual_path)
            if not visual_prim.IsValid():
                missing.append(link_name)
                continue
            UsdShade.MaterialBindingAPI.Apply(visual_prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            mesh_bound += _bind_mesh_visuals(visual_prim, material, Gf.Vec3f(0.0, 0.85, 1.0))
            bound.append(visual_path)

        print(
            f"[INFO] pressure pad visual material applied: {len(bound)}/{len(visual_links)} links "
            f"mesh_prims={mesh_bound} material={material_path}",
            flush=True,
        )
        if missing:
            print(f"[WARN] pressure pad visual links missing in stage: {missing}", flush=True)

        target_links: list[str] = []
        press_link = str(getattr(cfg, "press_touch_link", ""))
        if cfg.enable_press_motion and press_link:
            target_links.append(press_link)
            if press_link.endswith("_touch_link"):
                target_links.append(press_link.replace("_touch_link", "_rubber_link"))
                target_links.append(press_link.replace("_touch_link", "_tubber_link"))
        target_links = [name for name in target_links if name in visual_links]
        if target_links:
            target_material_path = Sdf.Path("/World/Looks/pressure_pad_target_yellow")
            target_material = UsdShade.Material.Define(stage, target_material_path)
            target_shader = UsdShade.Shader.Define(stage, target_material_path.AppendPath("PreviewSurface"))
            target_shader.CreateIdAttr("UsdPreviewSurface")
            target_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.85, 0.0))
            target_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.12, 0.10, 0.0))
            target_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
            target_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.30)
            target_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            target_material.CreateSurfaceOutput().ConnectToSource(target_shader.ConnectableAPI(), "surface")
            target_bound = []
            target_mesh_bound = 0
            for link_name in target_links:
                visual_path = f"{robot_path}/{link_name}/visuals"
                visual_prim = stage.GetPrimAtPath(visual_path)
                if not visual_prim.IsValid():
                    visual_path = f"{robot_path}/{link_name}"
                    visual_prim = stage.GetPrimAtPath(visual_path)
                if not visual_prim.IsValid():
                    continue
                UsdShade.MaterialBindingAPI.Apply(visual_prim).Bind(
                    target_material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )
                target_mesh_bound += _bind_mesh_visuals(visual_prim, target_material, Gf.Vec3f(1.0, 0.85, 0.0))
                target_bound.append(visual_path)
            print(
                f"[INFO] selected pressure pad visual material applied: {target_bound} "
                f"mesh_prims={target_mesh_bound} material={target_material_path}",
                flush=True,
            )
        if taxel_markers_enabled:
            taxel_layout_urdf = str(getattr(cfg, "pressure_pad_taxel_layout_urdf", None) or urdf_path)
            taxel_points_by_link = _pressure_pad_taxel_points_by_link_from_urdf(taxel_layout_urdf)
            taxel_material_path = Sdf.Path("/World/Looks/pressure_pad_taxel_yellow")
            taxel_material = UsdShade.Material.Define(stage, taxel_material_path)
            taxel_shader = UsdShade.Shader.Define(stage, taxel_material_path.AppendPath("PreviewSurface"))
            taxel_shader.CreateIdAttr("UsdPreviewSurface")
            taxel_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.85, 0.0))
            taxel_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.20, 0.12, 0.0))
            taxel_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
            taxel_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
            taxel_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            taxel_material.CreateSurfaceOutput().ConnectToSource(taxel_shader.ConnectableAPI(), "surface")
            marker_color = Gf.Vec3f(1.0, 0.85, 0.0)
            radius = float(getattr(cfg, "pressure_pad_taxel_point_radius", 0.00045))
            marked_links = 0
            marked_points = 0
            for link_name, points_l in sorted(taxel_points_by_link.items()):
                link_path = Sdf.Path(f"{robot_path}/{link_name}")
                link_prim = stage.GetPrimAtPath(link_path)
                if not link_prim.IsValid():
                    print(f"[WARN] pressure pad taxel link missing in stage: {link_name}", flush=True)
                    continue
                group_path = link_path.AppendPath("pressure_pad_taxel_points_yellow")
                group = UsdGeom.Xform.Define(stage, group_path)
                UsdGeom.Imageable(group.GetPrim()).MakeVisible()
                for point_index, point_l in enumerate(points_l):
                    marker_path = group_path.AppendPath(f"taxel_{point_index:04d}")
                    marker = UsdGeom.Sphere.Define(stage, marker_path)
                    marker.GetRadiusAttr().Set(radius)
                    marker_xform = UsdGeom.Xformable(marker.GetPrim())
                    marker_xform.ClearXformOpOrder()
                    marker_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*point_l))
                    gprim = UsdGeom.Gprim(marker.GetPrim())
                    gprim.CreateDisplayColorAttr(Vt.Vec3fArray([marker_color]))
                    UsdGeom.Imageable(marker.GetPrim()).MakeVisible()
                    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(
                        taxel_material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    )
                    marked_points += 1
                marked_links += 1
            print(
                f"[INFO] pressure pad yellow taxel markers applied: "
                f"{marked_points} points across {marked_links}/{len(taxel_points_by_link)} links "
                f"layout_urdf={taxel_layout_urdf}",
                flush=True,
            )
        center_markers_enabled = bool(getattr(cfg, "show_pressure_pad_centers", False))
        if center_markers_enabled:
            center_links = _pressure_pad_center_links_from_urdf(urdf_path)
            surface_axis_by_link = _pressure_pad_surface_axis_by_link_from_urdf(urdf_path)
            origin_by_link = _pressure_pad_origin_by_link_from_urdf(urdf_path)
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True,
            )
            marker_color = Gf.Vec3f(1.0, 0.0, 0.85)
            radius = float(getattr(cfg, "pressure_pad_center_marker_radius", 0.003))
            marked: list[tuple[str, str]] = []
            for link_name in center_links:
                visual_path = f"{robot_path}/{link_name}/visuals"
                visual_prim = stage.GetPrimAtPath(visual_path)
                if not visual_prim.IsValid():
                    visual_path = f"{robot_path}/{link_name}"
                    visual_prim = stage.GetPrimAtPath(visual_path)
                if not visual_prim.IsValid():
                    continue
                try:
                    local_box = bbox_cache.ComputeLocalBound(visual_prim).ComputeAlignedBox()
                    local_min, local_max = local_box.GetMin(), local_box.GetMax()
                    normal_axis, normal_sign = surface_axis_by_link.get(link_name, (0, 1.0))
                    explicit_origin_l = origin_by_link.get(link_name)
                    if explicit_origin_l is None:
                        center_l = Gf.Vec3d(
                            *_bbox_surface_center(
                                local_min,
                                local_max,
                                normal_axis=normal_axis,
                                normal_sign=normal_sign,
                            )
                        )
                        center_source = "bbox_surface"
                    else:
                        center_l = Gf.Vec3d(*explicit_origin_l)
                        center_source = "pressure_pad_origin"
                    marker_path = Sdf.Path(f"{robot_path}/{link_name}").AppendPath("pressure_pad_center_marker")
                    marker = UsdGeom.Sphere.Define(stage, marker_path)
                    marker.GetRadiusAttr().Set(radius)
                    marker_xform = UsdGeom.Xformable(marker.GetPrim())
                    marker_xform.ClearXformOpOrder()
                    marker_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(center_l)
                    UsdGeom.Gprim(marker.GetPrim()).CreateDisplayColorAttr(Vt.Vec3fArray([marker_color]))
                    UsdGeom.Imageable(marker.GetPrim()).MakeVisible()
                    center_w_np, _ = self._sim_utils.resolve_prim_pose(marker.GetPrim())
                    center_w = [float(value) for value in center_w_np[:3]]
                    print(
                        f"[PRESSURE_PAD_CENTER] {link_name} "
                        f"surface_xyz=({center_w[0]:.6f}, {center_w[1]:.6f}, {center_w[2]:.6f}) "
                        f"normal_axis={int(normal_axis)} normal_sign={float(normal_sign):.1f} "
                        f"source={center_source}",
                        flush=True,
                    )
                    marker_path_text = marker_path.pathString
                    marked.append((link_name, marker_path_text))
                except Exception as exc:
                    print(f"[WARN] Could not mark pressure pad center for {link_name}: {exc}", flush=True)
            print(f"[INFO] pressure pad center markers applied: {len(marked)}/{len(center_links)}", flush=True)

    def _apply_touch_contact_materials(self, cfg: AlohaTactileEnvCfg, sim_utils) -> None:
        """Bind compliant contact material to Revo touch surface collision prims."""
        if not bool(getattr(cfg, "enable_touch_compliant_material", True)):
            return

        material_cfg = sim_utils.RigidBodyMaterialCfg(
            compliant_contact_stiffness=float(cfg.compliant_contact_stiffness),
            compliant_contact_damping=float(cfg.compliant_contact_damping),
        )
        collision_cfg = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=float(cfg.touch_contact_offset),
            rest_offset=float(cfg.touch_rest_offset),
        )

        robot_path = str(cfg.robot_prim_path).rstrip("/")
        try:
            sim_utils.make_uninstanceable(robot_path)
        except Exception as exc:
            print(f"[WARN] Could not make {robot_path} uninstanceable before material binding: {exc}", flush=True)

        for rel_path in cfg.touch_collision_paths:
            collision_path = f"{robot_path}/{str(rel_path).lstrip('/')}"
            material_path = f"{collision_path}/compliant_material"
            try:
                material_cfg.func(material_path, material_cfg)
                sim_utils.modify_collision_properties(collision_path, collision_cfg)
                sim_utils.bind_physics_material(collision_path, material_path)
                print(
                    f"[INFO] touch compliant material applied: {collision_path}, "
                    f"stiffness={cfg.compliant_contact_stiffness}, damping={cfg.compliant_contact_damping}, "
                    f"contact_offset={cfg.touch_contact_offset}, rest_offset={cfg.touch_rest_offset}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[WARN] Could not bind touch compliant material on {collision_path}: {exc}", flush=True)


    def _find_elastomer_links(self, cfg: AlohaTactileEnvCfg, sim_utils, UsdPhysics, PhysxSchema) -> list[str]:
        """Return tactile surface link prim paths (strings)."""
        bodies = sim_utils.get_all_matching_child_prims(
            cfg.robot_prim_path,
            predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAPI(PhysxSchema.PhysxContactReportAPI),
            traverse_instance_prims=False,
        )
        keywords = tuple(str(k).lower() for k in getattr(cfg, "tactile_link_keywords", ("elastomer",)))
        tactile_links = sorted(
            [
                p.GetPath().pathString
                for p in bodies
                if any(keyword in p.GetPath().pathString.lower() for keyword in keywords)
            ]
        )
        if not tactile_links:
            raise RuntimeError(f"No tactile links found on robot. Tried keywords: {keywords}")
        return tactile_links

    def _sort_elastomer_links(self, elastomers: list[str]) -> list[str]:
        """Sort elastomer paths into deterministic arm/finger order."""
        def sort_key(path: str):
            slot = _sensor_slot(path)
            if slot is not None:
                return (0, slot, path)

            arm = _infer_arm(path)
            arm_order = 0 if arm == "left" else 1

            finger = _infer_finger(path)
            finger_order = 0 if finger == "left_finger" else 1

            return (1, arm_order, finger_order, path)

        return sorted(elastomers, key=sort_key)
    
    def _build_target_query_paths(
        self,
        cfg: AlohaTactileEnvCfg,
        selected_links: list[str],
        prim_utils,
        sim_utils,
    ) -> tuple[list[str], list[str]]:
        """Return (target_root_paths, target_query_paths) per selected elastomer link."""
        left_target = cfg.left_arm_target_mesh_prim if self._socket_obj else None
        right_target = cfg.right_arm_target_mesh_prim if self._plug_obj else None
        pressure_pad_presser_target = (
            str(getattr(cfg, "pressure_pad_presser_prim_path", "/World/PressurePadPresser"))
            if bool(getattr(cfg, "enable_pressure_pad_presser", False)) and self._pressure_pad_presser_obj is not None
            else None
        )
        pressure_pad_presser_link = str(getattr(cfg, "pressure_pad_presser_link", "")).strip()

        target_root_paths: list[str] = []
        for link_path in selected_links:
            arm = _infer_arm(link_path)
            if (
                pressure_pad_presser_target
                and pressure_pad_presser_link
                and pressure_pad_presser_link in str(link_path)
            ):
                target_root_paths.append(pressure_pad_presser_target)
            elif arm == "left" and left_target:
                target_root_paths.append(left_target)
            elif arm == "right" and right_target:
                target_root_paths.append(right_target)
            else:
                target_root_paths.append(cfg.robot_prim_path)

        target_query_paths: list[str] = []
        for root in target_root_paths:
            try:
                qp, _ = _resolve_mesh_prim(root, prim_utils=prim_utils, sim_utils=sim_utils)
            except RuntimeError:
                qp = root
            target_query_paths.append(qp)

        return target_root_paths, target_query_paths

    def _compute_patch_transform(self, link_path: str, cfg: AlohaTactileEnvCfg, math_utils):
        """Compute patch offset pos/quat in body frame for a given elastomer link."""
        lp = link_path.lower()
        patch_spec = self._press_link_patch_spec(link_path, cfg)
        if (cfg.enable_press_motion or cfg.enable_sample_point_view) and patch_spec is not None:
            center_l, normal_l, is_primary = patch_spec
            if is_primary and getattr(cfg, "press_patch_pos_l", None) is not None:
                patch_pos_l = tuple(float(v) for v in cfg.press_patch_pos_l)
            else:
                patch_pos_l = tuple(float(v) for v in center_l)
            if is_primary and getattr(cfg, "press_patch_quat_l", None) is not None:
                patch_quat = tuple(float(v) for v in cfg.press_patch_quat_l)
                return patch_pos_l, patch_quat
            axis = np.zeros(3, dtype=np.float64)
            axis[int(cfg.normal_axis)] = 1.0
            patch_quat = self._quat_from_vectors(axis, normal_l)
            return patch_pos_l, patch_quat

        if "_left_finger_link" in lp:
            side = "left"
        elif "_right_finger_link" in lp:
            side = "right"
        else:
            return cfg.patch_offset_pos, cfg.patch_offset_quat

        base_xyz, base_rpy = self._urdf_origins[side]

        base_xyz_t = torch.tensor(base_xyz, dtype=torch.float32).unsqueeze(0)
        base_rpy_t = torch.tensor(base_rpy, dtype=torch.float32).unsqueeze(0)
        user_pos_t = torch.tensor(cfg.patch_offset_pos, dtype=torch.float32).unsqueeze(0)
        user_quat_t = torch.tensor(cfg.patch_offset_quat, dtype=torch.float32).unsqueeze(0)

        q_be = math_utils.quat_from_euler_xyz(base_rpy_t[:, 0], base_rpy_t[:, 1], base_rpy_t[:, 2])
        pos_bp = (base_xyz_t + math_utils.quat_apply(q_be, user_pos_t)).squeeze(0)
        quat_bp = math_utils.quat_mul(q_be, user_quat_t).squeeze(0)

        return tuple(float(v) for v in pos_bp), tuple(float(v) for v in quat_bp)

    def _press_link_patch_spec(
        self,
        link_path: str,
        cfg: AlohaTactileEnvCfg,
    ) -> tuple[np.ndarray, np.ndarray, bool] | None:
        lp = str(link_path).lower()
        primary = str(getattr(cfg, "press_touch_link", "")).strip()
        primary_lower = primary.lower()
        if primary_lower and primary_lower in lp:
            center_l, normal_l = self._load_press_center_and_normal(cfg)
            return center_l, normal_l, True

        for entry in getattr(cfg, "press_link_patch_specs", ()) or ():
            if len(entry) != 3:
                continue
            link_name, center_values, normal_values = entry
            if str(link_name).strip().lower() not in lp:
                continue
            center = np.asarray(center_values, dtype=np.float64)
            normal = np.asarray(normal_values, dtype=np.float64)
            norm = np.linalg.norm(normal)
            if center.shape != (3,) or normal.shape != (3,) or norm < 1.0e-12:
                continue
            return center.astype(np.float32), (normal / norm).astype(np.float32), False
        return None

    def _create_tactile_sensors(
        self,
        cfg: AlohaTactileEnvCfg,
        selected_links: list[str],
        target_root_paths: list[str],
        target_query_paths: list[str],
        WarpSdfTactileSensor,
        WarpSdfTactileSensorCfg,
        math_utils,
    ):
        """Create tactile sensors and return (sensors, slot_order)."""
        sensors: list = []
        slot_order: list[int] = []
        self._physx_contact_pressure_sensors = []

        for i, link_path in enumerate(selected_links):
            patch_pos, patch_quat = self._compute_patch_transform(link_path, cfg, math_utils)

            sensor_cfg = WarpSdfTactileSensorCfg(
                prim_path=cfg.robot_prim_path,
                elastomer_prim_paths=[link_path],
                num_rows=cfg.num_rows,
                num_cols=cfg.num_cols,
                point_distance=cfg.point_distance,
                row_distance=cfg.row_distance,
                col_distance=cfg.col_distance,
                normal_axis=cfg.normal_axis,
                normal_offset=cfg.normal_offset,
                normal_sign=cfg.normal_sign,
                patch_offset_pos_b=patch_pos,
                patch_offset_quat_b=patch_quat,
                target_mesh_prim_path=target_query_paths[i],
                mesh_max_dist=cfg.mesh_max_dist,
                mesh_use_signed_distance=cfg.mesh_signed,
                mesh_signed_distance_method=cfg.mesh_signed_distance_method,
                mesh_smooth_normals=True,
                mesh_shell_thickness=cfg.mesh_shell_thickness,
                mesh_unsigned_shell_as_contact=cfg.mesh_unsigned_shell_as_contact,
                mesh_unsigned_contact_mode=cfg.mesh_unsigned_contact_mode,
                penetration_deadband=cfg.penetration_deadband,
                pressure_response_model=cfg.pressure_response_model,
                stiffness=cfg.stiffness,
                damping=cfg.damping,
                max_force=cfg.max_force,
                pressure_damping_dt=float(cfg.physics_dt),
                pressure_gain=cfg.pressure_gain,
                pressure_bias=cfg.pressure_bias,
                pressure_gamma=cfg.pressure_gamma,
                pressure_threshold=cfg.pressure_threshold,
                taxel_area=cfg.taxel_area,
                normalize_forces=True,
                debug_vis=cfg.debug_vis,
                debug_vis_env_id=0,
                debug_vis_point_radius=cfg.debug_vis_point_radius,
                debug_vis_force_threshold=cfg.debug_vis_force_threshold,
                debug_vis_show_all_taxels=cfg.debug_vis_show_all_taxels,
                debug_vis_show_axes=cfg.debug_vis_show_axes,
                debug_vis_axes_scale=cfg.debug_vis_axes_scale,
            )
            sensors.append(WarpSdfTactileSensor(sensor_cfg))

            slot = i if (cfg.enable_press_motion or cfg.enable_sample_point_view) else _sensor_slot(link_path)
            if slot is None or slot in slot_order:
                slot = len(slot_order)
            slot_order.append(slot)

            if bool(getattr(cfg, "enable_physx_contact_force_map", False)):
                contact_cfg = self._ContactSensorCfg(
                    prim_path=str(link_path),
                    update_period=0.0,
                    history_length=0,
                    debug_vis=False,
                    track_pose=True,
                    track_contact_points=False,
                    track_friction_forces=False,
                    max_contact_data_count_per_prim=int(cfg.physx_contact_max_data_count_per_prim),
                    filter_prim_paths_expr=[str(target_root_paths[i])],
                )
                self._physx_contact_pressure_sensors.append(self._ContactSensor(contact_cfg))

        return sensors, slot_order

    def _post_spawn_init(self, cfg: AlohaTactileEnvCfg, sim_utils, target_query_paths: list[str]) -> None:
        """Final initialization after spawning robot/objects/sensors."""
        # Reset simulation (triggers sensor PLAY callbacks)
        self._sim.reset()
        dt = cfg.physics_dt

        # First update pass so buffers are valid
        self._robot.update(dt)
        if self._plug_obj:
            self._plug_obj.update(dt)
        if self._socket_obj:
            self._socket_obj.update(dt)
        if self._pressure_pad_presser_obj:
            self._pressure_pad_presser_obj.update(dt)

        self._setup_press_motion(cfg)
        self._setup_pressure_pad_presser_motion(cfg)
        if cfg.enable_press_motion:
            self._press_counter = 0
            self._press_prev_object_pos_w = None
            self._press_prev_object_quat_w = None
            self._pressure_pad_presser_prev_pos_w = None
            self._pressure_pad_presser_prev_quat_w = None
            self._press_prev_robot_root_pos_w = None
            self._press_robot_base_pose_w = None
            if self._press_object_control_mode() in ("scripted", "initial_only"):
                self._write_press_object_pose()
                if self._plug_obj:
                    self._plug_obj.update(dt)
            else:
                self._press_object_pose_w = None
            if bool(getattr(cfg, "enable_pressure_pad_presser", False)):
                self._write_pressure_pad_presser_pose()
                if self._pressure_pad_presser_obj:
                    self._pressure_pad_presser_obj.update(dt)

        # Set up dynamic tracking
        self._stage = sim_utils.get_current_stage()
        self._per_sensor_target_prims = [self._stage.GetPrimAtPath(p) for p in target_query_paths]
        self._log_robot_visual_scene(cfg, label="post_spawn")
        self._validate_press_scene(cfg, target_query_paths)
        self._setup_dynamic_tracking()
        for sensor in self._physx_contact_pressure_sensors:
            sensor.update(dt=dt, force_recompute=True)

        # Resolve joint mapping (dataset order). Press-only runs with alternate
        # hand URDFs do not need replay joint targets.
        try:
            self._dataset_joint_ids = _resolve_joint_ids(self._robot, DATASET_JOINT_ORDER)
        except RuntimeError:
            if not cfg.enable_press_motion:
                raise
            self._dataset_joint_ids = []
            if bool(getattr(cfg, "lock_press_finger_joints", False)):
                self._press_hold_joint_ids = list(range(len(self._robot.joint_names)))
                hold_pose = str(getattr(cfg, "press_hold_joint_pose", "zero")).lower()
                self._press_hold_joint_pos = self._robot.data.default_joint_pos.clone()
                if hold_pose == "zero":
                    self._press_hold_joint_pos.zero_()
                elif hold_pose != "default":
                    raise ValueError(f"Unsupported press_hold_joint_pose={hold_pose!r}; expected 'zero' or 'default'")
                self._apply_press_initial_joint_degrees(cfg)
                self._press_hold_joint_vel = torch.zeros_like(self._robot.data.default_joint_vel)
                self._robot.write_joint_state_to_sim(self._press_hold_joint_pos, self._press_hold_joint_vel)
                self._robot.update(0.0)
                override_text = (
                    " with initial joint overrides"
                    if getattr(cfg, "press_initial_joint_degrees", ()) else ""
                )
                print(
                    "[WARN] Dataset joint mapping skipped for press-only robot URDF; "
                    f"holding {len(self._press_hold_joint_ids)} robot joints at {hold_pose} positions{override_text} "
                    "because lock_press_finger_joints=True.",
                    flush=True,
                )
                self._log_robot_visual_scene(cfg, label=f"post_{hold_pose}_joint_hold")
            else:
                print(
                    "[WARN] Dataset joint mapping skipped for press-only robot URDF; "
                    "robot joints are not held. Pass --lock-press-finger-joints to hold them.",
                    flush=True,
                )
        self._setup_press_finger_motion(cfg)
        self._action_dim = len(self._dataset_joint_ids)
        self._setup_locked_press_finger_joints(cfg)
        self._set_locked_press_finger_joint_targets()
        self._write_locked_press_finger_joint_state()
        self._tactile_sensor_count = max(
            len(self._tactile_sensors),
            max(self._sensor_slot_order) + 1 if self._sensor_slot_order else 0,
        )

        # Logging
        print(
            f"[INFO] {len(self._dataset_joint_ids)} joints mapped, "
            f"{len(self._tactile_sensors)} tactile sensors",
            flush=True,
        )
        if self._camera:
            print(f"[INFO] Camera: {cfg.camera_width}x{cfg.camera_height}", flush=True)

    def _log_robot_visual_scene(self, cfg: AlohaTactileEnvCfg, *, label: str = "") -> None:
        if self._stage is None:
            return

        try:
            from pxr import UsdGeom, UsdShade
        except Exception as exc:
            print(f"[WARN] Could not import USD geometry APIs for robot visual diagnostics: {exc}", flush=True)
            UsdGeom = None
            UsdShade = None

        robot_path = str(cfg.robot_prim_path).rstrip("/")
        mesh_paths: list[str] = []
        visual_paths: list[str] = []
        for prim in self._stage.Traverse():
            path = prim.GetPath().pathString
            if not path.startswith(f"{robot_path}/"):
                continue
            if "/visuals" in path:
                visual_paths.append(path)
            if prim.GetTypeName() == "Mesh":
                mesh_paths.append(path)

        finger_mesh_count = sum(1 for path in mesh_paths if "/right_" in path)
        sample = ", ".join(mesh_paths[:8])
        print(
            f"[INFO] Robot visual scene{f' ({label})' if label else ''}: "
            f"visual_prims={len(visual_paths)}, mesh_prims={len(mesh_paths)}, "
            f"right_hand_mesh_prims={finger_mesh_count}, sample=[{sample}]",
            flush=True,
        )
        if UsdGeom is None:
            return

        bbox_cache = UsdGeom.BBoxCache(
            0.0,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
            useExtentsHint=False,
        )

        def _range_to_arrays(prim):
            try:
                bbox_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            except Exception:
                return None
            if bbox_range.IsEmpty():
                return None
            min_v = bbox_range.GetMin()
            max_v = bbox_range.GetMax()
            mn = np.array([float(min_v[0]), float(min_v[1]), float(min_v[2])], dtype=np.float64)
            mx = np.array([float(max_v[0]), float(max_v[1]), float(max_v[2])], dtype=np.float64)
            return mn, mx

        def _fmt_vec(values: np.ndarray) -> str:
            return "[" + ", ".join(f"{float(v):.4g}" for v in values) + "]"

        right_ranges = []
        for path in mesh_paths:
            if "/right_" not in path:
                continue
            values = _range_to_arrays(self._stage.GetPrimAtPath(path))
            if values is not None:
                right_ranges.append(values)
        if right_ranges:
            mn = np.min(np.stack([item[0] for item in right_ranges], axis=0), axis=0)
            mx = np.max(np.stack([item[1] for item in right_ranges], axis=0), axis=0)
            print(
                f"[INFO] Robot visual bounds{f' ({label})' if label else ''}: "
                f"right_hand_world_min={_fmt_vec(mn)}, right_hand_world_max={_fmt_vec(mx)}, "
                f"right_hand_size={_fmt_vec(mx - mn)}",
                flush=True,
            )
        else:
            print(
                f"[WARN] Robot visual bounds{f' ({label})' if label else ''}: "
                "no valid right-hand mesh world bounds",
                flush=True,
            )

        focus_paths: list[str] = []
        press_link = str(getattr(cfg, "press_touch_link", "")).strip()
        for link_name in ("base_link", press_link, press_link.replace("_touch_link", "_rubber_link")):
            if not link_name:
                continue
            for suffix in ("", "/visuals"):
                path = f"{robot_path}/{link_name}{suffix}"
                if self._stage.GetPrimAtPath(path).IsValid() and path not in focus_paths:
                    focus_paths.append(path)
        for path in mesh_paths:
            if press_link and f"/{press_link}/" in path and path not in focus_paths:
                focus_paths.append(path)
        for path in mesh_paths:
            rubber_link = press_link.replace("_touch_link", "_rubber_link")
            if rubber_link and f"/{rubber_link}/" in path and path not in focus_paths:
                focus_paths.append(path)
        focus_paths = focus_paths[:8]

        for path in focus_paths:
            prim = self._stage.GetPrimAtPath(path)
            imageable = UsdGeom.Imageable(prim)
            try:
                visibility = str(imageable.ComputeVisibility())
            except Exception:
                visibility = "unknown"
            try:
                purpose = str(imageable.ComputePurpose())
            except Exception:
                purpose = "unknown"
            values = _range_to_arrays(prim)
            if values is None:
                bbox_text = "empty"
            else:
                mn, mx = values
                bbox_text = f"min={_fmt_vec(mn)}, max={_fmt_vec(mx)}, size={_fmt_vec(mx - mn)}"
            material_path = ""
            if UsdShade is not None:
                try:
                    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                    material_path = material.GetPath().pathString if material and material.GetPrim().IsValid() else ""
                except Exception:
                    material_path = ""
            print(
                f"[INFO] Robot visual focus{f' ({label})' if label else ''}: "
                f"path={path}, type={prim.GetTypeName()}, visibility={visibility}, purpose={purpose}, "
                f"bbox={bbox_text}, material={material_path or 'none'}",
                flush=True,
            )

    def _validate_press_scene(self, cfg: AlohaTactileEnvCfg, target_query_paths: list[str]) -> None:
        if not cfg.enable_press_motion or self._stage is None:
            return

        selected_links = [str(path) for path in getattr(self, "_selected_links", [])]
        press_link = str(cfg.press_touch_link)
        if not any(press_link in selected_link for selected_link in selected_links):
            raise RuntimeError(
                f"Press scene selected link mismatch: expected {press_link!r}, selected={selected_links!r}"
            )

        plug_prim = self._stage.GetPrimAtPath("/World/Plug")
        if self._plug_obj is None or not plug_prim.IsValid():
            raise RuntimeError(
                "Press mode requires the cylinder presser at /World/Plug, but it was not spawned. "
                f"press_object_usd_path={cfg.press_object_usd_path!r}"
            )

        target_queries = [str(path) for path in target_query_paths if str(path)]
        missing_queries: list[str] = []
        for target_query in target_queries:
            target_prim = self._stage.GetPrimAtPath(target_query)
            if target_prim is None or not target_prim.IsValid():
                missing_queries.append(target_query)
        if not target_queries or missing_queries:
            raise RuntimeError(
                f"Press mode target mesh is missing: target_query={missing_queries or target_queries!r}, "
                f"press_object_usd_path={cfg.press_object_usd_path!r}"
            )

        print(
            "[INFO] Press scene check: "
            f"selected_touch={selected_links}, target_root=/World/Plug, target_query={target_queries}, "
            f"press_object_usd={cfg.press_object_usd_path}",
            flush=True,
        )

    def _build_spaces(self, cfg: AlohaTactileEnvCfg) -> None:
        tactile_shape = (self._tactile_sensor_count, cfg.num_rows, cfg.num_cols)

        def map_box() -> gymnasium.spaces.Box:
            return gymnasium.spaces.Box(-np.inf, np.inf, shape=tactile_shape, dtype=np.float32)

        obs_spaces = {
            "tactile": map_box(),
            "pressure_force_map": map_box(),
            "pressure_force_map_raw": map_box(),
            "pressure_signed_distance_map": map_box(),
            "pressure_penetration_map": map_box(),
            "pressure_penetration_velocity_map": map_box(),
            "physx_contact_force_map": map_box(),
            "physx_contact_force_map_raw": map_box(),
            "physx_contact_count": gymnasium.spaces.Box(
                0, np.inf, shape=(self._tactile_sensor_count,), dtype=np.float32
            ),
            "joint_pos": gymnasium.spaces.Box(-np.inf, np.inf, shape=(self._action_dim,), dtype=np.float32),
            "joint_vel": gymnasium.spaces.Box(-np.inf, np.inf, shape=(self._action_dim,), dtype=np.float32),
            "plug_pose": gymnasium.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "socket_pose": gymnasium.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "press_touch_pose": gymnasium.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        }
        if self._camera:
            obs_spaces["rgb"] = gymnasium.spaces.Box(
                0, 255, shape=(cfg.camera_height, cfg.camera_width, 3), dtype=np.uint8
            )

        self.observation_space = gymnasium.spaces.Dict(obs_spaces)
        self.action_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(self._action_dim,), dtype=np.float32)


def _resolve_mesh_prim(root_path, *, prim_utils, sim_utils):
    """Find the first Mesh prim under *root_path* for SDF queries."""
    from pxr import UsdGeom

    root_prim = prim_utils.get_prim_at_path(str(root_path))
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Invalid target mesh prim: {root_path}")

    query_path = str(root_path)
    if not root_prim.IsA(UsdGeom.Mesh):
        children = sim_utils.get_all_matching_child_prims(
            query_path, predicate=lambda p: p.IsA(UsdGeom.Mesh),
            traverse_instance_prims=True,
        )
        if children:
            query_path = children[0].GetPath().pathString

    query_prim = prim_utils.get_prim_at_path(query_path)
    if not query_prim or not query_prim.IsValid():
        raise RuntimeError(f"No Mesh prim found under: {root_path}")
    return query_path, query_prim


def _to_numpy_1d(x, expected):
    x = to_numpy(x, shape=(-1,))
    if x.size != expected:
        raise ValueError(...)
    return x

def to_numpy(x, *, dtype=None, shape=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
    if shape is not None:
        x = x.reshape(shape)
    return x


def _obj_pose_numpy(obj) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    if obj is None:
        return pose
    root_state = getattr(obj.data, "root_state_w", None)
    if root_state is not None:
        pose[:] = to_numpy(root_state[0, :7], dtype=np.float32, shape=(7,))
        return pose
    # these should exist for IsaacLab RigidObject; if not, pose remains zeros
    root_pos = getattr(obj.data, "root_pos_w", None)
    root_quat = getattr(obj.data, "root_quat_w", None)
    if root_pos is None or root_quat is None:
        return pose
    pose[:3] = to_numpy(root_pos[0], dtype=np.float32, shape=(3,))
    pose[3:] = to_numpy(root_quat[0], dtype=np.float32, shape=(4,))
    return pose


def _rigid_prim_world_pose(rp):
    """Get (pos, quat) as numpy from a RigidPrim (handles API variations)."""
    fn = getattr(rp, "get_world_pose", None) or getattr(rp, "get_world_poses", None)
    if fn is None:
        raise AttributeError("RigidPrim has neither get_world_pose nor get_world_poses")
    pos, quat = fn()
    return _to_numpy_1d(pos, 3), _to_numpy_1d(quat, 4)
