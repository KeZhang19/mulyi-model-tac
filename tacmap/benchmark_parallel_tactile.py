from __future__ import annotations

"""Benchmark TacMap raw-depth invariance under parallel Isaac Lab environments."""

import argparse
import csv
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


TACMAP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TACMAP_ROOT.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tactile_parallel_benchmark"
DEFAULT_PRESS_INFO = TACMAP_ROOT / "assets" / "test_case" / "square_4_horizontal_145_20250904212453_990_1020.json"

FINGER_MAPS = {
    "middle": {
        "touch_link": "right_middle_touch_link",
        "points": TACMAP_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_point.npy",
        "normals": TACMAP_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_normal.npy",
    },
    "thumb": {
        "touch_link": "right_thumb_touch_link",
        "points": TACMAP_ROOT / "assets" / "tactilesensor_map" / "right_thumb_touch_point.npy",
        "normals": TACMAP_ROOT / "assets" / "tactilesensor_map" / "right_thumb_touch_normal.npy",
    },
}

SLIDE_AXES = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def _path_arg(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _vec3(text: str) -> tuple[float, float, float]:
    parts = [float(x) for x in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three floats")
    return (parts[0], parts[1], parts[2])


parser = argparse.ArgumentParser(description="Record and compare multi-env TacMap raw-depth traces.")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--finger", choices=tuple(FINGER_MAPS), default="middle")
parser.add_argument("--press-info", "--press_info", dest="press_info", type=Path, default=DEFAULT_PRESS_INFO)
parser.add_argument("--robot-usd", type=Path, default=None, help="Override the Revo2 robot USD used by the benchmark.")
parser.add_argument("--ray-mode", choices=("link_surface", "surface_normal"), default="link_surface")
parser.add_argument("--env-spacing", type=float, default=0.75)
parser.add_argument("--resolution-step", type=int, default=1, help="surface_normal stride; link_surface uses width/height")
parser.add_argument("--press-start-offset", type=float, default=0.025)
parser.add_argument("--press-end-offset", type=float, default=0.018)
parser.add_argument("--press-steps", type=int, default=700)
parser.add_argument("--press-slide-distance", type=float, default=0.004)
parser.add_argument("--press-slide-steps", type=int, default=300)
parser.add_argument("--press-slide-axis", choices=tuple(SLIDE_AXES), default="+y")
parser.add_argument("--max-steps", type=int, default=None, help="Default is press_steps + slide_steps - 1")
parser.add_argument("--record-every", type=int, default=1)
parser.add_argument(
    "--save-env-indices",
    type=str,
    default="",
    help="Comma list/ranges of env indices whose full TacMap images are saved; default saves all envs.",
)
parser.add_argument("--flip-axis", choices=("none", "x", "y", "z"), default="y")
parser.add_argument("--tacmap-max-distance", type=float, default=0.008)
parser.add_argument("--link-surface-width", type=int, default=240)
parser.add_argument("--link-surface-height", type=int, default=240)
parser.add_argument("--link-surface-ray-axis", choices=("+x", "-x", "+y", "-y", "+z", "-z"), default="+x")
parser.add_argument("--link-surface-ray-direction", type=_vec3, default=None)
parser.add_argument("--link-surface-grid-u-axis", choices=("+x", "-x", "+y", "-y", "+z", "-z"), default="+y")
parser.add_argument("--link-surface-grid-v-axis", choices=("+x", "-x", "+y", "-y", "+z", "-z"), default="+z")
parser.add_argument("--link-surface-grid-u-size", type=float, default=0.014)
parser.add_argument("--link-surface-grid-v-size", type=float, default=0.020)
parser.add_argument("--link-surface-grid-center", type=_vec3, default=(-0.008, 0.0, 0.0012))
parser.add_argument("--replicate-physics", action="store_true")
parser.add_argument(
    "--clone-in-fabric",
    action="store_true",
    help="Use IsaacLab/Fabric cloning when replicate physics is enabled. Diagnostic for high env counts.",
)
parser.add_argument("--disable-contact-sensor", action="store_true", help="Skip ContactSensor creation and save zero-width force/point arrays.")
parser.add_argument("--disable-touch-materials", action="store_true", help="Skip per-env touch material/collision mutation.")
parser.add_argument("--disable-robot-contact-sensors", action="store_true", help="Do not activate contact reporter APIs on robot spawn.")
parser.add_argument("--use-xform-anchor", action="store_true", help="Attach TacMap rays to per-env Xform anchors instead of rigid touch-link views.")
parser.add_argument("--static-surface-target", action="store_true", help="Do not create physics views for the touch-link surface target.")
parser.add_argument("--static-object-target", action="store_true", help="Do not create physics views for the object mesh target. Diagnostic mode for high env counts.")
parser.add_argument("--debug-viz", action="store_true")
parser.add_argument("--reference", type=Path, default=None)
parser.add_argument("--reference-env-index", type=int, default=0)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--no-compress", action="store_true")
parser.add_argument("--sync-cuda-timing", action="store_true")
parser.add_argument("--active-threshold", type=float, default=1.0e-9)
parser.add_argument("--active-iou-threshold", type=float, default=0.95)
parser.add_argument("--centroid-threshold-px", type=float, default=1.0)
parser.add_argument("--depth-max-abs-floor-m", type=float, default=0.00003)
parser.add_argument("--depth-max-rel-threshold", type=float, default=0.05)
parser.add_argument("--depth-rmse-floor-m", type=float, default=0.00002)
parser.add_argument("--depth-rmse-rel-threshold", type=float, default=0.03)
parser.add_argument("--volume-rel-threshold", type=float, default=0.05)
parser.add_argument("--onset-frame-threshold", type=int, default=1)
parser.add_argument("--slide-centroid-rmse-threshold-px", type=float, default=2.0)
parser.add_argument("--worst-median-ratio-threshold", type=float, default=2.0)
parser.add_argument("--precontact-leakage-threshold", type=float, default=0.001)
parser.add_argument("--surface-drift-threshold-m", type=float, default=0.00002)
parser.add_argument("--fail-on-threshold", action="store_true")
parser.add_argument("--gui", action="store_true", help="Launch Isaac Sim with the local GUI instead of the default headless benchmark mode.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()
if args_cli.gui:
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from revo2_tactile_env import Revo2TactilePressEnv  # noqa: E402
from revo2_tactile_env_cfg import Revo2TactilePressEnvCfg  # noqa: E402


def apply_finger_cfg(env_cfg: Revo2TactilePressEnvCfg, finger: str) -> None:
    spec = FINGER_MAPS[finger]
    env_cfg.finger = finger
    env_cfg.touch_link = spec["touch_link"]
    env_cfg.points_npy = str(spec["points"])
    env_cfg.normals_npy = str(spec["normals"])
    env_cfg.contact_sensor[0].prim_path = f"/World/envs/env_.*/Robot/{spec['touch_link']}"
    env_cfg.vbts_sensor[0].prim_path = f"/World/envs/env_.*/Robot/{spec['touch_link']}"
    env_cfg.vbts_sensor[0].points_npy = str(spec["points"])
    env_cfg.vbts_sensor[0].normals_npy = str(spec["normals"])


def apply_press_info(env_cfg: Revo2TactilePressEnvCfg, press_info_path: Path | None) -> Path | None:
    if press_info_path is None:
        return None
    path = Path(press_info_path).expanduser()
    if not path.is_absolute():
        candidates = [(Path.cwd() / path).resolve(), (TACMAP_ROOT / path).resolve(), (REPO_ROOT / path).resolve()]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    if not path.exists():
        raise FileNotFoundError(f"Press-info JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        press_info = json.load(f)

    presser_name = press_info.get("presser_name")
    if presser_name:
        usd_path = TACMAP_ROOT / "assets" / "presser" / f"{presser_name}.usd"
        if not usd_path.exists():
            raise FileNotFoundError(f"Presser USD not found: {usd_path}")
        env_cfg.presser_name = presser_name
        env_cfg.object_cfg.spawn.usd_path = str(usd_path)

    presser_init_rot = press_info.get("presser_init_rot")
    if presser_init_rot is not None:
        env_cfg.object_rot_in_touch_frame = tuple(float(x) for x in presser_init_rot)

    env_cfg.press_info = path.stem
    return path


def flip_axis_to_quat(axis: str) -> tuple[float, float, float, float]:
    return {
        "none": (1.0, 0.0, 0.0, 0.0),
        "x": (0.0, 1.0, 0.0, 0.0),
        "y": (0.0, 0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 0.0, 1.0),
    }[axis]


def build_env_cfg(args: argparse.Namespace) -> tuple[Revo2TactilePressEnvCfg, Path | None]:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.record_every <= 0:
        raise ValueError("--record-every must be positive")
    if args.press_steps <= 1:
        raise ValueError("--press-steps must be greater than 1")
    if args.resolution_step <= 0 or 240 % int(args.resolution_step) != 0:
        raise ValueError("--resolution-step must be a positive divisor of 240")

    env_cfg = Revo2TactilePressEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.scene.env_spacing = float(args.env_spacing)
    env_cfg.scene.replicate_physics = bool(args.replicate_physics)
    env_cfg.scene.clone_in_fabric = bool(args.clone_in_fabric)
    env_cfg.enable_contact_sensor = not bool(args.disable_contact_sensor)
    env_cfg.enable_touch_materials = not bool(args.disable_touch_materials)
    env_cfg.tacmap_use_xform_anchor = bool(args.use_xform_anchor)
    env_cfg.tacmap_surface_track_mesh_transforms = not bool(args.static_surface_target)
    env_cfg.seed = int(args.seed)
    if bool(args.disable_robot_contact_sensors):
        env_cfg.robot_cfg.spawn.activate_contact_sensors = False
    if args.robot_usd is not None:
        robot_usd = args.robot_usd.expanduser().resolve()
        if not robot_usd.exists():
            raise FileNotFoundError(f"--robot-usd does not exist: {robot_usd}")
        env_cfg.robot_cfg.spawn.usd_path = str(robot_usd)
    if args.device is not None:
        env_cfg.sim.device = args.device

    apply_finger_cfg(env_cfg, args.finger)
    press_info_path = apply_press_info(env_cfg, args.press_info)
    env_cfg.object_flip_quat_in_touch_frame = flip_axis_to_quat(args.flip_axis)

    env_cfg.press_start_offset = float(args.press_start_offset)
    env_cfg.press_end_offset = float(args.press_end_offset)
    env_cfg.press_steps = int(args.press_steps)
    env_cfg.press_slide_distance = float(args.press_slide_distance)
    env_cfg.press_slide_steps = int(args.press_slide_steps)
    env_cfg.press_slide_axis_l = SLIDE_AXES[str(args.press_slide_axis)]
    env_cfg.resolution_step = int(args.resolution_step)
    env_cfg.tacmap_ray_mode = str(args.ray_mode)
    env_cfg.tacmap_link_surface_width = int(args.link_surface_width)
    env_cfg.tacmap_link_surface_height = int(args.link_surface_height)
    env_cfg.tacmap_link_surface_ray_axis = str(args.link_surface_ray_axis)
    env_cfg.tacmap_link_surface_ray_direction = args.link_surface_ray_direction
    env_cfg.tacmap_link_surface_grid_u_axis = str(args.link_surface_grid_u_axis)
    env_cfg.tacmap_link_surface_grid_v_axis = str(args.link_surface_grid_v_axis)
    env_cfg.tacmap_link_surface_grid_u_size = float(args.link_surface_grid_u_size)
    env_cfg.tacmap_link_surface_grid_v_size = float(args.link_surface_grid_v_size)
    env_cfg.tacmap_link_surface_grid_center = tuple(float(v) for v in args.link_surface_grid_center)
    env_cfg.vbts_sensor[0].resolution_step = int(args.resolution_step)
    env_cfg.vbts_sensor[0].max_distance = float(args.tacmap_max_distance)
    env_cfg.vbts_sensor[0].debug_viz = bool(args.debug_viz)
    env_cfg.vbts_sensor[0].debug_viz_normals = bool(args.debug_viz)
    if bool(args.use_xform_anchor):
        env_cfg.tacmap_anchor_prim_path = "/World/envs/env_.*/TacmapAnchor"
        env_cfg.tacmap_surface_prim_path = f"/World/envs/env_.*/Robot/{env_cfg.touch_link}"
        env_cfg.vbts_sensor[0].prim_path = env_cfg.tacmap_anchor_prim_path
    if bool(getattr(args, "static_object_target", False)):
        for target in env_cfg.vbts_sensor[0].mesh_prim_paths:
            target.track_mesh_transforms = False
    if "distance_along_normal_raw" not in env_cfg.vbts_sensor[0].data_types:
        env_cfg.vbts_sensor[0].data_types = list(env_cfg.vbts_sensor[0].data_types) + ["distance_along_normal_raw"]
    return env_cfg, press_info_path


def _extract_obs(step_out: Any) -> dict[str, Any]:
    return step_out[0] if isinstance(step_out, tuple) else step_out


def _to_numpy(tensor: torch.Tensor, dtype: np.dtype | type | None = None) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr.copy()


def _record_step_indices(max_steps: int, record_every: int) -> list[int]:
    indices = [0]
    for step in range(1, max_steps + 1):
        if step % record_every == 0 or step == max_steps:
            indices.append(step)
    return indices


def _parse_save_env_indices(text: str, num_envs: int) -> list[int]:
    if not text or text.strip().lower() == "all":
        return list(range(num_envs))
    out: set[int] = set()
    for raw_token in text.split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token in ("first", "env0"):
            out.add(0)
        elif token in ("mid", "middle"):
            out.add(num_envs // 2)
        elif token == "last":
            out.add(num_envs - 1)
        elif ":" in token:
            parts = [int(p) if p else None for p in token.split(":")]
            if len(parts) > 3:
                raise ValueError(f"invalid --save-env-indices slice: {raw_token}")
            start, stop, step = (parts + [None, None, None])[:3]
            rng = range(*slice(start, stop, step).indices(num_envs))
            out.update(rng)
        elif "-" in token and not token.startswith("-"):
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    invalid = [idx for idx in out if idx < 0 or idx >= num_envs]
    if invalid:
        raise ValueError(f"--save-env-indices out of range for num_envs={num_envs}: {invalid}")
    if not out:
        raise ValueError("--save-env-indices selected no environments")
    return sorted(out)


def _obs_tensor(obs: dict[str, Any], name: str, fallback: str | None = None) -> torch.Tensor:
    value = obs.get(name)
    if value is None and fallback is not None:
        value = obs[fallback]
    if value is None:
        raise KeyError(name)
    return value


def _obs_array(
    obs: dict[str, Any],
    name: str,
    fallback: str | None = None,
    env_indices: torch.Tensor | None = None,
) -> np.ndarray:
    value = _obs_tensor(obs, name, fallback)
    if env_indices is not None:
        value = value.index_select(0, env_indices)
    dtype = np.uint8 if name in ("tacmap", "vbts_deform") else np.float32
    return _to_numpy(value, dtype)


def _tactile_scalar_snapshot(obs: dict[str, Any], active_threshold: float, pixel_area_m2: float) -> dict[str, np.ndarray]:
    raw = _obs_tensor(obs, "tacmap_raw").float()
    active = raw > float(active_threshold)
    active_f = active.float()
    height = raw.shape[-2]
    width = raw.shape[-1]
    x = torch.arange(width, dtype=torch.float32, device=raw.device).reshape(1, 1, 1, width)
    y = torch.arange(height, dtype=torch.float32, device=raw.device).reshape(1, 1, height, 1)
    count = active_f.sum(dim=(-2, -1))
    cx = torch.where(count > 0, (active_f * x).sum(dim=(-2, -1)) / count.clamp_min(1.0), torch.full_like(count, float("nan")))
    cy = torch.where(count > 0, (active_f * y).sum(dim=(-2, -1)) / count.clamp_min(1.0), torch.full_like(count, float("nan")))
    surface = _obs_tensor(obs, "tacmap_surface_raw").float()
    obj = _obs_tensor(obs, "tacmap_object_raw").float()
    formula = torch.clamp(surface - obj, min=0.0)
    return {
        "all_env_active_pixels": _to_numpy(count, np.float32),
        "all_env_max_depth_m": _to_numpy(raw.amax(dim=(-2, -1)), np.float32),
        "all_env_volume_m3": _to_numpy(raw.sum(dim=(-2, -1)) * float(pixel_area_m2), np.float32),
        "all_env_centroid_x_px": _to_numpy(cx, np.float32),
        "all_env_centroid_y_px": _to_numpy(cy, np.float32),
        "all_env_formula_max_error_m": _to_numpy(torch.abs(raw - formula).amax(dim=(-2, -1)), np.float32),
    }


def _store_snapshot(
    arrays: dict[str, np.ndarray],
    record_i: int,
    obs: dict[str, Any],
    env_indices: torch.Tensor | None,
    active_threshold: float,
    pixel_area_m2: float,
) -> None:
    arrays["tacmap"][record_i] = _obs_array(obs, "tacmap", "vbts_deform", env_indices)
    arrays["vbts_deform"][record_i] = _obs_array(obs, "vbts_deform", "tacmap", env_indices)
    arrays["tacmap_raw"][record_i] = _obs_array(obs, "tacmap_raw", env_indices=env_indices)
    arrays["tacmap_surface_raw"][record_i] = _obs_array(obs, "tacmap_surface_raw", env_indices=env_indices)
    arrays["tacmap_object_raw"][record_i] = _obs_array(obs, "tacmap_object_raw", env_indices=env_indices)
    arrays["tactile_forces"][record_i] = _obs_array(obs, "tactile_forces", env_indices=env_indices)
    arrays["tactile_points"][record_i] = _obs_array(obs, "tactile_points", env_indices=env_indices)
    for name, value in _tactile_scalar_snapshot(obs, active_threshold, pixel_area_m2).items():
        arrays[name][record_i] = value


def _git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_metadata() -> dict[str, Any]:
    status = _git_value(["status", "--porcelain"])
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "short_commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "dirty": bool(status),
    }


def _press_total_steps(env_cfg: Revo2TactilePressEnvCfg) -> int:
    return max(1, int(env_cfg.press_steps) + max(0, int(env_cfg.press_slide_steps)))


def _pixel_area(env_cfg: Revo2TactilePressEnvCfg) -> float:
    if str(env_cfg.tacmap_ray_mode) == "link_surface":
        return (float(env_cfg.tacmap_link_surface_grid_u_size) / float(env_cfg.tacmap_link_surface_width)) * (
            float(env_cfg.tacmap_link_surface_grid_v_size) / float(env_cfg.tacmap_link_surface_height)
        )
    return float(env_cfg.vbts_sensor[0].correction_scale) ** 2


def run_trace(args: argparse.Namespace, env_cfg: Revo2TactilePressEnvCfg, press_info_path: Path | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    max_steps = int(args.max_steps) if args.max_steps is not None else max(1, _press_total_steps(env_cfg) - 1)
    planned_indices = _record_step_indices(max_steps, int(args.record_every))
    planned_set = set(planned_indices)
    record_count = len(planned_indices)
    env = Revo2TactilePressEnv(env_cfg)
    step_times: list[float] = []
    actual_indices: list[int] = []

    try:
        obs = _extract_obs(env.reset())
        save_env_indices = _parse_save_env_indices(str(args.save_env_indices), int(env_cfg.scene.num_envs))
        env_indices_tensor = torch.tensor(save_env_indices, dtype=torch.long, device=env.device)
        if len(save_env_indices) == int(env_cfg.scene.num_envs) and save_env_indices[0] == 0:
            env_indices_tensor_or_none = None
        else:
            env_indices_tensor_or_none = env_indices_tensor
        pixel_area_m2 = float(_pixel_area(env_cfg))
        scalar_shape = tuple(_obs_tensor(obs, "tacmap_raw").shape[:2])
        arrays = {
            "tacmap": np.empty((record_count, *tuple(_obs_array(obs, "tacmap", "vbts_deform", env_indices_tensor_or_none).shape)), dtype=np.uint8),
            "vbts_deform": np.empty((record_count, *tuple(_obs_array(obs, "vbts_deform", "tacmap", env_indices_tensor_or_none).shape)), dtype=np.uint8),
            "tacmap_raw": np.empty((record_count, *tuple(_obs_array(obs, "tacmap_raw", env_indices=env_indices_tensor_or_none).shape)), dtype=np.float32),
            "tacmap_surface_raw": np.empty((record_count, *tuple(_obs_array(obs, "tacmap_surface_raw", env_indices=env_indices_tensor_or_none).shape)), dtype=np.float32),
            "tacmap_object_raw": np.empty((record_count, *tuple(_obs_array(obs, "tacmap_object_raw", env_indices=env_indices_tensor_or_none).shape)), dtype=np.float32),
            "tactile_forces": np.empty((record_count, *tuple(_obs_array(obs, "tactile_forces", env_indices=env_indices_tensor_or_none).shape)), dtype=np.float32),
            "tactile_points": np.empty((record_count, *tuple(_obs_array(obs, "tactile_points", env_indices=env_indices_tensor_or_none).shape)), dtype=np.float32),
            "all_env_active_pixels": np.empty((record_count, *scalar_shape), dtype=np.float32),
            "all_env_max_depth_m": np.empty((record_count, *scalar_shape), dtype=np.float32),
            "all_env_volume_m3": np.empty((record_count, *scalar_shape), dtype=np.float32),
            "all_env_centroid_x_px": np.empty((record_count, *scalar_shape), dtype=np.float32),
            "all_env_centroid_y_px": np.empty((record_count, *scalar_shape), dtype=np.float32),
            "all_env_formula_max_error_m": np.empty((record_count, *scalar_shape), dtype=np.float32),
        }
        record_i = 0
        _store_snapshot(arrays, record_i, obs, env_indices_tensor_or_none, float(args.active_threshold), pixel_area_m2)
        actual_indices.append(0)
        record_i += 1

        for step in range(1, max_steps + 1):
            if not simulation_app.is_running():
                break
            actions = torch.zeros(env_cfg.scene.num_envs, env_cfg.action_space, device=env.device)
            if args.sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            step_out = env.step(actions)
            if args.sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - start)
            if step in planned_set:
                obs = _extract_obs(step_out)
                _store_snapshot(arrays, record_i, obs, env_indices_tensor_or_none, float(args.active_threshold), pixel_area_m2)
                actual_indices.append(step)
                record_i += 1

        if record_i < record_count:
            arrays = {name: value[:record_i] for name, value in arrays.items()}

        cuda_memory_allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        cuda_memory_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
        try:
            env_origins = _to_numpy(env.scene.env_origins, np.float32).tolist()
        except Exception:
            env_origins = None
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "git": _git_metadata(),
            "num_envs": int(env_cfg.scene.num_envs),
            "seed": int(env_cfg.seed),
            "finger": str(env_cfg.finger),
            "touch_link": str(env_cfg.touch_link),
            "press_info": _display_path(press_info_path),
            "press_info_stem": press_info_path.stem if press_info_path else "",
            "presser_name": str(getattr(env_cfg, "presser_name", "")),
            "ray_mode": str(env_cfg.tacmap_ray_mode),
            "env_spacing": float(env_cfg.scene.env_spacing),
            "env_origins": env_origins,
            "saved_env_indices": save_env_indices,
            "saved_env_count": len(save_env_indices),
            "saved_env_origins": [env_origins[i] for i in save_env_indices] if env_origins is not None else None,
            "flip_axis": str(args.flip_axis),
            "resolution_step": int(env_cfg.resolution_step),
            "map_shape": list(arrays["tacmap_raw"].shape[-2:]),
            "press_start_offset": float(env_cfg.press_start_offset),
            "press_end_offset": float(env_cfg.press_end_offset),
            "press_steps": int(env_cfg.press_steps),
            "press_slide_distance": float(env_cfg.press_slide_distance),
            "press_slide_steps": int(env_cfg.press_slide_steps),
            "press_slide_axis": str(args.press_slide_axis),
            "max_steps": int(max_steps),
            "record_every": int(args.record_every),
            "recorded_step_indices": actual_indices,
            "includes_reset_frame": True,
            "device": str(env_cfg.sim.device),
            "headless": bool(args.headless),
            "replicate_physics": bool(env_cfg.scene.replicate_physics),
            "tacmap_max_distance": float(env_cfg.vbts_sensor[0].max_distance),
            "pixel_area_m2": pixel_area_m2,
            "link_surface": {
                "width": int(env_cfg.tacmap_link_surface_width),
                "height": int(env_cfg.tacmap_link_surface_height),
                "ray_axis": str(env_cfg.tacmap_link_surface_ray_axis),
                "ray_direction": env_cfg.tacmap_link_surface_ray_direction,
                "grid_u_axis": str(env_cfg.tacmap_link_surface_grid_u_axis),
                "grid_v_axis": str(env_cfg.tacmap_link_surface_grid_v_axis),
                "grid_u_size": float(env_cfg.tacmap_link_surface_grid_u_size),
                "grid_v_size": float(env_cfg.tacmap_link_surface_grid_v_size),
                "grid_center": list(env_cfg.tacmap_link_surface_grid_center),
            },
            "cuda_memory_allocated": cuda_memory_allocated,
            "cuda_memory_reserved": cuda_memory_reserved,
        }
        arrays["step_times_s"] = np.asarray(step_times, dtype=np.float64)
        arrays["recorded_step_indices"] = np.asarray(actual_indices, dtype=np.int32)
        return metadata, arrays
    finally:
        env.close()


def _slug(text: str) -> str:
    safe = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text]
    return "".join(safe).strip("_") or "run"


def trace_output_path(args: argparse.Namespace, metadata: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        timestamp,
        f"finger-{metadata['finger']}",
        f"case-{metadata['press_info_stem'] or 'manual'}",
        f"mode-{metadata['ray_mode']}",
        f"envs-{metadata['num_envs']:04d}",
        f"seed-{metadata['seed']}",
    ]
    if args.label:
        parts.insert(1, _slug(args.label))
    return _path_arg(args.output_dir) / ("__".join(_slug(p) for p in parts) + ".npz")


def save_trace(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    path.with_name(path.stem + "__args.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_trace(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = _path_arg(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        arrays = {name: data[name].copy() for name in data.files if name != "metadata"}
    return metadata, arrays


def _centroids(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = mask.shape[-2]
    width = mask.shape[-1]
    y = np.arange(height, dtype=np.float64).reshape((1, 1, 1, height, 1))
    x = np.arange(width, dtype=np.float64).reshape((1, 1, 1, 1, width))
    area = mask.sum(axis=(-2, -1), dtype=np.float64)
    cx = np.divide((mask * x).sum(axis=(-2, -1)), area, out=np.full_like(area, np.nan), where=area > 0)
    cy = np.divide((mask * y).sum(axis=(-2, -1)), area, out=np.full_like(area, np.nan), where=area > 0)
    return cx, cy, area


def _first_active_steps(mask: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
    active = mask.any(axis=(-2, -1))
    out = np.full(active.shape[1:], -1, dtype=np.int32)
    for env_i in range(active.shape[1]):
        for sensor_i in range(active.shape[2]):
            hits = np.nonzero(active[:, env_i, sensor_i])[0]
            if hits.size:
                out[env_i, sensor_i] = int(step_indices[hits[0]])
    return out


def _last_active_steps(mask: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
    active = mask.any(axis=(-2, -1))
    out = np.full(active.shape[1:], -1, dtype=np.int32)
    for env_i in range(active.shape[1]):
        for sensor_i in range(active.shape[2]):
            hits = np.nonzero(active[:, env_i, sensor_i])[0]
            if hits.size:
                out[env_i, sensor_i] = int(step_indices[hits[-1]])
    return out


def _bbox_sizes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_shape = mask.shape[:-2]
    widths = np.zeros(flat_shape, dtype=np.float64)
    heights = np.zeros(flat_shape, dtype=np.float64)
    for index in np.ndindex(flat_shape):
        ys, xs = np.nonzero(mask[index])
        if xs.size:
            widths[index] = float(xs.max() - xs.min() + 1)
            heights[index] = float(ys.max() - ys.min() + 1)
    return widths, heights


def _principal_angles(mask: np.ndarray) -> np.ndarray:
    height = mask.shape[-2]
    width = mask.shape[-1]
    y = np.arange(height, dtype=np.float64).reshape((1, 1, 1, height, 1))
    x = np.arange(width, dtype=np.float64).reshape((1, 1, 1, 1, width))
    area = mask.sum(axis=(-2, -1), dtype=np.float64)
    sum_x = (mask * x).sum(axis=(-2, -1), dtype=np.float64)
    sum_y = (mask * y).sum(axis=(-2, -1), dtype=np.float64)
    sum_xx = (mask * (x * x)).sum(axis=(-2, -1), dtype=np.float64)
    sum_yy = (mask * (y * y)).sum(axis=(-2, -1), dtype=np.float64)
    sum_xy = (mask * (x * y)).sum(axis=(-2, -1), dtype=np.float64)
    cx = np.divide(sum_x, area, out=np.zeros_like(area), where=area > 0)
    cy = np.divide(sum_y, area, out=np.zeros_like(area), where=area > 0)
    cov_xx = np.divide(sum_xx, area, out=np.zeros_like(area), where=area > 0) - np.square(cx)
    cov_yy = np.divide(sum_yy, area, out=np.zeros_like(area), where=area > 0) - np.square(cy)
    cov_xy = np.divide(sum_xy, area, out=np.zeros_like(area), where=area > 0) - cx * cy
    angle = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)
    angle[area < 2] = np.nan
    return angle


def _principal_angle_diff_deg(current_angle: np.ndarray, reference_angle: np.ndarray) -> np.ndarray:
    diff = current_angle - reference_angle
    diff = (diff + (math.pi / 2.0)) % math.pi - (math.pi / 2.0)
    return np.abs(np.rad2deg(diff))


def _monotonicity_metrics(peak_depth: np.ndarray, step_indices: np.ndarray, press_steps: int) -> dict[str, Any]:
    press_mask = step_indices <= max(0, int(press_steps) - 1)
    if np.count_nonzero(press_mask) < 2:
        return {"depth_monotonicity_violation_count": 0, "depth_monotonicity_max_drop_m": 0.0}
    deltas = np.diff(peak_depth[press_mask], axis=0)
    drops = -deltas[deltas < -1.0e-10]
    return {
        "depth_monotonicity_violation_count": int(drops.size),
        "depth_monotonicity_max_drop_m": float(drops.max(initial=0.0)),
    }


def _slide_stability_metrics(peak_depth: np.ndarray, step_indices: np.ndarray, press_steps: int) -> dict[str, Any]:
    slide_mask = step_indices >= max(0, int(press_steps) - 1)
    if np.count_nonzero(slide_mask) < 2:
        return {"slide_peak_depth_std_m_mean": 0.0, "slide_peak_depth_std_m_max": 0.0}
    std = np.std(peak_depth[slide_mask], axis=0)
    return {
        "slide_peak_depth_std_m_mean": float(std.mean()) if std.size else 0.0,
        "slide_peak_depth_std_m_max": float(std.max(initial=0.0)),
    }


def _env_drift_metrics(rmse_env: np.ndarray, env_origins: list[list[float]] | None) -> dict[str, Any]:
    if rmse_env.size == 0:
        return {
            "env0_rmse_m": 0.0,
            "middle_env_index": -1,
            "middle_env_rmse_m": 0.0,
            "last_env_rmse_m": 0.0,
            "worst_env_index": -1,
            "worst_env_origin_norm_m": 0.0,
            "env_origin_rmse_slope_m_per_m": 0.0,
            "worst_envs": [],
        }

    middle_index = int(rmse_env.size // 2)
    worst_index = int(np.argmax(rmse_env))
    origin_norms = None
    if env_origins is not None:
        origins = np.asarray(env_origins, dtype=np.float64)
        if origins.ndim == 2 and origins.shape[0] >= rmse_env.size and origins.shape[1] >= 2:
            origin_norms = np.linalg.norm(origins[: rmse_env.size, :3], axis=1)

    origin_slope = 0.0
    if origin_norms is not None and rmse_env.size > 1 and np.ptp(origin_norms) > 1.0e-12:
        origin_slope = float(np.polyfit(origin_norms, rmse_env, deg=1)[0])

    worst_order = np.argsort(rmse_env)[-min(8, rmse_env.size) :][::-1]
    worst_envs = []
    for env_i in worst_order:
        worst_envs.append(
            {
                "env_index": int(env_i),
                "rmse_m": float(rmse_env[env_i]),
                "origin_norm_m": float(origin_norms[env_i]) if origin_norms is not None else None,
            }
        )

    return {
        "env0_rmse_m": float(rmse_env[0]),
        "middle_env_index": middle_index,
        "middle_env_rmse_m": float(rmse_env[middle_index]),
        "last_env_rmse_m": float(rmse_env[-1]),
        "worst_env_index": worst_index,
        "worst_env_origin_norm_m": float(origin_norms[worst_index]) if origin_norms is not None else 0.0,
        "env_origin_rmse_slope_m_per_m": origin_slope,
        "worst_envs": worst_envs,
    }


def depth_metrics(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    active_threshold: float,
    reference_env_index: int,
    tacmap_max_distance: float,
    pixel_area_m2: float,
    step_indices: np.ndarray,
    press_steps: int,
    env_origins: list[list[float]] | None = None,
) -> dict[str, Any]:
    if reference_env_index < 0 or reference_env_index >= reference.shape[1]:
        raise ValueError(f"reference env index {reference_env_index} out of bounds for reference shape {reference.shape}")
    if current.shape[0] != reference.shape[0] or current.shape[2:] != reference.shape[2:]:
        raise ValueError(f"incompatible raw trace shapes: current={current.shape}, reference={reference.shape}")

    ref = reference[:, reference_env_index : reference_env_index + 1]
    diff = current.astype(np.float64) - ref.astype(np.float64)
    abs_diff = np.abs(diff)
    cur_active = current > active_threshold
    ref_active = ref > active_threshold

    intersection = np.logical_and(cur_active, ref_active).sum(dtype=np.int64)
    union = np.logical_or(cur_active, ref_active).sum(dtype=np.int64)
    iou_global = 1.0 if union == 0 else float(intersection / union)
    inter_env = np.logical_and(cur_active, ref_active).sum(axis=(0, 2, 3, 4), dtype=np.int64)
    union_env = np.logical_or(cur_active, ref_active).sum(axis=(0, 2, 3, 4), dtype=np.int64)
    iou_env = np.where(union_env > 0, inter_env / np.maximum(union_env, 1), 1.0)

    cx_cur, cy_cur, area_cur = _centroids(cur_active)
    cx_ref, cy_ref, area_ref = _centroids(ref_active)
    both_active = (area_cur > 0) & (area_ref > 0)
    one_empty = (area_cur > 0) ^ (area_ref > 0)
    centroid_shift = np.sqrt(np.square(cx_cur - cx_ref) + np.square(cy_cur - cy_ref))
    valid_shift = centroid_shift[both_active]

    bbox_w_cur, bbox_h_cur = _bbox_sizes(cur_active)
    bbox_w_ref, bbox_h_ref = _bbox_sizes(ref_active)
    bbox_w_diff = np.abs(bbox_w_cur - bbox_w_ref)
    bbox_h_diff = np.abs(bbox_h_cur - bbox_h_ref)
    angle_cur = _principal_angles(cur_active)
    angle_ref = _principal_angles(ref_active)
    angle_diff = _principal_angle_diff_deg(angle_cur, angle_ref)
    valid_angle = np.isfinite(angle_diff) & both_active

    peak_cur = current.max(axis=(-2, -1))
    peak_ref = ref.max(axis=(-2, -1))
    peak_diff = np.abs(peak_cur - peak_ref)
    ref_peak_global = float(ref.max(initial=0.0))

    volume_cur = current.sum(axis=(-2, -1), dtype=np.float64) * float(pixel_area_m2)
    volume_ref = ref.sum(axis=(-2, -1), dtype=np.float64) * float(pixel_area_m2)
    volume_abs = np.abs(volume_cur - volume_ref)
    volume_rel = np.divide(volume_abs, np.maximum(np.abs(volume_ref), 1.0e-18))
    volume_active = np.broadcast_to(volume_ref > 1.0e-18, volume_rel.shape)
    volume_rel_active = volume_rel[volume_active]

    onset_cur = _first_active_steps(cur_active, step_indices)
    onset_ref = _first_active_steps(ref_active, step_indices)
    onset_valid = (onset_cur >= 0) & (onset_ref >= 0)
    onset_error = np.abs(onset_cur - onset_ref)
    offset_cur = _last_active_steps(cur_active, step_indices)
    offset_ref = _last_active_steps(ref_active, step_indices)
    offset_valid = (offset_cur >= 0) & (offset_ref >= 0)
    offset_error = np.abs(offset_cur - offset_ref)

    slide_mask = step_indices >= max(0, int(press_steps) - 1)
    slide_shift = centroid_shift[slide_mask]
    slide_valid = both_active[slide_mask]
    slide_values = slide_shift[slide_valid]

    rmse_env = np.sqrt(np.mean(np.square(diff), axis=(0, 2, 3, 4)))
    median_env = float(np.median(rmse_env)) if rmse_env.size else 0.0
    worst_env = float(np.max(rmse_env)) if rmse_env.size else 0.0
    worst_median_ratio = 0.0 if median_env <= 1.0e-18 else float(worst_env / median_env)
    env_index_slope = 0.0
    if rmse_env.size > 1:
        env_index_slope = float(np.polyfit(np.arange(rmse_env.size, dtype=np.float64), rmse_env, deg=1)[0])

    precontact = ~ref_active.any(axis=(1, 2, 3, 4))
    precontact_active_pixels = int(np.count_nonzero(cur_active[precontact])) if np.any(precontact) else 0
    precontact_total_pixels = int(cur_active[precontact].size) if np.any(precontact) else 0
    precontact_leakage_rate = 0.0 if precontact_total_pixels == 0 else float(precontact_active_pixels / precontact_total_pixels)

    metrics = {
        "shape": list(current.shape),
        "nan_count": int(np.count_nonzero(np.isnan(current))),
        "inf_count": int(np.count_nonzero(np.isinf(current))),
        "negative_count": int(np.count_nonzero(current < 0.0)),
        "over_max_distance_count": int(np.count_nonzero(current > float(tacmap_max_distance))),
        "active_iou_global": iou_global,
        "active_iou_min_env": float(np.min(iou_env)) if iou_env.size else 1.0,
        "active_iou_per_env": [float(v) for v in iou_env.tolist()],
        "active_pixel_count_abs_error_mean": float(np.abs(area_cur - area_ref).mean()) if area_cur.size else 0.0,
        "active_pixel_count_abs_error_max": float(np.abs(area_cur - area_ref).max(initial=0.0)),
        "centroid_error_px_mean": float(valid_shift.mean()) if valid_shift.size else 0.0,
        "centroid_error_px_max": float(valid_shift.max(initial=0.0)) if valid_shift.size else 0.0,
        "centroid_missing_count": int(np.count_nonzero(one_empty)),
        "bbox_width_error_px_max": float(bbox_w_diff.max(initial=0.0)),
        "bbox_height_error_px_max": float(bbox_h_diff.max(initial=0.0)),
        "principal_axis_angle_error_deg_mean": float(angle_diff[valid_angle].mean()) if np.any(valid_angle) else 0.0,
        "principal_axis_angle_error_deg_max": float(angle_diff[valid_angle].max(initial=0.0)) if np.any(valid_angle) else 0.0,
        "depth_mae_m": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "depth_rmse_m": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0,
        "depth_p95_abs_error_m": float(np.percentile(abs_diff, 95)) if abs_diff.size else 0.0,
        "max_depth_error_m": float(peak_diff.max(initial=0.0)),
        "gt_max_depth_m": ref_peak_global,
        "indentation_volume_abs_error_m3_max": float(volume_abs.max(initial=0.0)),
        "indentation_volume_rel_error_max": float(volume_rel_active.max(initial=0.0)) if volume_rel_active.size else 0.0,
        "contact_onset_error_frames_max": int(onset_error[onset_valid].max(initial=0)) if np.any(onset_valid) else 0,
        "contact_offset_error_frames_max": int(offset_error[offset_valid].max(initial=0)) if np.any(offset_valid) else 0,
        "slide_centroid_rmse_px": float(np.sqrt(np.mean(np.square(slide_values)))) if slide_values.size else 0.0,
        "precontact_active_pixels": precontact_active_pixels,
        "precontact_leakage_rate": precontact_leakage_rate,
        "env_rmse_m": [float(v) for v in rmse_env.tolist()],
        "worst_env_rmse_m": worst_env,
        "median_env_rmse_m": median_env,
        "worst_median_rmse_ratio": worst_median_ratio,
        "env_index_rmse_slope": env_index_slope,
    }
    metrics.update(_monotonicity_metrics(peak_cur, step_indices, press_steps))
    metrics.update(_slide_stability_metrics(peak_cur, step_indices, press_steps))
    metrics.update(_env_drift_metrics(rmse_env, env_origins))
    return metrics


def link_surface_metrics(arrays: dict[str, np.ndarray], tacmap_max_distance: float) -> dict[str, Any]:
    raw = arrays["tacmap_raw"].astype(np.float64)
    surface = arrays["tacmap_surface_raw"].astype(np.float64)
    obj = arrays["tacmap_object_raw"].astype(np.float64)
    surface_valid = np.isfinite(surface) & (surface > 0.0)
    object_valid = np.isfinite(obj) & (obj > 0.0)
    valid_pair = surface_valid & object_valid
    before_surface = valid_pair & (obj <= surface)
    formula = np.where(before_surface, surface - obj, 0.0)
    formula_error = np.abs(raw - formula)
    inactive_frames = ~raw.any(axis=(1, 2, 3, 4))
    surface_drift = 0.0
    if np.any(inactive_frames):
        baseline = surface[0:1]
        surface_drift = float(np.nanmax(np.abs(surface[inactive_frames] - baseline)))
    return {
        "surface_nan_count": int(np.count_nonzero(np.isnan(surface))),
        "surface_inf_count": int(np.count_nonzero(np.isinf(surface))),
        "object_nan_count": int(np.count_nonzero(np.isnan(obj))),
        "object_inf_count": int(np.count_nonzero(np.isinf(obj))),
        "surface_finite_rate": float(np.count_nonzero(surface_valid) / surface.size) if surface.size else 1.0,
        "object_hit_rate": float(np.count_nonzero(object_valid) / obj.size) if obj.size else 1.0,
        "before_surface_count": int(np.count_nonzero(before_surface)),
        "object_hit_no_penetration_count": int(np.count_nonzero(object_valid & ~before_surface)),
        "penetration_formula_max_error_m": float(formula_error.max(initial=0.0)),
        "surface_baseline_drift_m": surface_drift,
        "impossible_surface_count": int(np.count_nonzero(surface < 0.0) + np.count_nonzero(surface > float(tacmap_max_distance))),
        "impossible_object_count": int(np.count_nonzero(obj < 0.0) + np.count_nonzero(obj > float(tacmap_max_distance))),
    }


def image_metrics(current: np.ndarray, reference: np.ndarray, active_threshold: float, reference_env_index: int) -> dict[str, Any]:
    ref = reference[:, reference_env_index : reference_env_index + 1]
    diff = np.abs(current.astype(np.int16) - ref.astype(np.int16))
    cur_active = current > active_threshold
    ref_active = ref > active_threshold
    union = np.logical_or(cur_active, ref_active).sum(dtype=np.int64)
    intersection = np.logical_and(cur_active, ref_active).sum(dtype=np.int64)
    return {
        "shape": list(current.shape),
        "max_abs_error": int(diff.max(initial=0)),
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "active_iou_global": 1.0 if union == 0 else float(intersection / union),
    }


def vector_metrics(current: np.ndarray, reference: np.ndarray, reference_env_index: int) -> dict[str, Any]:
    ref = reference[:, reference_env_index : reference_env_index + 1]
    diff_norm = np.linalg.norm(current.astype(np.float64) - ref.astype(np.float64), axis=-1)
    return {
        "shape": list(current.shape),
        "abs_error_mean": float(diff_norm.mean()) if diff_norm.size else 0.0,
        "abs_error_max": float(diff_norm.max(initial=0.0)),
    }


def evaluate_thresholds(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    raw = metrics["tacmap_raw"]
    depth_max_threshold = max(float(args.depth_max_abs_floor_m), raw["gt_max_depth_m"] * float(args.depth_max_rel_threshold))
    depth_rmse_threshold = max(float(args.depth_rmse_floor_m), raw["gt_max_depth_m"] * float(args.depth_rmse_rel_threshold))
    checks = [
        (raw["nan_count"] == 0, f"tacmap_raw nan_count={raw['nan_count']}"),
        (raw["inf_count"] == 0, f"tacmap_raw inf_count={raw['inf_count']}"),
        (raw["negative_count"] == 0, f"tacmap_raw negative_count={raw['negative_count']}"),
        (raw["over_max_distance_count"] == 0, f"tacmap_raw over_max_distance_count={raw['over_max_distance_count']}"),
        (raw["precontact_leakage_rate"] <= args.precontact_leakage_threshold, f"precontact_leakage_rate={raw['precontact_leakage_rate']:.6g}"),
        (raw["active_iou_global"] >= args.active_iou_threshold, f"active_iou_global={raw['active_iou_global']:.6g}"),
        (raw["centroid_error_px_max"] <= args.centroid_threshold_px, f"centroid_error_px_max={raw['centroid_error_px_max']:.6g}"),
        (raw["max_depth_error_m"] <= depth_max_threshold, f"max_depth_error_m={raw['max_depth_error_m']:.6g} threshold={depth_max_threshold:.6g}"),
        (raw["depth_rmse_m"] <= depth_rmse_threshold, f"depth_rmse_m={raw['depth_rmse_m']:.6g} threshold={depth_rmse_threshold:.6g}"),
        (raw["indentation_volume_rel_error_max"] <= args.volume_rel_threshold, f"volume_rel_error_max={raw['indentation_volume_rel_error_max']:.6g}"),
        (raw["contact_onset_error_frames_max"] <= args.onset_frame_threshold, f"onset_error={raw['contact_onset_error_frames_max']}"),
        (raw["contact_offset_error_frames_max"] <= args.onset_frame_threshold, f"offset_error={raw['contact_offset_error_frames_max']}"),
        (raw["slide_centroid_rmse_px"] <= args.slide_centroid_rmse_threshold_px, f"slide_centroid_rmse_px={raw['slide_centroid_rmse_px']:.6g}"),
        (raw["worst_median_rmse_ratio"] <= args.worst_median_ratio_threshold, f"worst_median_rmse_ratio={raw['worst_median_rmse_ratio']:.6g}"),
    ]
    for ok, text in checks:
        if not ok:
            failures.append(text)
    link = metrics.get("link_surface")
    if link is not None:
        for name in ("surface_nan_count", "surface_inf_count", "object_nan_count", "object_inf_count"):
            if link[name] > 0:
                failures.append(f"{name}={link[name]}")
        if link["penetration_formula_max_error_m"] > 1.0e-9:
            failures.append(f"penetration_formula_max_error_m={link['penetration_formula_max_error_m']:.6g}")
        if link["surface_baseline_drift_m"] > args.surface_drift_threshold_m:
            failures.append(f"surface_baseline_drift_m={link['surface_baseline_drift_m']:.6g}")
        if link["impossible_surface_count"] > 0:
            failures.append(f"impossible_surface_count={link['impossible_surface_count']}")
        if link["impossible_object_count"] > 0:
            failures.append(f"impossible_object_count={link['impossible_object_count']}")
    return failures


def compute_metrics(current_metadata: dict[str, Any], current_arrays: dict[str, np.ndarray], reference_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    reference_metadata, reference_arrays = load_trace(reference_path)
    cur_indices = current_arrays["recorded_step_indices"]
    ref_indices = reference_arrays["recorded_step_indices"]
    if not np.array_equal(cur_indices, ref_indices):
        raise ValueError(f"recorded step indices differ: current={cur_indices.tolist()}, reference={ref_indices.tolist()}")
    metrics: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "current_metadata": current_metadata,
        "reference_metadata": reference_metadata,
        "reference_path": str(_path_arg(reference_path)),
        "thresholds": vars(args),
    }
    metrics["tacmap_raw"] = depth_metrics(
        current_arrays["tacmap_raw"],
        reference_arrays["tacmap_raw"],
        active_threshold=float(args.active_threshold),
        reference_env_index=int(args.reference_env_index),
        tacmap_max_distance=float(current_metadata["tacmap_max_distance"]),
        pixel_area_m2=float(current_metadata["pixel_area_m2"]),
        step_indices=cur_indices,
        press_steps=int(current_metadata["press_steps"]),
        env_origins=current_metadata.get("saved_env_origins") or current_metadata.get("env_origins"),
    )
    metrics["tacmap"] = image_metrics(current_arrays["tacmap"], reference_arrays["tacmap"], 0.0, int(args.reference_env_index))
    metrics["tactile_forces_optional"] = vector_metrics(current_arrays["tactile_forces"], reference_arrays["tactile_forces"], int(args.reference_env_index))
    metrics["tactile_points_optional"] = vector_metrics(current_arrays["tactile_points"], reference_arrays["tactile_points"], int(args.reference_env_index))
    if str(current_metadata.get("ray_mode")) == "link_surface":
        metrics["link_surface"] = link_surface_metrics(current_arrays, float(current_metadata["tacmap_max_distance"]))
    failures = evaluate_thresholds(metrics, args)
    metrics["pass"] = not failures
    metrics["failures"] = failures
    return metrics


def compute_within_run_metrics(arrays: dict[str, np.ndarray], metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if arrays["tacmap_raw"].shape[1] <= 1:
        return None
    fake_ref = arrays["tacmap_raw"][:, :1]
    env_origins = metadata.get("saved_env_origins") or metadata.get("env_origins")
    if env_origins is not None:
        env_origins = env_origins[1:]
    return depth_metrics(
        arrays["tacmap_raw"][:, 1:],
        fake_ref,
        active_threshold=float(args.active_threshold),
        reference_env_index=0,
        tacmap_max_distance=float(metadata["tacmap_max_distance"]),
        pixel_area_m2=float(metadata["pixel_area_m2"]),
        step_indices=arrays["recorded_step_indices"],
        press_steps=int(metadata["press_steps"]),
        env_origins=env_origins,
    )


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    csv_path = path.with_name(path.stem + ".csv")
    raw = metrics.get("tacmap_raw", {})
    row = {
        "pass": metrics.get("pass"),
        "num_envs": metrics.get("current_metadata", {}).get("num_envs"),
        "saved_env_count": metrics.get("current_metadata", {}).get("saved_env_count"),
        "ray_mode": metrics.get("current_metadata", {}).get("ray_mode"),
        "active_iou_global": raw.get("active_iou_global"),
        "depth_rmse_m": raw.get("depth_rmse_m"),
        "max_depth_error_m": raw.get("max_depth_error_m"),
        "centroid_error_px_max": raw.get("centroid_error_px_max"),
        "principal_axis_angle_error_deg_max": raw.get("principal_axis_angle_error_deg_max"),
        "contact_offset_error_frames_max": raw.get("contact_offset_error_frames_max"),
        "slide_centroid_rmse_px": raw.get("slide_centroid_rmse_px"),
        "env_index_rmse_slope": raw.get("env_index_rmse_slope"),
        "env_origin_rmse_slope_m_per_m": raw.get("env_origin_rmse_slope_m_per_m"),
        "worst_median_rmse_ratio": raw.get("worst_median_rmse_ratio"),
    }
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _step_time_summary(step_times: np.ndarray) -> dict[str, float]:
    if step_times.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {"mean": float(np.mean(step_times)), "p95": float(np.percentile(step_times, 95)), "max": float(np.max(step_times))}


def print_summary(trace_path: Path, metadata: dict[str, Any], metrics: dict[str, Any] | None) -> None:
    print(f"[RESULT] trace: {trace_path}", flush=True)
    print(
        f"[RESULT] envs={metadata['num_envs']} finger={metadata['finger']} mode={metadata['ray_mode']} "
        f"records={len(metadata['recorded_step_indices'])}",
        flush=True,
    )
    times = metadata.get("step_times_summary")
    if times:
        print(f"[RESULT] step_time_s mean={times['mean']:.6f} p95={times['p95']:.6f} max={times['max']:.6f}", flush=True)
    if metrics is not None:
        raw = metrics["tacmap_raw"]
        print(
            f"[RESULT] compare pass={metrics['pass']} iou={raw['active_iou_global']:.6g} "
            f"rmse_m={raw['depth_rmse_m']:.6g} max_depth_err_m={raw['max_depth_error_m']:.6g} "
            f"centroid_px={raw['centroid_error_px_max']:.6g}",
            flush=True,
        )
        for failure in metrics.get("failures", []):
            print(f"[FAIL] {failure}", flush=True)


def main() -> int:
    env_cfg, press_info_path = build_env_cfg(args_cli)
    metadata, arrays = run_trace(args_cli, env_cfg, press_info_path)
    metadata["step_times_summary"] = _step_time_summary(arrays["step_times_s"])
    output_path = trace_output_path(args_cli, metadata)
    save_trace(output_path, metadata, arrays, compress=not args_cli.no_compress)

    metrics = None
    summary_path = output_path.with_name(output_path.stem + "__summary.json")
    if args_cli.reference is not None:
        metrics = compute_metrics(metadata, arrays, args_cli.reference, args_cli)
        metrics["current_path"] = str(output_path)
        save_metrics(output_path.with_name(output_path.stem + "__metrics.json"), metrics)
        save_metrics(summary_path, metrics)
    else:
        within = compute_within_run_metrics(arrays, metadata, args_cli)
        summary = {"created_at": datetime.now().isoformat(timespec="seconds"), "current_path": str(output_path), "metadata": metadata}
        if within is not None:
            summary["within_run_env0_reference"] = within
        save_metrics(summary_path, summary)

    print_summary(output_path, metadata, metrics)
    if metrics is not None and args_cli.fail_on_threshold and not metrics["pass"]:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
