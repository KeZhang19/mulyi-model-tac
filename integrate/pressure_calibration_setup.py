"""Small pressure-calibration helpers for the integrated tactile runner."""

from __future__ import annotations

from typing import Any

import numpy as np


SIGNED_AXIS_VALUES = {
    "+u", "-u", "+v", "-v", "+ray", "-ray",
    "+x", "-x", "+y", "-y", "+z", "-z",
}
SIGNED_AXIS_OPTIONS = {
    "--press-slide-axis",
    "--press-hand-axis",
    "--link-surface-ray-axis",
    "--link-surface-grid-u-axis",
    "--link-surface-grid-v-axis",
}
BALL_PROBE_PRESSURE_PAD_QUAT_WXYZ = (0.7071067811865476, 0.7071067811865475, 0.0, 0.0)


def normalize_signed_axis_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in SIGNED_AXIS_OPTIONS and i + 1 < len(argv) and argv[i + 1] in SIGNED_AXIS_VALUES:
            normalized.append(f"{arg}={argv[i + 1]}")
            i += 2
            continue
        normalized.append(arg)
        i += 1
    return normalized


def press_depth_stats_line(summary: dict[str, object]) -> str:
    if not summary:
        return ""
    phase = str(summary.get("phase", "unknown"))
    parts = [f"phase:{phase}"]
    if phase == "indent" and summary.get("cmd_indent_m", summary.get("indent_depth_m")) is not None:
        parts.append(f"cmd_indent:{float(summary.get('cmd_indent_m', summary.get('indent_depth_m'))) * 1000.0:.3f}mm")
    elif phase == "search" and summary.get("search_depth_m") is not None:
        parts.append(f"search:{float(summary['search_depth_m']) * 1000.0:.3f}mm")
    if summary.get("lateral_error_m") is not None:
        parts.append(f"lateral:{float(summary['lateral_error_m']) * 1000.0:.3f}mm")
    if summary.get("target_indent_m") is not None:
        parts.append(f"target:{float(summary['target_indent_m']) * 1000.0:.3f}mm")
    if summary.get("axis_source") is not None:
        parts.append(f"axis:{summary['axis_source']}")
    parts.append(f"sdf_pen_max:{float(summary.get('sdf_penetration_max_m', 0.0)) * 1000.0:.4f}mm")
    return "press_depth=" + " ".join(parts)


def _tensor_row_to_numpy(value: Any, *, ndim: int = 2) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    except Exception:
        return None
    arr = np.asarray(arr)
    if arr.ndim == ndim and arr.shape[0] > 0:
        return arr[0]
    if arr.ndim == ndim - 1:
        return arr
    return None


def print_rubber_link_poses(env) -> None:
    robot = getattr(env, "_robot", None)
    if robot is None:
        return
    names = [str(name) for name in getattr(robot, "body_names", [])]
    state = _tensor_row_to_numpy(getattr(getattr(robot, "data", None), "body_link_state_w", None), ndim=3)
    if state is None:
        return
    print("[LINK_POSE] rubber/tubber link world poses, meters:", flush=True)
    for i, name in enumerate(names):
        if i >= state.shape[0] or ("_rubber_link" not in name and "_tubber_link" not in name):
            continue
        pose = np.asarray(state[i, :7], dtype=np.float64)
        print(
            f"[LINK_POSE] {name} xyz=({pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}) "
            f"quat_wxyz=({pose[3]:.6f}, {pose[4]:.6f}, {pose[5]:.6f}, {pose[6]:.6f})",
            flush=True,
        )


def _xyz_text(value) -> str:
    if value is None:
        return "None"
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3 or not np.isfinite(arr[:3]).all():
        return "None"
    return f"({arr[0]:.6f}, {arr[1]:.6f}, {arr[2]:.6f})"


def _prim_bbox_center_w(env, prim_path: str) -> np.ndarray | None:
    try:
        from pxr import Usd, UsdGeom

        stage = getattr(env, "_sim_utils").get_current_stage()
        prim = stage.GetPrimAtPath(str(prim_path))
        if not prim or not prim.IsValid():
            return None
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn, mx = box.GetMin(), box.GetMax()
        return np.array(
            [
                0.5 * (float(mn[0]) + float(mx[0])),
                0.5 * (float(mn[1]) + float(mx[1])),
                0.5 * (float(mn[2]) + float(mx[2])),
            ],
            dtype=np.float64,
        )
    except Exception:
        return None


def _prim_world_xyz(env, prim_path: str) -> np.ndarray | None:
    try:
        from pxr import UsdGeom

        stage = getattr(env, "_sim_utils").get_current_stage()
        prim = stage.GetPrimAtPath(str(prim_path))
        if not prim or not prim.IsValid():
            return None
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        t = mat.ExtractTranslation()
        return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
    except Exception:
        return None


def _pressure_center_xyz(env, link_name: str) -> tuple[str | None, np.ndarray | None]:
    if not link_name:
        return None, None
    cfg = getattr(env, "_cfg", None)
    robot_path = str(getattr(cfg, "robot_prim_path", "/World/Robot")).rstrip("/")
    marker_path = f"{robot_path}/{link_name}/pressure_pad_center_marker"
    return str(link_name), _prim_bbox_center_w(env, marker_path)


def print_press_coordinates(env, *, phase: str, step: int) -> None:
    cfg = getattr(env, "_cfg", None)
    touch_link = str(getattr(cfg, "press_touch_link", ""))
    touch_pose = None
    touch_getter = getattr(env, "_press_touch_pose_numpy", None)
    if callable(touch_getter):
        touch_pose = touch_getter()
    plug_pose = None
    plug_getter = getattr(env, "press_object_pose_w_numpy", None)
    if callable(plug_getter):
        plug_pose = plug_getter()
    active_center_link, center_xyz = _pressure_center_xyz(env, touch_link)
    robot_path = str(getattr(cfg, "robot_prim_path", "/World/Robot")).rstrip("/")
    center_root_xyz = (
        _prim_world_xyz(env, f"{robot_path}/{active_center_link}")
        if active_center_link is not None
        else None
    )
    plug_mesh_center = _prim_bbox_center_w(env, "/World/Plug/geometry/mesh")
    if plug_mesh_center is None:
        plug_mesh_center = _prim_bbox_center_w(env, "/World/Plug")
    delta_center = None
    if center_xyz is not None and plug_mesh_center is not None:
        delta_center = np.asarray(plug_mesh_center, dtype=np.float64) - np.asarray(center_xyz, dtype=np.float64)
    print(
        f"[PRESS_COORD] phase={phase} step={int(step)} "
        f"touch_link={touch_link} touch_root={_xyz_text(None if touch_pose is None else np.asarray(touch_pose)[:3])} "
        f"center_link={active_center_link} center_root={_xyz_text(center_root_xyz)} center={_xyz_text(center_xyz)} "
        f"ball_root={_xyz_text(None if plug_pose is None else np.asarray(plug_pose)[:3])} "
        f"ball_mesh_center={_xyz_text(plug_mesh_center)} "
        f"ball_mesh_minus_center={_xyz_text(delta_center)}",
        flush=True,
    )
