from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .pressure_sources import GeometryNormalRayPenetrationSource, triangle_mesh_topology_diagnostics
from .pressure_taxel_map import PressureCalibration, calibrate_penetration, pressure_map_stats
from .pressure_trace_metrics import load_pressure_trace_npz


def apply_geometry_normal_ray_to_trace(
    trace: Mapping[str, Any] | str | Path,
    *,
    geometry: Mapping[str, Any],
    output_prefix: str = "geometry_normal_ray",
    pressure_points_key: str = "pressure_taxel_points_l_m",
    pressure_normals_key: str = "pressure_taxel_normals_l",
    rest_distance_m: float | Any,
    calibration: PressureCalibration | None = None,
    contact_deadband_m: float = 0.0,
    max_distance_m: float | None = None,
    invert_ray_directions: bool = False,
) -> dict[str, np.ndarray]:
    """Add geometry normal-ray pressure arrays to a pressure trace.

    The caller supplies explicit object geometry in the same local frame as the
    pressure taxel layout. This path does not read TacMap deformation values.
    """

    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if pressure_points_key not in data:
        raise KeyError(f"pressure_points_key {pressure_points_key!r} not found in trace")
    if pressure_normals_key not in data:
        raise KeyError(f"pressure_normals_key {pressure_normals_key!r} not found in trace")

    points = _as_shw3(data[pressure_points_key], name=pressure_points_key)
    normals = _as_shw3(data[pressure_normals_key], name=pressure_normals_key)
    if normals.shape != points.shape:
        raise ValueError(f"{pressure_normals_key} shape {normals.shape} does not match {pressure_points_key} {points.shape}")
    if invert_ray_directions:
        normals = -normals

    steps = _infer_trace_steps(data, geometry)
    origins = np.broadcast_to(points[None, ...], (steps, *points.shape)).astype(np.float32)
    directions = np.broadcast_to(normals[None, ...], (steps, *normals.shape)).astype(np.float32)

    source = GeometryNormalRayPenetrationSource(
        rest_distance_m=rest_distance_m,
        contact_deadband_m=max(0.0, float(contact_deadband_m)),
        max_distance_m=max_distance_m,
    )
    kind = str(geometry.get("kind", "")).lower()
    if kind == "sphere":
        frame = source.frame_from_sphere(
            origins,
            directions,
            center_l=_time_vector(geometry["center_l_m"], steps),
            radius_m=geometry["radius_m"],
        )
    elif kind == "box":
        frame = source.frame_from_box(
            origins,
            directions,
            center_l=_time_vector(geometry["center_l_m"], steps),
            half_extents_l=geometry["half_extents_l_m"],
        )
    elif kind == "mesh":
        frame = source.frame_from_triangle_mesh(
            origins,
            directions,
            vertices_l=geometry["vertices_l_m"],
            triangles=geometry.get("triangles"),
        )
    else:
        raise ValueError("geometry kind must be 'sphere', 'box', or 'mesh'")

    penetration = _as_tshw(frame.penetration_m, name="geometry_normal_ray_penetration_m")
    signed_distance = _as_tshw(frame.signed_distance_m, name="geometry_normal_ray_signed_distance_m")
    raw, pressure = calibrate_penetration(
        penetration,
        calibration or PressureCalibration(stiffness=1.0, max_force=1.0),
        normalize=True,
    )
    raw = np.asarray(raw, dtype=np.float32)
    pressure = np.asarray(pressure, dtype=np.float32)
    total_force, center = pressure_map_stats(raw)

    out = {str(key): value for key, value in data.items()}
    prefix = str(output_prefix).strip() or "geometry_normal_ray"
    out[f"{prefix}_source_kind"] = np.asarray(kind)
    out[f"{prefix}_penetration_m"] = penetration.astype(np.float32)
    out[f"{prefix}_signed_distance_m"] = signed_distance.astype(np.float32)
    out[f"{prefix}_contact_mask"] = (penetration > 0.0).astype(np.uint8)
    out[f"{prefix}_pressure_raw_n"] = raw
    out[f"{prefix}_pressure_norm"] = pressure
    out[f"{prefix}_total_force_n"] = np.asarray(total_force, dtype=np.float32)
    out[f"{prefix}_center_of_pressure_px"] = np.asarray(center, dtype=np.float32)
    return out


def write_geometry_normal_ray_trace(
    trace: Mapping[str, Any] | str | Path,
    out_path: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write a trace with geometry-normal-ray pressure arrays."""

    out = apply_geometry_normal_ray_to_trace(trace, **kwargs)
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)

    prefix = str(kwargs.get("output_prefix", "geometry_normal_ray")).strip() or "geometry_normal_ray"
    penetration = _as_tshw(out[f"{prefix}_penetration_m"], name=f"{prefix}_penetration_m")
    pressure = _as_tshw(out[f"{prefix}_pressure_norm"], name=f"{prefix}_pressure_norm")
    summary = {
        "out_trace": str(path),
        "output_prefix": prefix,
        "geometry_kind": str(np.asarray(out[f"{prefix}_source_kind"]).item()),
        "penetration_shape": [int(v) for v in penetration.shape],
        "pressure_shape": [int(v) for v in pressure.shape],
        "max_penetration_m": float(np.max(penetration)) if penetration.size else 0.0,
        "max_pressure_norm": float(np.max(pressure)) if pressure.size else 0.0,
        "active_taxel_count": int(np.count_nonzero(pressure > 0.0)),
    }
    geometry = kwargs.get("geometry")
    if isinstance(geometry, Mapping) and str(geometry.get("kind", "")).lower() == "mesh":
        summary["geometry_topology"] = triangle_mesh_topology_diagnostics(
            geometry["vertices_l_m"],
            geometry.get("triangles"),
        )
    return path, summary


def _infer_trace_steps(data: Mapping[str, Any], geometry: Mapping[str, Any]) -> int:
    center = geometry.get("center_l_m")
    if center is not None:
        center_arr = np.asarray(center)
        if center_arr.ndim == 2 and center_arr.shape[-1] == 3:
            return int(center_arr.shape[0])
    for key in ("step", "pressure_norm", "penetration_m"):
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    return 1


def _time_vector(value: Any, steps: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (3,):
        return arr
    if arr.shape == (steps, 3):
        return arr
    raise ValueError(f"expected vector shape (3,) or ({steps},3), got {arr.shape}")


def _as_shw3(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr[None, ...]
    raise ValueError(f"{name} must have shape (S,H,W,3) or (H,W,3), got {arr.shape}")


def _as_tshw(value: Any, *, name: str) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 2:
        return arr[None, None, :, :]
    raise ValueError(f"{name} must have shape (T,S,H,W), (T,H,W), or (H,W); got {arr.shape}")
