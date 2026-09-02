from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .pressure_trace_metrics import load_pressure_trace_npz


def align_reference_to_pressure_taxels(
    trace: Mapping[str, Any] | str | Path,
    *,
    reference_key: str = "tacmap_raw_m",
    output_key: str = "tacmap_raw_aligned_m",
    pressure_points_key: str = "pressure_taxel_points_l_m",
    reference_points_key: str = "tacmap_grid_points_l_m",
    reference_axes_key: str = "tacmap_grid_axes_l",
    pressure_valid_key: str = "pressure_taxel_layout_valid",
    reference_valid_key: str = "tacmap_grid_layout_valid",
    distance_mode: str = "plane",
    max_distance_m: float | None = None,
    chunk_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Resample a dense reference map onto the pressure taxel layout.

    ``distance_mode="plane"`` compares points in the reference grid's local
    u/v tangent plane. This is the right default for TacMap link_surface traces:
    TacMap stores ray-start points, while pressure taxels live on the tactile
    surface and can be separated by millimeters along the ray axis.
    ``distance_mode="local3d"`` keeps full local-coordinate nearest neighbor
    matching for debugging layouts that already share the same surface.
    """

    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if reference_key not in data:
        raise KeyError(f"reference_key {reference_key!r} not found in trace")
    if pressure_points_key not in data:
        raise KeyError(f"pressure_points_key {pressure_points_key!r} not found in trace")
    if reference_points_key not in data:
        raise KeyError(f"reference_points_key {reference_points_key!r} not found in trace")

    mode = str(distance_mode).lower()
    if mode not in {"plane", "local3d"}:
        raise ValueError("distance_mode must be 'plane' or 'local3d'")

    reference = _as_tshw(data[reference_key], name=reference_key)
    source_index, valid_mask, nn_distance, summary = build_reference_to_pressure_alignment(
        data[pressure_points_key],
        data[reference_points_key],
        reference_axes=data.get(reference_axes_key),
        pressure_valid=data.get(pressure_valid_key),
        reference_valid=data.get(reference_valid_key),
        distance_mode=mode,
        max_distance_m=max_distance_m,
        chunk_size=chunk_size,
    )
    aligned = apply_reference_alignment_to_values(reference, source_index, valid_mask)

    out = {str(key): value for key, value in data.items()}
    out[output_key] = aligned
    out[f"{output_key}_valid_mask"] = valid_mask
    out[f"{output_key}_nn_distance_m"] = nn_distance
    out[f"{output_key}_source_index"] = source_index

    summary.update(
        {
            "reference_key": reference_key,
            "output_key": output_key,
            "pressure_points_key": pressure_points_key,
            "reference_points_key": reference_points_key,
            "reference_axes_key": reference_axes_key,
            "reference_shape": [int(v) for v in reference.shape],
            "aligned_shape": [int(v) for v in aligned.shape],
            "max_distance_m": None if max_distance_m is None else float(max_distance_m),
        }
    )
    return out, summary


def build_reference_to_pressure_alignment(
    pressure_points: Any,
    reference_points: Any,
    *,
    reference_axes: Any | None = None,
    pressure_valid: Any | None = None,
    reference_valid: Any | None = None,
    distance_mode: str = "plane",
    max_distance_m: float | None = None,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Precompute nearest-neighbor indices from reference pixels to taxels."""

    pressure_points_arr = _as_shw3(pressure_points, name="pressure_points")
    reference_points_arr = _as_shw3(reference_points, name="reference_points")
    reference_axes_arr = _optional_s33(reference_axes, name="reference_axes")
    mode = str(distance_mode).lower()
    if mode not in {"plane", "local3d"}:
        raise ValueError("distance_mode must be 'plane' or 'local3d'")

    sensors = min(int(pressure_points_arr.shape[0]), int(reference_points_arr.shape[0]))
    if reference_axes_arr is not None:
        sensors = min(sensors, int(reference_axes_arr.shape[0]))
    if sensors <= 0:
        raise ValueError("no overlapping sensors between reference and layout arrays")

    pressure_h = int(pressure_points_arr.shape[1])
    pressure_w = int(pressure_points_arr.shape[2])
    source_index = np.full((sensors, pressure_h, pressure_w), -1, dtype=np.int64)
    nn_distance = np.full((sensors, pressure_h, pressure_w), np.inf, dtype=np.float32)
    valid_mask = np.zeros((sensors, pressure_h, pressure_w), dtype=np.uint8)
    pressure_valid_arr = _valid_sensors(pressure_valid, count=sensors)
    reference_valid_arr = _valid_sensors(reference_valid, count=sensors)

    summaries: list[dict[str, Any]] = []
    valid_sensor_indices: list[int] = []
    chunk = max(1, int(chunk_size))
    for sensor_idx in range(sensors):
        if not pressure_valid_arr[sensor_idx] or not reference_valid_arr[sensor_idx]:
            summaries.append({"sensor": sensor_idx, "available": False, "reason": "layout marked invalid"})
            continue

        query_points = np.asarray(pressure_points_arr[sensor_idx], dtype=np.float32).reshape(-1, 3)
        source_points = np.asarray(reference_points_arr[sensor_idx], dtype=np.float32).reshape(-1, 3)
        finite_query = np.all(np.isfinite(query_points), axis=-1)
        finite_source = np.all(np.isfinite(source_points), axis=-1)
        source_valid_indices = np.flatnonzero(finite_source)
        if not np.any(finite_query) or source_valid_indices.size == 0:
            summaries.append({"sensor": sensor_idx, "available": False, "reason": "empty finite point grid"})
            continue

        axes = reference_axes_arr[sensor_idx] if reference_axes_arr is not None and mode == "plane" else None
        query_features, source_features = _alignment_features(
            query_points,
            source_points,
            source_valid_indices=source_valid_indices,
            axes=axes,
        )
        nearest_valid_idx = np.full((query_points.shape[0],), -1, dtype=np.int64)
        nearest_dist = np.full((query_points.shape[0],), np.inf, dtype=np.float32)
        finite_query_indices = np.flatnonzero(finite_query)
        for start in range(0, finite_query_indices.size, chunk):
            query_idx = finite_query_indices[start : start + chunk]
            diff = query_features[query_idx, None, :] - source_features[None, :, :]
            dist2 = np.einsum("qrd,qrd->qr", diff, diff, optimize=True)
            local_idx = np.argmin(dist2, axis=1)
            nearest_valid_idx[query_idx] = source_valid_indices[local_idx]
            nearest_dist[query_idx] = np.sqrt(dist2[np.arange(local_idx.size), local_idx]).astype(np.float32)

        sensor_valid = nearest_valid_idx >= 0
        if max_distance_m is not None:
            sensor_valid &= nearest_dist <= float(max_distance_m)
        if not np.any(sensor_valid):
            summaries.append({"sensor": sensor_idx, "available": False, "reason": "no taxels within max distance"})
            continue

        source_index[sensor_idx] = nearest_valid_idx.reshape(pressure_h, pressure_w)
        nn_distance[sensor_idx] = nearest_dist.reshape(pressure_h, pressure_w)
        valid_mask[sensor_idx] = sensor_valid.reshape(pressure_h, pressure_w).astype(np.uint8)
        finite_dist = nearest_dist[sensor_valid]
        summaries.append(
            {
                "sensor": sensor_idx,
                "available": True,
                "valid_taxel_count": int(np.count_nonzero(sensor_valid)),
                "valid_taxel_fraction": float(np.count_nonzero(sensor_valid) / max(1, sensor_valid.size)),
                "nearest_distance_m_min": float(np.min(finite_dist)),
                "nearest_distance_m_mean": float(np.mean(finite_dist)),
                "nearest_distance_m_max": float(np.max(finite_dist)),
            }
        )
        valid_sensor_indices.append(sensor_idx)

    summary = {
        "distance_mode": mode if reference_axes_arr is not None or mode == "local3d" else "local3d_fallback",
        "pressure_shape": [int(v) for v in pressure_points_arr.shape],
        "reference_points_shape": [int(v) for v in reference_points_arr.shape],
        "max_distance_m": None if max_distance_m is None else float(max_distance_m),
        "valid_sensor_indices": valid_sensor_indices,
        "sensors": summaries,
    }
    return source_index, valid_mask, nn_distance, summary


def apply_reference_alignment_to_values(
    reference: Any,
    source_index: Any,
    valid_mask: Any,
) -> np.ndarray:
    """Apply a precomputed reference-to-pressure alignment to dense values."""

    ref = _as_tshw(reference, name="reference")
    source_idx = np.asarray(source_index, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    if source_idx.ndim != 3:
        raise ValueError(f"source_index must have shape (S,H,W); got {source_idx.shape}")
    if valid.shape != source_idx.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match source_index {source_idx.shape}")

    steps = int(ref.shape[0])
    sensors = min(int(ref.shape[1]), int(source_idx.shape[0]))
    out = np.zeros((steps, sensors, int(source_idx.shape[1]), int(source_idx.shape[2])), dtype=np.float32)
    for sensor_idx in range(sensors):
        flat_ref = ref[:, sensor_idx].reshape(steps, -1)
        flat_out = out[:, sensor_idx].reshape(steps, -1)
        flat_source = source_idx[sensor_idx].reshape(-1)
        flat_valid = valid[sensor_idx].reshape(-1) & (flat_source >= 0) & (flat_source < flat_ref.shape[1])
        flat_out[:, flat_valid] = flat_ref[:, flat_source[flat_valid]]
    return out


def aligned_reference_points_for_pressure_taxels(
    reference_points: Any,
    source_index: Any,
    valid_mask: Any,
    *,
    fallback_points: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map reference-grid points onto the pressure taxel grid.

    This reuses the same nearest-neighbor alignment as dense TacMap values. It
    is useful when a backend needs a physical surface origin while the pressure
    taxel points are offset along the local normal.
    """

    ref_points = _as_shw3(reference_points, name="reference_points")
    source_idx = np.asarray(source_index, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    if source_idx.ndim != 3:
        raise ValueError(f"source_index must have shape (S,H,W); got {source_idx.shape}")
    if valid.shape != source_idx.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match source_index {source_idx.shape}")

    sensors = min(int(ref_points.shape[0]), int(source_idx.shape[0]))
    out_shape = (sensors, int(source_idx.shape[1]), int(source_idx.shape[2]), 3)
    if fallback_points is None:
        out = np.zeros(out_shape, dtype=np.float32)
    else:
        fallback = _as_shw3(fallback_points, name="fallback_points")
        if fallback.shape[:3] != out_shape[:3]:
            raise ValueError(f"fallback_points shape {fallback.shape} does not match aligned shape {out_shape}")
        out = np.asarray(fallback[:sensors], dtype=np.float32).copy()

    out_valid = np.zeros(out_shape[:3], dtype=np.uint8)
    for sensor_idx in range(sensors):
        flat_ref = ref_points[sensor_idx].reshape(-1, 3)
        flat_out = out[sensor_idx].reshape(-1, 3)
        flat_source = source_idx[sensor_idx].reshape(-1)
        flat_valid = valid[sensor_idx].reshape(-1) & (flat_source >= 0) & (flat_source < flat_ref.shape[0])
        flat_out[flat_valid] = flat_ref[flat_source[flat_valid]]
        out_valid[sensor_idx].reshape(-1)[flat_valid] = 1
    return out.astype(np.float32), out_valid


def write_geometry_aligned_trace(
    trace_path: str | Path,
    out_path: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write a new trace with geometry-aligned reference arrays."""

    aligned, summary = align_reference_to_pressure_taxels(trace_path, **kwargs)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **aligned)
    return out, summary


def _alignment_features(
    query_points: np.ndarray,
    source_points: np.ndarray,
    *,
    source_valid_indices: np.ndarray,
    axes: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if axes is None:
        return query_points, source_points[source_valid_indices]

    u_axis = _normalize(axes[0])
    v_axis = _normalize(axes[1])
    center = np.mean(source_points[source_valid_indices], axis=0)
    query_centered = query_points - center
    source_centered = source_points[source_valid_indices] - center
    query_uv = np.stack([query_centered @ u_axis, query_centered @ v_axis], axis=-1).astype(np.float32)
    source_uv = np.stack([source_centered @ u_axis, source_centered @ v_axis], axis=-1).astype(np.float32)
    return query_uv, source_uv


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-8:
        raise ValueError("reference plane axis has near-zero length")
    return arr / norm


def _as_tshw(value: Any, *, name: str) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 2:
        return arr[None, None, :, :]
    raise ValueError(f"{name} must have shape (T,S,H,W), (T,H,W), or (H,W); got {arr.shape}")


def _as_shw3(value: Any, *, name: str) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr[None, :, :, :]
    raise ValueError(f"{name} must have shape (S,H,W,3) or (H,W,3); got {arr.shape}")


def _optional_s33(value: Any | None, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
        return arr
    if arr.ndim == 2 and arr.shape == (3, 3):
        return arr[None, :, :]
    raise ValueError(f"{name} must have shape (S,3,3) or (3,3); got {arr.shape}")


def _valid_sensors(value: Any | None, *, count: int) -> np.ndarray:
    if value is None:
        return np.ones((count,), dtype=bool)
    arr = np.asarray(value).reshape(-1)
    if arr.size < count:
        out = np.zeros((count,), dtype=bool)
        out[: arr.size] = arr.astype(bool)
        return out
    return arr[:count].astype(bool)
