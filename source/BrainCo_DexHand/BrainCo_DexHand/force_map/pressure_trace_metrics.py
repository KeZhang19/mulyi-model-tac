from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DENSE_REFERENCE_ACCEPTANCE_LAYERS = ("L0_analytic", "L1_model_reference", "L2_offline_oracle")
REAL_CALIBRATION_REFERENCE_TOKENS = (
    "tekscan",
    "pps",
    "pressure_film",
    "real_pad",
    "load_cell",
    "gelsight",
    "digit",
    "sparsh",
    "tacbench",
    "feats",
    "feelanyforce",
    "vision_tactile",
)
MODEL_SENSOR_REFERENCE_TOKENS = (
    "tacsl",
    "visuo_tactile",
    "visuotactile",
    "vbts",
)
# Legacy public name: this now means every inferred known layer needs an
# explicit allow flag before it can be manually relabeled.
UNSAFE_REFERENCE_OVERRIDE_LAYERS = (
    "L0_analytic",
    "L1_model_reference",
    "L1_sparse_sanity",
    "L2_offline_oracle",
    "L3_real_calibration",
    "benchmark_reference",
)


def load_pressure_trace_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a saved pressure trace into memory."""

    with np.load(Path(path).expanduser(), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def validate_pressure_trace_v1(trace: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate the stable pressure-trace v1 data contract.

    The validator is intentionally stricter about array shapes than metadata.
    Older traces may not include the v1 metadata keys yet, so missing metadata is
    reported as warnings while malformed core arrays are reported as errors.
    """

    trace_path = Path(trace).expanduser() if isinstance(trace, (str, Path)) else None
    data = load_pressure_trace_npz(trace_path) if trace_path is not None else dict(trace)
    metadata = _pressure_trace_metadata(data, trace_path)
    errors: list[str] = []
    warnings: list[str] = []

    required_tshw = (
        "penetration_m",
        "signed_distance_m",
        "penetration_velocity_mps",
        "pressure_raw_n",
        "pressure_norm",
    )
    required_summary = ("total_force_n", "center_of_pressure_px")
    required_layout = ("pressure_taxel_points_l_m", "pressure_taxel_normals_l")

    for key in required_tshw:
        if key not in data:
            errors.append(f"missing required array {key!r}")
    if errors:
        return _pressure_trace_validation_result(errors, warnings, metadata, None)

    try:
        pressure_shape = _as_tshw(data["pressure_norm"], name="pressure_norm").shape
    except ValueError as exc:
        errors.append(str(exc))
        return _pressure_trace_validation_result(errors, warnings, metadata, None)
    steps, sensors, rows, cols = (int(v) for v in pressure_shape)
    for key in required_tshw:
        try:
            shape = _as_tshw(data[key], name=key).shape
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if tuple(shape) != tuple(pressure_shape):
            errors.append(f"{key!r} shape {tuple(shape)} must match pressure_norm shape {tuple(pressure_shape)}")

    for key in required_summary:
        if key not in data:
            errors.append(f"missing required array {key!r}")
    if "total_force_n" in data:
        shape = tuple(np.asarray(data["total_force_n"]).shape)
        if shape != (steps, sensors):
            errors.append(f"'total_force_n' shape {shape} must be {(steps, sensors)}")
    if "center_of_pressure_px" in data:
        shape = tuple(np.asarray(data["center_of_pressure_px"]).shape)
        if shape != (steps, sensors, 2):
            errors.append(f"'center_of_pressure_px' shape {shape} must be {(steps, sensors, 2)}")

    for key in required_layout:
        if key not in data:
            errors.append(f"missing required layout array {key!r}")
    if all(key in data for key in required_layout):
        layout_shape = tuple(np.asarray(data["pressure_taxel_points_l_m"]).shape)
        normal_shape = tuple(np.asarray(data["pressure_taxel_normals_l"]).shape)
        expected_layout_shape = (sensors, rows, cols, 3)
        legacy_layout_shape = (sensors, rows * cols, 3)
        if layout_shape not in (expected_layout_shape, legacy_layout_shape):
            errors.append(
                f"'pressure_taxel_points_l_m' shape {layout_shape} must be "
                f"{expected_layout_shape} or {legacy_layout_shape}"
            )
        if normal_shape not in (expected_layout_shape, legacy_layout_shape):
            errors.append(
                f"'pressure_taxel_normals_l' shape {normal_shape} must be "
                f"{expected_layout_shape} or {legacy_layout_shape}"
            )

    optional_tshw = (
        "tacmap_raw_m",
        "normal_ray_penetration_m",
        "normal_ray_pressure_raw_n",
        "normal_ray_pressure_norm",
        "geometry_normal_ray_penetration_m",
        "geometry_normal_ray_pressure_raw_n",
        "geometry_normal_ray_pressure_norm",
        "geometry_normal_ray_sample_support_fraction",
        "geometry_normal_ray_sample_active_count",
        "geometry_normal_ray_sample_mean_penetration_m",
        "geometry_normal_ray_sample_positive_mean_penetration_m",
        "geometry_normal_ray_sample_max_penetration_m",
    )
    for key in optional_tshw:
        if key in data and np.asarray(data[key]).size:
            try:
                shape = _as_tshw(data[key], name=key).shape
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if tuple(shape[:2]) != (steps, sensors):
                errors.append(f"{key!r} leading shape {tuple(shape[:2])} must be {(steps, sensors)}")

    optional_samples = {
        "geometry_normal_ray_sample_penetrations_m": (steps, sensors, rows, cols),
        "geometry_normal_ray_sample_offsets_l": (steps, sensors, rows, cols),
        "geometry_normal_ray_sample_points_l_m": (steps, sensors, rows, cols),
    }
    for key, prefix in optional_samples.items():
        if key not in data or not np.asarray(data[key]).size:
            continue
        shape = tuple(np.asarray(data[key]).shape)
        if shape[:4] != prefix:
            errors.append(f"{key!r} leading shape {shape[:4]} must be {prefix}")
        if key.endswith(("_offsets_l", "_points_l_m")) and (len(shape) != 6 or shape[-1] != 3):
            errors.append(f"{key!r} must have shape (T,S,H,W,K,3), got {shape}")
        if key.endswith("_penetrations_m") and len(shape) != 5:
            errors.append(f"{key!r} must have shape (T,S,H,W,K), got {shape}")

    for key, value in data.items():
        if not (str(key).endswith("_valid_mask") or str(key).endswith("_layout_valid")):
            continue
        shape = tuple(np.asarray(value).shape)
        allowed = {
            (sensors,),
            (sensors, rows, cols),
            (steps, sensors, rows, cols),
        }
        if sensors == 1:
            allowed.add((rows, cols))
        if shape not in allowed:
            errors.append(f"{key!r} shape {shape} must be one of {sorted(allowed)}")

    metadata_required = (
        "pressure_trace_schema_version",
        "pressure_backend_id",
        "pressure_layout_id",
        "pressure_calibration_id",
    )
    for key in metadata_required:
        if key not in metadata:
            warnings.append(f"metadata missing {key!r}")
    if metadata.get("metadata_json_parse_error"):
        warnings.append("metadata_json parse failed")
    if metadata.get("metadata_json_not_object"):
        warnings.append("metadata_json is not an object")
    if metadata.get("metadata_file_parse_error"):
        warnings.append("metadata sidecar parse failed")
    if metadata.get("metadata_file_not_object"):
        warnings.append("metadata sidecar is not an object")
    schema_version = metadata.get("pressure_trace_schema_version")
    if schema_version is not None and str(schema_version) != "pressure_trace_v1":
        errors.append(f"unsupported pressure_trace_schema_version {schema_version!r}")

    shape_summary = {
        "num_steps": steps,
        "sensor_count": sensors,
        "image_shape": [rows, cols],
    }
    return _pressure_trace_validation_result(errors, warnings, metadata, shape_summary)


def pressure_trace_report(
    trace: Mapping[str, Any] | str | Path,
    *,
    pressure_key: str = "pressure_norm",
    raw_pressure_key: str = "pressure_raw_n",
    penetration_key: str = "penetration_m",
    reference_key: str | None = None,
    reference_valid_mask_key: str | None = None,
    reference_layer: str | None = None,
    active_threshold: float = 1.0e-6,
    penetration_threshold: float = 0.0,
    reference_threshold: float = 0.0,
) -> dict[str, Any]:
    """Compute pressure-trace quality metrics.

    The report is intentionally independent from Isaac. It can be used on raw
    ``--save-pressure-trace`` files, short smoke tests, or future real/FEA
    reference dumps as long as arrays follow the ``(T, S, H, W)`` pressure-map
    contract.
    """

    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    pressure = _as_tshw(data[pressure_key], name=pressure_key)
    active = pressure > float(active_threshold)
    active_counts = active.sum(axis=(-2, -1)).astype(np.int64)

    report: dict[str, Any] = {
        "pressure_key": pressure_key,
        "num_steps": int(pressure.shape[0]),
        "sensor_count": int(pressure.shape[1]),
        "image_shape": [int(pressure.shape[2]), int(pressure.shape[3])],
        "active_threshold": float(active_threshold),
        "active_taxels_per_step": active_counts.tolist(),
        "max_active_taxels": int(active_counts.max()) if active_counts.size else 0,
        "onset_step_by_sensor": _onset_steps(active_counts),
        "offset_step_by_sensor": _offset_steps(active_counts),
    }

    penetration = None
    if penetration_key in data:
        penetration = _as_tshw(data[penetration_key], name=penetration_key)
        if penetration.shape == pressure.shape:
            report.update(
                _precontact_leakage_metrics(
                    active,
                    active_counts,
                    penetration,
                    penetration_threshold=float(penetration_threshold),
                )
            )

    total_force = _total_force(data, raw_pressure_key=raw_pressure_key)
    if total_force is not None:
        report["total_force_n_max_by_sensor"] = np.nanmax(total_force, axis=0).astype(np.float64).tolist()
        if penetration is not None and penetration.shape == pressure.shape:
            spearman = _force_depth_spearman(total_force, penetration)
            report["force_depth_spearman_by_sensor"] = spearman
            finite = [value for value in spearman if value is not None]
            report["force_depth_spearman_min"] = min(finite) if finite else None

    if reference_key:
        if reference_key not in data:
            raise KeyError(f"reference_key {reference_key!r} not found in trace")
        resolved_valid_mask_key = _resolve_reference_valid_mask_key(
            reference_key,
            data,
            requested_key=reference_valid_mask_key,
        )
        reference_valid_mask = data.get(resolved_valid_mask_key)
        if reference_valid_mask_key is not None and reference_valid_mask is None:
            raise KeyError(f"reference_valid_mask_key {reference_valid_mask_key!r} not found in trace")
        inferred_reference_layer = infer_pressure_reference_layer(reference_key)
        requested_reference_layer = None if reference_layer is None else str(reference_layer)
        resolved_reference_layer = requested_reference_layer or inferred_reference_layer
        override_conflict = (
            requested_reference_layer is not None
            and inferred_reference_layer is not None
            and requested_reference_layer != inferred_reference_layer
        )
        try:
            reference_report = reference_mask_metrics(
                pressure,
                data[reference_key],
                pressure_threshold=float(active_threshold),
                reference_threshold=float(reference_threshold),
                predicted_depth_m=penetration,
                reference_valid_mask=reference_valid_mask,
            )
        except ValueError as exc:
            reference_report = {
                "available": False,
                "reason": str(exc),
                "reference_shape": [int(value) for value in np.asarray(data[reference_key]).shape],
            }
        reference_report["reference_key"] = str(reference_key)
        reference_report["reference_valid_mask_key"] = (
            str(resolved_valid_mask_key) if reference_valid_mask is not None else None
        )
        reference_report["reference_layer"] = resolved_reference_layer
        reference_report["inferred_reference_layer"] = inferred_reference_layer
        reference_report["requested_reference_layer"] = requested_reference_layer
        reference_report["reference_layer_override_conflict"] = bool(override_conflict)
        if _has_reference_origin_alignment_inputs(data):
            origin_alignment = pressure_reference_origin_alignment_diagnostics(data)
            reference_report["origin_alignment"] = origin_alignment
            report["reference_origin_alignment"] = origin_alignment
        report["reference_key"] = str(reference_key)
        report["reference_layer"] = resolved_reference_layer
        report["inferred_reference_layer"] = inferred_reference_layer
        report["requested_reference_layer"] = requested_reference_layer
        report["reference_layer_override_conflict"] = bool(override_conflict)
        report["reference"] = reference_report

    return report


def pressure_reference_origin_alignment_diagnostics(
    trace: Mapping[str, Any] | str | Path,
    *,
    pressure_points_key: str = "pressure_taxel_points_l_m",
    pressure_normals_key: str = "pressure_taxel_normals_l",
    reference_points_key: str = "tacmap_grid_points_l_m",
    source_index_key: str = "normal_ray_alignment_source_index",
    valid_mask_key: str = "normal_ray_alignment_valid_mask",
    nn_distance_key: str = "normal_ray_alignment_nn_distance_m",
) -> dict[str, Any]:
    """Measure the origin mismatch between pressure taxels and aligned reference rays."""

    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    required = (
        pressure_points_key,
        pressure_normals_key,
        reference_points_key,
        source_index_key,
        valid_mask_key,
    )
    missing = [key for key in required if key not in data]
    if missing:
        return {
            "available": False,
            "reason": "missing required arrays",
            "missing_keys": missing,
        }

    try:
        source_index, _source_shape = _as_sensor_index(data[source_index_key], name=source_index_key)
        valid_mask = _as_sensor_bool_mask(data[valid_mask_key], name=valid_mask_key, source_index=data[source_index_key])
        pressure_points = _as_sensor_points(
            data[pressure_points_key],
            name=pressure_points_key,
            sensor_count_hint=source_index.shape[0],
        )
        pressure_normals = _as_sensor_points(
            data[pressure_normals_key],
            name=pressure_normals_key,
            sensor_count_hint=source_index.shape[0],
        )
        reference_points = _as_sensor_points(
            data[reference_points_key],
            name=reference_points_key,
            sensor_count_hint=source_index.shape[0],
        )
        nn_distance = None
        if nn_distance_key in data:
            nn_distance, _ = _as_sensor_index(data[nn_distance_key], name=nn_distance_key)
    except ValueError as exc:
        return {
            "available": False,
            "reason": str(exc),
        }

    sensors = min(
        int(source_index.shape[0]),
        int(valid_mask.shape[0]),
        int(pressure_points.shape[0]),
        int(pressure_normals.shape[0]),
        int(reference_points.shape[0]),
    )
    by_sensor: list[dict[str, Any]] = []
    all_delta_norm: list[np.ndarray] = []
    all_normal_delta: list[np.ndarray] = []
    all_tangent_delta: list[np.ndarray] = []
    all_nn_distance: list[np.ndarray] = []
    total_taxels = 0
    valid_taxels = 0
    for sensor_idx in range(sensors):
        taxel_count = min(
            int(source_index.shape[1]),
            int(valid_mask.shape[1]),
            int(pressure_points.shape[1]),
            int(pressure_normals.shape[1]),
        )
        total_taxels += taxel_count
        if taxel_count <= 0:
            by_sensor.append({"sensor": int(sensor_idx), "total_taxels": 0, "valid_taxels": 0})
            continue
        source = np.asarray(source_index[sensor_idx, :taxel_count], dtype=np.int64)
        valid = np.asarray(valid_mask[sensor_idx, :taxel_count], dtype=bool)
        valid &= source >= 0
        valid &= source < int(reference_points.shape[1])
        valid_count = int(np.count_nonzero(valid))
        valid_taxels += valid_count
        sensor_report: dict[str, Any] = {
            "sensor": int(sensor_idx),
            "total_taxels": int(taxel_count),
            "valid_taxels": valid_count,
            "valid_fraction": float(valid_count / max(1, taxel_count)),
        }
        if valid_count <= 0:
            by_sensor.append(sensor_report)
            continue

        pressure_selected = pressure_points[sensor_idx, :taxel_count][valid]
        normal_selected = pressure_normals[sensor_idx, :taxel_count][valid]
        reference_selected = reference_points[sensor_idx, source[valid]]
        normal_selected = _normalize_vectors(normal_selected)
        delta = reference_selected - pressure_selected
        normal_delta = np.sum(delta * normal_selected, axis=-1)
        tangent_delta = np.linalg.norm(delta - normal_delta[:, None] * normal_selected, axis=-1)
        delta_norm = np.linalg.norm(delta, axis=-1)
        all_delta_norm.append(delta_norm)
        all_normal_delta.append(normal_delta)
        all_tangent_delta.append(tangent_delta)
        sensor_report.update(_metric_stats("delta_norm_m", delta_norm))
        sensor_report.update(_metric_stats("normal_delta_m", normal_delta))
        sensor_report.update(_metric_stats("abs_normal_delta_m", np.abs(normal_delta)))
        sensor_report.update(_metric_stats("tangent_delta_m", tangent_delta))
        if nn_distance is not None and sensor_idx < int(nn_distance.shape[0]):
            nn_count = min(taxel_count, int(nn_distance.shape[1]))
            if nn_count == taxel_count:
                selected_nn = np.asarray(nn_distance[sensor_idx, :taxel_count], dtype=np.float64)[valid]
                all_nn_distance.append(selected_nn)
                sensor_report.update(_metric_stats("nn_distance_m", selected_nn))
        by_sensor.append(sensor_report)

    if valid_taxels <= 0:
        return {
            "available": False,
            "reason": "no valid aligned reference origins",
            "sensor_count": int(sensors),
            "total_taxels": int(total_taxels),
            "valid_taxels": 0,
        }

    delta_norm_all = np.concatenate(all_delta_norm) if all_delta_norm else np.zeros((0,), dtype=np.float64)
    normal_delta_all = np.concatenate(all_normal_delta) if all_normal_delta else np.zeros((0,), dtype=np.float64)
    tangent_delta_all = np.concatenate(all_tangent_delta) if all_tangent_delta else np.zeros((0,), dtype=np.float64)
    report: dict[str, Any] = {
        "available": True,
        "pressure_points_key": str(pressure_points_key),
        "pressure_normals_key": str(pressure_normals_key),
        "reference_points_key": str(reference_points_key),
        "source_index_key": str(source_index_key),
        "valid_mask_key": str(valid_mask_key),
        "delta_direction": f"{reference_points_key} - {pressure_points_key}",
        "normal_delta_sign": "positive means reference origin is along the pressure taxel normal",
        "sensor_count": int(sensors),
        "total_taxels": int(total_taxels),
        "valid_taxels": int(valid_taxels),
        "valid_fraction": float(valid_taxels / max(1, total_taxels)),
        "by_sensor": by_sensor,
    }
    report.update(_metric_stats("delta_norm_m", delta_norm_all))
    report.update(_metric_stats("normal_delta_m", normal_delta_all))
    report.update(_metric_stats("abs_normal_delta_m", np.abs(normal_delta_all)))
    report.update(_metric_stats("tangent_delta_m", tangent_delta_all))
    if all_nn_distance:
        report.update(_metric_stats("nn_distance_m", np.concatenate(all_nn_distance)))
    return report


def infer_pressure_reference_layer(reference_key: str | None) -> str | None:
    """Infer the validation layer for common pressure trace reference arrays.

    The returned value is metadata only. It prevents L1 model references such as
    TacMap/normal-ray traces from being mistaken for physical ground truth.
    """

    if not reference_key:
        return None
    key = str(reference_key).lower()
    signed_distance_like = "signed_distance" in key or ("signed" in key and "distance" in key)
    velocity_like = "velocity" in key
    raw_sample_penetrations_like = "sample_penetrations" in key
    depth_like = (
        any(token in key for token in ("penetration", "depth", "deformation"))
        and not signed_distance_like
        and not velocity_like
        and not raw_sample_penetrations_like
    )
    if any(token in key for token in MODEL_SENSOR_REFERENCE_TOKENS):
        return "benchmark_reference"
    if any(token in key for token in REAL_CALIBRATION_REFERENCE_TOKENS):
        return "L3_real_calibration"
    if "analytic" in key and depth_like:
        return "L0_analytic"
    if any(token in key for token in ("hydroelastic", "fem", "uipc", "ipc")) and depth_like:
        return "L2_offline_oracle"
    if any(token in key for token in ("tacmap", "normal_ray", "warpsdf", "sdf", "geometry")) and (
        depth_like or key in {"tacmap_raw_m", "tacmap_raw_aligned_m"}
    ):
        return "L1_model_reference"
    if "physx" in key or "contact_force" in key or "contact_count" in key:
        return "L1_sparse_sanity"
    return "benchmark_reference"


def reference_mask_metrics(
    pressure: np.ndarray,
    reference: Any,
    *,
    pressure_threshold: float,
    reference_threshold: float,
    predicted_depth_m: Any | None = None,
    reference_valid_mask: Any | None = None,
) -> dict[str, Any]:
    """Compare pressure active masks and optional depth against a dense reference map."""

    pressure = _as_tshw(pressure, name="pressure")
    ref = _as_tshw(reference, name="reference")
    pred_depth = _as_tshw(predicted_depth_m, name="predicted_depth_m") if predicted_depth_m is not None else None
    valid_mask_arr = None if reference_valid_mask is None else np.asarray(reference_valid_mask, dtype=bool)
    if ref.shape[1] == 0:
        return {"available": False, "reason": "reference has zero sensors"}

    steps = min(int(pressure.shape[0]), int(ref.shape[0]))
    sensors = min(int(pressure.shape[1]), int(ref.shape[1]))
    if pred_depth is not None:
        steps = min(steps, int(pred_depth.shape[0]))
        sensors = min(sensors, int(pred_depth.shape[1]))
    if steps <= 0 or sensors <= 0:
        return {"available": False, "reason": "no overlapping steps or sensors"}

    pressure_mask = pressure[:steps, :sensors] > float(pressure_threshold)
    ref_values = ref[:steps, :sensors]
    ref_mask = ref_values > float(reference_threshold)
    target_shape = tuple(int(v) for v in pressure_mask.shape[-2:])

    ious: list[float] = []
    centroid_errors: list[float] = []
    bbox_errors: list[float] = []
    depth_rmse_values: list[float] = []
    alignment_valid_fractions: list[float] = []
    alignment_contact_valid_fractions: list[float] = []
    alignment_invalid_contact_fractions: list[float] = []
    alignment_invalid_zero_fill_fractions: list[float] = []
    onset_pressure = _onset_steps(pressure_mask.sum(axis=(-2, -1)).astype(np.int64))
    offset_pressure = _offset_steps(pressure_mask.sum(axis=(-2, -1)).astype(np.int64))
    onset_ref = []
    offset_ref = []
    onset_errors = []
    offset_errors = []
    for sensor_idx in range(sensors):
        ref_counts = []
        for step_idx in range(steps):
            ref_step_values = _resize_values_to_shape(ref_values[step_idx, sensor_idx], target_shape)
            ref_step = ref_step_values > float(reference_threshold)
            ref_counts.append(int(np.count_nonzero(ref_step)))
            pressure_step = pressure_mask[step_idx, sensor_idx]
            union = pressure_step | ref_step
            if valid_mask_arr is not None:
                valid_step = _reference_valid_mask_step(
                    valid_mask_arr,
                    step_idx=step_idx,
                    sensor_idx=sensor_idx,
                    target_shape=target_shape,
                )
                invalid_step = ~valid_step
                alignment_valid_fractions.append(
                    float(np.count_nonzero(valid_step) / max(1, valid_step.size))
                )
                zero_like_ref = np.asarray(ref_step_values) <= float(reference_threshold)
                alignment_invalid_zero_fill_fractions.append(
                    float(np.count_nonzero(invalid_step & zero_like_ref) / max(1, valid_step.size))
                )
                if np.any(union):
                    union_count = max(1, int(np.count_nonzero(union)))
                    alignment_contact_valid_fractions.append(
                        float(np.count_nonzero(valid_step & union) / union_count)
                    )
                    alignment_invalid_contact_fractions.append(
                        float(np.count_nonzero(invalid_step & union) / union_count)
                    )
            if not np.any(union):
                continue
            intersection = pressure_step & ref_step
            ious.append(float(np.count_nonzero(intersection) / max(1, np.count_nonzero(union))))
            if np.any(pressure_step) and np.any(ref_step):
                centroid_errors.append(float(np.linalg.norm(_mask_centroid(pressure_step) - _mask_centroid(ref_step))))
                bbox_errors.append(_bbox_error_px(_mask_bbox(pressure_step), _mask_bbox(ref_step)))
                if pred_depth is not None:
                    pred_step = _resize_values_to_shape(pred_depth[step_idx, sensor_idx], target_shape)
                    diff = np.asarray(pred_step, dtype=np.float64) - np.asarray(ref_step_values, dtype=np.float64)
                    depth_rmse_values.append(float(np.sqrt(np.mean(np.square(diff[union])))))
        ref_counts_arr = np.asarray(ref_counts, dtype=np.int64)
        sensor_ref_onset = _first_positive(ref_counts_arr)
        sensor_ref_offset = _last_positive(ref_counts_arr)
        onset_ref.append(sensor_ref_onset)
        offset_ref.append(sensor_ref_offset)
        pressure_onset = onset_pressure[sensor_idx]
        pressure_offset = offset_pressure[sensor_idx]
        onset_errors.append(None if pressure_onset is None or sensor_ref_onset is None else int(abs(int(pressure_onset) - int(sensor_ref_onset))))
        offset_errors.append(None if pressure_offset is None or sensor_ref_offset is None else int(abs(int(pressure_offset) - int(sensor_ref_offset))))

    report = {
        "available": True,
        "reference_shape": [int(v) for v in ref.shape],
        "compared_steps": int(steps),
        "compared_sensors": int(sensors),
        "frame_count": int(len(ious)),
        "active_mask_iou_min": min(ious) if ious else None,
        "active_mask_iou_mean": float(np.mean(ious)) if ious else None,
        "centroid_error_px_max": max(centroid_errors) if centroid_errors else None,
        "centroid_error_px_mean": float(np.mean(centroid_errors)) if centroid_errors else None,
        "bbox_error_px_max": max(bbox_errors) if bbox_errors else None,
        "bbox_error_px_mean": float(np.mean(bbox_errors)) if bbox_errors else None,
        "depth_rmse_m_max": max(depth_rmse_values) if depth_rmse_values else None,
        "depth_rmse_m_mean": float(np.mean(depth_rmse_values)) if depth_rmse_values else None,
        "pressure_onset_step_by_sensor": onset_pressure[:sensors],
        "pressure_offset_step_by_sensor": offset_pressure[:sensors],
        "reference_onset_step_by_sensor": onset_ref,
        "reference_offset_step_by_sensor": offset_ref,
        "onset_error_frames_by_sensor": onset_errors,
        "offset_error_frames_by_sensor": offset_errors,
    }
    if valid_mask_arr is not None:
        report.update(
            {
                "alignment_valid_mask_available": True,
                "alignment_valid_taxel_fraction_min": (
                    min(alignment_valid_fractions) if alignment_valid_fractions else None
                ),
                "alignment_valid_taxel_fraction_mean": (
                    float(np.mean(alignment_valid_fractions)) if alignment_valid_fractions else None
                ),
                "alignment_contact_region_valid_fraction_min": (
                    min(alignment_contact_valid_fractions) if alignment_contact_valid_fractions else None
                ),
                "alignment_contact_region_valid_fraction_mean": (
                    float(np.mean(alignment_contact_valid_fractions))
                    if alignment_contact_valid_fractions
                    else None
                ),
                "alignment_invalid_contact_fraction_max": (
                    max(alignment_invalid_contact_fractions) if alignment_invalid_contact_fractions else None
                ),
                "alignment_invalid_zero_fill_fraction_max": (
                    max(alignment_invalid_zero_fill_fractions) if alignment_invalid_zero_fill_fractions else None
                ),
                "alignment_invalid_zero_fill_fraction_mean": (
                    float(np.mean(alignment_invalid_zero_fill_fractions))
                    if alignment_invalid_zero_fill_fractions
                    else None
                ),
            }
        )
    return report


def pressure_reference_frame_diagnostics(
    trace: Mapping[str, Any] | str | Path,
    *,
    pressure_key: str = "pressure_norm",
    penetration_key: str = "penetration_m",
    reference_key: str,
    reference_layer: str | None = None,
    active_threshold: float = 1.0e-6,
    reference_threshold: float = 0.0,
    top_k: int = 8,
) -> dict[str, Any]:
    """Return per-frame reference mismatch diagnostics for a pressure trace."""

    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if reference_key not in data:
        raise KeyError(f"reference_key {reference_key!r} not found in trace")

    inferred_reference_layer = infer_pressure_reference_layer(reference_key)
    requested_reference_layer = None if reference_layer is None else str(reference_layer)
    resolved_reference_layer = requested_reference_layer or inferred_reference_layer
    override_conflict = (
        requested_reference_layer is not None
        and inferred_reference_layer is not None
        and requested_reference_layer != inferred_reference_layer
    )
    pressure = _as_tshw(data[pressure_key], name=pressure_key)
    try:
        reference = _as_tshw(data[reference_key], name=reference_key)
    except ValueError as exc:
        return {
            "summary": {
                "available": False,
                "reason": str(exc),
                "pressure_key": str(pressure_key),
                "penetration_key": str(penetration_key),
                "reference_key": str(reference_key),
                "reference_shape": [int(value) for value in np.asarray(data[reference_key]).shape],
                "reference_layer": resolved_reference_layer,
                "inferred_reference_layer": inferred_reference_layer,
                "requested_reference_layer": requested_reference_layer,
                "reference_layer_override_conflict": bool(override_conflict),
            },
            "frames": [],
        }
    pred_depth = _as_tshw(data[penetration_key], name=penetration_key) if penetration_key in data else pressure
    steps = min(int(pressure.shape[0]), int(reference.shape[0]), int(pred_depth.shape[0]))
    sensors = min(int(pressure.shape[1]), int(reference.shape[1]), int(pred_depth.shape[1]))
    target_shape = tuple(int(v) for v in pressure.shape[-2:])
    if steps <= 0 or sensors <= 0:
        return {
            "summary": {
                "available": False,
                "reason": "no overlapping steps or sensors",
                "pressure_key": str(pressure_key),
                "penetration_key": str(penetration_key),
                "reference_key": str(reference_key),
                "reference_layer": resolved_reference_layer,
                "inferred_reference_layer": inferred_reference_layer,
                "requested_reference_layer": requested_reference_layer,
                "reference_layer_override_conflict": bool(override_conflict),
            },
            "frames": [],
        }

    rows: list[dict[str, Any]] = []
    pressure_counts = np.zeros((steps, sensors), dtype=np.int64)
    reference_counts = np.zeros((steps, sensors), dtype=np.int64)
    false_positive_total = 0
    false_negative_total = 0
    depth_rmse_values: list[float] = []
    depth_bias_values: list[float] = []
    for step_idx in range(steps):
        for sensor_idx in range(sensors):
            pred_mask = pressure[step_idx, sensor_idx] > float(active_threshold)
            ref_values = _resize_values_to_shape(reference[step_idx, sensor_idx], target_shape)
            ref_mask = ref_values > float(reference_threshold)
            pred_values = _resize_values_to_shape(pred_depth[step_idx, sensor_idx], target_shape)
            union = pred_mask | ref_mask
            intersection = pred_mask & ref_mask
            false_positive = pred_mask & ~ref_mask
            false_negative = ref_mask & ~pred_mask
            pred_active = int(np.count_nonzero(pred_mask))
            ref_active = int(np.count_nonzero(ref_mask))
            false_positive_count = int(np.count_nonzero(false_positive))
            false_negative_count = int(np.count_nonzero(false_negative))
            pressure_counts[step_idx, sensor_idx] = pred_active
            reference_counts[step_idx, sensor_idx] = ref_active
            false_positive_total += false_positive_count
            false_negative_total += false_negative_count

            iou = None
            centroid_error = None
            bbox_error = None
            depth_rmse = None
            depth_bias = None
            if np.any(union):
                iou = float(np.count_nonzero(intersection) / max(1, int(np.count_nonzero(union))))
                diff = np.asarray(pred_values, dtype=np.float64) - np.asarray(ref_values, dtype=np.float64)
                union_diff = diff[union]
                depth_rmse = float(np.sqrt(np.mean(np.square(union_diff))))
                depth_bias = float(np.mean(union_diff))
                depth_rmse_values.append(depth_rmse)
                depth_bias_values.append(depth_bias)
            if pred_active > 0 and ref_active > 0:
                centroid_error = float(np.linalg.norm(_mask_centroid(pred_mask) - _mask_centroid(ref_mask)))
                bbox_error = _bbox_error_px(_mask_bbox(pred_mask), _mask_bbox(ref_mask))

            rows.append(
                {
                    "step": int(step_idx),
                    "sensor": int(sensor_idx),
                    "pred_active_taxels": pred_active,
                    "reference_active_taxels": ref_active,
                    "true_positive_taxels": int(np.count_nonzero(intersection)),
                    "false_positive_taxels": false_positive_count,
                    "false_negative_taxels": false_negative_count,
                    "iou": iou,
                    "centroid_error_px": centroid_error,
                    "bbox_error_px": bbox_error,
                    "depth_rmse_m": depth_rmse,
                    "depth_bias_m": depth_bias,
                    "pred_depth_sum_m": float(np.sum(np.where(pred_mask, pred_values, 0.0))),
                    "reference_depth_sum_m": float(np.sum(np.where(ref_mask, ref_values, 0.0))),
                    "false_positive_depth_sum_m": float(np.sum(np.where(false_positive, pred_values, 0.0))),
                    "false_negative_reference_depth_sum_m": float(np.sum(np.where(false_negative, ref_values, 0.0))),
                    "classification": _reference_frame_classification(
                        pred_active,
                        ref_active,
                        false_positive_count,
                        false_negative_count,
                    ),
                }
            )

    summary = {
        "available": True,
        "pressure_key": str(pressure_key),
        "penetration_key": str(penetration_key),
        "reference_key": str(reference_key),
        "reference_layer": resolved_reference_layer,
        "inferred_reference_layer": inferred_reference_layer,
        "requested_reference_layer": requested_reference_layer,
        "reference_layer_override_conflict": bool(override_conflict),
        "num_steps": int(steps),
        "sensor_count": int(sensors),
        "image_shape": [int(v) for v in target_shape],
        "active_threshold": float(active_threshold),
        "reference_threshold": float(reference_threshold),
        "pressure_onset_step_by_sensor": _onset_steps(pressure_counts),
        "reference_onset_step_by_sensor": _onset_steps(reference_counts),
        "pressure_offset_step_by_sensor": _offset_steps(pressure_counts),
        "reference_offset_step_by_sensor": _offset_steps(reference_counts),
        "false_positive_taxels_total": int(false_positive_total),
        "false_negative_taxels_total": int(false_negative_total),
        "frames_with_false_positive": int(sum(1 for row in rows if row["false_positive_taxels"] > 0)),
        "frames_with_false_negative": int(sum(1 for row in rows if row["false_negative_taxels"] > 0)),
        "active_frame_count": int(sum(1 for row in rows if row["pred_active_taxels"] > 0 or row["reference_active_taxels"] > 0)),
        "depth_rmse_m_max": max(depth_rmse_values) if depth_rmse_values else None,
        "depth_rmse_m_mean": float(np.mean(depth_rmse_values)) if depth_rmse_values else None,
        "depth_bias_m_mean": float(np.mean(depth_bias_values)) if depth_bias_values else None,
        "worst_iou_frames": _top_frame_rows(rows, key="iou", top_k=top_k, reverse=False),
        "worst_depth_rmse_frames": _top_frame_rows(rows, key="depth_rmse_m", top_k=top_k, reverse=True),
        "largest_false_positive_frames": _top_frame_rows(rows, key="false_positive_taxels", top_k=top_k, reverse=True),
        "largest_false_negative_frames": _top_frame_rows(rows, key="false_negative_taxels", top_k=top_k, reverse=True),
    }
    if _has_reference_origin_alignment_inputs(data):
        summary["origin_alignment"] = pressure_reference_origin_alignment_diagnostics(data)
    return {"summary": summary, "frames": rows}


def pressure_sample_boundary_diagnostics(
    trace: Mapping[str, Any] | str | Path,
    *,
    pressure_key: str = "geometry_normal_ray_pressure_norm",
    penetration_key: str = "geometry_normal_ray_penetration_m",
    reference_key: str = "normal_ray_penetration_m",
    sample_penetrations_key: str = "geometry_normal_ray_sample_penetrations_m",
    active_threshold: float = 1.0e-6,
    reference_threshold: float = 0.0,
    top_k: int = 8,
) -> dict[str, Any]:
    """Explain dense-reference boundary mismatches with raw per-taxel samples.

    ``sample_penetrations_key`` is expected to use shape ``(T, S, H, W, K)``.
    The returned group summaries make false positives/negatives auditable
    without promoting the chosen dense reference to physical ground truth.
    """

    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if reference_key not in data:
        raise KeyError(f"reference_key {reference_key!r} not found in trace")
    if sample_penetrations_key not in data:
        raise KeyError(f"sample_penetrations_key {sample_penetrations_key!r} not found in trace")

    inferred_reference_layer = infer_pressure_reference_layer(reference_key)
    pressure = _as_tshw(data[pressure_key], name=pressure_key)
    try:
        reference = _as_tshw(data[reference_key], name=reference_key)
    except ValueError as exc:
        return {
            "summary": {
                "available": False,
                "reason": str(exc),
                "pressure_key": str(pressure_key),
                "penetration_key": str(penetration_key),
                "reference_key": str(reference_key),
                "reference_shape": [int(value) for value in np.asarray(data[reference_key]).shape],
                "reference_layer": inferred_reference_layer,
                "sample_penetrations_key": str(sample_penetrations_key),
            },
            "frames": [],
        }
    pred_depth = _as_tshw(data[penetration_key], name=penetration_key) if penetration_key in data else pressure
    samples = np.nan_to_num(
        np.asarray(data[sample_penetrations_key], dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if samples.ndim != 5:
        raise ValueError(
            f"{sample_penetrations_key} must have shape (T,S,H,W,K); got {samples.shape}"
        )

    steps = min(int(pressure.shape[0]), int(reference.shape[0]), int(pred_depth.shape[0]), int(samples.shape[0]))
    sensors = min(int(pressure.shape[1]), int(reference.shape[1]), int(pred_depth.shape[1]), int(samples.shape[1]))
    target_shape = tuple(int(v) for v in pressure.shape[-2:])
    if steps <= 0 or sensors <= 0:
        return {
            "summary": {
                "available": False,
                "reason": "no overlapping steps or sensors",
                "pressure_key": pressure_key,
                "penetration_key": penetration_key,
                "reference_key": reference_key,
                "sample_penetrations_key": sample_penetrations_key,
            },
            "frames": [],
        }
    if tuple(samples.shape[2:4]) != target_shape:
        raise ValueError(
            f"{sample_penetrations_key} spatial shape {samples.shape[2:4]} does not match pressure shape {target_shape}"
        )

    pressure = pressure[:steps, :sensors]
    pred_depth = pred_depth[:steps, :sensors]
    samples = samples[:steps, :sensors]
    sample_active = samples > 0.0
    sample_active_count = np.count_nonzero(sample_active, axis=-1).astype(np.float32)
    sample_count = int(samples.shape[-1])
    sample_support = sample_active_count / float(max(sample_count, 1))
    if sample_count > 0:
        sample_mean = np.mean(samples, axis=-1).astype(np.float32)
        sample_max = np.max(samples, axis=-1).astype(np.float32)
        sample_positive_mean = (
            np.sum(np.where(sample_active, samples, 0.0), axis=-1) / np.maximum(sample_active_count, 1.0)
        ).astype(np.float32)
    else:
        sample_mean = np.zeros(samples.shape[:-1], dtype=np.float32)
        sample_max = np.zeros(samples.shape[:-1], dtype=np.float32)
        sample_positive_mean = np.zeros(samples.shape[:-1], dtype=np.float32)

    groups = {
        "true_positive": _SampleBoundaryAccumulator(),
        "false_positive": _SampleBoundaryAccumulator(),
        "false_negative": _SampleBoundaryAccumulator(),
    }
    rows: list[dict[str, Any]] = []
    pressure_counts = np.zeros((steps, sensors), dtype=np.int64)
    reference_counts = np.zeros((steps, sensors), dtype=np.int64)
    for step_idx in range(steps):
        for sensor_idx in range(sensors):
            pred_mask = pressure[step_idx, sensor_idx] > float(active_threshold)
            ref_values = _resize_values_to_shape(reference[step_idx, sensor_idx], target_shape)
            ref_mask = ref_values > float(reference_threshold)
            pred_values = _resize_values_to_shape(pred_depth[step_idx, sensor_idx], target_shape)
            true_positive = pred_mask & ref_mask
            false_positive = pred_mask & ~ref_mask
            false_negative = ~pred_mask & ref_mask
            union = pred_mask | ref_mask
            pred_active = int(np.count_nonzero(pred_mask))
            ref_active = int(np.count_nonzero(ref_mask))
            pressure_counts[step_idx, sensor_idx] = pred_active
            reference_counts[step_idx, sensor_idx] = ref_active

            frame_groups = {
                "true_positive": _sample_boundary_group_stats(
                    true_positive,
                    sample_support[step_idx, sensor_idx],
                    sample_active_count[step_idx, sensor_idx],
                    sample_mean[step_idx, sensor_idx],
                    sample_positive_mean[step_idx, sensor_idx],
                    sample_max[step_idx, sensor_idx],
                    pred_values,
                    ref_values,
                ),
                "false_positive": _sample_boundary_group_stats(
                    false_positive,
                    sample_support[step_idx, sensor_idx],
                    sample_active_count[step_idx, sensor_idx],
                    sample_mean[step_idx, sensor_idx],
                    sample_positive_mean[step_idx, sensor_idx],
                    sample_max[step_idx, sensor_idx],
                    pred_values,
                    ref_values,
                ),
                "false_negative": _sample_boundary_group_stats(
                    false_negative,
                    sample_support[step_idx, sensor_idx],
                    sample_active_count[step_idx, sensor_idx],
                    sample_mean[step_idx, sensor_idx],
                    sample_positive_mean[step_idx, sensor_idx],
                    sample_max[step_idx, sensor_idx],
                    pred_values,
                    ref_values,
                ),
            }
            for group_name, stats in frame_groups.items():
                groups[group_name].add(stats)

            iou = None
            if np.any(union):
                iou = float(np.count_nonzero(true_positive) / max(1, int(np.count_nonzero(union))))
            row = {
                "step": int(step_idx),
                "sensor": int(sensor_idx),
                "pred_active_taxels": pred_active,
                "reference_active_taxels": ref_active,
                "true_positive_taxels": int(np.count_nonzero(true_positive)),
                "false_positive_taxels": int(np.count_nonzero(false_positive)),
                "false_negative_taxels": int(np.count_nonzero(false_negative)),
                "iou": iou,
                "classification": _reference_frame_classification(
                    pred_active,
                    ref_active,
                    int(np.count_nonzero(false_positive)),
                    int(np.count_nonzero(false_negative)),
                ),
            }
            for group_name, stats in frame_groups.items():
                row.update({f"{group_name}_{key}": value for key, value in stats.items()})
            rows.append(row)

    summary = {
        "available": True,
        "pressure_key": str(pressure_key),
        "penetration_key": str(penetration_key),
        "reference_key": str(reference_key),
        "reference_layer": inferred_reference_layer,
        "sample_penetrations_key": str(sample_penetrations_key),
        "sample_count": sample_count,
        "num_steps": int(steps),
        "sensor_count": int(sensors),
        "image_shape": [int(v) for v in target_shape],
        "active_threshold": float(active_threshold),
        "reference_threshold": float(reference_threshold),
        "pressure_onset_step_by_sensor": _onset_steps(pressure_counts),
        "reference_onset_step_by_sensor": _onset_steps(reference_counts),
        "pressure_offset_step_by_sensor": _offset_steps(pressure_counts),
        "reference_offset_step_by_sensor": _offset_steps(reference_counts),
        "groups": {name: accumulator.summary() for name, accumulator in groups.items()},
        "worst_iou_frames": _top_frame_rows(rows, key="iou", top_k=top_k, reverse=False),
        "largest_false_positive_frames": _top_frame_rows(rows, key="false_positive_taxels", top_k=top_k, reverse=True),
        "largest_false_negative_frames": _top_frame_rows(rows, key="false_negative_taxels", top_k=top_k, reverse=True),
    }
    return {"summary": summary, "frames": rows}


def evaluate_pressure_trace_report(
    report: Mapping[str, Any],
    *,
    precontact_leakage_threshold: float = 0.001,
    reference_iou_threshold: float = 0.95,
    reference_alignment_valid_fraction_threshold: float | None = None,
    reference_contact_valid_fraction_threshold: float | None = 1.0,
    reference_invalid_contact_fraction_threshold: float | None = 0.0,
    centroid_error_threshold_px: float = 1.5,
    onset_error_threshold_frames: int = 1,
    offset_error_threshold_frames: int = 1,
    bbox_error_threshold_px: float = 2.0,
    depth_rmse_threshold_m: float = 2.0e-5,
    force_depth_spearman_threshold: float = 0.95,
    dense_reference_layers: tuple[str, ...] | list[str] | set[str] | None = DENSE_REFERENCE_ACCEPTANCE_LAYERS,
    allow_reference_layer_override: bool = False,
    unsafe_reference_override_layers: tuple[str, ...] | list[str] | set[str] = UNSAFE_REFERENCE_OVERRIDE_LAYERS,
) -> dict[str, Any]:
    """Evaluate a report against acceptance thresholds."""

    checks: list[dict[str, Any]] = []

    def add(name: str, value: Any, threshold: Any, passed: bool) -> None:
        checks.append({"name": name, "value": value, "threshold": threshold, "passed": bool(passed)})

    if "precontact_leakage_fraction" in report:
        value = float(report["precontact_leakage_fraction"])
        add("precontact_leakage_fraction", value, precontact_leakage_threshold, value <= precontact_leakage_threshold)

    spearman = report.get("force_depth_spearman_min")
    if spearman is not None:
        value = float(spearman)
        add("force_depth_spearman_min", value, force_depth_spearman_threshold, value >= force_depth_spearman_threshold)

    ref = report.get("reference")
    if isinstance(ref, Mapping) and not ref.get("available", True):
        add(
            "reference.available",
            ref.get("reason"),
            "reference must be a dense (T,S,H,W) compatible map",
            False,
        )
    if isinstance(ref, Mapping) and ref.get("available"):
        layer = ref.get("reference_layer") or report.get("reference_layer")
        inferred_layer = ref.get("inferred_reference_layer") or report.get("inferred_reference_layer")
        requested_layer = ref.get("requested_reference_layer") or report.get("requested_reference_layer")
        override_conflict = bool(
            ref.get("reference_layer_override_conflict", report.get("reference_layer_override_conflict", False))
        )
        unsafe_layers = tuple(str(value) for value in unsafe_reference_override_layers)
        if override_conflict and inferred_layer is not None and str(inferred_layer) in unsafe_layers:
            add(
                "reference.layer_override_is_safe",
                {
                    "inferred": str(inferred_layer),
                    "requested": None if requested_layer is None else str(requested_layer),
                    "resolved": None if layer is None else str(layer),
                },
                "no manual reference-layer override without explicit allow",
                bool(allow_reference_layer_override),
            )
        if dense_reference_layers is not None:
            allowed_layers = tuple(str(value) for value in dense_reference_layers)
            add(
                "reference.layer_allows_dense_acceptance",
                None if layer is None else str(layer),
                allowed_layers,
                layer is not None and str(layer) in allowed_layers,
            )
        iou = ref.get("active_mask_iou_min")
        if iou is not None:
            add("reference.active_mask_iou_min", float(iou), reference_iou_threshold, float(iou) >= reference_iou_threshold)
        alignment_valid = ref.get("alignment_valid_taxel_fraction_min")
        if alignment_valid is not None and reference_alignment_valid_fraction_threshold is not None:
            add(
                "reference.alignment_valid_taxel_fraction_min",
                float(alignment_valid),
                float(reference_alignment_valid_fraction_threshold),
                float(alignment_valid) >= float(reference_alignment_valid_fraction_threshold),
            )
        contact_valid = ref.get("alignment_contact_region_valid_fraction_min")
        if contact_valid is not None and reference_contact_valid_fraction_threshold is not None:
            add(
                "reference.alignment_contact_region_valid_fraction_min",
                float(contact_valid),
                float(reference_contact_valid_fraction_threshold),
                float(contact_valid) >= float(reference_contact_valid_fraction_threshold),
            )
        invalid_contact = ref.get("alignment_invalid_contact_fraction_max")
        if invalid_contact is not None and reference_invalid_contact_fraction_threshold is not None:
            add(
                "reference.alignment_invalid_contact_fraction_max",
                float(invalid_contact),
                float(reference_invalid_contact_fraction_threshold),
                float(invalid_contact) <= float(reference_invalid_contact_fraction_threshold),
            )
        centroid = ref.get("centroid_error_px_max")
        if centroid is not None:
            add(
                "reference.centroid_error_px_max",
                float(centroid),
                centroid_error_threshold_px,
                float(centroid) <= centroid_error_threshold_px,
            )
        bbox = ref.get("bbox_error_px_max")
        if bbox is not None:
            add("reference.bbox_error_px_max", float(bbox), bbox_error_threshold_px, float(bbox) <= bbox_error_threshold_px)
        depth_rmse = ref.get("depth_rmse_m_max")
        if depth_rmse is not None:
            add("reference.depth_rmse_m_max", float(depth_rmse), depth_rmse_threshold_m, float(depth_rmse) <= depth_rmse_threshold_m)
        onset_errors = [value for value in ref.get("onset_error_frames_by_sensor", []) if value is not None]
        if onset_errors:
            worst_onset = max(int(value) for value in onset_errors)
            add("reference.onset_error_frames_max", worst_onset, onset_error_threshold_frames, worst_onset <= onset_error_threshold_frames)
        offset_errors = [value for value in ref.get("offset_error_frames_by_sensor", []) if value is not None]
        if offset_errors:
            worst_offset = max(int(value) for value in offset_errors)
            add("reference.offset_error_frames_max", worst_offset, offset_error_threshold_frames, worst_offset <= offset_error_threshold_frames)

    return {"passed": all(check["passed"] for check in checks), "checks": checks}


_SAMPLE_BOUNDARY_METRICS = (
    "sample_support_fraction",
    "sample_active_count",
    "sample_mean_penetration_m",
    "sample_positive_mean_penetration_m",
    "sample_max_penetration_m",
    "predicted_depth_m",
    "reference_depth_m",
)


class _SampleBoundaryAccumulator:
    def __init__(self) -> None:
        self.taxels = 0
        self._sums = {name: 0.0 for name in _SAMPLE_BOUNDARY_METRICS}
        self._mins = {name: float("inf") for name in _SAMPLE_BOUNDARY_METRICS}
        self._maxs = {name: float("-inf") for name in _SAMPLE_BOUNDARY_METRICS}

    def add(self, stats: Mapping[str, Any]) -> None:
        count = int(stats.get("taxels") or 0)
        if count <= 0:
            return
        self.taxels += count
        for name in _SAMPLE_BOUNDARY_METRICS:
            total = stats.get(f"{name}_sum")
            minimum = stats.get(f"{name}_min")
            maximum = stats.get(f"{name}_max")
            if total is None or minimum is None or maximum is None:
                continue
            self._sums[name] += float(total)
            self._mins[name] = min(self._mins[name], float(minimum))
            self._maxs[name] = max(self._maxs[name], float(maximum))

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"taxels": int(self.taxels)}
        for name in _SAMPLE_BOUNDARY_METRICS:
            if self.taxels <= 0:
                out[f"{name}_mean"] = None
                out[f"{name}_min"] = None
                out[f"{name}_max"] = None
                out[f"{name}_sum"] = None
            else:
                out[f"{name}_mean"] = float(self._sums[name] / float(self.taxels))
                out[f"{name}_min"] = float(self._mins[name])
                out[f"{name}_max"] = float(self._maxs[name])
                out[f"{name}_sum"] = float(self._sums[name])
        return out


def _sample_boundary_group_stats(
    mask: np.ndarray,
    support_fraction: np.ndarray,
    active_count: np.ndarray,
    mean_penetration: np.ndarray,
    positive_mean_penetration: np.ndarray,
    max_penetration: np.ndarray,
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(mask))
    out: dict[str, Any] = {"taxels": count}
    values = {
        "sample_support_fraction": support_fraction,
        "sample_active_count": active_count,
        "sample_mean_penetration_m": mean_penetration,
        "sample_positive_mean_penetration_m": positive_mean_penetration,
        "sample_max_penetration_m": max_penetration,
        "predicted_depth_m": predicted_depth,
        "reference_depth_m": reference_depth,
    }
    for name, arr in values.items():
        selected = np.asarray(arr, dtype=np.float64)[mask]
        if selected.size <= 0:
            out[f"{name}_mean"] = None
            out[f"{name}_min"] = None
            out[f"{name}_max"] = None
            out[f"{name}_sum"] = None
        else:
            out[f"{name}_mean"] = float(np.mean(selected))
            out[f"{name}_min"] = float(np.min(selected))
            out[f"{name}_max"] = float(np.max(selected))
            out[f"{name}_sum"] = float(np.sum(selected))
    return out


def _as_tshw(value: Any, *, name: str) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 2:
        return arr[None, None, :, :]
    raise ValueError(f"{name} must have shape (T,S,H,W), (T,H,W), or (H,W); got {arr.shape}")


def _has_reference_origin_alignment_inputs(data: Mapping[str, Any]) -> bool:
    return all(
        key in data
        for key in (
            "pressure_taxel_points_l_m",
            "pressure_taxel_normals_l",
            "tacmap_grid_points_l_m",
            "normal_ray_alignment_source_index",
            "normal_ray_alignment_valid_mask",
        )
    )


def _as_sensor_points(value: Any, *, name: str, sensor_count_hint: int | None = None) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[-1:] != (3,):
        raise ValueError(f"{name} must end in xyz dimension 3; got {arr.shape}")
    if arr.ndim == 4:
        return arr.reshape(int(arr.shape[0]), -1, 3).astype(np.float32)
    if arr.ndim == 3:
        if sensor_count_hint is not None and int(arr.shape[0]) == int(sensor_count_hint):
            return arr.reshape(int(arr.shape[0]), -1, 3).astype(np.float32)
        return arr.reshape(1, -1, 3).astype(np.float32)
    if arr.ndim == 2:
        return arr.reshape(1, -1, 3).astype(np.float32)
    raise ValueError(f"{name} must have shape (S,H,W,3), (S,N,3), (H,W,3), or (N,3); got {arr.shape}")


def _as_sensor_index(value: Any, *, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(value)
    if arr.ndim == 3:
        return arr.reshape(int(arr.shape[0]), -1), tuple(int(v) for v in arr.shape)
    if arr.ndim == 2:
        return arr.reshape(int(arr.shape[0]), -1), tuple(int(v) for v in arr.shape)
    if arr.ndim == 1:
        return arr.reshape(1, -1), tuple(int(v) for v in arr.shape)
    raise ValueError(f"{name} must have shape (S,H,W), (S,N), or (N,); got {arr.shape}")


def _as_sensor_bool_mask(value: Any, *, name: str, source_index: Any) -> np.ndarray:
    source_arr = np.asarray(source_index)
    source_flat, _ = _as_sensor_index(source_arr, name="source_index")
    arr = np.asarray(value, dtype=bool)
    if arr.shape == source_arr.shape:
        return arr.reshape(source_flat.shape)
    if arr.ndim == 2 and source_arr.ndim == 3 and source_arr.shape[0] == 1 and arr.shape == source_arr.shape[1:]:
        return arr.reshape(1, -1)
    if arr.ndim == 1 and source_flat.shape[0] == 1 and arr.size == source_flat.shape[1]:
        return arr.reshape(1, -1)
    if arr.shape == source_flat.shape:
        return arr.reshape(source_flat.shape)
    raise ValueError(f"{name} shape {arr.shape} does not match {source_arr.shape} or {source_flat.shape}")


def _normalize_vectors(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1.0e-12)


def _metric_stats(prefix: str, values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def _precontact_leakage_metrics(
    active: np.ndarray,
    active_counts: np.ndarray,
    penetration: np.ndarray,
    *,
    penetration_threshold: float,
) -> dict[str, Any]:
    inactive_penetration = penetration <= float(penetration_threshold)
    leaked = active & inactive_penetration
    denom = int(np.count_nonzero(inactive_penetration))
    leakage_fraction = float(np.count_nonzero(leaked) / denom) if denom > 0 else 0.0
    precontact_step_mask = np.max(penetration, axis=(-2, -1)) <= float(penetration_threshold)
    precontact_steps = int(np.count_nonzero(precontact_step_mask))
    precontact_hit_steps = int(np.count_nonzero((active_counts > 0) & precontact_step_mask))
    return {
        "penetration_threshold_m": float(penetration_threshold),
        "precontact_leakage_fraction": leakage_fraction,
        "precontact_steps": precontact_steps,
        "precontact_active_steps": precontact_hit_steps,
    }


def _total_force(data: Mapping[str, Any], *, raw_pressure_key: str) -> np.ndarray | None:
    if raw_pressure_key in data:
        raw = _as_tshw(data[raw_pressure_key], name=raw_pressure_key)
        return raw.sum(axis=(-2, -1))
    if "total_force_n" in data:
        arr = np.asarray(data["total_force_n"], dtype=np.float32)
        if arr.ndim == 2:
            return arr
    return None


def _force_depth_spearman(total_force: np.ndarray, penetration: np.ndarray) -> list[float | None]:
    volume = penetration.sum(axis=(-2, -1))
    steps = min(int(total_force.shape[0]), int(volume.shape[0]))
    sensors = min(int(total_force.shape[1]), int(volume.shape[1]))
    out: list[float | None] = []
    for sensor_idx in range(sensors):
        out.append(_spearman(total_force[:steps, sensor_idx], volume[:steps, sensor_idx]))
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr) if np.isfinite(corr) else None


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _onset_steps(counts: np.ndarray) -> list[int | None]:
    return [_first_positive(counts[:, sensor_idx]) for sensor_idx in range(counts.shape[1])]


def _offset_steps(counts: np.ndarray) -> list[int | None]:
    out: list[int | None] = []
    for sensor_idx in range(counts.shape[1]):
        active = np.flatnonzero(counts[:, sensor_idx] > 0)
        out.append(int(active[-1]) if active.size else None)
    return out


def _first_positive(values: np.ndarray) -> int | None:
    active = np.flatnonzero(np.asarray(values) > 0)
    return int(active[0]) if active.size else None


def _resize_values_to_shape(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if tuple(values.shape) == tuple(target_shape):
        return values
    src_h, src_w = values.shape
    dst_h, dst_w = target_shape
    if src_h % dst_h == 0 and src_w % dst_w == 0:
        block_h = src_h // dst_h
        block_w = src_w // dst_w
        return values.reshape(dst_h, block_h, dst_w, block_w).mean(axis=(1, 3))
    row_idx = np.clip(np.round((np.arange(dst_h) + 0.5) * src_h / dst_h - 0.5).astype(int), 0, src_h - 1)
    col_idx = np.clip(np.round((np.arange(dst_w) + 0.5) * src_w / dst_w - 0.5).astype(int), 0, src_w - 1)
    return values[np.ix_(row_idx, col_idx)]


def _last_positive(values: np.ndarray) -> int | None:
    active = np.flatnonzero(np.asarray(values) > 0)
    return int(active[-1]) if active.size else None


def _pressure_trace_metadata(data: Mapping[str, Any], trace_path: Path | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if "metadata_json" in data:
        raw = data["metadata_json"]
        try:
            if isinstance(raw, np.ndarray):
                raw = raw.item() if raw.shape == () else raw.tolist()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if isinstance(raw, dict):
                metadata.update(raw)
            elif isinstance(raw, str):
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    metadata.update(parsed)
                else:
                    metadata["metadata_json_not_object"] = True
            else:
                metadata["metadata_json_not_object"] = True
        except Exception:
            metadata["metadata_json_parse_error"] = True
    if trace_path is not None:
        metadata_path = trace_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            try:
                parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    metadata.update(parsed)
                else:
                    metadata["metadata_file_not_object"] = True
            except Exception:
                metadata["metadata_file_parse_error"] = True
    return metadata


def _pressure_trace_validation_result(
    errors: list[str],
    warnings: list[str],
    metadata: Mapping[str, Any],
    shape_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "pressure_trace_v1",
        "passed": not errors,
        "errors": list(errors),
        "warnings": list(warnings),
        "metadata": {
            key: metadata.get(key)
            for key in (
                "pressure_trace_schema_version",
                "pressure_backend_id",
                "pressure_layout_id",
                "pressure_calibration_id",
                "reference_layer",
                "metadata_json_parse_error",
                "metadata_json_not_object",
                "metadata_file_parse_error",
                "metadata_file_not_object",
            )
            if key in metadata
        },
    }
    if shape_summary is not None:
        result.update(shape_summary)
    return result


def _resolve_reference_valid_mask_key(
    reference_key: str,
    data: Mapping[str, Any],
    *,
    requested_key: str | None,
) -> str:
    if requested_key is not None:
        return str(requested_key)
    key = str(reference_key)
    candidates = [f"{key}_valid_mask"]
    if key.startswith("normal_ray_"):
        candidates.append("normal_ray_alignment_valid_mask")
    for candidate in candidates:
        if candidate in data:
            return candidate
    return candidates[0]


def _reference_frame_classification(
    pred_active: int,
    ref_active: int,
    false_positive_count: int,
    false_negative_count: int,
) -> str:
    if pred_active <= 0 and ref_active <= 0:
        return "both_empty"
    if pred_active > 0 and ref_active <= 0:
        return "false_positive_only"
    if pred_active <= 0 and ref_active > 0:
        return "false_negative_only"
    if false_positive_count <= 0 and false_negative_count <= 0:
        return "perfect_overlap"
    if false_positive_count > 0 and false_negative_count > 0:
        return "mixed_boundary_error"
    if false_positive_count > 0:
        return "over_prediction"
    return "under_prediction"


def _top_frame_rows(rows: list[dict[str, Any]], *, key: str, top_k: int, reverse: bool) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get(key) is not None]
    if key in {"false_positive_taxels", "false_negative_taxels"}:
        candidates = [row for row in candidates if float(row[key]) > 0.0]
    if not candidates:
        return []
    limit = max(0, int(top_k))
    if limit <= 0:
        return []
    return [
        dict(row)
        for row in sorted(
            candidates,
            key=lambda row: (float(row[key]), int(row["step"]), int(row["sensor"])),
            reverse=bool(reverse),
        )[:limit]
    ]


def _resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if tuple(mask.shape) == tuple(target_shape):
        return mask
    src_h, src_w = mask.shape
    dst_h, dst_w = target_shape
    if src_h % dst_h == 0 and src_w % dst_w == 0:
        block_h = src_h // dst_h
        block_w = src_w // dst_w
        return mask.reshape(dst_h, block_h, dst_w, block_w).any(axis=(1, 3))
    row_idx = np.clip(np.round((np.arange(dst_h) + 0.5) * src_h / dst_h - 0.5).astype(int), 0, src_h - 1)
    col_idx = np.clip(np.round((np.arange(dst_w) + 0.5) * src_w / dst_w - 0.5).astype(int), 0, src_w - 1)
    return mask[np.ix_(row_idx, col_idx)]


def _reference_valid_mask_step(
    valid_mask: np.ndarray,
    *,
    step_idx: int,
    sensor_idx: int,
    target_shape: tuple[int, int],
) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim == 4:
        if step_idx >= mask.shape[0] or sensor_idx >= mask.shape[1]:
            return np.zeros(target_shape, dtype=bool)
        return _resize_mask_to_shape(mask[step_idx, sensor_idx], target_shape)
    if mask.ndim == 3:
        if sensor_idx >= mask.shape[0]:
            return np.zeros(target_shape, dtype=bool)
        return _resize_mask_to_shape(mask[sensor_idx], target_shape)
    if mask.ndim == 2:
        return _resize_mask_to_shape(mask, target_shape)
    raise ValueError(f"reference_valid_mask must have shape (S,H,W), (T,S,H,W), or (H,W); got {mask.shape}")


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return np.asarray([float(np.mean(rows)), float(np.mean(cols))], dtype=np.float64)


def _mask_bbox(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return np.asarray([rows.min(), cols.min(), rows.max(), cols.max()], dtype=np.float64)


def _bbox_error_px(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))
