from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .pressure_trace_metrics import (
    evaluate_pressure_trace_report,
    load_pressure_trace_npz,
    pressure_trace_report,
)


@dataclass(frozen=True)
class PressureDeadbandFitConfig:
    """Configuration for fitting SDF penetration deadband against an L1 model reference."""

    pressure_key: str = "pressure_norm"
    raw_pressure_key: str = "pressure_raw_n"
    penetration_key: str = "penetration_m"
    reference_key: str = "tacmap_raw_m"
    reference_layer: str | None = None
    reference_valid_mask_key: str | None = None
    active_threshold: float = 1.0e-6
    reference_threshold: float = 0.0
    penetration_threshold: float = 0.0
    candidate_count: int = 64
    max_deadband_m: float | None = None
    reference_iou_threshold: float = 0.95
    centroid_error_threshold_px: float = 1.5
    bbox_error_threshold_px: float = 2.0
    depth_rmse_threshold_m: float = 2.0e-5
    onset_error_threshold_frames: int = 1
    offset_error_threshold_frames: int = 1


@dataclass(frozen=True)
class PressureMaskDepthFitConfig:
    """Configuration for fitting active mask and depth calibration separately."""

    pressure_key: str = "pressure_norm"
    raw_pressure_key: str = "pressure_raw_n"
    penetration_key: str = "penetration_m"
    reference_key: str = "tacmap_raw_m"
    reference_layer: str | None = None
    reference_valid_mask_key: str | None = None
    active_threshold: float = 1.0e-6
    reference_threshold: float = 0.0
    penetration_threshold: float = 0.0
    candidate_count: int = 32
    max_mask_deadband_m: float | None = None
    max_depth_deadband_m: float | None = None
    reference_iou_threshold: float = 0.95
    centroid_error_threshold_px: float = 1.5
    bbox_error_threshold_px: float = 2.0
    depth_rmse_threshold_m: float = 2.0e-5
    onset_error_threshold_frames: int = 1
    offset_error_threshold_frames: int = 1


@dataclass(frozen=True)
class PressureSampleSupportFitConfig:
    """Configuration for fitting a finite-area support/depth law."""

    pressure_key: str = "pressure_norm"
    raw_pressure_key: str = "pressure_raw_n"
    penetration_key: str = "penetration_m"
    reference_key: str = "tacmap_raw_m"
    support_key: str = "geometry_normal_ray_sample_support_fraction"
    default_depth_source_key: str = "geometry_normal_ray_sample_mean_penetration_m"
    reference_layer: str | None = None
    reference_valid_mask_key: str | None = None
    active_threshold: float = 1.0e-6
    reference_threshold: float = 0.0
    penetration_threshold: float = 0.0
    candidate_count: int = 32
    max_mask_deadband_m: float | None = None
    max_depth_deadband_m: float | None = None
    reference_iou_threshold: float = 0.95
    centroid_error_threshold_px: float = 1.5
    bbox_error_threshold_px: float = 2.0
    depth_rmse_threshold_m: float = 2.0e-5
    onset_error_threshold_frames: int = 1
    offset_error_threshold_frames: int = 1


@dataclass(frozen=True)
class PressureSpatialFootprintFitConfig:
    """Configuration for fitting a spatial footprint law in taxel coordinates."""

    pressure_key: str = "pressure_norm"
    raw_pressure_key: str = "pressure_raw_n"
    penetration_key: str = "penetration_m"
    reference_key: str = "tacmap_raw_m"
    support_key: str = "geometry_normal_ray_sample_support_fraction"
    default_depth_source_key: str = "geometry_normal_ray_sample_mean_penetration_m"
    reference_layer: str | None = None
    reference_valid_mask_key: str | None = None
    active_threshold: float = 1.0e-6
    reference_threshold: float = 0.0
    penetration_threshold: float = 0.0
    candidate_count: int = 32
    max_mask_deadband_m: float | None = None
    max_depth_deadband_m: float | None = None
    connectivity: int = 8
    reference_iou_threshold: float = 0.95
    centroid_error_threshold_px: float = 1.5
    bbox_error_threshold_px: float = 2.0
    depth_rmse_threshold_m: float = 2.0e-5
    onset_error_threshold_frames: int = 1
    offset_error_threshold_frames: int = 1


@dataclass(frozen=True)
class PressureAreaFractionFitConfig:
    """Configuration for fitting sampled area-fraction/depth integration."""

    pressure_key: str = "pressure_norm"
    raw_pressure_key: str = "pressure_raw_n"
    penetration_key: str = "penetration_m"
    reference_key: str = "tacmap_raw_m"
    support_key: str = "geometry_normal_ray_sample_support_fraction"
    mean_depth_key: str = "geometry_normal_ray_sample_mean_penetration_m"
    positive_mean_depth_key: str = "geometry_normal_ray_sample_positive_mean_penetration_m"
    max_depth_key: str = "geometry_normal_ray_sample_max_penetration_m"
    center_depth_key: str = "geometry_normal_ray_penetration_m"
    reference_layer: str | None = None
    reference_valid_mask_key: str | None = None
    active_threshold: float = 1.0e-6
    reference_threshold: float = 0.0
    penetration_threshold: float = 0.0
    candidate_count: int = 32
    max_mask_deadband_m: float | None = None
    max_depth_deadband_m: float | None = None
    reference_iou_threshold: float = 0.95
    centroid_error_threshold_px: float = 1.5
    bbox_error_threshold_px: float = 2.0
    depth_rmse_threshold_m: float = 2.0e-5
    onset_error_threshold_frames: int = 1
    offset_error_threshold_frames: int = 1


def fit_pressure_deadband_to_reference(
    trace: Mapping[str, Any] | str | Path,
    *,
    config: PressureDeadbandFitConfig | None = None,
    candidates_m: Iterable[float] | None = None,
    scales: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Search a penetration deadband that best aligns pressure maps to a reference.

    This is an L1 calibration helper. It does not mutate a simulation trace; it
    evaluates candidate corrected penetrations ``max(penetration - deadband, 0)``
    against an L1 model reference such as ``tacmap_raw_m`` and returns
    JSON-friendly metrics plus the suggested ``--penetration-deadband`` value.
    """

    cfg = config or PressureDeadbandFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if cfg.penetration_key not in data:
        raise KeyError(f"penetration key {cfg.penetration_key!r} not found")
    if cfg.reference_key not in data:
        raise KeyError(f"reference key {cfg.reference_key!r} not found")

    penetration = _as_tshw(data[cfg.penetration_key], name=cfg.penetration_key)
    reference = _as_dense_reference_tshw(data[cfg.reference_key], name=cfg.reference_key)
    candidate_values = _candidate_deadbands(
        penetration,
        candidate_count=int(cfg.candidate_count),
        max_deadband_m=cfg.max_deadband_m,
        candidates_m=candidates_m,
    )
    scale_values = _candidate_scales(scales)

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for scale in scale_values:
        for deadband_m in candidate_values:
            candidate_trace = apply_pressure_deadband_to_trace(
                data,
                deadband_m=float(deadband_m),
                scale=float(scale),
                pressure_key=cfg.pressure_key,
                raw_pressure_key=cfg.raw_pressure_key,
                penetration_key=cfg.penetration_key,
                output_pressure_key=cfg.pressure_key,
                output_raw_pressure_key=cfg.raw_pressure_key,
                output_penetration_key=cfg.penetration_key,
            )
            report = pressure_trace_report(
                candidate_trace,
                pressure_key=cfg.pressure_key,
                raw_pressure_key=cfg.raw_pressure_key,
                penetration_key=cfg.penetration_key,
                reference_key=cfg.reference_key,
                reference_layer=cfg.reference_layer,
                reference_valid_mask_key=cfg.reference_valid_mask_key,
                active_threshold=float(cfg.active_threshold),
                penetration_threshold=float(cfg.penetration_threshold),
                reference_threshold=float(cfg.reference_threshold),
            )
            evaluation = evaluate_pressure_trace_report(
                report,
                reference_iou_threshold=float(cfg.reference_iou_threshold),
                centroid_error_threshold_px=float(cfg.centroid_error_threshold_px),
                bbox_error_threshold_px=float(cfg.bbox_error_threshold_px),
                depth_rmse_threshold_m=float(cfg.depth_rmse_threshold_m),
                onset_error_threshold_frames=int(cfg.onset_error_threshold_frames),
                offset_error_threshold_frames=int(cfg.offset_error_threshold_frames),
            )
            entry = _candidate_summary(float(deadband_m), report, evaluation, scale=float(scale))
            candidates.append(entry)
            if best is None or (entry["score"], entry["scale"], entry["deadband_m"]) < (
                best["score"],
                best["scale"],
                best["deadband_m"],
            ):
                best = entry

    assert best is not None
    return {
        "best": best,
        "suggested_args": ["--penetration-deadband", f"{float(best['deadband_m']):.9g}"],
        "suggested_depth_scale": f"{float(best['scale']):.9g}",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "config": _config_dict(cfg),
        "scale_candidates": [float(v) for v in scale_values],
        "reference_shape": [int(v) for v in reference.shape],
        "penetration_shape": [int(v) for v in penetration.shape],
    }


def fit_pressure_mask_depth_to_reference(
    trace: Mapping[str, Any] | str | Path,
    *,
    config: PressureMaskDepthFitConfig | None = None,
    mask_deadbands_m: Iterable[float] | None = None,
    depth_deadbands_m: Iterable[float] | None = None,
    scales: Iterable[float] | None = None,
    active_floor_m: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Search a split active-mask/depth calibration against a dense reference.

    ``mask_deadband_m`` gates whether a taxel is active using the unscaled source
    penetration. ``depth_deadband_m`` and ``scale`` then calibrate the reported
    depth. This keeps support calibration from automatically erasing weak edge
    taxels during depth fitting.
    """

    cfg = config or PressureMaskDepthFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if cfg.penetration_key not in data:
        raise KeyError(f"penetration key {cfg.penetration_key!r} not found")
    if cfg.reference_key not in data:
        raise KeyError(f"reference key {cfg.reference_key!r} not found")

    penetration = _as_tshw(data[cfg.penetration_key], name=cfg.penetration_key)
    reference = _as_dense_reference_tshw(data[cfg.reference_key], name=cfg.reference_key)
    mask_values = _explicit_or_candidate_deadbands(
        penetration,
        candidate_count=int(cfg.candidate_count),
        max_deadband_m=cfg.max_mask_deadband_m,
        candidates_m=mask_deadbands_m,
    )
    depth_values = _explicit_or_candidate_deadbands(
        penetration,
        candidate_count=int(cfg.candidate_count),
        max_deadband_m=cfg.max_depth_deadband_m,
        candidates_m=depth_deadbands_m,
    )
    scale_values = _candidate_scales(scales)
    floor_values = _candidate_active_floors(active_floor_m)

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for scale in scale_values:
        for mask_deadband_m in mask_values:
            for depth_deadband_m in depth_values:
                for floor_m in floor_values:
                    candidate_trace = apply_pressure_mask_depth_to_trace(
                        data,
                        mask_deadband_m=float(mask_deadband_m),
                        depth_deadband_m=float(depth_deadband_m),
                        scale=float(scale),
                        active_floor_m=float(floor_m),
                        pressure_key=cfg.pressure_key,
                        raw_pressure_key=cfg.raw_pressure_key,
                        penetration_key=cfg.penetration_key,
                        output_pressure_key=cfg.pressure_key,
                        output_raw_pressure_key=cfg.raw_pressure_key,
                        output_penetration_key=cfg.penetration_key,
                    )
                    report = pressure_trace_report(
                        candidate_trace,
                        pressure_key=cfg.pressure_key,
                        raw_pressure_key=cfg.raw_pressure_key,
                        penetration_key=cfg.penetration_key,
                        reference_key=cfg.reference_key,
                        reference_layer=cfg.reference_layer,
                        reference_valid_mask_key=cfg.reference_valid_mask_key,
                        active_threshold=float(cfg.active_threshold),
                        penetration_threshold=float(cfg.penetration_threshold),
                        reference_threshold=float(cfg.reference_threshold),
                    )
                    evaluation = evaluate_pressure_trace_report(
                        report,
                        reference_iou_threshold=float(cfg.reference_iou_threshold),
                        centroid_error_threshold_px=float(cfg.centroid_error_threshold_px),
                        bbox_error_threshold_px=float(cfg.bbox_error_threshold_px),
                        depth_rmse_threshold_m=float(cfg.depth_rmse_threshold_m),
                        onset_error_threshold_frames=int(cfg.onset_error_threshold_frames),
                        offset_error_threshold_frames=int(cfg.offset_error_threshold_frames),
                    )
                    entry = _mask_depth_candidate_summary(
                        float(mask_deadband_m),
                        float(depth_deadband_m),
                        float(floor_m),
                        report,
                        evaluation,
                        scale=float(scale),
                    )
                    candidates.append(entry)
                    if best is None or (
                        entry["score"],
                        entry["scale"],
                        entry["mask_deadband_m"],
                        entry["depth_deadband_m"],
                        entry["active_floor_m"],
                    ) < (
                        best["score"],
                        best["scale"],
                        best["mask_deadband_m"],
                        best["depth_deadband_m"],
                        best["active_floor_m"],
                    ):
                        best = entry

    assert best is not None
    return {
        "best": best,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "config": _config_dict(cfg),
        "mask_deadband_candidates": [float(v) for v in mask_values],
        "depth_deadband_candidates": [float(v) for v in depth_values],
        "scale_candidates": [float(v) for v in scale_values],
        "active_floor_candidates": [float(v) for v in floor_values],
        "reference_shape": [int(v) for v in reference.shape],
        "penetration_shape": [int(v) for v in penetration.shape],
    }


def fit_pressure_sample_support_to_reference(
    trace: Mapping[str, Any] | str | Path,
    *,
    config: PressureSampleSupportFitConfig | None = None,
    support_thresholds: Iterable[float] | None = None,
    mask_deadbands_m: Iterable[float] | None = None,
    depth_deadbands_m: Iterable[float] | None = None,
    scales: Iterable[float] | None = None,
    support_powers: Iterable[float] | None = None,
    depth_source_keys: Iterable[str] | None = None,
    active_floor_m: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Search a finite-area taxel support/depth law against a dense reference.

    This fitter consumes the raw ``geometry_normal_ray_sample_*`` diagnostics.
    It is meant for L1 model-reference calibration: support controls whether a
    taxel is active, while the selected depth source and support exponent control
    the reported indentation magnitude.
    """

    cfg = config or PressureSampleSupportFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if cfg.support_key not in data:
        raise KeyError(f"support key {cfg.support_key!r} not found")
    if cfg.reference_key not in data:
        raise KeyError(f"reference key {cfg.reference_key!r} not found")

    support = _as_tshw(data[cfg.support_key], name=cfg.support_key)
    reference = _as_dense_reference_tshw(data[cfg.reference_key], name=cfg.reference_key)
    depth_keys = _candidate_depth_source_keys(data, cfg, depth_source_keys)
    depth_for_candidates = _as_tshw(data[depth_keys[0]], name=depth_keys[0])
    support_values = _candidate_support_thresholds(support_thresholds)
    mask_values = _explicit_or_candidate_deadbands(
        depth_for_candidates,
        candidate_count=int(cfg.candidate_count),
        max_deadband_m=cfg.max_mask_deadband_m,
        candidates_m=mask_deadbands_m,
    )
    depth_values = _explicit_or_candidate_deadbands(
        depth_for_candidates,
        candidate_count=int(cfg.candidate_count),
        max_deadband_m=cfg.max_depth_deadband_m,
        candidates_m=depth_deadbands_m,
    )
    scale_values = _candidate_scales(scales)
    power_values = _candidate_support_powers(support_powers)
    floor_values = _candidate_active_floors(active_floor_m)

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for depth_key in depth_keys:
        for scale in scale_values:
            for support_threshold in support_values:
                for support_power in power_values:
                    for mask_deadband_m in mask_values:
                        for depth_deadband_m in depth_values:
                            for floor_m in floor_values:
                                candidate_trace = apply_pressure_sample_support_to_trace(
                                    data,
                                    support_threshold=float(support_threshold),
                                    mask_deadband_m=float(mask_deadband_m),
                                    depth_deadband_m=float(depth_deadband_m),
                                    scale=float(scale),
                                    support_power=float(support_power),
                                    active_floor_m=float(floor_m),
                                    support_key=cfg.support_key,
                                    depth_source_key=depth_key,
                                    pressure_key=cfg.pressure_key,
                                    raw_pressure_key=cfg.raw_pressure_key,
                                    penetration_key=cfg.penetration_key,
                                    output_pressure_key=cfg.pressure_key,
                                    output_raw_pressure_key=cfg.raw_pressure_key,
                                    output_penetration_key=cfg.penetration_key,
                                )
                                report = pressure_trace_report(
                                    candidate_trace,
                                    pressure_key=cfg.pressure_key,
                                    raw_pressure_key=cfg.raw_pressure_key,
                                    penetration_key=cfg.penetration_key,
                                    reference_key=cfg.reference_key,
                                    reference_layer=cfg.reference_layer,
                                    reference_valid_mask_key=cfg.reference_valid_mask_key,
                                    active_threshold=float(cfg.active_threshold),
                                    penetration_threshold=float(cfg.penetration_threshold),
                                    reference_threshold=float(cfg.reference_threshold),
                                )
                                evaluation = evaluate_pressure_trace_report(
                                    report,
                                    reference_iou_threshold=float(cfg.reference_iou_threshold),
                                    centroid_error_threshold_px=float(cfg.centroid_error_threshold_px),
                                    bbox_error_threshold_px=float(cfg.bbox_error_threshold_px),
                                    depth_rmse_threshold_m=float(cfg.depth_rmse_threshold_m),
                                    onset_error_threshold_frames=int(cfg.onset_error_threshold_frames),
                                    offset_error_threshold_frames=int(cfg.offset_error_threshold_frames),
                                )
                                entry = _sample_support_candidate_summary(
                                    float(support_threshold),
                                    float(mask_deadband_m),
                                    float(depth_deadband_m),
                                    float(support_power),
                                    float(floor_m),
                                    depth_key,
                                    report,
                                    evaluation,
                                    scale=float(scale),
                                )
                                candidates.append(entry)
                                if best is None or (
                                    entry["score"],
                                    entry["scale"],
                                    entry["support_threshold"],
                                    entry["support_power"],
                                    entry["mask_deadband_m"],
                                    entry["depth_deadband_m"],
                                    entry["active_floor_m"],
                                    entry["depth_source_key"],
                                ) < (
                                    best["score"],
                                    best["scale"],
                                    best["support_threshold"],
                                    best["support_power"],
                                    best["mask_deadband_m"],
                                    best["depth_deadband_m"],
                                    best["active_floor_m"],
                                    best["depth_source_key"],
                                ):
                                    best = entry

    assert best is not None
    return {
        "best": best,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "config": _config_dict(cfg),
        "support_threshold_candidates": [float(v) for v in support_values],
        "mask_deadband_candidates": [float(v) for v in mask_values],
        "depth_deadband_candidates": [float(v) for v in depth_values],
        "scale_candidates": [float(v) for v in scale_values],
        "support_power_candidates": [float(v) for v in power_values],
        "active_floor_candidates": [float(v) for v in floor_values],
        "depth_source_keys": list(depth_keys),
        "reference_shape": [int(v) for v in reference.shape],
        "support_shape": [int(v) for v in support.shape],
    }


def fit_pressure_spatial_footprint_to_reference(
    trace: Mapping[str, Any] | str | Path,
    *,
    config: PressureSpatialFootprintFitConfig | None = None,
    support_thresholds: Iterable[float] | None = None,
    mask_deadbands_m: Iterable[float] | None = None,
    depth_deadbands_m: Iterable[float] | None = None,
    scales: Iterable[float] | None = None,
    support_powers: Iterable[float] | None = None,
    depth_source_keys: Iterable[str] | None = None,
    active_floor_m: Iterable[float] | None = None,
    transition_depths_m: Iterable[float] | None = None,
    early_ops: Iterable[str] | None = None,
    late_ops: Iterable[str] | None = None,
    early_iterations: Iterable[int] | None = None,
    late_iterations: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Search a spatial footprint law against a dense reference.

    The law starts from finite-area support/depth fields, then applies one
    morphology operation to shallow/early frames and another operation to deeper
    frames. It is a diagnostic candidate for footprint mismatch; it is not a
    ground-truth generator.
    """

    cfg = config or PressureSpatialFootprintFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if cfg.support_key not in data:
        raise KeyError(f"support key {cfg.support_key!r} not found")
    if cfg.reference_key not in data:
        raise KeyError(f"reference key {cfg.reference_key!r} not found")

    support = _as_tshw(data[cfg.support_key], name=cfg.support_key)
    reference = _as_dense_reference_tshw(data[cfg.reference_key], name=cfg.reference_key)
    depth_keys = _candidate_depth_source_keys(data, cfg, depth_source_keys)
    depth_for_candidates = _as_tshw(data[depth_keys[0]], name=depth_keys[0])
    support_values = _candidate_support_thresholds(support_thresholds)
    mask_values = _explicit_or_candidate_deadbands(
        depth_for_candidates,
        candidate_count=int(cfg.candidate_count),
        max_deadband_m=cfg.max_mask_deadband_m,
        candidates_m=mask_deadbands_m,
    )
    depth_values = _explicit_or_candidate_deadbands(
        depth_for_candidates,
        candidate_count=int(cfg.candidate_count),
        max_deadband_m=cfg.max_depth_deadband_m,
        candidates_m=depth_deadbands_m,
    )
    scale_values = _candidate_scales(scales)
    power_values = _candidate_support_powers(support_powers)
    floor_values = _candidate_active_floors(active_floor_m)
    transition_values = _candidate_transition_depths(depth_for_candidates, transition_depths_m)
    early_op_values = _candidate_morph_ops(early_ops)
    late_op_values = _candidate_morph_ops(late_ops)
    early_iter_values = _candidate_iterations(early_iterations)
    late_iter_values = _candidate_iterations(late_iterations)

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for depth_key in depth_keys:
        for scale in scale_values:
            for support_threshold in support_values:
                for support_power in power_values:
                    for mask_deadband_m in mask_values:
                        for depth_deadband_m in depth_values:
                            for floor_m in floor_values:
                                for transition_m in transition_values:
                                    for early_op in early_op_values:
                                        for early_iter in early_iter_values:
                                            for late_op in late_op_values:
                                                for late_iter in late_iter_values:
                                                    if early_op == "none" and int(early_iter) > 0:
                                                        continue
                                                    if late_op == "none" and int(late_iter) > 0:
                                                        continue
                                                    candidate_trace = apply_pressure_spatial_footprint_to_trace(
                                                        data,
                                                        support_threshold=float(support_threshold),
                                                        mask_deadband_m=float(mask_deadband_m),
                                                        depth_deadband_m=float(depth_deadband_m),
                                                        scale=float(scale),
                                                        support_power=float(support_power),
                                                        active_floor_m=float(floor_m),
                                                        transition_depth_m=float(transition_m),
                                                        early_op=str(early_op),
                                                        late_op=str(late_op),
                                                        early_iterations=int(early_iter),
                                                        late_iterations=int(late_iter),
                                                        connectivity=int(cfg.connectivity),
                                                        support_key=cfg.support_key,
                                                        depth_source_key=depth_key,
                                                        pressure_key=cfg.pressure_key,
                                                        raw_pressure_key=cfg.raw_pressure_key,
                                                        penetration_key=cfg.penetration_key,
                                                        output_pressure_key=cfg.pressure_key,
                                                        output_raw_pressure_key=cfg.raw_pressure_key,
                                                        output_penetration_key=cfg.penetration_key,
                                                    )
                                                    report = pressure_trace_report(
                                                        candidate_trace,
                                                        pressure_key=cfg.pressure_key,
                                                        raw_pressure_key=cfg.raw_pressure_key,
                                                        penetration_key=cfg.penetration_key,
                                                        reference_key=cfg.reference_key,
                                                        reference_layer=cfg.reference_layer,
                                                        reference_valid_mask_key=cfg.reference_valid_mask_key,
                                                        active_threshold=float(cfg.active_threshold),
                                                        penetration_threshold=float(cfg.penetration_threshold),
                                                        reference_threshold=float(cfg.reference_threshold),
                                                    )
                                                    evaluation = evaluate_pressure_trace_report(
                                                        report,
                                                        reference_iou_threshold=float(cfg.reference_iou_threshold),
                                                        centroid_error_threshold_px=float(cfg.centroid_error_threshold_px),
                                                        bbox_error_threshold_px=float(cfg.bbox_error_threshold_px),
                                                        depth_rmse_threshold_m=float(cfg.depth_rmse_threshold_m),
                                                        onset_error_threshold_frames=int(cfg.onset_error_threshold_frames),
                                                        offset_error_threshold_frames=int(cfg.offset_error_threshold_frames),
                                                    )
                                                    entry = _spatial_footprint_candidate_summary(
                                                        float(support_threshold),
                                                        float(mask_deadband_m),
                                                        float(depth_deadband_m),
                                                        float(support_power),
                                                        float(floor_m),
                                                        float(transition_m),
                                                        str(early_op),
                                                        int(early_iter),
                                                        str(late_op),
                                                        int(late_iter),
                                                        depth_key,
                                                        report,
                                                        evaluation,
                                                        scale=float(scale),
                                                    )
                                                    candidates.append(entry)
                                                    if best is None or (
                                                        entry["score"],
                                                        entry["scale"],
                                                        entry["transition_depth_m"],
                                                        entry["support_threshold"],
                                                        entry["mask_deadband_m"],
                                                        entry["depth_deadband_m"],
                                                        entry["early_op"],
                                                        entry["early_iterations"],
                                                        entry["late_op"],
                                                        entry["late_iterations"],
                                                        entry["depth_source_key"],
                                                    ) < (
                                                        best["score"],
                                                        best["scale"],
                                                        best["transition_depth_m"],
                                                        best["support_threshold"],
                                                        best["mask_deadband_m"],
                                                        best["depth_deadband_m"],
                                                        best["early_op"],
                                                        best["early_iterations"],
                                                        best["late_op"],
                                                        best["late_iterations"],
                                                        best["depth_source_key"],
                                                    ):
                                                        best = entry

    assert best is not None
    return {
        "best": best,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "config": _config_dict(cfg),
        "support_threshold_candidates": [float(v) for v in support_values],
        "mask_deadband_candidates": [float(v) for v in mask_values],
        "depth_deadband_candidates": [float(v) for v in depth_values],
        "scale_candidates": [float(v) for v in scale_values],
        "support_power_candidates": [float(v) for v in power_values],
        "active_floor_candidates": [float(v) for v in floor_values],
        "transition_depth_candidates": [float(v) for v in transition_values],
        "early_ops": list(early_op_values),
        "late_ops": list(late_op_values),
        "early_iterations": [int(v) for v in early_iter_values],
        "late_iterations": [int(v) for v in late_iter_values],
        "depth_source_keys": list(depth_keys),
        "reference_shape": [int(v) for v in reference.shape],
        "support_shape": [int(v) for v in support.shape],
    }


def fit_pressure_area_fraction_to_reference(
    trace: Mapping[str, Any] | str | Path,
    *,
    config: PressureAreaFractionFitConfig | None = None,
    area_modes: Iterable[str] | None = None,
    blend_weights: Iterable[float] | None = None,
    support_thresholds: Iterable[float] | None = None,
    mask_deadbands_m: Iterable[float] | None = None,
    depth_deadbands_m: Iterable[float] | None = None,
    scales: Iterable[float] | None = None,
    active_floor_m: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Search sampled area-fraction/depth integration against a dense reference."""

    cfg = config or PressureAreaFractionFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    if cfg.support_key not in data:
        raise KeyError(f"support key {cfg.support_key!r} not found")
    if cfg.reference_key not in data:
        raise KeyError(f"reference key {cfg.reference_key!r} not found")

    support = _as_tshw(data[cfg.support_key], name=cfg.support_key)
    reference = _as_dense_reference_tshw(data[cfg.reference_key], name=cfg.reference_key)
    depth_for_candidates = _area_fraction_depth(data, cfg, mode="mean", blend_weight=0.0)
    support_values = _candidate_support_thresholds(support_thresholds)
    mask_values = _explicit_or_candidate_deadbands(
        depth_for_candidates,
        candidate_count=int(cfg.candidate_count),
        max_deadband_m=cfg.max_mask_deadband_m,
        candidates_m=mask_deadbands_m,
    )
    depth_values = _explicit_or_candidate_deadbands(
        depth_for_candidates,
        candidate_count=int(cfg.candidate_count),
        max_deadband_m=cfg.max_depth_deadband_m,
        candidates_m=depth_deadbands_m,
    )
    scale_values = _candidate_scales(scales)
    floor_values = _candidate_active_floors(active_floor_m)
    mode_values = _candidate_area_modes(data, cfg, area_modes)
    blend_values = _candidate_blend_weights(blend_weights)

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for area_mode in mode_values:
        for blend_weight in blend_values:
            if not str(area_mode).startswith("blend") and float(blend_weight) != 0.0:
                continue
            for scale in scale_values:
                for support_threshold in support_values:
                    for mask_deadband_m in mask_values:
                        for depth_deadband_m in depth_values:
                            for floor_m in floor_values:
                                candidate_trace = apply_pressure_area_fraction_to_trace(
                                    data,
                                    area_mode=str(area_mode),
                                    blend_weight=float(blend_weight),
                                    support_threshold=float(support_threshold),
                                    mask_deadband_m=float(mask_deadband_m),
                                    depth_deadband_m=float(depth_deadband_m),
                                    scale=float(scale),
                                    active_floor_m=float(floor_m),
                                    support_key=cfg.support_key,
                                    mean_depth_key=cfg.mean_depth_key,
                                    positive_mean_depth_key=cfg.positive_mean_depth_key,
                                    max_depth_key=cfg.max_depth_key,
                                    center_depth_key=cfg.center_depth_key,
                                    pressure_key=cfg.pressure_key,
                                    raw_pressure_key=cfg.raw_pressure_key,
                                    penetration_key=cfg.penetration_key,
                                    output_pressure_key=cfg.pressure_key,
                                    output_raw_pressure_key=cfg.raw_pressure_key,
                                    output_penetration_key=cfg.penetration_key,
                                )
                                report = pressure_trace_report(
                                    candidate_trace,
                                    pressure_key=cfg.pressure_key,
                                    raw_pressure_key=cfg.raw_pressure_key,
                                    penetration_key=cfg.penetration_key,
                                    reference_key=cfg.reference_key,
                                    reference_layer=cfg.reference_layer,
                                    reference_valid_mask_key=cfg.reference_valid_mask_key,
                                    active_threshold=float(cfg.active_threshold),
                                    penetration_threshold=float(cfg.penetration_threshold),
                                    reference_threshold=float(cfg.reference_threshold),
                                )
                                evaluation = evaluate_pressure_trace_report(
                                    report,
                                    reference_iou_threshold=float(cfg.reference_iou_threshold),
                                    centroid_error_threshold_px=float(cfg.centroid_error_threshold_px),
                                    bbox_error_threshold_px=float(cfg.bbox_error_threshold_px),
                                    depth_rmse_threshold_m=float(cfg.depth_rmse_threshold_m),
                                    onset_error_threshold_frames=int(cfg.onset_error_threshold_frames),
                                    offset_error_threshold_frames=int(cfg.offset_error_threshold_frames),
                                )
                                entry = _area_fraction_candidate_summary(
                                    str(area_mode),
                                    float(blend_weight),
                                    float(support_threshold),
                                    float(mask_deadband_m),
                                    float(depth_deadband_m),
                                    float(floor_m),
                                    report,
                                    evaluation,
                                    scale=float(scale),
                                )
                                candidates.append(entry)
                                if best is None or (
                                    entry["score"],
                                    entry["scale"],
                                    entry["area_mode"],
                                    entry["blend_weight"],
                                    entry["support_threshold"],
                                    entry["mask_deadband_m"],
                                    entry["depth_deadband_m"],
                                    entry["active_floor_m"],
                                ) < (
                                    best["score"],
                                    best["scale"],
                                    best["area_mode"],
                                    best["blend_weight"],
                                    best["support_threshold"],
                                    best["mask_deadband_m"],
                                    best["depth_deadband_m"],
                                    best["active_floor_m"],
                                ):
                                    best = entry

    assert best is not None
    return {
        "best": best,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "config": _config_dict(cfg),
        "area_modes": list(mode_values),
        "blend_weights": [float(v) for v in blend_values],
        "support_threshold_candidates": [float(v) for v in support_values],
        "mask_deadband_candidates": [float(v) for v in mask_values],
        "depth_deadband_candidates": [float(v) for v in depth_values],
        "scale_candidates": [float(v) for v in scale_values],
        "active_floor_candidates": [float(v) for v in floor_values],
        "reference_shape": [int(v) for v in reference.shape],
        "support_shape": [int(v) for v in support.shape],
    }


def apply_pressure_deadband_to_trace(
    trace: Mapping[str, Any],
    *,
    deadband_m: float,
    scale: float = 1.0,
    pressure_key: str = "pressure_norm",
    raw_pressure_key: str = "pressure_raw_n",
    penetration_key: str = "penetration_m",
    output_pressure_key: str = "pressure_norm_fit",
    output_raw_pressure_key: str = "pressure_raw_fit_n",
    output_penetration_key: str = "penetration_fit_m",
) -> dict[str, Any]:
    """Return a trace copy with depth-scale and deadband-corrected arrays."""

    out = dict(trace)
    penetration = _as_tshw(trace[penetration_key], name=penetration_key)
    corrected = np.maximum(float(scale) * penetration - max(0.0, float(deadband_m)), 0.0).astype(np.float32)
    norm = _normalize_per_sensor(corrected)
    out[output_penetration_key] = corrected
    out[output_raw_pressure_key] = corrected
    out[output_pressure_key] = norm
    if output_pressure_key != pressure_key and pressure_key not in out:
        out[pressure_key] = norm
    if output_raw_pressure_key != raw_pressure_key and raw_pressure_key not in out:
        out[raw_pressure_key] = corrected
    return out


def apply_pressure_mask_depth_to_trace(
    trace: Mapping[str, Any],
    *,
    mask_deadband_m: float,
    depth_deadband_m: float,
    scale: float = 1.0,
    active_floor_m: float = 0.0,
    pressure_key: str = "pressure_norm",
    raw_pressure_key: str = "pressure_raw_n",
    penetration_key: str = "penetration_m",
    output_pressure_key: str = "pressure_norm_mask_depth_fit",
    output_raw_pressure_key: str = "pressure_raw_mask_depth_fit_n",
    output_penetration_key: str = "penetration_mask_depth_fit_m",
) -> dict[str, Any]:
    """Return a trace copy with active mask and depth calibration split."""

    out = dict(trace)
    penetration = _as_tshw(trace[penetration_key], name=penetration_key)
    active = penetration > max(0.0, float(mask_deadband_m))
    corrected = np.maximum(float(scale) * penetration - max(0.0, float(depth_deadband_m)), 0.0)
    active_depth = np.where(active, np.maximum(corrected, max(0.0, float(active_floor_m))), 0.0).astype(np.float32)
    norm = _normalize_per_sensor(active_depth)
    out[output_penetration_key] = active_depth
    out[output_raw_pressure_key] = active_depth
    out[output_pressure_key] = norm
    if output_pressure_key != pressure_key and pressure_key not in out:
        out[pressure_key] = norm
    if output_raw_pressure_key != raw_pressure_key and raw_pressure_key not in out:
        out[raw_pressure_key] = active_depth
    return out


def apply_pressure_sample_support_to_trace(
    trace: Mapping[str, Any],
    *,
    support_threshold: float,
    mask_deadband_m: float,
    depth_deadband_m: float,
    scale: float = 1.0,
    support_power: float = 0.0,
    active_floor_m: float = 0.0,
    support_key: str = "geometry_normal_ray_sample_support_fraction",
    depth_source_key: str = "geometry_normal_ray_sample_mean_penetration_m",
    pressure_key: str = "pressure_norm",
    raw_pressure_key: str = "pressure_raw_n",
    penetration_key: str = "penetration_m",
    output_pressure_key: str = "pressure_norm_sample_support_fit",
    output_raw_pressure_key: str = "pressure_raw_sample_support_fit_n",
    output_penetration_key: str = "penetration_sample_support_fit_m",
) -> dict[str, Any]:
    """Return a trace copy with a finite-area support/depth correction."""

    out = dict(trace)
    support = np.clip(_as_tshw(trace[support_key], name=support_key), 0.0, 1.0)
    depth = _as_tshw(trace[depth_source_key], name=depth_source_key)
    if support.shape != depth.shape:
        raise ValueError(f"support/depth shapes must match, got {support.shape}/{depth.shape}")

    active = (support > 0.0) & (support >= max(0.0, float(support_threshold)))
    active &= depth > max(0.0, float(mask_deadband_m))
    power = max(0.0, float(support_power))
    support_weight = np.power(np.where(support > 0.0, support, 1.0), power).astype(np.float32)
    corrected = np.maximum(float(scale) * depth * support_weight - max(0.0, float(depth_deadband_m)), 0.0)
    active_depth = np.where(active, np.maximum(corrected, max(0.0, float(active_floor_m))), 0.0).astype(np.float32)
    norm = _normalize_per_sensor(active_depth)
    out[output_penetration_key] = active_depth
    out[output_raw_pressure_key] = active_depth
    out[output_pressure_key] = norm
    if output_pressure_key != pressure_key and pressure_key not in out:
        out[pressure_key] = norm
    if output_raw_pressure_key != raw_pressure_key and raw_pressure_key not in out:
        out[raw_pressure_key] = active_depth
    if output_penetration_key != penetration_key and penetration_key not in out:
        out[penetration_key] = active_depth
    return out


def apply_pressure_spatial_footprint_to_trace(
    trace: Mapping[str, Any],
    *,
    support_threshold: float,
    mask_deadband_m: float,
    depth_deadband_m: float,
    scale: float = 1.0,
    support_power: float = 0.0,
    active_floor_m: float = 0.0,
    transition_depth_m: float = 0.0,
    early_op: str = "none",
    late_op: str = "none",
    early_iterations: int = 0,
    late_iterations: int = 0,
    connectivity: int = 8,
    support_key: str = "geometry_normal_ray_sample_support_fraction",
    depth_source_key: str = "geometry_normal_ray_sample_mean_penetration_m",
    pressure_key: str = "pressure_norm",
    raw_pressure_key: str = "pressure_raw_n",
    penetration_key: str = "penetration_m",
    output_pressure_key: str = "pressure_norm_spatial_footprint_fit",
    output_raw_pressure_key: str = "pressure_raw_spatial_footprint_fit_n",
    output_penetration_key: str = "penetration_spatial_footprint_fit_m",
) -> dict[str, Any]:
    """Return a trace copy with a taxel-grid spatial footprint correction."""

    out = dict(trace)
    support = np.clip(_as_tshw(trace[support_key], name=support_key), 0.0, 1.0)
    depth = _as_tshw(trace[depth_source_key], name=depth_source_key)
    if support.shape != depth.shape:
        raise ValueError(f"support/depth shapes must match, got {support.shape}/{depth.shape}")

    active = (support > 0.0) & (support >= max(0.0, float(support_threshold)))
    active &= depth > max(0.0, float(mask_deadband_m))
    power = max(0.0, float(support_power))
    support_weight = np.power(np.where(support > 0.0, support, 1.0), power).astype(np.float32)
    corrected = np.maximum(float(scale) * depth * support_weight - max(0.0, float(depth_deadband_m)), 0.0)

    corrected_spread = _spread_depth_for_mask(corrected, max(0, int(max(early_iterations, late_iterations))), connectivity)
    peak_depth = np.max(np.where(active, depth, 0.0), axis=(-2, -1), keepdims=True)
    early_frames = peak_depth <= max(0.0, float(transition_depth_m))
    early_mask = _morph_mask(active, str(early_op), int(early_iterations), connectivity)
    late_mask = _morph_mask(active, str(late_op), int(late_iterations), connectivity)
    spatial_mask = np.where(early_frames, early_mask, late_mask)
    active_depth = np.where(spatial_mask, np.maximum(corrected_spread, max(0.0, float(active_floor_m))), 0.0).astype(np.float32)
    norm = _normalize_per_sensor(active_depth)
    out[output_penetration_key] = active_depth
    out[output_raw_pressure_key] = active_depth
    out[output_pressure_key] = norm
    if output_pressure_key != pressure_key and pressure_key not in out:
        out[pressure_key] = norm
    if output_raw_pressure_key != raw_pressure_key and raw_pressure_key not in out:
        out[raw_pressure_key] = active_depth
    if output_penetration_key != penetration_key and penetration_key not in out:
        out[penetration_key] = active_depth
    return out


def apply_pressure_area_fraction_to_trace(
    trace: Mapping[str, Any],
    *,
    area_mode: str,
    blend_weight: float = 0.0,
    support_threshold: float,
    mask_deadband_m: float,
    depth_deadband_m: float,
    scale: float = 1.0,
    active_floor_m: float = 0.0,
    support_key: str = "geometry_normal_ray_sample_support_fraction",
    mean_depth_key: str = "geometry_normal_ray_sample_mean_penetration_m",
    positive_mean_depth_key: str = "geometry_normal_ray_sample_positive_mean_penetration_m",
    max_depth_key: str = "geometry_normal_ray_sample_max_penetration_m",
    center_depth_key: str = "geometry_normal_ray_penetration_m",
    pressure_key: str = "pressure_norm",
    raw_pressure_key: str = "pressure_raw_n",
    penetration_key: str = "penetration_m",
    output_pressure_key: str = "pressure_norm_area_fraction_fit",
    output_raw_pressure_key: str = "pressure_raw_area_fraction_fit_n",
    output_penetration_key: str = "penetration_area_fraction_fit_m",
) -> dict[str, Any]:
    """Return a trace copy with sampled area-fraction depth integration."""

    out = dict(trace)
    support = np.clip(_as_tshw(trace[support_key], name=support_key), 0.0, 1.0)
    cfg = PressureAreaFractionFitConfig(
        support_key=support_key,
        mean_depth_key=mean_depth_key,
        positive_mean_depth_key=positive_mean_depth_key,
        max_depth_key=max_depth_key,
        center_depth_key=center_depth_key,
    )
    area_depth = _area_fraction_depth(trace, cfg, mode=str(area_mode), blend_weight=float(blend_weight))
    if support.shape != area_depth.shape:
        raise ValueError(f"support/depth shapes must match, got {support.shape}/{area_depth.shape}")

    active = (support > 0.0) & (support >= max(0.0, float(support_threshold)))
    active &= area_depth > max(0.0, float(mask_deadband_m))
    corrected = np.maximum(float(scale) * area_depth - max(0.0, float(depth_deadband_m)), 0.0)
    active_depth = np.where(active, np.maximum(corrected, max(0.0, float(active_floor_m))), 0.0).astype(np.float32)
    norm = _normalize_per_sensor(active_depth)
    out[output_penetration_key] = active_depth
    out[output_raw_pressure_key] = active_depth
    out[output_pressure_key] = norm
    if output_pressure_key != pressure_key and pressure_key not in out:
        out[pressure_key] = norm
    if output_raw_pressure_key != raw_pressure_key and raw_pressure_key not in out:
        out[raw_pressure_key] = active_depth
    if output_penetration_key != penetration_key and penetration_key not in out:
        out[penetration_key] = active_depth
    return out


def write_area_fraction_fit_trace(
    trace: Mapping[str, Any] | str | Path,
    out_path: str | Path,
    *,
    area_mode: str,
    blend_weight: float = 0.0,
    support_threshold: float,
    mask_deadband_m: float,
    depth_deadband_m: float,
    scale: float = 1.0,
    active_floor_m: float = 0.0,
    config: PressureAreaFractionFitConfig | None = None,
) -> Path:
    """Write a trace with additional sampled area-fraction fit arrays."""

    cfg = config or PressureAreaFractionFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    fitted = apply_pressure_area_fraction_to_trace(
        data,
        area_mode=str(area_mode),
        blend_weight=float(blend_weight),
        support_threshold=float(support_threshold),
        mask_deadband_m=float(mask_deadband_m),
        depth_deadband_m=float(depth_deadband_m),
        scale=float(scale),
        active_floor_m=float(active_floor_m),
        support_key=cfg.support_key,
        mean_depth_key=cfg.mean_depth_key,
        positive_mean_depth_key=cfg.positive_mean_depth_key,
        max_depth_key=cfg.max_depth_key,
        center_depth_key=cfg.center_depth_key,
        pressure_key=cfg.pressure_key,
        raw_pressure_key=cfg.raw_pressure_key,
        penetration_key=cfg.penetration_key,
    )
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in fitted.items() if _npz_safe(value)}
    np.savez_compressed(path, **payload)
    return path


def write_spatial_footprint_fit_trace(
    trace: Mapping[str, Any] | str | Path,
    out_path: str | Path,
    *,
    support_threshold: float,
    mask_deadband_m: float,
    depth_deadband_m: float,
    scale: float = 1.0,
    support_power: float = 0.0,
    active_floor_m: float = 0.0,
    transition_depth_m: float = 0.0,
    early_op: str = "none",
    late_op: str = "none",
    early_iterations: int = 0,
    late_iterations: int = 0,
    depth_source_key: str | None = None,
    config: PressureSpatialFootprintFitConfig | None = None,
) -> Path:
    """Write a trace with additional spatial footprint fit arrays."""

    cfg = config or PressureSpatialFootprintFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    depth_key = depth_source_key or cfg.default_depth_source_key
    fitted = apply_pressure_spatial_footprint_to_trace(
        data,
        support_threshold=float(support_threshold),
        mask_deadband_m=float(mask_deadband_m),
        depth_deadband_m=float(depth_deadband_m),
        scale=float(scale),
        support_power=float(support_power),
        active_floor_m=float(active_floor_m),
        transition_depth_m=float(transition_depth_m),
        early_op=str(early_op),
        late_op=str(late_op),
        early_iterations=int(early_iterations),
        late_iterations=int(late_iterations),
        connectivity=int(cfg.connectivity),
        support_key=cfg.support_key,
        depth_source_key=depth_key,
        pressure_key=cfg.pressure_key,
        raw_pressure_key=cfg.raw_pressure_key,
        penetration_key=cfg.penetration_key,
    )
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in fitted.items() if _npz_safe(value)}
    np.savez_compressed(path, **payload)
    return path


def write_sample_support_fit_trace(
    trace: Mapping[str, Any] | str | Path,
    out_path: str | Path,
    *,
    support_threshold: float,
    mask_deadband_m: float,
    depth_deadband_m: float,
    scale: float = 1.0,
    support_power: float = 0.0,
    active_floor_m: float = 0.0,
    depth_source_key: str | None = None,
    config: PressureSampleSupportFitConfig | None = None,
) -> Path:
    """Write a trace with additional finite-area support fit arrays."""

    cfg = config or PressureSampleSupportFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    depth_key = depth_source_key or cfg.default_depth_source_key
    fitted = apply_pressure_sample_support_to_trace(
        data,
        support_threshold=float(support_threshold),
        mask_deadband_m=float(mask_deadband_m),
        depth_deadband_m=float(depth_deadband_m),
        scale=float(scale),
        support_power=float(support_power),
        active_floor_m=float(active_floor_m),
        support_key=cfg.support_key,
        depth_source_key=depth_key,
        pressure_key=cfg.pressure_key,
        raw_pressure_key=cfg.raw_pressure_key,
        penetration_key=cfg.penetration_key,
    )
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in fitted.items() if _npz_safe(value)}
    np.savez_compressed(path, **payload)
    return path


def write_mask_depth_fit_trace(
    trace: Mapping[str, Any] | str | Path,
    out_path: str | Path,
    *,
    mask_deadband_m: float,
    depth_deadband_m: float,
    scale: float = 1.0,
    active_floor_m: float = 0.0,
    config: PressureMaskDepthFitConfig | None = None,
) -> Path:
    """Write a trace with additional split mask/depth fit arrays."""

    cfg = config or PressureMaskDepthFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    fitted = apply_pressure_mask_depth_to_trace(
        data,
        mask_deadband_m=float(mask_deadband_m),
        depth_deadband_m=float(depth_deadband_m),
        scale=float(scale),
        active_floor_m=float(active_floor_m),
        pressure_key=cfg.pressure_key,
        raw_pressure_key=cfg.raw_pressure_key,
        penetration_key=cfg.penetration_key,
    )
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in fitted.items() if _npz_safe(value)}
    np.savez_compressed(path, **payload)
    return path


def write_deadband_fit_trace(
    trace: Mapping[str, Any] | str | Path,
    out_path: str | Path,
    *,
    deadband_m: float,
    scale: float = 1.0,
    config: PressureDeadbandFitConfig | None = None,
) -> Path:
    """Write a trace with additional ``*_fit`` arrays for verifier re-use."""

    cfg = config or PressureDeadbandFitConfig()
    data = load_pressure_trace_npz(trace) if isinstance(trace, (str, Path)) else dict(trace)
    fitted = apply_pressure_deadband_to_trace(
        data,
        deadband_m=float(deadband_m),
        scale=float(scale),
        pressure_key=cfg.pressure_key,
        raw_pressure_key=cfg.raw_pressure_key,
        penetration_key=cfg.penetration_key,
    )
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in fitted.items() if _npz_safe(value)}
    np.savez_compressed(path, **payload)
    return path


def _candidate_deadbands(
    penetration: np.ndarray,
    *,
    candidate_count: int,
    max_deadband_m: float | None,
    candidates_m: Iterable[float] | None,
) -> np.ndarray:
    values: list[float] = []
    if candidates_m is not None:
        values.extend(float(v) for v in candidates_m)
    max_pen = float(np.nanmax(penetration)) if penetration.size else 0.0
    upper = max_pen if max_deadband_m is None else min(max_pen, max(0.0, float(max_deadband_m)))
    count = max(2, int(candidate_count))
    if upper > 0.0:
        values.extend(np.linspace(0.0, upper, count, dtype=np.float64).tolist())
        positive = penetration[penetration > 0.0]
        if positive.size:
            quantiles = np.linspace(0.0, 0.95, min(count, 32), dtype=np.float64)
            values.extend(np.quantile(positive.astype(np.float64), quantiles).tolist())
    else:
        values.append(0.0)
    arr = np.asarray(sorted({round(max(0.0, float(v)), 12) for v in values}), dtype=np.float64)
    return arr


def _explicit_or_candidate_deadbands(
    penetration: np.ndarray,
    *,
    candidate_count: int,
    max_deadband_m: float | None,
    candidates_m: Iterable[float] | None,
) -> np.ndarray:
    if candidates_m is not None:
        values = sorted({round(max(0.0, float(v)), 12) for v in candidates_m})
        if not values:
            raise ValueError("explicit deadband candidates must contain at least one value")
        return np.asarray(values, dtype=np.float64)
    return _candidate_deadbands(
        penetration,
        candidate_count=candidate_count,
        max_deadband_m=max_deadband_m,
        candidates_m=None,
    )


def _candidate_scales(scales: Iterable[float] | None) -> np.ndarray:
    if scales is None:
        return np.asarray([1.0], dtype=np.float64)
    values = sorted({round(float(v), 12) for v in scales if np.isfinite(float(v)) and float(v) > 0.0})
    if not values:
        raise ValueError("scale candidates must contain at least one positive finite value")
    return np.asarray(values, dtype=np.float64)


def _candidate_active_floors(values: Iterable[float] | None) -> np.ndarray:
    if values is None:
        return np.asarray([0.0, 1.0e-9], dtype=np.float64)
    floors = sorted({round(max(0.0, float(v)), 12) for v in values if np.isfinite(float(v))})
    if not floors:
        floors = [0.0]
    return np.asarray(floors, dtype=np.float64)


def _candidate_support_thresholds(values: Iterable[float] | None) -> np.ndarray:
    if values is None:
        return np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    thresholds = sorted({round(max(0.0, min(1.0, float(v))), 12) for v in values if np.isfinite(float(v))})
    if not thresholds:
        thresholds = [0.0]
    return np.asarray(thresholds, dtype=np.float64)


def _candidate_support_powers(values: Iterable[float] | None) -> np.ndarray:
    if values is None:
        return np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    powers = sorted({round(max(0.0, float(v)), 12) for v in values if np.isfinite(float(v))})
    if not powers:
        powers = [0.0]
    return np.asarray(powers, dtype=np.float64)


def _candidate_depth_source_keys(
    data: Mapping[str, Any],
    cfg: PressureSampleSupportFitConfig | PressureSpatialFootprintFitConfig,
    values: Iterable[str] | None,
) -> list[str]:
    requested = [str(v) for v in values] if values is not None else [
        cfg.default_depth_source_key,
        cfg.penetration_key,
        "geometry_normal_ray_sample_mean_penetration_m",
        "geometry_normal_ray_sample_positive_mean_penetration_m",
        "geometry_normal_ray_sample_max_penetration_m",
    ]
    keys: list[str] = []
    for key in requested:
        if key in data and key not in keys:
            keys.append(key)
    if not keys:
        raise KeyError("no requested depth source keys are present in the trace")
    return keys


def _candidate_transition_depths(depth: np.ndarray, values: Iterable[float] | None) -> np.ndarray:
    if values is not None:
        depths = sorted({round(max(0.0, float(v)), 12) for v in values if np.isfinite(float(v))})
        if not depths:
            depths = [0.0]
        return np.asarray(depths, dtype=np.float64)
    positive = depth[depth > 0.0]
    if positive.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    quantiles = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
    values_out = [0.0]
    values_out.extend(np.quantile(positive.astype(np.float64), quantiles).tolist())
    return np.asarray(sorted({round(max(0.0, float(v)), 12) for v in values_out}), dtype=np.float64)


def _candidate_morph_ops(values: Iterable[str] | None) -> list[str]:
    ops = [str(v).lower() for v in values] if values is not None else ["none", "erode", "dilate"]
    allowed = {"none", "erode", "dilate"}
    out: list[str] = []
    for op in ops:
        if op not in allowed:
            raise ValueError("morphology operations must be 'none', 'erode', or 'dilate'")
        if op not in out:
            out.append(op)
    return out or ["none"]


def _candidate_iterations(values: Iterable[int] | None) -> np.ndarray:
    if values is None:
        return np.asarray([0, 1], dtype=np.int64)
    iterations = sorted({max(0, int(v)) for v in values})
    if not iterations:
        iterations = [0]
    return np.asarray(iterations, dtype=np.int64)


def _candidate_area_modes(
    data: Mapping[str, Any],
    cfg: PressureAreaFractionFitConfig,
    values: Iterable[str] | None,
) -> list[str]:
    requested = [str(v).lower() for v in values] if values is not None else [
        "mean",
        "support_positive",
        "support_max",
        "support_center",
        "positive",
        "max",
        "blend_mean_support_max",
        "blend_mean_support_center",
    ]
    required = {
        "mean": (cfg.mean_depth_key,),
        "support_positive": (cfg.support_key, cfg.positive_mean_depth_key),
        "support_max": (cfg.support_key, cfg.max_depth_key),
        "support_center": (cfg.support_key, cfg.center_depth_key),
        "positive": (cfg.positive_mean_depth_key,),
        "max": (cfg.max_depth_key,),
        "blend_mean_support_max": (cfg.mean_depth_key, cfg.support_key, cfg.max_depth_key),
        "blend_mean_support_center": (cfg.mean_depth_key, cfg.support_key, cfg.center_depth_key),
    }
    out: list[str] = []
    for mode in requested:
        if mode not in required:
            raise ValueError(
                "area modes must be one of: "
                + ", ".join(sorted(required))
            )
        if all(key in data for key in required[mode]) and mode not in out:
            out.append(mode)
    if not out:
        raise KeyError("no requested area-fraction modes can be built from this trace")
    return out


def _candidate_blend_weights(values: Iterable[float] | None) -> np.ndarray:
    if values is None:
        return np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    weights = sorted({round(max(0.0, min(1.0, float(v))), 12) for v in values if np.isfinite(float(v))})
    if not weights:
        weights = [0.0]
    return np.asarray(weights, dtype=np.float64)


def _candidate_summary(
    deadband_m: float,
    report: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    ref = report.get("reference") if isinstance(report.get("reference"), Mapping) else {}
    checks = evaluation.get("checks", []) if isinstance(evaluation, Mapping) else []
    failed = [str(check.get("name")) for check in checks if not bool(check.get("passed"))]
    score = _score_reference_metrics(report)
    return {
        "deadband_m": float(deadband_m),
        "scale": float(scale),
        "score": float(score),
        "passed": bool(evaluation.get("passed", False)) if isinstance(evaluation, Mapping) else False,
        "failed_checks": failed,
        "reference_layer": ref.get("reference_layer", report.get("reference_layer")),
        "inferred_reference_layer": ref.get("inferred_reference_layer", report.get("inferred_reference_layer")),
        "requested_reference_layer": ref.get("requested_reference_layer", report.get("requested_reference_layer")),
        "reference_layer_override_conflict": bool(
            ref.get("reference_layer_override_conflict", report.get("reference_layer_override_conflict", False))
        ),
        "active_mask_iou_mean": _float_or_none(ref.get("active_mask_iou_mean")),
        "active_mask_iou_min": _float_or_none(ref.get("active_mask_iou_min")),
        "centroid_error_px_mean": _float_or_none(ref.get("centroid_error_px_mean")),
        "centroid_error_px_max": _float_or_none(ref.get("centroid_error_px_max")),
        "bbox_error_px_mean": _float_or_none(ref.get("bbox_error_px_mean")),
        "bbox_error_px_max": _float_or_none(ref.get("bbox_error_px_max")),
        "depth_rmse_m_mean": _float_or_none(ref.get("depth_rmse_m_mean")),
        "depth_rmse_m_max": _float_or_none(ref.get("depth_rmse_m_max")),
        "onset_error_frames_by_sensor": ref.get("onset_error_frames_by_sensor"),
        "offset_error_frames_by_sensor": ref.get("offset_error_frames_by_sensor"),
        "pressure_onset_step_by_sensor": ref.get("pressure_onset_step_by_sensor"),
        "reference_onset_step_by_sensor": ref.get("reference_onset_step_by_sensor"),
        "reference_valid_mask_key": ref.get("reference_valid_mask_key"),
        "alignment_valid_taxel_fraction_min": _float_or_none(ref.get("alignment_valid_taxel_fraction_min")),
        "alignment_contact_region_valid_fraction_min": _float_or_none(
            ref.get("alignment_contact_region_valid_fraction_min")
        ),
        "alignment_invalid_contact_fraction_max": _float_or_none(ref.get("alignment_invalid_contact_fraction_max")),
        "max_active_taxels": report.get("max_active_taxels"),
        "precontact_leakage_fraction": report.get("precontact_leakage_fraction"),
    }


def _mask_depth_candidate_summary(
    mask_deadband_m: float,
    depth_deadband_m: float,
    active_floor_m: float,
    report: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    entry = _candidate_summary(depth_deadband_m, report, evaluation, scale=scale)
    entry["mask_deadband_m"] = float(mask_deadband_m)
    entry["depth_deadband_m"] = float(depth_deadband_m)
    entry["active_floor_m"] = float(active_floor_m)
    entry.pop("deadband_m", None)
    return entry


def _sample_support_candidate_summary(
    support_threshold: float,
    mask_deadband_m: float,
    depth_deadband_m: float,
    support_power: float,
    active_floor_m: float,
    depth_source_key: str,
    report: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    entry = _candidate_summary(depth_deadband_m, report, evaluation, scale=scale)
    entry["support_threshold"] = float(support_threshold)
    entry["mask_deadband_m"] = float(mask_deadband_m)
    entry["depth_deadband_m"] = float(depth_deadband_m)
    entry["support_power"] = float(support_power)
    entry["active_floor_m"] = float(active_floor_m)
    entry["depth_source_key"] = str(depth_source_key)
    entry.pop("deadband_m", None)
    return entry


def _spatial_footprint_candidate_summary(
    support_threshold: float,
    mask_deadband_m: float,
    depth_deadband_m: float,
    support_power: float,
    active_floor_m: float,
    transition_depth_m: float,
    early_op: str,
    early_iterations: int,
    late_op: str,
    late_iterations: int,
    depth_source_key: str,
    report: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    entry = _sample_support_candidate_summary(
        support_threshold,
        mask_deadband_m,
        depth_deadband_m,
        support_power,
        active_floor_m,
        depth_source_key,
        report,
        evaluation,
        scale=scale,
    )
    entry["transition_depth_m"] = float(transition_depth_m)
    entry["early_op"] = str(early_op)
    entry["early_iterations"] = int(early_iterations)
    entry["late_op"] = str(late_op)
    entry["late_iterations"] = int(late_iterations)
    return entry


def _area_fraction_candidate_summary(
    area_mode: str,
    blend_weight: float,
    support_threshold: float,
    mask_deadband_m: float,
    depth_deadband_m: float,
    active_floor_m: float,
    report: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    entry = _candidate_summary(depth_deadband_m, report, evaluation, scale=scale)
    entry["area_mode"] = str(area_mode)
    entry["blend_weight"] = float(blend_weight)
    entry["support_threshold"] = float(support_threshold)
    entry["mask_deadband_m"] = float(mask_deadband_m)
    entry["depth_deadband_m"] = float(depth_deadband_m)
    entry["active_floor_m"] = float(active_floor_m)
    entry.pop("deadband_m", None)
    return entry


def _score_reference_metrics(report: Mapping[str, Any]) -> float:
    ref = report.get("reference") if isinstance(report.get("reference"), Mapping) else {}
    image_shape = report.get("image_shape", [1, 1])
    max_dim = max(1.0, float(max(image_shape))) if isinstance(image_shape, list) else 1.0
    steps = max(1.0, float(report.get("num_steps", 1)))
    iou = _finite_or(ref.get("active_mask_iou_mean"), 0.0)
    centroid = _finite_or(ref.get("centroid_error_px_mean"), max_dim)
    bbox = _finite_or(ref.get("bbox_error_px_mean"), max_dim)
    depth = _finite_or(ref.get("depth_rmse_m_mean"), 1.0)
    depth_scale = max(_finite_or(ref.get("depth_rmse_m_max"), 0.0), _finite_or(ref.get("depth_rmse_m_mean"), 0.0), 1.0e-6)
    onset = _mean_optional_ints(ref.get("onset_error_frames_by_sensor"))
    offset = _mean_optional_ints(ref.get("offset_error_frames_by_sensor"))
    leakage = _finite_or(report.get("precontact_leakage_fraction"), 0.0)
    return (
        4.0 * (1.0 - max(0.0, min(1.0, iou)))
        + centroid / max_dim
        + bbox / max_dim
        + depth / depth_scale
        + onset / steps
        + 0.5 * offset / steps
        + 10.0 * leakage
    )


def _normalize_per_sensor(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    denom = np.max(arr, axis=(-2, -1), keepdims=True)
    return np.divide(arr, denom + 1.0e-12, out=np.zeros_like(arr), where=denom > 0.0).astype(np.float32)


def _area_fraction_depth(
    trace: Mapping[str, Any],
    cfg: PressureAreaFractionFitConfig,
    *,
    mode: str,
    blend_weight: float,
) -> np.ndarray:
    support = np.clip(_as_tshw(trace[cfg.support_key], name=cfg.support_key), 0.0, 1.0)
    mode_name = str(mode).lower()
    if mode_name == "mean":
        return _as_tshw(trace[cfg.mean_depth_key], name=cfg.mean_depth_key)
    if mode_name == "support_positive":
        return (support * _as_tshw(trace[cfg.positive_mean_depth_key], name=cfg.positive_mean_depth_key)).astype(np.float32)
    if mode_name == "support_max":
        return (support * _as_tshw(trace[cfg.max_depth_key], name=cfg.max_depth_key)).astype(np.float32)
    if mode_name == "support_center":
        return (support * _as_tshw(trace[cfg.center_depth_key], name=cfg.center_depth_key)).astype(np.float32)
    if mode_name == "positive":
        return _as_tshw(trace[cfg.positive_mean_depth_key], name=cfg.positive_mean_depth_key)
    if mode_name == "max":
        return _as_tshw(trace[cfg.max_depth_key], name=cfg.max_depth_key)
    weight = max(0.0, min(1.0, float(blend_weight)))
    mean = _as_tshw(trace[cfg.mean_depth_key], name=cfg.mean_depth_key)
    if mode_name == "blend_mean_support_max":
        other = support * _as_tshw(trace[cfg.max_depth_key], name=cfg.max_depth_key)
    elif mode_name == "blend_mean_support_center":
        other = support * _as_tshw(trace[cfg.center_depth_key], name=cfg.center_depth_key)
    else:
        raise ValueError(f"unsupported area fraction mode: {mode}")
    return ((1.0 - weight) * mean + weight * other).astype(np.float32)


def _morph_mask(mask: np.ndarray, op: str, iterations: int, connectivity: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    mode = str(op).lower()
    count = max(0, int(iterations))
    if mode == "none" or count == 0:
        return out.copy()
    if mode not in {"erode", "dilate"}:
        raise ValueError("morphology operation must be 'none', 'erode', or 'dilate'")
    for _ in range(count):
        neighbors = _mask_neighbors(out, connectivity)
        if mode == "dilate":
            next_mask = np.zeros_like(out, dtype=bool)
            for neigh in neighbors:
                next_mask |= neigh
            out = next_mask
        else:
            next_mask = np.ones_like(out, dtype=bool)
            for neigh in neighbors:
                next_mask &= neigh
            out = next_mask
    return out


def _spread_depth_for_mask(depth: np.ndarray, iterations: int, connectivity: int) -> np.ndarray:
    out = np.asarray(depth, dtype=np.float32)
    count = max(0, int(iterations))
    for _ in range(count):
        neighbors = _value_neighbors(out, connectivity)
        out = np.maximum.reduce(neighbors).astype(np.float32)
    return out


def _mask_neighbors(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=bool)
    padded = np.pad(arr, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=False)
    offsets = _neighbor_offsets(connectivity)
    return [padded[:, :, 1 + dr:1 + dr + arr.shape[-2], 1 + dc:1 + dc + arr.shape[-1]] for dr, dc in offsets]


def _value_neighbors(values: np.ndarray, connectivity: int) -> list[np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    padded = np.pad(arr, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
    offsets = _neighbor_offsets(connectivity)
    return [padded[:, :, 1 + dr:1 + dr + arr.shape[-2], 1 + dc:1 + dc + arr.shape[-1]] for dr, dc in offsets]


def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if int(connectivity) == 4:
        return ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    return (
        (0, 0),
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )


def _as_tshw(value: Any, *, name: str) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 2:
        return arr[None, None, :, :]
    raise ValueError(f"{name} must have shape (T,S,H,W), (T,H,W), or (H,W); got {arr.shape}")


def _as_dense_reference_tshw(value: Any, *, name: str) -> np.ndarray:
    try:
        return _as_tshw(value, name=name)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a dense reference map with shape (T,S,H,W), (T,H,W), or (H,W); "
            "raw per-sample tensors are diagnostic-only"
        ) from exc


def _config_dict(cfg: PressureDeadbandFitConfig) -> dict[str, Any]:
    return {key: getattr(cfg, key) for key in cfg.__dataclass_fields__}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _finite_or(value: Any, default: float) -> float:
    out = _float_or_none(value)
    return float(default) if out is None else out


def _mean_optional_ints(values: Any) -> float:
    if not isinstance(values, list):
        return 0.0
    finite = [float(v) for v in values if v is not None]
    return float(np.mean(finite)) if finite else 0.0


def _npz_safe(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.dtype != object
