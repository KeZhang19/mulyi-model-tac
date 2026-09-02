#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from BrainCo_DexHand.force_map import (  # noqa: E402
    PressureAreaFractionFitConfig,
    fit_pressure_area_fraction_to_reference,
    write_area_fraction_fit_trace,
)
from pressure_fit_verify_command import pressure_fit_verify_command  # noqa: E402


def _float_list(text: str) -> list[float]:
    return [float(part) for part in text.replace(",", " ").split() if part]


def _str_list(text: str) -> list[str]:
    return [part.strip() for part in text.replace(",", " ").split() if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit sampled area-fraction/depth integration against a dense model reference."
    )
    parser.add_argument("trace", type=Path, help="Pressure trace .npz containing sample support/depth fields.")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--fit-trace-out", type=Path, default=None)
    parser.add_argument("--pressure-key", default="pressure_norm")
    parser.add_argument("--raw-pressure-key", default="pressure_raw_n")
    parser.add_argument("--penetration-key", default="penetration_m")
    parser.add_argument("--reference-key", default="tacmap_raw_m")
    parser.add_argument("--support-key", default="geometry_normal_ray_sample_support_fraction")
    parser.add_argument("--mean-depth-key", default="geometry_normal_ray_sample_mean_penetration_m")
    parser.add_argument("--positive-mean-depth-key", default="geometry_normal_ray_sample_positive_mean_penetration_m")
    parser.add_argument("--max-depth-key", default="geometry_normal_ray_sample_max_penetration_m")
    parser.add_argument("--center-depth-key", default="geometry_normal_ray_penetration_m")
    parser.add_argument("--reference-layer", default=None)
    parser.add_argument("--reference-valid-mask-key", default=None)
    parser.add_argument("--active-threshold", type=float, default=1.0e-6)
    parser.add_argument("--reference-threshold", type=float, default=0.0)
    parser.add_argument("--penetration-threshold", type=float, default=0.0)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--max-mask-deadband-m", type=float, default=None)
    parser.add_argument("--max-depth-deadband-m", type=float, default=None)
    parser.add_argument("--area-modes", type=_str_list, default=None)
    parser.add_argument("--blend-weights", type=_float_list, default=None)
    parser.add_argument("--support-thresholds", type=_float_list, default=None)
    parser.add_argument("--mask-deadbands-m", type=_float_list, default=None)
    parser.add_argument("--depth-deadbands-m", type=_float_list, default=None)
    parser.add_argument("--active-floors-m", type=_float_list, default=None)
    parser.add_argument("--scale-candidates", type=_float_list, default=None)
    parser.add_argument("--scale-min", type=float, default=1.0)
    parser.add_argument("--scale-max", type=float, default=1.0)
    parser.add_argument("--scale-count", type=int, default=1)
    parser.add_argument("--reference-iou-threshold", type=float, default=0.95)
    parser.add_argument("--centroid-error-threshold-px", type=float, default=1.5)
    parser.add_argument("--bbox-error-threshold-px", type=float, default=2.0)
    parser.add_argument("--depth-rmse-threshold-m", type=float, default=2.0e-5)
    parser.add_argument("--onset-error-threshold-frames", type=int, default=1)
    parser.add_argument("--offset-error-threshold-frames", type=int, default=1)
    parser.add_argument("--print-candidates", action="store_true", help="Print full candidate table to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PressureAreaFractionFitConfig(
        pressure_key=str(args.pressure_key),
        raw_pressure_key=str(args.raw_pressure_key),
        penetration_key=str(args.penetration_key),
        reference_key=str(args.reference_key),
        support_key=str(args.support_key),
        mean_depth_key=str(args.mean_depth_key),
        positive_mean_depth_key=str(args.positive_mean_depth_key),
        max_depth_key=str(args.max_depth_key),
        center_depth_key=str(args.center_depth_key),
        reference_layer=args.reference_layer,
        reference_valid_mask_key=args.reference_valid_mask_key,
        active_threshold=float(args.active_threshold),
        reference_threshold=float(args.reference_threshold),
        penetration_threshold=float(args.penetration_threshold),
        candidate_count=int(args.candidate_count),
        max_mask_deadband_m=args.max_mask_deadband_m,
        max_depth_deadband_m=args.max_depth_deadband_m,
        reference_iou_threshold=float(args.reference_iou_threshold),
        centroid_error_threshold_px=float(args.centroid_error_threshold_px),
        bbox_error_threshold_px=float(args.bbox_error_threshold_px),
        depth_rmse_threshold_m=float(args.depth_rmse_threshold_m),
        onset_error_threshold_frames=int(args.onset_error_threshold_frames),
        offset_error_threshold_frames=int(args.offset_error_threshold_frames),
    )
    scales = args.scale_candidates
    if scales is None and int(args.scale_count) > 1:
        import numpy as np

        scales = np.linspace(float(args.scale_min), float(args.scale_max), int(args.scale_count)).tolist()
    result = fit_pressure_area_fraction_to_reference(
        args.trace,
        config=cfg,
        area_modes=args.area_modes,
        blend_weights=args.blend_weights,
        support_thresholds=args.support_thresholds,
        mask_deadbands_m=args.mask_deadbands_m,
        depth_deadbands_m=args.depth_deadbands_m,
        scales=scales,
        active_floor_m=args.active_floors_m,
    )
    if args.fit_trace_out is not None:
        best = result["best"]
        fit_path = write_area_fraction_fit_trace(
            args.trace,
            args.fit_trace_out,
            area_mode=str(best["area_mode"]),
            blend_weight=float(best.get("blend_weight", 0.0)),
            support_threshold=float(best["support_threshold"]),
            mask_deadband_m=float(best["mask_deadband_m"]),
            depth_deadband_m=float(best["depth_deadband_m"]),
            scale=float(best.get("scale", 1.0)),
            active_floor_m=float(best.get("active_floor_m", 0.0)),
            config=cfg,
        )
        result["fit_trace"] = str(fit_path)
        result["fit_verify_command"] = pressure_fit_verify_command(
            fit_path,
            pressure_key="pressure_norm_area_fraction_fit",
            raw_pressure_key="pressure_raw_area_fraction_fit_n",
            penetration_key="penetration_area_fraction_fit_m",
            reference_key=str(args.reference_key),
            args=args,
        )
    printable = result if args.print_candidates else {key: value for key, value in result.items() if key != "candidates"}
    text = json.dumps(printable, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
