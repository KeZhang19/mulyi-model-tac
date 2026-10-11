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
    PressureDeadbandFitConfig,
    fit_pressure_deadband_to_reference,
    write_deadband_fit_trace,
)
from pressure_fit_verify_command import pressure_fit_verify_command  # noqa: E402


def _float_list(text: str) -> list[float]:
    return [float(part) for part in text.replace(",", " ").split() if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit SDF pressure penetration deadband against a dense L1 model reference map, not physical GT."
    )
    parser.add_argument("trace", type=Path, help="Pressure trace .npz containing penetration_m and a reference map.")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--fit-trace-out", type=Path, default=None, help="Optional .npz with penetration_fit_m/pressure_norm_fit arrays.")
    parser.add_argument("--pressure-key", default="pressure_norm")
    parser.add_argument("--raw-pressure-key", default="pressure_raw_n")
    parser.add_argument("--penetration-key", default="penetration_m")
    parser.add_argument("--reference-key", default="tacmap_raw_m")
    parser.add_argument(
        "--reference-layer",
        default=None,
        help="Optional validation layer metadata. Defaults to inference from --reference-key.",
    )
    parser.add_argument("--reference-valid-mask-key", default=None)
    parser.add_argument("--active-threshold", type=float, default=1.0e-6)
    parser.add_argument("--reference-threshold", type=float, default=0.0)
    parser.add_argument("--penetration-threshold", type=float, default=0.0)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--max-deadband-m", type=float, default=None)
    parser.add_argument("--candidates-m", type=_float_list, default=None, help="Explicit deadband candidates, e.g. '0,0.0005,0.001'.")
    parser.add_argument("--scale-candidates", type=_float_list, default=None, help="Explicit positive depth scales, e.g. '0.5,1.0,1.5'.")
    parser.add_argument("--scale-min", type=float, default=1.0)
    parser.add_argument("--scale-max", type=float, default=1.0)
    parser.add_argument("--scale-count", type=int, default=1)
    parser.add_argument(
        "--suggested-arg-name",
        default="--penetration-deadband",
        help="CLI arg name to report for the fitted deadband, e.g. --geometry-normal-ray-deadband.",
    )
    parser.add_argument("--reference-iou-threshold", type=float, default=0.95)
    parser.add_argument("--centroid-error-threshold-px", type=float, default=1.5)
    parser.add_argument("--bbox-error-threshold-px", type=float, default=2.0)
    parser.add_argument("--depth-rmse-threshold-m", type=float, default=2.0e-5)
    parser.add_argument("--onset-error-threshold-frames", type=int, default=1)
    parser.add_argument("--offset-error-threshold-frames", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PressureDeadbandFitConfig(
        pressure_key=str(args.pressure_key),
        raw_pressure_key=str(args.raw_pressure_key),
        penetration_key=str(args.penetration_key),
        reference_key=str(args.reference_key),
        reference_layer=args.reference_layer,
        reference_valid_mask_key=args.reference_valid_mask_key,
        active_threshold=float(args.active_threshold),
        reference_threshold=float(args.reference_threshold),
        penetration_threshold=float(args.penetration_threshold),
        candidate_count=int(args.candidate_count),
        max_deadband_m=args.max_deadband_m,
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
    result = fit_pressure_deadband_to_reference(args.trace, config=cfg, candidates_m=args.candidates_m, scales=scales)
    result["suggested_args"] = [str(args.suggested_arg_name), f"{float(result['best']['deadband_m']):.9g}"]
    if args.fit_trace_out is not None:
        fit_path = write_deadband_fit_trace(
            args.trace,
            args.fit_trace_out,
            deadband_m=float(result["best"]["deadband_m"]),
            scale=float(result["best"].get("scale", 1.0)),
            config=cfg,
        )
        result["fit_trace"] = str(fit_path)
        result["fit_verify_command"] = pressure_fit_verify_command(
            fit_path,
            pressure_key="pressure_norm_fit",
            raw_pressure_key="pressure_raw_fit_n",
            penetration_key="penetration_fit_m",
            reference_key=str(args.reference_key),
            args=args,
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
