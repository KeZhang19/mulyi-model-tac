from __future__ import annotations

from pathlib import Path
from typing import Any


def pressure_fit_verify_command(
    fit_path: str | Path,
    *,
    pressure_key: str,
    raw_pressure_key: str,
    penetration_key: str,
    reference_key: str,
    args: Any,
) -> list[str]:
    cmd = [
        "python3",
        "scripts/force_map/verify_pressure_trace.py",
        str(fit_path),
        "--pressure-key",
        str(pressure_key),
        "--raw-pressure-key",
        str(raw_pressure_key),
        "--penetration-key",
        str(penetration_key),
        "--reference-key",
        str(reference_key),
    ]
    if getattr(args, "reference_layer", None) is not None:
        cmd.extend(["--reference-layer", str(args.reference_layer)])
    if getattr(args, "reference_valid_mask_key", None) is not None:
        cmd.extend(["--reference-valid-mask-key", str(args.reference_valid_mask_key)])
    cmd.extend(
        [
            "--active-threshold",
            f"{float(args.active_threshold):.9g}",
            "--penetration-threshold",
            f"{float(args.penetration_threshold):.9g}",
            "--reference-threshold",
            f"{float(args.reference_threshold):.9g}",
            "--reference-iou-threshold",
            f"{float(args.reference_iou_threshold):.9g}",
            "--centroid-error-threshold-px",
            f"{float(args.centroid_error_threshold_px):.9g}",
            "--bbox-error-threshold-px",
            f"{float(args.bbox_error_threshold_px):.9g}",
            "--depth-rmse-threshold-m",
            f"{float(args.depth_rmse_threshold_m):.9g}",
            "--onset-error-threshold-frames",
            str(int(args.onset_error_threshold_frames)),
            "--offset-error-threshold-frames",
            str(int(args.offset_error_threshold_frames)),
            "--fail-on-threshold",
        ]
    )
    return cmd
