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
    PressureCalibration,
    write_normal_ray_reference_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a normal-ray pressure trace from an aligned deformation/reference map."
    )
    parser.add_argument("trace", type=Path, help="Input pressure trace .npz.")
    parser.add_argument("--out-trace", type=Path, default=None, help="Output .npz path. Defaults to <trace>.normal_ray.npz.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--deformation-key", default="tacmap_raw_aligned_m")
    parser.add_argument("--output-prefix", default="normal_ray")
    parser.add_argument("--contact-deadband-m", type=float, default=0.0)
    parser.add_argument("--stiffness", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--max-force", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--taxel-area", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_trace = args.out_trace
    if out_trace is None:
        out_trace = args.trace.with_name(args.trace.stem + ".normal_ray.npz")

    calibration = PressureCalibration(
        gain=float(args.gain),
        bias=float(args.bias),
        stiffness=float(args.stiffness),
        damping=float(args.damping),
        max_force=float(args.max_force),
        gamma=float(args.gamma),
        threshold=float(args.threshold),
        area=float(args.taxel_area),
    )
    path, summary = write_normal_ray_reference_trace(
        args.trace,
        out_trace,
        deformation_key=str(args.deformation_key),
        output_prefix=str(args.output_prefix),
        calibration=calibration,
        contact_deadband_m=float(args.contact_deadband_m),
    )
    prefix = str(args.output_prefix).strip() or "normal_ray"
    summary.update(
        {
            "trace": str(args.trace),
            "out_trace": str(path),
            "calibration": {
                "gain": float(args.gain),
                "bias": float(args.bias),
                "stiffness": float(args.stiffness),
                "damping": float(args.damping),
                "max_force": float(args.max_force),
                "gamma": float(args.gamma),
                "threshold": float(args.threshold),
                "taxel_area": float(args.taxel_area),
            },
            "verify_command": [
                "python3",
                "scripts/force_map/verify_pressure_trace.py",
                str(path),
                "--pressure-key",
                f"{prefix}_pressure_norm",
                "--raw-pressure-key",
                f"{prefix}_pressure_raw_n",
                "--penetration-key",
                f"{prefix}_penetration_m",
                "--reference-key",
                str(args.deformation_key),
                "--reference-layer",
                "L1_model_reference",
                "--reference-iou-threshold",
                "1",
                "--centroid-error-threshold-px",
                "0",
                "--bbox-error-threshold-px",
                "0",
                "--depth-rmse-threshold-m",
                "0",
                "--onset-error-threshold-frames",
                "0",
                "--offset-error-threshold-frames",
                "0",
                "--fail-on-threshold",
            ],
        }
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
