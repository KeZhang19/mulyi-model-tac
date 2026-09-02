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
    AnalyticPressTrajectory,
    AnalyticPresserSpec,
    PressureCalibration,
    PressureTaxelMap,
    generate_l0_pressure_trace,
    load_pressure_calibration_overrides,
    pressure_calibration_value_summary,
)


def _uv_pair(text: str) -> tuple[float, float]:
    parts = [float(v) for v in text.replace(",", " ").split()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two floats, e.g. '0.0,0.001'")
    return (parts[0], parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate L0 analytic pressure-map ground-truth traces.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "pressure_l0")
    parser.add_argument("--run-id", default="l0_pressure")
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--cols", type=int, default=30)
    parser.add_argument("--point-distance", type=float, default=0.001)
    parser.add_argument("--normal-axis", type=int, default=2)
    parser.add_argument("--link-name", default="l0_finger_tip")
    parser.add_argument("--presser", choices=("square", "rectangle", "cylinder", "sphere"), default="square")
    parser.add_argument("--size", type=float, default=0.004, help="Square side, cylinder diameter, or default sphere diameter in meters.")
    parser.add_argument("--radius", type=float, default=None, help="Sphere radius or cylinder radius in meters.")
    parser.add_argument("--width", type=float, default=None, help="Rectangle/square width in meters.")
    parser.add_argument("--height", type=float, default=None, help="Rectangle/square height in meters.")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--indent-start", type=float, default=0.0, help="Start indentation in meters.")
    parser.add_argument("--indent-end", type=float, default=0.002, help="End indentation in meters.")
    parser.add_argument("--center-start", type=_uv_pair, default=(0.0, 0.0))
    parser.add_argument("--center-end", type=_uv_pair, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--pressure-calib", type=Path, default=None)
    parser.add_argument("--stiffness", type=float, default=5000.0)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--max-force", type=float, default=10.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--taxel-area", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_shape = (int(args.rows), int(args.cols))
    overrides = load_pressure_calibration_overrides(args.pressure_calib, image_shape=image_shape)

    def value(name: str, default):
        return overrides.get(name, default)

    calibration = PressureCalibration(
        gain=value("gain", float(args.gain)),
        bias=value("bias", float(args.bias)),
        stiffness=value("stiffness", float(args.stiffness)),
        damping=value("damping", float(args.damping)),
        max_force=value("max_force", float(args.max_force)),
        gamma=value("gamma", float(args.gamma)),
        threshold=value("threshold", float(args.threshold)),
        area=value("taxel_area", float(args.taxel_area)),
    )
    taxel_map = PressureTaxelMap.from_grid(
        link_name=str(args.link_name),
        num_rows=int(args.rows),
        num_cols=int(args.cols),
        point_distance=float(args.point_distance),
        normal_axis=int(args.normal_axis),
        calibration=calibration,
    )
    presser = AnalyticPresserSpec(
        kind=str(args.presser),
        size_m=float(args.size),
        radius_m=float(args.radius) if args.radius is not None else None,
        width_m=float(args.width) if args.width is not None else None,
        height_m=float(args.height) if args.height is not None else None,
    )
    trajectory = AnalyticPressTrajectory(
        steps=int(args.steps),
        indentation_start_m=float(args.indent_start),
        indentation_end_m=float(args.indent_end),
        center_start_uv_m=tuple(float(v) for v in args.center_start),
        center_end_uv_m=tuple(float(v) for v in args.center_end) if args.center_end is not None else None,
        dt_s=float(args.dt),
    )
    trace = generate_l0_pressure_trace(
        taxel_map,
        presser,
        trajectory,
        metadata={
            "pressure_calib": str(args.pressure_calib or ""),
            "pressure_calib_overrides": {key: pressure_calibration_value_summary(value) for key, value in overrides.items()},
        },
    )
    npz_path, metadata_path = trace.save(args.out_dir, run_id=str(args.run_id))
    print(json.dumps({"trace": str(npz_path), "metadata": str(metadata_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
