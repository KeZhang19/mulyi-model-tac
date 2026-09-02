#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from BrainCo_DexHand.force_map import (  # noqa: E402
    PressureCalibration,
    write_geometry_normal_ray_trace,
)


def _vec3(text: str) -> tuple[float, float, float]:
    parts = [float(v) for v in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three floats, e.g. '0,0,0.01'")
    return (parts[0], parts[1], parts[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a geometry-normal-ray pressure trace from explicit object geometry."
    )
    parser.add_argument("trace", type=Path, help="Input pressure trace/layout .npz.")
    parser.add_argument("--out-trace", type=Path, default=None, help="Output .npz path. Defaults to <trace>.geometry_normal_ray.npz.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--output-prefix", default="geometry_normal_ray")
    parser.add_argument("--geometry", choices=("sphere", "box", "mesh"), required=True)
    parser.add_argument("--center", type=_vec3, default=None, help="Object center in pressure-link local meters.")
    parser.add_argument("--center-npy", type=Path, default=None, help="Optional center trajectory .npy with shape (T,3).")
    parser.add_argument("--radius", type=float, default=None, help="Sphere radius in meters.")
    parser.add_argument("--half-extents", type=_vec3, default=None, help="Box half extents in pressure-link local meters.")
    parser.add_argument("--vertices-npy", type=Path, default=None, help="Mesh vertices .npy, shape (V,3) or triangle vertices (T,3,3).")
    parser.add_argument("--triangles-npy", type=Path, default=None, help="Mesh triangle indices .npy, shape (T,3).")
    parser.add_argument("--pressure-points-key", default="pressure_taxel_points_l_m")
    parser.add_argument("--pressure-normals-key", default="pressure_taxel_normals_l")
    parser.add_argument("--rest-distance-m", type=float, required=True)
    parser.add_argument("--contact-deadband-m", type=float, default=0.0)
    parser.add_argument("--max-distance-m", type=float, default=None)
    parser.add_argument("--invert-ray-directions", action="store_true")
    parser.add_argument("--stiffness", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--max-force", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--taxel-area", type=float, default=1.0)
    return parser.parse_args()


def _geometry_from_args(args: argparse.Namespace) -> dict[str, object]:
    geometry: dict[str, object] = {"kind": str(args.geometry)}
    if args.geometry in {"sphere", "box"}:
        if args.center_npy is not None:
            geometry["center_l_m"] = np.load(args.center_npy).astype(np.float32)
        elif args.center is not None:
            geometry["center_l_m"] = np.asarray(args.center, dtype=np.float32)
        else:
            raise SystemExit("--center or --center-npy is required for sphere/box geometry")

    if args.geometry == "sphere":
        if args.radius is None:
            raise SystemExit("--radius is required for sphere geometry")
        geometry["radius_m"] = float(args.radius)
    elif args.geometry == "box":
        if args.half_extents is None:
            raise SystemExit("--half-extents is required for box geometry")
        geometry["half_extents_l_m"] = np.asarray(args.half_extents, dtype=np.float32)
    elif args.geometry == "mesh":
        if args.vertices_npy is None:
            raise SystemExit("--vertices-npy is required for mesh geometry")
        geometry["vertices_l_m"] = np.load(args.vertices_npy).astype(np.float32)
        if args.triangles_npy is not None:
            geometry["triangles"] = np.load(args.triangles_npy).astype(np.int64)
    return geometry


def main() -> int:
    args = parse_args()
    out_trace = args.out_trace
    if out_trace is None:
        out_trace = args.trace.with_name(args.trace.stem + ".geometry_normal_ray.npz")

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
    geometry = _geometry_from_args(args)
    path, summary = write_geometry_normal_ray_trace(
        args.trace,
        out_trace,
        geometry=geometry,
        output_prefix=str(args.output_prefix),
        pressure_points_key=str(args.pressure_points_key),
        pressure_normals_key=str(args.pressure_normals_key),
        rest_distance_m=float(args.rest_distance_m),
        calibration=calibration,
        contact_deadband_m=float(args.contact_deadband_m),
        max_distance_m=args.max_distance_m,
        invert_ray_directions=bool(args.invert_ray_directions),
    )
    prefix = str(args.output_prefix).strip() or "geometry_normal_ray"
    summary.update(
        {
            "trace": str(args.trace),
            "out_trace": str(path),
            "geometry": _jsonable_geometry(geometry),
            "rest_distance_m": float(args.rest_distance_m),
            "contact_deadband_m": float(args.contact_deadband_m),
            "max_distance_m": None if args.max_distance_m is None else float(args.max_distance_m),
            "invert_ray_directions": bool(args.invert_ray_directions),
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


def _jsonable_geometry(geometry: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in geometry.items():
        if isinstance(value, np.ndarray):
            out[key] = {
                "shape": [int(v) for v in value.shape],
                "min": float(np.min(value)) if value.size else None,
                "max": float(np.max(value)) if value.size else None,
            }
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    raise SystemExit(main())
