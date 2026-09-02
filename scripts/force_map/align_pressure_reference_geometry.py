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

from BrainCo_DexHand.force_map import write_geometry_aligned_trace  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample a dense reference map, such as TacMap link_surface, onto pressure taxel geometry."
    )
    parser.add_argument("trace", type=Path, help="Input pressure trace .npz containing layout point grids.")
    parser.add_argument("--out-trace", type=Path, default=None, help="Output .npz path. Defaults to <trace>.geom_aligned.npz.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional alignment summary JSON path.")
    parser.add_argument("--reference-key", default="tacmap_raw_m")
    parser.add_argument("--output-key", default="tacmap_raw_aligned_m")
    parser.add_argument("--pressure-points-key", default="pressure_taxel_points_l_m")
    parser.add_argument("--reference-points-key", default="tacmap_grid_points_l_m")
    parser.add_argument("--reference-axes-key", default="tacmap_grid_axes_l")
    parser.add_argument("--pressure-valid-key", default="pressure_taxel_layout_valid")
    parser.add_argument("--reference-valid-key", default="tacmap_grid_layout_valid")
    parser.add_argument("--distance-mode", choices=("plane", "local3d"), default="plane")
    parser.add_argument("--max-distance-m", type=float, default=None, help="Optional nearest-neighbor distance cutoff.")
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_trace = args.out_trace
    if out_trace is None:
        out_trace = args.trace.with_name(args.trace.stem + ".geom_aligned.npz")

    path, summary = write_geometry_aligned_trace(
        args.trace,
        out_trace,
        reference_key=str(args.reference_key),
        output_key=str(args.output_key),
        pressure_points_key=str(args.pressure_points_key),
        reference_points_key=str(args.reference_points_key),
        reference_axes_key=str(args.reference_axes_key),
        pressure_valid_key=str(args.pressure_valid_key),
        distance_mode=str(args.distance_mode),
        reference_valid_key=str(args.reference_valid_key),
        max_distance_m=args.max_distance_m,
        chunk_size=int(args.chunk_size),
    )
    summary["trace"] = str(args.trace)
    summary["out_trace"] = str(path)
    summary["verify_command"] = [
        "python3",
        "scripts/force_map/verify_pressure_trace.py",
        str(path),
        "--reference-key",
        str(args.output_key),
        "--reference-valid-mask-key",
        f"{args.output_key}_valid_mask",
    ]
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
