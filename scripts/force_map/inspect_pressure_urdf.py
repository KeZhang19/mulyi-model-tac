#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from BrainCo_DexHand.force_map import (  # noqa: E402
    load_pressure_pad_specs_from_urdf,
    load_pressure_touch_links_from_urdf,
    load_touch_links_from_urdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect pressure_pad metadata declared on existing hand toucher/touch links."
    )
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument(
        "--require-pressure-layouts",
        action="store_true",
        help="Exit 2 if pressure-capable toucher links lack explicit pressure_pad layout metadata.",
    )
    parser.add_argument(
        "--check-layout-files",
        action="store_true",
        help="Load points_npy/normals_npy files and validate their shape/count using the taxel-map loader.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = load_pressure_pad_specs_from_urdf(args.urdf, base_dir=args.base_dir, require_files=False)
    pressure_touch_links = load_pressure_touch_links_from_urdf(args.urdf)
    spec_link_counts = Counter(spec.link_name for spec in specs)
    spec_links = {spec.link_name for spec in specs}
    missing_pad_links = [link for link in pressure_touch_links if link not in spec_links]
    duplicate_pad_links = sorted(link for link, count in spec_link_counts.items() if count > 1)
    fingertip_pad_links = sorted(link for link in spec_links if _is_fingertip_link(link))
    extra_pad_links = sorted(link for link in spec_links if link not in set(pressure_touch_links))
    non_surface_origin_pads = [
        {"link_name": spec.link_name, "origin_semantics": spec.origin_semantics}
        for spec in specs
        if spec.origin_semantics != "pad_surface"
    ]
    missing_layout_links = [
        spec.link_name
        for spec in specs
        if not ((spec.points_npy is not None and spec.normals_npy is not None) or _has_grid_layout(spec))
    ]
    taxel_count_mismatches = [
        {
            "link_name": spec.link_name,
            "taxel_count": spec.taxel_count,
            "layout_count": int(spec.num_rows) * int(spec.num_cols),
        }
        for spec in specs
        if _taxel_count_mismatch(spec)
    ]
    invalid_grid_layouts = []
    for spec in specs:
        error = _grid_taxel_map_error(spec)
        if error is not None:
            invalid_grid_layouts.append(
                {
                    "link_name": spec.link_name,
                    "rows": spec.num_rows,
                    "cols": spec.num_cols,
                    "point_distance": spec.point_distance,
                    "row_distance": spec.row_distance,
                    "col_distance": spec.col_distance,
                    "pad_size": list(spec.pad_size) if spec.pad_size is not None else None,
                    "pad_size_semantics": spec.pad_size_semantics,
                    "origin_xyz": list(spec.origin_xyz),
                    "origin_rpy": list(spec.origin_rpy),
                    "normal_axis": spec.normal_axis,
                    "error": error,
                }
            )
    invalid_file_layouts = []
    if args.check_layout_files:
        for spec in specs:
            error = _file_taxel_map_error(spec)
            if error is not None:
                invalid_file_layouts.append(
                    {
                        "link_name": spec.link_name,
                        "points_npy": str(spec.points_npy) if spec.points_npy is not None else None,
                        "normals_npy": str(spec.normals_npy) if spec.normals_npy is not None else None,
                        "error": error,
                    }
                )
    passed = (
        not missing_pad_links
        and not missing_layout_links
        and not duplicate_pad_links
        and not fingertip_pad_links
        and not non_surface_origin_pads
        and not taxel_count_mismatches
        and not invalid_grid_layouts
        and not invalid_file_layouts
    )
    payload = {
        "urdf": str(args.urdf),
        "touch_links": load_touch_links_from_urdf(args.urdf),
        "pressure_touch_links": pressure_touch_links,
        "pressure_pad_count": len(specs),
        "pressure_pad_links": sorted(spec_links),
        "pressure_pad_links_not_in_pressure_touch_candidates": extra_pad_links,
        "duplicate_pressure_pad_links": duplicate_pad_links,
        "fingertip_pressure_pad_links": fingertip_pad_links,
        "pressure_pads_non_surface_origin": non_surface_origin_pads,
        "missing_pressure_pad_links": missing_pad_links,
        "pressure_pads_missing_layout": missing_layout_links,
        "pressure_pads_taxel_count_mismatch": taxel_count_mismatches,
        "pressure_pads_invalid_grid_layout": invalid_grid_layouts,
        "pressure_pads_invalid_file_layout": invalid_file_layouts,
        "passed": passed,
        "pressure_pads": [
            {
                "name": spec.name,
                "link_name": spec.link_name,
                "rows": spec.num_rows,
                "cols": spec.num_cols,
                "taxel_count": spec.taxel_count,
                "origin_semantics": spec.origin_semantics,
                "point_distance": spec.point_distance,
                "row_distance": spec.row_distance,
                "col_distance": spec.col_distance,
                "pad_size": list(spec.pad_size) if spec.pad_size is not None else None,
                "pad_size_semantics": spec.pad_size_semantics,
                "origin_xyz": list(spec.origin_xyz),
                "origin_rpy": list(spec.origin_rpy),
                "normal_sign": spec.normal_sign,
                "points_npy": str(spec.points_npy) if spec.points_npy is not None else None,
                "normals_npy": str(spec.normals_npy) if spec.normals_npy is not None else None,
                "calibration": {
                    "gain": spec.calibration.gain,
                    "bias": spec.calibration.bias,
                    "stiffness": spec.calibration.stiffness,
                    "damping": spec.calibration.damping,
                    "max_force": spec.calibration.max_force,
                    "gamma": spec.calibration.gamma,
                    "threshold": spec.calibration.threshold,
                    "area": spec.calibration.area,
                },
            }
            for spec in specs
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_pressure_layouts and not passed:
        return 2
    return 0


def _has_grid_layout(spec) -> bool:
    return (
        spec.num_rows is not None
        and spec.num_cols is not None
        and (spec.point_distance is not None or (spec.row_distance is not None and spec.col_distance is not None))
    )


def _taxel_count_mismatch(spec) -> bool:
    return (
        spec.taxel_count is not None
        and spec.num_rows is not None
        and spec.num_cols is not None
        and int(spec.taxel_count) != int(spec.num_rows) * int(spec.num_cols)
    )


def _grid_taxel_map_error(spec) -> str | None:
    if not _has_grid_layout(spec) or _taxel_count_mismatch(spec):
        return None
    try:
        spec.to_taxel_map()
    except ValueError as exc:
        return str(exc)
    return None


def _file_taxel_map_error(spec) -> str | None:
    if spec.points_npy is None or spec.normals_npy is None:
        return None
    try:
        spec.to_taxel_map()
    except (OSError, ValueError, IndexError) as exc:
        return str(exc)
    return None


def _is_fingertip_link(link_name: str) -> bool:
    name = str(link_name).lower()
    return "fingertip" in name or "_tip" in name or "tip_" in name


if __name__ == "__main__":
    raise SystemExit(main())
