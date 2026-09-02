#!/usr/bin/env python3
"""Render URDF pressure-pad taxel layouts as a simple SVG contact sheet."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from BrainCo_DexHand.force_map import load_pressure_pad_specs_from_urdf  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize URDF pressure-pad taxel grid layouts.")
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--link", action="append", default=None, help="Restrict output to one link. Can be repeated.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "pressure_pad_layout")
    parser.add_argument("--svg", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--require-files", action="store_true", help="Require declared points_npy/normals_npy files to exist.")
    parser.add_argument("--include-points", action="store_true", help="Include full point/normal arrays in JSON.")
    return parser.parse_args(argv)


def build_layout_report(
    urdf: Path,
    *,
    base_dir: Path | None = None,
    links: Sequence[str] | None = None,
    require_files: bool = False,
    include_points: bool = False,
) -> dict[str, Any]:
    specs = load_pressure_pad_specs_from_urdf(urdf, base_dir=base_dir, require_files=require_files)
    selected_links = set(str(link) for link in links or [])
    if selected_links:
        specs = [spec for spec in specs if spec.link_name in selected_links]
    pads = []
    for spec in specs:
        taxel_map = spec.to_taxel_map()
        points = np.asarray(taxel_map.points_l, dtype=np.float64)
        normals = np.asarray(taxel_map.normals_l, dtype=np.float64)
        rows, cols = taxel_map.image_shape
        projection_axes = projection_axes_for_points(points, normal_axis=int(spec.normal_axis))
        uv = points[:, list(projection_axes)].reshape(rows, cols, 2)
        pad = {
            "link_name": spec.link_name,
            "name": spec.map_name,
            "rows": int(rows),
            "cols": int(cols),
            "taxel_count": int(points.shape[0]),
            "origin_semantics": spec.origin_semantics,
            "point_distance_m": spec.point_distance,
            "row_distance_m": spec.row_distance,
            "col_distance_m": spec.col_distance,
            "pad_size_m": list(spec.pad_size) if spec.pad_size is not None else None,
            "pad_size_semantics": spec.pad_size_semantics,
            "origin_xyz_m": list(spec.origin_xyz),
            "origin_rpy_rad": list(spec.origin_rpy),
            "normal_axis": int(spec.normal_axis),
            "normal_sign": float(spec.normal_sign),
            "projection_axes": list(projection_axes),
            "taxel_uv_m": uv.tolist(),
            "bounds_l_m": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
            },
            "normal_mean_l": normals.mean(axis=0).tolist(),
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
        if include_points:
            pad["points_l_m"] = points.reshape(rows, cols, 3).tolist()
            pad["normals_l"] = normals.reshape(rows, cols, 3).tolist()
        pads.append(pad)
    missing_requested_links = sorted(selected_links - {pad["link_name"] for pad in pads})
    return {
        "passed": len(pads) > 0 and not missing_requested_links,
        "urdf": str(Path(urdf).expanduser().resolve()),
        "pad_count": len(pads),
        "requested_links": sorted(selected_links),
        "missing_requested_links": missing_requested_links,
        "pads": pads,
    }


def projection_axes_for_points(points: np.ndarray, *, normal_axis: int) -> tuple[int, int]:
    if normal_axis in (0, 1, 2):
        axes = [0, 1, 2]
        axes.remove(int(normal_axis))
        return int(axes[0]), int(axes[1])
    ranges = np.ptp(points, axis=0)
    order = np.argsort(ranges)[::-1]
    return int(order[0]), int(order[1])


def render_layout_svg(report: dict[str, Any]) -> str:
    pads = list(report.get("pads", []))
    panel_w = 280
    panel_h = 240
    margin = 24
    cols = 2 if len(pads) > 1 else 1
    rows = max(1, (len(pads) + cols - 1) // cols)
    width = cols * panel_w + 2 * margin
    height = rows * panel_h + 2 * margin + 28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#17202a} .small{font-size:10px;fill:#52616f}",
        ".title{font-size:15px;font-weight:700}.dot{fill:#00d5ff;stroke:#00586b;stroke-width:0.7}",
        ".frame{fill:#f8fafc;stroke:#9aa6b2;stroke-width:1}.axis{stroke:#6b7280;stroke-width:1}",
        "</style>",
        f'<text class="title" x="{margin}" y="22">Pressure pad taxel layout: {html.escape(Path(str(report.get("urdf", ""))).name)}</text>',
    ]
    for idx, pad in enumerate(pads):
        col = idx % cols
        row = idx // cols
        x0 = margin + col * panel_w
        y0 = margin + 28 + row * panel_h
        parts.extend(render_pad_panel(pad, x0=x0, y0=y0, width=panel_w - 18, height=panel_h - 18))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_pad_panel(pad: dict[str, Any], *, x0: int, y0: int, width: int, height: int) -> list[str]:
    rows = int(pad["rows"])
    cols = int(pad["cols"])
    plot_x = x0 + 24
    plot_y = y0 + 54
    plot_w = width - 48
    plot_h = height - 78
    axes = tuple(int(axis) for axis in pad["projection_axes"])
    pitch_text = pitch_label(pad)
    points = taxel_uv_points(pad, rows=rows, cols=cols)
    min_u = min(point[0] for point in points)
    max_u = max(point[0] for point in points)
    min_v = min(point[1] for point in points)
    max_v = max(point[1] for point in points)
    pad_u = max((max_u - min_u) * 0.08, 1.0e-6)
    pad_v = max((max_v - min_v) * 0.08, 1.0e-6)
    min_u -= pad_u
    max_u += pad_u
    min_v -= pad_v
    max_v += pad_v

    out = [
        f'<rect class="frame" x="{x0}" y="{y0}" width="{width}" height="{height}" rx="6"/>',
        f'<text class="title" x="{x0 + 12}" y="{y0 + 22}">{html.escape(str(pad["link_name"]))}</text>',
        (
            f'<text class="small" x="{x0 + 12}" y="{y0 + 40}">'
            f'{rows}x{cols} taxels, origin={html.escape(str(pad["origin_semantics"]))}, '
            f'pitch={html.escape(pitch_text)}, axes={axes[0]}/{axes[1]}</text>'
        ),
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#c5ced8"/>',
    ]
    for r, c, u, v in points:
        x = plot_x + (u - min_u) / max(max_u - min_u, 1.0e-12) * plot_w
        y = plot_y + plot_h - (v - min_v) / max(max_v - min_v, 1.0e-12) * plot_h
        out.append(f'<circle class="dot" cx="{x:.2f}" cy="{y:.2f}" r="4"><title>row={r} col={c}</title></circle>')
    out.extend(
        [
            f'<line class="axis" x1="{plot_x}" y1="{plot_y + plot_h + 10}" x2="{plot_x + 32}" y2="{plot_y + plot_h + 10}"/>',
            f'<text class="small" x="{plot_x + 38}" y="{plot_y + plot_h + 14}">local axis {axes[0]}</text>',
            f'<line class="axis" x1="{plot_x}" y1="{plot_y + plot_h + 10}" x2="{plot_x}" y2="{plot_y + plot_h - 22}"/>',
            f'<text class="small" x="{plot_x + 6}" y="{plot_y + plot_h - 26}">axis {axes[1]}</text>',
        ]
    )
    return out


def pitch_label(pad: dict[str, Any]) -> str:
    row_distance = pad.get("row_distance_m")
    col_distance = pad.get("col_distance_m")
    if row_distance is None or col_distance is None:
        scalar = pad.get("point_distance_m")
        return "unknown" if scalar is None else f"{float(scalar) * 1000.0:.3g}mm"
    return f"{float(row_distance) * 1000.0:.3g}/{float(col_distance) * 1000.0:.3g}mm"


def taxel_uv_points(pad: dict[str, Any], *, rows: int, cols: int) -> list[tuple[int, int, float, float]]:
    uv = pad.get("taxel_uv_m")
    if uv is None:
        return []
    return [
        (r, c, float(uv[r][c][0]), float(uv[r][c][1]))
        for r in range(rows)
        for c in range(cols)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = Path(args.svg).expanduser().resolve() if args.svg is not None else out_dir / "pressure_pad_layout.svg"
    json_path = Path(args.json).expanduser().resolve() if args.json is not None else out_dir / "pressure_pad_layout.json"
    report = build_layout_report(
        Path(args.urdf),
        base_dir=args.base_dir,
        links=args.link,
        require_files=bool(args.require_files),
        include_points=bool(args.include_points),
    )
    report["svg"] = str(svg_path)
    report["json"] = str(json_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(render_layout_svg(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
