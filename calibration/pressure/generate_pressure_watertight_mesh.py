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

from BrainCo_DexHand.force_map import triangle_mesh_topology_diagnostics  # noqa: E402


def _vec3(text: str) -> tuple[float, float, float]:
    parts = [float(value) for value in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three floats, e.g. '0,0,0.009'")
    return (parts[0], parts[1], parts[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple watertight pressure-reference meshes.")
    parser.add_argument("--kind", choices=("box", "cylinder"), required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--center", type=_vec3, default=(0.0, 0.0, 0.009))
    parser.add_argument("--half-extents", type=_vec3, default=(0.002, 0.002, 0.001))
    parser.add_argument("--radius", type=float, default=0.002)
    parser.add_argument("--height", type=float, default=0.002)
    parser.add_argument("--segments", type=int, default=32)
    return parser.parse_args()


def box_mesh(*, center: tuple[float, float, float], half_extents: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = (float(value) for value in center)
    hx, hy, hz = (float(value) for value in half_extents)
    vertices = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, triangles


def cylinder_mesh(*, center: tuple[float, float, float], radius: float, height: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    if float(radius) <= 0.0:
        raise ValueError("radius must be positive")
    if float(height) <= 0.0:
        raise ValueError("height must be positive")
    n = max(3, int(segments))
    cx, cy, cz = (float(value) for value in center)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * float(radius)
    bottom_z = cz - 0.5 * float(height)
    top_z = cz + 0.5 * float(height)
    bottom = np.column_stack([cx + xy[:, 0], cy + xy[:, 1], np.full(n, bottom_z, dtype=np.float32)])
    top = np.column_stack([cx + xy[:, 0], cy + xy[:, 1], np.full(n, top_z, dtype=np.float32)])
    vertices = np.vstack([bottom, top, np.array([[cx, cy, bottom_z], [cx, cy, top_z]], dtype=np.float32)]).astype(np.float32)
    bottom_center = 2 * n
    top_center = 2 * n + 1
    faces: list[list[int]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
        faces.append([bottom_center, j, i])
        faces.append([top_center, n + i, n + j])
    return vertices, np.asarray(faces, dtype=np.int64)


def main() -> int:
    args = parse_args()
    if args.kind == "box":
        vertices, triangles = box_mesh(center=args.center, half_extents=args.half_extents)
        params = {"center": list(args.center), "half_extents": list(args.half_extents)}
    else:
        vertices, triangles = cylinder_mesh(
            center=args.center,
            radius=float(args.radius),
            height=float(args.height),
            segments=int(args.segments),
        )
        params = {"center": list(args.center), "radius": float(args.radius), "height": float(args.height), "segments": max(3, int(args.segments))}

    prefix = args.out_prefix.expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    vertices_path = prefix.with_suffix(".vertices.npy")
    triangles_path = prefix.with_suffix(".triangles.npy")
    summary_path = prefix.with_suffix(".json")
    np.save(vertices_path, vertices)
    np.save(triangles_path, triangles)
    summary = {
        "kind": str(args.kind),
        "params": params,
        "vertices_npy": str(vertices_path),
        "triangles_npy": str(triangles_path),
        "topology": triangle_mesh_topology_diagnostics(vertices, triangles),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
