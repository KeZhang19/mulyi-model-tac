#!/usr/bin/env python3
"""Generate TacMap point/normal maps from a USD link mesh.

Run this in an Isaac Sim / Isaac Lab Python environment with `pxr` and `numpy`.
The generated point map is saved in millimeters by default to match
SharpaTacmapCfg.correction_scale=1e-3.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np

try:
    from pxr import Gf, Usd, UsdGeom
except ModuleNotFoundError:
    Gf = None
    Usd = None
    UsdGeom = None


AXES = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a regular TacMap grid on a USD link mesh by ray projection."
    )
    parser.add_argument("--usd", default=None, help="Path to the robot USD file.")
    parser.add_argument(
        "--stl",
        default=None,
        help="Path to a mesh STL file. Use this when USD Python bindings are unavailable.",
    )
    parser.add_argument(
        "--link",
        default=None,
        help="Link prim path or prim name, e.g. /World/Robot/right_thumb_DIP_Link or right_thumb_DIP_Link.",
    )
    parser.add_argument(
        "--mesh-regex",
        default=None,
        help="Optional regex to keep only matching descendant mesh prim paths.",
    )
    parser.add_argument(
        "--ray-dir",
        required=True,
        choices=sorted(AXES),
        help="Ray direction in link frame, from the virtual sensing plane into the mesh.",
    )
    parser.add_argument(
        "--u-axis",
        required=True,
        choices=["x", "y", "z"],
        help="Horizontal TacMap axis in link frame.",
    )
    parser.add_argument(
        "--v-axis",
        required=True,
        choices=["x", "y", "z"],
        help="Vertical TacMap axis in link frame.",
    )
    parser.add_argument("--height", type=int, default=240, help="TacMap rows.")
    parser.add_argument("--width", type=int, default=240, help="TacMap columns.")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.005,
        help="Distance from mesh bounds to the virtual sensing plane, in link units.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Padding added to auto u/v bounds, in link units.",
    )
    parser.add_argument(
        "--normal-min-dot",
        type=float,
        default=0.0,
        help=(
            "Keep only triangles whose outward normal faces the virtual sensing plane "
            "by at least this dot product. Try 0.35-0.75 to remove side/back walls."
        ),
    )
    parser.add_argument(
        "--normal-both-sides",
        action="store_true",
        help="Use absolute normal alignment for --normal-min-dot, keeping front and back faces.",
    )
    parser.add_argument("--u-min", type=float, default=None, help="Override u lower bound in link units.")
    parser.add_argument("--u-max", type=float, default=None, help="Override u upper bound in link units.")
    parser.add_argument("--v-min", type=float, default=None, help="Override v lower bound in link units.")
    parser.add_argument("--v-max", type=float, default=None, help="Override v upper bound in link units.")
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix. Writes <prefix>_point.npy and <prefix>_normal.npy.",
    )
    parser.add_argument(
        "--point-scale",
        type=float,
        default=1000.0,
        help="Scale applied to output points. Use 1000 for meters->millimeters, 1 for no scaling.",
    )
    parser.add_argument(
        "--preview-ply",
        default=None,
        help="Optional colored PLY path for previewing hit points in MeshLab/CloudCompare.",
    )
    parser.add_argument(
        "--preview-usda",
        default=None,
        help="Optional USD ASCII path for previewing hit points in Isaac Sim.",
    )
    parser.add_argument(
        "--preview-stride",
        type=int,
        default=1,
        help="Preview every Nth row/column. Use 1 to preview all sampled hits.",
    )
    parser.add_argument(
        "--preview-point-width",
        type=float,
        default=0.0007,
        help="UsdGeom.Points width for --preview-usda, in link units.",
    )
    parser.add_argument(
        "--preview-color-mode",
        choices=["stripes", "gradient", "red"],
        default="stripes",
        help="Preview point coloring mode.",
    )
    parser.add_argument(
        "--preview-stripe-interval",
        type=int,
        default=10,
        help="Row/column interval for stripe preview coloring.",
    )
    parser.add_argument(
        "--preview-swap-stripe-colors",
        action="store_true",
        help="Swap red row stripes and blue column stripes in preview outputs.",
    )
    return parser.parse_args()


def find_link(stage: Usd.Stage, link: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(link)
    if prim and prim.IsValid():
        return prim

    matches = [p for p in stage.Traverse() if p.GetName() == link]
    if not matches:
        raise RuntimeError(f"Could not find link prim by path or name: {link}")
    if len(matches) > 1:
        print("[warn] Multiple prims matched link name; using first:")
        for p in matches:
            print(f"  {p.GetPath()}")
    return matches[0]


def transform_point(mat: Gf.Matrix4d, point: np.ndarray) -> np.ndarray:
    p = mat.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
    return np.array([p[0], p[1], p[2]], dtype=np.float64)


def triangulate_mesh_points(mesh: UsdGeom.Mesh) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)

    triangles = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        for j in range(1, count - 1):
            triangles.append([face[0], face[j], face[j + 1]])

    tri_idx = np.asarray(triangles, dtype=np.int64)
    return points, tri_idx


def collect_triangles_in_link(link_prim: Usd.Prim, mesh_regex: str | None) -> tuple[np.ndarray, np.ndarray]:
    cache = UsdGeom.XformCache()
    link_world = cache.GetLocalToWorldTransform(link_prim)
    world_link = link_world.GetInverse()
    regex = re.compile(mesh_regex) if mesh_regex else None

    all_triangles = []
    mesh_count = 0
    for prim in Usd.PrimRange(link_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if regex and not regex.search(str(prim.GetPath())):
            continue

        mesh = UsdGeom.Mesh(prim)
        points, tri_idx = triangulate_mesh_points(mesh)
        if tri_idx.size == 0:
            continue

        mesh_world = cache.GetLocalToWorldTransform(prim)
        points_link = np.vstack([
            transform_point(world_link, transform_point(mesh_world, p)) for p in points
        ])
        all_triangles.append(points_link[tri_idx])
        mesh_count += 1
        print(f"[mesh] {prim.GetPath()} triangles={len(tri_idx)}")

    if not all_triangles:
        raise RuntimeError("No UsdGeom.Mesh descendants found under the requested link.")

    triangles = np.concatenate(all_triangles, axis=0)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-12
    triangles = triangles[keep]
    normals = normals[keep] / lengths[keep, None]
    print(f"[info] collected_meshes={mesh_count}, triangles={len(triangles)}")
    return triangles, normals


def normals_from_triangles(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-12
    return triangles[keep], normals[keep] / lengths[keep, None]


def load_stl_triangles(stl_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(stl_path)
    data = path.read_bytes()
    triangles = None

    if len(data) >= 84:
        tri_count = int(np.frombuffer(data[80:84], dtype="<u4", count=1)[0])
        expected_size = 84 + tri_count * 50
        if expected_size == len(data):
            raw = np.frombuffer(data[84:], dtype=np.dtype([
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attr", "<u2"),
            ]), count=tri_count)
            triangles = raw["vertices"].astype(np.float64)

    if triangles is None:
        vertices = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(vertices) % 3 != 0 or not vertices:
            raise RuntimeError(f"Could not parse STL as binary or ASCII: {stl_path}")
        triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)

    triangles, normals = normals_from_triangles(triangles)
    print(f"[stl] {path} triangles={len(triangles)}")
    return triangles, normals


def ray_triangle_nearest(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    candidate_mask: np.ndarray | None = None,
) -> tuple[int, float] | None:
    eps = 1e-10
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    mask = np.abs(det) > eps
    if candidate_mask is not None:
        mask &= candidate_mask
    if not np.any(mask):
        return None

    inv_det = np.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    mask &= (u >= 0.0) & (u <= 1.0)
    if not np.any(mask):
        return None

    qvec = np.cross(tvec, e1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    mask &= (v >= 0.0) & ((u + v) <= 1.0)
    if not np.any(mask):
        return None

    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    mask &= t > eps
    if not np.any(mask):
        return None

    candidates = np.nonzero(mask)[0]
    best_local = np.argmin(t[candidates])
    tri_id = int(candidates[best_local])
    return tri_id, float(t[tri_id])


def axis_vector(name: str) -> np.ndarray:
    sign = 1.0
    axis = name
    if name.startswith("-"):
        sign = -1.0
        axis = name[1:]
    if name.startswith("+"):
        axis = name[1:]
    out = np.zeros(3, dtype=np.float64)
    out["xyz".index(axis)] = sign
    return out


def generate_maps(args: argparse.Namespace, triangles: np.ndarray, tri_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ray_dir = AXES[args.ray_dir].astype(np.float64)
    u_axis = axis_vector(args.u_axis)
    v_axis = axis_vector(args.v_axis)
    if abs(float(np.dot(ray_dir, u_axis))) > 1e-6 or abs(float(np.dot(ray_dir, v_axis))) > 1e-6:
        raise RuntimeError("u-axis and v-axis must be perpendicular to ray-dir.")
    if abs(float(np.dot(u_axis, v_axis))) > 1e-6:
        raise RuntimeError("u-axis and v-axis must be different axes.")

    candidate_mask = None
    if args.normal_min_dot > 0.0:
        plane_normal = -ray_dir
        signed_alignment = tri_normals @ plane_normal
        normal_alignment = np.abs(signed_alignment) if args.normal_both_sides else signed_alignment
        candidate_mask = normal_alignment >= float(args.normal_min_dot)
        kept = int(np.sum(candidate_mask))
        print(
            f"[filter] normal_min_dot={args.normal_min_dot:.3f}, "
            f"normal_both_sides={args.normal_both_sides}, "
            f"kept_triangles={kept}/{len(triangles)}"
        )

    vertices = triangles.reshape(-1, 3)
    u_coords = vertices @ u_axis
    v_coords = vertices @ v_axis
    d_coords = vertices @ ray_dir

    u_min = float(np.min(u_coords) - args.padding) if args.u_min is None else args.u_min
    u_max = float(np.max(u_coords) + args.padding) if args.u_max is None else args.u_max
    v_min = float(np.min(v_coords) - args.padding) if args.v_min is None else args.v_min
    v_max = float(np.max(v_coords) + args.padding) if args.v_max is None else args.v_max
    d_start = float(np.min(d_coords) - args.offset)

    print(f"[bounds] u=[{u_min:.6g}, {u_max:.6g}], v=[{v_min:.6g}, {v_max:.6g}], d_start={d_start:.6g}")

    us = np.linspace(u_min, u_max, args.width)
    vs = np.linspace(v_min, v_max, args.height)
    point_map = np.zeros((args.height, args.width, 3), dtype=np.float32)
    normal_map = np.zeros((args.height, args.width, 3), dtype=np.float32)
    hit_mask = np.zeros((args.height, args.width), dtype=bool)

    total = args.height * args.width
    for i, v in enumerate(vs):
        if i % max(1, args.height // 20) == 0:
            print(f"[raycast] row {i}/{args.height}")
        for j, u in enumerate(us):
            origin = u * u_axis + v * v_axis + d_start * ray_dir
            hit = ray_triangle_nearest(origin, ray_dir, triangles, candidate_mask)
            if hit is None:
                continue
            tri_id, depth = hit
            point = origin + depth * ray_dir
            normal = tri_normals[tri_id].copy()
            # Orient normals toward the virtual sensing plane, i.e. opposite the ray direction.
            if np.dot(normal, -ray_dir) < 0.0:
                normal *= -1.0
            point_map[i, j] = point.astype(np.float32)
            normal_map[i, j] = normal.astype(np.float32)
            hit_mask[i, j] = True

    hits = int(np.sum(hit_mask))
    print(f"[info] hits={hits}/{total} ({100.0 * hits / max(1, total):.2f}%)")
    return point_map, normal_map, hit_mask


def preview_colors(height: int, width: int, mode: str, stripe_interval: int, swap_stripe_colors: bool) -> np.ndarray:
    colors = np.zeros((height, width, 3), dtype=np.float32)
    if mode == "red":
        colors[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return colors

    rows, cols = np.indices((height, width))
    if mode == "gradient":
        denom_row = max(1, height - 1)
        denom_col = max(1, width - 1)
        colors[..., 0] = cols / denom_col
        colors[..., 1] = rows / denom_row
        colors[..., 2] = 1.0 - rows / denom_row
        return colors

    interval = max(1, int(stripe_interval))
    colors[:] = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    row_stripe = rows % interval == 0
    col_stripe = cols % interval == 0
    if swap_stripe_colors:
        colors[row_stripe] = np.array([0.0, 0.2, 1.0], dtype=np.float32)
        colors[col_stripe] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        colors[row_stripe] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        colors[col_stripe] = np.array([0.0, 0.2, 1.0], dtype=np.float32)
    colors[row_stripe & col_stripe] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    return colors


def preview_arrays(points: np.ndarray, hit_mask: np.ndarray, colors: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(stride))
    preview_mask = np.zeros_like(hit_mask, dtype=bool)
    preview_mask[::stride, ::stride] = hit_mask[::stride, ::stride]
    return points[preview_mask], colors[preview_mask]


def save_preview_ply(path: str, points: np.ndarray, colors: np.ndarray):
    rgb = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"[write] {path}")


def save_preview_usda(path: str, points: np.ndarray, colors: np.ndarray, point_width: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#usda 1.0\n")
        f.write("(\n    defaultPrim = \"TacMapPreview\"\n    upAxis = \"Z\"\n)\n\n")
        f.write("def Xform \"TacMapPreview\"\n{\n")
        f.write("    def Points \"sampled_points\"\n    {\n")
        f.write("        point3f[] points = [\n")
        for p in points:
            f.write(f"            ({p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}),\n")
        f.write("        ]\n")
        f.write("        float[] widths = [\n")
        for _ in points:
            f.write(f"            {point_width:.9g},\n")
        f.write("        ]\n")
        f.write("        color3f[] primvars:displayColor = [\n")
        for c in colors:
            f.write(f"            ({c[0]:.6g}, {c[1]:.6g}, {c[2]:.6g}),\n")
        f.write("        ] (\n            interpolation = \"vertex\"\n        )\n")
        f.write("    }\n")
        f.write("}\n")
    print(f"[write] {path}")


def main():
    args = parse_args()

    if args.stl:
        triangles, tri_normals = load_stl_triangles(args.stl)
    else:
        if args.usd is None:
            raise RuntimeError("Please provide either --stl or --usd.")
        if Usd is None or UsdGeom is None or Gf is None:
            raise RuntimeError(
                "USD Python bindings are not available in this environment. "
                "Use Isaac Sim's python.sh, or pass --stl to sample a mesh file directly."
            )
        if args.link is None:
            raise RuntimeError("--link is required when using --usd.")
        stage = Usd.Stage.Open(args.usd)
        if stage is None:
            raise RuntimeError(f"Could not open USD: {args.usd}")

        link_prim = find_link(stage, args.link)
        print(f"[link] {link_prim.GetPath()}")
        triangles, tri_normals = collect_triangles_in_link(link_prim, args.mesh_regex)
    point_map, normal_map, hit_mask = generate_maps(args, triangles, tri_normals)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    point_path = out_prefix.with_name(out_prefix.name + "_point.npy")
    normal_path = out_prefix.with_name(out_prefix.name + "_normal.npy")
    np.save(point_path, (point_map * args.point_scale).astype(np.float32))
    np.save(normal_path, normal_map.astype(np.float32))
    print(f"[write] {point_path}")
    print(f"[write] {normal_path}")

    if args.preview_ply or args.preview_usda:
        colors = preview_colors(
            args.height,
            args.width,
            args.preview_color_mode,
            args.preview_stripe_interval,
            args.preview_swap_stripe_colors,
        )
        preview_points, preview_point_colors = preview_arrays(
            point_map, hit_mask, colors, args.preview_stride
        )
        print(f"[preview] points={len(preview_points)}, stride={max(1, args.preview_stride)}")
        if args.preview_ply:
            Path(args.preview_ply).parent.mkdir(parents=True, exist_ok=True)
            save_preview_ply(args.preview_ply, preview_points, preview_point_colors)
        if args.preview_usda:
            save_preview_usda(
                args.preview_usda,
                preview_points,
                preview_point_colors,
                args.preview_point_width,
            )


if __name__ == "__main__":
    main()
