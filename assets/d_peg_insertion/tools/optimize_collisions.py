#!/usr/bin/env python3
"""Build opt-in socket collision candidates without changing the canonical asset.

The original visual/tactile meshes and rigid-body/material settings are copied
verbatim. NumPy, SciPy and trimesh suffice; no CAD or simulator is needed.
Candidates require the separate force-controlled physics validation before use.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.spatial import ConvexHull
import trimesh


PACKAGE = Path(__file__).resolve().parents[1]
COLLISION_SCOPE = '    def Scope "Collisions"\n    {\n'


def mesh_block(text, name):
    start = text.index(f'        def Mesh "{name}"')
    opening = text.index("{", start)
    closing = text.index("\n        }", opening) + len("\n        }")
    return text[start:closing]


def mesh_arrays(block):
    def array(attribute):
        match = re.search(re.escape(attribute) + r" = (\[[^\n]*\])", block)
        if match is None:
            raise ValueError(f"Missing mesh attribute: {attribute}")
        return np.asarray(ast.literal_eval(match[1]))
    vertices = array("point3f[] points").astype(np.float64) * 1000.0
    counts = array("int[] faceVertexCounts")
    if not np.all(counts == 3):
        raise ValueError("Only triangulated source meshes are supported")
    return vertices, array("int[] faceVertexIndices").reshape(-1, 3).astype(np.int64)


def d_radius(angles, radius, flat):
    cosine = np.cos(angles)
    return np.minimum(radius, flat / np.maximum(cosine, 1e-12))


def wall_cells(segments, dimensions):
    """Union a beveled flat wall with circular wall sectors.

    Keeping the flat separate avoids convex-hulling the nonconvex transition
    where the circle/flat junction moves through the entrance chamfer.
    """
    radius = dimensions["hole_radius"] * 1000
    flat = dimensions["hole_flat_x"] * 1000
    chamfer = dimensions["entry_chamfer"] * 1000
    width, _, height = np.asarray(dimensions["socket_size"]) * 1000
    bottom = height - dimensions["hole_depth"] * 1000
    junction = math.acos((flat + chamfer) / (radius + chamfer))
    angles = [value for value in np.linspace(0, 2 * math.pi, segments, endpoint=False)
              if junction < value < 2 * math.pi - junction]
    angles.extend(value for value in (math.pi / 4 + i * math.pi / 2 for i in range(4))
                  if junction < value < 2 * math.pi - junction)
    angles.extend((junction, 2 * math.pi - junction))
    angles = np.asarray(sorted(set(round(value, 12) for value in angles)))
    cells = []
    raw_cells = []
    for index, (first, second) in enumerate(zip(angles[:-1], angles[1:])):
        directions = np.stack((np.cos([first, second]), np.sin([first, second])), axis=1)
        outer = directions * (width / 2 / np.max(np.abs(directions), axis=1))[:, None]
        vertices = []
        for z in (bottom, height - chamfer, height):
            bevel = max(0.0, z - height + chamfer)
            inner = directions * (radius + bevel)
            vertices.extend((*xy, z) for xy in (inner[0], outer[0], outer[1], inner[1]))
        raw_cells.append((f"Wall_{index:03d}", vertices))
    flat_vertices = []
    for z in (bottom, height - chamfer, height):
        inner_x = flat + max(0.0, z - height + chamfer)
        flat_vertices.extend((x, y, z) for x in (inner_x, width / 2) for y in (-width / 2, width / 2))
    raw_cells.append(("FlatWall", flat_vertices))
    for name, vertices in raw_cells:
        vertices = np.unique(np.asarray(vertices), axis=0)
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        flipped = np.sum(normals * hull.equations[:, :3], axis=1) < 0
        faces[flipped] = faces[flipped, ::-1]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
        cells.append((name, mesh, hull.equations))
    return angles, cells


def collision_mesh_text(name, vertices_mm, faces, approximation):
    def tuples(array):
        return "[" + ", ".join("(" + ", ".join(format(float(value), ".12g") for value in row) + ")" for row in array) + "]"
    vertices = vertices_mm * .001
    return f'''        def Mesh "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"])
        {{
            point3f[] points = {tuples(vertices)}
            int[] faceVertexCounts = [{", ".join("3" for _ in faces)}]
            int[] faceVertexIndices = [{", ".join(str(int(value)) for value in faces.ravel())}]
            float3[] extent = {tuples([vertices.min(axis=0), vertices.max(axis=0)])}
            uniform token subdivisionScheme = "none"
            uniform token orientation = "rightHanded"
            token visibility = "invisible"
            bool physics:collisionEnabled = true
            uniform token physics:approximation = "{approximation}"
            float physxCollision:contactOffset = 0.0001
            float physxCollision:restOffset = 0
            rel material:binding:physics = </DSocket/Materials/InsertionContact>
        }}'''


def cavity_validation(angles, cells, dimensions, samples):
    """Ray-intersect actual convex hull halfspaces, including the 3-D chamfer."""
    radius, flat, chamfer = (dimensions[key] * 1000 for key in ("hole_radius", "hole_flat_x", "entry_chamfer"))
    height = dimensions["socket_size"][2] * 1000
    bottom = height - dimensions["hole_depth"] * 1000
    theta = (np.arange(samples) + .5) * 2 * np.pi / samples
    directions = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    levels = np.unique(np.concatenate((np.linspace(bottom + 1e-6, height - chamfer, 9), np.linspace(height - chamfer, height, 65))))
    cavity = np.full((len(levels), samples), np.inf)
    for index, (name, _, equations) in enumerate(cells):
        selected = np.ones(samples, dtype=bool) if name == "FlatWall" else (theta >= angles[index]) & (theta < angles[index + 1])
        denominator = directions[selected] @ equations[:, :2].T
        numerator = -(levels[:, None] * equations[None, :, 2] + equations[None, :, 3])
        with np.errstate(divide="ignore", invalid="ignore"):
            distances = numerator[:, None, :] / denominator[None, :, :]
        lower = np.max(np.where(denominator[None, :, :] < -1e-10, distances, -np.inf), axis=2)
        upper = np.min(np.where(denominator[None, :, :] > 1e-10, distances, np.inf), axis=2)
        valid = np.isfinite(lower) & (lower >= 0) & (lower <= upper + 1e-7)
        candidate = np.where(valid, lower, np.inf)
        cavity[:, selected] = np.minimum(cavity[:, selected], candidate)
    assert np.all(np.isfinite(cavity))
    bevels = np.maximum(levels - height + chamfer, 0)
    ideal = d_radius(theta[None, :], radius + bevels[:, None], flat + bevels[:, None])
    error = cavity - ideal
    # The minimum clearance uses the full-size shaft; the tip chamfer only adds clearance.
    shaft = d_radius(theta, dimensions["shaft_radius"] * 1000, dimensions["shaft_flat_x"] * 1000)
    base_clearance = cavity[0] - shaft
    flat_mask = np.abs(theta - np.pi * 2 * np.round(theta / (np.pi * 2))) < .8
    flat_x_error = np.max(np.abs(cavity[:, flat_mask] * np.cos(theta[flat_mask]) - (flat + bevels[:, None])))
    worst = np.unravel_index(np.argmin(error), error.shape)
    results = {
        "ray_samples": int(samples), "height_samples": int(len(levels)),
        "max_inward_error_um": float(-error.min() * 1000),
        "max_outward_error_um": float(max(0.0, error.max()) * 1000),
        "worst_angle_rad": float(theta[worst[1]]), "worst_height_mm": float(levels[worst[0]]),
        "minimum_full_shaft_radial_clearance_mm": float(base_clearance.min()),
        "flat_x_max_error_um": float(flat_x_error * 1000),
        "floor_height_mm": float(bottom), "mouth_height_mm": float(height),
        "entry_chamfer_mm": float(chamfer), "aligned_full_shaft_fits": bool(np.all(base_clearance > 0)),
        "validation": "Actual 3-D convex-hull halfspace intersections of arc sectors union flat wall; 64 subdivisions across entrance chamfer.",
    }
    assert results["max_outward_error_um"] < 1e-4, results
    assert results["max_inward_error_um"] < 100.0, results
    assert results["flat_x_max_error_um"] < 1e-4, results
    assert results["minimum_full_shaft_radial_clearance_mm"] > .65, results
    return results


def build(args):
    source = args.source.read_text()
    metadata = json.loads((PACKAGE / "metadata.json").read_text())
    dimensions = metadata["dimensions_m"]
    prefix, _ = source.split(COLLISION_SCOPE, 1)
    original_bottom = mesh_block(source, "Bottom")
    original_visual = mesh_block(source, "VisualMesh")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    for segments in args.segments:
        angles, cells = wall_cells(segments, dimensions)
        blocks = [original_bottom] + [collision_mesh_text(name, mesh.vertices, mesh.faces, "convexHull") for name, mesh, _ in cells]
        text = prefix + COLLISION_SCOPE + "\n".join(blocks) + "\n    }\n}\n"
        destination = args.output_dir / f"socket_{segments}.usd"
        destination.write_text(text)
        assert text.split(COLLISION_SCOPE)[0] == prefix
        assert mesh_block(text, "Bottom") == original_bottom
        variants[str(segments)] = {
            "path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "collider_count": len(cells) + 1, "approximation": "convexHull",
            "visual_tactile_and_body_settings_identical": True, "floor_collider_identical": True,
            "geometry_validation": cavity_validation(angles, cells, dimensions, args.samples),
        }
        print(json.dumps(variants[str(segments)]), flush=True)
    vertices, faces = mesh_arrays(original_visual)
    visual_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert visual_mesh.is_watertight and visual_mesh.is_winding_consistent and visual_mesh.volume > 0
    triangle_text = prefix + COLLISION_SCOPE + collision_mesh_text("SocketMesh", vertices, faces, "none") + "\n    }\n}\n"
    triangle_path = args.output_dir / "socket_triangle.usd"
    triangle_path.write_text(triangle_text)
    cooked_vertices, cooked_faces = mesh_arrays(mesh_block(triangle_text, "SocketMesh"))
    assert np.allclose(cooked_vertices, vertices, atol=1e-9, rtol=0) and np.array_equal(cooked_faces, faces)
    variants["triangle"] = {
        "path": str(triangle_path), "sha256": hashlib.sha256(triangle_path.read_bytes()).hexdigest(),
        "collider_count": 1, "approximation": "none", "collision_triangles": len(faces),
        "visual_tactile_and_body_settings_identical": True,
        "collision_mesh_matches_original_visual": True,
        "max_vertex_deviation_from_original_visual_um": float(np.abs(cooked_vertices - vertices).max() * 1000),
        "watertight": bool(visual_mesh.is_watertight),
        "physics_support_reference": "https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/RigidBodyCollision.html",
        "risk": "Kinematic triangle mesh is supported by PhysX. Must measure actual GPU contact/solver performance and force-controlled insertion. It matches the CAD tessellated visual cavity, not the previous independent 133-convex approximation.",
    }
    report = {
        "source": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "original_collider_count": source.count("bool physics:collisionEnabled = true"),
        "variants": variants, "production_asset_modified": False, "physics_validated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(variants["triangle"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PACKAGE / "socket.usd")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "collision_variants")
    parser.add_argument("--segments", nargs="+", type=int, default=[48, 32])
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--report", type=Path, default=PACKAGE.parents[1] / "outputs/d_peg_speed_20260912/collision_candidates.json")
    options = parser.parse_args()
    if any(count < 32 for count in options.segments):
        parser.error("At least 32 angular segments are required by the 100 micrometre error bound")
    build(options)
