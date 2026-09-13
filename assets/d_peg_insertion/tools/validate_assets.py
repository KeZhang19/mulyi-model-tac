#!/usr/bin/env python3
"""CPU-only structural/geometry checks for the D-peg assets; no simulator launch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree


PACKAGE = Path(__file__).resolve().parents[1]


def load_usd_modules():
    """Expose Isaac's installed USD libraries without starting Kit or a GPU."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
    except ImportError:
        if os.environ.get("DPEG_USD_BOOTSTRAPPED") == "1":
            raise
        spec = importlib.util.find_spec("isaacsim")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("Use the brainco Python environment (or an environment with usd-core).")
        isaac = Path(next(iter(spec.submodule_search_locations)))
        candidates = sorted((isaac / "extscache").glob("omni.usd.libs-*"))
        if not candidates:
            raise RuntimeError("Isaac USD libraries were not found.")
        usd = candidates[-1]
        env = dict(os.environ)
        env["DPEG_USD_BOOTSTRAPPED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join((str(usd), env.get("PYTHONPATH", "")))
        env["LD_LIBRARY_PATH"] = os.pathsep.join((str(Path(sys.prefix) / "lib"), str(usd / "bin"),
                                                 str(isaac / "kit"), env.get("LD_LIBRARY_PATH", "")))
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    return Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils


def inside_any(points, hulls, tolerance=1e-8):
    result = np.zeros(len(points), dtype=bool)
    for hull in hulls:
        result |= ((points @ hull.equations[:, :3].T + hull.equations[:, 3]) <= tolerance).all(axis=1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="Optionally write the checked geometry summary.")
    args = parser.parse_args()
    Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = load_usd_modules()
    metadata = json.loads((PACKAGE / "metadata.json").read_text())
    results = {"schema_version": 1, "validation": "CPU-only static geometry and native USD checks",
               "physics_rollout_tested": False, "assets": {}}
    usd_hulls = {}
    for name in ("peg", "socket"):
        config = metadata[name]
        path = PACKAGE / config["usd"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config["usd_sha256"]
        stage = Usd.Stage.Open(str(path))
        assert stage is not None
        root = stage.GetDefaultPrim()
        root_path = str(root.GetPath())
        assert root_path == config["default_prim"]
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        assert UsdGeom.GetStageUpAxis(stage) == "Z"
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert bodies == [root]
        assert bool(UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()) == config["kinematic"]
        assert not [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        if name == "peg":
            assert abs(UsdPhysics.MassAPI(root).GetMassAttr().Get() - .1) < 1e-7
            assert not root.GetAttribute("physxRigidBody:disableGravity").Get()
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
        assert len(layers) == 1 and not assets and not unresolved
        visual = UsdGeom.Mesh.Get(stage, root_path + "/" + config["visual_mesh_path"])
        tactile = UsdGeom.Mesh.Get(stage, root_path + "/" + config["tactile_mesh_path"])
        for attr in ("GetPointsAttr", "GetFaceVertexCountsAttr", "GetFaceVertexIndicesAttr"):
            np.testing.assert_array_equal(np.asarray(getattr(visual, attr)().Get()),
                                          np.asarray(getattr(tactile, attr)().Get()))
        assert tactile.GetVisibilityAttr().Get() == "invisible"
        assert not tactile.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        points = np.asarray(visual.GetPointsAttr().Get())
        faces = np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        mesh = trimesh.Trimesh(points, faces, process=True)
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
        stl = trimesh.load_mesh(PACKAGE / "meshes" / f"{name}.stl", process=True)
        assert stl.is_watertight and stl.is_winding_consistent
        maximum_stl_error = float(cKDTree(points).query(stl.vertices * .001)[0].max())
        assert maximum_stl_error < 1e-7
        collisions = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        expected_count = 2 if name == "peg" else config["collision_cell_count"]
        assert len(collisions) == expected_count
        hulls, maximum_hull_volume_error, maximum_hull_surface_error = [], 0.0, 0.0
        for prim in collisions:
            assert str(prim.GetPath()).startswith(root_path + "/Collisions/")
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
            assert abs(prim.GetAttribute("physxCollision:contactOffset").Get() - .0001) < 1e-9
            assert prim.GetAttribute("physxCollision:restOffset").Get() == 0
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            physics_material = UsdPhysics.MaterialAPI(material.GetPrim())
            expected_mu = (.9, .8) if prim.GetName() == "Grip" else (.5, .4)
            assert abs(physics_material.GetStaticFrictionAttr().Get() - expected_mu[0]) < 1e-6
            assert abs(physics_material.GetDynamicFrictionAttr().Get() - expected_mu[1]) < 1e-6
            collider = UsdGeom.Mesh(prim)
            vertices_mm = np.asarray(collider.GetPointsAttr().Get(), dtype=float) * 1000
            triangle_indices = np.asarray(collider.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            body = trimesh.Trimesh(vertices_mm, triangle_indices, process=True)
            assert body.is_watertight and body.is_winding_consistent
            hull = ConvexHull(vertices_mm)
            error = abs(hull.volume - body.volume)
            maximum_hull_volume_error = max(maximum_hull_volume_error, float(error))
            hull_mesh = trimesh.Trimesh(vertices_mm, hull.simplices, process=False)
            _, distances, _ = trimesh.proximity.closest_point_naive(hull_mesh, body.triangles_center)
            surface_error = float(distances.max())
            maximum_hull_surface_error = max(maximum_hull_surface_error, surface_error)
            assert surface_error < .03, (prim.GetPath(), surface_error)
            hulls.append(hull)
        usd_hulls[name] = hulls
        results["assets"][name] = {
            "default_prim": root_path, "rigid_bodies": len(bodies), "joints": 0,
            "collision_cells": len(collisions), "watertight": True,
            "canonical_visual_equals_tactile": True, "max_usd_stl_vertex_error_m": maximum_stl_error,
            "max_cell_hull_volume_difference_mm3": maximum_hull_volume_error,
            "max_cell_hull_surface_difference_mm": maximum_hull_surface_error,
            "bounds_m": mesh.bounds.tolist(),
        }
    # A filled socket hull would fail both the open-cavity and wrong-yaw probes.
    xy = np.stack(np.meshgrid(np.arange(-13, 8, .25), np.arange(-13, 13.1, .25)), axis=-1).reshape(-1, 2)
    checked = 0
    for z in (20.01, 25., 35., 49., 49.5, 49.99):
        bevel = max(0, z - 49)
        r, flat = 12.75 + bevel, 6.75 + bevel
        keep = (np.linalg.norm(xy, axis=1) < r - .03) & (xy[:, 0] < flat - .03)
        points = np.c_[xy[keep], np.full(keep.sum(), z)]
        assert not inside_any(points, usd_hulls["socket"]).any(), ("cavity was filled", z)
        checked += len(points)
    assert inside_any(np.array([[0, 0, 10], [30, 0, 30], [11, 0, 30]]), usd_hulls["socket"]).all()
    assert not inside_any(np.array([[0, 0, 30], [-11, 0, 30]]), usd_hulls["socket"]).any()
    shaft = trimesh.load_mesh(PACKAGE / "meshes" / "shaft_collision.stl", process=True)
    tip = shaft.vertices[np.isclose(shaft.vertices[:, 2], 0, atol=1e-5)]
    assert abs(np.linalg.norm(tip[:, :2], axis=1).max() - 11) < .03
    assert abs(tip[:, 0].max() - 5) < .03
    shaft_at_target = shaft.vertices + [0, 0, 22]
    overlap = shaft_at_target[shaft_at_target[:, 2] < 49]
    assert not inside_any(overlap, usd_hulls["socket"]).any()
    results["geometric_insertion"] = {
        "empty_cavity_probes": checked, "correct_pose_collision_free": True,
        "wrong_180_degree_yaw_wall_probe_blocked": True, "blind_bottom_solid": True,
        "target_depth_m": .028, "target_tip_bottom_clearance_m": .002,
        "tip_chamfer_profile": "radius 11 mm / flat x=5 mm at z=0; normal inset 1 mm",
    }
    text = json.dumps(results, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
