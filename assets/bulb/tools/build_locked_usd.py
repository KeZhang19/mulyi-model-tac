#!/usr/bin/env python3
"""Build self-contained, joint-free USD assets for the bulb package.

Run this script with an Isaac Sim Python environment.  It intentionally keeps
all geometry inside each output USD so the package can be moved without
breaking asset references.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from isaacsim import SimulationApp


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LIGHTBULB_ROOT = PACKAGE_ROOT / "lightbulb_with_socket"
HIVEBOARD_ROOT = PACKAGE_ROOT / "hiveboard_lamp"


def _mesh_arrays(path: Path, scale=(1.0, 1.0, 1.0), translate=(0.0, 0.0, 0.0)):
    loaded = trimesh.load_mesh(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh in {path}, got {type(loaded).__name__}")
    if loaded.faces.shape[1] != 3:
        loaded = loaded.triangulate()

    points = np.asarray(loaded.vertices, dtype=np.float32)
    points = points * np.asarray(scale, dtype=np.float32)
    points = points + np.asarray(translate, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    return points, faces


def _author_mesh(stage, path, source, *, scale, translate, color, opacity, collision):
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    points, faces = _mesh_arrays(source, scale=scale, translate=translate)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    mesh.CreateDisplayOpacityAttr(Vt.FloatArray([opacity]))

    if collision:
        UsdGeom.Imageable(mesh.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")


def build_lightbulb():
    from pxr import Usd, UsdGeom, UsdPhysics

    output = LIGHTBULB_ROOT / "lightbulb_with_socket.usd"
    output.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/LightbulbWithSocket")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetDocumentation(
        "Joint-free compound lightbulb. Visual transforms from the Dexonomy "
        "MJCF adapter are baked into the mesh points."
    )
    root.GetPrim().SetCustomDataByKey("geometryLocked", True)
    root.GetPrim().SetCustomDataByKey("sourceUnits", "metres")
    rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.1565223603)

    UsdGeom.Scope.Define(stage, "/LightbulbWithSocket/Visuals")
    UsdGeom.Scope.Define(stage, "/LightbulbWithSocket/Collisions")
    mesh_root = LIGHTBULB_ROOT / "meshes"

    # These are the exact scale/translation values used by Dexonomy's
    # body.xml. Baking them removes all internal runtime transforms.
    _author_mesh(
        stage,
        "/LightbulbWithSocket/Visuals/Glass",
        mesh_root / "lightbulb_head.stl",
        scale=(1.01, 1.0, 1.0),
        translate=(0.0, 0.0, 0.0),
        color=(0.93, 0.99, 0.97),
        opacity=0.70,
        collision=False,
    )
    _author_mesh(
        stage,
        "/LightbulbWithSocket/Visuals/Socket",
        mesh_root / "lightbulb_socket.stl",
        scale=(1.05, 1.0, 1.0),
        translate=(0.002, 0.0, 0.0),
        color=(0.50, 0.50, 0.50),
        opacity=1.0,
        collision=False,
    )
    _author_mesh(
        stage,
        "/LightbulbWithSocket/Collisions/GlassCollision",
        mesh_root / "contact0.stl",
        scale=(1.0, 1.0, 1.0),
        translate=(0.0, 0.0, 0.0),
        color=(0.3, 0.4, 0.5),
        opacity=1.0,
        collision=True,
    )
    _author_mesh(
        stage,
        "/LightbulbWithSocket/Collisions/SocketCollision",
        mesh_root / "contact1.stl",
        scale=(1.0, 1.0, 1.0),
        translate=(0.0, 0.0, 0.0),
        color=(0.3, 0.4, 0.5),
        opacity=1.0,
        collision=True,
    )

    stage.GetRootLayer().Save()
    return output


def _remove_child_rigid_body_apis(stage, root_path):
    from pxr import PhysxSchema, UsdPhysics

    for prim in stage.Traverse():
        if prim.GetPath() == root_path:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        for property_name in (
            "physics:angularVelocity",
            "physics:kinematicEnabled",
            "physics:rigidBodyEnabled",
            "physics:startsAsleep",
            "physics:velocity",
        ):
            prim.RemoveProperty(property_name)


def build_hiveboard_lamp():
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    source_path = HIVEBOARD_ROOT / "source" / "Simulation" / "Lamp_Assembly.usd"
    output = HIVEBOARD_ROOT / "hiveboard_lamp_locked.usd"
    source = Usd.Stage.Open(str(source_path))
    if source is None:
        raise RuntimeError(f"Could not open {source_path}")

    output.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    root = UsdGeom.Xform.Define(stage, "/HiveboardLamp")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetDocumentation(
        "HiveBoard lamp collapsed at its zero joint pose into one compound rigid body. "
        "There are no internal articulation degrees of freedom."
    )
    root.GetPrim().SetCustomDataByKey("geometryLocked", True)
    root.GetPrim().SetCustomDataByKey("sourceJointPose", "RevoluteJoint=0 rad; PrismaticJoint=0 m")
    root.GetPrim().SetCustomDataByKey("sourceUnits", "metres")

    source_root = Sdf.Path("/World/Lamp_Assembly")
    destination_root = Sdf.Path("/HiveboardLamp")
    for child_name in ("base_pivot", "lamp_pivot", "rotation_pivot"):
        copied = Sdf.CopySpec(
            source.GetRootLayer(),
            source_root.AppendChild(child_name),
            stage.GetRootLayer(),
            destination_root.AppendChild(child_name),
        )
        if not copied:
            raise RuntimeError(f"Failed to copy HiveBoard prim {child_name}")

    _remove_child_rigid_body_apis(stage, destination_root)

    # Keep the detailed render meshes, but use the latest source URDF's cheap
    # collision strategy. Convex decomposition of all three shell meshes takes
    # minutes during the first PhysX spawn and is unsuitable for parallel RL.
    base_mesh = stage.GetPrimAtPath("/HiveboardLamp/base_pivot/Mesh")
    UsdPhysics.MeshCollisionAPI(base_mesh).CreateApproximationAttr("convexHull")
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith("/HiveboardLamp/lamp_pivot/"):
            continue
        for api in (
            UsdPhysics.CollisionAPI,
            UsdPhysics.MeshCollisionAPI,
            PhysxSchema.PhysxCollisionAPI,
            PhysxSchema.PhysxTriangleMeshCollisionAPI,
            PhysxSchema.PhysxConvexHullCollisionAPI,
            PhysxSchema.PhysxConvexDecompositionCollisionAPI,
        ):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
        for property_name in tuple(prop.GetName() for prop in prim.GetProperties()):
            if property_name.startswith("physics:") or property_name.startswith("physxCollision:"):
                prim.RemoveProperty(property_name)

    UsdGeom.Scope.Define(stage, "/HiveboardLamp/Collisions")
    for name, radius, height, center in (
        ("SocketCollision", 0.014, 0.030, (-0.015, 0.0, 0.0)),
        ("BulbCollision", 0.028, 0.080, (0.035, 0.0, 0.0)),
    ):
        cylinder = UsdGeom.Cylinder.Define(stage, f"/HiveboardLamp/Collisions/{name}")
        cylinder.CreateAxisAttr(UsdGeom.Tokens.x)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        UsdGeom.Xformable(cylinder.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*center))
        UsdGeom.Imageable(cylinder.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim()).CreateCollisionEnabledAttr(True)

    rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.5505)

    stage.GetRootLayer().Save()
    return output


def validate_outputs(paths):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    failures = []
    for path in paths:
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            failures.append(f"could not open {path}")
            continue
        root = stage.GetDefaultPrim()
        joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
        rigid_bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        collisions = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))

        if not root or not root.GetCustomDataByKey("geometryLocked"):
            failures.append(f"{path}: missing locked default prim")
        if joints:
            failures.append(f"{path}: expected zero joints, found {len(joints)}")
        if rigid_bodies != [root]:
            failures.append(f"{path}: expected the default prim to be the only rigid body")
        if not collisions:
            failures.append(f"{path}: no collision geometry")
        if assets or unresolved or len(layers) != 1:
            failures.append(f"{path}: output USD is not self-contained")
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
            failures.append(f"{path}: stage units are not metres")

    if failures:
        raise RuntimeError("\n".join(failures))


def main():
    app = SimulationApp({"headless": True})
    try:
        outputs = [build_lightbulb(), build_hiveboard_lamp()]
        validate_outputs(outputs)
        for output in outputs:
            print(f"built {output}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
