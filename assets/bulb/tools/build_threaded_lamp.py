#!/usr/bin/env python3
"""Build a native PhysX screw articulation, preserving HiveBoard thread surfaces.

The source USD has millimetre vertices under a 0.001 assembly transform.
All mesh transforms are baked in metres before authoring the new articulation.
Run with an Isaac Sim Python environment; no runtime Python controller is needed.
"""

from pathlib import Path
import hashlib
import json

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
ASSET = PACKAGE / "hiveboard_lamp"
SOURCE = ASSET / "source/Simulation/Lamp_Assembly.usd"
OUTPUT = ASSET / "hiveboard_lamp_threaded.usd"
PITCH = 0.006
TRAVEL = 0.024
SEATED_EXTENSION = 0.0003
INITIAL_EXTENSION = TRAVEL + SEATED_EXTENSION
ROOT = "/HiveboardLampThreaded"
MESH_MAP = {
    "base_pivot/Mesh": "Support/InternalThread",
    "lamp_pivot/Lamp_Assembly/Corpo1/Mesh": "Bulb/ExternalThread",
    "lamp_pivot/Lamp_Assembly/tn__Corpo11_j7m1J/Mesh": "Bulb/ShellA",
    "lamp_pivot/Lamp_Assembly/tn__Corpo12_j7m1J/Mesh": "Bulb/ShellB",
}


def source_meshes(stage):
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    meshes = {}
    for source_path, target in MESH_MAP.items():
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Lamp_Assembly/" + source_path))
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = np.array(cache.GetLocalToWorldTransform(mesh.GetPrim()))
        world = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        meshes[target] = (
            world.astype(np.float32),
            np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32),
            np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32),
        )
    return meshes


def make_body(stage, name, position, mass, inertia):
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema

    body = UsdGeom.Xform.Define(stage, ROOT + "/" + name)
    body.AddTranslateOp().Set(Gf.Vec3d(*position))
    body.AddOrientOp().Set(Gf.Quatf(0, 1, 0, 0) if name == "Bulb" else Gf.Quatf(1))
    prim = body.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    props = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    props.CreateDisableGravityAttr(True)
    props.CreateMaxDepenetrationVelocityAttr(0.2)
    props.CreateSleepThresholdAttr(0.0)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    if name == "Bulb":
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.03, 0, 0))
    report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    report.CreateThresholdAttr(0.0)
    return prim


def make_mesh(stage, path, arrays, color, material, collision=True):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt

    points, counts, indices = arrays
    mesh = UsdGeom.Mesh.Define(stage, ROOT + "/" + path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0).tolist()), Gf.Vec3f(*points.max(axis=0).tolist())])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = mesh.GetPrim()
    if not collision:
        return prim
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    if path.startswith("Support"):
        # Fixed support: exact triangle collision preserves the female bore,
        # including source meshes with small open boundaries.
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    else:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
        sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        sdf.CreateSdfResolutionAttr(512 if "ExternalThread" in path else 256)
        sdf.CreateSdfSubgridResolutionAttr(6)
        sdf.CreateSdfEnableRemeshingAttr(False)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact.CreateContactOffsetAttr(0.0001)
    contact.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    source_path = "base_pivot/Mesh" if path.startswith("Support") else next(
        (k for k, v in MESH_MAP.items() if v == path), "STL/Lamp_Screw.stl")
    prim.SetCustomDataByKey("sourceMesh", source_path)
    return prim


def make_joint(stage, schema, name, parent, child, anchor=(0, 0, 0)):
    from pxr import Gf, Sdf

    joint = schema.Define(stage, ROOT + "/Joints/" + name)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(ROOT + "/" + parent)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(ROOT + "/" + child)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr("X")
    return joint


def build():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    source = Usd.Stage.Open(str(SOURCE))
    arrays = source_meshes(source)
    # In-memory authoring + export avoids deleting an existing asset on failure.
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    stage.SetDefaultPrim(root)
    root.SetDocumentation("HiveBoard screw-in lamp: intact female and male thread meshes, SDF collisions, native screw coupling.")
    root.SetCustomDataByKey("pitchMetres", PITCH)
    root.SetCustomDataByKey("travelMetres", TRAVEL)
    root.SetCustomDataByKey("requiresGpuDynamics", True)
    root.SetCustomDataByKey("threadConvention", "positive rotation about +X advances +X; negative rotation inserts")

    base = make_body(stage, "Base", (0, 0, 0), .3005, (.000196, .000244, .00034))
    make_body(stage, "AxialCarrier", (INITIAL_EXTENSION, 0, 0), .01, (1e-6, 1e-6, 1e-6))
    make_body(stage, "Bulb", (INITIAL_EXTENSION, 0, 0), .20, (7.84e-5, 1.46e-4, 1.46e-4))
    # Keep the world joint inside the articulation-root subtree so PhysX
    # recognizes a fixed-base articulation, not a floating base plus constraint.
    fixed = UsdPhysics.FixedJoint.Define(stage, ROOT + "/Base/WorldFixedJoint")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(0))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
    fixed.CreateLocalRot0Attr(Gf.Quatf(1))
    fixed.CreateLocalRot1Attr(Gf.Quatf(1))
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(fixed.GetPrim())
    articulation.CreateEnabledSelfCollisionsAttr(True)
    articulation.CreateSolverPositionIterationCountAttr(32)
    articulation.CreateSolverVelocityIterationCountAttr(8)

    material = UsdShade.Material.Define(stage, ROOT + "/Materials/ThreadContact")
    friction = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    friction.CreateStaticFrictionAttr(.15)
    friction.CreateDynamicFrictionAttr(.10)
    friction.CreateRestitutionAttr(0)
    fingerprints = {}
    for name, data in arrays.items():
        color = (.22, .29, .36) if name.startswith("Support") else ((.55, .57, .60) if "Thread" in name else (.90, .85, .67))
        make_mesh(stage, name, data, color, material, collision=("Thread" not in name))
        fingerprints[name] = {
            "vertex_count": len(data[0]),
            "face_count": len(data[1]),
            "points_sha256": hashlib.sha256(data[0].tobytes()).hexdigest(),
            "indices_sha256": hashlib.sha256(data[2].tobytes()).hexdigest(),
            "bounds_m": [data[0].min(axis=0).tolist(), data[0].max(axis=0).tolist()],
        }

    # Preserve the visual female thread exactly. Give only the collision
    # entrance a 1 mm lead-in chamfer: the source mouth locally clips male
    # crests by ~0.2 mm. The remaining bore/thread triangles are unchanged.
    points, counts, indices = arrays["Support/InternalThread"]
    relieved = points.copy()
    radius = np.linalg.norm(points[:, 1:], axis=1)
    mouth_x = float(points[:, 0].max())
    blend = np.clip((points[:, 0] - (mouth_x - .001)) / .001, 0, 1)
    minimum_radius = .01335 + .0005 * blend
    selection = (blend > 0) & (radius < minimum_radius) & (radius > .01)
    relieved[selection, 1:] *= (minimum_radius[selection] / radius[selection])[:, None]
    female_collider = make_mesh(stage, "Support/InternalThreadCollider", (relieved, counts, indices),
                               (.22, .29, .36), material)
    UsdGeom.Imageable(female_collider).CreateVisibilityAttr("invisible")
    print("ENTRANCE collision vertices relieved", int(selection.sum()), flush=True)

    # The packaged high-resolution screw STL is closed; the decimated render
    # OBJ has boundary edges. Use the original full STL for SDF, preserving its
    # thread detail instead of filling holes in the visual mesh.
    import trimesh
    screw = trimesh.load_mesh(ASSET / "source/STL/Lamp_Screw.stl", process=True)
    assert screw.is_watertight
    transform = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(
        source.GetPrimAtPath("/World/Lamp_Assembly/lamp_pivot/Lamp_Assembly/Corpo1/Mesh")))
    points = (np.c_[screw.vertices, np.ones(len(screw.vertices))] @ transform)[:, :3].astype(np.float32)
    collision = make_mesh(stage, "Bulb/ExternalThreadCollider", (
        points, np.full(len(screw.faces), 3, dtype=np.int32), screw.faces.astype(np.int32).ravel(),
    ), (.5, .5, .5), material)
    UsdGeom.Imageable(collision).CreateVisibilityAttr("invisible")

    slide = make_joint(stage, UsdPhysics.PrismaticJoint, "screw_slide", "Base", "AxialCarrier", (INITIAL_EXTENSION, 0, 0))
    slide.CreateLowerLimitAttr(-TRAVEL)
    slide.CreateUpperLimitAttr(0)
    turn = make_joint(stage, UsdPhysics.RevoluteJoint, "screw_turn", "AxialCarrier", "Bulb")
    # Source zero rotation puts male crests against female crests. A half-turn
    # aligns crests with grooves; this rigid pose change preserves every vertex.
    turn.CreateLocalRot1Attr(Gf.Quatf(0, 1, 0, 0))
    turn.CreateLowerLimitAttr(-360 * TRAVEL / PITCH)
    turn.CreateUpperLimitAttr(0)
    # USD uses degrees here, whereas Isaac Lab joint states use radians.
    # theta_deg - (360 / pitch_m) * displacement_m = 0.
    coupling = PhysxSchema.PhysxMimicJointAPI.Apply(turn.GetPrim(), "rotX")
    coupling.CreateReferenceJointRel().SetTargets([slide.GetPath()])
    coupling.CreateGearingAttr(-360.0 / PITCH)
    coupling.CreateOffsetAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(turn.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.05)
    drive.CreateDampingAttr(0.005)
    drive.CreateMaxForceAttr(.25)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)

    # Check invariants before writing either deliverable.
    for name, data in arrays.items():
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(ROOT + "/" + name))
        np.testing.assert_array_equal(np.array(mesh.GetPointsAttr().Get()), data[0])
        np.testing.assert_array_equal(np.array(mesh.GetFaceVertexIndicesAttr().Get()), data[2])
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 3
    stage.GetRootLayer().Export(str(OUTPUT))
    metadata = {
        "usd": OUTPUT.name, "default_prim": ROOT,
        "physics": "PhysX GPU fixed-base articulation; native mimic screw coupling; static female triangle collision and dynamic male SDF collision",
        "pitch_m": PITCH, "pitch_basis": "Approx. 6 mm measured from consecutive external-thread radial-profile peaks; nominal value, not a certified CAD specification",
        "travel_m": TRAVEL, "turns": TRAVEL / PITCH,
        "initial_extension_m": INITIAL_EXTENSION,
        "seated_extension_from_source_zero_m": SEATED_EXTENSION,
        "initial_bulb_rotation_about_x_deg": 180,
        "limits": {"screw_turn_deg": [-1440, 0], "screw_slide_m": [-TRAVEL, 0]},
        "coupling": "screw_slide_m = screw_turn_rad * pitch_m / (2*pi)",
        "meshes": fingerprints,
        "collision": {
            "female": "Source base triangles with an entrance-only lead-in; bore thread topology retained",
            "female_entrance_length_m": .001,
            "female_entrance_min_radius_m": .01385,
            "female_modified_vertex_count": int(selection.sum()),
            "female_max_vertex_displacement_m": float(np.linalg.norm(relieved - arrays["Support/InternalThread"][0], axis=1).max()),
            "male": "Unmodified watertight packaged STL/Lamp_Screw.stl, baked to metres, SDF resolution 512",
            "male_face_count": len(screw.faces),
            "male_source_sha256": hashlib.sha256((ASSET / "source/STL/Lamp_Screw.stl").read_bytes()).hexdigest(),
            "shells": "Source meshes, SDF resolution 256",
            "contact_offset_m": .0001, "rest_offset_m": 0,
            "static_friction": .15, "dynamic_friction": .10,
        },
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "limitations": ["Guided and pre-engaged assembly: axis alignment is constrained and complete detachment is not supported.",
                        "Screw advance is enforced by a native mimic constraint; SDF contacts retain the thread surfaces but do not solely determine the lead.",
                        "Female entrance collision only has a 1 mm lead-in relief; all source visual vertices/faces are unchanged.",
                        "Fully seated pose retains 0.3 mm axial clearance from source CAD zero to avoid end-face interference.",
                        "GPU dynamics are required for SDF contact. Masses, friction and contact resolution are simulation defaults, not calibrated manufacturing data."],
    }
    (ASSET / "threaded_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("BUILT", OUTPUT, flush=True)


if __name__ == "__main__":
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        build()
    except Exception:
        import os
        import sys
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)  # SimulationApp.close() otherwise masks the build failure.
    app.close(skip_cleanup=True)
