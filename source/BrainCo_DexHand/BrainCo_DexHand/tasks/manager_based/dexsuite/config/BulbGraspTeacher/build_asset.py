#!/usr/bin/env python3
"""Derive a centered free bulb and deterministic teacher point cloud from the assembly.

Run with numpy and OpenUSD (usd-core), or the Isaac Sim Python environment.
The assembly is preserved. Convex hulls on each of the three bulb components
are suitable for grasping; this derivative is not a screw-assembly asset.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
_simulation_app = None
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ModuleNotFoundError as exc:
    if exc.name != "pxr" or __name__ != "__main__":
        raise
    from isaaclab.app import AppLauncher
    _simulation_app = AppLauncher(headless=True).app
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def build(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    src = Usd.Stage.Open(str(source))
    meshes = []
    for name in ('ExternalThread', 'ShellA', 'ShellB'):
        mesh = UsdGeom.Mesh(src.GetPrimAtPath('/HiveboardLampThreaded/Bulb/' + name))
        meshes.append((name, mesh, np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)))
    all_points = np.concatenate([points for _, _, points in meshes])
    lower, upper = all_points.min(0), all_points.max(0)
    center = (lower + upper) / 2
    usd_path = output / 'bulb_grasp.usd'
    stage = Usd.Stage.CreateNew(str(usd_path))
    root = UsdGeom.Xform.Define(stage, '/Bulb').GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root.SetCustomDataByKey('v2_test_object_name', 'bulb')
    body = UsdGeom.Xform.Define(stage, '/Bulb/base_link').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.2)
    material = UsdShade.Material.Define(stage, '/Bulb/Material')
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(0.5)
    physics.CreateDynamicFrictionAttr(0.5)
    physics.CreateRestitutionAttr(0.0)
    triangles = []
    for name, original, points in meshes:
        path = '/Bulb/base_link/' + name
        Sdf.CopySpec(src.GetRootLayer(), original.GetPath(), stage.GetRootLayer(), path)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        prim = mesh.GetPrim()
        # Discard the assembly's SDF and low-friction contact overrides.
        prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([]))
        for prop in list(prim.GetProperties()):
            if prop.GetName().startswith(('physics:', 'physx', 'material:binding', 'xformOp')):
                prim.RemoveProperty(prop.GetName())
        points = points - center
        mesh.GetPointsAttr().Set(points.astype(np.float32).tolist())
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
        counts = original.GetFaceVertexCountsAttr().Get()
        indices = np.asarray(original.GetFaceVertexIndicesAttr().Get())
        offset = 0
        for count in counts:
            face = indices[offset:offset + count]
            triangles.extend(points[[face[0], face[i], face[i + 1]]] for i in range(1, count - 1))
            offset += count
    stage.GetRootLayer().Save()
    tri = np.asarray(triangles)
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(42)
    sampled = tri[rng.choice(len(tri), 8192, p=areas / areas.sum())]
    u, v = rng.random((2, len(sampled)))
    u = np.sqrt(u)
    candidates = ((1-u)[:, None]*sampled[:, 0] + (u*(1-v))[:, None]*sampled[:, 1] + (u*v)[:, None]*sampled[:, 2])
    selected, distance, idx = [], np.full(len(candidates), np.inf), 0
    for _ in range(64):
        selected.append(candidates[idx])
        distance = np.minimum(distance, ((candidates-candidates[idx])**2).sum(1))
        idx = int(distance.argmax())
    pc_dir = output / 'point_clouds_64'
    pc_dir.mkdir(exist_ok=True)
    pc_file = pc_dir / 'bulb.npy'
    np.save(pc_file, np.asarray(selected, dtype=np.float32))
    manifest = {'version': 1, 'shape': [64, 3], 'dtype': 'float32', 'objects': [
        {'name': 'bulb', 'file': pc_file.name, 'file_sha256': hashlib.sha256(pc_file.read_bytes()).hexdigest()}]}
    (pc_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'usd_sha256': hashlib.sha256(usd_path.read_bytes()).hexdigest(),
              'bounds_m': [(-0.5*(upper-lower)).tolist(), (0.5*(upper-lower)).tolist()],
              'source_center_m': center.tolist(), 'collision': 'three component convex hulls',
              'mass_kg': 0.2, 'sample_seed': 42, 'point_cloud': manifest}
    (output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 1
    assert not any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse())
    assert len(np.unique(np.asarray(selected, dtype=np.float32), axis=0)) == 64
    assert not stage.GetRootLayer().GetExternalReferences()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[8]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=repo / 'assets/bulb/hiveboard_lamp/hiveboard_lamp_threaded.usd')
    parser.add_argument('--output', type=Path, default=repo / 'assets/bulb_grasp_teacher')
    args = parser.parse_args()
    build(args.source, args.output)

    if _simulation_app is not None:
        _simulation_app.close()
