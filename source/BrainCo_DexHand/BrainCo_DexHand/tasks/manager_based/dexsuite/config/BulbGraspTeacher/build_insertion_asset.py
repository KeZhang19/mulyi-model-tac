"""Build a free bulb with the source assembly's actual SDF collision surfaces.

Keep the centered grasp visual geometry, mass/material, and point cloud. No
articulation, mimic joint, pose constraint, or enlarged socket is introduced.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def build_insertion_asset(root: Path):
    source = root / "assets/bulb/hiveboard_lamp/hiveboard_lamp_threaded.usd"
    folder = root / "assets/bulb_grasp_teacher"
    output = folder / "bulb_insert.usd"
    metadata = json.loads((folder / "metadata.json").read_text())
    center = np.asarray(metadata["source_center_m"])
    src = Usd.Stage.Open(str(source))
    grasp = Usd.Stage.Open(str(folder / "bulb_grasp.usd"))
    grasp.GetRootLayer().Export(str(output))
    stage = Usd.Stage.Open(str(output))
    # The visible thread is replaced by its closed STL-derived collision mesh.
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath("/Bulb/base_link/ExternalThread")).GetCollisionEnabledAttr().Set(False)
    material = UsdShade.Material(stage.GetPrimAtPath("/Bulb/Material"))
    for name in ("ExternalThreadCollider", "ShellA", "ShellB"):
        path = f"/Bulb/base_link/{name}"
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
        Sdf.CopySpec(src.GetRootLayer(), f"/HiveboardLampThreaded/Bulb/{name}", stage.GetRootLayer(), path)
        prim = stage.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64) - center
        mesh.GetPointsAttr().Set(points.astype(np.float32).tolist())
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
        for prop in list(prim.GetProperties()):
            if prop.GetName().startswith("material:binding"):
                prim.RemoveProperty(prop.GetName())
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()
    thread = np.asarray(UsdGeom.Mesh(src.GetPrimAtPath("/HiveboardLampThreaded/Bulb/ExternalThreadCollider")).GetPointsAttr().Get())
    tip_x = float(thread[:, 0].min())
    tip_offset = np.asarray([tip_x, 0., 0.]) - center
    report = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "asset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "bulb_tip_offset_m": tip_offset.tolist(), "source_thread_tip_x_m": tip_x,
              "reference_extension_m": .0243, "reference_spin_rad": float(np.pi), "pitch_m": .006,
              "collision": "Original male closed-thread SDF512 and shell SDF256; original female triangles unchanged",
              "free_rigid_body": True, "no_guiding_joints": True}
    (folder / "insertion_geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_insertion_asset(Path(__file__).resolve().parents[8]), indent=2))
