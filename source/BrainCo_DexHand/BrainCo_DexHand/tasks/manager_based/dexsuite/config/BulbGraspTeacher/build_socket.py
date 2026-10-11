"""CPU-only extraction of the original socket, preserving both collision meshes."""
from pathlib import Path
import hashlib
import json

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def build_socket(source: Path, output: Path):
    src = Usd.Stage.Open(str(source))
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, "/Socket").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    physics = UsdPhysics.RigidBodyAPI.Apply(root)
    physics.CreateKinematicEnabledAttr(True)
    for name in ("Support", "Materials"):
        Sdf.CopySpec(src.GetRootLayer(), f"/HiveboardLampThreaded/{name}",
                     stage.GetRootLayer(), f"/Socket/{name}")
    # CopySpec preserves absolute material targets; remap within this derivative.
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if targets:
                rel.SetTargets([p.ReplacePrefix(Sdf.Path("/HiveboardLampThreaded"), Sdf.Path("/Socket"))
                                for p in targets])
    stage.GetRootLayer().Save()
    report = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "socket_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "source_prim": "/HiveboardLampThreaded/Support",
              "thread_direction_bulb_local": [-1, 0, 0],
              "socket_mouth_x_m": -3.2485189876751974e-05,
              "socket_bottom_x_m": -0.044732484966516495,
              "collision": "original socket triangle mesh; kinematic rigid body; no bulb or joints"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[8]
    print(json.dumps(build_socket(root / "assets/bulb/hiveboard_lamp/hiveboard_lamp_threaded.usd",
                                  root / "assets/bulb_grasp_teacher/socket.usd"), indent=2))
