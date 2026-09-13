#!/usr/bin/env python3
"""Validate the portable and locked geometry contract of assets/bulb."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from isaacsim import SimulationApp


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def validate_urdf(path: Path):
    root = ET.parse(path).getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    if len(links) != 1 or joints:
        raise AssertionError(f"{path}: expected one link and zero joints")
    for mesh in root.findall(".//mesh"):
        target = (path.parent / mesh.attrib["filename"]).resolve()
        if not target.is_file():
            raise AssertionError(f"{path}: missing mesh {target}")


def validate_usd(path: Path, expected_default_prim: str):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise AssertionError(f"Could not open {path}")
    root = stage.GetDefaultPrim()
    if str(root.GetPath()) != expected_default_prim:
        raise AssertionError(f"{path}: default prim is {root.GetPath()}")
    if root.GetCustomDataByKey("geometryLocked") is not True:
        raise AssertionError(f"{path}: geometryLocked metadata is missing")

    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    rigid_bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    collisions = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if joints:
        raise AssertionError(f"{path}: expected zero joints, found {len(joints)}")
    if rigid_bodies != [root]:
        raise AssertionError(f"{path}: default prim is not the only rigid body")
    if not collisions:
        raise AssertionError(f"{path}: no collision geometry")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise AssertionError(f"{path}: units are not metres")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise AssertionError(f"{path}: up axis is not Z")

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    if len(layers) != 1 or assets or unresolved:
        raise AssertionError(f"{path}: USD is not self-contained")


def main():
    manifest = json.loads((PACKAGE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    app = SimulationApp({"headless": True})
    try:
        for name, asset in manifest["assets"].items():
            usd_path = PACKAGE_ROOT / asset["recommended_usd"]
            urdf_path = PACKAGE_ROOT / asset["urdf"]
            validate_usd(usd_path, asset["default_prim"])
            validate_urdf(urdf_path)
            print(f"validated {name}: {usd_path.relative_to(PACKAGE_ROOT)}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
