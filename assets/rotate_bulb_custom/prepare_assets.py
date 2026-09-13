"""Copy the donor robot and express existing tactile assets in its link frames.

Offline only; requires numpy, scipy and pxr (Isaac Sim or usd-core). Runtime
loads the generated files without importing this script or the donor project.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation


DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
CUSTOM = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom"
CONTRACT = runpy.run_path(str(CUSTOM / "robot_contract.py"))


def relative_frame(stage, name, base):
    cache = UsdGeom.XformCache()
    root = stage.GetDefaultPrim().GetPath()
    link = stage.GetPrimAtPath(root.AppendChild(name))
    parent = stage.GetPrimAtPath(root.AppendChild(base))
    if not link or not parent:
        raise ValueError(f"Missing CAD frame: {name}, {base}")
    return np.array(cache.GetLocalToWorldTransform(link) * cache.GetLocalToWorldTransform(parent).GetInverse()).T


def prepare(source_root):
    source_robot = source_root / "flexiv_rizon_description"
    robot_dir = DEST / "robot"
    shutil.copytree(source_robot / "usd", robot_dir / "usd", dirs_exist_ok=True)
    shutil.copy2(source_robot / "LICENSE.flexiv_description", robot_dir)
    original = Usd.Stage.Open(str(ROOT / "assets/urdf/Tianji_Revo3/urdf/Tianji_Revo3_Right_visual.usd"))
    target = Usd.Stage.Open(str(robot_dir / "usd/rizon4_training.usda"))
    transforms = {}
    for old, new in CONTRACT["TACTILE_LINK_MAP"].items():
        transform = np.linalg.inv(relative_frame(target, new, "right_base_link")) @ relative_frame(original, old, "base_link")
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
        transforms[old] = transform

    # Keep the existing sensor keys and taxel ordering; only link coordinates
    # change. Pressure NPY points are pad-local, so transform their URDF origin.
    pressure_src = ROOT / "assets/revo21_right_touch/pressure_taxels/dv2"
    pressure_dst = DEST / "tactile/pressure"
    pressure_dst.mkdir(parents=True, exist_ok=True)
    for path in pressure_src.glob("*.npy"):
        shutil.copy2(path, pressure_dst / path.name)
    layout = ET.parse(pressure_src / "dv2_pressure_taxel_layout.urdf")
    for link in layout.getroot().findall("link"):
        origin = link.find("pressure_pad/origin")
        pose = np.eye(4)
        pose[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
        pose[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
        pose = transforms[link.get("name")] @ pose
        origin.set("xyz", " ".join(f"{v:.12g}" for v in pose[:3, 3]))
        origin.set("rpy", " ".join(f"{v:.12g}" for v in Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")))
    layout.write(pressure_dst / "dv2_pressure_taxel_layout.urdf", encoding="utf-8", xml_declaration=True)

    tacmap_src = ROOT / "tacmap/assets/tactilesensor_map/revo21_dv2"
    tacmap_dst = DEST / "tactile/tacmap"
    tacmap_dst.mkdir(parents=True, exist_ok=True)
    for path in tacmap_src.glob("*.npy"):
        shutil.copy2(path, tacmap_dst / path.name)
    # Read the old task's literal grid geometry, without importing its config.
    original_setup = CUSTOM.parent / "Revo3/dexsuite_revo3_env_cfg_grasp.py"
    node = next(n for n in ast.parse(original_setup.read_text()).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TIANJI_TACMAP_LINK_SURFACE_DEFAULTS" for t in n.targets))
    grids = ast.literal_eval(node.value)
    for old in grids:
        transform = transforms[old]
        rotation, translation = transform[:3, :3], transform[:3, 3]
        stem = old.removesuffix("_link")
        for kind in ("point", "normal"):
            values = np.load(tacmap_src / f"{stem}_{kind}.npy")
            converted = values @ rotation.T
            if kind == "point":
                converted += translation * 1000.0  # TacMap assets are millimetres.
            np.save(tacmap_dst / f"{stem}_{kind}.npy", converted.astype(values.dtype))
        grids[old]["grid_center"] = (rotation @ grids[old]["grid_center"] + translation).tolist()
        for field in ("ray_axis", "grid_u_axis", "grid_v_axis"):
            axis = grids[old][field]
            direction = rotation[:, "xyz".index(axis[-1])] * (-1 if axis[0] == "-" else 1)
            index = int(np.argmax(np.abs(direction)))
            grids[old][field] = ("+" if direction[index] > 0 else "-") + "xyz"[index]
    (tacmap_dst / "grid_defaults.json").write_text(json.dumps(grids, indent=2) + "\n")

    marker_src = ROOT / "assets/revo21_right_touch/marker_positions/vitai_4fingers"
    marker_dst = DEST / "tactile/markers"
    marker_dst.mkdir(parents=True, exist_ok=True)
    with np.load(marker_src / "marker_positions.npz") as archive:
        values = {key: archive[key].copy() for key in archive.files}
    inverse_links = {new: old for old, new in CONTRACT["TACTILE_LINK_MAP"].items()}
    for finger, new in CONTRACT["FINGER_LINKS"].items():
        transform = transforms[inverse_links[new]]
        rotation, translation = transform[:3, :3], transform[:3, 3]
        for key, data in values.items():
            if not key.startswith(finger + "_"):
                continue
            if key.endswith("_link_m"):
                converted = data @ rotation.T + translation
            elif key.endswith("_rotation_link"):
                converted = rotation @ data
            elif key.endswith(("_normals_link", "_directions_link")):
                converted = data @ rotation.T
            else:
                continue  # Camera coordinates, pixels and marker ordering stay fixed.
            values[key] = converted.astype(data.dtype)
    np.savez_compressed(marker_dst / "marker_positions.npz", **values)
    shutil.copy2(marker_src / "camera_ray_rectangles_320x240.json", marker_dst)

    manifest = {
        "source_project": str(source_root), "source_task": CONTRACT["SOURCE_TASK"],
        "robot_usd": "robot/usd/rizon4_training.usda",
        "self_collisions": True, "source_collision_filters_preserved": True,
        "source_robot_sha256": {
            str(path.relative_to(source_robot)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((source_robot / "usd").rglob("*")) if path.is_file()
        },
        "tactile_frame_convention": "new_link_from_old_link; column vectors; translation in metres; common CAD hand-base frame",
        "tactile_bindings": {
            old: {"body": CONTRACT["TACTILE_LINK_MAP"][old], "new_from_old": matrix.tolist()}
            for old, matrix in transforms.items()
        },
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared robot and {len(transforms)} tactile frame bindings in {DEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    prepare(parser.parse_args().source_root.resolve())
