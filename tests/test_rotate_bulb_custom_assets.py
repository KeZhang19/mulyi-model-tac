"""Offline checks for the asset port; USD checks also run with standalone usd-core."""

import ast
import hashlib
import json
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DEX = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite"
CUSTOM = DEX / "config/RotateBulbCustom"
ASSETS = ROOT / "assets/rotate_bulb_custom"
CONTRACT = runpy.run_path(str(CUSTOM / "robot_contract.py"))
MANIFEST = json.loads((ASSETS / "manifest.json").read_text())


def test_robot_files_are_exact_donor_copies_including_collision_filters():
    files = MANIFEST["source_robot_sha256"]
    assert "usd/configuration/rizon4_collision_filters.usda" in files
    for name, digest in files.items():
        assert hashlib.sha256((ASSETS / "robot" / name).read_bytes()).hexdigest() == digest


def test_usd_dependencies_joint_limits_and_sensor_bodies_resolve():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdPhysics, UsdUtils

    path = ASSETS / MANIFEST["robot_usd"]
    layers, external, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    assert len(layers) == 7
    assert not external
    assert set(unresolved) <= {"OmniPBR.mdl"}  # Isaac's built-in material module.
    for layer in layers:
        assert Path(layer.realPath).is_relative_to(ASSETS)
    stage = Usd.Stage.Open(str(path))
    joints = {p.GetName(): UsdPhysics.RevoluteJoint(p) for p in stage.Traverse()
              if p.IsA(UsdPhysics.RevoluteJoint)}
    assert set(joints) == set(CONTRACT["JOINT_NAMES"]) == set(CONTRACT["INITIAL_JOINT_POS"])
    assert len(joints) == 28
    for name, joint in joints.items():
        angle = np.rad2deg(CONTRACT["INITIAL_JOINT_POS"][name])
        assert joint.GetLowerLimitAttr().Get() <= angle <= joint.GetUpperLimitAttr().Get(), name
    root = stage.GetDefaultPrim()
    for name in set(CONTRACT["TACTILE_LINK_MAP"].values()) | set(CONTRACT["HAND_POINT_NAMES"]):
        body = root.GetChild(name)
        assert body and body.HasAPI(UsdPhysics.RigidBodyAPI), name
    articulation_roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(articulation_roots) == 1
    assert articulation_roots[0].GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert any(p.GetRelationship("physics:filteredPairs").GetTargets() for p in stage.Traverse())


def _pad_points(layout_path):
    result = {}
    for link in ET.parse(layout_path).getroot().findall("link"):
        pad = link.find("pressure_pad")
        origin = pad.find("origin")
        rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
        translation = np.fromstring(origin.get("xyz"), sep=" ")
        points = np.load(layout_path.parent / pad.get("points_npy")) * float(pad.get("correction_scale"))
        normals = np.load(layout_path.parent / pad.get("normals_npy"))
        result[link.get("name")] = (points @ rotation.T + translation, normals @ rotation.T)
    return result


def test_pressure_taxel_count_order_and_hand_frame_positions_are_preserved():
    old = _pad_points(ROOT / "assets/revo21_right_touch/pressure_taxels/dv2/dv2_pressure_taxel_layout.urdf")
    new = _pad_points(ASSETS / "tactile/pressure/dv2_pressure_taxel_layout.urdf")
    assert list(new) == list(old)
    assert sum(points.reshape(-1, 3).shape[0] for points, _ in new.values()) == 285
    for name, (points, normals) in old.items():
        matrix = np.array(MANIFEST["tactile_bindings"][name]["new_from_old"])
        np.testing.assert_allclose(new[name][0], points @ matrix[:3, :3].T + matrix[:3, 3], atol=1e-8)
        np.testing.assert_allclose(new[name][1], normals @ matrix[:3, :3].T, atol=1e-8)


def test_tactile_points_markers_and_camera_coordinates_remain_consistent():
    old_markers = ROOT / "assets/revo21_right_touch/marker_positions/vitai_4fingers"
    with np.load(old_markers / "marker_positions.npz") as old, np.load(ASSETS / "tactile/markers/marker_positions.npz") as new:
        assert old.files == new.files
        for key in old.files:
            assert old[key].shape == new[key].shape, key
            if "_link" not in key:
                np.testing.assert_array_equal(old[key], new[key])
        reverse = {new_name: old_name for old_name, new_name in CONTRACT["TACTILE_LINK_MAP"].items()}
        for finger, body in CONTRACT["FINGER_LINKS"].items():
            matrix = np.array(MANIFEST["tactile_bindings"][reverse[body]]["new_from_old"])
            points = new[finger + "_points_link_m"]
            assert points.shape == (100, 3)
            np.testing.assert_allclose(points, old[finger + "_points_link_m"] @ matrix[:3, :3].T + matrix[:3, 3], atol=1e-8)
            # Independent geometric check: link-frame conversion must leave the
            # camera-frame marker coordinates and calibrated viewing rays fixed.
            for prefix in ("_points", "_ray_starts", "_marker_ray_starts"):
                before = (old[finger + prefix + "_link_m"] - old[finger + "_camera_origin_link_m"]) @ old[finger + "_camera_rotation_link"]
                after = (new[finger + prefix + "_link_m"] - new[finger + "_camera_origin_link_m"]) @ new[finger + "_camera_rotation_link"]
                np.testing.assert_allclose(after, before, atol=2e-8)
            stem = reverse[body].removesuffix("_link")
            for kind in ("point", "normal"):
                original = np.load(ROOT / f"tacmap/assets/tactilesensor_map/revo21_dv2/{stem}_{kind}.npy")
                copied = np.load(ASSETS / f"tactile/tacmap/{stem}_{kind}.npy")
                expected = original @ matrix[:3, :3].T
                if kind == "point":
                    expected += 1000 * matrix[:3, 3]
                np.testing.assert_allclose(copied, expected, atol=5e-6)
    camera_file = "camera_ray_rectangles_320x240.json"
    assert (old_markers / camera_file).read_bytes() == (ASSETS / "tactile/markers" / camera_file).read_bytes()


def test_asset_swap_retains_original_actions_observation_definitions_and_rewards():
    def definitions(path):
        nodes = ast.parse(path.read_text()).body
        if path == CUSTOM / "env_cfg.py":
            for node in nodes:
                if isinstance(node, ast.ClassDef) and node.name in {
                    "DexsuiteRevo3RotateBulbEnvCfg", "DexsuiteRevo3RotateBulbEnvCfg_PLAY",
                }:
                    # The Custom-only opt-in is the sole additional environment
                    # field; continue comparing every other default and method.
                    node.body = [n for n in node.body if not (
                        isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                        and n.target.id == "tactile_policy_enabled")]
        return {n.name: ast.dump(n, include_attributes=False) for n in nodes
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    original_robot = definitions(DEX / "config/Revo3/dexsuite_revo3_env_cfg_grasp.py")
    custom_robot = definitions(CUSTOM / "robot_setup.py")
    assert custom_robot["Revo3RelJointPosActionCfg"] == original_robot["Revo3RelJointPosActionCfg"]
    original_base = definitions(DEX / "dexsuite_env_cfg_grasp_tianji.py")
    custom_base = definitions(CUSTOM / "base_env_cfg.py")
    assert custom_base == original_base
    original_env = definitions(DEX / "config/Revo3/dexsuite_revo3_env_cfg_rotate_bulb.py")
    custom_env = definitions(CUSTOM / "env_cfg.py")
    # Custom additionally has independent turn rewards and episode metrics.
    for name in original_env.keys() - {"configure_rotate_bulb_scene", "reset_lamp_on_table",
                                       "release_unscrewed_bulb", "RotateBulbPoseCommand",
                                       "DexsuiteRevo3RotateBulbRewardCfg"}:
        assert custom_env[name] == original_env[name], name
    # This port deliberately excludes the donor's action/observation modules.
    for path in CUSTOM.rglob("*.py"):
        assert "flexiv_v5" not in path.read_text()
