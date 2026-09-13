"""Compare the saved editor grasp against independent USD joint-frame FK.

These checks do not launch Isaac Sim and do not reuse the task's reset math.
USD's float32 joint frames and the editor's serialized CAD axes account for
the micrometre / 2e-5 rad tolerances; joint targets themselves stay exact.
"""

import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/rotate_bulb_custom"
CUSTOM = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom"
REFERENCE = json.loads((ASSETS / "grasp_reference/simulation_state.json").read_text())


def _pose_matrix(position, quaternion_wxyz):
    matrix = np.eye(4)
    matrix[:3, 3] = position
    matrix[:3, :3] = Rotation.from_quat(np.roll(quaternion_wxyz, -1)).as_matrix()
    return matrix


def _usd_joint_frame(position, quaternion):
    return _pose_matrix(position, [quaternion.GetReal(), *quaternion.GetImaginary()])


def _usd_forward_kinematics(stage, angles, root_pose):
    from pxr import UsdPhysics

    poses = {"base_link": root_pose}
    pending = [UsdPhysics.Joint(prim) for prim in stage.Traverse()
               if prim.IsA(UsdPhysics.Joint) and prim.GetName() != "root_joint"]
    while pending:
        previous_count = len(pending)
        for joint in pending.copy():
            parent = joint.GetBody0Rel().GetTargets()[0].name
            child = joint.GetBody1Rel().GetTargets()[0].name
            if parent not in poses:
                continue
            motion = np.eye(4)
            if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
                # The donor's arbitrary CAD axes use localRot0/localRot1 and
                # an empty token, which the PhysX USD importer treats as X.
                axis_name = UsdPhysics.RevoluteJoint(joint).GetAxisAttr().Get() or "X"
                axis = np.eye(3)["XYZ".index(axis_name)]
                motion[:3, :3] = Rotation.from_rotvec(axis * angles[joint.GetPrim().GetName()]).as_matrix()
            frame0 = _usd_joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            frame1 = _usd_joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            poses[child] = poses[parent] @ frame0 @ motion @ np.linalg.inv(frame1)
            pending.remove(joint)
        assert len(pending) < previous_count, "Robot USD joint tree is disconnected or cyclic"
    return poses


def test_exported_grasp_joint_mapping_and_base_are_exact():
    contract = runpy.run_path(str(CUSTOM / "robot_contract.py"))
    robot = REFERENCE["robot"]
    expected = dict(zip(robot["joint_names"], robot["joint_positions_rad"], strict=True))
    assert len(expected) == 28
    assert contract["INITIAL_JOINT_POS"] == expected
    np.testing.assert_array_equal(contract["ROBOT_POSITION"], robot["base"]["position_m"])
    np.testing.assert_array_equal(contract["ROBOT_ORIENTATION"], robot["base"]["quaternion_wxyz"])


def test_exported_grasp_matches_robot_usd_forward_kinematics():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdPhysics

    robot = REFERENCE["robot"]
    angles = dict(zip(robot["joint_names"], robot["joint_positions_rad"], strict=True))
    stage = Usd.Stage.Open(str(ASSETS / "robot/usd/rizon4_training.usda"))
    assert stage
    joints = {prim.GetName(): UsdPhysics.RevoluteJoint(prim) for prim in stage.Traverse()
              if prim.IsA(UsdPhysics.RevoluteJoint)}
    assert set(joints) == set(angles)
    for name, joint in joints.items():
        assert joint.GetLowerLimitAttr().Get() <= np.rad2deg(angles[name]) <= joint.GetUpperLimitAttr().Get(), name
    base = robot["base"]
    poses = _usd_forward_kinematics(stage, angles, _pose_matrix(base["position_m"], base["quaternion_wxyz"]))
    aliases = {"right_hand_base_link": "right_base_link", "right_palm": "palm"}
    # The editor splits its calibrated connector into two extra fixed frames;
    # the USD has a different connector CAD frame. Compare the flange and
    # wrist on either side, plus every arm/hand link and all five fingertips.
    connector_frames = {"right_connector_link", "right_connector_tip"}
    checked = []
    skipped = set()
    for link in robot["links"]:
        name = link["body_name"].removeprefix("hand-")
        if name in connector_frames:
            skipped.add(name)
            continue
        name = aliases.get(name, name)
        actual = poses[name]
        expected = _pose_matrix(link["position_m"], link["quaternion_wxyz"])
        assert np.linalg.norm(actual[:3, 3] - expected[:3, 3]) < 2e-6, name
        # Rotation matrices compare q and -q as the same physical orientation.
        angle_error = Rotation.from_matrix(actual[:3, :3].T @ expected[:3, :3]).magnitude()
        assert angle_error < 2e-5, name
        checked.append(name)
    assert len(checked) == 37
    assert skipped == connector_frames
    assert {"flange", "right_base_link", "palm"}.issubset(checked)
    assert sum(name.endswith("_tip_Link") for name in checked) == 5


def test_exported_lamp_geometry_is_the_task_source_usd():
    source = REFERENCE["assembly"]["source_configuration"]
    usd_path = ROOT / "assets/bulb/hiveboard_lamp/hiveboard_lamp_threaded.usd"
    assert hashlib.sha256(usd_path.read_bytes()).hexdigest() == source["source_usd_sha256"]


def test_exported_support_and_bulb_match_screw_geometry():
    source = REFERENCE["assembly"]["source_configuration"]
    support, = [fixture for fixture in REFERENCE["fixtures"] if fixture["name"] == "lamp_support"]
    bulb = REFERENCE["object"]
    screw_turn = -2 * np.pi * source["initial_screw_turns"]
    screw_slide = -source["pitch_m"] * source["initial_screw_turns"]
    np.testing.assert_allclose(screw_turn, source["screw_turn_rad"], atol=1e-14, rtol=0)
    np.testing.assert_allclose(screw_slide, source["screw_slide_m"], atol=1e-14, rtol=0)
    relative = np.eye(4)
    relative[0, 3] = source["screw_exit_extension_m"] + screw_slide
    relative[:3, :3] = Rotation.from_rotvec([np.pi + screw_turn, 0, 0]).as_matrix()
    world_from_support = _pose_matrix(support["position_m"], support["quaternion_wxyz"])
    world_from_bulb = _pose_matrix(bulb["position_m"], bulb["quaternion_wxyz"])
    np.testing.assert_allclose(world_from_support @ relative, world_from_bulb, atol=1e-12, rtol=0)
