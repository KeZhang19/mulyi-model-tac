"""Test editor-to-training pose conversion and rejection of incompatible exports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/d_peg_insertion"
REFERENCE = ASSETS / "flexiv_grasp_reference/20260913_142053_9886dc87/simulation_state.json"
spec = importlib.util.spec_from_file_location("flexiv_import", ASSETS / "tools/import_flexiv_pregrasp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def destination(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("peg.usd", "socket.usd", "metadata.json"):
        shutil.copyfile(ASSETS / name, assets / name)
    (assets / "pregrasp_flexiv.json").write_text('{"previous": true}\n')
    return assets


def test_import_preserves_named_joints_exact_poses_and_failed_validation(destination):
    value = module.import_flexiv_pregrasp(REFERENCE, destination)
    exported = json.loads(REFERENCE.read_text())
    assert value["robot_joint_positions"] == dict(zip(exported["robot"]["joint_names"], exported["robot"]["joint_positions_rad"]))
    assert value["robot_joint_targets"] == value["robot_joint_positions"]
    assert value["peg_pose"]["position"] == exported["object"]["position_m"]
    assert value["socket_pose"]["quaternion"] == exported["fixtures"][0]["quaternion_wxyz"]
    assert value["source_grasp_reference_sha256"] == hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    assert not value["validated"] and not value["validation"]["physics_validated"]
    assert not value["validation"]["source_export_pose_candidate_pass"]
    assert value["validation"]["optimization"] == exported["validation"]["optimization"]
    archive = destination / value["source_grasp_reference"]
    assert archive.read_bytes() == REFERENCE.read_bytes()
    assert json.loads((archive.parent / "previous_pregrasp_flexiv.json").read_text()) == {"previous": True}


@pytest.mark.parametrize("problem", ["wrong_joint", "wrong_task", "scaled_peg", "bad_quaternion", "changed_geometry", "wrong_base"])
def test_incompatible_export_is_rejected_before_replacing_default(destination, tmp_path, problem):
    value = copy.deepcopy(json.loads(REFERENCE.read_text()))
    if problem == "wrong_joint":
        value["robot"]["joint_names"][0] = "Joint1_R"
    elif problem == "wrong_task":
        value["assembly"]["source_configuration"]["task"] = "bulb"
    elif problem == "scaled_peg":
        value["object"]["mesh_scale_xyz"] = [.001] * 3
    elif problem == "bad_quaternion":
        value["object"]["quaternion_wxyz"] = [2., 0., 0., 0.]
    elif problem == "wrong_base":
        value["robot"]["base"]["position_m"][0] += 1.
    else:
        (destination / "peg.usd").write_text("different geometry")
    path = tmp_path / "source.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.import_flexiv_pregrasp(path, destination)
    assert json.loads((destination / "pregrasp_flexiv.json").read_text()) == {"previous": True}
