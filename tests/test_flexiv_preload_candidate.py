"""Regression checks for the Flexiv pregrasp preload candidate."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "assets/d_peg_insertion/tools/tune_flexiv_preload.py"
SPEC = importlib.util.spec_from_file_location("tune_flexiv_preload", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_preload_keeps_measured_pose_and_adds_finger_only_targets(tmp_path):
    source = ROOT / "assets/d_peg_insertion/pregrasp_flexiv.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    output = tmp_path / "candidate.json"
    result = MODULE.tune(source, output)
    assert result["robot_joint_positions"] == original["robot_joint_positions"]
    assert result["peg_pose"] == original["peg_pose"]
    assert result["socket_pose"] == original["socket_pose"]
    for name in (f"joint{i}" for i in range(1, 8)):
        assert result["robot_joint_targets"][name] == result["robot_joint_positions"][name]
    assert result["robot_joint_targets"]["right_index_MCP_joint"] > result["robot_joint_positions"]["right_index_MCP_joint"]
    assert result["validation"]["status"] == "preload_candidate"
    assert result["validation"]["physics_validated"] is False
