"""Static contract checks for the isolated Flexiv D-peg task."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/InsertDPegCustom"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_task_registration_is_new_and_isolated():
    source = _source(TASK_DIR / "__init__.py")
    assert 'id="BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0"' in source
    assert "DexsuiteFlexivInsertDPegEnvCfg" in source
    assert "DexsuiteFlexivInsertDPegPPORunnerCfg" in source
    assert "Revo3-Right-Insert-D-Peg-v0" not in source


def test_robot_asset_and_pregrasp_use_rotate_bulb_flexiv_contract():
    asset = _source(ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/robot_asset_cfg.py")
    env = _source(TASK_DIR / "env_cfg.py")
    assert "rizon4_training.usda" in asset
    assert "pregrasp_flexiv.json" in env
    assert "ROBOT_POSITION" in env and "ROBOT_ORIENTATION" in env
    assert "GRASP_REFERENCE[\"table\"]" in env


def test_flexiv_pregrasp_has_exact_runtime_joint_inventory():
    data = json.loads((ROOT / "assets/d_peg_insertion/pregrasp_flexiv.json").read_text(encoding="utf-8"))
    positions = data["robot_joint_positions"]
    targets = data["robot_joint_targets"]
    assert len(positions) == len(targets) == 28
    assert set(positions) == set(targets)
    assert {f"joint{i}" for i in range(1, 8)} <= set(positions)
    assert "right_thumb_CMP_joint" in positions
    assert "right_little_DIP_joint" in positions
    assert set(data["peg_pose"]) == {"position", "quaternion"}
    assert set(data["socket_pose"]) == {"position", "quaternion"}
    assert len(data["peg_pose"]["position"]) == 3
    assert len(data["socket_pose"]["quaternion"]) == 4


def test_ppo_config_is_task_owned():
    tree = ast.parse(_source(TASK_DIR / "agents/rsl_rl_ppo_cfg.py"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    runner = next(node for node in classes if node.name == "DexsuiteFlexivInsertDPegPPORunnerCfg")
    names = {node.targets[0].id: node.value for node in runner.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert ast.literal_eval(names["experiment_name"]) == "dexsuite_flexiv_insert_d_peg_custom"
    assert ast.literal_eval(names["max_iterations"]) == 15000
    assert ast.literal_eval(names["save_interval"]) == 250
