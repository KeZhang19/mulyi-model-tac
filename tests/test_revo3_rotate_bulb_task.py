from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REVO3_CONFIG_ROOT = (
    REPO_ROOT
    / "source"
    / "BrainCo_DexHand"
    / "BrainCo_DexHand"
    / "tasks"
    / "manager_based"
    / "dexsuite"
    / "config"
    / "Revo3"
)


def _class_node(path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)


def test_rotate_bulb_task_uses_its_own_environment_and_agent_configs():
    registration = (REVO3_CONFIG_ROOT / "__init__.py").read_text(encoding="utf-8")

    assert 'id="BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0"' in registration
    assert "dexsuite_revo3_env_cfg_rotate_bulb:DexsuiteRevo3RotateBulbEnvCfg" in registration
    assert "rl_games_ppo_cfg_rotate_bulb.yaml" in registration
    assert "rsl_rl_ppo_cfg_rotate_bulb:DexsuiteRevo3RotateBulbPPORunnerCfg" in registration
    assert "rsl_rl_distillation_cfg_rotate_bulb:" in registration
    assert "DexsuiteRevo3RotateBulbDistillationRunnerCfg" in registration


def test_rotate_bulb_scene_and_rewards_are_task_owned():
    env_cfg_path = REVO3_CONFIG_ROOT / "dexsuite_revo3_env_cfg_rotate_bulb.py"
    tree = ast.parse(env_cfg_path.read_text(encoding="utf-8"))
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert {
        "DexsuiteRevo3RotateBulbSceneCfg",
        "DexsuiteRevo3RotateBulbRewardCfg",
        "DexsuiteRevo3RotateBulbEnvCfg",
        "DexsuiteRevo3RotateBulbEnvCfg_PLAY",
    } <= class_names
    assert {
        "rotate_bulb_hand_contacts",
        "rotate_bulb_position_command_error_tanh",
        "rotate_bulb_orientation_command_error_tanh",
    } <= function_names

    reward_cfg = _class_node(env_cfg_path, "DexsuiteRevo3RotateBulbRewardCfg")
    reward_source = ast.unparse(reward_cfg)
    assert "weight=1.0" in reward_source
    assert "func=RotateBulbUnscrewProgress" in reward_source
    assert "func=RotateBulbStableSuccess" in reward_source
    assert "orientation_tracking = None" in reward_source
    assert "func=mdp.success_reward" not in reward_source
    assert "'std': 0.2" in reward_source
    assert "'pos_std': 0.03" in reward_source


def test_rotate_bulb_model_and_training_parameters_start_from_lift_values():
    agents_root = REVO3_CONFIG_ROOT / "agents"
    ppo_cfg = (agents_root / "rsl_rl_ppo_cfg_rotate_bulb.py").read_text(encoding="utf-8")
    distillation_cfg = (agents_root / "rsl_rl_distillation_cfg_rotate_bulb.py").read_text(encoding="utf-8")
    rl_games_cfg = (agents_root / "rl_games_ppo_cfg_rotate_bulb.yaml").read_text(encoding="utf-8")

    assert "actor_hidden_dims=[512, 256, 128]" in ppo_cfg
    assert "critic_hidden_dims=[512, 256, 128]" in ppo_cfg
    assert "learning_rate=1.0e-3" in ppo_cfg
    assert "student_hidden_dims=[256, 256, 128]" in distillation_cfg
    assert "teacher_hidden_dims=[1024, 512, 256, 128]" in distillation_cfg
    assert "units: [512, 256, 128]" in rl_games_cfg
    assert "name: tianji_revo3_rotate_bulb" in rl_games_cfg
