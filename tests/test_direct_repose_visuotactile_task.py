from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DIRECT_ROOT = (
    REPO_ROOT
    / "source"
    / "BrainCo_DexHand"
    / "BrainCo_DexHand"
    / "tasks"
    / "direct"
)
BRAINCO_ROOT = DIRECT_ROOT / "brainco"
AGENTS_ROOT = BRAINCO_ROOT / "agents"


def test_visuotactile_repose_task_has_independent_entry_points():
    registration = (BRAINCO_ROOT / "__init__.py").read_text(encoding="utf-8")

    assert registration.count('id="BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0"') == 1
    assert "visuotactile_inhand_manipulation_env:VisuotactileInHandManipulationEnv" in registration
    assert "brainco_hand_visuotactile_env_cfg:BrainCoVisuotactileHandEnvCfg" in registration
    assert "visuotactile_rl_games_ppo_cfg.yaml" in registration
    assert "visuotactile_rsl_rl_ppo_cfg:BrainCoVisuotactilePPORunnerCfg" in registration
    assert "visuotactile_skrl_ppo_cfg.yaml" in registration

    # The original state-only task keeps its original environment and configurations.
    original_registration = registration.split(
        'id="BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0"',
        maxsplit=1,
    )[0]
    assert 'id="BrainCo-Direct-Revo3-Repose-Cube-v0"' in original_registration
    assert "inhand_manipulation_env:InHandManipulationEnv" in original_registration
    assert "brainco_hand_env_cfg:BrainCoHandEnvCfg" in original_registration


def test_visuotactile_observation_configuration_is_task_owned():
    cfg_path = BRAINCO_ROOT / "brainco_hand_visuotactile_env_cfg.py"
    cfg_source = cfg_path.read_text(encoding="utf-8")
    cfg_tree = ast.parse(cfg_source)
    class_names = {node.name for node in cfg_tree.body if isinstance(node, ast.ClassDef)}

    assert "BrainCoVisuotactileHandEnvCfg" in class_names
    assert 'visuotactile_observation_terms = ("state", "aligned_tactile_z")' in cfg_source
    assert "observation_history_length = 1" in cfg_source
    assert "state_observation_dim = 152" in cfg_source
    assert "tactile_projection_dim = 64" in cfg_source
    assert "tactile_policy_checkpoint" in cfg_source
    assert "tactile_reconstruction_diagnostics = False" in cfg_source
    assert "num_envs=32" in cfg_source

    calibration_dir = REPO_ROOT / "assets/revo21_right_touch/marker_positions/vitai_4fingers"
    assert (calibration_dir / "marker_positions.npz").is_file()
    assert (calibration_dir / "camera_ray_rectangles_320x240.json").is_file()


def test_visuotactile_environment_materializes_all_modalities_and_history():
    env_path = DIRECT_ROOT / "visuotactile_inhand_manipulation_env.py"
    env_source = env_path.read_text(encoding="utf-8")
    ast.parse(env_source)

    encoder_path = REPO_ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tactile_representation/policy.py"
    encoder_source = encoder_path.read_text(encoding="utf-8")
    ast.parse(encoder_source)

    assert "class VisuotactileInHandManipulationEnv(InHandManipulationEnv)" in env_source
    assert "DirectVisuotactileRuntime(self)" in env_source
    assert "FrozenTactilePolicyEncoder" in env_source
    assert "TactileObservationHistory" in env_source
    assert "FrozenFiveFingerResNet18" not in env_source
    assert "self.requires_grad_(False)" in encoder_source
    assert env_source.index("cfg.observation_space =") < env_source.index("super().__init__(cfg")
    runtime = (BRAINCO_ROOT / "visuotactile_runtime.py").read_text()
    assert "ours_rl_hydroshear_obs" in runtime
    assert "ours_rl_taxim_rgb_obs" in runtime
    assert "prepare_direct_calibration" in runtime


def test_visuotactile_agent_configs_are_independent_without_changing_the_mlp():
    rsl_source = (AGENTS_ROOT / "visuotactile_rsl_rl_ppo_cfg.py").read_text(encoding="utf-8")
    rl_games_source = (AGENTS_ROOT / "visuotactile_rl_games_ppo_cfg.yaml").read_text(encoding="utf-8")
    skrl_source = (AGENTS_ROOT / "visuotactile_skrl_ppo_cfg.yaml").read_text(encoding="utf-8")

    assert "class BrainCoVisuotactilePPORunnerCfg" in rsl_source
    assert "actor_hidden_dims=[1024, 512, 256, 128]" in rsl_source
    assert "critic_hidden_dims=[1024, 512, 256, 128]" in rsl_source
    assert "learning_rate=5.0e-4" in rsl_source
    assert "units: [1024, 512, 256, 128]" in rl_games_source
    assert "layers: [1024, 512, 256, 128]" in skrl_source
    assert "brainco_repose_cube_visuotactile" in rsl_source
    assert "brainco_repose_cube_visuotactile" in rl_games_source
    assert "brainco_repose_cube_visuotactile" in skrl_source
