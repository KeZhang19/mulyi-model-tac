"""Custom tactile opt-in must skip expensive work and guard policy compatibility."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from test_rotate_bulb_tactile_encoder import file_sha256, prepare_policy_run


ROOT = Path(__file__).resolve().parents[1]
DEX = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite"
CUSTOM = DEX / "config/RotateBulbCustom"


@pytest.fixture
def runtime():
    class Term:
        def __init__(self, cfg, env):
            self._env = env

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled tactile observations performed expensive work")

    path = CUSTOM / "tactile.py"
    definitions = [n for n in ast.parse(path.read_text()).body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    ns = dict(torch=torch, copy=copy, Path=Path, ManagerTermBase=Term,
              file_sha256=file_sha256, load_sim_policy_encoder=forbidden,
              ensure_taxim_scatter=forbidden,
              tactile_obs=SimpleNamespace(invalidate_ours_tactile_cache_on_reset=Mock()))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), ns)
    ns["observe_rotate_bulb_tactile"] = forbidden
    return ns


def environment():
    return SimpleNamespace(
        num_envs=2, device="cpu", common_step_counter=0,
        cfg=SimpleNamespace(tactile_policy_enabled=False,
                            tactile_policy_checkpoint="missing-checkpoint.pt",
                            tactile_encoder_chunk_size=7, tactile_taxim_chunk_size=3,
                            tactile_reconstruction_diagnostics=False),
    )


@pytest.mark.parametrize("name", ["DexsuiteRevo3RotateBulbEnvCfg", "DexsuiteRevo3RotateBulbEnvCfg_PLAY"])
def test_train_and_play_default_to_disabled(name):
    tree = ast.parse((CUSTOM / "env_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    default = next(n.value for n in cls.body if isinstance(n, ast.AnnAssign)
                   and n.target.id == "tactile_policy_enabled")
    assert ast.literal_eval(default) is False


@pytest.mark.parametrize("diagnostics", [False, True])
def test_disabled_skips_model_assets_sensors_and_diagnostics_across_steps_and_resets(runtime, diagnostics):
    env = environment()
    env.cfg.tactile_reconstruction_diagnostics = diagnostics
    # No valid checkpoint or calibration assets are needed with the pathway off.
    term = runtime["RotateBulbPretrainedTactile"](SimpleNamespace(params={"component_cfg": {}}), env)
    assert term.encoder is None and env._rotate_bulb_tactile_term is term
    for ids in (None, torch.tensor([1])):
        empty = term(env, {})
        assert empty.shape == (2, 0) and empty.dtype == torch.float32
        remaining = torch.randn(2, 31)
        torch.testing.assert_close(torch.cat((remaining, empty), dim=1), remaining)
        env.common_step_counter += 1
        term.reset(ids)
        assert term(env, {}) is empty
    assert runtime["tactile_obs"].invalidate_ours_tactile_cache_on_reset.call_count == 2
    assert env.latest_tactile_inputs is env.latest_tactile_latent is env.latest_tactile_reconstruction is None


@pytest.mark.parametrize("diagnostics", [False, True])
def test_enable_after_config_creation_restores_features_caching_and_selected_reset(runtime, diagnostics):
    env = environment()
    env.cfg.tactile_policy_enabled = True  # Like a late Hydra override, before manager creation.
    env.cfg.tactile_reconstruction_diagnostics = diagnostics
    env._rl_ours_taxim_rgb_background_cache = {"real": torch.zeros(240, 320, 3)}
    reset_ids = []
    env._brainco_rl_hydroshear_adapter = SimpleNamespace(
        _batch_state_valid=torch.ones(10, dtype=torch.bool), _reset_hydrosoft_state=reset_ids.append)
    latent = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    encoder = Mock(normalization=SimpleNamespace(image_height=240, image_width=320, marker_count=100),
                   projection_dim=4, feature_mode="unaligned_sim", bundle_sha256="test-sha",
                   return_value=latent)
    encoder.to.return_value = encoder
    runtime["load_sim_policy_encoder"] = Mock(return_value=encoder)
    runtime["ensure_taxim_scatter"] = Mock()
    sample = {"rgb": torch.zeros(2, 5, 3, 240, 320, dtype=torch.uint8)}
    observe = runtime["observe_rotate_bulb_tactile"] = Mock(return_value=sample)
    original_cfg = {"hydroshear_cache_debug_output": True}
    term = runtime["RotateBulbPretrainedTactile"](
        SimpleNamespace(params={"component_cfg": original_cfg}), env)
    runtime["load_sim_policy_encoder"].assert_called_once_with("missing-checkpoint.pt", chunk_size=7)
    runtime["ensure_taxim_scatter"].assert_called_once_with()
    assert original_cfg == {"hydroshear_cache_debug_output": True}
    assert term.component_cfg["taxim_rgb_render_chunk_size"] == 3
    assert term.component_cfg["hydroshear_cache_debug_output"] is False
    first = term(env, original_cfg)
    torch.testing.assert_close(first, latent.flatten(1))
    assert term(env, original_cfg) is first and observe.call_count == 1
    env.common_step_counter += 1
    term(env, original_cfg)
    term.reset(torch.tensor([1]))
    assert reset_ids == list(range(5, 10))
    term(env, original_cfg)
    assert observe.call_count == encoder.call_count == 3
    assert encoder.reconstruct.call_count == (3 if diagnostics else 0)
    # Reset epoch invalidation also breaks a cache inside the same control step.
    env._rl_ours_tactile_reset_epoch = 1
    term(env, original_cfg)
    assert observe.call_count == 4
    if not diagnostics:
        assert env.latest_tactile_reconstruction is None
    encoder.return_value = torch.full_like(latent, float("nan"))
    env.common_step_counter += 1
    with pytest.raises(RuntimeError, match="non-finite"):
        term(env, original_cfg)


def test_contract_preserves_enabled_checkpoint_and_rejects_cross_mode_resume(runtime, tmp_path):
    marker = tmp_path / "marker.npz"
    camera = marker.with_name("camera_ray_rectangles_320x240.json")
    background = tmp_path / "background.png"
    for path in (marker, camera, background):
        path.write_bytes(b"test-calibration")
    component_cfg = {"hydroshear_marker_layout_path": str(marker),
                     "taxim_rgb_background_path": str(background), "taxim_rgb_render_chunk_size": 3}
    encoder = SimpleNamespace(projection_dim=4, bundle_sha256="test-sha",
                              observation_contract=lambda **kw: dict(schema_version=1, feature="normalized_h", **kw))
    env = environment()
    env.unwrapped = env
    env._rotate_bulb_tactile_term = SimpleNamespace(encoder=encoder, component_cfg=component_cfg)
    env.observation_manager = SimpleNamespace(
        group_obs_dim={"policy": (10,), "proprio": (45,), "perception": (5,)},
        active_terms={"policy": ["state"], "proprio": ["pressure", "pretrained_tactile"], "perception": ["cloud"]},
        group_obs_term_dim={"policy": [(10,)], "proprio": [(25,), (20,)], "perception": [(5,)]})
    # Enabled behavior is byte-for-byte compatible (as JSON) with the original contract.
    path = DEX / "mdp/rotate_bulb_tactile.py"
    legacy = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                  and n.name == "initialize_rotate_bulb_tactile_contract")
    old_ns = dict(Path=Path, file_sha256=file_sha256)
    exec(compile(ast.Module(body=[legacy], type_ignores=[]), str(path), "exec"), old_ns)
    old_ns[legacy.name](env)
    expected = dict(env.tactile_policy_contract, task="BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0")
    runtime[legacy.name](env)
    assert env.tactile_policy_contract == expected
    prepare_policy_run(env, tmp_path / "enabled")
    prepare_policy_run(env, tmp_path / "enabled_resume", resume_path=tmp_path / "enabled/model.pt")

    env._rotate_bulb_tactile_term.encoder = None
    env._rotate_bulb_tactile_term.component_cfg = {}  # Disabled contract never reads encoder assets.
    env.observation_manager.group_obs_dim["proprio"] = (25,)
    env.observation_manager.group_obs_term_dim["proprio"][-1] = (0,)
    runtime[legacy.name](env)
    disabled = env.tactile_policy_contract
    assert disabled["state_dim"] == disabled["observation_dim"] == 40
    assert disabled["feature"] == "disabled" and disabled["tactile_policy_enabled"] is False
    assert "simulation_encoder_sha256" not in disabled
    assert "simulation_sensor_assets" not in disabled
    assert json.loads(json.dumps(disabled)) == disabled
    prepare_policy_run(env, tmp_path / "disabled")
    prepare_policy_run(env, tmp_path / "disabled_resume", resume_path=tmp_path / "disabled/model.pt")
    with pytest.raises(ValueError, match="observation_dim"):
        prepare_policy_run(env, tmp_path / "bad", resume_path=tmp_path / "enabled/model.pt")
    env.tactile_policy_contract = expected
    with pytest.raises(ValueError, match="observation_dim"):
        prepare_policy_run(env, tmp_path / "bad", resume_path=tmp_path / "disabled/model.pt")
