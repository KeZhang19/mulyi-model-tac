"""Restoration opt-in must bypass all expensive dependencies and guard checkpoints."""

import ast
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from BrainCo_DexHand.tactile_representation.policy import file_sha256, prepare_policy_run
from test_d_peg_insertion_rewards import code  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
DEX = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite"


def load_runtime(code):
    class Manager:
        def __init__(self, cfg, env):
            self._env = env

    class EncoderRuntime(Manager):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.encoder = "enabled"

        def __call__(self, env, component_cfg):
            return torch.ones((env.num_envs, 1280))

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled restoration initialized encoder or sensor assets")

    path = DEX / "mdp/d_peg_tactile.py"
    definitions = [n for n in ast.parse(path.read_text()).body
                   if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    ns = dict(torch=torch, copy=copy, math=math, Path=Path, asdict=asdict,
              ManagerTermBase=Manager, RotateBulbPretrainedTactile=EncoderRuntime,
              tactile_obs=SimpleNamespace(invalidate_ours_tactile_cache_on_reset=Mock()),
              initialize_rotate_bulb_tactile_contract=forbidden, file_sha256=file_sha256,
              validate_d_peg_geometry_metadata=code.validate_d_peg_geometry_metadata,
              enable_d_peg_fast_taxim=forbidden)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), ns)
    return ns


def environment():
    assets = ROOT / "assets/d_peg_insertion"
    cfg = SimpleNamespace(tactile_policy_enabled=False, d_peg_fast_taxim=True,
        tactile_policy_checkpoint="missing-checkpoint.pt", tactile_reconstruction_diagnostics=True,
        geometry_metadata_path=assets / "metadata.json", insertion_depth_m=.028,
        socket_mouth_height_m=.05, peg_usd_path=assets / "peg.usd", socket_usd_path=assets / "socket.usd",
        pregrasp_path=assets / "pregrasp.json", socket_xy_randomization_m=.005,
        socket_yaw_randomization_deg=10., episode_length_s=15., d_peg_training_revision=3,
        actions=SimpleNamespace(action=SimpleNamespace(scale={"Joint[1-7]_R": .1, "right_.*": .02})))
    return SimpleNamespace(cfg=cfg, num_envs=2, device="cpu", common_step_counter=0)


def test_default_is_disabled_and_removes_only_restoration_columns(code):
    tree = ast.parse((DEX / "config/Revo3/dexsuite_revo3_env_cfg_insert_d_peg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DexsuiteRevo3InsertDPegEnvCfg")
    switch = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == "tactile_policy_enabled")
    assert ast.literal_eval(switch.value) is False
    ns, env = load_runtime(code), environment()
    term = ns["DPegPretrainedTactile"](SimpleNamespace(params={"component_cfg": {}}), env)
    assert term.encoder is None
    remaining = torch.randn(2, 669)
    for ids in (None, torch.tensor([1]), slice(None)):
        empty = term(env, {})
        assert empty.shape == (2, 0)
        torch.testing.assert_close(torch.cat((remaining, empty), -1), remaining)
        term.reset(ids)
        env.common_step_counter += 1
        assert term(env, {}) is empty
    assert ns["tactile_obs"].invalidate_ours_tactile_cache_on_reset.call_count == 3
    assert env.latest_tactile_inputs is env.latest_tactile_latent is env.latest_tactile_reconstruction is None


def test_late_enable_restores_original_encoder_pathway(code):
    ns, env = load_runtime(code), environment()
    env.cfg.tactile_policy_enabled = True
    env.cfg.d_peg_fast_taxim = False
    term = ns["DPegPretrainedTactile"](SimpleNamespace(params={"component_cfg": {}}), env)
    assert term.encoder == "enabled"
    assert term(env, {}).shape == (2, 1280)


def test_disabled_contract_skips_encoder_assets_and_rejects_enabled_or_v2_resume(code, tmp_path):
    ns, env = load_runtime(code), environment()
    dims = {"policy": 43, "proprio": 434, "perception": 192}
    env.observation_manager = SimpleNamespace(
        group_obs_dim={k: (v,) for k, v in dims.items()},
        active_terms={k: ["state"] for k in dims}, group_obs_term_dim={k: [(v,)] for k, v in dims.items()})
    env.action_manager = SimpleNamespace(get_term=lambda _: SimpleNamespace(
        action_contract=lambda: dict(schema_version=2, scale=env.cfg.actions.action.scale)))
    ns["initialize_d_peg_tactile_contract"](env)
    contract = env.tactile_policy_contract
    assert contract["observation_dim"] == 669 and contract["projection_dim"] == 0
    assert contract["task_schema_version"] == 3
    assert "simulation_encoder_sha256" not in contract and "simulation_sensor_assets" not in contract
    wrapped = SimpleNamespace(unwrapped=env)
    prepare_policy_run(wrapped, tmp_path / "off")
    for changed in (dict(contract, observation_dim=1949), dict(contract, task_schema_version=2)):
        other = tmp_path / "other"
        other.mkdir(exist_ok=True)
        (other / "tactile_policy_contract.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="contract mismatch"):
            prepare_policy_run(wrapped, tmp_path / "refused", resume_path=other / "model.pt")
        assert not (tmp_path / "refused").exists()
