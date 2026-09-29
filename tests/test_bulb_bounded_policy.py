"""Exercise Custom bulb checkpoint, PPO and native export semantics on CPU."""

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from bulb_training import (
    BulbActorCritic, bulb_inference_policy, bulb_policy_for_export,
    configure_bulb_checkpoint, prepare_bulb_policy_run,
)


def observations(count=16):
    return TensorDict({"policy": torch.randn(count, 669)}, batch_size=[count])


def make_policy(obs, legacy=False):
    return (ActorCritic if legacy else BulbActorCritic)(
        obs, {"policy": ["policy"], "critic": ["policy"]}, 28,
        actor_hidden_dims=[16, 16], critic_hidden_dims=[16, 16],
        actor_obs_normalization=True, critic_obs_normalization=True, init_noise_std=0.35)


def agent_config(**policy):
    return SimpleNamespace(class_name="OnPolicyRunner", algorithm=SimpleNamespace(class_name="PPO"),
                           policy=SimpleNamespace(**policy))


def env_stub():
    action = {"version": 3, "joint_names": [f"joint{i}" for i in range(28)],
              "delta_limits_rad": [1.0] * 28,
              "control_dt": 1 / 60, "decimation": 4}
    raw = SimpleNamespace(action_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(action_contract=lambda: dict(action))),
        observation_manager=SimpleNamespace(group_obs_dim={"policy": (41,), "proprio": (436,), "perception": (192,)}))
    raw.unwrapped = raw
    return raw, action


def save_legacy_config(directory):
    config = {"class_name": "OnPolicyRunner", "algorithm": {"class_name": "PPO"},
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]}, "policy": {
        "class_name": "ActorCritic", "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
        "activation": "elu", "actor_obs_normalization": True, "critic_obs_normalization": True,
        "init_noise_std": 0.35, "noise_std_type": "scalar", "state_dependent_std": False}}
    (directory / "params").mkdir()
    (directory / "params/agent.yaml").write_text(yaml.safe_dump(config))
    return config


def test_policy_identity_bounds_and_transformed_probabilities():
    torch.manual_seed(20)
    obs = observations(128).double()
    policy = make_policy(obs).double()
    actions = policy.act(obs)
    torch.testing.assert_close(policy.action_std, torch.full_like(policy.action_std, 0.35))
    assert "_bulb_policy_version" in policy.state_dict()
    assert "_repose_policy_version" not in policy.state_dict()
    reference = TransformedDistribution(Normal(policy.action_mean, policy.action_std), [TanhTransform()])
    log_prob = policy.get_actions_log_prob(actions).detach()
    torch.testing.assert_close(log_prob, reference.log_prob(actions).sum(-1))
    policy.act(obs)
    torch.testing.assert_close((policy.get_actions_log_prob(actions) - log_prob).exp(), torch.ones(128).double())
    with torch.no_grad():
        policy.noise_unconstrained.fill_(1e6)
        policy.actor[-1].bias.fill_(1e6)
    actions = policy.act(obs)
    assert actions.abs().max() <= 1.0
    assert policy.action_std.min() > 1e5
    assert policy.act_inference(obs).abs().max() < 0.965
    assert torch.isfinite(policy.entropy).all()
    assert torch.isfinite(policy.get_actions_log_prob(actions)).all()


@pytest.mark.parametrize("initial_std", [0.35, 1.5, 10.0])
def test_noise_above_old_ceiling_is_initializable_and_trainable(initial_std):
    obs = observations()
    policy = BulbActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 28,
                            actor_hidden_dims=[16], critic_hidden_dims=[16], init_noise_std=initial_std)
    policy.act(obs)
    torch.testing.assert_close(policy.action_std, torch.full_like(policy.action_std, initial_std))
    policy.action_std.mean().backward()
    assert (policy.noise_unconstrained.grad > 0).all()
    with torch.no_grad():
        policy.noise_unconstrained.fill_(-1e6)
    policy.act(obs)
    torch.testing.assert_close(policy.action_std, torch.full_like(policy.action_std, 0.05))


@pytest.mark.parametrize("task", [
    "BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0",
    "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0",
])
def test_both_bulb_contracts_record_unit_actions_and_uncapped_noise(tmp_path, task):
    raw, _ = env_stub()
    raw.spec = SimpleNamespace(id=task)
    contract = prepare_bulb_policy_run(raw, tmp_path)
    assert contract["task"] == task and contract["version"] == 3
    assert contract["policy_transform"] == "bounded_latent_mean_then_tanh"
    assert contract["latent_std_bounds"] == [0.05, None]
    assert contract["action"]["delta_limits_rad"] == [1.0] * 28


def test_v2_checkpoint_cannot_silently_change_noise_or_action_semantics(tmp_path):
    state = make_policy(observations()).state_dict()
    state["_bulb_policy_version"] = torch.tensor(2)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": state}, checkpoint)
    with pytest.raises(ValueError, match="v2 checkpoint uses small action scales and capped noise"):
        configure_bulb_checkpoint(agent_config(), str(checkpoint), training=True)


def test_real_ppo_update_is_finite_and_uses_bounded_actions():
    torch.manual_seed(12)
    obs = observations(8)
    policy = make_policy(obs)
    ppo = PPO(policy, num_learning_epochs=2, num_mini_batches=2, entropy_coef=0.001,
              learning_rate=5e-4, desired_kl=0.01, device="cpu")
    ppo.init_storage("rl", 8, 8, obs, [28])
    before = policy.actor[-1].weight.detach().clone()
    for step in range(8):
        with torch.no_grad():
            actions = ppo.act(obs)
            assert actions.abs().max() <= 1.0
            obs = observations(8)
            ppo.process_env_step(obs, -(actions - 0.2).square().mean(-1),
                                 torch.full((8,), step == 7), {})
    with torch.no_grad():
        ppo.compute_returns(obs)
    losses = ppo.update()
    assert all(math.isfinite(value) for value in losses.values())
    assert not torch.equal(before, policy.actor[-1].weight)
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())


def test_new_checkpoint_requires_matching_contract_and_reloads_exactly(tmp_path):
    obs = observations()
    policy = make_policy(obs)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)
    cfg = agent_config()
    with pytest.raises(ValueError, match="missing its action contract"):
        configure_bulb_checkpoint(cfg, str(checkpoint), training=True)
    raw, action = env_stub()
    prepare_bulb_policy_run(raw, tmp_path)
    assert not configure_bulb_checkpoint(cfg, str(checkpoint), training=True)
    assert cfg.policy.class_name == "BulbActorCritic"
    import rsl_rl.runners.on_policy_runner as runner
    assert runner.BulbActorCritic is BulbActorCritic
    restored = make_policy(obs)
    restored.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(policy.act_inference(obs), restored.act_inference(obs))
    prepare_bulb_policy_run(raw, tmp_path / "new_run", resume_path=str(checkpoint))
    action["decimation"] = 2
    with pytest.raises(ValueError, match="does not match"):
        prepare_bulb_policy_run(raw, tmp_path / "mismatched", resume_path=str(checkpoint))
    assert not (tmp_path / "mismatched").exists()


def test_legacy_requires_diagnostic_restores_config_and_preserves_weights(tmp_path):
    obs = observations()
    policy = make_policy(obs, legacy=True)
    policy.actor_obs_normalizer.update(obs["policy"] * 2 + 3)
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "model.pt"
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)
    source_bytes = checkpoint.read_bytes()
    saved = save_legacy_config(source)
    cfg = agent_config(actor_hidden_dims=[512], actor_obs_normalization=False)
    with pytest.raises(ValueError, match="requires explicit"):
        configure_bulb_checkpoint(cfg, str(checkpoint))
    with pytest.raises(ValueError, match="cannot resume"):
        configure_bulb_checkpoint(cfg, str(checkpoint), training=True)
    assert configure_bulb_checkpoint(cfg, str(checkpoint), legacy_diagnostic=True)
    assert cfg.obs_groups == saved["obs_groups"]
    assert cfg.policy.actor_hidden_dims == [16, 16]
    assert cfg.policy.actor_obs_normalization is True
    restored = ActorCritic(obs, cfg.obs_groups, 28, **vars(cfg.policy))
    restored.load_state_dict(policy.state_dict(), strict=True)
    original = {k: v.clone() for k, v in restored.state_dict().items()}
    obs = observations()
    expected = restored.act_inference(obs).clamp(-1, 1)
    torch.testing.assert_close(bulb_inference_policy(restored.act_inference, legacy_diagnostic=True)(obs), expected)
    raw, _ = env_stub()
    with pytest.raises(ValueError, match="outside the checkpoint directory"):
        prepare_bulb_policy_run(raw, source / "exported", resume_path=str(checkpoint), legacy_diagnostic=True)
    contract = prepare_bulb_policy_run(raw, tmp_path / "diagnostic", resume_path=str(checkpoint), legacy_diagnostic=True)
    assert contract["legacy_diagnostic"] and contract["source_checkpoint"] == str(checkpoint)
    assert checkpoint.read_bytes() == source_bytes
    assert all(torch.equal(v, original[k]) for k, v in restored.state_dict().items())


def test_unknown_or_repose_checkpoint_and_incorrect_observation_are_rejected(tmp_path):
    policy = make_policy(observations())
    state = policy.state_dict()
    state["_bulb_policy_version"] = torch.tensor(999)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": state}, checkpoint)
    with pytest.raises(ValueError, match="Unsupported Bulb policy"):
        configure_bulb_checkpoint(agent_config(), str(checkpoint))
    state["_repose_policy_version"] = torch.tensor(2)
    torch.save({"model_state_dict": state}, checkpoint)
    with pytest.raises(ValueError, match="Repose checkpoint"):
        configure_bulb_checkpoint(agent_config(), str(checkpoint))
    raw, _ = env_stub()
    raw.observation_manager.group_obs_dim["policy"] = (42,)
    with pytest.raises(ValueError, match="669-dimensional"):
        prepare_bulb_policy_run(raw, tmp_path / "bad")


def test_rotation_subgoal_contract_rejects_old_checkpoint_and_parameter_mismatch(tmp_path):
    raw, _ = env_stub()
    prepare_bulb_policy_run(raw, tmp_path / "old")
    raw.observation_manager.active_terms = {"policy": ["rotation_subgoal"]}
    dims = raw.observation_manager.group_obs_dim
    dims["policy"] = (dims["policy"][0] + 4,)
    params = {"segment_turns": .25, "force_scale": .5, "regrasp_grace_s": .2, "max_turn_rate": 1.}
    raw.reward_manager = SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(params=params))
    contract = prepare_bulb_policy_run(raw, tmp_path / "new")
    assert contract["observation_dim"] == 673
    assert contract["rotation_guidance"]["version"] == 1
    with pytest.raises(ValueError, match="does not match"):
        prepare_bulb_policy_run(raw, tmp_path / "bad_old", resume_path=tmp_path / "old/model.pt")
    prepare_bulb_policy_run(raw, tmp_path / "replay", resume_path=tmp_path / "new/model.pt")
    params["segment_turns"] = .5
    with pytest.raises(ValueError, match="does not match"):
        prepare_bulb_policy_run(raw, tmp_path / "bad_params", resume_path=tmp_path / "new/model.pt")


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("runner,algorithm", [("DistillationRunner", "PPO"), ("OnPolicyRunner", "Distillation")])
def test_other_runners_and_algorithms_cannot_use_bounded_bulb_policy(training, runner, algorithm):
    cfg = agent_config()
    cfg.class_name, cfg.algorithm.class_name = runner, algorithm
    with pytest.raises(ValueError, match="only OnPolicyRunner with PPO"):
        configure_bulb_checkpoint(cfg, training=training)


@pytest.mark.parametrize("legacy", [False, True])
def test_native_jit_and_onnx_exports_match_runtime_and_do_not_mutate_policy(tmp_path, legacy):
    pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator
    lab_spec = importlib.util.find_spec("isaaclab_rl")
    if lab_spec is None:
        pytest.skip("Native Isaac Lab exporters unavailable")
    path = Path(next(iter(lab_spec.submodule_search_locations))) / "rsl_rl/exporter.py"
    spec = importlib.util.spec_from_file_location("_bulb_native_exporter", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    obs = observations(1)
    policy = make_policy(obs, legacy=legacy)
    policy.actor_obs_normalizer.update(observations(32)["policy"] * 2 + 1)
    policy.eval()
    with torch.no_grad():
        policy.actor[-1].bias.fill_(5.0)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    expected = bulb_inference_policy(policy.act_inference, legacy_diagnostic=legacy)(obs).detach()
    adapted = bulb_policy_for_export(policy, legacy_diagnostic=legacy)
    exporter.export_policy_as_jit(adapted, normalizer=policy.actor_obs_normalizer, path=str(tmp_path))
    exporter.export_policy_as_onnx(adapted, normalizer=policy.actor_obs_normalizer, path=str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(scripted(obs["policy"]), expected)
    actual = ReferenceEvaluator(str(tmp_path / "policy.onnx")).run(None, {"obs": obs["policy"].numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected, atol=1e-6, rtol=1e-5)
    assert all(torch.equal(value, before[key]) for key, value in policy.state_dict().items())
