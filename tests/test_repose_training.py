"""Exercise the actual RSL-RL distribution/PPO and Isaac Lab export interfaces."""

import importlib.util
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl"))
from repose_training import (
    ReposeActorCritic,
    configure_repose_checkpoint,
    register_repose_policy,
    repose_policy_for_export,
)


def _observations(count=64):
    return TensorDict({"policy": torch.randn(count, 9)}, batch_size=[count])


def _policy(obs, policy_class=ReposeActorCritic):
    return policy_class(obs, {"policy": ["policy"], "critic": ["policy"]}, 4,
                        actor_hidden_dims=[16, 16], critic_hidden_dims=[16, 16], init_noise_std=0.35)


def test_actions_are_bounded_and_noise_cannot_run_away():
    obs = _observations(1024)
    policy = _policy(obs)
    policy.act(obs)
    torch.testing.assert_close(policy.action_std, torch.full((1024, 4), 0.35))
    assert policy.act_inference(obs).abs().max() < 0.05
    with torch.no_grad():
        policy.actor[-1].bias.fill_(1e6)
        policy.noise_logit.fill_(1e6)
    action = policy.act(obs)
    assert action.abs().max() <= 1
    assert policy.act_inference(obs).abs().max() < 0.965
    assert policy.action_std.max() <= 0.7
    assert policy.action_mean.abs().max() <= 2
    assert torch.isfinite(policy.get_actions_log_prob(action)).all()
    assert torch.isfinite(policy.get_actions_log_prob(torch.ones_like(action))).all()


def test_tanh_log_prob_matches_independent_distribution_and_replay():
    torch.manual_seed(20)
    obs = _observations()
    policy = _policy(obs).double()
    obs = obs.double()
    actions = policy.act(obs)
    reference = TransformedDistribution(
        Normal(policy.action_mean, policy.action_std), [TanhTransform(cache_size=0)])
    old_log_prob = policy.get_actions_log_prob(actions).detach()
    torch.testing.assert_close(old_log_prob, reference.log_prob(actions).sum(-1))
    # PPO resamples to refresh the distribution, then evaluates stored actions.
    policy.act(obs)
    ratio = (policy.get_actions_log_prob(actions) - old_log_prob).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio))


def test_latent_kl_matches_transformed_density_ratio():
    torch.manual_seed(19)
    obs = _observations(100000)
    policy = _policy(obs)
    with torch.no_grad():
        old_actions = policy.act(obs)
        old_log_prob = policy.get_actions_log_prob(old_actions)
        old_mean, old_std = policy.action_mean.clone(), policy.action_std.clone()
        policy.actor[-1].bias.add_(0.12)
        policy.noise_logit.add_(0.15)
        policy.act(obs)
        new_mean, new_std = policy.action_mean, policy.action_std
        analytic = (torch.log(new_std / old_std)
                    + (old_std.square() + (old_mean - new_mean).square()) / (2 * new_std.square()) - 0.5).sum(-1)
        measured = old_log_prob - policy.get_actions_log_prob(old_actions)
    assert abs(float(analytic.mean() - measured.mean())) < 0.007


def test_transformed_entropy_is_finite_and_has_location_gradient():
    torch.manual_seed(17)
    obs = _observations(8192)
    policy = _policy(obs)
    with torch.no_grad():
        policy.actor[-1].weight.zero_()
        policy.actor[-1].bias.fill_(1.0)
    policy.act(obs)
    policy.entropy.mean().backward()
    # Entropy should push a shifted distribution back toward the action centre.
    assert (policy.actor[-1].bias.grad < 0).all()
    assert torch.isfinite(policy.noise_logit.grad).all()


def test_real_ppo_updates_use_consistent_bounded_actions_and_finite_losses():
    torch.manual_seed(4)
    obs = _observations(16)
    policy = _policy(obs)
    ppo = PPO(policy, num_learning_epochs=2, num_mini_batches=2, entropy_coef=0.001,
              learning_rate=5e-4, desired_kl=0.016, device="cpu")
    ppo.init_storage("rl", 16, 8, obs, [4])
    before = policy.actor[-1].weight.detach().clone()
    for _ in range(2):
        for step in range(8):
            with torch.no_grad():
                actions = ppo.act(obs)
                assert actions.abs().max() <= 1
                rewards = -(actions - 0.2).square().sum(-1)
                obs = _observations(16)
                dones = torch.full((16,), step == 7)
                ppo.process_env_step(obs, rewards, dones, {})
        with torch.no_grad():
            ppo.compute_returns(obs)
        result = ppo.update()
        assert all(math.isfinite(value) for value in result.values())
        assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
    assert not torch.equal(before, policy.actor[-1].weight)


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_family_selection_and_strict_reload(tmp_path, legacy):
    obs = _observations()
    original = _policy(obs, ActorCritic if legacy else ReposeActorCritic)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": original.state_dict()}, checkpoint)
    cfg = SimpleNamespace(policy=SimpleNamespace(class_name="ReposeActorCritic", noise_std_type="log"))
    revision = configure_repose_checkpoint(cfg, str(checkpoint))
    assert revision == (1 if legacy else 2)
    import rsl_rl.runners.on_policy_runner as runner_module

    restored = _policy(obs, getattr(runner_module, cfg.policy.class_name))
    restored.load_state_dict(original.state_dict())
    torch.testing.assert_close(original.act_inference(obs), restored.act_inference(obs))
    assert configure_repose_checkpoint(cfg) == 2


def test_native_jit_and_onnx_exports_include_action_transform(tmp_path):
    pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator

    lab_spec = importlib.util.find_spec("isaaclab_rl")
    if lab_spec is None:
        pytest.skip("Native Isaac Lab exporters are unavailable")
    # Load the pure exporter without importing simulator-dependent wrappers.
    path = Path(next(iter(lab_spec.submodule_search_locations))) / "rsl_rl" / "exporter.py"
    spec = importlib.util.spec_from_file_location("repose_native_exporter", path)
    exporter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exporter
    spec.loader.exec_module(exporter)
    obs = _observations(1)
    policy = ReposeActorCritic(
        obs, {"policy": ["policy"], "critic": ["policy"]}, 4,
        actor_hidden_dims=[16, 16], critic_hidden_dims=[16, 16], actor_obs_normalization=True)
    policy.actor_obs_normalizer.update(_observations(64)["policy"] * 2.0 + 1.0)
    with torch.no_grad():
        policy.actor[-1].bias.fill_(2.0)
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    adapted = repose_policy_for_export(policy)
    expected = policy.act_inference(obs).detach()
    exporter.export_policy_as_jit(adapted, normalizer=policy.actor_obs_normalizer, path=str(tmp_path))
    exporter.export_policy_as_onnx(adapted, normalizer=policy.actor_obs_normalizer, path=str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(scripted(obs["policy"]), expected)
    onnx_actions = ReferenceEvaluator(str(tmp_path / "policy.onnx")).run(
        None, {"obs": obs["policy"].numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(onnx_actions), expected, atol=1e-6, rtol=1e-5)
    assert all(torch.equal(before[key], value) for key, value in policy.state_dict().items())


@pytest.mark.parametrize("load_optimizer", [False, True])
def test_real_runner_curriculum_checkpoint_roundtrip(tmp_path, load_optimizer):
    from rsl_rl.runners import OnPolicyRunner
    from repose_run_state import install_repose_run_state

    register_repose_policy()
    obs = _observations(16)
    env = SimpleNamespace(num_envs=16, num_actions=4, get_observations=lambda: obs)
    config = {
        "num_steps_per_env": 8, "save_interval": 1,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "policy": {"class_name": "ReposeActorCritic", "actor_hidden_dims": [16, 16],
                   "critic_hidden_dims": [16, 16], "actor_obs_normalization": True},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1},
    }
    runner = OnPolicyRunner(env, config, device="cpu")
    # The real learn() sets this field when constructing its writer.
    runner.logger_type = "tensorboard"
    runner.current_learning_iteration = 23
    runner.alg.policy.update_normalization(obs)
    state = {"revision": 2, "stage": 3, "window_completed": 300.0, "window_successful": 199.0}
    restored = {}
    runtime = SimpleNamespace(state_dict=lambda: dict(state),
                              load_state_dict=lambda payload: restored.update(payload))
    install_repose_run_state(runner, SimpleNamespace(_repose_training=runtime))
    before = {key: value.clone() for key, value in runner.alg.policy.state_dict().items()}
    checkpoint = tmp_path / "curriculum.pt"
    infos = {"caller_information": "preserved"}
    runner.save(str(checkpoint), infos=infos)
    assert infos == {"caller_information": "preserved"}
    with torch.no_grad():
        runner.alg.policy.actor[-1].bias.add_(1.0)
    runner.current_learning_iteration = 0
    loaded = runner.load(str(checkpoint), load_optimizer=load_optimizer, map_location="cpu")
    assert loaded == {**infos, "repose_training_state": state}
    assert restored == state
    assert runner.current_learning_iteration == 23
    assert all(torch.equal(before[key], value) for key, value in runner.alg.policy.state_dict().items())
    # V2 files cannot silently lose their curriculum stage and restart at easy goals.
    incomplete = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incomplete["infos"] = {}
    torch.save(incomplete, checkpoint)
    with pytest.raises(ValueError, match="missing its curriculum state"):
        runner.load(str(checkpoint), load_optimizer=load_optimizer, map_location="cpu")
