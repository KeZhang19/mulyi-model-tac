"""PPO hooks must expose drift without changing samples, loss or updates."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from d_peg_training import initialize_d_peg_policy, install_d_peg_training_diagnostics, reduce_episode_diagnostics


def _runner(log_dir, normalize=True):
    pytest.importorskip("rsl_rl")
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict

    torch.manual_seed(17)
    obs = TensorDict({"policy": torch.randn(4, 6)}, batch_size=[4])
    policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 28,
                         actor_hidden_dims=[16], critic_hidden_dims=[16], init_noise_std=.15,
                         actor_obs_normalization=normalize, critic_obs_normalization=normalize)
    algorithm = PPO(policy, num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-4,
                    schedule="adaptive", desired_kl=.01, device="cpu")
    algorithm.init_storage("rl", 4, 4, obs, [28])
    env = SimpleNamespace(tactile_policy_contract={"task": "BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0"})
    return SimpleNamespace(alg=algorithm, env=SimpleNamespace(unwrapped=env), device="cpu", log_dir=str(log_dir),
                           current_learning_iteration=0, writer=None), obs


def _learn(runner, obs):
    from tensordict import TensorDict
    with torch.inference_mode():
        for step in range(4):
            actions = runner.alg.act(obs)
            obs = TensorDict({"policy": torch.randn(4, 6) + step * 3}, batch_size=[4])
            runner.alg.process_env_step(obs, -actions.square().mean(-1), torch.zeros(4), {})
        runner.alg.compute_returns(obs)
    return runner.alg.update()


@pytest.mark.parametrize("normalize", [False, True])
def test_diagnostics_preserve_exact_rng_model_optimizer_and_losses(tmp_path, normalize):
    plain, obs = _runner(tmp_path / "plain", normalize)
    expected_loss = _learn(plain, obs)
    expected_rng = torch.get_rng_state().clone()
    wrapped, obs = _runner(tmp_path / "wrapped", normalize)
    install_d_peg_training_diagnostics(wrapped)
    actual_loss = _learn(wrapped, obs)
    torch.testing.assert_close(wrapped.alg.policy.state_dict(), plain.alg.policy.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(wrapped.alg.optimizer.state_dict(), plain.alg.optimizer.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    record = json.loads((tmp_path / "wrapped/d_peg_training.jsonl").read_text())
    assert record["minibatches_per_rank"] == 4
    assert 0 <= record["ppo"]["ratio_clip_frac"] <= 1
    first_kl = record["ppo"]["first_minibatch_pre_update_kl_mean"]
    if normalize:
        assert first_kl > .01  # Running moments alone moved the old action distribution.
        assert record["ppo"]["normalizer_rollout_actor_mean_max"] > 1
    else:
        assert first_kl == pytest.approx(28 * torch.log(torch.tensor(1 + 1e-5)).item(), abs=2e-6)


def test_fresh_actor_zero_preserves_critic_noise_and_respects_task(tmp_path):
    runner, obs = _runner(tmp_path)
    critic_before = {k: v.clone() for k, v in runner.alg.policy.critic.state_dict().items()}
    noise_before = runner.alg.policy.std.clone()
    initialize_d_peg_policy(runner)
    assert torch.equal(runner.alg.policy.act_inference(obs), torch.zeros(4, 28))
    torch.testing.assert_close(runner.alg.policy.critic.state_dict(), critic_before, rtol=0, atol=0)
    assert torch.equal(runner.alg.policy.std, noise_before)
    runner.env.unwrapped.tactile_policy_contract["task"] = "rotate_bulb"
    with pytest.raises(ValueError, match="D-peg tactile policy contract"):
        initialize_d_peg_policy(runner)
    with pytest.raises(ValueError, match="D-peg tactile policy contract"):
        install_d_peg_training_diagnostics(runner)


def _episode_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        payload = {"count": rank + 1, "sums": {"success": 1, "duration_s": 2 + 8 * rank},
                   "maxima": {"max_valid_depth_m": .01 + .015 * rank}}
        result = reduce_episode_diagnostics(payload, "cpu")
        assert result["completed_episodes"] == 3
        assert result["means"]["success"] == pytest.approx(2 / 3)
        assert result["means"]["duration_s"] == 4
        assert result["maxima"]["max_valid_depth_m"] == .025
        empty = reduce_episode_diagnostics({"count": 0, "sums": {"success": 0},
                                            "maxima": {"max_valid_depth_m": 0}}, "cpu")
        assert empty["means"]["success"] is None
        assert empty["maxima"]["max_valid_depth_m"] is None
        if rank == 0:
            Path(output).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_episode_reduction_weights_actual_completed_counts_across_ranks(tmp_path):
    mp.spawn(_episode_worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "result.json")),
             nprocs=2, join=True)
    assert json.loads((tmp_path / "result.json").read_text())["completed_episodes"] == 3


def _ppo_diagnostics_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        runner, obs = _runner(output)
        # The policies deliberately have different inputs/updates. This test
        # checks pooled diagnostics, while the existing audit tests gradients.
        obs["policy"] += rank * 5
        runner.alg.schedule = "fixed"
        install_d_peg_training_diagnostics(runner, episode_metrics_provider=lambda: {
            "count": rank + 1, "sums": {"held_fraction": .5 + rank}, "maxima": {"max_valid_depth_m": .01 * rank}})
        _learn(runner, obs)
        record = runner.d_peg_training_diagnostics
        assert record["world_size"] == 2
        assert record["minibatches_per_rank"] == 4
        assert record["episodes"]["completed_episodes"] == 3
        assert record["episodes"]["means"]["held_fraction"] == pytest.approx(2 / 3)
        records = [None, None]
        dist.all_gather_object(records, record)
        assert records[0] == records[1]
        if rank == 0:
            assert len((Path(output) / "d_peg_training.jsonl").read_text().splitlines()) == 1
    finally:
        dist.destroy_process_group()


def test_real_ppo_metrics_and_episode_sums_pool_identically_on_two_ranks(tmp_path):
    mp.spawn(_ppo_diagnostics_worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "log")),
             nprocs=2, join=True)


def test_dual_gpu_launcher_default_and_override_without_launching_gpu(tmp_path):
    import os
    capture = tmp_path / "python"
    capture.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    capture.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/launch_d_peg_dual_gpu.sh"
    env = {**os.environ, "D_PEG_PYTHON": str(capture)}
    env.pop("D_PEG_ENVS_PER_GPU", None)
    env.pop("D_PEG_MAX_ITERATIONS", None)
    default = subprocess.check_output(["bash", str(script)], env=env, text=True).splitlines()
    assert default[default.index("--num_envs") + 1] == "512"
    assert default[default.index("--max_iterations") + 1] == "15000"
    assert "/dexsuite_revo3_insert_d_peg_v3/" in default[default.index("--log_dir") + 1]
    changed = subprocess.check_output(["bash", str(script), str(tmp_path / "run")],
              env={**env, "D_PEG_ENVS_PER_GPU": "4", "D_PEG_MAX_ITERATIONS": "3"}, text=True).splitlines()
    assert changed[changed.index("--num_envs") + 1] == "4"
    assert changed[changed.index("--max_iterations") + 1] == "3"
