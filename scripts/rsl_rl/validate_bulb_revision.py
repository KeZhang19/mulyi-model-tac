"""Two-environment native acceptance probe for the revised Custom bulb task.

Writes evidence to a new directory and runs only the requested smoke updates.
It never resumes or modifies a production training run.
"""

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True)
parser.add_argument("--legacy-checkpoint", required=True)
parser.add_argument("--pure", action="store_true")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--updates", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app
env = None
report = {"passed": False, "pure_collision": args.pure}

try:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import BrainCo_DexHand.tasks
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom import env_cfg as custom
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.agents.rsl_rl_ppo_cfg import (
        DexsuiteRevo3RotateBulbPPORunnerCfg,
    )
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.self_collision import (
        restore_bulb_arm_self_collisions,
    )
    from BrainCo_DexHand.tactile_representation.policy import prepare_policy_run
    from bulb_training import configure_bulb_checkpoint, prepare_bulb_policy_run, bulb_inference_policy
    from bulb_diagnostics import BulbDiagnosticRecorder

    cfg = parse_env_cfg("BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0", device=args.device, num_envs=2)
    cfg.seed = 42
    cfg.bulb_pure_collision = args.pure
    cfg.tactile_policy_enabled = False
    cfg.tactile_reconstruction_diagnostics = False
    cfg.log_dir = str(out)
    agent_cfg = DexsuiteRevo3RotateBulbPPORunnerCfg()
    agent_cfg.device = args.device
    configure_bulb_checkpoint(agent_cfg, training=True)
    env = gym.make("BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0", cfg=cfg)
    raw = env.unwrapped
    prepare_policy_run(env, out / "fresh")
    contract = prepare_bulb_policy_run(env, out / "fresh")
    report["contract"] = contract
    report["loaded_source"] = custom.__file__
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    robot = raw.scene["robot"]
    action = raw.action_manager.get_term("action")
    report["clone_collision_filters"] = [
        restore_bulb_arm_self_collisions(raw.sim.stage, f"/World/envs/env_{i}/Robot") for i in range(2)
    ]
    assert all(value["removed_count"] == 0 for value in report["clone_collision_filters"])

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(out / "fresh"), device=args.device)
    network = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    samples = []
    original_apply = action.apply_actions

    def capture_apply():
        samples.append(action.position_targets.clone())
        original_apply()

    action.apply_actions = capture_apply
    joint_ids = action.joint_ids
    arm = action.arm_action_indices
    hand = action.hand_action_indices
    maximum_delta = torch.zeros(28, device=raw.device)
    report["fixed_step_trace"] = []
    with torch.inference_mode():
        raw.reset(seed=42)
        obs = wrapped.get_observations()
        for step in range(args.steps):
            before = robot.data.joint_pos[:, joint_ids].clone()
            controls = network.act(obs)
            if step < 2:
                controls[:] = 10.0 if step == 0 else -10.0
            samples.clear()
            obs, rewards, dones, _ = wrapped.step(controls)
            assert len(samples) == cfg.decimation == 4
            assert all(torch.equal(samples[0], target) for target in samples[1:])
            limits = robot.data.soft_joint_pos_limits[:, joint_ids]
            delta_limits = torch.tensor(contract["action"]["delta_limits_rad"], device=raw.device)
            expected = torch.clamp(before + delta_limits * controls.clamp(-1, 1),
                                   min=limits[..., 0], max=limits[..., 1])
            torch.testing.assert_close(samples[0], expected)
            # A measured position slightly outside a soft limit may need a
            # larger correction; the normalized incremental request stays bounded.
            maximum_delta = torch.maximum(maximum_delta, (delta_limits * controls.clamp(-1, 1)).abs().amax(0))
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(rewards).all()
            if step % 15 == 0 or dones.any():
                report["fixed_step_trace"].append({
                    "step": step, "reward": rewards.tolist(), "done": dones.tolist(),
                    "std_min": float(network.action_std.min()), "std_max": float(network.action_std.max()),
                    "contacts": custom.rotate_bulb_contact_scores(raw).tolist(),
                    "termination": {name: raw.termination_manager.get_term(name).tolist()
                                    for name in raw.termination_manager.active_terms},
                    "failure_reward": (raw.reward_manager._step_reward[:,
                        raw.reward_manager.active_terms.index("failure")] * raw.step_dt).tolist(),
                })
        report["maximum_requested_delta_rad"] = {
            "arm": float(maximum_delta[arm].max()), "hand": float(maximum_delta[hand].max())}
        assert maximum_delta[arm].max() <= 1.0000001
        assert maximum_delta[hand].max() <= 1.0000001
        try:
            action.process_actions(torch.full((2, 28), float("nan"), device=raw.device))
        except ValueError:
            report["nonfinite_action_rejected"] = True
        else:
            raise AssertionError("Nonfinite actions reached physics")
    action.apply_actions = original_apply

    # A real optimizer update exercises rollout storage, transformed log
    # probabilities and the latent KL interface on the server's RSL-RL version.
    with torch.inference_mode():
        raw.reset(seed=43)
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
    assert all(torch.isfinite(value).all() for value in network.state_dict().values())
    report["ppo_updates"] = args.updates
    saved = out / "fresh" / "acceptance_model.pt"
    runner.save(str(saved))
    replay_cfg = DexsuiteRevo3RotateBulbPPORunnerCfg()
    replay_cfg.device = args.device
    configure_bulb_checkpoint(replay_cfg, str(saved))
    prepare_bulb_policy_run(env, out / "bounded_replay", resume_path=str(saved))
    replay_runner = OnPolicyRunner(wrapped, replay_cfg.to_dict(), log_dir=None, device=args.device)
    replay_runner.load(str(saved), map_location=args.device)
    replay_network = replay_runner.alg.policy if hasattr(replay_runner.alg, "policy") else replay_runner.alg.actor_critic
    reference_state, replay_state = network.state_dict(), replay_network.state_dict()
    assert reference_state.keys() == replay_state.keys()
    for name, expected in reference_state.items():
        # Includes both observation normalizers and the independent Bulb marker.
        torch.testing.assert_close(replay_state[name], expected, rtol=0, atol=0)
    reference_policy = runner.get_inference_policy(device=raw.device)
    replay_policy = replay_runner.get_inference_policy(device=raw.device)
    with torch.inference_mode():
        replay_obs = wrapped.get_observations()
        expected_actions = reference_policy(replay_obs)
        replay_actions = replay_policy(replay_obs)
        torch.testing.assert_close(replay_actions, expected_actions)
        assert replay_actions.abs().max() <= 1.0
    report["bounded_reload_state_tensors"] = len(replay_state)
    report["bounded_reload_max_action_error"] = float((replay_actions - expected_actions).abs().max())
    report["bounded_reload_contract"] = True

    legacy_cfg = DexsuiteRevo3RotateBulbPPORunnerCfg()
    legacy_cfg.device = args.device
    assert configure_bulb_checkpoint(legacy_cfg, args.legacy_checkpoint, legacy_diagnostic=True)
    prepare_policy_run(env, out / "legacy", resume_path=args.legacy_checkpoint)
    prepare_bulb_policy_run(env, out / "legacy", resume_path=args.legacy_checkpoint, legacy_diagnostic=True)
    legacy_runner = OnPolicyRunner(wrapped, legacy_cfg.to_dict(), log_dir=None, device=args.device)
    legacy_runner.load(args.legacy_checkpoint, map_location=args.device)
    policy = bulb_inference_policy(legacy_runner.get_inference_policy(device=raw.device), legacy_diagnostic=True)
    recorder = BulbDiagnosticRecorder(env, out / "legacy")
    with torch.inference_mode():
        raw.reset(seed=42)
        obs = wrapped.get_observations()
        for step in range(args.steps):
            controls = policy(obs)
            assert controls.abs().max() <= 1.0
            obs, rewards, dones, _ = wrapped.step(controls)
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(rewards).all()
            recorder.record(controls, rewards, dones)
    recorder.close()
    report["legacy_steps"] = args.steps
    report["passed"] = True
except Exception:
    report["error"] = traceback.format_exc()
    traceback.print_exc()
finally:
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("BULB_ACCEPTANCE", json.dumps({key: value for key, value in report.items()
                                        if key != "fixed_step_trace"}), flush=True)
    if env is not None:
        env.close()
    app.close(wait_for_replicator=False, skip_cleanup=True)

if not report["passed"]:
    raise SystemExit(1)
