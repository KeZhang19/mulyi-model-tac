#!/usr/bin/env python3
"""Run reset, PPO save/resume and play checks with optional restoration observations.

Default suite: train/PLAY reset checks (2 envs each), 3 PPO iterations (4 envs),
2 resumed iterations in a new process, then 300 PLAY steps. This is a pipeline
and numerical-stability check; it does not claim the policy learned insertion.
"""

import argparse
from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]
TASK = "BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0"
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def finite_tree(value, name):
    import torch
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"Non-finite tensor: {name}")
    elif isinstance(value, Mapping) or hasattr(value, "items"):
        for key, child in value.items():
            finite_tree(child, f"{name}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            finite_tree(child, f"{name}/{index}")


def tensor_hash(named_values):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(named_values):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def policy_module(runner):
    if hasattr(runner.alg, "policy"):
        return runner.alg.policy
    return runner.alg.actor_critic


def install_step_audit(env, report):
    import torch
    from distributed_normalization import install_finite_action_guard
    install_finite_action_guard(env)
    original = env.step
    report["steps_checked"] = 0
    report["reward_min"] = None
    report["reward_max"] = None
    report["completed_episodes"] = 0

    def step(actions):
        if actions.shape != (env.num_envs, 28):
            raise AssertionError(f"Expected {env.num_envs}x28 actions, got {actions.shape}")
        result = original(actions)
        finite_tree(result[:3], "step")
        observations, rewards, dones = result[:3]
        report["steps_checked"] += 1
        low, high = float(rewards.min()), float(rewards.max())
        report["reward_min"] = low if report["reward_min"] is None else min(low, report["reward_min"])
        report["reward_max"] = high if report["reward_max"] is None else max(high, report["reward_max"])
        report["completed_episodes"] += int(dones.sum())
        finite_tree(env.unwrapped.scene["robot"].data.joint_pos, "joint_pos")
        finite_tree(env.unwrapped.scene["object"].data.root_state_w, "peg_root_state")
        if torch.cuda.is_available():
            report["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(env.unwrapped.device)
        return result

    env.step = step


def tactile_signal_diagnostics(env):
    """Inspect the encoder's actual cached reset inputs without advancing any sensor."""
    import torch

    sample = env.latest_tactile_inputs
    latent = env.latest_tactile_latent.detach()
    depth = sample["depth_m"].detach()
    marker = sample["marker"].detach()
    valid = sample["marker_valid"].bool()
    order = tuple(env.tactile_policy_contract["finger_order"])
    assert order == ("little", "ring", "middle", "index", "thumb"), order
    assert depth.ndim == 5 and depth.shape[:3] == (env.num_envs, 5, 1), depth.shape
    assert latent.ndim == 3 and latent.shape[:2] == (env.num_envs, 5), latent.shape
    assert marker.ndim == 4 and marker.shape[:2] == (env.num_envs, 5) and marker.shape[-1] == 5
    assert valid.shape == marker.shape[:-1], (valid.shape, marker.shape)
    finite_tree(sample, "reset_tactile_inputs")
    finite_tree(latent, "reset_tactile_latent")
    active_threshold_m = 1.0e-6
    active_pixels = (depth > active_threshold_m).flatten(2).sum(-1)
    nonzero_pixels = (depth > 0).flatten(2).sum(-1)
    depth_max = depth.flatten(2).amax(-1)
    flow = marker[..., 2:4]
    magnitude = flow.norm(dim=-1)
    count = valid.sum(-1)
    has_valid = count > 0
    flow_min = flow.masked_fill(~valid[..., None], float("inf")).amin(-2)
    flow_max = flow.masked_fill(~valid[..., None], -float("inf")).amax(-2)
    flow_min = torch.where(has_valid[..., None], flow_min, 0)
    flow_max = torch.where(has_valid[..., None], flow_max, 0)
    magnitude_max = magnitude.masked_fill(~valid, 0).amax(-1)
    magnitude_mean = magnitude.masked_fill(~valid, 0).sum(-1) / count.clamp_min(1)
    fingers = {}
    for index, name in enumerate(order):
        fingers[name] = {
            "latent_l2_per_env": latent[:, index].norm(dim=-1).cpu().tolist(),
            "depth_max_m_per_env": depth_max[:, index].cpu().tolist(),
            "depth_active_pixels_per_env": active_pixels[:, index].cpu().tolist(),
            "depth_positive_pixels_per_env": nonzero_pixels[:, index].cpu().tolist(),
            "marker_valid_count_per_env": count[:, index].cpu().tolist(),
            "marker_displacement_max_px_per_env": magnitude_max[:, index].cpu().tolist(),
            "marker_displacement_mean_px_per_env": magnitude_mean[:, index].cpu().tolist(),
            "marker_dx_dy_min_px_per_env": flow_min[:, index].cpu().tolist(),
            "marker_dx_dy_max_px_per_env": flow_max[:, index].cpu().tolist(),
        }
    return {
        "sample": "cached_encoder_inputs_immediately_after_reset",
        "common_step_counter": int(env.common_step_counter),
        "finger_order": list(order),
        "depth_active_threshold_m": active_threshold_m,
        "depth_has_contact_per_env": (active_pixels.sum(-1) > 0).cpu().tolist(),
        "input_depth_shape": list(depth.shape),
        "latent_shape": list(latent.shape),
        "per_finger": fingers,
        "resolved_pressure_target_meshes": {
            name: str(sensor._resolved_target_mesh_prim_path)
            for name, sensor in env.scene.sensors.items()
            if getattr(sensor, "_resolved_target_mesh_prim_path", None) is not None
        },
    }


def check_contract(env, report):
    contract = env.tactile_policy_contract
    enabled = env.cfg.tactile_policy_enabled
    expected = {"policy": 43, "proprio": 1714 if enabled else 434, "perception": 192}
    total = sum(expected.values())
    dimensions = {name: int(env.observation_manager.group_obs_dim[name][0]) for name in expected}
    assert dimensions == expected, dimensions
    assert contract["observation_dim"] == total and contract["task"] == TASK
    assert env.action_manager.total_action_dim == 28
    assert env.cfg.curriculum is None
    assert env.step_dt == 1 / 30 and env.physics_dt == 1 / 240
    encoder = env._rotate_bulb_tactile_term.encoder
    report.update(observation_dimensions=dimensions, observation_dim=total, action_dim=28,
                  contract=contract, tactile_policy_enabled=enabled)
    if not enabled:
        assert encoder is None
        assert env.latest_tactile_inputs is env.latest_tactile_latent is None
        assert not hasattr(env, "_brainco_rl_hydroshear_adapter")
        assert not hasattr(env, "_rl_ours_taxim_rgb_adapter")
        report["restoration_runtime_skipped"] = True
        return
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    report.update(encoder_frozen=True,
                  encoder_state_sha256_before=tensor_hash(encoder.state_dict().items()))
    signal = tactile_signal_diagnostics(env)
    report["reset_tactile_signals"] = signal
    assert all(signal["depth_has_contact_per_env"]), (
        "Pregrasp tactile depth has no contact pixels above 1 micrometre in at least one environment: "
        f"{signal['depth_has_contact_per_env']}"
    )


def check_resets(env, args, report):
    import torch
    from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude

    expected = env.cfg._d_peg_pregrasp
    robot, peg, socket = (env.scene[name] for name in ("robot", "object", "socket"))
    samples = []

    def verify(ids):
        q = torch.tensor([expected["robot_joint_positions"][n] for n in robot.joint_names], device=env.device)
        targets = torch.tensor([expected["robot_joint_targets"][n] for n in robot.joint_names], device=env.device)
        torch.testing.assert_close(robot.data.joint_pos[ids], q.expand(len(ids), -1), atol=1e-5, rtol=0)
        torch.testing.assert_close(robot.data.joint_pos_target[ids], targets.expand(len(ids), -1), atol=1e-5, rtol=0)
        pose = expected["peg_pose"]
        p = torch.tensor(pose["position"], device=env.device)
        qpeg = torch.tensor(pose["quaternion"], device=env.device).expand(len(ids), -1)
        torch.testing.assert_close(peg.data.root_pos_w[ids] - env.scene.env_origins[ids], p.expand(len(ids), -1), atol=1e-5, rtol=0)
        assert quat_error_magnitude(peg.data.root_quat_w[ids], qpeg).max().item() < 1e-4
        local = socket.data.root_pos_w[ids] - env.scene.env_origins[ids]
        assert (local[:, 0] - .45).abs().max() <= env.cfg.socket_xy_randomization_m + 1e-6
        assert (local[:, 1] - .10).abs().max() <= env.cfg.socket_xy_randomization_m + 1e-6
        torch.testing.assert_close(local[:, 2], torch.full_like(local[:, 2], .76), atol=1e-6, rtol=0)
        command = env.command_manager.get_command("object_pose")[ids]
        goal, quat = combine_frame_transforms(robot.data.root_pos_w[ids], robot.data.root_quat_w[ids],
                                               command[:, :3], command[:, 3:])
        offset = torch.zeros((len(ids), 3), device=env.device)
        offset[:, 2] = env.cfg.socket_mouth_height_m - env.cfg.insertion_depth_m
        correct, _ = combine_frame_transforms(socket.data.root_pos_w[ids], socket.data.root_quat_w[ids], offset)
        torch.testing.assert_close(goal, correct, atol=1e-5, rtol=0)
        assert quat_error_magnitude(quat, socket.data.root_quat_w[ids]).max().item() < 1e-4

    ids = torch.arange(env.num_envs, device=env.device)
    for index in range(args.resets):
        observations, _ = env.reset()
        finite_tree(observations, f"reset/{index}")
        verify(ids)
        samples.append((socket.data.root_pos_w - env.scene.env_origins).detach().cpu())
    sampled = torch.stack(samples)
    assert sampled[..., :2].std(dim=0).min() > .0002, "Socket randomization did not vary across resets"

    # No simulator step is permitted during a partial reset. Snapshot the other
    # environments' physics/control/task buffers and ensure exact isolation.
    selected, untouched = ids[:1], ids[1:]
    if not len(untouched):
        raise ValueError("Partial-reset isolation needs at least two environments")
    state = env._d_peg_state
    def snapshot():
        result = {
            "robot_joint_pos": robot.data.joint_pos[untouched].clone(),
            "robot_joint_vel": robot.data.joint_vel[untouched].clone(),
            "robot_targets": robot.data.joint_pos_target[untouched].clone(),
            "peg": peg.data.root_state_w[untouched].clone(),
            "socket": socket.data.root_state_w[untouched].clone(),
            "actions": env.action_manager.action[untouched].clone(),
            "commands": env.command_manager.get_command("object_pose")[untouched].clone(),
            "episode_length": env.episode_length_buf[untouched].clone(),
        }
        for name in ("best_depth", "hold_time", "last_step", "success", "entered"):
            result[f"reward/{name}"] = getattr(state, name)[untouched].clone()
        adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
        if adapter is not None:
            for name in ("_batch_state_valid", "_batch_hydrosoft_forces"):
                value = getattr(adapter, name, None)
                if value is not None:
                    result[f"tactile/{name}"] = value.reshape(env.num_envs, 5, *value.shape[1:])[untouched].clone()
        return result
    before = snapshot()
    counter = env.common_step_counter
    env._reset_idx(selected)
    after = snapshot()
    for name, value in before.items():
        torch.testing.assert_close(after[name], value, atol=0, rtol=0, msg=f"Partial reset changed {name}")
    assert env.common_step_counter == counter
    verify(selected)
    # In the actual RL path observations are recomputed after reset. The shear
    # history of an untouched environment must survive that call too.
    observations = env.observation_manager.compute(update_history=False)
    finite_tree(observations, "partial_reset_observations")
    after_observation = snapshot()
    for name, value in before.items():
        torch.testing.assert_close(after_observation[name], value, atol=0, rtol=0,
                                   msg=f"Observation after partial reset changed {name}")
    assert env.common_step_counter == counter
    report.update(full_resets=args.resets, partial_reset_isolated=True,
                  partial_reset_observation_recompute_isolated=True,
                  isolated_buffers=list(before), sampled_socket_min=sampled.amin(dim=(0, 1)).tolist(),
                  sampled_socket_max=sampled.amax(dim=(0, 1)).tolist())


def run_stage(args, report):
    import gymnasium as gym
    import torch
    import BrainCo_DexHand.tasks  # noqa: F401
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from BrainCo_DexHand.tactile_representation.policy import prepare_policy_run, validate_policy_contract
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.dexsuite_revo3_env_cfg_insert_d_peg import (
        DexsuiteRevo3InsertDPegEnvCfg, DexsuiteRevo3InsertDPegEnvCfg_PLAY,
    )
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.agents.rsl_rl_ppo_cfg_insert_d_peg import DexsuiteRevo3InsertDPegPPORunnerCfg
    from duration_logging import DurationLoggingOnPolicyRunner
    from distributed_normalization import install_distributed_normalization_fix

    play_cfg = args.mode in ("play", "reset_play")
    cfg = (DexsuiteRevo3InsertDPegEnvCfg_PLAY if play_cfg else DexsuiteRevo3InsertDPegEnvCfg)()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.tactile_policy_enabled = args.tactile_policy_enabled
    cfg.log_dir = str(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "checkpoint.pt").exists() and args.mode in ("train", "resume"):
        raise FileExistsError(f"Use a new validation output directory: {args.output}")
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    runner = None
    try:
        observations, _ = env.reset()
        finite_tree(observations, "initial_observations")
        check_contract(raw, report)
        prepare_policy_run(env, args.output, resume_path=args.checkpoint)
        dump_yaml(str(args.output / "params/env.yaml"), cfg)
        if args.mode.startswith("reset"):
            check_resets(raw, args, report)
        else:
            agent = DexsuiteRevo3InsertDPegPPORunnerCfg()
            agent.device = args.device
            agent.seed = args.seed
            agent.logger = "tensorboard"
            agent.save_interval = 1
            dump_yaml(str(args.output / "params/agent.yaml"), agent)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            install_step_audit(wrapped, report)
            runner = DurationLoggingOnPolicyRunner(wrapped, agent.to_dict(),
                                                   log_dir=None if args.mode == "play" else str(args.output),
                                                   device=agent.device)
            from d_peg_training import initialize_d_peg_policy, install_d_peg_training_diagnostics
            from BrainCo_DexHand.tasks.manager_based.dexsuite.mdp.d_peg_insertion import d_peg_pop_episode_diagnostics
            if not args.checkpoint:
                initialize_d_peg_policy(runner)
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                runner.load(str(args.checkpoint), map_location=agent.device)
                loaded = policy_module(runner).state_dict()
                assert tensor_hash(loaded.items()) == tensor_hash(checkpoint["model_state_dict"].items())
                torch.testing.assert_close(runner.alg.optimizer.state_dict(), checkpoint["optimizer_state_dict"],
                                           check_device=False, atol=0, rtol=0)
                saved_contract = json.loads((args.checkpoint.parent / "tactile_policy_contract.json").read_text())
                validate_policy_contract(raw.tactile_policy_contract, saved_contract)
                report.update(checkpoint_loaded=str(args.checkpoint), loaded_policy_exact=True, loaded_optimizer_exact=True,
                              loaded_iteration=int(runner.current_learning_iteration), contract_match=True)
            install_distributed_normalization_fix(runner)
            if args.mode != "play":
                install_d_peg_training_diagnostics(
                    runner, episode_metrics_provider=lambda: d_peg_pop_episode_diagnostics(raw),
                )
            policy = policy_module(runner)
            before = tensor_hash(policy.named_parameters())
            report["policy_parameters_sha256_before"] = before
            if args.mode == "play":
                inference = runner.get_inference_policy(device=raw.device)
                observations = wrapped.get_observations()
                for _ in range(args.play_steps):
                    with torch.inference_mode():
                        actions = inference(observations)
                        observations, _, dones, _ = wrapped.step(actions)
                        policy.reset(dones)
                assert tensor_hash(policy.named_parameters()) == before
                report["play_steps"] = args.play_steps
                report["policy_unchanged_during_play"] = True
            else:
                runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
                finite_tree(policy.state_dict(), "trained_policy")
                after = tensor_hash(policy.named_parameters())
                assert before != after, "PPO did not update any policy parameter"
                report.update(iterations_requested=args.iterations,
                              final_iteration=int(runner.current_learning_iteration),
                              policy_parameters_sha256_after=after, policy_weights_changed=True)
                checkpoint_path = args.output / "checkpoint.pt"
                runner.save(str(checkpoint_path), infos={"validation": True, "task": TASK})
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                assert tensor_hash(saved["model_state_dict"].items()) == tensor_hash(policy.state_dict().items())
                report.update(checkpoint=str(checkpoint_path), saved_policy_exact=True,
                              optimizer_state_saved=bool(saved["optimizer_state_dict"]["state"]))
                assert report["optimizer_state_saved"]
        encoder = raw._rotate_bulb_tactile_term.encoder
        if cfg.tactile_policy_enabled:
            report["encoder_state_sha256_after"] = tensor_hash(encoder.state_dict().items())
            assert report["encoder_state_sha256_before"] == report["encoder_state_sha256_after"]
            assert not encoder.training and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
            report["encoder_unchanged"] = True
        else:
            assert encoder is None
            assert raw.latest_tactile_inputs is raw.latest_tactile_latent is None
            assert not hasattr(raw, "_brainco_rl_hydroshear_adapter")
            assert not hasattr(raw, "_rl_ours_taxim_rgb_adapter")
    finally:
        if runner is not None and getattr(runner, "writer", None) is not None:
            runner.writer.close()
        env.close()


def run_suite(args, report):
    report["stages"] = []
    stages = [
        ("reset_train", 2, None, 0),
        ("reset_play", 2, None, 0),
        ("train", args.num_envs, None, args.iterations),
        ("resume", args.num_envs, args.output / "train/checkpoint.pt", args.resume_iterations),
        ("play", args.num_envs, args.output / "resume/checkpoint.pt", 0),
    ]
    for mode, count, checkpoint, iterations in stages:
        stage_dir = args.output / mode
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode,
                   "--output", str(stage_dir), "--device", args.device,
                   "--num-envs", str(count), "--seed", str(args.seed),
                   "--iterations", str(max(1, iterations)), "--resets", str(args.resets),
                   "--play-steps", str(args.play_steps)]
        if args.headless:
            command.append("--headless")
        if args.tactile_policy_enabled:
            command.append("--tactile-policy-enabled")
        if checkpoint:
            command.extend(("--checkpoint", str(checkpoint)))
        stage_dir.mkdir(parents=True, exist_ok=True)
        print(f"VALIDATION_STAGE {mode}: {' '.join(command)}", flush=True)
        with (stage_dir / "console.log").open("w") as console:
            completed = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
        stage_report = stage_dir / "report.json"
        item = json.loads(stage_report.read_text()) if stage_report.exists() else {
            "passed": False, "error": "Child exited before writing a report",
        }
        item.update(mode=mode, returncode=completed.returncode, console_log=str(stage_dir / "console.log"))
        report["stages"].append(item)
        write_report(args.output / "report.json", report)
        if completed.returncode != 0 or not item.get("passed", False):
            raise RuntimeError(f"Validation stage {mode} failed; see {stage_dir / 'console.log'}")


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "train", "resume", "play", "reset_train", "reset_play"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/diagnostics" / f"d_peg_validation_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--resume-iterations", type=int, default=2)
    parser.add_argument("--play-steps", type=int, default=300)
    parser.add_argument("--resets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tactile-policy-enabled", action="store_true",
                        help="Include the frozen multimodal restoration pathway (default: disabled)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if min(args.num_envs, args.iterations, args.resume_iterations, args.play_steps, args.resets) <= 0:
        parser.error("Environment, iteration, step and reset counts must be positive")
    if args.mode in ("resume", "play") and args.checkpoint is None:
        parser.error("--checkpoint is required for resume/play")
    report = dict(task=TASK, mode=args.mode, passed=False, device=args.device, num_envs=args.num_envs,
                  scope="Pipeline and stability validation only; not trained insertion success")
    started, app = time.monotonic(), None
    try:
        if args.mode == "all":
            run_suite(args, report)
        else:
            app = AppLauncher(args).app
            run_stage(args, report)
        report["passed"] = True
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(args.output / "report.json", report)
        print("D_PEG_VALIDATION_REPORT", str(args.output / "report.json"), report["passed"], flush=True)
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
