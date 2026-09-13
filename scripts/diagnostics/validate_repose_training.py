"""Exercise the Repose v2 environment in Isaac Sim before a fresh PPO run.

Run with the same Isaac Lab Python and encoder as the production job. This
validator never loads an RL checkpoint or updates a policy. Its controlled
reward probes run only after the real simulation rollout has completed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import MethodType


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "BrainCo_DexHand"))


def validate_repose_training() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--encoder", type=str, default=str(
        ROOT / "runs/revo3_encoder_exports/sim_policy_encoder_unaligned.pt"
    ))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 4 or args.steps < 310:
        parser.error("Validation requires at least 4 environments and 310 control steps")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "task": args.task, "num_envs": args.num_envs,
        "steps_requested": args.steps, "seed": args.seed, "policy_checkpoint": None,
        "optimizer_updates": 0, "checks": {}, "diagnostics": {},
    }
    started = time.monotonic()
    app = None
    env = None
    try:
        launcher = AppLauncher(args)
        app = launcher.app
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        import BrainCo_DexHand  # noqa: F401 -- Gym registrations

        cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.tactile_policy_checkpoint = args.encoder
        cfg.log_dir = str(output)
        assert cfg.repose_training_revision == 2, "Expected the Repose v2 production configuration"
        env = gym.make(args.task, cfg=cfg)
        raw = env.unwrapped
        report["max_episode_length"] = raw.max_episode_length
        report["tactile_policy_contract"] = raw.tactile_policy_contract
        device = raw.device
        all_ids = torch.arange(raw.num_envs, device=device)
        checks = report["checks"]
        diag = report["diagnostics"]
        reset_kind = "initial"
        reset_calls = {"initial": 0, "partial": 0, "natural": 0, "controlled": 0}
        reset_envs = dict.fromkeys(reset_calls, 0)
        reset_angles = []
        joint_limit_max_violation = 0.0
        reward_error = 0.0
        reward_sums = {}
        reward_samples = 0
        last_reward = None
        last_reached = None
        last_fallen = None
        original_reset = raw._reset_idx
        original_reward = raw._get_rewards

        def check_finite(value, label):
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all().item(), f"Non-finite {label}"
            elif hasattr(value, "items"):
                for key, child in value.items():
                    check_finite(child, f"{label}.{key}")

        def snapshot(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if hasattr(value, "items"):
                return {key: snapshot(child) for key, child in value.items()}
            return value

        def assert_unchanged(value, saved, label):
            if isinstance(saved, torch.Tensor):
                assert torch.equal(value, saved), f"Retained {label} mutated"
            elif isinstance(saved, dict):
                assert value.keys() == saved.keys(), f"Retained {label} keys mutated"
                for key, child in saved.items():
                    assert_unchanged(value[key], child, f"{label}.{key}")
            else:
                assert value == saved, f"Retained {label} mutated"

        def reset_checked(self, env_ids):
            nonlocal joint_limit_max_violation
            ids = all_ids if env_ids is None else torch.as_tensor(env_ids, device=device, dtype=torch.long)
            result = original_reset(env_ids)
            if not ids.numel():
                return result
            positions = self.repose_reset_joint_positions[ids]
            lower, upper = self.hand_dof_lower_limits[ids], self.hand_dof_upper_limits[ids]
            violation = torch.maximum(lower - positions, positions - upper).clamp_min(0)
            joint_limit_max_violation = max(joint_limit_max_violation, violation.max().item())
            assert joint_limit_max_violation <= 1e-6, "Reset joint command exceeds physical limits"
            angles = self.repose_reset_goal_angle[ids]
            check_finite(positions, "reset positions")
            check_finite(angles, "reset target angle")
            if self.repose_curriculum_stage == 0:
                assert angles.min().item() >= math.radians(20) - 2e-4, "Stage 0 target is below 20 degrees"
                assert angles.max().item() <= math.radians(45) + 2e-4, "Stage 0 target exceeds 45 degrees"
            reset_angles.extend(torch.rad2deg(angles).cpu().tolist())
            reset_calls[reset_kind] += 1
            reset_envs[reset_kind] += ids.numel()
            return result

        def rewards_checked(self):
            nonlocal reward_error, reward_samples, last_reward, last_reached, last_fallen
            reward = original_reward()
            terms = self.repose_reward_terms
            expected_keys = {"position", "orientation", "progress", "action", "action_delta", "success", "fall", "total"}
            assert set(terms) == expected_keys, f"Unexpected reward components: {set(terms)}"
            check_finite(terms, "reward components")
            summed = sum(value for key, value in terms.items() if key != "total")
            difference = max((summed - reward).abs().max().item(), (terms["total"] - reward).abs().max().item())
            reward_error = max(reward_error, difference)
            assert difference < 2e-4, f"Reward decomposition error: {difference}"
            last_reward = reward.clone()
            # Preserve terminal diagnostics before DirectRLEnv resets done rows.
            last_reached = self.repose_last_reached.clone()
            last_fallen = self.repose_last_fallen.clone()
            if reset_kind == "natural":
                for key, value in terms.items():
                    reward_sums[key] = reward_sums.get(key, 0.0) + value.sum().item()
                reward_samples += self.num_envs
            return reward

        raw._reset_idx = MethodType(reset_checked, raw)
        raw._get_rewards = MethodType(rewards_checked, raw)
        with torch.inference_mode():
            obs, extras = env.reset(seed=args.seed)
            check_finite(obs, "initial observations")
            checks["initial_joint_commands_within_limits"] = True
            checks["initial_relative_target_20_to_45_degrees"] = True

            reset_kind = "partial"
            partial_ids = all_ids[::2]
            untouched_ids = all_ids[1::2]
            retained_obs = obs
            retained_copy = snapshot(obs)
            old_history = raw._history.frames[untouched_ids].clone()
            old_goal = raw.goal_rot[untouched_ids].clone()
            raw._reset_idx(partial_ids)
            assert_unchanged(retained_obs, retained_copy, "observation after partial reset")
            assert torch.equal(raw._history.frames[untouched_ids], old_history), "Partial reset changed another environment's history"
            assert torch.equal(raw.goal_rot[untouched_ids], old_goal), "Partial reset changed another environment's goal"
            assert raw._history.needs_fill[partial_ids].all().item(), "Partial reset did not mark history for refill"
            obs = raw._get_observations()
            checks["partial_reset_isolation_and_retained_observation"] = True

            reset_kind = "natural"
            contact_samples = 0
            contact_positive = 0
            finger_contact_positive = torch.zeros(5, device=device)
            max_depth = 0.0
            latent_delta_sum = 0.0
            latent_delta_max = 0.0
            done_count = 0
            timeout_count = 0
            fall_count = 0
            reached_count = 0
            max_target_step = 0.0
            previous_log = None
            previous_log_saved = None
            joints = len(raw.actuated_dof_indices)
            phase = torch.arange(joints, device=device, dtype=torch.float).unsqueeze(0)
            phase = phase + all_ids.float().unsqueeze(1) * 0.19
            for step in range(args.steps):
                # Rest first, then use smooth bounded motions to exercise sensors,
                # action targets, falls, and real episode timeouts without a policy.
                actions = torch.zeros(raw.num_envs, joints, device=device)
                if step >= 50:
                    actions = 0.35 * torch.sin(phase + step * 0.045)
                old_obs, old_obs_saved = obs, snapshot(obs)
                old_latent = raw.latest_tactile_latent.clone()
                old_targets = raw.cur_targets[:, raw.actuated_dof_indices].clone()
                obs, reward, terminated, truncated, extras = env.step(actions)
                assert_unchanged(old_obs, old_obs_saved, "observation after step")
                if previous_log is not None:
                    assert extras["log"] is not previous_log, "Step reused the preceding log dictionary"
                    assert_unchanged(previous_log, previous_log_saved, "log dictionary")
                previous_log = extras["log"]
                previous_log_saved = snapshot(previous_log)
                for label, value in (("observation", obs), ("reward", reward), ("log", previous_log),
                                     ("joint state", raw.hand.data.joint_pos),
                                     ("object state", raw.object.data.root_state_w),
                                     ("tactile depth", raw.latest_tactile_depth),
                                     ("tactile latent", raw.latest_tactile_latent)):
                    check_finite(value, label)
                assert last_reward is not None and torch.equal(reward, last_reward)
                dones = terminated | truncated
                active = ~dones
                if active.any():
                    step_delta = (raw.cur_targets[:, raw.actuated_dof_indices] - old_targets).abs()[active].max().item()
                    max_target_step = max(max_target_step, step_delta)
                    assert step_delta <= 1.5 * raw.step_dt + 1e-5, "Action integrated more than once per control step"
                done_count += dones.sum().item()
                timeout_count += truncated.sum().item()
                fall_count += last_fallen.sum().item()
                reached_count += last_reached.sum().item()
                depth = raw.latest_tactile_depth.reshape(raw.num_envs, 5, -1)
                finger_max = depth.amax(-1)
                contact_positive += (finger_max > 1e-6).sum().item()
                finger_contact_positive += (finger_max > 1e-6).sum(dim=0)
                contact_samples += finger_max.numel()
                max_depth = max(max_depth, finger_max.max().item())
                latent_delta = (raw.latest_tactile_latent - old_latent).norm(dim=-1)
                latent_delta_sum += latent_delta.mean().item()
                latent_delta_max = max(latent_delta_max, latent_delta.max().item())
                if (step + 1) % 50 == 0:
                    print("VALIDATION_PROGRESS", json.dumps({"step": step + 1, "dones": done_count,
                          "timeouts": timeout_count, "target_hits": reached_count,
                          "contact_fraction": contact_positive / contact_samples}), flush=True)

            assert reset_envs["natural"] > 0 and done_count > 0, "Rollout did not exercise a natural episode reset"
            checks["natural_episode_resets_exercised"] = True
            checks["rollout_finite"] = True
            checks["reward_components_equal_returned_reward"] = True
            checks["log_dictionary_is_independent_each_step"] = True
            checks["prior_observations_unchanged_across_steps"] = True
            checks["action_targets_integrated_once_per_control_step"] = True
            diag.update({
                "reset_calls": reset_calls, "reset_environment_count": reset_envs,
                "reset_joint_max_limit_violation_rad": joint_limit_max_violation,
                "reset_relative_angle_min_deg": min(reset_angles),
                "reset_relative_angle_max_deg": max(reset_angles),
                "reward_decomposition_max_error": reward_error,
                "reward_component_per_step_mean": {key: value / reward_samples for key, value in reward_sums.items()},
                "completed_episodes": done_count, "timeouts": timeout_count,
                "falls": fall_count, "held_target_hits": reached_count,
                "action_target_max_control_step_delta_rad": max_target_step,
                "tactile_finger_contact_fraction_depth_over_1um": contact_positive / contact_samples,
                "tactile_contact_fraction_by_finger": dict(zip(
                    ("little", "ring", "middle", "index", "thumb"),
                    (finger_contact_positive / (raw.num_envs * args.steps)).cpu().tolist(),
                )),
                "tactile_depth_max_m": max_depth,
                "tactile_latent_mean_step_delta_l2": latent_delta_sum / args.steps,
                "tactile_latent_max_step_delta_l2": latent_delta_max,
                "contact_diagnostic_note": "Contact fraction and latent variation are measurements; no minimum contact is assumed for an untrained controller.",
                "curriculum_stage_after_rollout": raw.repose_curriculum_stage,
            })
            reset_kind = "controlled"
            controlled_repose_checks(raw, checks, diag)

        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("VALIDATION_RESULT", json.dumps(report), flush=True)
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()


def controlled_repose_checks(raw, checks, diagnostics):
    """Probe hold gating and curriculum thresholds after the physical rollout."""
    import torch
    from BrainCo_DexHand.tasks.direct.repose_training_math import (
        quaternion_angle_distance, sample_relative_goal_quaternions,
    )

    cfg, runtime = raw.cfg, raw._repose_training
    ids = torch.arange(raw.num_envs, device=raw.device)
    raw._reset_idx(ids)
    raw.reset_buf.zero_()
    raw.reset_time_outs.zero_()
    runtime.filtered_actions.zero_()
    runtime.previous_actions.zero_()
    raw.actions.zero_()

    def stable_state():
        # Replace cached intermediate tensors, without teleporting the simulation.
        # These are reward callback probes, explicitly separate from rollout data.
        raw.object_pos = raw.in_hand_pos.clone()
        raw.object_rot = raw.goal_rot.clone()
        raw.object_linvel = torch.zeros_like(raw.object_linvel)
        raw.object_angvel = torch.zeros_like(raw.object_angvel)
        raw.repose_previous_angle.zero_()

    stable_state()
    raw.repose_hold_steps.zero_()
    successes_before = raw.successes.clone()
    goal_before = raw.goal_rot.clone()
    for step in range(cfg.repose_success_hold_steps):
        raw._get_rewards()
        expected = step == cfg.repose_success_hold_steps - 1
        assert raw.repose_last_reached.eq(expected).all().item(), f"Success hold fired at incorrect step {step + 1}"
        expected_bonus = cfg.repose_success_bonus if expected else 0.0
        torch.testing.assert_close(raw.repose_reward_terms["success"],
                                   torch.full_like(raw.successes, expected_bonus))
    torch.testing.assert_close(raw.successes, successes_before + 1)
    assert (quaternion_angle_distance(goal_before, raw.goal_rot) >= math.radians(20) - 2e-4).all().item()
    assert not raw.repose_hold_steps.any().item(), "Target refresh did not clear hold counters"
    raw._get_rewards()
    assert not raw.repose_last_reached.any().item(), "Target refresh emitted a duplicate success"
    assert raw.repose_reward_terms["progress"].abs().max().item() < 2e-5, "Target refresh generated artificial progress"
    checks["success_requires_consecutive_stable_hold"] = True
    checks["success_bonus_once_then_target_refresh"] = True
    checks["new_target_resets_progress_baseline"] = True

    for invalid in ("angle", "position", "linear_speed", "angular_speed", "fall"):
        stable_state()
        raw.repose_hold_steps.fill_(cfg.repose_success_hold_steps - 1)
        if invalid == "angle":
            error = cfg.success_tolerance + 0.05
            raw.object_rot = sample_relative_goal_quaternions(raw.goal_rot, error, error)
        elif invalid == "position":
            raw.object_pos[:, 0] += cfg.repose_success_position + 0.01
        elif invalid == "linear_speed":
            raw.object_linvel[:, 0] = cfg.repose_success_linear_speed + 0.01
        elif invalid == "angular_speed":
            raw.object_angvel[:, 0] = cfg.repose_success_angular_speed + 0.01
        else:
            raw.object_pos[:, 0] += cfg.fall_dist + 0.01
        raw._get_rewards()
        assert not raw.repose_last_reached.any().item(), f"Unstable {invalid} state was counted as success"
        assert not raw.repose_hold_steps.any().item(), f"Unstable {invalid} state did not clear hold"
        assert not raw.repose_reward_terms["success"].any().item(), f"Unstable {invalid} received a success bonus"
    checks["angle_position_velocity_and_fall_gate_success"] = True

    # Test the production episode logger with deliberately different episode and
    # target counts: four endings, two successful episodes, five target hits.
    # The successful+falling row is an episode that reached goals before falling.
    raw.successes.zero_()
    raw.successes[0], raw.successes[2] = 3, 2
    raw.reset_buf.zero_()
    raw.reset_buf[:4] = True
    raw.reset_time_outs.zero_()
    raw.reset_time_outs[2:4] = True
    fallen = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    fallen[:2] = True
    reached = torch.zeros_like(fallen)
    count_before = {name: getattr(runtime, name).clone() for name in (
        "completed", "successful", "falls", "timeouts", "target_hits", "completed_returns",
    )}
    previous_completed_terms = {key: value.clone() for key, value in runtime.completed_terms.items()}
    runtime.episode_returns.copy_(torch.arange(raw.num_envs, device=raw.device) + 10)
    for index, value in enumerate(runtime.episode_terms.values()):
        value.copy_(torch.arange(raw.num_envs, device=raw.device) + index)
    expected_returns = runtime.episode_returns[:4].sum().clone()
    expected_terms = {key: value[:4].sum().clone() for key, value in runtime.episode_terms.items()}
    saved_counter = raw.common_step_counter
    raw.common_step_counter = 1  # Keep this isolated logger probe outside a curriculum sync.
    angle = torch.linspace(0.25, 1.0, raw.num_envs, device=raw.device)
    distance = torch.linspace(0.01, 0.10, raw.num_envs, device=raw.device)
    runtime._log_global_metrics(raw.repose_reward_terms, angle, distance, reached, fallen)
    for name, increment in (("completed", 4), ("successful", 2), ("falls", 2),
                            ("timeouts", 2), ("target_hits", 5)):
        torch.testing.assert_close(getattr(runtime, name) - count_before[name],
                                   torch.full_like(count_before[name], increment))
    torch.testing.assert_close(runtime.completed_returns, count_before["completed_returns"] + expected_returns)
    for key, expected in expected_terms.items():
        torch.testing.assert_close(runtime.completed_terms[key], previous_completed_terms[key] + expected)
    torch.testing.assert_close(raw.extras["log"]["Metrics/angle_deg"], angle.mean() * (180 / math.pi))
    torch.testing.assert_close(raw.extras["log"]["Metrics/position_error_m"], distance.mean())
    torch.testing.assert_close(raw.extras["log"]["Episode/success_rate"], runtime.successful / runtime.completed)
    checks["episode_logging_counts_episodes_separately_from_target_hits"] = True
    checks["packed_logger_reward_and_metric_offsets"] = True

    minimum = cfg.repose_curriculum_min_episodes
    interval = cfg.repose_curriculum_interval
    raw.repose_curriculum_stage = 0
    runtime.window_completed.fill_(minimum)
    runtime.window_successful.fill_(minimum)
    raw.common_step_counter = interval - 1
    runtime.maybe_advance_curriculum()
    assert raw.repose_curriculum_stage == 0, "Curriculum advanced outside its synchronization interval"
    assert runtime.window_completed.item() == minimum

    raw.common_step_counter = interval
    runtime.window_completed.fill_(minimum - 1)
    runtime.window_successful.fill_(minimum - 1)
    runtime.maybe_advance_curriculum()
    assert raw.repose_curriculum_stage == 0, "Curriculum advanced without enough completed episodes"
    assert runtime.window_completed.item() == minimum - 1

    runtime.window_completed.fill_(minimum)
    runtime.window_successful.fill_(math.ceil(minimum * cfg.repose_curriculum_success_threshold) - 1)
    runtime.maybe_advance_curriculum()
    assert raw.repose_curriculum_stage == 0, "Curriculum advanced below its successful-episode threshold"
    assert runtime.window_completed.item() == 0 and runtime.window_successful.item() == 0

    tested_stages = []
    for expected_stage in range(1, len(cfg.repose_goal_max_degrees)):
        runtime.window_completed.fill_(minimum)
        runtime.window_successful.fill_(math.ceil(minimum * cfg.repose_curriculum_success_threshold))
        runtime.maybe_advance_curriculum()
        assert raw.repose_curriculum_stage == expected_stage
        assert runtime.window_completed.item() == 0 and runtime.window_successful.item() == 0
        samples = []
        for _ in range(math.ceil(64 / raw.num_envs)):
            runtime.reset_target(ids)
            samples.append(quaternion_angle_distance(raw.object_rot, raw.goal_rot))
        error = torch.cat(samples)
        assert error.min().item() >= math.radians(cfg.repose_goal_min_deg) - 2e-4
        assert error.max().item() <= math.radians(cfg.repose_goal_max_degrees[expected_stage]) + 2e-4
        # A broad sample should actually enter the newly added difficulty band.
        assert error.max().item() > math.radians(cfg.repose_goal_max_degrees[expected_stage - 1])
        tested_stages.append({"stage": expected_stage, "maximum_angle_deg": cfg.repose_goal_max_degrees[expected_stage],
                              "sampled_maximum_deg": torch.rad2deg(error).max().item()})
    runtime.window_completed.fill_(minimum)
    runtime.window_successful.fill_(minimum)
    runtime.maybe_advance_curriculum()
    assert raw.repose_curriculum_stage == len(cfg.repose_goal_max_degrees) - 1, "Curriculum exceeded its final stage"
    raw.common_step_counter = saved_counter
    checks["curriculum_interval_minimum_episodes_and_success_threshold"] = True
    checks["curriculum_advances_one_stage_and_stops_at_final_stage"] = True
    checks["all_curriculum_stage_target_ranges"] = True
    diagnostics["controlled_curriculum_stage_probes"] = tested_stages
    diagnostics["controlled_probe_note"] = "Synthetic cached reward states and explicit episode counters, run after the real rollout; excluded from reported rollout returns/contact/successes."


if __name__ == "__main__":
    validate_repose_training()
