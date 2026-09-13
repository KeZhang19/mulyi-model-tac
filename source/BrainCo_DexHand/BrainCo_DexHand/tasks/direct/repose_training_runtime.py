"""Task-owned Repose curriculum, incremental control and globally counted diagnostics."""
from __future__ import annotations

import math
import torch
from isaaclab.envs import DirectRLEnv

from .repose_training_math import (
    advance_curriculum, compute_repose_reward_terms, quaternion_angle_distance,
    sample_bounded_joint_positions, sample_relative_goal_quaternions, update_stable_success,
)


class ReposeTrainingRuntime:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg
        n, device = env.num_envs, env.device
        env.repose_curriculum_stage = 0
        env.repose_hold_steps = torch.zeros(n, dtype=torch.long, device=device)
        env.repose_previous_angle = torch.zeros(n, device=device)
        env.repose_last_reached = torch.zeros(n, dtype=torch.bool, device=device)
        env.repose_last_fallen = torch.zeros(n, dtype=torch.bool, device=device)
        env.repose_reset_joint_positions = torch.zeros_like(env.hand_dof_targets)
        env.repose_reset_goal_angle = torch.zeros(n, device=device)
        env.repose_reward_terms = {}
        self.filtered_actions = torch.zeros_like(env.actions)
        self.previous_actions = torch.zeros_like(env.actions)
        self.episode_steps = torch.zeros(n, device=device)
        self.episode_returns = torch.zeros(n, device=device)
        self.episode_terms = {k: torch.zeros(n, device=device) for k in (
            "progress", "orientation", "position", "action", "action_delta", "success", "fall",
        )}
        # These counters contain global sums, identical on all distributed ranks.
        self.completed = torch.zeros((), device=device)
        self.successful = torch.zeros((), device=device)
        self.falls = torch.zeros((), device=device)
        self.timeouts = torch.zeros((), device=device)
        self.target_hits = torch.zeros((), device=device)
        self.window_completed = torch.zeros((), device=device)
        self.window_successful = torch.zeros((), device=device)
        self.completed_returns = torch.zeros((), device=device)
        self.completed_terms = {k: torch.zeros((), device=device) for k in self.episode_terms}

    def prepare_action(self, actions):
        e, c = self.env, self.cfg
        self.previous_actions = self.filtered_actions.clone()
        e.actions = actions.clamp(-1.0, 1.0).clone()
        alpha = c.repose_action_filter
        self.filtered_actions = alpha * e.actions + (1.0 - alpha) * self.filtered_actions
        ids = e.actuated_dof_indices
        proposed = e.prev_targets[:, ids] + c.repose_action_speed * e.step_dt * self.filtered_actions
        e.cur_targets[:, ids] = torch.maximum(torch.minimum(proposed, e.hand_dof_upper_limits[:, ids]),
                                               e.hand_dof_lower_limits[:, ids])
        e.prev_targets[:, ids] = e.cur_targets[:, ids]
        self.target_saturation = (e.cur_targets[:, ids] != proposed).float().mean(-1)

    def apply_action(self):
        e = self.env
        e.hand.set_joint_position_target(e.cur_targets[:, e.actuated_dof_indices],
                                        joint_ids=e.actuated_dof_indices)

    def reset(self, env_ids):
        e, c = self.env, self.cfg
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=e.device)
        if env_ids.numel() == 0:
            return
        DirectRLEnv._reset_idx(e, env_ids)
        state = e.object.data.default_root_state[env_ids].clone()
        state[:, :3] += c.repose_position_noise * (2 * torch.rand_like(state[:, :3]) - 1)
        state[:, :3] += e.scene.env_origins[env_ids]
        state[:, 3:7] = sample_relative_goal_quaternions(
            state[:, 3:7], 0.0, math.radians(c.repose_object_rotation_noise_deg),
        )
        state[:, 7:] = 0
        e.object.write_root_pose_to_sim(state[:, :7], env_ids)
        e.object.write_root_velocity_to_sim(state[:, 7:], env_ids)
        dof_pos = sample_bounded_joint_positions(
            e.hand.data.default_joint_pos[env_ids], e.hand_dof_lower_limits[env_ids],
            e.hand_dof_upper_limits[env_ids], c.repose_joint_noise_fraction,
        )
        e.prev_targets[env_ids] = dof_pos
        e.cur_targets[env_ids] = dof_pos
        e.hand_dof_targets[env_ids] = dof_pos
        e.repose_reset_joint_positions[env_ids] = dof_pos
        e.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        e.hand.write_joint_state_to_sim(dof_pos, torch.zeros_like(dof_pos), env_ids=env_ids)
        e.actions[env_ids] = 0
        self.filtered_actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.episode_steps[env_ids] = 0
        self.episode_returns[env_ids] = 0
        for value in self.episode_terms.values():
            value[env_ids] = 0
        e.successes[env_ids] = 0
        e._compute_intermediate_values()
        # Goals must use the NEW episode's object pose, never the previous episode.
        self.reset_target(env_ids)
        e.repose_reset_goal_angle[env_ids] = e.repose_previous_angle[env_ids]

    def reset_target(self, env_ids):
        e, c = self.env, self.cfg
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=e.device)
        if env_ids.numel() == 0:
            return
        maximum = c.repose_goal_max_degrees[e.repose_curriculum_stage]
        e.goal_rot[env_ids] = sample_relative_goal_quaternions(
            e.object_rot[env_ids], math.radians(c.repose_goal_min_deg), math.radians(maximum),
        )
        e.repose_previous_angle[env_ids] = quaternion_angle_distance(e.object_rot[env_ids], e.goal_rot[env_ids])
        e.repose_hold_steps[env_ids] = 0
        e.reset_goal_buf[env_ids] = False
        goal_pos = e.goal_pos + e.scene.env_origins
        dot_pos = e.in_hand_pos + e.scene.env_origins
        dot_pos[:, 2] += 0.02
        marker_ids = torch.cat((torch.zeros(e.num_envs, device=e.device, dtype=torch.long),
                                torch.ones(e.num_envs, device=e.device, dtype=torch.long)))
        e.goal_markers.visualize(torch.cat((goal_pos, dot_pos)), torch.cat((e.goal_rot, e.goal_rot)),
                                 marker_indices=marker_ids)

    def rewards(self):
        e, c = self.env, self.cfg
        angle = quaternion_angle_distance(e.object_rot, e.goal_rot)
        distance = (e.object_pos - e.in_hand_pos).norm(dim=-1)
        fallen = distance >= c.fall_dist
        hold, reached = update_stable_success(
            angle, distance, e.object_linvel.norm(dim=-1), e.object_angvel.norm(dim=-1),
            e.repose_hold_steps, angle_tolerance=c.success_tolerance,
            position_tolerance=c.repose_success_position, max_linear_speed=c.repose_success_linear_speed,
            max_angular_speed=c.repose_success_angular_speed, required_hold_steps=c.repose_success_hold_steps,
            fallen=fallen,
        )
        e.repose_hold_steps = hold
        e.repose_last_reached = reached.clone()
        e.repose_last_fallen = fallen.clone()
        terms = compute_repose_reward_terms(
            e.repose_previous_angle, angle, distance, self.filtered_actions, self.previous_actions,
            reached, fallen, progress_scale=c.repose_progress_scale,
            orientation_scale=c.repose_orientation_scale, position_scale=c.repose_position_scale,
            position_deadzone=c.repose_position_deadzone, action_scale=c.repose_action_scale,
            action_delta_scale=c.repose_action_delta_scale, success_bonus=c.repose_success_bonus,
            fall_penalty=c.repose_fall_penalty,
        )
        e.repose_reward_terms = terms
        e.successes += reached.float()
        self.episode_steps += 1
        self.episode_returns += terms["total"]
        for k in self.episode_terms:
            self.episode_terms[k] += terms[k]
        self._log_global_metrics(terms, angle, distance, reached, fallen)
        e.repose_previous_angle = angle.clone()
        # Successful target changes are excluded from the next progress difference.
        self.reset_target(reached.nonzero(as_tuple=False).flatten())
        return terms["total"]

    def _log_global_metrics(self, terms, angle, distance, reached, fallen):
        e, c = self.env, self.cfg
        done = e.reset_buf.bool()
        contact = torch.zeros((), device=e.device)
        if hasattr(e, "latest_tactile_depth"):
            contact = (e.latest_tactile_depth.flatten(2).amax(-1) > 1e-6).float().mean()
        keys = list(terms)
        # Every rank calls this collective once per control step, including zero resets.
        packed = torch.stack([terms[k].sum() for k in keys] + [
            torch.tensor(float(e.num_envs), device=e.device), angle.sum(), distance.sum(),
            (e.actions.abs() >= 0.98).float().mean() * e.num_envs,
            getattr(self, "target_saturation", torch.zeros_like(angle)).sum(), contact * e.num_envs,
            done.float().sum(), (done & (e.successes > 0)).float().sum(),
            (done & fallen).float().sum(), (done & e.reset_time_outs).float().sum(),
            (e.successes * done).sum(), (self.episode_returns * done).sum(),
        ] + [(self.episode_terms[k] * done).sum() for k in self.episode_terms])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed)
        offset = len(keys)
        count = packed[offset]
        stats = packed[offset + 1:offset + 12]
        ended, won, falls, timeouts, hits, returns = stats[5:11]
        self.completed += ended
        self.successful += won
        self.falls += falls
        self.timeouts += timeouts
        self.target_hits += hits
        self.completed_returns += returns
        self.window_completed += ended
        self.window_successful += won
        for i, k in enumerate(self.episode_terms):
            self.completed_terms[k] += packed[offset + 12 + i]
        self.maybe_advance_curriculum()
        denominator = self.completed.clamp_min(1)
        # Fresh dict and scalar storage: RSL-RL retains each step's log until update.
        log = {"Reward/" + k: (packed[i] / count).clone() for i, k in enumerate(keys)}
        log.update({
            "Metrics/angle_deg": stats[0] / count * (180 / math.pi),
            "Metrics/position_error_m": stats[1] / count,
            "Actions/near_bound_fraction": stats[2] / count,
            "Actions/joint_limit_fraction": stats[3] / count,
            "Sensors/previous_contact_fraction": stats[4] / count,
            "Episode/success_rate": (self.successful / denominator).clone(),
            "Episode/fall_rate": (self.falls / denominator).clone(),
            "Episode/timeout_rate": (self.timeouts / denominator).clone(),
            "Episode/targets_per_episode": (self.target_hits / denominator).clone(),
            "Episode/completed_count": self.completed.clone(),
            "Episode/return": (self.completed_returns / denominator).clone(),
            "Curriculum/stage": torch.tensor(float(e.repose_curriculum_stage), device=e.device),
            "Curriculum/max_angle_deg": torch.tensor(c.repose_goal_max_degrees[e.repose_curriculum_stage], device=e.device),
        })
        for k in self.completed_terms:
            log["EpisodeReward/" + k] = (self.completed_terms[k] / denominator).clone()
        e.extras["log"] = log

    def maybe_advance_curriculum(self):
        e, c = self.env, self.cfg
        if not c.repose_curriculum_enabled or e.common_step_counter % c.repose_curriculum_interval:
            return
        stage, clear = advance_curriculum(
            e.repose_curriculum_stage, int(self.window_completed.item()), int(self.window_successful.item()),
            stage_count=len(c.repose_goal_max_degrees), minimum_episodes=c.repose_curriculum_min_episodes,
            success_threshold=c.repose_curriculum_success_threshold,
        )
        e.repose_curriculum_stage = stage
        if clear:
            self.window_completed.zero_()
            self.window_successful.zero_()

    def state_dict(self):
        return {"revision": 2, "stage": self.env.repose_curriculum_stage,
                "window_completed": self.window_completed.item(),
                "window_successful": self.window_successful.item()}

    def load_state_dict(self, state):
        if state.get("revision") != 2 or not 0 <= int(state["stage"]) < len(self.cfg.repose_goal_max_degrees):
            raise ValueError("Invalid Repose curriculum checkpoint")
        self.env.repose_curriculum_stage = int(state["stage"])
        self.window_completed.fill_(state["window_completed"])
        self.window_successful.fill_(state["window_successful"])
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.reset_target(env_ids)
        self.env._observation_cache_key = None
        if hasattr(self.env, "_history"):
            self.env._history.reset(env_ids)
