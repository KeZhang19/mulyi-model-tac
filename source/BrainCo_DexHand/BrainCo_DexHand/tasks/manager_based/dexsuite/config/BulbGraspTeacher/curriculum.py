"""Episode-local grasp above a socket -> shallow tip entry -> three screw turns.

The frozen teacher MDP remains the source of all original rewards and resets.
This module owns the new command, outcome and orientation reward only.
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply_inverse, subtract_frame_transforms

from .flexiv_rewards import flexiv_hand_contacts
from .insertion import bulb_tip_position_w, insertion_errors, insertion_tip_target
from .screwing import ScrewInsertionState, screw_contact_quality, socket_twist_phase
from .mdp.commands.pose_commands import ObjectUniformPoseCommand
from .mdp.curriculums import DifficultyScheduler


def reset_socket_from_target_ranges(env, env_ids, command_name: str = "object_pose"):
    """Sample the original goal XY in the robot frame; keep the socket on the table.

    Runs after the original robot/table reset events and before command reset.
    Socket randomization belongs to episode reset, never to command timer resampling.
    """
    cfg = getattr(env.cfg.commands, command_name)
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None:
        ids = ids[env_ids]
    if ids.numel() == 0:
        return
    robot, socket = env.scene[cfg.asset_name], env.scene[cfg.socket_name]
    xy_b = torch.zeros_like(robot.data.root_pos_w[ids])
    xy_b[:, 0].uniform_(*cfg.ranges.pos_x)
    xy_b[:, 1].uniform_(*cfg.ranges.pos_y)
    xy_w = robot.data.root_pos_w[ids] + quat_apply(robot.data.root_quat_w[ids], xy_b)
    pose = socket.data.default_root_state[ids, :7].clone()
    pose[:, :2] = xy_w[:, :2]
    # The inherited table randomization changes XY/size in XY only; Z scale is 1.
    table_top = env.scene["table"].data.root_pos_w[ids, 2] + env.cfg.scene.table.spawn.size[2] / 2
    pose[:, 2] = table_top - cfg.socket_bottom_x_m * env._bulb_socket_scales[ids]
    socket.write_root_pose_to_sim(pose, env_ids=ids)
    socket.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)


def thread_down_error(quat_w: torch.Tensor) -> torch.Tensor:
    """Tilt in radians of bulb-local -X toward world -Z; independent of spin."""
    axis = torch.zeros_like(quat_w[:, :3])
    axis[:, 0] = -1.0
    direction = quat_apply(quat_w, axis)
    # atan2 avoids acos amplification of float32 roundoff near 0 and 180 deg.
    return torch.atan2(torch.linalg.vector_norm(direction[:, :2], dim=-1), -direction[:, 2])


class BulbGraspCurriculumCommand(ObjectUniformPoseCommand):
    """7D command: root position in stage one, direct tip position thereafter."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.socket = env.scene[cfg.socket_name]
        self.scales = env._bulb_socket_scales
        self.stage_two = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stage_three = torch.zeros_like(self.stage_two)
        self.stage_two_bonus_credit = torch.zeros(self.num_envs, device=self.device)
        self.stage_three_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.screw_hold_steps = torch.zeros_like(self.stage_three_steps)
        self.final_bonus_paid = torch.zeros_like(self.stage_three)
        self.final_bonus_credit = torch.zeros(self.num_envs, device=self.device)
        self.screw = ScrewInsertionState(self.num_envs, self.device, self.scales, cfg)
        # The shallow-entry stage has no hold gate; only final screwing uses it.
        self.required_hold_steps = math.ceil(cfg.insertion_hold_s / env.step_dt)
        self.metrics["stage"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["stage1_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stage2_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["thread_down_error_deg"] = torch.zeros(self.num_envs, device=self.device)
        for name in ("insertion_depth_m", "insertion_radial_error_m", "insertion_tilt_deg",
                     "insertion_speed_m_s", "insertion_angular_speed_rad_s", "insertion_hold_s",
                     "insertion_candidate", "insertion_tip_radial_error_m",
                     "insertion_depth_ok", "insertion_tip_radial_ok", "shallow_tilt_ok", "screw_tilt_ok"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        for name in ("stage3_success", "screw_turns", "screw_angular_turns", "screw_best_turns",
                     "screw_depth_target_m", "screw_helix_error_m", "screw_hold_s", "stage3_elapsed_s"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        # The pre-reward hook already measured the terminal physical state.
        # Preserve its final success before base reset clears metrics/resamples.
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids]
        self.stage_two[ids] = False
        self.stage_three[ids] = False
        self.stage_two_bonus_credit[ids] = 0
        self.stage_three_steps[ids] = 0
        self.screw_hold_steps[ids] = 0
        self.final_bonus_paid[ids] = False
        self.final_bonus_credit[ids] = 0
        self.screw.reset(ids)
        return super().reset(ids)

    def _socket_target_pose_b(self, height_m):
        # Socket's local +X points out of the mouth; its root is at CAD x=0.
        mouth_offset = torch.zeros_like(self.socket.data.root_pos_w)
        mouth_offset[:, 0] = self.cfg.socket_mouth_x_m * self.scales
        target_w = self.socket.data.root_pos_w + quat_apply(self.socket.data.root_quat_w, mouth_offset)
        target_w[:, 2] += height_m
        # Rotating -90 degrees about Y maps bulb +X to world +Z (thread -X down).
        quat_w = torch.zeros_like(self.object.data.root_quat_w)
        quat_w[:, 0] = math.sqrt(0.5)
        quat_w[:, 2] = -math.sqrt(0.5)
        return subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, target_w, quat_w
        )

    def _set_stage_two_target(self):
        # Keep the measured axial phase: shallow entry must not command screwing.
        pos_w, quat_w = insertion_tip_target(
            self.socket, self.scales, self.cfg, phase=socket_twist_phase(self.object, self.socket))
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, pos_w, quat_w
        )
        self.pose_command_b[self.stage_two, :3] = pos_b[self.stage_two]
        self.pose_command_b[self.stage_two, 3:] = quat_b[self.stage_two]
        # Follow the measured helix with a small lead; do not command a straight
        # plunge to the final depth before the bulb has physically turned.
        goal_turns = (self.screw.turns + self.cfg.screw_goal_lead_turns).clamp(max=self.cfg.screw_target_turns)
        depth = self.screw.entry_depth + goal_turns * self.cfg.thread_pitch_m * self.scales
        phase = self.screw.entry_phase - 2 * math.pi * goal_turns
        pos_w, quat_w = insertion_tip_target(self.socket, self.scales, self.cfg, depth_m=depth, phase=phase)
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, pos_w, quat_w)
        self.pose_command_b[self.stage_three, :3] = pos_b[self.stage_three]
        self.pose_command_b[self.stage_three, 3:] = quat_b[self.stage_three]

    @property
    def tracked_position_w(self):
        """Actual reference point matching the stage's direct position command."""
        tip_w = bulb_tip_position_w(self.object, self.scales, self.cfg)
        return torch.where(self.stage_two[:, None], tip_w, self.object.data.root_pos_w)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        if not self.cfg.position_only:
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
            self.curr_visualizer.visualize(self.tracked_position_w, self.object.data.root_quat_w)
        else:
            # Color by actual stage criteria, never the inherited 5 cm shortcut.
            reached = torch.where(self.stage_two,
                self.metrics["insertion_candidate"].bool(), self.metrics["position_success"].bool())
            reached = torch.where(self.stage_three, self.metrics["success"].bool(), reached)
            indices = reached.int() + 1
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=indices)
            self.curr_visualizer.visualize(self.tracked_position_w, marker_indices=indices)

    def _resample_command(self, env_ids):
        # Both goals follow the same socket. A timer must never move it or demote.
        pos_b, quat_b = self._socket_target_pose_b(self.cfg.stage1_height_m)
        self.pose_command_b[env_ids, :3] = pos_b[env_ids]
        self.pose_command_b[env_ids, 3:] = quat_b[env_ids]
        self._set_stage_two_target()

    def advance(self):
        """Run after physics, before rewards/reset; do not teleport or end a stage."""
        self._update_metrics()
        reached = (~self.stage_two) & (self.metrics["position_error"] < self.cfg.position_tolerance_m)
        self.stage_two |= reached
        self._set_stage_two_target()
        self._update_metrics()
        # Exactly once per control step. command.compute() also measures metrics,
        # but must not count a second sample or change terminal outcome on reset.
        was_three = self.stage_three.clone()
        entered = self.metrics["insertion_candidate"].bool() & ~self.stage_three
        # stage_three latches until reset, so only the first valid entry pays.
        self.stage_two_bonus_credit[:] = entered.float()
        phase = socket_twist_phase(self.object, self.socket)
        self.screw.begin(entered, phase, self.metrics["insertion_depth_m"])
        self.stage_three |= entered
        self.stage_three_steps += was_three.long()
        self.screw.update(self.stage_three, phase, self.metrics["insertion_depth_m"],
                          self.metrics["insertion_radial_error_m"],
                          torch.deg2rad(self.metrics["insertion_tilt_deg"]),
                          screw_contact_quality(self._env, self.cfg.screw_force_scale_n), self._env.step_dt)
        self._set_stage_two_target()
        self._update_metrics()
        self.screw_hold_steps[:] = torch.where(self._screw_candidate, self.screw_hold_steps + 1, 0).clamp(
            max=self.required_hold_steps)
        self._update_metrics()
        final = self.metrics["success"].bool()
        self.final_bonus_credit[:] = (final & ~self.final_bonus_paid).float()
        self.final_bonus_paid |= final

    def _update_metrics(self):
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w,
            self.pose_command_b[:, :3], self.pose_command_b[:, 3:],
        )
        error = torch.linalg.vector_norm(self.pose_command_w[:, :3] - self.tracked_position_w, dim=-1)
        tilt = thread_down_error(self.object.data.root_quat_w)
        position_ok = error < self.cfg.position_tolerance_m
        depth, radial, insertion_tilt, speed, angular_speed = insertion_errors(
            self.object, self.socket, self.scales, self.cfg
        )
        tip_w = bulb_tip_position_w(self.object, self.scales, self.cfg)
        tip_s = quat_apply_inverse(self.socket.data.root_quat_w, tip_w - self.socket.data.root_pos_w)
        tip_radial = torch.linalg.vector_norm(tip_s[:, 1:], dim=-1)
        # Stage two checks a coarse tip-entry region. Full shaft alignment at
        # the lip and valid helix progress remain stage-three requirements.
        depth_ok = ((depth >= self.cfg.insertion_depth_range_m[0]) & (depth <= self.cfg.insertion_depth_range_m[1]))
        tip_radial_ok = tip_radial <= self.cfg.shallow_insertion_radial_tolerance_m
        insertion_position_ok = depth_ok & tip_radial_ok
        shallow_orientation_ok = insertion_tilt < math.radians(self.cfg.shallow_insertion_tilt_tolerance_deg)
        orientation_ok = insertion_tilt <= math.radians(self.cfg.insertion_tilt_tolerance_deg)
        candidate = self.stage_two & insertion_position_ok & shallow_orientation_ok
        final_depth = self.screw.entry_depth + self.cfg.screw_target_turns * self.cfg.thread_pitch_m * self.scales
        self._screw_candidate = (
            self.stage_three & self.screw.valid & (self.screw.net_turns >= self.cfg.screw_target_turns - 1e-5)
            & (self.stage_three_steps <= math.ceil(self.cfg.stage3_duration_s / self._env.step_dt))
            & ((depth - final_depth).abs() <= self.cfg.screw_final_depth_tolerance_m * self.scales)
            & (radial <= self.cfg.insertion_radial_tolerance_m) & orientation_ok
            & (speed <= self.cfg.insertion_speed_limit_m_s)
            & (angular_speed <= self.cfg.insertion_angular_speed_limit_rad_s))
        success = self._screw_candidate & (self.screw_hold_steps >= self.required_hold_steps)
        self.metrics["position_error"][:] = error
        self.metrics["orientation_error"][:] = tilt
        position_reached = torch.where(self.stage_two, insertion_position_ok, position_ok)
        position_reached = torch.where(self.stage_three,
            ((depth - final_depth).abs() <= self.cfg.screw_final_depth_tolerance_m * self.scales)
            & (radial <= self.cfg.insertion_radial_tolerance_m), position_reached)
        self.metrics["position_success"][:] = position_reached.float()
        self.metrics["orientation_success"][:] = torch.where(
            self.stage_two & ~self.stage_three, shallow_orientation_ok, orientation_ok).float()
        self.metrics["success"][:] = success.float()
        self.metrics["stage"][:] = 1.0 + self.stage_two.float() + self.stage_three.float()
        self.metrics["stage1_success"][:] = self.stage_two.float()
        self.metrics["stage2_success"][:] = self.stage_three.float()
        self.metrics["stage3_success"][:] = success.float()
        self.metrics["thread_down_error_deg"][:] = torch.rad2deg(tilt)
        self.metrics["insertion_depth_m"][:] = depth
        self.metrics["insertion_radial_error_m"][:] = radial
        self.metrics["insertion_tip_radial_error_m"][:] = tip_radial
        self.metrics["insertion_tilt_deg"][:] = torch.rad2deg(insertion_tilt)
        self.metrics["insertion_speed_m_s"][:] = speed
        self.metrics["insertion_angular_speed_rad_s"][:] = angular_speed
        self.metrics["insertion_candidate"][:] = candidate.float()
        self.metrics["insertion_depth_ok"][:] = depth_ok.float()
        self.metrics["insertion_tip_radial_ok"][:] = tip_radial_ok.float()
        self.metrics["shallow_tilt_ok"][:] = shallow_orientation_ok.float()
        self.metrics["screw_tilt_ok"][:] = orientation_ok.float()
        self.metrics["insertion_hold_s"][:] = 0.0  # Legacy metric; no stage-two hold requirement.
        self.metrics["screw_turns"][:] = self.screw.turns
        self.metrics["screw_angular_turns"][:] = self.screw.net_turns
        self.metrics["screw_best_turns"][:] = self.screw.best
        self.metrics["screw_depth_target_m"][:] = final_depth
        self.metrics["screw_helix_error_m"][:] = self.screw.helix_error
        self.metrics["screw_hold_s"][:] = self.screw_hold_steps * self._env.step_dt
        self.metrics["stage3_elapsed_s"][:] = self.stage_three_steps * self._env.step_dt
        self.success_visualizer.visualize(self.success_vis_asset.data.root_pos_w, marker_indices=success.int())


def update_grasp_curriculum(env, command_name: str = "object_pose") -> torch.Tensor:
    """Non-terminating hook: synchronize target/reward/outcome before auto reset.

    Isaac's ordinary command.compute runs *after* reward and reset. Updating only
    there would lose final-step success and reward the old target on promotion.
    """
    env.command_manager.get_term(command_name).advance()
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def grasp_screw_time_out(env, command_name: str = "object_pose") -> torch.Tensor:
    """Stages 1/2 share 8 s; stage 3 receives its own 6 s from promotion."""
    command = env.command_manager.get_term(command_name)
    return torch.where(command.stage_three,
                       command.stage_three_steps >= math.ceil(command.cfg.stage3_duration_s / env.step_dt),
                       env.episode_length_buf >= math.ceil(command.cfg.stage12_duration_s / env.step_dt))


def bulb_down_orientation_penalty(env, command_name: str = "object_pose") -> torch.Tensor:
    """Cosine tilt penalty on the bulb thread, in both stages without contact gating.

    Preserve the former palm penalty's [0, 1] shape: down=0, horizontal=0.5,
    up=1. Pair with a negative weight. Spin around the thread axis is free.
    """
    command = env.command_manager.get_term(command_name)
    tilt = thread_down_error(command.object.data.root_quat_w)
    return (0.5 * (1.0 - torch.cos(tilt))).clamp(0.0, 1.0)


def bulb_thread_down_reward(env, command_name: str = "object_pose") -> torch.Tensor:
    """Dense [0,1] tilt shaping in stage two, gated like position_tracking.

    Linear angle shaping retains a useful signal throughout 0..180 degrees.
    Contact is a reward gate, not an extra success criterion.
    """
    command = env.command_manager.get_term(command_name)
    tilt = thread_down_error(command.object.data.root_quat_w)
    quality = torch.nan_to_num(1.0 - tilt / math.pi, nan=0.0).clamp(0.0, 1.0)
    return quality * command.stage_two.float() * flexiv_hand_contacts(env, 1.0).float()


def stage_two_success_bonus(env):
    """Pay once on shallow tip-entry promotion; suppress physical failures."""
    credit = env.command_manager.get_term("object_pose").stage_two_bonus_credit
    if getattr(env, "termination_manager", None) is not None:
        credit = torch.where(env.termination_manager.terminated, 0., credit)
    # RewardManager multiplies rates by dt; weight=10 must pay exactly +10.
    return credit / env.step_dt


def curriculum_position_tracking_reward(env, std, command_name, asset_cfg, align_asset_cfg):
    """Original contact-gated kernel, measuring root/tip against its direct goal."""
    command = env.command_manager.get_term(command_name)
    goal_w, _ = combine_frame_transforms(
        command.robot.data.root_pos_w, command.robot.data.root_quat_w, command.command[:, :3])
    distance = torch.linalg.vector_norm(goal_w - command.tracked_position_w, dim=-1)
    return (1 - torch.tanh(distance / std)) * flexiv_hand_contacts(env, 1.0).float()


def curriculum_position_reward(env, command_name, asset_cfg, align_asset_cfg, pos_std, rot_std=None):
    """Original pose kernel with direct tip distance in insertion stages."""
    command = env.command_manager.get_term(command_name)
    goal_w, _ = combine_frame_transforms(
        command.robot.data.root_pos_w, command.robot.data.root_quat_w, command.command[:, :3])
    distance = torch.linalg.vector_norm(goal_w - command.tracked_position_w, dim=-1)
    value = 1 - torch.tanh(distance / pos_std)
    if not rot_std:
        return value.square()
    # Preserve the original orientation kernel, independent of the position point.
    from isaaclab.utils.math import compute_pose_error
    _, goal_quat = combine_frame_transforms(command.robot.data.root_pos_w,
        command.robot.data.root_quat_w, command.command[:, :3], command.command[:, 3:])
    _, rot_error = compute_pose_error(goal_w, goal_quat,
        command.object.data.root_pos_w, command.object.data.root_quat_w)
    return value * (1 - torch.tanh(torch.linalg.vector_norm(rot_error, dim=-1) / rot_std))


def stage_two_position_orientation_reward(
    env, command_name, asset_cfg, align_asset_cfg, pos_std,
    rot_std=None, orientation_floor: float = 0.2,
) -> torch.Tensor:
    """Scale the teacher's position reward by thread tilt only during stage two.

    Keep a positional foothold at every angle and preserve free axial spin.
    This is dense shaping, not the shallow-insertion success predicate. Stages
    one and three have no tilt multiplier. All retain the original kernels;
    insertion-stage distances use the direct tip reference.
    """
    if not math.isfinite(orientation_floor) or not 0.0 <= orientation_floor <= 1.0:
        raise ValueError("orientation_floor must be finite and in [0, 1]")
    value = curriculum_position_reward(env, command_name, asset_cfg, align_asset_cfg, pos_std, rot_std)
    command = env.command_manager.get_term(command_name)
    tilt = thread_down_error(command.object.data.root_quat_w)
    alignment = (1.0 - tilt / math.pi).clamp(0.0, 1.0)
    factor = orientation_floor + (1.0 - orientation_floor) * alignment.square()
    stage_two_only = command.stage_two & ~command.stage_three
    return value * torch.where(stage_two_only, factor, 1.0)


class BulbGraspCurriculumDifficulty(DifficultyScheduler):
    """Keep ADR's update rule but use the complete two-stage success predicate."""

    def __call__(self, env, env_ids, asset_cfg=None, object_cfg=None, pos_tol=0.05,
                 rot_tol=None, init_difficulty=0, min_difficulty=0, max_difficulty=10,
                 promotion_only=False, fixed_difficulty=None):
        if self.fixed_difficulty is not None or fixed_difficulty is not None:
            return super().__call__(env, env_ids, min_difficulty=min_difficulty,
                                    max_difficulty=max_difficulty, fixed_difficulty=fixed_difficulty)
        command = env.command_manager.get_term("object_pose")
        move_up = command.metrics["success"][env_ids].bool()
        current = self.current_adr_difficulties[env_ids]
        self.current_adr_difficulties[env_ids] = torch.where(
            move_up, current + 1, current if promotion_only else current - 1
        ).clamp(min=min_difficulty, max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac
