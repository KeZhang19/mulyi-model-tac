"""Independent tabletop bulb assembly state and reset helpers.

This module deliberately does not modify the legacy unscrew/release state
machine.  The assembly task starts with a free rigid bulb on the table and
only records a threaded connection after measured socket contact, alignment,
rotation, and axial progress have all occurred.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
)


PITCH_M = 0.006
THREAD_MIN_X_M = -0.03002103604376316
SOCKET_MOUTH_X_M = -0.000032485189876751974
ENTRY_ROOT_X_M = SOCKET_MOUTH_X_M - THREAD_MIN_X_M
DESKTOP_CLEARANCE_M = 0.002
DESKTOP_OFFSET_Y_M = -0.18
LIFT_HEIGHT_M = 0.06
AXIS_TOLERANCE_RAD = 0.20
RADIAL_TOLERANCE_M = 0.006
MOUTH_TOLERANCE_M = 0.006
ENGAGEMENT_TURN_MIN = 0.15
ENGAGEMENT_ADVANCE_MIN_M = 0.00025
HELICAL_TOLERANCE_M = 0.004
INSTALLATION_DEPTH_M = 0.018
HOLD_DURATION_S = 1.0

DESKTOP, GRASPED, LIFTED, ALIGNED, ENGAGED, INSTALLED = range(6)


def _ensure_ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


class AssemblyState:
    """Per-environment physical phase ledger for the full tabletop task."""

    def __init__(self, env):
        n, device = env.num_envs, env.device
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.previous_quat = torch.zeros((n, 4), device=device)
        self.previous_quat[:, 0] = 1.0
        self.relative_angle = torch.zeros(n, device=device)
        self.anchor_x = torch.zeros(n, device=device)
        self.best_advance = torch.zeros(n, device=device)
        self.helical_error = torch.full((n,), float("inf"), device=device)
        self.engagement_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self.grasped = torch.zeros(n, dtype=torch.bool, device=device)
        self.lifted = torch.zeros(n, dtype=torch.bool, device=device)
        self.aligned = torch.zeros(n, dtype=torch.bool, device=device)
        self.engaged = torch.zeros(n, dtype=torch.bool, device=device)
        self.installed = torch.zeros(n, dtype=torch.bool, device=device)
        self.success = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self.tracking_thread = torch.zeros(n, dtype=torch.bool, device=device)
        self.axis_error = torch.zeros(n, device=device)
        self.radial_error = torch.zeros(n, device=device)
        self.depth = torch.zeros(n, device=device)
        self.table_top = torch.zeros(n, device=device)

    def reset(self, env, ids):
        support, bulb, table = env.scene["support"], env.scene["object"], env.scene["table"]
        _, rel_quat = subtract_frame_transforms(
            support.data.root_pos_w[ids], support.data.root_quat_w[ids],
            bulb.data.root_pos_w[ids], bulb.data.root_quat_w[ids],
        )
        rel_pos, _ = subtract_frame_transforms(
            support.data.root_pos_w[ids], support.data.root_quat_w[ids],
            bulb.data.root_pos_w[ids], bulb.data.root_quat_w[ids],
        )
        self.phase[ids] = DESKTOP
        self.previous_quat[ids] = rel_quat
        self.relative_angle[ids] = 0.0
        self.anchor_x[ids] = ENTRY_ROOT_X_M
        self.tracking_thread[ids] = False
        self.depth[ids] = 0.0
        self.best_advance[ids] = 0.0
        self.helical_error[ids] = float("inf")
        self.engagement_contact[ids] = False
        self.grasped[ids] = False
        self.lifted[ids] = False
        self.aligned[ids] = False
        self.engaged[ids] = False
        self.installed[ids] = False
        self.success[ids] = False
        self.last_contact[ids] = False
        # The table asset root is its cuboid center.  Its top surface is one
        # half-height above that root; using subtraction placed the bulb
        # inside the table and produced hundreds of newtons of contact force
        # on the first step.
        self.table_top[ids] = table.data.root_pos_w[ids, 2] + env.cfg.scene.table.spawn.size[2] / 2

    @staticmethod
    def _contact_from_sensor(env):
        sensor = env.scene.sensors.get("bulb_support_contact")
        if sensor is None or sensor.data.force_matrix_w is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        forces = sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3)
        return torch.linalg.vector_norm(torch.nan_to_num(forces), dim=-1).amax(dim=-1) > 0.2

    def update(self, env):
        support, bulb = env.scene["support"], env.scene["object"]
        rel_pos, rel_quat = subtract_frame_transforms(
            support.data.root_pos_w, support.data.root_quat_w,
            bulb.data.root_pos_w, bulb.data.root_quat_w,
        )
        delta = quat_mul(rel_quat, quat_conjugate(self.previous_quat))
        twist = 2.0 * torch.atan2(delta[:, 1], delta[:, 0])
        twist = torch.remainder(twist + math.pi, 2 * math.pi) - math.pi
        self.previous_quat.copy_(rel_quat)

        axis = quat_apply(rel_quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(rel_pos))
        axis_error = torch.acos(axis[:, 0].clamp(-1.0, 1.0))
        radial = torch.linalg.vector_norm(rel_pos[:, 1:], dim=-1)
        self.axis_error = axis_error
        self.radial_error = radial
        self.depth = ENTRY_ROOT_X_M - rel_pos[:, 0]
        at_mouth = (rel_pos[:, 0] - ENTRY_ROOT_X_M).abs() < MOUTH_TOLERANCE_M
        contact = self._contact_from_sensor(env)
        self.last_contact = contact

        # Finger contact is an observation of a grasp, never a teleport/phase
        # write.  It is deliberately imported lazily to avoid a module cycle.
        from .env_cfg import rotate_bulb_hand_contacts
        hand_contact = rotate_bulb_hand_contacts(env, 0.5)
        self.grasped |= hand_contact
        self.lifted |= self.grasped & (
            bulb.data.root_pos_w[:, 2] > self.table_top + LIFT_HEIGHT_M
        )
        aligned_now = (radial < RADIAL_TOLERANCE_M) & (axis_error < AXIS_TOLERANCE_RAD)
        self.aligned |= self.lifted & at_mouth & aligned_now
        # Observe contact at the physical mouth before measuring the screw
        # relation. This only resets a ledger; it never moves or constrains a body.
        escaped = ~aligned_now | (self.depth < -MOUTH_TOLERANCE_M)
        self.tracking_thread &= ~escaped
        self.engaged &= ~escaped
        new_entry = self.lifted & at_mouth & aligned_now & contact & ~self.tracking_thread
        self.anchor_x = torch.where(new_entry, rel_pos[:, 0], self.anchor_x)
        self.relative_angle = torch.where(new_entry, 0.0, self.relative_angle)
        self.relative_angle += torch.where(self.tracking_thread, twist, 0.0)
        self.tracking_thread |= new_entry

        advance = self.anchor_x - rel_pos[:, 0]
        # Asset convention: positive rotation around support +X withdraws;
        # negative rotation inserts. Preserve sign so reverse rotation cannot
        # masquerade as forward progress.
        expected = -self.relative_angle * PITCH_M / (2 * math.pi)
        # Validate the *current* signed screw relationship.  Taking a
        # historical minimum would lock the error to zero on the first reset
        # sample (advance=angle=0), allowing a later non-helical motion to
        # satisfy the engagement gate forever.
        current_error = (advance - expected).abs()
        has_rotation = self.relative_angle.abs() > 1.0e-5
        self.helical_error = torch.where(
            has_rotation, current_error, torch.full_like(current_error, float("inf"))
        )
        valid_progress = advance > ENGAGEMENT_ADVANCE_MIN_M
        valid_progress &= -self.relative_angle / (2 * math.pi) > ENGAGEMENT_TURN_MIN
        valid_progress &= self.helical_error < HELICAL_TOLERANCE_M
        self.engaged |= self.tracking_thread & aligned_now & contact & valid_progress
        self.engagement_contact |= self.engaged & contact
        self.best_advance = torch.maximum(self.best_advance, torch.where(
            self.engaged & aligned_now & contact & valid_progress, self.depth.clamp_min(0.0), 0.0))
        self.installed = self.engaged & aligned_now & (self.depth >= INSTALLATION_DEPTH_M)
        self.phase = torch.where(self.installed, INSTALLED, self.phase)
        self.phase = torch.where(self.engaged & ~self.installed, ENGAGED, self.phase)
        self.phase = torch.where(self.aligned & ~self.engaged, ALIGNED, self.phase)
        self.phase = torch.where(self.lifted & ~self.aligned, LIFTED, self.phase)
        self.phase = torch.where(self.grasped & ~self.lifted, GRASPED, self.phase)
        return self


def reset_bulb_assembly_on_table(
    env, env_ids, *, x_range=(0.0, 0.0), y_range=(0.0, 0.0),
    table_half_height=0.0, support_bottom_offset=0.044732484966516495,
    support_pose=None, desktop_offset_y=DESKTOP_OFFSET_Y_M, yaw_range=(0.0, 0.0),
):
    """Reset the free bulb with world XY offsets (m) and world Z yaw (rad).

    Sample around the nominal desktop pose each episode, keeping the socket
    fixed and preserving the bulb's height and tilt before physical settling.
    """
    ids = _ensure_ids(env, env_ids)
    if ids.numel() == 0:
        return
    support, bulb, table = env.scene["support"], env.scene["object"], env.scene["table"]
    support_root = support.data.default_root_state[ids].clone()
    if support_pose is not None:
        support_root[:, :7] = torch.as_tensor(support_pose, device=env.device, dtype=support_root.dtype)
        support_root[:, :3] += env.scene.env_origins[ids]
    else:
        support_root[:, 2] = table.data.root_pos_w[ids, 2] + table_half_height + support_bottom_offset
    support.write_root_pose_to_sim(support_root[:, :7], env_ids=ids)
    support.write_root_velocity_to_sim(torch.zeros_like(support_root[:, 7:]), env_ids=ids)

    # The support frame's +X maps to world +Z in the frozen asset pose.  The
    # external-thread minimum X therefore gives a measured tabletop clearance.
    table_top = table.data.root_pos_w[ids, 2] + table_half_height
    offset = torch.zeros((len(ids), 3), device=env.device, dtype=support_root.dtype)
    if not math.isfinite(float(desktop_offset_y)):
        raise ValueError("desktop_offset_y must be finite")
    offset[:, 1] = float(desktop_offset_y)
    bulb_pos, bulb_quat = combine_frame_transforms(
        support_root[:, :3], support_root[:, 3:7], offset,
        support_root[:, 3:7].new_tensor([1.0, 0.0, 0.0, 0.0]).expand(len(ids), -1),
    )
    bulb_pos[:, 0] += torch.empty_like(bulb_pos[:, 0]).uniform_(*x_range)
    bulb_pos[:, 1] += torch.empty_like(bulb_pos[:, 1]).uniform_(*y_range)
    bulb_pos[:, 2] = table_top - THREAD_MIN_X_M + DESKTOP_CLEARANCE_M
    yaw = torch.empty_like(bulb_pos[:, 0]).uniform_(*yaw_range)
    yaw_quat = torch.zeros_like(bulb_quat)
    yaw_quat[:, 0] = torch.cos(yaw / 2)
    yaw_quat[:, 3] = torch.sin(yaw / 2)
    # Left multiplication rotates about the tabletop normal (world Z), not
    # the asset's local Z axis, which is tilted in the reference pose.
    bulb_quat = quat_mul(yaw_quat, bulb_quat)
    bulb.write_root_pose_to_sim(torch.cat((bulb_pos, bulb_quat), dim=-1), env_ids=ids)
    bulb.write_root_velocity_to_sim(torch.zeros_like(support_root[:, 7:]), env_ids=ids)

    state = getattr(env, "_bulb_assembly_state", None)
    if state is None:
        state = AssemblyState(env)
        env._bulb_assembly_state = state
    state.reset(env, ids)
    # Reset only the selected environments.  Manager-based partial resets are
    # normal during vectorized rollouts; clearing the whole vector here leaks
    # state between independent episodes.
    if not hasattr(env, "_bulb_released"):
        env._bulb_released = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_bulb_success"):
        env._bulb_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._bulb_released[ids] = False
    env._bulb_success[ids] = False
    for name in ("_assembly_hold_time", "_assembly_no_contact_time"):
        timer = getattr(env, name, None)
        if timer is None:
            timer = torch.zeros(env.num_envs, device=env.device)
            setattr(env, name, timer)
        timer[ids] = 0.0


def assembly_phase_observation(env):
    state = getattr(env, "_bulb_assembly_state", None)
    if state is None:
        return torch.zeros((env.num_envs, 2), device=env.device)
    state.update(env)
    return torch.stack((state.phase.float() / INSTALLED, state.best_advance / INSTALLATION_DEPTH_M), dim=-1).clamp(0, 1)


def assembly_screw_progress(env):
    state = getattr(env, "_bulb_assembly_state", None)
    if state is None:
        return torch.zeros((env.num_envs, 2), device=env.device)
    state.update(env)
    turns = -state.relative_angle / (2 * math.pi)
    return torch.stack((turns.clamp(0, 4), (state.best_advance / INSTALLATION_DEPTH_M).clamp(0, 1)), dim=-1)


def assembly_bulb_dropped(env, force_threshold=0.5):
    state = getattr(env, "_bulb_assembly_state", None)
    if state is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    state.update(env)
    forces = env.scene.sensors["bulb_table_contact"].data.force_matrix_w
    if forces is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    contact = torch.linalg.vector_norm(torch.nan_to_num(forces), dim=-1).reshape(env.num_envs, -1).amax(dim=-1)
    # Table contact is expected during DESKTOP.  It is a failure after lift.
    return (state.lifted & (contact > force_threshold)) | (state.lifted & (env.scene["object"].data.root_pos_w[:, 2] < state.table_top + 0.02))


def assembly_lost_grasp(env, no_contact_duration_s=0.75, far_distance=0.2):
    state = getattr(env, "_bulb_assembly_state", None)
    if state is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    state.update(env)
    from .env_cfg import rotate_bulb_finger_forces, rotate_bulb_hand_shell_distance
    active = state.grasped & ~state.installed & (env.episode_length_buf * env.step_dt > 0.5)
    contact = rotate_bulb_finger_forces(env).amax(dim=-1) > 0.1
    far = rotate_bulb_hand_shell_distance(env) > far_distance
    # Keep a tiny per-env timer; no-contact before a grasp is not a failure.
    if not hasattr(env, "_assembly_no_contact_time"):
        env._assembly_no_contact_time = torch.zeros(env.num_envs, device=env.device)
    env._assembly_no_contact_time = torch.where(active & ~contact, env._assembly_no_contact_time + env.step_dt, 0.0)
    return (env._assembly_no_contact_time >= no_contact_duration_s) | (active & far)


def assembly_success(env, hold_duration=HOLD_DURATION_S, pos_tolerance=0.004, max_linear_speed=0.05, max_angular_speed=0.5):
    state = getattr(env, "_bulb_assembly_state", None)
    if state is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    state.update(env)
    bulb = env.scene["object"]
    support = env.scene["support"]
    rel_pos, _ = subtract_frame_transforms(
        support.data.root_pos_w, support.data.root_quat_w,
        bulb.data.root_pos_w, bulb.data.root_quat_w,
    )
    current_advance = ENTRY_ROOT_X_M - rel_pos[:, 0]
    stable = bulb.data.root_lin_vel_w.norm(dim=-1) < max_linear_speed
    stable &= bulb.data.root_ang_vel_w.norm(dim=-1) < max_angular_speed
    # Installation is a live physical state, not a historical best-progress
    # latch.  A bulb that reached depth and then backed out or lost contact
    # must not earn the one-second hold after becoming stationary elsewhere.
    stable &= (current_advance - INSTALLATION_DEPTH_M).abs() < pos_tolerance
    stable &= state.last_contact & (state.helical_error < HELICAL_TOLERANCE_M)
    stable &= (state.radial_error < RADIAL_TOLERANCE_M) & (state.axis_error < AXIS_TOLERANCE_RAD)
    valid = state.installed & state.engaged & stable
    if not hasattr(env, "_assembly_hold_time"):
        env._assembly_hold_time = torch.zeros(env.num_envs, device=env.device)
    env._assembly_hold_time = torch.where(valid, env._assembly_hold_time + env.step_dt, 0.0)
    state.success = env._assembly_hold_time + 1e-6 >= hold_duration
    return state.success


class AssemblySuccessReward(ManagerTermBase):
    """Manager term wrapper kept local to the independent task."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        if hasattr(self._env, "_assembly_hold_time"):
            self._env._assembly_hold_time[ids] = 0.0

    def __call__(self, env, hold_duration=HOLD_DURATION_S):
        success = assembly_success(env, hold_duration=hold_duration)
        credit = success & ~self.paid
        self.paid |= success
        return credit.float() / env.step_dt


def assembly_phase_reward(env):
    state = getattr(env, "_bulb_assembly_state", None)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    state.update(env)
    return (state.phase.float() / INSTALLED).clamp(0, 1)
