"""Pose measurements and episode resets for the free, collision-only bulb.

Runtime measurements never write physical state or apply forces. The socket
frame's +X axis is the screw axis, normally aligned with world +Z.
"""

import math

import torch
from isaaclab.utils.math import (
    combine_frame_transforms, quat_apply_inverse, quat_conjugate,
    quat_from_euler_xyz, quat_mul, subtract_frame_transforms,
)


class CollisionScrewState:
    """Unwrap physical rotation and conservatively detect thread clearance."""

    def __init__(self, count, device, pitch, thread_bounds, socket_mouth):
        self.pitch = pitch
        bounds = torch.as_tensor(thread_bounds, device=device, dtype=torch.float32)
        self.thread_center = bounds.mean(dim=0)
        self.thread_half_size = (bounds[1] - bounds[0]) / 2
        self.socket_mouth = socket_mouth
        self.previous_quat = torch.zeros((count, 4), device=device)
        self.previous_quat[:, 0] = 1
        self.initial_x = torch.zeros(count, device=device)
        self.initial_turns = torch.ones(count, device=device)
        self.angle = torch.zeros(count, device=device)
        self.progress = torch.zeros((count, 2), device=device)
        self.effective_turns = torch.zeros(count, device=device)
        self.clearance = torch.zeros(count, device=device)
        self.cleared = torch.zeros(count, device=device, dtype=torch.bool)

    def reset(self, ids, relative_pos, relative_quat, turns):
        self.previous_quat[ids] = relative_quat
        self.initial_x[ids] = relative_pos[:, 0]
        self.initial_turns[ids] = turns
        self.angle[ids] = 0
        self.progress[ids] = 0
        self.effective_turns[ids] = 0
        self.clearance[ids] = 0
        self.cleared[ids] = False

    def measure(self, relative_pos, relative_quat, released):
        # Quaternion differences avoid the +/-pi wrap in absolute orientation.
        # Repeated reads of the same physical pose add zero rotation.
        delta = quat_mul(relative_quat, quat_conjugate(self.previous_quat))
        twist = 2 * torch.atan2(delta[:, 1], delta[:, 0])
        twist = torch.remainder(twist + math.pi, 2 * math.pi) - math.pi
        self.angle += torch.where(released, 0.0, twist)
        self.previous_quat.copy_(relative_quat)
        rotation_turns = self.angle / (2 * math.pi)
        axial_turns = (relative_pos[:, 0] - self.initial_x) / self.pitch
        progress = torch.stack((rotation_turns, axial_turns), dim=-1) / self.initial_turns[:, None]
        progress = progress.clamp(0, 1)
        self.progress.copy_(torch.where(released[:, None], self.progress, progress))
        effective = torch.minimum(rotation_turns, axial_turns).clamp_min(0)
        effective = torch.minimum(effective, self.initial_turns)
        self.effective_turns.copy_(torch.where(released, self.effective_turns, effective))
        # Project the thread's local bounding box on the socket axis. This is
        # conservative under tilt and O(num_envs), without scanning every mesh
        # vertex every step. It affects metrics only, never physical connection.
        axis = torch.zeros_like(relative_pos)
        axis[:, 0] = 1
        direction = quat_apply_inverse(relative_quat, axis)
        lowest = relative_pos[:, 0] + (direction * self.thread_center).sum(-1)
        lowest -= (direction.abs() * self.thread_half_size).sum(-1)
        self.clearance.copy_(lowest - self.socket_mouth)
        self.cleared.copy_(self.clearance >= 0)

    def update(self, env):
        support, bulb = env.scene["support"], env.scene["object"]
        pos, quat = subtract_frame_transforms(
            support.data.root_pos_w, support.data.root_quat_w,
            bulb.data.root_pos_w, bulb.data.root_quat_w,
        )
        self.measure(pos, quat, env._bulb_released)


def reset_collision_lamp(env, env_ids, *, x_range, y_range, table_half_height,
                         support_bottom_offset, support_pose, pitch, exit_extension):
    """Seat the free bulb at reset; all subsequent motion comes from physics."""
    lower, upper = env.cfg.initial_screw_turns_range
    if not 0 < lower <= upper <= 4.0:
        raise ValueError("initial_screw_turns_range must satisfy 0 < min <= max <= 4")
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if ids.numel() == 0:
        return
    state = getattr(env, "_bulb_collision_state", None)
    metrics = getattr(env, "_bulb_turn_metrics_command", None)
    if state is not None and metrics is not None:
        metrics.record_turn_metrics(ids)
    support, bulb, table = (env.scene[name] for name in ("support", "object", "table"))
    root = support.data.default_root_state[ids].clone()
    if support_pose is not None:
        root[:, :7] = torch.as_tensor(support_pose, device=env.device, dtype=root.dtype)
        root[:, :3] += env.scene.env_origins[ids]
    else:
        offset = torch.zeros_like(root[:, :3])
        offset[:, 0].uniform_(*x_range)
        offset[:, 1].uniform_(*y_range)
        offset[:, 2] = table_half_height + support_bottom_offset
        root[:, 3:7] = torch.as_tensor(env.cfg.scene.lamp.init_state.rot, device=env.device)
        root[:, :3], root[:, 3:7] = combine_frame_transforms(
            table.data.root_pos_w[ids], table.data.root_quat_w[ids], offset, root[:, 3:7])
    support.write_root_pose_to_sim(root[:, :7], env_ids=ids)
    support.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), env_ids=ids)
    turns = torch.empty(len(ids), device=env.device).uniform_(lower, upper)
    offset = torch.zeros_like(root[:, :3])
    offset[:, 0] = exit_extension - pitch * turns
    angle = math.pi - 2 * math.pi * turns
    zero = torch.zeros_like(angle)
    relative_quat = quat_from_euler_xyz(angle, zero, zero)
    pos, quat = combine_frame_transforms(root[:, :3], root[:, 3:7], offset, relative_quat)
    bulb.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=ids)
    bulb.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), env_ids=ids)
    if state is None:
        prefix = "/World/envs/env_0/Lamp"
        male = env.sim.stage.GetPrimAtPath(prefix + "/Bulb/ExternalThreadCollider")
        female = env.sim.stage.GetPrimAtPath(prefix + "/Support/InternalThreadCollider")
        points = torch.tensor(male.GetAttribute("points").Get())
        bounds = torch.stack((points.amin(dim=0), points.amax(dim=0)))
        mouth = max(point[0] for point in female.GetAttribute("points").Get())
        state = CollisionScrewState(env.num_envs, env.device, pitch, bounds, mouth)
        env._bulb_collision_state = state
        env._bulb_released = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._bulb_success = torch.zeros_like(env._bulb_released)
    state.reset(ids, offset, relative_quat, turns)
    env._bulb_released[ids] = False
    env._bulb_success[ids] = False
