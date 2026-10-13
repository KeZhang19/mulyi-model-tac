"""Current task state with fixed channel meanings across all reward stages."""
import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .insertion import bulb_tip_position_w, insertion_tip_target


def bulb_task_state(env, command_name: str = "object_pose") -> torch.Tensor:
    """15 channels: stage(3), lift error(3), entry error(3), axis(3), screw/time(3).

    Position errors are goal minus actual in the robot base, in metres. The
    entry goal always means shallow insertion, including during screwing.
    The inward thread axis is expressed in the socket frame (desired -X).
    Final scalars are screw-active, signed net turns / target turns, and time
    until the current timeout / 8 seconds. They use fixed, not running, scales.
    """
    c = env.command_manager.get_term(command_name)
    stage = c.stage_two.long() + c.stage_three.long()
    phase = torch.nn.functional.one_hot(stage, num_classes=3).to(c.command.dtype)

    lift_goal_b, _ = c._socket_target_pose_b(c.cfg.stage1_height_m)
    root_b = quat_apply_inverse(c.robot.data.root_quat_w,
                               c.object.data.root_pos_w - c.robot.data.root_pos_w)
    lift_error_b = lift_goal_b - root_b
    entry_goal_w, _ = insertion_tip_target(c.socket, c.scales, c.cfg)
    tip_w = bulb_tip_position_w(c.object, c.scales, c.cfg)
    entry_error_b = quat_apply_inverse(c.robot.data.root_quat_w, entry_goal_w - tip_w)

    inward = torch.zeros_like(tip_w)
    inward[:, 0] = -1.0
    axis_s = quat_apply_inverse(c.socket.data.root_quat_w,
                               quat_apply(c.object.data.root_quat_w, inward))
    active = c.screw_active
    turns = torch.where(active, c.screw.net_turns / c.cfg.screw_target_turns, 0.0)
    remaining_steps = torch.where(
        active,
        math.ceil(c.cfg.stage3_duration_s / env.step_dt) - c.stage_three_steps,
        math.ceil(c.cfg.stage12_duration_s / env.step_dt) - env.episode_length_buf,
    )
    remaining_time = remaining_steps.clamp(min=0).to(c.command.dtype) * (env.step_dt / 8.0)
    progress = torch.stack((active.to(c.command.dtype), turns, remaining_time), dim=-1)
    return torch.cat((phase, lift_error_b, entry_error_b, axis_s, progress), dim=-1)
