# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Isolated V2-test teacher port. Source and SHA256 are recorded in port_manifest.json.
from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation, RigidObject

from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import ContactSensor

from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_apply

FLEXIV_FINGERTIP_OBJECT_SENSOR_NAMES = (
    "right_thumb_DIP_Link_object_s",
    "right_index_DIP_Link_object_s",
    "right_middle_DIP_Link_object_s",
    "right_ring_DIP_Link_object_s",
    "right_little_DIP_Link_object_s",
)

def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)

def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)

def _rigid_object_linear_velocity_w(asset) -> torch.Tensor:
    """Return a rigid object's world-frame linear velocity across Isaac Lab versions."""

    linear_velocity = getattr(asset.data, "root_lin_vel_w", None)
    if linear_velocity is not None:
        return linear_velocity
    root_velocity = getattr(asset.data, "root_vel_w", None)
    if root_velocity is not None:
        return root_velocity[:, :3]
    return asset.data.root_state_w[:, 7:10]

def _rigid_object_angular_velocity_w(asset) -> torch.Tensor:
    """Return a rigid object's world-frame angular velocity across Isaac Lab versions."""

    angular_velocity = getattr(asset.data, "root_ang_vel_w", None)
    if angular_velocity is not None:
        return angular_velocity
    root_velocity = getattr(asset.data, "root_vel_w", None)
    if root_velocity is not None:
        return root_velocity[:, 3:6]
    return asset.data.root_state_w[:, 10:13]

def _flexiv_fingertip_contact_flags(
    env: ManagerBasedRLEnv,
    threshold: float,
    contact_sensor_names: tuple[str, ...],
) -> torch.Tensor:
    """Return per-fingertip object-contact flags in the requested stable order."""

    contact_flags = []
    for sensor_name in contact_sensor_names:
        if sensor_name not in env.scene.sensors:
            contact_flags.append(torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            continue
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        contact_force = contact_sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3)
        contact_force = torch.nan_to_num(contact_force, nan=0.0, posinf=0.0, neginf=0.0)
        contact_magnitude = torch.linalg.vector_norm(contact_force, dim=-1).amax(dim=1)
        contact_flags.append(contact_magnitude > threshold)
    return torch.stack(contact_flags, dim=1)

def _flexiv_carry_stability_state(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    table_asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    lift_clearance_m: float = 0.09,
    drop_clearance_m: float = 0.08,
    drop_palm_distance_m: float = 0.12,
    contact_grace_steps: int = 3,
    drop_contact_loss_steps: int = 5,
    hold_steps: int = 45,
    relative_velocity_clamp_max: float = 1000.0,
    contact_sensor_names: tuple[str, ...] = FLEXIV_FINGERTIP_OBJECT_SENSOR_NAMES,
    state_key: str = "_flexiv_v2_carry_stability_state",
) -> dict[str, torch.Tensor | int]:
    """Update and cache the V2-test carry-stability state once per environment step."""

    if len(contact_sensor_names) != 5:
        raise ValueError("carry stability requires exactly five fingertip contact sensors")
    if not math.isfinite(contact_threshold) or contact_threshold < 0.0:
        raise ValueError("contact_threshold must be finite and non-negative")
    if not math.isfinite(lift_clearance_m) or lift_clearance_m <= 0.0:
        raise ValueError("lift_clearance_m must be finite and positive")
    if not math.isfinite(drop_clearance_m) or not 0.0 < drop_clearance_m < lift_clearance_m:
        raise ValueError("drop_clearance_m must be finite, positive, and below lift_clearance_m")
    if not math.isfinite(drop_palm_distance_m) or drop_palm_distance_m <= 0.0:
        raise ValueError("drop_palm_distance_m must be finite and positive")
    if contact_grace_steps < 0 or drop_contact_loss_steps <= contact_grace_steps:
        raise ValueError("drop_contact_loss_steps must be greater than contact_grace_steps")
    if hold_steps <= 0:
        raise ValueError("hold_steps must be positive")
    if not math.isfinite(relative_velocity_clamp_max) or relative_velocity_clamp_max <= 0.0:
        raise ValueError("relative_velocity_clamp_max must be finite and positive")

    current_step = int(getattr(env, "common_step_counter", 0))
    state = getattr(env, state_key, None)
    if isinstance(state, dict) and state.get("step") == current_step:
        return state

    object_asset: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[palm_cfg.name]
    table: RigidObject = env.scene[table_asset_cfg.name]

    palm_position_w = robot.data.body_pos_w[:, palm_cfg.body_ids]
    palm_linear_velocity_w = robot.data.body_lin_vel_w[:, palm_cfg.body_ids]
    palm_angular_velocity_w = robot.data.body_ang_vel_w[:, palm_cfg.body_ids]
    if palm_position_w.ndim != 3 or palm_position_w.shape[1] != 1:
        raise ValueError("carry stability rewards require exactly one palm body")
    palm_position_w = palm_position_w[:, 0]
    palm_linear_velocity_w = palm_linear_velocity_w[:, 0]
    palm_angular_velocity_w = palm_angular_velocity_w[:, 0]

    object_position_w = object_asset.data.root_pos_w
    table_top_z_w = table.data.root_pos_w[:, 2] + 0.5 * float(table.cfg.spawn.size[2])
    finite_object_z = torch.nan_to_num(
        object_position_w[:, 2],
        nan=-1.0e6,
        posinf=1.0e6,
        neginf=-1.0e6,
    )
    finite_object_position_w = torch.nan_to_num(
        object_position_w,
        nan=1.0e6,
        posinf=1.0e6,
        neginf=-1.0e6,
    )
    finite_palm_position_w = torch.nan_to_num(
        palm_position_w,
        nan=-1.0e6,
        posinf=1.0e6,
        neginf=-1.0e6,
    )

    contact_flags = _flexiv_fingertip_contact_flags(env, contact_threshold, contact_sensor_names)
    thumb_contact = contact_flags[:, 0]
    other_contact_count = contact_flags[:, 1:].sum(dim=1)
    good_contact = thumb_contact & (other_contact_count > 0)
    above_lift_height = finite_object_z >= table_top_z_w + lift_clearance_m
    lifted_now = good_contact & above_lift_height

    new_episode = env.episode_length_buf <= 1
    zeros_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    zeros_long = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def previous_tensor(name: str, default: torch.Tensor) -> torch.Tensor:
        if not isinstance(state, dict):
            return default
        previous = state.get(name)
        if not isinstance(previous, torch.Tensor) or previous.shape != default.shape:
            return default
        return torch.where(new_episode, default, previous)

    previous_ever_lifted = previous_tensor("ever_lifted", zeros_bool)
    previous_contact_loss_steps = previous_tensor("contact_loss_steps", zeros_long)
    previous_hold_steps = previous_tensor("hold_steps", zeros_long)
    previous_dropped = previous_tensor("dropped", zeros_bool)

    ever_lifted = previous_ever_lifted | lifted_now
    contact_loss_steps = torch.where(
        ~ever_lifted | good_contact,
        zeros_long,
        previous_contact_loss_steps + 1,
    )
    effective_contact = good_contact | (ever_lifted & (contact_loss_steps <= contact_grace_steps))
    carry_active = ever_lifted & effective_contact
    accumulated_hold_steps = torch.where(carry_active, previous_hold_steps + 1, zeros_long).clamp(max=hold_steps)

    object_palm_distance = torch.linalg.vector_norm(
        finite_object_position_w - finite_palm_position_w,
        dim=1,
    )
    below_drop_height = finite_object_z < table_top_z_w + drop_clearance_m
    separated_from_palm = object_palm_distance > drop_palm_distance_m
    new_drop = (
        ever_lifted
        & (contact_loss_steps >= drop_contact_loss_steps)
        & (below_drop_height | separated_from_palm)
    )
    drop_event = new_drop & ~previous_dropped
    dropped = torch.where(lifted_now, zeros_bool, previous_dropped | new_drop)

    large_velocity = math.sqrt(relative_velocity_clamp_max)
    relative_linear_velocity = torch.nan_to_num(
        _rigid_object_linear_velocity_w(object_asset) - palm_linear_velocity_w,
        nan=large_velocity,
        posinf=large_velocity,
        neginf=-large_velocity,
    )
    relative_angular_velocity = torch.nan_to_num(
        _rigid_object_angular_velocity_w(object_asset) - palm_angular_velocity_w,
        nan=large_velocity,
        posinf=large_velocity,
        neginf=-large_velocity,
    )
    relative_velocity_gate = (ever_lifted & good_contact).float()
    relative_linear_velocity_l2 = (
        torch.sum(torch.square(relative_linear_velocity), dim=1).clamp(max=relative_velocity_clamp_max)
        * relative_velocity_gate
    )
    relative_angular_velocity_l2 = (
        torch.sum(torch.square(relative_angular_velocity), dim=1).clamp(max=relative_velocity_clamp_max)
        * relative_velocity_gate
    )

    state = {
        "step": current_step,
        "ever_lifted": ever_lifted,
        "contact_loss_steps": contact_loss_steps,
        "hold_steps": accumulated_hold_steps,
        "dropped": dropped,
        "drop_event": drop_event,
        "good_contact": good_contact,
        "contact_quality": above_lift_height.float() * thumb_contact.float() * other_contact_count.float() / 4.0,
        "hold_reward": accumulated_hold_steps.float() / float(hold_steps),
        "relative_linear_velocity_l2": relative_linear_velocity_l2,
        "relative_angular_velocity_l2": relative_angular_velocity_l2,
        "drop_after_lift": dropped.float(),
    }
    setattr(env, state_key, state)
    env._flexiv_v2_carry_ever_lifted = ever_lifted.detach()
    env._flexiv_v2_carry_good_contact = good_contact.detach()
    env._flexiv_v2_carry_hold_steps = accumulated_hold_steps.detach()
    env._flexiv_v2_carry_dropped = dropped.detach()
    env._flexiv_v2_carry_drop_event = drop_event.detach()
    env._flexiv_v2_carry_relative_linear_velocity_l2 = relative_linear_velocity_l2.detach()
    env._flexiv_v2_carry_relative_angular_velocity_l2 = relative_angular_velocity_l2.detach()
    return state

def flexiv_carry_contact_quality(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    table_asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    lift_clearance_m: float = 0.09,
) -> torch.Tensor:
    """Reward lifted thumb opposition in proportion to the other fingertips in contact."""

    return _flexiv_carry_stability_state(
        env,
        object_cfg,
        palm_cfg,
        table_asset_cfg,
        contact_threshold=contact_threshold,
        lift_clearance_m=lift_clearance_m,
    )["contact_quality"]

def flexiv_carry_hold(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    table_asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    lift_clearance_m: float = 0.09,
    contact_grace_steps: int = 3,
    hold_steps: int = 45,
) -> torch.Tensor:
    """Reward consecutive lifted grasp duration with short contact-loss hysteresis."""

    return _flexiv_carry_stability_state(
        env,
        object_cfg,
        palm_cfg,
        table_asset_cfg,
        contact_threshold=contact_threshold,
        lift_clearance_m=lift_clearance_m,
        contact_grace_steps=contact_grace_steps,
        hold_steps=hold_steps,
    )["hold_reward"]

def flexiv_carry_relative_linear_velocity_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    table_asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    lift_clearance_m: float = 0.09,
    relative_velocity_clamp_max: float = 1000.0,
) -> torch.Tensor:
    """Penalize object slip relative to the palm while carrying."""

    return _flexiv_carry_stability_state(
        env,
        object_cfg,
        palm_cfg,
        table_asset_cfg,
        contact_threshold=contact_threshold,
        lift_clearance_m=lift_clearance_m,
        relative_velocity_clamp_max=relative_velocity_clamp_max,
    )["relative_linear_velocity_l2"]

def flexiv_carry_relative_angular_velocity_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    table_asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    lift_clearance_m: float = 0.09,
    relative_velocity_clamp_max: float = 1000.0,
) -> torch.Tensor:
    """Penalize object rotation relative to the palm while carrying."""

    state = _flexiv_carry_stability_state(
        env,
        object_cfg,
        palm_cfg,
        table_asset_cfg,
        contact_threshold=contact_threshold,
        lift_clearance_m=lift_clearance_m,
        relative_velocity_clamp_max=relative_velocity_clamp_max,
    )
    return state["relative_angular_velocity_l2"]

def flexiv_drop_after_lift(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    table_asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    lift_clearance_m: float = 0.09,
    drop_clearance_m: float = 0.08,
    drop_palm_distance_m: float = 0.12,
    contact_grace_steps: int = 3,
    drop_contact_loss_steps: int = 5,
) -> torch.Tensor:
    """Return an ongoing recoverable drop penalty after a previously valid lifted grasp."""

    return _flexiv_carry_stability_state(
        env,
        object_cfg,
        palm_cfg,
        table_asset_cfg,
        contact_threshold=contact_threshold,
        lift_clearance_m=lift_clearance_m,
        drop_clearance_m=drop_clearance_m,
        drop_palm_distance_m=drop_palm_distance_m,
        contact_grace_steps=contact_grace_steps,
        drop_contact_loss_steps=drop_contact_loss_steps,
    )["drop_after_lift"]

def object_ee_distance_delta(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_offset: tuple[float, float, float] | None = None,
    body_name_to_offset: str = "base",
    clamp_max: float = 10.0,
    history_key: str = "_closest_ee_distance",
) -> torch.Tensor:
    """Reward improvement over the historical closest end-effector distance.

    Computes the distance between the object and the end-effector bodies (max over bodies),
    then rewards only when the current distance is smaller than the historical best.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()
    asset_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]

    if body_offset is not None:
        offset_tensor = torch.tensor(body_offset, device=env.device, dtype=asset_pos.dtype)
        for i, body_id in enumerate(asset_cfg.body_ids):
            body_name = asset.body_names[body_id]
            if body_name == body_name_to_offset:
                offset_world = quat_apply(asset_quat[:, i], offset_tensor.unsqueeze(0).expand(env.num_envs, -1))
                asset_pos[:, i] = asset_pos[:, i] + offset_world
                break

    object_pos = object.data.root_pos_w
    curr_dist = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values

    buf_name = history_key
    closest = getattr(env, buf_name, None)
    if closest is None or closest.shape != curr_dist.shape:
        closest = curr_dist.clone()

    # Reset per-episode buffers on first step of each episode.
    # IsaacLab increments episode_length_buf before reward computation, so the
    # first reward step is typically 1 (not 0). Use <= 1 for compatibility.
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_mask = env.episode_length_buf <= 1
        if torch.any(reset_mask):
            closest = closest.clone()
            closest[reset_mask] = curr_dist[reset_mask]

    delta = (closest - curr_dist).clamp(min=0.0, max=clamp_max)
    new_closest = torch.minimum(closest, curr_dist)
    setattr(env, buf_name, new_closest)

    # normalize by std for scale compatibility
    return delta / (std + 1e-6)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_offset: tuple[float, float, float] | None = None,
    body_name_to_offset: str = "base",
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()
    asset_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]

    if body_offset is not None:
        offset_tensor = torch.tensor(body_offset, device=env.device, dtype=asset_pos.dtype)
        for i, body_id in enumerate(asset_cfg.body_ids):
            body_name = asset.body_names[body_id]
            if body_name == body_name_to_offset:
                offset_world = quat_apply(asset_quat[:, i], offset_tensor.unsqueeze(0).expand(env.num_envs, -1))
                asset_pos[:, i] = asset_pos[:, i] + offset_world
                break

    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)

def any_finger_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
) -> torch.Tensor:
    """Reward any Revo3 fingertip contact with the object above threshold."""

    sensor_names = [
        "right_thumb_DIP_Link_object_s",
        "right_index_DIP_Link_object_s",
        "right_middle_DIP_Link_object_s",
        "right_ring_DIP_Link_object_s",
        "right_little_DIP_Link_object_s",
    ]

    contact_magnitudes = []
    for sensor_name in sensor_names:
        if sensor_name in env.scene.sensors:
            contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
            contact_force = contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
            contact_mag = torch.norm(contact_force, dim=-1)
            contact_magnitudes.append(contact_mag)
            
            # Print when force is detected (only for env 0 to avoid too many prints)
            # if contact_mag[0].item() > threshold:
            #     force_value = contact_mag[0].item()
            #     finger_name = sensor_name.replace('_object_s', '').replace('_', ' ').title()
            #     print(f"[CONTACT] {finger_name} detected force: {force_value:.3f} N (threshold: {threshold:.3f} N)")

    if not contact_magnitudes:
        return torch.zeros(env.num_envs, device=env.device)

    all_contacts = torch.stack(contact_magnitudes, dim=1)
    any_contact = (all_contacts > threshold).any(dim=1)
    return any_contact.float()

def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        # square is not necessary but this help to keep the final value between having rot_std or not roughly the same
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = torch.norm(rot_err, dim=1)
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))

def palm_down_orientation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm"]),
) -> torch.Tensor:
    """Penalize the palm's outward normal for pointing away from world-down.

    The Revo3 palm frame uses local ``+X`` as its outward tactile normal.  The
    returned magnitude is the normalized cosine error to world ``-Z``:

    ``0.5 * (1 - dot(palm_normal_w, world_down))``.

    It is zero when local ``+X`` points toward world ``-Z``, 0.5 when it is
    horizontal, and one when it points toward world ``+Z``.  Pair it with a
    negative reward weight.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    palm_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids]
    if palm_quat_w.ndim != 3 or palm_quat_w.shape[1] != 1:
        raise ValueError("palm_down_orientation_penalty requires exactly one palm body")

    palm_quat_w = palm_quat_w[:, 0]
    palm_normal_l = palm_quat_w.new_tensor((1.0, 0.0, 0.0)).expand(env.num_envs, -1)
    palm_normal_w = quat_apply(palm_quat_w, palm_normal_l)
    down_alignment = -palm_normal_w[:, 2]
    return (0.5 * (1.0 - down_alignment)).clamp(min=0.0, max=1.0)
