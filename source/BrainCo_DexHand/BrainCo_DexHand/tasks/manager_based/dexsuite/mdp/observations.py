# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import hashlib
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms
from isaaclab.utils.warp import raycast_dynamic_meshes

from .utils import sample_object_point_cloud

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


RL_FINGER_ORDER = ("middle", "index", "ring", "pinky", "thumb")
RL_PRESSURE_PAD_FINGER_STEMS = {
    "middle": "mid",
    "index": "index",
    "ring": "ring",
    "pinky": "pinky",
    "thumb": "thumb",
}
RL_PRESSURE_PAD_SEGMENTS = ("mcp", "pip")
RL_PRESSURE_PAD_LINK_ORDER = tuple(
    f"right_{RL_PRESSURE_PAD_FINGER_STEMS[finger]}{segment}_roll_touch_link"
    for finger in RL_FINGER_ORDER
    for segment in RL_PRESSURE_PAD_SEGMENTS
) + ("right_hand_rubber_link",)
RL_HYDROSHEAR_MARKER_ROWS = 16
RL_HYDROSHEAR_MARKER_COLS = 8
RL_HYDROSHEAR_RENDER_ROWS = 320
RL_HYDROSHEAR_RENDER_COLS = 240
RL_HYDROSHEAR_MARKER_MARGIN_X = 15.0
RL_HYDROSHEAR_MARKER_MARGIN_Y = 26.0 * RL_HYDROSHEAR_RENDER_ROWS / 240.0
RL_HYDROSHEAR_OBJECT_SAMPLE_COUNT = 32768
RL_HYDROSHEAR_OBJECT_SAMPLE_MODE = "poisson"
RL_HYDROSHEAR_OBJECT_SAMPLE_SEED = 17
RL_HYDROSHEAR_POISSON_RADIUS = 0.00075
RL_HYDROSHEAR_POISSON_INITIAL_COUNT = 5000
RL_HYDROSHEAR_OBJECT_SAMPLE_REFERENCE_COUNT = 4096
RL_HYDROSHEAR_OBJECT_SAMPLE_ROI_COUNT = 168
RL_FOTS_MARKER_ROWS = 16
RL_FOTS_MARKER_COLS = 8
RL_FOTS_RENDER_ROWS = 320
RL_FOTS_RENDER_COLS = 240
RL_TAXIM_RGB_RENDER_ROWS = 240
RL_TAXIM_RGB_RENDER_COLS = 320
RL_TAXIM_RGB_RENDER_CHUNK_SIZE = 128
RL_TACTILE_RESNET_OUTPUT_DIM = 128
RL_TACTILE_RESNET_INPUT_ROWS = 224
RL_TACTILE_RESNET_INPUT_COLS = 224
RL_TACTILE_RESNET_DEPTH_MAX_M = 0.002
RL_TACTILE_RESNET_CHUNK_SIZE = 128


def _tacsl_axis_vector(axis_name: str, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    sign = -1.0 if str(axis_name).startswith("-") else 1.0
    axis = str(axis_name)[-1:].lower()
    values = {
        "x": (sign, 0.0, 0.0),
        "y": (0.0, sign, 0.0),
        "z": (0.0, 0.0, sign),
    }.get(axis)
    if values is None:
        raise ValueError(f"Unsupported TacSL axis name: {axis_name!r}")
    return torch.tensor(values, device=device, dtype=dtype)


def _tacsl_angular_velocity_from_quat_delta_batched(
    prev_quat_wxyz: torch.Tensor,
    current_quat_wxyz: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Vectorized form of the integrated TacSL relative-pose angular velocity."""

    prev_quat = F.normalize(prev_quat_wxyz, dim=-1, eps=1.0e-8)
    current_quat = F.normalize(current_quat_wxyz, dim=-1, eps=1.0e-8)
    current_quat = torch.where(
        torch.sum(prev_quat * current_quat, dim=-1, keepdim=True) < 0.0,
        -current_quat,
        current_quat,
    )
    delta_quat = quat_mul(current_quat, quat_inv(prev_quat))
    delta_quat = torch.where(delta_quat[..., :1] < 0.0, -delta_quat, delta_quat)
    axis_scaled = delta_quat[..., 1:4]
    sin_half_angle = torch.linalg.norm(axis_scaled, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_angle, delta_quat[..., :1])
    angular_velocity = axis_scaled * (
        angle / (sin_half_angle.clamp_min(1.0e-8) * max(float(dt), 1.0e-8))
    )
    return torch.where(sin_half_angle > 1.0e-8, angular_velocity, torch.zeros_like(angular_velocity))


def _fots_signed_twist_angle_batched(quat_wxyz: torch.Tensor, axis_l: torch.Tensor) -> torch.Tensor:
    """Extract signed twist around each tactile ray axis without leaving Torch."""

    quat = F.normalize(quat_wxyz, dim=-1, eps=1.0e-8)
    axis = F.normalize(axis_l.to(device=quat.device, dtype=quat.dtype), dim=-1, eps=1.0e-8)
    projected = torch.sum(quat[..., 1:4] * axis, dim=-1)
    angle = 2.0 * torch.atan2(projected, quat[..., 0])
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _tacsl_force_field_from_relative_pose_batched(
    penetration_depth_m: torch.Tensor,
    closest_points_l: torch.Tensor,
    contact_pos_l: torch.Tensor,
    contact_quat_l: torch.Tensor,
    prev_contact_pos_l: torch.Tensor,
    prev_contact_quat_l: torch.Tensor,
    history_valid: torch.Tensor,
    *,
    ray_direction_l: torch.Tensor,
    row_axis_l: torch.Tensor,
    col_axis_l: torch.Tensor,
    dt: float,
    normal_contact_stiffness: float,
    tangential_stiffness: float,
    friction_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the integrated TacSL normal/shear law for a batch entirely in Torch."""

    depth = torch.nan_to_num(penetration_depth_m, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    points_l = torch.nan_to_num(closest_points_l, nan=0.0, posinf=0.0, neginf=0.0)
    valid_history = history_valid.to(device=depth.device, dtype=torch.bool)

    contact_linvel_l = (contact_pos_l - prev_contact_pos_l) / max(float(dt), 1.0e-8)
    contact_angvel_l = _tacsl_angular_velocity_from_quat_delta_batched(
        prev_contact_quat_l,
        contact_quat_l,
        dt,
    )
    contact_linvel_l = torch.where(valid_history[:, None], contact_linvel_l, torch.zeros_like(contact_linvel_l))
    contact_angvel_l = torch.where(valid_history[:, None], contact_angvel_l, torch.zeros_like(contact_angvel_l))

    point_shape = (contact_pos_l.shape[0],) + (1,) * (points_l.ndim - 2) + (3,)
    closest_relative_l = points_l - contact_pos_l.reshape(point_shape)
    contact_linvel_field = contact_linvel_l.reshape(point_shape)
    contact_angvel_field = contact_angvel_l.reshape(point_shape).expand_as(closest_relative_l)
    relative_velocity_l = -(
        contact_linvel_field + torch.linalg.cross(contact_angvel_field, closest_relative_l, dim=-1)
    )

    normal_l = F.normalize(ray_direction_l.to(device=depth.device, dtype=depth.dtype), dim=-1, eps=1.0e-8)
    normal_shape = (1,) * (relative_velocity_l.ndim - 1) + (3,)
    normal_l = normal_l.reshape(normal_shape)
    tangential_velocity_l = relative_velocity_l - normal_l * torch.sum(
        normal_l * relative_velocity_l,
        dim=-1,
        keepdim=True,
    )
    tangential_speed = torch.linalg.norm(tangential_velocity_l, dim=-1)

    contact = depth > 0.0
    normal_force = torch.where(
        contact,
        float(normal_contact_stiffness) * depth,
        torch.zeros_like(depth),
    )
    static_shear = float(tangential_stiffness) * tangential_speed
    friction_limit = float(friction_coefficient) * normal_force
    shear_magnitude = torch.minimum(static_shear, friction_limit)
    shear_force_l = -shear_magnitude.unsqueeze(-1) * tangential_velocity_l / tangential_speed.unsqueeze(-1).clamp_min(
        1.0e-9
    )
    shear_force_l = torch.where(contact.unsqueeze(-1), shear_force_l, torch.zeros_like(shear_force_l))

    row_axis = row_axis_l.to(device=depth.device, dtype=depth.dtype)
    col_axis = col_axis_l.to(device=depth.device, dtype=depth.dtype)
    shear_row = torch.sum(shear_force_l * row_axis, dim=-1)
    shear_col = torch.sum(shear_force_l * col_axis, dim=-1)
    shear_force = torch.stack((shear_row, shear_col), dim=-1)
    return normal_force, torch.nan_to_num(shear_force, nan=0.0, posinf=0.0, neginf=0.0)


def _pool_tacsl_force_field_batched(
    normal_force: torch.Tensor,
    shear_force: torch.Tensor,
    out_rows: int,
    out_cols: int,
) -> torch.Tensor:
    """GPU active-only pooling matching the integrated TacSL force-field aggregation."""

    if normal_force.ndim != 3 or shear_force.shape != (*normal_force.shape, 2):
        raise ValueError(
            "TacSL pooling expects normal=(B,H,W) and shear=(B,H,W,2), "
            f"got {tuple(normal_force.shape)} and {tuple(shear_force.shape)}."
        )
    batch, rows, cols = normal_force.shape
    out_rows = max(1, int(out_rows))
    out_cols = max(1, int(out_cols))
    if out_rows > rows or out_cols > cols:
        raise ValueError(f"TacSL output grid {out_rows}x{out_cols} exceeds input grid {rows}x{cols}.")

    device = normal_force.device
    row_edges = torch.round(torch.linspace(0, rows, out_rows + 1, device=device)).to(dtype=torch.long)
    col_edges = torch.round(torch.linspace(0, cols, out_cols + 1, device=device)).to(dtype=torch.long)
    row_bins = torch.arange(out_rows, device=device).repeat_interleave(row_edges[1:] - row_edges[:-1])
    col_bins = torch.arange(out_cols, device=device).repeat_interleave(col_edges[1:] - col_edges[:-1])
    bin_ids = (row_bins[:, None] * out_cols + col_bins[None, :]).reshape(-1)

    values = torch.cat((normal_force.unsqueeze(-1), shear_force), dim=-1)
    active = (normal_force > 0.0) | (torch.linalg.norm(shear_force, dim=-1) > 0.0)
    values = torch.where(active.unsqueeze(-1), values, torch.zeros_like(values)).reshape(batch, -1, 3)
    active_values = active.to(dtype=values.dtype).reshape(batch, -1, 1)
    scatter_ids = bin_ids.reshape(1, -1, 1).expand(batch, -1, 3)
    count_ids = bin_ids.reshape(1, -1, 1).expand(batch, -1, 1)

    pooled_sum = torch.zeros((batch, out_rows * out_cols, 3), device=device, dtype=values.dtype)
    pooled_count = torch.zeros((batch, out_rows * out_cols, 1), device=device, dtype=values.dtype)
    pooled_sum.scatter_add_(1, scatter_ids, values)
    pooled_count.scatter_add_(1, count_ids, active_values)
    pooled = pooled_sum / pooled_count.clamp_min(1.0)
    pooled = torch.where(pooled_count > 0.0, pooled, torch.zeros_like(pooled))
    return torch.nan_to_num(
        pooled.reshape(batch, out_rows, out_cols, 3),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Object position in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: object position [x, y, z] expressed in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)


def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 4)``: object quaternion ``(w, x, y, z)`` in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)


def body_state_b_with_offset(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] = (0.035, 0.0, 0.035),
    body_name_to_offset: str = "base",
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base frame with an optional offset."""

    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]

    body_pos_w_orig = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids]
    body_quat_w_orig = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids]
    body_lin_vel_w_orig = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids]
    body_ang_vel_w_orig = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids]

    offset_tensor = torch.tensor(offset, device=env.device, dtype=body_pos_w_orig.dtype)

    for i, body_id in enumerate(body_asset_cfg.body_ids):
        body_name = body_asset.body_names[body_id]
        if body_name == body_name_to_offset:
            offset_world = quat_apply(body_quat_w_orig[:, i], offset_tensor.unsqueeze(0).expand(env.num_envs, -1))
            body_pos_w_orig[:, i] = body_pos_w_orig[:, i] + offset_world
            break

    num_bodies = body_pos_w_orig.shape[1]
    body_pos_w = body_pos_w_orig.view(-1, 3)
    body_quat_w = body_quat_w_orig.view(-1, 4)
    body_lin_vel_w = body_lin_vel_w_orig.view(-1, 3)
    body_ang_vel_w = body_ang_vel_w_orig.view(-1, 3)

    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)

    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)

    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Points are pre-sampled on the object's surface in its local frame and transformed to world,
    then into the reference (e.g., robot) root frame. Optionally visualizes the points.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.
        static: If ``True``, cache world-space points on reset and reuse them (no per-step resampling).

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        # lazy initialize visualizer and point cloud
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        self.points_local = sample_object_point_cloud(
            env.num_envs, num_points, self.object.cfg.prim_path, device=env.device
        )
        self.points_w = torch.zeros_like(self.points_local)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
        visualize: bool = True,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Note:
            Points are pre-sampled at initialization using ``self.num_points``; the ``num_points`` argument is
            kept for API symmetry and does not change the sampled set at runtime.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Unused at runtime; see note above.
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If ``True``, draw markers for the points.

        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened if requested.
        """
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)

        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)
        # apply rotation + translation
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)

        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b.view(env.num_envs, -1)


def _warn_once(env: ManagerBasedRLEnv, key: str, message: str) -> None:
    attr = f"_brainco_dexsuite_warned_{key}"
    if getattr(env, attr, False):
        return
    setattr(env, attr, True)
    print(message, flush=True)


def _finite_flat(tensor: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    tensor = tensor.to(device=env.device, dtype=torch.float32)
    tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
    return tensor.reshape(env.num_envs, -1)


def _zeros(env: ManagerBasedRLEnv, dim: int) -> torch.Tensor:
    return torch.zeros((env.num_envs, max(0, int(dim))), device=env.device, dtype=torch.float32)


def _finite_tensor(tensor: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    tensor = tensor.to(device=env.device, dtype=torch.float32)
    return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))


def _resolve_env_regex_prim_path(env: ManagerBasedRLEnv, prim_path: str) -> str:
    env_ns = str(getattr(env.scene, "env_ns", "/World/envs"))
    return str(prim_path).replace("{ENV_REGEX_NS}", f"{env_ns}/env_.*")


def _hydroshear_candidate_sample_count(
    *,
    sample_count: int,
    sample_mode: str,
    poisson_initial_count: int,
) -> int:
    return int(poisson_initial_count) if str(sample_mode) == "poisson" else int(sample_count)


def _poisson_disk_downsample_points(points: np.ndarray, *, radius: float, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=-1)]
    if points.size == 0 or float(radius) <= 0.0:
        return points.astype(np.float32)

    try:
        import point_cloud_utils as pcu

        idx = pcu.downsample_point_cloud_poisson_disk(points, radius=float(radius), target_num_samples=-1)
        return points[np.asarray(idx, dtype=np.int64)].astype(np.float32)
    except Exception:
        pass

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(points.shape[0])
    ordered = points[order]
    cell_size = float(radius)
    coords = np.floor(ordered / cell_size).astype(np.int64)
    _, first = np.unique(coords, axis=0, return_index=True)
    candidates = ordered[np.sort(first)]

    selected: list[np.ndarray] = []
    grid: dict[tuple[int, int, int], list[int]] = {}
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    radius2 = float(radius) * float(radius)

    for point in candidates:
        cell = tuple(np.floor(point / cell_size).astype(np.int64).tolist())
        keep = True
        for offset in offsets:
            neighbor = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
            for selected_idx in grid.get(neighbor, []):
                delta = point - selected[selected_idx]
                if float(np.dot(delta, delta)) < radius2:
                    keep = False
                    break
            if not keep:
                break
        if keep:
            grid.setdefault(cell, []).append(len(selected))
            selected.append(point)

    if not selected:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(selected, dtype=np.float32)


def _hydroshear_object_surface_samples_l(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    *,
    sample_count: int,
    sample_mode: str,
    poisson_radius: float,
    poisson_initial_count: int,
    sample_seed: int,
) -> tuple[torch.Tensor | None, tuple]:
    prim_path = _resolve_env_regex_prim_path(env, env.scene[object_cfg.name].cfg.prim_path)
    sample_key = (
        prim_path,
        int(sample_count),
        str(sample_mode),
        float(poisson_radius),
        int(poisson_initial_count),
        int(sample_seed),
    )
    cached_key = getattr(env, "_brainco_rl_hydroshear_object_sample_key", None)
    cached_samples = getattr(env, "_brainco_rl_hydroshear_object_samples_l", None)
    if cached_key == sample_key and isinstance(cached_samples, torch.Tensor):
        return cached_samples, sample_key

    candidate_count = _hydroshear_candidate_sample_count(
        sample_count=int(sample_count),
        sample_mode=str(sample_mode),
        poisson_initial_count=int(poisson_initial_count),
    )
    if candidate_count <= 0:
        return None, sample_key

    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        np.random.seed(int(sample_seed))
        torch.manual_seed(int(sample_seed))
        samples = sample_object_point_cloud(1, candidate_count, prim_path, device="cpu")[0]
        samples_np = samples.detach().cpu().numpy().astype(np.float32)
    except Exception as exc:
        _warn_once(
            env,
            "hydroshear_object_surface_sampling_failed",
            f"[WARN] RL HydroShear object surface sampling failed for {prim_path} ({exc}); using TacMap ray-hit fallback.",
        )
        return None, sample_key
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

    if str(sample_mode) == "poisson":
        samples_np = _poisson_disk_downsample_points(samples_np, radius=float(poisson_radius), seed=int(sample_seed))

    samples_l = torch.as_tensor(samples_np, dtype=torch.float32, device=env.device).reshape(-1, 3)
    env._brainco_rl_hydroshear_object_sample_key = sample_key
    env._brainco_rl_hydroshear_object_samples_l = samples_l
    return samples_l, sample_key


def warpsdf_pressure_obs(
    env: ManagerBasedRLEnv,
    pressure_sensor_names: list[str] | tuple[str, ...],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pressure_rows: int = 8,
    pressure_cols: int = 12,
    pressure_taxel_counts: list[int] | tuple[int, ...] | None = None,
    pressure_diffusion_enabled: bool = False,
    pressure_diffusion_sigma_m: float = 0.003,
    pressure_diffusion_blend: float = 0.7,
    pressure_diffusion_radius_sigma: float = 3.0,
    pressure_diffusion_normal_power: float = 1.0,
) -> torch.Tensor:
    """Flatten pressure-pad WarpSDF maps as one fixed-size GPU RL observation.

    The expected order is middle/index/ring/pinky/thumb, each with mcp then pip, followed by the palm.
    ``pressure_taxel_counts`` supports irregular layouts; omitting it preserves the legacy shared-HxW behavior.
    When enabled, pressure diffusion uses the same physical-distance, normal-aware,
    force-conserving Gaussian rule as the pressure visualizer, but stays on the RL tensor device.
    """

    rows = max(1, int(pressure_rows))
    cols = max(1, int(pressure_cols))
    if pressure_taxel_counts is None:
        sensor_dims = [rows * cols] * len(pressure_sensor_names)
    else:
        if len(pressure_taxel_counts) != len(pressure_sensor_names):
            raise ValueError("pressure_taxel_counts must have one positive count per pressure sensor")
        sensor_dims = [int(count) for count in pressure_taxel_counts]
        if any(count <= 0 for count in sensor_dims):
            raise ValueError("pressure_taxel_counts must have one positive count per pressure sensor")
    obj: RigidObject = env.scene[object_cfg.name]
    chunks: list[torch.Tensor] = []

    for sensor_name, per_sensor_dim in zip(pressure_sensor_names, sensor_dims, strict=True):
        try:
            sensor = env.scene.sensors[str(sensor_name)]
        except KeyError:
            _warn_once(
                env,
                f"missing_pressure_{sensor_name}",
                f"[WARN] RL tactile pressure sensor {sensor_name!r} is missing; using zeros.",
            )
            chunks.append(_zeros(env, per_sensor_dim))
            continue

        if getattr(sensor, "_target_mesh_prim_path", None) is not None and hasattr(sensor, "set_target_pose"):
            sensor.set_target_pose(obj.data.root_pos_w, obj.data.root_quat_w)
        elif hasattr(sensor, "set_box_pose"):
            sensor.set_box_pose(obj.data.root_pos_w, obj.data.root_quat_w)
        if hasattr(sensor, "update"):
            sensor.update(0.0, force_recompute=True)

        data = sensor.data
        force_map = getattr(data, "pressure_force_map", None)
        if force_map is not None:
            source = force_map.detach()
            if source.ndim == 4:
                source = source[:, 0]
            elif source.ndim == 3 and source.shape[0] != env.num_envs:
                source = source[0].unsqueeze(0)
            if source.ndim >= 3:
                if pressure_taxel_counts is None:
                    source = source[: env.num_envs, :rows, :cols]
                    out = _zeros(env, per_sensor_dim).reshape(env.num_envs, rows, cols)
                    out[:, : source.shape[-2], : source.shape[-1]] = source
                else:
                    source = source[: env.num_envs].reshape(min(env.num_envs, source.shape[0]), -1)
                    out = _zeros(env, per_sensor_dim)
                    copy_count = min(per_sensor_dim, int(source.shape[-1]))
                    out[: source.shape[0], :copy_count] = source[:, :copy_count]
                if pressure_diffusion_enabled:
                    out = out.reshape(env.num_envs, -1)
                    kernel = _pressure_rl_diffusion_kernel_for_sensor(
                        env,
                        str(sensor_name),
                        sensor,
                        out,
                        sigma_m=pressure_diffusion_sigma_m,
                        radius_sigma=pressure_diffusion_radius_sigma,
                        normal_power=pressure_diffusion_normal_power,
                    )
                    out = diffuse_pressure_rl_values(
                        out,
                        kernel,
                        blend=pressure_diffusion_blend,
                    )
                chunks.append(_finite_flat(out, env))
                continue

        tactile_points = getattr(data, "tactile_points_w_per_sensor", None)
        if tactile_points is not None:
            source = tactile_points.detach()
            if source.ndim == 4:
                source = source[:, 0, :, 3]
            if source.ndim == 2:
                if source.shape[-1] == per_sensor_dim:
                    chunks.append(_finite_flat(source[: env.num_envs], env))
                    continue
                if pressure_taxel_counts is not None:
                    out = _zeros(env, per_sensor_dim)
                    copy_count = min(per_sensor_dim, int(source.shape[-1]))
                    out[: min(env.num_envs, source.shape[0]), :copy_count] = source[
                        : env.num_envs, :copy_count
                    ]
                    chunks.append(_finite_flat(out, env))
                    continue

        chunks.append(_zeros(env, per_sensor_dim))

    if not chunks:
        return _zeros(env, 0)
    return torch.cat(chunks, dim=1)


def build_pressure_rl_diffusion_kernel(
    points_l,
    normals_l,
    *,
    sigma_m: float,
    radius_sigma: float = 3.0,
    normal_power: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the visualizer-equivalent pressure diffusion kernel on a Torch device."""

    points = torch.as_tensor(points_l, device=device, dtype=dtype)
    normals = torch.as_tensor(normals_l, device=device, dtype=dtype)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3 or not bool(torch.isfinite(points).all()):
        raise ValueError(f"pressure taxel points must be finite (P, 3), got {tuple(points.shape)}")
    if normals.shape != points.shape or not bool(torch.isfinite(normals).all()):
        raise ValueError(
            f"pressure taxel normals must match finite points, got {tuple(normals.shape)}/{tuple(points.shape)}"
        )

    sigma = float(sigma_m)
    radius_scale = float(radius_sigma)
    alignment_power = float(normal_power)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("RL pressure diffusion sigma_m must be finite and positive")
    if not np.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("RL pressure diffusion radius_sigma must be finite and positive")
    if not np.isfinite(alignment_power) or alignment_power < 0.0:
        raise ValueError("RL pressure diffusion normal_power must be finite and non-negative")

    normal_norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
    if bool(torch.any(normal_norm <= 1.0e-8)):
        raise ValueError("pressure taxel normals must be non-zero")
    normals = normals / normal_norm

    # K[i, j] is the fraction of source taxel j assigned to target taxel i.
    delta = points[:, None, :] - points[None, :, :]
    normal_i = normals[:, None, :]
    normal_j = normals[None, :, :]
    tangent_i = delta - torch.sum(delta * normal_i, dim=-1, keepdim=True) * normal_i
    tangent_j = delta - torch.sum(delta * normal_j, dim=-1, keepdim=True) * normal_j
    distance_sq = 0.5 * (
        torch.sum(tangent_i * tangent_i, dim=-1) + torch.sum(tangent_j * tangent_j, dim=-1)
    )

    alignment = torch.clamp(normals @ normals.transpose(0, 1), min=0.0, max=1.0).pow(alignment_power)
    weights = torch.exp(-0.5 * distance_sq / max(sigma * sigma, 1.0e-12)) * alignment
    radius_m = radius_scale * sigma
    weights = torch.where(distance_sq <= radius_m * radius_m, weights, torch.zeros_like(weights))
    diagonal = torch.arange(points.shape[0], device=points.device)
    weights[diagonal, diagonal] = 1.0

    denominator = torch.sum(weights, dim=0, keepdim=True)
    return weights / denominator.clamp_min(1.0e-8)


def diffuse_pressure_rl_values(
    values: torch.Tensor,
    kernel: torch.Tensor,
    *,
    blend: float,
) -> torch.Tensor:
    """Apply force-conserving pressure diffusion to a batched GPU RL tensor."""

    raw = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    matrix = kernel.to(device=raw.device, dtype=raw.dtype)
    if raw.ndim != 2 or matrix.shape != (raw.shape[1], raw.shape[1]):
        raise ValueError(
            f"RL pressure diffusion kernel {tuple(matrix.shape)} does not match values {tuple(raw.shape)}"
        )
    amount = float(blend)
    if not np.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ValueError("RL pressure diffusion blend must be finite and within [0, 1]")
    if amount <= 0.0:
        return raw

    spread = raw @ matrix.transpose(0, 1)
    return ((1.0 - amount) * raw + amount * spread).clamp_min(0.0)


def _pressure_rl_diffusion_kernel_for_sensor(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    sensor,
    values: torch.Tensor,
    *,
    sigma_m: float,
    radius_sigma: float,
    normal_power: float,
) -> torch.Tensor:
    """Return one cached GPU diffusion kernel for a static pressure-pad layout."""

    cache = getattr(env, "_brainco_rl_pressure_diffusion_kernel_cache", None)
    if cache is None:
        cache = {}
        env._brainco_rl_pressure_diffusion_kernel_cache = cache
    key = (
        str(sensor_name),
        str(values.device),
        str(values.dtype),
        int(values.shape[1]),
        float(sigma_m),
        float(radius_sigma),
        float(normal_power),
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    taxel_maps = tuple(getattr(sensor, "pressure_taxel_maps", ()))
    if len(taxel_maps) != 1:
        raise RuntimeError(
            f"RL pressure diffusion requires exactly one taxel map for sensor {sensor_name!r}, "
            f"got {len(taxel_maps)}"
        )
    taxel_map = taxel_maps[0]
    kernel = build_pressure_rl_diffusion_kernel(
        taxel_map.points_l,
        taxel_map.normals_l,
        sigma_m=sigma_m,
        radius_sigma=radius_sigma,
        normal_power=normal_power,
        device=values.device,
        dtype=values.dtype,
    )
    if kernel.shape[0] != values.shape[1]:
        raise ValueError(
            f"RL pressure sensor {sensor_name!r} has {values.shape[1]} observation values "
            f"but {kernel.shape[0]} taxel geometry points"
        )
    cache[key] = kernel
    return kernel


def _tacmap_raw_image(sensor, env: ManagerBasedRLEnv, rows: int, cols: int) -> torch.Tensor | None:
    raw = getattr(sensor.data, "output", {}).get("distance_along_normal_raw")
    if raw is None:
        return None

    source = raw.detach()
    if source.ndim == 3 and source.shape[-1] == 1:
        source = source[..., 0]
    elif source.ndim == 1 and env.num_envs == 1:
        source = source.reshape(1, -1)
    if source.ndim != 2 or source.shape[-1] != rows * cols:
        return None
    return _finite_tensor(source[: env.num_envs].reshape(env.num_envs, rows, cols), env)


def _tacmap_vec_image(
    sensor,
    attr_name: str,
    env: ManagerBasedRLEnv,
    rows: int,
    cols: int,
) -> torch.Tensor | None:
    value = getattr(sensor, attr_name, None)
    if value is None:
        return None
    source = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if source.ndim != 3 or source.shape[1] != rows * cols or source.shape[-1] != 3:
        return None
    return _finite_tensor(source[: env.num_envs].reshape(env.num_envs, rows, cols, 3), env)


def _tacmap_valid_image(
    sensor,
    attr_name: str,
    env: ManagerBasedRLEnv,
    rows: int,
    cols: int,
) -> torch.Tensor | None:
    value = getattr(sensor, attr_name, None)
    if value is None:
        return None
    source = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if source.ndim == 3 and source.shape[-1] == 1:
        source = source[..., 0]
    if source.ndim != 2 or source.shape[-1] != rows * cols:
        return None
    return source[: env.num_envs].to(device=env.device, dtype=torch.bool).reshape(env.num_envs, rows, cols)


def _tacmap_mean_ray_direction(
    sensor,
    env: ManagerBasedRLEnv,
    rows: int,
    cols: int,
) -> torch.Tensor | None:
    """Return one stable world-space TacMap ray direction per environment."""

    directions = _tacmap_vec_image(sensor, "_ray_directions_w", env, rows, cols)
    if directions is None:
        return None
    norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
    valid = torch.isfinite(norms) & (norms > 1.0e-9)
    unit = torch.where(valid, directions / torch.clamp(norms, min=1.0e-9), torch.zeros_like(directions))
    count = torch.count_nonzero(valid.squeeze(-1), dim=(1, 2)).to(dtype=torch.float32)
    mean = unit.sum(dim=(1, 2)) / torch.clamp(count[:, None], min=1.0)
    mean_norm = torch.linalg.norm(mean, dim=-1, keepdim=True)
    mean_valid = (count > 0.0) & torch.isfinite(mean_norm.squeeze(-1)) & (mean_norm.squeeze(-1) > 1.0e-9)
    normalized = mean / torch.clamp(mean_norm, min=1.0e-9)
    return torch.where(mean_valid[:, None], normalized, torch.zeros_like(normalized)).to(dtype=torch.float32)


def _tacmap_surface_debug_requires_live_update(sensor) -> bool:
    """Keep native surface-ray debug drawings live when explicitly enabled."""

    cfg = getattr(sensor, "cfg", None)
    return cfg is not None and any(
        bool(getattr(cfg, name, False))
        for name in ("debug_viz_link_surfaces", "debug_viz_hits", "debug_viz_rays")
    )


def _tacmap_attached_ray_layouts_match(surface_sensor, object_sensor, expected_count: int) -> bool:
    """Check once that cached surface depths correspond one-to-one with object rays."""

    surface_cfg = getattr(surface_sensor, "cfg", None)
    object_cfg = getattr(object_sensor, "cfg", None)
    if surface_cfg is not None and object_cfg is not None:
        if str(getattr(surface_cfg, "prim_path", "")) != str(getattr(object_cfg, "prim_path", "")):
            return False
        surface_offset = getattr(surface_cfg, "offset", None)
        object_offset = getattr(object_cfg, "offset", None)
        if surface_offset is not None and object_offset is not None:
            surface_offset_signature = (
                tuple(getattr(surface_offset, "pos", ())),
                tuple(getattr(surface_offset, "rot", ())),
                str(getattr(surface_offset, "convention", "")),
            )
            object_offset_signature = (
                tuple(getattr(object_offset, "pos", ())),
                tuple(getattr(object_offset, "rot", ())),
                str(getattr(object_offset, "convention", "")),
            )
            if surface_offset_signature != object_offset_signature:
                return False

    surface_starts = getattr(surface_sensor, "ray_starts_att", None)
    surface_directions = getattr(surface_sensor, "ray_directions_att", None)
    object_starts = getattr(object_sensor, "ray_starts_att", None)
    object_directions = getattr(object_sensor, "ray_directions_att", None)
    tensors = (surface_starts, surface_directions, object_starts, object_directions)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    assert isinstance(surface_starts, torch.Tensor)
    assert isinstance(surface_directions, torch.Tensor)
    assert isinstance(object_starts, torch.Tensor)
    assert isinstance(object_directions, torch.Tensor)
    expected_shape = (expected_count, 3)
    if any(value.ndim != 3 or tuple(value.shape[1:]) != expected_shape for value in tensors):
        return False
    if surface_starts.shape[0] < 1 or object_starts.shape[0] < 1:
        return False
    return bool(
        torch.allclose(surface_starts[0], object_starts[0], rtol=0.0, atol=1.0e-7)
        and torch.allclose(surface_directions[0], object_directions[0], rtol=0.0, atol=1.0e-7)
    )


def _tacmap_fixed_surface_reference(
    env: ManagerBasedRLEnv,
    *,
    surface_name: str,
    object_name: str,
    surface_sensor,
    object_sensor,
    rows: int,
    cols: int,
    require_geometry: bool,
) -> dict | None:
    """Cache a rigid finger-surface ray result once in the sensor's attached frame."""

    if _tacmap_surface_debug_requires_live_update(surface_sensor):
        return None

    expected_count = int(rows) * int(cols)
    cache = getattr(env, "_brainco_rl_tacmap_fixed_surface_cache", None)
    if cache is None:
        cache = {}
        env._brainco_rl_tacmap_fixed_surface_cache = cache
    key = (
        str(surface_name),
        str(object_name),
        id(surface_sensor),
        id(object_sensor),
        int(rows),
        int(cols),
        str(env.device),
    )
    cached = cache.get(key)
    if cached is not None and (not require_geometry or bool(cached.get("has_geometry", False))):
        return cached

    if not _tacmap_attached_ray_layouts_match(surface_sensor, object_sensor, expected_count):
        _warn_once(
            env,
            f"tacmap_surface_cache_layout_mismatch_{surface_name}_{object_name}",
            f"[WARN] TacMap surface cache disabled for object={object_name!r}, surface={surface_name!r}: "
            "the attached ray layouts do not match.",
        )
        return None

    if hasattr(surface_sensor, "update"):
        surface_sensor.update(0.0, force_recompute=True)
    surface_dist = _tacmap_raw_image(surface_sensor, env, rows, cols)
    if surface_dist is None or surface_dist.shape[0] < 1:
        return None
    surface_valid = _tacmap_valid_image(surface_sensor, "ray_hit_valid", env, rows, cols)
    if surface_valid is None:
        surface_valid = _tacmap_valid_image(surface_sensor, "second_ray_hit_valid", env, rows, cols)
    if surface_valid is None:
        surface_valid = surface_dist > 0.0

    reference = {
        "surface_dist_m": surface_dist[:1].detach().clone(),
        "surface_valid": surface_valid[:1].detach().clone(),
        "has_geometry": False,
    }
    if require_geometry:
        starts_l = getattr(surface_sensor, "ray_starts_att", None)
        directions_l = getattr(surface_sensor, "ray_directions_att", None)
        surface_normals_w = _tacmap_vec_image(surface_sensor, "ray_normals_w", env, rows, cols)
        if surface_normals_w is None:
            surface_normals_w = _tacmap_vec_image(surface_sensor, "second_ray_normals_w", env, rows, cols)
        surface_state = getattr(surface_sensor, "_data", None)
        surface_quat_w = getattr(surface_state, "quat_w", None)
        if (
            isinstance(starts_l, torch.Tensor)
            and isinstance(directions_l, torch.Tensor)
            and isinstance(surface_normals_w, torch.Tensor)
            and isinstance(surface_quat_w, torch.Tensor)
            and surface_quat_w.ndim == 2
            and surface_quat_w.shape[0] >= 1
            and surface_quat_w.shape[-1] == 4
        ):
            starts_l_grid = starts_l[:1].detach().reshape(1, rows, cols, 3)
            directions_l_grid = F.normalize(
                directions_l[:1].detach().reshape(1, rows, cols, 3), dim=-1, eps=1.0e-8
            )
            points_l_grid = starts_l_grid + directions_l_grid * reference["surface_dist_m"].unsqueeze(-1)
            quat_grid = surface_quat_w[:1, None, None, :].expand(1, rows, cols, 4)
            normals_l_grid = quat_apply_inverse(
                quat_grid.reshape(-1, 4),
                surface_normals_w[:1].reshape(-1, 3),
            ).reshape(1, rows, cols, 3)
            normals_l_grid = F.normalize(normals_l_grid, dim=-1, eps=1.0e-8)
            valid_grid = reference["surface_valid"].unsqueeze(-1)
            points_l_grid = torch.where(valid_grid, points_l_grid, torch.zeros_like(points_l_grid))
            normals_l_grid = torch.where(valid_grid, normals_l_grid, torch.zeros_like(normals_l_grid))
            direction_norms = torch.linalg.norm(directions_l_grid, dim=-1, keepdim=True)
            direction_valid = torch.isfinite(direction_norms) & (direction_norms > 1.0e-9)
            direction_sum = torch.where(
                direction_valid,
                directions_l_grid / torch.clamp(direction_norms, min=1.0e-9),
                torch.zeros_like(directions_l_grid),
            ).sum(dim=(1, 2))
            mean_direction_l = F.normalize(direction_sum, dim=-1, eps=1.0e-8)
            reference.update(
                {
                    "surface_points_l": points_l_grid.detach().clone(),
                    "surface_normals_l": normals_l_grid.detach().clone(),
                    "mean_ray_direction_l": mean_direction_l.detach().clone(),
                    "has_geometry": True,
                }
            )

    if require_geometry and not bool(reference["has_geometry"]):
        _warn_once(
            env,
            f"tacmap_surface_cache_geometry_missing_{surface_name}_{object_name}",
            f"[WARN] TacMap surface cache disabled for object={object_name!r}, surface={surface_name!r}: "
            "local surface geometry could not be cached.",
        )
        return None

    typed_reference = {str(name): value for name, value in reference.items()}
    cache[key] = typed_reference
    return typed_reference


def _tacmap_cached_surface_world_fields(
    reference: dict,
    object_sensor,
    env: ManagerBasedRLEnv,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Move cached local surface geometry with the current attached-finger pose on GPU."""

    if not bool(reference.get("has_geometry", False)):
        return None
    sensor_state = getattr(object_sensor, "_data", None)
    sensor_pos_w = getattr(sensor_state, "pos_w", None)
    sensor_quat_w = getattr(sensor_state, "quat_w", None)
    if (
        not isinstance(sensor_pos_w, torch.Tensor)
        or not isinstance(sensor_quat_w, torch.Tensor)
        or sensor_pos_w.shape[0] < env.num_envs
        or sensor_quat_w.shape[0] < env.num_envs
    ):
        return None

    points_l = reference["surface_points_l"].expand(env.num_envs, -1, -1, -1)
    normals_l = reference["surface_normals_l"].expand(env.num_envs, -1, -1, -1)
    quat_grid = sensor_quat_w[: env.num_envs, None, None, :].expand(env.num_envs, rows, cols, 4)
    points_w = quat_apply(quat_grid.reshape(-1, 4), points_l.reshape(-1, 3)).reshape(
        env.num_envs, rows, cols, 3
    )
    points_w = points_w + sensor_pos_w[: env.num_envs, None, None, :]
    normals_w = quat_apply(quat_grid.reshape(-1, 4), normals_l.reshape(-1, 3)).reshape(
        env.num_envs, rows, cols, 3
    )
    normals_w = F.normalize(normals_w, dim=-1, eps=1.0e-8)
    valid = reference["surface_valid"].expand(env.num_envs, -1, -1).unsqueeze(-1)
    points_w = torch.where(valid, points_w, torch.zeros_like(points_w))
    normals_w = torch.where(valid, normals_w, torch.zeros_like(normals_w))
    mean_direction_l = reference["mean_ray_direction_l"].expand(env.num_envs, -1)
    mean_direction_w = quat_apply(sensor_quat_w[: env.num_envs], mean_direction_l)
    mean_direction_w = F.normalize(mean_direction_w, dim=-1, eps=1.0e-8)
    return points_w, normals_w, mean_direction_w


def _tacmap_static_surface_reference(
    env: ManagerBasedRLEnv,
    reference_npy: str,
    *,
    sensor_count: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Load fixed TacMap surface depths once, avoiding a runtime surface RayCaster."""

    path = Path(reference_npy).expanduser().resolve()
    cache = getattr(env, "_brainco_rl_tacmap_static_surface_references", None)
    if cache is None:
        cache = {}
        env._brainco_rl_tacmap_static_surface_references = cache
    key = (str(path), int(sensor_count), int(rows), int(cols), str(env.device))
    cached = cache.get(key)
    if isinstance(cached, torch.Tensor):
        return cached

    if not path.is_file():
        raise FileNotFoundError(f"TacMap static surface reference is missing: {path}")
    reference_np = np.load(path, allow_pickle=False)
    expected_shape = (int(sensor_count), int(rows), int(cols))
    if tuple(reference_np.shape) != expected_shape:
        raise ValueError(
            f"TacMap static surface reference must have shape {expected_shape}, "
            f"got {tuple(reference_np.shape)} from {path}"
        )
    reference = torch.as_tensor(
        np.array(reference_np, dtype=np.float32, copy=True),
        device=env.device,
        dtype=torch.float32,
    )
    reference = torch.where(
        torch.isfinite(reference) & (reference > 0.0),
        reference,
        torch.zeros_like(reference),
    )
    cache[key] = reference
    return reference


def _clear_tacmap_aux_cache(env: ManagerBasedRLEnv) -> None:
    for name in (
        "_rl_tacmap_surface_points_w",
        "_rl_tacmap_surface_normals_w",
        "_rl_tacmap_ray_directions_w",
        "_rl_tacmap_surface_valid",
        "_rl_tacmap_object_points_w",
        "_rl_tacmap_object_valid",
        "_rl_tacmap_surface_raw_m",
    ):
        setattr(env, name, None)


def _ensure_integrate_import_path() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "integrate" / "curved_hydroshear_adapter.py").is_file():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


def _tacmap_penetration_from_distances(
    surface_dist: torch.Tensor,
    object_dist: torch.Tensor,
    hit_valid: torch.Tensor,
    *,
    contact_shell_m: float = 0.0,
) -> torch.Tensor:
    """Return TacMap depth; optional shell handles non-penetrating contact."""

    gap = object_dist - surface_dist
    shell = max(0.0, float(contact_shell_m))
    if shell <= 0.0:
        return torch.where(hit_valid & (gap <= 0.0), -gap, torch.zeros_like(surface_dist))

    shell_valid = hit_valid & (gap <= shell)
    depth = torch.where(gap <= 0.0, -gap, shell - gap)
    return torch.where(shell_valid, torch.clamp_min(depth, 0.0), torch.zeros_like(surface_dist))


def _load_camera_ray_rectangles(
    marker_layout_path: str | Path,
) -> tuple[dict[str, np.ndarray], tuple[int, int], Path]:
    """Load the shared per-finger white camera-XY ray rectangles."""

    rectangle_path = Path(marker_layout_path).expanduser().resolve().with_name(
        "camera_ray_rectangles_320x240.json"
    )
    if not rectangle_path.is_file():
        raise FileNotFoundError(f"Adaptive TacMap camera-ray rectangles are missing: {rectangle_path}")
    payload = json.loads(rectangle_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported camera-ray rectangle schema in {rectangle_path}")
    if str(payload.get("coordinate_frame")) != "per_finger_camera_link":
        raise ValueError("Camera-ray rectangles must use per-finger camera-link coordinates")
    if str(payload.get("units")) != "m":
        raise ValueError("Camera-ray rectangles must use metres")
    resolution = payload.get("resolution", {})
    width_px = int(resolution.get("width", 0))
    height_px = int(resolution.get("height", 0))
    if width_px <= 0 or height_px <= 0:
        raise ValueError("Camera-ray rectangle resolution must be positive")
    ray_direction = np.asarray(payload.get("ray_direction_camera"), dtype=np.float64).reshape(-1)
    if ray_direction.shape != (3,) or not np.allclose(ray_direction, (0.0, 0.0, 1.0), atol=1.0e-12):
        raise ValueError("Camera-ray rectangle direction must be camera +Z")

    raw_rectangles = payload.get("rectangles", {})
    rectangles: dict[str, np.ndarray] = {}
    expected_aspect = float(width_px) / float(height_px)
    for finger in RL_FINGER_ORDER:
        entry = raw_rectangles.get(finger)
        if not isinstance(entry, dict):
            raise ValueError(f"Camera-ray rectangle is missing finger {finger!r}")
        corners = np.asarray(entry.get("corners_camera_m"), dtype=np.float64)
        if corners.shape != (4, 3) or not np.isfinite(corners).all():
            raise ValueError(f"Camera-ray rectangle for {finger!r} must contain four finite 3D corners")
        if not np.allclose(corners[:, 2], 0.0, atol=1.0e-9):
            raise ValueError(f"Camera-ray rectangle for {finger!r} must lie on camera Z=0")
        extent = np.ptp(corners[:, :2], axis=0)
        if np.any(extent <= 1.0e-12):
            raise ValueError(f"Camera-ray rectangle for {finger!r} is degenerate")
        if not np.isclose(float(extent[0] / extent[1]), expected_aspect, rtol=0.0, atol=1.0e-7):
            raise ValueError(
                f"Camera-ray rectangle for {finger!r} does not match {width_px}:{height_px}"
            )
        rectangles[finger] = corners.astype(np.float32)
    return rectangles, (height_px, width_px), rectangle_path


def _camera_plane_ray_grid_from_rectangle(
    rectangle_camera: np.ndarray,
    *,
    camera_origin_link: np.ndarray,
    camera_rotation_link: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a rows-by-cols grid on a fixed white camera-XY rectangle."""

    rectangle = np.asarray(rectangle_camera, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(camera_origin_link, dtype=np.float64).reshape(3)
    rotation = np.asarray(camera_rotation_link, dtype=np.float64).reshape(3, 3)
    rows = int(rows)
    cols = int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("Adaptive TacMap reference dimensions must be positive")
    if rectangle.shape != (4, 3) or not np.isfinite(rectangle).all():
        raise ValueError("Adaptive TacMap white rectangle must contain four finite camera points")
    if not np.allclose(rectangle[:, 2], rectangle[0, 2], atol=1.0e-9):
        raise ValueError("Adaptive TacMap white rectangle must be parallel to camera XY")

    x_min, y_min = np.min(rectangle[:, :2], axis=0)
    x_max, y_max = np.max(rectangle[:, :2], axis=0)
    x = x_min + (np.arange(cols, dtype=np.float64) + 0.5) / float(cols) * (
        x_max - x_min
    )
    y = y_min + (np.arange(rows, dtype=np.float64) + 0.5) / float(rows) * (
        y_max - y_min
    )
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    starts_camera = np.stack(
        (
            grid_x,
            grid_y,
            np.full((rows, cols), float(rectangle[0, 2]), dtype=np.float64),
        ),
        axis=-1,
    )
    starts_link = starts_camera @ rotation.T + origin.reshape(1, 1, 3)
    direction_link = rotation[:, 2]
    direction_norm = float(np.linalg.norm(direction_link))
    if direction_norm <= 1.0e-9:
        raise ValueError("Adaptive TacMap camera optical axis is degenerate")
    direction_link = direction_link / direction_norm
    directions_link = np.broadcast_to(direction_link, starts_link.shape).copy()
    return (
        starts_link.astype(np.float32),
        directions_link.astype(np.float32),
        1,
    )


def _local_tacmap_rays_to_world(
    sensor,
    ray_starts_l: torch.Tensor,
    ray_directions_l: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform a batched local ray layout with the sensor's current pose."""

    batch_size = int(ray_starts_l.shape[0])
    ray_count = int(ray_starts_l.shape[1])
    sensor_pos_w = sensor.data.pos_w[:batch_size].to(device=ray_starts_l.device, dtype=torch.float32)
    sensor_quat_w = sensor.data.quat_w[:batch_size].to(device=ray_starts_l.device, dtype=torch.float32)
    starts_w = quat_apply(sensor_quat_w.repeat(1, ray_count), ray_starts_l) + sensor_pos_w.unsqueeze(1)
    directions_w = quat_apply(sensor_quat_w.repeat(1, ray_count), ray_directions_l)
    directions_w = F.normalize(directions_w, dim=-1, eps=1.0e-8)
    return starts_w.contiguous(), directions_w.contiguous()


def _raycast_local_tacmap_distances(
    sensor,
    ray_starts_l: torch.Tensor,
    ray_directions_l: torch.Tensor,
    *,
    max_distance_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast one fixed-size local layout for every environment on the GPU."""

    starts_w, directions_w = _local_tacmap_rays_to_world(sensor, ray_starts_l, ray_directions_l)
    batch_size = int(starts_w.shape[0])
    hits_w, distance, _, _, _ = raycast_dynamic_meshes(
        starts_w,
        directions_w,
        mesh_ids_wp=sensor._mesh_ids_wp,
        max_dist=float(max_distance_m),
        mesh_positions_w=sensor._mesh_positions_w[:batch_size],
        mesh_orientations_w=sensor._mesh_orientations_w[:batch_size],
        return_distance=True,
        return_normal=False,
        return_mesh_id=False,
    )
    if distance is None:
        raise RuntimeError("Adaptive TacMap object raycast did not return distances")
    valid = torch.isfinite(distance) & (distance > 0.0) & (distance <= float(max_distance_m))
    clean_distance = torch.where(valid, distance, torch.zeros_like(distance))
    return clean_distance, valid, starts_w, hits_w


def _initialize_local_tacmap_reference(
    env: ManagerBasedRLEnv,
    *,
    tacmap_surface_sensor_names: tuple[str, ...],
    finger_link_names: tuple[str, ...],
    marker_layout_path: str | Path,
    coarse_rows: int,
    coarse_cols: int,
    reference_rows: int,
    reference_cols: int,
    max_distance_m: float,
) -> dict:
    """Scan each calibrated 240x320 white-rectangle rubber grid twice and cache it once."""

    layout_path = Path(marker_layout_path).expanduser().resolve()
    rectangle_path = layout_path.with_name("camera_ray_rectangles_320x240.json")
    key = (
        tuple(tacmap_surface_sensor_names),
        tuple(finger_link_names),
        str(layout_path),
        str(rectangle_path),
        int(coarse_rows),
        int(coarse_cols),
        int(reference_rows),
        int(reference_cols),
        float(max_distance_m),
        str(env.device),
    )
    cached = getattr(env, "_rl_local_tacmap_reference_state", None)
    if isinstance(cached, dict) and cached.get("key") == key:
        return cached
    if not layout_path.is_file():
        raise FileNotFoundError(f"Adaptive TacMap marker/camera calibration is missing: {layout_path}")
    camera_rectangles, rectangle_shape, rectangle_path = _load_camera_ray_rectangles(layout_path)
    if (int(reference_rows), int(reference_cols)) != rectangle_shape:
        raise ValueError(
            "Adaptive TacMap reference shape must match the calibrated white rectangle: "
            f"requested={(int(reference_rows), int(reference_cols))}, calibrated={rectangle_shape}"
        )

    sensor_count = min(len(tacmap_surface_sensor_names), len(RL_FINGER_ORDER))
    if sensor_count != len(RL_FINGER_ORDER):
        raise RuntimeError(
            f"Adaptive TacMap requires {len(RL_FINGER_ORDER)} surface sensors, got {sensor_count}"
        )
    if len(finger_link_names) < sensor_count:
        raise RuntimeError(
            f"Adaptive TacMap requires {sensor_count} calibrated finger link names, got {len(finger_link_names)}"
        )

    reference_starts_l: list[torch.Tensor] = []
    reference_directions_l: list[torch.Tensor] = []
    reference_depth_m: list[torch.Tensor] = []
    reference_valid: list[torch.Tensor] = []
    reference_xy_camera_m: list[torch.Tensor] = []
    coarse_xy_camera_m: list[torch.Tensor] = []
    full_camera_bounds_m: list[torch.Tensor] = []
    storage_row_axis_camera: list[int] = []

    with np.load(layout_path, allow_pickle=False) as layout:
        for finger_index, (finger, surface_name) in enumerate(
            zip(RL_FINGER_ORDER, tacmap_surface_sensor_names[:sensor_count], strict=True)
        ):
            try:
                surface_sensor = env.scene.sensors[str(surface_name)]
            except KeyError as exc:
                raise RuntimeError(f"Adaptive TacMap surface sensor is missing: {surface_name!r}") from exc
            camera_origin_l = np.asarray(layout[f"{finger}_camera_origin_link_m"], dtype=np.float32).reshape(3)
            camera_rotation_l = np.asarray(layout[f"{finger}_camera_rotation_link"], dtype=np.float32).reshape(3, 3)
            starts_np, directions_np, row_axis_camera = _camera_plane_ray_grid_from_rectangle(
                camera_rectangles[finger],
                camera_origin_link=camera_origin_l,
                camera_rotation_link=camera_rotation_l,
                rows=reference_rows,
                cols=reference_cols,
            )
            starts_l = torch.as_tensor(starts_np, device=env.device, dtype=torch.float32)
            directions_l = torch.as_tensor(directions_np, device=env.device, dtype=torch.float32)
            flat_starts_l = starts_l.reshape(1, reference_rows * reference_cols, 3)
            flat_directions_l = directions_l.reshape(1, reference_rows * reference_cols, 3)
            starts_w, directions_w = _local_tacmap_rays_to_world(
                surface_sensor,
                flat_starts_l,
                flat_directions_l,
            )
            _, first_depth, _, _, _ = raycast_dynamic_meshes(
                starts_w,
                directions_w,
                mesh_ids_wp=surface_sensor._mesh_ids_wp,
                max_dist=float(max_distance_m),
                mesh_positions_w=surface_sensor._mesh_positions_w[:1],
                mesh_orientations_w=surface_sensor._mesh_orientations_w[:1],
                return_distance=True,
                return_normal=False,
                return_mesh_id=False,
            )
            if first_depth is None:
                raise RuntimeError(f"Adaptive TacMap first surface scan failed for finger {finger!r}")
            first_valid = (
                torch.isfinite(first_depth)
                & (first_depth > 0.0)
                & (first_depth <= float(max_distance_m))
            )
            first_clean = torch.where(first_valid, first_depth, torch.zeros_like(first_depth))
            epsilon_m = max(0.0, float(getattr(surface_sensor.cfg, "second_hit_epsilon", 1.0e-5)))
            second_starts_w = starts_w + directions_w * (first_clean + epsilon_m).unsqueeze(-1)
            _, second_depth, _, _, _ = raycast_dynamic_meshes(
                second_starts_w,
                directions_w,
                mesh_ids_wp=surface_sensor._mesh_ids_wp,
                max_dist=float(max_distance_m),
                mesh_positions_w=surface_sensor._mesh_positions_w[:1],
                mesh_orientations_w=surface_sensor._mesh_orientations_w[:1],
                return_distance=True,
                return_normal=False,
                return_mesh_id=False,
            )
            if second_depth is None:
                raise RuntimeError(f"Adaptive TacMap second surface scan failed for finger {finger!r}")
            second_total = first_clean + epsilon_m + torch.nan_to_num(
                second_depth,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            second_valid = (
                first_valid
                & torch.isfinite(second_depth)
                & (second_depth > 0.0)
                & (second_total <= float(max_distance_m))
            )
            selected_valid = first_valid | second_valid
            selected_depth = torch.where(second_valid, second_total, first_clean)[0].reshape(
                reference_rows,
                reference_cols,
            )
            selected_valid = selected_valid[0].reshape(reference_rows, reference_cols)

            origin_l = torch.as_tensor(camera_origin_l, device=env.device, dtype=torch.float32)
            rotation_l = torch.as_tensor(camera_rotation_l, device=env.device, dtype=torch.float32)
            reference_camera = torch.matmul(starts_l - origin_l, rotation_l)
            coarse_starts_l = surface_sensor.ray_starts_att[0].detach().to(
                device=env.device,
                dtype=torch.float32,
            ).reshape(coarse_rows, coarse_cols, 3)
            coarse_camera = torch.matmul(coarse_starts_l - origin_l, rotation_l)
            reference_xy = reference_camera[..., :2]
            full_bounds = torch.stack(
                (
                    torch.amin(reference_xy[..., 0]),
                    torch.amax(reference_xy[..., 0]),
                    torch.amin(reference_xy[..., 1]),
                    torch.amax(reference_xy[..., 1]),
                )
            )

            reference_starts_l.append(starts_l)
            reference_directions_l.append(directions_l)
            reference_depth_m.append(selected_depth)
            reference_valid.append(selected_valid)
            reference_xy_camera_m.append(reference_xy)
            coarse_xy_camera_m.append(coarse_camera[..., :2])
            full_camera_bounds_m.append(full_bounds)
            storage_row_axis_camera.append(int(row_axis_camera))

    state = {
        "key": key,
        "reference_rows": int(reference_rows),
        "reference_cols": int(reference_cols),
        "camera_rectangle_path": str(rectangle_path),
        "coarse_rows": int(coarse_rows),
        "coarse_cols": int(coarse_cols),
        "reference_starts_l": torch.stack(reference_starts_l, dim=0),
        "reference_directions_l": torch.stack(reference_directions_l, dim=0),
        "reference_depth_m": torch.stack(reference_depth_m, dim=0),
        "reference_valid": torch.stack(reference_valid, dim=0),
        "reference_xy_camera_m": torch.stack(reference_xy_camera_m, dim=0),
        "coarse_xy_camera_m": torch.stack(coarse_xy_camera_m, dim=0),
        "full_camera_bounds_m": torch.stack(full_camera_bounds_m, dim=0),
        "storage_row_axis_camera": tuple(storage_row_axis_camera),
        "link_names": tuple(str(name) for name in finger_link_names[:sensor_count]),
    }
    env._rl_local_tacmap_reference_state = state
    return state


def _local_tacmap_contact_roi_batched(
    coarse_depth_m: torch.Tensor,
    coarse_xy_camera_m: torch.Tensor,
    full_camera_bounds_m: torch.Tensor,
    *,
    contact_threshold_m: float,
    roi_margin_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return metric and normalized ROIs for a whole environment batch."""

    depth = torch.nan_to_num(coarse_depth_m, nan=0.0, posinf=0.0, neginf=0.0).reshape(
        coarse_depth_m.shape[0],
        -1,
    )
    xy = coarse_xy_camera_m.to(device=depth.device, dtype=torch.float32).reshape(-1, 2)
    contact = depth > max(0.0, float(contact_threshold_m))
    active = torch.any(contact, dim=1)
    inf = torch.full_like(depth, torch.inf)
    neg_inf = torch.full_like(depth, -torch.inf)
    x = xy[:, 0].reshape(1, -1)
    y = xy[:, 1].reshape(1, -1)
    x_min = torch.amin(torch.where(contact, x, inf), dim=1)
    x_max = torch.amax(torch.where(contact, x, neg_inf), dim=1)
    y_min = torch.amin(torch.where(contact, y, inf), dim=1)
    y_max = torch.amax(torch.where(contact, y, neg_inf), dim=1)

    full = full_camera_bounds_m.to(device=depth.device, dtype=torch.float32).reshape(4)
    margin = max(0.0, float(roi_margin_m))
    expanded = torch.stack(
        (
            torch.clamp(x_min - margin, min=full[0], max=full[1]),
            torch.clamp(x_max + margin, min=full[0], max=full[1]),
            torch.clamp(y_min - margin, min=full[2], max=full[3]),
            torch.clamp(y_max + margin, min=full[2], max=full[3]),
        ),
        dim=1,
    )
    bounds = torch.where(active[:, None], expanded, full.reshape(1, 4).expand_as(expanded))
    x_extent = torch.clamp(full[1] - full[0], min=1.0e-9)
    y_extent = torch.clamp(full[3] - full[2], min=1.0e-9)
    normalized = torch.stack(
        (
            (bounds[:, 0] - full[0]) / x_extent,
            (bounds[:, 1] - full[0]) / x_extent,
            (bounds[:, 2] - full[2]) / y_extent,
            (bounds[:, 3] - full[2]) / y_extent,
        ),
        dim=1,
    )
    normalized = torch.where(active[:, None], normalized, torch.zeros_like(normalized))
    return bounds, normalized, active


def _constant_stride_integer_lattice(
    coord_low: torch.Tensor,
    coord_high: torch.Tensor,
    *,
    full_low: torch.Tensor,
    full_high: torch.Tensor,
    reference_size: int,
    sample_count: int,
) -> torch.Tensor:
    """Return one centered constant-integer-stride lattice per environment."""

    if reference_size <= 0 or sample_count <= 0 or sample_count > reference_size:
        raise ValueError("TacMap lattice sizes must satisfy reference >= samples > 0")
    extent = torch.clamp(full_high - full_low, min=1.0e-9)
    low_float = (coord_low - full_low) / extent * float(reference_size - 1)
    high_float = (coord_high - full_low) / extent * float(reference_size - 1)
    low = torch.clamp(torch.floor(low_float).to(dtype=torch.long), min=0, max=reference_size - 1)
    high = torch.clamp(torch.ceil(high_float).to(dtype=torch.long), min=0, max=reference_size - 1)
    if sample_count == 1:
        return torch.round(0.5 * (low + high).to(dtype=torch.float32)).to(dtype=torch.long)[:, None]

    max_stride = max(1, (reference_size - 1) // (sample_count - 1))
    required_span = torch.clamp(high - low, min=0)
    stride = torch.div(
        required_span + (sample_count - 2),
        sample_count - 1,
        rounding_mode="floor",
    )
    stride = torch.clamp(stride, min=1, max=max_stride)
    lattice_span = stride * (sample_count - 1)
    center = 0.5 * (low + high).to(dtype=torch.float32)
    start = torch.round(center - 0.5 * lattice_span.to(dtype=torch.float32)).to(dtype=torch.long)
    start = torch.maximum(start, torch.zeros_like(start))
    start = torch.minimum(start, (reference_size - 1) - lattice_span)
    offsets = torch.arange(sample_count, device=coord_low.device, dtype=torch.long).reshape(1, -1)
    return start[:, None] + offsets * stride[:, None]


def _sample_local_tacmap_reference_batched(
    reference_state: dict,
    finger_index: int,
    roi_bounds_m: torch.Tensor,
    *,
    local_rows: int,
    local_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Index a camera-XY-aligned 25x40 lattice from the white 240x320 reference."""

    device = roi_bounds_m.device
    batch_size = int(roi_bounds_m.shape[0])
    reference_rows = int(reference_state["reference_rows"])
    reference_cols = int(reference_state["reference_cols"])
    if local_rows <= 0 or local_cols <= 0:
        raise ValueError("Adaptive TacMap local rows/cols must be positive")
    if local_rows > reference_rows or local_cols > reference_cols:
        raise ValueError(
            "Adaptive TacMap constant-stride sampling requires local rows/cols not to exceed "
            f"the {reference_rows}x{reference_cols} reference"
        )

    full = reference_state["full_camera_bounds_m"][finger_index].to(device=device, dtype=torch.float32)
    storage_row_axis = int(reference_state["storage_row_axis_camera"][finger_index])
    if storage_row_axis == 1:
        storage_rows = _constant_stride_integer_lattice(
            roi_bounds_m[:, 2],
            roi_bounds_m[:, 3],
            full_low=full[2],
            full_high=full[3],
            reference_size=reference_rows,
            sample_count=local_rows,
        )
        storage_cols = _constant_stride_integer_lattice(
            roi_bounds_m[:, 0],
            roi_bounds_m[:, 1],
            full_low=full[0],
            full_high=full[1],
            reference_size=reference_cols,
            sample_count=local_cols,
        )
        pixel_indices = storage_rows.unsqueeze(-1) * reference_cols + storage_cols.unsqueeze(1)
    elif storage_row_axis == 0:
        storage_rows = _constant_stride_integer_lattice(
            roi_bounds_m[:, 0],
            roi_bounds_m[:, 1],
            full_low=full[0],
            full_high=full[1],
            reference_size=reference_rows,
            sample_count=local_cols,
        )
        storage_cols = _constant_stride_integer_lattice(
            roi_bounds_m[:, 2],
            roi_bounds_m[:, 3],
            full_low=full[2],
            full_high=full[3],
            reference_size=reference_cols,
            sample_count=local_rows,
        )
        storage_indices = storage_rows.unsqueeze(-1) * reference_cols + storage_cols.unsqueeze(1)
        pixel_indices = storage_indices.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unknown TacMap reference storage row axis: {storage_row_axis}")
    flat_indices = pixel_indices.reshape(batch_size, local_rows * local_cols)

    reference_starts = reference_state["reference_starts_l"][finger_index].reshape(-1, 3)
    reference_directions = reference_state["reference_directions_l"][finger_index].reshape(-1, 3)
    reference_depth = reference_state["reference_depth_m"][finger_index].reshape(-1)
    reference_valid = reference_state["reference_valid"][finger_index].reshape(-1)
    starts = reference_starts[flat_indices]
    directions = reference_directions[flat_indices]
    depth = reference_depth[flat_indices]
    valid = reference_valid[flat_indices]
    return starts, directions, depth, valid, flat_indices


def adaptive_local_tacmap_rl_obs(
    env: ManagerBasedRLEnv,
    *,
    tacmap_sensor_names: list[str] | tuple[str, ...],
    tacmap_surface_sensor_names: list[str] | tuple[str, ...],
    finger_link_names: list[str] | tuple[str, ...],
    marker_layout_path: str | Path,
    coarse_rows: int = 32,
    coarse_cols: int = 24,
    reference_rows: int = 240,
    reference_cols: int = 320,
    local_rows: int = 25,
    local_cols: int = 40,
    contact_threshold_m: float = 0.02e-3,
    roi_margin_m: float = 1.0e-3,
    penetration_deadband_m: float = 1.0e-6,
    max_distance_m: float = 0.05,
    cache_debug_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return five adaptive local Depth patches using GPU-parallel raycasts."""

    object_names = tuple(str(name) for name in tacmap_sensor_names)
    surface_names = tuple(str(name) for name in tacmap_surface_sensor_names)
    link_names = tuple(str(name) for name in finger_link_names)
    sensor_count = min(len(object_names), len(surface_names), len(RL_FINGER_ORDER))
    if sensor_count != len(RL_FINGER_ORDER):
        raise RuntimeError(
            f"Adaptive TacMap requires {len(RL_FINGER_ORDER)} object/surface sensor pairs, got {sensor_count}"
        )
    current_step = int(getattr(env, "common_step_counter", -1))
    cached_depth = getattr(env, "_rl_local_tacmap_depth_m", None)
    cached_roi = getattr(env, "_rl_local_tacmap_roi_norm", None)
    cached_active = getattr(env, "_rl_local_tacmap_active", None)
    if (
        getattr(env, "_rl_local_tacmap_cache_step", None) == current_step
        and isinstance(cached_depth, torch.Tensor)
        and isinstance(cached_roi, torch.Tensor)
        and isinstance(cached_active, torch.Tensor)
    ):
        return cached_depth, cached_roi, cached_active

    coarse = getattr(env, "_rl_tacmap_penetration_m", None)
    expected_coarse_shape = (env.num_envs, sensor_count, int(coarse_rows), int(coarse_cols))
    if not isinstance(coarse, torch.Tensor) or tuple(coarse.shape) != expected_coarse_shape:
        raise RuntimeError(
            f"Adaptive TacMap requires the current coarse depth cache {expected_coarse_shape}, "
            f"got {None if not isinstance(coarse, torch.Tensor) else tuple(coarse.shape)}"
        )
    reference_state = _initialize_local_tacmap_reference(
        env,
        tacmap_surface_sensor_names=surface_names,
        finger_link_names=link_names,
        marker_layout_path=marker_layout_path,
        coarse_rows=coarse_rows,
        coarse_cols=coarse_cols,
        reference_rows=reference_rows,
        reference_cols=reference_cols,
        max_distance_m=max_distance_m,
    )

    depth_chunks: list[torch.Tensor] = []
    roi_chunks: list[torch.Tensor] = []
    active_chunks: list[torch.Tensor] = []
    debug_pixel_indices: list[torch.Tensor] = []
    debug_reference_points_l: list[torch.Tensor] = []
    debug_object_depth_m: list[torch.Tensor] = []
    debug_object_valid: list[torch.Tensor] = []
    debug_roi_bounds_m: list[torch.Tensor] = []
    pixel_index_chunks: list[torch.Tensor] = []
    for finger_index, object_name in enumerate(object_names[:sensor_count]):
        try:
            object_sensor = env.scene.sensors[object_name]
        except KeyError as exc:
            raise RuntimeError(f"Adaptive TacMap object sensor is missing: {object_name!r}") from exc
        roi_bounds, roi_norm, active = _local_tacmap_contact_roi_batched(
            coarse[:, finger_index],
            reference_state["coarse_xy_camera_m"][finger_index],
            reference_state["full_camera_bounds_m"][finger_index],
            contact_threshold_m=contact_threshold_m,
            roi_margin_m=roi_margin_m,
        )
        starts_l, directions_l, surface_depth, surface_valid, pixel_indices = (
            _sample_local_tacmap_reference_batched(
                reference_state,
                finger_index,
                roi_bounds,
                local_rows=local_rows,
                local_cols=local_cols,
            )
        )
        object_depth, object_valid, _, _ = _raycast_local_tacmap_distances(
            object_sensor,
            starts_l,
            directions_l,
            max_distance_m=max_distance_m,
        )
        penetration = torch.where(
            surface_valid & object_valid & active[:, None],
            torch.clamp(surface_depth - object_depth, min=0.0),
            torch.zeros_like(surface_depth),
        )
        penetration = torch.where(
            penetration > max(0.0, float(penetration_deadband_m)),
            penetration,
            torch.zeros_like(penetration),
        )
        depth_chunks.append(penetration.reshape(env.num_envs, local_rows, local_cols))
        roi_chunks.append(roi_norm)
        active_chunks.append(active)
        pixel_index_chunks.append(pixel_indices.reshape(env.num_envs, local_rows, local_cols))

        if bool(cache_debug_output):
            debug_pixel_indices.append(pixel_indices[0].detach())
            debug_reference_points_l.append(
                (starts_l[0] + directions_l[0] * surface_depth[0].unsqueeze(-1)).detach()
            )
            debug_object_depth_m.append(object_depth[0].detach())
            debug_object_valid.append(object_valid[0].detach())
            debug_roi_bounds_m.append(roi_bounds[0].detach())

    depth_output = torch.stack(depth_chunks, dim=1)
    roi_output = torch.stack(roi_chunks, dim=1)
    active_output = torch.stack(active_chunks, dim=1)
    env._rl_local_tacmap_depth_m = depth_output
    env._rl_local_tacmap_roi_norm = roi_output
    env._rl_local_tacmap_active = active_output
    env._rl_local_tacmap_pixel_indices = torch.stack(pixel_index_chunks, dim=1)
    env._rl_local_tacmap_cache_step = current_step
    if bool(cache_debug_output):
        env._rl_local_tacmap_debug = {
            "pixel_indices": torch.stack(debug_pixel_indices, dim=0),
            "reference_points_l": torch.stack(debug_reference_points_l, dim=0),
            "object_depth_m": torch.stack(debug_object_depth_m, dim=0),
            "object_valid": torch.stack(debug_object_valid, dim=0),
            "roi_bounds_m": torch.stack(debug_roi_bounds_m, dim=0),
        }
    else:
        env._rl_local_tacmap_debug = None
    return depth_output, roi_output, active_output


def tacmap_rl_obs(
    env: ManagerBasedRLEnv,
    tacmap_sensor_names: list[str] | tuple[str, ...] = (),
    tacmap_surface_sensor_names: list[str] | tuple[str, ...] = (),
    tacmap_surface_reference_npy: str = "",
    tacmap_rows: int = 240,
    tacmap_cols: int = 240,
    cache_aux_fields: bool = False,
    contact_shell_m: float = 0.0,
) -> torch.Tensor:
    """Flatten TacMap link-surface penetration maps as GPU RL observation."""

    rows = max(0, int(tacmap_rows))
    cols = max(0, int(tacmap_cols))
    if rows <= 0 or cols <= 0:
        return _zeros(env, 0)

    per_sensor_dim = rows * cols
    chunks: list[torch.Tensor] = []
    object_names = tuple(str(name) for name in tacmap_sensor_names)
    surface_names = tuple(str(name) for name in tacmap_surface_sensor_names)
    static_reference_path = str(tacmap_surface_reference_npy).strip()
    if cache_aux_fields and static_reference_path:
        raise ValueError(
            "cache_aux_fields=True requires live TacMap surface sensors; "
            "a depth-only static surface reference has no points or normals"
        )
    if not object_names or (not surface_names and not static_reference_path):
        _warn_once(
            env,
            "missing_tacmap_rl_sensors",
            "[WARN] RL TacMap sensors are not configured in this ManagerBased env; using zeros.",
        )
        zero_penetration = torch.zeros(
            (env.num_envs, len(RL_FINGER_ORDER), rows, cols), device=env.device, dtype=torch.float32
        )
        env._rl_tacmap_penetration_m = zero_penetration
        if cache_aux_fields:
            zero_points = torch.zeros(
                (env.num_envs, len(RL_FINGER_ORDER), rows, cols, 3), device=env.device, dtype=torch.float32
            )
            zero_valid = torch.zeros((env.num_envs, len(RL_FINGER_ORDER), rows, cols), device=env.device, dtype=torch.bool)
            env._rl_tacmap_surface_points_w = zero_points
            env._rl_tacmap_surface_normals_w = zero_points
            env._rl_tacmap_ray_directions_w = torch.zeros(
                (env.num_envs, len(RL_FINGER_ORDER), 3), device=env.device, dtype=torch.float32
            )
            env._rl_tacmap_surface_valid = zero_valid
            env._rl_tacmap_object_points_w = zero_points
            env._rl_tacmap_object_valid = zero_valid
            env._rl_tacmap_surface_raw_m = zero_penetration
        else:
            _clear_tacmap_aux_cache(env)
        env._rl_tacmap_cache_step = int(getattr(env, "common_step_counter", -1))
        return zero_penetration.reshape(env.num_envs, -1)

    sensor_count = (
        len(object_names)
        if static_reference_path
        else max(len(object_names), len(surface_names))
    )
    static_surface_depth = (
        _tacmap_static_surface_reference(
            env,
            static_reference_path,
            sensor_count=sensor_count,
            rows=rows,
            cols=cols,
        )
        if static_reference_path
        else None
    )
    penetration_images: list[torch.Tensor] = []
    surface_points_images: list[torch.Tensor] = []
    surface_normals_images: list[torch.Tensor] = []
    ray_direction_images: list[torch.Tensor] = []
    surface_valid_images: list[torch.Tensor] = []
    object_points_images: list[torch.Tensor] = []
    object_valid_images: list[torch.Tensor] = []
    surface_raw_images: list[torch.Tensor] = []

    zero_image = torch.zeros((env.num_envs, rows, cols), device=env.device, dtype=torch.float32)
    zero_points = (
        torch.zeros((env.num_envs, rows, cols, 3), device=env.device, dtype=torch.float32)
        if cache_aux_fields
        else None
    )
    zero_valid = (
        torch.zeros((env.num_envs, rows, cols), device=env.device, dtype=torch.bool) if cache_aux_fields else None
    )
    zero_direction = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)

    for i in range(sensor_count):
        object_name = object_names[i] if i < len(object_names) else ""
        surface_name = surface_names[i] if i < len(surface_names) else ""
        try:
            object_sensor = env.scene.sensors[object_name]
            surface_sensor = None if static_surface_depth is not None else env.scene.sensors[surface_name]
        except KeyError:
            _warn_once(
                env,
                f"missing_tacmap_{object_name}_{surface_name}",
                f"[WARN] RL TacMap sensor pair object={object_name!r}, surface={surface_name!r} is missing; using zeros.",
            )
            chunks.append(_zeros(env, per_sensor_dim))
            penetration_images.append(zero_image)
            if cache_aux_fields:
                assert zero_points is not None
                assert zero_valid is not None
                surface_points_images.append(zero_points)
                surface_normals_images.append(zero_points)
                ray_direction_images.append(zero_direction)
                surface_valid_images.append(zero_valid)
                object_points_images.append(zero_points)
                object_valid_images.append(zero_valid)
                surface_raw_images.append(zero_image)
            continue

        surface_reference = (
            {
                "surface_dist_m": static_surface_depth[i : i + 1],
                "surface_valid": static_surface_depth[i : i + 1] > 0.0,
                "has_geometry": False,
            }
            if static_surface_depth is not None
            else _tacmap_fixed_surface_reference(
                env,
                surface_name=surface_name,
                object_name=object_name,
                surface_sensor=surface_sensor,
                object_sensor=object_sensor,
                rows=rows,
                cols=cols,
                require_geometry=cache_aux_fields,
            )
        )
        if surface_reference is None and surface_sensor is not None and hasattr(surface_sensor, "update"):
            surface_sensor.update(0.0, force_recompute=True)
        if hasattr(object_sensor, "update"):
            object_sensor.update(0.0, force_recompute=True)

        surface_dist = (
            surface_reference["surface_dist_m"].expand(env.num_envs, -1, -1)
            if surface_reference is not None
            else _tacmap_raw_image(surface_sensor, env, rows, cols)
        )
        object_dist = _tacmap_raw_image(object_sensor, env, rows, cols)
        if surface_dist is None or object_dist is None:
            chunks.append(_zeros(env, per_sensor_dim))
            penetration_images.append(zero_image)
            if cache_aux_fields:
                assert zero_points is not None
                assert zero_valid is not None
                surface_points_images.append(zero_points)
                surface_normals_images.append(zero_points)
                ray_direction_images.append(zero_direction)
                surface_valid_images.append(zero_valid)
                object_points_images.append(zero_points)
                object_valid_images.append(zero_valid)
                surface_raw_images.append(zero_image)
            continue

        if surface_reference is not None:
            surface_valid = surface_reference["surface_valid"].expand(env.num_envs, -1, -1)
        else:
            surface_valid = _tacmap_valid_image(surface_sensor, "ray_hit_valid", env, rows, cols)
            if surface_valid is None:
                surface_valid = _tacmap_valid_image(surface_sensor, "second_ray_hit_valid", env, rows, cols)
            if surface_valid is None:
                surface_valid = surface_dist > 0.0
        object_valid = object_dist > 0.0

        penetration = _tacmap_penetration_from_distances(
            surface_dist,
            object_dist,
            surface_valid & object_valid,
            contact_shell_m=contact_shell_m,
        )
        chunks.append(_finite_flat(penetration, env))
        penetration_images.append(penetration)
        if cache_aux_fields:
            assert zero_points is not None
            cached_world_fields = (
                _tacmap_cached_surface_world_fields(surface_reference, object_sensor, env, rows, cols)
                if surface_reference is not None
                else None
            )
            if cached_world_fields is not None:
                surface_points, surface_normals, ray_direction = cached_world_fields
            else:
                if surface_reference is not None and surface_sensor is not None and hasattr(surface_sensor, "update"):
                    surface_sensor.update(0.0, force_recompute=True)
                surface_points = _tacmap_vec_image(surface_sensor, "ray_hits_w", env, rows, cols)
                if surface_points is None:
                    surface_points = _tacmap_vec_image(surface_sensor, "second_ray_hits_w", env, rows, cols)
                if surface_points is None:
                    surface_points = zero_points
                surface_normals = _tacmap_vec_image(surface_sensor, "ray_normals_w", env, rows, cols)
                if surface_normals is None:
                    surface_normals = _tacmap_vec_image(surface_sensor, "second_ray_normals_w", env, rows, cols)
                if surface_normals is None:
                    surface_normals = zero_points
                ray_direction = _tacmap_mean_ray_direction(surface_sensor, env, rows, cols)
                if ray_direction is None:
                    ray_direction = zero_direction
            object_points = _tacmap_vec_image(object_sensor, "ray_hits_w", env, rows, cols)
            if object_points is None:
                object_points = zero_points
            surface_points_images.append(surface_points)
            surface_normals_images.append(surface_normals)
            ray_direction_images.append(ray_direction)
            surface_valid_images.append(surface_valid)
            object_points_images.append(object_points)
            object_valid_images.append(object_valid)
            surface_raw_images.append(surface_dist)

    env._rl_tacmap_penetration_m = torch.stack(penetration_images, dim=1) if penetration_images else torch.zeros(
        (env.num_envs, 0, rows, cols), device=env.device, dtype=torch.float32
    )
    if cache_aux_fields:
        env._rl_tacmap_surface_points_w = (
            torch.stack(surface_points_images, dim=1)
            if surface_points_images
            else torch.zeros((env.num_envs, 0, rows, cols, 3), device=env.device, dtype=torch.float32)
        )
        env._rl_tacmap_surface_normals_w = (
            torch.stack(surface_normals_images, dim=1)
            if surface_normals_images
            else torch.zeros((env.num_envs, 0, rows, cols, 3), device=env.device, dtype=torch.float32)
        )
        env._rl_tacmap_ray_directions_w = (
            torch.stack(ray_direction_images, dim=1)
            if ray_direction_images
            else torch.zeros((env.num_envs, 0, 3), device=env.device, dtype=torch.float32)
        )
        env._rl_tacmap_surface_valid = (
            torch.stack(surface_valid_images, dim=1)
            if surface_valid_images
            else torch.zeros((env.num_envs, 0, rows, cols), device=env.device, dtype=torch.bool)
        )
        env._rl_tacmap_object_points_w = (
            torch.stack(object_points_images, dim=1)
            if object_points_images
            else torch.zeros((env.num_envs, 0, rows, cols, 3), device=env.device, dtype=torch.float32)
        )
        env._rl_tacmap_object_valid = (
            torch.stack(object_valid_images, dim=1)
            if object_valid_images
            else torch.zeros((env.num_envs, 0, rows, cols), device=env.device, dtype=torch.bool)
        )
        env._rl_tacmap_surface_raw_m = (
            torch.stack(surface_raw_images, dim=1)
            if surface_raw_images
            else torch.zeros((env.num_envs, 0, rows, cols), device=env.device, dtype=torch.float32)
        )
    else:
        _clear_tacmap_aux_cache(env)
    env._rl_tacmap_cache_step = int(getattr(env, "common_step_counter", -1))

    return torch.cat(chunks, dim=1) if chunks else _zeros(env, 0)


def tacsl_baseline_rl_obs(
    env: ManagerBasedRLEnv,
    tacmap_sensor_names: list[str] | tuple[str, ...],
    tacmap_surface_sensor_names: list[str] | tuple[str, ...],
    finger_link_names: list[str] | tuple[str, ...],
    ray_direction_axes: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
    row_axis_names: list[str] | tuple[str, ...],
    col_axis_names: list[str] | tuple[str, ...],
    ray_rows: int = 320,
    ray_cols: int = 240,
    output_rows: int = 16,
    output_cols: int = 8,
    normal_contact_stiffness: float = 1.0,
    tangential_stiffness: float = 0.1,
    friction_coefficient: float = 2.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the GPU TacSL baseline force field, pooled from 320x240 link-surface rays."""

    rows = max(1, int(ray_rows))
    cols = max(1, int(ray_cols))
    out_rows = max(1, int(output_rows))
    out_cols = max(1, int(output_cols))
    object_names = tuple(str(name) for name in tacmap_sensor_names)
    surface_names = tuple(str(name) for name in tacmap_surface_sensor_names)
    link_names = tuple(str(name) for name in finger_link_names)
    sensor_count = min(
        len(object_names),
        len(surface_names),
        len(link_names),
        len(ray_direction_axes),
        len(row_axis_names),
        len(col_axis_names),
    )
    output_dim = sensor_count * out_rows * out_cols * 3
    cache_step = int(getattr(env, "common_step_counter", -1))
    cached = getattr(env, "_brainco_rl_tacsl_baseline_obs", None)
    if (
        isinstance(cached, torch.Tensor)
        and cached.shape == (env.num_envs, output_dim)
        and getattr(env, "_brainco_rl_tacsl_baseline_cache_step", None) == cache_step
    ):
        return cached
    if sensor_count == 0:
        return _zeros(env, 0)

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id_key = (tuple(robot.body_names), link_names)
    if getattr(env, "_brainco_rl_tacsl_body_id_key", None) != body_id_key:
        missing_links = [name for name in link_names[:sensor_count] if name not in robot.body_names]
        if missing_links:
            raise RuntimeError(f"TacSL baseline finger links are missing from the robot articulation: {missing_links}")
        env._brainco_rl_tacsl_body_ids = tuple(robot.body_names.index(name) for name in link_names[:sensor_count])
        env._brainco_rl_tacsl_body_id_key = body_id_key
    body_ids = tuple(int(value) for value in env._brainco_rl_tacsl_body_ids)

    finger_pos_w = robot.data.body_pos_w[:, body_ids]
    finger_quat_w = F.normalize(robot.data.body_quat_w[:, body_ids], dim=-1, eps=1.0e-8)
    object_pos_w = obj.data.root_pos_w[:, None, :].expand(-1, sensor_count, -1)
    object_quat_w = F.normalize(obj.data.root_quat_w[:, None, :].expand(-1, sensor_count, -1), dim=-1, eps=1.0e-8)
    flat_finger_quat_inv = quat_inv(finger_quat_w.reshape(-1, 4))
    contact_pos_l = quat_apply(
        flat_finger_quat_inv,
        (object_pos_w - finger_pos_w).reshape(-1, 3),
    ).reshape(env.num_envs, sensor_count, 3)
    contact_quat_l = quat_mul(flat_finger_quat_inv, object_quat_w.reshape(-1, 4)).reshape(
        env.num_envs, sensor_count, 4
    )
    contact_quat_l = F.normalize(contact_quat_l, dim=-1, eps=1.0e-8)

    state_shape = (env.num_envs, sensor_count)
    state_valid = getattr(env, "_brainco_rl_tacsl_state_valid", None)
    if not isinstance(state_valid, torch.Tensor) or state_valid.shape != state_shape:
        env._brainco_rl_tacsl_prev_contact_pos_l = torch.zeros_like(contact_pos_l)
        env._brainco_rl_tacsl_prev_contact_quat_l = torch.zeros_like(contact_quat_l)
        env._brainco_rl_tacsl_prev_contact_quat_l[..., 0] = 1.0
        env._brainco_rl_tacsl_state_valid = torch.zeros(state_shape, device=env.device, dtype=torch.bool)
    prev_contact_pos_l = env._brainco_rl_tacsl_prev_contact_pos_l
    prev_contact_quat_l = env._brainco_rl_tacsl_prev_contact_quat_l
    state_valid = env._brainco_rl_tacsl_state_valid
    episode_length = getattr(env, "episode_length_buf", None)
    if isinstance(episode_length, torch.Tensor):
        state_valid[episode_length[: env.num_envs] == 0] = False

    dt = max(float(getattr(env, "step_dt", 0.0)), 1.0e-8)
    pooled_fields: list[torch.Tensor] = []
    zero_field = torch.zeros((env.num_envs, out_rows, out_cols, 3), device=env.device, dtype=torch.float32)
    surface_cache = getattr(env, "_brainco_rl_tacsl_surface_cache", None)
    if not isinstance(surface_cache, dict):
        surface_cache = {}
        env._brainco_rl_tacsl_surface_cache = surface_cache
    for sensor_index in range(sensor_count):
        try:
            object_sensor = env.scene.sensors[object_names[sensor_index]]
            surface_sensor = env.scene.sensors[surface_names[sensor_index]]
        except KeyError:
            _warn_once(
                env,
                f"missing_tacsl_baseline_sensor_{sensor_index}",
                f"[WARN] TacSL baseline sensor pair {sensor_index} is missing; using zeros.",
            )
            pooled_fields.append(zero_field)
            continue

        if hasattr(object_sensor, "update"):
            object_sensor.update(0.0, force_recompute=True)
        surface_key = (surface_names[sensor_index], rows, cols)
        cached_surface = surface_cache.get(surface_key)
        if cached_surface is None:
            if hasattr(surface_sensor, "update"):
                surface_sensor.update(0.0, force_recompute=True)
            surface_dist_init = _tacmap_raw_image(surface_sensor, env, rows, cols)
            surface_valid_init = _tacmap_valid_image(surface_sensor, "ray_hit_valid", env, rows, cols)
            if surface_valid_init is None:
                surface_valid_init = _tacmap_valid_image(surface_sensor, "second_ray_hit_valid", env, rows, cols)
            if surface_dist_init is not None:
                if surface_valid_init is None:
                    surface_valid_init = surface_dist_init > 0.0
                cached_surface = (
                    surface_dist_init[:1].detach().clone(),
                    surface_valid_init[:1].detach().clone(),
                )
                surface_cache[surface_key] = cached_surface
        if cached_surface is None:
            pooled_fields.append(zero_field)
            continue
        surface_dist = cached_surface[0].expand(env.num_envs, -1, -1)
        surface_valid = cached_surface[1].expand(env.num_envs, -1, -1)
        object_dist = _tacmap_raw_image(object_sensor, env, rows, cols)
        object_points_w = _tacmap_vec_image(object_sensor, "ray_hits_w", env, rows, cols)
        if surface_dist is None or object_dist is None or object_points_w is None:
            pooled_fields.append(zero_field)
            continue

        object_valid = _tacmap_valid_image(object_sensor, "ray_hit_valid", env, rows, cols)
        if object_valid is None:
            object_valid = object_dist > 0.0
        hit_valid = surface_valid & object_valid
        depth = _tacmap_penetration_from_distances(surface_dist, object_dist, hit_valid, contact_shell_m=0.0)

        finger_pos = finger_pos_w[:, sensor_index]
        finger_quat_inv = quat_inv(finger_quat_w[:, sensor_index])
        point_count = rows * cols
        closest_points_l = quat_apply(
            finger_quat_inv[:, None, :].expand(-1, point_count, -1).reshape(-1, 4),
            (object_points_w.reshape(env.num_envs, point_count, 3) - finger_pos[:, None, :]).reshape(-1, 3),
        ).reshape(env.num_envs, rows, cols, 3)
        closest_points_l = torch.where(hit_valid.unsqueeze(-1), closest_points_l, torch.zeros_like(closest_points_l))

        ray_direction_l = torch.as_tensor(ray_direction_axes[sensor_index], device=env.device, dtype=torch.float32)
        row_axis_l = _tacsl_axis_vector(row_axis_names[sensor_index], device=env.device, dtype=torch.float32)
        col_axis_l = _tacsl_axis_vector(col_axis_names[sensor_index], device=env.device, dtype=torch.float32)
        normal_force, shear_force = _tacsl_force_field_from_relative_pose_batched(
            depth,
            closest_points_l,
            contact_pos_l[:, sensor_index],
            contact_quat_l[:, sensor_index],
            prev_contact_pos_l[:, sensor_index],
            prev_contact_quat_l[:, sensor_index],
            state_valid[:, sensor_index],
            ray_direction_l=ray_direction_l,
            row_axis_l=row_axis_l,
            col_axis_l=col_axis_l,
            dt=dt,
            normal_contact_stiffness=normal_contact_stiffness,
            tangential_stiffness=tangential_stiffness,
            friction_coefficient=friction_coefficient,
        )
        pooled_fields.append(_pool_tacsl_force_field_batched(normal_force, shear_force, out_rows, out_cols))

    prev_contact_pos_l.copy_(contact_pos_l.detach())
    prev_contact_quat_l.copy_(contact_quat_l.detach())
    state_valid.fill_(True)
    output = torch.stack(pooled_fields, dim=1).reshape(env.num_envs, -1)
    output = _finite_flat(output, env)
    env._brainco_rl_tacsl_baseline_obs = output
    env._brainco_rl_tacsl_baseline_cache_step = cache_step
    return output


def fots_baseline_rl_obs(
    env: ManagerBasedRLEnv,
    tacmap_sensor_names: list[str] | tuple[str, ...],
    tacmap_surface_sensor_names: list[str] | tuple[str, ...],
    finger_link_names: list[str] | tuple[str, ...],
    ray_direction_axes: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
    ray_rows: int = 32,
    ray_cols: int = 24,
    render_rows: int = RL_FOTS_RENDER_ROWS,
    render_cols: int = RL_FOTS_RENDER_COLS,
    marker_rows: int = RL_FOTS_MARKER_ROWS,
    marker_cols: int = RL_FOTS_MARKER_COLS,
    marker_margin_x: float = 15.0,
    marker_margin_y: float = 26.0 * RL_FOTS_RENDER_ROWS / 240.0,
    mm2pix: float = 19.58,
    lamb_dilate: float = 0.00125,
    lamb_shear: float = 0.00021,
    lamb_twist: float = 0.00038,
    contact_threshold_mm: float = 0.02,
    depth_scale: float = 1.0,
    track_contact_center: bool = True,
    finger_count: int = len(RL_FINGER_ORDER),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return batched FOTS marker displacement from native 32x24 TacMap depth."""

    rows = max(1, int(ray_rows))
    cols = max(1, int(ray_cols))
    marker_count = max(1, int(marker_rows)) * max(1, int(marker_cols))
    object_names = tuple(str(name) for name in tacmap_sensor_names)
    surface_names = tuple(str(name) for name in tacmap_surface_sensor_names)
    link_names = tuple(str(name) for name in finger_link_names)
    sensor_count = min(
        len(object_names),
        len(surface_names),
        len(link_names),
        len(ray_direction_axes),
        max(0, int(finger_count)),
    )
    output_dim = max(0, int(finger_count)) * marker_count * 2
    cache_step = int(getattr(env, "common_step_counter", -1))
    cached = getattr(env, "_brainco_rl_fots_baseline_obs", None)
    if (
        isinstance(cached, torch.Tensor)
        and cached.shape == (env.num_envs, output_dim)
        and getattr(env, "_brainco_rl_fots_baseline_cache_step", None) == cache_step
    ):
        return cached
    if sensor_count == 0:
        return _zeros(env, output_dim)

    depth = getattr(env, "_rl_tacmap_penetration_m", None)
    depth_cache_step = getattr(env, "_rl_tacmap_cache_step", None)
    if (
        not isinstance(depth, torch.Tensor)
        or depth_cache_step != cache_step
        or depth.shape != (env.num_envs, sensor_count, rows, cols)
    ):
        tacmap_rl_obs(
            env,
            tacmap_sensor_names=object_names[:sensor_count],
            tacmap_surface_sensor_names=surface_names[:sensor_count],
            tacmap_rows=rows,
            tacmap_cols=cols,
            cache_aux_fields=False,
            contact_shell_m=0.0,
        )
        depth = getattr(env, "_rl_tacmap_penetration_m", None)
    if not isinstance(depth, torch.Tensor) or depth.ndim != 4 or depth.shape[0] != env.num_envs:
        _warn_once(env, "missing_fots_depth", "[WARN] FOTS baseline TacMap depth is unavailable; using zeros.")
        return _zeros(env, output_dim)
    depth = _finite_tensor(depth[:, :sensor_count, :rows, :cols], env)

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id_key = (tuple(robot.body_names), link_names[:sensor_count])
    if getattr(env, "_brainco_rl_fots_body_id_key", None) != body_id_key:
        missing_links = [name for name in link_names[:sensor_count] if name not in robot.body_names]
        if missing_links:
            raise RuntimeError(f"FOTS baseline finger links are missing from the robot articulation: {missing_links}")
        env._brainco_rl_fots_body_ids = tuple(robot.body_names.index(name) for name in link_names[:sensor_count])
        env._brainco_rl_fots_body_id_key = body_id_key
    body_ids = tuple(int(value) for value in env._brainco_rl_fots_body_ids)

    finger_quat_w = F.normalize(robot.data.body_quat_w[:, body_ids], dim=-1, eps=1.0e-8)
    object_quat_w = F.normalize(obj.data.root_quat_w[:, None, :].expand(-1, sensor_count, -1), dim=-1, eps=1.0e-8)
    relative_quat_l = quat_mul(quat_inv(finger_quat_w.reshape(-1, 4)), object_quat_w.reshape(-1, 4)).reshape(
        env.num_envs, sensor_count, 4
    )
    ray_axes_l = torch.as_tensor(
        ray_direction_axes[:sensor_count], device=env.device, dtype=torch.float32
    ).reshape(1, sensor_count, 3)
    theta = _fots_signed_twist_angle_batched(relative_quat_l, ray_axes_l).reshape(-1)

    try:
        _ensure_integrate_import_path()
        from integrate.fots_adapter import RevoFotsCfg, RevoFotsGpuAdapter
    except Exception as exc:
        _warn_once(env, "fots_adapter_import_failed", f"[WARN] RL FOTS adapter import failed ({exc}); using zeros.")
        return _zeros(env, output_dim)

    cfg_key = (
        rows,
        cols,
        int(render_rows),
        int(render_cols),
        int(marker_rows),
        int(marker_cols),
        float(marker_margin_x),
        float(marker_margin_y),
        float(mm2pix),
        float(lamb_dilate),
        float(lamb_shear),
        float(lamb_twist),
        float(contact_threshold_mm),
        float(depth_scale),
        bool(track_contact_center),
        str(env.device),
    )
    adapter = getattr(env, "_brainco_rl_fots_baseline_adapter", None)
    if adapter is None or getattr(env, "_brainco_rl_fots_baseline_adapter_key", None) != cfg_key:
        adapter = RevoFotsGpuAdapter(
            RevoFotsCfg(
                width=max(1, int(render_cols)),
                height=max(1, int(render_rows)),
                marker_rows=max(1, int(marker_rows)),
                marker_cols=max(1, int(marker_cols)),
                marker_margin_x=float(marker_margin_x),
                marker_margin_y=float(marker_margin_y),
                mm2pix=float(mm2pix),
                lamb=(float(lamb_dilate), float(lamb_shear), float(lamb_twist)),
                contact_threshold_mm=float(contact_threshold_mm),
                depth_scale=float(depth_scale),
                track_contact_center=bool(track_contact_center),
                render_marker_image=False,
                render_overlay_image=False,
                device=str(env.device),
            )
        )
        env._brainco_rl_fots_baseline_adapter = adapter
        env._brainco_rl_fots_baseline_adapter_key = cfg_key

    episode_length = getattr(env, "episode_length_buf", None)
    if isinstance(episode_length, torch.Tensor):
        reset_env = episode_length[: env.num_envs] == 0
        adapter.reset(reset_env[:, None].expand(-1, sensor_count).reshape(-1))

    displacement = adapter.step_displacement_only(
        depth.reshape(env.num_envs * sensor_count, rows, cols),
        theta_rad=theta,
    ).reshape(env.num_envs, sensor_count, marker_count, 2)
    displacement = torch.nan_to_num(displacement, nan=0.0, posinf=0.0, neginf=0.0)
    full = torch.zeros(
        (env.num_envs, max(0, int(finger_count)), marker_count, 2),
        device=env.device,
        dtype=torch.float32,
    )
    full[:, :sensor_count] = displacement
    output = full.reshape(env.num_envs, -1)
    env._brainco_rl_fots_baseline_displacement_px = full
    env._rl_fots_baseline_output = adapter.last_output
    env._brainco_rl_fots_baseline_obs = output
    env._brainco_rl_fots_baseline_cache_step = cache_step
    return output


def _hydroshear_calibrated_marker_world(
    env: ManagerBasedRLEnv,
    *,
    marker_layout_path: str | Path,
    finger_link_names: list[str] | tuple[str, ...],
    sensor_count: int,
    robot_cfg: SceneEntityCfg,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout_path = Path(marker_layout_path).resolve()
    link_names = tuple(str(name) for name in finger_link_names[:sensor_count])
    if len(link_names) != int(sensor_count):
        raise ValueError(f"Expected {sensor_count} HydroShear finger links, got {len(link_names)}")

    cache_key = (str(layout_path), link_names)
    cache = getattr(env, "_brainco_rl_hydroshear_marker_layout", None)
    if cache is None or getattr(env, "_brainco_rl_hydroshear_marker_layout_key", None) != cache_key:
        with np.load(layout_path, allow_pickle=False) as layout:
            marker_uv = np.asarray(layout["pixels_distorted"], dtype=np.float32).reshape(-1, 2)
            distortion_valid = np.asarray(layout["distortion_valid"], dtype=bool).reshape(-1)
            marker_count = int(marker_uv.shape[0])
            points_l = np.zeros((sensor_count, marker_count, 3), dtype=np.float32)
            normals_l = np.zeros_like(points_l)
            valid = np.zeros((sensor_count, marker_count), dtype=bool)
            for finger_index, finger in enumerate(RL_FINGER_ORDER[:sensor_count]):
                points_camera = np.asarray(
                    layout[f"{finger}_points_camera_m"],
                    dtype=np.float32,
                ).reshape(-1, 3)
                camera_origin = np.asarray(
                    layout[f"{finger}_camera_origin_link_m"],
                    dtype=np.float32,
                ).reshape(3)
                camera_rotation = np.asarray(
                    layout[f"{finger}_camera_rotation_link"],
                    dtype=np.float32,
                ).reshape(3, 3)
                points = points_camera @ camera_rotation.T + camera_origin
                normals = np.asarray(layout[f"{finger}_normals_link"], dtype=np.float32).reshape(-1, 3)
                methods = np.asarray(layout[f"{finger}_method"]).astype(str).reshape(-1)
                if not (len(points) == len(normals) == len(methods) == len(distortion_valid) == marker_count):
                    raise ValueError(f"Calibrated marker array length mismatch for {finger}")
                normal_norm = np.linalg.norm(normals, axis=-1)
                exact = (
                    distortion_valid
                    & (methods == "ray_hit")
                    & np.isfinite(points).all(axis=-1)
                    & np.isfinite(normals).all(axis=-1)
                    & (normal_norm > 1.0e-9)
                )
                points_l[finger_index] = points
                normals_l[finger_index, exact] = normals[exact] / normal_norm[exact, None]
                valid[finger_index] = exact
        cache = (marker_uv, points_l, normals_l, valid)
        env._brainco_rl_hydroshear_marker_layout = cache
        env._brainco_rl_hydroshear_marker_layout_key = cache_key

    marker_uv, points_l_np, normals_l_np, valid_np = cache
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids = tuple(robot.body_names.index(name) for name in link_names)
    link_state = robot.data.body_link_state_w[:, body_ids, :7]
    marker_count = int(marker_uv.shape[0])
    points_l = torch.as_tensor(points_l_np, device=env.device, dtype=torch.float32)
    normals_l = torch.as_tensor(normals_l_np, device=env.device, dtype=torch.float32)
    points_l = points_l.unsqueeze(0).expand(env.num_envs, -1, -1, -1)
    normals_l = normals_l.unsqueeze(0).expand_as(points_l)
    quats = link_state[:, :, None, 3:7].expand(env.num_envs, sensor_count, marker_count, 4)
    points_w = link_state[:, :, None, :3] + quat_apply(
        quats.reshape(-1, 4),
        points_l.reshape(-1, 3),
    ).reshape(env.num_envs, sensor_count, marker_count, 3)
    normals_w = quat_apply(
        quats.reshape(-1, 4),
        normals_l.reshape(-1, 3),
    ).reshape(env.num_envs, sensor_count, marker_count, 3)
    normals_w = F.normalize(normals_w, dim=-1, eps=1.0e-8)
    marker_valid = torch.as_tensor(valid_np, device=env.device, dtype=torch.bool)
    marker_valid = marker_valid.unsqueeze(0).expand(env.num_envs, -1, -1)
    return (
        marker_uv,
        points_w.reshape(env.num_envs * sensor_count, marker_count, 3),
        normals_w.reshape(env.num_envs * sensor_count, marker_count, 3),
        marker_valid.reshape(env.num_envs * sensor_count, marker_count),
        link_state.reshape(env.num_envs * sensor_count, 7),
    )


def _hydroshear_marker_ray_measurements(
    env: ManagerBasedRLEnv,
    *,
    marker_sensor_names: list[str] | tuple[str, ...],
    marker_surface_sensor_names: list[str] | tuple[str, ...],
    marker_rows: int,
    marker_cols: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return independent marker-ray penetration and calibrated-surface validity."""

    object_names = tuple(str(name) for name in marker_sensor_names)
    surface_names = tuple(str(name) for name in marker_surface_sensor_names)
    if not object_names or not surface_names:
        return None

    rows = max(0, int(marker_rows))
    cols = max(0, int(marker_cols))
    if rows <= 0 or cols <= 0:
        return None
    sensor_count = min(len(object_names), len(surface_names), len(RL_FINGER_ORDER))
    zero_depth = torch.zeros((env.num_envs, rows, cols), device=env.device, dtype=torch.float32)
    zero_valid = torch.zeros((env.num_envs, rows, cols), device=env.device, dtype=torch.bool)
    depths: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []

    for object_name, surface_name in zip(object_names[:sensor_count], surface_names[:sensor_count], strict=True):
        try:
            object_sensor = env.scene.sensors[object_name]
            surface_sensor = env.scene.sensors[surface_name]
        except KeyError:
            _warn_once(
                env,
                f"missing_hydroshear_marker_rays_{object_name}_{surface_name}",
                f"[WARN] HydroShear marker ray pair object={object_name!r}, "
                f"surface={surface_name!r} is missing; marker depths use zeros.",
            )
            depths.append(zero_depth)
            valid_masks.append(zero_valid)
            continue

        surface_reference = _tacmap_fixed_surface_reference(
            env,
            surface_name=surface_name,
            object_name=object_name,
            surface_sensor=surface_sensor,
            object_sensor=object_sensor,
            rows=rows,
            cols=cols,
            require_geometry=False,
        )
        if surface_reference is None and hasattr(surface_sensor, "update"):
            surface_sensor.update(0.0, force_recompute=True)
        if hasattr(object_sensor, "update"):
            object_sensor.update(0.0, force_recompute=True)
        surface_dist = (
            surface_reference["surface_dist_m"].expand(env.num_envs, -1, -1)
            if surface_reference is not None
            else _tacmap_raw_image(surface_sensor, env, rows, cols)
        )
        object_dist = _tacmap_raw_image(object_sensor, env, rows, cols)
        if surface_dist is None or object_dist is None:
            depths.append(zero_depth)
            valid_masks.append(zero_valid)
            continue

        if surface_reference is not None:
            surface_valid = surface_reference["surface_valid"].expand(env.num_envs, -1, -1)
        else:
            surface_valid = _tacmap_valid_image(surface_sensor, "ray_hit_valid", env, rows, cols)
            if surface_valid is None:
                surface_valid = _tacmap_valid_image(surface_sensor, "second_ray_hit_valid", env, rows, cols)
            if surface_valid is None:
                surface_valid = surface_dist > 0.0
        object_valid = object_dist > 0.0
        depths.append(
            _tacmap_penetration_from_distances(
                surface_dist,
                object_dist,
                surface_valid & object_valid,
            )
        )
        valid_masks.append(surface_valid)

    if not depths:
        return None
    return torch.stack(depths, dim=1), torch.stack(valid_masks, dim=1)


def hydroshear_rl_obs(
    env: ManagerBasedRLEnv,
    attr_name: str = "_rl_hydroshear_displacement_m",
    tacmap_sensor_names: list[str] | tuple[str, ...] = (),
    tacmap_surface_sensor_names: list[str] | tuple[str, ...] = (),
    tacmap_rows: int = 240,
    tacmap_cols: int = 240,
    render_rows: int = RL_HYDROSHEAR_RENDER_ROWS,
    render_cols: int = RL_HYDROSHEAR_RENDER_COLS,
    marker_rows: int = RL_HYDROSHEAR_MARKER_ROWS,
    marker_cols: int = RL_HYDROSHEAR_MARKER_COLS,
    marker_margin_x: float = RL_HYDROSHEAR_MARKER_MARGIN_X,
    marker_margin_y: float = RL_HYDROSHEAR_MARKER_MARGIN_Y,
    finger_count: int = len(RL_FINGER_ORDER),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_sample_mode: str = RL_HYDROSHEAR_OBJECT_SAMPLE_MODE,
    object_sample_count: int = RL_HYDROSHEAR_OBJECT_SAMPLE_COUNT,
    object_sample_seed: int = RL_HYDROSHEAR_OBJECT_SAMPLE_SEED,
    object_poisson_radius: float = RL_HYDROSHEAR_POISSON_RADIUS,
    object_poisson_initial_count: int = RL_HYDROSHEAR_POISSON_INITIAL_COUNT,
    object_sample_reference_count: int = RL_HYDROSHEAR_OBJECT_SAMPLE_REFERENCE_COUNT,
    object_sample_roi_count: int = RL_HYDROSHEAR_OBJECT_SAMPLE_ROI_COUNT,
    use_object_surface_samples: bool = True,
    cache_debug_output: bool = False,
    render_debug_marker_images: bool = True,
    debug_sensor_index: int | None = None,
    cache_marker_depth_output: bool = False,
    marker_depth_output_attr_name: str = "_rl_hydroshear_marker_depth_m",
    marker_depth_valid_attr_name: str = "_rl_hydroshear_marker_depth_valid",
    algorithm: str = "curved",
    adapter_namespace: str = "hydroshear",
    output_attr_name: str = "_rl_hydroshear_output",
    marker_layout_path: str | Path | None = None,
    marker_finger_link_names: list[str] | tuple[str, ...] = (),
    marker_ray_sensor_names: list[str] | tuple[str, ...] = (),
    marker_ray_surface_sensor_names: list[str] | tuple[str, ...] = (),
    dilation_source_depth_m: torch.Tensor | None = None,
    dilation_source_roi_norm: torch.Tensor | None = None,
    dilation_source_active: torch.Tensor | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Flatten HydroShear marker displacements as a GPU RL observation."""

    algorithm = str(algorithm).strip().lower()
    if algorithm not in ("curved", "original"):
        raise ValueError(f"Unsupported RL HydroShear algorithm: {algorithm!r}")
    namespace = "".join(char if char.isalnum() else "_" for char in str(adapter_namespace)).strip("_")
    namespace = namespace or "hydroshear"
    adapter_attr_name = f"_brainco_rl_{namespace}_adapter"
    adapter_key_attr_name = f"{adapter_attr_name}_key"

    marker_count = max(0, int(marker_rows)) * max(0, int(marker_cols))
    dim = max(0, int(finger_count)) * marker_count * 3
    if dim == 0:
        return _zeros(env, 0)
    if bool(cache_debug_output):
        setattr(env, output_attr_name, None)
    if bool(cache_marker_depth_output):
        setattr(env, str(marker_depth_output_attr_name), None)
        setattr(env, str(marker_depth_valid_attr_name), None)

    depth = getattr(env, "_rl_tacmap_penetration_m", None)
    surface_points = getattr(env, "_rl_tacmap_surface_points_w", None)
    surface_valid = getattr(env, "_rl_tacmap_surface_valid", None)
    object_points = getattr(env, "_rl_tacmap_object_points_w", None)
    object_valid = getattr(env, "_rl_tacmap_object_valid", None)
    surface_raw = getattr(env, "_rl_tacmap_surface_raw_m", None)
    surface_normals = getattr(env, "_rl_tacmap_surface_normals_w", None)
    ray_directions = getattr(env, "_rl_tacmap_ray_directions_w", None)

    current_step = int(getattr(env, "common_step_counter", -1))
    cache_step = getattr(env, "_rl_tacmap_cache_step", None)
    if (
        tacmap_sensor_names
        and tacmap_surface_sensor_names
        and (
            not isinstance(depth, torch.Tensor)
            or cache_step != current_step
            or not isinstance(surface_points, torch.Tensor)
            or not isinstance(surface_valid, torch.Tensor)
            or not isinstance(object_points, torch.Tensor)
            or not isinstance(object_valid, torch.Tensor)
            or not isinstance(ray_directions, torch.Tensor)
        )
    ):
        tacmap_rl_obs(
            env,
            tacmap_sensor_names=tacmap_sensor_names,
            tacmap_surface_sensor_names=tacmap_surface_sensor_names,
            tacmap_rows=tacmap_rows,
            tacmap_cols=tacmap_cols,
            cache_aux_fields=True,
        )
        depth = getattr(env, "_rl_tacmap_penetration_m", None)
        surface_points = getattr(env, "_rl_tacmap_surface_points_w", None)
        surface_valid = getattr(env, "_rl_tacmap_surface_valid", None)
        object_points = getattr(env, "_rl_tacmap_object_points_w", None)
        object_valid = getattr(env, "_rl_tacmap_object_valid", None)
        surface_raw = getattr(env, "_rl_tacmap_surface_raw_m", None)
        surface_normals = getattr(env, "_rl_tacmap_surface_normals_w", None)
        ray_directions = getattr(env, "_rl_tacmap_ray_directions_w", None)

    if (
        isinstance(depth, torch.Tensor)
        and isinstance(surface_points, torch.Tensor)
        and isinstance(surface_valid, torch.Tensor)
        and isinstance(object_points, torch.Tensor)
        and isinstance(object_valid, torch.Tensor)
        and isinstance(ray_directions, torch.Tensor)
        and depth.ndim == 4
        and surface_points.ndim == 5
        and object_points.ndim == 5
        and ray_directions.ndim == 3
        and depth.shape[0] == env.num_envs
        and ray_directions.shape[0] == env.num_envs
        and ray_directions.shape[-1] == 3
        and depth.shape[1] > 0
    ):
        try:
            _ensure_integrate_import_path()
            from integrate.curved_hydroshear_adapter import RevoCurvedHydroShearAdapter, RevoCurvedHydroShearCfg
        except Exception as exc:
            _warn_once(
                env,
                "hydroshear_adapter_import_failed",
                f"[WARN] RL HydroShear adapter import failed ({exc}); using zeros.",
            )
            return _zeros(env, dim)

        depth = _finite_tensor(depth, env)
        surface_points = _finite_tensor(surface_points, env)
        object_points = _finite_tensor(object_points, env)
        surface_valid = surface_valid.to(device=env.device, dtype=torch.bool)
        object_valid = object_valid.to(device=env.device, dtype=torch.bool)
        surface_raw = _finite_tensor(surface_raw, env) if isinstance(surface_raw, torch.Tensor) else depth
        surface_normals = (
            _finite_tensor(surface_normals, env)
            if isinstance(surface_normals, torch.Tensor)
            else torch.zeros_like(surface_points)
        )
        ray_directions = _finite_tensor(ray_directions, env)

        sensor_count = min(int(depth.shape[1]), int(ray_directions.shape[1]), int(finger_count))
        rows = int(depth.shape[2])
        cols = int(depth.shape[3])
        calibrated_markers = (
            _hydroshear_calibrated_marker_world(
                env,
                marker_layout_path=marker_layout_path,
                finger_link_names=marker_finger_link_names,
                sensor_count=sensor_count,
                robot_cfg=robot_cfg,
            )
            if marker_layout_path is not None
            else None
        )
        marker_uv = None if calibrated_markers is None else calibrated_markers[0]
        if marker_uv is not None and int(marker_uv.shape[0]) != marker_count:
            raise ValueError(
                f"Configured HydroShear marker grid has {marker_count} slots, "
                f"but calibrated layout contains {int(marker_uv.shape[0])} markers"
            )
        curved_reference_state = None
        reference_candidate = getattr(env, "_rl_local_tacmap_reference_state", None)
        reference_field_names = (
            "reference_starts_l",
            "reference_directions_l",
            "reference_depth_m",
            "reference_valid",
            "coarse_xy_camera_m",
            "full_camera_bounds_m",
            "storage_row_axis_camera",
        )
        if (
            algorithm == "curved"
            and calibrated_markers is not None
            and isinstance(reference_candidate, dict)
            and all(reference_candidate.get(name) is not None for name in reference_field_names)
            and int(reference_candidate.get("coarse_rows", -1)) == rows
            and int(reference_candidate.get("coarse_cols", -1)) == cols
            and tuple(reference_candidate.get("link_names", ()))[:sensor_count]
            == tuple(str(name) for name in marker_finger_link_names[:sensor_count])
        ):
            curved_reference_state = reference_candidate
        marker_ray_measurements = (
            _hydroshear_marker_ray_measurements(
                env,
                marker_sensor_names=marker_ray_sensor_names,
                marker_surface_sensor_names=marker_ray_surface_sensor_names,
                marker_rows=marker_rows,
                marker_cols=marker_cols,
            )
            if algorithm == "curved" and calibrated_markers is not None
            else None
        )
        marker_ray_depth = None
        marker_ray_valid = None
        if marker_ray_measurements is not None:
            marker_ray_depth_grid, marker_ray_valid_grid = marker_ray_measurements
            if int(marker_ray_depth_grid.shape[1]) < sensor_count:
                raise ValueError(
                    f"Expected {sensor_count} independent marker ray sensors, "
                    f"got {int(marker_ray_depth_grid.shape[1])}"
                )
            marker_ray_depth = marker_ray_depth_grid[:, :sensor_count].reshape(
                env.num_envs * sensor_count,
                marker_count,
            )
            marker_ray_valid = marker_ray_valid_grid[:, :sensor_count].reshape(
                env.num_envs * sensor_count,
                marker_count,
            )
            if bool(cache_marker_depth_output):
                calibrated_valid_grid = calibrated_markers[3].reshape(
                    env.num_envs,
                    sensor_count,
                    marker_rows,
                    marker_cols,
                )
                setattr(
                    env,
                    str(marker_depth_output_attr_name),
                    marker_ray_depth_grid[:, :sensor_count].detach(),
                )
                setattr(
                    env,
                    str(marker_depth_valid_attr_name),
                    (marker_ray_valid_grid[:, :sensor_count] & calibrated_valid_grid).detach(),
                )
        object_samples_l = None
        object_sample_key = ("grid_fast_path",)
        if bool(cache_debug_output) or bool(use_object_surface_samples):
            object_samples_l, object_sample_key = _hydroshear_object_surface_samples_l(
                env,
                object_cfg,
                sample_count=int(object_sample_count),
                sample_mode=str(object_sample_mode),
                poisson_radius=float(object_poisson_radius),
                poisson_initial_count=int(object_poisson_initial_count),
                sample_seed=int(object_sample_seed),
            )
        object_sample_count_actual = 0 if object_samples_l is None else int(object_samples_l.shape[0])
        cfg_key = (
            algorithm,
            rows,
            cols,
            int(render_rows),
            int(render_cols),
            int(marker_rows),
            int(marker_cols),
            float(marker_margin_x),
            float(marker_margin_y),
            None if marker_layout_path is None else str(Path(marker_layout_path).resolve()),
            str(env.device),
            object_sample_key,
            object_sample_count_actual,
            int(object_sample_reference_count),
            int(object_sample_roi_count),
            bool(use_object_surface_samples),
            None if curved_reference_state is None else curved_reference_state.get("key"),
        )
        adapter = getattr(env, adapter_attr_name, None)
        if adapter is None or getattr(env, adapter_key_attr_name, None) != cfg_key:
            adapter = RevoCurvedHydroShearAdapter(
                RevoCurvedHydroShearCfg(
                    width=max(1, int(render_cols)),
                    height=max(1, int(render_rows)),
                    marker_rows=int(marker_rows),
                    marker_cols=int(marker_cols),
                    marker_margin_x=float(marker_margin_x),
                    marker_margin_y=float(marker_margin_y),
                    marker_uv=marker_uv,
                    object_sample_reference_count=int(object_sample_reference_count),
                    object_sample_roi_count=int(object_sample_roi_count),
                    object_sample_points_l=object_samples_l,
                    shear_reference_starts_l=(
                        None
                        if curved_reference_state is None
                        else curved_reference_state["reference_starts_l"][:sensor_count]
                    ),
                    shear_reference_directions_l=(
                        None
                        if curved_reference_state is None
                        else curved_reference_state["reference_directions_l"][:sensor_count]
                    ),
                    shear_reference_depth_m=(
                        None
                        if curved_reference_state is None
                        else curved_reference_state["reference_depth_m"][:sensor_count]
                    ),
                    shear_reference_valid=(
                        None
                        if curved_reference_state is None
                        else curved_reference_state["reference_valid"][:sensor_count]
                    ),
                    shear_coarse_xy_camera_m=(
                        None
                        if curved_reference_state is None
                        else curved_reference_state["coarse_xy_camera_m"][:sensor_count]
                    ),
                    shear_reference_bounds_camera_m=(
                        None
                        if curved_reference_state is None
                        else curved_reference_state["full_camera_bounds_m"][:sensor_count]
                    ),
                    shear_reference_row_axis_camera=(
                        None
                        if curved_reference_state is None
                        else curved_reference_state["storage_row_axis_camera"][:sensor_count]
                    ),
                    device=str(env.device),
                )
            )
            setattr(env, adapter_attr_name, adapter)
            setattr(env, adapter_key_attr_name, cfg_key)

        flat_shape = (env.num_envs * sensor_count, rows, cols)
        flat_vec_shape = (env.num_envs * sensor_count, rows, cols, 3)
        episode_length_buf = getattr(env, "episode_length_buf", None)
        if isinstance(episode_length_buf, torch.Tensor):
            reset_env_ids = torch.nonzero(episode_length_buf[: env.num_envs] == 0, as_tuple=False).flatten()
            if reset_env_ids.numel() >= env.num_envs:
                adapter.reset()
            elif reset_env_ids.numel() > 0:
                reset_one = getattr(adapter, "_reset_hydrosoft_state", None)
                batch_state = getattr(adapter, "_batch_state_valid", None)
                if isinstance(batch_state, torch.Tensor):
                    state_count = int(batch_state.shape[0])
                else:
                    state_count = len(getattr(adapter, "_prev_sdf", ()))
                if callable(reset_one):
                    for env_id in reset_env_ids.detach().cpu().tolist():
                        base_slot = int(env_id) * sensor_count
                        for finger_slot in range(sensor_count):
                            slot = base_slot + finger_slot
                            if 0 <= slot < state_count:
                                reset_one(slot)
                else:
                    adapter.reset()

        object_pose_wxyz = None
        try:
            obj: RigidObject = env.scene[object_cfg.name]
            object_pose = torch.cat((obj.data.root_pos_w, obj.data.root_quat_w), dim=-1)
            object_pose_wxyz = object_pose[:, None, :].expand(env.num_envs, sensor_count, 7).reshape(-1, 7)
        except Exception as exc:
            _warn_once(
                env,
                "hydroshear_object_pose_failed",
                f"[WARN] RL HydroShear object pose unavailable ({exc}); using TacMap ray-hit fallback.",
            )
        displacement_method = "step_original_displacement_only" if algorithm == "original" else "step_displacement_only"
        displacement_step = getattr(adapter, displacement_method, None)
        if callable(displacement_step):
            displacement_kwargs = {
                "object_pose_wxyz": object_pose_wxyz,
                "surface_raw_m": surface_raw[:, :sensor_count].reshape(flat_shape),
                "surface_normals_w": surface_normals[:, :sensor_count].reshape(flat_vec_shape),
                "ray_directions_w": ray_directions[:, :sensor_count].reshape(-1, 3),
            }
            dilation_source_count = sum(
                value is not None
                for value in (dilation_source_depth_m, dilation_source_roi_norm, dilation_source_active)
            )
            if dilation_source_count not in (0, 3):
                raise ValueError(
                    "HydroShear adaptive dilation depth, ROI, and active mask must be provided together"
                )
            if dilation_source_count and algorithm == "curved":
                assert dilation_source_depth_m is not None
                assert dilation_source_roi_norm is not None
                assert dilation_source_active is not None
                expected_prefix = (int(env.num_envs), int(sensor_count))
                if tuple(dilation_source_depth_m.shape[:2]) != expected_prefix or dilation_source_depth_m.ndim != 4:
                    raise ValueError(
                        "Expected adaptive dilation depth [num_envs,sensors,H,W], got "
                        f"{tuple(dilation_source_depth_m.shape)}"
                    )
                if tuple(dilation_source_roi_norm.shape) != (*expected_prefix, 4):
                    raise ValueError(
                        "Expected adaptive dilation ROI [num_envs,sensors,4], got "
                        f"{tuple(dilation_source_roi_norm.shape)}"
                    )
                if tuple(dilation_source_active.shape) != expected_prefix:
                    raise ValueError(
                        "Expected adaptive dilation active mask [num_envs,sensors], got "
                        f"{tuple(dilation_source_active.shape)}"
                    )
                displacement_kwargs.update(
                    dilation_source_depth_m=dilation_source_depth_m.to(
                        device=env.device,
                        dtype=torch.float32,
                    ).reshape(env.num_envs * sensor_count, *dilation_source_depth_m.shape[2:]),
                    dilation_source_roi_norm=dilation_source_roi_norm.to(
                        device=env.device,
                        dtype=torch.float32,
                    ).reshape(env.num_envs * sensor_count, 4),
                    dilation_source_active=dilation_source_active.to(
                        device=env.device,
                        dtype=torch.bool,
                    ).reshape(env.num_envs * sensor_count),
                )
            if calibrated_markers is not None and algorithm == "curved":
                displacement_kwargs.update(
                    marker_points_w=calibrated_markers[1],
                    marker_normals_w=calibrated_markers[2],
                    marker_valid=calibrated_markers[3],
                    surface_frame_pose_wxyz=calibrated_markers[4],
                )
                if marker_ray_depth is not None and marker_ray_valid is not None:
                    displacement_kwargs.update(
                        marker_depth_m=marker_ray_depth,
                        marker_depth_valid=marker_ray_valid,
                    )
            disp = displacement_step(
                depth[:, :sensor_count].reshape(flat_shape),
                surface_points[:, :sensor_count].reshape(flat_vec_shape),
                surface_valid[:, :sensor_count].reshape(flat_shape),
                object_points[:, :sensor_count].reshape(flat_vec_shape),
                object_valid[:, :sensor_count].reshape(flat_shape),
                **displacement_kwargs,
            )
        else:
            _warn_once(
                env,
                f"hydroshear_{algorithm}_gpu_step_unavailable",
                f"[WARN] HydroShear {algorithm} GPU step is unavailable; using zeros instead of another algorithm.",
            )
            disp = torch.zeros(
                (env.num_envs * sensor_count, marker_count, 3),
                device=env.device,
                dtype=torch.float32,
            )
        disp = disp.to(device=env.device, dtype=torch.float32)
        if disp.ndim == 3 and disp.shape[0] == env.num_envs * sensor_count:
            disp = disp.reshape(env.num_envs, sensor_count, disp.shape[1], 3)
        else:
            disp = disp.reshape(env.num_envs, sensor_count, -1, 3)
        disp = torch.where(torch.isfinite(disp), disp, torch.zeros_like(disp))

        if bool(cache_debug_output):
            render_output = getattr(adapter, "render_displacement_output", None)
            if callable(render_output):
                debug_batch_index = None
                if debug_sensor_index is not None and sensor_count > 0:
                    debug_batch_index = max(0, min(int(debug_sensor_index), sensor_count - 1))
                setattr(
                    env,
                    output_attr_name,
                    render_output(
                        disp.reshape(env.num_envs * sensor_count, -1, 3),
                        depth[:, :sensor_count].reshape(flat_shape),
                        surface_points[:, :sensor_count].reshape(flat_vec_shape),
                        surface_valid[:, :sensor_count].reshape(flat_shape),
                        marker_points_w=None if calibrated_markers is None else calibrated_markers[1],
                        marker_normals_w=None if calibrated_markers is None else calibrated_markers[2],
                        marker_valid=None if calibrated_markers is None else calibrated_markers[3],
                        render_marker_images=bool(render_debug_marker_images),
                        debug_batch_index=debug_batch_index,
                    ),
                )
            else:
                _warn_once(
                    env,
                    "hydroshear_training_render_unavailable",
                    "[WARN] HydroShear training displacement renderer is unavailable; marker UI is disabled.",
                )

        full = torch.zeros((env.num_envs, int(finger_count), marker_count, 3), device=env.device, dtype=torch.float32)
        copy_markers = min(marker_count, int(disp.shape[2]))
        full[:, :sensor_count, :copy_markers, :] = disp[:, :sensor_count, :copy_markers, :]
        setattr(env, attr_name, full)
        return full.reshape(env.num_envs, -1)

    value = getattr(env, attr_name, None)
    if value is None and getattr(env, "unwrapped", None) is not None:
        value = getattr(env.unwrapped, attr_name, None)
    if callable(value):
        value = value()
    if value is None:
        _warn_once(
            env,
            "missing_hydroshear_rl_displacement",
            "[WARN] RL HydroShear displacement is not configured in this ManagerBased env; using zeros.",
        )
        return _zeros(env, dim)

    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=env.device, dtype=torch.float32)
    tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))

    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        if tensor.shape[0] == env.num_envs:
            tensor = tensor[:, :finger_count, :marker_count, :]
            return _finite_flat(tensor, env)
        tensor = tensor.reshape(tensor.shape[0], -1, 3)

    if tensor.ndim == 3 and tensor.shape[-1] == 3:
        source = tensor[:finger_count, :marker_count, :].reshape(1, -1).expand(env.num_envs, -1)
        out = _zeros(env, dim)
        copy_dim = min(out.shape[1], source.shape[1])
        out[:, :copy_dim] = source[:, :copy_dim]
        return _finite_flat(out, env)

    flat = tensor.reshape(1, -1).expand(env.num_envs, -1)
    out = _zeros(env, dim)
    copy_dim = min(out.shape[1], flat.shape[1])
    out[:, :copy_dim] = flat[:, :copy_dim]
    return _finite_flat(out, env)


def hydroshear_baseline_rl_obs(
    env: ManagerBasedRLEnv,
    tacmap_sensor_names: list[str] | tuple[str, ...] = (),
    tacmap_surface_sensor_names: list[str] | tuple[str, ...] = (),
    tacmap_rows: int = 32,
    tacmap_cols: int = 24,
    render_rows: int = RL_HYDROSHEAR_RENDER_ROWS,
    render_cols: int = RL_HYDROSHEAR_RENDER_COLS,
    marker_rows: int = RL_HYDROSHEAR_MARKER_ROWS,
    marker_cols: int = RL_HYDROSHEAR_MARKER_COLS,
    marker_margin_x: float = RL_HYDROSHEAR_MARKER_MARGIN_X,
    marker_margin_y: float = RL_HYDROSHEAR_MARKER_MARGIN_Y,
    finger_count: int = len(RL_FINGER_ORDER),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_sample_mode: str = RL_HYDROSHEAR_OBJECT_SAMPLE_MODE,
    object_sample_count: int = RL_HYDROSHEAR_OBJECT_SAMPLE_COUNT,
    object_sample_seed: int = RL_HYDROSHEAR_OBJECT_SAMPLE_SEED,
    object_poisson_radius: float = RL_HYDROSHEAR_POISSON_RADIUS,
    object_poisson_initial_count: int = RL_HYDROSHEAR_POISSON_INITIAL_COUNT,
    object_sample_reference_count: int = RL_HYDROSHEAR_OBJECT_SAMPLE_REFERENCE_COUNT,
    cache_debug_output: bool = False,
) -> torch.Tensor:
    """Original Euclidean HydroShear baseline, kept separate from the OURS observation."""

    return hydroshear_rl_obs(
        env,
        attr_name="_rl_hydroshear_baseline_displacement_m",
        tacmap_sensor_names=tacmap_sensor_names,
        tacmap_surface_sensor_names=tacmap_surface_sensor_names,
        tacmap_rows=tacmap_rows,
        tacmap_cols=tacmap_cols,
        render_rows=render_rows,
        render_cols=render_cols,
        marker_rows=marker_rows,
        marker_cols=marker_cols,
        marker_margin_x=marker_margin_x,
        marker_margin_y=marker_margin_y,
        finger_count=finger_count,
        object_cfg=object_cfg,
        object_sample_mode=object_sample_mode,
        object_sample_count=object_sample_count,
        object_sample_seed=object_sample_seed,
        object_poisson_radius=object_poisson_radius,
        object_poisson_initial_count=object_poisson_initial_count,
        object_sample_reference_count=object_sample_reference_count,
        object_sample_roi_count=max(1, int(object_poisson_initial_count)),
        use_object_surface_samples=True,
        cache_debug_output=cache_debug_output,
        algorithm="original",
        adapter_namespace="hydroshear_baseline",
        output_attr_name="_rl_hydroshear_baseline_output",
    )


def ours_rl_obs(
    env: ManagerBasedRLEnv,
    pressure_sensor_names: list[str] | tuple[str, ...],
    tacmap_sensor_names: list[str] | tuple[str, ...],
    tacmap_surface_sensor_names: list[str] | tuple[str, ...],
    pressure_rows: int = 4,
    pressure_cols: int = 8,
    pressure_taxel_counts: list[int] | tuple[int, ...] | None = None,
    pressure_diffusion_enabled: bool = False,
    pressure_diffusion_sigma_m: float = 0.003,
    pressure_diffusion_blend: float = 0.7,
    pressure_diffusion_radius_sigma: float = 3.0,
    pressure_diffusion_normal_power: float = 1.0,
    tacmap_rows: int = 32,
    tacmap_cols: int = 24,
    tacmap_contact_shell_m: float = 0.0,
    local_tacmap_enabled: bool = False,
    local_tacmap_reference_rows: int = 240,
    local_tacmap_reference_cols: int = 320,
    local_tacmap_rows: int = 25,
    local_tacmap_cols: int = 40,
    local_tacmap_contact_threshold_m: float = 0.02e-3,
    local_tacmap_roi_margin_m: float = 1.0e-3,
    local_tacmap_penetration_deadband_m: float = 1.0e-6,
    local_tacmap_max_distance_m: float = 0.05,
    local_tacmap_cache_debug_output: bool = False,
    hydroshear_marker_rows: int = RL_HYDROSHEAR_MARKER_ROWS,
    hydroshear_marker_cols: int = RL_HYDROSHEAR_MARKER_COLS,
    hydroshear_render_rows: int = RL_HYDROSHEAR_RENDER_ROWS,
    hydroshear_render_cols: int = RL_HYDROSHEAR_RENDER_COLS,
    hydroshear_marker_margin_x: float = RL_HYDROSHEAR_MARKER_MARGIN_X,
    hydroshear_marker_margin_y: float = RL_HYDROSHEAR_MARKER_MARGIN_Y,
    hydroshear_object_sample_mode: str = RL_HYDROSHEAR_OBJECT_SAMPLE_MODE,
    hydroshear_object_sample_count: int = RL_HYDROSHEAR_OBJECT_SAMPLE_COUNT,
    hydroshear_object_sample_seed: int = RL_HYDROSHEAR_OBJECT_SAMPLE_SEED,
    hydroshear_object_poisson_radius: float = RL_HYDROSHEAR_POISSON_RADIUS,
    hydroshear_object_poisson_initial_count: int = RL_HYDROSHEAR_POISSON_INITIAL_COUNT,
    hydroshear_object_sample_reference_count: int = RL_HYDROSHEAR_OBJECT_SAMPLE_REFERENCE_COUNT,
    hydroshear_object_sample_roi_count: int = RL_HYDROSHEAR_OBJECT_SAMPLE_ROI_COUNT,
    hydroshear_use_object_surface_samples: bool = True,
    hydroshear_cache_debug_output: bool = False,
    hydroshear_render_debug_marker_images: bool = True,
    hydroshear_debug_sensor_index: int | None = None,
    hydroshear_cache_marker_depth_output: bool = False,
    hydroshear_marker_layout_path: str | Path | None = None,
    hydroshear_marker_finger_link_names: list[str] | tuple[str, ...] = (),
    hydroshear_marker_sensor_names: list[str] | tuple[str, ...] = (),
    hydroshear_marker_surface_sensor_names: list[str] | tuple[str, ...] = (),
    cache_pressure_output: bool = False,
    pressure_output_attr_name: str = "_rl_pressure_observation",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Concatenate the current BrainCo tactile stack as one ``ours`` observation."""

    pressure = warpsdf_pressure_obs(
        env,
        pressure_sensor_names=pressure_sensor_names,
        object_cfg=object_cfg,
        pressure_rows=pressure_rows,
        pressure_cols=pressure_cols,
        pressure_taxel_counts=pressure_taxel_counts,
        pressure_diffusion_enabled=pressure_diffusion_enabled,
        pressure_diffusion_sigma_m=pressure_diffusion_sigma_m,
        pressure_diffusion_blend=pressure_diffusion_blend,
        pressure_diffusion_radius_sigma=pressure_diffusion_radius_sigma,
        pressure_diffusion_normal_power=pressure_diffusion_normal_power,
    )
    if bool(cache_pressure_output):
        # Debug visualizers may retain a detached view of the exact pressure
        # tensor concatenated into the policy observation.  This is opt-in so
        # ordinary RL training keeps the original observation path unchanged.
        setattr(env, str(pressure_output_attr_name), pressure.detach())
    coarse_tacmap = tacmap_rl_obs(
        env,
        tacmap_sensor_names=tacmap_sensor_names,
        tacmap_surface_sensor_names=tacmap_surface_sensor_names,
        tacmap_rows=tacmap_rows,
        tacmap_cols=tacmap_cols,
        cache_aux_fields=True,
        contact_shell_m=tacmap_contact_shell_m,
    )
    local_depth = None
    local_roi = None
    local_active = None
    if bool(local_tacmap_enabled):
        if hydroshear_marker_layout_path is None:
            raise ValueError("Adaptive local TacMap requires the calibrated marker/camera layout path")
        local_depth, local_roi, local_active = adaptive_local_tacmap_rl_obs(
            env,
            tacmap_sensor_names=tacmap_sensor_names,
            tacmap_surface_sensor_names=tacmap_surface_sensor_names,
            finger_link_names=hydroshear_marker_finger_link_names,
            marker_layout_path=hydroshear_marker_layout_path,
            coarse_rows=tacmap_rows,
            coarse_cols=tacmap_cols,
            reference_rows=local_tacmap_reference_rows,
            reference_cols=local_tacmap_reference_cols,
            local_rows=local_tacmap_rows,
            local_cols=local_tacmap_cols,
            contact_threshold_m=local_tacmap_contact_threshold_m,
            roi_margin_m=local_tacmap_roi_margin_m,
            penetration_deadband_m=local_tacmap_penetration_deadband_m,
            max_distance_m=local_tacmap_max_distance_m,
            cache_debug_output=local_tacmap_cache_debug_output,
        )
        tacmap_policy = torch.cat(
            (
                local_depth.reshape(env.num_envs, -1),
                local_roi.reshape(env.num_envs, -1),
                local_active.to(dtype=torch.float32).reshape(env.num_envs, -1),
            ),
            dim=1,
        )
    else:
        tacmap_policy = coarse_tacmap
    hydroshear = hydroshear_rl_obs(
        env,
        tacmap_sensor_names=tacmap_sensor_names,
        tacmap_surface_sensor_names=tacmap_surface_sensor_names,
        tacmap_rows=tacmap_rows,
        tacmap_cols=tacmap_cols,
        render_rows=hydroshear_render_rows,
        render_cols=hydroshear_render_cols,
        marker_rows=hydroshear_marker_rows,
        marker_cols=hydroshear_marker_cols,
        marker_margin_x=hydroshear_marker_margin_x,
        marker_margin_y=hydroshear_marker_margin_y,
        object_cfg=object_cfg,
        object_sample_mode=hydroshear_object_sample_mode,
        object_sample_count=hydroshear_object_sample_count,
        object_sample_seed=hydroshear_object_sample_seed,
        object_poisson_radius=hydroshear_object_poisson_radius,
        object_poisson_initial_count=hydroshear_object_poisson_initial_count,
        object_sample_reference_count=hydroshear_object_sample_reference_count,
        object_sample_roi_count=hydroshear_object_sample_roi_count,
        use_object_surface_samples=hydroshear_use_object_surface_samples,
        cache_debug_output=hydroshear_cache_debug_output,
        render_debug_marker_images=hydroshear_render_debug_marker_images,
        debug_sensor_index=hydroshear_debug_sensor_index,
        cache_marker_depth_output=hydroshear_cache_marker_depth_output,
        marker_layout_path=hydroshear_marker_layout_path,
        marker_finger_link_names=hydroshear_marker_finger_link_names,
        marker_ray_sensor_names=hydroshear_marker_sensor_names,
        marker_ray_surface_sensor_names=hydroshear_marker_surface_sensor_names,
        dilation_source_depth_m=local_depth,
        dilation_source_roi_norm=local_roi,
        dilation_source_active=local_active,
        robot_cfg=robot_cfg,
    )
    return torch.cat((pressure, tacmap_policy, hydroshear), dim=1)


def _ours_tacmap_aux_cache_is_current(
    env: ManagerBasedRLEnv,
    component_cfg: dict,
) -> bool:
    """Return whether this step already has all TacMap fields needed by OURS/HydroShear."""

    object_names = tuple(str(name) for name in component_cfg.get("tacmap_sensor_names", ()))
    surface_names = tuple(str(name) for name in component_cfg.get("tacmap_surface_sensor_names", ()))
    sensor_count = max(len(object_names), len(surface_names))
    if sensor_count <= 0:
        return False
    rows = max(0, int(component_cfg.get("tacmap_rows", 32)))
    cols = max(0, int(component_cfg.get("tacmap_cols", 24)))
    current_step = int(getattr(env, "common_step_counter", -1))
    if getattr(env, "_rl_tacmap_cache_step", None) != current_step:
        return False
    reset_epoch = int(getattr(env, "_rl_ours_tactile_reset_epoch", 0))
    if getattr(env, "_rl_ours_tacmap_cache_epoch", None) != reset_epoch:
        return False

    expected_image = (env.num_envs, sensor_count, rows, cols)
    expected_points = (*expected_image, 3)
    expected_shapes = {
        "_rl_tacmap_penetration_m": expected_image,
        "_rl_tacmap_surface_points_w": expected_points,
        "_rl_tacmap_surface_normals_w": expected_points,
        "_rl_tacmap_ray_directions_w": (env.num_envs, sensor_count, 3),
        "_rl_tacmap_surface_valid": expected_image,
        "_rl_tacmap_object_points_w": expected_points,
        "_rl_tacmap_object_valid": expected_image,
        "_rl_tacmap_surface_raw_m": expected_image,
    }
    return all(
        isinstance(value := getattr(env, attr_name, None), torch.Tensor)
        and tuple(value.shape) == expected_shape
        for attr_name, expected_shape in expected_shapes.items()
    )


def _ours_local_tacmap_cache_is_current(
    env: ManagerBasedRLEnv,
    component_cfg: dict,
) -> bool:
    """Return whether this step already has the adaptive local TacMap tensors."""

    current_step = int(getattr(env, "common_step_counter", -1))
    if getattr(env, "_rl_local_tacmap_cache_step", None) != current_step:
        return False
    reset_epoch = int(getattr(env, "_rl_ours_tactile_reset_epoch", 0))
    if getattr(env, "_rl_ours_local_tacmap_cache_epoch", None) != reset_epoch:
        return False
    sensor_count = min(
        len(tuple(component_cfg.get("tacmap_sensor_names", ()))),
        len(tuple(component_cfg.get("tacmap_surface_sensor_names", ()))),
        len(RL_FINGER_ORDER),
    )
    local_rows = max(1, int(component_cfg.get("local_tacmap_rows", 25)))
    local_cols = max(1, int(component_cfg.get("local_tacmap_cols", 40)))
    expected_shapes = {
        "_rl_local_tacmap_depth_m": (env.num_envs, sensor_count, local_rows, local_cols),
        "_rl_local_tacmap_roi_norm": (env.num_envs, sensor_count, 4),
        "_rl_local_tacmap_active": (env.num_envs, sensor_count),
        "_rl_local_tacmap_pixel_indices": (env.num_envs, sensor_count, local_rows, local_cols),
    }
    return all(
        isinstance(value := getattr(env, attr_name, None), torch.Tensor)
        and tuple(value.shape) == expected_shape
        for attr_name, expected_shape in expected_shapes.items()
    )


def _ours_tacmap_policy_components(
    env: ManagerBasedRLEnv,
    component_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Compute/reuse only the TacMap dependency shared by policy depth and HydroShear."""

    cfg = dict(component_cfg)
    reset_epoch = int(getattr(env, "_rl_ours_tactile_reset_epoch", 0))
    if _ours_tacmap_aux_cache_is_current(env, cfg):
        coarse_tacmap = env._rl_tacmap_penetration_m.reshape(env.num_envs, -1)
    else:
        coarse_tacmap = tacmap_rl_obs(
            env,
            tacmap_sensor_names=cfg.get("tacmap_sensor_names", ()),
            tacmap_surface_sensor_names=cfg.get("tacmap_surface_sensor_names", ()),
            tacmap_rows=cfg.get("tacmap_rows", 32),
            tacmap_cols=cfg.get("tacmap_cols", 24),
            cache_aux_fields=True,
            contact_shell_m=cfg.get("tacmap_contact_shell_m", 0.0),
        )
        env._rl_ours_tacmap_cache_epoch = reset_epoch

    if not bool(cfg.get("local_tacmap_enabled", False)):
        return coarse_tacmap, None, None, None

    marker_layout_path = cfg.get("hydroshear_marker_layout_path")
    if marker_layout_path is None:
        raise ValueError("Adaptive local TacMap requires the calibrated marker/camera layout path")
    if _ours_local_tacmap_cache_is_current(env, cfg):
        local_depth = env._rl_local_tacmap_depth_m
        local_roi = env._rl_local_tacmap_roi_norm
        local_active = env._rl_local_tacmap_active
    else:
        if getattr(env, "_rl_ours_local_tacmap_cache_epoch", None) != reset_epoch:
            env._rl_local_tacmap_cache_step = None
        local_depth, local_roi, local_active = adaptive_local_tacmap_rl_obs(
            env,
            tacmap_sensor_names=cfg.get("tacmap_sensor_names", ()),
            tacmap_surface_sensor_names=cfg.get("tacmap_surface_sensor_names", ()),
            finger_link_names=cfg.get("hydroshear_marker_finger_link_names", ()),
            marker_layout_path=marker_layout_path,
            coarse_rows=cfg.get("tacmap_rows", 32),
            coarse_cols=cfg.get("tacmap_cols", 24),
            reference_rows=cfg.get("local_tacmap_reference_rows", 240),
            reference_cols=cfg.get("local_tacmap_reference_cols", 320),
            local_rows=cfg.get("local_tacmap_rows", 25),
            local_cols=cfg.get("local_tacmap_cols", 40),
            contact_threshold_m=cfg.get("local_tacmap_contact_threshold_m", 0.02e-3),
            roi_margin_m=cfg.get("local_tacmap_roi_margin_m", 1.0e-3),
            penetration_deadband_m=cfg.get("local_tacmap_penetration_deadband_m", 1.0e-6),
            max_distance_m=cfg.get("local_tacmap_max_distance_m", 0.05),
            cache_debug_output=cfg.get("local_tacmap_cache_debug_output", False),
        )
        env._rl_ours_local_tacmap_cache_epoch = reset_epoch
    tacmap_policy = torch.cat(
        (
            local_depth.reshape(env.num_envs, -1),
            local_roi.reshape(env.num_envs, -1),
            local_active.to(dtype=torch.float32).reshape(env.num_envs, -1),
        ),
        dim=1,
    )
    return tacmap_policy, local_depth, local_roi, local_active


def _dense_local_tacmap_chunk_for_taxim(
    local_depth_m: torch.Tensor,
    pixel_indices: torch.Tensor,
    active: torch.Tensor,
    reference_valid: torch.Tensor,
    storage_row_axis: torch.Tensor,
) -> torch.Tensor:
    """Interpolate local lattices into their calibrated dense camera-image locations."""

    if local_depth_m.ndim != 3:
        raise ValueError(f"Taxim local Depth must be BxRxC, got {tuple(local_depth_m.shape)}")
    if pixel_indices.shape != local_depth_m.shape:
        raise ValueError(
            "Taxim local pixel indices must match local Depth, got "
            f"{tuple(pixel_indices.shape)} versus {tuple(local_depth_m.shape)}"
        )
    batch_size = int(local_depth_m.shape[0])
    if active.shape != (batch_size,):
        raise ValueError(f"Taxim local active flags must have shape {(batch_size,)}, got {tuple(active.shape)}")
    if reference_valid.ndim != 3 or int(reference_valid.shape[0]) != batch_size:
        raise ValueError("Taxim dense reference validity must be BxHxW")
    if storage_row_axis.shape != (batch_size,):
        raise ValueError("Taxim storage-row axes must have one entry per local Depth image")

    render_rows, render_cols = (int(value) for value in reference_valid.shape[-2:])
    dense = torch.zeros(
        (batch_size, render_rows, render_cols),
        device=local_depth_m.device,
        dtype=torch.float32,
    )
    output_rows = torch.arange(render_rows, device=local_depth_m.device, dtype=torch.float32).reshape(1, -1, 1)
    output_cols = torch.arange(render_cols, device=local_depth_m.device, dtype=torch.float32).reshape(1, 1, -1)

    for row_axis in (0, 1):
        selected = torch.nonzero(storage_row_axis == row_axis, as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        depth_patch = local_depth_m.index_select(0, selected).to(dtype=torch.float32)
        index_patch = pixel_indices.index_select(0, selected).to(device=local_depth_m.device, dtype=torch.long)
        if row_axis == 0:
            # Local Depth is always camera-Y by camera-X.  A reference stored
            # with camera X on image rows therefore needs one transpose.
            depth_patch = depth_patch.transpose(1, 2)
            index_patch = index_patch.transpose(1, 2)

        pixel_rows = torch.div(index_patch, render_cols, rounding_mode="floor")
        pixel_cols = torch.remainder(index_patch, render_cols)
        row_low = torch.amin(pixel_rows, dim=(1, 2)).to(dtype=torch.float32)
        row_high = torch.amax(pixel_rows, dim=(1, 2)).to(dtype=torch.float32)
        col_low = torch.amin(pixel_cols, dim=(1, 2)).to(dtype=torch.float32)
        col_high = torch.amax(pixel_cols, dim=(1, 2)).to(dtype=torch.float32)
        row_span = torch.clamp(row_high - row_low, min=1.0)
        col_span = torch.clamp(col_high - col_low, min=1.0)

        grid_y = 2.0 * (output_rows - row_low[:, None, None]) / row_span[:, None, None] - 1.0
        grid_x = 2.0 * (output_cols - col_low[:, None, None]) / col_span[:, None, None] - 1.0
        grid = torch.stack(
            (
                grid_x.expand(-1, render_rows, -1),
                grid_y.expand(-1, -1, render_cols),
            ),
            dim=-1,
        )
        interpolated = F.grid_sample(
            depth_patch.unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        inside = (
            (output_rows >= row_low[:, None, None])
            & (output_rows <= row_high[:, None, None])
            & (output_cols >= col_low[:, None, None])
            & (output_cols <= col_high[:, None, None])
        )
        valid = reference_valid.index_select(0, selected).to(device=local_depth_m.device, dtype=torch.bool)
        is_active = active.index_select(0, selected).to(device=local_depth_m.device, dtype=torch.bool)
        interpolated = torch.where(
            inside & valid & is_active[:, None, None],
            torch.clamp(interpolated, min=0.0),
            torch.zeros_like(interpolated),
        )
        dense.index_copy_(0, selected, interpolated)

    unknown_axes = (storage_row_axis != 0) & (storage_row_axis != 1)
    if bool(torch.any(unknown_axes)):
        values = torch.unique(storage_row_axis[unknown_axes]).detach().cpu().tolist()
        raise ValueError(f"Unsupported Taxim reference storage-row axes: {values}")
    return dense


def _ours_dense_tacmap_depth_from_local(
    env: ManagerBasedRLEnv,
    local_depth_m: torch.Tensor,
    local_active: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """Return the five full-resolution metric Depth images used by visualization and Taxim."""

    if local_depth_m.ndim != 4 or local_active.shape != local_depth_m.shape[:2]:
        raise ValueError("Dense RL Depth requires local Depth [env,finger,row,col] and matching active flags")
    num_envs, finger_count = (int(value) for value in local_depth_m.shape[:2])
    reference = getattr(env, "_rl_local_tacmap_reference_state", None)
    if not isinstance(reference, dict):
        raise RuntimeError("Dense RL Depth requires the calibrated local TacMap reference state")
    render_rows = int(reference.get("reference_rows", 0))
    render_cols = int(reference.get("reference_cols", 0))
    configured_rows = max(1, int(cfg.get("taxim_rgb_render_rows", RL_TAXIM_RGB_RENDER_ROWS)))
    configured_cols = max(1, int(cfg.get("taxim_rgb_render_cols", RL_TAXIM_RGB_RENDER_COLS)))
    if (render_rows, render_cols) != (configured_rows, configured_cols):
        raise ValueError(
            "Dense RL Depth shape must match the calibrated visualization reference: "
            f"reference={(render_rows, render_cols)}, configured={(configured_rows, configured_cols)}"
        )

    current_step = int(getattr(env, "common_step_counter", -1))
    reset_epoch = int(getattr(env, "_rl_ours_tactile_reset_epoch", 0))
    expected_shape = (num_envs, finger_count, render_rows, render_cols)
    cached = getattr(env, "_rl_ours_dense_tacmap_depth_m", None)
    if (
        getattr(env, "_rl_ours_dense_tacmap_cache_step", None) == current_step
        and getattr(env, "_rl_ours_dense_tacmap_cache_epoch", None) == reset_epoch
        and isinstance(cached, torch.Tensor)
        and tuple(cached.shape) == expected_shape
    ):
        return cached

    pixel_indices = getattr(env, "_rl_local_tacmap_pixel_indices", None)
    if not isinstance(pixel_indices, torch.Tensor) or pixel_indices.shape != local_depth_m.shape:
        raise RuntimeError(
            "Dense RL Depth requires the current adaptive TacMap pixel-index cache; "
            f"expected {tuple(local_depth_m.shape)}, got "
            f"{None if not isinstance(pixel_indices, torch.Tensor) else tuple(pixel_indices.shape)}"
        )
    reference_valid = reference.get("reference_valid")
    row_axes = tuple(int(value) for value in reference.get("storage_row_axis_camera", ()))
    if not isinstance(reference_valid, torch.Tensor) or tuple(reference_valid.shape) != (
        finger_count,
        render_rows,
        render_cols,
    ):
        raise RuntimeError("Dense RL Depth reference validity has the wrong shape")
    if len(row_axes) != finger_count:
        raise RuntimeError("Dense RL Depth reference storage axes do not match the finger count")

    flat_count = num_envs * finger_count
    flat_depth = local_depth_m.reshape(flat_count, *local_depth_m.shape[-2:])
    flat_indices = pixel_indices.reshape_as(flat_depth)
    flat_active = local_active.reshape(-1)
    reference_valid = reference_valid.to(device=local_depth_m.device, dtype=torch.bool)
    reference_axes = torch.as_tensor(row_axes, device=local_depth_m.device, dtype=torch.long)
    chunk_size = max(1, int(cfg.get("taxim_rgb_render_chunk_size", RL_TAXIM_RGB_RENDER_CHUNK_SIZE)))
    dense_flat = torch.zeros(
        (flat_count, render_rows, render_cols),
        device=local_depth_m.device,
        dtype=torch.float32,
    )
    for start in range(0, flat_count, chunk_size):
        stop = min(start + chunk_size, flat_count)
        chunk_fingers = torch.arange(start, stop, device=local_depth_m.device) % finger_count
        dense_flat[start:stop] = _dense_local_tacmap_chunk_for_taxim(
            flat_depth[start:stop],
            flat_indices[start:stop],
            flat_active[start:stop],
            reference_valid.index_select(0, chunk_fingers),
            reference_axes.index_select(0, chunk_fingers),
        )
    dense = dense_flat.reshape(expected_shape)
    env._rl_ours_dense_tacmap_depth_m = dense
    env._rl_ours_dense_tacmap_cache_step = current_step
    env._rl_ours_dense_tacmap_cache_epoch = reset_epoch
    return dense


def _ours_taxim_rgb_adapter(env: ManagerBasedRLEnv, cfg: dict):
    """Create/reuse the vendored TacEx GPU-Taxim renderer for RL observations."""

    render_rows = max(1, int(cfg.get("taxim_rgb_render_rows", RL_TAXIM_RGB_RENDER_ROWS)))
    render_cols = max(1, int(cfg.get("taxim_rgb_render_cols", RL_TAXIM_RGB_RENDER_COLS)))
    key = (
        render_rows,
        render_cols,
        float(cfg.get("taxim_rgb_depth_scale", 1.0)),
        bool(cfg.get("taxim_rgb_with_shadow", False)),
        str(cfg.get("taxim_rgb_device", env.device)),
        str(cfg.get("taxim_rgb_calib_dir", "")),
        str(cfg.get("taxim_rgb_sim_dir", "")),
    )
    adapter = getattr(env, "_rl_ours_taxim_rgb_adapter", None)
    if adapter is not None and getattr(env, "_rl_ours_taxim_rgb_adapter_key", None) == key:
        return adapter

    _ensure_integrate_import_path()
    from integrate.tacex_rgb_adapter import RevoTacExRgbAdapter, RevoTacExRgbCfg

    adapter_cfg = RevoTacExRgbCfg(
        width=render_cols,
        height=render_rows,
        depth_scale=float(cfg.get("taxim_rgb_depth_scale", 1.0)),
        with_shadow=bool(cfg.get("taxim_rgb_with_shadow", False)),
        device=str(cfg.get("taxim_rgb_device", env.device)),
    )
    calib_dir = str(cfg.get("taxim_rgb_calib_dir", "")).strip()
    sim_dir = str(cfg.get("taxim_rgb_sim_dir", "")).strip()
    if calib_dir:
        adapter_cfg.calib_dir = calib_dir
    if sim_dir:
        adapter_cfg.taxim_sim_dir = sim_dir
    adapter = RevoTacExRgbAdapter(adapter_cfg)
    env._rl_ours_taxim_rgb_adapter = adapter
    env._rl_ours_taxim_rgb_adapter_key = key
    env._rl_ours_taxim_rgb_background_cache = None
    return adapter


def _ours_taxim_rgb_background(
    env: ManagerBasedRLEnv,
    adapter,
    cfg: dict,
    *,
    device: torch.device,
    render_rows: int,
    render_cols: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load the real marker background and Taxim no-contact frame once."""

    background_path = str(cfg.get("taxim_rgb_background_path", "")).strip()
    if not background_path:
        return None
    path = Path(background_path).expanduser().resolve()
    key = (id(adapter), str(path), render_rows, render_cols, str(device))
    cached = getattr(env, "_rl_ours_taxim_rgb_background_cache", None)
    if isinstance(cached, dict) and cached.get("key") == key:
        return cached["real"], cached["taxim_zero"]
    if not path.is_file():
        raise FileNotFoundError(f"RL Taxim RGB marker background does not exist: {path}")

    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (render_cols, render_rows):
            image = image.resize((render_cols, render_rows), resample=Image.Resampling.BILINEAR)
        real = torch.as_tensor(
            np.array(image, dtype=np.uint8, copy=True),
            device=device,
            dtype=torch.float32,
        )
    taxim_zero = adapter.step(
        torch.zeros((1, render_rows, render_cols), device=device, dtype=torch.float32)
    ).tactile_rgb[0].detach().to(device=device, dtype=torch.float32)
    env._rl_ours_taxim_rgb_background_cache = {
        "key": key,
        "real": real,
        "taxim_zero": taxim_zero,
    }
    return real, taxim_zero


def _ours_taxim_rgb_policy_from_local(
    env: ManagerBasedRLEnv,
    local_depth_m: torch.Tensor,
    local_active: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """Render and return the five full-resolution RGB images used by visualization."""

    if local_depth_m.ndim != 4 or local_active.shape != local_depth_m.shape[:2]:
        raise ValueError("RL Taxim RGB requires local Depth [env,finger,row,col] and matching active flags")
    reference = getattr(env, "_rl_local_tacmap_reference_state", None)
    if not isinstance(reference, dict):
        raise RuntimeError("RL Taxim RGB requires the calibrated local TacMap reference state")

    num_envs, finger_count = (int(value) for value in local_depth_m.shape[:2])
    render_rows = int(reference.get("reference_rows", 0))
    render_cols = int(reference.get("reference_cols", 0))
    configured_rows = max(1, int(cfg.get("taxim_rgb_render_rows", RL_TAXIM_RGB_RENDER_ROWS)))
    configured_cols = max(1, int(cfg.get("taxim_rgb_render_cols", RL_TAXIM_RGB_RENDER_COLS)))
    if (render_rows, render_cols) != (configured_rows, configured_cols):
        raise ValueError(
            "RL Taxim RGB render shape must match the calibrated dense TacMap reference: "
            f"reference={(render_rows, render_cols)}, configured={(configured_rows, configured_cols)}"
        )
    dense_depth = _ours_dense_tacmap_depth_from_local(env, local_depth_m, local_active, cfg)

    adapter = _ours_taxim_rgb_adapter(env, cfg)
    background = _ours_taxim_rgb_background(
        env,
        adapter,
        cfg,
        device=local_depth_m.device,
        render_rows=render_rows,
        render_cols=render_cols,
    )
    chunk_size = max(1, int(cfg.get("taxim_rgb_render_chunk_size", RL_TAXIM_RGB_RENDER_CHUNK_SIZE)))

    if background is not None:
        default_rgb = background[0]
    else:
        zero_key = (id(adapter), render_rows, render_cols, str(local_depth_m.device))
        zero_cache = getattr(env, "_rl_ours_taxim_rgb_zero_cache", None)
        if not isinstance(zero_cache, dict) or zero_cache.get("key") != zero_key:
            zero_rgb = adapter.step(
                torch.zeros(
                    (1, render_rows, render_cols),
                    device=local_depth_m.device,
                    dtype=torch.float32,
                )
            ).tactile_rgb[0].detach().to(device=local_depth_m.device, dtype=torch.float32)
            zero_cache = {"key": zero_key, "rgb": zero_rgb}
            env._rl_ours_taxim_rgb_zero_cache = zero_cache
        default_rgb = zero_cache["rgb"]
    default_chw = default_rgb.permute(2, 0, 1).unsqueeze(0) / 255.0
    flat_depth = dense_depth.reshape(num_envs * finger_count, render_rows, render_cols)
    flat_active = local_active.reshape(-1)

    # An inactive local TacMap is exactly a no-contact Taxim frame.  Fill those
    # entries from the cached background and render only active env/finger pairs.
    full_rgb_flat = default_chw.expand(num_envs * finger_count, -1, -1, -1).clone()
    active_indices = torch.nonzero(flat_active, as_tuple=False).flatten()
    for start in range(0, int(active_indices.numel()), chunk_size):
        selected = active_indices[start : start + chunk_size]
        taxim_rgb = adapter.step(flat_depth.index_select(0, selected)).tactile_rgb.to(
            device=local_depth_m.device,
            dtype=torch.float32,
        )
        if background is not None:
            real, taxim_zero = background
            taxim_rgb = torch.clamp(real.unsqueeze(0) + taxim_rgb - taxim_zero.unsqueeze(0), 0.0, 255.0)
        rgb_chw = taxim_rgb.permute(0, 3, 1, 2) / 255.0
        full_rgb_flat.index_copy_(0, selected, rgb_chw.to(dtype=torch.float32))

    full_rgb = full_rgb_flat.reshape(
        num_envs,
        finger_count,
        3,
        render_rows,
        render_cols,
    )
    return torch.nan_to_num(full_rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0).reshape(num_envs, -1)


class _FrozenTactileResNet18(nn.Module):
    """Frozen ImageNet ResNet-18 with deterministic five-finger 128-D reduction."""

    def __init__(
        self,
        *,
        weights_path: Path | None,
        output_dim: int,
        input_rows: int,
        input_cols: int,
        chunk_size: int,
    ):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=None)
        if weights_path is None:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            weight_source = str(ResNet18_Weights.DEFAULT)
        else:
            try:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
            except RuntimeError as exc:
                if "legacy .tar format" not in str(exc):
                    raise
                expected_prefix = weights_path.stem.rsplit("-", 1)[-1].lower()
                digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
                if len(expected_prefix) < 8 or not digest.startswith(expected_prefix):
                    raise RuntimeError(
                        "Refusing to load legacy pretrained weights without a matching filename hash: "
                        f"path={weights_path}, sha256={digest}"
                    ) from exc
                state = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)
            weight_source = str(weights_path)
        self.backbone = nn.Sequential(*tuple(model.children())[:-1])
        self.output_dim = int(output_dim)
        self.input_rows = int(input_rows)
        self.input_cols = int(input_cols)
        self.chunk_size = int(chunk_size)
        self.weight_source = weight_source
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        """Keep all pretrained BatchNorm layers frozen in evaluation mode."""

        del mode
        return super().train(False)

    def forward(self, finger_images: torch.Tensor) -> torch.Tensor:
        if finger_images.ndim != 5 or int(finger_images.shape[2]) not in (1, 3):
            raise ValueError(
                "Frozen tactile ResNet expects [env,finger,1|3,row,col], "
                f"got {tuple(finger_images.shape)}"
            )
        num_envs, finger_count, channels, rows, cols = (
            int(value) for value in finger_images.shape
        )
        flat = finger_images.reshape(num_envs * finger_count, channels, rows, cols)
        features = []
        use_amp = flat.device.type == "cuda"
        for start in range(0, int(flat.shape[0]), self.chunk_size):
            images = flat[start : start + self.chunk_size].to(dtype=torch.float32)
            if channels == 1:
                images = images.expand(-1, 3, -1, -1)
            if tuple(images.shape[-2:]) != (self.input_rows, self.input_cols):
                images = F.interpolate(
                    images,
                    size=(self.input_rows, self.input_cols),
                    mode="bilinear",
                    align_corners=False,
                )
            images = (images - self.imagenet_mean) / self.imagenet_std
            with torch.autocast(device_type=flat.device.type, dtype=torch.float16, enabled=use_amp):
                encoded = self.backbone(images).flatten(1)
            features.append(encoded.to(dtype=torch.float32))
        ordered_finger_features = torch.cat(features, dim=0).reshape(num_envs, finger_count * 512)
        reduced = F.adaptive_avg_pool1d(
            ordered_finger_features.unsqueeze(1),
            self.output_dim,
        ).squeeze(1)
        return F.normalize(reduced, p=2.0, dim=-1, eps=1.0e-8)


def _resolve_frozen_resnet18_weights(cfg: dict) -> Path | None:
    configured = str(cfg.get("tactile_resnet_weights_path", "")).strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Frozen tactile ResNet-18 weights do not exist: {path}")
        return path

    checkpoint_dir = Path(torch.hub.get_dir()).expanduser() / "checkpoints"
    preferred_names = (
        "resnet18-f37072fd.pth",
        "resnet18-5c106cde.pth",
    )
    for name in preferred_names:
        path = checkpoint_dir / name
        if path.is_file():
            return path.resolve()
    candidates = sorted(checkpoint_dir.glob("resnet18-*.pth")) if checkpoint_dir.is_dir() else []
    return candidates[0].resolve() if candidates else None


def _ours_frozen_tactile_resnet(
    env: ManagerBasedRLEnv,
    cfg: dict,
    *,
    modality: str,
) -> _FrozenTactileResNet18:
    output_dim = max(1, int(cfg.get("tactile_resnet_output_dim", RL_TACTILE_RESNET_OUTPUT_DIM)))
    input_rows = max(1, int(cfg.get("tactile_resnet_input_rows", RL_TACTILE_RESNET_INPUT_ROWS)))
    input_cols = max(1, int(cfg.get("tactile_resnet_input_cols", RL_TACTILE_RESNET_INPUT_COLS)))
    chunk_size = max(1, int(cfg.get("tactile_resnet_chunk_size", RL_TACTILE_RESNET_CHUNK_SIZE)))
    weights_path = _resolve_frozen_resnet18_weights(cfg)
    key = (
        str(modality),
        None if weights_path is None else str(weights_path),
        str(env.device),
        output_dim,
        input_rows,
        input_cols,
        chunk_size,
    )
    attr_name = f"_rl_ours_{modality}_frozen_resnet18"
    key_name = f"{attr_name}_key"
    encoder = getattr(env, attr_name, None)
    if isinstance(encoder, _FrozenTactileResNet18) and getattr(env, key_name, None) == key:
        return encoder
    encoder = _FrozenTactileResNet18(
        weights_path=weights_path,
        output_dim=output_dim,
        input_rows=input_rows,
        input_cols=input_cols,
        chunk_size=chunk_size,
    ).to(device=env.device)
    encoder.eval()
    setattr(env, attr_name, encoder)
    setattr(env, key_name, key)
    print(
        f"[INFO] Frozen tactile ResNet-18 loaded: modality={modality}, "
        f"output={output_dim}, weights={encoder.weight_source}",
        flush=True,
    )
    return encoder


def _ours_frozen_resnet_embedding(
    env: ManagerBasedRLEnv,
    cfg: dict,
    images: torch.Tensor,
    *,
    modality: str,
) -> torch.Tensor:
    encoder = _ours_frozen_tactile_resnet(env, cfg, modality=modality)
    with torch.inference_mode():
        output = encoder(images)
    return output.detach().to(device=env.device, dtype=torch.float32)


def invalidate_ours_tactile_cache_on_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | None,
) -> None:
    """Invalidate step-local OURS TacMap caches after any environment reset.

    TacMap tensors are batched across all environments, so a partial reset
    conservatively invalidates the shared cache for the whole batch.  The
    HydroShear adapter keeps its per-environment history reset in
    :func:`hydroshear_rl_obs`, using ``episode_length_buf == 0``.
    """

    del env_ids
    env._rl_ours_tactile_reset_epoch = int(getattr(env, "_rl_ours_tactile_reset_epoch", 0)) + 1
    env._rl_tacmap_cache_step = None
    env._rl_local_tacmap_cache_step = None
    env._rl_ours_tacmap_cache_epoch = None
    env._rl_ours_local_tacmap_cache_epoch = None
    env._rl_ours_dense_tacmap_cache_step = None
    env._rl_ours_dense_tacmap_cache_epoch = None
    env._rl_ours_taxim_rgb_cache_step = None
    env._rl_ours_taxim_rgb_cache_epoch = None
    env._rl_ours_tacmap_resnet_cache_step = None
    env._rl_ours_tacmap_resnet_cache_epoch = None
    env._rl_ours_taxim_resnet_cache_step = None
    env._rl_ours_taxim_resnet_cache_epoch = None


def ours_rl_pressure_obs(
    env: ManagerBasedRLEnv,
    component_cfg: dict,
) -> torch.Tensor:
    """Compute only the pressure component of the OURS tactile observation."""

    cfg = dict(component_cfg)
    pressure = warpsdf_pressure_obs(
        env,
        pressure_sensor_names=cfg.get("pressure_sensor_names", ()),
        object_cfg=cfg.get("object_cfg", SceneEntityCfg("object")),
        pressure_rows=cfg.get("pressure_rows", 4),
        pressure_cols=cfg.get("pressure_cols", 8),
        pressure_taxel_counts=cfg.get("pressure_taxel_counts"),
        pressure_diffusion_enabled=cfg.get("pressure_diffusion_enabled", False),
        pressure_diffusion_sigma_m=cfg.get("pressure_diffusion_sigma_m", 0.003),
        pressure_diffusion_blend=cfg.get("pressure_diffusion_blend", 0.7),
        pressure_diffusion_radius_sigma=cfg.get("pressure_diffusion_radius_sigma", 3.0),
        pressure_diffusion_normal_power=cfg.get("pressure_diffusion_normal_power", 1.0),
    )
    if bool(cfg.get("cache_pressure_output", False)):
        setattr(env, str(cfg.get("pressure_output_attr_name", "_rl_pressure_observation")), pressure.detach())
    return pressure


def ours_rl_tacmap_policy_obs(
    env: ManagerBasedRLEnv,
    component_cfg: dict,
) -> torch.Tensor:
    """Return the full metric Depth images, with compact legacy output available by config."""

    cfg = dict(component_cfg)
    tacmap_policy, local_depth, _, local_active = _ours_tacmap_policy_components(env, cfg)
    if not bool(cfg.get("tacmap_policy_full_depth_image", False)):
        return tacmap_policy
    if local_depth is None or local_active is None:
        raise RuntimeError("Full-resolution RL Depth requires local_tacmap_enabled=True")
    dense_depth = _ours_dense_tacmap_depth_from_local(env, local_depth, local_active, cfg)
    return dense_depth.reshape(env.num_envs, -1)


def ours_rl_taxim_rgb_obs(
    env: ManagerBasedRLEnv,
    component_cfg: dict,
) -> torch.Tensor:
    """Return five full channel-first Taxim RGB images normalized to [0, 1]."""

    cfg = dict(component_cfg)
    current_step = int(getattr(env, "common_step_counter", -1))
    reset_epoch = int(getattr(env, "_rl_ours_tactile_reset_epoch", 0))
    render_rows = max(1, int(cfg.get("taxim_rgb_render_rows", RL_TAXIM_RGB_RENDER_ROWS)))
    render_cols = max(1, int(cfg.get("taxim_rgb_render_cols", RL_TAXIM_RGB_RENDER_COLS)))
    expected_shape = (env.num_envs, len(RL_FINGER_ORDER) * 3 * render_rows * render_cols)
    cached = getattr(env, "_rl_ours_taxim_rgb_observation", None)
    if (
        getattr(env, "_rl_ours_taxim_rgb_cache_step", None) == current_step
        and getattr(env, "_rl_ours_taxim_rgb_cache_epoch", None) == reset_epoch
        and isinstance(cached, torch.Tensor)
        and tuple(cached.shape) == expected_shape
    ):
        return cached

    _, local_depth, _, local_active = _ours_tacmap_policy_components(env, cfg)
    if local_depth is None or local_active is None:
        raise RuntimeError("RL Taxim RGB requires local_tacmap_enabled=True")
    output = _ours_taxim_rgb_policy_from_local(env, local_depth, local_active, cfg)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"RL Taxim RGB produced {tuple(output.shape)}, expected {expected_shape}")
    env._rl_ours_taxim_rgb_observation = output
    env._rl_ours_taxim_rgb_cache_step = current_step
    env._rl_ours_taxim_rgb_cache_epoch = reset_epoch
    return output


def ours_rl_tacmap_resnet_obs(
    env: ManagerBasedRLEnv,
    component_cfg: dict,
) -> torch.Tensor:
    """Return a frozen pretrained ResNet-18 embedding of the five Depth images."""

    cfg = dict(component_cfg)
    current_step = int(getattr(env, "common_step_counter", -1))
    reset_epoch = int(getattr(env, "_rl_ours_tactile_reset_epoch", 0))
    output_dim = max(1, int(cfg.get("tactile_resnet_output_dim", RL_TACTILE_RESNET_OUTPUT_DIM)))
    expected_shape = (env.num_envs, output_dim)
    cached = getattr(env, "_rl_ours_tacmap_resnet_observation", None)
    if (
        getattr(env, "_rl_ours_tacmap_resnet_cache_step", None) == current_step
        and getattr(env, "_rl_ours_tacmap_resnet_cache_epoch", None) == reset_epoch
        and isinstance(cached, torch.Tensor)
        and tuple(cached.shape) == expected_shape
    ):
        return cached

    cfg["tacmap_policy_full_depth_image"] = True
    render_rows = max(1, int(cfg.get("taxim_rgb_render_rows", RL_TAXIM_RGB_RENDER_ROWS)))
    render_cols = max(1, int(cfg.get("taxim_rgb_render_cols", RL_TAXIM_RGB_RENDER_COLS)))
    depth = ours_rl_tacmap_policy_obs(env, cfg).reshape(
        env.num_envs,
        len(RL_FINGER_ORDER),
        1,
        render_rows,
        render_cols,
    )
    depth_max_m = max(
        1.0e-8,
        float(cfg.get("tactile_resnet_depth_max_m", RL_TACTILE_RESNET_DEPTH_MAX_M)),
    )
    normalized_depth = torch.clamp(depth / depth_max_m, 0.0, 1.0)
    output = _ours_frozen_resnet_embedding(
        env,
        cfg,
        normalized_depth,
        modality="depth",
    )
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Frozen Depth ResNet produced {tuple(output.shape)}, expected {expected_shape}")
    env._rl_ours_tacmap_resnet_observation = output
    env._rl_ours_tacmap_resnet_cache_step = current_step
    env._rl_ours_tacmap_resnet_cache_epoch = reset_epoch
    return output


def ours_rl_taxim_resnet_obs(
    env: ManagerBasedRLEnv,
    component_cfg: dict,
) -> torch.Tensor:
    """Return a frozen pretrained ResNet-18 embedding of the five Taxim RGB images."""

    cfg = dict(component_cfg)
    current_step = int(getattr(env, "common_step_counter", -1))
    reset_epoch = int(getattr(env, "_rl_ours_tactile_reset_epoch", 0))
    output_dim = max(1, int(cfg.get("tactile_resnet_output_dim", RL_TACTILE_RESNET_OUTPUT_DIM)))
    expected_shape = (env.num_envs, output_dim)
    cached = getattr(env, "_rl_ours_taxim_resnet_observation", None)
    if (
        getattr(env, "_rl_ours_taxim_resnet_cache_step", None) == current_step
        and getattr(env, "_rl_ours_taxim_resnet_cache_epoch", None) == reset_epoch
        and isinstance(cached, torch.Tensor)
        and tuple(cached.shape) == expected_shape
    ):
        return cached

    render_rows = max(1, int(cfg.get("taxim_rgb_render_rows", RL_TAXIM_RGB_RENDER_ROWS)))
    render_cols = max(1, int(cfg.get("taxim_rgb_render_cols", RL_TAXIM_RGB_RENDER_COLS)))
    rgb = ours_rl_taxim_rgb_obs(env, cfg).reshape(
        env.num_envs,
        len(RL_FINGER_ORDER),
        3,
        render_rows,
        render_cols,
    )
    output = _ours_frozen_resnet_embedding(
        env,
        cfg,
        torch.clamp(rgb, 0.0, 1.0),
        modality="rgb",
    )
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Frozen RGB ResNet produced {tuple(output.shape)}, expected {expected_shape}")
    env._rl_ours_taxim_resnet_observation = output
    env._rl_ours_taxim_resnet_cache_step = current_step
    env._rl_ours_taxim_resnet_cache_epoch = reset_epoch
    return output


def ours_rl_hydroshear_obs(
    env: ManagerBasedRLEnv,
    component_cfg: dict,
) -> torch.Tensor:
    """Compute HydroShear, materializing only its required TacMap dependency."""

    cfg = dict(component_cfg)
    _, local_depth, local_roi, local_active = _ours_tacmap_policy_components(env, cfg)
    return hydroshear_rl_obs(
        env,
        tacmap_sensor_names=cfg.get("tacmap_sensor_names", ()),
        tacmap_surface_sensor_names=cfg.get("tacmap_surface_sensor_names", ()),
        tacmap_rows=cfg.get("tacmap_rows", 32),
        tacmap_cols=cfg.get("tacmap_cols", 24),
        render_rows=cfg.get("hydroshear_render_rows", RL_HYDROSHEAR_RENDER_ROWS),
        render_cols=cfg.get("hydroshear_render_cols", RL_HYDROSHEAR_RENDER_COLS),
        marker_rows=cfg.get("hydroshear_marker_rows", RL_HYDROSHEAR_MARKER_ROWS),
        marker_cols=cfg.get("hydroshear_marker_cols", RL_HYDROSHEAR_MARKER_COLS),
        marker_margin_x=cfg.get("hydroshear_marker_margin_x", RL_HYDROSHEAR_MARKER_MARGIN_X),
        marker_margin_y=cfg.get("hydroshear_marker_margin_y", RL_HYDROSHEAR_MARKER_MARGIN_Y),
        object_cfg=cfg.get("object_cfg", SceneEntityCfg("object")),
        object_sample_mode=cfg.get("hydroshear_object_sample_mode", RL_HYDROSHEAR_OBJECT_SAMPLE_MODE),
        object_sample_count=cfg.get("hydroshear_object_sample_count", RL_HYDROSHEAR_OBJECT_SAMPLE_COUNT),
        object_sample_seed=cfg.get("hydroshear_object_sample_seed", RL_HYDROSHEAR_OBJECT_SAMPLE_SEED),
        object_poisson_radius=cfg.get("hydroshear_object_poisson_radius", RL_HYDROSHEAR_POISSON_RADIUS),
        object_poisson_initial_count=cfg.get(
            "hydroshear_object_poisson_initial_count", RL_HYDROSHEAR_POISSON_INITIAL_COUNT
        ),
        object_sample_reference_count=cfg.get(
            "hydroshear_object_sample_reference_count", RL_HYDROSHEAR_OBJECT_SAMPLE_REFERENCE_COUNT
        ),
        object_sample_roi_count=cfg.get(
            "hydroshear_object_sample_roi_count", RL_HYDROSHEAR_OBJECT_SAMPLE_ROI_COUNT
        ),
        use_object_surface_samples=cfg.get("hydroshear_use_object_surface_samples", True),
        cache_debug_output=cfg.get("hydroshear_cache_debug_output", False),
        render_debug_marker_images=cfg.get("hydroshear_render_debug_marker_images", True),
        debug_sensor_index=cfg.get("hydroshear_debug_sensor_index"),
        cache_marker_depth_output=cfg.get("hydroshear_cache_marker_depth_output", False),
        marker_layout_path=cfg.get("hydroshear_marker_layout_path"),
        marker_finger_link_names=cfg.get("hydroshear_marker_finger_link_names", ()),
        marker_ray_sensor_names=cfg.get("hydroshear_marker_sensor_names", ()),
        marker_ray_surface_sensor_names=cfg.get("hydroshear_marker_surface_sensor_names", ()),
        dilation_source_depth_m=local_depth,
        dilation_source_roi_norm=local_roi,
        dilation_source_active=local_active,
        robot_cfg=cfg.get("robot_cfg", SceneEntityCfg("robot")),
    )
