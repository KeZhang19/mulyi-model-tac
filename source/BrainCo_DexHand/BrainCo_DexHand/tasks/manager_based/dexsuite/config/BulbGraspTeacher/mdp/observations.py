# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Isolated V2-test teacher port. Source and SHA256 are recorded in port_manifest.json.
from __future__ import annotations

import hashlib

import json

from pathlib import Path

import numpy as np

import torch

from isaaclab.assets import Articulation, RigidObject

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms

from .utils import sample_object_point_cloud

V2_TEST_OBJECT_METADATA_KEY = "v2_test_object_name"

OFFLINE_POINT_CLOUD_MANIFEST_VERSION = 1

def _load_offline_object_point_clouds(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg,
    num_points: int,
    point_cloud_dir: str | Path,
) -> torch.Tensor:
    """Load and select object-local point clouds for all parallel environments."""

    directory = Path(point_cloud_dir).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Offline point-cloud manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Unable to read offline point-cloud manifest {manifest_path}: {exc}") from exc

    if manifest.get("version") != OFFLINE_POINT_CLOUD_MANIFEST_VERSION:
        raise ValueError(
            f"Offline point-cloud manifest {manifest_path} has version={manifest.get('version')!r}; "
            f"expected {OFFLINE_POINT_CLOUD_MANIFEST_VERSION}."
        )
    expected_shape = [num_points, 3]
    if manifest.get("shape") != expected_shape or manifest.get("dtype") != "float32":
        raise ValueError(
            f"Offline point-cloud manifest {manifest_path} has shape={manifest.get('shape')!r}, "
            f"dtype={manifest.get('dtype')!r}; expected shape={expected_shape}, dtype='float32'."
        )
    object_entries = manifest.get("objects")
    if not isinstance(object_entries, list) or not object_entries:
        raise ValueError(f"Offline point-cloud manifest {manifest_path} has no object entries.")

    points_by_name: dict[str, np.ndarray] = {}
    for entry in object_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid object entry in offline point-cloud manifest {manifest_path}: {entry!r}")
        object_name = entry.get("name")
        filename = entry.get("file")
        expected_hash = entry.get("file_sha256")
        if not all(isinstance(value, str) and value for value in (object_name, filename, expected_hash)):
            raise ValueError(f"Incomplete object entry in offline point-cloud manifest {manifest_path}: {entry!r}")
        if object_name in points_by_name:
            raise ValueError(f"Duplicate object {object_name!r} in offline point-cloud manifest {manifest_path}.")

        point_path = (directory / filename).resolve()
        if point_path.parent != directory:
            raise ValueError(f"Offline point-cloud path escapes its directory: {point_path}")
        if not point_path.is_file():
            raise FileNotFoundError(f"Offline point cloud for object {object_name!r} not found: {point_path}")
        payload = point_path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Offline point cloud for object {object_name!r} failed SHA-256 validation at {point_path}: "
                f"expected {expected_hash}, got {actual_hash}."
            )
        try:
            points = np.load(point_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Unable to load offline point cloud for object {object_name!r} at {point_path}: {exc}"
            ) from exc
        if points.shape != (num_points, 3) or points.dtype != np.float32:
            raise ValueError(
                f"Offline point cloud for object {object_name!r} at {point_path} has "
                f"shape={points.shape}, dtype={points.dtype}; expected ({num_points}, 3), float32."
            )
        if not np.isfinite(points).all():
            raise ValueError(f"Offline point cloud for object {object_name!r} contains non-finite values: {point_path}")
        if len(np.unique(points, axis=0)) != num_points:
            raise ValueError(f"Offline point cloud for object {object_name!r} contains duplicate points: {point_path}")
        points_by_name[object_name] = points

    object_asset: RigidObject = env.scene[object_cfg.name]
    object_prim_name = object_asset.cfg.prim_path.rsplit("/", 1)[-1]
    selected_points: list[np.ndarray] = []
    for env_id, env_prim_path in enumerate(env.scene.env_prim_paths):
        object_prim_path = f"{env_prim_path}/{object_prim_name}"
        object_prim = env.scene.stage.GetPrimAtPath(object_prim_path)
        if not object_prim.IsValid():
            raise RuntimeError(f"Object prim for environment {env_id} does not exist: {object_prim_path}")
        object_name = object_prim.GetCustomDataByKey(V2_TEST_OBJECT_METADATA_KEY)
        if not isinstance(object_name, str) or not object_name:
            raise RuntimeError(
                f"Object prim {object_prim_path} is missing string metadata {V2_TEST_OBJECT_METADATA_KEY!r}."
            )
        if object_name not in points_by_name:
            raise RuntimeError(
                f"Object prim {object_prim_path} selects unsupported offline point cloud {object_name!r}; "
                f"available objects are {sorted(points_by_name)} in {directory}."
            )

        scale_attr = object_prim.GetAttribute("xformOp:scale")
        scale_value = scale_attr.Get() if scale_attr else None
        scale = np.ones(3, dtype=np.float32) if scale_value is None else np.asarray(scale_value, dtype=np.float32)
        if scale.shape != (3,) or not np.isfinite(scale).all():
            raise RuntimeError(f"Object prim {object_prim_path} has invalid xformOp:scale={scale_value!r}.")
        selected_points.append(points_by_name[object_name] * scale[None, :])

    if len(selected_points) != env.num_envs:
        raise RuntimeError(
            f"Loaded offline point clouds for {len(selected_points)} environments, expected {env.num_envs}."
        )
    return torch.as_tensor(np.stack(selected_points), dtype=torch.float32, device=env.device)

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

def processed_last_action(
    env: ManagerBasedRLEnv,
    action_name: str = "action",
) -> torch.Tensor:
    """Return the post-scale, post-clip action produced by one action term.

    This is useful when an action term has per-joint clipping that cannot be
    represented by the observation manager's single scalar clip interval.
    """

    action_term = env.action_manager.get_term(action_name)
    processed_actions = action_term.processed_actions
    if not getattr(action_term.cfg, "use_zero_offset", False):
        return processed_actions

    # Isaac Lab clears JointAction.raw_actions during reset but leaves the
    # processed buffer untouched. Relative actions with a zero offset must
    # nevertheless expose zero action history for those freshly reset rows.
    zero_raw_action = action_term.raw_actions.eq(0.0).all(dim=-1, keepdim=True)
    return torch.where(zero_raw_action, torch.zeros_like(processed_actions), processed_actions)

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

    Points are either loaded from deterministic offline samples or pre-sampled on the object's
    surface in its local frame. They are transformed to world, then into the reference (e.g.,
    robot) root frame. Optionally visualizes the points.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.
        offline_point_cloud_dir: Optional directory containing ``manifest.json`` and per-object
            ``.npy`` files. If provided, offline loading is mandatory and never falls back to sampling.

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
        offline_point_cloud_dir = cfg.params.get("offline_point_cloud_dir")
        if offline_point_cloud_dir is None:
            self.points_local = sample_object_point_cloud(
                env.num_envs, num_points, self.object.cfg.prim_path, device=env.device
            )
        else:
            self.points_local = _load_offline_object_point_clouds(
                env,
                object_cfg=self.object_cfg,
                num_points=num_points,
                point_cloud_dir=offline_point_cloud_dir,
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
        offline_point_cloud_dir: str | None = None,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Note:
            Points are loaded or pre-sampled at initialization. The ``num_points`` and
            ``offline_point_cloud_dir`` arguments are retained for observation-manager parameter
            forwarding and do not change the point set at runtime.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Unused at runtime; see note above.
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If ``True``, draw markers for the points.
            offline_point_cloud_dir: Unused at runtime; see note above.

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
