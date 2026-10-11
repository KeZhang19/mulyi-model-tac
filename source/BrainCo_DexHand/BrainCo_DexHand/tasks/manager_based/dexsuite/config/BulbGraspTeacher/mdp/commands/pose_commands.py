# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Isolated V2-test teacher port. Source and SHA256 are recorded in port_manifest.json.
from __future__ import annotations

import torch

from collections.abc import Sequence

from isaaclab.assets import Articulation, RigidObject

from isaaclab.managers import CommandTerm

from isaaclab.markers import VisualizationMarkers

from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

V2_TEST_OBJECT_NAMES = (
    "cube",
    "bread",
    "cardboard_box",
    "peach",
    "pomegranate",
    "tissue",
)

V2_TEST_OBJECT_METADATA_KEY = "v2_test_object_name"

class ObjectUniformPoseCommand(CommandTerm):
    """Uniform pose command generator for an object (in the robot base frame).

    This command term samples target object poses by:
      • Drawing (x, y, z) uniformly within configured Cartesian bounds, and
      • Drawing roll-pitch-yaw uniformly within configured ranges, then converting
        to a quaternion (w, x, y, z). Optionally makes quaternions unique by enforcing
        a positive real part.

    Frames:
        Targets are defined in the robot's *base frame*. For metrics/visualization,
        targets are transformed into the *world frame* using the robot root pose.

    Outputs:
        The command buffer has shape (num_envs, 7): `(x, y, z, qw, qx, qy, qz)`.

    Metrics:
        `position_error` and `orientation_error` are computed between the commanded
        world-frame pose and the object's current world-frame pose.

    Config:
        `cfg` must provide the sampling ranges, whether to enforce quaternion uniqueness,
        and optional visualization settings.
    """

    cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.success_vis_asset: RigidObject = env.scene[cfg.success_vis_asset_name]
        self.v2_test_object_names = tuple(cfg.object_metadata_names or V2_TEST_OBJECT_NAMES)
        self.v2_test_object_ids = self._read_v2_test_object_ids(env)

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_success"] = torch.zeros(self.num_envs, device=self.device)
        # 0/1 success indicator (can be averaged to compute success_rate)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        # Snapshot populated by reset() before CommandTerm clears the terminal metric.
        self.terminal_success = torch.zeros(self.num_envs, device=self.device)

        self.success_visualizer = VisualizationMarkers(self.cfg.success_visualizer_cfg)
        self.success_visualizer.set_visibility(True)

    def _read_v2_test_object_ids(self, env: ManagerBasedEnv) -> torch.Tensor | None:
        """Read the scene-time V2-test asset choice for every environment."""

        object_prim_name = self.object.cfg.prim_path.rsplit("/", 1)[-1]
        object_names_by_env: list[str | None] = []
        for env_prim_path in env.scene.env_prim_paths:
            prim = env.scene.stage.GetPrimAtPath(f"{env_prim_path}/{object_prim_name}")
            object_name = (
                prim.GetCustomDataByKey(V2_TEST_OBJECT_METADATA_KEY)
                if prim.IsValid()
                else None
            )
            object_names_by_env.append(object_name if isinstance(object_name, str) else None)

        if all(object_name is None for object_name in object_names_by_env):
            return None
        missing_env_ids = [
            env_id
            for env_id, object_name in enumerate(object_names_by_env)
            if object_name is None
        ]
        if missing_env_ids:
            raise RuntimeError(
                "Incomplete V2-test object metadata: missing "
                f"{V2_TEST_OBJECT_METADATA_KEY!r} for environment IDs {missing_env_ids[:8]}."
            )

        unknown_names = sorted(
            {
                object_name
                for object_name in object_names_by_env
                if object_name not in self.v2_test_object_names
            }
        )
        if unknown_names:
            raise RuntimeError(
                "Unsupported V2-test object metadata values: "
                f"{unknown_names}; expected one of {self.v2_test_object_names}."
            )

        name_to_id = {name: index for index, name in enumerate(self.v2_test_object_names)}
        return torch.tensor(
            [name_to_id[object_name] for object_name in object_names_by_env],
            dtype=torch.long,
            device=self.device,
        )

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Snapshot terminal success per environment before clearing command metrics."""

        resolved_env_ids = slice(None) if env_ids is None else env_ids
        self.terminal_success[resolved_env_ids] = self.metrics["success"][
            resolved_env_ids
        ]
        return super().reset(env_ids=env_ids)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        position_success = self.metrics["position_error"] < 0.05
        orientation_success = self.metrics["orientation_error"] < 0.5
        self.metrics["position_success"] = position_success.to(dtype=torch.float32)
        self.metrics["orientation_success"] = orientation_success.to(dtype=torch.float32)

        success_id = position_success
        if not self.cfg.position_only:
            success_id &= orientation_success
        self.metrics["success"] = success_id.to(dtype=torch.float32)
        self.success_visualizer.visualize(self.success_vis_asset.data.root_pos_w, marker_indices=success_id.int())

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                # -- goal pose
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.curr_visualizer = VisualizationMarkers(self.cfg.curr_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_visualizer.set_visibility(True)
            self.curr_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
                self.curr_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        if not self.cfg.position_only:
            # -- goal pose
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
            # -- current object pose
            self.curr_visualizer.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
        else:
            distance = torch.norm(self.pose_command_w[:, :3] - self.object.data.root_pos_w[:, :3], dim=1)
            success_id = (distance < 0.05).int()
            # note: since marker indices for position is 1(far) and 2(near), we can simply shift the success_id by 1.
            # -- goal position
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=success_id + 1)
            # -- current object position
            self.curr_visualizer.visualize(self.object.data.root_pos_w, marker_indices=success_id + 1)
