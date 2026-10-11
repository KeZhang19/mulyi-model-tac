# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Isolated V2-test teacher port. Source and SHA256 are recorded in port_manifest.json.
from __future__ import annotations

import math

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

ABNORMAL_ROBOT_DEFAULT_VELOCITY_LIMIT_SCALE = 2.0

def _validate_abnormal_robot_velocity_limit_scale(velocity_limit_scale: float) -> float:
    """Return a validated abnormal-robot velocity-limit multiplier."""
    if (
        isinstance(velocity_limit_scale, bool)
        or not isinstance(velocity_limit_scale, (int, float))
        or not math.isfinite(float(velocity_limit_scale))
        or float(velocity_limit_scale) <= 0.0
    ):
        raise ValueError("velocity_limit_scale must be a finite number greater than zero")
    return float(velocity_limit_scale)

def _abnormal_robot_joint_mask(robot: Articulation, velocity_limit_scale: float) -> torch.Tensor:
    """Return the strict per-joint abnormal-velocity mask."""
    scale = _validate_abnormal_robot_velocity_limit_scale(velocity_limit_scale)
    return robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * scale)

def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds

class abnormal_robot_state_with_diagnostics(ManagerTermBase):
    """Stable-teacher abnormal termination with reset-relative diagnostic snapshots.

    Detection deliberately remains at the environment-step boundary. The independent
    counter is necessary because RSL-RL randomizes ``episode_length_buf`` at training
    startup, which would otherwise corrupt a reset-relative first-event step.
    """

    provides_abnormal_robot_diagnostics = True

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        robot: Articulation = env.scene[asset_cfg.name]
        self.diagnostic_joint_names = tuple(robot.joint_names)
        self._steps_since_reset = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        self._decimation = int(env.cfg.decimation)
        if self._decimation < 1:
            raise ValueError(f"decimation must be positive; got {self._decimation}")

        self.last_abnormal_joint_mask = torch.zeros(
            (env.num_envs, len(self.diagnostic_joint_names)),
            dtype=torch.bool,
            device=env.device,
        )
        self.last_abnormal_env_step = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        self.last_abnormal_physics_step = torch.zeros_like(self.last_abnormal_env_step)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        velocity_limit_scale: float = ABNORMAL_ROBOT_DEFAULT_VELOCITY_LIMIT_SCALE,
    ) -> torch.Tensor:
        self._steps_since_reset += 1
        robot: Articulation = env.scene[asset_cfg.name]
        joint_mask = _abnormal_robot_joint_mask(robot, velocity_limit_scale)
        abnormal = joint_mask.any(dim=1)

        # These snapshots describe only this call. Automatic reset clears the
        # next-episode counter but intentionally leaves the terminal snapshot
        # intact until the runner consumes it after env.step().
        self.last_abnormal_joint_mask = joint_mask.detach()
        self.last_abnormal_env_step = torch.where(
            abnormal,
            self._steps_since_reset,
            torch.zeros_like(self._steps_since_reset),
        )
        self.last_abnormal_physics_step = self.last_abnormal_env_step * self._decimation
        return abnormal

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._steps_since_reset[env_ids] = 0

def non_finite_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when NaN/Inf appears in key simulation states.

    This typically happens when reset randomization spawns interpenetrating bodies (hand/object/table),
    or when contacts/constraints blow up numerically. Ending the episode early prevents PPO from
    ingesting NaN/Inf rewards/observations and "collapsing".
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Robot joint states
    jp = robot.data.joint_pos
    jv = robot.data.joint_vel

    # Root states
    rpos = robot.data.root_pos_w
    rquat = robot.data.root_quat_w
    opos = obj.data.root_pos_w
    oquat = obj.data.root_quat_w

    finite_robot = (
        torch.isfinite(jp).all(dim=1)
        & torch.isfinite(jv).all(dim=1)
        & torch.isfinite(rpos).all(dim=1)
        & torch.isfinite(rquat).all(dim=1)
    )
    finite_obj = torch.isfinite(opos).all(dim=1) & torch.isfinite(oquat).all(dim=1)

    return ~(finite_robot & finite_obj)
