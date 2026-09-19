"""Bounded, once-per-control-step joint targets owned by the Custom bulb task."""

from __future__ import annotations

import math
import re

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass

from .robot_contract import JOINT_NAMES


BULB_ACTION_VERSION = 2
ARM_DELTA_LIMIT = 0.025
HAND_DELTA_LIMIT = 0.05


class BulbJointPositionAction(JointAction):
    """Hold one bounded position target throughout all physics substeps.

    Actions are normalized joint increments. Targets are formed from measured
    positions once in ``process_actions`` and clipped to the joint limits;
    ``apply_actions`` never integrates an increment again.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if len(self._joint_names) != len(JOINT_NAMES) or set(self._joint_names) != set(JOINT_NAMES):
            raise ValueError("Custom bulb actions must resolve exactly the 28 Flexiv/Revo3 joints")
        self.arm_action_indices = [i for i, name in enumerate(self._joint_names)
                                   if re.fullmatch(r"joint[1-7]", name)]
        self.hand_action_indices = [i for i, name in enumerate(self._joint_names)
                                    if re.fullmatch(r"right_.*_joint", name)]
        if len(self.arm_action_indices) != 7 or len(self.hand_action_indices) != 21:
            raise ValueError("Custom bulb action mapping must contain seven arm and 21 hand joints")
        self._delta_limits = torch.as_tensor(self._scale, device=self.device).expand_as(self._raw_actions).clone()
        if not torch.isfinite(self._delta_limits).all() or not (self._delta_limits > 0).all():
            raise ValueError("Custom bulb joint increment limits must be finite and positive")
        if torch.as_tensor(self._offset).ne(0).any():
            raise ValueError("Custom bulb incremental actions require zero offset")
        if getattr(cfg, "clip", None) is not None:
            raise ValueError("Set Custom bulb increment bounds through scale, not an additional action clip")
        self._position_targets = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._control_dt = float(env.step_dt)
        self._decimation = int(env.cfg.decimation)
        if not math.isfinite(self._control_dt) or self._control_dt <= 0 or self._decimation < 1:
            raise ValueError("Custom bulb action timing must be finite and positive")

    @property
    def position_targets(self) -> torch.Tensor:
        """The actual position request held across the current physics substeps."""
        return self._position_targets

    @property
    def processed_actions(self) -> torch.Tensor:
        """Absolute targets, as submitted to the articulation drive."""
        return self._position_targets

    @property
    def action_deltas(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_ids(self):
        return self._joint_ids

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected Custom bulb action shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("Custom bulb received non-finite actions")
        self._raw_actions.copy_(actions.clamp(-1.0, 1.0))
        self._processed_actions.copy_(self._raw_actions * self._delta_limits)
        positions = self._asset.data.joint_pos[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._position_targets.copy_(torch.clamp(
            positions + self._processed_actions, min=limits[..., 0], max=limits[..., 1]))

    def apply_actions(self):
        self._asset.set_joint_position_target(self._position_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        positions = self._asset.data.joint_pos[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._position_targets[ids] = torch.clamp(
            positions[ids], min=limits[ids, :, 0], max=limits[ids, :, 1])

    def action_contract(self) -> dict:
        """JSON contract consumed by train/play and exported-policy users."""
        return {
            "version": BULB_ACTION_VERSION,
            "joint_names": list(self._joint_names),
            "arm_action_indices": list(self.arm_action_indices),
            "hand_action_indices": list(self.hand_action_indices),
            "delta_limits_rad": self._delta_limits[0].detach().cpu().tolist(),
            "soft_joint_limits_rad": self._asset.data.soft_joint_pos_limits[0, self._joint_ids].detach().cpu().tolist(),
            "normalized_action_range": [-1.0, 1.0],
            "control_dt": self._control_dt,
            "decimation": self._decimation,
            "target_rule": "measured_q_plus_delta_once_per_control_step_hold_substeps",
            "joint_limit_rule": "clamp_target_to_soft_joint_pos_limits",
        }


@configclass
class BulbJointPositionActionCfg(JointActionCfg):
    """Custom bulb action limits; independent of the shared D-Peg robot mixin."""

    class_type: type = BulbJointPositionAction
    asset_name: str = "robot"
    joint_names: list[str] = [".*"]
    scale: dict[str, float] = {"joint[1-7]": ARM_DELTA_LIMIT, "right_.*_joint": HAND_DELTA_LIMIT}
    offset: float = 0.0
    preserve_order: bool = False
