"""Task-owned position residuals around the calibrated, preloaded D-peg grasp."""

import math

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class DPegPregraspResidualAction(JointPositionAction):
    """Apply ``q_target = calibrated_drive_target + scale * action``.

    All 28 joints share this non-integrating reference. Zero actions preserve
    the gravity and finger preload from physical calibration; measured joint
    deflection never changes that reference. A policy can translate the arm
    and adjust or release its grasp by predicting a residual. This does not
    constrain the free peg or add forces beyond the existing joint drives.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        targets = env.cfg._d_peg_pregrasp["robot_joint_targets"]
        if self.action_dim != 28 or set(self._joint_names) != set(targets):
            raise ValueError("D-peg residual control requires all 28 calibrated joints")
        reference = [float(targets[name]) for name in self._joint_names]
        if not all(math.isfinite(value) for value in reference):
            raise ValueError("D-peg drive targets must be finite")
        # Do not substitute default_joint_pos: measured rest positions and
        # calibrated drive targets differ by the required grasp preload.
        self._offset = torch.tensor(reference, device=self.device).expand(self.num_envs, -1)
        self._processed_actions[:] = self._offset

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = self._offset[ids]

    def action_contract(self):
        """Describe actual runtime action ordering and anchors for checkpoint validation."""
        return {
            "schema_version": 2,
            "type": "pregrasp_target_residual_position",
            "formula": "q_target = q_pregrasp_drive_target + scale * action",
            "joint_order": list(self._joint_names),
            "reference_targets_rad": self._offset[0].detach().cpu().tolist(),
            "scale": self.cfg.scale,
            "clip": self.cfg.clip,
            "integrates_actions": False,
        }


@configclass
class DPegPregraspResidualActionCfg(JointPositionActionCfg):
    class_type = DPegPregraspResidualAction
    use_default_offset: bool = False
    offset: float = 0.0

