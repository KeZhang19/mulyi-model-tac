"""Reward-only phase split; physics, actions and observations stay unchanged."""
from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase

from ..RotateBulbCustom import env_cfg as original
from . import mdp


class ReleasedContactGuidance(ManagerTermBase):
    """Bounded dense grasp guidance immediately after physical detachment.

    Only bulb-filtered forces count. A single side earns at most one quarter
    of the rate; the thumb plus any opposing fingertip earns the full rate.
    No clearance/speed/hold prerequisite blocks the first post-release contact.
    The existing stable-grasp, transport and success terms remain stricter.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, device=env.device)
        self.credit = torch.zeros_like(self.paid)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = 0
        self.credit[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, force_scale=.5, reward_rate=1., max_reward=4.):
        quality = original.rotate_bulb_grasp_quality(env, force_scale)
        active = env._bulb_released & ~original.rotate_bulb_failed(env)
        quality = torch.where(active, quality, 0.)
        credit = original.rotate_bulb_bounded_credit(env, quality, self.paid, reward_rate, max_reward)
        fresh = self.last_step != env.common_step_counter
        self.paid += torch.where(fresh, credit, 0.)
        self.credit.copy_(torch.where(fresh, credit, self.credit))
        self.last_step[fresh] = env.common_step_counter
        env.extras.setdefault("log", {}).update({
            "Metrics/phase_contact/released_fraction": active.float().mean(),
            "Metrics/phase_contact/quality": quality.mean(),
            "Metrics/phase_contact/quality_when_released": quality.sum() / active.sum().clamp_min(1),
            "Metrics/phase_contact/budget_spent": self.paid.mean(),
        })
        return torch.where(active, self.credit / env.step_dt, 0.)


def prerelease_arm_motion(env, position_deadband=.05, position_scale=.15,
                          velocity_deadband=.15, velocity_scale=.6):
    """Damp arm speed before release even without an opposing grasp.

    Approach retains freedom to move away from its initial pose. Once a real
    grasp is latched, retain the existing drift cost relative to that grasp.
    After detachment this term is zero so the arm can transport the bulb.
    """
    if not all(math.isfinite(x) and x >= 0 for x in (position_deadband, velocity_deadband)):
        raise ValueError("arm motion deadbands must be finite and non-negative")
    if not all(math.isfinite(x) and x > 0 for x in (position_scale, velocity_scale)):
        raise ValueError("arm motion scales must be finite and positive")
    state = mdp.state(env)
    state.update()
    data = env.scene["robot"].data
    offset = data.joint_pos[:, state.arm_ids] - state.reference
    pose = ((offset.abs() - position_deadband).clamp_min(0) / position_scale).clamp_max(1).square().mean(-1)
    speed = ((data.joint_vel[:, state.arm_ids].abs() - velocity_deadband).clamp_min(0)
             / velocity_scale).clamp_max(1).square().mean(-1)
    cost = torch.nan_to_num(speed, nan=1.) + torch.where(state.grasped, torch.nan_to_num(pose, nan=1.), 0.)
    active = ~env._bulb_released & ~original.rotate_bulb_failed(env)
    return torch.where(active, cost, 0.)
