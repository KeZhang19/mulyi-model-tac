"""Contact-gated hover and insertion shaping for the tabletop assembly task.

These terms only read physics. The hover latch belongs to the reward manager,
not the assembly success ledger, and is cleared on full or partial resets.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

from . import assembly
from .env_cfg import rotate_bulb_finger_forces


@configclass
class AssemblyGuidanceCfg:
    """Shared geometry and soft-gate scales for both sequential reward terms."""

    hover_clearance_m: float = 0.10
    hover_axial_tolerance_m: float = 0.01
    hover_radial_tolerance_m: float = assembly.RADIAL_TOLERANCE_M
    hover_axis_tolerance_rad: float = assembly.AXIS_TOLERANCE_RAD
    contact_force_scale_n: float = 1.0
    position_std_m: float = 0.2
    # Map the full free-space descent to the original reward distance. This
    # keeps feedback at a higher hover from saturating tanh while preserving
    # the physical-depth reward scale once the thread reaches the mouth.
    descent_reward_length_m: float = 0.04
    insertion_std_m: float = 0.04
    radial_gate_std_m: float = 0.012
    axis_gate_std_rad: float = 0.35


def assembly_soft_grasp_gate(env, force_scale_n: float = 1.0) -> torch.Tensor:
    """Continuous thumb AND best-other-finger contact, using bulb-only forces.

    Zero on either side gives zero credit. Both sides at the force scale give
    full credit; increasing force beyond that cannot increase the reward.
    """
    if not math.isfinite(force_scale_n) or force_scale_n <= 0.0:
        raise ValueError("force_scale_n must be positive and finite")
    forces = rotate_bulb_finger_forces(env)
    opposing_force = torch.minimum(forces[:, 0], forces[:, 1:].amax(dim=-1))
    return (opposing_force / force_scale_n).clamp(0.0, 1.0)


def assembly_thread_downward(env) -> torch.Tensor:
    """Global axis-only reward: bulb local -X should point along world -Z.

    The shell lies along local +X and the screw tip along -X. Rewarding only
    the axis leaves rotation about the thread completely unconstrained.
    There is deliberately no grasp, lift or phase gate on this term.
    """
    quat = env.scene["object"].data.root_quat_w
    axis = torch.zeros((env.num_envs, 3), device=env.device, dtype=quat.dtype)
    axis[:, 0] = -1.0
    thread_direction = assembly.quat_apply(quat, axis)
    reward = (0.5 * (1.0 - thread_direction[:, 2])).clamp(0.0, 1.0)
    valid = torch.isfinite(quat).all(dim=-1) & (quat.norm(dim=-1) > 1.0e-6)
    return torch.where(valid, reward, torch.zeros_like(reward))


class AssemblyGuidanceReward(ManagerTermBase):
    """Track above the mouth, latch full position credit, then guide insertion.

    Each component owns an identical hover latch and reads the same shared
    config/physical state. This avoids reward-order dependencies or a reset
    event mutating another term's history. A latch never changes the physical
    ALIGNED/ENGAGED flags or the independent success test.

    Values are reward rates in [0, 1]; RewardManager applies weight * step_dt.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.params.get("component") not in ("position", "insertion"):
            raise ValueError("component must be 'position' or 'insertion'")
        self.guidance = env.cfg.assembly_guidance
        for name in (
            "hover_clearance_m", "hover_axial_tolerance_m", "hover_radial_tolerance_m",
            "hover_axis_tolerance_rad", "contact_force_scale_n", "position_std_m",
            "descent_reward_length_m", "insertion_std_m", "radial_gate_std_m", "axis_gate_std_rad",
        ):
            value = getattr(self.guidance, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"assembly_guidance.{name} must be positive and finite")
        if self.guidance.hover_axial_tolerance_m >= self.guidance.hover_clearance_m:
            raise ValueError("hover_axial_tolerance_m must be smaller than hover_clearance_m")
        self.hover_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.hover_reached[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env, component: str):
        state = getattr(env, "_bulb_assembly_state", None)
        if state is None:
            return torch.zeros(env.num_envs, device=env.device)
        state.update(env)
        cfg = self.guidance
        grasp = assembly_soft_grasp_gate(env, cfg.contact_force_scale_n)
        # depth=0 is first thread entry; negative depth is above the mouth.
        # Socket +X maps to world +Z for this asset. Use the measured socket
        # frame, so translated/cloned environments have identical guidance.
        axial_error = state.depth + cfg.hover_clearance_m
        finite = torch.isfinite(state.depth) & torch.isfinite(state.radial_error) & torch.isfinite(state.axis_error)
        reached_now = (
            finite & state.lifted & (grasp >= 1.0)
            & (axial_error.abs() < cfg.hover_axial_tolerance_m)
            & (state.radial_error < cfg.hover_radial_tolerance_m)
            & (state.axis_error < cfg.hover_axis_tolerance_rad)
        )
        self.hover_reached |= reached_now

        if component == "position":
            distance = torch.sqrt(axial_error.square() + state.radial_error.square())
            tracking = grasp * (1.0 - torch.tanh(distance / cfg.position_std_m))
            # Full constant credit removes the incentive to remain above the
            # socket or to climb back up while descending to engage the thread.
            reward = torch.where(self.hover_reached, torch.ones_like(tracking), tracking)
        elif component == "insertion":
            alignment = torch.exp(
                -(state.radial_error / cfg.radial_gate_std_m).square()
                -(state.axis_error / cfg.axis_gate_std_rad).square()
            )
            approach = (cfg.hover_clearance_m + state.depth).clamp(0.0, cfg.hover_clearance_m)
            # Before engagement, reward approaching the mouth. Below it,
            # axial travel must have matching signed bulb-relative rotation:
            # downward pushing, free spinning and reverse turning cannot add
            # insertion credit. No hard ENGAGED threshold blocks early turns.
            rotation_depth = (-state.relative_angle * assembly.PITCH_M / (2.0 * math.pi)).clamp(
                0.0, assembly.INSTALLATION_DEPTH_M,
            )
            measured_depth = state.depth.clamp(0.0, assembly.INSTALLATION_DEPTH_M)
            screw_gate = torch.exp(-(state.helical_error / assembly.HELICAL_TOLERANCE_M).square())
            screw_gate = torch.nan_to_num(screw_gate, nan=0.0, posinf=0.0, neginf=0.0)
            screw_gate *= (state.tracking_thread & state.last_contact).float()
            insertion = torch.minimum(measured_depth, rotation_depth) * screw_gate
            overtravel = (state.depth - assembly.INSTALLATION_DEPTH_M).clamp_min(0.0)
            descent_remaining = (
                (cfg.hover_clearance_m - approach) / cfg.hover_clearance_m
                * cfg.descent_reward_length_m
            )
            remaining = descent_remaining + assembly.INSTALLATION_DEPTH_M - insertion + overtravel
            reward = (
                self.hover_reached.float() * grasp * alignment
                * (1.0 - torch.tanh(remaining / cfg.insertion_std_m))
            )
        else:
            raise ValueError("component must be 'position' or 'insertion'")
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        return torch.where(finite, reward, torch.zeros_like(reward))
