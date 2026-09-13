"""Simulator-independent reset, reward, and curriculum math for Repose training.

All angles are in radians and all quaternions use Isaac Lab's ``wxyz`` order.
Reward scales multiply their terms directly: penalty scales should be negative.
The environment owns target changes, per-episode state, and distributed counts.
"""

from __future__ import annotations

import math

import torch


def sample_bounded_joint_positions(
    default_pos: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    noise_fraction: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Perturb a valid default grasp symmetrically without crossing joint limits.

    ``noise_fraction`` is a fraction of the smaller distance to either limit,
    in [0, 1]. A default outside the limits is first projected into the valid
    interval. Joints resting on a limit remain there; clipping random samples
    instead would systematically shift their mean away from the default grasp.
    """
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("noise_fraction must be between zero and one")
    center = torch.maximum(torch.minimum(default_pos, upper), lower)
    radius = torch.minimum(center - lower, upper - center).clamp_min(0.0) * noise_fraction
    noise = torch.rand(
        center.shape, dtype=center.dtype, device=center.device, generator=generator
    ) * 2.0 - 1.0
    return torch.maximum(torch.minimum(center + radius * noise, upper), lower)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for batched ``wxyz`` quaternions."""
    left, right = torch.broadcast_tensors(left, right)
    scalar = left[..., :1] * right[..., :1] - (left[..., 1:] * right[..., 1:]).sum(-1, keepdim=True)
    vector = (
        left[..., :1] * right[..., 1:]
        + right[..., :1] * left[..., 1:]
        + torch.cross(left[..., 1:], right[..., 1:], dim=-1)
    )
    return torch.cat((scalar, vector), dim=-1)


def quaternion_angle_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return the shortest orientation error in [0, pi], invariant to q/-q.

    atan2 remains accurate for almost identical orientations, where acos(dot)
    loses precision. Inputs may have minor normalization error from simulation.
    """
    first = torch.nn.functional.normalize(first, dim=-1)
    second = torch.nn.functional.normalize(second, dim=-1)
    conjugate = torch.cat((second[..., :1], -second[..., 1:]), dim=-1)
    relative = _quaternion_multiply(first, conjugate)
    return 2.0 * torch.atan2(torch.linalg.vector_norm(relative[..., 1:], dim=-1), relative[..., 0].abs())


def sample_relative_goal_quaternions(
    object_quat: torch.Tensor,
    min_angle: float,
    max_angle: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample relative goals with a uniform sphere axis and uniform angle.

    The sampled rotation is applied in the world frame to the *current* object
    orientation. Call this after object reset, then initialize the progress
    baseline against the returned goal. The distribution intentionally samples
    angles uniformly within the curriculum, rather than uniformly on SO(3).
    """
    if not 0.0 <= min_angle <= max_angle <= math.pi:
        raise ValueError("goal angles must satisfy 0 <= min_angle <= max_angle <= pi")
    axis = torch.randn(
        object_quat.shape[:-1] + (3,),
        dtype=object_quat.dtype,
        device=object_quat.device,
        generator=generator,
    )
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    fallback = torch.zeros_like(axis)
    fallback[..., 0] = 1.0
    axis = torch.where(norm > 1.0e-8, axis / norm.clamp_min(1.0e-8), fallback)
    angle = torch.rand(
        object_quat.shape[:-1] + (1,),
        dtype=object_quat.dtype,
        device=object_quat.device,
        generator=generator,
    ) * (max_angle - min_angle) + min_angle
    delta = torch.cat((torch.cos(angle * 0.5), axis * torch.sin(angle * 0.5)), dim=-1)
    return torch.nn.functional.normalize(
        _quaternion_multiply(delta, torch.nn.functional.normalize(object_quat, dim=-1)), dim=-1
    )


def update_stable_success(
    angle: torch.Tensor,
    position_distance: torch.Tensor,
    linear_speed: torch.Tensor,
    angular_speed: torch.Tensor,
    hold_steps: torch.Tensor,
    *,
    angle_tolerance: float,
    position_tolerance: float,
    max_linear_speed: float,
    max_angular_speed: float,
    required_hold_steps: int,
    fallen: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require consecutive stable steps; emit one pulse on reaching the hold.

    Counters saturate at the hold requirement. The environment changes the goal
    and clears its counter after a pulse. Instability or a fall clears the hold,
    including a fall on what would otherwise be the successful final step.
    """
    if required_hold_steps < 1:
        raise ValueError("required_hold_steps must be positive")
    stable = (
        (angle <= angle_tolerance)
        & (position_distance < position_tolerance)
        & (linear_speed <= max_linear_speed)
        & (angular_speed <= max_angular_speed)
        & ~fallen.bool()
    )
    new_hold = torch.where(
        stable,
        hold_steps.clamp(max=required_hold_steps - 1) + 1,
        torch.zeros_like(hold_steps),
    )
    reached = stable & (hold_steps < required_hold_steps) & (new_hold >= required_hold_steps)
    return new_hold, reached


def compute_repose_reward_terms(
    previous_angle: torch.Tensor,
    current_angle: torch.Tensor,
    position_distance: torch.Tensor,
    applied_actions: torch.Tensor,
    previous_actions: torch.Tensor,
    reached: torch.Tensor,
    fallen: torch.Tensor,
    *,
    progress_scale: float,
    orientation_scale: float,
    position_scale: float,
    position_deadzone: float,
    action_scale: float,
    action_delta_scale: float,
    success_bonus: float,
    fall_penalty: float,
) -> dict[str, torch.Tensor]:
    """Return weighted per-environment terms and their total.

    Progress is signed: backing away cancels earlier approach reward. Both angle
    inputs MUST refer to the same goal. On every target/object reset, initialize
    the previous angle to the error for the new object and new goal before the
    next transition. This function never rewards an instantaneous target change.

    Action terms use mean squared *applied normalized actions*, so their scale
    does not grow with the number of joints. A bounded policy owns latent action
    limits; latent Gaussian values must not be substituted for applied actions.
    """
    progress = progress_scale * (previous_angle - current_angle)
    orientation = orientation_scale * (1.0 - current_angle / math.pi).clamp(0.0, 1.0)
    position = position_scale * (position_distance - position_deadzone).clamp_min(0.0)
    action = action_scale * applied_actions.square().mean(dim=-1)
    action_delta = action_delta_scale * (applied_actions - previous_actions).square().mean(dim=-1)
    success = success_bonus * (reached.bool() & ~fallen.bool()).to(current_angle.dtype)
    fall = fall_penalty * fallen.to(current_angle.dtype)
    return {
        "progress": progress,
        "orientation": orientation,
        "position": position,
        "action": action,
        "action_delta": action_delta,
        "success": success,
        "fall": fall,
        "total": progress + orientation + position + action + action_delta + success + fall,
    }


def advance_curriculum(
    stage: int,
    completed_episodes: int,
    successful_episodes: int,
    *,
    stage_count: int,
    minimum_episodes: int,
    success_threshold: float,
) -> tuple[int, bool]:
    """Evaluate one completed global episode window and optionally advance.

    Returns ``(new_stage, reset_window)``. A fully observed window is consumed
    whether it passes or fails, avoiding an ever-growing lifetime average that
    hides recent learning. Only one stage can be advanced per window. Distributed
    environments must supply all-reduced episode counts on every rank.
    """
    if stage_count < 1 or not 0 <= stage < stage_count:
        raise ValueError("stage must index a nonempty curriculum")
    if minimum_episodes < 1 or not 0.0 <= success_threshold <= 1.0:
        raise ValueError("invalid curriculum window or success threshold")
    if completed_episodes < 0 or not 0 <= successful_episodes <= completed_episodes:
        raise ValueError("successful episodes must be a subset of completed episodes")
    if completed_episodes < minimum_episodes:
        return stage, False
    passed = successful_episodes / completed_episodes >= success_threshold
    return min(stage + int(passed), stage_count - 1), True
