"""Numerical regression tests for Repose without an Isaac Sim installation."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/repose_training_math.py"
)
SPEC = importlib.util.spec_from_file_location("repose_training_math", MODULE_PATH)
repose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repose)


def _generator(seed=17):
    return torch.Generator().manual_seed(seed)


def _axis_angle(angle, axis=(0.0, 0.0, 1.0)):
    axis = torch.tensor(axis, dtype=torch.float64)
    axis /= torch.linalg.vector_norm(axis)
    return torch.cat((torch.tensor([math.cos(angle / 2)], dtype=torch.float64), axis * math.sin(angle / 2)))


def _reward(previous, current, **kwargs):
    current = torch.as_tensor(current, dtype=torch.float64)
    defaults = dict(
        position_distance=torch.zeros_like(current),
        applied_actions=torch.zeros(current.shape + (12,), dtype=torch.float64),
        previous_actions=torch.zeros(current.shape + (12,), dtype=torch.float64),
        reached=torch.zeros_like(current, dtype=torch.bool),
        fallen=torch.zeros_like(current, dtype=torch.bool),
        progress_scale=10.0,
        orientation_scale=0.25,
        position_scale=-5.0,
        position_deadzone=0.04,
        action_scale=-0.01,
        action_delta_scale=-0.02,
        success_bonus=25.0,
        fall_penalty=-10.0,
    )
    defaults.update(kwargs)
    return repose.compute_repose_reward_terms(torch.as_tensor(previous, dtype=torch.float64), current, **defaults)


def _hold(hold_steps, **kwargs):
    hold_steps = torch.as_tensor(hold_steps, dtype=torch.long)
    values = dict(
        angle=torch.full(hold_steps.shape, 0.1),
        position_distance=torch.full(hold_steps.shape, 0.02),
        linear_speed=torch.zeros(hold_steps.shape),
        angular_speed=torch.zeros(hold_steps.shape),
        hold_steps=hold_steps,
        angle_tolerance=0.2,
        position_tolerance=0.08,
        max_linear_speed=0.2,
        max_angular_speed=2.0,
        required_hold_steps=3,
        fallen=torch.zeros(hold_steps.shape, dtype=torch.bool),
    )
    values.update(kwargs)
    return repose.update_stable_success(**values)


def test_joint_resets_are_legal_symmetric_and_keep_boundary_grasp():
    count = 100_000
    default = torch.tensor([0.0, 0.3, 0.95, 1.01, -1.02], dtype=torch.float64).repeat(count, 1)
    lower, upper = torch.full_like(default, -1.0), torch.ones_like(default)
    samples = repose.sample_bounded_joint_positions(default, lower, upper, 0.8, generator=_generator())
    assert torch.all(samples >= lower)
    assert torch.all(samples <= upper)
    center = default[0].clamp(-1.0, 1.0)
    torch.testing.assert_close(samples.mean(0), center, atol=0.003, rtol=0.0)
    positive_fraction = (samples[:, :3] > center[:3]).double().mean(0)
    torch.testing.assert_close(positive_fraction, torch.full((3,), 0.5, dtype=torch.float64), atol=0.005, rtol=0.0)
    assert torch.all(samples[:, 3] == 1.0)
    assert torch.all(samples[:, 4] == -1.0)
    assert samples[:, 0].std() > 0.4


def test_zero_reset_noise_projects_only_illegal_defaults():
    default = torch.tensor([[0.25, -2.0, 2.0]])
    samples = repose.sample_bounded_joint_positions(default, torch.tensor([-1.0]), torch.tensor([1.0]), 0.0)
    torch.testing.assert_close(samples, torch.tensor([[0.25, -1.0, 1.0]]))


@pytest.mark.parametrize("angle", [0.0, 1.0e-8, 0.2, math.pi / 2, math.pi - 1.0e-8, math.pi])
def test_quaternion_distance_preserves_small_and_large_angles_and_sign(angle):
    identity = _axis_angle(0.0)
    quaternion = _axis_angle(angle, (1.0, -2.0, 3.0))
    for sign in (-1.0, 1.0):
        actual = repose.quaternion_angle_distance(quaternion * sign * 1.0001, identity * 0.9999)
        assert actual.item() == pytest.approx(angle, abs=1.0e-12)


def test_relative_goals_are_relative_to_actual_object_and_cover_the_curriculum():
    count = 20_000
    objects = _axis_angle(2.2, (1.0, 2.0, -0.3)).repeat(count, 1)
    minimum, maximum = math.radians(20.0), math.radians(45.0)
    goals = repose.sample_relative_goal_quaternions(objects, minimum, maximum, generator=_generator())
    error = repose.quaternion_angle_distance(objects, goals)
    assert error.min() >= minimum - 1.0e-12
    assert error.max() <= maximum + 1.0e-12
    assert error.mean().item() == pytest.approx((minimum + maximum) / 2, abs=0.003)
    torch.testing.assert_close(torch.linalg.vector_norm(goals, dim=-1), torch.ones(count, dtype=goals.dtype))
    assert repose.quaternion_angle_distance(_axis_angle(0.0), goals).mean() > 1.0


def test_goal_axes_are_unbiased_and_full_rotation_limit_is_valid():
    count = 30_000
    objects = _axis_angle(0.0).repeat(count, 1)
    goals = repose.sample_relative_goal_quaternions(objects, math.pi, math.pi, generator=_generator())
    axes = goals[:, 1:]
    torch.testing.assert_close(axes.mean(0), torch.zeros(3, dtype=axes.dtype), atol=0.012, rtol=0.0)
    torch.testing.assert_close(axes.square().mean(0), torch.full((3,), 1 / 3, dtype=axes.dtype), atol=0.008, rtol=0.0)
    torch.testing.assert_close(repose.quaternion_angle_distance(objects, goals), torch.full((count,), math.pi, dtype=goals.dtype))


@pytest.mark.parametrize("minimum, maximum", [(-0.1, 0.5), (0.5, 0.4), (0.2, math.pi + 0.1)])
def test_invalid_goal_curriculum_is_rejected(minimum, maximum):
    with pytest.raises(ValueError):
        repose.sample_relative_goal_quaternions(_axis_angle(0.0), minimum, maximum)


def test_signed_progress_cancels_backtracking_and_uses_new_target_baseline():
    approach = _reward([1.0], [0.5])["progress"]
    retreat = _reward([0.5], [1.0])["progress"]
    torch.testing.assert_close(approach + retreat, torch.zeros_like(approach))
    # Goal A ended at 0.1 rad. Sampling B at 0.7 rad is not a physical transition:
    # reset the baseline to 0.7, then only the subsequent 0.7 -> 0.6 counts.
    transition_without_motion = _reward([0.7], [0.7])["progress"]
    approach_new_goal = _reward([0.7], [0.6])["progress"]
    assert transition_without_motion.item() == 0.0
    assert approach_new_goal.item() == pytest.approx(1.0)


def test_position_deadzone_allows_regrasp_without_losing_position_reward():
    result = _reward([1.0] * 4, [1.0] * 4, position_distance=torch.tensor([0.0, 0.02, 0.04, 0.06]))
    torch.testing.assert_close(result["position"], torch.tensor([0.0, 0.0, 0.0, -0.1]), atol=1.0e-7, rtol=0.0)


def test_reward_terms_are_bounded_scaled_and_add_up_with_mean_action_cost():
    result = _reward(
        [1.0, math.pi], [0.0, math.pi],
        applied_actions=torch.tensor([[1.0] * 12, [0.5] * 12], dtype=torch.float64),
        previous_actions=torch.tensor([[0.0] * 12, [0.5] * 12], dtype=torch.float64),
        reached=torch.tensor([True, True]), fallen=torch.tensor([False, True]),
    )
    torch.testing.assert_close(result["orientation"], torch.tensor([0.25, 0.0], dtype=torch.float64))
    torch.testing.assert_close(result["action"], torch.tensor([-0.01, -0.0025], dtype=torch.float64))
    torch.testing.assert_close(result["action_delta"], torch.tensor([-0.02, 0.0], dtype=torch.float64))
    torch.testing.assert_close(result["success"], torch.tensor([25.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(result["fall"], torch.tensor([0.0, -10.0], dtype=torch.float64))
    torch.testing.assert_close(result["total"], sum(value for key, value in result.items() if key != "total"))


def test_stable_success_requires_three_steps_and_pulses_once_without_overflow():
    count, reached = _hold([0])
    assert count.item() == 1 and not reached.item()
    count, reached = _hold(count)
    assert count.item() == 2 and not reached.item()
    count, reached = _hold(count)
    assert count.item() == 3 and reached.item()
    for _ in range(10):
        count, reached = _hold(count)
        assert count.item() == 3 and not reached.item()
    count, reached = _hold([torch.iinfo(torch.long).max])
    assert count.item() == 3 and not reached.item()


@pytest.mark.parametrize(
    "invalid",
    [dict(angle=torch.tensor([0.201])), dict(position_distance=torch.tensor([0.08])),
     dict(linear_speed=torch.tensor([0.201])), dict(angular_speed=torch.tensor([2.01])),
     dict(fallen=torch.tensor([True]))],
)
def test_instability_or_fall_clears_success_hold(invalid):
    count, reached = _hold([2], **invalid)
    assert count.item() == 0 and not reached.item()
    count, reached = _hold(count)
    assert count.item() == 1 and not reached.item()


def test_success_state_is_independent_per_environment():
    count, reached = _hold([2, 2, 0, 3], fallen=torch.tensor([False, True, False, False]))
    assert count.tolist() == [3, 0, 1, 3]
    assert reached.tolist() == [True, False, False, False]


def test_curriculum_consumes_full_windows_and_advances_at_most_one_stage():
    config = dict(stage_count=5, minimum_episodes=100, success_threshold=0.6)
    assert repose.advance_curriculum(0, 99, 99, **config) == (0, False)
    assert repose.advance_curriculum(0, 100, 59, **config) == (0, True)
    assert repose.advance_curriculum(0, 100, 60, **config) == (1, True)
    assert repose.advance_curriculum(0, 1000, 1000, **config) == (1, True)
    assert repose.advance_curriculum(4, 100, 100, **config) == (4, True)


def test_curriculum_success_count_means_episodes_not_repeated_goals():
    with pytest.raises(ValueError, match="subset"):
        repose.advance_curriculum(0, 100, 101, stage_count=5, minimum_episodes=100, success_threshold=0.6)
