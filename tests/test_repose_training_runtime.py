"""Exercise real Repose runtime transitions with small CPU simulator doubles."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from collections.abc import Sequence

import pytest
import torch


DIRECT_ROOT = (Path(__file__).resolve().parents[1]
               / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct")


@pytest.fixture
def runtime_module(monkeypatch):
    class DirectRLEnv:
        def _reset_idx(self, env_ids):
            self.base_resets.append(env_ids.clone())
            self.episode_length_buf[env_ids] = 0

    isaaclab = ModuleType("isaaclab")
    envs = ModuleType("isaaclab.envs")
    envs.DirectRLEnv = DirectRLEnv
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", envs)
    package = ModuleType("_repose_runtime_test_package")
    package.__path__ = [str(DIRECT_ROOT)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    name = package.__name__ + ".repose_training_runtime"
    spec = importlib.util.spec_from_file_location(name, DIRECT_ROOT / "repose_training_runtime.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    return module


class _Hand:
    def __init__(self, n):
        self.data = SimpleNamespace(default_joint_pos=torch.tensor([0.2, -0.3, 1.02]).repeat(n, 1))
        self._ALL_INDICES = torch.arange(n)
        self.targets = []
        self.states = []

    def set_joint_position_target(self, value, **kwargs):
        self.targets.append((value.clone(), kwargs))

    def write_joint_state_to_sim(self, position, velocity, env_ids):
        self.states.append((position.clone(), velocity.clone(), env_ids.clone()))


class _Object:
    def __init__(self, n):
        self.data = SimpleNamespace(default_root_state=torch.zeros(n, 13))
        self.data.default_root_state[:, 3] = 1.0
        self.state = self.data.default_root_state.clone()

    def write_root_pose_to_sim(self, pose, env_ids):
        self.state[env_ids, :7] = pose

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.state[env_ids, 7:] = velocity


class _Markers:
    """Match VisualizationMarkers' argument order and tensor shape contract."""

    def __init__(self):
        self.calls = []

    def visualize(self, translations=None, orientations=None, scales=None, marker_indices=None):
        if translations is not None:
            assert translations.ndim == 2 and translations.shape[1] == 3
        if orientations is not None:
            assert orientations.ndim == 2 and orientations.shape[1] == 4
        if scales is not None:
            assert scales.ndim == 2 and scales.shape[1] == 3
        if marker_indices is not None:
            assert marker_indices.ndim == 1
            assert marker_indices.shape[0] == translations.shape[0]
        self.calls.append((translations, orientations, scales, marker_indices))


class _Env:
    def __init__(self, n=3):
        self.cfg = SimpleNamespace(
            repose_action_filter=0.5, repose_action_speed=1.5,
            repose_position_noise=0.0, repose_object_rotation_noise_deg=5.0,
            repose_joint_noise_fraction=0.1, repose_goal_min_deg=20.0,
            repose_goal_max_degrees=(45.0, 90.0, 180.0),
            repose_success_position=0.08, repose_success_linear_speed=0.2,
            repose_success_angular_speed=2.0, repose_success_hold_steps=3,
            success_tolerance=0.2, fall_dist=0.24,
            repose_progress_scale=10.0, repose_orientation_scale=0.25,
            repose_position_scale=-5.0, repose_position_deadzone=0.04,
            repose_action_scale=-0.01, repose_action_delta_scale=-0.02,
            repose_success_bonus=25.0, repose_fall_penalty=-10.0,
            repose_curriculum_enabled=True, repose_curriculum_interval=10,
            repose_curriculum_min_episodes=4, repose_curriculum_success_threshold=0.5,
        )
        self.device, self.num_envs, self.step_dt = "cpu", n, 0.03
        self.actions = torch.zeros(n, 2)
        self.hand_dof_targets = torch.zeros(n, 3)
        self.prev_targets = self.hand_dof_targets.clone()
        self.cur_targets = self.hand_dof_targets.clone()
        self.actuated_dof_indices = [0, 2]
        self.hand_dof_lower_limits = torch.full((n, 3), -1.0)
        self.hand_dof_upper_limits = torch.ones(n, 3)
        self.hand = _Hand(n)
        self.object = _Object(n)
        self.scene = SimpleNamespace(env_origins=torch.arange(n).float()[:, None].expand(n, 3).clone())
        self.object.state[:, :3] += self.scene.env_origins
        self.in_hand_pos = torch.zeros(n, 3)
        self.goal_pos = torch.zeros(n, 3)
        self.goal_rot = torch.zeros(n, 4)
        self.goal_rot[:, 0] = 1.0
        self.goal_markers = _Markers()
        self.successes = torch.zeros(n)
        self.reset_goal_buf = torch.zeros(n, dtype=torch.bool)
        self.reset_buf = torch.zeros(n, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(n, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.common_step_counter = 0
        self.base_resets = []
        self.extras = {}
        self._compute_intermediate_values()

    def _compute_intermediate_values(self):
        self.object_pos = self.object.state[:, :3] - self.scene.env_origins
        self.object_rot = self.object.state[:, 3:7]
        self.object_linvel = self.object.state[:, 7:10]
        self.object_angvel = self.object.state[:, 10:13]


def _runtime(module, n=3):
    env = _Env(n)
    runtime = module.ReposeTrainingRuntime(env)
    runtime.reset(torch.arange(n))
    return env, runtime


def _step(runtime):
    runtime.env.common_step_counter += 1
    return runtime.rewards()


def test_action_integration_happens_once_before_all_physics_substeps(runtime_module):
    env, runtime = _runtime(runtime_module)
    before = env.cur_targets.clone()
    actions = torch.tensor([[0.8, -0.6]]).repeat(env.num_envs, 1)
    runtime.prepare_action(actions)
    target = env.cur_targets.clone()
    expected = before[:, env.actuated_dof_indices] + 1.5 * 0.03 * 0.5 * actions
    torch.testing.assert_close(target[:, env.actuated_dof_indices], expected)
    for _ in range(6):
        runtime.apply_action()
        torch.testing.assert_close(env.cur_targets, target)
    torch.testing.assert_close(env.cur_targets[:, 1], before[:, 1])
    _step(runtime)
    torch.testing.assert_close(runtime.episode_steps, torch.ones(env.num_envs))


def test_initial_and_partial_resets_use_new_object_pose_and_isolate_episode_rows(runtime_module):
    env, runtime = _runtime(runtime_module)
    runtime.prepare_action(torch.full_like(env.actions, 0.4))
    _step(runtime)
    untouched = {"target": env.cur_targets[1].clone(), "goal": env.goal_rot[1].clone(),
                 "return": runtime.episode_returns[1].clone(), "action": runtime.filtered_actions[1].clone()}
    # The old object pose is deliberately unrelated to the new default pose.
    env.object.state[0, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    runtime.reset([0, 2])
    assert len(env.base_resets) == 2
    torch.testing.assert_close(env.cur_targets[1], untouched["target"])
    torch.testing.assert_close(env.goal_rot[1], untouched["goal"])
    torch.testing.assert_close(runtime.episode_returns[1], untouched["return"])
    torch.testing.assert_close(runtime.filtered_actions[1], untouched["action"])
    assert runtime.episode_steps.tolist() == [0.0, 1.0, 0.0]
    assert torch.count_nonzero(runtime.episode_returns[[0, 2]]) == 0
    assert torch.count_nonzero(runtime.filtered_actions[[0, 2]]) == 0
    assert torch.all(env.repose_reset_joint_positions >= env.hand_dof_lower_limits)
    assert torch.all(env.repose_reset_joint_positions <= env.hand_dof_upper_limits)
    angle = runtime_module.quaternion_angle_distance(env.object_rot, env.goal_rot)
    torch.testing.assert_close(env.repose_previous_angle[[0, 2]], angle[[0, 2]])
    assert angle[[0, 2]].min() >= math.radians(20.0) - 1e-6
    assert angle[[0, 2]].max() <= math.radians(45.0) + 1e-6
    before = len(env.base_resets)
    runtime.reset([])
    assert len(env.base_resets) == before


def test_success_changes_target_without_rewarding_the_target_change(runtime_module):
    env, runtime = _runtime(runtime_module, n=1)
    env.goal_rot[:] = env.object_rot
    env.repose_previous_angle.zero_()
    env.repose_hold_steps[:] = 2
    reward = _step(runtime)
    assert env.successes.item() == 1
    assert env.repose_last_reached.item()
    assert env.repose_hold_steps.item() == 0
    assert reward.item() == pytest.approx(25.25)
    fresh_angle = runtime_module.quaternion_angle_distance(env.object_rot, env.goal_rot)
    torch.testing.assert_close(env.repose_previous_angle, fresh_angle)
    _step(runtime)
    assert env.repose_reward_terms["progress"].item() == 0.0
    assert env.repose_reward_terms["success"].item() == 0.0
    assert env.successes.item() == 1


def test_global_episode_statistics_count_resets_once_and_include_terminal_rewards(runtime_module, monkeypatch):
    env, runtime = _runtime(runtime_module)
    env.cfg.repose_curriculum_enabled = False
    calls = []

    def identical_second_rank(packed):
        calls.append(packed.clone())
        packed.mul_(2)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", identical_second_rank)
    first = _step(runtime).clone()
    env.successes[:] = torch.tensor([2.0, 0.0, 0.0])
    env.object.state[1, 0] += 0.25
    env._compute_intermediate_values()
    env.reset_buf[:] = torch.tensor([True, True, False])
    env.reset_time_outs[:] = torch.tensor([True, False, False])
    second = _step(runtime).clone()
    expected_sum = 2 * (first[:2].sum() + second[:2].sum())
    assert runtime.completed.item() == 4
    assert runtime.successful.item() == 2
    assert runtime.falls.item() == 2
    assert runtime.timeouts.item() == 2
    assert runtime.target_hits.item() == 4
    torch.testing.assert_close(runtime.completed_returns, expected_sum)
    assert env.extras["log"]["Episode/success_rate"].item() == 0.5
    assert env.extras["log"]["Episode/targets_per_episode"].item() == 1.0
    torch.testing.assert_close(env.extras["log"]["Reward/total"], second.mean())
    torch.testing.assert_close(sum(runtime.completed_terms.values()), runtime.completed_returns)
    runtime.reset([0, 1])
    env.reset_buf.zero_()
    env.reset_time_outs.zero_()
    third = _step(runtime).clone()
    assert runtime.completed.item() == 4
    assert runtime.episode_steps.tolist() == [1.0, 1.0, 3.0]
    env.reset_buf[2] = True
    env.reset_time_outs[2] = True
    fourth = _step(runtime).clone()
    assert runtime.completed.item() == 6
    assert runtime.successful.item() == 2
    expected_sum += 2 * (first[2] + second[2] + third[2] + fourth[2])
    torch.testing.assert_close(runtime.completed_returns, expected_sum)
    torch.testing.assert_close(env.extras["log"]["Episode/return"], expected_sum / 6)
    assert len(calls) == 4


def test_reward_logs_remain_independent_after_steps_resets_and_curriculum_updates(runtime_module):
    env, runtime = _runtime(runtime_module)
    _step(runtime)
    retained = env.extras["log"]
    snapshot = {k: v.clone() for k, v in retained.items()}
    env.reset_buf[:] = True
    _step(runtime)
    assert env.extras["log"] is not retained
    runtime.reset(torch.arange(env.num_envs))
    runtime.completed.add_(100)
    runtime.window_completed.fill_(100)
    runtime.window_successful.fill_(100)
    env.common_step_counter = 10
    runtime.maybe_advance_curriculum()
    assert env.repose_curriculum_stage == 1
    for key, old_value in snapshot.items():
        torch.testing.assert_close(retained[key], old_value)


def test_curriculum_uses_global_episode_windows_and_state_restore_resets_goal_baseline(runtime_module, monkeypatch):
    env, runtime = _runtime(runtime_module, n=2)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda packed: packed.mul_(2))
    env.common_step_counter = 9
    env.reset_buf[:] = True
    env.reset_time_outs[:] = True
    env.successes[0] = 2  # Multiple targets in one successful episode still count once.
    _step(runtime)
    assert env.repose_curriculum_stage == 1
    assert runtime.completed.item() == 4
    assert runtime.successful.item() == 2
    assert runtime.window_completed.item() == 0
    runtime.window_completed.fill_(2)
    runtime.window_successful.fill_(1)
    state = runtime.state_dict()
    other_env, other = _runtime(runtime_module, n=2)
    other.load_state_dict(state)
    assert other_env.repose_curriculum_stage == 1
    assert other.window_completed.item() == 2
    assert other.window_successful.item() == 1
    angle = runtime_module.quaternion_angle_distance(other_env.object_rot, other_env.goal_rot)
    torch.testing.assert_close(other_env.repose_previous_angle, angle)


def test_goal_markers_use_marker_indices_argument_and_preserve_world_origins(runtime_module):
    env, runtime = _runtime(runtime_module)
    translation, orientation, scales, marker_indices = env.goal_markers.calls[-1]
    assert scales is None
    assert marker_indices.tolist() == [0, 0, 0, 1, 1, 1]
    torch.testing.assert_close(translation[:env.num_envs], env.goal_pos + env.scene.env_origins)
    torch.testing.assert_close(orientation[:env.num_envs], env.goal_rot)
    torch.testing.assert_close(orientation[env.num_envs:], env.goal_rot)


def test_curriculum_restore_invalidates_cached_history_without_mutating_published_observation(runtime_module):
    # Execute the production history class, including its copy-before-reset rule.
    policy_file = DIRECT_ROOT.parents[1] / "tactile_representation/policy.py"
    tree = ast.parse(policy_file.read_text())
    definitions = [node for node in tree.body if (
        isinstance(node, ast.ClassDef) and node.name == "TactileObservationHistory"
        or isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "FINGER_ORDER" for target in node.targets
        )
    )]
    namespace = {"torch": torch, "Sequence": Sequence}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(policy_file), "exec"), namespace)
    env, runtime = _runtime(runtime_module)
    env._history = namespace["TactileObservationHistory"](
        env.num_envs, state_dim=4, projection_dim=2, history_length=3, device=env.device,
    )
    returned_observation = env._history.append(torch.ones(env.num_envs, 4), torch.ones(env.num_envs, 5, 2))
    saved_observation = returned_observation.clone()
    assert not env._history.needs_fill.any()
    env._observation_cache_key = (env.common_step_counter, 11)
    runtime.load_state_dict({"revision": 2, "stage": 1, "window_completed": 2, "window_successful": 1})
    assert env._observation_cache_key is None
    assert env._history.needs_fill.all()
    torch.testing.assert_close(returned_observation, saved_observation)
    torch.testing.assert_close(
        env.repose_previous_angle,
        runtime_module.quaternion_angle_distance(env.object_rot, env.goal_rot),
    )


def test_legacy_environment_callbacks_keep_original_dispatch(runtime_module):
    class Legacy:
        def _pre_physics_step(self, actions):
            self.calls.append(("prepare", actions))

        def _apply_action(self):
            self.calls.append(("apply",))

        def _get_rewards(self):
            self.calls.append(("reward",))
            return "legacy reward"

        def _reset_target_pose(self, env_ids):
            self.calls.append(("goal", env_ids))

        def _reset_idx(self, env_ids):
            self.calls.append(("reset", env_ids))

    tree = ast.parse((DIRECT_ROOT / "visuotactile_inhand_manipulation_env.py").read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "VisuotactileInHandManipulationEnv")
    callback_names = {"_pre_physics_step", "_apply_action", "_get_rewards", "_reset_target_pose", "_reset_idx"}
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in callback_names]
    namespace = {"InHandManipulationEnv": Legacy, "torch": torch, "Sequence": Sequence}
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(DIRECT_ROOT), "exec"), namespace)
    env = namespace[original.name]()
    env._repose_training = None
    env.calls = []
    env.hand = SimpleNamespace(_ALL_INDICES=torch.arange(3))
    env._pre_physics_step(torch.zeros(3, 2))
    env._apply_action()
    assert env._get_rewards() == "legacy reward"
    env._reset_target_pose([1])
    env._reset_idx(None)
    assert [call[0] for call in env.calls] == ["prepare", "apply", "reward", "goal", "reset"]
    assert env.calls[-1][1].tolist() == [0, 1, 2]
