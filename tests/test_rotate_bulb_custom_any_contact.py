"""Any-finger auxiliary reward: target filtering, scale, phases and penalty balance."""

import ast
from types import SimpleNamespace

import pytest
import torch

from test_rotate_bulb_custom_turns import code, configured
from test_rotate_bulb_reward_balance import _task_env


FINGERS = ("thumb", "index", "mid", "ring", "pinky")


def clear_contact(env):
    for sensor in env.scene.sensors.values():
        sensor.data.force_matrix_w.zero_()
        # Large non-bulb contacts must not qualify for the auxiliary reward.
        sensor.data.net_forces_w = torch.full((env.num_envs, 1, 3), 100.0)


def force(env, finger):
    return env.scene.sensors[f"right_{finger}dip_roll_rubber_link_object_s"].data.force_matrix_w


@pytest.mark.parametrize("finger", FINGERS)
def test_every_fingertip_qualifies_without_opposing_grasp_and_saturates(code, configured, finger):
    env = _task_env(count=5)
    clear_contact(env)
    term = configured.any_finger_contact
    assert not term.func(env, **term.params).any()
    force(env, finger)[:, 0, 0, 2] = torch.tensor([0., .1, .25, .5, 10.])
    scores = term.func(env, **term.params)
    torch.testing.assert_close(scores, torch.tensor([0., .04, .1, .2, .2]))
    assert not code.rotate_bulb_hand_contacts(env, 1.).any()
    for other in FINGERS:
        force(env, other)[:] = force(env, finger)
    torch.testing.assert_close(term.func(env, **term.params), scores * 5)


def test_invalid_forces_are_not_full_contact_and_do_not_hide_other_valid_fingers(code):
    env = _task_env(count=4)
    clear_contact(env)
    force(env, "thumb")[:, 0, 0, 0] = torch.tensor([float("nan"), float("inf"), -float("inf"), float("nan")])
    force(env, "ring")[3, 0, 0, 2] = .25
    torch.testing.assert_close(code.rotate_bulb_any_finger_contact(env), torch.tensor([0., 0., 0., .1]))


@pytest.mark.parametrize("scale", [0., -1., float("nan"), float("inf")])
def test_force_scale_must_be_positive_and_finite(code, scale):
    with pytest.raises(ValueError, match="force_scale"):
        code.rotate_bulb_any_finger_contact(_task_env(), force_scale=scale)


def test_reward_covers_attached_and_released_contact_but_excludes_drop_and_bounds(code):
    env = _task_env(count=5)
    env._bulb_released[1:] = True
    env.failure_terms["bulb_dropped"][2] = True
    env.failure_terms["object_out_of_bound"][3] = True
    env.failure_terms["time_out"][4] = True
    torch.testing.assert_close(code.rotate_bulb_any_finger_contact(env), torch.tensor([.4, .4, 0., 0., .4]))
    clear_contact(env)
    assert not code.rotate_bulb_any_finger_contact(env).any()
    # There is no cached contact or per-episode budget: recontact resumes immediately.
    force(env, "pinky")[4, 0, 0, 0] = .5
    torch.testing.assert_close(code.rotate_bulb_any_finger_contact(env), torch.tensor([0., 0., 0., 0., .2]))


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_contact_continues_after_grasp_budget_and_integrates_to_two_over_twenty_seconds(code, configured, hz):
    env = _task_env(hz=hz)
    for finger in FINGERS:
        force(env, finger)[:, 0, 0, 2] = 2.
    grasp = code.RotateBulbGraspBudget(SimpleNamespace(params={}), env)
    term = configured.any_finger_contact
    # Accumulate rates in float64 so thousands of float32 additions do not
    # obscure the control-frequency comparison with summation roundoff.
    auxiliary, initial_grasp = torch.zeros(1, dtype=torch.float64), torch.zeros(1)
    for _ in range(20 * hz):
        auxiliary += term.func(env, **term.params) * term.weight * env.step_dt
        initial_grasp += grasp(env) * env.step_dt
    torch.testing.assert_close(initial_grasp, torch.ones(1))
    torch.testing.assert_close(auxiliary, torch.full_like(auxiliary, 2.0), atol=1e-6, rtol=0)
    assert grasp(env).item() == 0
    assert term.func(env, **term.params).item() == 1


@pytest.mark.parametrize("count", range(6))
def test_each_contacting_finger_adds_point_zero_two_per_second(configured, count):
    env = _task_env()
    clear_contact(env)
    for finger in FINGERS[:count]:
        force(env, finger)[:, 0, 0, 2] = .5
    term = configured.any_finger_contact
    assert (term.func(env, **term.params) * term.weight).item() == pytest.approx(count * .02)
    # Extra force on a contacting finger must not substitute for a missing one.
    if count:
        force(env, FINGERS[0])[:, 0, 0, 2] = 100.
        assert (term.func(env, **term.params) * term.weight).item() == pytest.approx(count * .02)


def test_auxiliary_exceeds_real_penalties_for_28_joint_unit_actions_without_changing_weights(code, configured):
    path = code.source_path.parents[2] / "mdp/rewards.py"
    definitions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                   and n.name in {"action_l2_clamped", "action_rate_l2_clamped"}]
    namespace = dict(torch=torch, ManagerBasedRLEnv=object)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    env = _task_env()
    env.action_manager = SimpleNamespace(action=torch.ones(1, 28), prev_action=-torch.ones(1, 28))
    assert configured.action_l2.weight == -.0001
    assert configured.action_rate_l2.weight == -.0002
    penalty = (namespace["action_l2_clamped"](env) * configured.action_l2.weight
               + namespace["action_rate_l2_clamped"](env) * configured.action_rate_l2.weight) * env.step_dt
    torch.testing.assert_close(penalty, torch.tensor([-.0252 * env.step_dt]))
    term = configured.any_finger_contact
    auxiliary = term.func(env, **term.params) * term.weight * env.step_dt
    assert (auxiliary + penalty).item() > 0
    clear_contact(env)
    assert (term.func(env, **term.params) * term.weight * env.step_dt + penalty).item() < 0
    assert configured.unscrew_turns.weight == .5 and configured.unscrew_progress.weight == 1.5
    assert configured.success.weight == 10
