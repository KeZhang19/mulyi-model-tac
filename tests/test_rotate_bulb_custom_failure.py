"""Custom grasp abandonment, position-only success and terminal reward priorities."""

import ast
from types import SimpleNamespace

import pytest
import torch

from test_rotate_bulb_custom_turns import code, configured, _set_turns
from test_rotate_bulb_reward_balance import _task_env, _term
from test_rotate_bulb_rewards import _params


def _loss_env(count=2, hz=60, distance=.04):
    env = _task_env(count=count, hz=hz)
    # All fingertip distances start inside the near-field region. Only filtered
    # bulb forces count; unrelated net forces must not preserve the attempt.
    env.scene["robot"].data.body_pos_w[:] = torch.tensor([.04, distance, 0.])
    for sensor in env.scene.sensors.values():
        sensor.data.force_matrix_w.zero_()
        sensor.data.net_forces_w = torch.full((count, 1, 3), 100.)
    return env


def _step(env, loss):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    result = loss(env)
    env.failure_terms["lost_grasp"][:] = result
    env.termination_manager.dones = torch.stack(list(env.failure_terms.values())).any(dim=0)
    return result


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_initial_grace_and_no_contact_timeout_use_seconds_once_per_step(code, hz):
    env = _loss_env(hz=hz, distance=.21)
    env._bulb_released[:] = True
    env.command[:, 2] = .2  # Outside the unchanged goal-dwell exemption.
    term = _term(code, "RotateBulbGraspLoss", env)
    for _ in range(round(.5 * hz)):
        assert not _step(env, term).any()
        assert not term.no_contact_time.any()
    # At frequencies where the duration is fractional, fire on the first step
    # whose elapsed time reaches the threshold, never an earlier sample.
    import math
    steps = math.ceil(.75 * hz - 1e-6)
    for i in range(1, steps + 1):
        result = _step(env, term)
        assert result.tolist() == [i == steps] * 2
        before = term.no_contact_time.clone()
        assert torch.equal(term(env), result)  # Reward/diagnostic rereads cannot advance time.
        torch.testing.assert_close(term.no_contact_time, before)
    assert steps / hz >= .75 and steps / hz < .75 + env.step_dt + 1e-6


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_brief_regrasp_and_partial_reset_do_not_end_other_environment(code, hz):
    env = _loss_env(hz=hz, distance=.21)
    env._bulb_released[:] = True
    env.command[:, 2] = .2
    term = _term(code, "RotateBulbGraspLoss", env)
    for _ in range(round(.7 * hz)):
        assert not _step(env, term).any()
    thumb = env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w
    thumb[0, ..., 2] = .1  # A weak single-finger contact is enough to avoid full contact loss.
    _step(env, term)
    assert term.no_contact_time[0] == 0
    assert term.no_contact_time[1] > .19
    before = term.no_contact_time[1].clone()
    env.episode_length_buf[0] = 0
    term.reset(torch.tensor([0]))
    term(env)  # Partial reset and same-step reread do not touch the other timer.
    assert term.no_contact_time[0] == 0
    torch.testing.assert_close(term.no_contact_time[1], before)
    for _ in range(round(.5 * hz)):
        _step(env, term)
    assert term.no_contact_time[0] == 0
    assert not term.failed[0]


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_far_termination_and_cost_boundaries(code, hz):
    env = _loss_env(count=3, hz=hz)
    env._bulb_released[:] = True
    env.command[:, 2] = .2
    # Keep real bulb contact to isolate the independent far timer.
    env.scene.sensors["right_indexdip_roll_rubber_link_object_s"].data.force_matrix_w[..., 2] = 1.
    env.scene["robot"].data.body_pos_w[:, :, 1] = torch.tensor([.08, .14, .21])[:, None]
    term = _term(code, "RotateBulbGraspLoss", env)
    for _ in range(round(.5 * hz)):
        _step(env, term)
        assert not code.rotate_bulb_hand_far(env).any()
    _step(env, term)
    torch.testing.assert_close(code.rotate_bulb_hand_far(env), torch.tensor([0., .25, 1.]), atol=1e-6, rtol=0)
    import math
    for _ in range(math.ceil(.25 * hz - 1e-6) - 1):
        _step(env, term)
    assert not term.failed.any()  # Distance alone is never abandonment.
    env.scene.sensors["right_indexdip_roll_rubber_link_object_s"].data.force_matrix_w.zero_()
    for _ in range(math.ceil(.75 * hz - 1e-6)):
        _step(env, term)
    assert term.failed.tolist() == [False, False, True]  # Both absence and distance are required.
    assert code.rotate_bulb_hand_far(env)[2] == 0  # No extra far cost on the terminal step.


def test_distance_uses_named_best_opposing_pair_and_rotates_with_bulb(code):
    env = _loss_env(count=1)
    robot = env.scene["robot"]
    # Unused fingers can be far away; a near thumb/index pair remains valid.
    robot.data.body_pos_w[:, 2:] = 1.
    assert code.rotate_bulb_hand_shell_distance(env).item() == pytest.approx(.04)
    robot.data.body_pos_w[:, 0, 1] = .10
    assert code.rotate_bulb_hand_shell_distance(env).item() == pytest.approx(.10)
    env.scene["object"].data.root_quat_w[:] = torch.tensor([0., 0., 0., 1.])
    robot.data.body_pos_w[:, :2] = torch.tensor([-.04, .03, 0.])
    assert code.rotate_bulb_hand_shell_distance(env).item() == pytest.approx(.03)


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_configured_far_cost_applies_before_release_and_rearms_per_environment(configured, hz):
    env = _loss_env(count=4, hz=hz)
    env.command[:, 2] = .2  # Keep the goal-dwell exemption inactive in both phases.
    env.scene["robot"].data.body_pos_w[:, :, 1] = torch.tensor([.04, .08, .14, .21])[:, None]
    env.episode_length_buf[:] = hz
    term = configured.hand_far
    rates = term.func(env, **term.params) * term.weight
    torch.testing.assert_close(rates, torch.tensor([0., 0., -.075, -.3]), atol=1e-6, rtol=0)
    env._bulb_released[:] = True
    torch.testing.assert_close(term.func(env, **term.params) * term.weight, rates)
    # Resetting one environment restores only its initial grace.
    env._bulb_released[3] = False
    env.episode_length_buf[3] = 0
    torch.testing.assert_close(term.func(env, **term.params) * term.weight,
                               torch.tensor([0., 0., -.075, 0.]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_configured_far_cost_has_the_same_twenty_second_budget_at_each_control_rate(configured, hz):
    env = _loss_env(count=1, hz=hz, distance=.4)
    term = configured.hand_far
    total = 0.
    for step in range(1, 20 * hz + 1):
        env.episode_length_buf[:] = step
        total += (term.func(env, **term.params) * term.weight * env.step_dt).item()
    # The first half-second is free; no terminal frame is injected in this rate integral.
    assert total == pytest.approx(-5.85, abs=1e-6)


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_position_only_goal_dwell_clears_loss_and_allows_original_success(code, hz):
    env = _loss_env(hz=hz, distance=.4)
    params = dict(_params(), pos_std=.03, hold_duration=1., once=True)
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(params=params))
    env._bulb_released[:] = True
    env.command[:, 2] = .1
    env.scene["object"].data.root_pos_w[:, 2] = .1
    loss = _term(code, "RotateBulbGraspLoss", env)
    done = _term(code, "RotateBulbSuccessTermination", env)
    success = _term(code, "RotateBulbStableSuccess", env)
    for i in range(1, hz + 1):
        assert not _step(env, loss).any()
        assert not loss.no_contact_time.any()
        assert not loss.far_time.any()
        assert not code.rotate_bulb_hand_far(env).any()
        assert done(env).tolist() == [i == hz] * 2
        expected = torch.full((2,), 10. if i == hz else 0.)
        torch.testing.assert_close(success(env, **params) * env.step_dt * 10, expected)


def test_goal_exemption_reads_late_overrides_and_restarts_timers_on_exit(code):
    env = _loss_env(count=1, distance=.4)
    env._bulb_released[:] = True
    env.command[:, 0] = .025
    params = dict(_params(), pos_std=.03)
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(params=params))
    loss = _term(code, "RotateBulbGraspLoss", env)
    for _ in range(60):
        assert not _step(env, loss).any()
    params["pos_std"] = .02
    for _ in range(10):
        assert not _step(env, loss).any()
    assert loss.far_time.item() == pytest.approx(10 / 60)
    params["pos_std"] = .03
    _step(env, loss)
    assert not loss.no_contact_time.any() and not loss.far_time.any()
    params["pos_std"] = .02
    _step(env, loss)
    assert loss.far_time.item() == pytest.approx(1 / 60)


@pytest.mark.parametrize("failure_name", ["bulb_dropped", "object_out_of_bound", "lost_grasp", "non_finite"])
@pytest.mark.parametrize("released", [False, True])
def test_each_physical_failure_costs_once_and_suppresses_all_positive_rewards(code, configured, failure_name, released):
    env = _loss_env()
    configured.fingers_to_object_delta.params["asset_cfg"].body_ids = list(range(5))
    terms = {name: term for name, term in vars(type(configured)).items()
             if isinstance(term, SimpleNamespace) and hasattr(term, "weight") and term.weight > 0}
    functions = {name: term.func(term, env) if isinstance(term.func, type) else term.func
                 for name, term in terms.items()}
    penalty = _term(code, "RotateBulbFailurePenalty", env)
    _set_turns(env, [2., 2.])
    env._bulb_released[:] = released
    env.scene["object"].data.root_pos_w[:, 2] = .1
    env.command[:, 2] = .1
    for sensor in env.scene.sensors.values():
        sensor.data.force_matrix_w[..., 2] = 2.
    env.failure_terms[failure_name][:] = True
    if failure_name == "non_finite":
        env.scene["object"].data.root_lin_vel_w[:] = float("nan")
        env.scene["object"].data.root_ang_vel_w[:] = float("nan")
    if failure_name == "non_finite":
        assert not code.rotate_bulb_failed(env).any()
        assert not penalty(env).any()
        name = "position_tracking" if released else "unscrew_progress"
        if released:
            functions[name](env, **terms[name].params)  # Establish release baseline.
            env.command[:, 2] = .3
            functions[name].reset()
            functions[name](env, **terms[name].params)
            env.scene["object"].data.root_pos_w[:, 2] = .2
        credit = functions[name](env, **terms[name].params)
        assert torch.isfinite(credit).all() and (credit > 0).all()
        return
    for _ in range(61):
        for name, term in terms.items():
            value = functions[name](env, **term.params)
            assert torch.isfinite(value).all(), name
            assert not value.any(), name
    torch.testing.assert_close(penalty(env) * env.step_dt * -6, torch.full((2,), -6.))
    assert not penalty(env).any()
    for flags in env.failure_terms.values():
        flags[:] = True
    assert not penalty(env).any()  # Overlapping causes cannot charge twice.
    penalty.reset(torch.tensor([0]))
    torch.testing.assert_close(penalty(env) * env.step_dt * -6, torch.tensor([-6., 0.]))


@pytest.mark.parametrize("failure_name", ["bulb_dropped", "object_out_of_bound", "non_finite"])
def test_goal_dwell_never_exempts_physical_failures_on_success_step(code, failure_name):
    env = _loss_env(count=1, distance=.4)
    env._bulb_released[:] = True
    params = dict(_params(), pos_std=.03, hold_duration=1., once=True)
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(params=params))
    loss = _term(code, "RotateBulbGraspLoss", env)
    done = _term(code, "RotateBulbSuccessTermination", env)
    for _ in range(59):
        _step(env, loss)
        assert not done(env).any()
    env.failure_terms[failure_name][:] = True
    assert not _step(env, loss).any()
    disabled = failure_name == "non_finite"
    assert done(env).tolist() == [disabled]
    assert code.rotate_bulb_failed(env).tolist() == [not disabled]


def test_balanced_action_cost_follows_named_indices_and_has_no_cap(code, configured):
    env = _loss_env(count=3)
    arm_ids = [1, 4, 7, 10, 13, 16, 19]  # Deliberately not the first seven columns.
    hand_ids = [i for i in range(28) if i not in arm_ids]
    action = torch.zeros(3, 28)
    action[0, arm_ids] = 1.
    action[1, hand_ids] = 1.
    action[2] = 100.
    env.action_manager = SimpleNamespace(action=action, prev_action=-action,
                                        get_term=lambda _: SimpleNamespace(arm_action_indices=arm_ids,
                                                                          hand_action_indices=hand_ids))
    torch.testing.assert_close(code.rotate_bulb_action_l2_balanced(env), torch.tensor([.5, .5, 10000.]))
    torch.testing.assert_close(code.rotate_bulb_action_rate_l2_balanced(env), torch.tensor([2., 2., 40000.]))
    assert configured.action_l2.weight == -.05 and configured.action_rate_l2.weight == -.02


def test_custom_failure_terms_are_registered_before_success(code):
    tree = ast.parse(code.source_path.read_text())
    configure = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "configure_rotate_bulb_scene")
    names = [ast.unparse(node.targets[0]) for node in configure.body if isinstance(node, ast.Assign)]
    assert names.index("cfg.terminations.bulb_dropped") < names.index("cfg.terminations.success")
    assert names.index("cfg.terminations.lost_grasp") < names.index("cfg.terminations.success")
    assignment = next(node for node in configure.body if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == "cfg.terminations.lost_grasp")
    namespace = dict(DoneTerm=lambda **kw: SimpleNamespace(**kw), RotateBulbGraspLoss=code.RotateBulbGraspLoss)
    term_cfg = eval(compile(ast.Expression(assignment.value), str(code.source_path), "eval"), namespace)
    assert term_cfg.func is code.RotateBulbGraspLoss
    assert term_cfg.params["no_contact_duration_s"] == .75
    assignment = next(node for node in configure.body if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == "cfg.terminations.non_finite")
    assert ast.literal_eval(assignment.value) is None


def _finite_env():
    env = _loss_env()
    robot = env.scene["robot"].data
    robot.joint_pos = torch.zeros(2, 28)
    robot.joint_vel = torch.zeros(2, 28)
    robot.root_lin_vel_w = torch.zeros(2, 3)
    robot.root_ang_vel_w = torch.zeros(2, 3)
    robot.body_quat_w = torch.zeros(2, 5, 4)
    robot.body_quat_w[:, :, 0] = 1.
    robot.body_lin_vel_w = torch.zeros(2, 5, 3)
    robot.body_ang_vel_w = torch.zeros(2, 5, 3)
    env.scene["lamp"].data.joint_vel = torch.zeros(2, 2)
    env.scene.sensors["bulb_table_contact"] = SimpleNamespace(
        data=SimpleNamespace(force_matrix_w=torch.zeros(2, 1, 1, 3)))
    return env


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("asset, field", [
    ("robot", "joint_pos"), ("robot", "joint_vel"), ("robot", "root_quat_w"),
    ("robot", "body_pos_w"), ("robot", "body_quat_w"),
    ("robot", "body_lin_vel_w"), ("robot", "body_ang_vel_w"),
    ("object", "root_lin_vel_w"), ("object", "root_ang_vel_w"),
    ("lamp", "joint_pos"), ("lamp", "joint_vel"),
    ("right_thumbdip_roll_rubber_link_object_s", "force_matrix_w"),
    ("bulb_table_contact", "force_matrix_w"),
])
def test_nonfinite_covers_kinematics_velocities_and_filtered_sensors(code, asset, field, value):
    env = _finite_env()
    assert not code.rotate_bulb_non_finite(env).any()
    target = env.scene.sensors[asset] if asset in env.scene.sensors else env.scene[asset]
    getattr(target.data, field).reshape(2, -1)[0, 0] = value
    flags = code.rotate_bulb_non_finite(env)
    assert flags.tolist() == [True, False]
    env.failure_terms["non_finite"][:] = flags
    assert not code.rotate_bulb_failed(env).any()  # Detection is retained but disabled as a failure cause.
    assert code.rotate_bulb_any_finger_contact(env)[0] == 0


@pytest.mark.parametrize("sensor_name", ["right_thumbdip_roll_rubber_link_object_s", "bulb_table_contact"])
def test_missing_filtered_force_matrix_fails_without_crashing_rewards(code, sensor_name):
    env = _finite_env()
    env.scene.sensors[sensor_name].data.force_matrix_w = None
    env.failure_terms["non_finite"][:] = code.rotate_bulb_non_finite(env)
    assert env.failure_terms["non_finite"].all()
    assert torch.isfinite(code.rotate_bulb_finger_forces(env)).all()
    assert not code.rotate_bulb_hand_contacts(env, 1.).any()
    assert not code.rotate_bulb_any_finger_contact(env).any()
    assert not code.rotate_bulb_dropped(env).any()


def test_nonfinite_pure_collision_does_not_require_removed_lamp_articulation(code):
    env = _finite_env()
    env.cfg = SimpleNamespace(bulb_pure_collision=True)
    del env.scene["lamp"]
    assert not code.rotate_bulb_non_finite(env).any()


def test_invalid_reach_cannot_poison_progress_or_reset_baseline(code):
    env = _loss_env()
    params = dict(std=.2, asset_cfg=SimpleNamespace(name="robot", body_ids=list(range(5))))
    term = _term(code, "RotateBulbReachProgress", env, **params)
    env.scene["robot"].data.body_pos_w[0] = float("nan")
    env.failure_terms["non_finite"][0] = True
    assert not term(env, **params).any()
    assert torch.isfinite(term.best).all()
    env.scene["robot"].data.body_pos_w[0] = torch.tensor([.04, .02, 0.])
    term.reset(torch.tensor([0]))  # Isaac Lab's failure flags still refer to the old episode.
    baseline = code.rotate_bulb_reach_shell(env, **params)
    torch.testing.assert_close(term.best[0], baseline[0])
    env.failure_terms["non_finite"][0] = False
    assert not term(env, **params).any()
