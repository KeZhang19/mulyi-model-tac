"""Check physical-time reward budgets and complete-task incentives on CPU."""

from types import SimpleNamespace

import pytest
import torch

from test_rotate_bulb_rewards import _env, _params, reward_code  # noqa: F401


def _task_env(count=1, hz=60):
    env = _env((0.,) * count)
    env.step_dt = 1 / hz
    env._bulb_start_joints = torch.tensor([[-6 * torch.pi, -.018]]).repeat(count, 1)
    env.scene["lamp"].data.joint_pos[:] = env._bulb_start_joints
    env.failure_terms = {
        name: torch.zeros(count, dtype=torch.bool)
        for name in ("bulb_dropped", "object_out_of_bound", "time_out")
    }
    env.termination_manager = SimpleNamespace(get_term=env.failure_terms.__getitem__)
    return env


def _term(code, name, env, **params):
    return getattr(code, name)(SimpleNamespace(params=params), env)


def _hold_released(env, height=.1):
    env._bulb_released[:] = True
    env.scene["object"].data.root_pos_w[:, 2] = height
    env.command[:, 2] = height


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_contact_and_stability_budgets_are_time_based_and_reset_independently(reward_code, hz):
    env = _task_env(count=2, hz=hz)
    grasp = _term(reward_code, "RotateBulbGraspBudget", env)
    total = torch.zeros(2)
    for _ in range(5 * hz):
        total += grasp(env) * env.step_dt
    torch.testing.assert_close(total, torch.ones(2))
    assert not grasp(env).any()  # Waiting longer cannot earn another budget.
    grasp.reset(torch.tensor([0]))
    total.zero_()
    for _ in range(hz):
        total += grasp(env) * env.step_dt
    torch.testing.assert_close(total, torch.tensor([.25, 0.]))

    _hold_released(env)
    assert not grasp(env).any()  # Pre-release shaping cannot leak into carrying.
    stable = _term(reward_code, "RotateBulbReleasedStability", env)
    total.zero_()
    for _ in range(5 * hz):
        total += stable(env) * env.step_dt
    torch.testing.assert_close(total, torch.full((2,), 2.))
    assert not stable(env).any()
    stable.reset(torch.tensor([0]))
    total.zero_()
    for _ in range(hz):
        total += stable(env) * env.step_dt
    torch.testing.assert_close(total, torch.tensor([.5, 0.]))


def test_released_stability_allows_carrying_but_rejects_unsupported_and_failed_states(reward_code):
    env = _task_env(count=8)
    _hold_released(env)
    env._bulb_released[0] = False
    env.scene["object"].data.root_pos_w[1, 2] = .005
    env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w[2] = 0
    env.scene["object"].data.root_lin_vel_w[3, 0] = .15
    env.scene["object"].data.root_ang_vel_w[5, 0] = 4.
    env.failure_terms["bulb_dropped"][6] = True
    env.scene["object"].data.root_lin_vel_w[7, 0] = 1.
    stable = _term(reward_code, "RotateBulbReleasedStability", env)
    rates = stable(env)
    assert rates[:3].tolist() == [0., 0., 0.]
    assert rates[6] == 0
    assert rates[3] == pytest.approx(rates[4].item())
    assert 0 < rates[5] < rates[4]
    assert 0 <= rates[7] < rates[4]


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_failure_is_one_event_even_with_overlap_and_ignores_timeout(reward_code, hz):
    env = _task_env(count=5, hz=hz)
    env._bulb_released[:] = torch.tensor([True, True, True, False, True])
    env.failure_terms["bulb_dropped"][:] = torch.tensor([True, False, False, True, True])
    env.failure_terms["object_out_of_bound"][:] = torch.tensor([True, True, False, True, False])
    env.failure_terms["time_out"][:] = True
    assert reward_code.rotate_bulb_failed(env).tolist() == [True, True, False, False, True]
    penalty = _term(reward_code, "RotateBulbFailurePenalty", env)
    torch.testing.assert_close(penalty(env) * env.step_dt * -6, torch.tensor([-6., -6., 0., 0., -6.]))
    assert not penalty(env).any()
    penalty.reset(torch.tensor([0, 3]))
    torch.testing.assert_close(penalty(env) * env.step_dt * -6, torch.tensor([-6., 0., 0., 0., 0.]))


def test_release_bonus_requires_uninterrupted_physical_grasp_and_only_rearms_on_reset(reward_code):
    env = _task_env(count=2, hz=60)
    env.scene["lamp"].data.joint_pos.zero_()
    env.scene["object"].data.root_pos_w[:, 2] = .1
    release = _term(reward_code, "RotateBulbReleaseGrasp", env)
    for _ in range(30):
        assert not release(env).any()  # An attached bulb cannot pass the milestone.
    env._bulb_released[:] = True
    for _ in range(12):
        assert not release(env).any()
    thumb = env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w
    thumb.zero_()
    assert not release(env).any()
    thumb[..., 2] = 2.
    for _ in range(12):
        assert not release(env).any()  # Earlier interrupted holding does not count.
    total = torch.zeros(2)
    for _ in range(30):
        total += release(env) * env.step_dt * 2
    torch.testing.assert_close(total, torch.full((2,), 2.))
    release.reset(torch.tensor([0]))
    total.zero_()
    for _ in range(30):
        total += release(env) * env.step_dt * 2
    torch.testing.assert_close(total, torch.tensor([2., 0.]))


@pytest.mark.parametrize("failure", ["bulb_dropped", "object_out_of_bound"])
def test_failure_suppresses_release_milestone_and_final_success(reward_code, failure):
    env = _task_env()
    _hold_released(env)
    env.failure_terms[failure][:] = True
    release = _term(reward_code, "RotateBulbReleaseGrasp", env)
    success = _term(reward_code, "RotateBulbStableSuccess", env)
    for _ in range(30):
        assert not release(env).any()
        assert not success(env, once=True, pos_std=.03, **_params()).any()
    assert not env._bulb_success.any()


def test_transport_tracks_current_distance_after_release(reward_code):
    env = _task_env()
    env.command[:, 2] = .3
    env.scene["object"].data.root_pos_w[:, 2] = .1
    params = dict(_params(), std=.1)
    transport = _term(reward_code, "RotateBulbTransportProgress", env, **params)
    assert not transport(env, **params).any()
    env._bulb_released[:] = True
    far = transport(env, **params).item()
    assert far > 0
    # The absolute tracking term remains active on stationary steps and rises
    # smoothly as the bulb approaches the commanded point.
    env.scene["object"].data.root_pos_w[:, 2] = .2
    mid = transport(env, **params).item()
    env.scene["object"].data.root_pos_w[:, 2] = .3
    near = transport(env, **params).item()
    assert 0 < far < mid < near <= 1
    thumb = env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w
    thumb.zero_()
    assert not transport(env, **params).any()
    # A failed or unreleased bulb cannot receive transport shaping.
    env.failure_terms["bulb_dropped"][:] = True
    assert not transport(env, **params).any()


def test_transport_is_independent_between_environments(reward_code):
    env = _task_env(count=2)
    _hold_released(env)
    env.command[:, 2] = .3
    params = dict(_params(), std=.1)
    transport = _term(reward_code, "RotateBulbTransportProgress", env, **params)
    original = transport(env, **params)
    assert (original > 0).all()
    env._bulb_released[0] = False
    current = transport(env, **params)
    torch.testing.assert_close(current[0], torch.tensor(0.))
    torch.testing.assert_close(current[1], original[1])


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_final_success_event_has_fixed_value_and_current_success_can_be_lost(reward_code, hz):
    env = _task_env(count=2, hz=hz)
    _hold_released(env)
    success = _term(reward_code, "RotateBulbStableSuccess", env)
    params = dict(_params(), once=True, pos_std=.03)
    total = torch.zeros(2)
    for _ in range(hz):
        total += success(env, **params) * env.step_dt * 10
    torch.testing.assert_close(total, torch.full((2,), 10.))
    assert env._bulb_success.all()
    env.scene["object"].data.root_ang_vel_w[:, 0] = 2.
    # The relaxed criterion intentionally ignores velocity once the position
    # has been reached; only leaving the target clears the hold timer.
    assert success(env, **params).all()
    env.scene["object"].data.root_pos_w[:, 0] = .1
    assert not success(env, **params).any()
    env.scene["object"].data.root_ang_vel_w.zero_()
    env.scene["object"].data.root_pos_w[:, 0] = 0.
    for _ in range(hz):
        assert not success(env, **params).any()  # Regaining the goal is not a new success event.
    assert env._bulb_success.all()
    success.reset(torch.tensor([0]))
    total.zero_()
    for _ in range(hz):
        total += success(env, **params) * env.step_dt * 10
    torch.testing.assert_close(total, torch.tensor([10., 0.]))


def _run_three_turn_attempt(code, hz, drop):
    env = _task_env(hz=hz)
    spin = _term(code, "RotateBulbUnscrewProgress", env)
    grasp = _term(code, "RotateBulbGraspBudget", env)
    release = _term(code, "RotateBulbReleaseGrasp", env)
    stable = _term(code, "RotateBulbReleasedStability", env)
    failure = _term(code, "RotateBulbFailurePenalty", env)
    success = _term(code, "RotateBulbStableSuccess", env)
    params = dict(_params(), std=.1)
    transport = _term(code, "RotateBulbTransportProgress", env, **params)
    total = 0.
    spin_total = 0.
    for step in range(1, hz + 1):
        env.scene["lamp"].data.joint_pos[:] = env._bulb_start_joints * (1 - step / hz)
        credit = (spin(env, completion_bonus=0.) * env.step_dt * 2).item()
        spin_total += credit
        total += credit + (grasp(env) * env.step_dt).item()
    assert spin_total == pytest.approx(6., abs=1e-5)
    _hold_released(env, height=.03)
    env.command[:, 2] = .15
    if drop:
        env.failure_terms["bulb_dropped"][:] = True
        env.failure_terms["object_out_of_bound"][:] = True
    total += (failure(env) * env.step_dt * -6).item()
    if drop:
        assert not transport(env, **params).any()
        assert not release(env).any()
        assert not stable(env).any()
        assert not success(env, once=True, pos_std=.03, **_params()).any()
        return total
    assert (transport(env, **params) > 0).all()
    # Hold after release, carry 12 cm at 0.12 m/s, then settle at the target.
    for phase, steps in (("hold", round(.4 * hz)), ("carry", hz), ("settle", round(.4 * hz))):
        for step in range(1, steps + 1):
            if phase == "carry":
                env.scene["object"].data.root_pos_w[:, 2] = .03 + .12 * step / steps
                env.scene["object"].data.root_lin_vel_w[:, 2] = .12
            else:
                env.scene["object"].data.root_lin_vel_w.zero_()
            total += (release(env) * env.step_dt * 2).item()
            total += (stable(env) * env.step_dt).item()
            total += (transport(env, **params) * env.step_dt * 4).item()
            total += (success(env, once=True, pos_std=.03, **_params()) * env.step_dt * 10).item()
    assert env._bulb_success.all()
    return total


def test_complete_task_dominates_spin_then_drop_at_all_control_frequencies(reward_code):
    complete = [_run_three_turn_attempt(reward_code, hz, drop=False) for hz in (30, 60, 120)]
    dropped = [_run_three_turn_attempt(reward_code, hz, drop=True) for hz in (30, 60, 120)]
    # Dense tracking is paid throughout the released carry phase, so the
    # completed trajectory now includes several points of transport shaping.
    assert all(23 < value < 28 for value in complete)
    assert all(-.01 < value < .3 for value in dropped)
    # Grasp closes on the final attached step, so its quadrature differs by at most one control step.
    assert max(complete) - min(complete) < .01
    assert max(dropped) - min(dropped) < .01
    assert min(complete) > max(dropped) + 20


def test_actual_configured_pipeline_keeps_task_completion_above_early_phase_and_failure(reward_code):
    """Execute configured terms/parameters/weights, including the real action penalties."""
    import ast
    source_path = reward_code.source_path

    namespace = dict(vars(reward_code))
    action_tree = ast.parse((source_path.parents[2] / "mdp/rewards.py").read_text())
    action_defs = [node for node in action_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"action_l2_clamped", "action_rate_l2_clamped"}]
    exec(compile(ast.Module(body=action_defs, type_ignores=[]), "actual_action_rewards", "exec"), namespace)
    namespace["mdp"] = SimpleNamespace(**{node.name: namespace[node.name] for node in action_defs})
    namespace["SceneEntityCfg"] = lambda name: SimpleNamespace(name=name, body_ids=list(range(5)))
    namespace["RewTerm"] = lambda func, weight, params=None: SimpleNamespace(
        func=func, weight=weight, params=params or {},
    )
    tree = ast.parse(source_path.read_text())
    cfg_node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "DexsuiteRevo3RotateBulbRewardCfg")
    cfg_node.decorator_list = []  # Simulator config decoration is not reward computation.
    exec(compile(ast.Module(body=[cfg_node], type_ignores=[]), str(source_path), "exec"), namespace)
    cfg = namespace[cfg_node.name]()
    assert cfg.success.params["once"] is True
    assert cfg.success.params["pos_std"] == 0.03
    assert cfg.success.params["hold_duration"] == 1.0
    assert cfg.unscrew_progress.params["completion_bonus"] == 0
    assert cfg.position_tracking.func is reward_code.RotateBulbTransportProgress
    assert cfg.position_tracking.weight == 0.5
    assert cfg.position_tracking.params["std"] == 0.2
    assert cfg.position_tracking_fine is None
    assert cfg.fingers_to_object.weight == cfg.good_finger_contact.weight == 0
    assert -.005 < cfg.action_l2.weight < 0
    assert -.005 < cfg.action_rate_l2.weight < 0

    env = _task_env(count=3)
    env.action_manager = SimpleNamespace(action=torch.zeros(3, 20), prev_action=torch.zeros(3, 20))
    configured = {name: term for name, term in vars(type(cfg)).items()
                  if isinstance(term, SimpleNamespace) and hasattr(term, "weight") and term.weight != 0}
    functions = {name: term.func(term, env) if isinstance(term.func, type) else term.func
                 for name, term in configured.items()}
    sums = {name: torch.zeros(3) for name in configured}

    def step_configured_rewards():
        for name, term in configured.items():
            sums[name] += functions[name](env, **term.params) * term.weight * env.step_dt

    # Environments 0/1 turn three times; environment 2 waits without rotating.
    for step in range(1, 61):
        env.scene["lamp"].data.joint_pos[:2] = env._bulb_start_joints[:2] * (1 - step / 60)
        step_configured_rewards()
    env._bulb_released[:2] = True
    env.scene["object"].data.root_pos_w[:2, 2] = .03
    env.command[:, 2] = .15
    env.failure_terms["bulb_dropped"][1] = True
    env.failure_terms["object_out_of_bound"][1] = True
    step_configured_rewards()  # Establish release baseline; charge failure once.
    for phase, steps in (("hold", 24), ("carry", 60), ("settle", 360)):
        for step in range(1, steps + 1):
            if phase == "carry":
                env.scene["object"].data.root_pos_w[0, 2] = .03 + .12 * step / steps
                env.scene["object"].data.root_lin_vel_w[0, 2] = .12
            else:
                env.scene["object"].data.root_lin_vel_w.zero_()
            step_configured_rewards()

    spin_sum = sums["unscrew_progress"] + sums.get("unscrew_turns", 0.)
    torch.testing.assert_close(spin_sum, torch.tensor([6., 6., 0.]))
    torch.testing.assert_close(sums["success"], torch.tensor([10., 0., 0.]))
    torch.testing.assert_close(sums["release_grasp"], torch.tensor([2., 0., 0.]))
    torch.testing.assert_close(sums["failure"], torch.tensor([0., -6., 0.]))
    torch.testing.assert_close(sums["released_stability"], torch.tensor([2., 0., 0.]))
    assert sums["position_tracking"][0] > 0
    assert sums["position_tracking"][1:].tolist() == [0., 0.]
    total = sum(sums.values())
    assert total[0] > 20
    assert 0 <= total[1] < 1
    auxiliary_contact = sums.get("any_finger_contact", torch.zeros(3))
    assert total[2] - auxiliary_contact[2] <= 1.00001  # The initial grasp budget stays bounded.
    assert 0 <= auxiliary_contact[2] <= 2.0  # Custom adds a small continuous contact rate.
    early = sums["fingers_to_object_delta"][0] + sums["grasp_shaping"][0] + spin_sum[0]
    assert total[0] - early > 2 * early


def test_transport_nan_contact_is_zero_credit_even_when_phase_or_failure_masks_apply(reward_code):
    env = _task_env(count=4)
    env.command[:, 2] = .3
    env.scene["object"].data.root_pos_w[:, 2] = .1
    env._bulb_released[1:] = True
    params = dict(_params(), std=.1)
    transport = _term(reward_code, "RotateBulbTransportProgress", env, **params)
    assert not transport(env, **params).any()
    env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w[:3] = float("nan")
    env.failure_terms["bulb_dropped"][2] = True
    env.scene["object"].data.root_pos_w[:, 2] = .2
    credit = transport(env, **params) * env.step_dt
    assert torch.isfinite(credit).all()
    assert credit[:3].tolist() == [0., 0., 0.]
    assert credit[3] > 0


@pytest.mark.parametrize("hz", [30, 60, 120])
def test_success_termination_and_bonus_coincide_in_physical_time_and_after_partial_reset(reward_code, hz):
    env = _task_env(count=2, hz=hz)
    _hold_released(env)
    reward_params = dict(_params(), pos_std=.03, hold_duration=.3, once=True)
    done_params = dict(reward_params, once=False, as_termination=True)
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(success=SimpleNamespace(params=reward_params)))
    done = _term(reward_code, "RotateBulbStableSuccess", env, **done_params)
    success = _term(reward_code, "RotateBulbStableSuccess", env, **reward_params)
    hold_steps = round(.3 * hz)
    for step in range(1, hold_steps + 1):
        terminated = done(env, **done_params)  # Manager lifecycle computes terminations first.
        bonus = success(env, **reward_params) * env.step_dt * 10
        assert terminated.dtype == torch.bool
        if step < hold_steps:
            assert not terminated.any()
            assert not bonus.any()
        else:
            assert terminated.all()
            torch.testing.assert_close(bonus, torch.full((2,), 10.))
    # Each manager resets its own timer for selected environments only.
    done.reset(torch.tensor([0]))
    success.reset(torch.tensor([0]))
    for step in range(1, hold_steps + 1):
        terminated = done(env, **done_params)
        bonus = success(env, **reward_params) * env.step_dt * 10
        assert terminated.tolist() == [step == hold_steps, True]
        expected = torch.tensor([10. if step == hold_steps else 0., 0.])
        torch.testing.assert_close(bonus, expected)


@pytest.mark.parametrize("hz", [30, 60, 120])
@pytest.mark.parametrize("failure_name", ["bulb_dropped", "object_out_of_bound"])
def test_failure_on_success_threshold_step_wins_over_success_termination_and_bonus(reward_code, hz, failure_name):
    env = _task_env(count=2, hz=hz)
    _hold_released(env)
    reward_params = dict(_params(), pos_std=.03, hold_duration=.3, once=True)
    done_params = dict(reward_params, once=False, as_termination=True)
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(success=SimpleNamespace(params=reward_params)))
    done = _term(reward_code, "RotateBulbStableSuccess", env, **done_params)
    success = _term(reward_code, "RotateBulbStableSuccess", env, **reward_params)
    failure = _term(reward_code, "RotateBulbFailurePenalty", env)
    for _ in range(round(.3 * hz) - 1):
        assert not done(env, **done_params).any()
        assert not success(env, **reward_params).any()
    env.failure_terms[failure_name][0] = True
    assert done(env, **done_params).tolist() == [False, True]
    torch.testing.assert_close(success(env, **reward_params) * env.step_dt * 10, torch.tensor([0., 10.]))
    torch.testing.assert_close(failure(env) * env.step_dt * -6, torch.tensor([-6., 0.]))
    assert env._bulb_success.tolist() == [False, True]


def test_success_termination_uses_late_reward_parameters_without_mutating_event_mode(reward_code):
    import ast
    source_path = reward_code.source_path

    tree = ast.parse(source_path.read_text())
    configure = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "configure_rotate_bulb_scene")
    assignment = next(node for node in configure.body if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == "cfg.terminations.success")
    cfg = SimpleNamespace(terminations=SimpleNamespace())
    namespace = dict(cfg=cfg, RotateBulbSuccessTermination=reward_code.RotateBulbSuccessTermination,
                     DoneTerm=lambda **kwargs: SimpleNamespace(**kwargs))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source_path), "exec"), namespace)
    assert cfg.terminations.success.func is reward_code.RotateBulbSuccessTermination

    env = _task_env(count=2, hz=60)
    _hold_released(env)
    env.command[0, 0] = .025
    params = dict(_params(), pos_std=.03, rot_std=None, hold_duration=.3, once=True)
    actual_reward_cfg = SimpleNamespace(params=params)
    # Return the real mutable params object, as the reward manager does.
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda _: actual_reward_cfg)
    done = _term(reward_code, "RotateBulbSuccessTermination", env)
    success = _term(reward_code, "RotateBulbStableSuccess", env, **params)
    params.update(pos_std=.02, hold_duration=.4)  # Override after both terms exist.
    for step in range(1, 25):
        terminated = done(env)
        bonus = success(env, **params) * env.step_dt * 10
        assert terminated.dtype == torch.bool
        assert terminated.tolist() == [False, step == 24]
        torch.testing.assert_close(bonus, torch.tensor([0., 10. if step == 24 else 0.]))
    assert actual_reward_cfg.params is params
    assert params["once"] is True
    assert "as_termination" not in params
    assert params["pos_std"] == .02
    assert params["hold_duration"] == .4


def test_command_reset_logs_completed_episode_outcome_after_scene_success_state_clears(reward_code):
    import ast
    source_path = reward_code.source_path

    class CommandParent:
        def __init__(self, cfg, env):
            self._env = env
            self.num_envs, self.device = env.num_envs, env.device
            self.reset_ids = None

        def reset(self, env_ids=None):
            self.reset_ids = env_ids
            return {"success": 0., "position_error": .02}

    tree = ast.parse(source_path.read_text())
    command_node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "RotateBulbPoseCommand")
    namespace = dict(torch=torch, ObjectUniformPoseCommand=CommandParent)
    exec(compile(ast.Module(body=[command_node], type_ignores=[]), str(source_path), "exec"), namespace)
    env = _task_env(count=3)
    env._bulb_success = torch.zeros(3, dtype=torch.bool)  # Scene reset has already run.
    env.failure_terms["success"] = torch.tensor([True, False, True])
    env.episode_length_buf = torch.zeros(3, dtype=torch.long)
    env.common_step_counter = 0
    command = namespace[command_node.name](None, env)
    selected = torch.tensor([0, 1])
    metrics = command.reset(selected)
    assert command.reset_ids is selected
    assert metrics["success"] == .5 and metrics["position_error"] == .02
    assert all(value.numel() == 0 for key, value in metrics.items() if key.startswith("unscrew_"))
    assert command.reset(torch.tensor([2]))["success"] == 1.
    assert command.reset()["success"] == pytest.approx(2 / 3)
