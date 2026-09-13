"""Exercise task reward code on CPU without launching Isaac Sim."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


CONFIG = (Path(__file__).resolve().parents[1] / "source/BrainCo_DexHand/BrainCo_DexHand/tasks"
          / "manager_based/dexsuite/config/Revo3/dexsuite_revo3_env_cfg_rotate_bulb.py")
CUSTOM_CONFIG = CONFIG.parents[1] / "RotateBulbCustom/env_cfg.py"


class _Term:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


def _combine(pos, quat, offset):
    xyz = quat[:, 1:]
    rotated = offset + 2 * torch.cross(xyz, torch.cross(xyz, offset, dim=-1)
                                     + quat[:, :1] * offset, dim=-1)
    return pos + rotated, quat


@pytest.fixture(params=[CONFIG, CUSTOM_CONFIG], ids=["standard", "custom"])
def reward_code(monkeypatch, request):
    # Execute the actual definitions. Only the simulator integration types and
    # frame-transform import are substituted; all reward logic is unmodified.
    math_module = ModuleType("isaaclab.utils.math")
    math_module.combine_frame_transforms = _combine
    assets_module = ModuleType("isaaclab.assets")
    assets_module.RigidObject = object
    monkeypatch.setitem(__import__("sys").modules, "isaaclab.utils.math", math_module)
    monkeypatch.setitem(__import__("sys").modules, "isaaclab.assets", assets_module)
    source_path = request.param
    tree = ast.parse(source_path.read_text())
    definitions = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and (node.name.startswith("rotate_bulb_")
                                              or node.name == "release_unscrewed_bulb")
        or isinstance(node, ast.ClassDef) and node.name.startswith("RotateBulb")
        and node.name != "RotateBulbPoseCommand"
    )]
    namespace = dict(torch=torch, math=math, ManagerTermBase=_Term, ManagerBasedRLEnv=object,
                     SceneEntityCfg=object, ContactSensor=object, combine_frame_transforms=_combine,
                     _INITIAL_SCREW_TURN=-8 * math.pi, _SCREW_TRAVEL=0.024, _SCREW_PITCH=0.006,
                     _SCREW_RELEASE_TOLERANCE=0.001)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"), namespace)
    return SimpleNamespace(**namespace, source_path=source_path)


class _Scene(dict):
    pass


def _env(progress=(0.0, 0.0)):
    n = len(progress)
    joint_pos = torch.tensor([[(-8 * math.pi) * (1 - p), -0.024 * (1 - p)] for p in progress])
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
    bulb = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=root_quat.clone(),
                           root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3)))
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=root_quat.clone(),
                            body_pos_w=torch.full((n, 5, 3), 0.3)))
    scene = _Scene(lamp=SimpleNamespace(joint_names=["screw_turn", "screw_slide"],
                                      data=SimpleNamespace(joint_pos=joint_pos,
                                          default_joint_pos=torch.tensor([[-8 * math.pi, -.024]]).repeat(n, 1))),
                   object=bulb, robot=robot, support=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(n, 3))))
    scene.sensors = {}
    for finger in ("thumb", "index", "mid", "ring", "pinky"):
        force = torch.zeros(n, 1, 1, 3)
        if finger in ("thumb", "index"):
            force[..., 2] = 2.0
        scene.sensors[f"right_{finger}dip_roll_rubber_link_object_s"] = SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=force))
    command = torch.zeros(n, 7)
    command[:, 3] = 1
    return SimpleNamespace(scene=scene, num_envs=n, device="cpu", step_dt=0.1,
                           _bulb_released=torch.zeros(n, dtype=torch.bool),
                           command=command, command_manager=SimpleNamespace(get_command=lambda _: command))


def _set_progress(env, progress):
    p = torch.as_tensor(progress)
    env.scene["lamp"].data.joint_pos[:, 0] = -8 * math.pi * (1 - p)
    env.scene["lamp"].data.joint_pos[:, 1] = -0.024 * (1 - p)


def _params():
    return dict(command_name="object_pose", asset_cfg=SimpleNamespace(name="robot"),
                align_asset_cfg=SimpleNamespace(name="object"))


def test_progress_counts_full_turns_and_requires_axial_motion(reward_code):
    env = _env((0.0, 0.25, 0.5, 1.0))
    torch.testing.assert_close(reward_code.rotate_bulb_screw_progress(env)[:, 0], torch.tensor([0., .25, .5, 1.]))
    env.scene["lamp"].data.joint_pos[-1, 1] = -0.012
    assert not reward_code.rotate_bulb_screw_complete(env).any()


def test_unscrew_cannot_reward_backtracking_or_stationary_contact(reward_code):
    env = _env()
    term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
    _set_progress(env, (.5, .25))
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([2., 1.]))
    _set_progress(env, (.2, .1))
    assert not term(env).any()
    _set_progress(env, (.5, .25))
    assert not term(env).any()
    _set_progress(env, (1., 1.))
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([2.5, 3.5]))
    assert not term(env).any()


def test_unheld_rotation_and_regrasp_do_not_repay_old_progress(reward_code):
    env = _env()
    term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
    thumb = env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w
    thumb.zero_()
    _set_progress(env, (.5, .5))
    assert not term(env).any()
    thumb[..., 2] = 2
    assert not term(env).any()


def test_partial_reset_only_clears_selected_environment_progress(reward_code):
    env = _env()
    term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
    _set_progress(env, (1., 1.))
    term(env)
    _set_progress(env, (.1, 1.))
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.best, torch.tensor([.4, 4.]))
    torch.testing.assert_close(term.contact_memory, torch.tensor([0., 1.]))
    _set_progress(env, (.25, 1.))
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([.6, 0.]))
    assert term.completion_paid.tolist() == [False, True]
    _set_progress(env, (0., 0.))
    term.reset()
    assert not term.best.any()
    assert not term.completion_paid.any()
    assert not term.contact_memory.any()


def test_progress_credit_is_independent_of_control_frequency(reward_code):
    totals = []
    for dt in (.1, .02):
        env = _env((0.,))
        env.step_dt = dt
        term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
        total = 0.
        for p in torch.linspace(0., 1., round(1 / dt) + 1)[1:]:
            _set_progress(env, (p,))
            total += (term(env) * dt).item()
        totals.append(total)
    assert totals == pytest.approx([4.5, 4.5])


def test_equal_rotation_pays_equal_credit_at_different_starting_depths(reward_code):
    env = _env((0., 0., 0., 0.))
    depths = torch.tensor([.25, .5, 3., 4.])
    env._bulb_start_joints = torch.stack((-2 * math.pi * depths, -.006 * depths), dim=-1)
    env.scene["lamp"].data.joint_pos[:] = env._bulb_start_joints
    term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
    # Each bulb rotates 45 degrees and rises 0.75 mm, without reaching its exit.
    env.scene["lamp"].data.joint_pos += torch.tensor([math.pi / 4, .00075])
    torch.testing.assert_close(reward_code.rotate_bulb_unscrewed_turns(env), torch.full((4,), .125))
    torch.testing.assert_close(term(env) * env.step_dt * 40, torch.full((4,), 5.))
    assert not term.completion_paid.any()


def test_turn_credit_requires_matching_axial_travel_and_stays_within_start(reward_code):
    env = _env((0.,) * 6)
    # Angle-only, slide-only, lagging slide, lagging angle, deeper motion, overshoot.
    angular_turns = torch.tensor([1., 0., 2., 1., -1., 5.])
    axial_turns = torch.tensor([0., 1., 1., 2., 1., 5.])
    env.scene["lamp"].data.joint_pos += torch.stack(
        (2 * math.pi * angular_turns, .006 * axial_turns), dim=-1,
    )
    torch.testing.assert_close(
        reward_code.rotate_bulb_unscrewed_turns(env), torch.tensor([0., 0., 1., 1., 0., 4.]),
    )


def test_regrasp_credit_decays_to_zero_after_grace_and_does_not_repay(reward_code):
    env = _env((0.,))
    env.step_dt = .05
    term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
    assert not term(env).any()  # Establish opposing contact without moving.
    thumb = env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w
    thumb.zero_()
    credits = []
    for step in range(1, 6):
        _set_progress(env, (step / 16,))  # A quarter turn every 50 ms.
        credits.append((term(env) * env.step_dt).item())
    assert credits == pytest.approx([.1875, .125, .0625, 0., 0.])
    assert not term.contact_memory.any()
    thumb[..., 2] = 2
    assert not term(env).any()  # Expired or partly credited travel stays consumed.


def test_zero_grace_disables_contact_memory(reward_code):
    env = _env((0.,))
    term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
    term(env, regrasp_grace_s=0.)
    env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w.zero_()
    _set_progress(env, (.25,))
    assert not term(env, regrasp_grace_s=0.).any()
    assert not term.contact_memory.any()


def test_release_freezes_turn_history_and_clears_regrasp_memory(reward_code):
    env = _env((.99,))
    term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
    term(env)
    prior_best = term.best.clone()
    env._bulb_released[:] = True
    _set_progress(env, (1.,))
    # Strictly held release can pay the milestone, never the free follower's travel.
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([.5]))
    torch.testing.assert_close(term.best, prior_best)
    assert not term.contact_memory.any()
    for progress in (.5, 1., 1.1):
        _set_progress(env, (progress,))
        assert not term(env).any()
        torch.testing.assert_close(term.best, prior_best)


def test_grasp_shaping_is_continuous_saturating_and_stops_at_completion(reward_code):
    env = _env((0.,) * 5)
    forces_by_finger = {
        "thumb": [0., .5, 1., 2., .2],
        "index": [0., 0., .5, 2., .2],
        "mid": [0., 0., .75, 0., .6],
    }
    for finger, forces in forces_by_finger.items():
        env.scene.sensors[f"right_{finger}dip_roll_rubber_link_object_s"].data.force_matrix_w[:, 0, 0, 2] = (
            torch.tensor(forces)
        )
    torch.testing.assert_close(
        reward_code.rotate_bulb_contact_scores(env),
        torch.tensor([[0., 0.], [.5, 0.], [1., .75], [1., 1.], [.2, .6]]),
    )
    torch.testing.assert_close(
        reward_code.rotate_bulb_grasp_shaping(env), torch.tensor([0., .125, .8125, 1., .3]),
    )
    torch.testing.assert_close(
        reward_code.rotate_bulb_contact_scores(env, force_scale=2.)[3], torch.ones(2),
    )
    _set_progress(env, (1.,) * 5)
    assert not reward_code.rotate_bulb_grasp_shaping(env).any()
    env._bulb_released[:] = True
    _set_progress(env, (0.,) * 5)
    assert not reward_code.rotate_bulb_grasp_shaping(env).any()


def test_soft_contact_pays_progress_but_completion_and_success_remain_strict(reward_code):
    env = _env((0.,) * 3)
    weak_forces = torch.tensor([.1, .5, 1.])
    for finger in ("thumb", "index"):
        env.scene.sensors[f"right_{finger}dip_roll_rubber_link_object_s"].data.force_matrix_w[:, 0, 0, 2] = weak_forces
    term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
    _set_progress(env, (.25,) * 3)
    torch.testing.assert_close(term(env) * env.step_dt, weak_forces)
    _set_progress(env, (1.,) * 3)
    torch.testing.assert_close(term(env) * env.step_dt, 3 * weak_forces)
    assert not term.completion_paid.any()
    assert not reward_code.rotate_bulb_hand_contacts(env, 1.).any()
    env._bulb_released[:] = True
    env.scene["object"].data.root_pos_w[:, 2] = .1
    env.command[:, 2] = .1
    success = reward_code.RotateBulbStableSuccess(SimpleNamespace(params={}), env)
    for _ in range(5):
        assert not success(env, pos_std=.03, **_params()).any()
    for finger in ("thumb", "index"):
        env.scene.sensors[f"right_{finger}dip_roll_rubber_link_object_s"].data.force_matrix_w[..., 2] = 2.
    torch.testing.assert_close(term(env) * env.step_dt, torch.full((3,), .5))
    assert not term(env).any()
    for _ in range(3):
        result = success(env, pos_std=.03, **_params())
    assert result.all()


@pytest.mark.parametrize("force_scale", [0., -1., float("inf"), float("nan")])
def test_contact_scores_reject_invalid_force_scales(reward_code, force_scale):
    with pytest.raises(ValueError, match="force_scale"):
        reward_code.rotate_bulb_contact_scores(_env(), force_scale=force_scale)


@pytest.mark.parametrize("grace", [-.1, float("inf"), float("nan")])
def test_unscrew_rejects_invalid_grace_durations(reward_code, grace):
    env = _env()
    term = reward_code.RotateBulbUnscrewProgress(SimpleNamespace(params={}), env)
    with pytest.raises(ValueError, match="regrasp_grace_s"):
        term(env, regrasp_grace_s=grace)


def test_transport_needs_completed_screw_and_opposing_contacts(reward_code):
    env = _env((0., .5, 1., 1., 1.))
    env._bulb_released[2:4] = True
    env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"].data.force_matrix_w[3] = 0
    # Even exact target coincidence cannot earn transport reward too early.
    actual = reward_code.rotate_bulb_position_command_error_tanh(env, std=.2, **_params())
    torch.testing.assert_close(actual, torch.tensor([0., 0., 1., 0., 0.]))
    env.scene["robot"].data.root_pos_w[:, 0] = 2
    env.command[:, 0] = -2
    torch.testing.assert_close(reward_code.rotate_bulb_position_command_error_tanh(env, std=.2, **_params()), actual)


def test_approach_targets_shell_and_stops_after_unscrewing(reward_code):
    env = _env()
    cfg = SimpleNamespace(body_ids=list(range(5)), name="robot")
    distant = reward_code.rotate_bulb_reach_shell(env, .2, cfg)
    env.scene["object"].data.root_quat_w[:] = torch.tensor([math.sqrt(.5), 0., -math.sqrt(.5), 0.])
    env.scene["robot"].data.body_pos_w[:] = torch.tensor([0., 0., .04])
    close = reward_code.rotate_bulb_reach_shell(env, .2, cfg)
    assert (close > distant).all()
    torch.testing.assert_close(close, torch.ones(2))
    _set_progress(env, (1., 1.))
    assert not reward_code.rotate_bulb_reach_shell(env, .2, cfg).any()


def test_success_requires_position_and_continuous_hold(reward_code):
    env = _env((1., 1.))
    env._bulb_released[:] = True
    env.scene["object"].data.root_pos_w[:, 2] = .1
    env.command[:, 2] = .1
    term = reward_code.RotateBulbStableSuccess(SimpleNamespace(params={}), env)
    params = dict(_params(), pos_std=.03, hold_duration=.3)
    for _ in range(2):
        assert not term(env, **params).any()
    assert term(env, **params).all()
    term.reset(torch.tensor([0]))
    assert term(env, **params).tolist() == [0., 1.]
    env.scene["object"].data.root_pos_w[:, 0] = .1
    assert not term(env, **params).any()
    assert not term.hold_time.any()
    env.scene["object"].data.root_pos_w[:, 0] = 0.
    for _ in range(2):
        assert not term(env, **params).any()
    assert term(env, **params).all()


def test_success_ignores_legacy_clearance_contact_and_speed_gates(reward_code):
    env = _env((1.,))
    env._bulb_released[:] = True
    env.scene["object"].data.root_pos_w[:, 2] = .1
    env.command[:, 2] = .1
    _set_progress(env, (0.,))
    env.scene.sensors["right_indexdip_roll_rubber_link_object_s"].data.force_matrix_w.zero_()
    env.scene["object"].data.root_lin_vel_w[:, 0] = 1.
    env.scene["object"].data.root_ang_vel_w[:, 2] = 2.
    term = reward_code.RotateBulbStableSuccess(SimpleNamespace(params={}), env)
    for _ in range(2):
        assert not term(env, pos_std=.03, **_params()).any()
    assert term(env, pos_std=.03, **_params()).all()


def test_progress_uses_each_environments_actual_start_and_survives_release(reward_code):
    env = _env()
    env._bulb_start_joints = torch.tensor([[-math.pi / 2, -.0015], [-8 * math.pi, -.024]])
    env.scene["lamp"].data.joint_pos[:] = env._bulb_start_joints / 2
    torch.testing.assert_close(reward_code.rotate_bulb_screw_progress(env), torch.full((2, 2), .5))
    env._bulb_released[0] = True
    # The empty screw follower can move after detachment without undoing the task phase.
    env.scene["lamp"].data.joint_pos[0] = env._bulb_start_joints[0]
    torch.testing.assert_close(reward_code.rotate_bulb_screw_progress(env), torch.tensor([[1., 1.], [.5, .5]]))


def test_release_changes_only_completed_connections_and_is_idempotent(reward_code):
    env = _env((.5, 1.))
    calls = [[], []]
    env._bulb_attachment_attrs = [SimpleNamespace(Set=values.append) for values in calls]
    reward_code.release_unscrewed_bulb(env)
    reward_code.release_unscrewed_bulb(env)
    assert calls == [[], [False]]
    assert env._bulb_released.tolist() == [False, True]


def test_success_requires_physical_release(reward_code):
    env = _env((1.,))
    env.scene["object"].data.root_pos_w[:, 2] = .1
    env.command[:, 2] = .1
    term = reward_code.RotateBulbStableSuccess(SimpleNamespace(params={}), env)
    for _ in range(2):
        assert not term(env, pos_std=.03, **_params()).any()
    assert not term(env, pos_std=.03, **_params()).any()


def test_drop_detection_requires_release_and_table_contact(reward_code):
    env = _env((0., 1., 1.))
    env._bulb_released[1:] = True
    forces = torch.zeros(3, 1, 1, 3)
    forces[:2, ..., 2] = 2
    env.scene.sensors["bulb_table_contact"] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=forces))
    assert reward_code.rotate_bulb_dropped(env).tolist() == [False, True, False]


def test_phase_observation_distinguishes_release_and_time_remaining(reward_code):
    env = _env()
    env._bulb_released[1] = True
    env.episode_length_buf = torch.tensor([0, 300])
    env.max_episode_length = 600
    torch.testing.assert_close(reward_code.rotate_bulb_phase_state(env), torch.tensor([[0., 1.], [1., .5]]))
