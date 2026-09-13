"""Physical geometry and reward-history contracts without starting Isaac Sim."""

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


MODULE = (Path(__file__).resolve().parents[1] / "source/BrainCo_DexHand/BrainCo_DexHand/tasks"
          / "manager_based/dexsuite/mdp/d_peg_insertion.py")


@pytest.fixture
def code(monkeypatch):
    managers = ModuleType("isaaclab.managers")

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self._env, self.cfg = env, cfg

    managers.ManagerTermBase = ManagerTermBase
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    spec = importlib.util.spec_from_file_location("_d_peg_reward_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def snapshot(code, depths=(-.02,), xy=None, yaw=None, tilt=None, held=True,
             linear_speed=0., angular_speed=0., socket_force=0., dropped=False,
             outbound=False):
    n = len(depths)
    peg_pos = torch.zeros(n, 3)
    peg_pos[:, 2] = .05 - torch.tensor(depths)
    if xy is not None:
        peg_pos[:, :2] = torch.as_tensor(xy)
    q = torch.zeros(n, 4)
    q[:, 0] = 1
    if yaw is not None:
        angles = torch.as_tensor(yaw).expand(n)
        q[:, 0], q[:, 3] = (angles / 2).cos(), (angles / 2).sin()
    if tilt is not None:
        angles = torch.as_tensor(tilt).expand(n)
        q[:, 0], q[:, 2] = (angles / 2).cos(), (angles / 2).sin()
    socket_pos = torch.zeros(n, 3)
    socket_quat = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    geometry = code.d_peg_geometry(peg_pos, q, socket_pos, socket_quat)
    return dict(geometry=geometry, held=torch.as_tensor(held, dtype=torch.bool).expand(n).clone(),
                finite=torch.ones(n, dtype=torch.bool),
                linear_speed=torch.as_tensor(linear_speed).expand(n).clone(),
                angular_speed=torch.as_tensor(angular_speed).expand(n).clone(),
                socket_force=torch.as_tensor(socket_force).expand(n).clone(),
                dropped=torch.as_tensor(dropped, dtype=torch.bool).expand(n).clone(),
                outbound=torch.as_tensor(outbound, dtype=torch.bool).expand(n).clone())


def state(code, n=1):
    result = code.DPegRewardState(n, "cpu")
    result.reset(snapshot(code, (-.02,) * n), step_id=0)
    return result


def test_d_profile_rejects_centered_wrong_key_rotation_and_off_axis_lowering(code):
    good = snapshot(code, (.01,))["geometry"]
    assert good["contained"].item()
    wrong_yaw = snapshot(code, (.01,), yaw=math.pi)["geometry"]
    assert not wrong_yaw["contained"].item()
    outside = snapshot(code, (.01,), xy=((.03, 0.),))["geometry"]
    assert not outside["contained"].item()
    assert good["depth"].item() == pytest.approx(wrong_yaw["depth"].item())


def test_rim_cross_section_rejects_tilt_even_when_tip_center_fits(code):
    g = snapshot(code, (.015,), tilt=math.radians(12))["geometry"]
    assert g["xy"].item() == 0
    assert not g["contained"].item()
    assert g["max_violation"].item() > 0


def test_chamfer_admits_shallow_lead_in_but_not_deep_misalignment(code):
    shallow = snapshot(code, (.0005,), xy=((.001, 0.),))["geometry"]
    deep = snapshot(code, (.01,), xy=((.001, 0.),))["geometry"]
    assert shallow["contained"].item()
    assert not deep["contained"].item()


def test_depth_cannot_reward_floor_penetration_or_upside_down_peg(code):
    assert not snapshot(code, (.032,))["geometry"]["contained"].item()
    assert not snapshot(code, (.005,), tilt=math.pi)["geometry"]["contained"].item()


def test_geometry_is_invariant_to_socket_world_pose(code):
    pos = torch.tensor([[0., 0., .022]])
    identity = torch.tensor([[1., 0., 0., 0.]])
    original = code.d_peg_geometry(pos, identity, torch.zeros_like(pos), identity)
    q = torch.tensor([[math.cos(.7), 0., 0., math.sin(.7)]])
    offset = torch.tensor([[1., -2., .8]])
    changed = code.d_peg_geometry(code._rotate(q, pos) + offset, q, offset, q)
    for name in ("depth", "xy", "tilt", "yaw", "approach", "align"):
        torch.testing.assert_close(original[name], changed[name], atol=2e-6, rtol=0)
    assert torch.equal(original["contained"], changed["contained"])


def test_initial_or_side_inserted_pose_is_not_a_valid_entry(code):
    s = code.DPegRewardState(1, "cpu")
    s.reset(snapshot(code, (.005,), xy=((.03, 0.),)), step_id=0)
    s.update(snapshot(code, (.01,)), .1, 1)
    assert not s.entered.any()
    assert s.components["depth"].item() == 0
    assert s.components["entry"].item() == 0
    for step in range(2, 10):
        s.update(snapshot(code, (.028,)), .1, step)
    assert not s.success.any()


def test_unheld_entry_consumes_milestone_and_progress_without_paying(code):
    s = state(code)
    s.update(snapshot(code, (.003,), held=False), .1, 1)
    assert s.entered.item()
    assert s.entry_paid.item()
    assert s.components["depth"].item() == 0
    s.update(snapshot(code, (.003,)), .1, 2)
    assert s.components["depth"].item() == 0
    assert s.components["entry"].item() == 0
    s.update(snapshot(code, (-.001,)), .1, 3)
    s.update(snapshot(code, (.003,)), .1, 4)
    assert s.entered.item()
    assert s.components["entry"].item() == 0
    assert s.components["depth"].item() == 0


def test_progress_cannot_be_repaid_by_retreat_or_unheld_regrasp(code):
    s = state(code)
    s.update(snapshot(code, (.005,)), .1, 1)
    first = s.components["depth"].item()
    assert first == pytest.approx(40 * .005 / .028)
    for step, depth in enumerate((.002, .005), 2):
        s.update(snapshot(code, (depth,)), .1, step)
        assert s.components["depth"].item() == 0
        assert s.components["entry"].item() == 0
    s.update(snapshot(code, (.012,), held=False), .1, 4)
    assert s.components["depth"].item() == 0
    s.update(snapshot(code, (.012,)), .1, 5)
    assert s.components["depth"].item() == 0
    s.update(snapshot(code, (.014,)), .1, 6)
    assert s.components["depth"].item() == pytest.approx(40 * .002 / .028, abs=1e-5)


def test_repeated_reads_in_same_step_do_not_advance_hold_or_progress(code):
    s = state(code)
    current = snapshot(code, (.028,))
    first = s.update(current, .1, 1).clone()
    hold = s.hold_time.clone()
    for _ in range(10):
        torch.testing.assert_close(s.update(current, .1, 1), first)
        torch.testing.assert_close(s.hold_time, hold)
    assert not s.success.any()
    for step in range(2, 6):
        s.update(current, .1, step)
    assert s.success.item()
    assert s.components["success"].item() == 80
    s.update(current, .1, 6)
    assert s.reward_rate.item() == 0
    assert s.components["success"].item() == 0


@pytest.mark.parametrize("change", [dict(linear_speed=.03), dict(angular_speed=.3),
                                   dict(held=False), dict(xy=((.0005, 0.),)),
                                   dict(yaw=math.radians(4)), dict(depths=(.026,))])
def test_success_requires_continuous_strict_stability(code, change):
    s = state(code)
    for step in range(1, 4):
        s.update(snapshot(code, (.028,)), .1, step)
    args = dict(depths=(.028,))
    args.update(change)
    s.update(snapshot(code, **args), .1, 4)
    assert not s.success.any()
    assert not s.hold_time.any()


def test_partial_reset_preserves_every_other_environment_history(code):
    s = state(code, 3)
    for step in range(1, 4):
        s.update(snapshot(code, (.028, .028, .028)), .1, step)
    saved = {name: value.clone() for name, value in vars(s).items() if isinstance(value, torch.Tensor)}
    components = {name: value.clone() for name, value in s.components.items()}
    s.reset(snapshot(code, (.028, -.02, .028)), torch.tensor([1]), step_id=3)
    for name, old in saved.items():
        torch.testing.assert_close(getattr(s, name)[[0, 2]], old[[0, 2]])
    for name, old in components.items():
        torch.testing.assert_close(s.components[name][[0, 2]], old[[0, 2]])
    assert s.best_depth[1].item() == 0
    assert not s.entered[1]
    assert s.hold_time[1].item() == 0
    s.update(snapshot(code, (.028, -.02, .028)), .1, 3)
    assert s.hold_time[1].item() == 0


def test_progress_and_event_totals_do_not_depend_on_control_dt(code):
    totals = []
    for dt in (.1, .02):
        s = state(code)
        total = dict(depth=0., entry=0., success=0.)
        for step in range(1, round(1 / dt) + 1):
            depth = min(.028, -.02 + .096 * step * dt)
            s.update(snapshot(code, (depth,)), dt, step)
            for name in total:
                total[name] += s.components[name].item()
        totals.append(total)
    assert totals[0] == pytest.approx(dict(depth=40., entry=5., success=80.), abs=1e-4)
    assert totals[1] == pytest.approx(totals[0], abs=1e-4)


def test_force_cost_is_linear_then_capped_and_time_is_per_second(code):
    s = state(code, 4)
    s.update(snapshot(code, (-.02,) * 4, socket_force=[30., 45., 60., 100.]), .1, 1)
    torch.testing.assert_close(s.components["force"], torch.tensor([0., -.1, -.2, -.2]))
    torch.testing.assert_close(s.components["time"], torch.full((4,), -.01))


@pytest.mark.parametrize("failure", ["dropped", "outbound"])
def test_failure_is_a_single_20_point_event(code, failure):
    s = state(code)
    kwargs = {failure: True}
    s.update(snapshot(code, **kwargs), .1, 1)
    assert s.reward_rate.item() * .1 == pytest.approx(-20)
    assert s.failed.item()
    assert not s.nonfinite.item()
    s.update(snapshot(code, **kwargs), .1, 2)
    assert s.reward_rate.item() == 0


def test_invalid_measurement_terminates_without_nonfinite_reward(code):
    s = state(code)
    current = snapshot(code, (float("nan"),))
    s.update(current, .1, 1)
    assert s.failed.item()
    assert s.nonfinite.item()
    assert torch.isfinite(s.reward_rate).all()
    assert s.components["failure"].item() == -20
    assert s.reward_rate.item() * .1 == pytest.approx(-20)


@pytest.mark.parametrize("dt", [0., -.1, float("nan"), float("inf")])
def test_invalid_dt_is_rejected(code, dt):
    with pytest.raises(ValueError, match="step_dt"):
        state(code).update(snapshot(code), dt, 1)


def fake_env(n=2):
    class Scene(dict):
        pass

    quat = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    peg = SimpleNamespace(root_pos_w=torch.tensor([[0., 0., .07]]).repeat(n, 1),
                          root_quat_w=quat.clone(), root_lin_vel_w=torch.zeros(n, 3),
                          root_ang_vel_w=torch.zeros(n, 3))
    socket = SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=quat.clone())
    robot = SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=quat.clone(),
                            joint_pos=torch.zeros(n, 28), joint_vel=torch.zeros(n, 28))
    scene = Scene(object=SimpleNamespace(data=peg), socket=SimpleNamespace(data=socket),
                  robot=SimpleNamespace(data=robot))
    scene.env_origins = torch.zeros(n, 3)
    scene.sensors = {}
    for finger in ("thumb", "index", "mid", "ring", "pinky"):
        forces = torch.zeros(n, 1, 1, 3)
        if finger in ("thumb", "index"):
            forces[..., 2] = 2
        scene.sensors[f"right_{finger}dip_roll_rubber_link_object_s"] = SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=forces))
    for name in ("peg_socket_contact", "peg_table_contact"):
        scene.sensors[name] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(n, 1, 1, 3)))
    cfg = SimpleNamespace(insertion_depth_m=.028, socket_mouth_height_m=.05,
                          terminations=SimpleNamespace(object_out_of_bound=SimpleNamespace(
                              params={"in_bound_range": {"x": (-1., 1.), "y": (-1., 1.), "z": (-1., 1.)}})))
    return SimpleNamespace(scene=scene, cfg=cfg, num_envs=n, device="cpu", common_step_counter=0,
                           step_dt=.1, episode_length_buf=torch.zeros(n), max_episode_length=100,
                           extras={})


def test_observations_can_be_created_before_reward_manager(code):
    env = fake_env()
    assert code.d_peg_insertion_progress(env).shape == (2, 2)
    assert code.d_peg_insertion_phase(env).shape == (2, 2)
    assert not hasattr(env, "_d_peg_state")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pose_only_geometry_matches_full_geometry_for_random_and_invalid_poses(code, dtype):
    generator = torch.Generator().manual_seed(715)
    peg = torch.randn(128, 3, generator=generator, dtype=dtype) * .04
    socket = torch.randn(128, 3, generator=generator, dtype=dtype) * .02
    peg_q = torch.randn(128, 4, generator=generator, dtype=dtype)
    socket_q = torch.randn(128, 4, generator=generator, dtype=dtype)
    peg[0, 2], socket[1, 0] = torch.nan, torch.inf
    peg_q[2], socket_q[3] = 0., 0.
    peg_q[4, 1], socket_q[5, 0] = torch.nan, torch.inf
    full = code.d_peg_geometry(peg, peg_q, socket, socket_q)
    pose = code.d_peg_geometry(peg, peg_q, socket, socket_q, compute_containment=False)
    for name in ("tip", "depth", "xy", "tilt", "yaw", "finite"):
        torch.testing.assert_close(pose[name], full[name], atol=0, rtol=0, equal_nan=True)


def test_observations_skip_containment_and_read_resets_within_the_same_step(code, monkeypatch):
    env = fake_env()
    reward = code.DPegInsertionState(SimpleNamespace(), env)
    peg = env.scene["object"].data
    peg.root_pos_w[:, 2] = .022
    env.common_step_counter = 1
    code.d_peg_success(env)
    before = code.d_peg_insertion_progress(env).clone()
    # Physical reset precedes RewardManager.reset without advancing this step.
    peg.root_pos_w[0] = torch.tensor([.01, 0., .07])
    reward.reset(torch.tensor([0]))

    def reject_containment(*args, **kwargs):
        raise AssertionError("observations must not construct or clip the shaft")

    monkeypatch.setattr(code, "_shaft_mesh", reject_containment)
    after = code.d_peg_insertion_progress(env)
    phase = code.d_peg_insertion_phase(env)
    assert env.common_step_counter == 1
    torch.testing.assert_close(after[0], torch.tensor([0., .01 / .01275]))
    torch.testing.assert_close(after[1], before[1])
    assert phase[:, 0].tolist() == [0., 1.]
    assert reward.state.last_step.tolist() == [1, 1]


def test_static_shaft_cache_separates_geometry_and_dtype_without_pose_mutation(code):
    from dataclasses import replace

    cfg = code.DPegGeometryConfig()
    original = tuple(value.clone() for value in code._shaft_mesh(cfg, torch.device("cpu"), torch.float32))
    changed_cfg = replace(cfg, shaft_radius_m=.014, shaft_length_m=.04)
    changed = code._shaft_mesh(changed_cfg, torch.device("cpu"), torch.float64)
    assert changed[0].dtype == torch.float64
    assert changed[0][:, 2].amax().item() == pytest.approx(.04)
    snapshot(code, (.015,), tilt=math.radians(12))
    restored = code._shaft_mesh(cfg, torch.device("cpu"), torch.float32)
    for before, after in zip(original, restored):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    # Warm a fresh cache inside inference mode, then use it with grad enabled.
    code._shaft_mesh.cache_clear()
    with torch.inference_mode():
        snapshot(code, (.01,))
    pos = torch.tensor([[0., 0., .04]], requires_grad=True)
    quat = torch.tensor([[1., 0., 0., 0.]])
    geometry = code.d_peg_geometry(pos, quat, torch.zeros_like(pos), quat)
    geometry["max_violation"].sum().backward()
    assert torch.isfinite(pos.grad).all()


def test_termination_reward_and_metrics_share_one_cached_update(code):
    env = fake_env()
    reward = code.DPegInsertionState(SimpleNamespace(), env)
    env.scene["object"].data.root_pos_w[:, 2] = .022
    env.common_step_counter += 1
    assert not code.d_peg_success(env).any()  # Terminations run before rewards.
    hold = reward.state.hold_time.clone()
    paid = reward(env).clone()
    assert not code.d_peg_dropped(env).any()
    assert not code.d_peg_out_of_bounds(env).any()
    metrics = code.d_peg_state_metrics(env)
    torch.testing.assert_close(reward.state.hold_time, hold)
    torch.testing.assert_close(reward(env), paid)
    torch.testing.assert_close(metrics["reward_entry"], torch.full((2,), 5.))
    assert "DPeg/reward_depth" in env.extras["log"]


def test_nonfinite_public_termination_and_partial_adapter_reset(code):
    env = fake_env()
    reward = code.DPegInsertionState(SimpleNamespace(), env)
    env.scene["object"].data.root_pos_w[0, 2] = float("nan")
    env.common_step_counter += 1
    assert code.d_peg_nonfinite(env).tolist() == [True, False]
    torch.testing.assert_close(reward(env), torch.tensor([-200., -.1]))
    env.scene["object"].data.root_pos_w[0, 2] = .07
    reward.reset(torch.tensor([0]))
    assert not code.d_peg_nonfinite(env).any()
    assert reward.state.reward_rate[0].item() == 0
    assert reward.state.reward_rate[1].item() == pytest.approx(-.1)


@pytest.mark.parametrize("field", ["joint_pos", "joint_vel", "root_pos_w", "root_quat_w"])
def test_robot_nonfinite_states_are_not_lost_by_task_termination_override(code, field):
    env = fake_env()
    reward = code.DPegInsertionState(SimpleNamespace(), env)
    getattr(env.scene["robot"].data, field)[1, 0] = float("nan")
    env.common_step_counter += 1
    assert code.d_peg_nonfinite(env).tolist() == [False, True]
    assert reward(env)[1].item() * env.step_dt == pytest.approx(-20)


def test_terminal_success_and_episode_components_survive_manager_reset_order(code):
    env = fake_env()
    command = SimpleNamespace(metrics={"success": torch.zeros(2)})
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    reward = code.DPegInsertionState(SimpleNamespace(), env)
    env.scene["object"].data.root_pos_w[0, 2] = .022
    for step in range(1, 6):
        env.common_step_counter = step
        code.d_peg_success(env)
        reward(env)
    assert reward.state.success.tolist() == [True, False]
    untouched = {name: value[1].clone() for name, value in reward.state.episode_sums.items()}
    # Isaac Lab clears extras after termination, resets the physical state, then
    # resets rewards before resetting/logging the command manager.
    env.extras["log"] = {}
    env.scene["object"].data.root_pos_w[0, 2] = .07
    reward.reset(torch.tensor([0]))
    assert env.extras["log"]["DPeg/EpisodeSuccess"].item() == 1
    assert env.extras["log"]["DPeg/EpisodeReward_success"].item() == 80
    assert env.extras["log"]["DPeg/EpisodeReward_depth"].item() == pytest.approx(40)
    assert command.metrics["success"].tolist() == [1., 0.]
    assert not reward.state.success.any()
    for name, before in untouched.items():
        torch.testing.assert_close(reward.state.episode_sums[name][1], before)
        assert reward.state.episode_sums[name][0].item() == 0


def test_reward_geometry_contract_matches_checked_in_asset_metadata(code, tmp_path):
    metadata_path = MODULE.parents[7] / "assets/d_peg_insertion/metadata.json"
    # Resolve from repository rather than a simulator/package installation.
    metadata_path = Path(__file__).resolve().parents[1] / "assets/d_peg_insertion/metadata.json"
    cfg = code.validate_d_peg_geometry_metadata(metadata_path)
    assert cfg == code.DPegGeometryConfig()
    metadata = json.loads(metadata_path.read_text())
    metadata["dimensions_m"]["shaft_radius"] = .011
    altered = tmp_path / "metadata.json"
    altered.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="shaft_radius"):
        code.validate_d_peg_geometry_metadata(altered)
    with pytest.raises(ValueError, match="insertion_depth_m"):
        code.validate_d_peg_geometry_metadata(metadata_path, insertion_depth_m=.015)
    with pytest.raises(ValueError, match="socket_mouth_height_m"):
        code.validate_d_peg_geometry_metadata(metadata_path, socket_mouth_height_m=.04)


def test_approach_and_alignment_reward_continues_from_old_peak_to_real_mouth(code):
    s = state(code)
    for step, height in enumerate((.015, .010, .005, .002, .001, 0.), 1):
        s.update(snapshot(code, (-height,)), .1, step)
        assert s.components["approach"].item() > 0
        assert s.components["alignment"].item() > 0
    assert s.best_approach.item() == pytest.approx(1.)
    assert s.best_align.item() == pytest.approx(1.)
    assert s.episode_sums["approach"].item() <= 5
    assert s.episode_sums["alignment"].item() <= 8
    # Neither returning to an already visited height nor waiting pays again.
    for step, height in enumerate((.01, .002, 0., 0.), 7):
        s.update(snapshot(code, (-height,)), .1, step)
        assert s.components["approach"].item() == 0
        assert s.components["alignment"].item() == 0


def test_unheld_approach_consumes_progress_and_below_mouth_shaping_needs_entry(code):
    s = state(code)
    s.update(snapshot(code, (-.001,), held=False), .1, 1)
    s.update(snapshot(code, (-.001,)), .1, 2)
    assert s.components["approach"].item() == 0
    assert s.components["alignment"].item() == 0
    # Move below the rim outside the hole, then tunnel sideways to its center.
    s.update(snapshot(code, (.01,), xy=((.03, 0.),)), .1, 3)
    s.update(snapshot(code, (.01,)), .1, 4)
    assert not s.entered.item()
    for name in ("approach", "alignment", "entry", "depth"):
        assert s.components[name].item() == 0


def test_new_approach_budget_is_control_period_independent(code):
    totals = []
    for dt in (.1, .02):
        s = code.DPegRewardState(1, "cpu")
        s.reset(snapshot(code, (-.03,)), step_id=0)
        for step in range(1, round(.6 / dt) + 1):
            s.update(snapshot(code, (min(0., -.03 + .05 * step * dt),)), dt, step)
        totals.append([s.episode_sums[name].item() for name in ("approach", "alignment")])
    assert totals[0] == pytest.approx([5 * (1 - math.exp(-1)), 8 * (1 - math.exp(-1))], abs=1e-5)
    assert totals[1] == pytest.approx(totals[0], abs=1e-5)


def test_episode_diagnostics_integrate_once_and_partial_reset_only_finishes_selected(code):
    s = state(code, 2)
    first = snapshot(code, (.003, -.02))
    first["fingertip_forces"] = torch.tensor([[2., 4., 0., 0., 0.], [3., 5., 0., 0., 0.]])
    s.update(first, .1, 1)
    second = snapshot(code, (.01, -.02), held=[False, True], socket_force=[20., 0.])
    second["fingertip_forces"] = torch.tensor([[0., 4., 0., 0., 0.], [3., 5., 0., 0., 0.]])
    s.update(second, .1, 2)
    saved = {name: value.clone() for name, value in s.episode_diagnostics.items()}
    for _ in range(3):
        s.update(second, .1, 2)
    for name, before in saved.items():
        torch.testing.assert_close(s.episode_diagnostics[name], before)
    s.reset(snapshot(code, (-.02, -.02)), torch.tensor([0]), step_id=2)
    summary = s.completed_episode_diagnostics()
    assert summary["count"].item() == 1
    values = {name: value.item() for name, value in summary["sums"].items()}
    assert values["duration_s"] == pytest.approx(.2)
    assert values["held_fraction"] == pytest.approx(.5)
    assert values["mean_thumb_force_n"] == pytest.approx(1.)
    assert values["mean_other_force_n"] == pytest.approx(4.)
    assert values["mean_socket_force_n"] == pytest.approx(10.)
    assert values["first_grasp_loss_s"] == pytest.approx(.2)
    assert values["experienced_grasp_loss"] == 1
    assert values["max_valid_depth_m"] == pytest.approx(.01, abs=1e-6)
    assert values["held_entry"] == 1
    assert values["valid_entry"] == 1
    for name, before in saved.items():
        torch.testing.assert_close(s.episode_diagnostics[name][1], before[1])
    # A second reset with no intervening simulation never counts an episode.
    s.reset(snapshot(code, (-.02, -.02)), torch.tensor([0]), step_id=2)
    assert s.completed_episode_diagnostics()["count"].item() == 1


def test_pop_diagnostics_returns_fixed_keys_without_changing_lifetime_or_active_episodes(code):
    s = state(code, 2)
    empty = s.completed_episode_diagnostics(pop=True)
    assert empty["count"].item() == 0
    assert all(value.item() == 0 for value in empty["sums"].values())
    s.update(snapshot(code, (-.02, -.02)), .1, 1)
    s.reset(snapshot(code, (-.02, -.02)), torch.tensor([0]), step_id=1)
    active_duration = s.episode_diagnostics["elapsed_s"][1].clone()
    window = s.completed_episode_diagnostics(pop=True)
    assert window["count"].item() == 1
    assert window["sums"]["experienced_grasp_loss"].item() == 0
    assert window["sums"]["first_grasp_loss_s"].item() == pytest.approx(.1)
    again = s.completed_episode_diagnostics(pop=True)
    assert set(again["sums"]) == set(empty["sums"]) == set(window["sums"])
    assert again["count"].item() == 0
    assert s.completed_episode_diagnostics()["count"].item() == 1
    torch.testing.assert_close(s.episode_diagnostics["elapsed_s"][1], active_duration)
    window["sums"]["duration_s"].zero_()
    assert s.completed_episode_diagnostics()["sums"]["duration_s"].item() == pytest.approx(.1)


def test_episode_depth_diagnostic_rejects_tunneling_and_stops_on_terminal_state(code):
    s = state(code)
    s.update(snapshot(code, (.01,), xy=((.03, 0.),)), .1, 1)
    s.update(snapshot(code, (.028,)), .1, 2)
    assert s.episode_diagnostics["max_valid_depth_m"].item() == 0
    s.update(snapshot(code, dropped=True), .1, 3)
    before = {name: value.clone() for name, value in s.episode_diagnostics.items()}
    for step in range(4, 7):
        s.update(snapshot(code, (.028,)), .1, step)
    for name, value in before.items():
        torch.testing.assert_close(s.episode_diagnostics[name], value)
    s.reset(snapshot(code), step_id=6)
    summary = s.completed_episode_diagnostics()
    assert summary["sums"]["dropped"].item() == 1
    assert summary["sums"]["duration_s"].item() == pytest.approx(.3)
    assert summary["maxima"]["max_valid_depth_m"].item() == 0


def test_nonfinite_terminal_sample_keeps_completed_diagnostics_finite(code):
    s = state(code)
    s.update(snapshot(code, (float("nan"),)), .1, 1)
    s.reset(snapshot(code), step_id=1)
    summary = s.completed_episode_diagnostics()
    assert summary["count"].item() == 1
    assert summary["sums"]["nonfinite"].item() == 1
    assert all(torch.isfinite(value) for value in summary["sums"].values())
    assert all(torch.isfinite(value) for value in summary["maxima"].values())


def test_public_completed_episode_diagnostics_do_not_advance_physics_or_cached_state(code):
    env = fake_env()
    reward = code.DPegInsertionState(SimpleNamespace(), env)
    env.common_step_counter = 1
    reward(env)
    reward.reset(torch.tensor([0]))
    env.common_step_counter = 2
    before = reward.state.last_step.clone()
    totals = code.d_peg_diagnostic_totals(env)
    window = code.d_peg_pop_episode_diagnostics(env)
    assert totals["count"].item() == window["count"].item() == 1
    torch.testing.assert_close(reward.state.last_step, before)
    assert code.d_peg_pop_episode_diagnostics(env)["count"].item() == 0
