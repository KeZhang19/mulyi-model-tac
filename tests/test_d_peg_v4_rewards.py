"""Full insertion completion, independent of grip and extra pose/speed gates."""

import math
from types import SimpleNamespace

import pytest
import torch

from test_d_peg_insertion_rewards import code, fake_env, snapshot, state  # noqa: F401


@pytest.mark.parametrize("dt", [1 / 30, .1, .02])
@pytest.mark.parametrize("depth", [.027, .028, .030005])
def test_full_insertion_requires_half_second_and_pays_once(code, dt, depth):
    s = state(code)
    current = snapshot(code, (depth,))
    # The last case is within the existing 10 um floor tolerance: containment,
    # rather than an independent 30 mm depth cutoff, is the upper constraint.
    assert current["geometry"]["contained"].item()
    steps = round(.5 / dt)
    for step in range(1, steps):
        s.update(current, dt, step)
        assert not s.success.item()
        assert s.components["success"].item() == 0
    assert s.hold_time.item() == pytest.approx((steps - 1) * dt)
    s.update(current, dt, steps)
    assert s.success.item()
    assert s.hold_time.item() == pytest.approx(.5)
    assert s.components["success"].item() == 80
    for step in (steps, steps, steps + 1, steps + 2):
        s.update(current, dt, step)
    assert s.episode_sums["success"].item() == 80
    assert s.components["success"].item() == 0


@pytest.mark.parametrize("change", [
    dict(held=False), dict(linear_speed=.03), dict(angular_speed=.3),
    dict(xy=((.0005, 0.),)), dict(yaw=math.radians(4)),
    dict(tilt=math.radians(3.05), xy=((-0.0007, 0.),)),
])
def test_contained_insertion_ignores_old_grip_precision_and_speed_gates(code, change):
    s = state(code)
    current = snapshot(code, (.027,), **change)
    assert current["geometry"]["contained"].item()
    for step in range(1, 16):
        s.update(current, 1 / 30, step)
    assert s.success.item()
    assert s.components["success"].item() == 80
    if change.get("held") is False:
        # Completion can be rewarded after release without changing the grip
        # requirements or rates of any process reward/penalty.
        for name in ("approach", "alignment", "fine_alignment", "entry", "depth", "stability"):
            assert s.episode_sums[name].item() == 0
        assert s.episode_sums["grasp_loss"].item() == pytest.approx(-.1)
        assert s.episode_sums["time"].item() == pytest.approx(-.05)


@pytest.mark.parametrize("interruption", [dict(depths=(.0269,)), dict(depths=(.027,), xy=((.001, 0.),))])
def test_depth_or_containment_loss_restarts_the_entire_hold(code, interruption):
    s = state(code)
    current = snapshot(code, (.027,))
    for step in range(1, 15):
        s.update(current, 1 / 30, step)
    s.update(snapshot(code, **interruption), 1 / 30, 15)
    assert s.hold_time.item() == 0
    assert not s.success.item()
    assert s.entry_provenance.item()  # Both interruptions remain in the aperture.
    for step in range(16, 30):
        s.update(current, 1 / 30, step)
        assert not s.success.item()
    s.update(current, 1 / 30, 30)
    assert s.success.item()
    assert s.episode_sums["success"].item() == 80


@pytest.mark.parametrize("initial, current", [
    (dict(depths=(.027,)), dict(depths=(.027,))),  # Initially inserted.
    (dict(depths=(.005,), xy=((.03, 0.),)), dict(depths=(.027,))),  # Side entry.
    (dict(depths=(-.02,)), dict(depths=(.027,), xy=((.03, 0.),))),  # Outside hole.
    (dict(depths=(-.02,)), dict(depths=(.027,), xy=((.001, 0.),))),  # Through wall.
    (dict(depths=(-.02,)), dict(depths=(.031,))),  # Through floor.
])
def test_depth_alone_cannot_bypass_entry_and_physical_containment(code, initial, current):
    s = code.DPegRewardState(1, "cpu")
    s.reset(snapshot(code, **initial), step_id=0)
    for step in range(1, 21):
        s.update(snapshot(code, **current), 1 / 30, step)
        assert not s.success.item()
        assert s.hold_time.item() == 0
        assert s.components["success"].item() == 0


@pytest.mark.parametrize("failure", ["dropped", "outbound", "finite"])
def test_failure_on_completion_step_prevents_final_success_reward(code, failure):
    s = state(code)
    current = snapshot(code, (.027,))
    for step in range(1, 15):
        s.update(current, 1 / 30, step)
    terminal = snapshot(code, (.027,))
    terminal[failure] = torch.tensor([failure != "finite"])
    s.update(terminal, 1 / 30, 15)
    assert s.failed.item()
    assert not s.success.item()
    assert s.hold_time.item() == 0
    assert s.components["success"].item() == 0
    assert s.components["failure"].item() == -20
    assert s.reward_rate.item() / 30 == pytest.approx(-20)
    s.update(current, 1 / 30, 16)
    assert not s.success.item()
    assert s.reward_rate.item() == 0


def test_released_success_is_shared_by_termination_reward_and_episode_metrics(code):
    env = fake_env()
    env.step_dt = 1 / 30
    command = SimpleNamespace(metrics={"success": torch.zeros(2)})
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    reward = code.DPegInsertionState(SimpleNamespace(), env)
    peg = env.scene["object"].data
    peg.root_pos_w[0] = torch.tensor([.0005, 0., .023])
    peg.root_lin_vel_w[0, 0] = .03
    peg.root_ang_vel_w[0, 2] = .3
    for sensor in env.scene.sensors.values():
        sensor.data.force_matrix_w[0].zero_()
    for step in range(1, 16):
        env.common_step_counter = step
        assert code.d_peg_success(env).tolist() == [step == 15, False]
        hold = reward.state.hold_time.clone()
        reward(env)
        metrics = code.d_peg_state_metrics(env)
        torch.testing.assert_close(reward.state.hold_time, hold)
        assert metrics["success"].tolist() == [float(step == 15), 0.]
    assert env.extras["log"]["DPeg/success"].item() == .5
    assert metrics["reward_success"].tolist() == [80., 0.]
    env.extras["log"] = {}
    peg.root_pos_w[0] = torch.tensor([0., 0., .07])
    reward.reset(torch.tensor([0]))
    assert env.extras["log"]["DPeg/EpisodeSuccess"].item() == 1
    assert env.extras["log"]["DPeg/EpisodeReward_success"].item() == 80
    assert command.metrics["success"].tolist() == [1., 0.]
    completed = code.d_peg_pop_episode_diagnostics(env)
    assert completed["count"].item() == 1
    assert completed["sums"]["success"].item() == 1
    assert code.d_peg_pop_episode_diagnostics(env)["count"].item() == 0
    assert not reward.state.success.any()
    assert reward.state.hold_time[0].item() == 0
