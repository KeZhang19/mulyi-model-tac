"""Insertion recovery and precision shaping without weakening physical success."""

import math

import pytest

from test_d_peg_insertion_rewards import code, snapshot, state  # noqa: F401


def test_contact_then_recenter_can_continue_insertion(code):
    s = state(code)
    s.update(snapshot(code, (.003,)), 1 / 30, 1)
    s.update(snapshot(code, (.004,), xy=((.001, 0.),)), 1 / 30, 2)
    assert not s.entered.item()
    assert s.entry_provenance.item()
    assert s.components["depth"].item() == 0
    assert s.hold_time.item() == 0
    s.update(snapshot(code, (.006,)), 1 / 30, 3)
    assert s.entered.item()
    assert s.components["depth"].item() == pytest.approx(40 * .003 / .028, abs=1e-5)
    assert s.components["entry"].item() == 0  # Already paid at 3 mm.
    for step in range(4, 20):
        s.update(snapshot(code, (.028,)), 1 / 30, step)
    assert s.success.item()


@pytest.mark.parametrize("bad", [dict(depths=(-.001,)), dict(depths=(.004,), xy=((.03, 0.),)),
                                 dict(depths=(.032,))])
def test_withdrawal_leaving_aperture_or_floor_penetration_clears_provenance(code, bad):
    s = state(code)
    s.update(snapshot(code, (.003,)), .1, 1)
    s.update(snapshot(code, **bad), .1, 2)
    assert not s.entry_provenance.item()
    if bad["depths"][0] > 0:
        s.update(snapshot(code, (.028,)), .1, 3)
        assert not s.entered.item()
        assert not s.success.item()
        assert s.components["depth"].item() == 0


def test_precision_reward_survives_two_mm_milestone_and_is_not_repeatable(code):
    s = state(code)
    s.update(snapshot(code, (.003,), xy=((.0005, 0.),)), .1, 1)
    assert s.entry_paid.item()
    before = s.episode_sums["fine_alignment"].item()
    s.update(snapshot(code, (.003,)), .1, 2)
    assert s.components["alignment"].item() == 0  # Coarse v2 shaping stopped here.
    assert s.components["fine_alignment"].item() > 0
    assert s.episode_sums["fine_alignment"].item() > before
    budget = s.episode_sums["fine_alignment"].item()
    for step, x in enumerate((.0005, 0., .0005, 0.), 3):
        s.update(snapshot(code, (.003,), xy=((x, 0.),)), .1, step)
        assert s.components["fine_alignment"].item() == 0
    assert budget <= 8


def test_precision_score_resolves_small_signed_pose_errors_without_free_credit(code):
    centered = snapshot(code, (0.,))["geometry"]
    offset = snapshot(code, (0.,), xy=((.001, 0.),))["geometry"]
    assert code.d_peg_fine_alignment(centered).item() > code.d_peg_fine_alignment(offset).item()
    s = state(code)
    s.update(snapshot(code, (-.001,), held=False), .1, 1)
    s.update(snapshot(code, (-.001,)), .1, 2)
    assert s.components["fine_alignment"].item() == 0
    s.update(snapshot(code, (.005,), xy=((.03, 0.),)), .1, 3)
    s.update(snapshot(code, (.005,)), .1, 4)
    assert s.components["fine_alignment"].item() == 0


def test_grasp_loss_cost_is_time_scaled_and_never_rewards_stationary_grasp(code):
    totals = []
    for dt in (.1, 1 / 30):
        s = state(code)
        for step in range(1, round(1 / dt) + 1):
            s.update(snapshot(code, (-.02,), held=False), dt, step)
        totals.append(s.episode_sums["grasp_loss"].item())
        s.update(snapshot(code, (-.02,)), dt, step + 1)
        assert s.components["grasp_loss"].item() == 0
    assert totals == pytest.approx([-.2, -.2], abs=1e-6)


def test_transient_contact_never_relaxes_success_tolerances(code):
    s = state(code)
    s.update(snapshot(code, (.003,)), .1, 1)
    for step in range(2, 12):
        s.update(snapshot(code, (.028,), xy=((.0005, 0.),)), .1, step)
        assert not s.success.item()
        assert s.hold_time.item() == 0


def test_stability_reward_is_gated_to_deep_held_entry(code):
    s = state(code)
    s.update(snapshot(code, (.003,)), 1 / 30, 1)
    s.update(snapshot(code, (.028,)), 1 / 30, 2)
    assert s.components["stability"].item() > 0
    s.update(snapshot(code, (.010,)), 1 / 30, 3)
    assert s.components["stability"].item() == 0
    s.update(snapshot(code, (.028,), held=False), 1 / 30, 4)
    assert s.components["stability"].item() == 0
