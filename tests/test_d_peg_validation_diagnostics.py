"""CPU checks that validation measures physical inputs rather than encoder bias."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "assets/d_peg_insertion/tools/validate_training.py"
SPEC = importlib.util.spec_from_file_location("d_peg_validation_diagnostics", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_env():
    depth = torch.zeros(2, 5, 1, 3, 4)
    depth[:, 4, 0, 1, 2] = .0003
    marker = torch.zeros(2, 5, 3, 5)
    marker[:, 4, 1, 2:4] = torch.tensor([3., -4.])
    # Large invalid flows must not make the reported valid contact flow nonzero.
    marker[:, 0, :, 2:4] = 999.
    valid = torch.zeros(2, 5, 3, dtype=torch.bool)
    valid[:, 4, 1] = True
    return SimpleNamespace(
        num_envs=2, common_step_counter=7,
        latest_tactile_inputs={"depth_m": depth, "marker": marker, "marker_valid": valid},
        latest_tactile_latent=torch.ones(2, 5, 4),
        tactile_policy_contract={"finger_order": ["little", "ring", "middle", "index", "thumb"]},
        scene=SimpleNamespace(sensors={"pressure": SimpleNamespace(
            _resolved_target_mesh_prim_path="/World/envs/env_0/Object/TactileSurface")}),
    )


def test_real_depth_and_valid_marker_statistics_are_in_policy_finger_order():
    env = make_env()
    original = {key: value.clone() for key, value in env.latest_tactile_inputs.items()}
    report = MODULE.tactile_signal_diagnostics(env)
    assert report["depth_has_contact_per_env"] == [True, True]
    thumb, little = report["per_finger"]["thumb"], report["per_finger"]["little"]
    assert thumb["latent_l2_per_env"] == [2., 2.]
    assert thumb["depth_max_m_per_env"] == pytest.approx([.0003, .0003])
    assert thumb["depth_active_pixels_per_env"] == [1, 1]
    assert thumb["marker_displacement_max_px_per_env"] == [5., 5.]
    assert thumb["marker_dx_dy_min_px_per_env"] == [[3., -4.], [3., -4.]]
    assert little["depth_active_pixels_per_env"] == [0, 0]
    assert little["marker_displacement_max_px_per_env"] == [0., 0.]
    assert little["marker_dx_dy_min_px_per_env"] == [[0., 0.], [0., 0.]]
    for name, value in original.items():
        torch.testing.assert_close(env.latest_tactile_inputs[name], value, atol=0, rtol=0)
    assert env.common_step_counter == 7


def test_nonzero_latent_does_not_hide_zero_depth_in_one_environment():
    env = make_env()
    env.latest_tactile_inputs["depth_m"][1] = 0
    result = MODULE.tactile_signal_diagnostics(env)
    assert result["depth_has_contact_per_env"] == [True, False]
    assert result["per_finger"]["thumb"]["latent_l2_per_env"] == [2., 2.]


def test_tactile_diagnostic_rejects_nonfinite_inputs():
    env = make_env()
    env.latest_tactile_inputs["depth_m"][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="depth_m"):
        MODULE.tactile_signal_diagnostics(env)
