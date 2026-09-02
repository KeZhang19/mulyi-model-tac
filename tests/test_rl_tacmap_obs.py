from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import torch
import pytest

pytest.importorskip("pxr")
from BrainCo_DexHand.tasks.manager_based.dexsuite.mdp.observations import (
    build_pressure_rl_diffusion_kernel,
    diffuse_pressure_rl_values,
    _fots_signed_twist_angle_batched,
    _pool_tacsl_force_field_batched,
    _tacsl_force_field_from_relative_pose_batched,
    _tacmap_mean_ray_direction,
    _tacmap_penetration_from_distances,
    _tacmap_static_surface_reference,
    warpsdf_pressure_obs,
)
from BrainCo_DexHand.tasks.manager_based.dexsuite.mdp import observations as observations_module
from integrate.tensor_image_utils import depth_weighted_center_tensor, tacmap_strip


class _FakeEnv:
    num_envs = 2
    device = "cpu"


class _FakeRaySensor:
    def __init__(self):
        self._ray_directions_w = torch.tensor(
            [
                [[0.0, 0.0, 2.0]] * 4,
                [[0.0, -3.0, 0.0]] * 4,
            ],
            dtype=torch.float32,
        )


def test_tacmap_mean_ray_direction_uses_world_sensor_rays():
    direction = _tacmap_mean_ray_direction(_FakeRaySensor(), _FakeEnv(), rows=2, cols=2)

    expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
    assert direction is not None
    assert torch.allclose(direction, expected)


def test_tacmap_contact_shell_preserves_zero_shell_behavior():
    surface = torch.tensor([[0.008, 0.008, 0.008]], dtype=torch.float32)
    obj = torch.tensor([[0.0075, 0.0082, 0.0200]], dtype=torch.float32)
    valid = torch.tensor([[True, True, False]])

    depth = _tacmap_penetration_from_distances(surface, obj, valid, contact_shell_m=0.0)

    assert torch.allclose(depth, torch.tensor([[0.0005, 0.0, 0.0]], dtype=torch.float32))


def test_tacmap_contact_shell_reports_near_nonpenetrating_contact():
    surface = torch.tensor([[0.008, 0.008, 0.008]], dtype=torch.float32)
    obj = torch.tensor([[0.0075, 0.0082, 0.0095]], dtype=torch.float32)
    valid = torch.tensor([[True, True, True]])

    depth = _tacmap_penetration_from_distances(surface, obj, valid, contact_shell_m=0.001)

    assert torch.allclose(depth, torch.tensor([[0.0005, 0.0008, 0.0]], dtype=torch.float32))


def test_tacmap_static_surface_reference_is_loaded_sanitized_and_cached(tmp_path):
    path = tmp_path / "surface.npy"
    np.save(
        path,
        np.asarray([[[0.008, np.nan], [-1.0, 0.006]]], dtype=np.float32),
        allow_pickle=False,
    )
    env = SimpleNamespace(device="cpu")

    first = _tacmap_static_surface_reference(env, str(path), sensor_count=1, rows=2, cols=2)
    path.unlink()
    second = _tacmap_static_surface_reference(env, str(path), sensor_count=1, rows=2, cols=2)

    assert first is second
    torch.testing.assert_close(
        first,
        torch.tensor([[[0.008, 0.0], [0.0, 0.006]]], dtype=torch.float32),
    )


def test_tacmap_display_center_stays_tensor_until_final_image_transfer():
    depth = torch.zeros((3, 4), dtype=torch.float32)
    depth[1, 2] = 1.0

    center, active = depth_weighted_center_tensor(depth)
    image = tacmap_strip(
        depth,
        tacmap_raw=depth,
        scale=4,
        gamma=1.0,
        view="raw",
        display_max_m=1.0,
    )

    assert torch.equal(center, torch.tensor([2.0, 1.0]))
    assert bool(active)
    assert isinstance(image, torch.Tensor)
    assert image.shape == (12, 16, 3)
    assert torch.count_nonzero(image[..., 0]) > 0


def test_tacmap_strip_jointly_interpolates_base_and_marker_depth():
    depth = torch.zeros((2, 2), dtype=torch.float32)
    marker_depth = torch.tensor([0.001], dtype=torch.float32)
    source_indices = torch.tensor(
        [
            [[0, 1, 4], [1, 3, 4]],
            [[0, 2, 4], [2, 3, 4]],
        ],
        dtype=torch.long,
    )
    source_weights = torch.full((2, 2, 3), 1.0 / 3.0, dtype=torch.float32)

    image = tacmap_strip(
        depth,
        tacmap_raw=depth,
        scale=1,
        gamma=1.0,
        view="raw",
        display_max_m=0.001,
        target_size=(2, 2),
        resize_mode="bilinear",
        joint_marker_depth_values=marker_depth,
        joint_marker_depth_valid=torch.tensor([True]),
        joint_source_indices=source_indices,
        joint_source_weights=source_weights,
        joint_pixel_valid=torch.ones((2, 2), dtype=torch.bool),
    )

    assert image.shape == (2, 2, 3)
    assert torch.all((image[..., 0] >= 84) & (image[..., 0] <= 85))
    assert torch.equal(image[..., 0], image[..., 1])
    assert torch.equal(image[..., 1], image[..., 2])


def _identity_quat(batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
    quat = torch.zeros((batch, 4), dtype=torch.float32, device=device)
    quat[:, 0] = 1.0
    return quat


def test_tacsl_force_field_matches_integrated_normal_and_friction_law():
    depth = torch.full((2, 2, 2), 0.001, dtype=torch.float32)
    points_l = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
    current_pos = torch.tensor([[0.0, 0.001, 0.0], [0.0, 0.001, 0.0]], dtype=torch.float32)
    previous_pos = torch.zeros_like(current_pos)
    quat = _identity_quat(2)

    normal, shear = _tacsl_force_field_from_relative_pose_batched(
        depth,
        points_l,
        current_pos,
        quat,
        previous_pos,
        quat,
        torch.tensor([True, False]),
        ray_direction_l=torch.tensor([1.0, 0.0, 0.0]),
        row_axis_l=torch.tensor([0.0, 0.0, 1.0]),
        col_axis_l=torch.tensor([0.0, 1.0, 0.0]),
        dt=0.01,
        normal_contact_stiffness=1.0,
        tangential_stiffness=0.1,
        friction_coefficient=2.0,
    )

    assert torch.allclose(normal, torch.full_like(normal, 0.001))
    assert torch.allclose(shear[0, ..., 0], torch.zeros((2, 2)))
    assert torch.allclose(shear[0, ..., 1], torch.full((2, 2), 0.002), atol=1.0e-7)
    assert torch.count_nonzero(shear[1]) == 0


def test_tacsl_force_field_ignores_nonpenetrating_points():
    depth = torch.tensor([[[0.001, 0.0]]], dtype=torch.float32)
    points_l = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    quat = _identity_quat(1)

    normal, shear = _tacsl_force_field_from_relative_pose_batched(
        depth,
        points_l,
        torch.tensor([[0.0, 0.001, 0.0]]),
        quat,
        torch.zeros((1, 3)),
        quat,
        torch.tensor([True]),
        ray_direction_l=torch.tensor([1.0, 0.0, 0.0]),
        row_axis_l=torch.tensor([0.0, 0.0, 1.0]),
        col_axis_l=torch.tensor([0.0, 1.0, 0.0]),
        dt=0.01,
        normal_contact_stiffness=1.0,
        tangential_stiffness=0.1,
        friction_coefficient=2.0,
    )

    assert normal[0, 0, 0] > 0.0
    assert shear[0, 0, 0, 1] > 0.0
    assert normal[0, 0, 1] == 0.0
    assert torch.count_nonzero(shear[0, 0, 1]) == 0


def test_tacsl_active_pooling_matches_integrated_block_mean():
    normal = torch.zeros((1, 4, 4), dtype=torch.float32)
    shear = torch.zeros((1, 4, 4, 2), dtype=torch.float32)
    normal[0, 0, 0] = 1.0
    normal[0, 1, 1] = 3.0
    shear[0, 0, 0] = torch.tensor([2.0, 4.0])
    shear[0, 1, 1] = torch.tensor([6.0, 8.0])
    normal[0, 3, 3] = 5.0

    pooled = _pool_tacsl_force_field_batched(normal, shear, out_rows=2, out_cols=2)

    assert pooled.shape == (1, 2, 2, 3)
    assert torch.allclose(pooled[0, 0, 0], torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(pooled[0, 1, 1], torch.tensor([5.0, 0.0, 0.0]))
    assert torch.count_nonzero(pooled[0, 0, 1]) == 0
    assert torch.count_nonzero(pooled[0, 1, 0]) == 0


def test_tacsl_baseline_defaults_match_shared_320x240_and_16x8_layout():
    params = inspect.signature(observations_module.tacsl_baseline_rl_obs).parameters

    assert params["ray_rows"].default == 320
    assert params["ray_cols"].default == 240
    assert params["output_rows"].default == 16
    assert params["output_cols"].default == 8
    assert 5 * params["output_rows"].default * params["output_cols"].default * 3 == 1920


def test_fots_baseline_defaults_use_32x24_depth_and_16x8_markers():
    params = inspect.signature(observations_module.fots_baseline_rl_obs).parameters

    assert params["ray_rows"].default == 32
    assert params["ray_cols"].default == 24
    assert params["render_rows"].default == 320
    assert params["render_cols"].default == 240
    assert params["marker_rows"].default == 16
    assert params["marker_cols"].default == 8
    assert 5 * params["marker_rows"].default * params["marker_cols"].default * 2 == 1280


def test_fots_signed_twist_extracts_rotation_around_ray_axis():
    half_angle = torch.tensor(torch.pi / 8.0)
    quat = torch.tensor([[[torch.cos(half_angle), torch.sin(half_angle), 0.0, 0.0]]])

    theta = _fots_signed_twist_angle_batched(quat, torch.tensor([[[1.0, 0.0, 0.0]]]))

    assert torch.allclose(theta, torch.tensor([[torch.pi / 4.0]]), atol=1.0e-6)


def test_hydroshear_baseline_wrapper_selects_original_gpu_path(monkeypatch):
    captured = {}

    def _fake_hydroshear_rl_obs(env, **kwargs):
        captured.update(kwargs)
        return torch.zeros((env.num_envs, 5 * 16 * 8 * 3), dtype=torch.float32)

    monkeypatch.setattr(observations_module, "hydroshear_rl_obs", _fake_hydroshear_rl_obs)
    result = observations_module.hydroshear_baseline_rl_obs(
        _FakeEnv(),
        tacmap_rows=32,
        tacmap_cols=24,
        marker_rows=16,
        marker_cols=8,
        object_poisson_initial_count=5000,
    )

    assert result.shape == (2, 1920)
    assert captured["algorithm"] == "original"
    assert captured["adapter_namespace"] == "hydroshear_baseline"
    assert captured["output_attr_name"] == "_rl_hydroshear_baseline_output"
    assert captured["tacmap_rows"] == 32
    assert captured["tacmap_cols"] == 24
    assert captured["render_rows"] == observations_module.RL_HYDROSHEAR_RENDER_ROWS
    assert captured["render_cols"] == observations_module.RL_HYDROSHEAR_RENDER_COLS
    assert captured["marker_margin_x"] == observations_module.RL_HYDROSHEAR_MARKER_MARGIN_X
    assert captured["marker_margin_y"] == observations_module.RL_HYDROSHEAR_MARKER_MARGIN_Y
    assert captured["object_sample_roi_count"] == 5000
    assert captured["use_object_surface_samples"] is True


def test_ours_pressure_visual_cache_is_opt_in_and_matches_policy_slice(monkeypatch):
    pressure = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    tacmap = torch.tensor([[5.0], [6.0]], dtype=torch.float32)
    hydroshear = torch.tensor([[7.0, 8.0], [9.0, 10.0]], dtype=torch.float32)

    monkeypatch.setattr(observations_module, "warpsdf_pressure_obs", lambda _env, **_kwargs: pressure)
    monkeypatch.setattr(observations_module, "tacmap_rl_obs", lambda _env, **_kwargs: tacmap)
    monkeypatch.setattr(observations_module, "hydroshear_rl_obs", lambda _env, **_kwargs: hydroshear)

    cached_env = SimpleNamespace(num_envs=2)
    cached_obs = observations_module.ours_rl_obs(
        cached_env,
        pressure_sensor_names=(),
        tacmap_sensor_names=(),
        tacmap_surface_sensor_names=(),
        cache_pressure_output=True,
    )

    assert torch.equal(cached_obs, torch.cat((pressure, tacmap, hydroshear), dim=1))
    assert torch.equal(cached_env._rl_pressure_observation, pressure)
    assert cached_env._rl_pressure_observation.data_ptr() == pressure.data_ptr()
    assert not cached_env._rl_pressure_observation.requires_grad

    normal_env = SimpleNamespace(num_envs=2)
    normal_obs = observations_module.ours_rl_obs(
        normal_env,
        pressure_sensor_names=(),
        tacmap_sensor_names=(),
        tacmap_surface_sensor_names=(),
    )

    assert torch.equal(normal_obs, cached_obs)
    assert not hasattr(normal_env, "_rl_pressure_observation")


def test_ours_pressure_term_does_not_compute_tacmap_or_hydroshear(monkeypatch):
    pressure = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    monkeypatch.setattr(observations_module, "warpsdf_pressure_obs", lambda _env, **_kwargs: pressure)
    monkeypatch.setattr(
        observations_module,
        "tacmap_rl_obs",
        lambda *_args, **_kwargs: pytest.fail("pressure term must not compute TacMap"),
    )
    monkeypatch.setattr(
        observations_module,
        "hydroshear_rl_obs",
        lambda *_args, **_kwargs: pytest.fail("pressure term must not compute HydroShear"),
    )

    env = SimpleNamespace(num_envs=1, common_step_counter=7)
    result = observations_module.ours_rl_pressure_obs(
        env,
        {"pressure_sensor_names": (), "cache_pressure_output": True},
    )

    assert torch.equal(result, pressure)
    assert torch.equal(env._rl_pressure_observation, pressure)


def test_ours_tacmap_term_does_not_compute_pressure_or_hydroshear(monkeypatch):
    tacmap = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32)
    captured = {}

    def _tacmap(_env, **kwargs):
        captured.update(kwargs)
        return tacmap

    monkeypatch.setattr(observations_module, "tacmap_rl_obs", _tacmap)
    monkeypatch.setattr(
        observations_module,
        "warpsdf_pressure_obs",
        lambda *_args, **_kwargs: pytest.fail("TacMap term must not compute pressure"),
    )
    monkeypatch.setattr(
        observations_module,
        "hydroshear_rl_obs",
        lambda *_args, **_kwargs: pytest.fail("TacMap term must not compute HydroShear"),
    )

    result = observations_module.ours_rl_tacmap_policy_obs(
        SimpleNamespace(num_envs=1, common_step_counter=7),
        {"tacmap_sensor_names": (), "tacmap_surface_sensor_names": ()},
    )

    assert torch.equal(result, tacmap)
    assert captured["cache_aux_fields"] is True


def test_ours_tacmap_and_hydroshear_share_current_step_tacmap_cache(monkeypatch):
    tacmap = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32)
    hydroshear = torch.tensor([[6.0]], dtype=torch.float32)
    calls = {"pressure": 0, "tacmap": 0, "hydroshear": 0}

    def _pressure(_env, **_kwargs):
        calls["pressure"] += 1
        return torch.zeros((1, 1), dtype=torch.float32)

    def _tacmap(env, **_kwargs):
        calls["tacmap"] += 1
        env._rl_tacmap_penetration_m = tacmap.reshape(1, 1, 1, 3)
        env._rl_tacmap_surface_points_w = torch.zeros((1, 1, 1, 3, 3), dtype=torch.float32)
        env._rl_tacmap_surface_normals_w = torch.zeros((1, 1, 1, 3, 3), dtype=torch.float32)
        env._rl_tacmap_ray_directions_w = torch.zeros((1, 1, 3), dtype=torch.float32)
        env._rl_tacmap_surface_valid = torch.zeros((1, 1, 1, 3), dtype=torch.bool)
        env._rl_tacmap_object_points_w = torch.zeros((1, 1, 1, 3, 3), dtype=torch.float32)
        env._rl_tacmap_object_valid = torch.zeros((1, 1, 1, 3), dtype=torch.bool)
        env._rl_tacmap_surface_raw_m = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        env._rl_tacmap_cache_step = env.common_step_counter
        return tacmap

    def _hydroshear(_env, **_kwargs):
        calls["hydroshear"] += 1
        return hydroshear

    monkeypatch.setattr(observations_module, "warpsdf_pressure_obs", _pressure)
    monkeypatch.setattr(observations_module, "tacmap_rl_obs", _tacmap)
    monkeypatch.setattr(observations_module, "hydroshear_rl_obs", _hydroshear)

    env = SimpleNamespace(num_envs=1, common_step_counter=7)
    component_cfg = {
        "tacmap_sensor_names": ("object",),
        "tacmap_surface_sensor_names": ("surface",),
        "tacmap_rows": 1,
        "tacmap_cols": 3,
    }
    tacmap_obs = observations_module.ours_rl_tacmap_policy_obs(env, component_cfg)
    hydroshear_obs = observations_module.ours_rl_hydroshear_obs(env, component_cfg)

    assert torch.equal(tacmap_obs, tacmap)
    assert torch.equal(hydroshear_obs, hydroshear)
    assert calls == {"pressure": 0, "tacmap": 1, "hydroshear": 1}

    observations_module.invalidate_ours_tactile_cache_on_reset(env, torch.tensor([0]))
    observations_module.ours_rl_hydroshear_obs(env, component_cfg)
    assert calls == {"pressure": 0, "tacmap": 2, "hydroshear": 2}


def test_ours_tacmap_and_hydroshear_share_adaptive_local_depth(monkeypatch):
    calls = {"tacmap": 0, "local": 0, "hydroshear": 0}
    local_depth = torch.arange(20, dtype=torch.float32).reshape(1, 5, 2, 2)
    local_roi = torch.zeros((1, 5, 4), dtype=torch.float32)
    local_active = torch.ones((1, 5), dtype=torch.bool)

    def _tacmap(env, **_kwargs):
        calls["tacmap"] += 1
        image_shape = (1, 5, 1, 1)
        point_shape = (*image_shape, 3)
        env._rl_tacmap_penetration_m = torch.zeros(image_shape, dtype=torch.float32)
        env._rl_tacmap_surface_points_w = torch.zeros(point_shape, dtype=torch.float32)
        env._rl_tacmap_surface_normals_w = torch.zeros(point_shape, dtype=torch.float32)
        env._rl_tacmap_ray_directions_w = torch.zeros((1, 5, 3), dtype=torch.float32)
        env._rl_tacmap_surface_valid = torch.zeros(image_shape, dtype=torch.bool)
        env._rl_tacmap_object_points_w = torch.zeros(point_shape, dtype=torch.float32)
        env._rl_tacmap_object_valid = torch.zeros(image_shape, dtype=torch.bool)
        env._rl_tacmap_surface_raw_m = torch.zeros(image_shape, dtype=torch.float32)
        env._rl_tacmap_cache_step = env.common_step_counter
        return env._rl_tacmap_penetration_m.reshape(1, -1)

    def _local(env, **_kwargs):
        calls["local"] += 1
        env._rl_local_tacmap_depth_m = local_depth
        env._rl_local_tacmap_roi_norm = local_roi
        env._rl_local_tacmap_active = local_active
        env._rl_local_tacmap_pixel_indices = torch.zeros_like(local_depth, dtype=torch.long)
        env._rl_local_tacmap_cache_step = env.common_step_counter
        return local_depth, local_roi, local_active

    def _hydroshear(_env, **kwargs):
        calls["hydroshear"] += 1
        assert kwargs["dilation_source_depth_m"] is local_depth
        assert kwargs["dilation_source_roi_norm"] is local_roi
        assert kwargs["dilation_source_active"] is local_active
        return torch.ones((1, 1), dtype=torch.float32)

    monkeypatch.setattr(observations_module, "tacmap_rl_obs", _tacmap)
    monkeypatch.setattr(observations_module, "adaptive_local_tacmap_rl_obs", _local)
    monkeypatch.setattr(observations_module, "hydroshear_rl_obs", _hydroshear)

    env = SimpleNamespace(num_envs=1, common_step_counter=7)
    component_cfg = {
        "tacmap_sensor_names": tuple(f"object_{index}" for index in range(5)),
        "tacmap_surface_sensor_names": tuple(f"surface_{index}" for index in range(5)),
        "tacmap_rows": 1,
        "tacmap_cols": 1,
        "local_tacmap_enabled": True,
        "local_tacmap_rows": 2,
        "local_tacmap_cols": 2,
        "hydroshear_marker_layout_path": "marker_positions.npz",
        "hydroshear_marker_finger_link_names": tuple(f"finger_{index}" for index in range(5)),
    }
    tacmap_policy = observations_module.ours_rl_tacmap_policy_obs(env, component_cfg)
    hydroshear = observations_module.ours_rl_hydroshear_obs(env, component_cfg)

    assert tacmap_policy.shape == (1, 5 * 2 * 2 + 5 * 4 + 5)
    assert torch.equal(hydroshear, torch.ones((1, 1), dtype=torch.float32))
    assert calls == {"tacmap": 1, "local": 1, "hydroshear": 1}

    observations_module.invalidate_ours_tactile_cache_on_reset(env, torch.tensor([0]))
    observations_module.ours_rl_hydroshear_obs(env, component_cfg)
    assert calls == {"tacmap": 2, "local": 2, "hydroshear": 2}


def test_ours_reset_cache_invalidation_advances_epoch_and_clears_step_keys():
    env = SimpleNamespace(
        _rl_ours_tactile_reset_epoch=4,
        _rl_tacmap_cache_step=9,
        _rl_local_tacmap_cache_step=9,
        _rl_ours_tacmap_cache_epoch=4,
        _rl_ours_local_tacmap_cache_epoch=4,
        _rl_ours_dense_tacmap_cache_step=9,
        _rl_ours_dense_tacmap_cache_epoch=4,
        _rl_ours_taxim_rgb_cache_step=9,
        _rl_ours_taxim_rgb_cache_epoch=4,
        _rl_ours_tacmap_resnet_cache_step=9,
        _rl_ours_tacmap_resnet_cache_epoch=4,
        _rl_ours_taxim_resnet_cache_step=9,
        _rl_ours_taxim_resnet_cache_epoch=4,
    )

    observations_module.invalidate_ours_tactile_cache_on_reset(env, torch.tensor([1]))

    assert env._rl_ours_tactile_reset_epoch == 5
    assert env._rl_tacmap_cache_step is None
    assert env._rl_local_tacmap_cache_step is None
    assert env._rl_ours_tacmap_cache_epoch is None
    assert env._rl_ours_local_tacmap_cache_epoch is None
    assert env._rl_ours_dense_tacmap_cache_step is None
    assert env._rl_ours_dense_tacmap_cache_epoch is None
    assert env._rl_ours_taxim_rgb_cache_step is None
    assert env._rl_ours_taxim_rgb_cache_epoch is None
    assert env._rl_ours_tacmap_resnet_cache_step is None
    assert env._rl_ours_tacmap_resnet_cache_epoch is None
    assert env._rl_ours_taxim_resnet_cache_step is None
    assert env._rl_ours_taxim_resnet_cache_epoch is None


def test_dense_local_tacmap_chunk_places_lattice_at_calibrated_pixels():
    local_depth = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    pixel_indices = torch.tensor(
        [
            [[6, 8], [16, 18]],
            [[6, 16], [8, 18]],
        ],
        dtype=torch.long,
    )
    dense = observations_module._dense_local_tacmap_chunk_for_taxim(
        local_depth,
        pixel_indices,
        torch.tensor([True, False]),
        torch.ones((2, 4, 5), dtype=torch.bool),
        torch.tensor([1, 0], dtype=torch.long),
    )

    assert dense.shape == (2, 4, 5)
    assert dense[0, 1, 1] == 1.0
    assert dense[0, 1, 3] == 2.0
    assert dense[0, 3, 1] == 3.0
    assert dense[0, 3, 3] == 4.0
    assert torch.count_nonzero(dense[0, :1]) == 0
    assert torch.count_nonzero(dense[1]) == 0


def test_ours_tacmap_policy_can_return_full_depth_image(monkeypatch):
    local_depth = torch.ones((1, 5, 2, 2), dtype=torch.float32)
    local_active = torch.ones((1, 5), dtype=torch.bool)
    dense_depth = torch.arange(5 * 4 * 5, dtype=torch.float32).reshape(1, 5, 4, 5)

    monkeypatch.setattr(
        observations_module,
        "_ours_tacmap_policy_components",
        lambda _env, _cfg: (torch.zeros((1, 45)), local_depth, torch.zeros((1, 5, 4)), local_active),
    )
    monkeypatch.setattr(
        observations_module,
        "_ours_dense_tacmap_depth_from_local",
        lambda _env, depth, active, _cfg: dense_depth
        if depth is local_depth and active is local_active
        else None,
    )

    output = observations_module.ours_rl_tacmap_policy_obs(
        SimpleNamespace(num_envs=1),
        {"tacmap_policy_full_depth_image": True},
    )

    assert output.shape == (1, 5 * 4 * 5)
    torch.testing.assert_close(output, dense_depth.reshape(1, -1))


def test_full_depth_cache_is_shared_in_step_and_rebuilt_after_reset(monkeypatch):
    calls = {"dense": 0}

    def _dense_chunk(local_depth, _indices, _active, reference_valid, _axes):
        calls["dense"] += 1
        return torch.ones(
            (local_depth.shape[0], *reference_valid.shape[-2:]),
            device=local_depth.device,
            dtype=torch.float32,
        )

    monkeypatch.setattr(observations_module, "_dense_local_tacmap_chunk_for_taxim", _dense_chunk)
    local_depth = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    local_active = torch.ones((1, 2), dtype=torch.bool)
    env = SimpleNamespace(
        common_step_counter=3,
        _rl_local_tacmap_pixel_indices=torch.zeros_like(local_depth, dtype=torch.long),
        _rl_local_tacmap_reference_state={
            "reference_rows": 4,
            "reference_cols": 5,
            "reference_valid": torch.ones((2, 4, 5), dtype=torch.bool),
            "storage_row_axis_camera": (1, 1),
        },
    )
    cfg = {"taxim_rgb_render_rows": 4, "taxim_rgb_render_cols": 5, "taxim_rgb_render_chunk_size": 8}

    first = observations_module._ours_dense_tacmap_depth_from_local(env, local_depth, local_active, cfg)
    second = observations_module._ours_dense_tacmap_depth_from_local(env, local_depth, local_active, cfg)

    assert first is second
    assert calls["dense"] == 1

    observations_module.invalidate_ours_tactile_cache_on_reset(env, torch.tensor([0]))
    third = observations_module._ours_dense_tacmap_depth_from_local(env, local_depth, local_active, cfg)

    assert third is not first
    assert calls["dense"] == 2


def test_ours_taxim_rgb_is_independent_and_cached_per_step_and_reset(monkeypatch):
    local_depth = torch.zeros((1, 5, 2, 2), dtype=torch.float32)
    local_active = torch.ones((1, 5), dtype=torch.bool)
    calls = {"components": 0, "rgb": 0}

    def _components(_env, _cfg):
        calls["components"] += 1
        return torch.zeros((1, 1)), local_depth, torch.zeros((1, 5, 4)), local_active

    def _rgb(_env, depth, active, _cfg):
        calls["rgb"] += 1
        assert depth is local_depth
        assert active is local_active
        return torch.full((1, 5 * 3 * 2 * 3), 0.25, dtype=torch.float32)

    monkeypatch.setattr(observations_module, "_ours_tacmap_policy_components", _components)
    monkeypatch.setattr(observations_module, "_ours_taxim_rgb_policy_from_local", _rgb)
    env = SimpleNamespace(num_envs=1, common_step_counter=11)
    cfg = {"taxim_rgb_render_rows": 2, "taxim_rgb_render_cols": 3}

    first = observations_module.ours_rl_taxim_rgb_obs(env, cfg)
    second = observations_module.ours_rl_taxim_rgb_obs(env, cfg)
    assert torch.equal(first, second)
    assert calls == {"components": 1, "rgb": 1}

    observations_module.invalidate_ours_tactile_cache_on_reset(env, torch.tensor([0]))
    third = observations_module.ours_rl_taxim_rgb_obs(env, cfg)
    assert torch.equal(first, third)
    assert calls == {"components": 2, "rgb": 2}


def test_frozen_depth_resnet_observation_is_128d_and_cached(monkeypatch):
    calls = {"depth": 0, "encoder": 0}
    dense_depth = torch.ones((1, 5 * 4 * 5), dtype=torch.float32)

    def _depth(_env, cfg):
        calls["depth"] += 1
        assert cfg["tacmap_policy_full_depth_image"] is True
        return dense_depth

    def _encode(_env, _cfg, images, *, modality):
        calls["encoder"] += 1
        assert modality == "depth"
        assert images.shape == (1, 5, 1, 4, 5)
        assert torch.all((images >= 0.0) & (images <= 1.0))
        return torch.full((1, 128), 0.25, dtype=torch.float32)

    monkeypatch.setattr(observations_module, "ours_rl_tacmap_policy_obs", _depth)
    monkeypatch.setattr(observations_module, "_ours_frozen_resnet_embedding", _encode)
    env = SimpleNamespace(num_envs=1, device="cpu", common_step_counter=3)
    cfg = {
        "taxim_rgb_render_rows": 4,
        "taxim_rgb_render_cols": 5,
        "tactile_resnet_output_dim": 128,
        "tactile_resnet_depth_max_m": 2.0,
    }

    first = observations_module.ours_rl_tacmap_resnet_obs(env, cfg)
    second = observations_module.ours_rl_tacmap_resnet_obs(env, cfg)
    assert first is second
    assert first.shape == (1, 128)
    assert calls == {"depth": 1, "encoder": 1}

    observations_module.invalidate_ours_tactile_cache_on_reset(env, torch.tensor([0]))
    third = observations_module.ours_rl_tacmap_resnet_obs(env, cfg)
    assert third.shape == (1, 128)
    assert calls == {"depth": 2, "encoder": 2}


def test_frozen_rgb_resnet_observation_is_128d_and_cached(monkeypatch):
    calls = {"rgb": 0, "encoder": 0}
    full_rgb = torch.full((1, 5 * 3 * 4 * 5), 0.5, dtype=torch.float32)

    def _rgb(_env, _cfg):
        calls["rgb"] += 1
        return full_rgb

    def _encode(_env, _cfg, images, *, modality):
        calls["encoder"] += 1
        assert modality == "rgb"
        assert images.shape == (1, 5, 3, 4, 5)
        return torch.full((1, 128), 0.5, dtype=torch.float32)

    monkeypatch.setattr(observations_module, "ours_rl_taxim_rgb_obs", _rgb)
    monkeypatch.setattr(observations_module, "_ours_frozen_resnet_embedding", _encode)
    env = SimpleNamespace(num_envs=1, device="cpu", common_step_counter=7)
    cfg = {
        "taxim_rgb_render_rows": 4,
        "taxim_rgb_render_cols": 5,
        "tactile_resnet_output_dim": 128,
    }

    first = observations_module.ours_rl_taxim_resnet_obs(env, cfg)
    second = observations_module.ours_rl_taxim_resnet_obs(env, cfg)
    assert first is second
    assert first.shape == (1, 128)
    assert calls == {"rgb": 1, "encoder": 1}

    observations_module.invalidate_ours_tactile_cache_on_reset(env, torch.tensor([0]))
    third = observations_module.ours_rl_taxim_resnet_obs(env, cfg)
    assert third.shape == (1, 128)
    assert calls == {"rgb": 2, "encoder": 2}


def test_ours_taxim_rgb_renders_only_active_finger_images(monkeypatch):
    class _FakeAdapter:
        def __init__(self):
            self.batch_sizes = []

        def step(self, depth):
            self.batch_sizes.append(int(depth.shape[0]))
            rgb = torch.full((*depth.shape, 3), 100, device=depth.device, dtype=torch.uint8)
            return SimpleNamespace(tactile_rgb=rgb)

    adapter = _FakeAdapter()
    monkeypatch.setattr(observations_module, "_ours_taxim_rgb_adapter", lambda _env, _cfg: adapter)
    monkeypatch.setattr(
        observations_module,
        "_ours_taxim_rgb_background",
        lambda *_args, **_kwargs: (
            torch.full((4, 5, 3), 50.0),
            torch.full((4, 5, 3), 20.0),
        ),
    )
    pixel_indices = torch.tensor(
        [[[[6, 8], [16, 18]], [[6, 8], [16, 18]]]],
        dtype=torch.long,
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        _rl_local_tacmap_pixel_indices=pixel_indices,
        _rl_local_tacmap_reference_state={
            "reference_rows": 4,
            "reference_cols": 5,
            "reference_valid": torch.ones((2, 4, 5), dtype=torch.bool),
            "storage_row_axis_camera": (1, 1),
        },
    )
    output = observations_module._ours_taxim_rgb_policy_from_local(
        env,
        torch.ones((1, 2, 2, 2), dtype=torch.float32),
        torch.tensor([[False, True]]),
        {
            "taxim_rgb_render_rows": 4,
            "taxim_rgb_render_cols": 5,
        },
    ).reshape(1, 2, 3, 4, 5)

    assert adapter.batch_sizes == [1]
    torch.testing.assert_close(output[0, 0], torch.full((3, 4, 5), 50.0 / 255.0))
    torch.testing.assert_close(output[0, 1], torch.full((3, 4, 5), 130.0 / 255.0))


def test_ours_hydroshear_only_materializes_tacmap_dependency(monkeypatch):
    calls = {"pressure": 0, "tacmap": 0, "hydroshear": 0}

    def _pressure(_env, **_kwargs):
        calls["pressure"] += 1
        return torch.zeros((1, 1), dtype=torch.float32)

    def _tacmap(_env, **_kwargs):
        calls["tacmap"] += 1
        return torch.zeros((1, 3), dtype=torch.float32)

    def _hydroshear(_env, **kwargs):
        calls["hydroshear"] += 1
        assert kwargs["dilation_source_depth_m"] is None
        return torch.tensor([[6.0]], dtype=torch.float32)

    monkeypatch.setattr(observations_module, "warpsdf_pressure_obs", _pressure)
    monkeypatch.setattr(observations_module, "tacmap_rl_obs", _tacmap)
    monkeypatch.setattr(observations_module, "hydroshear_rl_obs", _hydroshear)

    result = observations_module.ours_rl_hydroshear_obs(
        SimpleNamespace(num_envs=1, common_step_counter=7),
        {"tacmap_sensor_names": (), "tacmap_surface_sensor_names": ()},
    )

    assert torch.equal(result, torch.tensor([[6.0]], dtype=torch.float32))
    assert calls == {"pressure": 0, "tacmap": 1, "hydroshear": 1}


def test_ours_uses_local_25x40_depth_as_hydroshear_dilation_source(monkeypatch):
    pressure = torch.ones((2, 2), dtype=torch.float32)
    coarse_tacmap = torch.ones((2, 3), dtype=torch.float32)
    local_depth = torch.zeros((2, 5, 25, 40), dtype=torch.float32)
    local_depth[:, :, 12, 20] = 0.001
    local_roi = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32).expand(2, 5, 4).clone()
    local_active = torch.ones((2, 5), dtype=torch.bool)
    captured = {}

    monkeypatch.setattr(observations_module, "warpsdf_pressure_obs", lambda _env, **_kwargs: pressure)
    monkeypatch.setattr(observations_module, "tacmap_rl_obs", lambda _env, **_kwargs: coarse_tacmap)
    monkeypatch.setattr(
        observations_module,
        "adaptive_local_tacmap_rl_obs",
        lambda _env, **_kwargs: (local_depth, local_roi, local_active),
    )

    def _fake_hydroshear(_env, **kwargs):
        captured.update(kwargs)
        return torch.zeros((2, 4), dtype=torch.float32)

    monkeypatch.setattr(observations_module, "hydroshear_rl_obs", _fake_hydroshear)

    output = observations_module.ours_rl_obs(
        SimpleNamespace(num_envs=2),
        pressure_sensor_names=(),
        tacmap_sensor_names=(),
        tacmap_surface_sensor_names=(),
        local_tacmap_enabled=True,
        hydroshear_marker_layout_path="marker_positions.npz",
    )

    expected_tacmap_dim = 5 * 25 * 40 + 5 * 4 + 5
    assert output.shape == (2, 2 + expected_tacmap_dim + 4)
    assert captured["dilation_source_depth_m"] is local_depth
    assert captured["dilation_source_roi_norm"] is local_roi
    assert captured["dilation_source_active"] is local_active


def test_hydroshear_marker_depth_visual_cache_defaults_off_for_training():
    hydroshear_signature = inspect.signature(observations_module.hydroshear_rl_obs)
    ours_signature = inspect.signature(observations_module.ours_rl_obs)

    assert hydroshear_signature.parameters["cache_marker_depth_output"].default is False
    assert ours_signature.parameters["hydroshear_cache_marker_depth_output"].default is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tacsl_force_and_pooling_stay_on_gpu():
    device = torch.device("cuda")
    depth = torch.full((3, 8, 6), 0.001, dtype=torch.float32, device=device)
    points_l = torch.zeros((3, 8, 6, 3), dtype=torch.float32, device=device)
    quat = _identity_quat(3, device=device)
    normal, shear = _tacsl_force_field_from_relative_pose_batched(
        depth,
        points_l,
        torch.tensor([[0.0, 0.001, 0.0]] * 3, device=device),
        quat,
        torch.zeros((3, 3), device=device),
        quat,
        torch.ones(3, dtype=torch.bool, device=device),
        ray_direction_l=torch.tensor([1.0, 0.0, 0.0], device=device),
        row_axis_l=torch.tensor([0.0, 0.0, 1.0], device=device),
        col_axis_l=torch.tensor([0.0, 1.0, 0.0], device=device),
        dt=0.01,
        normal_contact_stiffness=1.0,
        tangential_stiffness=0.1,
        friction_coefficient=2.0,
    )
    pooled = _pool_tacsl_force_field_batched(normal, shear, out_rows=4, out_cols=3)

    assert normal.is_cuda
    assert shear.is_cuda
    assert pooled.is_cuda
    assert torch.isfinite(pooled).all()


class _FakePressureScene:
    def __init__(self, sensors):
        self.sensors = sensors
        self._object = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.zeros((2, 3), dtype=torch.float32),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float32),
            )
        )

    def __getitem__(self, name):
        assert name == "object"
        return self._object


class _FakePressureEnv(_FakeEnv):
    def __init__(self, sensors):
        self.scene = _FakePressureScene(sensors)


def _fake_pressure_sensor(values, *, points_l=None, normals_l=None):
    taxel_maps = ()
    if points_l is not None and normals_l is not None:
        taxel_maps = (
            SimpleNamespace(
                points_l=torch.as_tensor(points_l, dtype=torch.float32),
                normals_l=torch.as_tensor(normals_l, dtype=torch.float32),
            ),
        )
    return SimpleNamespace(
        data=SimpleNamespace(pressure_force_map=torch.tensor(values, dtype=torch.float32)),
        pressure_taxel_maps=taxel_maps,
    )


def test_warpsdf_pressure_obs_flattens_variable_taxel_counts_in_sensor_order():
    env = _FakePressureEnv(
        {
            "first": _fake_pressure_sensor([[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]]),
            "second": _fake_pressure_sensor(
                [[[[7.0, 8.0], [9.0, 10.0]]], [[[11.0, 12.0], [13.0, 14.0]]]]
            ),
        }
    )

    obs = warpsdf_pressure_obs(
        env,
        pressure_sensor_names=("first", "second"),
        pressure_taxel_counts=(3, 4),
    )

    assert obs.shape == (2, 7)
    assert torch.equal(obs[0], torch.tensor([1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0]))
    assert torch.equal(obs[1], torch.tensor([4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0]))


def test_warpsdf_pressure_obs_zero_fills_only_missing_sensor_count():
    env = _FakePressureEnv({"present": _fake_pressure_sensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])})

    obs = warpsdf_pressure_obs(
        env,
        pressure_sensor_names=("missing", "present"),
        pressure_taxel_counts=(3, 2),
    )

    assert obs.shape == (2, 5)
    assert torch.equal(obs[:, :3], torch.zeros((2, 3)))
    assert torch.equal(obs[:, 3:], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_pressure_rl_diffusion_matches_display_geometry_and_conserves_total():
    points_l = torch.tensor(
        [
            [0.0, 0.000, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.020, 0.0],
        ],
        dtype=torch.float32,
    )
    normals_l = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float32)
    kernel = build_pressure_rl_diffusion_kernel(
        points_l,
        normals_l,
        sigma_m=0.002,
        radius_sigma=3.0,
        normal_power=1.0,
    )
    raw = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    diffused = diffuse_pressure_rl_values(raw, kernel, blend=1.0)

    assert torch.allclose(torch.sum(kernel, dim=0), torch.ones(3), atol=1.0e-6)
    assert torch.allclose(torch.sum(diffused, dim=1), torch.sum(raw, dim=1), atol=1.0e-6)
    assert diffused[0, 0] > diffused[0, 1] > 0.0
    assert diffused[0, 2] == 0.0


def test_warpsdf_pressure_obs_applies_and_caches_rl_diffusion_before_flatten():
    points_l = [
        [0.0, 0.000, 0.0],
        [0.0, 0.002, 0.0],
        [0.0, 0.020, 0.0],
    ]
    normals_l = [[1.0, 0.0, 0.0]] * 3
    env = _FakePressureEnv(
        {
            "pressure": _fake_pressure_sensor(
                [[[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0]]]],
                points_l=points_l,
                normals_l=normals_l,
            )
        }
    )
    kwargs = {
        "pressure_sensor_names": ("pressure",),
        "pressure_taxel_counts": (3,),
        "pressure_diffusion_enabled": True,
        "pressure_diffusion_sigma_m": 0.002,
        "pressure_diffusion_blend": 1.0,
        "pressure_diffusion_radius_sigma": 3.0,
        "pressure_diffusion_normal_power": 1.0,
    }

    obs = warpsdf_pressure_obs(env, **kwargs)
    cached_kernel = next(iter(env._brainco_rl_pressure_diffusion_kernel_cache.values()))
    obs_again = warpsdf_pressure_obs(env, **kwargs)

    assert obs.shape == (2, 3)
    assert torch.allclose(torch.sum(obs, dim=1), torch.ones(2), atol=1.0e-6)
    assert next(iter(env._brainco_rl_pressure_diffusion_kernel_cache.values())) is cached_kernel
    assert torch.equal(obs_again, obs)


def test_warpsdf_pressure_obs_diffuses_legacy_shared_grid_layout():
    points_l = [
        [0.0, 0.000, 0.000],
        [0.0, 0.002, 0.000],
        [0.0, 0.000, 0.002],
        [0.0, 0.002, 0.002],
    ]
    env = _FakePressureEnv(
        {
            "pressure": _fake_pressure_sensor(
                [[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 1.0], [0.0, 0.0]]]],
                points_l=points_l,
                normals_l=[[1.0, 0.0, 0.0]] * 4,
            )
        }
    )

    obs = warpsdf_pressure_obs(
        env,
        pressure_sensor_names=("pressure",),
        pressure_rows=2,
        pressure_cols=2,
        pressure_diffusion_enabled=True,
        pressure_diffusion_sigma_m=0.002,
        pressure_diffusion_blend=1.0,
    )

    assert obs.shape == (2, 4)
    assert torch.allclose(torch.sum(obs, dim=1), torch.ones(2), atol=1.0e-6)
