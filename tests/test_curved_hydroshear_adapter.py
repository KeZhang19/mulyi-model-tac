from __future__ import annotations

import numpy as np
import torch

from integrate.curved_hydroshear_adapter import RevoCurvedHydroShearAdapter, RevoCurvedHydroShearCfg


def _direction_inputs(batch_count: int = 2):
    surface_points = torch.zeros((batch_count, 2, 2, 3), dtype=torch.float32)
    object_points = torch.zeros_like(surface_points)
    surface_valid = torch.ones((batch_count, 2, 2), dtype=torch.bool)
    object_valid = torch.zeros_like(surface_valid)
    depth = torch.zeros((batch_count, 2, 2), dtype=torch.float32)
    return surface_points, object_points, surface_valid, object_valid, depth


def test_configured_ray_direction_is_used_before_contact():
    adapter = RevoCurvedHydroShearAdapter(RevoCurvedHydroShearCfg(device="cpu"))
    inputs = _direction_inputs()
    configured = torch.tensor([[0.0, 0.0, 2.0], [0.0, -3.0, 0.0]], dtype=torch.float32)

    direction = adapter._batched_ray_direction(*inputs, ray_directions_w=configured)

    expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(direction, expected)


def test_configured_ray_direction_does_not_switch_when_contact_appears():
    adapter = RevoCurvedHydroShearAdapter(RevoCurvedHydroShearCfg(device="cpu"))
    surface_points, object_points, surface_valid, object_valid, depth = _direction_inputs(batch_count=1)
    surface_points[..., 0] = 1.0
    object_valid[:] = True
    depth[:] = 0.001
    configured = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    direction = adapter._batched_ray_direction(
        surface_points,
        object_points,
        surface_valid,
        object_valid,
        depth,
        ray_directions_w=configured,
    )

    assert torch.allclose(direction, configured)


def test_missing_configured_ray_direction_keeps_legacy_fallback():
    adapter = RevoCurvedHydroShearAdapter(RevoCurvedHydroShearCfg(device="cpu"))
    inputs = _direction_inputs(batch_count=1)

    direction = adapter._batched_ray_direction(*inputs)

    assert torch.allclose(direction, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))


def test_render_displacement_output_is_stateless_and_preserves_training_values():
    cfg = RevoCurvedHydroShearCfg(
        device="cpu",
        width=24,
        height=32,
        marker_rows=2,
        marker_cols=2,
        marker_margin_x=4.0,
        marker_margin_y=4.0,
    )
    adapter = RevoCurvedHydroShearAdapter(cfg)
    surface_points = torch.zeros((1, 32, 24, 3), dtype=torch.float32)
    surface_points[0, :, :, 1] = torch.arange(32, dtype=torch.float32)[:, None] * 0.001
    surface_points[0, :, :, 2] = torch.arange(24, dtype=torch.float32)[None, :] * 0.001
    surface_valid = torch.ones((1, 32, 24), dtype=torch.bool)
    depth = torch.full((1, 32, 24), 0.001, dtype=torch.float32)
    displacement = torch.tensor(
        [[[0.0, 0.0001, 0.0], [0.0, 0.0, 0.0002], [0.0, -0.0001, 0.0], [0.0, 0.0, -0.0002]]],
        dtype=torch.float32,
    )

    output = adapter.render_displacement_output(displacement, depth, surface_points, surface_valid)

    assert torch.equal(output.displacement_m, displacement)
    assert tuple(output.marker_flow.shape) == (1, 2, 4, 2)
    assert output.marker_images.shape == (1, 32, 24, 3)
    assert output.marker_images[0, 0, 0].tolist() == [0, 0, 0]
    assert output.marker_images[..., 1].max() == 255
    assert output.marker_images[..., 0].max() == 0
    assert output.marker_images[..., 2].max() == 0
    for marker_x, marker_y in adapter._marker_uv:
        pixel = output.marker_images[0, int(round(float(marker_y))), int(round(float(marker_x)))]
        assert pixel.tolist() == [0, 255, 0]
    assert adapter._batch_state_valid is None
    assert adapter._batch_prev_sdf is None
    assert adapter._prev_sdf == []


def test_render_displacement_output_can_skip_images_and_filter_cpu_debug_geometry():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            width=24,
            height=32,
            marker_rows=2,
            marker_cols=2,
            marker_margin_x=4.0,
            marker_margin_y=4.0,
        )
    )
    surface_points = torch.zeros((2, 32, 24, 3), dtype=torch.float32)
    surface_points[:, :, :, 1] = torch.arange(32, dtype=torch.float32)[None, :, None] * 0.001
    surface_points[:, :, :, 2] = torch.arange(24, dtype=torch.float32)[None, None, :] * 0.001
    surface_valid = torch.ones((2, 32, 24), dtype=torch.bool)
    depth = torch.full((2, 32, 24), 0.001, dtype=torch.float32)
    displacement = torch.zeros((2, 4, 3), dtype=torch.float32)

    output = adapter.render_displacement_output(
        displacement,
        depth,
        surface_points,
        surface_valid,
        render_marker_images=False,
        debug_batch_index=1,
    )

    assert torch.equal(output.displacement_m, displacement)
    assert tuple(output.marker_flow.shape) == (2, 2, 4, 2)
    assert output.marker_images.shape == (0, 32, 24, 3)
    assert output.debug_marker_points_w[0].shape == (0, 3)
    assert output.debug_marker_normals_w[0].shape == (0, 3)
    assert output.debug_marker_points_w[1].shape == (4, 3)
    assert output.debug_marker_normals_w[1].shape == (4, 3)


def test_render_displacement_output_draws_static_marker_points():
    cfg = RevoCurvedHydroShearCfg(
        device="cpu",
        width=240,
        height=240,
        marker_rows=16,
        marker_cols=8,
    )
    adapter = RevoCurvedHydroShearAdapter(cfg)
    surface_points = torch.zeros((1, 32, 24, 3), dtype=torch.float32)
    surface_points[0, :, :, 1] = torch.arange(32, dtype=torch.float32)[:, None] * 0.001
    surface_points[0, :, :, 2] = torch.arange(24, dtype=torch.float32)[None, :] * 0.001
    surface_valid = torch.ones((1, 32, 24), dtype=torch.bool)
    depth = torch.zeros((1, 32, 24), dtype=torch.float32)
    displacement = torch.zeros((1, 128, 3), dtype=torch.float32)

    output = adapter.render_displacement_output(displacement, depth, surface_points, surface_valid)

    image = output.marker_images[0]
    assert adapter._marker_uv.shape == (128, 2)
    for marker_x, marker_y in adapter._marker_uv:
        pixel = image[int(round(float(marker_y))), int(round(float(marker_x)))]
        assert pixel.tolist() == [0, 255, 0]


def test_vector_field_render_padding_keeps_marker_arrows_inside_frame():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(device="cpu", width=320, height=240)
    )
    marker_flow = np.asarray([[[160.0, 232.0]], [[160.0, 254.0]]], dtype=np.float32)

    image = adapter._render_vector_field_image(
        marker_flow,
        color=(0, 255, 0),
        draw_points=True,
        padding_px=32,
    )

    green = image[..., 1] > 0
    assert image.shape == (304, 384, 3)
    assert image[286, 192].tolist() == [0, 255, 0]
    assert not green[[0, -1], :].any()
    assert not green[:, [0, -1]].any()


def test_calibrated_marker_uv_and_world_geometry_replace_regular_grid():
    marker_uv = np.asarray(((20.0, 30.0), (280.0, 210.0)), dtype=np.float32)
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            width=320,
            height=240,
            marker_rows=1,
            marker_cols=2,
            marker_uv=marker_uv,
        )
    )
    surface_points = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    surface_points[0, :, :, 1] = torch.arange(4, dtype=torch.float32)[:, None] * 0.001
    surface_points[0, :, :, 2] = torch.arange(4, dtype=torch.float32)[None, :] * 0.001
    surface_valid = torch.ones((1, 4, 4), dtype=torch.bool)
    surface_valid[0, 0, 0] = False
    depth = torch.zeros((1, 4, 4), dtype=torch.float32)
    displacement = torch.zeros((1, 2, 3), dtype=torch.float32)
    marker_points = torch.tensor([[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]], dtype=torch.float32)
    marker_normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    marker_valid = torch.tensor([[True, False]])

    output = adapter.render_displacement_output(
        displacement,
        depth,
        surface_points,
        surface_valid,
        marker_points_w=marker_points,
        marker_normals_w=marker_normals,
        marker_valid=marker_valid,
    )

    assert np.array_equal(adapter._marker_uv, marker_uv)
    assert np.allclose(output.debug_marker_points_w[0], marker_points[0, :1].numpy())
    assert np.allclose(output.debug_marker_normals_w[0], marker_normals[0, :1].numpy())
    assert output.marker_images[0, 30, 20].tolist() == [0, 255, 0]
    assert output.marker_images[0, 210, 280].tolist() == [0, 0, 0]


def test_independent_marker_depth_replaces_grid_sample_without_changing_grid():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            width=4,
            height=4,
            marker_rows=1,
            marker_cols=2,
            marker_uv=np.asarray(((0.0, 0.0), (3.0, 3.0)), dtype=np.float32),
        )
    )
    depth = torch.zeros((1, 4, 4), dtype=torch.float32)
    depth[0, 0, 0] = 0.009
    depth[0, 3, 3] = 0.008
    surface_points = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    surface_valid = torch.ones((1, 4, 4), dtype=torch.bool)
    marker_points = torch.tensor([[[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]]], dtype=torch.float32)
    marker_normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)

    samples = adapter._batched_marker_samples(
        depth,
        surface_points,
        surface_valid,
        None,
        marker_points,
        marker_normals,
        torch.tensor([[True, True]]),
        torch.tensor([[0.001, 0.002]]),
        torch.tensor([[True, False]]),
    )

    assert torch.equal(samples[2], torch.tensor([[0.001, 0.002]]))
    assert torch.equal(samples[3], torch.tensor([[True, False]]))
    assert depth[0, 0, 0] == 0.009
    assert depth[0, 3, 3] == 0.008


def test_adaptive_25x40_depth_spreads_to_fixed_100_marker_targets():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            width=320,
            height=240,
            marker_rows=10,
            marker_cols=10,
        )
    )
    rows, cols = 32, 24
    yy, xx = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    surface_points = torch.stack(
        (
            torch.zeros((rows, cols), dtype=torch.float32),
            yy.to(dtype=torch.float32) * 0.001,
            xx.to(dtype=torch.float32) * 0.001,
        ),
        dim=-1,
    ).unsqueeze(0)
    surface_valid = torch.ones((1, rows, cols), dtype=torch.bool)
    coarse_depth = torch.zeros((1, rows, cols), dtype=torch.float32)
    local_depth = torch.zeros((1, 25, 40), dtype=torch.float32)
    local_depth[0, 12, 20] = 0.001

    displacement = adapter.step_displacement_only(
        coarse_depth,
        surface_points,
        surface_valid,
        torch.zeros_like(surface_points),
        torch.zeros_like(surface_valid),
        dilation_source_depth_m=local_depth,
        dilation_source_roi_norm=torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
        dilation_source_active=torch.tensor([True]),
    )

    assert displacement.shape == (1, 100, 3)
    assert torch.isfinite(displacement).all()
    assert torch.count_nonzero(displacement) > 0


def test_non_affine_ray_grid_coordinates_match_visualized_row_boundaries():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            object_sample_roi_boundary_padding_cells=0.0,
        )
    )
    starts = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    starts[0, 0, :, 1] = torch.tensor([-2.0, 0.0, 2.0])
    starts[0, 1, :, 1] = torch.tensor([-1.0, 0.0, 1.0])
    starts[0, 1, :, 2] = 1.0
    points = torch.tensor(
        [
            [
                [0.5, 1.5, 0.0],
                [0.5, 1.5, 1.0],
                [0.5, 1.5, 0.5],
                [0.5, 1.6, 0.5],
            ]
        ],
        dtype=torch.float32,
    )

    x, y, point_depth, inside = adapter._ray_grid_coordinates_batched(
        points,
        starts,
        torch.ones((1, 2, 3), dtype=torch.bool),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    )

    torch.testing.assert_close(x, torch.tensor([[1.75, 2.5, 2.0, 2.0666666]]))
    torch.testing.assert_close(y, torch.tensor([[0.0, 1.0, 0.5, 0.5]]))
    torch.testing.assert_close(point_depth, torch.full((1, 4), 0.5))
    assert inside.tolist() == [[True, False, True, False]]


def test_all_object_sdf_uses_visualized_non_affine_roi_boundary():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            object_sample_roi_boundary_padding_cells=0.0,
        )
    )
    starts = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    starts[0, 0, :, 1] = torch.tensor([-2.0, 0.0, 2.0])
    starts[0, 1, :, 1] = torch.tensor([-1.0, 0.0, 1.0])
    starts[0, 1, :, 2] = 1.0
    ray_dir = torch.tensor([[1.0, 0.0, 0.0]])
    surface_raw = torch.ones((1, 2, 3), dtype=torch.float32)
    surface_points = starts + surface_raw[..., None] * ray_dir[:, None, None, :]
    candidates = torch.tensor(
        [[[0.5, 1.5, 0.0], [0.5, 1.5, 1.0]]],
        dtype=torch.float32,
    )

    sdf, normals, slot_active = adapter._estimate_all_object_sample_sdf_batched(
        candidates,
        torch.full((1, 2, 3), 0.001, dtype=torch.float32),
        surface_points,
        torch.ones((1, 2, 3), dtype=torch.bool),
        surface_points - 0.001 * ray_dir[:, None, None, :],
        torch.ones((1, 2, 3), dtype=torch.bool),
        surface_raw,
        ray_dir,
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    )

    torch.testing.assert_close(sdf, torch.tensor([[-0.5, 1.0]]))
    torch.testing.assert_close(normals, ray_dir[:, None, :].expand(1, 2, 3))
    assert slot_active.tolist() == [True]


def test_default_ray_grid_full_cell_padding_expands_all_four_boundaries():
    adapter = RevoCurvedHydroShearAdapter(RevoCurvedHydroShearCfg(device="cpu"))
    starts = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    starts[0, 0, :, 1] = torch.tensor([-2.0, 0.0, 2.0])
    starts[0, 1, :, 1] = torch.tensor([-1.0, 0.0, 1.0])
    starts[0, 1, :, 2] = 1.0
    points = torch.tensor(
        [
            [
                [0.5, 4.0, -1.0],
                [0.5, 2.0, 2.0],
                [0.5, 4.01, -1.0],
                [0.5, 2.0, 2.01],
            ]
        ],
        dtype=torch.float32,
    )

    x, y, _point_depth, inside = adapter._ray_grid_coordinates_batched(
        points,
        starts,
        torch.ones((1, 2, 3), dtype=torch.bool),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    )

    torch.testing.assert_close(x[:, :2], torch.tensor([[3.0, 3.0]]))
    torch.testing.assert_close(y[:, :2], torch.tensor([[-1.0, 2.0]]))
    assert inside.tolist() == [[True, True, False, False]]


def test_padded_object_sdf_reuses_nearest_boundary_depth():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            object_sample_roi_boundary_padding_cells=0.5,
        )
    )
    starts = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    starts[0, :, :, 1] = torch.tensor([[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]])
    starts[0, 1, :, 2] = 1.0
    ray_dir = torch.tensor([[1.0, 0.0, 0.0]])
    surface_raw = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    surface_points = starts + surface_raw[..., None] * ray_dir[:, None, None, :]
    candidates = torch.tensor(
        [
            [
                [0.5, -3.0, -0.5],
                [5.5, 3.0, 1.5],
                [0.5, -3.01, -0.5],
            ]
        ],
        dtype=torch.float32,
    )

    sdf, _normals, slot_active = adapter._estimate_all_object_sample_sdf_batched(
        candidates,
        torch.full((1, 2, 3), 0.001, dtype=torch.float32),
        surface_points,
        torch.ones((1, 2, 3), dtype=torch.bool),
        surface_points - 0.001 * ray_dir[:, None, None, :],
        torch.ones((1, 2, 3), dtype=torch.bool),
        surface_raw,
        ray_dir,
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    )

    torch.testing.assert_close(sdf, torch.tensor([[-0.5, -0.5, 1.0]]))
    assert slot_active.tolist() == [True]


def test_adaptive_dilation_falls_back_per_slot_when_dense_depth_has_no_contact():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            width=320,
            height=240,
            marker_rows=1,
            marker_cols=2,
        )
    )
    marker_uv = adapter._scaled_marker_uv_t(24, 32)
    marker_fallback = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        dtype=torch.float32,
    )
    local_depth = torch.zeros((2, 25, 40), dtype=torch.float32)
    local_depth[0, 12, 20] = 0.001
    selected = adapter._select_dilation_source_batched(
        marker_fallback,
        marker_uv,
        local_depth,
        torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.2, 0.8, 0.2, 0.8]], dtype=torch.float32),
        torch.tensor([True, True]),
        torch.tensor([[0.0, 0.001, 0.0], [0.0, 0.001, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.001], [0.0, 0.0, 0.001]], dtype=torch.float32),
        grid_width=24,
        grid_height=32,
    )

    assert torch.isfinite(selected).all()
    assert not torch.equal(selected[0], marker_fallback[0])
    assert torch.equal(selected[1], marker_fallback[1])


def test_adaptive_dense_source_area_weight_keeps_marker_scale_comparable():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            width=320,
            height=240,
            marker_rows=10,
            marker_cols=10,
        )
    )
    marker_uv = adapter._scaled_marker_uv_t(24, 32)
    row_metric = torch.tensor([[0.0, 0.001, 0.0]], dtype=torch.float32)
    col_metric = torch.tensor([[0.0, 0.0, 0.001]], dtype=torch.float32)
    marker_depth = torch.full((1, 100), 0.001, dtype=torch.float32)
    marker_dilation = adapter._compute_dilation_batched(
        marker_uv,
        marker_depth,
        torch.ones_like(marker_depth, dtype=torch.bool),
        row_metric,
        col_metric,
    )
    dense_dilation, use_dense = adapter._compute_adaptive_roi_dilation_batched(
        marker_uv,
        torch.full((1, 25, 40), 0.001, dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
        torch.tensor([True]),
        row_metric,
        col_metric,
        grid_width=24,
        grid_height=32,
    )

    scale_ratio = torch.linalg.norm(dense_dilation, dim=-1).amax() / torch.linalg.norm(
        marker_dilation,
        dim=-1,
    ).amax()
    assert use_dense.tolist() == [True]
    assert 0.5 < float(scale_ratio) < 2.0


def test_direct_320x240_render_preserves_legacy_tacmap_marker_samples():
    legacy = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            width=240,
            height=240,
            marker_rows=16,
            marker_cols=8,
            marker_margin_x=15.0,
            marker_margin_y=26.0,
        )
    )
    direct = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            width=240,
            height=320,
            marker_rows=16,
            marker_cols=8,
            marker_margin_x=15.0,
            marker_margin_y=26.0 * 320.0 / 240.0,
        )
    )

    assert torch.allclose(legacy._scaled_marker_uv_t(24, 32), direct._scaled_marker_uv_t(24, 32))


def test_marker_render_projection_uses_one_frame_per_sensor_slot():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(device="cpu", marker_rows=2, marker_cols=2, width=24, height=32)
    )
    surface_points = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    surface_points[:, :, :, 1] = torch.arange(4, dtype=torch.float32)[None, :, None] * 0.001
    surface_points[:, :, :, 2] = torch.arange(4, dtype=torch.float32)[None, None, :] * 0.001
    surface_points[1, :, :, 0] = torch.arange(4, dtype=torch.float32)[None, None, :] ** 2 * 0.0001
    surface_valid = torch.ones((2, 4, 4), dtype=torch.bool)
    marker_row_axes = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.8, 0.2], [0.0, 0.6, 0.4], [0.0, 0.4, 0.6]],
            [[0.2, 1.0, 0.0], [0.4, 0.8, 0.2], [0.6, 0.6, 0.4], [0.8, 0.4, 0.6]],
        ],
        dtype=torch.float32,
    )
    marker_col_axes = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.2, 0.0, 0.8], [0.4, 0.0, 0.6], [0.6, 0.0, 0.4]],
            [[0.0, 0.0, 1.0], [0.2, 0.2, 0.8], [0.4, 0.4, 0.6], [0.6, 0.6, 0.4]],
        ],
        dtype=torch.float32,
    )

    row_axes, col_axes = adapter._batched_uniform_marker_projection_axes(
        surface_points,
        surface_valid,
        marker_row_axes,
        marker_col_axes,
    )

    assert torch.allclose(row_axes, row_axes[:, :1].expand_as(row_axes))
    assert torch.allclose(col_axes, col_axes[:, :1].expand_as(col_axes))
    assert torch.allclose(torch.linalg.norm(row_axes, dim=-1), torch.ones((2, 4)), atol=1.0e-6)
    assert torch.allclose(torch.linalg.norm(col_axes, dim=-1), torch.ones((2, 4)), atol=1.0e-6)
    assert torch.allclose(torch.sum(row_axes * col_axes, dim=-1), torch.zeros((2, 4)), atol=1.0e-6)


def test_batched_shear_ignores_noncontact_nan_uv_slots():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(device="cpu", marker_rows=1, marker_cols=2, width=24, height=32)
    )
    marker_uv = torch.tensor([[6.0, 16.0], [18.0, 16.0]], dtype=torch.float32)
    marker_normals = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    marker_valid = torch.ones((1, 2), dtype=torch.bool)
    obj = torch.zeros((1, 2, 3), dtype=torch.float32)
    sdf = torch.tensor([[-0.001, 1.0]], dtype=torch.float32)
    fbar = torch.tensor([[[1.0, 0.0002, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    normals = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    sample_uv = torch.tensor([[[12.0, 16.0], [float("nan"), float("nan")]]], dtype=torch.float32)
    ray_row_vec = torch.tensor([[0.0, 0.001, 0.0]], dtype=torch.float32)
    ray_col_vec = torch.tensor([[0.0, 0.0, 0.001]], dtype=torch.float32)

    with_invalid_slot = adapter._compute_shear_from_samples_batched(
        marker_uv,
        marker_normals,
        marker_valid,
        obj,
        sdf,
        fbar,
        normals,
        sample_uv,
        ray_row_vec,
        ray_col_vec,
    )
    contact_only = adapter._compute_shear_from_samples_batched(
        marker_uv,
        marker_normals,
        marker_valid,
        obj[:, :1],
        sdf[:, :1],
        fbar[:, :1],
        normals[:, :1],
        sample_uv[:, :1],
        ray_row_vec,
        ray_col_vec,
    )

    assert torch.isfinite(with_invalid_slot).all()
    assert torch.count_nonzero(with_invalid_slot) > 0
    assert torch.allclose(with_invalid_slot, contact_only)


def _curved_shear_reference_cfg(depth: torch.Tensor) -> dict:
    row, col = torch.meshgrid(
        torch.arange(2, dtype=torch.float32),
        torch.arange(2, dtype=torch.float32),
        indexing="ij",
    )
    starts = torch.stack((torch.zeros_like(row), col, torch.zeros_like(row)), dim=-1).unsqueeze(0)
    directions = torch.zeros_like(starts)
    directions[..., 2] = 1.0
    coarse_xy = torch.stack((col, row), dim=-1).unsqueeze(0)
    return {
        "shear_reference_starts_l": starts,
        "shear_reference_directions_l": directions,
        "shear_reference_depth_m": depth.reshape(1, 2, 2),
        "shear_reference_valid": torch.ones((1, 2, 2), dtype=torch.bool),
        "shear_coarse_xy_camera_m": coarse_xy,
        "shear_reference_bounds_camera_m": torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
        "shear_reference_row_axis_camera": (1,),
    }


def test_affected_uv_queries_reference_curve_and_applies_link_pose():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = _curved_shear_reference_cfg(torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    for name in (
        "shear_reference_starts_l",
        "shear_reference_directions_l",
        "shear_reference_valid",
        "shear_coarse_xy_camera_m",
        "shear_reference_bounds_camera_m",
    ):
        reference[name] = reference[name].repeat(2, *([1] * (reference[name].ndim - 1)))
    reference["shear_reference_depth_m"] = torch.tensor(
        [[[1.0, 2.0], [1.0, 2.0]], [[3.0, 4.0], [3.0, 4.0]]]
    )
    reference["shear_reference_row_axis_camera"] = (1, 1)
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device=device,
            **reference,
        )
    )
    pose = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], device=device)

    points_w, valid = adapter._curved_surface_points_from_uv_batched(
        torch.tensor([[[0.25, 0.5]]], device=device),
        pose,
        batch_offset=1,
    )

    assert valid is not None and valid.item()
    assert points_w is not None
    torch.testing.assert_close(points_w[0, 0], torch.tensor([1.0, 2.25, 6.25], device=device))


def test_curved_shear_distance_matches_plane_and_attenuates_across_bulge():
    common = {
        "device": "cpu",
        "marker_rows": 1,
        "marker_cols": 1,
        "lambda_shear": 1.0,
        "shear_scale": 1.0,
    }
    flat_adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            **common,
            **_curved_shear_reference_cfg(torch.ones((2, 2))),
        )
    )
    curved_adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            **common,
            **_curved_shear_reference_cfg(torch.tensor([[1.0, 2.0], [1.0, 2.0]])),
        )
    )
    marker_uv = torch.tensor([[1.0, 0.0]])
    marker_normals = torch.tensor([[[1.0, 0.0, 0.0]]])
    marker_valid = torch.tensor([[True]])
    obj = torch.zeros((1, 1, 3))
    sdf = torch.tensor([[-1.0]])
    fbar = torch.tensor([[[1.0, 0.2, 0.0]]])
    normals = torch.tensor([[[1.0, 0.0, 0.0]]])
    sample_uv = torch.tensor([[[0.0, 0.0]]])
    ray_row_vec = torch.tensor([[0.0, 0.0, 1.0]])
    ray_col_vec = torch.tensor([[0.0, 1.0, 0.0]])
    pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    args = (
        marker_uv,
        marker_normals,
        marker_valid,
        obj,
        sdf,
        fbar,
        normals,
        sample_uv,
        ray_row_vec,
        ray_col_vec,
    )

    planar_result = flat_adapter._compute_shear_from_samples_batched(*args)
    flat_curve_result = flat_adapter._compute_shear_from_samples_batched(
        *args,
        marker_points=torch.tensor([[[0.0, 1.0, 1.0]]]),
        surface_frame_pose_wxyz=pose,
    )
    curved_result = curved_adapter._compute_shear_from_samples_batched(
        *args,
        marker_points=torch.tensor([[[0.0, 1.0, 2.0]]]),
        surface_frame_pose_wxyz=pose,
    )

    torch.testing.assert_close(flat_curve_result, planar_result)
    torch.testing.assert_close(planar_result[0, 0, 1], -0.2 * torch.exp(torch.tensor(-0.64)))
    torch.testing.assert_close(curved_result[0, 0, 1], -0.2 * torch.exp(torch.tensor(-1.28)))
    assert torch.linalg.norm(curved_result) < torch.linalg.norm(flat_curve_result)


def test_large_curve_lookup_chunks_preserve_small_propagation_chunk_results():
    reference = _curved_shear_reference_cfg(torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    common = {
        "device": "cpu",
        "marker_rows": 1,
        "marker_cols": 2,
        "batch_slot_chunk_size": 2,
        "lambda_shear": 1.0,
        "shear_scale": 1.0,
        **reference,
    }
    small_lookup = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(curved_surface_lookup_chunk_size=2, **common)
    )
    large_lookup = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(curved_surface_lookup_chunk_size=5, **common)
    )
    batch_count = 7
    marker_uv = torch.tensor([[0.2, 0.2], [0.8, 0.8]])
    marker_normals = torch.tensor([[[1.0, 0.0, 0.0]]]).expand(batch_count, 2, 3).clone()
    marker_valid = torch.ones((batch_count, 2), dtype=torch.bool)
    marker_points = torch.tensor([[[0.0, 0.2, 1.2], [0.0, 0.8, 1.8]]]).expand(
        batch_count,
        -1,
        -1,
    ).clone()
    obj = torch.zeros((batch_count, 3, 3))
    sdf = -torch.ones((batch_count, 3))
    fbar = torch.tensor(
        [[[1.0, 0.02, 0.01], [1.0, 0.03, 0.02], [1.0, 0.04, 0.01]]]
    ).expand(batch_count, -1, -1).clone()
    normals = torch.tensor([[[1.0, 0.0, 0.0]]]).expand(batch_count, 3, 3).clone()
    sample_uv = torch.tensor([[[0.1, 0.1], [0.3, 0.4], [0.6, 0.7]]]).expand(
        batch_count,
        -1,
        -1,
    ).clone()
    ray_row_vec = torch.tensor([[0.0, 0.0, 1.0]]).expand(batch_count, -1).clone()
    ray_col_vec = torch.tensor([[0.0, 1.0, 0.0]]).expand(batch_count, -1).clone()
    pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).expand(
        batch_count,
        -1,
    ).clone()
    args = (
        marker_uv,
        marker_normals,
        marker_valid,
        obj,
        sdf,
        fbar,
        normals,
        sample_uv,
        ray_row_vec,
        ray_col_vec,
    )
    lookup_calls: list[tuple[int, int]] = []
    original_lookup = large_lookup._curved_surface_points_from_uv_batched

    def _tracked_lookup(affected_uv, surface_pose, *, batch_offset=0):
        lookup_calls.append((int(affected_uv.shape[0]), int(batch_offset)))
        return original_lookup(affected_uv, surface_pose, batch_offset=batch_offset)

    large_lookup._curved_surface_points_from_uv_batched = _tracked_lookup

    expected = small_lookup._compute_shear_from_samples_batched(
        *args,
        marker_points=marker_points,
        surface_frame_pose_wxyz=pose,
    )
    actual = large_lookup._compute_shear_from_samples_batched(
        *args,
        marker_points=marker_points,
        surface_frame_pose_wxyz=pose,
    )

    assert lookup_calls == [(5, 0), (2, 5)]
    torch.testing.assert_close(actual, expected)


def test_roi_selector_new_contact_replaces_weak_precontact_without_moving_other_slots():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            object_sample_roi_count=3,
            object_sample_roi_replace_margin_m=0.0002,
        )
    )
    adapter._batch_prev_sample_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    roi_candidate = torch.ones((1, 5), dtype=torch.bool)
    roi_priority = torch.tensor([[0.0010, 0.0015, 0.0020, 0.0005, 0.0025]], dtype=torch.float32)
    sample_contact = torch.tensor([[True, False, False, False, True]], dtype=torch.bool)

    selected, valid, history = adapter._select_object_sample_roi_batched(
        roi_candidate,
        roi_priority,
        sample_contact,
    )

    assert selected.tolist() == [[0, 4, 2]]
    assert valid.tolist() == [[True, True, True]]
    assert history.tolist() == [[True, False, True]]
    assert torch.unique(selected[0]).numel() == 3


def test_roi_selector_hysteresis_and_contact_tier_prevent_slot_churn():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            object_sample_roi_count=2,
            object_sample_roi_replace_margin_m=0.0002,
        )
    )
    adapter._batch_prev_sample_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    roi_candidate = torch.tensor(
        [[True, True, True], [True, True, True]],
        dtype=torch.bool,
    )
    roi_priority = torch.tensor(
        [[0.0010, 0.0015, 0.0011], [0.0010, 0.0015, 0.0100]],
        dtype=torch.float32,
    )
    sample_contact = torch.tensor(
        [[False, False, False], [True, True, False]],
        dtype=torch.bool,
    )

    selected, valid, history = adapter._select_object_sample_roi_batched(
        roi_candidate,
        roi_priority,
        sample_contact,
    )

    assert selected.tolist() == [[0, 1], [0, 1]]
    assert valid.all()
    assert history.all()


def test_roi_selector_fills_empty_slot_before_replacing_retained_history():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            object_sample_roi_count=3,
            object_sample_roi_replace_margin_m=0.0002,
        )
    )
    adapter._batch_prev_sample_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    roi_candidate = torch.tensor([[False, True, True, True, True]], dtype=torch.bool)
    roi_priority = torch.tensor(
        [[float("-inf"), 0.0010, 0.0020, 0.0015, 0.0030]],
        dtype=torch.float32,
    )
    sample_contact = torch.tensor([[False, True, False, False, True]], dtype=torch.bool)

    selected, valid, history = adapter._select_object_sample_roi_batched(
        roi_candidate,
        roi_priority,
        sample_contact,
    )

    assert selected.tolist() == [[4, 1, 2]]
    assert valid.tolist() == [[True, True, True]]
    assert history.tolist() == [[False, True, True]]


def test_roi_selector_stronger_contact_can_replace_weaker_contact_after_margin():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            object_sample_roi_count=2,
            object_sample_roi_replace_margin_m=0.0002,
        )
    )
    adapter._batch_prev_sample_ids = torch.tensor([[0, 1]], dtype=torch.long)
    roi_candidate = torch.ones((1, 4), dtype=torch.bool)
    roi_priority = torch.tensor([[0.0010, 0.0030, 0.0011, 0.0013]], dtype=torch.float32)
    sample_contact = torch.ones((1, 4), dtype=torch.bool)

    selected, valid, history = adapter._select_object_sample_roi_batched(
        roi_candidate,
        roi_priority,
        sample_contact,
    )

    assert selected.tolist() == [[3, 1]]
    assert valid.tolist() == [[True, True]]
    assert history.tolist() == [[False, True]]


def test_original_dilation_batched_matches_integrated_single_sensor_formula():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(device="cpu", marker_rows=2, marker_cols=2)
    )
    marker_points = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.002, 0.0], [0.0, 0.0, 0.003], [0.0, 0.002, 0.003]],
            [[0.001, 0.0, 0.0], [0.001, 0.003, 0.0], [0.001, 0.0, 0.002], [0.001, 0.003, 0.002]],
        ],
        dtype=torch.float32,
    )
    marker_depth = torch.tensor(
        [[0.0010, 0.0005, 0.0, 0.0002], [0.0, 0.0007, 0.0003, 0.0004]],
        dtype=torch.float32,
    )
    marker_contact = marker_depth > 0.0001

    batched = adapter._compute_original_dilation_batched(marker_points, marker_depth, marker_contact)
    expected = torch.stack(
        [
            adapter._compute_original_dilation_t(marker_points[i], marker_depth[i], marker_contact[i])
            for i in range(marker_points.shape[0])
        ],
        dim=0,
    )

    assert torch.isfinite(batched).all()
    assert torch.allclose(batched, expected, atol=1.0e-8, rtol=1.0e-6)


def test_original_shear_batched_matches_integrated_single_sensor_formula():
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(device="cpu", marker_rows=1, marker_cols=2, object_chunk_size=2)
    )
    marker_points = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.003, 0.0]],
            [[0.001, 0.0, 0.0], [0.001, 0.0, 0.003]],
        ],
        dtype=torch.float32,
    )
    marker_valid = torch.tensor([[True, True], [True, False]])
    obj = torch.tensor(
        [
            [[-0.0005, 0.0010, 0.0], [-0.0002, 0.0025, 0.0], [0.002, 0.0, 0.0]],
            [[0.0005, 0.0, 0.0010], [0.0008, 0.0, 0.0025], [0.003, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    sdf = torch.tensor([[-0.0005, -0.0002, 1.0], [-0.0004, -0.0001, 1.0]], dtype=torch.float32)
    fbar = torch.tensor(
        [
            [[0.0004, 0.0001, 0.0], [0.0002, -0.0001, 0.0], [0.0, 0.0, 0.0]],
            [[0.0003, 0.0, 0.0001], [0.0001, 0.0, -0.0001], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    normals = torch.tensor(
        [
            [[1.0, 0.0, 0.0]] * 3,
            [[1.0, 0.0, 0.0]] * 3,
        ],
        dtype=torch.float32,
    )

    batched = adapter._compute_original_shear_batched(marker_points, marker_valid, obj, sdf, fbar, normals)
    expected = torch.stack(
        [
            adapter._original_hydroshear_shear_t(
                marker_points[i], marker_valid[i], obj[i], sdf[i], fbar[i], normals[i]
            )[0]
            for i in range(marker_points.shape[0])
        ],
        dim=0,
    )

    assert torch.isfinite(batched).all()
    assert torch.count_nonzero(batched) > 0
    assert torch.allclose(batched, expected, atol=1.0e-8, rtol=1.0e-6)


def test_original_public_step_uses_all_samples_and_keeps_fixed_history_slots():
    sample_points_l = torch.tensor(
        [[0.0, 0.001, 0.001], [0.0, 0.002, 0.002], [0.0, 0.003, 0.003]],
        dtype=torch.float32,
    )
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(
            device="cpu",
            width=24,
            height=32,
            marker_rows=2,
            marker_cols=2,
            object_sample_points_l=sample_points_l,
        )
    )
    rows, cols = 32, 24
    yy, xx = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    surface_points = torch.stack(
        (
            torch.full((rows, cols), 0.01),
            yy.to(torch.float32) * 0.001,
            xx.to(torch.float32) * 0.001,
        ),
        dim=-1,
    ).unsqueeze(0)
    object_points = surface_points.clone()
    object_points[..., 0] -= 0.0005
    valid = torch.ones((1, rows, cols), dtype=torch.bool)
    depth = torch.full((1, rows, cols), 0.0005, dtype=torch.float32)
    surface_raw = torch.full((1, rows, cols), 0.01, dtype=torch.float32)
    pose = torch.tensor([[0.0095, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    first = adapter.step_original_displacement_only(
        depth,
        surface_points,
        valid,
        object_points,
        valid,
        object_pose_wxyz=pose,
        surface_raw_m=surface_raw,
        ray_directions_w=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    pose[:, 1] += 0.0002
    second = adapter.step_original_displacement_only(
        depth,
        surface_points,
        valid,
        object_points,
        valid,
        object_pose_wxyz=pose,
        surface_raw_m=surface_raw,
        ray_directions_w=torch.tensor([[1.0, 0.0, 0.0]]),
    )

    assert first.shape == (1, 4, 3)
    assert second.shape == (1, 4, 3)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert adapter._batch_prev_sdf is not None
    assert adapter._batch_prev_sdf.shape == (1, 3)
    assert adapter._batch_state_valid is not None
    assert adapter._batch_state_valid.all()


@torch.no_grad()
def test_original_hydroshear_helpers_keep_cuda_outputs_on_gpu():
    if not torch.cuda.is_available():
        return
    adapter = RevoCurvedHydroShearAdapter(
        RevoCurvedHydroShearCfg(device="cuda", marker_rows=1, marker_cols=2, object_chunk_size=2)
    )
    marker_points = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.003, 0.0]]], device="cuda", dtype=torch.float32
    )
    marker_depth = torch.tensor([[0.001, 0.0005]], device="cuda", dtype=torch.float32)
    marker_valid = torch.ones((1, 2), device="cuda", dtype=torch.bool)
    obj = torch.tensor(
        [[[-0.0005, 0.001, 0.0], [-0.0002, 0.002, 0.0]]], device="cuda", dtype=torch.float32
    )
    sdf = torch.tensor([[-0.0005, -0.0002]], device="cuda", dtype=torch.float32)
    fbar = torch.tensor(
        [[[0.0004, 0.0001, 0.0], [0.0002, -0.0001, 0.0]]], device="cuda", dtype=torch.float32
    )
    normals = torch.tensor([[[1.0, 0.0, 0.0]] * 2], device="cuda", dtype=torch.float32)

    mdilate = adapter._compute_original_dilation_batched(marker_points, marker_depth, marker_valid)
    mshear = adapter._compute_original_shear_batched(marker_points, marker_valid, obj, sdf, fbar, normals)

    assert mdilate.is_cuda
    assert mshear.is_cuda
    assert torch.isfinite(mdilate).all()
    assert torch.isfinite(mshear).all()
