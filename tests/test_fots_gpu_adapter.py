from __future__ import annotations

import pytest
import torch

from integrate.fots_adapter import RevoFotsAdapter, RevoFotsCfg, RevoFotsGpuAdapter


def _cfg(device: str = "cpu") -> RevoFotsCfg:
    return RevoFotsCfg(
        width=240,
        height=320,
        marker_rows=16,
        marker_cols=8,
        marker_margin_x=15.0,
        marker_margin_y=26.0 * 320.0 / 240.0,
        track_contact_center=True,
        render_marker_image=False,
        render_overlay_image=False,
        device=device,
    )


def _contact_depth(batch: int, *, device: str = "cpu") -> torch.Tensor:
    depth = torch.zeros((batch, 32, 24), dtype=torch.float32, device=device)
    depth[:, 10:22, 6:18] = 0.001
    return depth


def test_fots_gpu_adapter_returns_zero_for_no_contact():
    adapter = RevoFotsGpuAdapter(_cfg())

    displacement = adapter.step_displacement_only(torch.zeros((3, 32, 24), dtype=torch.float32))

    assert displacement.shape == (3, 128, 2)
    assert adapter.last_output is not None
    assert adapter.last_output.marker_flow.shape == (3, 2, 128, 2)
    assert torch.count_nonzero(displacement) == 0
    assert torch.isfinite(displacement).all()


def test_fots_gpu_adapter_fully_resizes_native_depth():
    adapter = RevoFotsGpuAdapter(_cfg())
    native_depth_mm = _contact_depth(2) * 1000.0

    resized = adapter._resize_depth_mm(native_depth_mm)

    assert resized.shape == (2, 320, 240)
    assert resized.device == native_depth_mm.device
    assert torch.amax(resized) > 0.0


def test_fots_gpu_dilation_matches_integrated_adapter():
    cfg = _cfg()
    depth = _contact_depth(1)
    reference = RevoFotsAdapter(cfg).step(depth).marker_flow
    expected = reference[:, 1] - reference[:, 0]

    actual = RevoFotsGpuAdapter(cfg).step_displacement_only(depth)

    assert torch.linalg.norm(actual, dim=-1).amax() > 0.0
    assert torch.allclose(actual, expected, atol=2.0e-5, rtol=1.0e-5)


def test_fots_partial_reset_only_clears_selected_batch_history():
    cfg = _cfg()
    adapter = RevoFotsGpuAdapter(cfg)
    first = _contact_depth(2)
    adapter.step_displacement_only(first)
    moved = torch.roll(first, shifts=2, dims=-1)
    adapter.reset(torch.tensor([True, False]))

    actual = adapter.step_displacement_only(moved)
    fresh = RevoFotsGpuAdapter(cfg).step_displacement_only(moved[:1])

    assert torch.allclose(actual[:1], fresh, atol=2.0e-5, rtol=1.0e-5)
    assert not torch.allclose(actual[1:2], fresh, atol=1.0e-4, rtol=1.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fots_batched_path_stays_on_gpu():
    adapter = RevoFotsGpuAdapter(_cfg("cuda"))

    displacement = adapter.step_displacement_only(_contact_depth(6, device="cuda"))

    assert displacement.is_cuda
    assert adapter.last_output is not None
    assert adapter.last_output.marker_flow.is_cuda
    assert torch.isfinite(displacement).all()
