"""Tensor helpers shared by calibrated Direct and ManagerBased tactile inputs."""

import sys
import types

import torch


def ensure_taxim_scatter() -> None:
    """Provide Taxim's scatter-min reduction when the optional extension is absent."""
    try:
        __import__("torch_scatter")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "torch_scatter":
            raise

    module = types.ModuleType("torch_scatter")

    def scatter_min(src, index, dim=-1, out=None, dim_size=None, **kwargs):
        if out is None:
            shape = list(src.shape)
            shape[dim] = int(dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0))
            out = torch.full(shape, torch.inf, dtype=src.dtype, device=src.device)
        index = index.to(device=src.device, dtype=torch.long)
        # Taxim reduces the last dimension and supplies one shared index per column.
        while index.ndim < src.ndim:
            index = index.unsqueeze(0)
        out.scatter_reduce_(dim, index.expand_as(src), src, reduce="amin", include_self=True)
        return out, None

    module.scatter_min = scatter_min
    sys.modules["torch_scatter"] = module


def project_marker_flow(adapter, displacement, depth, points, valid, marker_points, marker_normals, marker_valid):
    """Project HydroShear's metric displacement into the collector's pixel flow."""
    _, _, _, marker_valid, row_axes, col_axes, _ = adapter._batched_marker_samples(
        depth, points, valid, None, marker_points, marker_normals, marker_valid,
    )
    row_axes, col_axes = adapter._batched_uniform_marker_projection_axes(points, valid, row_axes, col_axes)
    displacement = torch.where(marker_valid[..., None], displacement, torch.zeros_like(displacement))
    # The collector projects each sensor separately. Keep the same reductions,
    # but launch them once for the entire environment/finger batch.
    scale = float(adapter.cfg.arrow_scale_px_per_m)
    du = torch.sum(displacement * col_axes, dim=-1) * scale
    dv = torch.sum(displacement * row_axes, dim=-1) * scale
    delta_uv = torch.stack((du, dv), dim=-1).to(dtype=torch.float32)
    marker_uv = adapter._marker_uv_t.to(dtype=torch.float32).expand_as(delta_uv)
    return torch.stack((marker_uv, marker_uv + delta_uv), dim=1)
