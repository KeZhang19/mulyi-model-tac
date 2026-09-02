"""Small torch image helpers for tactile debug panes.

The integrated viewer ultimately has to hand CPU uint8 bytes to omni.ui or
OpenCV, but keeping pane construction as tensors avoids bouncing every
intermediate image through NumPy.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_device(*values: Any, fallback: torch.device | str | None = None) -> torch.device:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device(fallback) if fallback is not None else default_device()


def as_float_tensor(value: Any, *, device: torch.device | str | None = None) -> torch.Tensor:
    target = tensor_device(value, fallback=device)
    if isinstance(value, torch.Tensor):
        out = value.detach().to(device=target, dtype=torch.float32)
    else:
        out = torch.as_tensor(value, dtype=torch.float32, device=target)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def as_bool_tensor(value: Any, *, device: torch.device | str | None = None) -> torch.Tensor:
    target = tensor_device(value, fallback=device)
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=target, dtype=torch.bool)
    return torch.as_tensor(value, dtype=torch.bool, device=target)


def as_uint8_tensor(value: Any, *, device: torch.device | str | None = None) -> torch.Tensor:
    target = tensor_device(value, fallback=device)
    if isinstance(value, torch.Tensor):
        out = value.detach().to(device=target)
        if out.dtype == torch.uint8:
            return out
        return torch.clamp(out, 0, 255).to(torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device=target)


def to_numpy_uint8_rgb(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack((arr, arr, arr), axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr, dtype=np.uint8)


def to_numpy_float32(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def zeros_rgb(height: int, width: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.zeros((max(1, int(height)), max(1, int(width)), 3), dtype=torch.uint8, device=tensor_device(fallback=device))


def full_rgb(
    height: int,
    width: int,
    value: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return torch.full(
        (max(1, int(height)), max(1, int(width)), 3),
        int(value),
        dtype=torch.uint8,
        device=tensor_device(fallback=device),
    )


def jet_colormap(values: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(values.to(dtype=torch.float32), 0.0, 1.0)
    r = torch.clamp(1.5 - torch.abs(4.0 * values - 3.0), 0.0, 1.0)
    g = torch.clamp(1.5 - torch.abs(4.0 * values - 2.0), 0.0, 1.0)
    b = torch.clamp(1.5 - torch.abs(4.0 * values - 1.0), 0.0, 1.0)
    return torch.clamp(torch.stack((r, g, b), dim=-1) * 255.0, 0.0, 255.0).to(torch.uint8)


def normalize_grid(grid: Any, gamma: float, *, device: torch.device | str | None = None) -> torch.Tensor:
    values = as_float_tensor(grid, device=device)
    vmax = torch.amax(values) if values.numel() else torch.tensor(0.0, dtype=torch.float32, device=values.device)
    norm = torch.where(vmax > 0.0, values / (vmax + 1.0e-8), torch.zeros_like(values))
    return torch.pow(torch.clamp(norm, 0.0, 1.0), max(1.0e-6, float(gamma)))


def fixed_scale_grid(grid: Any, vmax: float, gamma: float, *, device: torch.device | str | None = None) -> torch.Tensor:
    values = as_float_tensor(grid, device=device)
    if float(vmax) <= 0.0:
        return normalize_grid(values, gamma, device=values.device)
    norm = torch.clamp(values / float(vmax), 0.0, 1.0)
    return torch.pow(norm, max(1.0e-6, float(gamma)))


def _surface_spacing(points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor | None:
    distances = []
    if points.shape[1] > 1:
        mask = valid[:, 1:] & valid[:, :-1]
        vals = torch.linalg.norm(points[:, 1:] - points[:, :-1], dim=-1)
        distances.append(vals[mask])
    if points.shape[0] > 1:
        mask = valid[1:, :] & valid[:-1, :]
        vals = torch.linalg.norm(points[1:, :] - points[:-1, :], dim=-1)
        distances.append(vals[mask])
    distances = [d[torch.isfinite(d) & (d > 1.0e-9)] for d in distances if d.numel()]
    if not distances:
        return None
    joined = torch.cat(distances)
    if joined.numel() == 0:
        return None
    return torch.median(joined)


def _surface_weighted_resize_grid(
    grid: torch.Tensor,
    surface_points: torch.Tensor | None,
    surface_valid: torch.Tensor | None,
    target_h: int,
    target_w: int,
) -> torch.Tensor | None:
    if surface_points is None or surface_valid is None or grid.ndim != 2:
        return None
    source_h = int(grid.shape[0])
    source_w = int(grid.shape[1])
    if surface_points.shape[:2] != (source_h, source_w) or surface_points.shape[-1] != 3:
        return None
    if surface_valid.shape[:2] != (source_h, source_w):
        return None

    points = torch.nan_to_num(surface_points.to(device=grid.device, dtype=torch.float32))
    valid = surface_valid.to(device=grid.device, dtype=torch.bool) & torch.isfinite(surface_points).all(dim=-1).to(
        device=grid.device
    )
    if not bool(torch.any(valid).detach().cpu()):
        return None
    spacing = _surface_spacing(points, valid)
    if spacing is None or not bool(torch.isfinite(spacing).detach().cpu()) or float(spacing.detach().cpu()) <= 1.0e-9:
        return None

    ys = (torch.arange(target_h, dtype=torch.float32, device=grid.device) + 0.5) * source_h / float(target_h) - 0.5
    xs = (torch.arange(target_w, dtype=torch.float32, device=grid.device) + 0.5) * source_w / float(target_w) - 0.5
    ys = torch.clamp(ys, 0.0, max(0.0, float(source_h - 1)))
    xs = torch.clamp(xs, 0.0, max(0.0, float(source_w - 1)))
    y0 = torch.floor(ys).to(torch.long)
    x0 = torch.floor(xs).to(torch.long)
    y1 = torch.clamp(y0 + 1, max=source_h - 1)
    x1 = torch.clamp(x0 + 1, max=source_w - 1)
    wy = (ys - y0.to(torch.float32)).view(target_h, 1)
    wx = (xs - x0.to(torch.float32)).view(1, target_w)

    yy0 = y0.view(target_h, 1).expand(target_h, target_w)
    yy1 = y1.view(target_h, 1).expand(target_h, target_w)
    xx0 = x0.view(1, target_w).expand(target_h, target_w)
    xx1 = x1.view(1, target_w).expand(target_h, target_w)

    base_w = torch.stack(
        ((1.0 - wy) * (1.0 - wx), (1.0 - wy) * wx, wy * (1.0 - wx), wy * wx),
        dim=-1,
    )
    neighbor_points = torch.stack((points[yy0, xx0], points[yy0, xx1], points[yy1, xx0], points[yy1, xx1]), dim=-2)
    neighbor_valid = torch.stack((valid[yy0, xx0], valid[yy0, xx1], valid[yy1, xx0], valid[yy1, xx1]), dim=-1)
    valid_w = base_w * neighbor_valid.to(torch.float32)
    target_den = torch.sum(valid_w, dim=-1, keepdim=True)
    target_points = torch.sum(neighbor_points * valid_w.unsqueeze(-1), dim=-2) / torch.clamp(target_den, min=1.0e-8)

    neighbor_values = torch.stack((grid[yy0, xx0], grid[yy0, xx1], grid[yy1, xx0], grid[yy1, xx1]), dim=-1)
    bilinear_values = torch.sum(neighbor_values * base_w, dim=-1)
    d2 = torch.sum((neighbor_points - target_points.unsqueeze(-2)) ** 2, dim=-1)
    sigma2 = torch.clamp((spacing * 0.75) ** 2, min=1.0e-12)
    surface_w = base_w * torch.exp(-0.5 * d2 / sigma2) * neighbor_valid.to(torch.float32)
    den = torch.sum(surface_w, dim=-1)
    values = torch.sum(neighbor_values * surface_w, dim=-1) / torch.clamp(den, min=1.0e-8)
    return torch.where(den > 1.0e-8, values, bilinear_values)


def force_strip(tactile: Any, *, scale: int, gamma: float) -> torch.Tensor:
    values = as_float_tensor(tactile)
    if values.ndim == 2:
        values = values[None, ...]
    imgs = []
    for i in range(int(values.shape[0])):
        rgb = jet_colormap(normalize_grid(torch.rot90(values[i], k=1, dims=(0, 1)), gamma, device=values.device))
        if int(scale) != 1:
            rgb = rgb.repeat_interleave(max(1, int(scale)), dim=0).repeat_interleave(max(1, int(scale)), dim=1)
        imgs.append(rgb)
    return torch.cat(imgs, dim=1) if imgs else zeros_rgb(1, 1, device=values.device)


def depth_weighted_center_tensor(depth: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(center_xy, active)`` without synchronizing a CUDA tensor."""

    weights = torch.clamp(as_float_tensor(depth), min=0.0)
    total = torch.sum(weights)
    active = total > 1.0e-12
    safe_total = torch.clamp(total, min=1.0e-12)
    ys = torch.arange(weights.shape[0], dtype=torch.float32, device=weights.device).reshape(-1, 1)
    xs = torch.arange(weights.shape[1], dtype=torch.float32, device=weights.device).reshape(1, -1)
    cx = torch.sum(xs * weights) / safe_total
    cy = torch.sum(ys * weights) / safe_total
    center = torch.stack((cx, cy))
    return torch.where(active, center, torch.zeros_like(center)), active


def depth_weighted_center_px(depth: Any) -> tuple[float, float] | None:
    center, active = depth_weighted_center_tensor(depth)
    packed = torch.cat((center, active.to(dtype=torch.float32).reshape(1))).detach().cpu()
    if not bool(packed[2]):
        return None
    return float(packed[0]), float(packed[1])


def draw_ring(
    img: torch.Tensor,
    center_xy: tuple[Any, Any] | torch.Tensor,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 7,
    thickness: int = 2,
    enabled: Any = True,
) -> None:
    cx, cy = center_xy
    cx = torch.as_tensor(cx, dtype=torch.float32, device=img.device)
    cy = torch.as_tensor(cy, dtype=torch.float32, device=img.device)
    yy = torch.arange(img.shape[0], dtype=torch.float32, device=img.device).reshape(-1, 1)
    xx = torch.arange(img.shape[1], dtype=torch.float32, device=img.device).reshape(1, -1)
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    half = max(0.5, float(thickness) * 0.5)
    mask = (dist >= max(0.0, float(radius) - half)) & (dist <= float(radius) + half)
    mask = mask & torch.as_tensor(enabled, dtype=torch.bool, device=img.device)
    img[mask] = torch.tensor(color, dtype=torch.uint8, device=img.device)


def _joint_depth_interpolation(
    base_depth: torch.Tensor,
    marker_depth: torch.Tensor,
    marker_valid: torch.Tensor,
    source_indices: torch.Tensor,
    source_weights: torch.Tensor,
    pixel_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate base-grid and Marker depths together using precomputed triangles."""

    base = torch.nan_to_num(base_depth.to(dtype=torch.float32)).reshape(-1)
    markers = marker_depth.to(device=base.device, dtype=torch.float32).reshape(-1)
    marker_finite = torch.isfinite(markers)
    markers = torch.nan_to_num(markers)
    marker_mask = marker_valid.to(device=base.device, dtype=torch.bool).reshape(-1)
    marker_count = min(int(markers.numel()), int(marker_mask.numel()))
    markers = markers[:marker_count]
    marker_mask = marker_mask[:marker_count] & marker_finite[:marker_count]
    sample_values = torch.cat((base, markers), dim=0)
    sample_valid = torch.cat(
        (
            torch.ones(base.shape, dtype=torch.bool, device=base.device),
            marker_mask,
        ),
        dim=0,
    )

    indices = source_indices.to(device=base.device, dtype=torch.long)
    weights = source_weights.to(device=base.device, dtype=torch.float32)
    valid = pixel_valid.to(device=base.device, dtype=torch.bool)
    if indices.ndim != 3 or int(indices.shape[-1]) != 3 or weights.shape != indices.shape:
        raise ValueError("Joint TacMap interpolation expects HxWx3 indices and weights")
    if valid.shape != indices.shape[:-1]:
        raise ValueError("Joint TacMap interpolation pixel mask shape mismatch")

    gathered_values = sample_values[indices]
    gathered_valid = sample_valid[indices]
    interpolated = torch.sum(gathered_values * weights, dim=-1)
    valid = valid & torch.all(gathered_valid, dim=-1) & torch.isfinite(interpolated)
    return torch.nan_to_num(interpolated), valid


def tacmap_strip(
    tacmap: Any,
    *,
    tacmap_raw: Any | None = None,
    surface_points: Any | None = None,
    surface_valid: Any | None = None,
    scale: int,
    gamma: float,
    view: str,
    display_max_m: float,
    target_size: tuple[int, int] | None = None,
    resize_mode: str = "nearest",
    joint_marker_depth_values: Any | None = None,
    joint_marker_depth_valid: Any | None = None,
    joint_source_indices: Any | None = None,
    joint_source_weights: Any | None = None,
    joint_pixel_valid: Any | None = None,
) -> torch.Tensor:
    values = as_float_tensor(tacmap)
    if values.ndim == 2:
        values = values[None, ...]
    raw = None if tacmap_raw is None else as_float_tensor(tacmap_raw, device=values.device)
    if raw is not None and raw.ndim == 2:
        raw = raw[None, ...]
    points = None if surface_points is None else as_float_tensor(surface_points, device=values.device)
    if points is not None and points.ndim == 3:
        points = points[None, ...]
    valid = None if surface_valid is None else as_bool_tensor(surface_valid, device=values.device)
    if valid is not None and valid.ndim == 2:
        valid = valid[None, ...]
    marker_depth = (
        None if joint_marker_depth_values is None else as_float_tensor(joint_marker_depth_values, device=values.device)
    )
    if marker_depth is not None and marker_depth.ndim == 1:
        marker_depth = marker_depth[None, ...]
    marker_valid = (
        None if joint_marker_depth_valid is None else as_bool_tensor(joint_marker_depth_valid, device=values.device)
    )
    if marker_valid is not None and marker_valid.ndim == 1:
        marker_valid = marker_valid[None, ...]
    source_indices = None if joint_source_indices is None else torch.as_tensor(joint_source_indices, device=values.device)
    if source_indices is not None and source_indices.ndim == 3:
        source_indices = source_indices[None, ...]
    source_weights = None if joint_source_weights is None else as_float_tensor(joint_source_weights, device=values.device)
    if source_weights is not None and source_weights.ndim == 3:
        source_weights = source_weights[None, ...]
    pixel_valid = None if joint_pixel_valid is None else as_bool_tensor(joint_pixel_valid, device=values.device)
    if pixel_valid is not None and pixel_valid.ndim == 2:
        pixel_valid = pixel_valid[None, ...]

    imgs = []
    for i in range(int(values.shape[0])):
        if view == "raw" and raw is not None and i < int(raw.shape[0]):
            gray = torch.clamp(fixed_scale_grid(raw[i], display_max_m, gamma, device=values.device) * 255.0, 0.0, 255.0)
        elif view == "mask" and raw is not None and i < int(raw.shape[0]):
            gray = (raw[i] > 0.0).to(torch.float32) * 255.0
        elif view == "quantized":
            gray = torch.clamp(values[i], 0.0, 255.0)
        else:
            gray = torch.clamp(normalize_grid(values[i], gamma, device=values.device) * 255.0, 0.0, 255.0)
        original_h = max(1, int(gray.shape[-2]))
        original_w = max(1, int(gray.shape[-1]))
        scale_y = 1.0
        scale_x = 1.0
        pressure_center = None
        pressure_center_active = None
        if raw is not None and i < int(raw.shape[0]):
            pressure_center, pressure_center_active = depth_weighted_center_tensor(raw[i])
        if target_size is not None and int(target_size[0]) > 0 and int(target_size[1]) > 0:
            target_h = max(1, int(target_size[0]))
            target_w = max(1, int(target_size[1]))
            if target_h != original_h or target_w != original_w:
                mode = str(resize_mode).lower()
                source = gray.to(torch.float32).unsqueeze(0).unsqueeze(0)
                surface_gray = None
                if mode == "surface" and points is not None and valid is not None and i < int(points.shape[0]):
                    surface_gray = _surface_weighted_resize_grid(
                        gray.to(torch.float32),
                        points[i],
                        valid[i] if i < int(valid.shape[0]) else None,
                        target_h,
                        target_w,
                    )
                if surface_gray is not None:
                    resized = surface_gray
                elif mode in ("surface", "bilinear"):
                    resized = torch.nn.functional.interpolate(
                        source,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                else:
                    resized = torch.nn.functional.interpolate(source, size=(target_h, target_w), mode="nearest")[0, 0]
                gray = torch.clamp(resized, 0.0, 255.0)
            scale_y = target_h / float(original_h)
            scale_x = target_w / float(original_w)
        elif int(scale) != 1:
            scale_factor = max(1, int(scale))
            gray = gray.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)
            scale_y = float(scale_factor)
            scale_x = float(scale_factor)
        if (
            view == "raw"
            and raw is not None
            and marker_depth is not None
            and marker_valid is not None
            and source_indices is not None
            and source_weights is not None
            and pixel_valid is not None
            and i < int(raw.shape[0])
            and i < int(marker_depth.shape[0])
            and i < int(marker_valid.shape[0])
            and i < int(source_indices.shape[0])
            and i < int(source_weights.shape[0])
            and i < int(pixel_valid.shape[0])
        ):
            joint_depth, joint_valid = _joint_depth_interpolation(
                raw[i],
                marker_depth[i],
                marker_valid[i],
                source_indices[i],
                source_weights[i],
                pixel_valid[i],
            )
            if tuple(joint_depth.shape) != tuple(gray.shape):
                raise ValueError(
                    f"Joint TacMap interpolation shape {tuple(joint_depth.shape)} does not match display {tuple(gray.shape)}"
                )
            joint_gray = torch.clamp(
                fixed_scale_grid(joint_depth, display_max_m, gamma, device=values.device) * 255.0,
                0.0,
                255.0,
            )
            gray = torch.where(joint_valid, joint_gray, gray)
        gray = torch.clamp(gray, 0.0, 255.0).to(torch.uint8)
        rgb = torch.stack((gray, gray, gray), dim=-1)
        if pressure_center is not None:
            center = (
                (pressure_center[0] + 0.5) * scale_x - 0.5,
                (pressure_center[1] + 0.5) * scale_y - 0.5,
            )
            min_cells = max(1.0, float(min(original_h, original_w)))
            radius_cells = min(7.0, max(1.5, 0.08 * min_cells))
            ring_scale = max(1.0, min(scale_x, scale_y))
            draw_ring(
                rgb,
                center,
                radius=max(1, int(round(radius_cells * ring_scale))),
                thickness=max(1, int(round(max(0.35, 0.25 * radius_cells) * ring_scale))),
                enabled=pressure_center_active,
            )
        imgs.append(rgb)
    return torch.cat(imgs, dim=1) if imgs else zeros_rgb(1, 1, device=values.device)


def image_strip(images: Any, *, scale: int = 1) -> np.ndarray:
    arr = to_numpy_uint8_rgb(images)
    if arr.ndim == 2:
        arr = to_numpy_uint8_rgb(arr)[None, ...]
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[None, ...]
        else:
            arr = np.stack((arr, arr, arr), axis=-1)
    elif arr.ndim != 4:
        arr = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    imgs = []
    for i in range(int(arr.shape[0])):
        rgb = to_numpy_uint8_rgb(arr[i])[:, :, :3]
        if int(scale) != 1:
            repeat = max(1, int(scale))
            rgb = np.repeat(np.repeat(rgb, repeat, axis=0), repeat, axis=1)
        imgs.append(rgb)
    return np.concatenate(imgs, axis=1) if imgs else np.zeros((1, 1, 3), dtype=np.uint8)


def resize_nearest(img: Any, height: int) -> np.ndarray:
    arr = to_numpy_uint8_rgb(img)
    target_h = max(1, int(height))
    if int(arr.shape[0]) == target_h:
        return arr
    repeat = max(1, int(round(target_h / max(1, int(arr.shape[0])))))
    resized = np.repeat(arr, repeat, axis=0)
    if int(resized.shape[0]) > target_h:
        return resized[:target_h]
    if int(resized.shape[0]) < target_h:
        pad = np.zeros((target_h - int(resized.shape[0]), int(resized.shape[1]), 3), dtype=np.uint8)
        return np.concatenate((resized, pad), axis=0)
    return resized


def resize_nearest_to_shape(img: Any, height: int, width: int) -> np.ndarray:
    arr = to_numpy_uint8_rgb(img)
    target_h = max(1, int(height))
    target_w = max(1, int(width))
    repeat_y = max(1, int(math.ceil(target_h / max(1, int(arr.shape[0])))))
    repeat_x = max(1, int(math.ceil(target_w / max(1, int(arr.shape[1])))))
    resized = np.repeat(np.repeat(arr, repeat_y, axis=0), repeat_x, axis=1)
    resized = resized[:target_h, :target_w]
    pad_h = max(0, target_h - int(resized.shape[0]))
    pad_w = max(0, target_w - int(resized.shape[1]))
    if pad_h or pad_w:
        resized = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    return resized


def combined_image(*panes: Any) -> np.ndarray:
    valid = [pane for pane in panes if pane is not None]
    if not valid:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    arrays = [to_numpy_uint8_rgb(pane)[:, :, :3] for pane in valid]
    height = max(int(pane.shape[0]) for pane in arrays)
    resized = [resize_nearest(pane, height) for pane in arrays]
    gap = np.full((height, 8, 3), 32, dtype=np.uint8)
    out = resized[0]
    for pane in resized[1:]:
        out = np.concatenate((out, gap, pane), axis=1)
    return out
