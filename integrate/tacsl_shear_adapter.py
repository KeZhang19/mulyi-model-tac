"""TacSL SDF force-field visualization for RevoLab.

This mirrors IsaacLab TacSL's ``compute_tactile_shear_image`` layout: arrows
visualize ``tactile_shear_force`` on a marker-like grid.  The display uses the
same black-background, green-vector style as the HydroShear debug marker view;
normal force is still computed upstream and exposed in the output data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


_MARKER_GREEN = (0, 255, 0)


@dataclass
class TacslShearOutput:
    shear_images: np.ndarray
    normal_force: torch.Tensor
    shear_force: torch.Tensor
    penetration_depth_m: torch.Tensor
    max_depth_mm: torch.Tensor
    active_taxels: torch.Tensor
    max_normal_force: torch.Tensor
    max_shear_force: torch.Tensor


@dataclass
class RevoTacslShearCfg:
    width: int = 240
    height: int = 240
    normal_force_threshold: float = 0.00008
    shear_force_threshold: float = 0.001
    resolution: int = 8
    render_rows: int = 9
    render_cols: int = 11
    render_stride: int = 0
    marker_margin_x: float = 15.0
    marker_margin_y: float = 26.0
    marker_radius: int = 3
    arrow_thickness: int = 2
    arrow_tip_length: float = 0.4
    min_arrow_length_px: float = 0.0
    device: str | None = None


class RevoTacslShearAdapter:
    """Render IsaacLab TacSL-style normal/shear force-field images."""

    def __init__(self, cfg: RevoTacslShearCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def reset(self) -> None:
        return

    def step(
        self,
        tactile_normal_force: np.ndarray | torch.Tensor,
        tactile_shear_force: np.ndarray | torch.Tensor,
        penetration_depth_m: np.ndarray | torch.Tensor | None = None,
    ) -> TacslShearOutput:
        normal = self._to_tensor(tactile_normal_force)
        shear = self._to_tensor(tactile_shear_force)
        if normal.ndim == 2:
            normal = normal[None, ...]
        if shear.ndim == 3:
            shear = shear[None, ...]
        if penetration_depth_m is None:
            depth = torch.zeros_like(normal, dtype=torch.float32)
        else:
            depth = self._to_tensor(penetration_depth_m)
            if depth.ndim == 2:
                depth = depth[None, ...]

        sensor_count = min(normal.shape[0], shear.shape[0], depth.shape[0])
        images = []
        max_depth_mm = []
        active_taxels = []
        max_normal = []
        max_shear = []

        for i in range(sensor_count):
            normal_i = torch.nan_to_num(normal[i], nan=0.0, posinf=0.0, neginf=0.0)
            shear_i = torch.nan_to_num(shear[i], nan=0.0, posinf=0.0, neginf=0.0)
            depth_i = torch.nan_to_num(depth[i], nan=0.0, posinf=0.0, neginf=0.0)
            render_normal, render_shear = self._prepare_render_grid(normal_i, shear_i)
            images.append(self._render_force_field(render_normal, render_shear))
            max_depth_mm.append(torch.amax(depth_i) * 1000.0 if depth_i.numel() else torch.tensor(0.0, device=self.device))
            active_taxels.append(torch.count_nonzero(normal_i > 0.0).to(torch.float32))
            max_normal.append(torch.amax(torch.abs(normal_i)) if normal_i.numel() else torch.tensor(0.0, device=self.device))
            shear_norm = torch.linalg.norm(shear_i, dim=-1) if shear_i.numel() else torch.zeros((), dtype=torch.float32, device=self.device)
            max_shear.append(torch.amax(shear_norm) if shear_norm.numel() else torch.tensor(0.0, device=self.device))

        if not images:
            images = [np.zeros((1, 1, 3), dtype=np.uint8)]
            zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            max_depth_mm = [zero]
            active_taxels = [zero]
            max_normal = [zero]
            max_shear = [zero]

        return TacslShearOutput(
            shear_images=np.stack(images, axis=0),
            normal_force=normal[:sensor_count],
            shear_force=shear[:sensor_count],
            penetration_depth_m=depth[:sensor_count],
            max_depth_mm=torch.stack(max_depth_mm).to(dtype=torch.float32),
            active_taxels=torch.stack(active_taxels).to(dtype=torch.float32),
            max_normal_force=torch.stack(max_normal).to(dtype=torch.float32),
            max_shear_force=torch.stack(max_shear).to(dtype=torch.float32),
        )

    def _to_tensor(self, value, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            out = value.detach().to(device=self.device, dtype=dtype)
        else:
            out = torch.as_tensor(value, dtype=dtype, device=self.device)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0) if out.is_floating_point() else out

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _prepare_render_grid(self, normal_force: torch.Tensor, shear_force: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        normal_force = np.nan_to_num(self._to_numpy(normal_force).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        shear_force = np.nan_to_num(self._to_numpy(shear_force).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        rows, cols = normal_force.shape
        target_rows = int(self.cfg.render_rows)
        target_cols = int(self.cfg.render_cols)
        if target_rows > 0 and target_cols > 0:
            return self._aggregate_to_grid(normal_force, shear_force, target_rows, target_cols)

        stride = int(self.cfg.render_stride)
        if stride <= 0:
            stride = max(1, int(round(max(rows, cols) / 30.0))) if max(rows, cols) > 60 else 1
        if stride <= 1:
            return normal_force, shear_force

        out_rows = int(math.ceil(rows / stride))
        out_cols = int(math.ceil(cols / stride))
        return self._aggregate_to_grid(normal_force, shear_force, out_rows, out_cols)

    def _aggregate_to_grid(
        self,
        normal_force: np.ndarray,
        shear_force: np.ndarray,
        out_rows: int,
        out_cols: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = normal_force.shape
        normal_out = np.zeros((out_rows, out_cols), dtype=np.float32)
        shear_out = np.zeros((out_rows, out_cols, 2), dtype=np.float32)
        row_edges = np.rint(np.linspace(0, rows, out_rows + 1)).astype(np.int32)
        col_edges = np.rint(np.linspace(0, cols, out_cols + 1)).astype(np.int32)

        for row in range(out_rows):
            r0 = int(row_edges[row])
            r1 = max(r0 + 1, int(row_edges[row + 1]))
            r1 = min(rows, r1)
            for col in range(out_cols):
                c0 = int(col_edges[col])
                c1 = max(c0 + 1, int(col_edges[col + 1]))
                c1 = min(cols, c1)
                normal_block = normal_force[r0:r1, c0:c1]
                shear_block = shear_force[r0:r1, c0:c1]
                active = (normal_block > 0.0) | (np.linalg.norm(shear_block, axis=-1) > 0.0)
                if not bool(np.any(active)):
                    continue
                normal_out[row, col] = np.mean(normal_block[active])
                shear_out[row, col] = np.mean(shear_block[active], axis=0)

        return normal_out, shear_out

    def _render_force_field(self, normal_force: np.ndarray, shear_force: np.ndarray) -> np.ndarray:
        rows, cols = normal_force.shape
        resolution = max(1, int(self.cfg.resolution))
        fixed_canvas = int(self.cfg.width) > 0 and int(self.cfg.height) > 0
        if fixed_canvas:
            img = np.zeros((int(self.cfg.height), int(self.cfg.width), 3), dtype=np.uint8)
            xs = np.linspace(
                float(self.cfg.marker_margin_x),
                float(self.cfg.width) - float(self.cfg.marker_margin_x),
                cols,
                dtype=np.float32,
            ).astype(np.float32)
            ys = np.linspace(
                float(self.cfg.marker_margin_y),
                float(self.cfg.height) - float(self.cfg.marker_margin_y),
                rows,
            ).astype(np.float32)
            x_spacing = float(np.mean(np.diff(xs))) if cols > 1 else float(resolution)
            y_spacing = float(np.mean(np.diff(ys))) if rows > 1 else float(resolution)
            arrow_scale = max(1.0, 0.5 * (abs(x_spacing) + abs(y_spacing)))
        else:
            img = np.zeros((rows * resolution, cols * resolution, 3), dtype=np.uint8)
            xs = np.arange(cols, dtype=np.float32) * resolution + resolution // 2
            ys = np.arange(rows, dtype=np.float32) * resolution + resolution // 2
            arrow_scale = float(resolution)
        shear_threshold = max(float(self.cfg.shear_force_threshold), 1.0e-12)

        for row in range(rows):
            for col in range(cols):
                n_value = float(normal_force[row, col])
                shear_vec = shear_force[row, col].astype(np.float32)
                shear_norm = float(np.linalg.norm(shear_vec))
                if not (n_value > 0.0 or shear_norm > 0.0):
                    continue

                loc0_x = float(ys[row])
                loc0_y = float(xs[col])
                loc1_x = loc0_x + float(shear_vec[0]) / shear_threshold * arrow_scale
                loc1_y = loc0_y + float(shear_vec[1]) / shear_threshold * arrow_scale
                start = np.array((loc0_y, loc0_x), dtype=np.float32)
                end = np.array((loc1_y, loc1_x), dtype=np.float32)
                visual_delta = end - start
                visual_length = float(np.linalg.norm(visual_delta))
                min_arrow_length = max(0.0, float(self.cfg.min_arrow_length_px))
                if shear_norm > 0.0 and 0.0 < visual_length < min_arrow_length:
                    end = start + visual_delta / max(visual_length, 1.0e-12) * min_arrow_length
                    visual_length = min_arrow_length

                if visual_length < 0.5:
                    _draw_circle(img, start, radius=max(1, int(self.cfg.arrow_thickness)), color=_MARKER_GREEN)
                else:
                    _draw_arrow(
                        img,
                        start,
                        end,
                        color=_MARKER_GREEN,
                        thickness=int(self.cfg.arrow_thickness),
                        tip_length=float(self.cfg.arrow_tip_length),
                    )

        return img


def _draw_circle(img: np.ndarray, center: np.ndarray, *, radius: int, color: tuple[int, int, int]) -> None:
    try:
        import cv2

        pt = (int(round(float(center[0]))), int(round(float(center[1]))))
        cv2.circle(img, pt, max(1, int(radius)), color, thickness=-1, lineType=cv2.LINE_AA)
        return
    except Exception:
        pass

    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    r = max(1, int(radius))
    y0 = max(0, y - r)
    y1 = min(img.shape[0], y + r + 1)
    x0 = max(0, x - r)
    x1 = min(img.shape[1], x + r + 1)
    img[y0:y1, x0:x1] = color


def _draw_line(
    img: np.ndarray, start: np.ndarray, end: np.ndarray, *, color: tuple[int, int, int], thickness: int = 1
) -> None:
    x0, y0 = int(round(float(start[0]))), int(round(float(start[1])))
    x1, y1 = int(round(float(end[0]))), int(round(float(end[1])))
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).round().astype(int)
    ys = np.linspace(y0, y1, steps + 1).round().astype(int)
    radius = max(0, int(thickness) // 2)
    for x, y in zip(xs, ys):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            y0p = max(0, y - radius)
            y1p = min(img.shape[0], y + radius + 1)
            x0p = max(0, x - radius)
            x1p = min(img.shape[1], x + radius + 1)
            img[y0p:y1p, x0p:x1p] = color


def _draw_arrow(
    img: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int,
    tip_length: float,
) -> None:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1.0e-3:
        return

    try:
        import cv2

        pt1 = (int(round(float(start[0]))), int(round(float(start[1]))))
        pt2 = (int(round(float(end[0]))), int(round(float(end[1]))))
        cv2.arrowedLine(img, pt1, pt2, color, int(thickness), tipLength=float(tip_length))
        return
    except Exception:
        pass

    _draw_line(img, start, end, color=color, thickness=thickness)
    direction = delta / length
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    head_len = max(4.0, float(tip_length) * length)
    head_width = 0.55 * head_len
    left = end - direction * head_len + normal * head_width
    right = end - direction * head_len - normal * head_width
    _draw_line(img, end, left, color=color, thickness=thickness)
    _draw_line(img, end, right, color=color, thickness=thickness)
