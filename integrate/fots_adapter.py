"""Self-contained FOTS-style marker motion from TacMap raw indentation depth.

The marker motion equations are adapted from TacEx's MarkerMotion wrapper,
which itself cites:

    FOTS: A Fast Optical Tactile Simulator for Sim2Real Learning of
    Tactile-motor Robot Manipulation Skills.

This module intentionally does not import TacEx or files from an external
TacEx checkout. It keeps the small subset needed by RevoLab local to this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F


_MARKER_GREEN = (0, 255, 0)


@dataclass
class FotsOutput:
    marker_flow: torch.Tensor
    marker_images: np.ndarray
    marker_overlay_images: np.ndarray
    depth_mm: torch.Tensor
    contact_mask: torch.Tensor
    max_depth_mm: torch.Tensor
    active_pixels: torch.Tensor
    theta_rad: torch.Tensor
    theta_delta_rad: torch.Tensor


@dataclass
class FotsBatchOutput:
    """GPU-resident batched FOTS state used by RL and its visualizer."""

    marker_flow: torch.Tensor
    marker_displacement: torch.Tensor
    max_depth_mm: torch.Tensor
    active_pixels: torch.Tensor
    active_markers: torch.Tensor
    theta_rad: torch.Tensor
    theta_delta_rad: torch.Tensor


@dataclass
class RevoFotsCfg:
    width: int = 240
    height: int = 240
    marker_cols: int = 11
    marker_rows: int = 9
    marker_margin_x: float = 15.0
    marker_margin_y: float = 26.0
    mm2pix: float = 19.58
    lamb: tuple[float, float, float] = (0.00125, 0.00021, 0.00038)
    contact_threshold_mm: float = 0.02
    depth_scale: float = 1.0
    theta: float = 0.0
    track_contact_center: bool = False
    show_depth_background: bool = False
    marker_arrow_scale: float = 1.0
    marker_size: float = 3.0
    marker_size_depth_gain: float = 2.0
    marker_min_size: float = 1.5
    marker_max_size: float = 5.5
    depth_background_max_mm: float = 15.0
    render_marker_image: bool = True
    render_overlay_image: bool = True
    device: str | None = None


class RevoFotsGpuAdapter:
    """Batched Torch implementation of the local FOTS marker equations.

    The native TacMap depth grid is fully resized on the GPU before contact
    extraction. Gaussian dilation, shear, and twist are then evaluated in the
    configured tactile-image pixel coordinates.
    """

    def __init__(self, cfg: RevoFotsCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        marker_x = torch.linspace(
            float(cfg.marker_margin_x),
            float(cfg.width) - float(cfg.marker_margin_x),
            max(1, int(cfg.marker_cols)),
            device=self.device,
            dtype=torch.float32,
        ).to(torch.long)
        marker_y = torch.linspace(
            float(cfg.marker_margin_y),
            float(cfg.height) - float(cfg.marker_margin_y),
            max(1, int(cfg.marker_rows)),
            device=self.device,
            dtype=torch.float32,
        ).to(torch.long)
        marker_y, marker_x = torch.meshgrid(marker_y, marker_x, indexing="ij")
        marker_x = marker_x.clamp(0, max(0, int(cfg.width) - 1)).to(torch.float32).reshape(-1)
        marker_y = marker_y.clamp(0, max(0, int(cfg.height) - 1)).to(torch.float32).reshape(-1)
        self._marker_uv = torch.stack((marker_x, marker_y), dim=-1)

        source_dx = marker_x[:, None] - marker_x[None, :]
        source_dy = marker_y[:, None] - marker_y[None, :]
        self._dilation_dx = source_dx
        self._dilation_dy = source_dy
        self._dilation_gaussian = torch.exp(
            -float(cfg.lamb[0]) * (source_dx.square() + source_dy.square())
        )

        self._state_valid: torch.Tensor | None = None
        self._start_center_mm: torch.Tensor | None = None
        self._start_theta_rad: torch.Tensor | None = None
        self._last_theta_rad: torch.Tensor | None = None
        self.last_output: FotsBatchOutput | None = None

    @property
    def marker_count(self) -> int:
        return int(self._marker_uv.shape[0])

    @property
    def marker_uv(self) -> torch.Tensor:
        return self._marker_uv

    def reset(self, batch_mask: torch.Tensor | None = None) -> None:
        if self._state_valid is None:
            return
        if batch_mask is None:
            self._state_valid.zero_()
            return
        mask = batch_mask.to(device=self._state_valid.device, dtype=torch.bool).reshape(-1)
        if mask.shape != self._state_valid.shape:
            raise ValueError(
                f"FOTS reset mask must have shape {tuple(self._state_valid.shape)}, got {tuple(mask.shape)}."
            )
        self._state_valid[mask] = False

    def _ensure_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> None:
        state_shape = (int(batch_size),)
        if (
            isinstance(self._state_valid, torch.Tensor)
            and self._state_valid.shape == state_shape
            and self._state_valid.device == device
        ):
            return
        self._state_valid = torch.zeros(state_shape, device=device, dtype=torch.bool)
        self._start_center_mm = torch.zeros((batch_size, 2), device=device, dtype=dtype)
        self._start_theta_rad = torch.zeros(state_shape, device=device, dtype=dtype)
        self._last_theta_rad = torch.zeros(state_shape, device=device, dtype=dtype)

    @staticmethod
    def _theta_values(
        theta_rad: torch.Tensor | np.ndarray | float | None,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        fallback: float,
    ) -> torch.Tensor:
        if theta_rad is None:
            return torch.full((batch_size,), float(fallback), device=device, dtype=dtype)
        theta = torch.as_tensor(theta_rad, device=device, dtype=dtype).reshape(-1)
        if theta.numel() == 0:
            return torch.full((batch_size,), float(fallback), device=device, dtype=dtype)
        if theta.numel() == 1:
            theta = theta.expand(batch_size)
        elif theta.numel() < batch_size:
            theta = torch.cat(
                (theta, torch.full((batch_size - theta.numel(),), float(fallback), device=device, dtype=dtype))
            )
        else:
            theta = theta[:batch_size]
        return torch.nan_to_num(theta, nan=float(fallback), posinf=float(fallback), neginf=float(fallback))

    def _resize_depth_mm(self, depth_mm: torch.Tensor) -> torch.Tensor:
        target_shape = (max(1, int(self.cfg.height)), max(1, int(self.cfg.width)))
        if tuple(depth_mm.shape[-2:]) == target_shape:
            return depth_mm
        return F.interpolate(
            depth_mm[:, None],
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    def _sample_marker_depth_mm(self, depth_mm: torch.Tensor) -> torch.Tensor:
        marker_uv = self._marker_uv.to(device=depth_mm.device)
        xs = marker_uv[:, 0].to(dtype=torch.long).clamp(0, int(self.cfg.width) - 1)
        ys = marker_uv[:, 1].to(dtype=torch.long).clamp(0, int(self.cfg.height) - 1)
        return depth_mm[:, ys, xs]

    def _contact_center_mm(self, contact_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, rows, cols = contact_mask.shape
        weights = contact_mask.to(dtype=torch.float32)
        count = weights.sum(dim=(1, 2))
        x_native = torch.arange(cols, device=contact_mask.device, dtype=torch.float32)
        y_native = torch.arange(rows, device=contact_mask.device, dtype=torch.float32)
        mean_x = (weights.sum(dim=1) * x_native[None]).sum(dim=1) / count.clamp_min(1.0)
        mean_y = (weights.sum(dim=2) * y_native[None]).sum(dim=1) / count.clamp_min(1.0)
        center_x_px = (mean_x + 0.5) * float(self.cfg.width) / float(cols) - 0.5
        center_y_px = (mean_y + 0.5) * float(self.cfg.height) / float(rows) - 0.5
        center_mm = torch.stack(
            (
                (center_x_px - float(self.cfg.width) / 2.0) / float(self.cfg.mm2pix),
                (center_y_px - float(self.cfg.height) / 2.0) / float(self.cfg.mm2pix),
            ),
            dim=-1,
        )
        return center_mm.to(dtype=torch.float32), count > 0.0

    @torch.no_grad()
    def step_displacement_only(
        self,
        tacmap_raw_m: torch.Tensor | np.ndarray,
        theta_rad: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        depth_m = torch.as_tensor(tacmap_raw_m, device=self.device, dtype=torch.float32)
        if depth_m.ndim == 2:
            depth_m = depth_m.unsqueeze(0)
        if depth_m.ndim != 3:
            raise ValueError(f"Batched FOTS expects depth=(B,H,W), got {tuple(depth_m.shape)}.")
        depth_m = torch.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        depth_mm = depth_m * (1000.0 * float(self.cfg.depth_scale))
        depth_mm = self._resize_depth_mm(depth_mm)
        batch_size = int(depth_mm.shape[0])
        self._ensure_state(batch_size, device=depth_mm.device, dtype=depth_mm.dtype)
        assert self._state_valid is not None
        assert self._start_center_mm is not None
        assert self._start_theta_rad is not None
        assert self._last_theta_rad is not None

        theta = self._theta_values(
            theta_rad,
            batch_size,
            device=depth_mm.device,
            dtype=depth_mm.dtype,
            fallback=float(self.cfg.theta),
        )
        contact_mask = depth_mm > float(self.cfg.contact_threshold_mm)
        center_mm, has_contact = self._contact_center_mm(contact_mask)

        sampled_depth_mm = self._sample_marker_depth_mm(depth_mm)
        sampled_depth_mm = sampled_depth_mm - depth_mm.amin(dim=(1, 2), keepdim=False)[:, None]
        marker_contact = sampled_depth_mm > float(self.cfg.contact_threshold_mm)
        marker_response_active = marker_contact.any(dim=1)
        contact_depth = torch.where(marker_contact, sampled_depth_mm / 10.0, torch.zeros_like(sampled_depth_mm))

        dilation_dx = self._dilation_dx.to(device=depth_mm.device, dtype=depth_mm.dtype)
        dilation_dy = self._dilation_dy.to(device=depth_mm.device, dtype=depth_mm.dtype)
        dilation_g = self._dilation_gaussian.to(device=depth_mm.device, dtype=depth_mm.dtype)
        x_dilate = contact_depth @ (dilation_dx * dilation_g).transpose(0, 1)
        y_dilate = contact_depth @ (dilation_dy * dilation_g).transpose(0, 1)

        state_was_valid = self._state_valid.clone()
        theta_delta = torch.remainder(theta - self._last_theta_rad + math.pi, 2.0 * math.pi) - math.pi
        theta_unwrapped = torch.where(state_was_valid, self._last_theta_rad + theta_delta, theta)
        first_contact = has_contact & ~state_was_valid
        start_center_mm = torch.where(first_contact[:, None], center_mm, self._start_center_mm)
        start_theta = torch.where(first_contact, theta_unwrapped, self._start_theta_rad)
        has_trajectory = has_contact & state_was_valid & bool(self.cfg.track_contact_center)

        marker_x = self._marker_uv[:, 0].to(device=depth_mm.device, dtype=depth_mm.dtype)
        marker_y = self._marker_uv[:, 1].to(device=depth_mm.device, dtype=depth_mm.dtype)
        start_center_px = torch.trunc(
            start_center_mm * float(self.cfg.mm2pix)
            + torch.tensor(
                (float(self.cfg.width) / 2.0, float(self.cfg.height) / 2.0),
                device=depth_mm.device,
                dtype=depth_mm.dtype,
            )
        )
        shear_px = torch.trunc((center_mm - start_center_mm) * float(self.cfg.mm2pix)).clamp(-10.0, 10.0)
        shear_g = torch.exp(
            -float(self.cfg.lamb[1])
            * (
                (marker_x[None] - start_center_px[:, 0:1]).square()
                + (marker_y[None] - start_center_px[:, 1:2]).square()
            )
        )
        x_shear = shear_px[:, 0:1] * shear_g
        y_shear = shear_px[:, 1:2] * shear_g

        current_center_px = torch.trunc(
            center_mm * float(self.cfg.mm2pix)
            + torch.tensor(
                (float(self.cfg.width) / 2.0, float(self.cfg.height) / 2.0),
                device=depth_mm.device,
                dtype=depth_mm.dtype,
            )
        )
        twist = (theta_unwrapped - start_theta).clamp(-math.pi / 3.0, math.pi / 3.0)
        offset_x = marker_x[None] - current_center_px[:, 0:1]
        offset_y = marker_y[None] - current_center_px[:, 1:2]
        twist_g = torch.exp(-float(self.cfg.lamb[2]) * (offset_x.square() + offset_y.square()))
        cos_theta = torch.cos(twist)[:, None]
        sin_theta = torch.sin(twist)[:, None]
        x_twist = (offset_x * (cos_theta - 1.0) - offset_y * sin_theta) * twist_g
        y_twist = (offset_x * sin_theta + offset_y * (cos_theta - 1.0)) * twist_g

        trajectory_mask = has_trajectory[:, None].to(dtype=depth_mm.dtype)
        displacement = torch.stack(
            (
                x_dilate + trajectory_mask * (x_shear + x_twist),
                y_dilate + trajectory_mask * (y_shear + y_twist),
            ),
            dim=-1,
        )
        displacement = torch.where(
            marker_response_active[:, None, None],
            displacement,
            torch.zeros_like(displacement),
        )
        displacement = torch.nan_to_num(displacement, nan=0.0, posinf=0.0, neginf=0.0)

        if bool(self.cfg.track_contact_center):
            self._start_center_mm.copy_(torch.where(first_contact[:, None], center_mm, self._start_center_mm))
            self._start_theta_rad.copy_(torch.where(first_contact, theta_unwrapped, self._start_theta_rad))
            self._last_theta_rad.copy_(torch.where(has_contact, theta_unwrapped, theta))
            self._state_valid.copy_(has_contact)
        else:
            self._state_valid.zero_()

        marker_uv = self._marker_uv.to(device=depth_mm.device, dtype=depth_mm.dtype)
        initial = marker_uv[None].expand(batch_size, -1, -1)
        flow = torch.stack((initial, initial + displacement), dim=1)
        theta_output_delta = torch.where(has_trajectory, theta_unwrapped - start_theta, torch.zeros_like(theta))
        self.last_output = FotsBatchOutput(
            marker_flow=flow,
            marker_displacement=displacement,
            max_depth_mm=depth_mm.amax(dim=(1, 2)),
            active_pixels=contact_mask.sum(dim=(1, 2)).to(dtype=torch.float32),
            active_markers=marker_contact.sum(dim=1).to(dtype=torch.float32),
            theta_rad=theta,
            theta_delta_rad=theta_output_delta,
        )
        return displacement


class _MarkerMotion:
    def __init__(self, cfg: RevoFotsCfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.contact: list[list[float]] = []

        marker_x_idx = np.linspace(cfg.marker_margin_x, cfg.width - cfg.marker_margin_x, cfg.marker_cols, dtype=int)
        marker_y_idx = np.linspace(cfg.marker_margin_y, cfg.height - cfg.marker_margin_y, cfg.marker_rows, dtype=int)
        marker_x_idx, marker_y_idx = np.meshgrid(marker_x_idx, marker_y_idx)
        marker_x_idx = np.clip(marker_x_idx.reshape([1, -1])[0], 0, cfg.width - 1).astype(np.int16)
        marker_y_idx = np.clip(marker_y_idx.reshape([1, -1])[0], 0, cfg.height - 1).astype(np.int16)

        self.init_marker_x_pos = torch.as_tensor(
            marker_x_idx.reshape([cfg.marker_rows, cfg.marker_cols]),
            dtype=torch.float32,
            device=self.device,
        )
        self.init_marker_y_pos = torch.as_tensor(
            marker_y_idx.reshape([cfg.marker_rows, cfg.marker_cols]),
            dtype=torch.float32,
            device=self.device,
        )

    def marker_flow(self) -> torch.Tensor:
        init_pos = torch.stack((self.init_marker_x_pos, self.init_marker_y_pos), dim=-1).reshape(-1, 2)
        current_pos = torch.stack((self.init_marker_x_pos, self.init_marker_y_pos), dim=-1).reshape(-1, 2)
        return torch.stack((init_pos, current_pos), dim=0).to(dtype=torch.float32)

    def _dilate(
        self,
        lamb: float,
        xx: torch.Tensor,
        yy: torch.Tensor,
        contact_x: torch.Tensor,
        contact_y: torch.Tensor,
        contact_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if contact_x.numel() <= 0:
            return torch.zeros_like(xx), torch.zeros_like(yy)
        dx_src = xx[..., None] - contact_x.reshape(1, 1, -1)
        dy_src = yy[..., None] - contact_y.reshape(1, 1, -1)
        g = torch.exp(-float(lamb) * (dx_src * dx_src + dy_src * dy_src))
        depth = contact_depth.reshape(1, 1, -1)
        return torch.sum(depth * dx_src * g, dim=-1), torch.sum(depth * dy_src * g, dim=-1)

    def _shear(
        self,
        center_x: int,
        center_y: int,
        lamb: float,
        shear_x: int,
        shear_y: int,
        xx: torch.Tensor,
        yy: torch.Tensor,
        shear_max: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.exp(-float(lamb) * ((xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2))
        shear_x = max(-float(shear_max), min(float(shear_x), float(shear_max)))
        shear_y = max(-float(shear_max), min(float(shear_y), float(shear_max)))
        return shear_x * g, shear_y * g

    def _twist(
        self,
        center_x: int,
        center_y: int,
        lamb: float,
        theta: float,
        xx: torch.Tensor,
        yy: torch.Tensor,
        theta_max_deg: float = 60,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = max(-theta_max_deg / 180.0 * math.pi, min(float(theta), theta_max_deg / 180.0 * math.pi))
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        offset_x = xx - float(center_x)
        offset_y = yy - float(center_y)
        g = torch.exp(-float(lamb) * (offset_x**2 + offset_y**2))
        # Return displacement from the original marker location, so theta=0
        # must produce zero twist motion.
        dx = offset_x * (torch.cos(theta_t) - 1.0) - offset_y * torch.sin(theta_t)
        dy = offset_x * torch.sin(theta_t) + offset_y * (torch.cos(theta_t) - 1.0)
        return dx * g, dy * g

    def marker_sim(self, depth_map_mm: torch.Tensor, contact_mask: torch.Tensor, traj: list[list[float]]) -> torch.Tensor:
        marker_x_pos = self.init_marker_x_pos
        marker_y_pos = self.init_marker_y_pos

        depth_map = depth_map_mm.to(device=self.device, dtype=torch.float32)
        depth_map = depth_map - torch.amin(depth_map)
        depth_map = depth_map / 10.0

        xs = torch.clamp(torch.round(marker_x_pos).to(torch.long), 0, int(self.cfg.width) - 1)
        ys = torch.clamp(torch.round(marker_y_pos).to(torch.long), 0, int(self.cfg.height) - 1)
        active = contact_mask[ys, xs].bool()
        if not bool(torch.any(active).detach().cpu()):
            return self.marker_flow()
        contact_x = marker_x_pos[active]
        contact_y = marker_y_pos[active]
        contact_depth = depth_map[ys[active], xs[active]]

        x_dd, y_dd = self._dilate(self.cfg.lamb[0], marker_x_pos, marker_y_pos, contact_x, contact_y, contact_depth)
        new_x_pos = marker_x_pos + x_dd
        new_y_pos = marker_y_pos + y_dd

        if len(traj) >= 2:
            x_ds, y_ds = self._shear(
                int(traj[0][0] * self.cfg.mm2pix + self.cfg.width / 2),
                int(traj[0][1] * self.cfg.mm2pix + self.cfg.height / 2),
                self.cfg.lamb[1],
                int((traj[-1][0] - traj[0][0]) * self.cfg.mm2pix),
                int((traj[-1][1] - traj[0][1]) * self.cfg.mm2pix),
                marker_x_pos,
                marker_y_pos,
            )
            new_x_pos += x_ds
            new_y_pos += y_ds

            x_dt, y_dt = self._twist(
                int(traj[-1][0] * self.cfg.mm2pix + self.cfg.width / 2),
                int(traj[-1][1] * self.cfg.mm2pix + self.cfg.height / 2),
                self.cfg.lamb[2],
                traj[-1][2] - traj[0][2],
                marker_x_pos,
                marker_y_pos,
            )
            new_x_pos += x_dt
            new_y_pos += y_dt

        init_pos = torch.stack((self.init_marker_x_pos, self.init_marker_y_pos), dim=-1).reshape(-1, 2)
        current_pos = torch.stack((new_x_pos, new_y_pos), dim=-1).reshape(-1, 2)
        return torch.stack((init_pos, current_pos), dim=0).to(dtype=torch.float32)


class RevoFotsAdapter:
    def __init__(self, cfg: RevoFotsCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._markers: list[_MarkerMotion] = []
        self._traj: list[list[list[float]]] = []
        self._marker_patch_array = _generate_marker_patch_array()

    def reset(self) -> None:
        self._traj.clear()

    def step(self, tacmap_raw_m: np.ndarray | torch.Tensor, theta_rad: np.ndarray | torch.Tensor | float | None = None) -> FotsOutput:
        raw = self._to_tensor(tacmap_raw_m)
        if raw.ndim == 2:
            raw = raw[None, ...]

        sensor_count = raw.shape[0]
        theta_values = self._theta_values(theta_rad, sensor_count)
        while len(self._markers) < sensor_count:
            self._markers.append(_MarkerMotion(self.cfg, self.device))
            self._traj.append([])

        flows = []
        images = []
        overlay_images = []
        depths = []
        masks = []
        max_depths = []
        active_pixels = []
        theta_deltas = []

        for i in range(sensor_count):
            depth_mm = self._prepare_depth_mm(raw[i])
            contact_mask = depth_mm > float(self.cfg.contact_threshold_mm)
            center = self._contact_center_mm(contact_mask)
            theta_i = float(theta_values[i].detach().cpu())
            if self.cfg.track_contact_center and center is not None:
                if self._traj[i]:
                    theta_i = self._unwrap_angle(theta_i, self._traj[i][-1][2])
                self._traj[i].append([center[0], center[1], theta_i])
                traj = self._traj[i]
            elif center is not None:
                traj = [[center[0], center[1], theta_i]]
            else:
                self._traj[i].clear()
                traj = []
            theta_deltas.append(float(traj[-1][2] - traj[0][2]) if len(traj) >= 2 else 0.0)

            flow = self._markers[i].marker_sim(depth_mm, contact_mask, traj)
            flows.append(flow)
            blank = np.zeros((int(self.cfg.height), int(self.cfg.width), 3), dtype=np.uint8)
            images.append(self.render_markers(flow, depth_mm) if self.cfg.render_marker_image else blank)
            overlay_images.append(self.render_marker_overlay(flow, depth_mm) if self.cfg.render_overlay_image else blank)
            depths.append(depth_mm)
            masks.append(contact_mask.to(torch.uint8))
            max_depths.append(torch.amax(depth_mm))
            active_pixels.append(torch.count_nonzero(contact_mask).to(torch.float32))

        return FotsOutput(
            marker_flow=torch.stack(flows, dim=0),
            marker_images=np.stack(images, axis=0),
            marker_overlay_images=np.stack(overlay_images, axis=0),
            depth_mm=torch.stack(depths, dim=0),
            contact_mask=torch.stack(masks, dim=0),
            max_depth_mm=torch.stack(max_depths).to(dtype=torch.float32),
            active_pixels=torch.stack(active_pixels).to(dtype=torch.float32),
            theta_rad=theta_values.to(dtype=torch.float32),
            theta_delta_rad=torch.as_tensor(theta_deltas, dtype=torch.float32, device=self.device),
        )

    def _to_tensor(self, value, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=self.device, dtype=dtype)
        else:
            tensor = torch.as_tensor(value, dtype=dtype, device=self.device)
        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _theta_values(self, theta_rad: np.ndarray | torch.Tensor | float | None, sensor_count: int) -> torch.Tensor:
        fallback = float(self.cfg.theta)
        if theta_rad is None:
            return torch.full((sensor_count,), fallback, dtype=torch.float32, device=self.device)

        theta = self._to_tensor(theta_rad).reshape(-1)
        if theta.numel() == 0:
            return torch.full((sensor_count,), fallback, dtype=torch.float32, device=self.device)
        if theta.numel() == 1 and sensor_count != 1:
            theta = theta[0].repeat(sensor_count)
        elif theta.numel() < sensor_count:
            pad = torch.full((sensor_count - theta.numel(),), fallback, dtype=torch.float32, device=self.device)
            theta = torch.cat((theta, pad), dim=0)
        else:
            theta = theta[:sensor_count]
        return torch.nan_to_num(theta, nan=fallback, posinf=fallback, neginf=fallback).to(dtype=torch.float32)

    @staticmethod
    def _unwrap_angle(angle: float, reference: float) -> float:
        delta = (float(angle) - float(reference) + np.pi) % (2.0 * np.pi) - np.pi
        return float(reference) + float(delta)

    def _prepare_depth_mm(self, depth_m: torch.Tensor) -> torch.Tensor:
        depth = torch.clamp(self._to_tensor(depth_m), min=0.0) * 1000.0 * float(self.cfg.depth_scale)
        if tuple(depth.shape) != (self.cfg.height, self.cfg.width):
            depth = torch.nn.functional.interpolate(
                depth.reshape(1, 1, int(depth.shape[0]), int(depth.shape[1])),
                size=(int(self.cfg.height), int(self.cfg.width)),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        return depth

    def _contact_center_mm(self, contact_mask: torch.Tensor) -> tuple[float, float] | None:
        ys, xs = torch.nonzero(contact_mask, as_tuple=True)
        if xs.numel() == 0:
            return None
        x_mm = (float(xs.to(torch.float32).mean().detach().cpu()) - self.cfg.width / 2.0) / float(self.cfg.mm2pix)
        y_mm = (float(ys.to(torch.float32).mean().detach().cpu()) - self.cfg.height / 2.0) / float(self.cfg.mm2pix)
        return x_mm, y_mm

    def render_markers(self, marker_flow: torch.Tensor, depth_mm: torch.Tensor | None = None) -> np.ndarray:
        img = np.zeros((int(self.cfg.height), int(self.cfg.width), 3), dtype=np.uint8)
        self._draw_flow_arrows(img, self._to_numpy(marker_flow).astype(np.float32))
        return img

    def render_marker_overlay(self, marker_flow: torch.Tensor, depth_mm: torch.Tensor | None = None) -> np.ndarray:
        img = np.zeros((int(self.cfg.height), int(self.cfg.width), 3), dtype=np.uint8)
        self._draw_flow_arrows(img, self._to_numpy(marker_flow).astype(np.float32))
        return img

    def render_marker_mask(self, marker_uv: np.ndarray, depth_mm: np.ndarray | None = None) -> np.ndarray:
        marker_uv_compensated = np.asarray(marker_uv, dtype=np.float32) + np.array([0.5, 0.5], dtype=np.float32)
        marker_image = np.ones((self.cfg.height + 24, self.cfg.width + 24), dtype=np.uint8) * 255
        marker_sizes = self._marker_sizes(marker_uv, depth_mm)

        if self._marker_patch_array is None:
            rgb = np.stack((marker_image, marker_image, marker_image), axis=-1)
            for uv, marker_size in zip(marker_uv_compensated, marker_sizes):
                _draw_square(rgb, uv + 12.0, radius=max(1, int(round(float(marker_size)))), color=(0, 0, 0))
            return rgb[12:-12, 12:-12, 0]

        patch_array_dict = self._marker_patch_array
        for uv, marker_size in zip(marker_uv_compensated, marker_sizes):
            u = float(uv[0]) + 12.0
            v = float(uv[1]) + 12.0
            patch_id_u = math.floor((u - math.floor(u)) * patch_array_dict["super_resolution_ratio"])
            patch_id_v = math.floor((v - math.floor(v)) * patch_array_dict["super_resolution_ratio"])
            patch_id_w = math.floor(
                (marker_size - patch_array_dict["base_circle_radius"]) * patch_array_dict["super_resolution_ratio"]
            )
            patch_id_w = int(np.clip(patch_id_w, 0, patch_array_dict["size_slot_num"] - 1))
            current_patch = patch_array_dict["patch_array"][patch_id_u, patch_id_v, patch_id_w]
            patch_coord_u = math.floor(u) - 6
            patch_coord_v = math.floor(v) - 6
            if marker_image.shape[1] - 12 > patch_coord_u >= 0 and marker_image.shape[0] - 12 > patch_coord_v >= 0:
                marker_image[patch_coord_v : patch_coord_v + 12, patch_coord_u : patch_coord_u + 12] = current_patch

        marker_image = marker_image[12:-12, 12:-12]
        return marker_image

    def _draw_flow_arrows(self, img: np.ndarray, marker_flow: np.ndarray) -> None:
        for start, end in zip(marker_flow[0], marker_flow[1]):
            visual_end = end + (end - start) * float(self.cfg.marker_arrow_scale)
            if float(np.linalg.norm(visual_end - start)) < 1.0e-6:
                continue
            if not _point_in_image(img, visual_end):
                continue
            _draw_arrow(img, start, visual_end, color=_MARKER_GREEN, thickness=2, tip_length=0.2)

    def _marker_sizes(self, marker_uv: np.ndarray, depth_mm: np.ndarray | None) -> np.ndarray:
        marker_uv = np.asarray(marker_uv, dtype=np.float32)
        sizes = np.full(marker_uv.shape[0], float(self.cfg.marker_size), dtype=np.float32)
        if depth_mm is None or float(np.max(depth_mm)) <= 0.0:
            return sizes

        xs = np.rint(marker_uv[:, 0]).astype(np.int32)
        ys = np.rint(marker_uv[:, 1]).astype(np.int32)
        valid = (0 <= xs) & (xs < self.cfg.width) & (0 <= ys) & (ys < self.cfg.height)
        local_depth = np.zeros(marker_uv.shape[0], dtype=np.float32)
        local_depth[valid] = depth_mm[ys[valid], xs[valid]]
        depth_norm = local_depth / self._depth_display_denominator(depth_mm)
        sizes = sizes + float(self.cfg.marker_size_depth_gain) * np.clip(depth_norm, 0.0, 1.0)
        return np.clip(sizes, float(self.cfg.marker_min_size), float(self.cfg.marker_max_size))

    def _depth_background(self, depth_mm: np.ndarray) -> np.ndarray:
        norm = np.clip(depth_mm / self._depth_display_denominator(depth_mm), 0.0, 1.0)
        gray = (245 - 105 * norm).astype(np.uint8)
        blue = np.clip(gray.astype(np.int16) + 6, 0, 255).astype(np.uint8)
        return np.stack((gray, gray, blue), axis=-1)

    def _depth_display_denominator(self, depth_mm: np.ndarray) -> float:
        fixed_max = float(self.cfg.depth_background_max_mm)
        if fixed_max > 0.0:
            return fixed_max + 1.0e-8
        return float(np.max(depth_mm)) + 1.0e-8


def _resize_float(img: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    except Exception:
        y_idx = np.linspace(0, img.shape[0] - 1, height).round().astype(int)
        x_idx = np.linspace(0, img.shape[1] - 1, width).round().astype(int)
        return img[np.ix_(y_idx, x_idx)].astype(np.float32)


def _draw_square(img: np.ndarray, point: np.ndarray, *, radius: int, color: tuple[int, int, int]) -> None:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    y0 = max(0, y - radius)
    y1 = min(img.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(img.shape[1], x + radius + 1)
    img[y0:y1, x0:x1] = color


def _point_in_image(img: np.ndarray, point: np.ndarray) -> bool:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    return 0 <= x < img.shape[1] and 0 <= y < img.shape[0]


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


def _generate_marker_patch_array(super_resolution_ratio: int = 10):
    try:
        import cv2
    except Exception:
        return None

    circle_radius = 3
    size_slot_num = 50
    base_circle_radius = 1.5
    patch_array = np.zeros(
        (
            super_resolution_ratio,
            super_resolution_ratio,
            size_slot_num,
            4 * circle_radius,
            4 * circle_radius,
        ),
        dtype=np.uint8,
    )
    for u in range(super_resolution_ratio):
        for v in range(super_resolution_ratio):
            for w in range(size_slot_num):
                img_highres = (
                    np.ones(
                        (
                            4 * circle_radius * super_resolution_ratio,
                            4 * circle_radius * super_resolution_ratio,
                        ),
                        dtype=np.uint8,
                    )
                    * 255
                )
                center = np.array(
                    [
                        circle_radius * super_resolution_ratio * 2,
                        circle_radius * super_resolution_ratio * 2,
                    ],
                    dtype=np.uint8,
                )
                center_offseted = center + np.array([u, v])
                radius = round(base_circle_radius * super_resolution_ratio + w)
                img_highres = cv2.circle(
                    img_highres,
                    tuple(center_offseted),
                    radius,
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                    lineType=cv2.LINE_AA,
                )
                img_highres = cv2.GaussianBlur(img_highres, (17, 17), 15)
                img_lowres = cv2.resize(
                    img_highres,
                    (4 * circle_radius, 4 * circle_radius),
                    interpolation=cv2.INTER_CUBIC,
                )
                patch_array[u, v, w, ...] = img_lowres

    return {
        "base_circle_radius": base_circle_radius,
        "circle_radius": circle_radius,
        "size_slot_num": size_slot_num,
        "patch_array": patch_array,
        "super_resolution_ratio": super_resolution_ratio,
    }
