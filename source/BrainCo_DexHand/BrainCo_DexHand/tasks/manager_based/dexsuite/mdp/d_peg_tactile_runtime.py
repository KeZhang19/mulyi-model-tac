"""Opt-in D-peg Taxim acceleration with unchanged calibration and RGB values."""

from functools import cached_property
import inspect
from pathlib import Path
from types import MethodType

import torch


def _where_deformation(self, height_map):
    pressing_depth_mm = -height_map.amin(-1).amin(-1)
    contact_mask = height_map < 0
    gel = self._TaximTorch__get_gel_map_cached(height_map.shape[1:])
    joined = torch.minimum(height_map, gel)
    mask = torch.logical_and(joined - gel < -pressing_depth_mm[..., None, None] * self.sim_params.contact_scale, contact_mask)
    blurred = joined
    for sigma in zip(*self.sim_params.deform_pyramid_sigma(height_map.shape[1:])):
        blurred = self._TaximTorch__gaussian_blur(blurred.unsqueeze(0), sigma)[0]
        # Preserve the FFT result's strides and subsequent FFT plan.
        torch.where(mask, joined, blurred, out=blurred)
    blurred = self._TaximTorch__gaussian_blur(blurred.unsqueeze(0), self.sim_params.deform_final_sigma(height_map.shape[1:]))[0]
    return blurred, mask


def _where_normals(self, height_map):
    height, width = height_map.shape[-2:]
    top = height_map[..., 0:height - 2, 1:width - 1]
    bottom = height_map[..., 2:height, 1:width - 1]
    left = height_map[..., 1:height - 1, 0:width - 2]
    right = height_map[..., 1:height - 1, 2:width]
    dx = (bottom - top) / 2.0
    dy = (right - left) / 2.0
    dx_norm = dx * height_map.shape[1] / self.height
    dy_norm = dy * height_map.shape[2] / self.width
    magnitude = torch.sqrt(dx_norm ** 2 + dy_norm ** 2)
    grad_magnitude = torch.arctan(magnitude)
    valid = magnitude != 0
    denominator = torch.where(valid, magnitude, torch.ones_like(magnitude))
    direction = torch.atan2(dx_norm / denominator, dy_norm / denominator)
    direction = torch.where(valid, direction, torch.zeros_like(direction))
    return (torch.nn.functional.pad(grad_magnitude, (1, 1, 1, 1), "replicate"),
            torch.nn.functional.pad(direction, (1, 1, 1, 1), "replicate"))


class DPegTaximRgbOutput:
    """Keep the adapter's result attributes; compute unused statistics on demand."""

    def __init__(self, rgb, height, depth):
        self.tactile_rgb = rgb
        self.taxim_height_map_mm = height
        self.depth_mm = depth

    @cached_property
    def max_depth_mm(self):
        return torch.amax(self.depth_mm, dim=(1, 2)).to(dtype=torch.float32)

    @cached_property
    def active_pixels(self):
        return torch.count_nonzero(self.depth_mm > 0.0, dim=(1, 2)).to(dtype=torch.float32)


class DPegFastTaximRgbAdapter:
    """Wrap only one D-peg adapter and its independently owned Taxim instance."""

    def __init__(self, adapter):
        from integrate.tacex_rgb_adapter import DEFAULT_TACEX_SIM_DIR
        taxim = adapter._taxim
        expected = (DEFAULT_TACEX_SIM_DIR / "taxim_torch.py").resolve()
        if Path(inspect.getfile(type(taxim))).resolve() != expected or type(taxim).__name__ != "TaximTorch":
            raise ValueError("D-peg fast Taxim requires the vendored, clipped [0,1] Torch renderer")
        if adapter.cfg.with_shadow:
            raise ValueError("D-peg fast Taxim is validated only with_shadow=False")
        self._adapter = adapter
        self.cfg = adapter.cfg
        self._taxim = taxim
        self._original_deformation = taxim._TaximTorch__compute_gel_pad_deformation
        self._original_normals = taxim._TaximTorch__generate_normals
        self._checked_real_contact = False
        self.first_contact_rgb_exact = None
        self._install_fast_methods()

    def _install_fast_methods(self):
        self._taxim._TaximTorch__compute_gel_pad_deformation = MethodType(_where_deformation, self._taxim)
        self._taxim._TaximTorch__generate_normals = MethodType(_where_normals, self._taxim)

    @torch.no_grad()
    def step(self, tacmap_raw_m):
        height, depth = self._adapter._prepare_taxim_height_map(tacmap_raw_m)
        # Background initialization can render zero before a real contact. Test
        # the first nonzero batch against the unmodified adapter, once only.
        verify = not self._checked_real_contact and bool(torch.any(depth > 0).cpu())
        if verify:
            self._taxim._TaximTorch__compute_gel_pad_deformation = self._original_deformation
            self._taxim._TaximTorch__generate_normals = self._original_normals
            try:
                reference = self._adapter.step(tacmap_raw_m).tactile_rgb
            finally:
                self._install_fast_methods()
        rgb = self._taxim.render_direct(height, with_shadow=False, press_depth=None, orig_hm_fmt=False)
        rgb = rgb.detach().permute(0, 2, 3, 1).to(self._adapter._device)
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
        rgb = torch.clamp(rgb * 255.0, 0.0, 255.0).to(torch.uint8)
        if verify:
            self.first_contact_rgb_exact = torch.equal(reference, rgb)
            if not self.first_contact_rgb_exact:
                raise RuntimeError("D-peg fast Taxim changed RGB on the first real depth batch")
            self._checked_real_contact = True
        return DPegTaximRgbOutput(rgb, height, depth)


def enable_d_peg_fast_taxim(adapter):
    """Idempotent installation; a calibrated-adapter replacement gets a fresh guard."""
    return adapter if isinstance(adapter, DPegFastTaximRgbAdapter) else DPegFastTaximRgbAdapter(adapter)
