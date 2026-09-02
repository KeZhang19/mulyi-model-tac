"""Denoising reconstruction objectives for the tri-modal cross autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .data import TactileNormalization
from .objectives import MODALITY_NAMES, degrade_tactile_inputs


@dataclass(frozen=True)
class TriModalAutoencoderLossCfg:
    """Balanced clean-target losses applied to all three decoder outputs."""

    rgb_weight: float = 1.0
    depth_weight: float = 1.0
    marker_weight: float = 1.0
    rgb_reference_weight: float = 0.75
    rgb_change_threshold: float = 0.02
    rgb_change_boost: float = 4.0
    depth_foreground_weight: float = 4.0
    depth_structure_weight: float = 0.25
    depth_foreground_threshold: float = 0.02
    depth_mask_temperature: float = 0.02
    depth_mask_pos_weight: float = 32.0
    marker_direction_weight: float = 0.1
    marker_direction_threshold_px: float = 0.5
    marker_prediction_floor_px: float = 0.25

    def __post_init__(self) -> None:
        non_negative = {
            name: float(value)
            for name, value in vars(self).items()
            if name not in {"depth_mask_temperature", "marker_prediction_floor_px"}
        }
        invalid = {name: value for name, value in non_negative.items() if value < 0.0}
        if invalid:
            raise ValueError(f"Loss settings must be non-negative: {invalid}")
        if sum((self.rgb_weight, self.depth_weight, self.marker_weight)) <= 0.0:
            raise ValueError("At least one modality loss weight must be positive")
        if not 0.0 <= float(self.rgb_reference_weight) <= 1.0:
            raise ValueError("rgb_reference_weight must be in [0, 1]")
        if float(self.depth_foreground_weight) < 1.0:
            raise ValueError("depth_foreground_weight must be at least 1")
        if float(self.depth_mask_temperature) <= 0.0:
            raise ValueError("depth_mask_temperature must be positive")
        if float(self.depth_mask_pos_weight) < 1.0:
            raise ValueError("depth_mask_pos_weight must be at least 1")
        if float(self.marker_prediction_floor_px) <= 0.0:
            raise ValueError("marker_prediction_floor_px must be positive")


def tri_modal_reconstruction_loss(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    normalization: TactileNormalization,
    cfg: TriModalAutoencoderLossCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare every decoded modality with its clean observation target."""

    rgb_error = (output["rgb_recon"] - batch["rgb"]).abs()
    rgb_global = rgb_error.mean(dim=(1, 2, 3))
    rgb_reference = batch.get("rgb_reference")
    rgb_changed = rgb_global
    rgb_reference_aware = rgb_global
    if rgb_reference is not None and float(cfg.rgb_reference_weight) > 0.0:
        change_strength = (batch["rgb"] - rgb_reference).abs().mean(dim=1)
        changed_mask = change_strength > float(cfg.rgb_change_threshold)
        changed_weight = changed_mask.to(rgb_error.dtype)
        strength = (
            change_strength / max(float(cfg.rgb_change_threshold), 1.0e-6)
        ).clamp(0.0, 1.0)
        pixel_weight = 1.0 + float(cfg.rgb_change_boost) * strength * changed_weight
        pixel_error = rgb_error.mean(dim=1)
        rgb_reference_aware = (
            (pixel_error * pixel_weight).sum(dim=(1, 2))
            / pixel_weight.sum(dim=(1, 2)).clamp_min(1.0)
        )
        rgb_changed = (
            (pixel_error * changed_weight).sum(dim=(1, 2))
            / changed_weight.sum(dim=(1, 2)).clamp_min(1.0)
        )
    reference_weight = float(cfg.rgb_reference_weight)
    rgb_per_sample = (
        (1.0 - reference_weight) * rgb_global
        + reference_weight * rgb_reference_aware
    )
    rgb_mse = (output["rgb_recon"] - batch["rgb"]).square().mean(dim=(1, 2, 3))

    depth_target = batch["depth"]
    depth_prediction = output["depth_recon"]
    foreground = depth_target > float(cfg.depth_foreground_threshold)
    depth_pixel_error = F.smooth_l1_loss(
        depth_prediction,
        depth_target,
        reduction="none",
        beta=0.02,
    )
    depth_weights = 1.0 + foreground.to(depth_pixel_error.dtype) * (
        float(cfg.depth_foreground_weight) - 1.0
    )
    depth_pixel_per_sample = (
        (depth_pixel_error * depth_weights).sum(dim=(1, 2, 3))
        / depth_weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
    )
    foreground_count = foreground.flatten(1).sum(dim=1).to(depth_pixel_error.dtype)
    background_count = (~foreground).flatten(1).sum(dim=1).to(depth_pixel_error.dtype)
    positive_weight = (background_count / foreground_count.clamp_min(1.0)).clamp(
        min=1.0,
        max=float(cfg.depth_mask_pos_weight),
    )
    depth_mask_logits = (
        depth_prediction - float(cfg.depth_foreground_threshold)
    ) / float(cfg.depth_mask_temperature)
    depth_structure_per_sample = F.binary_cross_entropy_with_logits(
        depth_mask_logits,
        foreground.to(depth_pixel_error.dtype),
        reduction="none",
        pos_weight=positive_weight[:, None, None, None],
    ).mean(dim=(1, 2, 3))
    depth_per_sample = depth_pixel_per_sample + (
        float(cfg.depth_structure_weight) * depth_structure_per_sample
    )
    depth_mae = (depth_prediction - depth_target).abs().mean(dim=(1, 2, 3))
    predicted_foreground = depth_prediction > float(cfg.depth_foreground_threshold)
    intersection = (predicted_foreground & foreground).flatten(1).sum(dim=1).to(
        depth_pixel_error.dtype
    )
    union = (predicted_foreground | foreground).flatten(1).sum(dim=1).to(
        depth_pixel_error.dtype
    )
    depth_iou = intersection / union.clamp_min(1.0)

    marker_target = batch["marker"][..., 2:4]
    marker_prediction = output["marker_recon"]
    marker_valid = batch["marker_valid"].to(torch.bool)
    marker_valid_weight = marker_valid.to(marker_prediction.dtype)
    marker_count = marker_valid.sum(dim=1).clamp_min(1).to(marker_prediction.dtype)
    marker_element_error = F.smooth_l1_loss(
        marker_prediction,
        marker_target,
        reduction="none",
        beta=0.1,
    ).mean(dim=-1)
    marker_motion_per_sample = (
        (marker_element_error * marker_valid_weight).sum(dim=1) / marker_count
    )

    marker_scale = float(normalization.marker_motion_scale_px)
    predicted_px = marker_prediction * marker_scale
    target_px = marker_target * marker_scale
    predicted_norm = torch.linalg.vector_norm(predicted_px, dim=-1)
    target_norm = torch.linalg.vector_norm(target_px, dim=-1)
    direction_mask = marker_valid & (
        target_norm > float(cfg.marker_direction_threshold_px)
    )
    target_unit = target_px / target_norm.clamp_min(1.0e-6).unsqueeze(-1)
    predicted_unit = predicted_px / predicted_norm.clamp_min(
        float(cfg.marker_prediction_floor_px)
    ).unsqueeze(-1)
    direction_cosine = (target_unit * predicted_unit).sum(dim=-1).clamp(-1.0, 1.0)
    direction_strength = (target_norm / max(marker_scale, 1.0e-6)).clamp(0.0, 1.0)
    direction_weights = direction_mask.to(marker_prediction.dtype) * direction_strength
    direction_per_sample = (
        ((1.0 - direction_cosine) * direction_weights).sum(dim=1)
        / direction_weights.sum(dim=1).clamp_min(1.0)
    )
    marker_per_sample = marker_motion_per_sample + (
        float(cfg.marker_direction_weight) * direction_per_sample
    )
    marker_epe = torch.linalg.vector_norm(predicted_px - target_px, dim=-1)
    marker_epe_per_sample = (
        (marker_epe * marker_valid_weight).sum(dim=1) / marker_count
    )
    direction_dot = (predicted_px * target_px).sum(dim=-1)
    direction_failure = (direction_dot <= 0.0) | (
        predicted_norm < float(cfg.marker_prediction_floor_px)
    )
    direction_count = direction_mask.sum(dim=1).clamp_min(1)
    direction_error_per_sample = (
        (direction_failure & direction_mask).sum(dim=1).to(marker_prediction.dtype)
        / direction_count.to(marker_prediction.dtype)
    )

    weight_sum = float(cfg.rgb_weight + cfg.depth_weight + cfg.marker_weight)
    total_per_sample = (
        float(cfg.rgb_weight) * rgb_per_sample
        + float(cfg.depth_weight) * depth_per_sample
        + float(cfg.marker_weight) * marker_per_sample
    ) / weight_sum
    total = total_per_sample.mean()
    return total, {
        "loss/total": total,
        "loss/rgb": rgb_per_sample.mean(),
        "loss/depth": depth_per_sample.mean(),
        "loss/depth_pixel": depth_pixel_per_sample.mean(),
        "loss/depth_structure": depth_structure_per_sample.mean(),
        "loss/marker": marker_per_sample.mean(),
        "loss/marker_motion": marker_motion_per_sample.mean(),
        "loss/marker_direction": direction_per_sample.mean(),
        "metric/rgb_mae": rgb_global.mean(),
        "metric/rgb_reference_weighted_mae": rgb_reference_aware.mean(),
        "metric/rgb_changed_mae": rgb_changed.mean(),
        "metric/rgb_psnr_db": (-10.0 * torch.log10(rgb_mse.clamp_min(1.0e-10))).mean(),
        "metric/depth_mae_normalized": depth_mae.mean(),
        "metric/depth_mae_mm": depth_mae.mean()
        * float(normalization.depth_scale_m)
        * 1000.0,
        "metric/depth_foreground_iou": depth_iou.mean(),
        "metric/marker_motion_epe_px": marker_epe_per_sample.mean(),
        "metric/marker_direction_error_rate": direction_error_per_sample.mean(),
    }


def compute_tri_modal_autoencoder_objective(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    normalization: TactileNormalization,
    cfg: TriModalAutoencoderLossCfg,
    generator: torch.Generator | None = None,
    degraded_modality: int | None = None,
    min_degradation_severity: float = 0.15,
    max_degradation_severity: float = 0.45,
    clean_probability: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Corrupt at most one input, then reconstruct all three clean targets."""

    degraded, flags, _, severity = degrade_tactile_inputs(
        batch,
        normalization=normalization,
        generator=generator,
        degraded_modality=degraded_modality,
        min_severity=min_degradation_severity,
        max_severity=max_degradation_severity,
        clean_probability=clean_probability,
    )
    output = model(
        rgb=degraded["rgb"],
        depth=degraded["depth"],
        marker=degraded["marker"],
        marker_valid_mask=degraded["marker_valid"],
    )
    loss, metrics = tri_modal_reconstruction_loss(output, batch, normalization, cfg)
    metrics.update(
        {
            "metric/clean_fraction": (~flags.any(dim=1)).to(batch["rgb"].dtype).mean(),
            "metric/degradation_severity": severity.mean(),
            **{
                f"metric/degraded_{name}_fraction": flags[:, index]
                .to(batch["rgb"].dtype)
                .mean()
                for index, name in enumerate(MODALITY_NAMES)
            },
        }
    )
    return loss, metrics
