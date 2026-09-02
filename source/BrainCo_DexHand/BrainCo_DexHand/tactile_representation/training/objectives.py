"""Single-modality observation degradation and cross-modal restoration objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .data import TactileNormalization


MODALITY_NAMES = ("rgb", "depth", "marker")


@dataclass(frozen=True)
class CrossModalLossCfg:
    """Weights for clean-target restoration, latent consistency, and quality estimation."""

    rgb_weight: float = 1.0
    # Keep a global RGB fidelity term, but emphasize pixels whose clean RGB
    # differs from the unpressed reference.  A zero value reproduces the old
    # uniformly averaged image loss.
    rgb_residual_loss_weight: float = 0.75
    rgb_change_threshold: float = 0.02
    rgb_change_boost: float = 4.0
    depth_weight: float = 4.0
    marker_weight: float = 2.0
    marker_motion_weight: float = 1.0
    # Explicitly penalize reversed arrows for markers with visible motion.
    # Smooth L1 alone can make a small sign error artificially cheap.
    marker_direction_weight: float = 0.2
    depth_foreground_weight: float = 4.0
    empty_depth_weight: float = 2.0
    # Depth is a sparse image.  The structural term prevents the pixel average
    # from being minimized by predicting a faint background everywhere.
    depth_structure_weight: float = 1.0
    depth_foreground_threshold: float = 0.02
    depth_mask_temperature: float = 0.02
    depth_mask_pos_weight: float = 32.0
    consistency_weight: float = 0.1
    quality_weight: float = 0.2
    # Preserve genuinely clean observations if the quality detector briefly
    # selects a modality for restoration during early training.
    clean_identity_weight: float = 1.0
    # Decode all three clean targets from the shared latent alone.  The model
    # withholds dynamic RGB/Depth/Marker spatial contexts on this path, making
    # the reconstruction gradient an explicit test of latent sufficiency.
    # Zero preserves objectives stored in schema-v6 and older checkpoints.
    latent_sufficiency_weight: float = 0.0

    def __post_init__(self) -> None:
        non_negative = {name: float(value) for name, value in vars(self).items()}
        invalid = {name: value for name, value in non_negative.items() if value < 0.0}
        if invalid:
            raise ValueError(f"Loss weights must be non-negative: {invalid}")
        if float(self.rgb_residual_loss_weight) > 1.0:
            raise ValueError("rgb_residual_loss_weight must be in [0, 1]")
        if not 0.0 <= float(self.depth_foreground_threshold) <= 1.0:
            raise ValueError("depth_foreground_threshold must be in [0, 1]")
        if float(self.depth_mask_temperature) <= 0.0:
            raise ValueError("depth_mask_temperature must be positive")
        if float(self.depth_mask_pos_weight) < 1.0:
            raise ValueError("depth_mask_pos_weight must be at least 1")


def sample_degraded_modality_indices(
    batch_size: int,
    device: torch.device | str,
    *,
    generator: torch.Generator | None = None,
    degraded_modality: int | None = None,
) -> torch.Tensor:
    """Return a shuffled, count-balanced degraded-modality index for every row."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if degraded_modality is not None:
        if not 0 <= int(degraded_modality) < len(MODALITY_NAMES):
            raise ValueError(f"degraded_modality must be in [0, {len(MODALITY_NAMES) - 1}]")
        return torch.full(
            (int(batch_size),), int(degraded_modality), device=device, dtype=torch.int64
        )
    indices = torch.arange(int(batch_size), device=device, dtype=torch.int64)
    indices.remainder_(len(MODALITY_NAMES))
    permutation = torch.randperm(int(batch_size), device=device, generator=generator)
    return indices.index_select(0, permutation)


def _rand(
    shape: tuple[int, ...],
    reference: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.rand(
        shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _randn_like(reference: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def degrade_tactile_inputs(
    batch: dict[str, torch.Tensor],
    *,
    normalization: TactileNormalization,
    generator: torch.Generator | None = None,
    degraded_modality: int | None = None,
    min_severity: float = 0.15,
    max_severity: float = 0.45,
    clean_probability: float = 0.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mildly degrade at most one observation while retaining all input tensors.

    With ``clean_probability > 0``, some rows remain fully clean and receive
    an all-false degradation flag plus quality target ``[1,1,1]``.  Other rows
    receive exactly one degraded modality.  Static marker calibration
    coordinates x0/y0 are never corrupted.
    """

    if not 0.0 <= float(min_severity) <= float(max_severity) <= 1.0:
        raise ValueError("severity bounds must satisfy 0 <= min_severity <= max_severity <= 1")
    if not 0.0 <= float(clean_probability) <= 1.0:
        raise ValueError("clean_probability must be in [0, 1]")
    rgb, depth, marker = batch["rgb"], batch["depth"], batch["marker"]
    marker_valid = batch["marker_valid"]
    batch_size = int(rgb.shape[0])
    degraded_indices = sample_degraded_modality_indices(
        batch_size,
        rgb.device,
        generator=generator,
        degraded_modality=degraded_modality,
    )
    degradation_flags = F.one_hot(
        degraded_indices, num_classes=len(MODALITY_NAMES)
    ).to(torch.bool)
    severity = float(min_severity) + (
        float(max_severity) - float(min_severity)
    ) * _rand((batch_size,), rgb, generator)
    quality_target = torch.ones(
        (batch_size, len(MODALITY_NAMES)), device=rgb.device, dtype=rgb.dtype
    )
    quality_target.scatter_(1, degraded_indices[:, None], (1.0 - severity)[:, None])
    if float(clean_probability) > 0.0:
        clean_rows = _rand((batch_size,), rgb, generator) < float(clean_probability)
        degradation_flags = degradation_flags & ~clean_rows[:, None]
        severity = severity.masked_fill(clean_rows, 0.0)
        quality_target = torch.where(
            clean_rows[:, None], torch.ones_like(quality_target), quality_target
        )

    rgb_severity = severity[:, None, None, None]
    rgb_gain_direction = _rand((batch_size, rgb.shape[1], 1, 1), rgb, generator) * 2.0 - 1.0
    rgb_bias_direction = _rand((batch_size, rgb.shape[1], 1, 1), rgb, generator) * 2.0 - 1.0
    # Mild but visible illumination drift: roughly 5--10% gain, 1--2% bias,
    # and 0.5--1% pixel noise over the default severity interval.
    rgb_gain = 1.0 + rgb_gain_direction * (0.03 + 0.15 * rgb_severity)
    rgb_bias = rgb_bias_direction * (0.005 + 0.035 * rgb_severity)
    rgb_noise = _randn_like(rgb, generator) * (0.003 + 0.015 * rgb_severity)
    # A smooth, channel-dependent color patch models local illumination or
    # reflection changes without creating a hard synthetic rectangle.
    local_x = torch.linspace(
        -1.0, 1.0, rgb.shape[-1], device=rgb.device, dtype=rgb.dtype
    ).view(1, 1, 1, rgb.shape[-1])
    local_y = torch.linspace(
        -1.0, 1.0, rgb.shape[-2], device=rgb.device, dtype=rgb.dtype
    ).view(1, 1, rgb.shape[-2], 1)
    local_center_x = _rand((batch_size, 1, 1, 1), rgb, generator) * 1.2 - 0.6
    local_center_y = _rand((batch_size, 1, 1, 1), rgb, generator) * 1.2 - 0.6
    local_radius = 0.18 + 0.17 * _rand((batch_size, 1, 1, 1), rgb, generator)
    local_mask = torch.exp(
        -0.5
        * (
            (local_x - local_center_x).square()
            + (local_y - local_center_y).square()
        )
        / local_radius.square()
    )
    local_color_direction = (
        _rand((batch_size, rgb.shape[1], 1, 1), rgb, generator) * 2.0 - 1.0
    )
    local_color_direction = local_color_direction - local_color_direction.mean(
        dim=1, keepdim=True
    )
    local_color_direction = local_color_direction / local_color_direction.abs().amax(
        dim=1, keepdim=True
    ).clamp_min(0.25)
    local_color_shift = (
        local_mask * local_color_direction * (0.04 + 0.10 * rgb_severity)
    )
    rgb_candidate = (
        rgb * rgb_gain + rgb_bias + rgb_noise + local_color_shift
    ).clamp(0.0, 1.0)
    degraded_rgb = torch.where(
        degradation_flags[:, 0, None, None, None], rgb_candidate, rgb
    )

    depth_severity = severity[:, None, None, None].to(depth.dtype)
    depth_gain_direction = _rand((batch_size, 1, 1, 1), depth, generator) * 2.0 - 1.0
    depth_bias_direction = _rand((batch_size, 1, 1, 1), depth, generator) * 2.0 - 1.0
    # At the default 3 mm normalization scale the additive terms correspond
    # to roughly 0.01--0.03 mm, with only 1--3% sparse invalid pixels.
    depth_gain = 1.0 + depth_gain_direction * (0.025 + 0.10 * depth_severity)
    depth_bias = depth_bias_direction * (0.004 + 0.012 * depth_severity)
    depth_noise = _randn_like(depth, generator) * (0.002 + depth_severity * 0.008)
    holes = _rand(tuple(depth.shape), depth, generator) < (
        0.005 + depth_severity * 0.04
    )
    depth_candidate = (depth * depth_gain + depth_bias + depth_noise).clamp(0.0, 1.0)
    depth_candidate = depth_candidate.masked_fill(holes, 0.0)
    degraded_depth = torch.where(
        degradation_flags[:, 1, None, None, None], depth_candidate, depth
    )

    marker_severity = severity[:, None, None].to(marker.dtype)
    marker_motion_scale = marker.new_tensor(
        (1.0 / float(normalization.marker_motion_scale_px),) * 2
    ).view(1, 1, 2)
    marker_candidate = marker.clone()
    marker_candidate[..., 2:4] = marker_candidate[..., 2:4] + _randn_like(
        marker_candidate[..., 2:4], generator
    ) * marker_motion_scale * (1.00 + 5.00 * marker_severity)
    # Marker calibration anchors and validity are sensor-layout properties.
    # Only the measured motion vector is degraded.
    degraded_marker_valid = marker_valid
    marker_candidate[..., 4] = degraded_marker_valid.to(marker.dtype)
    marker_candidate[..., 2:4].masked_fill_(~degraded_marker_valid.unsqueeze(-1), 0.0)
    degraded_marker = torch.where(
        degradation_flags[:, 2, None, None], marker_candidate, marker
    )

    return (
        {
            "rgb": degraded_rgb,
            "depth": degraded_depth,
            "marker": degraded_marker,
            "marker_valid": degraded_marker_valid,
        },
        degradation_flags,
        quality_target,
        severity,
    )


def _selected_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    weights = selected.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def restoration_loss(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    degradation_flags: torch.Tensor,
    normalization: TactileNormalization,
    cfg: CrossModalLossCfg,
    rgb_reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruct the clean targets selected for each sample.

    Restoration training selects at most one degraded modality per row.  The
    latent-sufficiency path selects all three; both cases use the same physical
    modality losses so their scales stay directly comparable.
    """

    batch_size = int(batch["rgb"].shape[0])
    if tuple(degradation_flags.shape) != (batch_size, len(MODALITY_NAMES)):
        raise ValueError(
            f"degradation_flags must have shape {(batch_size, len(MODALITY_NAMES))}"
        )
    selected_count = degradation_flags.sum(dim=1)

    rgb_absolute_error = (output["rgb_recon"] - batch["rgb"]).abs()
    rgb_global_error = rgb_absolute_error.mean(dim=(1, 2, 3))
    rgb_residual_error = rgb_global_error
    rgb_change_error = rgb_global_error
    if rgb_reference is not None and float(cfg.rgb_residual_loss_weight) > 0.0:
        expected_rgb = tuple(batch["rgb"].shape)
        if tuple(rgb_reference.shape) != expected_rgb:
            raise ValueError(
                f"rgb_reference must have shape {expected_rgb}, got {tuple(rgb_reference.shape)}"
            )
        # The decoder is reference-conditioned, so this is the same error in
        # residual coordinates: (R_hat-R_ref) - (R_clean-R_ref).  The explicit
        # form makes the intended objective auditable and lets us weight the
        # contact-induced change region instead of changing the target.
        target_residual = batch["rgb"] - rgb_reference
        predicted_residual = output["rgb_recon"] - rgb_reference
        residual_absolute_error = (predicted_residual - target_residual).abs()
        residual_pixel_error = residual_absolute_error.mean(dim=1)
        change_strength = target_residual.abs().mean(dim=1)
        change_fraction = (change_strength > float(cfg.rgb_change_threshold)).to(
            residual_pixel_error.dtype
        )
        # A soft strength term avoids a discontinuous target while the binary
        # threshold prevents tiny sensor quantization noise from being boosted.
        strength = (
            change_strength / max(float(cfg.rgb_change_threshold), 1.0e-6)
        ).clamp(0.0, 1.0) * change_fraction
        pixel_weights = 1.0 + float(cfg.rgb_change_boost) * strength
        rgb_residual_error = (
            (residual_pixel_error * pixel_weights).sum(dim=(1, 2))
            / pixel_weights.sum(dim=(1, 2)).clamp_min(1.0)
        )
        rgb_change_error = (
            (residual_pixel_error * change_fraction).sum(dim=(1, 2))
            / change_fraction.sum(dim=(1, 2)).clamp_min(1.0)
        )
        residual_weight = float(cfg.rgb_residual_loss_weight)
        rgb_error = (
            (1.0 - residual_weight) * rgb_global_error
            + residual_weight * rgb_residual_error
        )
    else:
        rgb_error = rgb_global_error
    rgb_mse = (output["rgb_recon"] - batch["rgb"]).square().mean(dim=(1, 2, 3))

    depth_error = F.smooth_l1_loss(
        output["depth_recon"], batch["depth"], reduction="none", beta=0.02
    )
    foreground = batch["depth"] > float(cfg.depth_foreground_threshold)
    depth_weights = 1.0 + foreground.to(depth_error.dtype) * float(cfg.depth_foreground_weight)
    depth_per_sample = (depth_error * depth_weights).sum(dim=(1, 2, 3)) / depth_weights.sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)
    contact = batch.get("contact")
    if contact is None:
        contact = foreground.flatten(1).any(dim=1)
    no_contact = ~contact.to(device=depth_error.device, dtype=torch.bool)
    empty_depth_per_sample = output["depth_recon"].abs().mean(dim=(1, 2, 3))
    depth_per_sample = depth_per_sample + (
        float(cfg.empty_depth_weight)
        * empty_depth_per_sample
        * no_contact.to(empty_depth_per_sample.dtype)
    )
    # Treat foreground recovery as a separate task.  A differentiable mask is
    # derived from the predicted depth so this remains checkpoint-compatible
    # and does not add a second decoder head.  Positive weighting is computed
    # per sample and capped because only a few percent of pixels are foreground.
    depth_mask_logits = (
        output["depth_recon"] - float(cfg.depth_foreground_threshold)
    ) / float(cfg.depth_mask_temperature)
    foreground_count = foreground.flatten(1).sum(dim=1).to(depth_error.dtype)
    background_count = (~foreground).flatten(1).sum(dim=1).to(depth_error.dtype)
    dynamic_pos_weight = (
        background_count / foreground_count.clamp_min(1.0)
    ).clamp(min=1.0, max=float(cfg.depth_mask_pos_weight))
    depth_mask_loss = F.binary_cross_entropy_with_logits(
        depth_mask_logits,
        foreground.to(depth_error.dtype),
        reduction="none",
        pos_weight=dynamic_pos_weight[:, None, None, None],
    ).mean(dim=(1, 2, 3))
    depth_per_sample = depth_per_sample + float(cfg.depth_structure_weight) * depth_mask_loss
    depth_mae = (output["depth_recon"] - batch["depth"]).abs().mean(dim=(1, 2, 3))
    depth_abs_error = (output["depth_recon"] - batch["depth"]).abs()
    foreground_mae = (
        (depth_abs_error * foreground.to(depth_abs_error.dtype)).sum(dim=(1, 2, 3))
        / foreground_count.clamp_min(1.0)
    )
    background_count_full = (~foreground).flatten(1).sum(dim=1).to(depth_error.dtype)
    background_mae = (
        (depth_abs_error * (~foreground).to(depth_abs_error.dtype)).sum(dim=(1, 2, 3))
        / background_count_full.clamp_min(1.0)
    )
    predicted_foreground = output["depth_recon"] > float(cfg.depth_foreground_threshold)
    intersection = (predicted_foreground & foreground).flatten(1).sum(dim=1).to(depth_error.dtype)
    union = (predicted_foreground | foreground).flatten(1).sum(dim=1).to(depth_error.dtype)
    depth_mask_iou = intersection / union.clamp_min(1.0)

    valid = batch["marker_valid"]
    motion_error = F.smooth_l1_loss(
        output["marker_recon"],
        batch["marker"][..., 2:4],
        reduction="none",
        beta=1.0,
    ).mean(dim=-1)
    valid_weight = valid.to(motion_error.dtype)
    valid_count_per_sample = valid.sum(dim=1).clamp_min(1).to(motion_error.dtype)
    marker_per_sample = (
        (motion_error * valid_weight).sum(dim=1) / valid_count_per_sample
    ) * float(cfg.marker_motion_weight)
    marker_motion_scale = batch["marker"].new_full(
        (1, 1, 2), float(normalization.marker_motion_scale_px)
    )
    marker_epe = torch.linalg.vector_norm(
        (output["marker_recon"] - batch["marker"][..., 2:4]) * marker_motion_scale,
        dim=-1,
    )
    marker_epe_per_sample = (marker_epe * valid_weight).sum(dim=1) / valid_count_per_sample
    target_motion_px = batch["marker"][..., 2:4] * marker_motion_scale
    predicted_motion_px = output["marker_recon"] * marker_motion_scale
    target_norm = torch.linalg.vector_norm(target_motion_px, dim=-1)
    predicted_norm = torch.linalg.vector_norm(predicted_motion_px, dim=-1)
    direction_target = valid & (target_norm > 0.5)
    direction_valid = direction_target & (predicted_norm > 1.0e-6)
    direction_cosine = F.cosine_similarity(
        predicted_motion_px, target_motion_px, dim=-1, eps=1.0e-6
    )
    direction_loss_per_marker = (1.0 - direction_cosine).clamp(0.0, 2.0)
    direction_target_count = direction_target.sum(dim=1)
    direction_loss_per_sample = (
        direction_loss_per_marker
        * direction_target.to(direction_loss_per_marker.dtype)
    ).sum(dim=1) / direction_target_count.clamp_min(1).to(direction_loss_per_marker.dtype)
    marker_per_sample = marker_per_sample + (
        float(cfg.marker_direction_weight) * direction_loss_per_sample
    )
    direction_dot = (target_motion_px * predicted_motion_px).sum(dim=-1)
    direction_wrong = direction_dot < 0.0
    direction_count = direction_valid.sum(dim=1)
    direction_error_per_sample = (
        (direction_wrong & direction_valid).sum(dim=1).to(marker_epe.dtype)
        / direction_count.clamp_min(1).to(marker_epe.dtype)
    )
    has_direction_target = direction_count > 0

    selected_rgb = degradation_flags[:, 0]
    selected_depth = degradation_flags[:, 1]
    selected_marker = degradation_flags[:, 2]
    weighted_per_sample = (
        float(cfg.rgb_weight) * rgb_error * selected_rgb.to(rgb_error.dtype)
        + float(cfg.depth_weight) * depth_per_sample * selected_depth.to(depth_per_sample.dtype)
        + float(cfg.marker_weight)
        * marker_per_sample
        * selected_marker.to(marker_per_sample.dtype)
    )
    selected_rows = selected_count > 0
    restoration = weighted_per_sample.sum() / selected_rows.sum().clamp_min(1).to(
        weighted_per_sample.dtype
    )

    selected_rgb_mse = _selected_mean(rgb_mse, selected_rgb)
    return restoration, {
        "loss/restoration": restoration,
        "loss/restore_rgb": _selected_mean(rgb_error, selected_rgb),
        "loss/restore_depth": _selected_mean(depth_per_sample, selected_depth),
        "loss/restore_depth_empty": _selected_mean(
            empty_depth_per_sample, selected_depth & no_contact
        ),
        "loss/restore_depth_structure": _selected_mean(depth_mask_loss, selected_depth),
        "loss/restore_marker": _selected_mean(marker_per_sample, selected_marker),
        "loss/restore_marker_direction": _selected_mean(
            direction_loss_per_sample, selected_marker & (direction_target_count > 0)
        ),
        "metric/rgb_mae": _selected_mean(rgb_global_error, selected_rgb),
        "metric/rgb_residual_mae": _selected_mean(rgb_residual_error, selected_rgb),
        "metric/rgb_change_mae": _selected_mean(rgb_change_error, selected_rgb),
        "metric/rgb_psnr_db": -10.0 * torch.log10(selected_rgb_mse.clamp_min(1.0e-10)),
        "metric/depth_mae_normalized": _selected_mean(depth_mae, selected_depth),
        "metric/depth_mae_mm": _selected_mean(depth_mae, selected_depth)
        * float(normalization.depth_scale_m)
        * 1000.0,
        "metric/depth_foreground_mae": _selected_mean(foreground_mae, selected_depth),
        "metric/depth_background_mae": _selected_mean(background_mae, selected_depth),
        "metric/depth_foreground_iou": _selected_mean(depth_mask_iou, selected_depth),
        "metric/marker_motion_epe_px": _selected_mean(marker_epe_per_sample, selected_marker),
        "metric/marker_direction_error_rate": _selected_mean(
            direction_error_per_sample, selected_marker & has_direction_target
        ),
    }


def clean_identity_loss(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    clean_rows: torch.Tensor,
    normalization: TactileNormalization,
    cfg: CrossModalLossCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize any change made to a genuinely clean three-modality input."""

    batch_size = int(batch["rgb"].shape[0])
    if tuple(clean_rows.shape) != (batch_size,):
        raise ValueError(f"clean_rows must have shape {(batch_size,)}")
    clean_rows = clean_rows.to(device=batch["rgb"].device, dtype=torch.bool)
    rgb_drift = (output["restored_rgb"] - batch["rgb"]).abs().mean(dim=(1, 2, 3))
    depth_drift = (output["restored_depth"] - batch["depth"]).abs().mean(
        dim=(1, 2, 3)
    )
    valid = batch["marker_valid"]
    valid_weight = valid.to(batch["marker"].dtype)
    marker_drift_px = torch.linalg.vector_norm(
        (output["restored_marker_motion"] - batch["marker"][..., 2:4])
        * float(normalization.marker_motion_scale_px),
        dim=-1,
    )
    marker_drift_px_per_sample = (
        marker_drift_px * valid_weight
    ).sum(dim=1) / valid_weight.sum(dim=1).clamp_min(1.0)
    marker_drift = marker_drift_px_per_sample / float(
        normalization.marker_motion_scale_px
    )
    per_sample = (
        float(cfg.rgb_weight) * rgb_drift
        + float(cfg.depth_weight) * depth_drift
        + float(cfg.marker_weight) * marker_drift
    )
    identity = _selected_mean(per_sample, clean_rows)
    return identity, {
        "loss/clean_identity": identity,
        "metric/clean_rgb_drift": _selected_mean(rgb_drift, clean_rows),
        "metric/clean_depth_drift_mm": _selected_mean(depth_drift, clean_rows)
        * float(normalization.depth_scale_m)
        * 1000.0,
        "metric/clean_marker_drift_px": _selected_mean(
            marker_drift_px_per_sample, clean_rows
        ),
    }


def _latent_distance(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def _stack_reliability(output: dict[str, Any]) -> torch.Tensor:
    return torch.stack([output["reliability"][name] for name in MODALITY_NAMES], dim=1)


def compute_cross_modal_objective(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    teacher_model: nn.Module | None = None,
    normalization: TactileNormalization,
    cfg: CrossModalLossCfg,
    degraded_modality: int | None = None,
    min_degradation_severity: float = 0.15,
    max_degradation_severity: float = 0.45,
    clean_probability: float = 0.0,
    generator: torch.Generator | None = None,
    use_predicted_restoration: bool | None = None,
    restoration_quality_threshold: float | None = None,
    report_oracle_upper_bound: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train restoration without exposing the degradation label to the model.

    In predicted mode, ground-truth degradation flags are used only for quality
    supervision and for selecting the clean target loss.  Fusion and output
    restoration depend solely on predicted qualities.  The legacy oracle-mask
    branch remains available for loading schema-v4 checkpoints.
    """

    batch_size = int(batch["rgb"].shape[0])
    device = batch["rgb"].device
    clean_inputs = {
        "rgb": batch["rgb"],
        "depth": batch["depth"],
        "marker": batch["marker"],
        "marker_valid_mask": batch["marker_valid"],
    }
    degraded, degradation_flags, quality_target, severity = degrade_tactile_inputs(
        batch,
        normalization=normalization,
        generator=generator,
        degraded_modality=degraded_modality,
        min_severity=min_degradation_severity,
        max_severity=max_degradation_severity,
        clean_probability=clean_probability,
    )
    full_mask = torch.ones((batch_size, len(MODALITY_NAMES)), device=device, dtype=torch.bool)
    degraded_inputs = {
        "rgb": degraded["rgb"],
        "rgb_reference": batch.get("rgb_reference"),
        "depth": degraded["depth"],
        "marker": degraded["marker"],
        "marker_valid_mask": degraded["marker_valid"],
    }
    rgb_reference_for_loss = (
        batch.get("rgb_reference")
        if bool(getattr(getattr(model, "cfg", None), "rgb_reference_residual", False))
        else None
    )

    if use_predicted_restoration is None:
        model_cfg = getattr(model, "cfg", None)
        use_predicted_restoration = bool(
            getattr(model_cfg, "predicted_soft_restoration", False)
        )

    if use_predicted_restoration:
        model_cfg = getattr(model, "cfg", None)
        if restoration_quality_threshold is None:
            restoration_quality_threshold = float(
                getattr(model_cfg, "reliability_gate_threshold", 0.9)
            )
        restoration_output = model.restore_degraded_observation(
            rgb=degraded_inputs["rgb"],
            rgb_reference=degraded_inputs["rgb_reference"],
            depth=degraded_inputs["depth"],
            marker=degraded_inputs["marker"],
            marker_valid_mask=degraded_inputs["marker_valid_mask"],
            quality_threshold=restoration_quality_threshold,
        )
        candidate_restoration, candidate_metrics = restoration_loss(
            restoration_output,
            batch,
            degradation_flags,
            normalization,
            cfg,
            rgb_reference=rgb_reference_for_loss,
        )
        end_to_end_output = {
            "rgb_recon": restoration_output["restored_rgb"],
            "depth_recon": restoration_output["restored_depth"],
            "marker_recon": restoration_output["restored_marker_motion"],
        }
        end_to_end_restoration, metrics = restoration_loss(
            end_to_end_output,
            batch,
            degradation_flags,
            normalization,
            cfg,
            rgb_reference=rgb_reference_for_loss,
        )
        metrics["loss/candidate_restoration"] = candidate_restoration
        metrics["loss/end_to_end_restoration"] = end_to_end_restoration
        for name, value in candidate_metrics.items():
            if name != "loss/restoration":
                metrics[f"candidate/{name}"] = value
        detection = restoration_output
        restoration_mask = full_mask
        optimization_restoration = candidate_restoration

        if report_oracle_upper_bound:
            with torch.no_grad():
                oracle_output = model(
                    **degraded_inputs, modality_mask=~degradation_flags
                )
                oracle_restoration, _ = restoration_loss(
                    oracle_output,
                    batch,
                    degradation_flags,
                    normalization,
                    cfg,
                    rgb_reference=rgb_reference_for_loss,
                )
            metrics["loss/oracle_upper_bound_restoration"] = oracle_restoration
    else:
        # Schema-v4 compatibility only: this is intentionally not the default
        # for new training runs.
        detection = model.encode_with_diagnostics(
            rgb=degraded_inputs["rgb"],
            depth=degraded_inputs["depth"],
            marker=degraded_inputs["marker"],
            marker_valid_mask=degraded_inputs["marker_valid_mask"],
            modality_mask=full_mask,
        )
        restoration_mask = ~degradation_flags
        restoration_output = model(**degraded_inputs, modality_mask=restoration_mask)
        optimization_restoration, metrics = restoration_loss(
            restoration_output,
            batch,
            degradation_flags,
            normalization,
            cfg,
            rgb_reference=rgb_reference_for_loss,
        )

    degraded_rows = degradation_flags.any(dim=1)
    clean_rows = ~degraded_rows
    if use_predicted_restoration:
        clean_identity, clean_metrics = clean_identity_loss(
            restoration_output,
            batch,
            clean_rows,
            normalization,
            cfg,
        )
    else:
        clean_identity = restoration_output["latent"].sum() * 0.0
        clean_metrics = {
            "loss/clean_identity": clean_identity,
            "metric/clean_rgb_drift": clean_identity,
            "metric/clean_depth_drift_mm": clean_identity,
            "metric/clean_marker_drift_px": clean_identity,
        }
    metrics.update(clean_metrics)

    latent_sufficiency = restoration_output["latent"].sum() * 0.0
    if float(cfg.latent_sufficiency_weight) > 0.0:
        decode_from_latent = getattr(model, "decode_from_latent", None)
        if decode_from_latent is None:
            raise TypeError(
                "latent_sufficiency_weight requires model.decode_from_latent()"
            )
        latent_only_output = decode_from_latent(
            restoration_output["latent"],
            rgb_reference=batch.get("rgb_reference"),
            marker_positions=batch["marker"][..., :2],
            marker_valid_mask=batch["marker_valid"],
        )
        all_clean_targets = torch.ones_like(degradation_flags, dtype=torch.bool)
        latent_sufficiency, latent_metrics = restoration_loss(
            latent_only_output,
            batch,
            all_clean_targets,
            normalization,
            cfg,
            rgb_reference=rgb_reference_for_loss,
        )
        for name, value in latent_metrics.items():
            if name != "loss/restoration":
                metrics[f"latent/{name}"] = value
    metrics["loss/latent_sufficiency"] = latent_sufficiency
    metrics["loss/latent_sufficiency_weighted"] = (
        float(cfg.latent_sufficiency_weight) * latent_sufficiency
    )

    teacher = model if teacher_model is None else teacher_model
    with torch.no_grad():
        clean_latent = teacher.encode(**clean_inputs, modality_mask=full_mask)
    consistency = _latent_distance(restoration_output["latent"], clean_latent)

    predicted_quality = _stack_reliability(detection)
    quality_loss = F.smooth_l1_loss(
        predicted_quality, quality_target, reduction="mean", beta=0.1
    )
    quality_mae = (predicted_quality - quality_target).abs().mean()
    degraded_indices = degradation_flags.to(torch.int64).argmax(dim=1)
    detected_indices = predicted_quality.argmin(dim=1)
    detection_accuracy = _selected_mean(
        (detected_indices == degraded_indices).to(torch.float32), degraded_rows
    )
    row_quality_mae = (predicted_quality - quality_target).abs().mean(dim=1)
    clean_min_quality = predicted_quality.min(dim=1).values
    restoration_applied = (
        restoration_output["restoration_applied"].to(torch.float32)
        if use_predicted_restoration
        else degraded_rows.to(torch.float32)
    )

    total = (
        optimization_restoration
        + float(cfg.clean_identity_weight) * clean_identity
        + float(cfg.latent_sufficiency_weight) * latent_sufficiency
        + float(cfg.consistency_weight) * consistency
        + float(cfg.quality_weight) * quality_loss
    )
    metrics.update(
        {
            "loss/total": total,
            "loss/consistency": consistency,
            "loss/quality": quality_loss,
            "metric/quality_mae": quality_mae,
            "metric/degraded_quality_mae": _selected_mean(
                row_quality_mae, degraded_rows
            ),
            "metric/clean_quality_mae": _selected_mean(row_quality_mae, clean_rows),
            "metric/degraded_modality_accuracy": detection_accuracy,
            "metric/degradation_severity": _selected_mean(severity, degraded_rows),
            "metric/clean_fraction": clean_rows.to(torch.float32).mean(),
            "metric/clean_false_restoration_rate": _selected_mean(
                restoration_applied, clean_rows
            ),
            "metric/clean_min_quality": _selected_mean(clean_min_quality, clean_rows),
            "metric/degraded_restoration_applied_fraction": _selected_mean(
                restoration_applied, degraded_rows
            ),
            "metric/modalities_per_restoration": restoration_mask.sum(dim=1)
            .to(torch.float32)
            .mean(),
            "metric/degraded_rgb_fraction": degradation_flags[:, 0].to(torch.float32).mean(),
            "metric/degraded_depth_fraction": degradation_flags[:, 1].to(torch.float32).mean(),
            "metric/degraded_marker_fraction": degradation_flags[:, 2].to(torch.float32).mean(),
        }
    )
    if use_predicted_restoration:
        metrics.update(
            {
                "metric/restoration_applied_fraction": restoration_output[
                    "restoration_applied"
                ]
                .to(torch.float32)
                .mean(),
                "metric/restoration_blend": restoration_output["restoration_blend"]
                .sum(dim=1)
                .mean(),
            }
        )
    return total, metrics
