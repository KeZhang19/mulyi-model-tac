"""End-to-end robust cross-modal tactile representation network."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..config import CrossModalTactileNetworkCfg
from .decoders import ConditionedImageDecoder, MarkerDecoder
from .encoders import ImageTokenEncoder, MarkerTokenEncoder
from .fusion import ReliabilityAwareFusion
from .reliability import CrossModalReliabilityHead, ReliabilityHead, masked_token_mean


MODALITIES = ("rgb", "depth", "marker")


class RobustCrossModalTactileNetwork(nn.Module):
    """Learn one tactile latent from any non-empty subset of three modalities."""

    def __init__(self, cfg: CrossModalTactileNetworkCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CrossModalTactileNetworkCfg()
        cfg = self.cfg
        self.rgb_encoder = ImageTokenEncoder(
            in_channels=cfg.rgb_channels,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            base_channels=cfg.image_base_channels,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )
        self.depth_encoder = ImageTokenEncoder(
            in_channels=cfg.depth_channels,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            base_channels=cfg.image_base_channels,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            # Exact-zero depth is a valid no-contact observation. Passing it through
            # stacked GroupNorm blocks creates zero variance at every stage and an
            # exponentially large backward scale, so encode it with a finite sentinel.
            empty_input_value=-1.0,
        )
        self.marker_encoder = MarkerTokenEncoder(
            marker_count=cfg.marker_count,
            input_dim=cfg.marker_input_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.marker_transformer_layers,
            ffn_ratio=cfg.ffn_ratio,
            summary_tokens=cfg.marker_summary_tokens,
            dropout=cfg.dropout,
        )
        self.modality_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(1, 1, cfg.d_model)) for name in MODALITIES}
        )
        reliability_head = (
            CrossModalReliabilityHead if cfg.cross_modal_reliability else ReliabilityHead
        )
        self.reliability_heads = nn.ModuleDict(
            {name: reliability_head(cfg.d_model) for name in MODALITIES}
        )
        self.fusion = ReliabilityAwareFusion(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.fusion_layers,
            ffn_ratio=cfg.ffn_ratio,
            dropout=cfg.dropout,
        )
        self.rgb_decoder = ConditionedImageDecoder(
            latent_dim=cfg.d_model,
            output_channels=cfg.rgb_channels,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            base_channels=cfg.decoder_base_channels,
            reference_residual=cfg.rgb_reference_residual,
            reference_residual_scale=cfg.rgb_reference_max_delta,
            spatial_condition_dim=cfg.d_model if cfg.rgb_depth_spatial_skip else None,
        )
        self.depth_decoder = ConditionedImageDecoder(
            latent_dim=cfg.d_model,
            output_channels=cfg.depth_channels,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            base_channels=cfg.decoder_base_channels,
            spatial_condition_dim=cfg.d_model if cfg.depth_rgb_spatial_skip else None,
        )
        # Depth is sparse and normalized so that no contact is exactly zero.
        # A default sigmoid head starts near 0.5, which produces a very large
        # foreground-structure loss and can overflow the first fp16 update.
        # Start near 0.01 while retaining a small non-zero path to upstream
        # decoder features from the first optimization step.
        nn.init.normal_(self.depth_decoder.output_head.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.depth_decoder.output_head.bias, -4.59511985013459)
        self.marker_decoder = MarkerDecoder(
            marker_count=cfg.marker_count,
            output_dim=cfg.marker_output_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.marker_transformer_layers,
            ffn_ratio=cfg.ffn_ratio,
            dropout=cfg.dropout,
            static_context=cfg.marker_static_context,
            image_spatial_context=cfg.marker_image_spatial_context,
        )
        for embedding in self.modality_embeddings.values():
            nn.init.trunc_normal_(embedding, std=0.02)

    def encode(
        self,
        *,
        rgb: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        marker: torch.Tensor | None = None,
        modality_mask: torch.Tensor | Mapping[str, Any] | None = None,
        marker_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only the policy-facing shared tactile latent."""

        return self.encode_with_diagnostics(
            rgb=rgb,
            depth=depth,
            marker=marker,
            modality_mask=modality_mask,
            marker_valid_mask=marker_valid_mask,
        )["latent"]

    def encode_with_diagnostics(
        self,
        *,
        rgb: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        marker: torch.Tensor | None = None,
        modality_mask: torch.Tensor | Mapping[str, Any] | None = None,
        marker_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Encode valid inputs and expose reliability/fusion diagnostics."""

        inputs = {"rgb": rgb, "depth": depth, "marker": marker}
        batch_size, device = self._batch_context(inputs)
        presence = self._resolve_modality_mask(inputs, modality_mask, batch_size, device)

        token_groups: dict[str, torch.Tensor] = {}
        token_validity: dict[str, torch.Tensor] = {}
        if rgb is not None and bool(presence[:, 0].any()):
            token_groups["rgb"] = self.rgb_encoder(rgb)
            token_validity["rgb"] = presence[:, 0, None].expand(-1, self.rgb_encoder.token_count)
        if depth is not None and bool(presence[:, 1].any()):
            token_groups["depth"] = self.depth_encoder(depth)
            token_validity["depth"] = presence[:, 1, None].expand(-1, self.depth_encoder.token_count)
        if marker is not None and bool(presence[:, 2].any()):
            marker_tokens, valid_markers = self.marker_encoder(marker, marker_valid_mask)
            marker_presence = presence[:, 2] & valid_markers.any(dim=1)
            presence = presence.clone()
            presence[:, 2] = marker_presence
            token_groups["marker"] = marker_tokens
            token_validity["marker"] = valid_markers & marker_presence[:, None]

        if bool((~presence.any(dim=1)).any()):
            bad_rows = torch.nonzero(~presence.any(dim=1), as_tuple=False).flatten().tolist()
            raise ValueError(f"Every sample needs at least one valid modality; empty rows={bad_rows}")

        summaries = {
            name: masked_token_mean(token_groups[name], token_validity[name])
            for name in token_groups
        }
        logits = torch.zeros((batch_size, len(MODALITIES)), device=device)
        reliabilities = torch.zeros_like(logits)
        for index, name in enumerate(MODALITIES):
            if name not in token_groups:
                continue
            if self.cfg.cross_modal_reliability:
                peer_names = [
                    peer for peer in MODALITIES if peer != name and peer in summaries
                ]
                peer_summaries = (
                    torch.stack([summaries[peer] for peer in peer_names], dim=1)
                    if peer_names
                    else None
                )
                peer_presence = (
                    torch.stack(
                        [presence[:, MODALITIES.index(peer)] for peer in peer_names],
                        dim=1,
                    )
                    if peer_names
                    else None
                )
                modality_logits = self.reliability_heads[name](
                    token_groups[name],
                    token_validity[name],
                    peer_summaries=peer_summaries,
                    peer_presence=peer_presence,
                )
            else:
                modality_logits = self.reliability_heads[name](
                    token_groups[name], token_validity[name]
                )
            logits[:, index] = modality_logits
            calibrated_logits = modality_logits / float(self.cfg.reliability_temperature)
            reliabilities[:, index] = torch.sigmoid(calibrated_logits) * presence[:, index]

        if self.cfg.predicted_soft_restoration:
            gate_probability = torch.sigmoid(
                (reliabilities - float(self.cfg.reliability_gate_threshold))
                / float(self.cfg.reliability_gate_temperature)
            )
            fusion_gates = (
                float(self.cfg.reliability_gate_floor)
                + (1.0 - float(self.cfg.reliability_gate_floor)) * gate_probability
            ) * presence.to(reliabilities.dtype)
        else:
            fusion_gates = reliabilities
        effective_gates = (
            fusion_gates.detach()
            if self.cfg.detach_reliability_gate
            else fusion_gates
        )
        weights = self._normalized_reliability_weights(effective_gates, presence)
        fusion_tokens = []
        fusion_masks = []
        token_reliability = []
        token_weights = []
        token_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for index, name in enumerate(MODALITIES):
            if name not in token_groups:
                continue
            tokens = token_groups[name] + self.modality_embeddings[name]
            count = int(tokens.shape[1])
            validity = token_validity[name]
            valid_count = validity.sum(dim=1, keepdim=True).clamp_min(1)
            per_token_weight = weights[:, index, None] / valid_count.to(dtype=weights.dtype)
            per_token_weight = per_token_weight.expand(-1, count).masked_fill(~validity, 0.0)
            fusion_tokens.append(tokens)
            fusion_masks.append(validity)
            token_reliability.append(effective_gates[:, index, None].expand(-1, count))
            token_weights.append(per_token_weight)
            token_slices[name] = (offset, offset + count)
            offset += count

        tokens = torch.cat(fusion_tokens, dim=1)
        latent, attention = self.fusion(
            tokens,
            token_mask=torch.cat(fusion_masks, dim=1),
            token_reliability=torch.cat(token_reliability, dim=1),
            token_modality_weight=torch.cat(token_weights, dim=1),
        )
        return {
            "latent": latent,
            "reliability": {
                name: reliabilities[:, index] for index, name in enumerate(MODALITIES)
            },
            "reliability_logits": {
                name: logits[:, index] for index, name in enumerate(MODALITIES)
            },
            "fusion_gate": {
                name: fusion_gates[:, index] for index, name in enumerate(MODALITIES)
            },
            "weights": {name: weights[:, index] for index, name in enumerate(MODALITIES)},
            "modality_mask": presence,
            "attention": attention,
            "token_slices": token_slices,
            # Keep raw modality token grids available to optional spatial
            # decoder skips.  Existing callers can ignore this diagnostic key.
            "token_groups": token_groups,
        }

    def forward(
        self,
        *,
        rgb: torch.Tensor | None = None,
        rgb_reference: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        marker: torch.Tensor | None = None,
        modality_mask: torch.Tensor | Mapping[str, Any] | None = None,
        marker_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        representation = self.encode_with_diagnostics(
            rgb=rgb,
            depth=depth,
            marker=marker,
            modality_mask=modality_mask,
            marker_valid_mask=marker_valid_mask,
        )
        latent = representation["latent"]

        def _decoder_reliability(name: str) -> torch.Tensor:
            """Use the same soft sensor-quality gate in decoder bypasses.

            When reliability-gate detachment is enabled, reconstruction must
            not be able to make a bad sensor look clean through any auxiliary
            spatial path.  The quality head still receives its explicit
            supervision in the objective.
            """

            gate = representation["fusion_gate"][name]
            return gate.detach() if self.cfg.detach_reliability_gate else gate

        # x0/y0 are fixed calibration coordinates, not corrupted dynamic
        # observations.  Keep them available to the decoder even when the
        # Marker modality is excluded from fusion.
        marker_positions = marker[..., :2] if marker is not None else None
        rgb_spatial_tokens = None
        if self.cfg.rgb_depth_spatial_skip and depth is not None:
            # Analogous to the RGB->Depth skip below, this keeps the local
            # contact geometry available to RGB residual reconstruction.  The
            # boolean modality mask is applied per row, so mixed batches cannot
            # leak a disabled Depth observation through this bypass.
            rgb_spatial_tokens = representation.get("token_groups", {}).get("depth")
            if rgb_spatial_tokens is not None:
                depth_presence = representation["modality_mask"][:, 1]
                rgb_spatial_tokens = rgb_spatial_tokens * (
                    depth_presence.to(rgb_spatial_tokens.dtype)
                    * _decoder_reliability("depth").to(rgb_spatial_tokens.dtype)
                )[:, None, None]

        depth_spatial_tokens = None
        if self.cfg.depth_rgb_spatial_skip and rgb is not None:
            # The mask is important for mixed batches where RGB is physically
            # present but disabled for only some rows.  Those rows must use the
            # latent-only fallback rather than leaking an unavailable image.
            depth_spatial_tokens = representation.get("token_groups", {}).get("rgb")
            if depth_spatial_tokens is not None:
                rgb_presence = representation["modality_mask"][:, 0]
                depth_spatial_tokens = depth_spatial_tokens * (
                    rgb_presence.to(depth_spatial_tokens.dtype)
                    * _decoder_reliability("rgb").to(depth_spatial_tokens.dtype)
                )[:, None, None]

        marker_image_context_tokens = None
        marker_image_context_mask = None
        marker_image_context_reliability = None
        if self.cfg.marker_image_spatial_context:
            context_tokens: list[torch.Tensor] = []
            context_masks: list[torch.Tensor] = []
            context_reliabilities: list[torch.Tensor] = []
            for index, name in ((0, "rgb"), (1, "depth")):
                modality_tokens = representation.get("token_groups", {}).get(name)
                if modality_tokens is None:
                    continue
                token_count = int(modality_tokens.shape[1])
                presence = representation["modality_mask"][:, index]
                # Include the modality embedding so the Marker cross-attention
                # can distinguish RGB geometry from a Depth surface token.
                context_tokens.append(modality_tokens + self.modality_embeddings[name])
                context_masks.append(presence[:, None].expand(-1, token_count))
                context_reliabilities.append(
                    _decoder_reliability(name)[:, None].expand(-1, token_count)
                )
            if context_tokens:
                marker_image_context_tokens = torch.cat(context_tokens, dim=1)
                marker_image_context_mask = torch.cat(context_masks, dim=1)
                marker_image_context_reliability = torch.cat(context_reliabilities, dim=1)
        representation.update(
            {
                "rgb_recon": self.rgb_decoder(
                    latent,
                    reference=rgb_reference,
                    spatial_tokens=rgb_spatial_tokens,
                ),
                "depth_recon": self.depth_decoder(
                    latent, spatial_tokens=depth_spatial_tokens
                ),
                "marker_recon": self.marker_decoder(
                    latent,
                    marker_positions=marker_positions,
                    marker_valid_mask=marker_valid_mask,
                    image_context_tokens=marker_image_context_tokens,
                    image_context_mask=marker_image_context_mask,
                    image_context_reliability=marker_image_context_reliability,
                ),
            }
        )
        return representation

    def decode_from_latent(
        self,
        latent: torch.Tensor,
        *,
        rgb_reference: torch.Tensor | None = None,
        marker_positions: torch.Tensor | None = None,
        marker_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode every clean target without dynamic spatial bypasses.

        This path is used by the training-only latent-sufficiency objective.  It
        deliberately withholds RGB/Depth token grids from the image decoders
        and image-context tokens from the Marker decoder, so reconstruction
        gradients must pass through the shared tactile latent.  Fixed sensor
        calibration remains available: the unpressed RGB reference carries
        static appearance and ``marker_positions`` carries fixed x0/y0.
        """

        expected_latent = (self.cfg.d_model,)
        if latent.ndim != 2 or tuple(latent.shape[1:]) != expected_latent:
            raise ValueError(
                f"latent must have shape [B,{self.cfg.d_model}], got {tuple(latent.shape)}"
            )
        if self.cfg.rgb_reference_residual and rgb_reference is None:
            raise ValueError(
                "rgb_reference is required for latent-only RGB residual decoding"
            )
        return {
            "rgb_recon": self.rgb_decoder(
                latent,
                reference=rgb_reference,
                spatial_tokens=None,
            ),
            "depth_recon": self.depth_decoder(latent, spatial_tokens=None),
            "marker_recon": self.marker_decoder(
                latent,
                marker_positions=marker_positions,
                marker_valid_mask=marker_valid_mask,
                image_context_tokens=None,
                image_context_mask=None,
                image_context_reliability=None,
            ),
        }

    def restore_degraded_observation(
        self,
        *,
        rgb: torch.Tensor,
        rgb_reference: torch.Tensor | None = None,
        depth: torch.Tensor,
        marker: torch.Tensor,
        marker_valid_mask: torch.Tensor | None = None,
        quality_threshold: float | None = 0.9,
    ) -> dict[str, Any]:
        """Detect and restore a degraded observation without an oracle mask.

        New checkpoints use predicted quality both to suppress unreliable fusion
        tokens and to blend the least-reliable observation with its clean-target
        reconstruction.  The legacy two-pass hard-mask path is retained only for
        old checkpoint compatibility.
        """

        if quality_threshold is not None and not 0.0 <= float(quality_threshold) <= 1.0:
            raise ValueError("quality_threshold must be in [0, 1] or None")
        if self.cfg.rgb_reference_residual and rgb_reference is None:
            raise ValueError(
                "rgb_reference is required when RGB reference-residual restoration is enabled"
            )
        inputs = {"rgb": rgb, "depth": depth, "marker": marker}
        batch_size, device = self._batch_context(inputs)
        full_mask = torch.ones((batch_size, len(MODALITIES)), device=device, dtype=torch.bool)
        if self.cfg.predicted_soft_restoration:
            restored = self.forward(
                rgb=rgb,
                rgb_reference=rgb_reference,
                depth=depth,
                marker=marker,
                marker_valid_mask=marker_valid_mask,
                modality_mask=full_mask,
            )
            quality = torch.stack(
                [restored["reliability"][name] for name in MODALITIES], dim=1
            )
            fusion_gates = torch.stack(
                [restored["fusion_gate"][name] for name in MODALITIES], dim=1
            )
            degraded_index = quality.argmin(dim=1)
            degraded_quality = quality.gather(1, degraded_index[:, None]).squeeze(1)
            apply_restoration = torch.ones(batch_size, device=device, dtype=torch.bool)
            if quality_threshold is not None:
                apply_restoration = degraded_quality < float(quality_threshold)
            selected = torch.nn.functional.one_hot(
                degraded_index, num_classes=len(MODALITIES)
            ).to(torch.bool)
            restoration_blend = (
                selected.to(fusion_gates.dtype)
                * apply_restoration[:, None].to(fusion_gates.dtype)
                * (1.0 - fusion_gates)
            )
            rgb_blend = restoration_blend[:, 0, None, None, None]
            depth_blend = restoration_blend[:, 1, None, None, None]
            marker_blend = restoration_blend[:, 2, None, None]
            restored.update(
                {
                    "observed_quality": quality,
                    "observed_fusion_gate": fusion_gates,
                    "detected_degraded_modality": degraded_index,
                    "detected_degraded_quality": degraded_quality,
                    "restoration_applied": apply_restoration,
                    # This is the physical availability mask; no predicted or
                    # ground-truth damage label is converted into hard absence.
                    "restoration_modality_mask": full_mask,
                    "restoration_blend": restoration_blend,
                    "restored_rgb": rgb * (1.0 - rgb_blend)
                    + restored["rgb_recon"] * rgb_blend,
                    "restored_depth": depth * (1.0 - depth_blend)
                    + restored["depth_recon"] * depth_blend,
                    "restored_marker_motion": marker[..., 2:4]
                    * (1.0 - marker_blend)
                    + restored["marker_recon"] * marker_blend,
                }
            )
            return restored

        detection = self.encode_with_diagnostics(
            rgb=rgb,
            depth=depth,
            marker=marker,
            marker_valid_mask=marker_valid_mask,
            modality_mask=full_mask,
        )
        quality = torch.stack(
            [detection["reliability"][name] for name in MODALITIES], dim=1
        )
        degraded_index = quality.argmin(dim=1)
        degraded_quality = quality.gather(1, degraded_index[:, None]).squeeze(1)
        apply_restoration = torch.ones(batch_size, device=device, dtype=torch.bool)
        if quality_threshold is not None:
            apply_restoration = degraded_quality < float(quality_threshold)
        restoration_mask = full_mask.clone()
        rows = torch.nonzero(apply_restoration, as_tuple=False).flatten()
        if int(rows.numel()) > 0:
            restoration_mask[rows, degraded_index.index_select(0, rows)] = False

        restored = self.forward(
            rgb=rgb,
            rgb_reference=rgb_reference,
            depth=depth,
            marker=marker,
            marker_valid_mask=marker_valid_mask,
            modality_mask=restoration_mask,
        )
        restore_rgb = apply_restoration & (degraded_index == 0)
        restore_depth = apply_restoration & (degraded_index == 1)
        restore_marker = apply_restoration & (degraded_index == 2)
        restored.update(
            {
                "observed_quality": quality,
                "detected_degraded_modality": degraded_index,
                "detected_degraded_quality": degraded_quality,
                "restoration_applied": apply_restoration,
                "restoration_modality_mask": restoration_mask,
                "restored_rgb": torch.where(
                    restore_rgb[:, None, None, None], restored["rgb_recon"], rgb
                ),
                "restored_depth": torch.where(
                    restore_depth[:, None, None, None], restored["depth_recon"], depth
                ),
                "restored_marker_motion": torch.where(
                    restore_marker[:, None, None],
                    restored["marker_recon"],
                    marker[..., 2:4],
                ),
            }
        )
        return restored

    def _batch_context(
        self,
        inputs: Mapping[str, torch.Tensor | None],
    ) -> tuple[int, torch.device]:
        available = [(name, value) for name, value in inputs.items() if value is not None]
        if not available:
            raise ValueError("At least one input tensor is required")
        batch_size = int(available[0][1].shape[0])
        device = available[0][1].device
        for name, value in available[1:]:
            if int(value.shape[0]) != batch_size:
                raise ValueError(f"{name} batch size {value.shape[0]} does not match {batch_size}")
            if value.device != device:
                raise ValueError(f"{name} is on {value.device}, expected {device}")
        return batch_size, device

    def _resolve_modality_mask(
        self,
        inputs: Mapping[str, torch.Tensor | None],
        modality_mask: torch.Tensor | Mapping[str, Any] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        input_presence = torch.tensor(
            [inputs[name] is not None for name in MODALITIES], device=device, dtype=torch.bool
        ).unsqueeze(0).expand(batch_size, -1)
        if modality_mask is None:
            mask = input_presence.clone()
        elif isinstance(modality_mask, torch.Tensor):
            mask = modality_mask.to(device=device, dtype=torch.bool)
            if tuple(mask.shape) == (len(MODALITIES),):
                mask = mask.unsqueeze(0).expand(batch_size, -1)
            if tuple(mask.shape) != (batch_size, len(MODALITIES)):
                raise ValueError(
                    f"modality_mask must have shape {(batch_size, len(MODALITIES))}, "
                    f"got {tuple(mask.shape)}"
                )
        elif isinstance(modality_mask, Mapping):
            columns = []
            for name in MODALITIES:
                value = modality_mask.get(name, inputs[name] is not None)
                if isinstance(value, torch.Tensor):
                    column = value.to(device=device, dtype=torch.bool)
                    if column.ndim == 0:
                        column = column.expand(batch_size)
                    if tuple(column.shape) != (batch_size,):
                        raise ValueError(
                            f"modality_mask[{name!r}] must be scalar or [{batch_size}], "
                            f"got {tuple(column.shape)}"
                        )
                else:
                    column = torch.full((batch_size,), bool(value), device=device, dtype=torch.bool)
                columns.append(column)
            mask = torch.stack(columns, dim=1)
        else:
            raise TypeError("modality_mask must be a tensor, mapping, or None")

        invalid = mask & ~input_presence
        if bool(invalid.any()):
            rows, columns = torch.nonzero(invalid, as_tuple=True)
            details = [(int(row), MODALITIES[int(column)]) for row, column in zip(rows, columns)]
            raise ValueError(f"modality_mask marks missing input tensors as present: {details}")
        return mask

    def _normalized_reliability_weights(
        self,
        reliabilities: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        # Reliability means "is this sensor trustworthy?", not "which modality
        # should win?". Normalizing sigmoid probabilities keeps all clean sensors
        # active while still suppressing a sensor whose reliability approaches zero.
        scores = reliabilities.masked_fill(~presence, 0.0)
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
