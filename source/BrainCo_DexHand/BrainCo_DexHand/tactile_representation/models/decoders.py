"""Latent-conditioned modality decoders."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import ResidualConvBlock, _group_count


class ConditionedUpsampleBlock(nn.Module):
    """Upsample a feature grid and condition it on the shared latent with FiLM."""

    def __init__(self, in_channels: int, out_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.normalization = nn.GroupNorm(_group_count(out_channels), out_channels, affine=False)
        self.condition = nn.Linear(latent_dim, out_channels * 2)
        self.activation = nn.GELU()
        self.refine = ResidualConvBlock(out_channels, out_channels)

    def forward(self, features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        features = F.interpolate(features, scale_factor=2.0, mode="bilinear", align_corners=False)
        features = self.normalization(self.convolution(features))
        scale, shift = self.condition(latent).chunk(2, dim=-1)
        scale = torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        features = self.activation(features * (1.0 + scale) + shift)
        return self.refine(features)


class ConditionedImageDecoder(nn.Module):
    """Decode RGB/Depth from the shared latent, optionally using an RGB reference."""

    def __init__(
        self,
        *,
        latent_dim: int,
        output_channels: int,
        image_height: int,
        image_width: int,
        base_channels: int,
        reference_residual: bool = False,
        reference_residual_scale: float = 0.25,
        spatial_condition_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.seed_rows = self.image_height // 16
        self.seed_cols = self.image_width // 16
        self.reference_residual = bool(reference_residual)
        self.reference_residual_scale = float(reference_residual_scale)
        if not 0.0 < self.reference_residual_scale <= 1.0:
            raise ValueError("reference_residual_scale must be in (0, 1]")
        self.spatial_condition_dim = (
            None if spatial_condition_dim is None else int(spatial_condition_dim)
        )
        seed_rows = self.seed_rows
        seed_cols = self.seed_cols
        channels = (
            base_channels,
            base_channels,
            max(base_channels // 2, 16),
            max(base_channels // 4, 16),
            max(base_channels // 8, 16),
        )
        self.seed = nn.Parameter(torch.empty(1, channels[0], seed_rows, seed_cols))
        self.seed_condition = nn.Linear(latent_dim, channels[0])
        self.blocks = nn.ModuleList(
            ConditionedUpsampleBlock(in_channels, out_channels, latent_dim)
            for in_channels, out_channels in zip(channels[:-1], channels[1:], strict=True)
        )
        self.output_head = nn.Conv2d(channels[-1], output_channels, 3, padding=1)
        if self.spatial_condition_dim is not None:
            self.spatial_projection = nn.Conv2d(
                self.spatial_condition_dim, channels[0], 1, bias=False
            )
        if self.reference_residual:
            self.reference_projection = nn.Sequential(
                nn.Conv2d(output_channels, channels[-1], 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(channels[-1]), channels[-1], affine=False),
                nn.GELU(),
            )
            # The residual branch starts as an exact copy of the reference.
            # This gives the network a useful identity solution and leaves the
            # latent pathway responsible only for contact-induced changes.
            self.reference_residual_head = nn.Conv2d(
                channels[-1] * 2, output_channels, 3, padding=1
            )
            nn.init.zeros_(self.reference_residual_head.weight)
            nn.init.zeros_(self.reference_residual_head.bias)
        nn.init.trunc_normal_(self.seed, std=0.02)

    def forward(
        self,
        latent: torch.Tensor,
        reference: torch.Tensor | None = None,
        spatial_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(latent.shape[0])
        condition = self.seed_condition(latent).unsqueeze(-1).unsqueeze(-1)
        features = self.seed.expand(batch_size, -1, -1, -1) + condition
        if spatial_tokens is not None:
            if self.spatial_condition_dim is None:
                raise ValueError("This decoder was not configured for spatial conditioning")
            expected_tokens = self.seed_rows * self.seed_cols
            expected_shape = (batch_size, expected_tokens, self.spatial_condition_dim)
            if tuple(spatial_tokens.shape) != expected_shape:
                raise ValueError(
                    "spatial_tokens must have shape "
                    f"{expected_shape}, got {tuple(spatial_tokens.shape)}"
                )
            spatial_grid = spatial_tokens.transpose(1, 2).reshape(
                batch_size, self.spatial_condition_dim, self.seed_rows, self.seed_cols
            )
            features = features + self.spatial_projection(spatial_grid)
        for block in self.blocks:
            features = block(features, latent)
        if reference is not None and self.reference_residual:
            if tuple(reference.shape[1:]) != (
                self.output_head.out_channels,
                self.image_height,
                self.image_width,
            ):
                raise ValueError(
                    "RGB reference must have shape "
                    f"[B, {self.output_head.out_channels}, {self.image_height}, "
                    f"{self.image_width}], got {tuple(reference.shape)}"
                )
            reference_features = self.reference_projection(reference)
            residual = self.reference_residual_head(
                torch.cat((features, reference_features), dim=1)
            )
            output = (
                reference + self.reference_residual_scale * torch.tanh(residual)
            ).clamp(0.0, 1.0)
        else:
            output = torch.sigmoid(self.output_head(features))
        if tuple(output.shape[-2:]) != (self.image_height, self.image_width):
            raise RuntimeError(
                f"Image decoder produced {tuple(output.shape[-2:])}, "
                f"expected {(self.image_height, self.image_width)}"
            )
        return output


class MarkerDecoder(nn.Module):
    """Decode one dynamic motion vector for each calibrated marker.

    The learned queries identify a marker and the shared latent captures the
    global tactile state.  Optionally, a query can also attend to RGB/Depth
    spatial tokens.  This retains local contact geometry when the dynamic
    Marker observation is the one being restored.
    """

    def __init__(
        self,
        *,
        marker_count: int,
        output_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_ratio: int,
        dropout: float = 0.0,
        static_context: bool = False,
        image_spatial_context: bool = False,
    ) -> None:
        super().__init__()
        self.marker_queries = nn.Parameter(torch.empty(1, marker_count, d_model))
        self.latent_projection = nn.Linear(d_model, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ffn_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.static_context = bool(static_context)
        if self.static_context:
            self.position_projection = nn.Sequential(
                nn.Linear(2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.valid_projection = nn.Linear(1, d_model, bias=False)
        self.image_spatial_context = bool(image_spatial_context)
        if self.image_spatial_context:
            self.context_query_norm = nn.LayerNorm(d_model)
            self.context_key_norm = nn.LayerNorm(d_model)
            self.context_attention = nn.MultiheadAttention(
                d_model,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.context_dropout = nn.Dropout(dropout)
            self.context_ffn_norm = nn.LayerNorm(d_model)
            self.context_ffn = nn.Sequential(
                nn.Linear(d_model, d_model * ffn_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ffn_ratio, d_model),
                nn.Dropout(dropout),
            )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )
        nn.init.trunc_normal_(self.marker_queries, std=0.02)

    def forward(
        self,
        latent: torch.Tensor,
        *,
        marker_positions: torch.Tensor | None = None,
        marker_valid_mask: torch.Tensor | None = None,
        image_context_tokens: torch.Tensor | None = None,
        image_context_mask: torch.Tensor | None = None,
        image_context_reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.marker_queries.expand(latent.shape[0], -1, -1)
        queries = queries + self.latent_projection(latent).unsqueeze(1)
        if self.static_context and marker_positions is not None:
            expected_positions = (queries.shape[0], queries.shape[1], 2)
            if tuple(marker_positions.shape) != expected_positions:
                raise ValueError(
                    f"marker_positions must have shape {expected_positions}, "
                    f"got {tuple(marker_positions.shape)}"
                )
            queries = queries + self.position_projection(marker_positions)
            if marker_valid_mask is not None:
                expected_valid = (queries.shape[0], queries.shape[1])
                if tuple(marker_valid_mask.shape) != expected_valid:
                    raise ValueError(
                        f"marker_valid_mask must have shape {expected_valid}, "
                        f"got {tuple(marker_valid_mask.shape)}"
                    )
                queries = queries + self.valid_projection(
                    marker_valid_mask.to(dtype=queries.dtype).unsqueeze(-1)
                )
        if image_context_tokens is not None:
            if not self.image_spatial_context:
                raise ValueError("This Marker decoder was not configured for image context")
            expected_prefix = (queries.shape[0],)
            if (
                image_context_tokens.ndim != 3
                or tuple(image_context_tokens.shape[:1]) != expected_prefix
                or int(image_context_tokens.shape[2]) != int(queries.shape[2])
            ):
                raise ValueError(
                    "image_context_tokens must have shape [B,N,D] with the Marker "
                    f"decoder embedding dimension, got {tuple(image_context_tokens.shape)}"
                )
            context_shape = tuple(image_context_tokens.shape[:2])
            if image_context_mask is None:
                image_context_mask = torch.ones(
                    context_shape, device=queries.device, dtype=torch.bool
                )
            elif tuple(image_context_mask.shape) != context_shape:
                raise ValueError(
                    f"image_context_mask must have shape {context_shape}, "
                    f"got {tuple(image_context_mask.shape)}"
                )
            else:
                image_context_mask = image_context_mask.to(
                    device=queries.device, dtype=torch.bool
                )
            # A mixed-modality batch can contain RGB/Depth context for some
            # rows while another row has only Marker.  MultiheadAttention
            # cannot accept an all-masked row, so provide one *zero* sentinel
            # token for those rows.  This is a latent-only fallback, not a
            # leak of a modality disabled by the per-sample mask.
            all_context_missing = ~image_context_mask.any(dim=1)
            if bool(all_context_missing.any()):
                image_context_tokens = image_context_tokens.clone()
                image_context_tokens[all_context_missing] = 0.0
                image_context_mask = image_context_mask.clone()
                image_context_mask[all_context_missing, 0] = True
            if image_context_reliability is not None:
                if tuple(image_context_reliability.shape) != context_shape:
                    raise ValueError(
                        f"image_context_reliability must have shape {context_shape}, "
                        f"got {tuple(image_context_reliability.shape)}"
                    )
                # Reliability is an observed, differentiable soft gate.  It
                # attenuates values without converting a suspected sensor into
                # an oracle-known missing modality.
                image_context_tokens = image_context_tokens * image_context_reliability.to(
                    device=queries.device, dtype=queries.dtype
                ).unsqueeze(-1)
            attended, _ = self.context_attention(
                self.context_query_norm(queries),
                self.context_key_norm(image_context_tokens),
                image_context_tokens,
                key_padding_mask=~image_context_mask,
                need_weights=False,
            )
            queries = queries + self.context_dropout(attended)
            queries = queries + self.context_ffn(self.context_ffn_norm(queries))
        return self.output_head(self.output_norm(self.transformer(queries)))
