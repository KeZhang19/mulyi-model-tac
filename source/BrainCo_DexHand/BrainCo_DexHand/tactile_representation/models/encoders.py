"""Modality-specific token encoders."""

from __future__ import annotations

import math

import torch
from torch import nn


def _group_count(channels: int, maximum: int = 8) -> int:
    """Choose the largest small GroupNorm divisor for ``channels``."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """Small-batch-friendly residual block used by both image encoders."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class ImageTokenEncoder(nn.Module):
    """Convert an RGB or Depth image into a spatial token sequence."""

    def __init__(
        self,
        *,
        in_channels: int,
        image_height: int,
        image_width: int,
        base_channels: int,
        d_model: int,
        dropout: float = 0.0,
        empty_input_value: float | None = None,
    ) -> None:
        super().__init__()
        if empty_input_value is not None:
            empty_input_value = float(empty_input_value)
            if not math.isfinite(empty_input_value) or empty_input_value == 0.0:
                raise ValueError("empty_input_value must be finite and non-zero")
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.in_channels = int(in_channels)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.empty_input_value = empty_input_value
        self.token_rows = self.image_height // 16
        self.token_cols = self.image_width // 16

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.GELU(),
        )
        self.stages = nn.Sequential(
            ResidualConvBlock(channels[0], channels[0]),
            ResidualConvBlock(channels[0], channels[1], stride=2),
            ResidualConvBlock(channels[1], channels[2], stride=2),
            ResidualConvBlock(channels[2], channels[3], stride=2),
            ResidualConvBlock(channels[3], channels[3]),
        )
        self.projection = nn.Conv2d(channels[3], d_model, 1)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.token_rows * self.token_cols, d_model)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @property
    def token_count(self) -> int:
        return self.token_rows * self.token_cols

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, self.image_height, self.image_width)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected:
            raise ValueError(
                f"Image encoder expects [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(image.shape)}"
            )
        if self.empty_input_value is not None:
            empty_rows = image.detach().abs().amax(dim=(1, 2, 3), keepdim=True) == 0.0
            image = torch.where(
                empty_rows,
                image.new_tensor(self.empty_input_value),
                image,
            )
        features = self.projection(self.stages(self.stem(image)))
        if tuple(features.shape[-2:]) != (self.token_rows, self.token_cols):
            raise RuntimeError(
                f"Image encoder produced spatial shape {tuple(features.shape[-2:])}, "
                f"expected {(self.token_rows, self.token_cols)}"
            )
        tokens = features.flatten(2).transpose(1, 2)
        return self.dropout(self.output_norm(tokens + self.position_embedding))


class MarkerTokenEncoder(nn.Module):
    """Encode marker motion separately from static calibration coordinates."""

    def __init__(
        self,
        *,
        marker_count: int,
        input_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_ratio: int,
        summary_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_dim) != 5:
            raise ValueError("MarkerTokenEncoder expects [x0,y0,dx,dy,valid] inputs")
        self.marker_count = int(marker_count)
        self.input_dim = int(input_dim)
        self.summary_token_count = int(summary_tokens)
        self.motion_mlp = nn.Sequential(
            # [dx, dy, magnitude, valid]. dx/dy have already been divided by
            # the physical marker-motion scale rather than the image size.
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.marker_id_embedding = nn.Parameter(torch.empty(1, marker_count, d_model))
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
        self.token_norm = nn.LayerNorm(d_model)
        self.summary_queries = nn.Parameter(torch.empty(1, self.summary_token_count, d_model))
        self.summary_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.summary_norm = nn.LayerNorm(d_model)
        self.summary_ffn_norm = nn.LayerNorm(d_model)
        self.summary_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout),
        )
        nn.init.trunc_normal_(self.marker_id_embedding, std=0.02)
        nn.init.trunc_normal_(self.summary_queries, std=0.02)

    def forward(
        self,
        markers: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.marker_count, self.input_dim)
        if markers.ndim != 3 or tuple(markers.shape[1:]) != expected:
            raise ValueError(
                f"Marker encoder expects [B,{expected[0]},{expected[1]}], got {tuple(markers.shape)}"
            )
        batch_size = int(markers.shape[0])
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch_size, self.marker_count), device=markers.device, dtype=torch.bool
            )
        else:
            if tuple(valid_mask.shape) != (batch_size, self.marker_count):
                raise ValueError(
                    f"marker_valid_mask must have shape {(batch_size, self.marker_count)}, "
                    f"got {tuple(valid_mask.shape)}"
                )
            valid_mask = valid_mask.to(device=markers.device, dtype=torch.bool)

        motion = markers[..., 2:4]
        magnitude = torch.linalg.vector_norm(motion, dim=-1, keepdim=True)
        motion_features = torch.cat(
            (motion, magnitude, valid_mask.to(dtype=markers.dtype).unsqueeze(-1)),
            dim=-1,
        )
        tokens = (
            self.motion_mlp(motion_features)
            + self.position_mlp(markers[..., :2])
            + self.marker_id_embedding
        )
        tokens = tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        # PyTorch attention cannot consume a row whose every key is masked.
        # Temporarily expose one zero token, then zero the row again below.
        padding_mask = ~valid_mask
        all_invalid = ~valid_mask.any(dim=1)
        if bool(all_invalid.any()):
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False
        tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        tokens = self.token_norm(tokens)
        tokens = tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        # A few learned summaries preserve distinct spatial motion modes instead
        # of forcing one fusion query to uniformly average all marker vectors.
        queries = self.summary_queries.expand(batch_size, -1, -1)
        attended, _ = self.summary_attention(
            queries,
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        summaries = self.summary_norm(queries + attended)
        summaries = summaries + self.summary_ffn(self.summary_ffn_norm(summaries))
        summary_valid = valid_mask.any(dim=1, keepdim=True).expand(
            -1, self.summary_token_count
        )
        summaries = summaries.masked_fill(~summary_valid.unsqueeze(-1), 0.0)
        return summaries, summary_valid
