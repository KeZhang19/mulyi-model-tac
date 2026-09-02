"""Reliability-aware cross-attention fusion."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ReliabilityBiasedCrossAttention(nn.Module):
    """Cross-attention with relative reliability bias and absolute value gating."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim**-0.5
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        token_reliability: torch.Tensor,
        token_modality_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, _ = tokens.shape
        expected = (batch_size, token_count)
        for name, value in (
            ("token_mask", token_mask),
            ("token_reliability", token_reliability),
            ("token_modality_weight", token_modality_weight),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        if bool((~token_mask.any(dim=1)).any()):
            raise ValueError("Every sample needs at least one valid fusion token")

        projected_query = self._split_heads(self.query_projection(query))
        projected_key = self._split_heads(self.key_projection(tokens))
        projected_value = self._split_heads(self.value_projection(tokens))
        projected_value = projected_value * token_reliability[:, None, :, None].to(
            dtype=projected_value.dtype
        )

        attention_logits = torch.matmul(projected_query, projected_key.transpose(-2, -1)) * self.scale
        reliability_bias = torch.log(token_modality_weight.clamp_min(1.0e-8))
        attention_logits = attention_logits + reliability_bias[:, None, None, :]
        attention_logits = attention_logits.masked_fill(~token_mask[:, None, None, :], -math.inf)
        attention = F.softmax(attention_logits, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        attended = torch.matmul(attention, projected_value)
        attended = attended.transpose(1, 2).reshape(batch_size, query.shape[1], self.d_model)
        return self.output_projection(attended), attention


class ReliabilityFusionBlock(nn.Module):
    """Pre-norm cross-attention followed by a Transformer feed-forward block."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        ffn_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.attention = ReliabilityBiasedCrossAttention(d_model, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        token_reliability: torch.Tensor,
        token_modality_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.attention(
            self.query_norm(query),
            self.token_norm(tokens),
            token_mask=token_mask,
            token_reliability=token_reliability,
            token_modality_weight=token_modality_weight,
        )
        query = query + self.attention_dropout(attended)
        query = query + self.ffn(self.ffn_norm(query))
        return query, attention


class ReliabilityAwareFusion(nn.Module):
    """Fuse all valid modality tokens into one shared TACT latent."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_ratio: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tact_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.layers = nn.ModuleList(
            ReliabilityFusionBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.tact_token, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        token_reliability: torch.Tensor,
        token_modality_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.tact_token.expand(tokens.shape[0], -1, -1)
        attention = torch.empty(0, device=tokens.device)
        for layer in self.layers:
            query, attention = layer(
                query,
                tokens,
                token_mask=token_mask,
                token_reliability=token_reliability,
                token_modality_weight=token_modality_weight,
            )
        latent = self.output_norm(query[:, 0])
        return latent, attention[:, :, 0]
