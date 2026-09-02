"""Self-only and cross-modal tactile reliability estimation."""

from __future__ import annotations

import torch
from torch import nn


class ReliabilityHead(nn.Module):
    """Predict one reliability logit from a token sequence."""

    def __init__(self, d_model: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"ReliabilityHead expects [B,N,D], got {tuple(tokens.shape)}")
        if token_mask is None:
            pooled = tokens.mean(dim=1)
        else:
            if tuple(token_mask.shape) != tuple(tokens.shape[:2]):
                raise ValueError(
                    f"token_mask must have shape {tuple(tokens.shape[:2])}, got {tuple(token_mask.shape)}"
                )
            weights = token_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
            pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.mlp(pooled).squeeze(-1)


def masked_token_mean(
    tokens: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool ``[B,N,D]`` tokens without leaking padded or absent rows."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,N,D] tokens, got {tuple(tokens.shape)}")
    if token_mask is None:
        return tokens.mean(dim=1)
    if tuple(token_mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(
            f"token_mask must have shape {tuple(tokens.shape[:2])}, "
            f"got {tuple(token_mask.shape)}"
        )
    weights = token_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class CrossModalReliabilityHead(nn.Module):
    """Estimate quality by comparing one modality with its available peers.

    The comparison happens before fusion, which avoids the circular design in
    which a fused latent would first consume a bad sensor and then be asked to
    decide whether that same sensor was bad.
    """

    def __init__(self, d_model: int, hidden_dim: int = 128) -> None:
        super().__init__()
        feature_dim = int(d_model) * 4 + 1
        self.input_norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start by trusting all sensors. This keeps content gradients healthy
        # before explicit quality supervision separates clean and degraded data.
        nn.init.constant_(self.mlp[-1].bias, 2.9444389791664403)  # logit(0.95)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        peer_summaries: torch.Tensor | None = None,
        peer_presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current = masked_token_mean(tokens, token_mask)
        if peer_summaries is None:
            context = torch.zeros_like(current)
            context_available = torch.zeros(
                (current.shape[0], 1), device=current.device, dtype=current.dtype
            )
        else:
            if peer_summaries.ndim != 3 or peer_summaries.shape[2] != current.shape[1]:
                raise ValueError(
                    "peer_summaries must have shape [B,K,D] matching the current summary"
                )
            expected = (current.shape[0], peer_summaries.shape[1])
            if peer_presence is None or tuple(peer_presence.shape) != expected:
                raise ValueError(f"peer_presence must have shape {expected}")
            weights = peer_presence.to(
                device=peer_summaries.device, dtype=peer_summaries.dtype
            ).unsqueeze(-1)
            context = (peer_summaries * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            context_available = peer_presence.any(dim=1, keepdim=True).to(current.dtype)
        features = torch.cat(
            (
                current,
                context,
                (current - context).abs(),
                current * context,
                context_available,
            ),
            dim=-1,
        )
        return self.mlp(self.input_norm(features)).squeeze(-1)
