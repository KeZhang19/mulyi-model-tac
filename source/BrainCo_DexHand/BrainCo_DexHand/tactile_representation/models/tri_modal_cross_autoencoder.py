"""Three-encoder, cross-attention, three-decoder tactile autoencoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..tri_modal_config import TriModalCrossAutoencoderCfg
from .decoders import ConditionedImageDecoder, MarkerDecoder
from .encoders import ImageTokenEncoder, MarkerTokenEncoder


MODALITIES = ("rgb", "depth", "marker")


def _masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class _CrossSummaryPool(nn.Module):
    """Compress one modality to a few tokens before peer cross-attention."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        summary_tokens: int,
        ffn_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, summary_tokens, d_model))
        self.token_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout),
        )
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
        expected_mask = tuple(tokens.shape[:2])
        if tuple(valid_mask.shape) != expected_mask:
            raise ValueError(
                f"valid_mask must have shape {expected_mask}, got {tuple(valid_mask.shape)}"
            )
        valid_mask = valid_mask.to(device=tokens.device, dtype=torch.bool)
        safe_tokens = tokens
        safe_mask = valid_mask
        all_invalid = ~valid_mask.any(dim=1)
        if bool(all_invalid.any()):
            safe_tokens = tokens.clone()
            safe_tokens[all_invalid] = 0.0
            safe_mask = valid_mask.clone()
            safe_mask[all_invalid, 0] = True

        queries = self.queries.expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(
            queries,
            self.token_norm(safe_tokens),
            safe_tokens,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        summaries = self.output_norm(queries + self.attention_dropout(attended))
        summaries = summaries + self.ffn(self.ffn_norm(summaries))
        summary_valid = valid_mask.any(dim=1, keepdim=True).expand(
            -1, summaries.shape[1]
        )
        summaries = summaries.masked_fill(~summary_valid.unsqueeze(-1), 0.0)
        return summaries, summary_valid


class TriModalCrossAttentionBlock(nn.Module):
    """Let every modality query compact summaries of the other two modalities."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        summary_tokens: int,
        ffn_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.summary_pools = nn.ModuleDict(
            {
                name: _CrossSummaryPool(
                    d_model=d_model,
                    num_heads=num_heads,
                    summary_tokens=summary_tokens,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for name in MODALITIES
            }
        )
        self.query_norms = nn.ModuleDict(
            {name: nn.LayerNorm(d_model) for name in MODALITIES}
        )
        self.context_norms = nn.ModuleDict(
            {name: nn.LayerNorm(d_model) for name in MODALITIES}
        )
        self.cross_attention = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(
                    d_model,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for name in MODALITIES
            }
        )
        self.attention_dropout = nn.ModuleDict(
            {name: nn.Dropout(dropout) for name in MODALITIES}
        )
        self.ffn_norms = nn.ModuleDict(
            {name: nn.LayerNorm(d_model) for name in MODALITIES}
        )
        self.ffns = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(d_model, d_model * ffn_ratio),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * ffn_ratio, d_model),
                    nn.Dropout(dropout),
                )
                for name in MODALITIES
            }
        )

    def forward(
        self,
        token_groups: dict[str, torch.Tensor],
        token_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if set(token_groups) != set(MODALITIES) or set(token_masks) != set(MODALITIES):
            raise ValueError("Cross-attention requires rgb, depth, and marker token groups")
        summaries: dict[str, torch.Tensor] = {}
        summary_masks: dict[str, torch.Tensor] = {}
        for name in MODALITIES:
            summaries[name], summary_masks[name] = self.summary_pools[name](
                token_groups[name], token_masks[name]
            )

        updated: dict[str, torch.Tensor] = {}
        for name in MODALITIES:
            peers = tuple(peer for peer in MODALITIES if peer != name)
            context = torch.cat([summaries[peer] for peer in peers], dim=1)
            context_mask = torch.cat([summary_masks[peer] for peer in peers], dim=1)
            if bool((~context_mask.any(dim=1)).any()):
                raise ValueError(f"{name} cross-attention needs at least one valid peer")
            attended, _ = self.cross_attention[name](
                self.query_norms[name](token_groups[name]),
                self.context_norms[name](context),
                context,
                key_padding_mask=~context_mask,
                need_weights=False,
            )
            tokens = token_groups[name] + self.attention_dropout[name](attended)
            tokens = tokens + self.ffns[name](self.ffn_norms[name](tokens))
            updated[name] = tokens.masked_fill(~token_masks[name].unsqueeze(-1), 0.0)
        return updated


class TriModalCrossAutoencoder(nn.Module):
    """Encode RGB/Depth/Marker, cross their tokens, and decode all modalities.

    Unlike :class:`RobustCrossModalTactileNetwork`, this is a deliberately
    simple comparison network: all three observations are required and it has
    no reliability detector, oracle mask, or restoration gate.
    """

    def __init__(self, cfg: TriModalCrossAutoencoderCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TriModalCrossAutoencoderCfg()
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
        self.cross_blocks = nn.ModuleList(
            TriModalCrossAttentionBlock(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                summary_tokens=cfg.cross_summary_tokens,
                ffn_ratio=cfg.ffn_ratio,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.cross_layers)
        )
        self.latent_projection = nn.Sequential(
            nn.Linear(cfg.d_model * len(MODALITIES), cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.rgb_decoder = ConditionedImageDecoder(
            latent_dim=cfg.d_model,
            output_channels=cfg.rgb_channels,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            base_channels=cfg.decoder_base_channels,
            spatial_condition_dim=cfg.d_model,
        )
        self.depth_decoder = ConditionedImageDecoder(
            latent_dim=cfg.d_model,
            output_channels=cfg.depth_channels,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            base_channels=cfg.decoder_base_channels,
            spatial_condition_dim=cfg.d_model,
        )
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
            static_context=True,
            image_spatial_context=True,
        )
        for embedding in self.modality_embeddings.values():
            nn.init.trunc_normal_(embedding, std=0.02)

    def encode(
        self,
        *,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        marker: torch.Tensor,
        marker_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the shared latent after all three cross-attention streams."""

        return self.encode_with_diagnostics(
            rgb=rgb,
            depth=depth,
            marker=marker,
            marker_valid_mask=marker_valid_mask,
        )["latent"]

    def encode_with_diagnostics(
        self,
        *,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        marker: torch.Tensor,
        marker_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Return the shared latent and the three crossed token streams."""

        self._validate_inputs(rgb=rgb, depth=depth, marker=marker)
        batch_size = int(rgb.shape[0])
        if marker_valid_mask is None:
            marker_valid_mask = marker[..., 4] > 0.5
        else:
            expected_valid = (batch_size, self.cfg.marker_count)
            if tuple(marker_valid_mask.shape) != expected_valid:
                raise ValueError(
                    f"marker_valid_mask must have shape {expected_valid}, "
                    f"got {tuple(marker_valid_mask.shape)}"
                )
            marker_valid_mask = marker_valid_mask.to(device=marker.device, dtype=torch.bool)

        marker_tokens, marker_token_mask = self.marker_encoder(marker, marker_valid_mask)
        token_groups = {
            "rgb": self.rgb_encoder(rgb) + self.modality_embeddings["rgb"],
            "depth": self.depth_encoder(depth) + self.modality_embeddings["depth"],
            "marker": marker_tokens + self.modality_embeddings["marker"],
        }
        token_masks = {
            "rgb": torch.ones(
                (batch_size, self.rgb_encoder.token_count),
                device=rgb.device,
                dtype=torch.bool,
            ),
            "depth": torch.ones(
                (batch_size, self.depth_encoder.token_count),
                device=depth.device,
                dtype=torch.bool,
            ),
            "marker": marker_token_mask,
        }
        for block in self.cross_blocks:
            token_groups = block(token_groups, token_masks)

        modality_latents = {
            name: _masked_mean(token_groups[name], token_masks[name])
            for name in MODALITIES
        }
        latent = self.latent_projection(
            torch.cat([modality_latents[name] for name in MODALITIES], dim=-1)
        )
        return {
            "latent": latent,
            "modality_latents": modality_latents,
            "cross_tokens": token_groups,
            "token_masks": token_masks,
            "marker_valid_mask": marker_valid_mask,
        }

    def forward(
        self,
        *,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        marker: torch.Tensor,
        marker_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        representation = self.encode_with_diagnostics(
            rgb=rgb,
            depth=depth,
            marker=marker,
            marker_valid_mask=marker_valid_mask,
        )
        latent = representation["latent"]
        cross_tokens = representation["cross_tokens"]
        token_masks = representation["token_masks"]
        image_context_tokens = torch.cat(
            (cross_tokens["rgb"], cross_tokens["depth"]), dim=1
        )
        image_context_mask = torch.cat(
            (token_masks["rgb"], token_masks["depth"]), dim=1
        )
        representation.update(
            {
                "rgb_recon": self.rgb_decoder(
                    latent,
                    spatial_tokens=cross_tokens["rgb"],
                ),
                "depth_recon": self.depth_decoder(
                    latent,
                    spatial_tokens=cross_tokens["depth"],
                ),
                "marker_recon": self.marker_decoder(
                    latent,
                    marker_positions=marker[..., :2],
                    marker_valid_mask=representation["marker_valid_mask"],
                    image_context_tokens=image_context_tokens,
                    image_context_mask=image_context_mask,
                ),
            }
        )
        return representation

    def _validate_inputs(
        self,
        *,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        marker: torch.Tensor,
    ) -> None:
        cfg = self.cfg
        expected = {
            "rgb": (cfg.rgb_channels, cfg.image_height, cfg.image_width),
            "depth": (cfg.depth_channels, cfg.image_height, cfg.image_width),
            "marker": (cfg.marker_count, cfg.marker_input_dim),
        }
        inputs = {"rgb": rgb, "depth": depth, "marker": marker}
        batch_size = int(rgb.shape[0]) if rgb.ndim > 0 else -1
        device = rgb.device
        for name, tensor in inputs.items():
            if tensor.ndim != len(expected[name]) + 1 or tuple(tensor.shape[1:]) != expected[name]:
                raise ValueError(
                    f"{name} must have shape [B,{','.join(map(str, expected[name]))}], "
                    f"got {tuple(tensor.shape)}"
                )
            if int(tensor.shape[0]) != batch_size:
                raise ValueError(
                    f"{name} batch size {tensor.shape[0]} does not match RGB batch size "
                    f"{batch_size}"
                )
            if tensor.device != device:
                raise ValueError(f"{name} is on {tensor.device}, expected {device}")
