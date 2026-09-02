"""Configuration for the plain three-modality cross-attention autoencoder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriModalCrossAutoencoderCfg:
    """Shape and capacity settings for :class:`TriModalCrossAutoencoder`.

    This configuration is intentionally separate from the robust network's
    reliability and restoration settings.  The model always receives RGB,
    Depth, and Marker observations, crosses their encoded tokens, and decodes
    all three clean targets.
    """

    image_height: int = 240
    image_width: int = 320
    rgb_channels: int = 3
    depth_channels: int = 1
    marker_count: int = 100
    marker_input_dim: int = 5
    marker_output_dim: int = 2
    marker_summary_tokens: int = 4

    d_model: int = 256
    num_heads: int = 8
    image_base_channels: int = 32
    marker_transformer_layers: int = 2
    cross_layers: int = 2
    cross_summary_tokens: int = 4
    ffn_ratio: int = 4
    decoder_base_channels: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        positive = {
            "image_height": self.image_height,
            "image_width": self.image_width,
            "rgb_channels": self.rgb_channels,
            "depth_channels": self.depth_channels,
            "marker_count": self.marker_count,
            "marker_input_dim": self.marker_input_dim,
            "marker_output_dim": self.marker_output_dim,
            "marker_summary_tokens": self.marker_summary_tokens,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "image_base_channels": self.image_base_channels,
            "marker_transformer_layers": self.marker_transformer_layers,
            "cross_layers": self.cross_layers,
            "cross_summary_tokens": self.cross_summary_tokens,
            "ffn_ratio": self.ffn_ratio,
            "decoder_base_channels": self.decoder_base_channels,
        }
        invalid = {name: value for name, value in positive.items() if int(value) <= 0}
        if invalid:
            raise ValueError(f"Network dimensions must be positive, got {invalid}")
        if self.image_height % 16 != 0 or self.image_width % 16 != 0:
            raise ValueError("image_height and image_width must both be divisible by 16")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.marker_input_dim != 5 or self.marker_output_dim != 2:
            raise ValueError(
                "marker_input_dim must be [x0,y0,dx,dy,valid] (5 values) and "
                "marker_output_dim must be [dx,dy] (2 values)"
            )
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def token_rows(self) -> int:
        return self.image_height // 16

    @property
    def token_cols(self) -> int:
        return self.image_width // 16

    @property
    def image_token_count(self) -> int:
        return self.token_rows * self.token_cols
