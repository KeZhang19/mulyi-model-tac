"""Configuration for the robust cross-modal tactile network."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrossModalTactileNetworkCfg:
    """Shape and capacity settings for a single-finger tactile network.

    Five-finger observations can be folded into the batch dimension so the
    same network is shared across all fingers.
    """

    image_height: int = 240
    image_width: int = 320
    rgb_channels: int = 3
    depth_channels: int = 1
    marker_count: int = 100
    # Encoder input: [x0, y0, dx, dy, valid], matching the collector's 2-D
    # marker flow. The decoder predicts only dynamic [dx, dy]; x0/y0 are fixed
    # calibration coordinates and valid is an input/loss mask, not a target.
    marker_input_dim: int = 5
    marker_output_dim: int = 2
    marker_summary_tokens: int = 4

    d_model: int = 256
    num_heads: int = 8
    image_base_channels: int = 32
    marker_transformer_layers: int = 2
    fusion_layers: int = 3
    ffn_ratio: int = 4
    decoder_base_channels: int = 128
    dropout: float = 0.0
    reliability_temperature: float = 1.0
    # Compare every modality with the pooled summaries of the other available
    # modalities.  False preserves the original self-only reliability heads and
    # their checkpoint parameter shapes.
    cross_modal_reliability: bool = False
    # Convert predicted quality into a differentiable fusion gate.  The
    # degradation label is never used to build this gate.
    predicted_soft_restoration: bool = False
    reliability_gate_threshold: float = 0.9
    reliability_gate_temperature: float = 0.05
    reliability_gate_floor: float = 0.01
    # Prevent the reconstruction objective from making every quality gate look
    # clean.  Reliability still learns from its explicit quality supervision.
    detach_reliability_gate: bool = False
    # When enabled, the RGB decoder receives the unpressed reference frame and
    # predicts a bounded residual on top of it.  False preserves the original
    # latent-only decoder and its checkpoint layout.
    rgb_reference_residual: bool = False
    # Old checkpoints used 0.25. New training raises this to 0.5 because the
    # recorded reference differences reach roughly 0.4 in a small set of
    # contact pixels.
    rgb_reference_max_delta: float = 0.25
    # The uncorrupted depth field gives the RGB residual decoder an explicit
    # spatial cue for the local contact deformation.  This is intentionally a
    # separate, optional path: the shared latent remains the policy feature and
    # RGB-only inference still has a valid latent-only fallback.
    rgb_depth_spatial_skip: bool = False
    # Fixed marker calibration coordinates remain available as decoder context;
    # physical marker absence falls back to learned marker queries.
    marker_static_context: bool = False
    # When RGB is available, expose its spatial token grid to the Depth
    # decoder.  A single pooled latent is intentionally kept as the fallback
    # for RGB-missing inference and for compatibility with old checkpoints.
    depth_rgb_spatial_skip: bool = False
    # Give every calibrated Marker query direct cross-attention to RGB/Depth
    # spatial tokens.  A single pooled latent is not expressive enough to
    # recover a detailed 100-point local motion field when Marker motion is
    # unreliable.  The context intentionally excludes dynamic Marker tokens,
    # since Marker itself may be the observation being restored.
    marker_image_spatial_context: bool = False

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
            "fusion_layers": self.fusion_layers,
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
        if float(self.reliability_temperature) <= 0.0:
            raise ValueError("reliability_temperature must be positive")
        if not 0.0 <= float(self.reliability_gate_threshold) <= 1.0:
            raise ValueError("reliability_gate_threshold must be in [0, 1]")
        if float(self.reliability_gate_temperature) <= 0.0:
            raise ValueError("reliability_gate_temperature must be positive")
        if not 0.0 <= float(self.reliability_gate_floor) < 1.0:
            raise ValueError("reliability_gate_floor must be in [0, 1)")
        if not 0.0 < float(self.rgb_reference_max_delta) <= 1.0:
            raise ValueError("rgb_reference_max_delta must be in (0, 1]")

    @property
    def token_rows(self) -> int:
        return self.image_height // 16

    @property
    def token_cols(self) -> int:
        return self.image_width // 16

    @property
    def image_token_count(self) -> int:
        return self.token_rows * self.token_cols
