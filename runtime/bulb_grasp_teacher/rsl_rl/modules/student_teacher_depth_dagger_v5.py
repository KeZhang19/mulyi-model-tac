# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DEXTRAH-style recurrent depth student for Flexiv online distillation."""

from __future__ import annotations

import glob
import math
import os
import warnings
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import ResNet18_Weights, resnet18

from rsl_rl.networks import EmpiricalNormalization, MLP


V5_DEPTH_OUTPUT_SIZE = (120, 212)
V5_LEGACY_DEPTH_OUTPUT_SIZE = (120, 160)
V5_SUPPORTED_DEPTH_OUTPUT_SIZES = (
    V5_DEPTH_OUTPUT_SIZE,
    V5_LEGACY_DEPTH_OUTPUT_SIZE,
)
V5_INPUT_CONTRACT_SCHEMA_VERSION = 1
# This version covers the V5 student's action-space contract: both its raw
# action output and the previous-action component of its low-dimensional input.
V5_ACTION_OUTPUT_CONTRACT_VERSION = 3
V5_POLICY_DEPTH_RANGE_M = (0.3, 1.5)
V2_TEST_POLICY_DEPTH_RANGE_M = (0.01, 1.2)
V5_DEPTH_RESIZE_MODE = "validity_weighted_bilinear_full_frame"
DUAL_DEPTH_INPUT_CONTRACT_SCHEMA_VERSION = 2
DUAL_DEPTH_CAMERA_ORDER = ("261822074231", "317222071012")
DUAL_DEPTH_CHANNELS = ("normalized_metric_depth", "validity_mask")
DUAL_DEPTH_FUSION_MODE = "shared_resnet18_cross_only_attention"
DUAL_DEPTH_TOKENS_PER_VIEW = 28
DUAL_DEPTH_TOKEN_DIM = 128
DUAL_DEPTH_ATTENTION_LAYERS = 2
DUAL_DEPTH_ATTENTION_HEADS = 4
DUAL_DEPTH_ATTENTION_DROPOUT = 0.1
DUAL_DEPTH_VISUAL_FEATURE_DIM = 64
DUAL_DEPTH_RIGHT_VIEW_TRANSFORM = "none"
VTGEN_TACTILE_INPUT_CONTRACT_SCHEMA_VERSION = 1
VTGEN_TACTILE_CHECKPOINT_SHA256 = (
    "72807954570690b8b6f6285f16627450f982c0d94f6607bce67eee33684267d4"
)
VTGEN_TACTILE_ARCHITECTURE = "joint_replaces_position"
VTGEN_TACTILE_IMAGE_SIZE = (128, 128)
VTGEN_TACTILE_HISTORY_FRAMES = 3
VTGEN_TACTILE_JOINT_DIM = 28
VTGEN_TACTILE_FEATURE_DIM = 64


def _validated_depth_range(
    depth_range_m,
    *,
    name: str,
) -> tuple[float, float]:
    try:
        values = tuple(float(value) for value in depth_range_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain two numbers") from exc
    if (
        len(values) != 2
        or not all(math.isfinite(value) for value in values)
        or not values[0] < values[1]
    ):
        raise ValueError(f"{name} must contain two finite increasing numbers")
    return values


def make_v5_input_contract(
    depth_output_size: tuple[int, int],
    *,
    depth_range_m=V5_POLICY_DEPTH_RANGE_M,
) -> dict[str, object]:
    """Build the serializable depth-input contract stored in V5 checkpoints."""

    output_size = tuple(int(value) for value in depth_output_size)
    if output_size not in V5_SUPPORTED_DEPTH_OUTPUT_SIZES:
        raise ValueError(
            f"unsupported V5 depth output size {output_size}; "
            f"expected one of {V5_SUPPORTED_DEPTH_OUTPUT_SIZES}."
        )
    depth_range = _validated_depth_range(
        depth_range_m,
        name="depth_range_m",
    )
    return {
        "schema_version": V5_INPUT_CONTRACT_SCHEMA_VERSION,
        "depth_shape": [2, *output_size],
        "depth_range_m": list(depth_range),
        "resize_mode": V5_DEPTH_RESIZE_MODE,
    }


def v5_checkpoint_depth_shape(
    checkpoint_payload: Mapping[str, object],
    *,
    expected_depth_range_m=V5_POLICY_DEPTH_RANGE_M,
    allow_missing_contract: bool = True,
) -> tuple[int, int, int]:
    """Read a strict V5 input contract, treating metadata-free checkpoints as legacy."""

    expected_depth_range = _validated_depth_range(
        expected_depth_range_m,
        name="expected_depth_range_m",
    )
    infos = checkpoint_payload.get("infos")
    if infos is None or (isinstance(infos, Mapping) and "v5_input_contract" not in infos):
        if not allow_missing_contract:
            raise ValueError("checkpoint is missing required v5_input_contract metadata")
        return (2, *V5_LEGACY_DEPTH_OUTPUT_SIZE)
    if not isinstance(infos, Mapping):
        raise ValueError("V5 checkpoint infos must be a mapping or null")
    contract = infos.get("v5_input_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("V5 checkpoint infos.v5_input_contract must be a mapping")
    schema_version = contract.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or schema_version != V5_INPUT_CONTRACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "V5 checkpoint input-contract schema_version must be "
            f"{V5_INPUT_CONTRACT_SCHEMA_VERSION}"
        )
    raw_depth_shape = contract.get("depth_shape", ())
    if not isinstance(raw_depth_shape, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_depth_shape
    ):
        raise ValueError("V5 checkpoint depth_shape must contain three integers")
    depth_shape = tuple(raw_depth_shape)
    supported_shapes = tuple((2, *size) for size in V5_SUPPORTED_DEPTH_OUTPUT_SIZES)
    if depth_shape not in supported_shapes:
        raise ValueError(
            f"V5 checkpoint depth_shape must be one of {supported_shapes}, got {depth_shape}"
        )
    depth_range = _validated_depth_range(
        contract.get("depth_range_m", ()),
        name="checkpoint depth_range_m",
    )
    if depth_range != expected_depth_range:
        raise ValueError(
            "checkpoint depth_range_m must match deployment: "
            f"expected={expected_depth_range}, got={depth_range}"
        )
    if contract.get("resize_mode") != V5_DEPTH_RESIZE_MODE:
        raise ValueError(
            f"V5 checkpoint resize_mode must be {V5_DEPTH_RESIZE_MODE!r}"
        )
    return depth_shape


def make_dual_depth_input_contract(
    depth_output_size: tuple[int, int],
    *,
    depth_range_m=V2_TEST_POLICY_DEPTH_RANGE_M,
) -> dict[str, object]:
    """Build the strict, ordered two-camera input contract."""

    output_size = tuple(int(value) for value in depth_output_size)
    if output_size != V5_DEPTH_OUTPUT_SIZE:
        raise ValueError(
            f"dual depth output size must be {V5_DEPTH_OUTPUT_SIZE}, got {output_size}"
        )
    depth_range = _validated_depth_range(depth_range_m, name="depth_range_m")
    return {
        "schema_version": DUAL_DEPTH_INPUT_CONTRACT_SCHEMA_VERSION,
        "camera_order": list(DUAL_DEPTH_CAMERA_ORDER),
        "per_camera_depth_shape": [2, *output_size],
        "combined_depth_shape": [2, 2, *output_size],
        "channels": list(DUAL_DEPTH_CHANNELS),
        "depth_range_m": list(depth_range),
        "resize_mode": V5_DEPTH_RESIZE_MODE,
        "fusion_mode": DUAL_DEPTH_FUSION_MODE,
        "per_camera_token_shape": [
            DUAL_DEPTH_TOKENS_PER_VIEW,
            DUAL_DEPTH_TOKEN_DIM,
        ],
        "attention": {
            "mode": "cross_only",
            "layers": DUAL_DEPTH_ATTENTION_LAYERS,
            "heads": DUAL_DEPTH_ATTENTION_HEADS,
            "dropout": DUAL_DEPTH_ATTENTION_DROPOUT,
        },
        "fused_visual_dim": DUAL_DEPTH_VISUAL_FEATURE_DIM,
        "right_view_transform": DUAL_DEPTH_RIGHT_VIEW_TRANSFORM,
    }


def dual_depth_checkpoint_shape(
    checkpoint_payload: Mapping[str, object],
) -> tuple[int, int, int, int]:
    """Validate and return the ordered dual-camera checkpoint input shape."""

    infos = checkpoint_payload.get("infos")
    if not isinstance(infos, Mapping):
        raise ValueError("dual-camera checkpoint infos must be a mapping")
    contract = infos.get("dual_depth_input_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(
            "dual-camera checkpoint is missing required dual_depth_input_contract metadata"
        )
    expected = make_dual_depth_input_contract(V5_DEPTH_OUTPUT_SIZE)
    actual = dict(contract)
    if actual != expected:
        raise ValueError(
            "dual-camera checkpoint input contract is incompatible: "
            f"checkpoint={actual!r}, expected={expected!r}"
        )
    return tuple(expected["combined_depth_shape"])


def make_vtgen_tactile_input_contract(
    checkpoint_sha256: str = VTGEN_TACTILE_CHECKPOINT_SHA256,
    *,
    tactile_feature_dim: int = VTGEN_TACTILE_FEATURE_DIM,
) -> dict[str, object]:
    """Build the strict frozen-generator and trainable-fusion contract."""

    digest = str(checkpoint_sha256).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("VT-Gen checkpoint SHA-256 must be a 64-character hex digest")
    if int(tactile_feature_dim) != VTGEN_TACTILE_FEATURE_DIM:
        raise ValueError(
            f"VT-Gen tactile feature dim must be {VTGEN_TACTILE_FEATURE_DIM}, "
            f"got {tactile_feature_dim}"
        )
    return {
        "schema_version": VTGEN_TACTILE_INPUT_CONTRACT_SCHEMA_VERSION,
        "generator": {
            "architecture": VTGEN_TACTILE_ARCHITECTURE,
            "checkpoint_sha256": digest,
            "frozen": True,
            "embedded_in_student_checkpoint": False,
        },
        "history_order": "oldest_to_newest",
        "history_frames": VTGEN_TACTILE_HISTORY_FRAMES,
        "depth_source": "primary_front_camera_only",
        "depth_history_shape": [VTGEN_TACTILE_HISTORY_FRAMES, 2, *V5_DEPTH_OUTPUT_SIZE],
        "joint_history_shape": [VTGEN_TACTILE_HISTORY_FRAMES, VTGEN_TACTILE_JOINT_DIM],
        "preprocessing": {
            "normalized_depth_range": [-1.0, 1.0],
            "validity_channel": 1,
            "invalid_pixel_value": 0.0,
            "uint8_semantics": "round((depth+1)*127.5)/255",
            "resize": "pil_semantic_bilinear_128x128_uint8",
            "rgb_mode": "repeat_grayscale_three_channels",
            "generator_visual_shape": [9, *VTGEN_TACTILE_IMAGE_SIZE],
        },
        "generator_output_shape": [3, *VTGEN_TACTILE_IMAGE_SIZE],
        "policy_tactile_shape": [1, *VTGEN_TACTILE_IMAGE_SIZE],
        "policy_output_channel": "last",
        "policy_pixel_range": [0.0, 1.0],
        "tactile_encoder": "imagenet_resnet18_single_channel",
        "tactile_feature_dim": VTGEN_TACTILE_FEATURE_DIM,
        "fusion_order": ["student_low_dim_123", "stereo_visual_64", "tactile_64"],
        "lstm_current_input_dim": 251,
        "auxiliary_object_position_input": "stereo_visual_64_only",
    }


def validate_vtgen_tactile_input_contract(
    contract: object,
    *,
    checkpoint_sha256: str = VTGEN_TACTILE_CHECKPOINT_SHA256,
    tactile_feature_dim: int = VTGEN_TACTILE_FEATURE_DIM,
) -> dict[str, object]:
    """Reject missing, legacy stereo, or incompatible tactile checkpoints."""

    expected = make_vtgen_tactile_input_contract(
        checkpoint_sha256,
        tactile_feature_dim=tactile_feature_dim,
    )
    if not isinstance(contract, Mapping):
        raise ValueError(
            "predicted-tactile student checkpoint is missing required "
            "vtgen_tactile_input_contract metadata; an old stereo student "
            "cannot warm-start or resume this task"
        )
    actual = dict(contract)
    if actual != expected:
        raise ValueError(
            "predicted-tactile student checkpoint contract is incompatible: "
            f"checkpoint={actual!r}, expected={expected!r}"
        )
    return actual


class DepthMaskResNet18(nn.Module):
    """ImageNet-initialized ResNet18 adapted to depth, mask, and optional conditioning channels."""

    def __init__(
        self,
        output_dim: int = 128,
        load_pretrained: bool = True,
        input_channels: int = 2,
    ):
        super().__init__()
        input_channels = int(input_channels)
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        backbone = resnet18(weights=None)
        # Deployment restores every encoder parameter from the V5 checkpoint.
        # Let that path explicitly skip the ImageNet lookup so an offline robot
        # process can never attempt a network download during startup.
        state_dict = self._load_pretrained_state_dict() if load_pretrained else None
        if state_dict is not None:
            backbone.load_state_dict(state_dict)

        rgb_conv = backbone.conv1
        depth_mask_conv = nn.Conv2d(
            input_channels,
            rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if state_dict is not None:
                depth_mask_conv.weight[:, :1].copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
                depth_mask_conv.weight[:, 1:].zero_()
            else:
                nn.init.kaiming_normal_(depth_mask_conv.weight, mode="fan_out", nonlinearity="relu")
                depth_mask_conv.weight[:, 1:].zero_()
        backbone.conv1 = depth_mask_conv
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(),
        )
        self.output_dim = int(output_dim)
        self.input_channels = input_channels
        self.pretrained_loaded = state_dict is not None

    @staticmethod
    def _load_pretrained_state_dict() -> dict[str, torch.Tensor] | None:
        checkpoint_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
        candidates = sorted(glob.glob(os.path.join(checkpoint_dir, "resnet18-*.pth")))
        if candidates:
            try:
                return torch.load(candidates[0], map_location="cpu", weights_only=True)
            except RuntimeError as exc:
                if "legacy .tar format" not in str(exc):
                    raise
                # Official older torchvision ResNet checkpoints used the legacy
                # serialization container and cannot be opened in weights-only mode.
                return torch.load(candidates[0], map_location="cpu", weights_only=False)
        try:
            return ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        except Exception as exc:  # pragma: no cover - depends on external cache/network state
            warnings.warn(
                f"Could not load ImageNet ResNet18 weights; v5 will initialize the encoder from scratch: {exc}",
                RuntimeWarning,
            )
            return None

    def forward(self, depth_mask: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(depth_mask))


class DepthMaskResNet18TokenBackbone(nn.Module):
    """Shared two-channel ResNet18 that preserves the native 4 x 7 token grid."""

    def __init__(
        self,
        token_dim: int = DUAL_DEPTH_TOKEN_DIM,
        load_pretrained: bool = True,
    ):
        super().__init__()
        if int(token_dim) != DUAL_DEPTH_TOKEN_DIM:
            raise ValueError(
                f"stereo depth token_dim must be {DUAL_DEPTH_TOKEN_DIM}, got {token_dim}"
            )
        backbone = resnet18(weights=None)
        state_dict = (
            DepthMaskResNet18._load_pretrained_state_dict()
            if load_pretrained
            else None
        )
        if state_dict is not None:
            backbone.load_state_dict(state_dict)

        rgb_conv = backbone.conv1
        depth_mask_conv = nn.Conv2d(
            2,
            rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if state_dict is not None:
                depth_mask_conv.weight[:, :1].copy_(
                    rgb_conv.weight.mean(dim=1, keepdim=True)
                )
            else:
                nn.init.kaiming_normal_(
                    depth_mask_conv.weight[:, :1],
                    mode="fan_out",
                    nonlinearity="relu",
                )
            depth_mask_conv.weight[:, 1:].zero_()
        backbone.conv1 = depth_mask_conv
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        # Keep one ResNet instance for both cameras. Forward stops at layer4,
        # before the classifier's global pooling and fully-connected layer.
        self.backbone = backbone
        self.token_projection = nn.Sequential(
            nn.Conv2d(512, DUAL_DEPTH_TOKEN_DIM, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.token_dim = DUAL_DEPTH_TOKEN_DIM
        self.tokens_per_view = DUAL_DEPTH_TOKENS_PER_VIEW
        self.native_feature_shape = (4, 7)
        self.pretrained_loaded = state_dict is not None

    def forward(self, depth_mask: torch.Tensor) -> torch.Tensor:
        if depth_mask.ndim != 4 or tuple(depth_mask.shape[1:]) != (
            2,
            *V5_DEPTH_OUTPUT_SIZE,
        ):
            raise ValueError(
                "stereo depth view must have shape "
                f"[B,2,{V5_DEPTH_OUTPUT_SIZE[0]},{V5_DEPTH_OUTPUT_SIZE[1]}], "
                f"got {tuple(depth_mask.shape)}"
            )
        x = self.backbone.conv1(depth_mask)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        if tuple(x.shape[-2:]) != self.native_feature_shape:
            raise RuntimeError(
                "ResNet18 layer4 must produce a native 4x7 grid for 120x212 input, "
                f"got {tuple(x.shape[-2:])}"
            )
        x = self.token_projection(x)
        return x.flatten(2).transpose(1, 2)


class CrossOnlyTransformerBlock(nn.Module):
    """Pre-norm transformer block using an externally supplied visibility mask."""

    def __init__(
        self,
        token_dim: int = DUAL_DEPTH_TOKEN_DIM,
        num_heads: int = DUAL_DEPTH_ATTENTION_HEADS,
        dropout: float = DUAL_DEPTH_ATTENTION_DROPOUT,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 4, token_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        tokens = tokens + self.attention_dropout(attended)
        return tokens + self.mlp(self.mlp_norm(tokens))


class StereoDepthTransformerEncoder(nn.Module):
    """Fuse ordered depth views with shared ResNet tokens and cross-only attention."""

    def __init__(
        self,
        token_dim: int = DUAL_DEPTH_TOKEN_DIM,
        output_dim: int = DUAL_DEPTH_VISUAL_FEATURE_DIM,
        depth_encode_batch_size: int = 32,
        use_gradient_checkpointing: bool = False,
        load_pretrained_backbone: bool = True,
    ):
        super().__init__()
        if int(token_dim) != DUAL_DEPTH_TOKEN_DIM:
            raise ValueError(
                f"stereo depth token_dim must be {DUAL_DEPTH_TOKEN_DIM}, got {token_dim}"
            )
        if int(output_dim) != DUAL_DEPTH_VISUAL_FEATURE_DIM:
            raise ValueError(
                "stereo fused visual output must be "
                f"{DUAL_DEPTH_VISUAL_FEATURE_DIM}, got {output_dim}"
            )
        self.shared_backbone = DepthMaskResNet18TokenBackbone(
            token_dim=token_dim,
            load_pretrained=load_pretrained_backbone,
        )
        self.primary_position_embedding = nn.Parameter(
            torch.empty(1, DUAL_DEPTH_TOKENS_PER_VIEW, DUAL_DEPTH_TOKEN_DIM)
        )
        self.secondary_position_embedding = nn.Parameter(
            torch.empty(1, DUAL_DEPTH_TOKENS_PER_VIEW, DUAL_DEPTH_TOKEN_DIM)
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, DUAL_DEPTH_TOKEN_DIM))
        self.transformer_blocks = nn.ModuleList(
            CrossOnlyTransformerBlock()
            for _ in range(DUAL_DEPTH_ATTENTION_LAYERS)
        )
        self.output_norm = nn.LayerNorm(DUAL_DEPTH_TOKEN_DIM)
        self.output_projection = nn.Sequential(
            nn.Linear(DUAL_DEPTH_TOKEN_DIM, DUAL_DEPTH_TOKEN_DIM),
            nn.GELU(),
            nn.Linear(DUAL_DEPTH_TOKEN_DIM, DUAL_DEPTH_VISUAL_FEATURE_DIM),
        )
        self.register_buffer(
            "cross_only_attention_mask",
            self._make_cross_only_attention_mask(),
            persistent=True,
        )
        nn.init.trunc_normal_(self.primary_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.secondary_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.token_dim = DUAL_DEPTH_TOKEN_DIM
        self.tokens_per_view = DUAL_DEPTH_TOKENS_PER_VIEW
        self.output_dim = DUAL_DEPTH_VISUAL_FEATURE_DIM
        self.depth_encode_batch_size = max(1, int(depth_encode_batch_size))
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    @staticmethod
    def _make_cross_only_attention_mask() -> torch.Tensor:
        """Return a boolean MHA mask where True entries are forbidden."""

        tokens_per_view = DUAL_DEPTH_TOKENS_PER_VIEW
        total_tokens = 1 + 2 * tokens_per_view
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool)
        # The CLS token exchanges information with every token.
        mask[0, :] = False
        mask[:, 0] = False
        primary = slice(1, 1 + tokens_per_view)
        secondary = slice(1 + tokens_per_view, total_tokens)
        # Image tokens can only exchange information with the other camera.
        mask[primary, secondary] = False
        mask[secondary, primary] = False
        return mask

    def _encode_chunk(self, stereo_depth: torch.Tensor) -> torch.Tensor:
        batch_size = stereo_depth.shape[0]
        view_tokens = self.shared_backbone(
            stereo_depth.reshape(batch_size * 2, 2, *V5_DEPTH_OUTPUT_SIZE)
        ).reshape(batch_size, 2, DUAL_DEPTH_TOKENS_PER_VIEW, DUAL_DEPTH_TOKEN_DIM)
        primary = view_tokens[:, 0] + self.primary_position_embedding
        secondary = view_tokens[:, 1] + self.secondary_position_embedding
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls, primary, secondary), dim=1)
        for block in self.transformer_blocks:
            tokens = block(tokens, self.cross_only_attention_mask)
        return self.output_projection(self.output_norm(tokens[:, 0]))

    def forward(self, stereo_depth: torch.Tensor) -> torch.Tensor:
        expected_shape = (2, 2, *V5_DEPTH_OUTPUT_SIZE)
        if stereo_depth.ndim != 5 or tuple(stereo_depth.shape[1:]) != expected_shape:
            raise ValueError(
                f"stereo depth must have shape [B,{','.join(map(str, expected_shape))}], "
                f"got {tuple(stereo_depth.shape)}"
            )
        outputs = []
        for start in range(0, stereo_depth.shape[0], self.depth_encode_batch_size):
            depth_batch = stereo_depth[start : start + self.depth_encode_batch_size]
            if (
                self.training
                and self.use_gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                encoded = checkpoint(
                    self._encode_chunk,
                    depth_batch,
                    use_reentrant=False,
                )
            else:
                encoded = self._encode_chunk(depth_batch)
            outputs.append(encoded)
        return torch.cat(outputs, dim=0)


class DextrahStereoDepthTransformerStudent(nn.Module):
    """DEXTRAH-style ordered stereo student with visual/current-input actor skip."""

    def __init__(
        self,
        low_dim_obs_dim: int,
        num_actions: int,
        visual_feature_dim: int = DUAL_DEPTH_TOKEN_DIM,
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 1,
        actor_hidden_dims: list[int] | None = None,
        auxiliary_hidden_dim: int = 256,
        activation: str = "elu",
        depth_encode_batch_size: int = 32,
        use_gradient_checkpointing: bool = False,
        load_pretrained_backbone: bool = True,
        auxiliary_input_source: str = "visual",
        depth_view_count: int = 2,
        tactile_feature_dim: int = 0,
        tactile_image_size: tuple[int, int] = VTGEN_TACTILE_IMAGE_SIZE,
    ):
        super().__init__()
        actor_hidden_dims = actor_hidden_dims or [512, 512, 256]
        required = {
            "low_dim_obs_dim": (int(low_dim_obs_dim), 123),
            "num_actions": (int(num_actions), 28),
            "visual_feature_dim": (int(visual_feature_dim), DUAL_DEPTH_TOKEN_DIM),
            "rnn_hidden_dim": (int(rnn_hidden_dim), 512),
            "rnn_num_layers": (int(rnn_num_layers), 1),
            "auxiliary_hidden_dim": (int(auxiliary_hidden_dim), 256),
            "depth_view_count": (int(depth_view_count), 2),
        }
        for name, (actual, expected) in required.items():
            if actual != expected:
                raise ValueError(
                    f"stereo transformer {name} must be {expected}, got {actual}"
                )
        if list(actor_hidden_dims) != [512, 512, 256]:
            raise ValueError(
                "stereo transformer actor_hidden_dims must be [512, 512, 256]"
            )
        if auxiliary_input_source != "visual":
            raise ValueError("stereo transformer auxiliary head must use visual input")
        tactile_feature_dim = int(tactile_feature_dim)
        if tactile_feature_dim not in (0, VTGEN_TACTILE_FEATURE_DIM):
            raise ValueError(
                "stereo predicted-tactile feature dim must be either disabled (0) "
                f"or {VTGEN_TACTILE_FEATURE_DIM}, got {tactile_feature_dim}"
            )
        tactile_image_size = tuple(int(value) for value in tactile_image_size)
        if tactile_image_size != VTGEN_TACTILE_IMAGE_SIZE:
            raise ValueError(
                f"predicted tactile image size must be {VTGEN_TACTILE_IMAGE_SIZE}, "
                f"got {tactile_image_size}"
            )

        self.stereo_depth_encoder = StereoDepthTransformerEncoder(
            token_dim=DUAL_DEPTH_TOKEN_DIM,
            output_dim=DUAL_DEPTH_VISUAL_FEATURE_DIM,
            depth_encode_batch_size=depth_encode_batch_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            load_pretrained_backbone=load_pretrained_backbone,
        )
        self.tactile_encoder = (
            DepthMaskResNet18(
                output_dim=VTGEN_TACTILE_FEATURE_DIM,
                load_pretrained=load_pretrained_backbone,
                input_channels=1,
            )
            if tactile_feature_dim
            else None
        )
        current_input_dim = (
            123 + DUAL_DEPTH_VISUAL_FEATURE_DIM + tactile_feature_dim
        )
        self.memory = nn.LSTM(
            input_size=current_input_dim,
            hidden_size=512,
            num_layers=1,
        )
        self.memory_norm = nn.LayerNorm(512)
        self.actor = MLP(
            512 + current_input_dim,
            28,
            actor_hidden_dims,
            activation,
        )
        self.auxiliary_object_position = nn.Sequential(
            nn.Linear(DUAL_DEPTH_VISUAL_FEATURE_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 3),
        )
        self.auxiliary_input_source = "visual"
        self.depth_view_count = 2
        self.token_dim = DUAL_DEPTH_TOKEN_DIM
        self.tokens_per_view = DUAL_DEPTH_TOKENS_PER_VIEW
        self.visual_feature_dim = DUAL_DEPTH_VISUAL_FEATURE_DIM
        self.tactile_feature_dim = tactile_feature_dim
        self.tactile_image_size = tactile_image_size
        self.rnn_hidden_dim = 512
        self.rnn_num_layers = 1
        self.current_input_dim = current_input_dim

    @property
    def depth_encoder(self) -> StereoDepthTransformerEncoder:
        """Compatibility handle used by optimizer groups and diagnostics."""

        return self.stereo_depth_encoder

    def initial_hidden(self, batch_size: int, device, dtype=torch.float32):
        shape = (self.rnn_num_layers, batch_size, self.rnn_hidden_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        return self.stereo_depth_encoder(depth)

    def fused_and_visual_features(
        self,
        low_dim_obs: torch.Tensor,
        depth: torch.Tensor,
        tactile_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.encode_depth(depth)
        features = [low_dim_obs, visual]
        if self.tactile_encoder is not None:
            expected_shape = (1, *self.tactile_image_size)
            if (
                tactile_image is None
                or tactile_image.ndim != 4
                or tuple(tactile_image.shape[1:]) != expected_shape
            ):
                actual = None if tactile_image is None else tuple(tactile_image.shape)
                raise ValueError(
                    "predicted tactile must have shape "
                    f"[B,{','.join(map(str, expected_shape))}], got {actual}"
                )
            features.append(self.tactile_encoder(tactile_image))
        elif tactile_image is not None:
            raise ValueError("tactile image was provided to a stereo-only student")
        return torch.cat(features, dim=-1), visual

    def fused_features(
        self,
        low_dim_obs: torch.Tensor,
        depth: torch.Tensor,
        tactile_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused, _ = self.fused_and_visual_features(
            low_dim_obs,
            depth,
            tactile_image,
        )
        return fused

    def forward_step(
        self,
        low_dim_obs: torch.Tensor,
        depth: torch.Tensor,
        hidden_state=None,
        tactile_image: torch.Tensor | None = None,
    ):
        if hidden_state is None:
            hidden_state = self.initial_hidden(
                low_dim_obs.shape[0],
                low_dim_obs.device,
                low_dim_obs.dtype,
            )
        current_input, visual = self.fused_and_visual_features(
            low_dim_obs,
            depth,
            tactile_image,
        )
        recurrent, hidden_state = self.memory(current_input.unsqueeze(0), hidden_state)
        recurrent = self.memory_norm(recurrent.squeeze(0))
        actions = self.actor(torch.cat((recurrent, current_input), dim=-1))
        auxiliary = self.auxiliary_object_position(visual)
        return actions, auxiliary, hidden_state

    def forward_sequence(
        self,
        low_dim_obs: torch.Tensor,
        depth: torch.Tensor,
        dones: torch.Tensor,
        hidden_state=None,
        hidden_detach_mask: torch.Tensor | None = None,
        tactile_image: torch.Tensor | None = None,
    ):
        """Run a T x B sequence, resetting hidden state after terminal transitions."""

        sequence_length, batch_size = low_dim_obs.shape[:2]
        flat_low_dim = low_dim_obs.reshape(sequence_length * batch_size, -1)
        flat_depth = depth.reshape(sequence_length * batch_size, *depth.shape[2:])
        flat_tactile = (
            tactile_image.reshape(
                sequence_length * batch_size,
                *tactile_image.shape[2:],
            )
            if tactile_image is not None
            else None
        )
        flat_input, flat_visual = self.fused_and_visual_features(
            flat_low_dim,
            flat_depth,
            flat_tactile,
        )
        current_input = flat_input.reshape(sequence_length, batch_size, -1)
        visual = flat_visual.reshape(sequence_length, batch_size, -1)
        if hidden_state is None:
            hidden_state = self.initial_hidden(
                batch_size,
                low_dim_obs.device,
                low_dim_obs.dtype,
            )

        recurrent_outputs = []
        for step in range(sequence_length):
            recurrent, hidden_state = self.memory(
                current_input[step : step + 1],
                hidden_state,
            )
            recurrent_outputs.append(recurrent.squeeze(0))
            keep = (~dones[step].reshape(-1).bool()).to(
                low_dim_obs.dtype
            ).view(1, -1, 1)
            hidden_state = (
                hidden_state[0] * keep,
                hidden_state[1] * keep,
            )
            if hidden_detach_mask is not None:
                detach = hidden_detach_mask[step].reshape(-1).bool().view(1, -1, 1)
                hidden_state = tuple(
                    torch.where(detach, state.detach(), state)
                    for state in hidden_state
                )
        recurrent = self.memory_norm(torch.stack(recurrent_outputs, dim=0))
        actor_input = torch.cat((recurrent, current_input), dim=-1)
        return (
            self.actor(actor_input),
            self.auxiliary_object_position(visual),
            hidden_state,
        )


class DextrahDepthRecurrentStudent(nn.Module):
    """Single-frame visual encoder with raw-action and auxiliary prediction heads."""

    def __init__(
        self,
        low_dim_obs_dim: int,
        num_actions: int,
        visual_feature_dim: int = 128,
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 1,
        actor_hidden_dims: list[int] | None = None,
        auxiliary_hidden_dim: int = 256,
        activation: str = "elu",
        depth_encode_batch_size: int = 32,
        use_gradient_checkpointing: bool = False,
        load_pretrained_backbone: bool = True,
        auxiliary_input_source: str = "recurrent",
        depth_view_count: int = 1,
    ):
        super().__init__()
        actor_hidden_dims = actor_hidden_dims or [512, 512, 256]
        if auxiliary_input_source not in {"recurrent", "visual"}:
            raise ValueError(
                "auxiliary_input_source must be 'recurrent' or 'visual', "
                f"got {auxiliary_input_source!r}"
            )
        depth_view_count = int(depth_view_count)
        if depth_view_count not in (1, 2):
            raise ValueError(
                f"depth_view_count must be 1 or 2, got {depth_view_count}"
            )
        self.depth_encoder = DepthMaskResNet18(
            visual_feature_dim,
            load_pretrained=load_pretrained_backbone,
        )
        fused_visual_dim = visual_feature_dim * depth_view_count
        self.fusion_norm = nn.LayerNorm(low_dim_obs_dim + fused_visual_dim)
        self.memory = nn.LSTM(
            input_size=low_dim_obs_dim + fused_visual_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
        )
        self.memory_norm = nn.LayerNorm(rnn_hidden_dim)
        self.actor = MLP(
            rnn_hidden_dim,
            num_actions,
            actor_hidden_dims,
            activation,
        )
        auxiliary_input_dim = (
            fused_visual_dim
            if auxiliary_input_source == "visual"
            else rnn_hidden_dim
        )
        self.auxiliary_object_position = nn.Sequential(
            nn.Linear(auxiliary_input_dim, auxiliary_hidden_dim),
            nn.ELU(),
            nn.Linear(auxiliary_hidden_dim, 3),
        )
        self.auxiliary_input_source = auxiliary_input_source
        self.depth_view_count = depth_view_count
        self.visual_feature_dim = int(visual_feature_dim)
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.rnn_num_layers = int(rnn_num_layers)
        self.depth_encode_batch_size = max(1, int(depth_encode_batch_size))
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    def initial_hidden(self, batch_size: int, device, dtype=torch.float32):
        shape = (self.rnn_num_layers, batch_size, self.rnn_hidden_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if self.depth_view_count == 1:
            if depth.ndim != 4:
                raise ValueError(
                    f"single-camera depth must have shape [B,C,H,W], got {tuple(depth.shape)}"
                )
            merged_depth = depth
            batch_size = depth.shape[0]
        else:
            if depth.ndim != 5 or depth.shape[1] != self.depth_view_count:
                raise ValueError(
                    "dual-camera depth must have shape [B,2,C,H,W], "
                    f"got {tuple(depth.shape)}"
                )
            batch_size = depth.shape[0]
            merged_depth = depth.reshape(
                batch_size * self.depth_view_count,
                *depth.shape[2:],
            )
        outputs = []
        for start in range(0, merged_depth.shape[0], self.depth_encode_batch_size):
            depth_batch = merged_depth[start : start + self.depth_encode_batch_size]
            if self.training and self.use_gradient_checkpointing and torch.is_grad_enabled():
                encoded = checkpoint(self.depth_encoder, depth_batch, use_reentrant=False)
            else:
                encoded = self.depth_encoder(depth_batch)
            outputs.append(encoded)
        encoded = torch.cat(outputs, dim=0)
        if self.depth_view_count == 1:
            return encoded
        return encoded.reshape(batch_size, self.depth_view_count, -1).flatten(1)

    def fused_and_visual_features(
        self,
        low_dim_obs: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_features = self.encode_depth(depth)
        fused = self.fusion_norm(torch.cat((low_dim_obs, visual_features), dim=-1))
        return fused, visual_features

    def fused_features(self, low_dim_obs: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        fused, _ = self.fused_and_visual_features(low_dim_obs, depth)
        return fused

    def auxiliary_features(
        self,
        visual_features: torch.Tensor,
        recurrent_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.auxiliary_input_source == "visual":
            return visual_features
        return recurrent_features

    def forward_step(self, low_dim_obs: torch.Tensor, depth: torch.Tensor, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.initial_hidden(low_dim_obs.shape[0], low_dim_obs.device, low_dim_obs.dtype)
        fused, visual = self.fused_and_visual_features(low_dim_obs, depth)
        fused = fused.unsqueeze(0)
        recurrent, hidden_state = self.memory(fused, hidden_state)
        recurrent = self.memory_norm(recurrent.squeeze(0))
        auxiliary = self.auxiliary_object_position(
            self.auxiliary_features(visual, recurrent)
        )
        return self.actor(recurrent), auxiliary, hidden_state

    def forward_sequence(
        self,
        low_dim_obs: torch.Tensor,
        depth: torch.Tensor,
        dones: torch.Tensor,
        hidden_state=None,
        hidden_detach_mask: torch.Tensor | None = None,
    ):
        """Run a T x B sequence, resetting hidden state after terminal transitions."""

        sequence_length, batch_size = low_dim_obs.shape[:2]
        flat_low_dim = low_dim_obs.reshape(sequence_length * batch_size, -1)
        flat_depth = depth.reshape(sequence_length * batch_size, *depth.shape[2:])
        flat_fused, flat_visual = self.fused_and_visual_features(
            flat_low_dim,
            flat_depth,
        )
        fused = flat_fused.reshape(sequence_length, batch_size, -1)
        visual = flat_visual.reshape(sequence_length, batch_size, -1)
        if hidden_state is None:
            hidden_state = self.initial_hidden(batch_size, low_dim_obs.device, low_dim_obs.dtype)

        recurrent_outputs = []
        for step in range(sequence_length):
            recurrent, hidden_state = self.memory(fused[step : step + 1], hidden_state)
            recurrent_outputs.append(recurrent.squeeze(0))
            if step < sequence_length - 1:
                keep = (~dones[step].reshape(-1).bool()).to(low_dim_obs.dtype).view(1, -1, 1)
                hidden_state = (hidden_state[0] * keep, hidden_state[1] * keep)
            if hidden_detach_mask is not None:
                detach = hidden_detach_mask[step].reshape(-1).bool().view(1, -1, 1)
                hidden_state = tuple(
                    torch.where(detach, state.detach(), state)
                    for state in hidden_state
                )
        recurrent = self.memory_norm(torch.stack(recurrent_outputs, dim=0))
        auxiliary = self.auxiliary_object_position(
            self.auxiliary_features(visual, recurrent)
        )
        return self.actor(recurrent), auxiliary, hidden_state


class StudentTeacherDepthDaggerV5(nn.Module):
    """Recurrent visual student that emits raw actions in the PPO teacher's units."""

    is_recurrent = True
    uses_tbptt_distillation = True

    @staticmethod
    def _student_model_class():
        # Resolve the module symbol at construction time so downstream policy
        # variants and tests can select an independent student architecture.
        return DextrahDepthRecurrentStudent

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_low_dim_obs_group: str = "student_low_dim",
        student_depth_obs_group: str = "depth_image",
        student_secondary_depth_obs_group: str | None = None,
        student_tactile_image_obs_group: str | None = None,
        student_aux_obs_group: str = "student_aux",
        teacher_obs_groups: list[str] | None = None,
        student_obs_normalization: bool = True,
        teacher_obs_normalization: bool = True,
        teacher_hidden_dims: list[int] | None = None,
        actor_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        visual_feature_dim: int = 128,
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 1,
        auxiliary_hidden_dim: int = 256,
        auxiliary_loss_coefficient: float = 0.0,
        encoder_learning_rate: float = 1.0e-3,
        student_learning_rate: float = 1.0e-3,
        depth_output_size: tuple[int, int] = V5_DEPTH_OUTPUT_SIZE,
        depth_encode_batch_size: int = 32,
        use_gradient_checkpointing: bool = False,
        tactile_feature_dim: int = VTGEN_TACTILE_FEATURE_DIM,
        tactile_image_size: tuple[int, int] = VTGEN_TACTILE_IMAGE_SIZE,
        vtgen_tactile_checkpoint_sha256: str = VTGEN_TACTILE_CHECKPOINT_SHA256,
        auxiliary_input_source: str = "recurrent",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        student_hidden_dims: list[int] | None = None,
        enable_hybrid_rl: bool = False,
        critic_obs_groups: list[str] | None = None,
        critic_hidden_dims: list[int] | None = None,
        critic_obs_normalization: bool = True,
        rl_learning_rate: float = 2.0e-4,
        min_action_std: float = 0.05,
        max_action_std: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        # These are part of the common Isaac Lab policy config interface.  V5 is
        # deterministic and uses ``actor_hidden_dims`` for the student instead.
        del noise_std_type, student_hidden_dims
        if kwargs:
            print(f"StudentTeacherDepthDaggerV5 ignored unexpected arguments: {sorted(kwargs)}")
        teacher_hidden_dims = teacher_hidden_dims or [512, 256, 128]
        actor_hidden_dims = actor_hidden_dims or [512, 512, 256]
        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.student_low_dim_obs_group = student_low_dim_obs_group
        self.student_depth_obs_group = student_depth_obs_group
        self.student_secondary_depth_obs_group = student_secondary_depth_obs_group
        self.student_tactile_image_obs_group = student_tactile_image_obs_group
        self.student_aux_obs_group = student_aux_obs_group
        self.teacher_obs_groups = teacher_obs_groups or list(obs_groups["teacher"])
        self.enable_hybrid_rl = bool(enable_hybrid_rl)
        self.critic_obs_groups = critic_obs_groups or list(self.teacher_obs_groups)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.rl_learning_rate = float(rl_learning_rate)
        self.min_action_std = float(min_action_std)
        self.max_action_std = float(max_action_std)
        if not 0.0 < self.min_action_std <= self.max_action_std:
            raise ValueError(
                "hybrid action std bounds must satisfy 0 < min_action_std <= max_action_std"
            )
        self.auxiliary_loss_coefficient = float(auxiliary_loss_coefficient)
        self.encoder_learning_rate = float(encoder_learning_rate)
        self.student_learning_rate = float(student_learning_rate)
        self.depth_output_size = tuple(int(value) for value in depth_output_size)
        if self.depth_output_size not in V5_SUPPORTED_DEPTH_OUTPUT_SIZES:
            raise ValueError(
                f"v5 depth_output_size must be one of {V5_SUPPORTED_DEPTH_OUTPUT_SIZES}, "
                f"got {self.depth_output_size}."
            )
        self.depth_input_shape = (2, *self.depth_output_size)
        self.depth_view_count = 2 if student_secondary_depth_obs_group else 1
        self.combined_depth_input_shape = (
            self.depth_view_count,
            *self.depth_input_shape,
        )
        self.tactile_feature_dim = int(tactile_feature_dim)
        self.tactile_image_size = tuple(int(value) for value in tactile_image_size)
        self.vtgen_tactile_checkpoint_sha256 = str(
            vtgen_tactile_checkpoint_sha256
        ).lower()

        low_dim_shape = obs[student_low_dim_obs_group].shape
        depth_shape = obs[student_depth_obs_group].shape
        aux_shape = obs[student_aux_obs_group].shape
        if len(low_dim_shape) != 2 or low_dim_shape[-1] != 123:
            raise ValueError(f"v5 requires 123 low-dimensional observations, got {tuple(low_dim_shape)}.")
        if len(depth_shape) != 4 or tuple(depth_shape[1:]) != self.depth_input_shape:
            raise ValueError(
                f"v5 requires depth [N, {self.depth_input_shape[0]}, "
                f"{self.depth_input_shape[1]}, {self.depth_input_shape[2]}], "
                f"got {tuple(depth_shape)}."
            )
        if self.student_secondary_depth_obs_group is not None:
            secondary_depth_shape = obs[self.student_secondary_depth_obs_group].shape
            if (
                len(secondary_depth_shape) != 4
                or tuple(secondary_depth_shape[1:]) != self.depth_input_shape
            ):
                raise ValueError(
                    "dual-camera v2-test requires secondary depth "
                    f"[N, {self.depth_input_shape[0]}, {self.depth_input_shape[1]}, "
                    f"{self.depth_input_shape[2]}], got {tuple(secondary_depth_shape)}."
                )
        if self.student_tactile_image_obs_group is not None:
            if self.depth_view_count != 2:
                raise ValueError(
                    "predicted tactile is supported only by the dual-camera student"
                )
            tactile_shape = obs[self.student_tactile_image_obs_group].shape
            expected_tactile_shape = (1, *VTGEN_TACTILE_IMAGE_SIZE)
            if (
                len(tactile_shape) != 4
                or tuple(tactile_shape[1:]) != expected_tactile_shape
            ):
                raise ValueError(
                    "predicted tactile must be [N,1,128,128], got "
                    f"{tuple(tactile_shape)}."
                )
            # Constructing the contract also validates SHA and feature width.
            make_vtgen_tactile_input_contract(
                self.vtgen_tactile_checkpoint_sha256,
                tactile_feature_dim=self.tactile_feature_dim,
            )
        if len(aux_shape) != 2 or aux_shape[-1] != 3:
            raise ValueError(f"v5 requires object-position auxiliary targets [N, 3], got {tuple(aux_shape)}.")

        student_model_class = self._student_model_class()
        student_kwargs = dict(
            low_dim_obs_dim=123,
            num_actions=num_actions,
            visual_feature_dim=visual_feature_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            actor_hidden_dims=actor_hidden_dims,
            auxiliary_hidden_dim=auxiliary_hidden_dim,
            activation=activation,
            depth_encode_batch_size=depth_encode_batch_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            auxiliary_input_source=auxiliary_input_source,
            depth_view_count=self.depth_view_count,
        )
        if self.student_tactile_image_obs_group is not None:
            if student_model_class is not DextrahStereoDepthTransformerStudent:
                raise TypeError(
                    "predicted tactile requires DextrahStereoDepthTransformerStudent"
                )
            student_kwargs.update(
                tactile_feature_dim=self.tactile_feature_dim,
                tactile_image_size=self.tactile_image_size,
            )
        self.student = student_model_class(**student_kwargs)
        self.student_obs_normalization = bool(student_obs_normalization)
        self.student_obs_normalizer = (
            EmpiricalNormalization(123) if student_obs_normalization else nn.Identity()
        )

        num_teacher_obs = sum(int(obs[group].shape[-1]) for group in self.teacher_obs_groups)
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher_obs_normalization = bool(teacher_obs_normalization)
        self.teacher_obs_normalizer = (
            EmpiricalNormalization(num_teacher_obs) if teacher_obs_normalization else nn.Identity()
        )
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

        if self.enable_hybrid_rl:
            critic_hidden_dims = critic_hidden_dims or [512, 256, 128]
            num_critic_obs = sum(
                int(obs[group].shape[-1]) for group in self.critic_obs_groups
            )
            self.student_critic = MLP(
                num_critic_obs,
                1,
                critic_hidden_dims,
                activation,
            )
            self.student_critic_obs_normalizer = (
                EmpiricalNormalization(num_critic_obs)
                if self.critic_obs_normalization
                else nn.Identity()
            )
            initial_std = min(
                self.max_action_std,
                max(self.min_action_std, float(init_noise_std)),
            )
            self.student_log_std = nn.Parameter(
                torch.full((num_actions,), math.log(initial_std))
            )
        else:
            self.student_critic = None
            self.student_critic_obs_normalizer = None
            self.register_parameter("student_log_std", None)
        self.hybrid_teacher_rl_initialized = False

        # This buffer deliberately makes V5 student checkpoints with a
        # different action-space contract fail to load. Their tensors have the
        # same shapes, but bounded/scaled outputs or normalized/clipped
        # previous-action inputs would make silent reuse incorrect.
        self.register_buffer(
            "action_output_contract_version",
            torch.tensor(V5_ACTION_OUTPUT_CONTRACT_VERSION, dtype=torch.int64),
        )
        self.register_buffer("teacher_action_std", torch.ones(num_actions))
        self.register_buffer("teacher_action_weights", torch.ones(num_actions))
        self.register_buffer("student_action_std", torch.zeros(num_actions))
        self._hidden_state = None
        self._last_action_mean = torch.zeros(1, num_actions)
        self.last_action_loss = 0.0
        self.last_auxiliary_loss = 0.0

    @property
    def action_mean(self):
        return self._last_action_mean

    @property
    def action_std(self):
        if self.enable_hybrid_rl:
            return self.hybrid_action_std()
        return self.student_action_std

    @property
    def entropy(self):
        if self.enable_hybrid_rl:
            std = self.hybrid_action_std().expand_as(self._last_action_mean)
            return torch.distributions.Normal(self._last_action_mean, std).entropy().sum(-1)
        return torch.zeros(self._last_action_mean.shape[0], device=self._last_action_mean.device)

    def hybrid_action_std(self) -> torch.Tensor:
        if not self.enable_hybrid_rl or self.student_log_std is None:
            raise RuntimeError("hybrid RL is not enabled for this policy")
        return self.student_log_std.exp().clamp(
            self.min_action_std,
            self.max_action_std,
        )

    def hybrid_act(self, obs, stochastic: bool):
        """Advance the recurrent actor once and return raw PPO distribution data."""

        deterministic_action = self.act_inference(obs)
        raw_mean = getattr(self, "unclipped_action_mean", self._last_action_mean)
        std = self.hybrid_action_std().expand_as(raw_mean)
        distribution = torch.distributions.Normal(raw_mean, std)
        raw_action = distribution.sample() if stochastic else raw_mean
        executed_action = (
            self.effective_raw_action(raw_action)
            if stochastic and hasattr(self, "effective_raw_action")
            else deterministic_action
        )
        return (
            executed_action,
            raw_action,
            raw_mean,
            std,
            distribution.log_prob(raw_action).sum(dim=-1, keepdim=True),
        )

    def initialize_hidden_state(self, obs):
        batch_size = int(obs[self.student_low_dim_obs_group].shape[0])
        device = obs[self.student_low_dim_obs_group].device
        if self._hidden_state is None or self._hidden_state[0].shape[1] != batch_size:
            self._hidden_state = self.student.initial_hidden(batch_size, device)

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is not None:
            student_state = hidden_states[0] if len(hidden_states) == 2 else hidden_states
            self._hidden_state = student_state
            return
        if dones is None:
            self._hidden_state = None
            return
        if self._hidden_state is None:
            return
        keep = (~dones.reshape(-1).bool()).to(self._hidden_state[0].dtype).view(1, -1, 1)
        self._hidden_state = (self._hidden_state[0] * keep, self._hidden_state[1] * keep)

    def get_hidden_states(self):
        return self._hidden_state, None

    def detach_hidden_states(self, dones=None):
        if self._hidden_state is not None:
            self._hidden_state = tuple(state.detach() for state in self._hidden_state)

    def _depth_observation(self, obs):
        primary = obs[self.student_depth_obs_group].to(dtype=torch.float32)
        secondary_group = getattr(
            self, "student_secondary_depth_obs_group", None
        )
        if secondary_group is None:
            return primary
        secondary = obs[secondary_group].to(dtype=torch.float32)
        return torch.stack((primary, secondary), dim=primary.ndim - 3)

    def get_student_obs(self, obs):
        low_dim = self.student_obs_normalizer(obs[self.student_low_dim_obs_group])
        depth = self._depth_observation(obs)
        return low_dim, depth

    def _tactile_observation(self, obs) -> torch.Tensor | None:
        group = self.student_tactile_image_obs_group
        if group is None:
            return None
        return obs[group].to(dtype=torch.float32)

    def get_teacher_obs(self, obs):
        return torch.cat([obs[group] for group in self.teacher_obs_groups], dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[group] for group in self.critic_obs_groups], dim=-1)

    def evaluate_student_critic(self, obs):
        if not self.enable_hybrid_rl or self.student_critic is None:
            raise RuntimeError("hybrid RL critic is not enabled for this policy")
        critic_obs = self.get_critic_obs(obs)
        return self.student_critic(self.student_critic_obs_normalizer(critic_obs))

    def evaluate_student_critic_sequence(self, obs):
        critic_obs = self.get_critic_obs(obs)
        shape = critic_obs.shape
        values = self.student_critic(
            self.student_critic_obs_normalizer(critic_obs.reshape(-1, shape[-1]))
        )
        return values.reshape(*shape[:-1], 1)

    def act(self, obs):
        return self.act_inference(obs)

    def act_inference(self, obs):
        self.initialize_hidden_state(obs)
        low_dim, depth = self.get_student_obs(obs)
        tactile = self._tactile_observation(obs)
        raw_actions, _, self._hidden_state = self.student.forward_step(
            low_dim,
            depth,
            self._hidden_state,
            tactile_image=tactile,
        )
        self._last_action_mean = raw_actions
        return raw_actions

    def evaluate(self, obs):
        teacher_obs = self.teacher_obs_normalizer(self.get_teacher_obs(obs))
        with torch.no_grad():
            return self.teacher(teacher_obs)

    def sequence_predictions(
        self,
        obs,
        dones,
        initial_hidden=None,
        hidden_detach_mask: torch.Tensor | None = None,
    ):
        """Return student action and auxiliary predictions for a time-major sequence."""

        low_dim = obs[self.student_low_dim_obs_group]
        shape = low_dim.shape
        low_dim = self.student_obs_normalizer(low_dim.reshape(-1, shape[-1])).reshape(shape)
        depth = self._depth_observation(obs)
        tactile = self._tactile_observation(obs)
        student_actions, object_position, hidden = self.student.forward_sequence(
            low_dim,
            depth,
            dones,
            initial_hidden,
            hidden_detach_mask=hidden_detach_mask,
            tactile_image=tactile,
        )
        return student_actions, object_position, hidden

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if loss_mask is None:
            return F.mse_loss(prediction, target)
        mask = loss_mask.to(device=prediction.device, dtype=prediction.dtype)
        if tuple(mask.shape) != tuple(prediction.shape[: mask.ndim]):
            raise ValueError(
                f"loss_mask shape {tuple(mask.shape)} does not prefix prediction "
                f"shape {tuple(prediction.shape)}"
            )
        while mask.ndim < prediction.ndim:
            mask = mask.unsqueeze(-1)
        elements_per_item = prediction[0, 0].numel() if prediction.ndim >= 2 else 1
        denominator = mask.sum() * elements_per_item
        if not bool(denominator > 0):
            raise ValueError("loss_mask must select at least one target")
        return ((prediction - target).square() * mask).sum() / denominator

    def compute_sequence_loss(
        self,
        obs,
        teacher_actions,
        dones,
        initial_hidden=None,
        loss_mask: torch.Tensor | None = None,
        hidden_detach_mask: torch.Tensor | None = None,
    ):
        student_actions, object_position, _ = self.sequence_predictions(
            obs,
            dones,
            initial_hidden,
            hidden_detach_mask=hidden_detach_mask,
        )
        # Match the BrainCo1 DAgger objective: a direct, unweighted MSE in the
        # teacher's raw action space.
        action_loss = self._masked_mse(
            student_actions,
            teacher_actions,
            loss_mask,
        )
        if self.auxiliary_loss_coefficient:
            auxiliary_target = obs[self.student_aux_obs_group]
            auxiliary_loss = self._masked_mse(
                object_position,
                auxiliary_target,
                loss_mask,
            )
        else:
            auxiliary_loss = action_loss.new_zeros(())
        total_loss = action_loss + self.auxiliary_loss_coefficient * auxiliary_loss
        self.last_action_loss = float(action_loss.detach().item())
        self.last_auxiliary_loss = float(auxiliary_loss.detach().item())
        return total_loss, action_loss.detach(), auxiliary_loss.detach()

    def compute_behavior_loss(self, obs, teacher_actions, loss_fn=None):
        low_dim, depth = self.get_student_obs(obs)
        student_actions, object_position, _ = self.student.forward_step(
            low_dim,
            depth,
            tactile_image=self._tactile_observation(obs),
        )
        action_loss = F.mse_loss(student_actions, teacher_actions)
        if self.auxiliary_loss_coefficient:
            auxiliary_loss = F.mse_loss(object_position, obs[self.student_aux_obs_group])
        else:
            auxiliary_loss = action_loss.new_zeros(())
        return action_loss + self.auxiliary_loss_coefficient * auxiliary_loss

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            self.student_obs_normalizer.update(obs[self.student_low_dim_obs_group])
        if self.enable_hybrid_rl and self.critic_obs_normalization:
            self.student_critic_obs_normalizer.update(self.get_critic_obs(obs))

    def optimizer_parameters(self):
        encoder_parameters = list(self.student.depth_encoder.parameters())
        tactile_encoder = getattr(self.student, "tactile_encoder", None)
        if tactile_encoder is not None:
            encoder_parameters.extend(tactile_encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        other_parameters = [
            parameter
            for parameter in self.student.parameters()
            if id(parameter) not in encoder_ids
        ]
        parameter_groups = [
            {
                "params": encoder_parameters,
                "lr": self.encoder_learning_rate,
                "name": "encoder",
            },
            {
                "params": other_parameters,
                "lr": self.student_learning_rate,
                "name": "student",
            },
        ]
        if self.enable_hybrid_rl:
            parameter_groups.extend(
                [
                    {
                        "params": self.student_critic.parameters(),
                        "lr": self.rl_learning_rate,
                        "name": "critic",
                    },
                    {
                        "params": [self.student_log_std],
                        "lr": self.rl_learning_rate,
                        "name": "std",
                    },
                ]
            )
        return parameter_groups

    def _set_teacher_action_std(self, action_std: torch.Tensor):
        action_std = action_std.detach().to(self.teacher_action_std).abs().clamp_min(1.0e-6)
        if action_std.numel() == 1:
            action_std = action_std.expand_as(self.teacher_action_std)
        # Retain these buffers only so raw-action V5 checkpoints stay loadable.
        # The BrainCo1-matched MSE objective deliberately does not use them.
        weights = torch.ones_like(action_std)
        self.teacher_action_std.copy_(action_std)
        self.teacher_action_weights.copy_(weights)
        if self.enable_hybrid_rl:
            with torch.no_grad():
                self.student_log_std.copy_(
                    action_std.clamp(self.min_action_std, self.max_action_std).log()
                )

    def load_state_dict(self, state_dict, strict=True):
        if any(key.startswith("actor.") for key in state_dict):
            teacher_state = {
                key.removeprefix("actor."): value
                for key, value in state_dict.items()
                if key.startswith("actor.")
            }
            teacher_normalizer_state = {
                key.removeprefix("actor_obs_normalizer."): value
                for key, value in state_dict.items()
                if key.startswith("actor_obs_normalizer.")
            }
            self.teacher.load_state_dict(teacher_state, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_normalizer_state, strict=strict)
            if self.enable_hybrid_rl:
                critic_state = {
                    key.removeprefix("critic."): value
                    for key, value in state_dict.items()
                    if key.startswith("critic.")
                }
                critic_normalizer_state = {
                    key.removeprefix("critic_obs_normalizer."): value
                    for key, value in state_dict.items()
                    if key.startswith("critic_obs_normalizer.")
                }
                if not critic_state or not critic_normalizer_state:
                    raise ValueError(
                        "Hybrid DAgger-PPO requires critic.* and "
                        "critic_obs_normalizer.* in the teacher checkpoint."
                    )
                self.student_critic.load_state_dict(critic_state, strict=strict)
                self.student_critic_obs_normalizer.load_state_dict(
                    critic_normalizer_state,
                    strict=strict,
                )
            if "std" in state_dict:
                self._set_teacher_action_std(state_dict["std"])
            elif "log_std" in state_dict:
                self._set_teacher_action_std(state_dict["log_std"].exp())
            else:
                warnings.warn("Teacher checkpoint has no action std; v5 will use uniform action weights.")
            if self.enable_hybrid_rl:
                self.hybrid_teacher_rl_initialized = True
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        if any(key.startswith("student.") for key in state_dict):
            output_contract = state_dict.get("action_output_contract_version")
            if (
                not isinstance(output_contract, torch.Tensor)
                or output_contract.numel() != 1
                or int(output_contract.item()) != V5_ACTION_OUTPUT_CONTRACT_VERSION
            ):
                raise ValueError(
                    "This V5 student checkpoint uses an incompatible action-space contract "
                    "(legacy action output or normalized/clipped previous-action input). "
                    "It cannot be resumed or played; retrain V5 with the current raw teacher-action contract."
                )
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            self._hidden_state = None
            return True
        raise ValueError("state_dict does not contain student or teacher parameters")

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()
        return self

    def diagnostics(self):
        return {
            "v5_action_loss": self.last_action_loss,
            "v5_auxiliary_object_position_loss": self.last_auxiliary_loss,
            "v5_action_output_contract_version": float(self.action_output_contract_version.item()),
        }
