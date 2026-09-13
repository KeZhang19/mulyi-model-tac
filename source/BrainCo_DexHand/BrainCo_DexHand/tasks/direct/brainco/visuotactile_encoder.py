# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local frozen image encoders for direct visuotactile observations."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class FrozenFiveFingerResNet18(nn.Module):
    """Encode five ordered tactile images into one normalized feature vector."""

    def __init__(
        self,
        *,
        output_dim: int,
        input_rows: int,
        input_cols: int,
        chunk_size: int,
        weights_path: str = "",
        use_imagenet_weights: bool = True,
    ):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        configured_path = Path(weights_path).expanduser().resolve() if weights_path.strip() else None
        if configured_path is not None:
            if not configured_path.is_file():
                raise FileNotFoundError(f"Tactile ResNet-18 weights do not exist: {configured_path}")
            model = resnet18(weights=None)
            state = torch.load(configured_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)
            self.weight_source = str(configured_path)
        elif use_imagenet_weights:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.weight_source = str(ResNet18_Weights.DEFAULT)
        else:
            model = resnet18(weights=None)
            self.weight_source = "random-frozen"

        self.backbone = nn.Sequential(*tuple(model.children())[:-1])
        self.output_dim = max(1, int(output_dim))
        self.input_rows = max(1, int(input_rows))
        self.input_cols = max(1, int(input_cols))
        self.chunk_size = max(1, int(chunk_size))
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        """Keep the pretrained BatchNorm statistics frozen."""

        del mode
        return super().train(False)

    def forward(self, finger_images: torch.Tensor) -> torch.Tensor:
        if finger_images.ndim != 5 or int(finger_images.shape[2]) not in (1, 3):
            raise ValueError(
                "Tactile ResNet expects [env,finger,1|3,row,col], "
                f"got {tuple(finger_images.shape)}"
            )

        num_envs, finger_count, channels, rows, cols = (int(value) for value in finger_images.shape)
        flat_images = finger_images.reshape(num_envs * finger_count, channels, rows, cols)
        encoded_chunks = []
        use_amp = flat_images.device.type == "cuda"
        for start in range(0, int(flat_images.shape[0]), self.chunk_size):
            images = flat_images[start : start + self.chunk_size].to(dtype=torch.float32)
            if channels == 1:
                images = images.expand(-1, 3, -1, -1)
            if tuple(images.shape[-2:]) != (self.input_rows, self.input_cols):
                images = F.interpolate(
                    images,
                    size=(self.input_rows, self.input_cols),
                    mode="bilinear",
                    align_corners=False,
                )
            images = (images - self.imagenet_mean) / self.imagenet_std
            with torch.autocast(
                device_type=flat_images.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                features = self.backbone(images).flatten(1)
            encoded_chunks.append(features.to(dtype=torch.float32))

        ordered_finger_features = torch.cat(encoded_chunks, dim=0).reshape(num_envs, finger_count * 512)
        reduced = F.adaptive_avg_pool1d(
            ordered_finger_features.unsqueeze(1),
            self.output_dim,
        ).squeeze(1)
        return F.normalize(reduced, p=2.0, dim=-1, eps=1.0e-8)
