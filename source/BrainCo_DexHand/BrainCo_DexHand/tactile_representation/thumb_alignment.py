"""Isolated SDK-native thumb input contract and explicit weight initialization."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from .policy import build_tactile_encoder, checkpoint_model_type

THUMB_INPUT_CONTRACT = {
    "version": "revo3_thumb_sdk_180x320_no_padding_v2",
    "finger": "thumb",
    "camera_serial": "WTUVL2198X260001A",
    "image_height": 180,
    "image_width": 320,
    "marker_count": 100,
    "input_spatial_transform": "none: no resize, padding or crop",
    "encoder_spatial_sizes": [[180, 320], [90, 160], [45, 80], [23, 40], [12, 20]],
    "decoder_spatial_sizes": [[12, 20], [23, 40], [45, 80], [90, 160], [180, 320]],
}


def validate_thumb_checkpoint(source: dict) -> None:
    cfg, norm = source["model_cfg"], source["normalization"]
    for key in ("image_height", "image_width", "marker_count"):
        if cfg[key] != THUMB_INPUT_CONTRACT[key] or norm[key] != THUMB_INPUT_CONTRACT[key]:
            raise ValueError(f"Thumb {key} must be {THUMB_INPUT_CONTRACT[key]} in model and normalization")
    if "pad_to_stride" in cfg or cfg.get("native_image_size") is not True:
        raise ValueError("Thumb encoder requires native_image_size=True; padded checkpoints are obsolete")
    if source.get("thumb_input_contract") != THUMB_INPUT_CONTRACT:
        raise ValueError("Thumb checkpoint must carry the current no-padding input contract")
    if checkpoint_model_type(source) not in ("robust", "tri_modal"):
        raise ValueError("Thumb CLIP supports the original robust/tri_modal towers")


def initialize_thumb_checkpoint(source_path: Path, output_path: Path) -> dict:
    """Transfer old convolution weights; interpolate only spatial parameters.

    This creates a new initialization, never resumes the old optimizer or
    declares physical calibration or cross-domain alignment.
    """
    source_path, output_path = Path(source_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    kind = checkpoint_model_type(source)
    if kind not in ("robust", "tri_modal"):
        raise ValueError("Thumb initialization supports robust/tri_modal checkpoints only")
    old_cfg = source["model_cfg"]
    if (old_cfg["image_height"], old_cfg["image_width"], old_cfg["marker_count"]) != (240, 320, 100):
        raise ValueError("Expected an original 240x320, 100-marker source checkpoint")
    if "pad_to_stride" in old_cfg:
        raise ValueError("Initialize from the original checkpoint, not the obsolete padded attempt")
    cfg = dict(old_cfg, image_height=180, image_width=320, native_image_size=True)
    norm = dict(source["normalization"], image_height=180, image_width=320)
    if (source["normalization"]["image_height"], source["normalization"]["image_width"],
            source["normalization"]["marker_count"]) != (240, 320, 100):
        raise ValueError("Source model and normalization disagree")
    model = build_tactile_encoder(kind, cfg)
    expected = model.state_dict()
    spatial = {"rgb_encoder.position_embedding", "depth_encoder.position_embedding",
               "rgb_decoder.seed", "depth_decoder.seed"}
    state, changed = {}, []
    if source["model_state"].keys() != expected.keys():
        raise ValueError("Source and thumb model state keys differ")
    for name, value in source["model_state"].items():
        target = expected[name]
        if value.shape != target.shape:
            if name not in spatial:
                raise ValueError(f"Unexpected thumb weight shape change: {name}")
            if name.endswith("position_embedding"):
                grid = value.transpose(1, 2).reshape(1, value.shape[-1], 15, 20)
                value = F.interpolate(grid, (12, 20), mode="bilinear", align_corners=False)
                value = value.flatten(2).transpose(1, 2)
            else:
                value = F.interpolate(value, (12, 20), mode="bilinear", align_corners=False)
            changed.append(name)
        state[name] = value.detach().cpu()
    model.load_state_dict(state, strict=True)
    result = {"schema_version": 2, "network_type": kind,
              "training_objective": "thumb_native_no_padding_initialization_v2",
              "model_cfg": cfg, "normalization": norm, "model_state": state,
              "epoch": 0, "initialization_only": True,
              "thumb_input_contract": deepcopy(THUMB_INPUT_CONTRACT),
              "initialization": {"source": str(source_path.resolve()), "source_sha256": digest,
                                 "interpolated_parameters": changed,
                                 "optimizer_restored": False, "alignment_validated": False},
              "args": {"initialization_from": str(source_path.resolve())}}
    validate_thumb_checkpoint(result)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source checkpoint changed during initialization")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return result["initialization"]
