#!/usr/bin/env python3
"""Train single-observation degradation detection and cross-modal restoration."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from BrainCo_DexHand.tactile_representation import (  # noqa: E402
    CrossModalTactileNetworkCfg,
    RobustCrossModalTactileNetwork,
)
from BrainCo_DexHand.tactile_representation.training import (  # noqa: E402
    CrossModalLossCfg,
    EpisodeSplits,
    MODALITY_NAMES,
    NpzTactileDataset,
    TactileNormalization,
    compute_cross_modal_objective,
    load_complete_manifest,
    manifest_sha256,
    split_episode_ids,
)


DEGRADED_NAMES = MODALITY_NAMES
LEGACY_OBJECTIVE = "single_modality_degradation_restoration_v1"
PREDICTED_OBJECTIVE_V2 = "predicted_quality_soft_restoration_v2"
PREDICTED_OBJECTIVE = "clean_calibrated_predicted_quality_restoration_v3"
LATENT_SUFFICIENT_OBJECTIVE = "latent_sufficient_predicted_quality_restoration_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train RGB/Depth/Marker single-observation degradation detection and restoration."
        )
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/revo3_index_sweep_parallel_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/revo3_cross_modal_restoration_mild_v3")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    parser.add_argument(
        "--dataset-backend",
        choices=("auto", "mmap", "npz"),
        default="auto",
        help="Use a validated mmap cache when available, or explicitly select a backend.",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("fixed", "cosine"),
        default="fixed",
        help="Keep learning rate fixed, or apply warmup followed by cosine decay.",
    )
    parser.add_argument("--lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--depth-scale-mm", type=float, default=3.0)
    parser.add_argument("--marker-motion-scale-px", type=float, default=5.0)
    parser.add_argument(
        "--rgb-depth-spatial-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed clean Depth spatial tokens to the RGB residual decoder when available.",
    )
    parser.add_argument(
        "--depth-rgb-spatial-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed RGB spatial tokens to the Depth decoder when RGB is available.",
    )
    parser.add_argument(
        "--marker-image-spatial-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let calibrated Marker queries attend to RGB/Depth spatial tokens.",
    )
    parser.add_argument(
        "--min-degradation-severity",
        type=float,
        default=0.15,
        help="Minimum mild observation degradation severity in [0, 1].",
    )
    parser.add_argument(
        "--max-degradation-severity",
        type=float,
        default=0.45,
        help="Maximum mild observation degradation severity in [0, 1].",
    )
    parser.add_argument(
        "--clean-sample-probability",
        type=float,
        default=0.25,
        help="Fraction of training rows kept fully clean to calibrate no-restoration behavior.",
    )
    parser.add_argument(
        "--balance-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample contact and no-contact training rows with equal probability.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Stop after this many non-improving restoration epochs; zero disables early stopping.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--max-train-batches", type=int, default=None, help="Debug/smoke-test batch limit."
    )
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--image-base-channels", type=int, default=32)
    parser.add_argument("--decoder-base-channels", type=int, default=128)
    parser.add_argument("--marker-transformer-layers", type=int, default=2)
    parser.add_argument("--marker-summary-tokens", type=int, default=4)
    parser.add_argument("--fusion-layers", type=int, default=3)
    parser.add_argument("--ffn-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--reliability-temperature", type=float, default=1.0)
    parser.add_argument(
        "--cross-modal-reliability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate each sensor's quality by comparing it with the other sensors.",
    )
    parser.add_argument(
        "--predicted-soft-restoration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use predicted quality gates instead of the ground-truth degradation mask.",
    )
    parser.add_argument("--reliability-gate-threshold", type=float, default=0.9)
    parser.add_argument("--reliability-gate-temperature", type=float, default=0.05)
    parser.add_argument("--reliability-gate-floor", type=float, default=0.01)
    parser.add_argument(
        "--detach-reliability-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep reconstruction gradients from collapsing the reliability detector.",
    )
    parser.add_argument(
        "--rgb-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use each episode's unpressed RGB frame as a residual-decoder reference. "
            "Use --no-rgb-reference for compatibility with latent-only checkpoints."
        ),
    )
    parser.add_argument(
        "--marker-static-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep fixed marker x0/y0/valid as independent decoder context. "
            "Use --no-marker-static-context for old MarkerDecoder checkpoints."
        ),
    )

    parser.add_argument("--rgb-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--rgb-residual-loss-weight",
        type=float,
        default=0.75,
        help="Blend weight for the reference-change-aware RGB residual loss.",
    )
    parser.add_argument(
        "--rgb-change-threshold",
        type=float,
        default=0.02,
        help="Normalized per-pixel RGB change above which residual emphasis starts.",
    )
    parser.add_argument(
        "--rgb-change-boost",
        type=float,
        default=4.0,
        help="Maximum extra weight assigned to pixels changed from the RGB reference.",
    )
    parser.add_argument(
        "--rgb-residual-max-delta",
        type=float,
        default=0.5,
        help="Maximum normalized RGB change predicted on top of the unpressed reference.",
    )
    parser.add_argument("--depth-loss-weight", type=float, default=4.0)
    parser.add_argument("--depth-foreground-weight", type=float, default=4.0)
    parser.add_argument(
        "--depth-structure-weight",
        type=float,
        default=1.0,
        help="Weight of the sparse Depth foreground-mask loss.",
    )
    parser.add_argument(
        "--depth-foreground-threshold-mm",
        type=float,
        default=0.06,
        help="Depth above this physical value is treated as foreground for structure loss.",
    )
    parser.add_argument(
        "--depth-mask-temperature",
        type=float,
        default=0.02,
        help="Temperature in normalized Depth units for the differentiable foreground mask.",
    )
    parser.add_argument("--depth-mask-pos-weight", type=float, default=32.0)
    parser.add_argument("--marker-loss-weight", type=float, default=2.0)
    parser.add_argument("--marker-motion-weight", type=float, default=1.0)
    parser.add_argument("--marker-direction-weight", type=float, default=0.2)
    parser.add_argument("--empty-depth-weight", type=float, default=2.0)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--quality-weight", type=float, default=0.2)
    parser.add_argument("--clean-identity-weight", type=float, default=1.0)
    parser.add_argument(
        "--latent-sufficiency-weight",
        type=float,
        default=0.1,
        help=(
            "Weight for reconstructing all clean modalities from the shared latent "
            "with every dynamic spatial decoder bypass disabled."
        ),
    )
    parser.add_argument(
        "--clean-selection-weight",
        type=float,
        default=1.0,
        help="Weight of clean quality/identity loss in best-checkpoint selection.",
    )
    parser.add_argument(
        "--worst-modality-weight",
        type=float,
        default=0.5,
        help="Weight of the worst degraded-modality restoration in checkpoint selection.",
    )
    args = parser.parse_args()

    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "shard-cache-size": args.shard_cache_size,
        "learning-rate": args.learning_rate,
        "depth-scale-mm": args.depth_scale_mm,
        "marker-motion-scale-px": args.marker_motion_scale_px,
        "depth-mask-temperature": args.depth_mask_temperature,
        "depth-mask-pos-weight": args.depth_mask_pos_weight,
        "save-every": args.save_every,
        "grad-accum-steps": args.grad_accum_steps,
        "marker-summary-tokens": args.marker_summary_tokens,
    }
    invalid = {name: value for name, value in positive.items() if float(value) <= 0.0}
    if invalid:
        parser.error(f"These arguments must be positive: {invalid}")
    if (
        args.num_workers < 0
        or args.lr_warmup_epochs < 0
        or args.early_stopping_patience < 0
    ):
        parser.error("Worker and warmup counts must be non-negative")
    if (
        args.gradient_clip < 0.0
        or args.weight_decay < 0.0
        or args.worst_modality_weight < 0.0
        or args.depth_foreground_weight < 0.0
        or args.depth_structure_weight < 0.0
        or args.depth_foreground_threshold_mm < 0.0
        or args.depth_mask_temperature <= 0.0
        or args.depth_mask_pos_weight < 1.0
        or args.marker_direction_weight < 0.0
        or args.clean_identity_weight < 0.0
        or args.latent_sufficiency_weight < 0.0
        or args.clean_selection_weight < 0.0
    ):
        parser.error("Gradient, decay, and worst-modality weights must be non-negative")
    if not 0.0 <= float(args.rgb_residual_loss_weight) <= 1.0:
        parser.error("--rgb-residual-loss-weight must be in [0, 1]")
    if float(args.rgb_change_threshold) < 0.0 or float(args.rgb_change_boost) < 0.0:
        parser.error("RGB change threshold and boost must be non-negative")
    if not 0.0 < float(args.rgb_residual_max_delta) <= 1.0:
        parser.error("--rgb-residual-max-delta must be in (0, 1]")
    if not (
        0.0
        <= float(args.min_degradation_severity)
        <= float(args.max_degradation_severity)
        <= 1.0
    ):
        parser.error(
            "degradation severity must satisfy 0 <= --min-degradation-severity "
            "<= --max-degradation-severity <= 1"
        )
    if not 0.0 <= float(args.clean_sample_probability) < 1.0:
        parser.error("--clean-sample-probability must be in [0, 1)")
    if not 0.0 <= float(args.ema_decay) < 1.0:
        parser.error("--ema-decay must be in [0, 1)")
    if not 0.0 <= float(args.reliability_gate_threshold) <= 1.0:
        parser.error("--reliability-gate-threshold must be in [0, 1]")
    if float(args.reliability_gate_temperature) <= 0.0:
        parser.error("--reliability-gate-temperature must be positive")
    if not 0.0 <= float(args.reliability_gate_floor) < 1.0:
        parser.error("--reliability-gate-floor must be in [0, 1)")
    for name in ("max_train_batches", "max_val_batches", "max_test_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive when provided")
    if args.evaluate_only and args.resume is None:
        parser.error("--evaluate-only requires --resume")
    return args


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {value!r} was requested, but torch.cuda.is_available() is false. "
            "Use --device cpu only for smoke tests."
        )
    return device


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def model_config(args: argparse.Namespace, norm: TactileNormalization) -> CrossModalTactileNetworkCfg:
    return CrossModalTactileNetworkCfg(
        image_height=norm.image_height,
        image_width=norm.image_width,
        marker_count=norm.marker_count,
        marker_summary_tokens=int(args.marker_summary_tokens),
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        image_base_channels=int(args.image_base_channels),
        decoder_base_channels=int(args.decoder_base_channels),
        marker_transformer_layers=int(args.marker_transformer_layers),
        fusion_layers=int(args.fusion_layers),
        ffn_ratio=int(args.ffn_ratio),
        dropout=float(args.dropout),
        reliability_temperature=float(args.reliability_temperature),
        cross_modal_reliability=bool(args.cross_modal_reliability),
        predicted_soft_restoration=bool(args.predicted_soft_restoration),
        reliability_gate_threshold=float(args.reliability_gate_threshold),
        reliability_gate_temperature=float(args.reliability_gate_temperature),
        reliability_gate_floor=float(args.reliability_gate_floor),
        detach_reliability_gate=bool(args.detach_reliability_gate),
        rgb_reference_residual=bool(args.rgb_reference),
        rgb_reference_max_delta=float(args.rgb_residual_max_delta),
        rgb_depth_spatial_skip=bool(args.rgb_depth_spatial_skip),
        marker_static_context=bool(args.marker_static_context),
        depth_rgb_spatial_skip=bool(args.depth_rgb_spatial_skip),
        marker_image_spatial_context=bool(args.marker_image_spatial_context),
    )


def loss_config(args: argparse.Namespace) -> CrossModalLossCfg:
    return CrossModalLossCfg(
        rgb_weight=float(args.rgb_loss_weight),
        rgb_residual_loss_weight=float(args.rgb_residual_loss_weight),
        rgb_change_threshold=float(args.rgb_change_threshold),
        rgb_change_boost=float(args.rgb_change_boost),
        depth_weight=float(args.depth_loss_weight),
        depth_foreground_weight=float(args.depth_foreground_weight),
        depth_structure_weight=float(args.depth_structure_weight),
        depth_foreground_threshold=(
            float(args.depth_foreground_threshold_mm)
            / max(float(args.depth_scale_mm), 1.0e-8)
        ),
        depth_mask_temperature=float(args.depth_mask_temperature),
        depth_mask_pos_weight=float(args.depth_mask_pos_weight),
        marker_weight=float(args.marker_loss_weight),
        marker_motion_weight=float(args.marker_motion_weight),
        marker_direction_weight=float(args.marker_direction_weight),
        empty_depth_weight=float(args.empty_depth_weight),
        consistency_weight=float(args.consistency_weight),
        quality_weight=float(args.quality_weight),
        clean_identity_weight=float(args.clean_identity_weight),
        latent_sufficiency_weight=float(args.latent_sufficiency_weight),
    )


def _json_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved}")
    try:
        return torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(resolved, map_location="cpu")


def _make_loader(
    dataset: NpzTactileDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if bool(shuffle) and bool(args.balance_contact):
        labels = torch.tensor(dataset.contact_labels, dtype=torch.int64)
        counts = torch.bincount(labels, minlength=2)
        if bool((counts > 0).all()):
            class_weights = counts.sum().to(torch.float64) / (2.0 * counts.to(torch.float64))
            sample_weights = class_weights.index_select(0, labels)
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=bool(shuffle) and sampler is None,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        generator=generator,
        drop_last=False,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def _batch_limit(loader: DataLoader, requested: int | None) -> int:
    return min(len(loader), int(requested)) if requested is not None else len(loader)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


@torch.no_grad()
def _update_ema_teacher(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """Update the frozen clean-input teacher after one optimizer step."""

    student_parameters = dict(student.named_parameters())
    for name, teacher_parameter in teacher.named_parameters():
        teacher_parameter.mul_(float(decay)).add_(
            student_parameters[name].detach(), alpha=1.0 - float(decay)
        )
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        teacher_buffer.copy_(student_buffers[name].detach())


class MetricAccumulator:
    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.samples = 0

    def update(self, metrics: dict[str, torch.Tensor], batch_size: int) -> None:
        self.samples += int(batch_size)
        for name, value in metrics.items():
            scalar = float(value.detach().to(torch.float32).cpu())
            self.total[name] = self.total.get(name, 0.0) + scalar * int(batch_size)

    def averages(self) -> dict[str, float]:
        if self.samples <= 0:
            raise RuntimeError("No metrics were accumulated")
        return {name: value / self.samples for name, value in self.total.items()}


def train_epoch(
    model: nn.Module,
    teacher_model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: AdamW,
    scaler: Any,
    device: torch.device,
    normalization: TactileNormalization,
    objective_cfg: CrossModalLossCfg,
    min_degradation_severity: float,
    max_degradation_severity: float,
    clean_sample_probability: float,
    amp_enabled: bool,
    grad_accum_steps: int,
    gradient_clip: float,
    ema_decay: float,
    seed: int,
    max_batches: int | None,
) -> tuple[dict[str, float], int]:
    model.train()
    metrics = MetricAccumulator()
    generator = _generator(device, seed)
    limit = _batch_limit(loader, max_batches)
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= limit:
            break
        batch = _move_batch(cpu_batch, device)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast_context:
            loss, batch_metrics = compute_cross_modal_objective(
                model,
                batch,
                teacher_model=teacher_model,
                normalization=normalization,
                cfg=objective_cfg,
                min_degradation_severity=min_degradation_severity,
                max_degradation_severity=max_degradation_severity,
                clean_probability=clean_sample_probability,
                generator=generator,
            )
            group_start = (batch_index // int(grad_accum_steps)) * int(grad_accum_steps)
            group_size = min(int(grad_accum_steps), limit - group_start)
            scaled_loss = loss / group_size
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Training loss became non-finite at batch {batch_index}")
        scaler.scale(scaled_loss).backward()
        should_step = (batch_index + 1) % int(grad_accum_steps) == 0 or batch_index + 1 == limit
        if should_step:
            scaler.unscale_(optimizer)
            if float(gradient_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            scale_before_step = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_was_skipped = (
                bool(scaler.is_enabled())
                and float(scaler.get_scale()) < scale_before_step
            )
            if optimizer_step_was_skipped:
                skipped_optimizer_steps += 1
            else:
                _update_ema_teacher(teacher_model, model, ema_decay)
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
        metrics.update(batch_metrics, int(batch["rgb"].shape[0]))
    averages = metrics.averages()
    averages["metric/optimizer_skipped_steps"] = float(skipped_optimizer_steps)
    return averages, optimizer_steps


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    teacher_model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: TactileNormalization,
    objective_cfg: CrossModalLossCfg,
    amp_enabled: bool,
    seed: int,
    max_batches: int | None,
    min_degradation_severity: float,
    max_degradation_severity: float,
    worst_modality_weight: float,
    clean_selection_weight: float,
) -> dict[str, float]:
    """Evaluate each degraded modality plus the fully clean no-trigger case."""

    model.eval()
    teacher_model.eval()
    modality_accumulators = {name: MetricAccumulator() for name in DEGRADED_NAMES}
    clean_accumulator = MetricAccumulator()
    generator = _generator(device, seed)
    limit = _batch_limit(loader, max_batches)
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= limit:
            break
        batch = _move_batch(cpu_batch, device)
        batch_size = int(batch["rgb"].shape[0])
        for degraded_index, degraded_name in enumerate(DEGRADED_NAMES):
            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            ):
                loss, batch_metrics = compute_cross_modal_objective(
                    model,
                    batch,
                    teacher_model=teacher_model,
                    normalization=normalization,
                    cfg=objective_cfg,
                    degraded_modality=degraded_index,
                    min_degradation_severity=min_degradation_severity,
                    max_degradation_severity=max_degradation_severity,
                    clean_probability=0.0,
                    generator=generator,
                    report_oracle_upper_bound=bool(
                        getattr(
                            getattr(model, "cfg", None),
                            "predicted_soft_restoration",
                            False,
                        )
                    ),
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Evaluation loss became non-finite at batch {batch_index}, "
                    f"degraded_modality={degraded_name}"
                )
            modality_accumulators[degraded_name].update(batch_metrics, batch_size)
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        ):
            clean_loss, clean_metrics = compute_cross_modal_objective(
                model,
                batch,
                teacher_model=teacher_model,
                normalization=normalization,
                cfg=objective_cfg,
                degraded_modality=0,
                min_degradation_severity=min_degradation_severity,
                max_degradation_severity=max_degradation_severity,
                clean_probability=1.0,
                generator=generator,
                report_oracle_upper_bound=False,
            )
        if not bool(torch.isfinite(clean_loss)):
            raise FloatingPointError(
                f"Clean evaluation loss became non-finite at batch {batch_index}"
            )
        clean_accumulator.update(clean_metrics, batch_size)

    by_modality = {
        name: accumulator.averages() for name, accumulator in modality_accumulators.items()
    }
    clean_values = clean_accumulator.averages()
    aggregate_keys = [
        "loss/total",
        "loss/restoration",
        "loss/latent_sufficiency",
        "loss/latent_sufficiency_weighted",
        "loss/consistency",
        "loss/quality",
        "metric/quality_mae",
        "metric/degraded_quality_mae",
        "metric/degraded_modality_accuracy",
        "metric/degradation_severity",
        "metric/modalities_per_restoration",
    ]
    predicted_mode = bool(
        getattr(getattr(model, "cfg", None), "predicted_soft_restoration", False)
    )
    if predicted_mode:
        aggregate_keys.extend(
            (
                "loss/candidate_restoration",
                "loss/end_to_end_restoration",
                "loss/oracle_upper_bound_restoration",
                "metric/restoration_applied_fraction",
                "metric/restoration_blend",
            )
        )
    result = {
        key: sum(values[key] for values in by_modality.values()) / len(by_modality)
        for key in aggregate_keys
    }
    mean_total = sum(values["loss/total"] for values in by_modality.values()) / len(
        by_modality
    )
    mean_restoration = sum(
        values["loss/restoration"] for values in by_modality.values()
    ) / len(by_modality)
    worst_restoration = max(
        values["loss/restoration"] for values in by_modality.values()
    )
    mean_latent_sufficiency_weighted = sum(
        values["loss/latent_sufficiency_weighted"] for values in by_modality.values()
    ) / len(by_modality)
    if predicted_mode:
        selection_loss = (
            mean_restoration
            + float(worst_modality_weight) * worst_restoration
            + mean_latent_sufficiency_weighted
            + float(clean_selection_weight) * clean_values["loss/total"]
        )
    else:
        selection_loss = (
            mean_total
            + float(worst_modality_weight) * worst_restoration
        )
    result.update(
        {
            "loss/total": mean_total,
            "loss/restoration": mean_restoration,
            "loss/worst_modality_restoration": worst_restoration,
            "loss/latent_selection": mean_latent_sufficiency_weighted,
            "loss/clean_selection": clean_values["loss/total"],
            "loss/selection": selection_loss,
        }
    )
    for degraded_name, keys in {
        "rgb": (
            "loss/restore_rgb",
            "metric/rgb_mae",
            "metric/rgb_residual_mae",
            "metric/rgb_change_mae",
            "metric/rgb_psnr_db",
        ),
        "depth": (
            "loss/restore_depth",
            "loss/restore_depth_empty",
            "loss/restore_depth_structure",
            "metric/depth_mae_normalized",
            "metric/depth_mae_mm",
            "metric/depth_foreground_mae",
            "metric/depth_background_mae",
            "metric/depth_foreground_iou",
        ),
        "marker": (
            "loss/restore_marker",
            "metric/marker_motion_epe_px",
            "metric/marker_direction_error_rate",
        ),
    }.items():
        for key in keys:
            result[key] = by_modality[degraded_name][key]
    for degraded_name, values in by_modality.items():
        for key, value in values.items():
            result[f"degraded/{degraded_name}/{key}"] = value
    for key, value in clean_values.items():
        result[f"clean/{key}"] = value
    return result


def _make_scaler(enabled: bool) -> Any:
    # Sparse-Depth structure loss plus transformer paths can overflow the first
    # fp16 backward pass at PyTorch's default 65536 scale. Start conservatively
    # and let GradScaler grow automatically after stable optimizer updates.
    initial_scale = 4096.0
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=initial_scale)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=initial_scale)


def _lr_lambda(
    epoch: int,
    *,
    schedule: str,
    epochs: int,
    warmup_epochs: int,
) -> float:
    if schedule == "fixed":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {schedule!r}")
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    decay_epochs = max(1, epochs - warmup_epochs)
    progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(decay_epochs)))
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _make_scheduler(
    optimizer: AdamW,
    args: argparse.Namespace,
    *,
    last_epoch: int = -1,
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _lr_lambda(
            epoch,
            schedule=str(args.lr_schedule),
            epochs=int(args.epochs),
            warmup_epochs=int(args.lr_warmup_epochs),
        ),
        last_epoch=int(last_epoch),
    )


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    model: nn.Module,
    teacher_model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Any,
    network_cfg: CrossModalTactileNetworkCfg,
    objective_cfg: CrossModalLossCfg,
    normalization: TactileNormalization,
    splits: EpisodeSplits,
    dataset_fingerprint: str,
    args: argparse.Namespace,
    validation_metrics: dict[str, float],
    epochs_without_improvement: int,
    objective_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 7,
        "training_objective": objective_name,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
        "model_state": model.state_dict(),
        "teacher_model_state": teacher_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_cfg": asdict(network_cfg),
        "loss_cfg": asdict(objective_cfg),
        "normalization": normalization.to_json(),
        "episode_splits": splits.to_json(),
        "dataset_manifest_sha256": dataset_fingerprint,
        "args": _json_args(args),
        "validation_metrics": validation_metrics,
        "epochs_without_improvement": int(epochs_without_improvement),
    }


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    dataset_root, manifest = load_complete_manifest(args.dataset)
    fingerprint = manifest_sha256(dataset_root)
    normalization = TactileNormalization.from_manifest(
        manifest,
        depth_scale_m=float(args.depth_scale_mm) * 1.0e-3,
        marker_motion_scale_px=float(args.marker_motion_scale_px),
    )
    resume_payload = _load_checkpoint(args.resume) if args.resume is not None else None
    if resume_payload is not None:
        checkpoint_schema = int(resume_payload.get("schema_version", 1))
        if checkpoint_schema not in (4, 5, 6, 7):
            raise ValueError(
                "Checkpoint uses an incompatible training objective and cannot be resumed "
                "by this restoration trainer; start a new output directory"
            )
        if resume_payload.get("dataset_manifest_sha256") != fingerprint:
            raise ValueError("Resume checkpoint was created from a different dataset manifest")
        # Checkpoints written before the reference branch was introduced do not
        # contain its config key.  Reopen them in the original latent-only mode
        # instead of making old evaluation/resume commands fail.
        checkpoint_model_cfg = resume_payload.get("model_cfg", {})
        if "rgb_depth_spatial_skip" not in checkpoint_model_cfg:
            args.rgb_depth_spatial_skip = False
        if "depth_rgb_spatial_skip" not in checkpoint_model_cfg:
            args.depth_rgb_spatial_skip = False
        if "marker_image_spatial_context" not in checkpoint_model_cfg:
            args.marker_image_spatial_context = False
        if "rgb_reference_residual" not in checkpoint_model_cfg:
            args.rgb_reference = False
        if "marker_static_context" not in checkpoint_model_cfg:
            args.marker_static_context = False
        if "rgb_reference_max_delta" not in checkpoint_model_cfg:
            # All reference-residual checkpoints written before schema-v6 used
            # the original hard-coded +/-0.25 decoder range.
            args.rgb_residual_max_delta = 0.25
        if checkpoint_schema == 4:
            # Schema-v4 was trained with an oracle restoration mask and
            # self-only reliability. Preserve that exact architecture when an
            # old run is resumed or evaluated.
            args.cross_modal_reliability = False
            args.predicted_soft_restoration = False
            args.detach_reliability_gate = False
            args.reliability_gate_threshold = 0.9
            args.reliability_gate_temperature = 0.05
            args.reliability_gate_floor = 0.01
            # v4 used the uniformly averaged RGB image loss. Keep that exact
            # objective when an old checkpoint is resumed or evaluated.
            args.rgb_residual_loss_weight = 0.0
        checkpoint_loss_cfg = resume_payload.get("loss_cfg", {})
        if "rgb_residual_loss_weight" not in checkpoint_loss_cfg:
            # Early restoration checkpoints optimized absolute RGB only.
            args.rgb_residual_loss_weight = 0.0
        # The sparse Depth structure term was introduced after schema-v4 and
        # the first schema-v5 runs. Preserve those checkpoints' original
        # pixel-only objective when they are evaluated or resumed.
        if "depth_structure_weight" not in checkpoint_loss_cfg:
            args.depth_structure_weight = 0.0
            args.depth_foreground_threshold_mm = 0.0
        if "marker_direction_weight" not in checkpoint_loss_cfg:
            args.marker_direction_weight = 0.0
        if "clean_identity_weight" not in checkpoint_loss_cfg:
            args.clean_identity_weight = 0.0
        if "latent_sufficiency_weight" not in checkpoint_loss_cfg:
            # Schema-v6 and older checkpoints never decoded all three clean
            # targets from the shared latent alone.  Preserve their exact
            # objective for evaluation and strict resume compatibility.
            args.latent_sufficiency_weight = 0.0
        checkpoint_args = resume_payload.get("args", {})
        if "clean_sample_probability" not in checkpoint_args:
            args.clean_sample_probability = 0.0
        if "clean_selection_weight" not in checkpoint_args:
            args.clean_selection_weight = 0.0
        if not bool(args.evaluate_only):
            checkpoint_min_severity = float(
                checkpoint_args.get("min_degradation_severity", 0.2)
            )
            # Checkpoints written before mild bounded degradation had an
            # implicit maximum severity of 1.0.
            checkpoint_max_severity = float(
                checkpoint_args.get("max_degradation_severity", 1.0)
            )
            requested_bounds = (
                float(args.min_degradation_severity),
                float(args.max_degradation_severity),
            )
            checkpoint_bounds = (
                checkpoint_min_severity,
                checkpoint_max_severity,
            )
            if any(
                not math.isclose(old, new, rel_tol=0.0, abs_tol=1.0e-12)
                for old, new in zip(checkpoint_bounds, requested_bounds, strict=True)
            ):
                raise ValueError(
                    "Resume checkpoint used degradation severity bounds "
                    f"{checkpoint_bounds}, but this run requested {requested_bounds}. "
                    "Start a new output directory because validation selection losses "
                    "are not comparable across degradation distributions."
                )
            checkpoint_clean_probability = float(
                checkpoint_args.get("clean_sample_probability", 0.0)
            )
            checkpoint_clean_selection_weight = float(
                checkpoint_args.get("clean_selection_weight", 0.0)
            )
            if not math.isclose(
                checkpoint_clean_probability,
                float(args.clean_sample_probability),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                checkpoint_clean_selection_weight,
                float(args.clean_selection_weight),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "Resume checkpoint used clean calibration settings "
                    f"(probability={checkpoint_clean_probability}, "
                    f"selection_weight={checkpoint_clean_selection_weight}), but this run "
                    f"requested (probability={float(args.clean_sample_probability)}, "
                    f"selection_weight={float(args.clean_selection_weight)}). Start a new "
                    "output directory because checkpoint-selection losses are not comparable."
                )
        splits = EpisodeSplits.from_json(resume_payload["episode_splits"])
    else:
        splits = split_episode_ids(
            manifest,
            seed=int(args.seed),
            validation_fraction=float(args.validation_fraction),
            test_fraction=float(args.test_fraction),
        )

    output_dir = args.output.expanduser().resolve()
    if resume_payload is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory or --resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = NpzTactileDataset(
        dataset_root,
        episode_ids=splits.train,
        normalization=normalization,
        cache_size=int(args.shard_cache_size),
        backend=str(args.dataset_backend),
    )
    validation_dataset = NpzTactileDataset(
        dataset_root,
        episode_ids=splits.validation,
        normalization=normalization,
        cache_size=int(args.shard_cache_size),
        backend=str(args.dataset_backend),
    )
    test_dataset = NpzTactileDataset(
        dataset_root,
        episode_ids=splits.test,
        normalization=normalization,
        cache_size=int(args.shard_cache_size),
        backend=str(args.dataset_backend),
    )
    dataset_backends = {
        train_dataset.backend,
        validation_dataset.backend,
        test_dataset.backend,
    }
    if len(dataset_backends) != 1:
        raise RuntimeError(f"Dataset splits selected inconsistent backends: {dataset_backends}")
    dataset_backend = next(iter(dataset_backends))
    train_loader = _make_loader(
        train_dataset, args=args, device=device, shuffle=True, seed=int(args.seed)
    )
    validation_loader = _make_loader(
        validation_dataset, args=args, device=device, shuffle=False, seed=int(args.seed) + 1
    )
    test_loader = _make_loader(
        test_dataset, args=args, device=device, shuffle=False, seed=int(args.seed) + 2
    )

    network_cfg = model_config(args, normalization)
    objective_cfg = loss_config(args)
    model = RobustCrossModalTactileNetwork(network_cfg).to(device)
    teacher_model = copy.deepcopy(model).eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    optimizer = AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    scheduler = _make_scheduler(optimizer, args)
    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = _make_scaler(amp_enabled)
    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    if resume_payload is not None:
        checkpoint_model_cfg = dict(resume_payload["model_cfg"])
        checkpoint_model_cfg.setdefault("rgb_reference_residual", False)
        checkpoint_model_cfg.setdefault("marker_static_context", False)
        checkpoint_model_cfg.setdefault("cross_modal_reliability", False)
        checkpoint_model_cfg.setdefault("predicted_soft_restoration", False)
        checkpoint_model_cfg.setdefault("reliability_gate_threshold", 0.9)
        checkpoint_model_cfg.setdefault("reliability_gate_temperature", 0.05)
        checkpoint_model_cfg.setdefault("reliability_gate_floor", 0.01)
        checkpoint_model_cfg.setdefault("detach_reliability_gate", False)
        checkpoint_model_cfg.setdefault("rgb_depth_spatial_skip", False)
        checkpoint_model_cfg.setdefault("depth_rgb_spatial_skip", False)
        checkpoint_model_cfg.setdefault("marker_image_spatial_context", False)
        checkpoint_model_cfg.setdefault("rgb_reference_max_delta", 0.25)
        if checkpoint_model_cfg != asdict(network_cfg):
            raise ValueError("Resume checkpoint model configuration does not match CLI arguments")
        checkpoint_loss_cfg = dict(resume_payload.get("loss_cfg", {}))
        current_loss_cfg = asdict(objective_cfg)
        changed_loss_cfg = {
            name: {"checkpoint": value, "requested": current_loss_cfg[name]}
            for name, value in checkpoint_loss_cfg.items()
            if name in current_loss_cfg
            and not math.isclose(
                float(value),
                float(current_loss_cfg[name]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        }
        if changed_loss_cfg:
            raise ValueError(
                "Resume checkpoint loss configuration does not match CLI arguments: "
                f"{changed_loss_cfg}"
            )
        if resume_payload["normalization"] != normalization.to_json():
            raise ValueError("Resume checkpoint normalization does not match CLI arguments")
        model.load_state_dict(resume_payload["model_state"], strict=True)
        teacher_model.load_state_dict(resume_payload["teacher_model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scaler.load_state_dict(resume_payload.get("scaler_state", {}))
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload.get("global_step", 0))
        best_validation_loss = float(resume_payload.get("best_validation_loss", math.inf))
        epochs_without_improvement = int(resume_payload.get("epochs_without_improvement", 0))
        original_args = resume_payload.get("args", {})
        original_epochs = int(original_args.get("epochs", args.epochs))
        original_schedule = str(original_args.get("lr_schedule", "cosine"))
        original_learning_rate = float(
            original_args.get("learning_rate", args.learning_rate)
        )
        scheduler_is_compatible = (
            original_epochs == int(args.epochs)
            and original_schedule == str(args.lr_schedule)
            and math.isclose(
                original_learning_rate,
                float(args.learning_rate),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        )
        if scheduler_is_compatible:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        else:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = float(args.learning_rate)
                parameter_group["initial_lr"] = float(args.learning_rate)
            scheduler = _make_scheduler(optimizer, args, last_epoch=start_epoch - 1)

    if resume_payload is not None and resume_payload.get("training_objective"):
        objective_name = str(resume_payload["training_objective"])
    elif (
        resume_payload is not None
        and int(resume_payload.get("schema_version", 1)) == 5
        and network_cfg.predicted_soft_restoration
    ):
        objective_name = PREDICTED_OBJECTIVE_V2
    elif (
        network_cfg.predicted_soft_restoration
        and float(objective_cfg.latent_sufficiency_weight) > 0.0
    ):
        objective_name = LATENT_SUFFICIENT_OBJECTIVE
    elif network_cfg.predicted_soft_restoration:
        objective_name = PREDICTED_OBJECTIVE
    else:
        objective_name = LEGACY_OBJECTIVE
    run_summary = {
        "training_objective": objective_name,
        "dataset": str(dataset_root),
        "dataset_manifest_sha256": fingerprint,
        "dataset_backend": dataset_backend,
        "dataset_samples": int(manifest["sample_count"]),
        "split_episodes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "split_samples": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "rgb_reference_fallback_episodes": sorted(
            set(train_dataset.rgb_reference_fallback_episodes)
            | set(validation_dataset.rgb_reference_fallback_episodes)
            | set(test_dataset.rgb_reference_fallback_episodes)
        ),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_cfg": asdict(network_cfg),
        "loss_cfg": asdict(objective_cfg),
        "normalization": normalization.to_json(),
        "episode_splits": splits.to_json(),
        "args": _json_args(args),
    }
    config_name = (
        "evaluation_config.json"
        if args.evaluate_only
        else "resume_config.json" if resume_payload is not None else "run_config.json"
    )
    _write_json_atomic(output_dir / config_name, run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)

    if args.evaluate_only:
        test_metrics = evaluate_epoch(
            model,
            teacher_model,
            test_loader,
            device=device,
            normalization=normalization,
            objective_cfg=objective_cfg,
            amp_enabled=amp_enabled,
            seed=int(args.seed) + 20_000,
            max_batches=args.max_test_batches,
            min_degradation_severity=float(args.min_degradation_severity),
            max_degradation_severity=float(args.max_degradation_severity),
            worst_modality_weight=float(args.worst_modality_weight),
            clean_selection_weight=float(args.clean_selection_weight),
        )
        _write_json_atomic(output_dir / "evaluation.json", test_metrics)
        print("[test] " + json.dumps(test_metrics, sort_keys=True), flush=True)
        return

    if start_epoch >= int(args.epochs):
        raise ValueError(
            f"Checkpoint has already completed epoch {start_epoch}; increase --epochs to continue"
        )
    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, int(args.epochs)):
        epoch_start = time.monotonic()
        train_metrics, optimizer_steps = train_epoch(
            model,
            teacher_model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            normalization=normalization,
            objective_cfg=objective_cfg,
            min_degradation_severity=float(args.min_degradation_severity),
            max_degradation_severity=float(args.max_degradation_severity),
            clean_sample_probability=float(args.clean_sample_probability),
            amp_enabled=amp_enabled,
            grad_accum_steps=int(args.grad_accum_steps),
            gradient_clip=float(args.gradient_clip),
            ema_decay=float(args.ema_decay),
            seed=int(args.seed) + epoch * 101,
            max_batches=args.max_train_batches,
        )
        global_step += optimizer_steps
        validation_metrics = evaluate_epoch(
            model,
            teacher_model,
            validation_loader,
            device=device,
            normalization=normalization,
            objective_cfg=objective_cfg,
            amp_enabled=amp_enabled,
            seed=int(args.seed) + 10_000,
            max_batches=args.max_val_batches,
            min_degradation_severity=float(args.min_degradation_severity),
            max_degradation_severity=float(args.max_degradation_severity),
            worst_modality_weight=float(args.worst_modality_weight),
            clean_selection_weight=float(args.clean_selection_weight),
        )
        validation_loss = float(validation_metrics["loss/selection"])
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        if optimizer_steps > 0:
            scheduler.step()
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "training_objective": objective_name,
            "learning_rate": current_learning_rate,
            "seconds": time.monotonic() - epoch_start,
            "train": train_metrics,
            "validation": validation_metrics,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        with history_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, sort_keys=True) + "\n")
        payload = _checkpoint_payload(
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            model=model,
            teacher_model=teacher_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            network_cfg=network_cfg,
            objective_cfg=objective_cfg,
            normalization=normalization,
            splits=splits,
            dataset_fingerprint=fingerprint,
            args=args,
            validation_metrics=validation_metrics,
            epochs_without_improvement=epochs_without_improvement,
            objective_name=objective_name,
        )
        _atomic_torch_save(output_dir / "last.pt", payload)
        if improved:
            _atomic_torch_save(output_dir / "best.pt", payload)
        if (epoch + 1) % int(args.save_every) == 0:
            _atomic_torch_save(output_dir / f"epoch_{epoch + 1:04d}.pt", payload)
        print(
            f"[epoch {epoch + 1:03d}/{int(args.epochs):03d}] mode=restoration "
            f"train={train_metrics['loss/total']:.6f} "
            f"val={validation_loss:.6f} best={best_validation_loss:.6f} "
            f"lr={current_learning_rate:.3e} seconds={record['seconds']:.1f}",
            flush=True,
        )
        if (
            int(args.early_stopping_patience) > 0
            and epochs_without_improvement >= int(args.early_stopping_patience)
        ):
            print(
                f"[early-stop] validation did not improve for "
                f"{epochs_without_improvement} restoration epochs",
                flush=True,
            )
            break

    best_payload = _load_checkpoint(output_dir / "best.pt")
    model.load_state_dict(best_payload["model_state"], strict=True)
    teacher_model.load_state_dict(best_payload["teacher_model_state"], strict=True)
    test_metrics = evaluate_epoch(
        model,
        teacher_model,
        test_loader,
        device=device,
        normalization=normalization,
        objective_cfg=objective_cfg,
        amp_enabled=amp_enabled,
        seed=int(args.seed) + 20_000,
        max_batches=args.max_test_batches,
        min_degradation_severity=float(args.min_degradation_severity),
        max_degradation_severity=float(args.max_degradation_severity),
        worst_modality_weight=float(args.worst_modality_weight),
        clean_selection_weight=float(args.clean_selection_weight),
    )
    final_summary = {
        "best_epoch": int(best_payload["epoch"]) + 1,
        "best_epoch_index": int(best_payload["epoch"]),
        "best_validation_loss": float(best_payload["best_validation_loss"]),
        "test": test_metrics,
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
    }
    _write_json_atomic(output_dir / "evaluation.json", final_summary)
    print("[done] " + json.dumps(final_summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
