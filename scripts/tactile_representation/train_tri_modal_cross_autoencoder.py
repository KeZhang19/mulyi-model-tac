#!/usr/bin/env python3
"""Train the independent RGB/Depth/Marker cross-attention autoencoder."""

from __future__ import annotations

import argparse
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
    TriModalCrossAutoencoder,
    TriModalCrossAutoencoderCfg,
)
from BrainCo_DexHand.tactile_representation.training import (  # noqa: E402
    EpisodeSplits,
    NpzTactileDataset,
    TactileNormalization,
    load_complete_manifest,
    manifest_sha256,
    split_episode_ids,
)
from BrainCo_DexHand.tactile_representation.training.tri_modal_objectives import (  # noqa: E402
    TriModalAutoencoderLossCfg,
    compute_tri_modal_autoencoder_objective,
)


NETWORK_TYPE = "tri_modal_cross_autoencoder"
OBJECTIVE_NAME = "single_degradation_all_modality_denoising_v1"
EVALUATION_MODES = ("clean", "rgb", "depth", "marker")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train three modality encoders, peer cross-attention, and three clean decoders."
        )
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/revo3_index_sweep_parallel_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/revo3_tri_modal_cross_v1")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    parser.add_argument(
        "--dataset-backend", choices=("auto", "mmap", "npz"), default="auto"
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lr-schedule", choices=("fixed", "cosine"), default="fixed")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--depth-scale-mm", type=float, default=3.0)
    parser.add_argument("--marker-motion-scale-px", type=float, default=5.0)
    parser.add_argument("--min-degradation-severity", type=float, default=0.15)
    parser.add_argument("--max-degradation-severity", type=float, default=0.45)
    parser.add_argument("--clean-sample-probability", type=float, default=0.25)
    parser.add_argument(
        "--balance-contact", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--image-base-channels", type=int, default=32)
    parser.add_argument("--decoder-base-channels", type=int, default=128)
    parser.add_argument("--marker-transformer-layers", type=int, default=2)
    parser.add_argument("--marker-summary-tokens", type=int, default=4)
    parser.add_argument("--cross-layers", type=int, default=2)
    parser.add_argument("--cross-summary-tokens", type=int, default=4)
    parser.add_argument("--ffn-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--rgb-loss-weight", type=float, default=1.0)
    parser.add_argument("--depth-loss-weight", type=float, default=1.0)
    parser.add_argument("--marker-loss-weight", type=float, default=1.0)
    parser.add_argument("--rgb-reference-weight", type=float, default=0.75)
    parser.add_argument("--rgb-change-threshold", type=float, default=0.02)
    parser.add_argument("--rgb-change-boost", type=float, default=4.0)
    parser.add_argument("--depth-foreground-weight", type=float, default=4.0)
    parser.add_argument("--depth-structure-weight", type=float, default=0.25)
    parser.add_argument("--depth-foreground-threshold-mm", type=float, default=0.06)
    parser.add_argument("--depth-mask-temperature", type=float, default=0.02)
    parser.add_argument("--depth-mask-pos-weight", type=float, default=32.0)
    parser.add_argument("--marker-direction-weight", type=float, default=0.1)
    parser.add_argument("--marker-direction-threshold-px", type=float, default=0.5)
    parser.add_argument("--marker-prediction-floor-px", type=float, default=0.25)
    parser.add_argument("--worst-modality-weight", type=float, default=0.5)
    parser.add_argument("--clean-selection-weight", type=float, default=0.25)

    args = parser.parse_args(argv)
    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "grad-accum-steps": args.grad_accum_steps,
        "shard-cache-size": args.shard_cache_size,
        "learning-rate": args.learning_rate,
        "depth-scale-mm": args.depth_scale_mm,
        "marker-motion-scale-px": args.marker_motion_scale_px,
        "save-every": args.save_every,
        "log-every-batches": args.log_every_batches,
        "d-model": args.d_model,
        "num-heads": args.num_heads,
        "image-base-channels": args.image_base_channels,
        "decoder-base-channels": args.decoder_base_channels,
        "marker-transformer-layers": args.marker_transformer_layers,
        "marker-summary-tokens": args.marker_summary_tokens,
        "cross-layers": args.cross_layers,
        "cross-summary-tokens": args.cross_summary_tokens,
        "ffn-ratio": args.ffn_ratio,
    }
    invalid = {name: value for name, value in positive.items() if float(value) <= 0.0}
    if invalid:
        parser.error(f"These arguments must be positive: {invalid}")
    if args.num_workers < 0 or args.lr_warmup_epochs < 0 or args.early_stopping_patience < 0:
        parser.error("Worker, warmup, and patience counts must be non-negative")
    if args.gradient_clip < 0.0 or args.weight_decay < 0.0:
        parser.error("gradient-clip and weight-decay must be non-negative")
    if not 0.0 <= args.min_degradation_severity <= args.max_degradation_severity <= 1.0:
        parser.error("degradation severity bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= args.clean_sample_probability <= 1.0:
        parser.error("clean-sample-probability must be in [0, 1]")
    if args.evaluate_only and args.resume is None:
        parser.error("--evaluate-only requires --resume")
    return args


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {value!r} was requested but torch.cuda.is_available() is false"
        )
    return device


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def model_config(
    args: argparse.Namespace, normalization: TactileNormalization
) -> TriModalCrossAutoencoderCfg:
    return TriModalCrossAutoencoderCfg(
        image_height=normalization.image_height,
        image_width=normalization.image_width,
        marker_count=normalization.marker_count,
        marker_summary_tokens=int(args.marker_summary_tokens),
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        image_base_channels=int(args.image_base_channels),
        marker_transformer_layers=int(args.marker_transformer_layers),
        cross_layers=int(args.cross_layers),
        cross_summary_tokens=int(args.cross_summary_tokens),
        ffn_ratio=int(args.ffn_ratio),
        decoder_base_channels=int(args.decoder_base_channels),
        dropout=float(args.dropout),
    )


def loss_config(
    args: argparse.Namespace, normalization: TactileNormalization
) -> TriModalAutoencoderLossCfg:
    return TriModalAutoencoderLossCfg(
        rgb_weight=float(args.rgb_loss_weight),
        depth_weight=float(args.depth_loss_weight),
        marker_weight=float(args.marker_loss_weight),
        rgb_reference_weight=float(args.rgb_reference_weight),
        rgb_change_threshold=float(args.rgb_change_threshold),
        rgb_change_boost=float(args.rgb_change_boost),
        depth_foreground_weight=float(args.depth_foreground_weight),
        depth_structure_weight=float(args.depth_structure_weight),
        depth_foreground_threshold=float(args.depth_foreground_threshold_mm)
        / max(float(args.depth_scale_mm), 1.0e-8),
        depth_mask_temperature=float(args.depth_mask_temperature),
        depth_mask_pos_weight=float(args.depth_mask_pos_weight),
        marker_direction_weight=float(args.marker_direction_weight),
        marker_direction_threshold_px=float(args.marker_direction_threshold_px),
        marker_prediction_floor_px=float(args.marker_prediction_floor_px),
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


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(resolved, map_location="cpu")
    if payload.get("network_type") != NETWORK_TYPE:
        raise ValueError(
            f"Checkpoint network_type={payload.get('network_type')!r} is not {NETWORK_TYPE!r}"
        )
    return payload


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
    if shuffle and bool(args.balance_contact):
        labels = torch.tensor(dataset.contact_labels, dtype=torch.int64)
        counts = torch.bincount(labels, minlength=2)
        if bool((counts > 0).all()):
            class_weights = counts.sum().to(torch.float64) / (
                2.0 * counts.to(torch.float64)
            )
            sampler = WeightedRandomSampler(
                class_weights.index_select(0, labels),
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        generator=generator,
        drop_last=False,
    )


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


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


def _batch_limit(loader: DataLoader, requested: int | None) -> int:
    return min(len(loader), int(requested)) if requested is not None else len(loader)


def _autocast(amp_enabled: bool):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp_enabled
        else nullcontext()
    )


def _make_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=4096.0)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=4096.0)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: AdamW,
    scaler: Any,
    device: torch.device,
    normalization: TactileNormalization,
    objective_cfg: TriModalAutoencoderLossCfg,
    args: argparse.Namespace,
    amp_enabled: bool,
    seed: int,
    epoch: int,
) -> tuple[dict[str, float], int]:
    model.train()
    metrics = MetricAccumulator()
    generator = _generator(device, seed)
    limit = _batch_limit(loader, args.max_train_batches)
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    skipped_steps = 0
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= limit:
            break
        batch = _move_batch(cpu_batch, device)
        with _autocast(amp_enabled):
            loss, batch_metrics = compute_tri_modal_autoencoder_objective(
                model,
                batch,
                normalization=normalization,
                cfg=objective_cfg,
                generator=generator,
                min_degradation_severity=float(args.min_degradation_severity),
                max_degradation_severity=float(args.max_degradation_severity),
                clean_probability=float(args.clean_sample_probability),
            )
            group_start = (
                batch_index // int(args.grad_accum_steps)
            ) * int(args.grad_accum_steps)
            group_size = min(int(args.grad_accum_steps), limit - group_start)
            scaled_loss = loss / group_size
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Training loss is non-finite at batch {batch_index}")
        scaler.scale(scaled_loss).backward()
        should_step = (
            (batch_index + 1) % int(args.grad_accum_steps) == 0
            or batch_index + 1 == limit
        )
        if should_step:
            scaler.unscale_(optimizer)
            if float(args.gradient_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            skipped = bool(scaler.is_enabled()) and float(scaler.get_scale()) < scale_before
            if skipped:
                skipped_steps += 1
            else:
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
        metrics.update(batch_metrics, int(batch["rgb"].shape[0]))
        if (
            (batch_index + 1) % int(args.log_every_batches) == 0
            or batch_index + 1 == limit
        ):
            print(
                f"[epoch {epoch + 1:03d}] batch={batch_index + 1}/{limit} "
                f"loss={float(loss.detach()):.6f}",
                flush=True,
            )
    averages = metrics.averages()
    averages["metric/optimizer_skipped_steps"] = float(skipped_steps)
    return averages, optimizer_steps


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: TactileNormalization,
    objective_cfg: TriModalAutoencoderLossCfg,
    amp_enabled: bool,
    seed: int,
    max_batches: int | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    accumulators = {name: MetricAccumulator() for name in EVALUATION_MODES}
    generator = _generator(device, seed)
    limit = _batch_limit(loader, max_batches)
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= limit:
            break
        batch = _move_batch(cpu_batch, device)
        batch_size = int(batch["rgb"].shape[0])
        for mode_index, mode in enumerate(EVALUATION_MODES):
            clean = mode == "clean"
            degraded_modality = 0 if clean else mode_index - 1
            with _autocast(amp_enabled):
                loss, batch_metrics = compute_tri_modal_autoencoder_objective(
                    model,
                    batch,
                    normalization=normalization,
                    cfg=objective_cfg,
                    generator=generator,
                    degraded_modality=degraded_modality,
                    min_degradation_severity=float(args.min_degradation_severity),
                    max_degradation_severity=float(args.max_degradation_severity),
                    clean_probability=1.0 if clean else 0.0,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Evaluation loss is non-finite at batch {batch_index}, mode={mode}"
                )
            accumulators[mode].update(batch_metrics, batch_size)
    by_mode = {name: accumulator.averages() for name, accumulator in accumulators.items()}
    degraded_losses = [by_mode[name]["loss/total"] for name in EVALUATION_MODES[1:]]
    degraded_mean = sum(degraded_losses) / len(degraded_losses)
    degraded_worst = max(degraded_losses)
    selection = (
        degraded_mean
        + float(args.worst_modality_weight) * degraded_worst
        + float(args.clean_selection_weight) * by_mode["clean"]["loss/total"]
    )
    result = {
        "loss/degraded_mean": degraded_mean,
        "loss/degraded_worst": degraded_worst,
        "loss/clean": by_mode["clean"]["loss/total"],
        "loss/selection": selection,
    }
    for mode, values in by_mode.items():
        for key, value in values.items():
            result[f"{mode}/{key}"] = value
    return result


def _lr_lambda(
    epoch: int, *, schedule: str, epochs: int, warmup_epochs: int
) -> float:
    if schedule == "fixed":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {schedule!r}")
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = min(
        1.0,
        max(
            0.0,
            float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs)),
        ),
    )
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _make_scheduler(
    optimizer: AdamW, args: argparse.Namespace, *, last_epoch: int = -1
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


def _checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Any,
    network_cfg: TriModalCrossAutoencoderCfg,
    objective_cfg: TriModalAutoencoderLossCfg,
    normalization: TactileNormalization,
    splits: EpisodeSplits,
    dataset_fingerprint: str,
    args: argparse.Namespace,
    validation_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "network_type": NETWORK_TYPE,
        "training_objective": OBJECTIVE_NAME,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
        "epochs_without_improvement": int(epochs_without_improvement),
        "model_state": model.state_dict(),
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
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
        if resume_payload.get("dataset_manifest_sha256") != fingerprint:
            raise ValueError("Resume checkpoint was created from a different dataset")
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

    datasets = {
        name: NpzTactileDataset(
            dataset_root,
            episode_ids=getattr(splits, name),
            normalization=normalization,
            cache_size=int(args.shard_cache_size),
            backend=str(args.dataset_backend),
        )
        for name in ("train", "validation", "test")
    }
    backends = {dataset.backend for dataset in datasets.values()}
    if len(backends) != 1:
        raise RuntimeError(f"Dataset splits selected inconsistent backends: {backends}")
    loaders = {
        "train": _make_loader(
            datasets["train"], args=args, device=device, shuffle=True, seed=int(args.seed)
        ),
        "validation": _make_loader(
            datasets["validation"],
            args=args,
            device=device,
            shuffle=False,
            seed=int(args.seed) + 1,
        ),
        "test": _make_loader(
            datasets["test"],
            args=args,
            device=device,
            shuffle=False,
            seed=int(args.seed) + 2,
        ),
    }

    network_cfg = model_config(args, normalization)
    objective_cfg = loss_config(args, normalization)
    model = TriModalCrossAutoencoder(network_cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = _make_scheduler(optimizer, args)
    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = _make_scaler(amp_enabled)
    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    if resume_payload is not None:
        if resume_payload["model_cfg"] != asdict(network_cfg):
            raise ValueError("Resume checkpoint model configuration does not match CLI")
        if resume_payload["loss_cfg"] != asdict(objective_cfg):
            raise ValueError("Resume checkpoint loss configuration does not match CLI")
        if resume_payload["normalization"] != normalization.to_json():
            raise ValueError("Resume checkpoint normalization does not match CLI")
        original_args = resume_payload.get("args", {})
        for name in (
            "min_degradation_severity",
            "max_degradation_severity",
            "clean_sample_probability",
            "weight_decay",
        ):
            if not math.isclose(
                float(original_args.get(name, getattr(args, name))),
                float(getattr(args, name)),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"Resume setting {name!r} does not match the checkpoint")
        model.load_state_dict(resume_payload["model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scaler.load_state_dict(resume_payload.get("scaler_state", {}))
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload.get("global_step", 0))
        best_validation_loss = float(resume_payload.get("best_validation_loss", math.inf))
        epochs_without_improvement = int(
            resume_payload.get("epochs_without_improvement", 0)
        )
        original_schedule = str(original_args.get("lr_schedule", "fixed"))
        original_epochs = int(original_args.get("epochs", args.epochs))
        original_lr = float(original_args.get("learning_rate", args.learning_rate))
        compatible_scheduler = (
            original_schedule == str(args.lr_schedule)
            and original_epochs == int(args.epochs)
            and math.isclose(
                original_lr,
                float(args.learning_rate),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = float(args.learning_rate)
            group["initial_lr"] = float(args.learning_rate)
            group["weight_decay"] = float(args.weight_decay)
        if compatible_scheduler:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        else:
            scheduler = _make_scheduler(optimizer, args, last_epoch=start_epoch - 1)

    run_summary = {
        "network_type": NETWORK_TYPE,
        "training_objective": OBJECTIVE_NAME,
        "dataset": str(dataset_root),
        "dataset_manifest_sha256": fingerprint,
        "dataset_backend": next(iter(backends)),
        "dataset_samples": int(manifest["sample_count"]),
        "split_episodes": {
            name: len(getattr(splits, name)) for name in ("train", "validation", "test")
        },
        "split_samples": {name: len(dataset) for name, dataset in datasets.items()},
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
        metrics = evaluate_epoch(
            model,
            loaders["test"],
            device=device,
            normalization=normalization,
            objective_cfg=objective_cfg,
            amp_enabled=amp_enabled,
            seed=int(args.seed) + 20_000,
            max_batches=args.max_test_batches,
            args=args,
        )
        _write_json_atomic(output_dir / "evaluation.json", metrics)
        print("[test] " + json.dumps(metrics, sort_keys=True), flush=True)
        return

    if start_epoch >= int(args.epochs):
        raise ValueError(
            f"Checkpoint already completed {start_epoch} epochs; increase --epochs"
        )
    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, int(args.epochs)):
        epoch_start = time.monotonic()
        train_metrics, optimizer_steps = train_epoch(
            model,
            loaders["train"],
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            normalization=normalization,
            objective_cfg=objective_cfg,
            args=args,
            amp_enabled=amp_enabled,
            seed=int(args.seed) + epoch * 101,
            epoch=epoch,
        )
        global_step += optimizer_steps
        validation_metrics = evaluate_epoch(
            model,
            loaders["validation"],
            device=device,
            normalization=normalization,
            objective_cfg=objective_cfg,
            amp_enabled=amp_enabled,
            seed=int(args.seed) + 10_000,
            max_batches=args.max_val_batches,
            args=args,
        )
        validation_loss = float(validation_metrics["loss/selection"])
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        current_lr = float(optimizer.param_groups[0]["lr"])
        if optimizer_steps > 0:
            scheduler.step()
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": current_lr,
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
            epochs_without_improvement=epochs_without_improvement,
            model=model,
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
        )
        _atomic_torch_save(output_dir / "last.pt", payload)
        if improved:
            _atomic_torch_save(output_dir / "best.pt", payload)
        if (epoch + 1) % int(args.save_every) == 0:
            _atomic_torch_save(output_dir / f"epoch_{epoch + 1:04d}.pt", payload)
        print(
            f"[epoch {epoch + 1:03d}/{int(args.epochs):03d}] "
            f"train={train_metrics['loss/total']:.6f} "
            f"val={validation_loss:.6f} best={best_validation_loss:.6f} "
            f"lr={current_lr:.3e} seconds={record['seconds']:.1f}",
            flush=True,
        )
        if (
            int(args.early_stopping_patience) > 0
            and epochs_without_improvement >= int(args.early_stopping_patience)
        ):
            print("[early-stop] validation did not improve", flush=True)
            break

    best_path = output_dir / "best.pt"
    best_payload = _load_checkpoint(best_path)
    model.load_state_dict(best_payload["model_state"], strict=True)
    test_metrics = evaluate_epoch(
        model,
        loaders["test"],
        device=device,
        normalization=normalization,
        objective_cfg=objective_cfg,
        amp_enabled=amp_enabled,
        seed=int(args.seed) + 20_000,
        max_batches=args.max_test_batches,
        args=args,
    )
    _write_json_atomic(output_dir / "test_metrics.json", test_metrics)
    print(
        "[done] "
        + json.dumps(
            {
                "best_checkpoint": str(best_path),
                "best_epoch": int(best_payload["epoch"]) + 1,
                "best_validation_loss": float(best_payload["best_validation_loss"]),
                "last_checkpoint": str(output_dir / "last.pt"),
                "test": test_metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
