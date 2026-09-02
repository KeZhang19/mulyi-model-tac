#!/usr/bin/env python3
"""Fit affine frozen-tower adapters for a tactile common latent space.

The paired training split is used to solve a regularized least-squares map from
the simulation encoder latent to the real encoder latent.  The real tower is
kept as the common-space anchor, so this remains a two-tower projection setup
while avoiding unnecessary nonlinear distortion when the pretrained latents
are already informative.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from BrainCo_DexHand.tactile_representation.models.latent_alignment import (  # noqa: E402
    AffineTactileLatentAlignmentNetwork,
)
from BrainCo_DexHand.tactile_representation.training import (  # noqa: E402
    NpzTactileDataset,
    TactileNormalization,
)

from train_latent_alignment import (  # noqa: E402
    PairedTactileDataset,
    _build_encoder,
    _load_checkpoint,
    _make_dataset,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-checkpoint", type=Path, required=True)
    parser.add_argument("--real-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/revo3_index_sweep_parallel_v1"))
    parser.add_argument("--sim-dataset", type=Path, default=None)
    parser.add_argument("--real-dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/revo3_latent_alignment_ridge_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dataset-backend", choices=("auto", "mmap", "npz"), default="auto")
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-4)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers must be non-negative")
    if args.ridge_lambda < 0.0:
        parser.error("ridge-lambda must be non-negative")
    return args


def _inputs(domain_batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "rgb": domain_batch["rgb"],
        "depth": domain_batch["depth"],
        "marker": domain_batch["marker"],
        "marker_valid_mask": domain_batch["marker_valid"],
    }


def _collect(
    loader: DataLoader,
    sim_encoder: torch.nn.Module,
    real_encoder: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sim_latents: list[torch.Tensor] = []
    real_latents: list[torch.Tensor] = []
    sim_encoder.eval()
    real_encoder.eval()
    with torch.no_grad():
        for batch in loader:
            sim_batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch["sim"].items()
            }
            real_batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch["real"].items()
            }
            sim_latents.append(
                sim_encoder.encode(**_inputs(sim_batch)).float().cpu()
            )
            real_latents.append(
                real_encoder.encode(**_inputs(real_batch)).float().cpu()
            )
    if not sim_latents:
        raise ValueError("No paired samples were collected")
    return torch.cat(sim_latents), torch.cat(real_latents)


def _retrieval_metrics(sim_latent: torch.Tensor, real_latent: torch.Tensor) -> dict[str, float]:
    similarity = torch.nn.functional.normalize(sim_latent, dim=-1) @ torch.nn.functional.normalize(
        real_latent, dim=-1
    ).transpose(0, 1)
    n = similarity.shape[0]
    ranks = similarity.argsort(dim=1, descending=True)
    target = torch.arange(n)[:, None]
    positive_rank = (ranks == target).nonzero(as_tuple=False)[:, 1]
    return {
        "r1": float((positive_rank < 1).float().mean()),
        "r5": float((positive_rank < 5).float().mean()),
        "r10": float((positive_rank < 10).float().mean()),
        "mean_rank": float(positive_rank.float().mean() + 1.0),
        "median_rank": float(positive_rank.median() + 1),
        "positive_cosine": float(similarity.diag().mean()),
    }


def _fit_affine(sim_latent: torch.Tensor, real_latent: torch.Tensor, ridge_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    sim_latent = sim_latent.double()
    real_latent = real_latent.double()
    design = torch.cat((sim_latent, torch.ones((len(sim_latent), 1), dtype=torch.float64)), dim=1)
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * float(ridge_lambda)
    weights = torch.linalg.solve(design.transpose(0, 1) @ design + regularizer, design.transpose(0, 1) @ real_latent)
    return weights[:-1].transpose(0, 1).float(), weights[-1].float()


def _make_loader(
    root: Path,
    checkpoint: Mapping[str, Any],
    other_root: Path,
    other_checkpoint: Mapping[str, Any],
    episode_ids: Sequence[int],
    backend: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    left = _make_dataset(root, checkpoint, episode_ids, backend)
    right = _make_dataset(other_root, other_checkpoint, episode_ids, backend)
    paired = PairedTactileDataset(left, right, episode_ids=episode_ids)
    return DataLoader(
        paired,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested but CUDA is unavailable")
    sim_checkpoint = _load_checkpoint(args.sim_checkpoint)
    real_checkpoint = _load_checkpoint(args.real_checkpoint)
    sim_split = sim_checkpoint.get("episode_splits", {})
    real_split = real_checkpoint.get("episode_splits", {})
    train_episodes = sorted(
        set(map(int, sim_split.get("train", []))) & set(map(int, real_split.get("train", [])))
    )
    validation_episodes = sorted(
        set(map(int, sim_split.get("validation", [])))
        & set(map(int, real_split.get("validation", [])))
    )
    test_episodes = sorted(
        set(map(int, sim_split.get("test", []))) & set(map(int, real_split.get("test", [])))
    )
    sim_root = args.sim_dataset or args.dataset
    real_root = args.real_dataset or args.dataset
    sim_encoder = _build_encoder("robust", sim_checkpoint).to(device)
    real_encoder = _build_encoder("tri_modal", real_checkpoint).to(device)
    sim_encoder.eval()
    real_encoder.eval()
    train_loader = _make_loader(
        sim_root, sim_checkpoint, real_root, real_checkpoint, train_episodes,
        args.dataset_backend, args.batch_size, args.num_workers
    )
    validation_loader = _make_loader(
        sim_root, sim_checkpoint, real_root, real_checkpoint, validation_episodes,
        args.dataset_backend, args.batch_size, args.num_workers
    )
    test_loader = _make_loader(
        sim_root, sim_checkpoint, real_root, real_checkpoint, test_episodes,
        args.dataset_backend, args.batch_size, args.num_workers
    )
    train_sim, train_real = _collect(train_loader, sim_encoder, real_encoder, device)
    validation_sim, validation_real = _collect(validation_loader, sim_encoder, real_encoder, device)
    test_sim, test_real = _collect(test_loader, sim_encoder, real_encoder, device)
    weight, bias = _fit_affine(train_sim, train_real, args.ridge_lambda)

    aligned_train_sim = train_sim @ weight.transpose(0, 1) + bias
    aligned_validation_sim = validation_sim @ weight.transpose(0, 1) + bias
    aligned_test_sim = test_sim @ weight.transpose(0, 1) + bias
    train_metrics = _retrieval_metrics(aligned_train_sim, train_real)
    validation_metrics = _retrieval_metrics(aligned_validation_sim, validation_real)
    test_metrics = _retrieval_metrics(aligned_test_sim, test_real)

    aligner = AffineTactileLatentAlignmentNetwork(
        sim_encoder,
        real_encoder,
        latent_dim=int(sim_checkpoint["model_cfg"]["d_model"]),
        projection_dim=int(real_checkpoint["model_cfg"]["d_model"]),
        freeze_encoders=True,
    ).to(device)
    with torch.no_grad():
        aligner.sim_projection.weight.copy_(weight)
        aligner.sim_projection.bias.copy_(bias)
        aligner.real_projection.weight.copy_(torch.eye(weight.shape[0]))
        aligner.real_projection.bias.zero_()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "training_objective": "ridge_affine_frozen_encoder_latent_alignment_v1",
            "model_state": aligner.state_dict(),
            "sim_checkpoint": str(args.sim_checkpoint),
            "real_checkpoint": str(args.real_checkpoint),
            "args": vars(args),
            "train_episodes": train_episodes,
            "validation_episodes": validation_episodes,
            "test_episodes": test_episodes,
            "train_pairs": len(train_sim),
            "validation_pairs": len(validation_sim),
            "test_pairs": len(test_sim),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        args.output / "best.pt",
    )
    print(
        f"train_pairs={len(train_sim)} validation_pairs={len(validation_sim)} test_pairs={len(test_sim)} "
        f"ridge_lambda={args.ridge_lambda}"
    )
    print("train", train_metrics)
    print("validation", validation_metrics)
    print("test", test_metrics)


if __name__ == "__main__":
    main()
