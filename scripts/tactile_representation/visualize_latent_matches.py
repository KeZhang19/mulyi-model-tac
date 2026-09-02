#!/usr/bin/env python3
"""Visualize top-1 sim-to-real latent retrievals.

Each row contains the simulation query, the real observation retrieved by the
alignment model, and the ground-truth paired real observation.  Correct and
high-confidence incorrect matches are written to separate contact sheets so
that a high retrieval score can be checked visually rather than inferred only
from a scalar metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from BrainCo_DexHand.tactile_representation.models.latent_alignment import (  # noqa: E402
    TactileLatentAlignmentNetwork,
)
from train_latent_alignment import (  # noqa: E402
    MODEL_TYPES,
    PairedTactileDataset,
    _build_encoder,
    _inputs,
    _load_checkpoint,
    _make_dataset,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-checkpoint", type=Path, required=True)
    parser.add_argument("--real-checkpoint", type=Path, required=True)
    parser.add_argument("--alignment-checkpoint", type=Path, required=True)
    parser.add_argument("--sim-model-type", choices=MODEL_TYPES, default="robust")
    parser.add_argument("--real-model-type", choices=MODEL_TYPES, default="tri_modal")
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/revo3_index_sweep_parallel_v1")
    )
    parser.add_argument("--sim-dataset", type=Path, default=None)
    parser.add_argument("--real-dataset", type=Path, default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-backend", choices=("auto", "mmap", "npz"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-correct", type=int, default=8)
    parser.add_argument("--num-incorrect", type=int, default=8)
    parser.add_argument(
        "--direction",
        choices=("sim-to-real", "real-to-sim"),
        default="sim-to-real",
        help="Query direction to visualize.",
    )
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers must be non-negative")
    if args.num_correct < 0 or args.num_incorrect < 0:
        parser.error("num-correct and num-incorrect must be non-negative")
    if args.dpi <= 0:
        parser.error("dpi must be positive")
    return args


def _load_alignment(
    checkpoint: dict[str, Any],
    sim_checkpoint: dict[str, Any],
    real_checkpoint: dict[str, Any],
    sim_model_type: str,
    real_model_type: str,
    device: torch.device,
) -> TactileLatentAlignmentNetwork:
    alignment_args = checkpoint.get("args", {})
    sim_dim = int(sim_checkpoint["model_cfg"]["d_model"])
    real_dim = int(real_checkpoint["model_cfg"]["d_model"])
    if sim_dim != real_dim:
        raise ValueError(f"Encoder latent dimensions differ: sim={sim_dim}, real={real_dim}")
    aligner = TactileLatentAlignmentNetwork(
        _build_encoder(sim_model_type, sim_checkpoint),
        _build_encoder(real_model_type, real_checkpoint),
        latent_dim=sim_dim,
        projection_dim=int(alignment_args.get("projection_dim", 64)),
        projection_hidden_dim=alignment_args.get("projection_hidden_dim"),
        projection_layernorm=bool(alignment_args.get("projection_layernorm", False)),
        temperature=float(alignment_args.get("temperature", 0.1)),
        freeze_encoders=False,
    ).to(device)
    aligner.load_state_dict(checkpoint["model_state"], strict=True)
    return aligner.eval()


def _collect_embeddings(
    aligner: TactileLatentAlignmentNetwork,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    sim_latents: list[torch.Tensor] = []
    real_latents: list[torch.Tensor] = []
    pair_ids: list[tuple[int, int]] = []
    with torch.no_grad():
        for batch in loader:
            sim_batch = {name: value.to(device, non_blocking=True) for name, value in batch["sim"].items()}
            real_batch = {name: value.to(device, non_blocking=True) for name, value in batch["real"].items()}
            outputs = aligner(_inputs(sim_batch), _inputs(real_batch))
            sim_latents.append(outputs["z_sim"].float().cpu())
            real_latents.append(outputs["z_real"].float().cpu())
            pair_ids.extend(tuple(map(int, value)) for value in batch["pair_id"].tolist())
    return torch.cat(sim_latents), torch.cat(real_latents), pair_ids


def _rgb(sample: dict[str, torch.Tensor]) -> np.ndarray:
    return sample["rgb"].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def _draw_sample(
    axis: Any,
    sample: dict[str, torch.Tensor],
    *,
    title: str,
    image_width: int,
    image_height: int,
    marker_motion_scale_px: float,
) -> None:
    image = _rgb(sample)
    axis.imshow(image)
    marker = sample["marker"].detach().cpu().numpy()
    valid = sample["marker_valid"].detach().cpu().numpy().astype(bool)
    marker = marker[valid]
    if marker.size:
        x = marker[:, 0] * max(1, image_width - 1)
        y = marker[:, 1] * max(1, image_height - 1)
        dx = marker[:, 2] * marker_motion_scale_px
        dy = marker[:, 3] * marker_motion_scale_px
        axis.quiver(
            x,
            y,
            dx,
            dy,
            color="#ff7f0e",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
        )
        axis.scatter(x, y, s=5, c="black", linewidths=0.3)
    axis.set_title(title, fontsize=9)
    axis.set_xlim(0, image_width - 1)
    axis.set_ylim(image_height - 1, 0)
    axis.axis("off")


def _metadata(sample: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "episode_id": int(sample["episode_id"]),
        "episode_step": int(sample["episode_step"]),
        "target_force_n": float(sample["target_force_n"]),
        "phase": int(sample["phase"]),
        "sweep_stage": int(sample["sweep_stage"]),
    }


def _render(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    title: str,
    image_width: int,
    image_height: int,
    marker_motion_scale_px: float,
    dpi: int,
) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(15.0, max(3.5, 3.6 * len(rows))),
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        status = "CORRECT" if row["correct"] else "WRONG"
        query_key = row["query_key"]
        predicted_key = row["predicted_key"]
        target_key = row["target_key"]
        score = row["score"]
        common = f"ep={query_key[0]} step={query_key[1]}"
        _draw_sample(
            axes[row_index, 0],
            row["query_sample"],
            title=f"query sim\n{common}",
            image_width=image_width,
            image_height=image_height,
            marker_motion_scale_px=marker_motion_scale_px,
        )
        _draw_sample(
            axes[row_index, 1],
            row["predicted_sample"],
            title=f"top-1 real | {status}\npred={predicted_key} cos={score:.3f}",
            image_width=image_width,
            image_height=image_height,
            marker_motion_scale_px=marker_motion_scale_px,
        )
        _draw_sample(
            axes[row_index, 2],
            row["target_sample"],
            title=f"ground-truth real\ntarget={target_key}",
            image_width=image_width,
            image_height=image_height,
            marker_motion_scale_px=marker_motion_scale_px,
        )
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _select_rows(
    similarity: torch.Tensor,
    paired: PairedTactileDataset,
    query_samples: list[dict[str, torch.Tensor]],
    candidate_samples: list[dict[str, torch.Tensor]],
    *,
    direction: str,
    count_correct: int,
    count_incorrect: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if direction == "sim-to-real":
        ranking = similarity
    else:
        ranking = similarity.transpose(0, 1)
    top1 = ranking.argmax(dim=1)
    target_indices = torch.arange(len(paired))
    correct = top1.eq(target_indices)
    score = ranking[torch.arange(len(paired)), top1]
    correct_order = torch.where(correct)[0][torch.argsort(score[correct], descending=True)]
    incorrect_order = torch.where(~correct)[0][torch.argsort(score[~correct], descending=True)]

    def make_row(query_index: int) -> dict[str, Any]:
        candidate_index = int(top1[query_index])
        if direction == "sim-to-real":
            query_sample = query_samples[query_index]
            predicted_sample = candidate_samples[candidate_index]
            target_sample = candidate_samples[query_index]
        else:
            query_sample = candidate_samples[query_index]
            predicted_sample = query_samples[candidate_index]
            target_sample = query_samples[query_index]
        return {
            "query_sample": query_sample,
            "predicted_sample": predicted_sample,
            "target_sample": target_sample,
            "query_key": paired.pair_ids[query_index],
            "predicted_key": paired.pair_ids[candidate_index],
            "target_key": paired.pair_ids[query_index],
            "score": float(score[query_index]),
            "correct": bool(correct[query_index]),
        }

    correct_rows = [make_row(int(index)) for index in correct_order[:count_correct].tolist()]
    incorrect_rows = [make_row(int(index)) for index in incorrect_order[:count_incorrect].tolist()]
    return correct_rows, incorrect_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    sim_checkpoint = _load_checkpoint(args.sim_checkpoint)
    real_checkpoint = _load_checkpoint(args.real_checkpoint)
    alignment_checkpoint = torch.load(
        args.alignment_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(alignment_checkpoint, dict) or "model_state" not in alignment_checkpoint:
        raise ValueError(f"Invalid alignment checkpoint: {args.alignment_checkpoint}")
    aligner = _load_alignment(
        alignment_checkpoint,
        sim_checkpoint,
        real_checkpoint,
        args.sim_model_type,
        args.real_model_type,
        device,
    )

    split_ids = sorted(
        set(int(value) for value in sim_checkpoint["episode_splits"][args.split])
        & set(int(value) for value in real_checkpoint["episode_splits"][args.split])
    )
    sim_root = args.sim_dataset or args.dataset
    real_root = args.real_dataset or args.dataset
    sim_dataset = _make_dataset(sim_root, sim_checkpoint, split_ids, args.dataset_backend)
    real_dataset = _make_dataset(real_root, real_checkpoint, split_ids, args.dataset_backend)
    paired = PairedTactileDataset(sim_dataset, real_dataset, episode_ids=split_ids)
    loader = DataLoader(
        paired,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    sim_latent, real_latent, pair_ids = _collect_embeddings(aligner, loader, device)
    if pair_ids != list(paired.pair_ids):
        raise RuntimeError("DataLoader order changed while collecting retrieval embeddings")
    similarity = sim_latent @ real_latent.transpose(0, 1)
    sim_samples = [sim_dataset[record[0]] for record in paired._records]
    real_samples = [real_dataset[record[1]] for record in paired._records]
    query_samples = sim_samples if args.direction == "sim-to-real" else real_samples
    candidate_samples = real_samples if args.direction == "sim-to-real" else sim_samples
    correct_rows, incorrect_rows = _select_rows(
        similarity,
        paired,
        query_samples,
        candidate_samples,
        direction=args.direction,
        count_correct=args.num_correct,
        count_incorrect=args.num_incorrect,
    )
    normalization = sim_dataset.normalization
    args.output.mkdir(parents=True, exist_ok=True)
    _render(
        correct_rows,
        output_path=args.output / f"{args.direction.replace('-', '_')}_correct.png",
        title=f"Latent matches | {args.split} | {args.direction} | correct top-1",
        image_width=normalization.image_width,
        image_height=normalization.image_height,
        marker_motion_scale_px=normalization.marker_motion_scale_px,
        dpi=args.dpi,
    )
    _render(
        incorrect_rows,
        output_path=args.output / f"{args.direction.replace('-', '_')}_incorrect.png",
        title=f"Latent matches | {args.split} | {args.direction} | high-confidence errors",
        image_width=normalization.image_width,
        image_height=normalization.image_height,
        marker_motion_scale_px=normalization.marker_motion_scale_px,
        dpi=args.dpi,
    )
    metadata = {
        "split": args.split,
        "direction": args.direction,
        "num_pairs": len(paired),
        "top1_accuracy": float((similarity.argmax(dim=1) == torch.arange(len(paired))).float().mean())
        if args.direction == "sim-to-real"
        else float((similarity.transpose(0, 1).argmax(dim=1) == torch.arange(len(paired))).float().mean()),
        "correct": [
            {
                "query_key": list(row["query_key"]),
                "predicted_key": list(row["predicted_key"]),
                "target_key": list(row["target_key"]),
                "cosine": row["score"],
            }
            for row in correct_rows
        ],
        "incorrect": [
            {
                "query_key": list(row["query_key"]),
                "predicted_key": list(row["predicted_key"]),
                "target_key": list(row["target_key"]),
                "cosine": row["score"],
            }
            for row in incorrect_rows
        ],
    }
    direction_stem = args.direction.replace("-", "_")
    (args.output / "matches.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"pairs={len(paired)} top1={metadata['top1_accuracy']:.6f}")
    print(f"correct_figure={args.output / f'{direction_stem}_correct.png'}")
    print(f"incorrect_figure={args.output / f'{direction_stem}_incorrect.png'}")
    print(f"metadata={args.output / 'matches.json'}")


if __name__ == "__main__":
    main()
