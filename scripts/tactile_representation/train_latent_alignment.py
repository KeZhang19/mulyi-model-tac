#!/usr/bin/env python3
"""Train CTTP-style alignment heads on two pretrained tactile encoders.

The encoders are loaded from checkpoints and frozen by default.  Passing
``--unfreeze-all-epoch`` enables CTTP-style end-to-end fine-tuning after the
requested warm-up epoch.  A paired dataset supplies one simulation sample and
one real sample for the same ``(episode_id, episode_step)`` key; all mismatched
keys in a minibatch become in-batch negatives for the symmetric InfoNCE
objective.

This script can be used as a two-checkpoint latent-alignment pilot when both
domains are represented by the same dataset.  For a genuine sim--real run,
pass separate dataset roots that share a canonical pair key.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from BrainCo_DexHand.tactile_representation import (  # noqa: E402
    CrossModalTactileNetworkCfg,
    RobustCrossModalTactileNetwork,
    TriModalCrossAutoencoder,
    TriModalCrossAutoencoderCfg,
)
from BrainCo_DexHand.tactile_representation.models.latent_alignment import (  # noqa: E402
    TactileLatentAlignmentNetwork,
)
from BrainCo_DexHand.tactile_representation.training import (  # noqa: E402
    NpzTactileDataset,
    TactileNormalization,
    build_positive_mask,
    symmetric_queued_infonce_loss,
)


MODEL_TYPES = ("robust", "tri_modal")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CTTP-style tactile latent alignment heads with optional encoder fine-tuning."
    )
    parser.add_argument("--sim-checkpoint", type=Path, required=True)
    parser.add_argument("--real-checkpoint", type=Path, required=True)
    parser.add_argument("--sim-model-type", choices=MODEL_TYPES, default="robust")
    parser.add_argument("--real-model-type", choices=MODEL_TYPES, default="tri_modal")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/revo3_index_sweep_parallel_v1"),
        help="Dataset root used for both towers when separate roots are not supplied.",
    )
    parser.add_argument("--sim-dataset", type=Path, default=None)
    parser.add_argument("--real-dataset", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/revo3_latent_alignment_pilot")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset-backend", choices=("auto", "mmap", "npz"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--projection-layernorm",
        action="store_true",
        help="Layer-normalize each encoder latent before its projection head.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=0,
        help="At this epoch, unfreeze only late fusion/projection encoder blocks; zero disables.",
    )
    parser.add_argument(
        "--unfreeze-all-epoch",
        type=int,
        default=0,
        help=(
            "At this epoch, unfreeze both encoders end-to-end, matching CTTP; "
            "zero disables."
        ),
    )
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=1.0e-5,
        help="Learning rate for encoder parameters after either unfreeze option.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=0,
        help="Detached cross-batch candidate queue size; zero disables the queue.",
    )
    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.0,
        help="Fraction of non-anchor batch slots sampled from the same episode.",
    )
    parser.add_argument(
        "--hard-negative-min-step-gap",
        type=int,
        default=2,
        help="Minimum frame-step gap for same-episode hard negatives.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args(argv)

    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("epochs and batch-size must be positive; num-workers must be non-negative")
    if args.learning_rate <= 0.0 or args.temperature <= 0.0:
        parser.error("learning-rate and temperature must be positive")
    if args.unfreeze_epoch < 0 or args.encoder_learning_rate <= 0.0:
        parser.error("unfreeze-epoch must be non-negative and encoder-learning-rate positive")
    if args.unfreeze_all_epoch < 0:
        parser.error("unfreeze-all-epoch must be non-negative")
    if args.unfreeze_epoch > 0 and args.unfreeze_all_epoch > 0:
        parser.error("choose either unfreeze-epoch or unfreeze-all-epoch, not both")
    if args.projection_dim <= 0 or (
        args.projection_hidden_dim is not None and args.projection_hidden_dim <= 0
    ):
        parser.error("projection dimensions must be positive")
    if args.gradient_clip < 0.0 or args.weight_decay < 0.0:
        parser.error("gradient-clip and weight-decay must be non-negative")
    if args.queue_size < 0:
        parser.error("queue-size must be non-negative")
    if not 0.0 <= args.hard_negative_fraction <= 1.0:
        parser.error("hard-negative-fraction must be in [0, 1]")
    if args.hard_negative_min_step_gap < 0:
        parser.error("hard-negative-min-step-gap must be non-negative")
    if args.save_every <= 0:
        parser.error("save-every must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable")
    return device


def _frame_key(sample: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    return (int(sample["episode_id"]), int(sample["episode_step"]))


class PairedTactileDataset(Dataset[dict[str, Any]]):
    """Join two tactile datasets by their canonical episode/frame key."""

    def __init__(
        self,
        sim_dataset: NpzTactileDataset,
        real_dataset: NpzTactileDataset,
        *,
        episode_ids: Sequence[int],
    ) -> None:
        self.sim_dataset = sim_dataset
        self.real_dataset = real_dataset
        allowed = {int(value) for value in episode_ids}
        if not allowed:
            raise ValueError("episode_ids must not be empty")

        sim_indices = self._index_dataset(sim_dataset, allowed, "simulation")
        real_indices = self._index_dataset(real_dataset, allowed, "real")
        keys = sorted(set(sim_indices).intersection(real_indices))
        if not keys:
            raise ValueError(
                "No paired frames found. The two datasets must share episode_id and episode_step."
            )
        self._records = [(sim_indices[key], real_indices[key], key) for key in keys]

    @staticmethod
    def _index_dataset(
        dataset: NpzTactileDataset,
        allowed_episode_ids: set[int],
        domain_name: str,
    ) -> dict[tuple[int, int], int]:
        indexed: dict[tuple[int, int], int] = {}
        for index in range(len(dataset)):
            # The metadata fields are scalar; the image tensors are released after
            # this pass.  The mmap backend keeps this indexing pass bounded in RAM.
            sample = dataset[index]
            key = _frame_key(sample)
            if key[0] not in allowed_episode_ids:
                continue
            if key in indexed:
                raise ValueError(f"Duplicate {domain_name} frame key {key}")
            indexed[key] = index
        return indexed

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sim_index, real_index, key = self._records[index]
        return {
            "sim": self.sim_dataset[sim_index],
            "real": self.real_dataset[real_index],
            "pair_id": torch.tensor(key, dtype=torch.int64),
        }

    @property
    def pair_ids(self) -> tuple[tuple[int, int], ...]:
        """Return the immutable canonical key for every paired record."""

        return tuple(record[2] for record in self._records)


class EpisodeHardNegativeBatchSampler(Sampler[list[int]]):
    """Build batches containing same-episode, different-frame negatives.

    Random in-batch negatives are often too easy for tactile trajectories.  This
    sampler keeps each anchor unique within an epoch, then fills a configurable
    fraction of the remaining slots with frames from the same episode while
    enforcing a minimum temporal gap.  The remaining slots are sampled from the
    full paired dataset.  No observations are duplicated inside a batch.
    """

    def __init__(
        self,
        dataset: PairedTactileDataset,
        batch_size: int,
        *,
        hard_negative_fraction: float,
        min_step_gap: int = 2,
        seed: int = 7,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= float(hard_negative_fraction) <= 1.0:
            raise ValueError("hard_negative_fraction must be in [0, 1]")
        if min_step_gap < 0:
            raise ValueError("min_step_gap must be non-negative")
        if len(dataset) == 0:
            raise ValueError("dataset must not be empty")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.min_step_gap = int(min_step_gap)
        self.seed = int(seed)
        self.epoch = 0
        self._pair_ids = dataset.pair_ids
        self._by_episode: dict[int, list[int]] = {}
        for index, (episode_id, _) in enumerate(self._pair_ids):
            self._by_episode.setdefault(int(episode_id), []).append(index)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1009)
        anchors = list(range(len(self.dataset)))
        rng.shuffle(anchors)
        all_indices = list(range(len(self.dataset)))
        for start in range(0, len(anchors), self.batch_size):
            target_size = min(self.batch_size, len(anchors) - start)
            anchor = anchors[start]
            selected = [anchor]
            episode_id, step = self._pair_ids[anchor]
            hard_candidates = [
                index
                for index in self._by_episode[int(episode_id)]
                if index != anchor
                and abs(int(self._pair_ids[index][1]) - int(step)) >= self.min_step_gap
            ]
            rng.shuffle(hard_candidates)
            hard_count = min(
                int(round((target_size - 1) * self.hard_negative_fraction)),
                len(hard_candidates),
            )
            selected.extend(hard_candidates[:hard_count])

            random_candidates = all_indices.copy()
            rng.shuffle(random_candidates)
            selected_set = set(selected)
            for index in random_candidates:
                if len(selected) >= target_size:
                    break
                if index not in selected_set:
                    selected.append(index)
                    selected_set.add(index)
            if len(selected) != target_size:
                raise RuntimeError("Could not construct a non-duplicated training batch")
            rng.shuffle(selected)
            yield selected


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload or "model_cfg" not in payload:
        raise ValueError(f"Checkpoint {path} is missing model_state/model_cfg")
    return payload


def _build_encoder(model_type: str, checkpoint: Mapping[str, Any]) -> nn.Module:
    cfg_values = dict(checkpoint["model_cfg"])
    if model_type == "robust":
        model: nn.Module = RobustCrossModalTactileNetwork(
            CrossModalTactileNetworkCfg(**cfg_values)
        )
    elif model_type == "tri_modal":
        model = TriModalCrossAutoencoder(TriModalCrossAutoencoderCfg(**cfg_values))
    else:  # pragma: no cover - argparse restricts this branch
        raise ValueError(f"Unknown model type: {model_type}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def _make_dataset(
    root: Path,
    checkpoint: Mapping[str, Any],
    episode_ids: Sequence[int],
    backend: str,
) -> NpzTactileDataset:
    normalization = TactileNormalization(**dict(checkpoint["normalization"]))
    return NpzTactileDataset(
        root,
        episode_ids=episode_ids,
        normalization=normalization,
        backend=backend,
    )


def _inputs(domain_batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "rgb": domain_batch["rgb"],
        "depth": domain_batch["depth"],
        "marker": domain_batch["marker"],
        "marker_valid_mask": domain_batch["marker_valid"],
    }


def _retrieval_recall(
    outputs: Mapping[str, torch.Tensor], positive_mask: torch.Tensor
) -> float:
    similarity = outputs["z_sim"] @ outputs["z_real"].transpose(0, 1)
    prediction = similarity.argmax(dim=1)
    row_index = torch.arange(similarity.shape[0], device=similarity.device)
    return float(positive_mask.to(device=similarity.device)[row_index, prediction].float().mean())


def _run_epoch(
    aligner: TactileLatentAlignmentNetwork,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    max_batches: int | None,
    queue_size: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    if training:
        aligner.train()
    else:
        aligner.eval()
    total_loss = 0.0
    total_recall = 0.0
    batches = 0
    queue_sim: torch.Tensor | None = None
    queue_real: torch.Tensor | None = None
    queue_ids: torch.Tensor | None = None
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            sim_batch = {name: value.to(device) for name, value in batch["sim"].items()}
            real_batch = {name: value.to(device) for name, value in batch["real"].items()}
            pair_ids = batch["pair_id"]
            outputs = aligner(_inputs(sim_batch), _inputs(real_batch))
            current_positive = build_positive_mask(pair_ids, pair_ids)
            if queue_size > 0 and queue_sim is not None and queue_real is not None:
                candidate_ids = torch.cat((pair_ids, queue_ids), dim=0)
                sim_positive = build_positive_mask(pair_ids, candidate_ids)
                real_positive = build_positive_mask(pair_ids, candidate_ids)
                loss = symmetric_queued_infonce_loss(
                    outputs["z_sim"],
                    torch.cat((outputs["z_real"], queue_real), dim=0),
                    sim_positive,
                    outputs["z_real"],
                    torch.cat((outputs["z_sim"], queue_sim), dim=0),
                    real_positive,
                    temperature=aligner.temperature,
                )
            else:
                loss = aligner.alignment_loss(outputs, current_positive)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip > 0.0:
                    nn.utils.clip_grad_norm_(aligner.trainable_parameters(), gradient_clip)
                optimizer.step()
                if queue_size > 0:
                    current_sim = outputs["z_sim"].detach()
                    current_real = outputs["z_real"].detach()
                    current_ids = pair_ids.detach()
                    if queue_sim is None:
                        queue_sim = current_sim
                        queue_real = current_real
                        queue_ids = current_ids
                    else:
                        queue_sim = torch.cat((queue_sim, current_sim), dim=0)[-queue_size:]
                        queue_real = torch.cat((queue_real, current_real), dim=0)[-queue_size:]
                        queue_ids = torch.cat((queue_ids, current_ids), dim=0)[-queue_size:]
            total_loss += float(loss.detach())
            total_recall += _retrieval_recall(outputs, current_positive)
            batches += 1
    if batches == 0:
        raise ValueError("No batches were processed; increase the batch limit or dataset size")
    return {"loss": total_loss / batches, "recall_at_1": total_recall / batches}


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    aligner: TactileLatentAlignmentNetwork,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    train_pairs: int,
    validation_pairs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    end_to_end = int(getattr(args, "unfreeze_all_epoch", 0)) > 0
    torch.save(
        {
            "schema_version": 2,
            "training_objective": (
                "cttp_style_end_to_end_tactile_latent_alignment_v1"
                if end_to_end
                else "cttp_style_frozen_encoder_latent_alignment_v1"
            ),
            "encoder_training": "end_to_end" if end_to_end else "frozen",
            "epoch": int(epoch),
            "model_state": aligner.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "train_metrics": dict(train_metrics),
            "validation_metrics": dict(validation_metrics),
            "train_pairs": int(train_pairs),
            "validation_pairs": int(validation_pairs),
        },
        path,
    )


def _save_encoder_checkpoint(
    path: Path,
    *,
    encoder: nn.Module,
    source_checkpoint: Mapping[str, Any],
    source_checkpoint_path: Path,
    role: str,
    alignment_epoch: int | None = None,
    alignment_args: argparse.Namespace | None = None,
) -> None:
    """Save one encoder in the same loadable format as its source checkpoint.

    Keeping the original model configuration, normalization, and episode split
    metadata makes the exported file usable anywhere that accepts a regular
    tactile encoder checkpoint, while omitting the other tower and projection
    heads.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": int(source_checkpoint.get("schema_version", 1)),
        "encoder_only": True,
        "encoder_role": str(role),
        "source_checkpoint": str(source_checkpoint_path),
        "source_epoch": source_checkpoint.get("epoch"),
        "alignment_epoch": alignment_epoch,
        "network_type": source_checkpoint.get("network_type"),
        "model_cfg": dict(source_checkpoint["model_cfg"]),
        "model_state": {
            name: parameter.detach().cpu()
            for name, parameter in encoder.state_dict().items()
        },
        "normalization": source_checkpoint.get("normalization"),
        "episode_splits": source_checkpoint.get("episode_splits"),
    }
    if alignment_args is not None:
        payload["alignment_args"] = vars(alignment_args)
    torch.save(payload, path)


def _add_encoder_optimizer_group(
    aligner: TactileLatentAlignmentNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    learning_rate: float,
    weight_decay: float,
) -> int:
    """Add newly trainable encoder parameters without duplicating head params."""

    existing_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    new_parameters = [
        parameter
        for parameter in aligner.parameters()
        if parameter.requires_grad and id(parameter) not in existing_ids
    ]
    if not new_parameters:
        raise RuntimeError("Encoder unfreezing found no new optimizer parameters")
    optimizer.add_param_group(
        {
            "params": new_parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
    )
    return len(new_parameters)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    sim_checkpoint = _load_checkpoint(args.sim_checkpoint)
    real_checkpoint = _load_checkpoint(args.real_checkpoint)

    sim_split = sim_checkpoint.get("episode_splits", {})
    real_split = real_checkpoint.get("episode_splits", {})
    train_episodes = sorted(
        set(int(value) for value in sim_split.get("train", []))
        & set(int(value) for value in real_split.get("train", []))
    )
    validation_episodes = sorted(
        set(int(value) for value in sim_split.get("validation", []))
        & set(int(value) for value in real_split.get("validation", []))
    )
    if not train_episodes or not validation_episodes:
        raise ValueError("The two checkpoints need overlapping train and validation episode splits")

    sim_root = args.sim_dataset or args.dataset
    real_root = args.real_dataset or args.dataset
    sim_train = _make_dataset(sim_root, sim_checkpoint, train_episodes, args.dataset_backend)
    real_train = _make_dataset(real_root, real_checkpoint, train_episodes, args.dataset_backend)
    sim_validation = _make_dataset(
        sim_root, sim_checkpoint, validation_episodes, args.dataset_backend
    )
    real_validation = _make_dataset(
        real_root, real_checkpoint, validation_episodes, args.dataset_backend
    )
    train_dataset = PairedTactileDataset(
        sim_train, real_train, episode_ids=train_episodes
    )
    validation_dataset = PairedTactileDataset(
        sim_validation, real_validation, episode_ids=validation_episodes
    )
    train_batch_sampler = None
    if args.hard_negative_fraction > 0.0:
        train_batch_sampler = EpisodeHardNegativeBatchSampler(
            train_dataset,
            args.batch_size,
            hard_negative_fraction=args.hard_negative_fraction,
            min_step_gap=args.hard_negative_min_step_gap,
            seed=args.seed,
        )
    train_loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if train_batch_sampler is None:
        train_loader_kwargs.update(
            {"batch_size": args.batch_size, "shuffle": True, "drop_last": False}
        )
    else:
        train_loader_kwargs["batch_sampler"] = train_batch_sampler
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    sim_encoder = _build_encoder(args.sim_model_type, sim_checkpoint)
    real_encoder = _build_encoder(args.real_model_type, real_checkpoint)
    sim_dim = int(sim_checkpoint["model_cfg"]["d_model"])
    real_dim = int(real_checkpoint["model_cfg"]["d_model"])
    if sim_dim != real_dim:
        raise ValueError(
            "The current aligner expects equal encoder latent dimensions; "
            f"got sim={sim_dim}, real={real_dim}"
        )
    aligner = TactileLatentAlignmentNetwork(
        sim_encoder,
        real_encoder,
        latent_dim=sim_dim,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_layernorm=args.projection_layernorm,
        temperature=args.temperature,
        freeze_encoders=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        aligner.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    resume_validation_loss = float("inf")
    encoder_group_added = False
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        resume_epoch = int(payload["epoch"])
        # A resumed checkpoint may already contain an encoder optimizer group.
        # Recreate that group before loading optimizer state so the parameter
        # groups have the same topology as when the checkpoint was written.
        optimizer_state = payload["optimizer_state"]
        checkpoint_has_encoder_group = len(optimizer_state.get("param_groups", [])) > 1
        if checkpoint_has_encoder_group:
            checkpoint_args = payload.get("args", {})
            if payload.get("encoder_training") == "end_to_end" or checkpoint_args.get(
                "unfreeze_all_epoch", 0
            ):
                aligner.set_encoder_trainable(True)
            else:
                aligner.set_encoder_prefixes_trainable(
                    ("fusion", "cross_blocks", "latent_projection")
                )
            _add_encoder_optimizer_group(
                aligner,
                optimizer,
                learning_rate=args.encoder_learning_rate,
                weight_decay=args.weight_decay,
            )
            encoder_group_added = True
        aligner.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(optimizer_state)
        start_epoch = resume_epoch + 1
        resume_validation_loss = float(
            payload.get("validation_metrics", {}).get("loss", float("inf"))
        )
        # If the source checkpoint was frozen, allow a new run to opt into
        # encoder fine-tuning after loading its one-group optimizer state.
        if not encoder_group_added and args.unfreeze_all_epoch > 0 and resume_epoch >= args.unfreeze_all_epoch:
            aligner.set_encoder_trainable(True)
            _add_encoder_optimizer_group(
                aligner,
                optimizer,
                learning_rate=args.encoder_learning_rate,
                weight_decay=args.weight_decay,
            )
            encoder_group_added = True
        elif not encoder_group_added and args.unfreeze_epoch > 0 and resume_epoch >= args.unfreeze_epoch:
            aligner.set_encoder_prefixes_trainable(
                ("fusion", "cross_blocks", "latent_projection")
            )
            _add_encoder_optimizer_group(
                aligner,
                optimizer,
                learning_rate=args.encoder_learning_rate,
                weight_decay=args.weight_decay,
            )
            encoder_group_added = True

    print(
        f"train_pairs={len(train_dataset)} validation_pairs={len(validation_dataset)} "
        f"device={device} projection_dim={args.projection_dim} "
        f"queue_size={args.queue_size} hard_negative_fraction={args.hard_negative_fraction} "
        f"unfreeze_epoch={args.unfreeze_epoch} "
        f"unfreeze_all_epoch={args.unfreeze_all_epoch} "
        f"encoder_lr={args.encoder_learning_rate} "
        f"encoders_frozen={all(not p.requires_grad for p in sim_encoder.parameters()) and all(not p.requires_grad for p in real_encoder.parameters())}"
    )
    best_validation_loss = resume_validation_loss
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if (
            args.unfreeze_all_epoch > 0
            and not encoder_group_added
            and epoch >= args.unfreeze_all_epoch
        ):
            aligner.set_encoder_trainable(True)
            parameter_count = _add_encoder_optimizer_group(
                aligner,
                optimizer,
                learning_rate=args.encoder_learning_rate,
                weight_decay=args.weight_decay,
            )
            encoder_group_added = True
            print(
                f"all_encoders_unfrozen epoch={epoch} parameters={parameter_count} "
                f"lr={args.encoder_learning_rate}",
                flush=True,
            )
        elif (
            args.unfreeze_epoch > 0
            and not encoder_group_added
            and epoch >= args.unfreeze_epoch
        ):
            enabled = aligner.set_encoder_prefixes_trainable(
                ("fusion", "cross_blocks", "latent_projection")
            )
            parameter_count = _add_encoder_optimizer_group(
                aligner,
                optimizer,
                learning_rate=args.encoder_learning_rate,
                weight_decay=args.weight_decay,
            )
            encoder_group_added = True
            print(
                f"late_encoder_unfrozen epoch={epoch} parameters={parameter_count} "
                f"matched_names={len(enabled)} lr={args.encoder_learning_rate}",
                flush=True,
            )
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            aligner,
            train_loader,
            device,
            optimizer=optimizer,
            gradient_clip=args.gradient_clip,
            max_batches=args.max_train_batches,
            queue_size=args.queue_size,
        )
        validation_metrics = _run_epoch(
            aligner,
            validation_loader,
            device,
            optimizer=None,
            gradient_clip=0.0,
            max_batches=args.max_val_batches,
        )
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_r1={train_metrics['recall_at_1']:.4f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_r1={validation_metrics['recall_at_1']:.4f}"
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            _save_checkpoint(
                args.output / "best.pt",
                epoch=epoch,
                aligner=aligner,
                optimizer=optimizer,
                args=args,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                train_pairs=len(train_dataset),
                validation_pairs=len(validation_dataset),
            )
            _save_encoder_checkpoint(
                args.output / "sim_encoder_best.pt",
                encoder=aligner.sim_encoder,
                source_checkpoint=sim_checkpoint,
                source_checkpoint_path=args.sim_checkpoint,
                role="simulation",
                alignment_epoch=epoch,
                alignment_args=args,
            )
            _save_encoder_checkpoint(
                args.output / "real_encoder_best.pt",
                encoder=aligner.real_encoder,
                source_checkpoint=real_checkpoint,
                source_checkpoint_path=args.real_checkpoint,
                role="real",
                alignment_epoch=epoch,
                alignment_args=args,
            )
        if epoch % args.save_every == 0 or epoch == args.epochs:
            _save_checkpoint(
                args.output / f"epoch_{epoch:04d}.pt",
                epoch=epoch,
                aligner=aligner,
                optimizer=optimizer,
                args=args,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                train_pairs=len(train_dataset),
                validation_pairs=len(validation_dataset),
            )


if __name__ == "__main__":
    main()
