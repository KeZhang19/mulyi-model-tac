# Copyright (c) 2021-2026, ETH Zurich, NVIDIA CORPORATION, and BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict episode and sequence loading for offline teacher rollouts."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .teacher_rollout_storage import (
    TEACHER_ROLLOUT_FORMAT,
    TEACHER_ROLLOUT_SCHEMA_VERSION,
)


REQUIRED_OBSERVATIONS = (
    "student_low_dim",
    "depth_image",
    "secondary_depth_image",
    "student_aux",
)
REQUIRED_TIME_FIELDS = (
    "env_id",
    "episode_id",
    "step_id",
    "control_step",
    "reset_mask",
    "done",
    "timeout",
    "teacher_target_action",
    "behavior_action",
    "executed_action",
    "applied_action",
    "object_id",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch_file(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Cannot read teacher-rollout episode {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Teacher-rollout episode must be a dictionary: {path}")
    return value


def _time_lengths(value: Any, prefix: str = "") -> dict[str, int]:
    lengths: dict[str, int] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else key
            if key in {"episode", "format", "schema_version", "task"}:
                continue
            lengths.update(_time_lengths(item, name))
    elif isinstance(value, torch.Tensor):
        if value.ndim < 1:
            raise ValueError(f"Time field {prefix!r} must have a leading time dimension")
        lengths[prefix] = int(value.shape[0])
    return lengths


def _tensor_schema(value: Any, prefix: str = "") -> dict[str, Any]:
    schema: dict[str, Any] = {}
    if not isinstance(value, dict):
        return schema
    for key, item in value.items():
        if key in {"episode", "format", "schema_version", "task"}:
            continue
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            schema.update(_tensor_schema(item, name))
        elif isinstance(item, torch.Tensor):
            schema[name] = {
                "shape": ["T", *list(item.shape[1:])],
                "dtype": str(item.dtype).removeprefix("torch."),
            }
    return schema


class TeacherRolloutEpisodeDataset(Dataset):
    """Load only complete episodes listed by a strict, versioned manifest."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        entries: Sequence[dict[str, Any]] | None = None,
        validate_hashes: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        manifest_path = self.data_dir / "manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Cannot read teacher-rollout manifest {manifest_path}: {exc}") from exc
        if self.manifest.get("format") != TEACHER_ROLLOUT_FORMAT:
            raise ValueError(
                f"Unsupported teacher-rollout format {self.manifest.get('format')!r}"
            )
        if self.manifest.get("schema_version") != TEACHER_ROLLOUT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported teacher-rollout schema_version "
                f"{self.manifest.get('schema_version')!r}; expected "
                f"{TEACHER_ROLLOUT_SCHEMA_VERSION}"
            )
        manifest_entries = self.manifest.get("episodes")
        if not isinstance(manifest_entries, list):
            raise ValueError("Teacher-rollout manifest episodes must be a list")
        self.entries = list(manifest_entries if entries is None else entries)
        self.validate_hashes = bool(validate_hashes)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        known_files = {item.get("file") for item in manifest_entries}
        for entry in self.entries:
            if not isinstance(entry, dict) or entry.get("file") not in known_files:
                raise ValueError("Dataset split contains an episode not owned by the manifest")
            filename = entry["file"]
            if Path(filename).name != filename or not filename.endswith(".pt"):
                raise ValueError(f"Unsafe teacher-rollout episode filename: {filename!r}")
            path = self.data_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Manifest episode is missing: {path}")
            expected_hash = entry.get("sha256")
            if self.validate_hashes and expected_hash is not None:
                actual_hash = _sha256_path(path)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Teacher-rollout episode hash mismatch for {path}: "
                        f"expected={expected_hash}, actual={actual_hash}"
                    )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        path = self.data_dir / self.entries[index]["file"]
        episode = _load_torch_file(path)
        self._validate_episode(
            episode,
            path,
            expected_task=self.manifest.get("task"),
            expected_schema=self.manifest.get("schema"),
        )
        entry = self.entries[index]
        header = episode["episode"]
        for key in ("env_id", "episode_id", "object_name", "length"):
            if entry.get(key) != header.get(key):
                raise ValueError(
                    f"Manifest/episode {key} mismatch for {path}: "
                    f"manifest={entry.get(key)!r}, episode={header.get(key)!r}"
                )
        self._cache[index] = episode
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return episode

    def split(
        self,
        validation_fraction: float,
        seed: int,
    ) -> tuple["TeacherRolloutEpisodeDataset", "TeacherRolloutEpisodeDataset"]:
        """Return deterministic, episode-disjoint train and validation datasets."""

        fraction = float(validation_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in self.entries:
            groups[str(entry.get("object_name", "unknown"))].append(entry)
        train_entries: list[dict[str, Any]] = []
        validation_entries: list[dict[str, Any]] = []
        for object_name, entries in sorted(groups.items()):
            ranked = sorted(
                entries,
                key=lambda entry: hashlib.sha256(
                    f"{int(seed)}:{object_name}:{entry['file']}".encode("utf-8")
                ).digest(),
            )
            validation_count = int(round(len(ranked) * fraction))
            if fraction > 0.0 and len(ranked) > 1:
                validation_count = min(max(validation_count, 1), len(ranked) - 1)
            validation_entries.extend(ranked[:validation_count])
            train_entries.extend(ranked[validation_count:])
        if fraction > 0.0 and not validation_entries and len(train_entries) > 1:
            ranked = sorted(
                train_entries,
                key=lambda entry: hashlib.sha256(
                    f"{int(seed)}:fallback:{entry['file']}".encode("utf-8")
                ).digest(),
            )
            validation_entries.append(ranked[0])
            train_entries.remove(ranked[0])
        return (
            type(self)(
                self.data_dir,
                entries=train_entries,
                validate_hashes=False,
            ),
            type(self)(
                self.data_dir,
                entries=validation_entries,
                validate_hashes=False,
            ),
        )

    @staticmethod
    def _validate_episode(
        episode: dict[str, Any],
        path: Path,
        *,
        expected_task: Any,
        expected_schema: Any,
    ) -> None:
        if episode.get("format") != TEACHER_ROLLOUT_FORMAT:
            raise ValueError(f"Episode has an invalid format: {path}")
        if episode.get("schema_version") != TEACHER_ROLLOUT_SCHEMA_VERSION:
            raise ValueError(f"Episode has an unsupported schema version: {path}")
        if episode.get("task") != expected_task:
            raise ValueError(f"Episode task does not match its manifest: {path}")
        header = episode.get("episode")
        if not isinstance(header, dict) or header.get("success") is not True:
            raise ValueError(f"Dataset episode is not marked successful: {path}")
        observations = episode.get("observations")
        if not isinstance(observations, dict):
            raise ValueError(f"Episode observations must be a dictionary: {path}")
        missing_obs = [key for key in REQUIRED_OBSERVATIONS if key not in observations]
        if missing_obs:
            raise ValueError(f"Episode is missing observations {missing_obs}: {path}")
        missing_fields = [key for key in REQUIRED_TIME_FIELDS if key not in episode]
        if missing_fields:
            raise ValueError(f"Episode is missing fields {missing_fields}: {path}")

        lengths = _time_lengths(episode)
        unique_lengths = set(lengths.values())
        expected_length = int(header.get("length", -1))
        if unique_lengths != {expected_length} or expected_length < 1:
            raise ValueError(
                f"Episode time lengths are inconsistent in {path}: "
                f"header={expected_length}, fields={lengths}"
            )
        actual_schema = _tensor_schema(episode)
        if actual_schema != expected_schema:
            raise ValueError(f"Episode tensor schema does not match its manifest: {path}")
        expected_steps = torch.arange(expected_length, dtype=torch.int64)
        if not torch.equal(episode["step_id"].to(torch.int64), expected_steps):
            raise ValueError(f"Episode step_id is not contiguous from zero: {path}")
        reset_mask = episode["reset_mask"].bool()
        expected_reset = torch.zeros(expected_length, dtype=torch.bool)
        expected_reset[0] = True
        if not torch.equal(reset_mask, expected_reset):
            raise ValueError(f"Episode reset_mask is invalid: {path}")
        dones = episode["done"].bool()
        if bool(dones[:-1].any()) or not bool(dones[-1]):
            raise ValueError(f"Episode terminal marker is not last-only: {path}")
        for key in ("env_id", "episode_id", "object_id"):
            values = episode[key].reshape(-1)
            if not torch.equal(values, values[:1].expand_as(values)):
                raise ValueError(f"Episode field {key!r} changes over time: {path}")
        if not torch.equal(
            episode["teacher_target_action"], episode.get("privileged_actions")
        ):
            raise ValueError(f"Teacher target alias mismatch: {path}")
        if not torch.equal(episode["behavior_action"], episode["raw_action"]):
            raise ValueError(f"Behavior/raw action mismatch: {path}")
        expected_behavior = episode["teacher_target_action"] + episode["action_noise"]
        if not torch.allclose(
            episode["behavior_action"], expected_behavior, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"Behavior action does not equal target plus noise: {path}")
        low_dim = observations["student_low_dim"]
        action_dim = int(episode["executed_action"].shape[-1])
        if low_dim.shape[-1] < action_dim:
            raise ValueError(f"Student low-dimensional observation cannot contain last_action: {path}")
        saved_last_action = low_dim[..., -action_dim:]
        if not torch.allclose(
            saved_last_action[0],
            torch.zeros_like(saved_last_action[0]),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(f"Episode initial last_action is not reset to zero: {path}")
        if expected_length > 1 and not torch.allclose(
            saved_last_action[1:],
            episode["executed_action"][:-1],
            rtol=0.0,
            atol=1.0e-5,
        ):
            maximum_error = float(
                (saved_last_action[1:] - episode["executed_action"][:-1])
                .abs()
                .max()
            )
            raise ValueError(
                f"Episode last_action is not the previous executed action "
                f"(max_error={maximum_error:g}): {path}"
            )
        for key in ("camera_frame_id", "secondary_camera_frame_id"):
            if key in episode and not torch.equal(
                episode[key].to(torch.int64), episode["control_step"].to(torch.int64)
            ):
                raise ValueError(f"Episode {key} is not aligned with control_step: {path}")
        if "camera_frame_id" in episode and "secondary_camera_frame_id" in episode:
            if not torch.equal(
                episode["camera_frame_id"], episode["secondary_camera_frame_id"]
            ):
                raise ValueError(f"Dual-camera frame ids differ: {path}")


class TeacherRolloutSequenceDataset(Dataset):
    """Sample loss windows with exact episode-prefix recurrent reconstruction."""

    def __init__(
        self,
        episodes: TeacherRolloutEpisodeDataset,
        *,
        sequence_length: int,
        burn_in_length: int = 0,
        stride: int | None = None,
    ) -> None:
        self.episodes = episodes
        self.sequence_length = int(sequence_length)
        self.burn_in_length = int(burn_in_length)
        self.stride = self.sequence_length if stride is None else int(stride)
        if self.sequence_length < 1 or self.burn_in_length < 0 or self.stride < 1:
            raise ValueError("sequence_length/stride must be positive and burn_in_length non-negative")
        self.windows: list[tuple[int, int, int, int]] = []
        for episode_index, entry in enumerate(episodes.entries):
            length = int(entry["length"])
            for loss_start in range(0, length, self.stride):
                loss_end = min(loss_start + self.sequence_length, length)
                # Replaying from step zero reconstructs the exact recurrent state
                # without storing stale network hidden tensors in the dataset.
                context_start = 0
                self.windows.append(
                    (episode_index, context_start, loss_start, loss_end)
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, context_start, loss_start, loss_end = self.windows[int(index)]
        episode = self.episodes[episode_index]
        relative_loss_start = loss_start - context_start

        def slice_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: slice_value(item)
                    for key, item in value.items()
                    if key not in {"episode", "format", "schema_version", "task"}
                }
            if isinstance(value, torch.Tensor):
                return value[context_start:loss_end]
            return value

        sample = slice_value(episode)
        sample_length = loss_end - context_start
        loss_mask = torch.zeros(sample_length, dtype=torch.bool)
        loss_mask[relative_loss_start:] = True
        hidden_detach_mask = torch.zeros(sample_length, dtype=torch.bool)
        detach_after = relative_loss_start - self.burn_in_length - 1
        if detach_after >= 0:
            hidden_detach_mask[detach_after] = True
        sample["loss_mask"] = loss_mask
        sample["hidden_detach_mask"] = hidden_detach_mask
        sample["padding_mask"] = torch.ones(sample_length, dtype=torch.bool)
        sample["window"] = {
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "context_start": torch.tensor(context_start, dtype=torch.int64),
            "loss_start": torch.tensor(loss_start, dtype=torch.int64),
            "loss_end": torch.tensor(loss_end, dtype=torch.int64),
        }
        return sample


def collate_teacher_rollout_sequences(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length sequences and return tensors in time-major T x B order."""

    if not samples:
        raise ValueError("Cannot collate an empty sequence batch")
    lengths = [int(sample["loss_mask"].shape[0]) for sample in samples]
    maximum_length = max(lengths)

    def collate_values(values: Sequence[Any], key: str = "") -> Any:
        first = values[0]
        if isinstance(first, dict):
            common_keys = first.keys()
            if any(value.keys() != common_keys for value in values[1:]):
                raise ValueError(f"Sequence field keys differ at {key}")
            return {
                name: collate_values(
                    [value[name] for value in values],
                    f"{key}/{name}" if key else name,
                )
                for name in common_keys
            }
        if not isinstance(first, torch.Tensor):
            raise TypeError(f"Unsupported sequence value at {key}: {type(first).__name__}")
        if key.startswith("window/"):
            return torch.stack(list(values), dim=0)
        fill_value: bool | float | int = False
        if key in {"done", "dones"}:
            fill_value = True
        padded = first.new_full((maximum_length, len(values), *first.shape[1:]), fill_value)
        for batch_index, value in enumerate(values):
            padded[: value.shape[0], batch_index] = value
        return padded.contiguous()

    return collate_values(samples)


__all__ = [
    "REQUIRED_OBSERVATIONS",
    "TeacherRolloutEpisodeDataset",
    "TeacherRolloutSequenceDataset",
    "collate_teacher_rollout_sequences",
]
