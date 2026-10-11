# Copyright (c) 2021-2026, ETH Zurich, NVIDIA CORPORATION, and BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode-buffered storage for successful offline teacher rollouts."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch


TEACHER_ROLLOUT_FORMAT = "rsl_rl_teacher_rollout"
TEACHER_ROLLOUT_SCHEMA_VERSION = 1


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TeacherRolloutDatasetWriter:
    """Buffer parallel episodes in memory and atomically save successful ones."""

    def __init__(
        self,
        output_dir: str,
        *,
        num_envs: int,
        task_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs!r}.")
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                "Teacher-rollout output directory must be new or empty to avoid "
                f"overwriting data: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.task_name = str(task_name)
        self.metadata = dict(metadata or {})
        self._buffers: list[list[dict[str, Any]]] = [
            [] for _ in range(self.num_envs)
        ]
        self._episode_ids = list(range(self.num_envs))
        self._next_episode_id = self.num_envs
        self.saved_episodes: list[dict[str, Any]] = []
        self.failed_episodes: list[dict[str, Any]] = []
        self.failed_episode_reasons: Counter[str] = Counter()
        self.completed_episodes = 0
        self.incomplete_episodes = 0
        self._schema: dict[str, Any] | None = None

    @property
    def num_saved_episodes(self) -> int:
        return len(self.saved_episodes)

    def episode_id(self, env_id: int) -> int:
        self._validate_env_id(env_id)
        return self._episode_ids[env_id]

    def step_id(self, env_id: int) -> int:
        self._validate_env_id(env_id)
        return len(self._buffers[env_id])

    def append(self, env_id: int, sample: dict[str, Any]) -> None:
        """Append one pre-step-aligned transition to an environment's episode."""

        self._validate_env_id(env_id)
        step_id = len(self._buffers[env_id])
        item = dict(sample)
        item.update(
            {
                "env_id": torch.tensor(env_id, dtype=torch.int64),
                "episode_id": torch.tensor(
                    self._episode_ids[env_id], dtype=torch.int64
                ),
                "step_id": torch.tensor(step_id, dtype=torch.int64),
                "reset_mask": torch.tensor(step_id == 0, dtype=torch.bool),
            }
        )
        self._buffers[env_id].append(self._to_cpu_clone(item))

    def complete_episode(
        self,
        env_id: int,
        *,
        success: bool,
        failure_reason: str,
        object_name: str,
        save_success: bool = True,
    ) -> Path | None:
        """Commit a complete successful episode, or discard it otherwise."""

        self._validate_env_id(env_id)
        samples = self._buffers[env_id]
        if not samples:
            raise RuntimeError(f"Cannot complete empty episode for env {env_id}.")
        if not bool(samples[-1]["done"].item()):
            raise RuntimeError(
                f"Episode {self._episode_ids[env_id]} for env {env_id} is not terminal."
            )

        self.completed_episodes += 1
        output_path = None
        if success and save_success:
            episode_id = self._episode_ids[env_id]
            data = self._stack_samples(samples)
            episode_length = len(samples)
            data["episode_success"] = torch.ones(
                episode_length, dtype=torch.bool
            )
            data["episode_length"] = torch.full(
                (episode_length,), episode_length, dtype=torch.int64
            )
            # Preserve the online DAgger label name alongside the explicit
            # collection name without copying tensor storage.
            data["privileged_actions"] = data["teacher_target_action"]
            data["dones"] = data["done"]
            data["rewards"] = data["reward"]
            schema = self._tensor_schema(data)
            if self._schema is not None and schema != self._schema:
                raise RuntimeError(
                    "Teacher-rollout schema changed between successful episodes."
                )
            payload = {
                "format": TEACHER_ROLLOUT_FORMAT,
                "schema_version": TEACHER_ROLLOUT_SCHEMA_VERSION,
                "task": self.task_name,
                "episode": {
                    "env_id": env_id,
                    "episode_id": episode_id,
                    "object_name": object_name,
                    "success": True,
                    "length": episode_length,
                },
                **data,
            }
            output_path = self.output_dir / (
                f"episode_{episode_id:08d}_env_{env_id:05d}.pt"
            )
            temporary_path = output_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary_path)
            os.replace(temporary_path, output_path)
            output_hash = _sha256_path(output_path)

            if self._schema is None:
                self._schema = schema
            self.saved_episodes.append(
                {
                    "file": output_path.name,
                    "env_id": env_id,
                    "episode_id": episode_id,
                    "object_name": object_name,
                    "length": episode_length,
                    "sha256": output_hash,
                    "size_bytes": output_path.stat().st_size,
                }
            )
        elif success:
            self.failed_episode_reasons["successful_surplus"] += 1
            self.failed_episodes.append(
                {
                    "env_id": env_id,
                    "episode_id": self._episode_ids[env_id],
                    "object_name": object_name,
                    "length": len(samples),
                    "reason": "successful_surplus",
                    "success": True,
                }
            )
        else:
            reason = failure_reason or "unsuccessful"
            self.failed_episode_reasons[reason] += 1
            self.failed_episodes.append(
                {
                    "env_id": env_id,
                    "episode_id": self._episode_ids[env_id],
                    "object_name": object_name,
                    "length": len(samples),
                    "reason": reason,
                    "success": False,
                }
            )

        self._buffers[env_id] = []
        self._episode_ids[env_id] = self._next_episode_id
        self._next_episode_id += 1
        # Keep every already committed successful episode discoverable even if
        # the simulator process is interrupted before the requested target.
        self.finalize(collection_complete=False)
        return output_path

    def finalize(self, *, collection_complete: bool = True) -> Path:
        """Atomically snapshot the manifest without serializing live buffers."""

        self.incomplete_episodes = sum(bool(buffer) for buffer in self._buffers)
        manifest = {
            "format": TEACHER_ROLLOUT_FORMAT,
            "schema_version": TEACHER_ROLLOUT_SCHEMA_VERSION,
            "task": self.task_name,
            "collection_complete": bool(collection_complete),
            "metadata": self.metadata,
            "statistics": {
                "saved_successful_episodes": self.num_saved_episodes,
                "completed_episodes": self.completed_episodes,
                "discarded_incomplete_episodes": self.incomplete_episodes,
                "discarded_episode_reasons": dict(
                    sorted(self.failed_episode_reasons.items())
                ),
            },
            "schema": self._schema or {},
            "episodes": self.saved_episodes,
            "failed_episodes": self.failed_episodes,
        }
        manifest_path = self.output_dir / "manifest.json"
        temporary_path = manifest_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, manifest_path)
        return manifest_path

    def write_artifact(self, filename: str, value: Any) -> dict[str, Any]:
        """Atomically write one dataset-level PyTorch artifact and return its descriptor."""

        if Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError(f"Invalid teacher-rollout artifact filename: {filename!r}")
        output_path = self.output_dir / filename
        if output_path.exists():
            raise FileExistsError(f"Teacher-rollout artifact already exists: {output_path}")
        temporary_path = output_path.with_suffix(".pt.tmp")
        torch.save(value, temporary_path)
        os.replace(temporary_path, output_path)
        return {
            "file": filename,
            "sha256": _sha256_path(output_path),
            "size_bytes": output_path.stat().st_size,
        }

    def _validate_env_id(self, env_id: int) -> None:
        if not 0 <= int(env_id) < self.num_envs:
            raise IndexError(
                f"env_id must be in [0, {self.num_envs}), got {env_id!r}."
            )

    @classmethod
    def _to_cpu_clone(cls, value):
        if isinstance(value, dict):
            return {key: cls._to_cpu_clone(item) for key, item in value.items()}
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "Teacher-rollout samples may contain only tensors or nested dictionaries; "
                f"got {type(value).__name__}."
            )
        return value.detach().to(device="cpu").clone()

    @classmethod
    def _stack_samples(cls, samples: list[dict[str, Any]]) -> dict[str, Any]:
        keys = samples[0].keys()
        if any(sample.keys() != keys for sample in samples[1:]):
            raise RuntimeError("Teacher-rollout sample fields changed within an episode.")
        result = {}
        for key in keys:
            values = [sample[key] for sample in samples]
            if isinstance(values[0], dict):
                result[key] = cls._stack_samples(values)
            else:
                result[key] = torch.stack(values, dim=0).contiguous()
        return result

    @classmethod
    def _tensor_schema(cls, value, prefix: str = "") -> dict[str, Any]:
        schema = {}
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(item, dict):
                schema.update(cls._tensor_schema(item, name))
            elif isinstance(item, torch.Tensor):
                schema[name] = {
                    "shape": ["T", *list(item.shape[1:])],
                    "dtype": str(item.dtype).removeprefix("torch."),
                }
        return schema
