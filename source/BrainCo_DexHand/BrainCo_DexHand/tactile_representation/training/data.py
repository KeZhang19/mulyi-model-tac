"""Manifest-backed dataset loading and leakage-safe episode splitting."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_FIELDS = (
    "rgb",
    "depth_m",
    "marker_2d",
    "marker_valid",
    "episode_id",
    "episode_step",
    "env_id",
    "phase",
    "contact",
    "target_force_n",
    "presser_id",
    "sweep_stage",
)

MMAP_CACHE_DIRNAME = "mmap"
MMAP_CACHE_SCHEMA_VERSION = 1


def load_complete_manifest(dataset_root: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load and validate a completed merged tactile dataset manifest."""

    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Dataset is not complete: status={manifest.get('status')!r}, path={manifest_path}"
        )
    if int(manifest.get("sample_count", 0)) <= 0:
        raise ValueError(f"Dataset contains no samples: {manifest_path}")
    if not manifest.get("shards"):
        raise ValueError(f"Dataset manifest contains no shards: {manifest_path}")
    schema = manifest.get("schema", {})
    missing = sorted(set(REQUIRED_FIELDS) - set(schema))
    if missing:
        raise ValueError(f"Dataset schema is missing required fields: {missing}")
    plans = manifest.get("metadata", {}).get("episode_plans", [])
    if not plans:
        raise ValueError("Dataset manifest does not contain metadata.episode_plans")
    return root, manifest


def manifest_sha256(dataset_root: str | Path) -> str:
    """Return a stable fingerprint used to reject incompatible resume attempts."""

    root = Path(dataset_root).expanduser().resolve()
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _expected_mmap_fields(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sample_count = int(manifest["sample_count"])
    schema = manifest["schema"]
    return {
        name: {
            "file": f"{name}.npy",
            "dtype": str(np.dtype(schema[name]["dtype"])),
            "shape": [sample_count, *[int(value) for value in schema[name]["sample_shape"]]],
        }
        for name in REQUIRED_FIELDS
    }


def _open_mmap_cache(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    required: bool,
) -> tuple[Path, dict[str, np.memmap]] | None:
    cache_root = root / MMAP_CACHE_DIRNAME
    cache_manifest_path = cache_root / "manifest.json"
    if not cache_root.exists():
        if required:
            raise FileNotFoundError(
                f"Memory-mapped cache does not exist: {cache_root}. "
                "Run scripts/tactile_representation/prepare_mmap_cache.py first."
            )
        return None
    if not cache_root.is_dir() or not cache_manifest_path.is_file():
        raise ValueError(f"Memory-mapped cache is incomplete: {cache_root}")

    with cache_manifest_path.open("r", encoding="utf-8") as file_obj:
        cache_manifest = json.load(file_obj)
    if int(cache_manifest.get("schema_version", -1)) != MMAP_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported mmap cache schema version: {cache_manifest.get('schema_version')!r}"
        )
    if cache_manifest.get("status") != "complete":
        raise ValueError(
            f"Memory-mapped cache is not complete: status={cache_manifest.get('status')!r}"
        )
    source_fingerprint = manifest_sha256(root)
    if cache_manifest.get("source_manifest_sha256") != source_fingerprint:
        raise ValueError(
            "Memory-mapped cache was created from a different dataset manifest; "
            f"remove {cache_root} and rebuild it"
        )
    if int(cache_manifest.get("sample_count", -1)) != int(manifest["sample_count"]):
        raise ValueError("Memory-mapped cache sample count does not match the dataset manifest")

    expected_fields = _expected_mmap_fields(manifest)
    cached_fields = cache_manifest.get("fields")
    if not isinstance(cached_fields, dict) or set(cached_fields) != set(expected_fields):
        raise ValueError("Memory-mapped cache field set does not match the training schema")

    arrays: dict[str, np.memmap] = {}
    for name, expected in expected_fields.items():
        recorded = cached_fields[name]
        if recorded != expected:
            raise ValueError(
                f"Memory-mapped cache field {name!r} has metadata {recorded}, expected {expected}"
            )
        path = (cache_root / str(recorded["file"])).resolve()
        if not path.is_relative_to(cache_root.resolve()):
            raise ValueError(f"Memory-mapped cache field escapes the cache root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Memory-mapped cache field does not exist: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.memmap):
            raise TypeError(f"Memory-mapped cache field {path} is not a .npy memmap")
        if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
            raise ValueError(
                f"Memory-mapped cache field {name!r} has dtype={array.dtype}, "
                f"shape={list(array.shape)}, expected={expected}"
            )
        arrays[name] = array
    return cache_root, arrays


def build_mmap_cache(
    dataset_root: str | Path,
    *,
    progress: Callable[[int, int, Path], None] | None = None,
) -> Path:
    """Convert NPZ shards once into validated, read-only memory-mapped arrays."""

    root, manifest = load_complete_manifest(dataset_root)
    existing = _open_mmap_cache(root, manifest, required=False)
    if existing is not None:
        return existing[0]

    cache_root = root / MMAP_CACHE_DIRNAME
    if cache_root.exists():
        raise FileExistsError(
            f"An invalid mmap cache already exists at {cache_root}; remove only that cache "
            "directory before rebuilding"
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{MMAP_CACHE_DIRNAME}-", dir=root))
    expected_fields = _expected_mmap_fields(manifest)
    destinations: dict[str, np.memmap] = {}
    try:
        for name, field in expected_fields.items():
            destinations[name] = np.lib.format.open_memmap(
                staging / str(field["file"]),
                mode="w+",
                dtype=np.dtype(field["dtype"]),
                shape=tuple(field["shape"]),
            )

        cursor = 0
        shards = manifest["shards"]
        for shard_number, shard in enumerate(shards, start=1):
            path = (root / str(shard["file"])).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"Dataset shard does not exist inside the root: {path}")
            shard_count = int(shard["sample_count"])
            stop = cursor + shard_count
            if stop > int(manifest["sample_count"]):
                raise ValueError(f"Shard sample counts exceed the manifest total at {path}")
            with np.load(path, allow_pickle=False) as source:
                for name, field in expected_fields.items():
                    source_array = np.asarray(source[name])
                    expected_shape = (shard_count, *tuple(field["shape"])[1:])
                    if source_array.shape != expected_shape:
                        raise ValueError(
                            f"Shard {path} field {name!r} has shape {source_array.shape}, "
                            f"expected {expected_shape}"
                        )
                    if str(source_array.dtype) != field["dtype"]:
                        raise ValueError(
                            f"Shard {path} field {name!r} has dtype={source_array.dtype}, "
                            f"expected {field['dtype']}"
                        )
                    destinations[name][cursor:stop] = source_array
            cursor = stop
            if progress is not None:
                progress(shard_number, len(shards), path)
        if cursor != int(manifest["sample_count"]):
            raise ValueError(
                f"Converted {cursor} samples, expected {int(manifest['sample_count'])}"
            )
        for array in destinations.values():
            array.flush()
        destinations.clear()

        cache_manifest = {
            "schema_version": MMAP_CACHE_SCHEMA_VERSION,
            "status": "complete",
            "source_manifest_sha256": manifest_sha256(root),
            "sample_count": int(manifest["sample_count"]),
            "fields": expected_fields,
        }
        _write_json_atomic(staging / "manifest.json", cache_manifest)
        os.replace(staging, cache_root)
        _open_mmap_cache(root, manifest, required=True)
        return cache_root
    except BaseException:
        destinations.clear()
        if staging.exists():
            shutil.rmtree(staging)
        raise


@dataclass(frozen=True)
class EpisodeSplits:
    """Episode IDs assigned to mutually exclusive training partitions."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
        }
        for left_name, left in groups.items():
            for right_name, right in groups.items():
                if left_name >= right_name:
                    continue
                overlap = left & right
                if overlap:
                    raise ValueError(
                        f"Episode splits {left_name} and {right_name} overlap: {sorted(overlap)}"
                    )
        if not self.train or not self.validation or not self.test:
            raise ValueError("Train, validation, and test episode splits must all be non-empty")

    def to_json(self) -> dict[str, list[int]]:
        return {name: list(values) for name, values in asdict(self).items()}

    @classmethod
    def from_json(cls, value: Mapping[str, Sequence[int]]) -> "EpisodeSplits":
        return cls(
            train=tuple(int(item) for item in value["train"]),
            validation=tuple(int(item) for item in value["validation"]),
            test=tuple(int(item) for item in value["test"]),
        )


def _condition_key(plan: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identify a physical condition while deliberately ignoring presser type."""

    return (
        str(plan.get("sweep_stage", "unknown")),
        float(plan["target_force_n"]),
        float(plan["offset_u_m"]),
        float(plan["offset_v_m"]),
        str(plan["tilt_axis"]),
        float(plan["tilt_deg"]),
        str(plan["slide_axis"]),
        float(plan["slide_distance_m"]),
        int(plan["baseline_steps"]),
        int(plan["press_steps"]),
        int(plan["slide_steps"]),
        int(plan["hold_steps"]),
    )


def _stage_holdout_allocations(
    stage_counts: Mapping[str, int], fraction: float
) -> dict[str, int]:
    """Use largest remainders to hit the global fraction while retaining every stage."""

    total_conditions = sum(int(value) for value in stage_counts.values())
    target = max(1, int(round(total_conditions * float(fraction))))
    allocations: dict[str, int] = {}
    capacities: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for stage, raw_count in stage_counts.items():
        count = int(raw_count)
        capacities[stage] = max(0, (count - 1) // 2)
        quota = count * float(fraction)
        allocations[stage] = min(capacities[stage], max(1, int(math.floor(quota))))
        remainders[stage] = quota - math.floor(quota)
    while sum(allocations.values()) < target:
        candidates = [
            stage for stage in allocations if allocations[stage] < capacities[stage]
        ]
        if not candidates:
            break
        selected = max(candidates, key=lambda stage: (remainders[stage], stage_counts[stage]))
        allocations[selected] += 1
        remainders[selected] = 0.0
    while sum(allocations.values()) > target:
        candidates = [stage for stage in allocations if allocations[stage] > 1]
        if not candidates:
            break
        selected = min(candidates, key=lambda stage: (remainders[stage], stage_counts[stage]))
        allocations[selected] -= 1
    return allocations


def split_episode_ids(
    manifest: Mapping[str, Any],
    *,
    seed: int = 7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> EpisodeSplits:
    """Split whole physical conditions, never adjacent frames, across partitions."""

    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if float(validation_fraction) + float(test_fraction) >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be smaller than 1")

    plans = manifest.get("metadata", {}).get("episode_plans", [])
    if not plans:
        raise ValueError("Cannot split a dataset without episode plans")
    condition_episodes: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for plan in plans:
        condition_episodes[_condition_key(plan)].append(int(plan["episode_id"]))

    by_stage: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for condition in condition_episodes:
        by_stage[str(condition[0])].append(condition)

    rng = random.Random(int(seed))
    assignments: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    stage_counts = {stage: len(conditions) for stage, conditions in by_stage.items()}
    validation_allocations = _stage_holdout_allocations(stage_counts, validation_fraction)
    test_allocations = _stage_holdout_allocations(stage_counts, test_fraction)
    for stage in sorted(by_stage):
        conditions = sorted(by_stage[stage], key=repr)
        rng.shuffle(conditions)
        validation_count = validation_allocations[stage]
        test_count = test_allocations[stage]
        validation_conditions = conditions[:validation_count]
        test_conditions = conditions[validation_count : validation_count + test_count]
        train_conditions = conditions[validation_count + test_count :]
        for split_name, selected in (
            ("train", train_conditions),
            ("validation", validation_conditions),
            ("test", test_conditions),
        ):
            for condition in selected:
                assignments[split_name].extend(condition_episodes[condition])

    expected = {int(plan["episode_id"]) for plan in plans}
    assigned = set().union(*(set(values) for values in assignments.values()))
    if assigned != expected:
        raise RuntimeError(
            f"Episode splitting lost or added IDs; missing={sorted(expected-assigned)}, "
            f"extra={sorted(assigned-expected)}"
        )
    return EpisodeSplits(
        train=tuple(sorted(assignments["train"])),
        validation=tuple(sorted(assignments["validation"])),
        test=tuple(sorted(assignments["test"])),
    )


@dataclass(frozen=True)
class TactileNormalization:
    """Numerical scales shared by the dataset, losses, and checkpoints."""

    image_height: int
    image_width: int
    marker_count: int
    depth_scale_m: float = 0.003
    # Marker motion is physically a few pixels, not a fraction of the full image.
    # Keeping this scale near the observed motion range prevents dx/dy from being
    # two orders of magnitude smaller than the static x/y calibration coordinates.
    marker_motion_scale_px: float = 5.0

    def __post_init__(self) -> None:
        if min(self.image_height, self.image_width, self.marker_count) <= 0:
            raise ValueError("Image dimensions and marker_count must be positive")
        if float(self.depth_scale_m) <= 0.0:
            raise ValueError("depth_scale_m must be positive")
        if float(self.marker_motion_scale_px) <= 0.0:
            raise ValueError("marker_motion_scale_px must be positive")

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        depth_scale_m: float = 0.003,
        marker_motion_scale_px: float = 5.0,
    ) -> "TactileNormalization":
        schema = manifest["schema"]
        rgb_shape = tuple(schema["rgb"]["sample_shape"])
        marker_shape = tuple(schema["marker_2d"]["sample_shape"])
        if len(rgb_shape) != 3 or len(marker_shape) != 2:
            raise ValueError(f"Unexpected RGB/Marker schema: rgb={rgb_shape}, marker={marker_shape}")
        return cls(
            image_height=int(rgb_shape[1]),
            image_width=int(rgb_shape[2]),
            marker_count=int(marker_shape[0]),
            depth_scale_m=float(depth_scale_m),
            marker_motion_scale_px=float(marker_motion_scale_px),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class NpzTactileDataset(Dataset[dict[str, torch.Tensor]]):
    """Read tactile samples from mmap when available, with NPZ as a compatible fallback."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        episode_ids: Sequence[int],
        normalization: TactileNormalization,
        cache_size: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if int(cache_size) <= 0:
            raise ValueError("cache_size must be positive")
        if str(backend) not in {"auto", "npz", "mmap"}:
            raise ValueError("backend must be one of: auto, npz, mmap")
        self.root, self.manifest = load_complete_manifest(dataset_root)
        self.normalization = normalization
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._mmap_arrays: dict[str, np.memmap] = {}
        self._contact_labels: list[bool] = []
        # The collector records an unpressed baseline phase for every episode.
        # Keep one RGB frame per episode in memory so the reference can be
        # returned with every sample without adding a new on-disk field or
        # changing the old collection format.
        phase_codes = self.manifest.get("metadata", {}).get("phase_codes", {})
        self._baseline_phase = int(phase_codes.get("baseline", 0))
        self._rgb_references: dict[int, np.ndarray] = {}
        self._rgb_reference_fallback_episodes: set[int] = set()
        selected = {int(value) for value in episode_ids}
        if not selected:
            raise ValueError("episode_ids must not be empty")

        self._shards: list[Path] = []
        self._entries: list[int | tuple[int, int]] = []
        found_episode_ids: set[int] = set()
        mmap_cache = (
            None
            if str(backend) == "npz"
            else _open_mmap_cache(
                self.root,
                self.manifest,
                required=str(backend) == "mmap",
            )
        )
        if mmap_cache is not None:
            self.backend = "mmap"
            _, self._mmap_arrays = mmap_cache
            episode_values = self._mmap_arrays["episode_id"]
            for offset, episode_id in enumerate(episode_values.tolist()):
                if int(episode_id) in selected:
                    self._entries.append(offset)
                    self._contact_labels.append(bool(self._mmap_arrays["contact"][offset]))
                    episode_id_int = int(episode_id)
                    found_episode_ids.add(episode_id_int)
                    if (
                        episode_id_int not in self._rgb_references
                        and int(self._mmap_arrays["phase"][offset]) == self._baseline_phase
                    ):
                        self._rgb_references[episode_id_int] = np.asarray(
                            self._mmap_arrays["rgb"][offset]
                        ).copy()
        else:
            self.backend = "npz"
            self._index_npz_entries(selected, found_episode_ids)
        missing = selected - found_episode_ids
        if missing:
            raise ValueError(f"Selected episode IDs are absent from the dataset: {sorted(missing)}")
        self._fill_missing_rgb_references()
        self._episode_ids = tuple(sorted(found_episode_ids))

    def _index_npz_entries(
        self,
        selected: set[int],
        found_episode_ids: set[int],
    ) -> None:
        for shard_index, shard in enumerate(self.manifest["shards"]):
            path = (self.root / str(shard["file"])).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"Shard path escapes the dataset root: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Dataset shard does not exist: {path}")
            self._shards.append(path)
            with np.load(path, allow_pickle=False) as arrays:
                episode_values = np.asarray(arrays["episode_id"], dtype=np.int64)
                contact_values = np.asarray(arrays["contact"], dtype=bool)
                phase_values = np.asarray(arrays["phase"], dtype=np.int8)
                rgb_values = np.asarray(arrays["rgb"])
            for offset, episode_id in enumerate(episode_values.tolist()):
                if int(episode_id) in selected:
                    self._entries.append((shard_index, offset))
                    self._contact_labels.append(bool(contact_values[offset]))
                    episode_id_int = int(episode_id)
                    found_episode_ids.add(episode_id_int)
                    if (
                        episode_id_int not in self._rgb_references
                        and int(phase_values[offset]) == self._baseline_phase
                    ):
                        self._rgb_references[episode_id_int] = np.asarray(
                            rgb_values[offset]
                        ).copy()

    def _fill_missing_rgb_references(self) -> None:
        """Use the first episode frame only for legacy datasets without baseline rows."""

        first_entries: dict[int, int | tuple[int, int]] = {}
        for entry in self._entries:
            if self.backend == "mmap":
                offset = int(entry)
                episode_id = int(self._mmap_arrays["episode_id"][offset])
            else:
                shard_index, offset = entry
                arrays = self._load_shard(int(shard_index))
                episode_id = int(arrays["episode_id"][offset])
            first_entries.setdefault(episode_id, entry)

        for episode_id, entry in first_entries.items():
            if episode_id in self._rgb_references:
                continue
            if self.backend == "mmap":
                rgb = self._mmap_arrays["rgb"][int(entry)]
            else:
                shard_index, offset = entry
                rgb = self._load_shard(int(shard_index))["rgb"][int(offset)]
            self._rgb_references[episode_id] = np.asarray(rgb).copy()
            self._rgb_reference_fallback_episodes.add(episode_id)

    @property
    def rgb_reference_fallback_episodes(self) -> tuple[int, ...]:
        """Episode IDs that had no explicit baseline and used a legacy fallback."""

        return tuple(sorted(self._rgb_reference_fallback_episodes))

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return self._episode_ids

    @property
    def contact_labels(self) -> tuple[bool, ...]:
        """Return lightweight labels used by the training sampler without loading images."""

        return tuple(self._contact_labels)

    def __len__(self) -> int:
        return len(self._entries)

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        cached = self._cache.pop(shard_index, None)
        if cached is not None:
            self._cache[shard_index] = cached
            return cached
        path = self._shards[shard_index]
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: np.asarray(source[name]) for name in REQUIRED_FIELDS}
        sample_count = int(arrays["episode_id"].shape[0])
        for name, array in arrays.items():
            if array.ndim < 1 or int(array.shape[0]) != sample_count:
                raise ValueError(
                    f"Shard {path} field {name!r} has inconsistent shape {array.shape}"
                )
        self._cache[shard_index] = arrays
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self._entries[index]
        if self.backend == "mmap":
            arrays = self._mmap_arrays
            offset = int(entry)
        else:
            shard_index, offset = entry
            arrays = self._load_shard(int(shard_index))
        norm = self.normalization

        rgb = torch.tensor(arrays["rgb"][offset], dtype=torch.float32).div_(255.0)
        episode_id = int(arrays["episode_id"][offset])
        rgb_reference = torch.tensor(
            self._rgb_references[episode_id], dtype=torch.float32
        ).div_(255.0)
        depth = torch.tensor(arrays["depth_m"][offset], dtype=torch.float32)
        depth = depth.div(float(norm.depth_scale_m)).clamp_(0.0, 1.0)
        marker_valid = torch.tensor(arrays["marker_valid"][offset], dtype=torch.bool)
        marker = torch.tensor(arrays["marker_2d"][offset], dtype=torch.float32)
        marker[..., 0].div_(max(1, norm.image_width - 1))
        marker[..., 1].div_(max(1, norm.image_height - 1))
        marker[..., 2].div_(float(norm.marker_motion_scale_px))
        marker[..., 3].div_(float(norm.marker_motion_scale_px))
        marker[..., 4] = marker_valid.to(marker.dtype)
        marker[~marker_valid, :4] = 0.0

        result = {
            "rgb": rgb,
            "rgb_reference": rgb_reference,
            "depth": depth,
            "marker": marker,
            "marker_valid": marker_valid,
        }
        for name in (
            "episode_id",
            "episode_step",
            "env_id",
            "phase",
            "presser_id",
            "sweep_stage",
        ):
            result[name] = torch.tensor(int(arrays[name][offset]), dtype=torch.int64)
        result["contact"] = torch.tensor(bool(arrays["contact"][offset]), dtype=torch.bool)
        result["target_force_n"] = torch.tensor(
            float(arrays["target_force_n"][offset]), dtype=torch.float32
        )
        return result
