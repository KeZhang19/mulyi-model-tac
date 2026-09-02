"""Pure-Python utilities for synchronized cross-modal tactile collection.

The Isaac Sim entrypoint lives in ``scripts/tactile_representation``.  This
module intentionally has no Isaac Lab dependency so collection plans, sample
validation, and dataset writing can be tested without starting the simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PHASE_CODES = {
    "baseline": 0,
    "loading": 1,
    "sliding": 2,
    "holding": 3,
}

SWEEP_STAGE_CODES = {
    "random": 0,
    "force": 1,
    "offset_u": 2,
    "offset_v": 3,
    "tilt": 4,
    "slide": 5,
}


@dataclass(frozen=True)
class EpisodePlan:
    """One reproducible index-finger contact trajectory."""

    episode_id: int
    presser: str
    target_force_n: float
    offset_u_m: float
    offset_v_m: float
    tilt_axis: str
    tilt_deg: float
    slide_axis: str
    slide_distance_m: float
    baseline_steps: int
    press_steps: int
    slide_steps: int
    hold_steps: int
    sweep_stage: str = "random"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _stratified_values(
    values: Sequence[Any],
    count: int,
    rng: np.random.Generator,
) -> list[Any]:
    """Repeat shuffled value blocks so short runs still cover every choice."""

    if not values:
        raise ValueError("At least one value is required")
    output: list[Any] = []
    while len(output) < count:
        indices = rng.permutation(len(values))
        output.extend(values[int(index)] for index in indices)
    return output[:count]


def make_episode_plans(
    *,
    episode_count: int,
    seed: int,
    pressers: Sequence[str],
    force_levels_n: Sequence[float],
    offset_u_range_m: float,
    offset_v_range_m: float,
    tilt_max_deg: float,
    tilt_probability: float,
    slide_max_m: float,
    slide_probability: float,
    baseline_steps: int,
    press_steps: int,
    slide_steps: int,
    hold_steps: int,
) -> list[EpisodePlan]:
    """Build deterministic, varied contact plans for the index fingertip."""

    if int(episode_count) <= 0:
        raise ValueError("episode_count must be positive")
    if any(int(value) < 0 for value in (baseline_steps, slide_steps, hold_steps)):
        raise ValueError("baseline_steps, slide_steps, and hold_steps must be non-negative")
    if int(press_steps) <= 0:
        raise ValueError("press_steps must be positive")
    if not 0.0 <= float(tilt_probability) <= 1.0:
        raise ValueError("tilt_probability must be in [0, 1]")
    if not 0.0 <= float(slide_probability) <= 1.0:
        raise ValueError("slide_probability must be in [0, 1]")
    if any(float(value) < 0.0 for value in (offset_u_range_m, offset_v_range_m, tilt_max_deg, slide_max_m)):
        raise ValueError("offset, tilt, and slide ranges must be non-negative")
    if not force_levels_n or any(float(value) <= 0.0 for value in force_levels_n):
        raise ValueError("force_levels_n must contain positive values")

    rng = np.random.default_rng(int(seed))
    count = int(episode_count)
    presser_values = _stratified_values(tuple(str(value) for value in pressers), count, rng)
    force_values = _stratified_values(tuple(float(value) for value in force_levels_n), count, rng)
    tilt_axes = ("+u", "-u", "+v", "-v")
    slide_axes = ("+u", "-u", "+v", "-v")

    plans: list[EpisodePlan] = []
    for episode_id in range(count):
        use_tilt = float(tilt_max_deg) > 0.0 and bool(rng.random() < float(tilt_probability))
        use_slide = float(slide_max_m) > 0.0 and bool(rng.random() < float(slide_probability))
        tilt_axis = str(rng.choice(tilt_axes)) if use_tilt else "none"
        slide_axis = str(rng.choice(slide_axes))
        plans.append(
            EpisodePlan(
                episode_id=episode_id,
                presser=presser_values[episode_id],
                target_force_n=force_values[episode_id],
                offset_u_m=float(rng.uniform(-offset_u_range_m, offset_u_range_m)),
                offset_v_m=float(rng.uniform(-offset_v_range_m, offset_v_range_m)),
                tilt_axis=tilt_axis,
                tilt_deg=float(rng.uniform(0.0, tilt_max_deg)) if use_tilt else 0.0,
                slide_axis=slide_axis,
                slide_distance_m=float(rng.uniform(0.0, slide_max_m)) if use_slide else 0.0,
                baseline_steps=int(baseline_steps),
                press_steps=int(press_steps),
                slide_steps=int(slide_steps) if use_slide else 0,
                hold_steps=int(hold_steps),
            )
        )
    return plans


def _regular_ladder(maximum: float, step: float, *, name: str) -> list[float]:
    """Return ``[0, step, ..., maximum]`` and reject irregular final steps."""

    maximum = float(maximum)
    step = float(step)
    if maximum < 0.0:
        raise ValueError(f"{name} maximum must be non-negative")
    if maximum == 0.0:
        return [0.0]
    if step <= 0.0:
        raise ValueError(f"{name} step must be positive when its maximum is non-zero")
    step_count = maximum / step
    rounded_count = int(round(step_count))
    if rounded_count <= 0 or not np.isclose(step_count, rounded_count, rtol=0.0, atol=1.0e-7):
        raise ValueError(f"{name} maximum must be an exact multiple of its step")
    return [float(index * step) for index in range(rounded_count + 1)]


def make_sweep_episode_plans(
    *,
    episode_count: int | None,
    pressers: Sequence[str],
    force_levels_n: Sequence[float],
    reference_force_n: float,
    offset_u_range_m: float,
    offset_v_range_m: float,
    offset_step_m: float,
    tilt_max_deg: float,
    tilt_step_deg: float,
    slide_max_m: float,
    slide_step_m: float,
    baseline_steps: int,
    press_steps: int,
    slide_steps: int,
    hold_steps: int,
) -> list[EpisodePlan]:
    """Build a deterministic one-factor-at-a-time staircase protocol.

    The full protocol scans force, U position, V position, tilt, and sliding in
    that order.  Every condition is emitted once for each presser before the
    next condition, so a truncated run remains balanced across pressers.
    """

    if episode_count is not None and int(episode_count) <= 0:
        raise ValueError("episode_count must be positive when provided")
    if not pressers:
        raise ValueError("pressers must contain at least one value")
    if not force_levels_n or any(float(value) <= 0.0 for value in force_levels_n):
        raise ValueError("force_levels_n must contain positive values")
    if float(reference_force_n) <= 0.0:
        raise ValueError("reference_force_n must be positive")
    if any(int(value) < 0 for value in (baseline_steps, slide_steps, hold_steps)):
        raise ValueError("baseline_steps, slide_steps, and hold_steps must be non-negative")
    if int(press_steps) <= 0:
        raise ValueError("press_steps must be positive")

    offset_u_positive = _regular_ladder(offset_u_range_m, offset_step_m, name="offset U")
    offset_v_positive = _regular_ladder(offset_v_range_m, offset_step_m, name="offset V")
    tilt_values = _regular_ladder(tilt_max_deg, tilt_step_deg, name="tilt")
    slide_values = _regular_ladder(slide_max_m, slide_step_m, name="slide")
    offset_u_values = [-value for value in reversed(offset_u_positive[1:])] + offset_u_positive
    offset_v_values = [-value for value in reversed(offset_v_positive[1:])] + offset_v_positive

    conditions: list[dict[str, Any]] = []

    def append_condition(
        stage: str,
        *,
        force_n: float = float(reference_force_n),
        offset_u_m: float = 0.0,
        offset_v_m: float = 0.0,
        tilt_axis: str = "none",
        tilt_deg: float = 0.0,
        slide_axis: str = "+u",
        slide_distance_m: float = 0.0,
    ) -> None:
        conditions.append(
            {
                "sweep_stage": stage,
                "target_force_n": float(force_n),
                "offset_u_m": float(offset_u_m),
                "offset_v_m": float(offset_v_m),
                "tilt_axis": str(tilt_axis),
                "tilt_deg": float(tilt_deg),
                "slide_axis": str(slide_axis),
                "slide_distance_m": float(slide_distance_m),
            }
        )

    for force_n in force_levels_n:
        append_condition("force", force_n=float(force_n))
    for offset_m in offset_u_values:
        append_condition("offset_u", offset_u_m=offset_m)
    for offset_m in offset_v_values:
        append_condition("offset_v", offset_v_m=offset_m)

    append_condition("tilt")
    for axis in ("+u", "-u", "+v", "-v"):
        for tilt_deg in tilt_values[1:]:
            append_condition("tilt", tilt_axis=axis, tilt_deg=tilt_deg)

    append_condition("slide")
    for axis in ("+u", "-u", "+v", "-v"):
        for distance_m in slide_values[1:]:
            append_condition("slide", slide_axis=axis, slide_distance_m=distance_m)

    expanded_conditions = [
        (str(presser), condition)
        for condition in conditions
        for presser in pressers
    ]
    count = len(expanded_conditions) if episode_count is None else int(episode_count)
    plans: list[EpisodePlan] = []
    for episode_id in range(count):
        presser, condition = expanded_conditions[episode_id % len(expanded_conditions)]
        distance_m = float(condition["slide_distance_m"])
        plans.append(
            EpisodePlan(
                episode_id=episode_id,
                presser=presser,
                target_force_n=float(condition["target_force_n"]),
                offset_u_m=float(condition["offset_u_m"]),
                offset_v_m=float(condition["offset_v_m"]),
                tilt_axis=str(condition["tilt_axis"]),
                tilt_deg=float(condition["tilt_deg"]),
                slide_axis=str(condition["slide_axis"]),
                slide_distance_m=distance_m,
                baseline_steps=int(baseline_steps),
                press_steps=int(press_steps),
                slide_steps=int(slide_steps) if distance_m > 0.0 else 0,
                hold_steps=int(hold_steps),
                sweep_stage=str(condition["sweep_stage"]),
            )
        )
    return plans


def batch_episode_plans(
    plans: Sequence[EpisodePlan],
    batch_size: int,
) -> list[list[EpisodePlan]]:
    """Group compatible plans into fixed-schedule multi-environment waves."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    grouped: dict[tuple[str, int, int, int, int], list[EpisodePlan]] = {}
    for plan in plans:
        signature = (
            str(plan.presser),
            int(plan.baseline_steps),
            int(plan.press_steps),
            int(plan.slide_steps),
            int(plan.hold_steps),
        )
        grouped.setdefault(signature, []).append(plan)
    waves: list[list[EpisodePlan]] = []
    for compatible in grouped.values():
        for start in range(0, len(compatible), int(batch_size)):
            waves.append(compatible[start : start + int(batch_size)])
    return waves


def marker_flow_to_features(
    marker_flow: np.ndarray,
    marker_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``[start,end]`` pixel flow to ``[x0,y0,dx,dy,valid]``."""

    flow = np.asarray(marker_flow, dtype=np.float32)
    valid = np.asarray(marker_valid, dtype=bool).reshape(-1)
    if flow.ndim != 3 or flow.shape[0] != 2 or flow.shape[-1] != 2:
        raise ValueError(f"marker_flow must have shape [2,N,2], got {flow.shape}")
    if int(flow.shape[1]) != int(valid.shape[0]):
        raise ValueError(
            f"marker_valid length {valid.shape[0]} does not match flow count {flow.shape[1]}"
        )
    finite = np.isfinite(flow).all(axis=(0, 2))
    valid = valid & finite
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    delta = flow[1] - flow[0]
    features = np.concatenate(
        (flow[0], delta, valid[:, None].astype(np.float32)),
        axis=-1,
    ).astype(np.float32, copy=False)
    features[~valid] = 0.0
    return np.ascontiguousarray(features), np.ascontiguousarray(valid)


def make_cross_modal_sample(
    *,
    rgb_chw: np.ndarray,
    depth_m: np.ndarray,
    marker_flow: np.ndarray,
    marker_valid: np.ndarray,
    marker_displacement_3d_m: np.ndarray | None,
    scalar_fields: Mapping[str, Any],
    expected_marker_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Normalize and validate one synchronized single-finger sample."""

    rgb = np.asarray(rgb_chw)
    if rgb.ndim != 3 or int(rgb.shape[0]) != 3:
        raise ValueError(f"rgb_chw must have shape [3,H,W], got {rgb.shape}")
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(np.rint(np.nan_to_num(rgb) * 255.0), 0.0, 255.0).astype(np.uint8)
    else:
        rgb = np.clip(np.nan_to_num(rgb), 0, 255).astype(np.uint8)

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[None, ...]
    if depth.ndim != 3 or int(depth.shape[0]) != 1:
        raise ValueError(f"depth_m must have shape [H,W] or [1,H,W], got {depth.shape}")
    if tuple(depth.shape[-2:]) != tuple(rgb.shape[-2:]):
        raise ValueError(f"RGB/depth spatial mismatch: {rgb.shape[-2:]} versus {depth.shape[-2:]}")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.maximum(depth, 0.0).astype(np.float32, copy=False)

    marker, valid = marker_flow_to_features(marker_flow, marker_valid)
    marker_count = int(marker.shape[0])
    if expected_marker_count is not None and marker_count != int(expected_marker_count):
        raise ValueError(f"Expected {expected_marker_count} markers, got {marker_count}")

    sample: dict[str, np.ndarray] = {
        "rgb": np.ascontiguousarray(rgb),
        "depth_m": np.ascontiguousarray(depth),
        "marker_2d": marker,
        "marker_valid": valid,
    }
    if marker_displacement_3d_m is not None:
        displacement = np.asarray(marker_displacement_3d_m, dtype=np.float32)
        if displacement.shape != (marker_count, 3):
            raise ValueError(
                f"marker_displacement_3d_m must have shape {(marker_count, 3)}, got {displacement.shape}"
            )
        displacement = np.nan_to_num(displacement, nan=0.0, posinf=0.0, neginf=0.0)
        displacement[~valid] = 0.0
        sample["marker_displacement_3d_m"] = np.ascontiguousarray(displacement)

    for name, value in scalar_fields.items():
        array = np.asarray(value)
        if array.dtype.kind in {"O", "U", "S"}:
            raise TypeError(f"Scalar field {name!r} must be numeric, got dtype {array.dtype}")
        if array.ndim != 0:
            raise ValueError(f"Scalar field {name!r} must be scalar, got shape {array.shape}")
        sample[str(name)] = array
    return sample


class NpzShardWriter:
    """Write fixed-schema samples to atomic, optionally compressed NPZ shards."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        shard_size: int,
        compressed: bool,
        metadata: Mapping[str, Any],
    ) -> None:
        if int(shard_size) <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {self.output_dir}. Use a new directory to avoid overwriting data."
            )
        self.shards_dir = self.output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.shard_size = int(shard_size)
        self.compressed = bool(compressed)
        self._buffer: dict[str, list[np.ndarray]] = {}
        self._schema: dict[str, dict[str, Any]] = {}
        self._sample_count = 0
        self._shard_index = 0
        self._closed = False
        self._manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "in_progress",
            "sample_count": 0,
            "shard_count": 0,
            "shard_size": self.shard_size,
            "compressed": self.compressed,
            "schema": self._schema,
            "shards": [],
            "metadata": dict(metadata),
        }
        self._write_manifest()

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def shard_count(self) -> int:
        return self._shard_index

    def append(self, sample: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed dataset writer")
        arrays = {str(name): np.asarray(value) for name, value in sample.items()}
        if not arrays:
            raise ValueError("A sample cannot be empty")
        if not self._schema:
            for name, array in arrays.items():
                if array.dtype.kind == "O":
                    raise TypeError(f"Object arrays are not allowed: {name}")
                self._schema[name] = {
                    "dtype": str(array.dtype),
                    "sample_shape": list(array.shape),
                }
                self._buffer[name] = []
        elif set(arrays) != set(self._schema):
            missing = sorted(set(self._schema) - set(arrays))
            extra = sorted(set(arrays) - set(self._schema))
            raise ValueError(f"Sample schema mismatch; missing={missing}, extra={extra}")

        for name, array in arrays.items():
            expected = self._schema[name]
            if list(array.shape) != expected["sample_shape"] or str(array.dtype) != expected["dtype"]:
                raise ValueError(
                    f"Field {name!r} changed from {expected} to "
                    f"dtype={array.dtype}, shape={list(array.shape)}"
                )
            stored = array.copy() if array.ndim == 0 else np.ascontiguousarray(array)
            self._buffer[name].append(stored)
        self._sample_count += 1
        if len(next(iter(self._buffer.values()))) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer or not next(iter(self._buffer.values())):
            return
        arrays = {name: np.stack(values, axis=0) for name, values in self._buffer.items()}
        shard_name = f"shard_{self._shard_index:06d}.npz"
        shard_path = self.shards_dir / shard_name
        temp_path = self.shards_dir / f".{shard_name}.tmp"
        save_fn = np.savez_compressed if self.compressed else np.savez
        try:
            with temp_path.open("wb") as file_obj:
                save_fn(file_obj, **arrays)
            os.replace(temp_path, shard_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        shard_samples = int(next(iter(arrays.values())).shape[0])
        self._manifest["shards"].append(
            {
                "file": f"shards/{shard_name}",
                "sample_count": shard_samples,
            }
        )
        self._shard_index += 1
        for values in self._buffer.values():
            values.clear()
        self._manifest["sample_count"] = self._sample_count
        self._manifest["shard_count"] = self._shard_index
        self._write_manifest()

    def close(self, *, status: str = "complete") -> None:
        if self._closed:
            return
        self.flush()
        self._manifest["status"] = str(status)
        self._manifest["sample_count"] = self._sample_count
        self._manifest["shard_count"] = self._shard_index
        self._write_manifest()
        self._closed = True

    def _write_manifest(self) -> None:
        temp_path = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as file_obj:
                json.dump(self._manifest, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
                file_obj.write("\n")
            os.replace(temp_path, self.manifest_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def __enter__(self) -> "NpzShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(status="complete" if exc_type is None else "interrupted")


def merge_part_datasets(
    output_dir: str | Path,
    part_dirs: Sequence[str | Path],
    *,
    metadata: Mapping[str, Any],
) -> Path:
    """Create one manifest over independently collected presser datasets."""

    root = Path(output_dir).expanduser().resolve()
    if not part_dirs:
        raise ValueError("At least one part dataset is required")
    schema: dict[str, Any] | None = None
    shards: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    sample_count = 0
    compressed_values: set[bool] = set()
    for value in part_dirs:
        part = Path(value).expanduser().resolve()
        try:
            relative_part = part.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Part dataset must be inside {root}: {part}") from exc
        manifest_path = part / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Part dataset is not complete: {manifest_path}")
        part_schema = manifest.get("schema")
        if schema is None:
            schema = part_schema
        elif part_schema != schema:
            raise ValueError(f"Part dataset schema mismatch: {manifest_path}")
        compressed_values.add(bool(manifest.get("compressed", False)))
        part_samples = int(manifest.get("sample_count", 0))
        parts.append(
            {
                "directory": relative_part.as_posix(),
                "sample_count": part_samples,
                "shard_count": int(manifest.get("shard_count", 0)),
            }
        )
        sample_count += part_samples
        for shard in manifest.get("shards", []):
            shard_copy = dict(shard)
            shard_copy["file"] = (relative_part / str(shard["file"])).as_posix()
            shards.append(shard_copy)

    master = {
        "schema_version": 1,
        "status": "complete",
        "sample_count": sample_count,
        "shard_count": len(shards),
        "compressed": compressed_values.pop() if len(compressed_values) == 1 else "mixed",
        "schema": schema or {},
        "shards": shards,
        "parts": parts,
        "metadata": dict(metadata),
    }
    manifest_path = root / "manifest.json"
    temp_path = root / ".manifest.json.tmp"
    try:
        with temp_path.open("w", encoding="utf-8") as file_obj:
            json.dump(master, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return manifest_path
