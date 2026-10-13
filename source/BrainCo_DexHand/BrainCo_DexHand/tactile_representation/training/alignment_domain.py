"""Validated, bounded-memory access to raw-unit CLIP domain shards."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = ("rgb", "depth_m", "marker_2d", "marker_valid", "pair_id",
          "pair_valid", "training_ready")


def contained_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping alignment data path: {name}")
    return path


class ShardField:
    """Array-like sample access; full array materialization remains explicit."""

    def __init__(self, domain, name, shape, dtype):
        self.domain, self.name = domain, name
        self.shape, self.dtype = shape, np.dtype(dtype)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            index = int(index)
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(index)
            shard = bisect_right(self.domain.stops, index)
            return self.domain.read_shard(shard)[self.name][index - self.domain.starts[shard]]
        indices = np.arange(len(self))[index]
        if np.asarray(indices).size == 0:
            return np.empty((*np.asarray(indices).shape, *self.shape[1:]), dtype=self.dtype)
        return np.stack([self[int(i)] for i in np.asarray(indices).reshape(-1)]).reshape(
            (*np.asarray(indices).shape, *self.shape[1:]))

    def __array__(self, dtype=None, copy=None):
        value = np.concatenate([self.domain.read_shard(i)[self.name]
                                for i in range(len(self.domain.paths))], axis=0)
        return value.astype(dtype, copy=False) if dtype is not None else value


class AlignmentDomain:
    """Read old single-NPZ or new sharded domain manifests without retaining all frames."""

    def __init__(self, root: Path, manifest_path: Path, normalization, *, domain: str):
        self.root = Path(root).resolve()
        self.path = contained_file(self.root, str(Path(manifest_path).relative_to(self.root)))
        self.info = json.loads(self.path.read_text())
        info = self.info
        self.count = int(info.get("sample_count", 0))
        if (info.get("schema") != "tactile_alignment_domain_npz_v1"
                or info.get("domain") != domain or info.get("status") != "complete" or self.count < 1):
            raise ValueError(f"Invalid {domain} alignment manifest")
        if ("shard" in info) == ("shards" in info):
            raise ValueError("Alignment domain must specify exactly one of shard/shards")
        shards = [info["shard"]] if "shard" in info else info["shards"]
        if not isinstance(shards, list) or not shards:
            raise ValueError("Alignment domain contains no shards")
        h, w, k = normalization.image_height, normalization.image_width, normalization.marker_count
        shapes = {"rgb": ((3, h, w), "uint8"), "depth_m": ((1, h, w), "float32"),
                  "marker_2d": ((k, 5), "float32"), "marker_valid": ((k,), "bool"),
                  "pair_id": ((2,), "int64"), "pair_valid": ((), "bool"), "training_ready": ((), "bool")}
        self.paths, self.starts, self.stops, self.fingerprint_parts = [], [], [], [self.path.read_bytes()]
        self.digests = []
        self._cache = OrderedDict()
        small = {name: [] for name in ("pair_id", "pair_valid", "training_ready")}
        cursor = 0
        for shard in shards:
            name = str(self.path.parent.relative_to(self.root) / shard["file"])
            path = contained_file(self.root, name)
            if path in self.paths:
                raise ValueError("Duplicate alignment shard path")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != shard["sha256"]:
                raise ValueError(f"{domain} alignment shard SHA256 mismatch")
            with np.load(path, allow_pickle=False) as source:
                if set(source.files) != set(FIELDS):
                    raise ValueError(f"Unexpected {domain} alignment fields")
                n = len(source["pair_id"])
                if n < 1 or int(shard.get("sample_count", n)) != n:
                    raise ValueError("Alignment shard sample count is inconsistent")
                values = {name: source[name] for name in FIELDS}
                for name, (shape, dtype) in shapes.items():
                    a = values[name]
                    if a.shape != (n, *shape) or str(a.dtype) != dtype:
                        raise ValueError(f"{domain}.{name} has incompatible shape/dtype")
                    if info.get("fields", {}).get(name) != {"dtype": dtype, "sample_shape": list(shape)}:
                        raise ValueError(f"{domain}.{name} schema disagrees with arrays")
                    if not np.isfinite(a).all():
                        raise ValueError(f"{domain}.{name} contains nonfinite values")
                if (values["depth_m"] < 0).any():
                    raise ValueError("Negative depth is outside the encoder input contract")
                if not np.array_equal(values["marker_2d"][..., 4], values["marker_valid"]):
                    raise ValueError("Marker validity fields disagree")
                for name in small:
                    small[name].append(values[name].copy())
            self.paths.append(path)
            self.digests.append(digest)
            self.starts.append(cursor)
            cursor += n
            self.stops.append(cursor)
            self.fingerprint_parts.append(digest.encode())
        if cursor != self.count:
            raise ValueError("Alignment domain sample count differs from shards")
        self.arrays = {name: np.concatenate(parts) for name, parts in small.items()}
        self.arrays.update({name: ShardField(self, name, (cursor, *shape), dtype)
                            for name, (shape, dtype) in shapes.items() if name not in small})

    def read_shard(self, index):
        values = self._cache.pop(index, None)
        if values is None:
            with self.paths[index].open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != self.digests[index]:
                    raise ValueError("Alignment shard changed after validation: SHA256 mismatch")
            with np.load(self.paths[index], allow_pickle=False) as source:
                values = {name: source[name] for name in FIELDS}
        self._cache[index] = values
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return values
