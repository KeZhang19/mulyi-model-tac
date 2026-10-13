"""Paired NPZ inputs for alignment without fabricated collection labels."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import TactileNormalization
from .alignment_domain import AlignmentDomain, FIELDS

SCHEMA = "tactile_alignment_pairs_v1"


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping alignment data path: {name}")
    return path


class AlignmentData:
    """Validate both domains and their shared split/pair contract once."""

    def __init__(self, root: Path, sim_normalization: TactileNormalization,
                 real_normalization: TactileNormalization, *, workflow_test: bool = False):
        self.root = Path(root).expanduser().resolve()
        path = contained_path(self.root, "pairs.json")
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
            raise ValueError("Expected a completed tactile_alignment_pairs_v1 dataset")
        quality = manifest.get("quality", {})
        quality_names = ("pair_valid", "training_ready", "geometry_validated",
                         "reference_unloaded_confirmed", "independent_validation")
        if any(type(quality.get(name)) is not bool for name in quality_names):
            raise ValueError("Alignment quality flags must be explicit booleans")
        if workflow_test:
            if manifest.get("association_kind") != "ordinal_workflow_test":
                raise ValueError("--workflow-test requires an ordinal_workflow_test dataset")
        elif (manifest.get("association_kind") != "verified_physical_pairs"
              or not all(quality[name] for name in quality_names)):
            raise ValueError("Unverified pairs require explicit --workflow-test; formal alignment is refused")
        self.records = manifest.get("pairs", [])
        count = int(manifest.get("sample_count", 0))
        if count < 2 or len(self.records) != count:
            raise ValueError("Alignment pair count is inconsistent or insufficient")
        ids = [tuple(record["pair_id"]) for record in self.records]
        if any(len(key) != 2 or any(type(v) is not int or v < 0 for v in key) for key in ids):
            raise ValueError("Pair IDs must be nonnegative integer [episode, step] keys")
        if len(set(ids)) != count:
            raise ValueError("Duplicate alignment pair ID")
        for record in self.records:
            if any(type(record.get(name)) is not bool for name in ("pair_valid", "training_ready")):
                raise ValueError("Per-pair validity must be explicit")
            if not workflow_test and not (record["pair_valid"] and record["training_ready"]):
                raise ValueError("Formal alignment contains an invalid pair")
        splits = manifest.get("splits", {})
        if set(splits) != {"train", "validation", "test"}:
            raise ValueError("Alignment splits must contain train/validation/test")
        self.indices = {}
        lookup = {key: index for index, key in enumerate(ids)}
        assigned = []
        for name, keys in splits.items():
            keys = [tuple(key) for key in keys]
            if (name != "test" or not workflow_test) and not keys:
                raise ValueError(f"Empty {name} alignment split")
            if any(key not in lookup for key in keys):
                raise ValueError("Split contains an unknown pair ID")
            self.indices[name] = [lookup[key] for key in keys]
            assigned.extend(keys)
        if len(assigned) != count or set(assigned) != set(ids):
            raise ValueError("Alignment splits overlap or fail to cover all pairs")
        # A physical episode cannot contribute different steps to two splits.
        episodes = {}
        for name, indices in self.indices.items():
            for index in indices:
                episode = ids[index][0]
                if episodes.setdefault(episode, name) != name:
                    raise ValueError("Alignment splits leak an episode")
        if not workflow_test:
            for group_name in ("source_session_id", "reference_id", "condition_id"):
                groups = {}
                for name, indices in self.indices.items():
                    for index in indices:
                        group = self.records[index].get(group_name)
                        if not group or groups.setdefault(group, name) != name:
                            raise ValueError(f"Formal alignment splits leak or lack {group_name}")
        fingerprint = hashlib.sha256(path.read_bytes())
        self.arrays = {}
        self.domains = {}
        self.normalizations = {"sim": sim_normalization, "real": real_normalization}
        for domain, normalization in self.normalizations.items():
            domain_path = contained_path(self.root, manifest["domains"][domain])
            reader = AlignmentDomain(self.root, domain_path, normalization, domain=domain)
            if reader.count != count:
                raise ValueError(f"Invalid {domain} alignment manifest")
            for part in reader.fingerprint_parts:
                fingerprint.update(part)
            arrays = reader.arrays
            if not np.array_equal(arrays["pair_id"], np.asarray(ids, dtype=np.int64)):
                raise ValueError(f"{domain} pair keys differ from the shared manifest")
            for name in ("pair_valid", "training_ready"):
                if not np.array_equal(arrays[name], [record[name] for record in self.records]):
                    raise ValueError(f"{domain} {name} differs from the shared manifest")
            self.domains[domain] = reader
            self.arrays[domain] = arrays
        self.contract = {"schema": SCHEMA, "fingerprint": fingerprint.hexdigest(),
                         "workflow_test": workflow_test, "splits": splits,
                         "quality": quality,
                         "normalization": {name: asdict(value) for name, value in self.normalizations.items()}}

    def dataset(self, split: str) -> "AlignmentPairDataset":
        return AlignmentPairDataset(self, self.indices[split])


class AlignmentPairDataset(Dataset):
    """Return the same sim/real/pair_id batch interface as the legacy join."""

    def __init__(self, data: AlignmentData, indices: list[int]):
        self.data, self.indices = data, tuple(indices)

    def __len__(self):
        return len(self.indices)

    @property
    def pair_ids(self):
        return tuple(tuple(self.data.records[index]["pair_id"]) for index in self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        result = {}
        for domain, norm in self.data.normalizations.items():
            arrays = self.data.arrays[domain]
            marker = torch.tensor(arrays["marker_2d"][source_index], dtype=torch.float32)
            valid = torch.tensor(arrays["marker_valid"][source_index], dtype=torch.bool)
            marker[:, 0] /= norm.image_width - 1
            marker[:, 1] /= norm.image_height - 1
            marker[:, 2:4] /= norm.marker_motion_scale_px
            marker[:, 4] = valid.float()
            marker[~valid, :4] = 0
            result[domain] = {
                "rgb": torch.tensor(arrays["rgb"][source_index], dtype=torch.float32).div_(255),
                "depth": torch.tensor(arrays["depth_m"][source_index], dtype=torch.float32).div_(norm.depth_scale_m).clamp_(0, 1),
                "marker": marker, "marker_valid": valid}
        result["pair_id"] = torch.tensor(self.data.records[source_index]["pair_id"], dtype=torch.int64)
        return result
