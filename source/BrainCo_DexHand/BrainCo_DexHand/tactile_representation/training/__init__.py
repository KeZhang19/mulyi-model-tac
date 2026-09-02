"""Offline training utilities for robust cross-modal tactile representations."""

from .data import (
    EpisodeSplits,
    NpzTactileDataset,
    TactileNormalization,
    build_mmap_cache,
    load_complete_manifest,
    manifest_sha256,
    split_episode_ids,
)
from .objectives import (
    CrossModalLossCfg,
    MODALITY_NAMES,
    compute_cross_modal_objective,
    degrade_tactile_inputs,
    restoration_loss,
    sample_degraded_modality_indices,
)
from .contrastive import (
    build_in_batch_pair_masks,
    build_negative_pair_indices,
    build_positive_mask,
    symmetric_masked_infonce_loss,
)

__all__ = [
    "CrossModalLossCfg",
    "EpisodeSplits",
    "MODALITY_NAMES",
    "NpzTactileDataset",
    "TactileNormalization",
    "build_mmap_cache",
    "compute_cross_modal_objective",
    "degrade_tactile_inputs",
    "load_complete_manifest",
    "manifest_sha256",
    "restoration_loss",
    "sample_degraded_modality_indices",
    "split_episode_ids",
    "build_in_batch_pair_masks",
    "build_negative_pair_indices",
    "build_positive_mask",
    "symmetric_masked_infonce_loss",
]
