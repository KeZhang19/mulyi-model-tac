from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
SCRIPT_ROOT = REPO_ROOT / "scripts" / "tactile_representation"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from BrainCo_DexHand.tactile_representation.training.contrastive import (  # noqa: E402
    build_in_batch_pair_masks,
    build_negative_pair_indices,
    build_positive_mask,
    symmetric_masked_infonce_loss,
    symmetric_queued_infonce_loss,
)
from BrainCo_DexHand.tactile_representation.models.latent_alignment import (  # noqa: E402
    TactileLatentAlignmentNetwork,
)
from train_latent_alignment import EpisodeHardNegativeBatchSampler  # noqa: E402


class _DummyEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def encode(self, *, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


def test_same_frame_is_positive_and_other_batch_rows_are_negative():
    frame_ids = torch.tensor([101, 102, 103])

    positive, negative = build_in_batch_pair_masks(frame_ids, frame_ids)

    assert torch.equal(positive, torch.eye(3, dtype=torch.bool))
    assert torch.equal(negative, ~torch.eye(3, dtype=torch.bool))
    sim_indices, real_indices = build_negative_pair_indices(positive)
    assert len(sim_indices) == len(real_indices) == 6
    assert all(int(i) != int(j) for i, j in zip(sim_indices, real_indices, strict=True))


def test_composite_episode_and_step_ids_match_all_components():
    sim_ids = torch.tensor([[1, 0], [1, 1], [2, 0]])
    real_ids = torch.tensor([[1, 1], [1, 0], [3, 0]])

    positive = build_positive_mask(sim_ids, real_ids)

    expected = torch.tensor(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    assert torch.equal(positive, expected)


def test_masked_infonce_supports_multiple_positive_views_and_gradients():
    sim = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    real = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    positive = torch.tensor([[True, True], [False, True]])

    loss = symmetric_masked_infonce_loss(sim, real, positive, temperature=0.2)

    assert torch.isfinite(loss)
    loss.backward()
    assert sim.grad is not None
    assert real.grad is not None


def test_queued_infonce_matches_masked_loss_without_queue():
    torch.manual_seed(11)
    sim = torch.randn(4, 6, requires_grad=True)
    real = torch.randn(4, 6, requires_grad=True)
    positive = torch.eye(4, dtype=torch.bool)
    queued = symmetric_queued_infonce_loss(
        sim,
        real,
        positive,
        real,
        sim,
        positive,
        temperature=0.2,
    )
    regular = symmetric_masked_infonce_loss(sim, real, positive, temperature=0.2)
    assert torch.allclose(queued, regular)


def test_queued_infonce_accepts_unmatched_memory_candidates():
    sim = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    real = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    memory = torch.tensor([[1.0, 1.0]])
    current_positive = torch.eye(2, dtype=torch.bool)
    positive_with_memory = torch.tensor([[True, False, False], [False, True, False]])
    loss = symmetric_queued_infonce_loss(
        sim,
        torch.cat((real, memory)),
        positive_with_memory,
        real,
        torch.cat((sim, memory)),
        positive_with_memory,
        temperature=0.2,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert sim.grad is not None
    assert real.grad is not None


def test_infonce_rejects_rows_without_a_positive_pair():
    latent = torch.eye(2)
    positive = torch.tensor([[True, False], [False, False]])

    with pytest.raises(ValueError, match="simulation row"):
        symmetric_masked_infonce_loss(latent, latent, positive)


def test_cttp_style_two_tower_alignment_keeps_h_and_z_separate():
    torch.manual_seed(2)
    aligner = TactileLatentAlignmentNetwork(
        _DummyEncoder(3, 4),
        _DummyEncoder(3, 4),
        latent_dim=4,
        projection_dim=5,
        temperature=0.2,
    )
    assert all(not parameter.requires_grad for parameter in aligner.sim_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in aligner.real_encoder.parameters())
    assert all(parameter.requires_grad for parameter in aligner.sim_projection.parameters())
    assert all(parameter.requires_grad for parameter in aligner.real_projection.parameters())
    inputs = {"features": torch.randn(3, 3)}
    outputs = aligner(inputs, inputs)
    loss = aligner.alignment_loss(outputs)

    assert set(outputs) == {"h_sim", "h_real", "z_sim", "z_real"}
    assert outputs["h_sim"].shape == (3, 4)
    assert outputs["z_real"].shape == (3, 5)
    assert torch.allclose(outputs["z_sim"].norm(dim=-1), torch.ones(3))
    assert torch.isfinite(loss)
    loss.backward()

    aligner.set_encoder_trainable(True)
    assert all(parameter.requires_grad for parameter in aligner.sim_encoder.parameters())
    assert all(parameter.requires_grad for parameter in aligner.real_encoder.parameters())


def test_partial_encoder_unfreezing_only_enables_requested_prefixes():
    aligner = TactileLatentAlignmentNetwork(
        _DummyEncoder(3, 4),
        _DummyEncoder(3, 4),
        latent_dim=4,
        projection_dim=5,
    )
    enabled = aligner.set_encoder_prefixes_trainable(("projection",))
    assert enabled == ("projection.weight", "projection.bias", "projection.weight", "projection.bias")
    assert all(parameter.requires_grad for parameter in aligner.sim_encoder.parameters())
    assert all(parameter.requires_grad for parameter in aligner.real_encoder.parameters())


def test_episode_hard_negative_sampler_keeps_unique_rows_and_same_episode_negatives():
    class _Dataset:
        pair_ids = ((1, 0), (1, 4), (1, 8), (2, 0), (2, 4), (2, 8), (3, 0), (3, 4))

        def __len__(self):
            return len(self.pair_ids)

    sampler = EpisodeHardNegativeBatchSampler(
        _Dataset(),
        batch_size=4,
        hard_negative_fraction=0.75,
        min_step_gap=2,
        seed=3,
    )
    batches = list(sampler)
    assert len(batches) == 2
    assert all(len(batch) == 4 for batch in batches)
    assert all(len(set(batch)) == len(batch) for batch in batches)
    same_episode_pairs = 0
    for batch in batches:
        for left_index in batch:
            for right_index in batch:
                if left_index >= right_index:
                    continue
                left = _Dataset.pair_ids[left_index]
                right = _Dataset.pair_ids[right_index]
                same_episode_pairs += int(
                    left[0] == right[0] and abs(left[1] - right[1]) >= 2
                )
    assert same_episode_pairs > 0
