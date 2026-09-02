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

from BrainCo_DexHand.tactile_representation.training.contrastive import (  # noqa: E402
    build_in_batch_pair_masks,
    build_negative_pair_indices,
    build_positive_mask,
    symmetric_masked_infonce_loss,
)
from BrainCo_DexHand.tactile_representation.models.latent_alignment import (  # noqa: E402
    TactileLatentAlignmentNetwork,
)


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
