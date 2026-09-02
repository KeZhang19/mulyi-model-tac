"""Pair-aware contrastive objectives for tactile sim--real alignment.

The dataset only needs to store aligned (positive) pairs.  During training,
the pair IDs in a batch define the positive mask; every cross-pair with a
different frame ID is an in-batch negative.  Keeping this as a mask avoids
materialising a second copy of the observations and also lets callers exclude
false negatives when multiple rows describe the same physical state.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _normalise_frame_ids(frame_ids: torch.Tensor, *, name: str) -> torch.Tensor:
    """Return frame IDs as ``[N]`` or composite IDs as ``[N, K]``."""

    if not torch.is_tensor(frame_ids):
        frame_ids = torch.as_tensor(frame_ids)
    if frame_ids.ndim == 0:
        frame_ids = frame_ids.reshape(1)
    if frame_ids.ndim not in (1, 2):
        raise ValueError(
            f"{name} must have shape [N] or [N,K], got {tuple(frame_ids.shape)}"
        )
    if frame_ids.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one frame ID")
    return frame_ids


def build_positive_mask(
    sim_frame_ids: torch.Tensor,
    real_frame_ids: torch.Tensor,
) -> torch.Tensor:
    """Match simulation and real observations that share a physical frame ID.

    A one-dimensional ID represents a scalar ``pair_id``.  A two-dimensional
    ID represents a composite key such as ``(episode_id, episode_step)``;
    all components must match.  The result is rectangular with shape
    ``[num_sim, num_real]`` and can contain more than one positive per row if
    the batch intentionally includes repeated views of a frame.
    """

    sim_ids = _normalise_frame_ids(sim_frame_ids, name="sim_frame_ids")
    real_ids = _normalise_frame_ids(real_frame_ids, name="real_frame_ids")
    if sim_ids.ndim != real_ids.ndim:
        raise ValueError(
            "sim_frame_ids and real_frame_ids must both be scalar IDs or both be composite IDs"
        )
    if sim_ids.ndim == 2 and sim_ids.shape[1] != real_ids.shape[1]:
        raise ValueError(
            "Composite frame IDs must have the same number of components: "
            f"sim={sim_ids.shape[1]}, real={real_ids.shape[1]}"
        )
    if real_ids.device != sim_ids.device:
        real_ids = real_ids.to(device=sim_ids.device)
    if sim_ids.ndim == 1:
        return sim_ids[:, None].eq(real_ids[None, :])
    return sim_ids[:, None, :].eq(real_ids[None, :, :]).all(dim=-1)


def build_in_batch_pair_masks(
    sim_frame_ids: torch.Tensor,
    real_frame_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive and negative pair masks for a batch.

    Negative pairs are all cross-domain pairs whose frame IDs do not match.
    Thus a batch of aligned positive pairs automatically supplies its
    negatives without a separate negative-observation dataset.
    """

    positive = build_positive_mask(sim_frame_ids, real_frame_ids)
    return positive, ~positive


def build_negative_pair_indices(
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sim_indices, real_indices)`` for every in-batch negative.

    This is useful for diagnostics or explicit negative mining.  The normal
    InfoNCE path should use the boolean mask directly instead of copying the
    observations indexed by this result.
    """

    if not torch.is_tensor(positive_mask) or positive_mask.ndim != 2:
        raise ValueError(
            "positive_mask must be a rank-2 torch.Tensor with shape [num_sim, num_real]"
        )
    return torch.where(~positive_mask.to(dtype=torch.bool))


def symmetric_masked_infonce_loss(
    sim_latent: torch.Tensor,
    real_latent: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute bidirectional InfoNCE with pair-ID-defined positives.

    ``sim_latent`` and ``real_latent`` are the two domain towers' outputs,
    before or after a projection head.  With the default diagonal mask, each
    row has one positive and all off-diagonal entries are negatives.  A custom
    mask supports repeated views or false-negative suppression.
    """

    if sim_latent.ndim != 2 or real_latent.ndim != 2:
        raise ValueError(
            "sim_latent and real_latent must have shape [batch, latent_dim]"
        )
    if sim_latent.shape[1] != real_latent.shape[1]:
        raise ValueError(
            "sim_latent and real_latent must have the same latent dimension: "
            f"sim={sim_latent.shape[1]}, real={real_latent.shape[1]}"
        )
    if sim_latent.shape[0] <= 0 or real_latent.shape[0] <= 0:
        raise ValueError("sim_latent and real_latent must contain at least one sample")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")

    logits = (
        F.normalize(sim_latent, dim=-1)
        @ F.normalize(real_latent, dim=-1).transpose(0, 1)
    ) / float(temperature)
    if positive_mask is None:
        if logits.shape[0] != logits.shape[1]:
            raise ValueError(
                "positive_mask is required when simulation and real batch sizes differ"
            )
        positive_mask = torch.eye(
            logits.shape[0], device=logits.device, dtype=torch.bool
        )
    else:
        if tuple(positive_mask.shape) != tuple(logits.shape):
            raise ValueError(
                "positive_mask shape must match the similarity matrix: "
                f"mask={tuple(positive_mask.shape)}, logits={tuple(logits.shape)}"
            )
        positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)

    if bool((~positive_mask.any(dim=1)).any()):
        raise ValueError("Every simulation row must have at least one positive real pair")
    if bool((~positive_mask.any(dim=0)).any()):
        raise ValueError("Every real row must have at least one positive simulation pair")

    def _directional_loss(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_denominator = torch.logsumexp(scores, dim=1)
        positive_scores = scores.masked_fill(~mask, float("-inf"))
        log_numerator = torch.logsumexp(positive_scores, dim=1)
        return -(log_numerator - log_denominator).mean()

    sim_to_real = _directional_loss(logits, positive_mask)
    real_to_sim = _directional_loss(logits.transpose(0, 1), positive_mask.transpose(0, 1))
    return 0.5 * (sim_to_real + real_to_sim)


def symmetric_queued_infonce_loss(
    sim_queries: torch.Tensor,
    real_candidates: torch.Tensor,
    sim_positive_mask: torch.Tensor,
    real_queries: torch.Tensor,
    sim_candidates: torch.Tensor,
    real_positive_mask: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Bidirectional InfoNCE with detached cross-batch candidate queues.

    ``sim_queries`` and ``real_queries`` are the current minibatch.  The
    candidate tensors may additionally contain embeddings from previous
    minibatches, which act as a memory queue.  Queue candidates do not need a
    positive in the current minibatch; only every query row must have at least
    one positive candidate.  Embeddings in the queue should be detached by the
    caller so gradients only update the current minibatch.
    """

    def _validate_pair(
        queries: torch.Tensor,
        candidates: torch.Tensor,
        positive_mask: torch.Tensor,
        query_name: str,
        candidate_name: str,
    ) -> None:
        if queries.ndim != 2 or candidates.ndim != 2:
            raise ValueError(
                f"{query_name} and {candidate_name} must have shape [N, D]"
            )
        if queries.shape[1] != candidates.shape[1]:
            raise ValueError(
                f"{query_name} and {candidate_name} must have the same latent dimension"
            )
        if queries.shape[0] <= 0 or candidates.shape[0] <= 0:
            raise ValueError("query and candidate tensors must be non-empty")
        if tuple(positive_mask.shape) != (queries.shape[0], candidates.shape[0]):
            raise ValueError(
                f"positive mask for {query_name} must have shape "
                f"{(queries.shape[0], candidates.shape[0])}, got {tuple(positive_mask.shape)}"
            )
        if not bool(positive_mask.any(dim=1).all()):
            raise ValueError(f"Every {query_name} row must have a positive candidate")

    _validate_pair(
        sim_queries,
        real_candidates,
        sim_positive_mask,
        "simulation queries",
        "real candidates",
    )
    _validate_pair(
        real_queries,
        sim_candidates,
        real_positive_mask,
        "real queries",
        "simulation candidates",
    )
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")

    def _directional_loss(
        queries: torch.Tensor,
        candidates: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = (
            F.normalize(queries, dim=-1)
            @ F.normalize(candidates, dim=-1).transpose(0, 1)
        ) / float(temperature)
        positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
        positive_scores = logits.masked_fill(~positive_mask, float("-inf"))
        return -(
            torch.logsumexp(positive_scores, dim=1)
            - torch.logsumexp(logits, dim=1)
        ).mean()

    return 0.5 * (
        _directional_loss(sim_queries, real_candidates, sim_positive_mask)
        + _directional_loss(real_queries, sim_candidates, real_positive_mask)
    )
