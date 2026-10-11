"""TactSpace RGB/Depth/Marker adaptation, not an official reproduction.

Reuse the local image/point token encoders. A shared latent MLP feeds a
four-channel FiLM image head and a fixed-query marker motion head. No decoder
receives observations, encoder tokens, reference images, or dynamic skips.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..tri_modal_config import TriModalCrossAutoencoderCfg
from .encoders import ImageTokenEncoder, MarkerTokenEncoder
from .decoders import ConditionedImageDecoder, MarkerDecoder
from ..training.contrastive import build_positive_mask, symmetric_masked_infonce_loss

MODALITIES = ("rgb", "depth", "marker")
NETWORK_TYPE = "tactspace"
OBJECTIVE_NAME = "tactspace_rgb_depth_marker_joint_v1"


@dataclass(frozen=True)
class TactSpaceCfg(TriModalCrossAutoencoderCfg):
    # Inherited cross_layers/cross_summary_tokens are unused; no fusion path.
    domains: tuple[str, ...] = ("sim",)
    input_modalities: tuple[str, ...] = MODALITIES

    def __post_init__(self):
        super().__post_init__()
        for name, allowed in (("domains", ("sim", "real")), ("input_modalities", MODALITIES)):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
                raise ValueError(f"Invalid {name}: {values}")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class TactSpaceLossCfg:
    reconstruction_weight: float = 1.0
    alignment_weight: float = 1.0
    temperature: float = 0.1
    rgb_weight: float = 1.0
    depth_weight: float = 1.0
    marker_weight: float = 1.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0 or (name == "temperature" and value == 0):
                raise ValueError(f"Invalid {name}: {value}")


def representation_contract(cfg: TactSpaceCfg) -> dict:
    return {
        "model_type": NETWORK_TYPE, "adaptation": "TactSpace RGB/Depth/Marker v1",
        "latent_dim": cfg.d_model, "input_domains": list(cfg.domains),
        "input_modalities": list(cfg.input_modalities),
        "aggregation": "valid_modality_mean",
        "normalization": "l2_each_latent_then_mean_then_l2",
        "finger_order": ["little", "ring", "middle", "index", "thumb"],
        "history": "encode_each_frame_independently; external_oldest_to_newest",
        "encoder_history_length": 1,
    }


class TactSpaceRGBDepthMarker(nn.Module):
    def __init__(self, cfg: TactSpaceCfg | None = None):
        super().__init__()
        self.cfg = cfg or TactSpaceCfg()
        c = self.cfg
        self.encoders = nn.ModuleDict()
        self.latent_heads = nn.ModuleDict()
        for domain in c.domains:
            branches = nn.ModuleDict()
            for name in c.input_modalities:
                if name == "marker":
                    branches[name] = MarkerTokenEncoder(
                        marker_count=c.marker_count, input_dim=c.marker_input_dim,
                        d_model=c.d_model, num_heads=c.num_heads,
                        num_layers=c.marker_transformer_layers, ffn_ratio=c.ffn_ratio,
                        summary_tokens=c.marker_summary_tokens, dropout=c.dropout,
                    )
                else:
                    branches[name] = ImageTokenEncoder(
                        in_channels=c.rgb_channels if name == "rgb" else c.depth_channels,
                        image_height=c.image_height, image_width=c.image_width,
                        base_channels=c.image_base_channels, d_model=c.d_model, dropout=c.dropout,
                        empty_input_value=-1.0 if name == "depth" else None,
                    )
                self.latent_heads[f"{domain}_{name}"] = nn.Sequential(
                    nn.Linear(c.d_model, c.d_model), nn.GELU(), nn.LayerNorm(c.d_model)
                )
            self.encoders[domain] = branches
        self.decoder_trunk = nn.Sequential(nn.Linear(c.d_model, c.d_model), nn.GELU(), nn.LayerNorm(c.d_model))
        self.image_head = ConditionedImageDecoder(
            latent_dim=c.d_model, output_channels=c.rgb_channels + c.depth_channels,
            image_height=c.image_height, image_width=c.image_width,
            base_channels=c.decoder_base_channels,
        )
        self.marker_head = MarkerDecoder(
            marker_count=c.marker_count, output_dim=c.marker_output_dim, d_model=c.d_model,
            num_heads=c.num_heads, num_layers=c.marker_transformer_layers,
            ffn_ratio=c.ffn_ratio, dropout=c.dropout,
            static_context=False, image_spatial_context=False,
        )
        # Counts are updated only after successful optimizer steps, never validation.
        self.register_buffer("trained_source_steps", torch.zeros(len(c.domains), 3, dtype=torch.long))

    def encode_modalities(self, *, rgb=None, depth=None, marker=None,
                          marker_valid_mask=None, modality_valid=None, domain="sim"):
        if domain not in self.cfg.domains:
            raise ValueError(f"Domain {domain!r} has no encoder")
        supplied = {k: v for k, v in dict(rgb=rgb, depth=depth, marker=marker).items()
                    if v is not None and k in self.cfg.input_modalities}
        if not supplied:
            raise ValueError("No enabled input modality supplied")
        first = next(iter(supplied.values()))
        b = first.shape[0]
        if modality_valid is None:
            modality_valid = torch.ones(b, 3, dtype=torch.bool, device=first.device)
        if modality_valid.shape != (b, 3):
            raise ValueError("modality_valid must have shape [B,3] in rgb/depth/marker order")
        latents, validities = {}, {}
        for name, value in supplied.items():
            if value.shape[0] != b or value.device != first.device:
                raise ValueError("Input modalities must have the same batch and device")
            valid = modality_valid[:, MODALITIES.index(name)].to(device=value.device, dtype=torch.bool)
            if name == "marker":
                if value.shape[1:] != (self.cfg.marker_count, 5):
                    raise ValueError("Marker must be [B,K,5] point features")
                points = value[..., 4] > 0.5 if marker_valid_mask is None else marker_valid_mask.bool()
                if points.shape != value.shape[:2]:
                    raise ValueError("marker_valid_mask must be [B,K]")
                points = points & torch.isfinite(value).all(-1) & valid[:, None]
                valid = valid & points.any(-1)
                clean = value.masked_fill(~points[..., None], 0.0)
                tokens, mask = self.encoders[domain][name](clean, points)
                pooled = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            else:
                valid = valid & torch.isfinite(value).flatten(1).all(1)
                clean = value.masked_fill(~valid[:, None, None, None], 0.0)
                pooled = self.encoders[domain][name](clean).mean(1)
            z = F.normalize(self.latent_heads[f"{domain}_{name}"](pooled), dim=-1)
            latents[name] = z.masked_fill(~valid[:, None], 0.0)
            validities[name] = valid
        return latents, validities

    @staticmethod
    def aggregate(latents, validities):
        count = torch.stack(list(validities.values())).sum(0)
        if bool((count == 0).any()):
            raise ValueError("No valid tactile input for one or more samples")
        mean = torch.stack(list(latents.values())).sum(0) / count[:, None]
        if bool((mean.norm(dim=-1) < 1e-8).any()):
            raise ValueError("Valid modality latents cancelled to a zero vector")
        return F.normalize(mean, dim=-1)

    def encode(self, **inputs):
        return self.aggregate(*self.encode_modalities(**inputs))

    def decode(self, latent):
        shared = self.decoder_trunk(latent)
        image = self.image_head(shared)
        return {"rgb_recon": image[:, :self.cfg.rgb_channels],
                "depth_recon": image[:, self.cfg.rgb_channels:],
                "marker_recon": self.marker_head(shared)}

    def forward(self, **inputs):
        latents, valid = self.encode_modalities(**inputs)
        latent = self.aggregate(latents, valid)
        return {"latent": latent, "modality_latents": latents, "modality_valid": valid,
                "source_reconstructions": {name: self.decode(z) for name, z in latents.items()},
                **self.decode(latent)}

    @torch.no_grad()
    def record_trained_sources(self, metrics, domain="sim"):
        for index, name in enumerate(MODALITIES):
            if float(metrics.get(f"count/trained_{name}", 0)) > 0:
                self.trained_source_steps[self.cfg.domains.index(domain), index] += 1


def batch_inputs(batch: Mapping, *, modalities=MODALITIES):
    supplied = {name: batch[name] for name in modalities if name in batch}
    if not supplied:
        raise ValueError("Batch has no input modalities")
    first = next(iter(supplied.values()))
    valid = batch.get("modality_valid", torch.ones(first.shape[0], 3, dtype=torch.bool, device=first.device)).bool().clone()
    for i, name in enumerate(MODALITIES):
        if f"{name}_valid" in batch and name != "marker":
            valid[:, i] &= batch[f"{name}_valid"].bool()
    if "training_ready" in batch:
        valid &= batch["training_ready"].bool()[:, None]
    return {**supplied, "marker_valid_mask": batch.get("marker_valid"), "modality_valid": valid}


def compute_tactspace_objective(model, batch, *, cfg: TactSpaceLossCfg, domain="sim", **_):
    """Joint same-frame loss; cross-domain joining is deliberately not inferred.

    The existing simulation collector stores synchronous modalities in one row.
    Composite frame IDs preserve repeated samples as multiple positives.
    Optional pair_valid gates alignment, training_ready gates all objectives.
    """
    latents, source_valid = model.encode_modalities(**batch_inputs(batch), domain=domain)
    first = next(iter(latents.values()))
    zero = first.sum() * 0.0
    losses, self_losses, cross_losses, align_losses = [], [], [], []
    paired_count = skipped_count = rec_pairs = 0
    trained = {name: 0 for name in MODALITIES}
    # Targets may include modalities not chosen as encoder inputs.
    targets = batch_inputs(batch)
    target_valid = targets["modality_valid"]
    for source, latent in latents.items():
        predictions = model.decode(latent)
        for ti, target in enumerate(MODALITIES):
            weight = getattr(cfg, f"{target}_weight")
            if target not in batch or weight == 0:
                continue
            truth = batch[target]
            rows = source_valid[source] & target_valid[:, ti]
            if target == "marker":
                points = batch.get("marker_valid", truth[..., 4] > 0.5).bool()
                mask = points & torch.isfinite(truth).all(-1) & rows[:, None]
                truth = truth[..., 2:4]
                mask = mask[..., None].expand_as(truth)
            else:
                rows &= torch.isfinite(truth).flatten(1).all(1)
                mask = rows.reshape(-1, *([1] * (truth.ndim - 1))).expand_as(truth)
            count = int(mask.sum())
            if count == 0:
                continue
            # Index BEFORE subtraction: invalid points/rows can contain NaNs.
            error = (predictions[f"{target}_recon"][mask].float() - truth[mask].float()).square().mean()
            losses.append((weight, error))
            (self_losses if source == target else cross_losses).append(error)
            rec_pairs += 1
            if cfg.reconstruction_weight > 0:
                trained[source] += int(mask.reshape(mask.shape[0], -1).any(1).sum())
    pair_gate = batch.get("pair_valid", torch.ones(first.shape[0], device=first.device, dtype=torch.bool)).bool()
    ids = batch.get("pair_id")
    if ids is None and "episode_id" in batch and "episode_step" in batch:
        ids = torch.stack((batch["episode_id"], batch["episode_step"]), dim=-1)
    for left, right in combinations(latents, 2):
        rows = source_valid[left] & source_valid[right] & pair_gate
        if ids is None or int(rows.sum()) < 2:
            skipped_count += 1
            continue
        selected_ids = ids[rows]
        positives = build_positive_mask(selected_ids, selected_ids)
        if bool(positives.all()):  # No negative physical stimulus, no alignment update.
            skipped_count += 1
            continue
        align_losses.append(symmetric_masked_infonce_loss(
            latents[left][rows].float(), latents[right][rows].float(), positives, temperature=cfg.temperature))
        paired_count += int(rows.sum())
        if cfg.alignment_weight > 0:
            trained[left] += int(rows.sum())
            trained[right] += int(rows.sum())
    rec = sum(w * loss for w, loss in losses) / sum(w for w, _ in losses) if losses else zero
    mean = lambda values: torch.stack(values).mean() if values else zero
    align = mean(align_losses)
    total = (cfg.reconstruction_weight * rec if cfg.reconstruction_weight else zero) + (cfg.alignment_weight * align if cfg.alignment_weight else zero)
    metrics = {"loss/total": total.detach(), "loss/reconstruction": rec.detach(),
               "loss/self_reconstruction": mean(self_losses).detach(),
               "loss/cross_reconstruction": mean(cross_losses).detach(), "loss/alignment": align.detach(),
               "count/valid_alignment_pairs": first.new_tensor(paired_count),
               "count/skipped_alignment_pairs": first.new_tensor(skipped_count),
               "count/skipped_alignment_batches": first.new_tensor(not align_losses),
               "count/reconstruction_pairs": first.new_tensor(rec_pairs),
               "count/active_sources": first.new_tensor(sum(v > 0 for v in trained.values()))}
    metrics.update({f"count/trained_{name}": first.new_tensor(count) for name, count in trained.items()})
    return total, metrics


def validate_sim_training_manifest(manifest):
    """Fail closed when the legacy loader cannot preserve a validity contract."""
    metadata = manifest.get("metadata", {})
    if manifest.get("training_ready") is False or metadata.get("training_ready") is False:
        raise ValueError("Dataset training_ready is false")
    domain = metadata.get("domain", manifest.get("domain"))
    # This is the local synchronous Isaac collector's existing manifest signature.
    if domain is None and metadata.get("dataset") == "revo3_index_cross_modal" and str(metadata.get("task", "")).startswith("BrainCo-"):
        domain = "sim"
    if domain != "sim":
        raise ValueError("TactSpace CLI currently supports synchronous simulation data only; real pairing is unverified")
    unsupported = set(manifest.get("schema", {})) & {
        "pair_valid", "training_ready", "modality_valid", "rgb_valid", "depth_valid", "pair_id"}
    if unsupported:
        raise ValueError(f"Legacy dataset loader does not preserve {sorted(unsupported)}; refusing to discard validity/pairing fields")
