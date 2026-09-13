"""Portable, frozen tactile towers and the policy observation contract.

Inputs use sensor units: RGB in [0,255], depth in metres, marker coordinates
and flow in pixels. This module has no simulator dependency.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PosixPath, WindowsPath
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .config import CrossModalTactileNetworkCfg
from .tri_modal_config import TriModalCrossAutoencoderCfg
from .models.network import RobustCrossModalTactileNetwork
from .models.tri_modal_cross_autoencoder import TriModalCrossAutoencoder
from .models.latent_alignment import ProjectionHead
from .training.data import TactileNormalization


FINGER_ORDER = ("little", "ring", "middle", "index", "thumb")
BUNDLE_FORMAT = "brainco_tactile_policy_encoder"
INPUT_SCHEMA = "rgb255_depth_m_marker_xy0_dxy_px_valid_v1"


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_tactile_checkpoint(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tactile checkpoint does not exist: {path}")
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Tactile checkpoint is a Git LFS pointer; fetch the weights first: {path}")
    # Original training checkpoints contain argparse Path values. Allow only
    # these extra types rather than disabling weights-only loading.
    with torch.serialization.safe_globals([PosixPath, WindowsPath]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a checkpoint dictionary: {path}")
    return payload


def build_tactile_encoder(network_type: str, model_cfg: Mapping) -> nn.Module:
    if network_type == "robust":
        return RobustCrossModalTactileNetwork(CrossModalTactileNetworkCfg(**dict(model_cfg)))
    if network_type == "tri_modal":
        return TriModalCrossAutoencoder(TriModalCrossAutoencoderCfg(**dict(model_cfg)))
    raise ValueError(f"Unsupported tactile network type: {network_type!r}")


def export_policy_encoder_bundle(
    output: str | Path, *, alignment: Mapping, source: Mapping,
    domain: str, network_type: str, alignment_id: str,
) -> None:
    """Export the FINAL aligned tower, including its projection and metadata."""
    if domain not in ("sim", "real"):
        raise ValueError("domain must be sim or real")
    args = alignment.get("args", {})
    configured_type = args.get(f"{domain}_model_type", network_type)
    if configured_type != network_type:
        raise ValueError(f"Alignment {domain} tower is {configured_type}, requested {network_type}")
    model_cfg = dict(source["model_cfg"])
    normalization = asdict(TactileNormalization(**dict(source["normalization"])))
    projection_cfg = {
        "input_dim": int(model_cfg["d_model"]),
        "projection_dim": int(args.get("projection_dim", 64)),
        "hidden_dim": args.get("projection_hidden_dim"),
        "use_layernorm": bool(args.get("projection_layernorm", False)),
    }
    state = alignment["model_state"]
    def tower_state(prefix: str) -> dict:
        return {k.removeprefix(prefix): v.detach().cpu() for k, v in state.items() if k.startswith(prefix)}
    bundle = {
        "format": BUNDLE_FORMAT, "schema_version": 1, "input_schema": INPUT_SCHEMA,
        "domain": domain, "network_type": network_type, "alignment_id": str(alignment_id),
        "finger_order": list(FINGER_ORDER), "model_cfg": model_cfg,
        "normalization": normalization, "projection_cfg": projection_cfg,
        "encoder_state": tower_state(f"{domain}_encoder."),
        "projection_state": tower_state(f"{domain}_projection."),
    }
    # Reject incomplete or mismatched exports before creating a file.
    FrozenTactilePolicyEncoder(bundle)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)


def export_unaligned_sim_policy_encoder_bundle(
    output: str | Path, *, source: Mapping, source_checkpoint_sha256: str,
) -> None:
    """Export pretrained simulation features without a learned alignment head."""
    bundle = make_unaligned_sim_policy_bundle(source, source_checkpoint_sha256=source_checkpoint_sha256)
    FrozenTactilePolicyEncoder(bundle, domain="sim")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)


def make_unaligned_sim_policy_bundle(source: Mapping, *, source_checkpoint_sha256: str) -> dict:
    """Use the same pretrained feature definition for direct loading and export."""
    latent_dim = int(source["model_cfg"]["d_model"])
    return {
        "format": BUNDLE_FORMAT, "schema_version": 1, "input_schema": INPUT_SCHEMA,
        "domain": "sim", "network_type": "robust", "alignment_id": None,
        "feature_mode": "unaligned_sim", "source_checkpoint_sha256": source_checkpoint_sha256,
        "finger_order": list(FINGER_ORDER), "model_cfg": dict(source["model_cfg"]),
        "normalization": asdict(TactileNormalization(**dict(source["normalization"]))),
        "projection_cfg": {"input_dim": latent_dim, "projection_dim": latent_dim},
        "encoder_state": {k: v.detach().cpu() for k, v in source["model_state"].items()},
        "projection_state": {},
    }


def load_sim_policy_encoder(checkpoint: str | Path, *, chunk_size: int = 32) -> FrozenTactilePolicyEncoder:
    """Load a pretrained robust checkpoint or an explicitly exported sim policy bundle.

    Raw checkpoints use normalized h without a projection. Missing, LFS-only,
    real-domain or incompatible weights fail instead of falling back to ResNet.
    """
    payload = load_tactile_checkpoint(checkpoint)
    digest = file_sha256(checkpoint)
    if payload.get("format") == BUNDLE_FORMAT:
        bundle = payload
    else:
        bundle = make_unaligned_sim_policy_bundle(payload, source_checkpoint_sha256=digest)
    encoder = FrozenTactilePolicyEncoder(bundle, domain="sim", chunk_size=chunk_size)
    encoder.bundle_sha256 = digest
    return encoder


def marker_flow_to_features_tensor(flow: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched equivalent of collection.marker_flow_to_features; no CPU copies."""
    if flow.shape[-3] != 2 or flow.shape[-1] != 2 or valid.shape != flow.shape[:-3] + flow.shape[-2:-1]:
        raise ValueError("Expected flow [...,2,K,2] and valid [...,K]")
    valid = valid.bool() & torch.isfinite(flow).all(dim=-1).all(dim=-2)
    clean = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    features = torch.cat((clean[..., 0, :, :], clean[..., 1, :, :] - clean[..., 0, :, :], valid[..., None]), dim=-1)
    return features.masked_fill(~valid[..., None], 0.0), valid


class FrozenTactilePolicyEncoder(nn.Module):
    """One domain tower, shared by five fingers, with the aligned z head."""

    def __init__(self, checkpoint: str | Path | Mapping, *, domain: str | None = None, chunk_size: int = 32):
        super().__init__()
        bundle = dict(checkpoint) if isinstance(checkpoint, Mapping) else load_tactile_checkpoint(checkpoint)
        if bundle.get("format") != BUNDLE_FORMAT or bundle.get("schema_version") != 1:
            raise ValueError("Expected a policy encoder bundle; export with --alignment-checkpoint or --unaligned-sim")
        if bundle.get("input_schema") != INPUT_SCHEMA or tuple(bundle.get("finger_order", ())) != FINGER_ORDER:
            raise ValueError("Incompatible tactile input schema or finger order")
        self.domain = bundle["domain"]
        if self.domain not in ("sim", "real") or (domain is not None and self.domain != domain):
            raise ValueError(f"Expected {domain!r} tower, got {self.domain!r}")
        self.feature_mode = bundle.get("feature_mode", "aligned")
        if self.feature_mode not in ("aligned", "unaligned_sim"):
            raise ValueError(f"Unsupported tactile feature mode: {self.feature_mode!r}")
        self.alignment_id = bundle["alignment_id"]
        self.source_checkpoint_sha256 = bundle.get("source_checkpoint_sha256")
        if self.feature_mode == "unaligned_sim":
            if self.domain != "sim" or bundle["network_type"] != "robust" or self.alignment_id is not None:
                raise ValueError("Unaligned features are a simulation-only baseline, without an alignment ID")
            if not isinstance(self.source_checkpoint_sha256, str) or len(self.source_checkpoint_sha256) != 64:
                raise ValueError("Unaligned simulation bundle must identify its pretrained source checkpoint")
        elif not isinstance(self.alignment_id, str) or not self.alignment_id:
            raise ValueError("Policy encoder bundle must identify its alignment checkpoint")
        self.network_type = bundle["network_type"]
        self.normalization = TactileNormalization(**bundle["normalization"])
        self.encoder = build_tactile_encoder(self.network_type, bundle["model_cfg"])
        self.projection_dim = int(bundle["projection_cfg"]["projection_dim"])
        projection_input_dim = int(bundle["projection_cfg"]["input_dim"])
        if self.feature_mode == "unaligned_sim":
            if self.projection_dim != projection_input_dim:
                raise ValueError("Unaligned simulation features must preserve the encoder latent dimension")
            self.projection = nn.Identity()
        else:
            self.projection = ProjectionHead(**bundle["projection_cfg"])
        cfg, norm = self.encoder.cfg, self.normalization
        if (cfg.image_height, cfg.image_width, cfg.marker_count) != (norm.image_height, norm.image_width, norm.marker_count):
            raise ValueError("Encoder shapes and normalization metadata disagree")
        if cfg.d_model != projection_input_dim:
            raise ValueError("Encoder latent dimension and projection input disagree")
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self.encoder.load_state_dict(bundle["encoder_state"], strict=True)
        self.projection.load_state_dict(bundle["projection_state"], strict=True)
        self.bundle_sha256 = None if isinstance(checkpoint, Mapping) else file_sha256(checkpoint)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def preprocess(self, *, rgb: torch.Tensor, depth_m: torch.Tensor, marker: torch.Tensor,
                   marker_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Match NpzTactileDataset normalization exactly, in any leading batch shape."""
        n = self.normalization
        if rgb.ndim < 4:
            raise ValueError("RGB must have leading batch dimensions and CHW image axes")
        leading = rgb.shape[:-3]
        expected = {
            "rgb": leading + (3, n.image_height, n.image_width),
            "depth_m": leading + (1, n.image_height, n.image_width),
            "marker": leading + (n.marker_count, 5),
        }
        for name, tensor in (("rgb", rgb), ("depth_m", depth_m), ("marker", marker)):
            if tensor.shape != expected[name]:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}; expected {expected[name]}")
            if tensor.device != rgb.device:
                raise ValueError("All tactile modalities must be on the same device")
        if not torch.isfinite(rgb).all() or bool(((rgb < 0) | (rgb > 255)).any()):
            raise ValueError("RGB must contain finite sensor values in [0,255]")
        if not torch.isfinite(depth_m).all():
            raise ValueError("Depth must be finite; sanitize missing ray hits in the sensor adapter")
        valid = marker[..., 4] > 0.5 if marker_valid is None else marker_valid.bool()
        if valid.shape != leading + (n.marker_count,) or valid.device != rgb.device:
            raise ValueError("marker_valid shape/device mismatch")
        valid = valid & torch.isfinite(marker).all(dim=-1)
        normalized_marker = marker.float().clone()
        normalized_marker[..., 0] /= max(1, n.image_width - 1)
        normalized_marker[..., 1] /= max(1, n.image_height - 1)
        normalized_marker[..., 2:4] /= n.marker_motion_scale_px
        normalized_marker[..., 4] = valid
        normalized_marker.masked_fill_(~valid[..., None], 0.0)
        return {
            "rgb": rgb.float().reshape(-1, 3, n.image_height, n.image_width) / 255.0,
            "depth": (depth_m.float() / n.depth_scale_m).clamp(0.0, 1.0).reshape(-1, 1, n.image_height, n.image_width),
            "marker": normalized_marker.reshape(-1, n.marker_count, 5),
            "marker_valid_mask": valid.reshape(-1, n.marker_count),
        }

    @torch.no_grad()
    def forward(self, *, rgb: torch.Tensor, depth_m: torch.Tensor, marker: torch.Tensor,
                marker_valid: torch.Tensor | None = None) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[1] != len(FINGER_ORDER):
            raise ValueError("Policy encoder expects RGB [batch,5,3,H,W] in FINGER_ORDER")
        inputs = self.preprocess(rgb=rgb, depth_m=depth_m, marker=marker, marker_valid=marker_valid)
        chunks = []
        for start in range(0, inputs["rgb"].shape[0], self.chunk_size):
            part = {k: v[start:start + self.chunk_size] for k, v in inputs.items()}
            h = self.encoder.encode(**part)
            chunks.append(F.normalize(self.projection(h), dim=-1))
        return torch.cat(chunks).reshape(rgb.shape[0], len(FINGER_ORDER), self.projection_dim)

    @torch.no_grad()
    def reconstruct(self, *, rgb: torch.Tensor, depth_m: torch.Tensor, marker: torch.Tensor,
                    marker_valid: torch.Tensor | None = None, rgb_reference: torch.Tensor | None = None) -> dict:
        """Diagnostic reconstruction in training-normalized units; never fed back to z."""
        inputs = self.preprocess(rgb=rgb, depth_m=depth_m, marker=marker, marker_valid=marker_valid)
        if self.network_type == "robust" and self.encoder.cfg.rgb_reference_residual:
            if rgb_reference is None or rgb_reference.shape != rgb.shape:
                raise ValueError("RGB reference matching the observation is required for reconstruction")
            inputs["rgb_reference"] = rgb_reference.float().reshape_as(inputs["rgb"]) / 255.0
        chunks = {name: [] for name in ("rgb_recon", "depth_recon", "marker_recon")}
        for start in range(0, inputs["rgb"].shape[0], self.chunk_size):
            output = self.encoder(**{k: v[start:start + self.chunk_size] for k, v in inputs.items()})
            for name in chunks:
                chunks[name].append(output[name])
        return {name: torch.cat(values).reshape(*rgb.shape[:2], *values[0].shape[1:]) for name, values in chunks.items()}

    def observation_contract(self, *, state_dim: int = 152, history_length: int = 4) -> dict:
        contract = {
            "schema_version": 1, "input_schema": INPUT_SCHEMA,
            "feature": "normalized_z", "alignment_id": self.alignment_id,
            "finger_order": list(FINGER_ORDER), "projection_dim": self.projection_dim,
            "state_dim": int(state_dim), "history_length": int(history_length),
            "frame_order": ["state", "finger_z"], "history_order": "oldest_to_newest",
            "observation_dim": (int(state_dim) + len(FINGER_ORDER) * self.projection_dim) * int(history_length),
        }
        if self.feature_mode == "unaligned_sim":
            contract.update(feature="normalized_h", feature_mode=self.feature_mode,
                            source_checkpoint_sha256=self.source_checkpoint_sha256)
        return contract

    def validate_observation_contract(self, saved: Mapping, *, state_dim: int = 152, history_length: int = 4) -> None:
        """Validate a sim-trained policy before using either aligned tower on hardware."""
        expected = self.observation_contract(state_dim=state_dim, history_length=history_length)
        validate_policy_contract(expected, {key: saved.get(key) for key in expected})
        if self.domain == "sim" and "simulation_encoder_sha256" in saved:
            if saved["simulation_encoder_sha256"] != self.bundle_sha256:
                raise ValueError("Simulation policy was trained with a different encoder bundle")


class TactileObservationHistory:
    """Shared simulation/hardware assembly, including per-environment reset."""

    def __init__(self, num_envs: int, state_dim: int, projection_dim: int, history_length: int, device):
        if min(num_envs, state_dim, projection_dim, history_length) < 1:
            raise ValueError("Observation history dimensions must be positive")
        self.state_dim, self.projection_dim = state_dim, projection_dim
        self.frames = torch.zeros(num_envs, history_length, state_dim + len(FINGER_ORDER) * projection_dim, device=device)
        self.needs_fill = torch.ones(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        # Published observations alias frames. PPO copies them only after env.step(),
        # so reset a new buffer to preserve observations from the ending episode.
        self.frames = self.frames.clone()
        self.frames[env_ids] = 0.0
        self.needs_fill[env_ids] = True

    def append(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch = self.frames.shape[0]
        if state.shape != (batch, self.state_dim) or z.shape != (batch, len(FINGER_ORDER), self.projection_dim):
            raise ValueError("State or five-finger latent shape does not match the policy contract")
        frame = torch.cat((state, z.flatten(1)), dim=-1)
        self.frames = torch.roll(self.frames, shifts=-1, dims=1)
        self.frames[:, -1] = frame
        self.frames[self.needs_fill] = frame[self.needs_fill, None, :]
        self.needs_fill.fill_(False)
        return self.frames.flatten(1)


def policy_contract_path(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    for parent in (checkpoint.parent, *checkpoint.parents):
        candidate = parent / "tactile_policy_contract.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing tactile_policy_contract.json beside PPO run: {checkpoint}; legacy policies require retraining")


def validate_policy_contract(expected: Mapping, saved: Mapping) -> None:
    if dict(expected) != dict(saved):
        keys = sorted(k for k in set(expected) | set(saved) if expected.get(k) != saved.get(k))
        raise ValueError(f"Tactile policy observation contract mismatch: {keys}")


def prepare_policy_run(env, log_dir: str | Path, *, resume_path: str | Path | None = None) -> None:
    """Called before loading a PPO checkpoint; a no-op for other tasks."""
    contract = getattr(env.unwrapped, "tactile_policy_contract", None)
    if contract is None:
        return
    if resume_path is not None:
        saved = json.loads(policy_contract_path(resume_path).read_text())
        validate_policy_contract(contract, saved)
    path = Path(log_dir) / "tactile_policy_contract.json"
    if path.exists():
        validate_policy_contract(contract, json.loads(path.read_text()))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contract, indent=2) + "\n")
