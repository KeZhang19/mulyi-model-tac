"""CTTP-style latent alignment between two tactile domain encoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..training.contrastive import symmetric_masked_infonce_loss


class ProjectionHead(nn.Module):
    """Two-layer projection head used only by the contrastive objective."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int | None = None,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        projection_dim = int(projection_dim)
        hidden_dim = projection_dim if hidden_dim is None else int(hidden_dim)
        if input_dim <= 0 or projection_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Projection dimensions must be positive")
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        if use_layernorm:
            layers.append(nn.LayerNorm(input_dim))
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, projection_dim),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError(
                f"ProjectionHead expects [batch, latent_dim], got {tuple(latent.shape)}"
            )
        return self.net(latent)


class TactileLatentAlignmentNetwork(nn.Module):
    """Align latent representations from simulation and real tactile towers.

    The two encoders may have different architectures, just as CLIP uses
    different image and text towers.  Each tower produces a task-facing
    representation ``h`` and has a private two-layer projection head that
    produces the normalized contrastive representation ``z``.  Only ``z`` is
    passed to InfoNCE; callers should keep using ``h`` for policies and
    downstream probes.
    """

    def __init__(
        self,
        sim_encoder: nn.Module,
        real_encoder: nn.Module,
        *,
        latent_dim: int,
        projection_dim: int = 256,
        projection_hidden_dim: int | None = None,
        projection_layernorm: bool = False,
        temperature: float = 0.1,
        freeze_encoders: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(sim_encoder, "encode") or not hasattr(real_encoder, "encode"):
            raise TypeError("Both domain encoders must expose encode(**inputs)")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        self.sim_encoder = sim_encoder
        self.real_encoder = real_encoder
        self.sim_projection = ProjectionHead(
            latent_dim,
            projection_dim,
            projection_hidden_dim,
            use_layernorm=projection_layernorm,
        )
        self.real_projection = ProjectionHead(
            latent_dim,
            projection_dim,
            projection_hidden_dim,
            use_layernorm=projection_layernorm,
        )
        self.temperature = float(temperature)
        self.freeze_encoders = bool(freeze_encoders)
        self.set_encoder_trainable(not self.freeze_encoders)

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze both domain encoders.

        Alignment defaults to frozen encoders so that the first experiment
        measures whether the already-trained representations can be aligned by
        projection heads alone.  A later fine-tuning stage can explicitly call
        ``set_encoder_trainable(True)``.
        """

        trainable = bool(trainable)
        self.freeze_encoders = not trainable
        for encoder in (self.sim_encoder, self.real_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(trainable)
        if not trainable:
            # A frozen encoder must also stay in eval mode when the wrapper is
            # put into train mode; otherwise dropout or running-stat updates
            # make the supposedly fixed latent stochastic.
            self.sim_encoder.eval()
            self.real_encoder.eval()

    def set_encoder_prefixes_trainable(self, prefixes: tuple[str, ...]) -> tuple[str, ...]:
        """Enable gradients only for selected late encoder submodules.

        The wrapper keeps the encoders in evaluation mode so all untouched
        backbone parameters remain deterministic.  This is intended for a
        low-learning-rate second stage after projection-head warm-up.
        """

        if not prefixes:
            raise ValueError("At least one encoder prefix is required")
        enabled: list[str] = []
        for encoder in (self.sim_encoder, self.real_encoder):
            for name, parameter in encoder.named_parameters():
                trainable = any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in prefixes
                )
                parameter.requires_grad_(trainable)
                if trainable:
                    enabled.append(name)
            encoder.eval()
        self.freeze_encoders = True
        if not enabled:
            raise ValueError(
                "None of the requested encoder prefixes matched a parameter: "
                f"{prefixes}"
            )
        return tuple(enabled)

    def train(self, mode: bool = True) -> "TactileLatentAlignmentNetwork":
        """Set train/eval mode while preserving frozen-encoder semantics."""

        super().train(mode)
        if self.freeze_encoders:
            self.sim_encoder.eval()
            self.real_encoder.eval()
        return self

    def trainable_parameters(self):
        """Yield parameters that should be passed to the alignment optimizer."""

        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _encode(encoder: nn.Module, inputs: Mapping[str, Any]) -> torch.Tensor:
        latent = encoder.encode(**dict(inputs))
        if not torch.is_tensor(latent) or latent.ndim != 2:
            raise ValueError(
                "Domain encoder encode(**inputs) must return [batch, latent_dim], "
                f"got {type(latent).__name__} with shape "
                f"{getattr(latent, 'shape', None)}"
            )
        return latent

    def forward(
        self,
        sim_inputs: Mapping[str, Any],
        real_inputs: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Return task latents ``h_*`` and projected latents ``z_*``."""

        h_sim = self._encode(self.sim_encoder, sim_inputs)
        h_real = self._encode(self.real_encoder, real_inputs)
        if h_sim.shape[1] != self.sim_projection.input_dim:
            raise ValueError(
                "Simulation latent dimension does not match latent_dim: "
                f"got {h_sim.shape[1]}, expected {self.sim_projection.net[0].in_features}"
            )
        if h_real.shape[1] != self.real_projection.input_dim:
            raise ValueError(
                "Real latent dimension does not match latent_dim: "
                f"got {h_real.shape[1]}, expected {self.real_projection.net[0].in_features}"
            )
        z_sim = torch.nn.functional.normalize(self.sim_projection(h_sim), dim=-1)
        z_real = torch.nn.functional.normalize(self.real_projection(h_real), dim=-1)
        return {"h_sim": h_sim, "h_real": h_real, "z_sim": z_sim, "z_real": z_real}

    def alignment_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        positive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE on a forward output dictionary."""

        return symmetric_masked_infonce_loss(
            outputs["z_sim"],
            outputs["z_real"],
            positive_mask,
            temperature=self.temperature,
        )


class AffineTactileLatentAlignmentNetwork(nn.Module):
    """Frozen-tower affine adapters for a ridge-fitted common latent space.

    This is the deterministic linear-projection counterpart to
    :class:`TactileLatentAlignmentNetwork`.  It is useful when the pretrained
    towers already contain the same physical information but their coordinate
    systems differ substantially.  Each tower has an independent affine map;
    the real adapter can be initialized to identity when the real tower is the
    chosen common-space anchor.
    """

    def __init__(
        self,
        sim_encoder: nn.Module,
        real_encoder: nn.Module,
        *,
        latent_dim: int,
        projection_dim: int | None = None,
        freeze_encoders: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(sim_encoder, "encode") or not hasattr(real_encoder, "encode"):
            raise TypeError("Both domain encoders must expose encode(**inputs)")
        latent_dim = int(latent_dim)
        projection_dim = latent_dim if projection_dim is None else int(projection_dim)
        if latent_dim <= 0 or projection_dim <= 0:
            raise ValueError("Latent and projection dimensions must be positive")
        self.sim_encoder = sim_encoder
        self.real_encoder = real_encoder
        self.sim_projection = nn.Linear(latent_dim, projection_dim)
        self.real_projection = nn.Linear(latent_dim, projection_dim)
        self.freeze_encoders = bool(freeze_encoders)
        self.set_encoder_trainable(not self.freeze_encoders)

    def set_encoder_trainable(self, trainable: bool) -> None:
        trainable = bool(trainable)
        self.freeze_encoders = not trainable
        for encoder in (self.sim_encoder, self.real_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(trainable)
            if not trainable:
                encoder.eval()

    def train(self, mode: bool = True) -> "AffineTactileLatentAlignmentNetwork":
        super().train(mode)
        if self.freeze_encoders:
            self.sim_encoder.eval()
            self.real_encoder.eval()
        return self

    @staticmethod
    def _encode(encoder: nn.Module, inputs: Mapping[str, Any]) -> torch.Tensor:
        latent = encoder.encode(**dict(inputs))
        if not torch.is_tensor(latent) or latent.ndim != 2:
            raise ValueError("Domain encoder encode(**inputs) must return [batch, latent_dim]")
        return latent

    def forward(
        self,
        sim_inputs: Mapping[str, Any],
        real_inputs: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        h_sim = self._encode(self.sim_encoder, sim_inputs)
        h_real = self._encode(self.real_encoder, real_inputs)
        z_sim = torch.nn.functional.normalize(self.sim_projection(h_sim), dim=-1)
        z_real = torch.nn.functional.normalize(self.real_projection(h_real), dim=-1)
        return {"h_sim": h_sim, "h_real": h_real, "z_sim": z_sim, "z_real": z_real}
