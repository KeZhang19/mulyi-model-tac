"""Bounded Repose PPO policy, checkpoint selection, and inference export.

PPO stores the *latent Gaussian* mean/std for its analytic KL scheduler. KL is
unchanged by the common invertible tanh transform. Executed actions and their
log probabilities are in the transformed action space. No PPO patch is needed.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import torch
from rsl_rl.modules import ActorCritic
from torch import nn
from torch.distributions import Normal


class _BoundLatentMean(nn.Module):
    """Prevent numerical tanh saturation without hard-clipping actor gradients."""

    def __init__(self, limit: float = 2.0):
        super().__init__()
        self.limit = limit

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.limit * torch.tanh(value / self.limit)


class ReposeActorCritic(ActorCritic):
    """Tanh Gaussian with bounded latent exploration and exact change of variables.

    ``action_mean`` and ``action_std`` intentionally retain the base class's
    Gaussian interpretation for RSL-RL's rollout storage and KL calculation.
    Deterministic actions are ``tanh(action_mean)``, not ``action_mean``.
    Entropy uses one reparameterized Monte Carlo sample per observation.
    """

    minimum_std = 0.05
    maximum_std = 0.7

    def __init__(self, *args, init_noise_std: float = 0.35,
                 noise_std_type: str = "log", state_dependent_std: bool = False, **kwargs):
        if state_dependent_std:
            raise ValueError("ReposeActorCritic requires a state-independent standard deviation")
        if not self.minimum_std < init_noise_std < self.maximum_std:
            raise ValueError("Initial Repose noise must lie strictly between 0.05 and 0.7")
        super().__init__(*args, init_noise_std=init_noise_std, noise_std_type="log",
                         state_dependent_std=False, **kwargs)
        # Keep the bounded parameterization smooth even at the exploration limit.
        fraction = (init_noise_std - self.minimum_std) / (self.maximum_std - self.minimum_std)
        self.noise_logit = nn.Parameter(torch.full_like(self.log_std, math.log(fraction / (1.0 - fraction))))
        del self.log_std
        self.latent_mean_bound = _BoundLatentMean()
        self.register_buffer("_repose_policy_version", torch.tensor(2, dtype=torch.int64))
        # With incremental joint targets a fresh policy should initially hold
        # its reset grasp. Small outputs retain gradients through the whole MLP.
        output_layer = [layer for layer in self.actor.modules() if isinstance(layer, nn.Linear)][-1]
        nn.init.orthogonal_(output_layer.weight, gain=0.01)
        nn.init.zeros_(output_layer.bias)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.latent_mean_bound(self.actor(obs))
        std = self.minimum_std + (self.maximum_std - self.minimum_std) * torch.sigmoid(self.noise_logit)
        self.distribution = Normal(mean, std.expand_as(mean))

    def act(self, obs, **kwargs) -> torch.Tensor:
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._update_distribution(obs)
        return torch.tanh(self.distribution.sample())

    def act_inference(self, obs) -> torch.Tensor:
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return torch.tanh(self.latent_mean_bound(self.actor(obs)))

    @staticmethod
    def _log_tanh_jacobian(latent: torch.Tensor) -> torch.Tensor:
        # Stable log(1 - tanh(z)^2), including near the action-space boundary.
        return 2.0 * (math.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Float32 can round extreme samples to +/-1. This guard also makes
        # boundary probes finite; rollout and replay use the identical inverse.
        epsilon = torch.finfo(actions.dtype).eps
        bounded = actions.clamp(-1.0 + epsilon, 1.0 - epsilon)
        latent = torch.atanh(bounded)
        return (self.distribution.log_prob(latent) - self._log_tanh_jacobian(latent)).sum(dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        latent = self.distribution.rsample()
        return (self.distribution.entropy() + self._log_tanh_jacobian(latent)).sum(dim=-1)


def register_repose_policy() -> None:
    """Register the independent class in RSL-RL's runner eval namespace."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.ReposeActorCritic = ReposeActorCritic


def configure_repose_checkpoint(agent_cfg, checkpoint_path: str | None = None) -> int:
    """Select the saved policy family before constructing a Repose environment.

    Return the task revision (1 for legacy checkpoints, 2 for new training).
    Callers must apply this revision to the Repose env config before gym.make.
    Checkpoint tensors are loaded onto CPU, never onto a running training GPU.
    """
    register_repose_policy()
    if checkpoint_path is None:
        agent_cfg.policy.class_name = "ReposeActorCritic"
        agent_cfg.policy.noise_std_type = "log"
        return 2
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    if "_repose_policy_version" in state:
        revision = int(state["_repose_policy_version"].item())
        if revision != 2:
            raise ValueError(f"Unsupported Repose policy revision: {revision}")
        agent_cfg.policy.class_name = "ReposeActorCritic"
        agent_cfg.policy.noise_std_type = "log"
        return revision
    agent_cfg.policy.class_name = "ActorCritic"
    agent_cfg.policy.noise_std_type = "log" if "log_std" in state else "scalar"
    return 1


def repose_policy_for_export(policy):
    """Include both latent mean bounding and action tanh in Isaac Lab exports.

    Isaac Lab exports only ``policy.actor`` and otherwise bypasses
    ``act_inference``. A flat Sequential preserves its actor[0].in_features
    convention for ONNX. The original training actor and normalizer are intact.
    """
    if not isinstance(policy, ReposeActorCritic):
        return policy
    # Sequential iteration preserves reused activations; children() deduplicates
    # modules and would silently omit ELU layers in RSL-RL's MLP.
    actor = nn.Sequential(*copy.deepcopy(list(policy.actor)),
                          copy.deepcopy(policy.latent_mean_bound), nn.Tanh())
    return SimpleNamespace(actor=actor, is_recurrent=False)
