# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


def _build_trunk(input_dim: int, hidden_dims: list[int], activation: str) -> tuple[nn.Sequential, int]:
    if len(hidden_dims) == 0:
        return nn.Sequential(nn.Identity()), input_dim

    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(resolve_nn_activation(activation))
        prev_dim = hidden_dim
    return nn.Sequential(*layers), prev_dim


def _build_head(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(resolve_nn_activation(activation))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCriticDualHead(nn.Module):
    is_recurrent = False
    is_transformerxl = False
    is_sequence_model = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        actor_head_hidden_dims=[128, 64],
        critic_head_hidden_dims=[128, 64],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_dependent_std=False,
        left_action_terms: list[str] | None = None,
        right_action_terms: list[str] | None = None,
        left_action_indices: list[int] | tuple[int, ...] | None = None,
        right_action_indices: list[int] | tuple[int, ...] | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDualHead.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if state_dependent_std:
            raise NotImplementedError("ActorCriticDualHead currently supports only global action standard deviations.")

        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type
        self.state_dependent_std = False
        self.left_action_terms = list(left_action_terms or [])
        self.right_action_terms = list(right_action_terms or [])

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "ActorCriticDualHead only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "ActorCriticDualHead only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        if left_action_indices is None or right_action_indices is None:
            raise ValueError("ActorCriticDualHead requires resolved left_action_indices and right_action_indices.")

        left_action_indices = tuple(int(index) for index in left_action_indices)
        right_action_indices = tuple(int(index) for index in right_action_indices)
        overlap = set(left_action_indices).intersection(right_action_indices)
        if overlap:
            raise ValueError(f"Left and right action indices overlap: {sorted(overlap)}")
        covered_indices = sorted(left_action_indices + right_action_indices)
        if covered_indices != list(range(num_actions)):
            raise ValueError(
                "Dual-head action partition must cover the full action vector exactly once. "
                f"Received indices: {covered_indices}, expected: {list(range(num_actions))}"
            )

        self.register_buffer(
            "left_action_index_tensor",
            torch.tensor(left_action_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "right_action_index_tensor",
            torch.tensor(right_action_indices, dtype=torch.long),
            persistent=False,
        )

        self.actor_trunk, actor_latent_dim = _build_trunk(num_actor_obs, actor_hidden_dims, activation)
        self.actor_left_head = _build_head(
            actor_latent_dim, actor_head_hidden_dims, len(left_action_indices), activation
        )
        self.actor_right_head = _build_head(
            actor_latent_dim, actor_head_hidden_dims, len(right_action_indices), activation
        )

        self.critic_trunk, critic_latent_dim = _build_trunk(num_critic_obs, critic_hidden_dims, activation)
        self.critic_left_head = _build_head(critic_latent_dim, critic_head_hidden_dims, 1, activation)
        self.critic_right_head = _build_head(critic_latent_dim, critic_head_hidden_dims, 1, activation)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"Actor trunk: {self.actor_trunk}")
        print(f"Actor left head: {self.actor_left_head}")
        print(f"Actor right head: {self.actor_right_head}")
        print(f"Critic trunk: {self.critic_trunk}")
        print(f"Critic left head: {self.critic_left_head}")
        print(f"Critic right head: {self.critic_right_head}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_mean(self, obs: torch.Tensor) -> torch.Tensor:
        actor_latent = self.actor_trunk(obs)
        left_mean = self.actor_left_head(actor_latent)
        right_mean = self.actor_right_head(actor_latent)
        mean = torch.zeros((obs.shape[0], self.num_actions), device=obs.device, dtype=obs.dtype)
        mean[:, self.left_action_index_tensor] = left_mean
        mean[:, self.right_action_index_tensor] = right_mean
        return mean

    def _critic_values(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        critic_latent = self.critic_trunk(obs)
        return self.critic_left_head(critic_latent), self.critic_right_head(critic_latent)

    def update_distribution(self, obs):
        mean = self._actor_mean(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self._actor_mean(obs)

    def evaluate_heads(self, obs, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self._critic_values(obs)

    def evaluate(self, obs, **kwargs):
        left_value, right_value = self.evaluate_heads(obs, **kwargs)
        return left_value + right_value

    def get_actor_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actions_log_prob_heads(self, actions) -> tuple[torch.Tensor, torch.Tensor]:
        log_prob = self.distribution.log_prob(actions)
        left_log_prob = log_prob[:, self.left_action_index_tensor].sum(dim=-1)
        right_log_prob = log_prob[:, self.right_action_index_tensor].sum(dim=-1)
        return left_log_prob, right_log_prob

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
