# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.networks.transformer import TransformerWithPositionalEncoding, LayerNorm

_default = object()


class ActorCriticTransformer(nn.Module):
    is_recurrent = False
    is_transformerxl = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_dependent_std=False,
        transformer_model_dim: int = 256,
        transformer_num_layers: int = 4,
        transformer_num_heads: int = 8,
        transformer_dropout: float = 0.0,
        max_seq_len: int = 128,  # Maximum sequence length for positional encoding
        # Note: CLIP Transformer doesn't use memory_length, pos_bias_max, or norm_type
        # These parameters are kept for compatibility but ignored
        memory_length: int = 128,
        transformer_pos_bias_max: float = 2.0,
        transformer_norm_type: str = "rms",
        **kwargs,
    ):
        # print(memory_length)
        # print("-"*60)
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys())),
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std
        self.max_seq_len = max_seq_len

        # get observation dimensions
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "Transformer module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "Transformer module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # transformer backbones (separate transformers for actor and critic)
        # Project raw observations to transformer model dimension using a
        # single hidden layer (2x model dim).
        self.obs_project_actor = MLP(num_actor_obs, transformer_model_dim, [transformer_model_dim * 2], activation)
        self.obs_project_critic = MLP(num_critic_obs, transformer_model_dim, [transformer_model_dim * 2], activation)

        # Build causal attention mask
        causal_mask = self.build_attention_mask()
        
        # Transformer with positional encoding (built-in)
        self.transformer_actor = TransformerWithPositionalEncoding(
            width=transformer_model_dim,
            layers=transformer_num_layers,
            heads=transformer_num_heads,
            max_seq_len=max_seq_len,
            attn_mask=causal_mask  # Causal mask for RL
        )
        self.transformer_critic = TransformerWithPositionalEncoding(
            width=transformer_model_dim,
            layers=transformer_num_layers,
            heads=transformer_num_heads,
            max_seq_len=max_seq_len,
            attn_mask=causal_mask  # Causal mask for RL
        )
        
        # Final normalization layers
        self.ln_final_actor = LayerNorm(transformer_model_dim)
        self.ln_final_critic = LayerNorm(transformer_model_dim)

        # actor
        if self.state_dependent_std:
            self.actor = MLP(transformer_model_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(transformer_model_dim, num_actions, actor_hidden_dims, activation)

        # actor normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # critic
        self.critic = MLP(transformer_model_dim, 1, critic_hidden_dims, activation)

        # critic normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def build_attention_mask(self):
        """Build causal attention mask for transformer.
        
        Creates an upper triangular mask where future positions are masked out (-inf).
        This ensures that each position can only attend to previous positions and itself.
        
        Returns:
            torch.Tensor: Causal attention mask of shape [max_seq_len, max_seq_len]
        """
        # Create causal attention mask
        # PyTorch uses additive attention mask; fill with -inf for masked positions
        mask = torch.empty(self.max_seq_len, self.max_seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # Zero out the lower diagonal (keep upper triangular as -inf)
        return mask

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None, hidden_states=None):
        pass  # CLIP Transformer doesn't use cache

    def forward(self):
        raise NotImplementedError

    def _apply_transformer(self, obs, transformer, ln_final, is_actor=True):
        """Run the Transformer backbone with positional encoding.
        
        Transformer expects [seq_len, batch, dim] (LND format).
        
        - Streaming mode (collecting data): obs is [num_envs, feature_dim].
        - Training mode: obs is [time, num_envs, feature_dim].
        """
        if obs.dim() == 2:
            # [B, D] -> [1, B, D] for Transformer (LND format)
            transformer_input = obs.unsqueeze(0)  # [1, B, D]
            squeeze_output = True
        elif obs.dim() == 3:
            # [T, B, D] -> already in LND format for Transformer
            transformer_input = obs  # [T, B, D]
            squeeze_output = False
        else:
            raise ValueError(f"Unsupported observation rank {obs.dim()}; expected 2 or 3.")

        # Transformer forward (positional encoding is applied inside TransformerWithPositionalEncoding)
        features = transformer(transformer_input)  # [T, B, D] or [1, B, D]
        
        # Apply final normalization
        features = ln_final(features)
        
        if squeeze_output:
            # [1, B, D] -> [B, D]
            features = features.squeeze(0)
        else:
            # Keep [T, B, D] format
            pass
        
        # Return None as state (Transformer doesn't use cache)
        return features, None

    def _update_distribution(self, features):
        if self.state_dependent_std:
            mean_and_std = self.actor(features)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(features)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, masks=None, hidden_states=_default):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        obs = self.obs_project_actor(obs)
        features, _ = self._apply_transformer(
            obs,
            self.transformer_actor,
            self.ln_final_actor,
            is_actor=True
        )

        self._update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        obs = self.obs_project_actor(obs)
        features, _ = self._apply_transformer(
            obs,
            self.transformer_actor,
            self.ln_final_actor,
            is_actor=True
        )

        if self.state_dependent_std:
            return self.actor(features)[..., 0, :]
        else:
            return self.actor(features)

    def evaluate(self, obs, masks=None, hidden_states=_default):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        obs = self.obs_project_critic(obs)
        features, _ = self._apply_transformer(
            obs,
            self.transformer_critic,
            self.ln_final_critic,
            is_actor=False
        )

        return self.critic(features)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return None, None

    def get_transformerxl_state(self):
        return None, None  # CLIP Transformer doesn't use cache

    def detach_hidden_states(self, dones=None):
        return None, None

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def print_state(self, state):
        # CLIP Transformer doesn't use cache, so this is a no-op
        pass
