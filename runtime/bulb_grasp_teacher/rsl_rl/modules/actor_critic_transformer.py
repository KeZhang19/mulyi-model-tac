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
    # We intentionally do NOT train on rollout-time sequences [T, num_envs, ...].
    # Instead, we tokenize the *per-step* stacked-history observation into multiple tokens
    # (e.g., history_length=5 -> 5 tokens) and keep PPO updates in the standard feedforward
    # minibatch mode (flattened [T * num_envs, ...]).
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
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_dependent_std=False,
        transformer_model_dim: int = 256,
        transformer_num_layers: int = 4,
        transformer_num_heads: int = 8,
        transformer_dropout: float = 0.0,
        max_seq_len: int = 128,  # Maximum sequence length for positional encoding
        # How many tokens to split ONE observation into.
        # If your env stacks history in the feature dimension (e.g. history_length=5),
        # set this to 5 so the transformer self-attention operates over that stacked history.
        obs_history_tokens: int = 5,
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
        
        # Print network information
        print("=" * 80)
        print("[INFO] Using ActorCriticTransformer Network")
        print(f"  - Transformer model dim: {transformer_model_dim}")
        print(f"  - Transformer layers: {transformer_num_layers}")
        print(f"  - Attention heads: {transformer_num_heads}")
        print(f"  - Max sequence length: {max_seq_len}")
        print(f"  - Obs tokens per step: {obs_history_tokens}")
        print(f"  - Actor hidden dims: {actor_hidden_dims}")
        print(f"  - Critic hidden dims: {critic_hidden_dims}")
        print("=" * 80)

        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std
        self.max_seq_len = max_seq_len
        self.obs_history_tokens = int(obs_history_tokens) if obs_history_tokens is not None else 1
        if self.obs_history_tokens < 1:
            self.obs_history_tokens = 1

        # get observation dimensions
        # We treat ONE env step as a token sequence of length H = obs_history_tokens.
        # Each observation group is expected to be flattened history: [B, H * d_step].
        # We'll reshape it to [B, H, d_step] and concatenate across groups on the feature axis.
        num_actor_obs_per_token = 0
        can_split_all_actor = True
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "Transformer module only supports 1D observations."
            group_dim = int(obs[obs_group].shape[-1])
            if self.obs_history_tokens > 1 and group_dim % self.obs_history_tokens == 0:
                num_actor_obs_per_token += group_dim // self.obs_history_tokens
            else:
                # Fallback: cannot split cleanly -> treat as static features repeated across tokens
                num_actor_obs_per_token += group_dim
                can_split_all_actor = False

        num_critic_obs_per_token = 0
        can_split_all_critic = True
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "Transformer module only supports 1D observations."
            group_dim = int(obs[obs_group].shape[-1])
            if self.obs_history_tokens > 1 and group_dim % self.obs_history_tokens == 0:
                num_critic_obs_per_token += group_dim // self.obs_history_tokens
            else:
                num_critic_obs_per_token += group_dim
                can_split_all_critic = False

        if self.obs_history_tokens > 1 and (not can_split_all_actor or not can_split_all_critic):
            print(
                "[WARN] Some observation groups cannot be evenly split by obs_history_tokens="
                f"{self.obs_history_tokens}. They will be broadcast (repeated) across tokens."
            )

        # transformer backbones (separate transformers for actor and critic)
        # Project raw observations to transformer model dimension using a
        # single hidden layer (2x model dim).
        self.obs_project_actor = MLP(
            num_actor_obs_per_token, transformer_model_dim, [transformer_model_dim * 2], activation
        )
        self.obs_project_critic = MLP(
            num_critic_obs_per_token, transformer_model_dim, [transformer_model_dim * 2], activation
        )

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
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_per_token)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # critic
        self.critic = MLP(transformer_model_dim, 1, critic_hidden_dims, activation)

        # critic normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs_per_token)
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

    def _apply_transformer_tokens(self, tokens_lbd, transformer, ln_final):
        """Run Transformer on token sequence.

        Args:
            tokens_lbd: Tensor shaped [L, B, D_model] (LND / LBD).
        Returns:
            pooled_features: Tensor shaped [B, D_model] (we use the last token).
        """
        if tokens_lbd.dim() != 3:
            raise ValueError(f"Expected tokens with rank 3 [L,B,D], got rank={tokens_lbd.dim()}")

        features = transformer(tokens_lbd)  # [L, B, D]
        features = ln_final(features)       # [L, B, D]

        # With causal attention, the last token can attend to all previous tokens,
        # so it forms a natural summary of the stacked history window.
        pooled = features[-1]  # [B, D]
        return pooled, None

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
        # Ensure standard deviation is valid (>= 0) to avoid torch.distributions.Normal crash.
        # Keep behavior consistent with MLP/RNN policies.
        std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
        if self.noise_std_type == "scalar":
            std = std.abs()
        std = torch.clamp(std, min=1e-6)
        self.distribution = Normal(mean, std)

    def act(self, obs, masks=None, hidden_states=_default):
        tokens = self.get_actor_obs_tokens(obs)                 # [L, B, D_token]
        tokens = self.actor_obs_normalizer(tokens)              # broadcast over [L,B,*]
        tokens = self.obs_project_actor(tokens)                 # [L, B, D_model]
        features, _ = self._apply_transformer_tokens(tokens, self.transformer_actor, self.ln_final_actor)

        self._update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None):
        tokens = self.get_actor_obs_tokens(obs)                 # [L, B, D_token]
        tokens = self.actor_obs_normalizer(tokens)
        tokens = self.obs_project_actor(tokens)                 # [L, B, D_model]
        features, _ = self._apply_transformer_tokens(tokens, self.transformer_actor, self.ln_final_actor)

        if self.state_dependent_std:
            return self.actor(features)[..., 0, :]
        else:
            return self.actor(features)

    def evaluate(self, obs, masks=None, hidden_states=_default):
        tokens = self.get_critic_obs_tokens(obs)                # [L, B, D_token]
        tokens = self.critic_obs_normalizer(tokens)
        tokens = self.obs_project_critic(tokens)                # [L, B, D_model]
        features, _ = self._apply_transformer_tokens(tokens, self.transformer_critic, self.ln_final_critic)

        return self.critic(features)

    def _group_to_tokens_bld(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a single obs-group tensor to [B, L, d] token format.

        Expected input:
          - [B, D_total] where D_total is either H*d (stacked history) or static features.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected group obs of rank 2 [B,D], got rank={x.dim()}")
        b, d_total = x.shape
        h = self.obs_history_tokens
        if h > 1 and d_total % h == 0:
            return x.reshape(b, h, d_total // h)
        # Fallback: broadcast static features across tokens
        return x.unsqueeze(1).expand(b, h, d_total)

    def get_actor_obs_tokens(self, obs) -> torch.Tensor:
        """Build actor token sequence from observation dict.

        Returns:
            Tensor shaped [L, B, D_token] where L = obs_history_tokens.
        """
        tokens_bld = []
        for obs_group in self.obs_groups["policy"]:
            tokens_bld.append(self._group_to_tokens_bld(obs[obs_group]))
        # [B, L, sum_d] -> [L, B, sum_d]
        tokens_bld = torch.cat(tokens_bld, dim=-1)
        return tokens_bld.permute(1, 0, 2).contiguous()

    def get_critic_obs_tokens(self, obs) -> torch.Tensor:
        """Build critic token sequence from observation dict.

        Returns:
            Tensor shaped [L, B, D_token] where L = obs_history_tokens.
        """
        tokens_bld = []
        for obs_group in self.obs_groups["critic"]:
            tokens_bld.append(self._group_to_tokens_bld(obs[obs_group]))
        tokens_bld = torch.cat(tokens_bld, dim=-1)
        return tokens_bld.permute(1, 0, 2).contiguous()

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
            actor_tokens = self.get_actor_obs_tokens(obs)  # [L, B, D]
            self.actor_obs_normalizer.update(actor_tokens.flatten(0, 1))
        if self.critic_obs_normalization:
            critic_tokens = self.get_critic_obs_tokens(obs)
            self.critic_obs_normalizer.update(critic_tokens.flatten(0, 1))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def print_state(self, state):
        # CLIP Transformer doesn't use cache, so this is a no-op
        pass