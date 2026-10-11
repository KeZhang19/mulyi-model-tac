# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class DualHeadRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL wrapper that exposes left/right split rewards for dual-head PPO."""

    def __init__(
        self,
        env,
        clip_actions: float | None = None,
        reward_left_terms: list[str] | tuple[str, ...] | None = None,
        reward_right_terms: list[str] | tuple[str, ...] | None = None,
        reward_shared_terms: list[str] | tuple[str, ...] | None = None,
    ):
        self.reward_left_terms = tuple(reward_left_terms) if reward_left_terms is not None else None
        self.reward_right_terms = tuple(reward_right_terms) if reward_right_terms is not None else None
        self.reward_shared_terms = tuple(reward_shared_terms) if reward_shared_terms is not None else None
        super().__init__(env, clip_actions=clip_actions)

    def step(self, actions: torch.Tensor):
        obs, rew, dones, extras = super().step(actions)

        reward_manager = getattr(self.unwrapped, "reward_manager", None)
        if reward_manager is None:
            raise ValueError("Dual-head wrapper requires an environment with a reward_manager.")

        term_names = reward_manager.active_terms
        per_term_reward = reward_manager._step_reward * self.unwrapped.step_dt

        left_unique_reward = self._sum_reward_terms(per_term_reward, term_names, self.reward_left_terms, prefix="left_")
        right_unique_reward = self._sum_reward_terms(per_term_reward, term_names, self.reward_right_terms, prefix="right_")
        shared_reward = self._sum_reward_terms(per_term_reward, term_names, self.reward_shared_terms, prefix="shared_")

        left_reward = left_unique_reward + shared_reward
        right_reward = right_unique_reward + shared_reward

        residual = torch.max(torch.abs((left_unique_reward + right_unique_reward + shared_reward) - rew))
        if residual.item() > 1.0:
            raise RuntimeError(
                "Dual-head reward split does not match the environment reward. "
                f"Max residual: {residual.item():.6e}"
            )

        extras = dict(extras)
        extras["reward_left"] = left_reward
        extras["reward_right"] = right_reward
        extras["reward_shared"] = shared_reward

        return obs, rew, dones, extras

    def _sum_reward_terms(
        self,
        per_term_reward: torch.Tensor,
        term_names: list[str],
        named_terms: tuple[str, ...] | None,
        prefix: str,
    ) -> torch.Tensor:
        if named_terms is not None:
            name_to_index = {name: idx for idx, name in enumerate(term_names)}
            indices = []
            for term_name in named_terms:
                if term_name not in name_to_index:
                    continue
                indices.append(name_to_index[term_name])
        else:
            indices = [idx for idx, term_name in enumerate(term_names) if term_name.startswith(prefix)]

        if not indices:
            return torch.zeros(per_term_reward.shape[0], device=per_term_reward.device, dtype=per_term_reward.dtype)

        return per_term_reward[:, indices].sum(dim=1)
