# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from .rollout_storage import RolloutStorage


class RolloutStorageDualHead(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.left_rewards = None
            self.right_rewards = None
            self.left_values = None
            self.right_values = None
            self.left_actions_log_prob = None
            self.right_actions_log_prob = None

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
    ):
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device=device)

        if training_type == "rl":
            self.left_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.right_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.left_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.right_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.left_actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.right_actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.left_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.right_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.left_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.right_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

    def add_transitions(self, transition: Transition):
        super().add_transitions(transition)
        if self.training_type != "rl":
            return

        step = self.step - 1
        self.left_rewards[step].copy_(transition.left_rewards.view(-1, 1))
        self.right_rewards[step].copy_(transition.right_rewards.view(-1, 1))
        self.left_values[step].copy_(transition.left_values)
        self.right_values[step].copy_(transition.right_values)
        self.left_actions_log_prob[step].copy_(transition.left_actions_log_prob.view(-1, 1))
        self.right_actions_log_prob[step].copy_(transition.right_actions_log_prob.view(-1, 1))

    def compute_returns(self, last_left_values, last_right_values, gamma, lam, normalize_advantage: bool = True):
        left_advantage = 0
        right_advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_left_values = last_left_values
                next_right_values = last_right_values
            else:
                next_left_values = self.left_values[step + 1]
                next_right_values = self.right_values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()

            left_delta = self.left_rewards[step] + next_is_not_terminal * gamma * next_left_values - self.left_values[step]
            right_delta = (
                self.right_rewards[step] + next_is_not_terminal * gamma * next_right_values - self.right_values[step]
            )

            left_advantage = left_delta + next_is_not_terminal * gamma * lam * left_advantage
            right_advantage = right_delta + next_is_not_terminal * gamma * lam * right_advantage

            self.left_returns[step] = left_advantage + self.left_values[step]
            self.right_returns[step] = right_advantage + self.right_values[step]

        self.left_advantages = self.left_returns - self.left_values
        self.right_advantages = self.right_returns - self.right_values

        if normalize_advantage:
            self.left_advantages = (self.left_advantages - self.left_advantages.mean()) / (
                self.left_advantages.std() + 1e-8
            )
            self.right_advantages = (self.right_advantages - self.right_advantages.mean()) / (
                self.right_advantages.std() + 1e-8
            )

        self.returns = self.left_returns + self.right_returns
        self.advantages = self.left_advantages + self.right_advantages

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        left_values = self.left_values.flatten(0, 1)
        right_values = self.right_values.flatten(0, 1)
        left_returns = self.left_returns.flatten(0, 1)
        right_returns = self.right_returns.flatten(0, 1)
        left_advantages = self.left_advantages.flatten(0, 1)
        right_advantages = self.right_advantages.flatten(0, 1)
        old_left_actions_log_prob = self.left_actions_log_prob.flatten(0, 1)
        old_right_actions_log_prob = self.right_actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    left_values[batch_idx],
                    right_values[batch_idx],
                    left_advantages[batch_idx],
                    right_advantages[batch_idx],
                    left_returns[batch_idx],
                    right_returns[batch_idx],
                    old_left_actions_log_prob[batch_idx],
                    old_right_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                )
