# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from .ppo import PPO
from rsl_rl.storage import RolloutStorageDualHead


class PPODualHead(PPO):
    """PPO variant with separate left/right actor and critic objectives."""

    def __init__(self, policy, *args, rnd_cfg=None, symmetry_cfg=None, **kwargs):
        if rnd_cfg is not None:
            raise NotImplementedError("PPODualHead does not currently support RND.")
        if symmetry_cfg is not None:
            raise NotImplementedError("PPODualHead does not currently support symmetry augmentation.")
        super().__init__(policy, *args, rnd_cfg=None, symmetry_cfg=None, **kwargs)
        self.transition = RolloutStorageDualHead.Transition()

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = RolloutStorageDualHead(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        self.transition.actions = self.policy.act(obs).detach()
        left_values, right_values = self.policy.evaluate_heads(obs)
        self.transition.left_values = left_values.detach()
        self.transition.right_values = right_values.detach()
        self.transition.values = (left_values + right_values).detach()

        left_log_prob, right_log_prob = self.policy.get_actions_log_prob_heads(self.transition.actions)
        self.transition.left_actions_log_prob = left_log_prob.detach()
        self.transition.right_actions_log_prob = right_log_prob.detach()
        self.transition.actions_log_prob = (left_log_prob + right_log_prob).detach()

        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        self.policy.update_normalization(obs)

        if "reward_left" not in extras or "reward_right" not in extras:
            raise ValueError("PPODualHead expects 'reward_left' and 'reward_right' in env extras.")

        reward_left = extras["reward_left"].to(self.device).clone()
        reward_right = extras["reward_right"].to(self.device).clone()
        reward_shared = extras.get("reward_shared", None)
        if reward_shared is None:
            reward_shared = torch.zeros_like(reward_left)
        else:
            reward_shared = reward_shared.to(self.device).clone()
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        residual = torch.max(torch.abs((reward_left + reward_right - reward_shared) - rewards))
        if residual.item() > 1.0:
            raise RuntimeError(
                "Left/right reward split does not match the environment reward. "
                f"Max residual: {residual.item():.6e}"
            )

        self.transition.left_rewards = reward_left
        self.transition.right_rewards = reward_right
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            time_outs = extras["time_outs"].unsqueeze(1).to(self.device)
            reward_left = reward_left + self.gamma * torch.squeeze(self.transition.left_values * time_outs, 1)
            reward_right = reward_right + self.gamma * torch.squeeze(self.transition.right_values * time_outs, 1)
            self.transition.left_rewards = reward_left
            self.transition.right_rewards = reward_right
            self.transition.rewards = reward_left + reward_right - reward_shared

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        last_left_values, last_right_values = self.policy.evaluate_heads(obs, masks="compute_returns")
        self.storage.compute_returns(
            last_left_values.detach(),
            last_right_values.detach(),
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self):
        if self.policy.is_recurrent or getattr(self.policy, "is_sequence_model", False):
            raise NotImplementedError("PPODualHead currently supports only feed-forward policies.")

        if (
            self.is_multi_gpu
            and not self.per_task_obs_normalizer
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            self._sync_empirical_normalization_buffers()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_left_value_loss = 0.0
        mean_right_value_loss = 0.0
        mean_left_surrogate_loss = 0.0
        mean_right_surrogate_loss = 0.0
        mean_left_value = 0.0
        mean_right_value = 0.0
        mean_left_return = 0.0
        mean_right_return = 0.0
        dbg_kl_mean = 0.0

        rollout_left_reward_mean = self.storage.left_rewards.mean().item()
        rollout_right_reward_mean = self.storage.right_rewards.mean().item()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for update_steps, (
            obs_batch,
            actions_batch,
            target_left_values_batch,
            target_right_values_batch,
            left_advantages_batch,
            right_advantages_batch,
            left_returns_batch,
            right_returns_batch,
            old_left_actions_log_prob_batch,
            old_right_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _hid_states_batch,
            _masks_batch,
        ) in enumerate(generator):
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    left_advantages_batch = (left_advantages_batch - left_advantages_batch.mean()) / (
                        left_advantages_batch.std() + 1e-8
                    )
                    right_advantages_batch = (right_advantages_batch - right_advantages_batch.mean()) / (
                        right_advantages_batch.std() + 1e-8
                    )

            self.policy.act(obs_batch)
            left_actions_log_prob_batch, right_actions_log_prob_batch = self.policy.get_actions_log_prob_heads(
                actions_batch
            )
            left_value_batch, right_value_batch = self.policy.evaluate_heads(obs_batch)

            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            with torch.inference_mode():
                kl_dbg = torch.sum(
                    torch.log((sigma_batch + 1.0e-8) / (old_sigma_batch[:original_batch_size] + 1.0e-8))
                    + (
                        torch.square(old_sigma_batch[:original_batch_size])
                        + torch.square(old_mu_batch[:original_batch_size] - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch) + 1.0e-8)
                    - 0.5,
                    axis=-1,
                )
                dbg_kl_mean += torch.mean(kl_dbg).item()

            left_ratio = torch.exp(left_actions_log_prob_batch - torch.squeeze(old_left_actions_log_prob_batch))
            right_ratio = torch.exp(right_actions_log_prob_batch - torch.squeeze(old_right_actions_log_prob_batch))

            left_surrogate = -torch.squeeze(left_advantages_batch) * left_ratio
            left_surrogate_clipped = -torch.squeeze(left_advantages_batch) * torch.clamp(
                left_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            left_surrogate_loss = torch.max(left_surrogate, left_surrogate_clipped).mean()

            right_surrogate = -torch.squeeze(right_advantages_batch) * right_ratio
            right_surrogate_clipped = -torch.squeeze(right_advantages_batch) * torch.clamp(
                right_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            right_surrogate_loss = torch.max(right_surrogate, right_surrogate_clipped).mean()
            surrogate_loss = 0.5 * (left_surrogate_loss + right_surrogate_loss)

            if self.use_clipped_value_loss:
                left_value_clipped = target_left_values_batch + (left_value_batch - target_left_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                left_value_losses = (left_value_batch - left_returns_batch).pow(2)
                left_value_losses_clipped = (left_value_clipped - left_returns_batch).pow(2)
                left_value_loss = torch.max(left_value_losses, left_value_losses_clipped).mean()

                right_value_clipped = target_right_values_batch + (
                    right_value_batch - target_right_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                right_value_losses = (right_value_batch - right_returns_batch).pow(2)
                right_value_losses_clipped = (right_value_clipped - right_returns_batch).pow(2)
                right_value_loss = torch.max(right_value_losses, right_value_losses_clipped).mean()
            else:
                left_value_loss = (left_returns_batch - left_value_batch).pow(2).mean()
                right_value_loss = (right_returns_batch - right_value_batch).pow(2).mean()

            value_loss = 0.5 * (left_value_loss + right_value_loss)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_left_value_loss += left_value_loss.item()
            mean_right_value_loss += right_value_loss.item()
            mean_left_surrogate_loss += left_surrogate_loss.item()
            mean_right_surrogate_loss += right_surrogate_loss.item()
            mean_left_value += left_value_batch.mean().item()
            mean_right_value += right_value_batch.mean().item()
            mean_left_return += left_returns_batch.mean().item()
            mean_right_return += right_returns_batch.mean().item()

        num_updates = max(update_steps, 1)
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_left_value_loss /= num_updates
        mean_right_value_loss /= num_updates
        mean_left_surrogate_loss /= num_updates
        mean_right_surrogate_loss /= num_updates
        mean_left_value /= num_updates
        mean_right_value /= num_updates
        mean_left_return /= num_updates
        mean_right_return /= num_updates

        self.storage.clear()

        if self.scheduler is not None:
            self.scheduler.step()
            self.learning_rate = self.optimizer.param_groups[0]["lr"]

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "value_function_left": mean_left_value_loss,
            "value_function_right": mean_right_value_loss,
            "surrogate_left": mean_left_surrogate_loss,
            "surrogate_right": mean_right_surrogate_loss,
            "value_left_mean": mean_left_value,
            "value_right_mean": mean_right_value,
            "returns_left_mean": mean_left_return,
            "returns_right_mean": mean_right_return,
            "reward_left_mean": rollout_left_reward_mean,
            "reward_right_mean": rollout_right_reward_mean,
        }
        if update_steps > 0:
            loss_dict["debug/kl"] = dbg_kl_mean / update_steps

        return loss_dict
