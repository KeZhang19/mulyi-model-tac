# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type in ("distillation", "hybrid"):
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type in ("rl", "hybrid"):
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type in ("distillation", "hybrid"):
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type in ("rl", "hybrid"):
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        if hidden_states[1] is None:
            hid_c = ()
        else:
            hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
        for i in range(len(hid_c)):
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for distillation
    def generator(self):
        if self.training_type not in ("distillation", "hybrid"):
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type not in ("rl", "hybrid"):
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # yield the mini-batch
                yield obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type not in ("rl", "hybrid"):
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj

    def sequence_batch_generator(self, num_epochs=1, num_mini_batches=1):
        """Return full rollout sequences without flattening the time dimension.

        This is intended for feedforward sequence models (e.g. Transformer
        backbones without explicit hidden-state cache) that want to see the
        entire [T, num_envs, ...] rollout at once during PPO updates.

        Args:
            num_epochs (int): Number of epochs to repeat the same data.
                             Default is 1 (single pass). Set to > 1 to reuse
                             the rollout data multiple times (similar to MLP training).
            num_mini_batches (int): Number of mini-batches to split the environments.
                                   Default is 1 (all environments in one batch).
                                   When > 1, environments are split along dim=1,
                                   preserving the time sequence structure (dim=0).

        Note:
            - Mini-batching is performed along the environment dimension (dim=1),
              preserving the time sequence structure (dim=0).
            - Hidden states are left as (None, None) and masks as None,
              since these models are not recurrent.
            - The same data is yielded num_epochs times to allow multiple
              gradient updates on the same rollout (improves data utilization).
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if self.num_envs == 0:
            return

        # Prepare data (shape: [T, num_envs, ...])
        obs_batch = self.observations
        actions_batch = self.actions
        old_mu_batch = self.mu
        old_sigma_batch = self.sigma
        returns_batch = self.returns
        advantages_batch = self.advantages
        values_batch = self.values
        old_actions_log_prob_batch = self.actions_log_prob

        # Calculate mini-batch size along environment dimension
        num_envs_per_batch = self.num_envs // num_mini_batches
        if num_envs_per_batch == 0:
            raise ValueError(
                f"num_mini_batches ({num_mini_batches}) cannot be larger than num_envs ({self.num_envs})"
            )

        # Repeat the same data for num_epochs times
        for epoch in range(num_epochs):
            # Split environments into mini-batches
            for i in range(num_mini_batches):
                start_env = i * num_envs_per_batch
                end_env = (i + 1) * num_envs_per_batch
                
                # Handle the last batch if num_envs is not divisible by num_mini_batches
                if i == num_mini_batches - 1:
                    end_env = self.num_envs

                # Slice along environment dimension (dim=1), preserving time dimension (dim=0)
                obs_mini = obs_batch[:, start_env:end_env]
                actions_mini = actions_batch[:, start_env:end_env]
                old_mu_mini = old_mu_batch[:, start_env:end_env]
                old_sigma_mini = old_sigma_batch[:, start_env:end_env]
                returns_mini = returns_batch[:, start_env:end_env]
                advantages_mini = advantages_batch[:, start_env:end_env]
                values_mini = values_batch[:, start_env:end_env]
                old_actions_log_prob_mini = old_actions_log_prob_batch[:, start_env:end_env]

                yield (
                    obs_mini,
                    actions_mini,
                    values_mini,
                    advantages_mini,
                    returns_mini,
                    old_actions_log_prob_mini,
                    old_mu_mini,
                    old_sigma_mini,
                    (None, None),
                    None,
                )

    def transformerxl_batch_generator(self, num_epochs=1, num_mini_batches=1):
        """Generator for Transformer-XL with support for multiple epochs and mini-batching.
        
        Args:
            num_epochs (int): Number of epochs to repeat the same data. Default is 1.
            num_mini_batches (int): Number of mini-batches to split environments. Default is 1.
        
        Note:
            - We keep the rollout structure intact (one sequence per environment) and rely on the dones buffer
              to expose episode boundaries. This yields data shaped like:
              [[a1, a2, a3, a4 | a5, a6],
               [b1, b2 | b3, b4, b5 | b6]]
            - Mini-batching is performed along the environment dimension (dim=1),
              preserving the time sequence structure (dim=0).
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if self.num_envs == 0:
            return

        # Prepare data (shape: [T, num_envs, ...])
        obs_batch = self.observations
        actions_batch = self.actions
        old_mu_batch = self.mu
        old_sigma_batch = self.sigma
        returns_batch = self.returns
        advantages_batch = self.advantages
        values_batch = self.values
        old_actions_log_prob_batch = self.actions_log_prob

        # Calculate mini-batch size along environment dimension
        num_envs_per_batch = self.num_envs // num_mini_batches
        if num_mini_batches > 1 and num_envs_per_batch == 0:
            raise ValueError(
                f"num_mini_batches ({num_mini_batches}) cannot be larger than num_envs ({self.num_envs})"
            )

        # Repeat the same data for num_epochs times
        for epoch in range(num_epochs):
            if num_mini_batches == 1:
                # Single batch - yield all environments at once
                yield (
                    obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (None, None),
                    None,
                )
            else:
                # Split environments into mini-batches
                for i in range(num_mini_batches):
                    start_env = i * num_envs_per_batch
                    end_env = (i + 1) * num_envs_per_batch
                    
                    # Handle the last batch if num_envs is not divisible by num_mini_batches
                    if i == num_mini_batches - 1:
                        end_env = self.num_envs

                    # Slice along environment dimension (dim=1), preserving time dimension (dim=0)
                    obs_mini = obs_batch[:, start_env:end_env]
                    actions_mini = actions_batch[:, start_env:end_env]
                    old_mu_mini = old_mu_batch[:, start_env:end_env]
                    old_sigma_mini = old_sigma_batch[:, start_env:end_env]
                    returns_mini = returns_batch[:, start_env:end_env]
                    advantages_mini = advantages_batch[:, start_env:end_env]
                    values_mini = values_batch[:, start_env:end_env]
                    old_actions_log_prob_mini = old_actions_log_prob_batch[:, start_env:end_env]

                    yield (
                        obs_mini,
                        actions_mini,
                        values_mini,
                        advantages_mini,
                        returns_mini,
                        old_actions_log_prob_mini,
                        old_mu_mini,
                        old_sigma_mini,
                        (None, None),
                        None,
                    )
