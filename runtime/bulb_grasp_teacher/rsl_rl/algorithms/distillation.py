# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import random
from collections import deque

import torch
import torch.nn as nn

from rsl_rl.modules import (
    StudentTeacher,
    StudentTeacherDiffusionPolicy,
    StudentTeacherRecurrent,
)
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class DistillationSequenceReplay:
    """CPU sequence replay stratified by rollout teacher-execution fraction."""

    def __init__(self, capacity_chunks: int = 128, depth_group: str = "depth_image"):
        self.num_buckets = 4
        per_bucket = max(1, int(math.ceil(capacity_chunks / self.num_buckets)))
        self.buckets = [deque(maxlen=per_bucket) for _ in range(self.num_buckets)]
        self.depth_group = depth_group
        self.total_env_sequences = 0

    @staticmethod
    def bucket_index(teacher_fraction: float) -> int:
        return min(3, max(0, int(float(teacher_fraction) * 4.0)))

    def add(self, observations: dict[str, torch.Tensor], actions: torch.Tensor, dones: torch.Tensor, bucket: int):
        cpu_observations = {}
        for key, value in observations.items():
            dtype = torch.float16 if key == self.depth_group else value.dtype
            cpu_observations[key] = value.detach().to(device="cpu", dtype=dtype).contiguous()
        chunk = {
            "observations": cpu_observations,
            "actions": actions.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            "dones": dones.detach().to(device="cpu", dtype=torch.bool).contiguous(),
        }
        self.buckets[int(bucket)].append(chunk)
        self.total_env_sequences += int(actions.shape[1])

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self.buckets)

    def bucket_sizes(self) -> list[int]:
        return [len(bucket) for bucket in self.buckets]

    def sample(self, num_envs: int, device: str | torch.device):
        nonempty = [index for index, bucket in enumerate(self.buckets) if bucket]
        if not nonempty or num_envs < 1:
            return None
        selected: list[tuple[dict, int]] = []
        base = num_envs // len(nonempty)
        remainder = num_envs % len(nonempty)
        for offset, bucket_index in enumerate(nonempty):
            count = base + int(offset < remainder)
            bucket = self.buckets[bucket_index]
            for _ in range(count):
                chunk = random.choice(tuple(bucket))
                env_index = random.randrange(int(chunk["actions"].shape[1]))
                selected.append((chunk, env_index))
        random.shuffle(selected)
        observation_keys = selected[0][0]["observations"].keys()
        observations = {
            key: torch.stack([chunk["observations"][key][:, env_index] for chunk, env_index in selected], dim=1)
            .to(device=device, dtype=torch.float32)
            for key in observation_keys
        }
        actions = torch.stack([chunk["actions"][:, env_index] for chunk, env_index in selected], dim=1).to(device)
        dones = torch.stack([chunk["dones"][:, env_index] for chunk, env_index in selected], dim=1).to(device)
        return observations, actions, dones


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: (
        StudentTeacher
        | StudentTeacherRecurrent
        | StudentTeacherDiffusionPolicy
    )
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        update_batch_size: int | None = None,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        optimizer="adam",
        replay_capacity_chunks: int = 0,
        replay_envs_per_rollout: int = 32,
        replay_sample_envs: int = 96,
        replay_num_learning_epochs: int = 3,
        tbptt_steps: int = 16,
        recurrent_minibatch_size: int = 16,
        diffusion_sample_minibatch_size: int = 128,
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later

        # initialize the optimizer
        optimizer_parameters = (
            self.policy.optimizer_parameters()
            if hasattr(self.policy, "optimizer_parameters")
            else self.policy.parameters()
        )
        self.optimizer = resolve_optimizer(optimizer)(optimizer_parameters, lr=learning_rate)

        # initialize the transition
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.update_batch_size = update_batch_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.replay_envs_per_rollout = max(1, int(replay_envs_per_rollout))
        self.replay_sample_envs = max(1, int(replay_sample_envs))
        self.replay_num_learning_epochs = max(1, int(replay_num_learning_epochs))
        self.tbptt_steps = max(1, int(tbptt_steps))
        self.recurrent_minibatch_size = max(1, int(recurrent_minibatch_size))
        self.diffusion_sample_minibatch_size = max(1, int(diffusion_sample_minibatch_size))
        self.sequence_replay = (
            DistillationSequenceReplay(
                capacity_chunks=int(replay_capacity_chunks),
                depth_group=getattr(policy, "student_depth_obs_group", "depth_image"),
            )
            if int(replay_capacity_chunks) > 0
            else None
        )

        # initialize the loss function
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        self.num_updates = 0
        self.last_student_grad_norm = 0.0
        self.last_depth_encoder_grad_norm = 0.0
        self._teacher_execution_masks: list[torch.Tensor] = []

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs, student_mask: torch.Tensor | None = None):
        # compute the actions
        if getattr(self.policy, "is_recurrent", False):
            if hasattr(self.policy, "initialize_hidden_state"):
                self.policy.initialize_hidden_state(obs)
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        if hasattr(self.policy, "observe_teacher_actions"):
            self.policy.observe_teacher_actions(self.transition.privileged_actions)
        if student_mask is not None and getattr(self.policy, "supports_masked_rollout", False):
            self.transition.actions = self.policy.act_masked(
                obs,
                student_mask,
                self.transition.privileged_actions,
            ).detach()
        else:
            self.transition.actions = self.policy.act(obs).detach()
        # record the observations
        self.transition.observations = obs
        return self.transition.actions

    def record_teacher_execution_mask(self, mask: torch.Tensor):
        if not getattr(self.policy, "uses_execution_mask", False):
            return
        self._teacher_execution_masks.append(mask.detach().to(device=self.device, dtype=torch.bool).clone())

    def discard_rollout(self):
        self.storage.clear()
        self._teacher_execution_masks.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

    def process_env_step(self, obs, rewards, dones, extras):
        # update the normalizers
        self.policy.update_normalization(obs)

        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        if getattr(self.policy, "uses_recurrent_diffusion", False):
            return self._update_recurrent_diffusion()
        if getattr(self.policy, "uses_tbptt_distillation", False):
            return self._update_tbptt_behavior_cloning()
        if getattr(self.policy, "uses_diffusion_policy", False):
            return self._update_diffusion_policy()
        self._archive_rollout_for_replay()
        loss_dict = self._update_behavior_cloning()
        replay_loss = self._update_behavior_cloning_replay()
        loss_dict["current_behavior"] = loss_dict["behavior"]
        loss_dict["replay_behavior"] = replay_loss
        if replay_loss is not None:
            loss_dict["behavior"] = 0.75 * loss_dict["current_behavior"] + 0.25 * replay_loss
        if self.sequence_replay is not None:
            loss_dict["replay_chunks"] = float(len(self.sequence_replay))
            loss_dict["replay_env_sequences_seen"] = float(self.sequence_replay.total_env_sequences)
            for index, size in enumerate(self.sequence_replay.bucket_sizes()):
                loss_dict[f"replay_bucket_{index}_size"] = float(size)
        return loss_dict

    def _update_tbptt_behavior_cloning(self):
        """Optimize recurrent visual students over truncated rollout sequences.

        ``gradient_length`` follows the baseline DAgger semantics: sum gradients
        from that many TBPTT mini-batches, then make one optimizer step.
        """

        num_steps = int(self.storage.step)
        num_envs = int(self.storage.num_envs)
        if num_steps < 1:
            return {"behavior": 0.0, "action": 0.0, "auxiliary_object_position": 0.0}

        behavior_sum = 0.0
        action_sum = 0.0
        auxiliary_sum = 0.0
        update_count = 0
        accumulated_batches = 0
        self.optimizer.zero_grad()
        for _ in range(self.num_learning_epochs):
            env_order = torch.randperm(num_envs, device=self.device)
            for sequence_start in range(0, num_steps, self.tbptt_steps):
                sequence_end = min(sequence_start + self.tbptt_steps, num_steps)
                for env_start in range(0, num_envs, self.recurrent_minibatch_size):
                    env_ids = env_order[env_start : env_start + self.recurrent_minibatch_size]
                    observations = {
                        key: value[sequence_start:sequence_end, env_ids]
                        for key, value in self.storage.observations.items()
                    }
                    teacher_actions = self.storage.privileged_actions[
                        sequence_start:sequence_end, env_ids
                    ]
                    dones = self.storage.dones[sequence_start:sequence_end, env_ids]
                    initial_hidden = self._tbptt_initial_hidden(sequence_start, env_ids)
                    behavior_loss, action_loss, auxiliary_loss = self.policy.compute_sequence_loss(
                        observations,
                        teacher_actions,
                        dones,
                        initial_hidden=initial_hidden,
                    )
                    # This is intentionally an unnormalised sum, matching the
                    # original BrainCo1 Distillation.update implementation.
                    behavior_loss.backward()
                    accumulated_batches += 1
                    if accumulated_batches == self.gradient_length:
                        self._optimizer_step_from_accumulated_gradients()
                        self.policy.detach_hidden_states()
                        accumulated_batches = 0
                    behavior_sum += float(behavior_loss.detach().item())
                    action_sum += float(action_loss.item())
                    auxiliary_sum += float(auxiliary_loss.item())
                    update_count += 1

        if accumulated_batches:
            self._optimizer_step_from_accumulated_gradients()
            self.policy.detach_hidden_states()
        self.storage.clear()
        self._teacher_execution_masks.clear()
        self.policy.detach_hidden_states()
        divisor = max(update_count, 1)
        return {
            "behavior": behavior_sum / divisor,
            "action": action_sum / divisor,
            "auxiliary_object_position": auxiliary_sum / divisor,
        }

    def _update_recurrent_diffusion(self):
        """Encode each TBPTT frame once, then supervise recurrent diffusion windows."""

        num_steps = int(self.storage.step)
        num_envs = int(self.storage.num_envs)
        if num_steps < 1:
            return {
                "behavior": 0.0,
                "diffusion_epsilon": 0.0,
                "diffusion_x0": 0.0,
                "auxiliary_object_position": 0.0,
                "diffusion_windows": 0.0,
            }
        if len(self._teacher_execution_masks) != num_steps:
            raise RuntimeError(
                f"Expected {num_steps} V6 execution masks, got {len(self._teacher_execution_masks)}."
            )
        execution_masks = torch.stack(self._teacher_execution_masks, dim=0)
        behavior_sum = 0.0
        epsilon_sum = 0.0
        x0_sum = 0.0
        auxiliary_sum = 0.0
        window_count = 0
        update_count = 0

        for _ in range(self.num_learning_epochs):
            env_order = torch.randperm(num_envs, device=self.device)
            for sequence_start in range(0, num_steps, self.tbptt_steps):
                sequence_end = min(sequence_start + self.tbptt_steps, num_steps)
                for env_start in range(0, num_envs, self.recurrent_minibatch_size):
                    env_ids = env_order[env_start : env_start + self.recurrent_minibatch_size]
                    observations = {
                        key: value[sequence_start:sequence_end, env_ids]
                        for key, value in self.storage.observations.items()
                    }
                    teacher_actions = self.storage.privileged_actions[
                        sequence_start:sequence_end, env_ids
                    ]
                    dones = self.storage.dones[sequence_start:sequence_end, env_ids]
                    masks = execution_masks[sequence_start:sequence_end, env_ids]
                    initial_hidden = self._tbptt_initial_hidden(sequence_start, env_ids)
                    behavior_loss, epsilon_loss, x0_loss, auxiliary_loss, windows = (
                        self.policy.compute_recurrent_diffusion_loss(
                            observations,
                            teacher_actions,
                            dones,
                            masks,
                            initial_hidden=initial_hidden,
                            diffusion_sample_minibatch_size=self.diffusion_sample_minibatch_size,
                        )
                    )
                    self._optimizer_step(behavior_loss)
                    behavior_sum += float(behavior_loss.detach().item())
                    epsilon_sum += float(epsilon_loss.item())
                    x0_sum += float(x0_loss.item())
                    auxiliary_sum += float(auxiliary_loss.item())
                    window_count += int(windows)
                    update_count += 1

        self.storage.clear()
        self._teacher_execution_masks.clear()
        self.policy.detach_hidden_states()
        divisor = max(update_count, 1)
        return {
            "behavior": behavior_sum / divisor,
            "diffusion_epsilon": epsilon_sum / divisor,
            "diffusion_x0": x0_sum / divisor,
            "auxiliary_object_position": auxiliary_sum / divisor,
            "diffusion_windows": float(window_count),
        }

    def _tbptt_initial_hidden(self, sequence_start: int, env_ids: torch.Tensor):
        saved = self.storage.saved_hidden_states_a
        if saved is not None and len(saved) >= 2:
            return tuple(
                state[sequence_start, :, env_ids, :].detach().contiguous()
                for state in saved[:2]
            )
        batch_size = int(env_ids.numel())
        return self.policy.student.initial_hidden(batch_size, self.device)

    def _compute_behavior_loss(self, obs, privileged_actions: torch.Tensor) -> torch.Tensor:
        if hasattr(self.policy, "compute_behavior_loss"):
            return self.policy.compute_behavior_loss(obs, privileged_actions, self.loss_fn)
        actions = self.policy.act_inference(obs)
        return self.loss_fn(actions, privileged_actions)

    def _archive_rollout_for_replay(self):
        replay = self.sequence_replay
        if replay is None or self.storage.step < 1:
            return
        num_envs = int(self.storage.num_envs)
        sample_count = min(self.replay_envs_per_rollout, num_envs)
        execution_masks = None
        if len(self._teacher_execution_masks) == self.storage.step:
            execution_masks = torch.stack(self._teacher_execution_masks, dim=0).float()
            teacher_fraction = execution_masks.mean(dim=0)
        else:
            teacher_fraction = torch.ones(num_envs, device=self.device)

        selected_ids: list[torch.Tensor] = []
        per_bucket = max(1, sample_count // 4)
        used = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        for bucket in range(4):
            lower, upper = bucket / 4.0, (bucket + 1) / 4.0
            candidates = ((teacher_fraction >= lower) & (teacher_fraction <= upper if bucket == 3 else teacher_fraction < upper)).nonzero(as_tuple=False).reshape(-1)
            if candidates.numel() == 0:
                continue
            take = min(per_bucket, int(candidates.numel()))
            ids = candidates[torch.randperm(candidates.numel(), device=candidates.device)[:take]]
            selected_ids.append(ids)
            used[ids] = True
        chosen = torch.cat(selected_ids) if selected_ids else torch.empty(0, dtype=torch.long, device=self.device)
        if chosen.numel() < sample_count:
            remaining = (~used).nonzero(as_tuple=False).reshape(-1)
            take = min(sample_count - int(chosen.numel()), int(remaining.numel()))
            if take > 0:
                chosen = torch.cat((chosen, remaining[torch.randperm(remaining.numel(), device=remaining.device)[:take]]))

        steps = int(self.storage.step)
        obs_groups = getattr(self.policy, "replay_obs_groups", list(self.storage.observations.keys()))
        for bucket in range(4):
            bucket_ids = chosen[
                torch.tensor(
                    [self.sequence_replay.bucket_index(float(teacher_fraction[index].item())) == bucket for index in chosen],
                    dtype=torch.bool,
                    device=chosen.device,
                )
            ]
            if bucket_ids.numel() == 0:
                continue
            observations = {key: self.storage.observations[key][:steps, bucket_ids] for key in obs_groups}
            replay.add(
                observations,
                self.storage.privileged_actions[:steps, bucket_ids],
                self.storage.dones[:steps, bucket_ids],
                bucket,
            )

    def _update_behavior_cloning_replay(self) -> float | None:
        if self.sequence_replay is None or len(self.sequence_replay) == 0:
            return None
        batch = self.sequence_replay.sample(self.replay_sample_envs, self.device)
        if batch is None:
            return None
        observations, privileged_actions, dones = batch
        mean_loss = 0.0
        count = 0
        loss = 0
        for _ in range(self.replay_num_learning_epochs):
            self.policy.reset(hidden_states=None)
            self.policy.detach_hidden_states()
            for step in range(int(privileged_actions.shape[0])):
                obs = {key: value[step] for key, value in observations.items()}
                behavior_loss = self._compute_behavior_loss(obs, privileged_actions[step])
                loss = loss + behavior_loss
                mean_loss += float(behavior_loss.item())
                count += 1
                if count % self.gradient_length == 0:
                    self._optimizer_step(loss)
                    loss = 0
                self.policy.reset(dones[step].reshape(-1))
                self.policy.detach_hidden_states(dones[step].reshape(-1))
        if torch.is_tensor(loss):
            self._optimizer_step(loss)
        self.policy.reset(hidden_states=None)
        return mean_loss / max(count, 1)

    def update_fixed_observation(
        self,
        obs,
        privileged_actions: torch.Tensor,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Train the student on one frozen observation/action batch."""

        self.num_updates += 1
        num_obs = int(privileged_actions.shape[0])
        if num_obs < 1:
            raise ValueError("Fixed-observation distillation requires at least one observation.")
        batch_size = num_obs if batch_size is None else max(1, min(int(batch_size), num_obs))

        self.policy.reset(hidden_states=self.last_hidden_states)
        self.policy.detach_hidden_states()
        self.policy.train()

        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(num_obs, device=privileged_actions.device)
            for start in range(0, num_obs, batch_size):
                batch_ids = indices[start : start + batch_size]
                obs_batch = self._index_observations(obs, batch_ids)
                action_targets = privileged_actions[batch_ids]

                actions = self.policy.act_inference(obs_batch)
                behavior_loss = nn.functional.mse_loss(actions, action_targets)

                self._optimizer_step(behavior_loss)
                self.policy.detach_hidden_states()

        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()
        return self._compute_fixed_observation_metrics(obs, privileged_actions, batch_size)

    def _update_behavior_cloning(self):
        if self.update_batch_size is not None:
            return self._update_behavior_cloning_mini_batch()

        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():

                # inference the student for gradient computation
                # behavior cloning loss
                behavior_loss = self._compute_behavior_loss(obs, privileged_actions)

                # total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    self._capture_and_clip_grad_norm()
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self._teacher_execution_masks.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict

    def _update_behavior_cloning_mini_batch(self):
        if getattr(self.policy, "is_recurrent", False):
            raise ValueError("Mini-batch distillation update is only supported for feed-forward policies.")

        mean_behavior_loss = 0.0
        cnt = 0
        update_batch_size = max(1, int(self.update_batch_size))
        self.optimizer.zero_grad()

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                batch_size = int(privileged_actions.shape[0])
                mini_batch_size = min(update_batch_size, batch_size)
                indices = torch.randperm(batch_size, requires_grad=False, device=privileged_actions.device)
                timestep_loss = 0.0

                for start in range(0, batch_size, mini_batch_size):
                    batch_ids = indices[start : start + mini_batch_size]
                    obs_batch = self._index_observations(obs, batch_ids)
                    action_targets = privileged_actions[batch_ids]

                    behavior_loss = self._compute_behavior_loss(obs_batch, action_targets)
                    scaled_loss = behavior_loss * (int(action_targets.shape[0]) / batch_size)
                    scaled_loss.backward()
                    timestep_loss += scaled_loss.detach().item()

                mean_behavior_loss += timestep_loss
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self._optimizer_step_from_accumulated_gradients()
                    self.policy.detach_hidden_states()

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        if cnt == 0:
            self.storage.clear()
            self._teacher_execution_masks.clear()
            self.last_hidden_states = self.policy.get_hidden_states()
            self.policy.detach_hidden_states()
            return {"behavior": 0.0}

        if cnt % self.gradient_length != 0:
            self._optimizer_step_from_accumulated_gradients()
            self.policy.detach_hidden_states()

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {"behavior": mean_behavior_loss}

    def _update_diffusion_policy(self):
        if self.update_batch_size is not None:
            return self._update_diffusion_policy_mini_batch()

        mean_behavior_loss = 0.0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            if not getattr(self.policy, "preserve_rollout_history_on_update", False):
                self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs_sequence, privileged_action_sequence, execution_mask_sequence in self._diffusion_sequence_generator():
                behavior_loss = self._compute_diffusion_loss(
                    obs_sequence, privileged_action_sequence, execution_mask_sequence
                )

                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self._optimizer_step(loss)
                    self.policy.detach_hidden_states()
                    loss = 0

        if cnt == 0:
            self.storage.clear()
            self._teacher_execution_masks.clear()
            self.last_hidden_states = self.policy.get_hidden_states()
            self.policy.detach_hidden_states()
            return {"behavior": 0.0}

        if torch.is_tensor(loss):
            self._optimizer_step(loss)
            self.policy.detach_hidden_states()

        mean_behavior_loss /= cnt
        self.storage.clear()
        self._teacher_execution_masks.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {"behavior": mean_behavior_loss}

    def _update_diffusion_policy_mini_batch(self):
        mean_behavior_loss = 0.0
        cnt = 0
        update_batch_size = max(1, int(self.update_batch_size))
        self.optimizer.zero_grad()

        for epoch in range(self.num_learning_epochs):
            if not getattr(self.policy, "preserve_rollout_history_on_update", False):
                self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs_sequence, privileged_action_sequence, execution_mask_sequence in self._diffusion_sequence_generator():
                batch_size = int(privileged_action_sequence.shape[0])
                mini_batch_size = min(update_batch_size, batch_size)
                indices = torch.randperm(
                    batch_size,
                    requires_grad=False,
                    device=privileged_action_sequence.device,
                )
                sequence_loss = 0.0

                for start in range(0, batch_size, mini_batch_size):
                    batch_ids = indices[start : start + mini_batch_size]
                    obs_batch = self._index_observations(obs_sequence, batch_ids)
                    action_targets = privileged_action_sequence[batch_ids]
                    execution_mask_batch = (
                        execution_mask_sequence[batch_ids] if execution_mask_sequence is not None else None
                    )

                    behavior_loss = self._compute_diffusion_loss(
                        obs_batch, action_targets, execution_mask_batch
                    )
                    scaled_loss = behavior_loss * (int(action_targets.shape[0]) / batch_size)
                    scaled_loss.backward()
                    sequence_loss += scaled_loss.detach().item()

                    del obs_batch, action_targets, behavior_loss, scaled_loss

                mean_behavior_loss += sequence_loss
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self._optimizer_step_from_accumulated_gradients()
                    self.policy.detach_hidden_states()

        if cnt == 0:
            self.storage.clear()
            self._teacher_execution_masks.clear()
            self.last_hidden_states = self.policy.get_hidden_states()
            self.policy.detach_hidden_states()
            return {"behavior": 0.0}

        if cnt % self.gradient_length != 0:
            self._optimizer_step_from_accumulated_gradients()
            self.policy.detach_hidden_states()

        mean_behavior_loss /= cnt
        self.storage.clear()
        self._teacher_execution_masks.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {"behavior": mean_behavior_loss}

    def _optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._optimizer_step_from_accumulated_gradients()

    def _optimizer_step_from_accumulated_gradients(self):
        if self.is_multi_gpu:
            self.reduce_parameters()
        self._capture_and_clip_grad_norm()
        self.optimizer.step()
        if hasattr(self.policy, "update_ema"):
            self.policy.update_ema()
        self.optimizer.zero_grad()

    def _compute_diffusion_loss(self, obs_sequence, action_sequence, execution_mask_sequence):
        if getattr(self.policy, "uses_execution_mask", False):
            return self.policy.compute_distillation_loss(
                obs_sequence,
                action_sequence,
                execution_mask_sequence=execution_mask_sequence,
            )
        return self.policy.compute_distillation_loss(obs_sequence, action_sequence)

    def _capture_and_clip_grad_norm(self):
        depth_encoder = getattr(getattr(self.policy, "student", None), "depth_encoder", None)
        if depth_encoder is not None:
            grad_sq = sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in depth_encoder.parameters()
                if parameter.grad is not None
            )
            self.last_depth_encoder_grad_norm = grad_sq**0.5
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.student.parameters(), self.max_grad_norm if self.max_grad_norm else float("inf")
        )
        self.last_student_grad_norm = float(grad_norm.item())

    @staticmethod
    def _index_observations(obs, indices: torch.Tensor):
        if isinstance(obs, dict):
            return {key: Distillation._index_observations(value, indices) for key, value in obs.items()}
        return obs[indices]

    def _compute_fixed_observation_metrics(
        self,
        obs,
        privileged_actions: torch.Tensor,
        batch_size: int,
    ) -> dict[str, float]:
        num_obs = int(privileged_actions.shape[0])
        numel = int(privileged_actions.numel())
        sum_sq_error = 0.0
        sum_abs_error = 0.0
        max_abs_error = 0.0
        sum_teacher_abs = 0.0
        sum_student_abs = 0.0

        with torch.no_grad():
            for start in range(0, num_obs, batch_size):
                end = min(start + batch_size, num_obs)
                batch_ids = torch.arange(start, end, device=privileged_actions.device)
                obs_batch = self._index_observations(obs, batch_ids)
                action_targets = privileged_actions[batch_ids]
                actions = self.policy.act_inference(obs_batch)
                abs_error = torch.abs(actions - action_targets)

                sum_sq_error += torch.sum((actions - action_targets) ** 2).item()
                sum_abs_error += torch.sum(abs_error).item()
                max_abs_error = max(max_abs_error, abs_error.max().item())
                sum_teacher_abs += torch.sum(torch.abs(action_targets)).item()
                sum_student_abs += torch.sum(torch.abs(actions)).item()

        return {
            "mse": sum_sq_error / numel,
            "mae": sum_abs_error / numel,
            "max_abs_error": max_abs_error,
            "teacher_action_abs_mean": sum_teacher_abs / numel,
            "student_action_abs_mean": sum_student_abs / numel,
        }

    def _diffusion_sequence_generator(self):
        horizon = int(self.policy.horizon)
        num_steps = self.storage.step
        if num_steps < horizon:
            return

        dones = self.storage.dones[:num_steps].bool()
        observations = self.storage.observations
        privileged_actions = self.storage.privileged_actions
        execution_masks = None
        if getattr(self.policy, "uses_execution_mask", False):
            if len(self._teacher_execution_masks) != num_steps:
                raise RuntimeError(
                    f"Expected {num_steps} teacher execution masks, got {len(self._teacher_execution_masks)}."
                )
            execution_masks = torch.stack(self._teacher_execution_masks, dim=0)

        for start in range(0, num_steps - horizon + 1):
            end = start + horizon
            if horizon > 1:
                invalid_envs = dones[start : end - 1].any(dim=0).reshape(-1)
            else:
                invalid_envs = torch.zeros(self.storage.num_envs, dtype=torch.bool, device=self.device)
            if execution_masks is not None:
                execution_window = execution_masks[start:end]
                consistent_execution = execution_window.all(dim=0) | (~execution_window).all(dim=0)
                invalid_envs |= ~consistent_execution
            valid_env_ids = (~invalid_envs).nonzero(as_tuple=False).reshape(-1)
            if valid_env_ids.numel() == 0:
                continue

            obs_sequence = {
                key: observations[key][start:end, valid_env_ids].transpose(0, 1).contiguous()
                for key in observations.keys()
            }
            action_sequence = privileged_actions[start:end, valid_env_ids].transpose(0, 1).contiguous()
            execution_mask_sequence = None
            if execution_masks is not None:
                execution_mask_sequence = execution_masks[start:end, valid_env_ids].transpose(0, 1).contiguous()
            yield obs_sequence, action_sequence, execution_mask_sequence

    """
    Helper functions
    """

    def broadcast_parameters(self, exclude_obs_normalizers: bool = False):
        """Broadcast model parameters to all GPUs.

        Args:
            exclude_obs_normalizers: If True, do not broadcast observation normalizer buffers.
                Useful when each rank intentionally maintains task-specific normalizers.
        """
        # obtain the model parameters on current GPU
        policy_state = self.policy.state_dict()
        if exclude_obs_normalizers:
            policy_state = {k: v for k, v in policy_state.items() if "obs_normalizer" not in k}
        model_params = [policy_state]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        # If we excluded obs-normalizer buffers, allow missing keys.
        self.policy.load_state_dict(model_params[0], strict=not exclude_obs_normalizers)

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
