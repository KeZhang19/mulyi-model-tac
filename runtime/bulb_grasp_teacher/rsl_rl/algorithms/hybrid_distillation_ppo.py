# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Critic-gated behavior cloning and recurrent PPO for the stereo student."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.networks.normalization import (
    EmpiricalNormalization,
    synchronize_empirical_normalization,
)

from .distillation import Distillation


HYBRID_DAGGER_PPO_SCHEMA_VERSION = 3


def hybrid_loss_schedule(
    iteration: int,
    *,
    critic_start_iteration: int = 500,
    critic_full_weight_iteration: int = 600,
    actor_earliest_iteration: int = 700,
    actor_ramp_progress: float = 0.0,
    min_bc_weight: float = 0.1,
    actor_enabled: bool = False,
) -> dict[str, float | str]:
    """Return phase and loss weights for one relative hybrid iteration."""

    iteration = int(iteration)
    if not (
        0 <= critic_start_iteration
        < critic_full_weight_iteration
        <= actor_earliest_iteration
    ):
        raise ValueError(
            "hybrid schedule must satisfy 0 <= critic_start < "
            "critic_full_weight <= actor_earliest"
        )
    if not 0.0 <= min_bc_weight <= 1.0:
        raise ValueError("min_bc_weight must be in [0, 1]")

    critic_progress = min(
        1.0,
        max(
            0.0,
            (iteration - critic_start_iteration)
            / (critic_full_weight_iteration - critic_start_iteration),
        ),
    )
    ramp_progress = min(1.0, max(0.0, float(actor_ramp_progress)))
    ppo_active = bool(actor_enabled and iteration >= actor_earliest_iteration)
    rl_weight = ramp_progress if ppo_active else 0.0
    bc_weight = max(1.0 - rl_weight, float(min_bc_weight))

    if iteration < critic_start_iteration:
        phase = "bc"
        phase_index = 0.0
    elif iteration < critic_full_weight_iteration:
        phase = "critic_ramp"
        phase_index = 1.0
    elif not ppo_active:
        phase = "critic_wait"
        phase_index = 2.0
    else:
        phase = "hybrid"
        phase_index = 3.0

    return {
        "phase": phase,
        "phase_index": phase_index,
        "critic_weight": critic_progress,
        "progress": ramp_progress if ppo_active else 0.0,
        "bc_weight": bc_weight,
        "rl_weight": rl_weight,
    }


class HybridDistillationPPO(Distillation):
    """Recurrent visual BC with a privileged, readiness-gated PPO critic.

    The actor and critic have disjoint parameter graphs. They intentionally use
    one optimizer with named parameter groups so normal checkpoint save/load
    remains compatible with :class:`DistillationRunner`, while learning rates
    and gradient clipping are applied independently per group.
    """

    supports_hybrid_rl = True

    def __init__(
        self,
        policy,
        *,
        critic_start_iteration: int = 500,
        critic_full_weight_iteration: int = 600,
        actor_earliest_iteration: int = 700,
        actor_ramp_iterations: int = 400,
        min_bc_weight: float = 0.1,
        critic_ev_ema_decay: float = 0.99,
        critic_ev_threshold: float = 0.6,
        critic_ready_streak: int = 3,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 5.0,
        entropy_coef: float = 0.0,
        bounds_loss_coef: float = 10.0,
        desired_kl: float = 0.01,
        use_clipped_value_loss: bool = True,
        hybrid_encoder_learning_rate: float = 1.0e-5,
        hybrid_actor_learning_rate: float = 2.0e-5,
        critic_learning_rate: float = 2.0e-4,
        std_learning_rate: float = 2.0e-5,
        **kwargs,
    ):
        if not getattr(policy, "enable_hybrid_rl", False):
            raise ValueError("HybridDistillationPPO requires policy.enable_hybrid_rl=True")
        super().__init__(policy, **kwargs)

        self.critic_start_iteration = int(critic_start_iteration)
        self.critic_full_weight_iteration = int(critic_full_weight_iteration)
        self.actor_earliest_iteration = int(actor_earliest_iteration)
        self.actor_ramp_iterations = int(actor_ramp_iterations)
        self.min_bc_weight = float(min_bc_weight)
        if not (
            0 <= self.critic_start_iteration
            < self.critic_full_weight_iteration
            <= self.actor_earliest_iteration
        ):
            raise ValueError(
                "hybrid schedule must satisfy 0 <= critic_start < "
                "critic_full_weight <= actor_earliest"
            )
        if self.actor_ramp_iterations < 1:
            raise ValueError("actor_ramp_iterations must be positive")
        if not 0.0 <= self.min_bc_weight <= 1.0:
            raise ValueError("min_bc_weight must be in [0, 1]")

        self.critic_ev_ema_decay = float(critic_ev_ema_decay)
        self.critic_ev_threshold = float(critic_ev_threshold)
        self.required_critic_ready_streak = int(critic_ready_streak)
        if not 0.0 <= self.critic_ev_ema_decay < 1.0:
            raise ValueError("critic_ev_ema_decay must be in [0, 1)")
        if not math.isfinite(self.critic_ev_threshold):
            raise ValueError("critic_ev_threshold must be finite")
        if self.required_critic_ready_streak < 1:
            raise ValueError("critic_ready_streak must be positive")

        self.clip_param = float(clip_param)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.bounds_loss_coef = float(bounds_loss_coef)
        self.desired_kl = float(desired_kl)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.hybrid_encoder_learning_rate = float(hybrid_encoder_learning_rate)
        self.hybrid_actor_learning_rate = float(hybrid_actor_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.std_learning_rate = float(std_learning_rate)
        for name, value in (
            ("hybrid_encoder_learning_rate", self.hybrid_encoder_learning_rate),
            ("hybrid_actor_learning_rate", self.hybrid_actor_learning_rate),
            ("critic_learning_rate", self.critic_learning_rate),
            ("std_learning_rate", self.std_learning_rate),
        ):
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")

        self._bc_learning_rates = {
            str(group.get("name", "student")): float(group["lr"])
            for group in self.optimizer.param_groups
        }
        self.training_iteration = 0
        self.schedule_origin_iteration = 0
        self.critic_ev_ema = 0.0
        self.critic_ev_initialized = False
        self.critic_ready_streak = 0
        self.actor_gate_enabled = False
        self.rollout_actor_enabled = False
        self.actor_ramp_active_iterations = 0
        self.last_kl_early_stop = False
        self.per_task_obs_normalizer = False
        self.last_group_grad_norms: dict[str, float] = {}
        self._warned_teacher_ppo_suppression = False
        self._apply_learning_rates()

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        del training_type
        return super().init_storage(
            "hybrid",
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
        )

    def set_training_iteration(self, iteration: int) -> None:
        self.training_iteration = int(iteration)
        self.rollout_actor_enabled = bool(
            self.actor_gate_enabled
            and self.relative_training_iteration() >= self.actor_earliest_iteration
        )
        self._set_hybrid_module_mode()
        self._apply_learning_rates()

    def set_schedule_origin(
        self,
        iteration: int,
        *,
        initial_relative_iteration: int = 0,
    ) -> None:
        initial_relative_iteration = int(initial_relative_iteration)
        if initial_relative_iteration < 0:
            raise ValueError("initial relative iteration must be non-negative")
        self.schedule_origin_iteration = int(iteration) - initial_relative_iteration
        self._set_hybrid_module_mode()
        self._apply_learning_rates()

    def relative_training_iteration(self, iteration: int | None = None) -> int:
        absolute_iteration = (
            self.training_iteration if iteration is None else int(iteration)
        )
        return max(0, absolute_iteration - self.schedule_origin_iteration)

    def actor_ramp_progress(self) -> float:
        return min(
            1.0,
            self.actor_ramp_active_iterations / self.actor_ramp_iterations,
        )

    def schedule_state(
        self,
        iteration: int | None = None,
        *,
        actor_enabled: bool | None = None,
    ) -> dict[str, float | str]:
        if actor_enabled is None:
            actor_enabled = self.rollout_actor_enabled
        return hybrid_loss_schedule(
            self.relative_training_iteration(iteration),
            critic_start_iteration=self.critic_start_iteration,
            critic_full_weight_iteration=self.critic_full_weight_iteration,
            actor_earliest_iteration=self.actor_earliest_iteration,
            actor_ramp_progress=self.actor_ramp_progress(),
            min_bc_weight=self.min_bc_weight,
            actor_enabled=actor_enabled,
        )

    def _set_hybrid_module_mode(self) -> None:
        # Once value learning starts, every stateful/stochastic feature module
        # must behave identically during rollout collection and PPO replay.
        frozen = self.relative_training_iteration() >= self.critic_start_iteration
        for name in (
            "student_obs_normalizer",
            "student_critic_obs_normalizer",
        ):
            normalizer = getattr(self.policy, name, None)
            if isinstance(normalizer, nn.Module):
                normalizer.eval() if frozen else normalizer.train()
        student = getattr(self.policy, "student", None)
        if isinstance(student, nn.Module):
            student.train()
            if frozen:
                frozen_module_types = (
                    nn.MultiheadAttention,
                    nn.modules.batchnorm._BatchNorm,
                    nn.modules.dropout._DropoutNd,
                )
                for module in student.modules():
                    if isinstance(module, frozen_module_types):
                        module.eval()

    def _critic_learning_progress(self) -> float:
        return float(
            hybrid_loss_schedule(
                self.relative_training_iteration(),
                critic_start_iteration=self.critic_start_iteration,
                critic_full_weight_iteration=self.critic_full_weight_iteration,
                actor_earliest_iteration=self.actor_earliest_iteration,
            )["critic_weight"]
        )

    def _apply_learning_rates(self) -> None:
        progress = self._critic_learning_progress()
        for group in self.optimizer.param_groups:
            name = str(group.get("name", "student"))
            if name == "encoder":
                initial = self._bc_learning_rates.get(name, self.learning_rate)
                group["lr"] = initial + progress * (
                    self.hybrid_encoder_learning_rate - initial
                )
            elif name == "student":
                initial = self._bc_learning_rates.get(name, self.learning_rate)
                group["lr"] = initial + progress * (
                    self.hybrid_actor_learning_rate - initial
                )
            elif name == "critic":
                group["lr"] = self.critic_learning_rate
            elif name == "std":
                group["lr"] = (
                    self.std_learning_rate if self.rollout_actor_enabled else 0.0
                )
        self.current_learning_rate = next(
            (
                float(group["lr"])
                for group in self.optimizer.param_groups
                if group.get("name") == "student"
            ),
            float(self.learning_rate),
        )

    def act(self, obs, student_mask: torch.Tensor | None = None):
        del student_mask
        if getattr(self.policy, "is_recurrent", False):
            if hasattr(self.policy, "initialize_hidden_state"):
                self.policy.initialize_hidden_state(obs)
            self.transition.hidden_states = self.policy.get_hidden_states()

        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        if hasattr(self.policy, "observe_teacher_actions"):
            self.policy.observe_teacher_actions(self.transition.privileged_actions)

        executed, raw_action, mean, std, log_prob = self.policy.hybrid_act(
            obs,
            stochastic=self.rollout_actor_enabled,
        )
        self.transition.actions = raw_action.detach()
        self.transition.values = self.policy.evaluate_student_critic(obs).detach()
        self.transition.actions_log_prob = log_prob.detach()
        self.transition.action_mean = mean.detach()
        self.transition.action_sigma = std.detach()
        self.transition.observations = obs
        return executed.detach()

    def record_teacher_execution_mask(self, mask: torch.Tensor):
        # Hybrid PPO needs this contract even though the V5 policy itself does
        # not use execution masks for its BC objective.
        self._teacher_execution_masks.append(
            mask.detach().to(device=self.device, dtype=torch.bool).clone()
        )

    def process_env_step(self, obs, rewards, dones, extras):
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device).reshape(-1)
            self.transition.rewards += (
                self.gamma
                * self.transition.values.reshape(-1)
                * time_outs
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        last_values = self.policy.evaluate_student_critic(obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=False,
        )

    @staticmethod
    def _bc_action_losses(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        squared_error = (prediction - target).square()
        return squared_error.sum(dim=-1).mean(), squared_error.mean()

    def _action_bound_loss(self, means: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.policy, "effective_raw_action_limits"):
            return means.new_zeros(())
        limits = self.policy.effective_raw_action_limits(means).to(means)
        excess = F.relu(means.abs() - limits)
        return excess.square().sum(dim=-1).mean()

    def _teacher_execution_fraction(self, num_steps: int) -> float:
        if not self._teacher_execution_masks:
            return 0.0
        if len(self._teacher_execution_masks) != num_steps:
            raise RuntimeError(
                "hybrid rollout execution-mask count does not match storage: "
                f"masks={len(self._teacher_execution_masks)}, steps={num_steps}"
            )
        masks = torch.stack(self._teacher_execution_masks, dim=0).float()
        fraction = masks.mean()
        if self.is_multi_gpu:
            torch.distributed.all_reduce(fraction, op=torch.distributed.ReduceOp.SUM)
            fraction /= self.gpu_world_size
        return float(fraction.item())

    def _stored_explained_variance(self, num_steps: int) -> float:
        returns = self.storage.returns[:num_steps].detach().double().reshape(-1)
        errors = (
            self.storage.returns[:num_steps] - self.storage.values[:num_steps]
        ).detach().double().reshape(-1)
        moments = torch.stack(
            (
                returns.new_tensor(float(returns.numel())),
                returns.sum(),
                returns.square().sum(),
                errors.sum(),
                errors.square().sum(),
            )
        )
        if self.is_multi_gpu:
            torch.distributed.all_reduce(moments, op=torch.distributed.ReduceOp.SUM)
        count, return_sum, return_sq_sum, error_sum, error_sq_sum = moments
        if count.item() <= 0.0:
            return 0.0
        return_var = return_sq_sum / count - (return_sum / count).square()
        error_var = error_sq_sum / count - (error_sum / count).square()
        if not torch.isfinite(return_var) or return_var.item() <= 1.0e-8:
            return 0.0
        explained_variance = 1.0 - error_var.clamp_min(0.0) / return_var
        return float(explained_variance.item())

    def _update_critic_gate(self, explained_variance: float) -> None:
        relative_iteration = self.relative_training_iteration()
        if relative_iteration >= self.critic_start_iteration and math.isfinite(
            explained_variance
        ):
            if not self.critic_ev_initialized:
                self.critic_ev_ema = float(explained_variance)
                self.critic_ev_initialized = True
            else:
                decay = self.critic_ev_ema_decay
                self.critic_ev_ema = (
                    decay * self.critic_ev_ema
                    + (1.0 - decay) * float(explained_variance)
                )

        ready = (
            relative_iteration >= self.actor_earliest_iteration
            and self.critic_ev_initialized
            and math.isfinite(self.critic_ev_ema)
            and self.critic_ev_ema >= self.critic_ev_threshold
        )
        if ready:
            self.critic_ready_streak += 1
        else:
            self.critic_ready_streak = 0
        self.actor_gate_enabled = (
            self.critic_ready_streak >= self.required_critic_ready_streak
        )

    def update(self):  # noqa: C901, PLR0912, PLR0915
        self.num_updates += 1
        num_steps = int(self.storage.step)
        num_envs = int(self.storage.num_envs)
        if num_steps < 1:
            losses = self._empty_metrics()
            losses.update(self._schedule_metrics(self.schedule_state()))
            return losses

        explained_variance = self._stored_explained_variance(num_steps)
        current_rollout_actor_enabled = bool(self.rollout_actor_enabled)
        teacher_execution_fraction = self._teacher_execution_fraction(num_steps)
        ppo_suppressed_by_teacher = bool(
            current_rollout_actor_enabled and teacher_execution_fraction > 0.0
        )
        if ppo_suppressed_by_teacher and not self._warned_teacher_ppo_suppression:
            warnings.warn(
                "Teacher actions were executed during an actor-enabled hybrid rollout; "
                "PPO actor/std updates are suppressed for this rollout. Ensure teacher "
                "mix is exactly zero before critic-gated PPO starts.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_teacher_ppo_suppression = True
        ppo_rollout_valid = bool(
            current_rollout_actor_enabled and not ppo_suppressed_by_teacher
        )
        self._update_critic_gate(explained_variance)

        schedule = self.schedule_state(actor_enabled=ppo_rollout_valid)
        bc_weight = float(schedule["bc_weight"])
        rl_weight = float(schedule["rl_weight"])
        critic_weight = float(schedule["critic_weight"])

        raw_advantages = self.storage.advantages[:num_steps]
        advantage_mean = float(raw_advantages.mean().item())
        advantage_std_tensor = raw_advantages.std(unbiased=False)
        advantage_std = float(advantage_std_tensor.item())
        advantages = (raw_advantages - raw_advantages.mean()) / (
            advantage_std_tensor + 1.0e-8
        )

        totals = {
            "bc_objective": 0.0,
            "bc_mse": 0.0,
            "auxiliary": 0.0,
            "ppo": 0.0,
            "bounds": 0.0,
            "entropy": 0.0,
            "value": 0.0,
            "weighted_bc": 0.0,
            "weighted_auxiliary": 0.0,
            "weighted_ppo": 0.0,
            "weighted_bounds": 0.0,
            "weighted_value": 0.0,
            "kl": 0.0,
            "ratio": 0.0,
            "clip_fraction": 0.0,
        }
        batch_count = 0
        kl_count = 0
        optimizer_steps = 0
        actor_optimizer_steps = 0
        critic_optimizer_steps = 0
        pending_batches = 0
        actor_updates_allowed = True
        kl_early_stop = False
        early_stop_kl = 0.0
        initial_recurrent_kl = 0.0
        initial_kl_recorded = False
        group_grad_norm_sums: dict[str, float] = {}
        group_grad_norm_counts: dict[str, int] = {}

        self.optimizer.zero_grad()
        for _epoch in range(self.num_learning_epochs):
            env_order = torch.randperm(num_envs, device=self.device)
            for sequence_start in range(0, num_steps, self.tbptt_steps):
                sequence_end = min(sequence_start + self.tbptt_steps, num_steps)
                for env_start in range(0, num_envs, self.recurrent_minibatch_size):
                    env_ids = env_order[
                        env_start : env_start + self.recurrent_minibatch_size
                    ]
                    observations = {
                        key: value[sequence_start:sequence_end, env_ids]
                        for key, value in self.storage.observations.items()
                    }
                    dones = self.storage.dones[
                        sequence_start:sequence_end, env_ids
                    ]
                    initial_hidden = self._tbptt_initial_hidden(
                        sequence_start,
                        env_ids,
                    )
                    means, object_position, _ = self.policy.sequence_predictions(
                        observations,
                        dones,
                        initial_hidden=initial_hidden,
                    )
                    teacher_actions = self.storage.privileged_actions[
                        sequence_start:sequence_end, env_ids
                    ]
                    bc_action_loss, bc_mse = self._bc_action_losses(
                        means,
                        teacher_actions,
                    )
                    if self.policy.auxiliary_loss_coefficient:
                        auxiliary_loss = F.mse_loss(
                            object_position,
                            observations[self.policy.student_aux_obs_group],
                        )
                    else:
                        auxiliary_loss = bc_action_loss.new_zeros(())
                    bound_loss = self._action_bound_loss(means)

                    ppo_loss = bc_action_loss.new_zeros(())
                    std_vector = self.policy.hybrid_action_std()
                    entropy = (
                        std_vector.detach().log()
                        + 0.5 * (1.0 + math.log(2.0 * math.pi))
                    ).sum()
                    approx_kl = bc_action_loss.new_zeros(())
                    ratio_mean = bc_action_loss.new_ones(())
                    clip_fraction = bc_action_loss.new_zeros(())
                    if ppo_rollout_valid:
                        std = std_vector.view(1, 1, -1).expand_as(means)
                        distribution = torch.distributions.Normal(means, std)
                        raw_actions = self.storage.actions[
                            sequence_start:sequence_end, env_ids
                        ]
                        new_log_prob = distribution.log_prob(raw_actions).sum(
                            dim=-1,
                            keepdim=True,
                        )
                        entropy = distribution.entropy().sum(dim=-1).mean()
                        old_log_prob = self.storage.actions_log_prob[
                            sequence_start:sequence_end, env_ids
                        ]
                        ratio = torch.exp(new_log_prob - old_log_prob)
                        advantage_batch = advantages[
                            sequence_start:sequence_end, env_ids
                        ]
                        surrogate = -advantage_batch * ratio
                        surrogate_clipped = -advantage_batch * ratio.clamp(
                            1.0 - self.clip_param,
                            1.0 + self.clip_param,
                        )
                        ppo_loss = torch.maximum(
                            surrogate,
                            surrogate_clipped,
                        ).mean()
                        ratio_mean = ratio.detach().mean()
                        clip_fraction = (
                            (ratio.detach() - 1.0).abs() > self.clip_param
                        ).float().mean()

                        old_mean = self.storage.mu[
                            sequence_start:sequence_end, env_ids
                        ]
                        old_std = self.storage.sigma[
                            sequence_start:sequence_end, env_ids
                        ]
                        with torch.no_grad():
                            # The first post-step KL can be much smaller than
                            # float32 cancellation error. Compute the closed
                            # form in float64 so the early-stop controller can
                            # still distinguish a real policy change from the
                            # rollout/update consistency check at zero.
                            std_kl = std.double()
                            old_std_kl = old_std.double()
                            mean_kl = means.double()
                            old_mean_kl = old_mean.double()
                            approx_kl = (
                                torch.log(std_kl / old_std_kl)
                                + (
                                    old_std_kl.square()
                                    + (old_mean_kl - mean_kl).square()
                                )
                                / (2.0 * std_kl.square())
                                - 0.5
                            ).sum(dim=-1).mean()
                            if self.is_multi_gpu:
                                torch.distributed.all_reduce(
                                    approx_kl,
                                    op=torch.distributed.ReduceOp.SUM,
                                )
                                approx_kl /= self.gpu_world_size

                        if not initial_kl_recorded:
                            initial_recurrent_kl = float(approx_kl.item())
                            initial_kl_recorded = True
                            mismatch_threshold = max(
                                2.0 * self.desired_kl,
                                1.0e-3,
                            )
                            if initial_recurrent_kl > mismatch_threshold:
                                raise RuntimeError(
                                    "recurrent PPO rollout/update policy mismatch before "
                                    "the first optimizer step: "
                                    f"KL={initial_recurrent_kl:.6g}, "
                                    f"threshold={mismatch_threshold:.6g}. Ensure actor "
                                    "normalizers, dropout, attention, and BatchNorm are frozen."
                                )
                        elif (
                            actor_optimizer_steps > 0
                            and actor_updates_allowed
                            and self.desired_kl > 0.0
                            and float(approx_kl.item()) > 2.0 * self.desired_kl
                        ):
                            actor_updates_allowed = False
                            kl_early_stop = True
                            early_stop_kl = float(approx_kl.item())

                    value_loss = bc_action_loss.new_zeros(())
                    if critic_weight > 0.0:
                        values = self.policy.evaluate_student_critic_sequence(
                            observations
                        )
                        old_values = self.storage.values[
                            sequence_start:sequence_end, env_ids
                        ]
                        returns = self.storage.returns[
                            sequence_start:sequence_end, env_ids
                        ]
                        if self.use_clipped_value_loss:
                            value_clipped = old_values + (
                                values - old_values
                            ).clamp(
                                -self.clip_param,
                                self.clip_param,
                            )
                            value_loss = torch.maximum(
                                (values - returns).square(),
                                (value_clipped - returns).square(),
                            ).mean()
                        else:
                            value_loss = (values - returns).square().mean()

                    weighted_bc = bc_weight * bc_action_loss
                    weighted_auxiliary = (
                        self.policy.auxiliary_loss_coefficient * auxiliary_loss
                    )
                    weighted_ppo = rl_weight * (
                        ppo_loss - self.entropy_coef * entropy
                    )
                    weighted_bounds = (
                        rl_weight * self.bounds_loss_coef * bound_loss
                    )
                    weighted_value = (
                        critic_weight * self.value_loss_coef * value_loss
                    )
                    actor_loss = (
                        weighted_bc + weighted_auxiliary + weighted_ppo + weighted_bounds
                    )
                    loss_terms = []
                    if actor_updates_allowed:
                        loss_terms.append(actor_loss)
                    if critic_weight > 0.0:
                        loss_terms.append(weighted_value)

                    if loss_terms:
                        total_loss = sum(loss_terms)
                        (total_loss / self.gradient_length).backward()
                        pending_batches += 1

                    totals["bc_objective"] += float(bc_action_loss.detach().item())
                    totals["bc_mse"] += float(bc_mse.detach().item())
                    totals["auxiliary"] += float(auxiliary_loss.detach().item())
                    totals["ppo"] += float(ppo_loss.detach().item())
                    totals["bounds"] += float(bound_loss.detach().item())
                    totals["entropy"] += float(entropy.detach().item())
                    totals["value"] += float(value_loss.detach().item())
                    totals["weighted_bc"] += float(weighted_bc.detach().item())
                    totals["weighted_auxiliary"] += float(
                        weighted_auxiliary.detach().item()
                    )
                    totals["weighted_ppo"] += float(weighted_ppo.detach().item())
                    totals["weighted_bounds"] += float(
                        weighted_bounds.detach().item()
                    )
                    totals["weighted_value"] += float(weighted_value.detach().item())
                    totals["ratio"] += float(ratio_mean.item())
                    totals["clip_fraction"] += float(clip_fraction.item())
                    if ppo_rollout_valid:
                        totals["kl"] += float(approx_kl.item())
                        kl_count += 1
                    batch_count += 1

                    if pending_batches == self.gradient_length:
                        step_info = self._step_accumulated_gradients(pending_batches)
                        pending_batches = 0
                        optimizer_steps += 1
                        actor_optimizer_steps += int(step_info["actor_updated"])
                        critic_optimizer_steps += int(step_info["critic_updated"])
                        self._accumulate_group_grad_norms(
                            step_info["grad_norms"],
                            group_grad_norm_sums,
                            group_grad_norm_counts,
                        )

        if pending_batches:
            step_info = self._step_accumulated_gradients(pending_batches)
            optimizer_steps += 1
            actor_optimizer_steps += int(step_info["actor_updated"])
            critic_optimizer_steps += int(step_info["critic_updated"])
            self._accumulate_group_grad_norms(
                step_info["grad_norms"],
                group_grad_norm_sums,
                group_grad_norm_counts,
            )

        if ppo_rollout_valid and actor_optimizer_steps > 0:
            self.actor_ramp_active_iterations = min(
                self.actor_ramp_iterations,
                self.actor_ramp_active_iterations + 1,
            )

        self._sync_empirical_normalization_buffers()
        self.last_kl_early_stop = kl_early_stop
        self.storage.clear()
        self._teacher_execution_masks.clear()
        self.policy.detach_hidden_states()

        divisor = max(batch_count, 1)
        mean_kl = totals["kl"] / max(kl_count, 1)
        losses = {
            "behavior": totals["bc_mse"] / divisor,
            "bc_loss": totals["bc_mse"] / divisor,
            "bc_action_objective": totals["bc_objective"] / divisor,
            "auxiliary_object_position": totals["auxiliary"] / divisor,
            "ppo_loss": totals["ppo"] / divisor,
            "bound_loss": totals["bounds"] / divisor,
            "entropy": totals["entropy"] / divisor,
            "value_loss": totals["value"] / divisor,
            "weighted_bc_actor_loss": totals["weighted_bc"] / divisor,
            "weighted_auxiliary_loss": totals["weighted_auxiliary"] / divisor,
            "weighted_ppo_loss": totals["weighted_ppo"] / divisor,
            "weighted_bound_loss": totals["weighted_bounds"] / divisor,
            "weighted_value_loss": totals["weighted_value"] / divisor,
            "approx_kl": max(mean_kl, early_stop_kl),
            "initial_recurrent_kl": initial_recurrent_kl,
            "ratio_mean": totals["ratio"] / divisor,
            "ratio_clip_fraction": totals["clip_fraction"] / divisor,
            "gae_mean": advantage_mean,
            "gae_std": advantage_std,
            "explained_variance": explained_variance,
            "critic_ev_ema": float(self.critic_ev_ema),
            "critic_ready_streak": float(self.critic_ready_streak),
            "actor_gate_next_rollout": float(self.actor_gate_enabled),
            "actor_ramp_active_iterations": float(
                self.actor_ramp_active_iterations
            ),
            "teacher_execution_fraction": teacher_execution_fraction,
            "ppo_suppressed_by_teacher_actions": float(
                ppo_suppressed_by_teacher
            ),
            "action_std_mean": float(self.policy.hybrid_action_std().mean().item()),
            "action_std_min": float(self.policy.hybrid_action_std().min().item()),
            "action_std_max": float(self.policy.hybrid_action_std().max().item()),
            "kl_early_stop": float(kl_early_stop),
            "optimizer_steps": float(optimizer_steps),
            "actor_optimizer_steps": float(actor_optimizer_steps),
            "critic_optimizer_steps": float(critic_optimizer_steps),
            "actor_learning_rate": float(self.current_learning_rate),
        }
        for name, total in group_grad_norm_sums.items():
            losses[f"{name}_grad_norm"] = total / max(
                group_grad_norm_counts.get(name, 0),
                1,
            )
        losses.update(self._schedule_metrics(schedule))
        return losses

    def _step_accumulated_gradients(self, pending_batches: int) -> dict[str, object]:
        if pending_batches < 1:
            raise ValueError("pending_batches must be positive")
        if pending_batches != self.gradient_length:
            scale = self.gradient_length / pending_batches
            for parameter in self.policy.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        if self.is_multi_gpu:
            self.reduce_parameters()

        grad_norms: dict[str, float] = {}
        actor_updated = False
        critic_updated = False
        max_norm = self.max_grad_norm if self.max_grad_norm else float("inf")
        for group in self.optimizer.param_groups:
            name = str(group.get("name", "student"))
            parameters = [
                parameter
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            if not parameters:
                continue
            grad_norm = nn.utils.clip_grad_norm_(parameters, max_norm)
            grad_norms[name] = float(grad_norm.item())
            actor_updated = actor_updated or name in {"encoder", "student", "std"}
            critic_updated = critic_updated or name == "critic"

        self.last_group_grad_norms = grad_norms
        self.last_depth_encoder_grad_norm = grad_norms.get("encoder", 0.0)
        self.last_student_grad_norm = math.sqrt(
            grad_norms.get("encoder", 0.0) ** 2
            + grad_norms.get("student", 0.0) ** 2
        )
        self.optimizer.step()
        self._clamp_action_std_parameter()
        self.optimizer.zero_grad()
        return {
            "grad_norms": grad_norms,
            "actor_updated": actor_updated,
            "critic_updated": critic_updated,
        }

    @staticmethod
    def _accumulate_group_grad_norms(
        grad_norms: dict[str, float],
        totals: dict[str, float],
        counts: dict[str, int],
    ) -> None:
        for name, value in grad_norms.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

    def _sync_empirical_normalization_buffers(self) -> None:
        if (
            not self.is_multi_gpu
            or self.per_task_obs_normalizer
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        for module in self.policy.modules():
            if isinstance(module, EmpiricalNormalization):
                synchronize_empirical_normalization(module)

    def _clamp_action_std_parameter(self) -> None:
        with torch.no_grad():
            self.policy.student_log_std.clamp_(
                math.log(self.policy.min_action_std),
                math.log(self.policy.max_action_std),
            )

    def _schedule_metrics(
        self,
        schedule: dict[str, float | str],
    ) -> dict[str, float]:
        return {
            "hybrid_phase": float(schedule["phase_index"]),
            "critic_weight": float(schedule["critic_weight"]),
            "bc_weight": float(schedule["bc_weight"]),
            "rl_weight": float(schedule["rl_weight"]),
            "rl_ramp_progress": float(schedule["progress"]),
            "schedule_iteration": float(self.relative_training_iteration()),
            "schedule_origin_iteration": float(self.schedule_origin_iteration),
            "normalizers_frozen": float(
                self.relative_training_iteration() >= self.critic_start_iteration
            ),
            "rollout_actor_enabled": float(self.rollout_actor_enabled),
        }

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        return {
            "behavior": 0.0,
            "bc_loss": 0.0,
            "bc_action_objective": 0.0,
            "auxiliary_object_position": 0.0,
            "ppo_loss": 0.0,
            "bound_loss": 0.0,
            "entropy": 0.0,
            "value_loss": 0.0,
            "weighted_bc_actor_loss": 0.0,
            "weighted_auxiliary_loss": 0.0,
            "weighted_ppo_loss": 0.0,
            "weighted_bound_loss": 0.0,
            "weighted_value_loss": 0.0,
            "approx_kl": 0.0,
            "initial_recurrent_kl": 0.0,
            "ratio_mean": 0.0,
            "ratio_clip_fraction": 0.0,
            "gae_mean": 0.0,
            "gae_std": 0.0,
            "explained_variance": 0.0,
            "critic_ev_ema": 0.0,
            "critic_ready_streak": 0.0,
            "actor_gate_next_rollout": 0.0,
            "actor_ramp_active_iterations": 0.0,
            "teacher_execution_fraction": 0.0,
            "ppo_suppressed_by_teacher_actions": 0.0,
            "kl_early_stop": 0.0,
            "optimizer_steps": 0.0,
            "actor_optimizer_steps": 0.0,
            "critic_optimizer_steps": 0.0,
        }

    def training_state(self) -> dict[str, float | int | bool]:
        return {
            "schedule_origin_iteration": self.schedule_origin_iteration,
            "critic_ev_ema": self.critic_ev_ema,
            "critic_ev_initialized": self.critic_ev_initialized,
            "critic_ready_streak": self.critic_ready_streak,
            "actor_gate_enabled": self.actor_gate_enabled,
            "actor_ramp_active_iterations": self.actor_ramp_active_iterations,
        }

    def validate_training_state(
        self,
        state: object,
    ) -> dict[str, float | int | bool]:
        if not isinstance(state, dict):
            raise ValueError("hybrid checkpoint is missing hybrid_training_state")

        origin = state.get("schedule_origin_iteration")
        if isinstance(origin, bool) or not isinstance(origin, int):
            raise ValueError(
                "hybrid checkpoint has invalid schedule origin: "
                f"{origin!r}"
            )
        ev = state.get("critic_ev_ema")
        if (
            isinstance(ev, bool)
            or not isinstance(ev, (int, float))
            or not math.isfinite(float(ev))
        ):
            raise ValueError(
                "hybrid checkpoint has invalid critic EV EMA: "
                f"{ev!r}"
            )
        initialized = state.get("critic_ev_initialized")
        gate = state.get("actor_gate_enabled")
        if not isinstance(initialized, bool) or not isinstance(gate, bool):
            raise ValueError("hybrid checkpoint has invalid critic/actor gate flags")
        streak = state.get("critic_ready_streak")
        ramp_iterations = state.get("actor_ramp_active_iterations")
        for name, value in (
            ("critic_ready_streak", streak),
            ("actor_ramp_active_iterations", ramp_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"hybrid checkpoint has invalid {name}: {value!r}"
                )
        if ramp_iterations > self.actor_ramp_iterations:
            raise ValueError(
                "hybrid checkpoint ramp progress exceeds the configured ramp"
            )
        if not initialized and (streak != 0 or gate):
            raise ValueError(
                "hybrid checkpoint enables critic readiness before EV initialization"
            )
        expected_gate = streak >= self.required_critic_ready_streak
        if gate != expected_gate:
            raise ValueError(
                "hybrid checkpoint actor gate disagrees with the critic-ready streak"
            )
        if gate and float(ev) < self.critic_ev_threshold:
            raise ValueError(
                "hybrid checkpoint enables PPO below the critic EV threshold"
            )
        return {
            "schedule_origin_iteration": origin,
            "critic_ev_ema": float(ev),
            "critic_ev_initialized": initialized,
            "critic_ready_streak": streak,
            "actor_gate_enabled": gate,
            "actor_ramp_active_iterations": ramp_iterations,
        }

    def load_training_state(self, state: object) -> None:
        validated = self.validate_training_state(state)
        self.schedule_origin_iteration = int(
            validated["schedule_origin_iteration"]
        )
        self.critic_ev_ema = float(validated["critic_ev_ema"])
        self.critic_ev_initialized = bool(
            validated["critic_ev_initialized"]
        )
        self.critic_ready_streak = int(validated["critic_ready_streak"])
        self.actor_gate_enabled = bool(validated["actor_gate_enabled"])
        self.actor_ramp_active_iterations = int(
            validated["actor_ramp_active_iterations"]
        )
        self._set_hybrid_module_mode()
        self._apply_learning_rates()

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "schema_version": HYBRID_DAGGER_PPO_SCHEMA_VERSION,
            "algorithm": "Critic-gated Hybrid DAgger-PPO",
            "loss_schedule": {
                "iteration_basis": "relative_to_schedule_origin",
                "critic_start_iteration": self.critic_start_iteration,
                "critic_full_weight_iteration": self.critic_full_weight_iteration,
                "actor_earliest_iteration": self.actor_earliest_iteration,
                "actor_ramp_iterations": self.actor_ramp_iterations,
                "min_bc_weight": self.min_bc_weight,
                "critic_ev_ema_decay": self.critic_ev_ema_decay,
                "critic_ev_threshold": self.critic_ev_threshold,
                "critic_ready_streak": self.required_critic_ready_streak,
            },
            "losses": {
                "bc_reduction": "mean_time_env_sum_action",
                "auxiliary_weight": self.policy.auxiliary_loss_coefficient,
                "value_loss_coef": self.value_loss_coef,
                "entropy_coef": self.entropy_coef,
                "bounds_loss_coef": self.bounds_loss_coef,
            },
            "ppo": {
                "clip_param": self.clip_param,
                "gamma": self.gamma,
                "lam": self.lam,
                "desired_kl": self.desired_kl,
                "kl_policy": "actor_early_stop_critic_continues",
                "max_grad_norm_per_parameter_group": self.max_grad_norm,
                "num_learning_epochs": self.num_learning_epochs,
                "gradient_length": self.gradient_length,
            },
            "learning_rates": {
                "hybrid_encoder": self.hybrid_encoder_learning_rate,
                "hybrid_actor": self.hybrid_actor_learning_rate,
                "critic": self.critic_learning_rate,
                "std": self.std_learning_rate,
                "adaptive_kl_scaling": False,
            },
            "normalization": {
                "freeze_iteration": self.critic_start_iteration,
                "iteration_basis": "relative_to_schedule_origin",
                "student_module_mode_after_freeze": (
                    "train_with_dropout_attention_and_batchnorm_eval"
                ),
            },
            "action_distribution": {
                "rollout": "unclipped_diagonal_gaussian_when_critic_gate_is_open",
                "storage_action": "unclipped_gaussian_sample",
                "environment_action": "contract_v5_effective_raw_clipped",
                "deployment": "deterministic_effective_mean",
                "min_std": self.policy.min_action_std,
                "max_std": self.policy.max_action_std,
                "teacher_mix_requirement": "zero_for_actor_enabled_rollouts",
            },
        }


__all__ = [
    "HYBRID_DAGGER_PPO_SCHEMA_VERSION",
    "HybridDistillationPPO",
    "hybrid_loss_schedule",
]
