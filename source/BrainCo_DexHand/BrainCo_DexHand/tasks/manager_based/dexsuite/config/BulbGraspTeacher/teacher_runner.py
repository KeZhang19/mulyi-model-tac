# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import statistics
import time
import torch
import warnings
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, PPODualHead
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticDualHead,
    ActorCriticRecurrent,
    ActorCriticTransformer,
    ActorCriticTransformerCache,
    ActorCriticTransformerXL,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.utils import resolve_obs_groups, store_code_state
from .policy import BulbGraspTeacherLSTM


class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # Keep a copy of the *raw* env observation groups for residual base policy inference.
        # (The residual policy may use augmented observations.)
        self._base_obs_groups = copy.deepcopy(self.cfg["obs_groups"])
        self._base_obs_example = obs

        # Residual RL (base policy + learned residual policy):
        # If residual_base_checkpoint is provided, we load a frozen base policy and at rollout time execute:
        #   a_env = a_base + alpha * a_residual
        # where a_residual is produced by the trainable PPO policy (self.alg.policy).
        self.residual_base_checkpoint: str | None = self.cfg.get("residual_base_checkpoint", None)
        self.residual_alpha: float = float(self.cfg.get("residual_alpha", 1.0))
        self.residual_clip_actions: bool = bool(self.cfg.get("residual_clip_actions", False))
        # If enabled, include base-policy action as an additional observation group for the residual policy.
        # The group includes a matching history stack so transformer tokenization sees consistent history.
        self.residual_base_action_as_obs: bool = bool(self.cfg.get("residual_base_action_as_obs", True))
        self.residual_base_action_obs_key: str = str(self.cfg.get("residual_base_action_obs_key", "residual_base_action"))
        # History length (H) for the base-action observation stack:
        # Priority:
        #   1) train_cfg["residual_base_action_history"] (explicit override)
        #   2) env.cfg.observations.<group>.history_length (match env's own stacked-history)
        #   3) policy_cfg["obs_history_tokens"] (transformer token count)
        explicit_h = self.cfg.get("residual_base_action_history", None)
        if explicit_h is None:
            inferred_h = self._infer_env_history_length()
            if inferred_h is None:
                inferred_h = self.policy_cfg.get("obs_history_tokens", 1) or 1
            self.residual_base_action_history = int(inferred_h)
        else:
            self.residual_base_action_history = int(explicit_h)
        if self.residual_base_action_history < 1:
            self.residual_base_action_history = 1
        # Runtime state: base-action history buffer is allocated on first learn().
        self._residual_base_action_hist = None

        # If residual-base-action-as-observation is enabled, augment the example observation and obs_groups
        # BEFORE constructing the algorithm so network input dimensions match.
        if self.residual_base_checkpoint and self.residual_base_action_as_obs:
            obs = self._augment_obs_for_residual(obs, base_action_flat=None)
            for set_name in ("policy", "critic"):
                if self.residual_base_action_obs_key not in self.cfg["obs_groups"][set_name]:
                    self.cfg["obs_groups"][set_name] = self.cfg["obs_groups"][set_name] + [self.residual_base_action_obs_key]

        # create the algorithm (may use augmented obs)
        self.alg = self._construct_algorithm(obs)

        self.residual_base_policy = None
        if self.residual_base_checkpoint:
            try:
                self._load_residual_base_policy(self.residual_base_checkpoint)
                # Print a clear one-liner so users can confirm residual RL is active.
                print(
                    "[ResidualRL] enabled: "
                    f"base='{self.residual_base_checkpoint}', alpha={self.residual_alpha}, "
                    f"clip_actions={self.residual_clip_actions}"
                )
            except Exception as e:
                self.residual_base_policy = None
                print(
                    "[ResidualRL] disabled: failed to load base checkpoint "
                    f"base='{self.residual_base_checkpoint}'. Error: {e}"
                )
        # Per-task (heterogeneous) observation normalizer support:
        # When enabled, each rank keeps its own observation normalizer buffers and we package them
        # into a single checkpoint at save-time (one entry per rank/task). This avoids "rank0 bias"
        # when different ranks run different robots/tasks.
        self.per_task_obs_normalizer: bool = bool(self.cfg.get("per_task_obs_normalizer", False))
        if hasattr(self.alg, "per_task_obs_normalizer"):
            self.alg.per_task_obs_normalizer = self.per_task_obs_normalizer

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        self._v2_gravity_curriculum_state = (
            {"trigger_iteration": None, "last_success_rate": None}
            if self._v2_gravity_curriculum_cfg() is not None
            else None
        )
        self._init_object_success_tracking()
        self._init_abnormal_robot_tracking()

    def _init_object_success_tracking(self) -> None:
        """Discover optional V2-test object IDs exposed by the command term."""

        self._object_success_term = None
        self._object_success_names: tuple[str, ...] = ()
        self._object_success_ids: torch.Tensor | None = None
        self._object_success_iteration_stats: torch.Tensor | None = None
        try:
            env = getattr(self.env, "unwrapped", self.env)
            term = env.command_manager.get_term("object_pose")
            object_ids = term.v2_test_object_ids
            object_names = tuple(term.v2_test_object_names)
        except (AttributeError, KeyError, ValueError):
            return
        if object_ids is None:
            return
        if object_ids.numel() != int(self.env.num_envs):
            raise RuntimeError(
                "V2-test object ID count does not match the vectorized environment: "
                f"ids={object_ids.numel()}, envs={self.env.num_envs}."
            )
        if not object_names:
            raise RuntimeError("V2-test object names cannot be empty when object IDs are present.")
        object_ids = object_ids.detach().reshape(-1).to(device=self.device, dtype=torch.long)
        if int(object_ids.min().item()) < 0 or int(object_ids.max().item()) >= len(object_names):
            raise RuntimeError(
                "V2-test object IDs are outside the configured name range: "
                f"ids=[{int(object_ids.min().item())}, {int(object_ids.max().item())}], "
                f"names={object_names}."
            )
        self._object_success_term = term
        self._object_success_names = object_names
        self._object_success_ids = object_ids
        # groups: overall, student-only, any-teacher. fields: success sum, episode count.
        self._object_success_iteration_stats = torch.zeros(
            (3, len(object_names), 2), device=self.device, dtype=torch.float32
        )

    def _record_terminal_object_success(
        self,
        dones: torch.Tensor,
        episode_used_teacher: torch.Tensor | None = None,
    ) -> None:
        """Accumulate exact terminal success sums/counts for completed episodes."""

        if self._object_success_iteration_stats is None or self._object_success_ids is None:
            return
        done_ids = dones.reshape(-1).bool().nonzero(as_tuple=False).reshape(-1)
        if done_ids.numel() == 0:
            return
        terminal_success = self._object_success_term.terminal_success.detach().reshape(-1).to(
            device=self.device, dtype=torch.float32
        )
        if terminal_success.numel() != self._object_success_ids.numel():
            raise RuntimeError(
                "V2-test terminal success count does not match object IDs: "
                f"success={terminal_success.numel()}, ids={self._object_success_ids.numel()}."
            )
        object_ids = self._object_success_ids[done_ids]
        successes = terminal_success[done_ids]
        counts = torch.ones_like(successes)
        self._object_success_iteration_stats[0, :, 0].scatter_add_(
            0, object_ids, successes
        )
        self._object_success_iteration_stats[0, :, 1].scatter_add_(
            0, object_ids, counts
        )

        if episode_used_teacher is None:
            return
        used_teacher = episode_used_teacher.reshape(-1).bool().to(self.device)[done_ids]
        for group_index, group_mask in ((1, ~used_teacher), (2, used_teacher)):
            if not bool(group_mask.any().item()):
                continue
            group_object_ids = object_ids[group_mask]
            group_successes = successes[group_mask]
            self._object_success_iteration_stats[group_index, :, 0].scatter_add_(
                0, group_object_ids, group_successes
            )
            self._object_success_iteration_stats[group_index, :, 1].scatter_add_(
                0, group_object_ids, torch.ones_like(group_successes)
            )

    def _finalize_object_success_stats(self) -> torch.Tensor | None:
        """Reduce the current iteration across ranks and reset local accumulators."""

        if self._object_success_iteration_stats is None:
            return None
        stats = self._object_success_iteration_stats.clone()
        if (
            self.is_distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        self._object_success_iteration_stats.zero_()
        return stats

    def _log_object_success_stats(
        self,
        stats: torch.Tensor | None,
        iteration: int,
        *,
        include_distillation_roles: bool = False,
    ) -> None:
        """Write per-object episode counts and non-empty success rates."""

        if self.writer is None or stats is None:
            return
        groups = (("overall", 0),)
        if include_distillation_roles:
            groups += (("student", 1), ("teacher", 2))
        for group_name, group_index in groups:
            for object_index, object_name in enumerate(self._object_success_names):
                success_sum = float(stats[group_index, object_index, 0].item())
                episode_count = float(stats[group_index, object_index, 1].item())
                self.writer.add_scalar(
                    f"ObjectEpisodes/{group_name}/{object_name}",
                    episode_count,
                    iteration,
                )
                if episode_count > 0.0:
                    self.writer.add_scalar(
                        f"ObjectSuccess/{group_name}/{object_name}",
                        success_sum / episode_count,
                        iteration,
                    )

    def _init_abnormal_robot_tracking(self) -> None:
        """Discover the optional Stable V2-test abnormal diagnostic term."""

        self._abnormal_robot_term = None
        self._abnormal_robot_joint_names: tuple[str, ...] = ()
        self._abnormal_robot_iteration_stats: torch.Tensor | None = None
        try:
            env = getattr(self.env, "unwrapped", self.env)
            term = env.termination_manager.get_term_cfg("abnormal_robot").func
        except (AttributeError, KeyError, ValueError):
            return
        if not bool(getattr(term, "provides_abnormal_robot_diagnostics", False)):
            return

        joint_names = tuple(str(name) for name in term.diagnostic_joint_names)
        joint_mask = term.last_abnormal_joint_mask
        env_steps = term.last_abnormal_env_step
        physics_steps = term.last_abnormal_physics_step
        if not joint_names:
            raise RuntimeError("Abnormal-robot diagnostics require at least one joint name.")
        if tuple(joint_mask.shape) != (int(self.env.num_envs), len(joint_names)):
            raise RuntimeError(
                "Abnormal-robot diagnostic joint-mask shape does not match the environment: "
                f"mask={tuple(joint_mask.shape)}, envs={self.env.num_envs}, joints={len(joint_names)}."
            )
        for label, values in (("env steps", env_steps), ("physics steps", physics_steps)):
            if values.numel() != int(self.env.num_envs):
                raise RuntimeError(
                    f"Abnormal-robot diagnostic {label} count does not match the environment: "
                    f"values={values.numel()}, envs={self.env.num_envs}."
                )

        self._abnormal_robot_term = term
        self._abnormal_robot_joint_names = joint_names
        # Fields: env-step sum, physics-step sum, event count, per-joint first-event counts.
        self._abnormal_robot_iteration_stats = torch.zeros(
            3 + len(joint_names),
            device=self.device,
            dtype=torch.float32,
        )

    def _record_abnormal_robot_diagnostics(self) -> None:
        """Accumulate exact first-abnormal sums and joint counts for one environment step."""

        if self._abnormal_robot_iteration_stats is None or self._abnormal_robot_term is None:
            return
        env_steps = self._abnormal_robot_term.last_abnormal_env_step.detach().reshape(-1).to(
            device=self.device,
            dtype=torch.float32,
        )
        event_mask = env_steps > 0
        physics_steps = self._abnormal_robot_term.last_abnormal_physics_step.detach().reshape(-1).to(
            device=self.device,
            dtype=torch.float32,
        )
        joint_mask = self._abnormal_robot_term.last_abnormal_joint_mask.detach().to(
            device=self.device,
            dtype=torch.float32,
        )
        expected_shape = (int(self.env.num_envs), len(self._abnormal_robot_joint_names))
        if tuple(joint_mask.shape) != expected_shape:
            raise RuntimeError(
                "Abnormal-robot diagnostic joint-mask shape changed during rollout: "
                f"mask={tuple(joint_mask.shape)}, expected={expected_shape}."
            )

        self._abnormal_robot_iteration_stats[0] += env_steps[event_mask].sum()
        self._abnormal_robot_iteration_stats[1] += physics_steps[event_mask].sum()
        self._abnormal_robot_iteration_stats[2] += event_mask.sum()
        self._abnormal_robot_iteration_stats[3:] += joint_mask[event_mask].sum(dim=0)

    def _finalize_abnormal_robot_stats(self) -> torch.Tensor | None:
        """Reduce the current iteration across ranks and reset local accumulators."""

        if self._abnormal_robot_iteration_stats is None:
            return None
        stats = self._abnormal_robot_iteration_stats.clone()
        if (
            self.is_distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        self._abnormal_robot_iteration_stats.zero_()
        return stats

    def _log_abnormal_robot_stats(self, stats: torch.Tensor | None, iteration: int) -> None:
        """Write Stable V2-test abnormal event, timing, and first-joint metrics."""

        if self.writer is None or stats is None:
            return
        event_count = float(stats[2].item())
        self.writer.add_scalar("AbnormalRobot/events", event_count, iteration)
        for joint_index, joint_name in enumerate(self._abnormal_robot_joint_names):
            joint_count = float(stats[3 + joint_index].item())
            self.writer.add_scalar(
                f"AbnormalRobot/first_joint_count/{joint_name}",
                joint_count,
                iteration,
            )
        if event_count <= 0.0:
            return

        self.writer.add_scalar(
            "AbnormalRobot/mean_env_step_since_reset",
            float(stats[0].item()) / event_count,
            iteration,
        )
        self.writer.add_scalar(
            "AbnormalRobot/mean_physics_step_since_reset",
            float(stats[1].item()) / event_count,
            iteration,
        )
        for joint_index, joint_name in enumerate(self._abnormal_robot_joint_names):
            joint_count = float(stats[3 + joint_index].item())
            self.writer.add_scalar(
                f"AbnormalRobot/first_joint_rate/{joint_name}",
                joint_count / event_count,
                iteration,
            )

    def _augment_obs_for_residual(self, obs, base_action_flat: torch.Tensor | None):
        """Return a copy of `obs` with `residual_base_action_obs_key` added.

        The added key has shape [num_envs, H * num_actions] where H = residual_base_action_history.
        If base_action_flat is None, it is initialized to zeros (used during algorithm construction).
        """
        if base_action_flat is None:
            sample_group = self.cfg["obs_groups"]["policy"][0]
            sample = obs[sample_group]
            base_action_flat = torch.zeros(
                (sample.shape[0], self.residual_base_action_history * int(self.env.num_actions)),
                device=sample.device,
                dtype=sample.dtype,
            )
        obs_aug = obs.copy()
        obs_aug[self.residual_base_action_obs_key] = base_action_flat
        return obs_aug

    def _infer_env_history_length(self) -> int | None:
        """Try to infer the env's stacked-history length from env.cfg.observations.*.history_length.

        This makes residual_base_action_history follow the env's own observation history stacking by default.
        If multiple groups are used by the policy/critic and have different history_length, we pick the max
        and print a warning.
        """
        cfg = getattr(self.env, "cfg", None)
        observations_cfg = getattr(cfg, "observations", None) if cfg is not None else None
        if observations_cfg is None:
            return None

        # Look up history_length for the obs groups that the policy/critic actually uses.
        groups_used = set(self.cfg["obs_groups"].get("policy", [])) | set(self.cfg["obs_groups"].get("critic", []))
        lengths: list[int] = []
        for group in groups_used:
            group_cfg = getattr(observations_cfg, group, None)
            h = getattr(group_cfg, "history_length", None) if group_cfg is not None else None
            if h is not None:
                try:
                    lengths.append(int(h))
                except Exception:
                    continue
        if not lengths:
            return None
        if len(set(lengths)) > 1:
            print(
                "[WARN] ResidualRL: env observation groups have different history_length values "
                f"{sorted(set(lengths))}. Using max={max(lengths)} for residual_base_action_history."
            )
        return max(lengths)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        self._apply_env_training_iteration(start_iter)

        # start learning
        obs_raw = self.env.get_observations().to(self.device)
        # If enabled, initialize base-action history and build augmented observation for the residual policy.
        if self.residual_base_policy is not None and self.residual_base_action_as_obs:
            self._residual_base_action_hist = torch.zeros(
                (self.env.num_envs, self.residual_base_action_history, int(self.env.num_actions)),
                device=self.device,
                dtype=torch.float32,
            )
            with torch.no_grad():
                base_a0 = self.residual_base_policy.act_inference(obs_raw).detach()
            # Fill the entire history with the initial base action for consistency.
            self._residual_base_action_hist[:] = base_a0.unsqueeze(1).repeat(1, self.residual_base_action_history, 1)
            base_action_flat0 = self._residual_base_action_hist.reshape(self.env.num_envs, -1)
            obs = self._augment_obs_for_residual(obs_raw, base_action_flat=base_action_flat0)
        else:
            obs = obs_raw
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        v2_gravity_curriculum_enabled = self._v2_gravity_curriculum_state is not None
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            # If each rank maintains task-specific obs normalizers, do NOT overwrite them via broadcast.
            self.alg.broadcast_parameters(exclude_obs_normalizers=self.per_task_obs_normalizer)

        # Start training
        for it in range(start_iter, tot_iter):
            if it != start_iter:
                self._apply_env_training_iteration(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Residual RL: execute composed action in the environment, but keep PPO storage on residual actions.
                    if self.residual_base_policy is not None:
                        # Base policy expects *raw* env observations (no residual augmentation).
                        base_actions = self.residual_base_policy.act_inference(obs_raw).detach()
                        actions_env = base_actions + (self.residual_alpha * actions)
                        if self.residual_clip_actions:
                            actions_env = torch.clamp(actions_env, -1.0, 1.0)
                    else:
                        actions_env = actions
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions_env.to(self.env.device))
                    # Move to device
                    obs_raw, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self._record_terminal_object_success(dones)
                    self._record_abnormal_robot_diagnostics()
                    # If enabled, compute base action for the *next* observation and update the history stack.
                    if self.residual_base_policy is not None and self.residual_base_action_as_obs:
                        with torch.no_grad():
                            base_actions_next = self.residual_base_policy.act_inference(obs_raw).detach()
                        # Reset history for environments that terminated.
                        if (dones > 0).any():
                            done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                            self._residual_base_action_hist[done_ids] = base_actions_next[done_ids].unsqueeze(1).repeat(
                                1, self.residual_base_action_history, 1
                            )
                        # Roll and append.
                        if self.residual_base_action_history > 1:
                            self._residual_base_action_hist = torch.roll(self._residual_base_action_hist, shifts=-1, dims=1)
                        self._residual_base_action_hist[:, -1, :] = base_actions_next
                        base_action_flat = self._residual_base_action_hist.reshape(self.env.num_envs, -1)
                        obs = self._augment_obs_for_residual(obs_raw, base_action_flat=base_action_flat)
                    else:
                        obs = obs_raw
                    # process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # book keeping
                    if self.log_dir is not None or v2_gravity_curriculum_enabled:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                    if self.log_dir is not None:
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                self.alg.compute_returns(obs)

            v2_gravity_success_rate = self._update_v2_gravity_curriculum(ep_infos, it)
            object_success_stats = self._finalize_object_success_stats()
            abnormal_robot_stats = self._finalize_abnormal_robot_stats()

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                self._log_v2_gravity_curriculum(it, v2_gravity_success_rate)
                self._log_object_success_stats(object_success_stats, it)
                self._log_abnormal_robot_stats(abnormal_robot_stats, it)
                # Save model
                if it % self.save_interval == 0:
                    # In per-task normalizer mode, all ranks must participate in saving (for all_gather),
                    # but only rank0 writes the actual file. The call below is safe for both modes.
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            elif self.log_dir is not None and self.per_task_obs_normalizer and self.is_distributed:
                # Participate in distributed save even when this rank isn't writing logs.
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            if not self.disable_logs:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            elif self.per_task_obs_normalizer and self.is_distributed:
                # Participate so rank0 can package all per-task normalizers.
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _apply_env_training_iteration(self, iteration: int) -> None:
        """Publish the RSL-RL iteration and trigger optional environment events."""

        self._publish_v2_gravity_curriculum_state()
        self.env.set_iteration(iteration)
        env = getattr(self.env, "unwrapped", self.env)
        env._rsl_rl_training_iteration = int(iteration)
        event_manager = getattr(env, "event_manager", None)
        if event_manager is not None and "training_iteration" in event_manager.available_modes:
            event_manager.apply(mode="training_iteration")

    def _v2_gravity_curriculum_cfg(self):
        """Return the optional V2 teacher gravity curriculum configuration."""

        env = getattr(self.env, "unwrapped", self.env)
        return getattr(getattr(env, "cfg", None), "v2_gravity_curriculum", None)

    def _publish_v2_gravity_curriculum_state(self) -> None:
        """Make runner-owned trigger state available to environment events."""

        if self._v2_gravity_curriculum_state is None:
            return
        env = getattr(self.env, "unwrapped", self.env)
        env._v2_gravity_curriculum_trigger_iteration = self._v2_gravity_curriculum_state["trigger_iteration"]

    def _v2_gravity_curriculum_values(self, iteration: int) -> tuple[float, float] | None:
        """Return ``(progress, gravity_z)`` for a V2 curriculum iteration."""

        curriculum_cfg = self._v2_gravity_curriculum_cfg()
        if curriculum_cfg is None or self._v2_gravity_curriculum_state is None:
            return None
        trigger_iteration = self._v2_gravity_curriculum_state["trigger_iteration"]
        if trigger_iteration is None:
            progress = 0.0
        else:
            ramp_iterations = int(curriculum_cfg.ramp_iterations)
            progress = min(max((int(iteration) - int(trigger_iteration)) / ramp_iterations, 0.0), 1.0)
        initial_gravity_z = float(curriculum_cfg.initial_gravity_z)
        target_gravity_z = float(curriculum_cfg.target_gravity_z)
        gravity_z = initial_gravity_z + progress * (target_gravity_z - initial_gravity_z)
        return progress, gravity_z

    @staticmethod
    def _v2_gravity_success_sum_and_count(ep_infos, device) -> torch.Tensor:
        """Extract V2 completed-episode success values from a rollout's info dictionaries."""

        values = []
        for ep_info in ep_infos:
            if not isinstance(ep_info, dict):
                continue
            success = ep_info.get("Metrics/object_pose/success")
            if success is None:
                continue
            if isinstance(success, torch.Tensor):
                value = success.detach().to(device=device, dtype=torch.float32).reshape(-1)
            else:
                value = torch.as_tensor(success, device=device, dtype=torch.float32).reshape(-1)
            if value.numel() > 0:
                values.append(value)

        if not values:
            return torch.zeros(2, device=device, dtype=torch.float32)
        successes = torch.cat(values)
        return torch.stack((successes.sum(), successes.new_tensor(float(successes.numel()))))

    def _update_v2_gravity_curriculum(self, ep_infos, iteration: int) -> float | None:
        """Latch the V2 gravity curriculum on its first successful rollout."""

        curriculum_cfg = self._v2_gravity_curriculum_cfg()
        if curriculum_cfg is None or self._v2_gravity_curriculum_state is None:
            return None

        success_stats = self._v2_gravity_success_sum_and_count(ep_infos, self.device)
        if (
            self.is_distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(success_stats, op=torch.distributed.ReduceOp.SUM)

        success_sum = float(success_stats[0].item())
        success_count = float(success_stats[1].item())
        if success_count == 0.0:
            return None
        success_rate = float((success_stats[0] / success_stats[1]).item())
        self._v2_gravity_curriculum_state["last_success_rate"] = success_rate
        if (
            self._v2_gravity_curriculum_state["trigger_iteration"] is None
            # Compare aggregate counts directly.  A float32 rate for exactly
            # 3/5 is slightly above 0.60 and would otherwise trigger a
            # curriculum whose contract requires a strictly higher rate.
            and success_sum > float(curriculum_cfg.success_threshold) * success_count
        ):
            self._v2_gravity_curriculum_state["trigger_iteration"] = int(iteration)
        return success_rate

    def _v2_gravity_curriculum_checkpoint_state(self) -> dict | None:
        """Return a serializable copy of the V2 gravity curriculum state."""

        if self._v2_gravity_curriculum_state is None:
            return None
        return dict(self._v2_gravity_curriculum_state)

    def _restore_v2_gravity_curriculum_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore the V2 curriculum latch so resumed runs keep the same ramp."""

        if self._v2_gravity_curriculum_state is None:
            return
        state = checkpoint.get("v2_gravity_curriculum_state")
        if state is None:
            print(
                "[WARN] Checkpoint has no V2 gravity curriculum state; "
                "starting from the configured initial gravity."
            )
            return
        if not isinstance(state, dict):
            raise ValueError("v2_gravity_curriculum_state in checkpoint must be a dictionary.")
        trigger_iteration = state.get("trigger_iteration")
        if trigger_iteration is not None:
            trigger_iteration = int(trigger_iteration)
        last_success_rate = state.get("last_success_rate")
        if last_success_rate is not None:
            last_success_rate = float(last_success_rate)
        self._v2_gravity_curriculum_state = {
            "trigger_iteration": trigger_iteration,
            "last_success_rate": last_success_rate,
        }
        self._publish_v2_gravity_curriculum_state()

    def _log_v2_gravity_curriculum(self, iteration: int, success_rate: float | None) -> None:
        """Write V2 gravity-curriculum diagnostics alongside regular PPO metrics."""

        if self.writer is None or self._v2_gravity_curriculum_state is None:
            return
        values = self._v2_gravity_curriculum_values(iteration)
        if values is None:
            return
        progress, gravity_z = values
        trigger_iteration = self._v2_gravity_curriculum_state["trigger_iteration"]
        self.writer.add_scalar("GravityCurriculum/triggered", float(trigger_iteration is not None), iteration)
        self.writer.add_scalar(
            "GravityCurriculum/trigger_iteration",
            -1 if trigger_iteration is None else int(trigger_iteration),
            iteration,
        )
        self.writer.add_scalar("GravityCurriculum/progress", progress, iteration)
        self.writer.add_scalar("GravityCurriculum/gravity_z", gravity_z, iteration)
        if success_rate is not None:
            self.writer.add_scalar("GravityCurriculum/success_rate", success_rate, iteration)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        ep_info_means: dict[str, torch.Tensor] = {}
        if locs["ep_infos"]:
            # Key iteration strategy:
            # - Default (backwards-compatible): iterate keys from the first episode-info dict to preserve
            #   the historical terminal print order.
            # - Per-task normalizer mode: iterate over the union of keys across ep_infos (sorted) so we don't
            #   silently drop keys that are missing in ep_infos[0] (common in heterogeneous setups).
            if getattr(self, "per_task_obs_normalizer", False):
                all_keys: set[str] = set()
                for ep_info in locs["ep_infos"]:
                    if isinstance(ep_info, dict):
                        all_keys.update(ep_info.keys())
                keys_iter = sorted(all_keys)
            else:
                keys_iter = list(locs["ep_infos"][0].keys())

            for key in keys_iter:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if infotensor.numel() == 0:
                    # no data for this key in this logging window
                    continue
                value = torch.mean(infotensor)
                ep_info_means[key] = value
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

            # Convenience terminal print: success rate
            # Only use explicit 0/1 success indicators (NOT reward terms).
            # For dexsuite pose command tasks we expose: "Metrics/object_pose/success".
            success_keys = [
                "Metrics/object_pose/success",
                "success_rate",
                "Episode/success_rate",
                "success",
                "is_success",
                "Episode/success",
                "Episode/is_success",
                "Metrics/success",
                "Metrics/is_success",
                "Episode_Termination/success",
            ]
            success_tensor = None
            for k in success_keys:
                if k in ep_info_means:
                    success_tensor = ep_info_means[k]
                    break
            if success_tensor is not None:
                success_rate = float(success_tensor.item())
                self.writer.add_scalar("Train/success_rate", success_rate, locs["it"])
                ep_string += f"""{'Success rate:':>{pad}} {success_rate * 100.0:.2f}%\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        v2_gravity_curriculum_state = self._v2_gravity_curriculum_checkpoint_state()
        if v2_gravity_curriculum_state is not None:
            saved_dict["v2_gravity_curriculum_state"] = v2_gravity_curriculum_state
        # -- Save per-task obs normalizers (one entry per rank) when enabled
        if self.per_task_obs_normalizer and self.is_distributed and torch.distributed.is_initialized():
            local_entry = {
                "rank": self.gpu_global_rank,
                "task_tag": self._infer_task_tag(),
                "obs_normalizers": self._get_obs_normalizer_state(),
            }
            gathered: list[dict] = [None for _ in range(self.gpu_world_size)]  # type: ignore[list-item]
            torch.distributed.all_gather_object(gathered, local_entry)
            saved_dict["task_normalizers"] = gathered
        # -- Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # Only rank0 writes the file (others may call save() only to participate in all_gather).
        if not self.is_distributed or self.gpu_global_rank == 0:
            torch.save(saved_dict, path)

        # upload model to external logging service (rank0 only; writer is only initialized on non-disabled ranks)
        if (
            not self.disable_logs
            and hasattr(self, "logger_type")
            and self.logger_type in ["neptune", "wandb"]
            and self.writer is not None
        ):
            self.writer.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
        model_only: bool = False,
    ):
        """Load a checkpoint, optionally using only its policy model parameters.

        ``model_only=True`` is a warm start: it loads only ``model_state_dict`` and
        resets the learning iteration to zero. Optimizers and all other checkpoint
        training state remain untouched.
        """

        # Robust device mapping:
        # - Checkpoints saved during multi-GPU training may contain storages on cuda:1/cuda:2/...
        # - When loading on a machine with fewer GPUs, torch.load() must remap storages.
        if map_location is None:
            # Default to this runner's device when possible; otherwise fall back to CPU.
            map_location = self.device
            if isinstance(map_location, str) and map_location.startswith("cuda"):
                if not torch.cuda.is_available():
                    map_location = "cpu"
                else:
                    # If requested cuda:N doesn't exist, map to cuda:0.
                    try:
                        idx = int(map_location.split(":", 1)[1])
                        if idx >= torch.cuda.device_count():
                            map_location = "cuda:0"
                    except Exception:
                        map_location = "cuda:0"

        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if model_only:
            self.current_learning_iteration = 0
            return None
        # -- If checkpoint contains per-task normalizers, select the best match for this env and apply it.
        if "task_normalizers" in loaded_dict:
            try:
                self._apply_task_normalizer_from_checkpoint(loaded_dict["task_normalizers"])
            except Exception as e:
                # Don't fail hard on resume/play if matching fails; fall back to whatever was in model_state_dict.
                print(f"[WARN] Failed to apply per-task normalizer from checkpoint: {e}")
        # -- Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        self._restore_v2_gravity_curriculum_checkpoint_state(loaded_dict)
        return loaded_dict["infos"]

    def _torch_load_checkpoint(self, path: str, map_location: str | None = None) -> dict:
        """Load a checkpoint dict with robust device mapping (shared logic for residual/base policies too)."""
        if map_location is None:
            map_location = self.device
            if isinstance(map_location, str) and map_location.startswith("cuda"):
                if not torch.cuda.is_available():
                    map_location = "cpu"
                else:
                    try:
                        idx = int(map_location.split(":", 1)[1])
                        if idx >= torch.cuda.device_count():
                            map_location = "cuda:0"
                    except Exception:
                        map_location = "cuda:0"
        return torch.load(path, weights_only=False, map_location=map_location)

    def _get_obs_normalizer_state_from_policy(self, policy) -> dict:
        state: dict[str, dict] = {}
        if hasattr(policy, "actor_obs_normalizer"):
            state["actor"] = policy.actor_obs_normalizer.state_dict()
        if hasattr(policy, "critic_obs_normalizer"):
            state["critic"] = policy.critic_obs_normalizer.state_dict()
        return state

    def _set_obs_normalizer_state_into_policy(self, policy, state: dict):
        if state is None:
            return
        if "actor" in state and hasattr(policy, "actor_obs_normalizer"):
            policy.actor_obs_normalizer.load_state_dict(state["actor"], strict=False)
        if "critic" in state and hasattr(policy, "critic_obs_normalizer"):
            policy.critic_obs_normalizer.load_state_dict(state["critic"], strict=False)

    def _apply_task_normalizer_from_checkpoint_to_policy(self, policy, task_normalizers: list[dict]):
        """Pick the normalizer entry matching this env and load it into the provided policy."""
        current_tag = self._normalize_task_tag(self._infer_task_tag())
        chosen = None
        for entry in task_normalizers:
            if entry and self._normalize_task_tag(entry.get("task_tag", "")) == current_tag:
                chosen = entry
                break
        if chosen is None:
            for entry in task_normalizers:
                if entry and entry.get("rank") == 0:
                    chosen = entry
                    break
        if chosen is None:
            raise ValueError(f"No usable task_normalizers entries found for task_tag={current_tag}")
        self._set_obs_normalizer_state_into_policy(policy, chosen.get("obs_normalizers", {}))

    def _load_residual_base_policy(self, checkpoint_path: str):
        """Load a frozen base policy for residual RL from checkpoint_path."""
        ckpt = self._torch_load_checkpoint(checkpoint_path, map_location=None)
        # Build a separate policy instance that matches *raw* env observations (no residual augmentation).
        actor_critic_class_name = self.policy_cfg.get("class_name")
        if actor_critic_class_name is None:
            raise ValueError("Missing policy.class_name in train_cfg (required to build residual base policy).")
        actor_critic_class = eval(actor_critic_class_name)
        policy_kwargs = dict(self.policy_cfg)
        policy_kwargs.pop("class_name", None)
        base_policy: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            self._base_obs_example, self._base_obs_groups, self.env.num_actions, **policy_kwargs
        ).to(self.device)
        # Load weights/buffers; allow strict=False for compatibility across minor arch diffs.
        if "model_state_dict" in ckpt:
            base_policy.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            # fallback: treat the checkpoint itself as a state_dict
            base_policy.load_state_dict(ckpt, strict=False)
        # If checkpoint contains per-task normalizers, select and apply for this env.
        if "task_normalizers" in ckpt:
            try:
                self._apply_task_normalizer_from_checkpoint_to_policy(base_policy, ckpt["task_normalizers"])
            except Exception as e:
                print(f"[WARN] Failed to apply per-task normalizer to residual base policy: {e}")
        base_policy.eval()
        for p in base_policy.parameters():
            p.requires_grad_(False)
        self.residual_base_policy = base_policy

    def _infer_task_tag(self) -> str:
        """Infer a stable task identifier for selecting per-task normalizers."""
        cfg = getattr(self.env, "cfg", None)
        if cfg is not None:
            return cfg.__class__.__name__
        return self.env.__class__.__name__

    @staticmethod
    def _normalize_task_tag(tag: str) -> str:
        """Normalize task tags across TRAIN/PLAY variants.

        IsaacLab commonly uses different cfg classes for training vs play/eval (e.g. FooEnvCfg vs FooEnvCfg_PLAY).
        For per-task normalizers we want these to match.
        """
        if not tag:
            return tag
        # common suffixes
        for suffix in ("_PLAY", "_Play", "_EVAL", "_Eval", "_STUDENT", "_Student"):
            if tag.endswith(suffix):
                tag = tag[: -len(suffix)]
        return tag

    def _get_obs_normalizer_state(self) -> dict:
        """Extract observation-normalizer states from the policy (actor/critic) if present."""
        policy = self.alg.policy
        state: dict[str, dict] = {}
        if hasattr(policy, "actor_obs_normalizer"):
            state["actor"] = policy.actor_obs_normalizer.state_dict()
        if hasattr(policy, "critic_obs_normalizer"):
            state["critic"] = policy.critic_obs_normalizer.state_dict()
        return state

    def _set_obs_normalizer_state(self, state: dict):
        """Load observation-normalizer states into the policy (actor/critic) if present."""
        policy = self.alg.policy
        if state is None:
            return
        if "actor" in state and hasattr(policy, "actor_obs_normalizer"):
            policy.actor_obs_normalizer.load_state_dict(state["actor"], strict=False)
        if "critic" in state and hasattr(policy, "critic_obs_normalizer"):
            policy.critic_obs_normalizer.load_state_dict(state["critic"], strict=False)

    def _apply_task_normalizer_from_checkpoint(self, task_normalizers: list[dict]):
        """Pick the normalizer entry matching this env and load it into the policy."""
        current_tag = self._normalize_task_tag(self._infer_task_tag())
        # Prefer exact (normalized) task_tag match; otherwise fall back to rank0 entry.
        chosen = None
        for entry in task_normalizers:
            if entry and self._normalize_task_tag(entry.get("task_tag", "")) == current_tag:
                chosen = entry
                break
        if chosen is None:
            for entry in task_normalizers:
                if entry and entry.get("rank") == 0:
                    chosen = entry
                    break
        if chosen is None:
            raise ValueError(f"No usable task_normalizers entries found for task_tag={current_tag}")
        self._set_obs_normalizer_state(chosen.get("obs_normalizers", {}))

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
            if self.residual_base_policy is not None:
                self.residual_base_policy.to(device)

        # If residual RL is not enabled, return the plain policy inference function.
        if self.residual_base_policy is None or not self.residual_base_action_as_obs:
            return self.alg.policy.act_inference

        # Residual RL inference wrapper:
        # - base policy runs on raw env observations
        # - residual policy runs on augmented observations (raw obs + base-action history)
        # - env executes composed action: a_env = a_base + alpha * a_res
        hist = None  # type: ignore[var-annotated]

        def _policy(obs_raw):
            nonlocal hist
            base_actions = self.residual_base_policy.act_inference(obs_raw).detach()
            num_envs = int(base_actions.shape[0])
            action_dim = int(base_actions.shape[-1])

            # Allocate or re-allocate history buffer as needed.
            if (
                hist is None
                or hist.device != base_actions.device
                or hist.shape[0] != num_envs
                or hist.shape[1] != self.residual_base_action_history
                or hist.shape[2] != action_dim
            ):
                hist = base_actions.unsqueeze(1).repeat(1, self.residual_base_action_history, 1).clone()

            # Reset history for envs that have just been reset (episode_length == 0).
            # This mirrors env-side history buffers (oldest->newest) and avoids drift during PLAY loops.
            ep_len_buf = getattr(self.env, "episode_length_buf", None)
            if isinstance(ep_len_buf, torch.Tensor):
                try:
                    reset_ids = (ep_len_buf.to(base_actions.device) == 0).nonzero(as_tuple=False).squeeze(-1)
                    if reset_ids.numel() > 0:
                        hist[reset_ids] = base_actions[reset_ids].unsqueeze(1).repeat(
                            1, self.residual_base_action_history, 1
                        )
                except Exception:
                    pass

            # Update sliding window: oldest -> newest, newest at tail.
            if self.residual_base_action_history > 1:
                hist = torch.roll(hist, shifts=-1, dims=1)
            hist[:, -1, :] = base_actions

            base_action_flat = hist.reshape(num_envs, -1)
            obs_aug = self._augment_obs_for_residual(obs_raw, base_action_flat=base_action_flat)

            residual_actions = self.alg.policy.act_inference(obs_aug)
            actions_env = base_actions + (self.residual_alpha * residual_actions)
            if self.residual_clip_actions:
                actions_env = torch.clamp(actions_env, -1.0, 1.0)
            return actions_env

        return _policy

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs) -> PPO:
        """Construct the actor-critic algorithm."""
        # resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # initialize the actor-critic
        actor_critic_class_name = self.policy_cfg.get("class_name")
        if actor_critic_class_name is None:
            raise ValueError("Missing policy.class_name in train_cfg.")
        actor_critic_class = eval(actor_critic_class_name)
        policy_kwargs = dict(self.policy_cfg)
        policy_kwargs.pop("class_name", None)
        if policy_kwargs.get("left_action_terms") is not None and policy_kwargs.get("left_action_indices") is None:
            policy_kwargs["left_action_indices"] = self._resolve_action_term_indices(policy_kwargs["left_action_terms"])
        if policy_kwargs.get("right_action_terms") is not None and policy_kwargs.get("right_action_indices") is None:
            policy_kwargs["right_action_indices"] = self._resolve_action_term_indices(policy_kwargs["right_action_terms"])
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **policy_kwargs
        ).to(self.device)

        # initialize the algorithm
        alg_class_name = self.alg_cfg.get("class_name")
        if alg_class_name is None:
            raise ValueError("Missing algorithm.class_name in train_cfg.")
        alg_class = eval(alg_class_name)
        alg_kwargs = dict(self.alg_cfg)
        alg_kwargs.pop("class_name", None)
        alg: PPO = alg_class(actor_critic, device=self.device, **alg_kwargs, multi_gpu_cfg=self.multi_gpu_cfg)

        # initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _resolve_action_term_indices(self, action_terms: list[str]) -> list[int]:
        """Resolve ordered flat action indices for the requested action terms."""

        action_manager = getattr(self.env.unwrapped, "action_manager", None)
        if action_manager is None:
            raise ValueError("Dual-head policies require an environment with an action_manager.")

        active_terms = action_manager.active_terms
        term_dims = action_manager.action_term_dim
        term_to_indices: dict[str, list[int]] = {}
        start = 0
        for term_name, term_dim in zip(active_terms, term_dims, strict=True):
            stop = start + term_dim
            term_to_indices[term_name] = list(range(start, stop))
            start = stop

        resolved_indices: list[int] = []
        for term_name in action_terms:
            if term_name not in term_to_indices:
                raise ValueError(
                    f"Unknown action term '{term_name}'. Available action terms: {active_terms}"
                )
            resolved_indices.extend(term_to_indices[term_name])

        return resolved_indices

    def _prepare_logging_writer(self):
        """Prepares the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
