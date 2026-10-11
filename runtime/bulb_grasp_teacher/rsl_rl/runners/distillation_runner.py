# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import os
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import Distillation, HybridDistillationPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    StudentTeacher,
    StudentTeacherDepthDagger,
    StudentTeacherDepthDaggerV2,
    StudentTeacherDepthDaggerV4,
    StudentTeacherDepthDaggerV5,
    StudentTeacherDepthDiffusionV6,
    StudentTeacherDepthDiffusionPolicy,
    StudentTeacherDepthDiffusionPolicyV3,
    StudentTeacherDiffusionPolicy,
    StudentTeacherRecurrent,
    StudentTeacherTransformer,
    StudentTeacherV2Distillation,
    StudentTeacherV2TestDualCameraDistillation,
    StudentTeacherV2TestDistillation,
    StudentTeacherV2TestDistillationV4,
)
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import TeacherRolloutDatasetWriter
from rsl_rl.utils import resolve_obs_groups, store_code_state
from rsl_rl.modules.student_teacher_depth_dagger_v5 import (
    DUAL_DEPTH_CAMERA_ORDER,
    DUAL_DEPTH_CHANNELS,
    V2_TEST_POLICY_DEPTH_RANGE_M,
    V5_LEGACY_DEPTH_OUTPUT_SIZE,
    V5_POLICY_DEPTH_RANGE_M,
    dual_depth_checkpoint_shape,
    make_dual_depth_input_contract,
    make_v5_input_contract,
    make_vtgen_tactile_input_contract,
    validate_vtgen_tactile_input_contract,
    v5_checkpoint_depth_shape,
)
from rsl_rl.modules.student_teacher_v2_distillation import (
    make_v2_distillation_action_contract,
    make_v2_test_dual_camera_distillation_action_contract,
    make_v2_test_distillation_action_contract,
    make_v2_test_distillation_v5_action_contract,
    validate_v2_distillation_action_contract,
    validate_v2_test_dual_camera_distillation_action_contract,
    validate_v2_test_distillation_action_contract,
    validate_v2_test_distillation_v5_action_contract,
    v2_test_distillation_checkpoint_action_contract_version,
)


HYBRID_STABLE_REWARD_CONTRACT = {
    "name": "stable_v2_test_carry_stability",
    "carry_stability_terms": [
        "carry_contact_quality",
        "carry_hold",
        "carry_relative_lin_vel_l2",
        "carry_relative_ang_vel_l2",
        "drop_after_lift",
    ],
    "disabled_terms": [
        "table_clearance",
        "arm_joint_velocity_soft_limit",
        "arm_joint_acceleration_soft_limit",
        "hand_joint_velocity_soft_limit",
    ],
    "palm_down_penalty": -0.2,
    "abnormal_termination": "abnormal_robot_state_with_diagnostics",
}


class DistillationRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of teacher-student training."""

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
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets=["teacher"])

        # Distillation does not currently support residual RL. Keep these attributes defined so
        # inherited helper methods remain safe to call during play/save/load.
        self.residual_base_checkpoint: str | None = None
        self.residual_alpha: float = 1.0
        self.residual_clip_actions: bool = False
        self.residual_base_action_as_obs: bool = False
        self.residual_base_policy = None
        # Most distillation tasks do not drive the V2 teacher gravity
        # curriculum. V2-test opts in by exposing the same environment config
        # as its teacher, so the inherited helpers can own and checkpoint it.
        self._v2_gravity_curriculum_state = (
            {"trigger_iteration": None, "last_success_rate": None}
            if self._v2_gravity_curriculum_cfg() is not None
            else None
        )

        # create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Keep feature parity with OnPolicyRunner.save(): when disabled this should still exist
        # so inherited checkpoint saving does not fail on attribute lookup.
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
        self.teacher_mix_schedule = self._validate_teacher_mix_schedule(
            self.cfg.get("teacher_mix_schedule", [])
        )
        self._teacher_episode_mask: torch.Tensor | None = None
        self.teacher_mix_mode = self.cfg.get("teacher_mix_mode", "step")
        self.teacher_mix_warmup_iterations = int(self.cfg.get("teacher_mix_warmup_iterations", 100))
        self.teacher_mix_zero_iteration = int(self.cfg.get("teacher_mix_zero_iteration", 3100))
        self.teacher_mix_force_zero_iteration = int(self.cfg.get("teacher_mix_force_zero_iteration", 3300))
        self.teacher_mix_eval_interval = int(self.cfg.get("teacher_mix_eval_interval", 50))
        self.teacher_mix_min_student_episodes = int(self.cfg.get("teacher_mix_min_student_episodes", 200))
        self.teacher_mix_recovery_increment = float(self.cfg.get("teacher_mix_recovery_increment", 0.10))
        self.teacher_mix_recovery_iterations = int(self.cfg.get("teacher_mix_recovery_iterations", 100))
        self.teacher_mix_success_drop_threshold = float(
            self.cfg.get("teacher_mix_success_drop_threshold", 0.10)
        )
        self.teacher_mix_oob_rise_threshold = float(self.cfg.get("teacher_mix_oob_rise_threshold", 0.03))
        self._teacher_mix_recovery_until = -1
        self._teacher_mix_bad_windows = 0
        self._stable_student_success: float | None = None
        self._stable_student_oob: float | None = None
        self._last_stable_checkpoint: str | None = None
        self._current_nominal_teacher_probability = 1.0
        self._current_actual_teacher_probability = 1.0
        self._dagger_role_stats = {
            "teacher": deque(maxlen=2000),
            "student": deque(maxlen=2000),
        }
        self._init_object_success_tracking()
        if isinstance(self.alg, HybridDistillationPPO):
            if (
                self.alg.actor_earliest_iteration
                < self.teacher_mix_zero_iteration
            ):
                raise ValueError(
                    "Hybrid PPO must start after DAgger teacher execution reaches zero: "
                    "actor_earliest_iteration="
                    f"{self.alg.actor_earliest_iteration}, "
                    f"teacher_mix_zero_iteration={self.teacher_mix_zero_iteration}."
                )
            self.hybrid_teacher_init_source: str | None = None
            self.hybrid_student_init_source: str | None = None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()
        # check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start from the checkpoint iteration so environment-side curricula are
        # resume-safe and do not restart from zero after loading a model.
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        self._apply_env_training_iteration(start_iter)
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        v2_gravity_curriculum_enabled = self._v2_gravity_curriculum_state is not None
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_success = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_oob = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_non_finite = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_used_teacher = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.device
        )

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        for it in range(start_iter, tot_iter):
            track_cuda_memory = torch.cuda.is_available() and torch.device(self.device).type == "cuda"
            if track_cuda_memory:
                torch.cuda.reset_peak_memory_stats(self.device)
            if it != start_iter:
                self._apply_env_training_iteration(it)
                # Recompute the current frame so the new curriculum strength is
                # active from the first rollout sample of this iteration.
                obs = self.env.get_observations().to(self.device)
            if hasattr(self.alg.policy, "set_training_iteration"):
                self.alg.policy.set_training_iteration(it)
            if hasattr(self.alg, "set_training_iteration"):
                self.alg.set_training_iteration(it)
            teacher_mix_probability = self.teacher_mix_probability(it)
            if (
                hasattr(self.alg.policy, "should_skip_distillation_update")
                and self.alg.policy.should_skip_distillation_update()
            ):
                teacher_mix_probability = 1.0
            if (
                getattr(self.alg.policy, "uses_recurrent_diffusion", False)
                and it >= self.teacher_mix_force_zero_iteration
                and self._teacher_episode_mask is not None
            ):
                # V6 reserves the final 200 iterations for strictly pure-student
                # rollout, including episodes that began before the boundary.
                self._teacher_episode_mask.fill_(False)
            teacher_action_count = 0.0
            rollout_action_count = 0
            action_sq_error_sum = torch.zeros(self.env.num_actions, device=self.device)
            action_abs_error_sum = torch.zeros(self.env.num_actions, device=self.device)
            normalized_action_sq_error_sum = 0.0
            student_action_sum = 0.0
            student_action_sq_sum = 0.0
            teacher_action_sum = 0.0
            teacher_action_sq_sum = 0.0
            student_action_min = float("inf")
            student_action_max = float("-inf")
            teacher_action_min = float("inf")
            teacher_action_max = float("-inf")
            action_error_samples = 0
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    student_mask = None
                    if getattr(self.alg.policy, "supports_masked_rollout", False):
                        teacher_episode_mask = self._ensure_teacher_episode_mask(
                            self.env.num_envs,
                            teacher_mix_probability,
                            obs.device,
                        )
                        student_mask = ~teacher_episode_mask
                    student_actions = self.alg.act(obs, student_mask=student_mask)
                    teacher_actions = self.alg.transition.privileged_actions
                    # Masked V6 rollout returns teacher actions for teacher-only
                    # rows.  Only rows where diffusion was the behavior policy
                    # are meaningful student/teacher error samples.
                    action_metric_mask = getattr(self.alg.policy, "last_student_action_mask", None)
                    if action_metric_mask is None:
                        action_metric_mask = torch.ones(
                            student_actions.shape[0], dtype=torch.bool, device=student_actions.device
                        )
                    else:
                        action_metric_mask = action_metric_mask.reshape(-1).bool()
                    metric_student_actions = student_actions[action_metric_mask]
                    metric_teacher_actions = teacher_actions[action_metric_mask]
                    if metric_student_actions.numel() > 0:
                        action_sq_error_sum += (
                            (metric_student_actions - metric_teacher_actions) ** 2
                        ).sum(dim=0)
                        action_abs_error_sum += (
                            metric_student_actions - metric_teacher_actions
                        ).abs().sum(dim=0)
                        student_action_sum += float(metric_student_actions.sum().item())
                        student_action_sq_sum += float(metric_student_actions.pow(2).sum().item())
                        teacher_action_sum += float(metric_teacher_actions.sum().item())
                        teacher_action_sq_sum += float(metric_teacher_actions.pow(2).sum().item())
                        student_action_min = min(
                            student_action_min, float(metric_student_actions.min().item())
                        )
                        student_action_max = max(
                            student_action_max, float(metric_student_actions.max().item())
                        )
                        teacher_action_min = min(
                            teacher_action_min, float(metric_teacher_actions.min().item())
                        )
                        teacher_action_max = max(
                            teacher_action_max, float(metric_teacher_actions.max().item())
                        )
                        if hasattr(self.alg.policy, "normalize_actions"):
                            normalized_error = self.alg.policy.normalize_actions(
                                metric_student_actions, clip=False
                            ) - self.alg.policy.normalize_actions(metric_teacher_actions, clip=False)
                            normalized_action_sq_error_sum += float(normalized_error.pow(2).sum().item())
                        action_error_samples += int(metric_student_actions.shape[0])
                    actions, used_teacher = self._mix_rollout_actions(
                        student_actions, teacher_actions, teacher_mix_probability
                    )
                    cur_episode_used_teacher.logical_or_(used_teacher.to(self.device))
                    self.alg.record_teacher_execution_mask(used_teacher)
                    teacher_action_count += float(used_teacher.sum().item())
                    rollout_action_count += int(used_teacher.numel())
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self._record_terminal_object_success(
                        dones,
                        episode_used_teacher=cur_episode_used_teacher,
                    )
                    completed_episode_ids = (dones > 0).nonzero(as_tuple=False)
                    # process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # book keeping
                    if self.log_dir is not None or v2_gravity_curriculum_enabled:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        self._accumulate_dagger_episode_metrics(
                            extras,
                            cur_episode_success,
                            cur_episode_oob,
                            cur_episode_non_finite,
                        )
                        self._record_dagger_episode_outcomes(
                            extras=extras,
                            dones=dones,
                            used_teacher=used_teacher,
                            reward_sums=cur_reward_sum,
                            episode_success=cur_episode_success,
                            episode_oob=cur_episode_oob,
                            episode_non_finite=cur_episode_non_finite,
                        )
                        # Clear data for completed episodes
                        new_ids = completed_episode_ids
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        cur_episode_success[new_ids] = 0
                        cur_episode_oob[new_ids] = 0
                        cur_episode_non_finite[new_ids] = 0
                    cur_episode_used_teacher[completed_episode_ids] = False
                    self._resample_teacher_mask_for_dones(dones, teacher_mix_probability)

                stop = time.time()
                collection_time = stop - start
                start = stop

            v2_gravity_success_rate = self._update_v2_gravity_curriculum(
                ep_infos,
                it,
            )
            object_success_stats = self._finalize_object_success_stats()

            # update policy
            if (
                hasattr(self.alg.policy, "should_skip_distillation_update")
                and self.alg.policy.should_skip_distillation_update()
            ):
                self.alg.discard_rollout()
                loss_dict = {"behavior": 0.0}
            else:
                if isinstance(self.alg, HybridDistillationPPO):
                    self.alg.compute_returns(obs)
                loss_dict = self.alg.update()
            loss_dict["teacher_mix_probability"] = teacher_mix_probability
            loss_dict["teacher_action_ratio"] = teacher_action_count / max(rollout_action_count, 1)
            if action_error_samples > 0:
                per_action_mse = action_sq_error_sum / action_error_samples
                loss_dict["student_teacher_action_mse"] = float(per_action_mse.mean().item())
                loss_dict["student_teacher_action_mae"] = float(
                    (action_abs_error_sum / action_error_samples).mean().item()
                )
                if hasattr(self.alg.policy, "normalize_actions"):
                    loss_dict["normalized_action_mse"] = normalized_action_sq_error_sum / (
                        action_error_samples * self.env.num_actions
                    )
                action_value_count = action_error_samples * self.env.num_actions
                student_mean = student_action_sum / action_value_count
                teacher_mean = teacher_action_sum / action_value_count
                loss_dict.update(
                    {
                        "student_action_min": student_action_min,
                        "student_action_max": student_action_max,
                        "student_action_mean": student_mean,
                        "student_action_std": max(
                            student_action_sq_sum / action_value_count - student_mean**2, 0.0
                        )
                        ** 0.5,
                        "teacher_action_min": teacher_action_min,
                        "teacher_action_max": teacher_action_max,
                        "teacher_action_mean": teacher_mean,
                        "teacher_action_std": max(
                            teacher_action_sq_sum / action_value_count - teacher_mean**2, 0.0
                        )
                        ** 0.5,
                    }
                )
                for action_index, value in enumerate(per_action_mse.tolist()):
                    loss_dict[f"action_{action_index}_mse"] = float(value)
            if hasattr(self.alg.policy, "diagnostics"):
                loss_dict.update(self.alg.policy.diagnostics())
            loss_dict["depth_encoder_grad_norm"] = float(
                getattr(self.alg, "last_depth_encoder_grad_norm", 0.0)
            )

            stop = time.time()
            learn_time = stop - start
            peak_cuda_memory_mib = (
                torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
                if track_cuda_memory
                else 0.0
            )
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                self.writer.add_scalar(
                    "Perf/peak_cuda_memory_mib", peak_cuda_memory_mib, it
                )
                print(
                    "[V5_DEPTH_PERF] "
                    f"iteration={it} "
                    f"iteration_s={collection_time + learn_time:.6f} "
                    f"peak_cuda_memory_mib={peak_cuda_memory_mib:.3f}",
                    flush=True,
                )
                self._log_dagger_diagnostics(it)
                self._log_env_curriculum(it)
                self._log_v2_gravity_curriculum(it, v2_gravity_success_rate)
                self._log_object_success_stats(
                    object_success_stats,
                    it,
                    include_distillation_roles=True,
                )
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            self._update_adaptive_teacher_mix(it)

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
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def learn_fixed_observation(self, num_learning_iterations: int, batch_size: int = 256):
        """Train the student against teacher actions on one frozen observation batch."""

        self._prepare_logging_writer()
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        obs = self.env.get_observations()
        if hasattr(obs, "to"):
            obs = obs.to(self.device)
        fixed_obs = self._clone_fixed_observations(obs)

        self.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.alg.policy.update_normalization(fixed_obs)

        with torch.inference_mode():
            privileged_actions = self.alg.policy.evaluate(fixed_obs).detach().clone()

        fixed_obs_size = int(privileged_actions.shape[0])
        print(
            "[INFO] Fixed-observation distillation debug: "
            f"num_obs={fixed_obs_size}, batch_size={batch_size}, "
            f"num_learning_epochs={self.alg.num_learning_epochs}"
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            loss_dict = self.alg.update_fixed_observation(fixed_obs, privileged_actions, batch_size=batch_size)
            learn_time = time.time() - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self._log_fixed_observation(
                    loss_dict=loss_dict,
                    it=it,
                    start_iter=start_iter,
                    tot_iter=tot_iter,
                    num_learning_iterations=num_learning_iterations,
                    learn_time=learn_time,
                    fixed_obs_size=fixed_obs_size,
                    batch_size=batch_size,
                )
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def collect_teacher_rollouts(  # noqa: C901
        self,
        *,
        num_successful_episodes: int,
        output_dir: str,
        task_name: str,
        teacher_checkpoint: str | None = None,
    ) -> dict[str, object]:
        """Collect pre-step student observations with deterministic teacher labels."""

        if not self.alg.policy.loaded_teacher:
            raise ValueError(
                "Teacher model parameters not loaded. Please load the teacher checkpoint "
                "before collecting rollouts."
            )
        num_successful_episodes = int(num_successful_episodes)
        if num_successful_episodes < 1:
            raise ValueError("num_successful_episodes must be positive.")
        if self.is_distributed:
            raise ValueError("Teacher-rollout collection does not support distributed runners.")
        if self.cfg.get("clip_actions") is not None:
            raise ValueError(
                "Teacher-rollout action alignment requires the task's unclipped RSL-RL "
                "wrapper (clip_actions=None)."
            )

        policy = self.alg.policy
        required_policy_attributes = (
            "student_low_dim_obs_group",
            "student_depth_obs_group",
            "student_secondary_depth_obs_group",
            "student_aux_obs_group",
            "effective_raw_action",
            "effective_raw_action_limits",
        )
        missing_attributes = [
            name for name in required_policy_attributes if not hasattr(policy, name)
        ]
        if missing_attributes:
            raise TypeError(
                "Teacher-rollout collection requires the v2-test dual-camera policy "
                f"contract; missing {missing_attributes}."
            )
        secondary_depth_group = policy.student_secondary_depth_obs_group
        if not secondary_depth_group:
            raise TypeError("Teacher-rollout collection requires a secondary depth observation.")

        noise_std = float(self.cfg.get("teacher_rollout_noise_std", 0.02))
        noise_clip = float(self.cfg.get("teacher_rollout_noise_clip", 0.05))
        noise_hold_steps = int(self.cfg.get("teacher_rollout_noise_hold_steps", 3))
        if not 0.0 <= noise_std <= noise_clip or noise_hold_steps < 1:
            raise ValueError(
                "Invalid teacher-rollout noise config: expected "
                f"0 <= std <= clip and hold_steps >= 1, got "
                f"std={noise_std}, clip={noise_clip}, hold_steps={noise_hold_steps}."
            )

        env = getattr(self.env, "unwrapped", self.env)
        command_term = env.command_manager.get_term("object_pose")
        action_term = env.action_manager.get_term("action")
        termination_manager = env.termination_manager
        object_indices = getattr(command_term, "v2_test_object_ids", None)
        object_names = tuple(getattr(command_term, "v2_test_object_names", ()))
        if not isinstance(object_indices, torch.Tensor) or not object_names:
            raise RuntimeError(
                "Teacher-rollout collection requires the enabled V2-test multi-object metadata."
            )
        object_indices = object_indices.detach().reshape(-1).to(
            device=self.device, dtype=torch.int64
        )
        if object_indices.numel() != int(self.env.num_envs):
            raise RuntimeError(
                "V2-test object metadata does not match the parallel environment count."
            )

        observation_keys = (
            policy.student_low_dim_obs_group,
            policy.student_depth_obs_group,
            secondary_depth_group,
            policy.student_aux_obs_group,
        )
        checkpoint_path = (
            os.path.abspath(os.path.expanduser(teacher_checkpoint))
            if teacher_checkpoint is not None
            else None
        )
        if checkpoint_path is not None and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Teacher checkpoint metadata path does not exist: {checkpoint_path}"
            )
        step_dt = float(getattr(env, "step_dt", 0.0))
        metadata = {
            "collection_mode": "teacher_rollout_collection",
            "random_seed": int(self.cfg.get("seed", 0)),
            "num_parallel_envs": int(self.env.num_envs),
            "device": str(self.device),
            "simulator_control": {
                "step_dt_s": step_dt if step_dt > 0.0 else None,
                "control_frequency_hz": (1.0 / step_dt) if step_dt > 0.0 else None,
            },
            "teacher_checkpoint": {
                "path": checkpoint_path,
                "sha256": (
                    self._sha256_file(checkpoint_path)
                    if checkpoint_path is not None
                    else None
                ),
            },
            "runner_config_snapshot": self._json_safe(self.cfg),
            "observation_keys": list(observation_keys),
            "depth_valid_mask_channel": 1,
            "camera_schema": {
                "order": list(DUAL_DEPTH_CAMERA_ORDER),
                "channels": list(DUAL_DEPTH_CHANNELS),
                "per_camera_shape": [2, *list(policy.depth_output_size)],
                "depth_range_m": list(V2_TEST_POLICY_DEPTH_RANGE_M),
                "storage_dtype": "float32",
                "frame_id_semantics": "environment common_step_counter",
            },
            "object_index_to_name": {
                str(index): name for index, name in enumerate(object_names)
            },
            "abnormal": {
                "enabled": "abnormal_robot" in termination_manager.active_terms,
                "termination_terms": list(termination_manager.active_terms),
            },
            "noise": {
                "coordinate_space": "per_dimension_normalized_effective_raw",
                "std": noise_std,
                "clip": noise_clip,
                "hold_steps": noise_hold_steps,
            },
            "action_semantics": {
                "teacher_target_action": (
                    "deterministic effective-raw teacher label used by online DAgger"
                ),
                "behavior_action": (
                    "teacher_target_action plus held bounded noise, passed to env.step"
                ),
                "raw_action": (
                    "exact raw action received by the action manager before action-term "
                    "scale and clip; equal to behavior_action because clip_actions=None"
                ),
                "executed_action": (
                    "effective-raw action after the existing per-joint action clip"
                ),
                "applied_action": (
                    "action-term processed action after existing scale and clip"
                ),
                "contract": self._json_safe(policy.action_contract()),
                "action_term_scale": self._json_safe(
                    getattr(getattr(action_term, "cfg", None), "scale", None)
                ),
                "action_term_clip": self._json_safe(
                    getattr(getattr(action_term, "cfg", None), "clip", None)
                ),
            },
        }
        writer = TeacherRolloutDatasetWriter(
            output_dir,
            num_envs=self.env.num_envs,
            task_name=task_name,
            metadata=metadata,
        )
        normalizer = getattr(policy, "student_obs_normalizer", None)
        if isinstance(normalizer, torch.nn.Module):
            writer.metadata["student_observation_normalizer"] = writer.write_artifact(
                "student_obs_normalizer.pt",
                {
                    "class": type(normalizer).__name__,
                    "state_dict": normalizer.state_dict(),
                    "observation_group": policy.student_low_dim_obs_group,
                    "applied_during_collection": False,
                    "stored_observations": "raw_pre_normalizer",
                },
            )
        writer.finalize(collection_complete=False)

        obs = self.env.get_observations()
        if hasattr(obs, "to"):
            obs = obs.to(self.device)
        missing_observations = [key for key in observation_keys if key not in obs]
        if missing_observations:
            raise KeyError(
                f"Teacher-rollout observation groups are missing: {missing_observations}."
            )

        self.eval_mode()
        episode_success = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.device
        )
        current_noise = torch.zeros(
            self.env.num_envs,
            self.env.num_actions,
            dtype=obs[policy.student_low_dim_obs_group].dtype,
            device=self.device,
        )
        noise_generator = torch.Generator(device=torch.device(self.device))
        noise_generator.manual_seed(int(self.cfg.get("seed", 0)) + 9_731)

        print(
            "[INFO] Teacher-rollout collection ACTIVE: teacher controls every env; "
            "student inference and optimizer updates are disabled."
        )
        print(
            "[INFO] Teacher behavior noise in normalized effective-raw coordinates: "
            f"std={noise_std:g}, clip={noise_clip:g}, hold_steps={noise_hold_steps}."
        )
        print(f"[INFO] Teacher-rollout dataset: {writer.output_dir}")

        with torch.inference_mode():
            while writer.num_saved_episodes < num_successful_episodes:
                control_step = int(getattr(env, "common_step_counter", -1))
                step_ids = torch.tensor(
                    [writer.step_id(env_id) for env_id in range(self.env.num_envs)],
                    dtype=torch.int64,
                    device=self.device,
                )
                refresh_noise = step_ids.remainder(noise_hold_steps).eq(0)

                # This is the exact deterministic teacher target used to fill
                # Distillation.transition.privileged_actions in online DAgger.
                teacher_target_action = policy.evaluate(obs).detach().clone()
                action_limits = policy.effective_raw_action_limits(
                    teacher_target_action
                )
                sampled_normalized_noise = torch.randn(
                    teacher_target_action.shape,
                    dtype=teacher_target_action.dtype,
                    device=teacher_target_action.device,
                    generator=noise_generator,
                ).mul_(noise_std).clamp_(-noise_clip, noise_clip)
                sampled_noise = sampled_normalized_noise * action_limits
                current_noise[refresh_noise] = sampled_noise[refresh_noise]
                behavior_action = teacher_target_action + current_noise
                raw_action = behavior_action.detach().clone()
                executed_action = policy.effective_raw_action(raw_action)

                # Materialize every input/label field before env.step so later
                # observation-buffer updates cannot turn x_t into x_{t+1}.
                pre_step_samples = []
                for env_id in range(self.env.num_envs):
                    student_observations = {
                        key: obs[key][env_id].detach().clone()
                        for key in observation_keys
                    }
                    primary_depth = student_observations[
                        policy.student_depth_obs_group
                    ]
                    secondary_depth = student_observations[secondary_depth_group]
                    pre_step_samples.append(
                        {
                            "observations": student_observations,
                            "depth_valid_mask": primary_depth[1].bool(),
                            "secondary_depth_valid_mask": secondary_depth[1].bool(),
                            "teacher_target_action": teacher_target_action[
                                env_id
                            ].clone(),
                            "behavior_action": behavior_action[env_id].clone(),
                            "action_noise": current_noise[env_id].clone(),
                            "raw_action": raw_action[env_id].clone(),
                            "executed_action": executed_action[env_id].clone(),
                            "control_step": torch.tensor(
                                control_step, dtype=torch.int64
                            ),
                            "camera_frame_id": torch.tensor(
                                control_step, dtype=torch.int64
                            ),
                            "secondary_camera_frame_id": torch.tensor(
                                control_step, dtype=torch.int64
                            ),
                            "object_id": object_indices[env_id].clone(),
                            "object_index": object_indices[env_id].clone(),
                        }
                    )

                next_obs, rewards, dones, _ = self.env.step(
                    raw_action.to(self.env.device)
                )
                # JointAction.reset() clears raw actions for completed rows but
                # deliberately leaves processed_actions intact. Reading this
                # buffer therefore preserves the terminal applied action without
                # enabling RecorderManager, whose pre-reset observation compute
                # would disturb camera/cache and temporal-history alignment.
                applied_action = action_term.processed_actions.detach().to(
                    device=self.device
                ).clone()
                dones = dones.to(device=self.device).reshape(-1).bool()
                live_success = command_term.metrics["success"].to(
                    device=self.device
                ).reshape(-1).bool()
                terminal_success = command_term.terminal_success.to(
                    device=self.device
                ).reshape(-1).bool()
                step_success = torch.where(dones, terminal_success, live_success)
                episode_success.logical_or_(step_success)
                termination = {
                    name: termination_manager.get_term(name)
                    .detach()
                    .to(device=self.device)
                    .reshape(-1)
                    .bool()
                    .clone()
                    for name in termination_manager.active_terms
                }
                timeouts = termination.get(
                    "time_out",
                    torch.zeros_like(dones),
                )
                abnormal = termination.get(
                    "abnormal_robot",
                    torch.zeros_like(dones),
                )

                for env_id in range(self.env.num_envs):
                    object_index = int(object_indices[env_id].item())
                    sample = pre_step_samples[env_id]
                    sample.update(
                        {
                            "applied_action": applied_action[env_id],
                            "done": dones[env_id],
                            "timeout": timeouts[env_id],
                            "reward": rewards[env_id].to(device=self.device),
                            "abnormal_robot": abnormal[env_id],
                            "termination": {
                                name: value[env_id]
                                for name, value in termination.items()
                            },
                        }
                    )
                    writer.append(
                        env_id,
                        sample,
                    )

                    if dones[env_id]:
                        active_terms = sorted(
                            name
                            for name, value in termination.items()
                            if bool(value[env_id].item())
                        )
                        failure_reason = "+".join(active_terms) or "unsuccessful"
                        output_path = writer.complete_episode(
                            env_id,
                            success=bool(episode_success[env_id].item()),
                            failure_reason=failure_reason,
                            object_name=object_names[object_index],
                            # Save every success completed by the final vectorized
                            # step. The requested count is therefore a lower bound;
                            # discarding same-step surplus successes would violate
                            # the complete-success filtering contract.
                            save_success=True,
                        )
                        episode_success[env_id] = False
                        if output_path is not None:
                            print(
                                "[INFO] Saved successful teacher episode "
                                f"{writer.num_saved_episodes}/{num_successful_episodes}: "
                                f"{output_path.name}"
                            )

                next_obs = next_obs.to(self.device) if hasattr(next_obs, "to") else next_obs
                next_last_action = next_obs[policy.student_low_dim_obs_group][
                    :, -self.env.num_actions :
                ]
                if (~dones).any() and not torch.allclose(
                    next_last_action[~dones],
                    executed_action[~dones],
                    rtol=0.0,
                    atol=1.0e-5,
                ):
                    raise RuntimeError(
                        "Student last_action does not match the previous executed action."
                    )
                if dones.any() and not torch.allclose(
                    next_last_action[dones],
                    torch.zeros_like(next_last_action[dones]),
                    rtol=0.0,
                    atol=1.0e-6,
                ):
                    raise RuntimeError(
                        "Student last_action was not cleared by episode reset."
                    )
                obs = next_obs

        manifest_path = writer.finalize()
        result = {
            "output_dir": str(writer.output_dir),
            "manifest": str(manifest_path),
            "saved_successful_episodes": writer.num_saved_episodes,
            "completed_episodes": writer.completed_episodes,
            "discarded_incomplete_episodes": writer.incomplete_episodes,
            "discarded_episode_reasons": dict(writer.failed_episode_reasons),
        }
        print(f"[INFO] Teacher-rollout collection complete: {result}")
        return result

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _json_safe(cls, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return repr(value)

    def get_inference_policy(self, device=None):
        """Return the student policy for deterministic inference."""
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    """
    Helper methods.
    """

    def _construct_algorithm(self, obs) -> Distillation:
        """Construct the distillation algorithm."""
        # initialize the actor-critic
        student_teacher_class = eval(self.policy_cfg.pop("class_name"))
        student_teacher: (
            StudentTeacher
            | StudentTeacherRecurrent
            | StudentTeacherDiffusionPolicy
            | StudentTeacherDepthDiffusionPolicy
            | StudentTeacherDepthDaggerV2
            | StudentTeacherDepthDaggerV4
            | StudentTeacherDepthDaggerV5
            | StudentTeacherDepthDiffusionPolicyV3
            | StudentTeacherDepthDiffusionV6
            | StudentTeacherV2Distillation
            | StudentTeacherV2TestDualCameraDistillation
            | StudentTeacherV2TestDistillation
            | StudentTeacherV2TestDistillationV4
        ) = student_teacher_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: Distillation = alg_class(
            student_teacher, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # initialize the storage
        alg.init_storage(
            "hybrid" if isinstance(alg, HybridDistillationPPO) else "distillation",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    @staticmethod
    def _validate_teacher_mix_schedule(schedule) -> list[tuple[int, float]]:
        validated = [(int(start), float(probability)) for start, probability in schedule]
        if not validated:
            return []
        if validated[0][0] != 0 or any(a[0] >= b[0] for a, b in zip(validated, validated[1:])):
            raise ValueError("teacher_mix_schedule must start at iteration 0 with increasing boundaries.")
        if any(not 0.0 <= probability <= 1.0 for _, probability in validated):
            raise ValueError("teacher mixing probabilities must be in [0, 1].")
        return validated

    def teacher_mix_probability(self, iteration: int) -> float:
        teacher_mix_mode = getattr(self, "teacher_mix_mode", "step")
        if teacher_mix_mode in ("linear", "linear_adaptive"):
            nominal = self.nominal_teacher_mix_probability(iteration)
            probability = nominal
            if teacher_mix_mode == "linear_adaptive":
                recovery_offset = (
                    self.teacher_mix_recovery_increment
                    if iteration < self._teacher_mix_recovery_until
                    else 0.0
                )
                probability = min(1.0, nominal + recovery_offset)
                if iteration >= self.teacher_mix_force_zero_iteration:
                    probability = 0.0
            self._current_nominal_teacher_probability = nominal
            self._current_actual_teacher_probability = probability
            return probability
        probability = 0.0
        for start, scheduled_probability in self.teacher_mix_schedule:
            if iteration < start:
                break
            probability = scheduled_probability
        return probability

    def nominal_teacher_mix_probability(self, iteration: int) -> float:
        if iteration < self.teacher_mix_warmup_iterations:
            return 1.0
        if iteration >= self.teacher_mix_zero_iteration:
            return 0.0
        span = max(self.teacher_mix_zero_iteration - self.teacher_mix_warmup_iterations, 1)
        progress = (iteration - self.teacher_mix_warmup_iterations) / span
        return max(0.0, 1.0 - progress)

    def _apply_env_training_iteration(self, iteration: int) -> None:
        """Publish the RSL-RL iteration and trigger optional environment events."""

        env = getattr(self.env, "unwrapped", self.env)
        env._rsl_rl_training_iteration = int(iteration)
        event_manager = getattr(env, "event_manager", None)
        if event_manager is not None and "training_iteration" in event_manager.available_modes:
            event_manager.apply(mode="training_iteration")

    def _log_env_curriculum(self, iteration: int) -> None:
        """Log optional camera curriculum diagnostics on rank zero."""

        if self.writer is None:
            return
        env = getattr(self.env, "unwrapped", self.env)
        self._log_noise_prediction_metrics(env, iteration)
        camera_strength = getattr(env, "_v5_camera_extrinsics_curriculum_strength", None)
        if camera_strength is None:
            return

        strength = float(camera_strength)
        self.writer.add_scalar("CameraCurriculum/training_iteration", iteration, iteration)
        self.writer.add_scalar("CameraCurriculum/strength", strength, iteration)
        curriculum_cfg = getattr(
            getattr(env, "cfg", None),
            "v5_camera_extrinsics_curriculum",
            None,
        )
        if curriculum_cfg is not None:
            self.writer.add_scalar(
                "CameraCurriculum/translation_limit_m",
                strength * float(curriculum_cfg.camera_translation_max_m),
                iteration,
            )
            self.writer.add_scalar(
                "CameraCurriculum/rotation_limit_deg",
                strength * float(curriculum_cfg.camera_rotation_max_deg),
                iteration,
            )

        observations_cfg = getattr(getattr(env, "cfg", None), "observations", None)
        depth_group_cfg = getattr(observations_cfg, "depth_image", None)
        depth_term_cfg = getattr(depth_group_cfg, "head_depth", None)
        depth_params = getattr(depth_term_cfg, "params", {}) or {}
        if bool(depth_params.get("enable_sensor_noise", False)):
            self.writer.add_scalar(
                "CameraCurriculum/depth_noise_strength",
                strength,
                iteration,
            )

    def _log_noise_prediction_metrics(self, env, iteration: int) -> None:
        """Publish opt-in online depth-noise timings and runtime memory."""

        metrics_by_sensor = getattr(env, "_noise_prediction_metrics_by_sensor", None)
        if not isinstance(metrics_by_sensor, dict):
            return
        metric_names = (
            "provider/ddim_steps",
            "provider/latency_preprocess_ms",
            "provider/latency_vda_ms",
            "provider/latency_turbo_condition_ms",
            "provider/latency_diffusion_ms",
            "provider/latency_decode_ms",
            "provider/latency_total_ms",
            "provider/loaded_model_memory_mib",
            "provider/process_memory_allocated_mib",
            "provider/process_peak_memory_allocated_mib",
        )
        for sensor_name, metrics in metrics_by_sensor.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name in metric_names:
                value = metrics.get(metric_name)
                if torch.is_tensor(value):
                    if value.numel() != 1:
                        continue
                    value = float(value.detach().item())
                elif isinstance(value, (float, int)):
                    value = float(value)
                else:
                    continue
                short_name = metric_name.removeprefix("provider/")
                self.writer.add_scalar(
                    f"NoisePrediction/{sensor_name}/{short_name}",
                    value,
                    iteration,
                )

    @staticmethod
    def _metric_tensor(extras, keys: tuple[str, ...], num_envs: int, device) -> torch.Tensor | None:
        metric_dict = extras.get("episode") or extras.get("log") or {}
        for key in keys:
            value = metric_dict.get(key)
            if torch.is_tensor(value):
                value = value.reshape(-1).to(device=device, dtype=torch.float32)
                if value.numel() == num_envs:
                    return value
        return None

    def _live_env_metric(self, name: str) -> torch.Tensor | None:
        try:
            env = getattr(self.env, "unwrapped", self.env)
            if name == "success":
                return env.command_manager.get_term("object_pose").metrics["success"].detach()
            termination_manager = env.termination_manager
            index = termination_manager.active_terms.index(name)
            return termination_manager._term_dones[:, index].detach()  # noqa: SLF001
        except (AttributeError, KeyError, ValueError):
            return None

    def _accumulate_dagger_episode_metrics(self, extras, success_buffer, oob_buffer, non_finite_buffer):
        num_envs = int(success_buffer.numel())
        success = self._live_env_metric("success")
        if success is None:
            success = self._metric_tensor(
                extras,
                ("Metrics/object_pose/success", "Train/success_rate", "success"),
                num_envs,
                success_buffer.device,
            )
        oob = self._live_env_metric("object_out_of_bound")
        if oob is None:
            oob = self._metric_tensor(
                extras,
                ("Episode_Termination/object_out_of_bound", "object_out_of_bound"),
                num_envs,
                success_buffer.device,
            )
        non_finite = self._live_env_metric("non_finite")
        if non_finite is None:
            non_finite = self._metric_tensor(
                extras,
                ("Episode_Termination/non_finite", "non_finite"),
                num_envs,
                success_buffer.device,
            )
        if success is not None:
            success = success.to(device=success_buffer.device, dtype=success_buffer.dtype)
            success_buffer.copy_(torch.maximum(success_buffer, success))
        if oob is not None:
            oob = oob.to(device=oob_buffer.device, dtype=oob_buffer.dtype)
            oob_buffer.copy_(torch.maximum(oob_buffer, oob))
        if non_finite is not None:
            non_finite = non_finite.to(device=non_finite_buffer.device, dtype=non_finite_buffer.dtype)
            non_finite_buffer.copy_(torch.maximum(non_finite_buffer, non_finite))

    def _record_dagger_episode_outcomes(
        self,
        extras,
        dones,
        used_teacher,
        reward_sums,
        episode_success=None,
        episode_oob=None,
        episode_non_finite=None,
    ):
        done_ids = dones.reshape(-1).bool().nonzero(as_tuple=False).reshape(-1)
        if done_ids.numel() == 0:
            return
        num_envs = int(dones.numel())
        success = episode_success if episode_success is not None else self._metric_tensor(
            extras,
            ("Metrics/object_pose/success", "Train/success_rate", "success"),
            num_envs,
            dones.device,
        )
        oob = episode_oob if episode_oob is not None else self._metric_tensor(
            extras,
            ("Episode_Termination/object_out_of_bound", "object_out_of_bound"),
            num_envs,
            dones.device,
        )
        non_finite = episode_non_finite if episode_non_finite is not None else self._metric_tensor(
            extras,
            ("Episode_Termination/non_finite", "non_finite"),
            num_envs,
            dones.device,
        )
        for env_id in done_ids.tolist():
            role = "teacher" if bool(used_teacher[env_id].item()) else "student"
            self._dagger_role_stats[role].append(
                {
                    "success": float(success[env_id].item()) if success is not None else 0.0,
                    "reward": float(reward_sums[env_id].item()),
                    "oob": float(oob[env_id].item()) if oob is not None else 0.0,
                    "non_finite": float(non_finite[env_id].item()) if non_finite is not None else 0.0,
                }
            )

    def _role_summary(self, role: str, limit: int | None = None) -> dict[str, float]:
        values = list(self._dagger_role_stats[role])
        if limit is not None:
            values = values[-limit:]
        if not values:
            return {"count": 0.0, "success": 0.0, "reward": 0.0, "oob": 0.0, "non_finite": 0.0}
        return {
            "count": float(len(values)),
            **{key: sum(item[key] for item in values) / len(values) for key in ("success", "reward", "oob", "non_finite")},
        }

    def _log_dagger_diagnostics(self, iteration: int):
        if self.writer is None:
            return
        for role in ("student", "teacher"):
            summary = self._role_summary(role, limit=200)
            self.writer.add_scalar(f"DAgger/{role}_episode_count", summary["count"], iteration)
            self.writer.add_scalar(f"DAgger/{role}_success_rate", summary["success"], iteration)
            self.writer.add_scalar(f"DAgger/{role}_mean_reward", summary["reward"], iteration)
            self.writer.add_scalar(f"DAgger/{role}_object_out_of_bound", summary["oob"], iteration)
            self.writer.add_scalar(f"DAgger/{role}_non_finite", summary["non_finite"], iteration)
        self.writer.add_scalar(
            "DAgger/nominal_teacher_probability", self._current_nominal_teacher_probability, iteration
        )
        self.writer.add_scalar(
            "DAgger/actual_teacher_probability", self._current_actual_teacher_probability, iteration
        )
        self.writer.add_scalar(
            "DAgger/recovery_offset",
            self.teacher_mix_recovery_increment if iteration < self._teacher_mix_recovery_until else 0.0,
            iteration,
        )

    def _update_adaptive_teacher_mix(self, iteration: int):
        if self.teacher_mix_mode != "linear_adaptive" or (iteration + 1) % self.teacher_mix_eval_interval != 0:
            return
        student = self._role_summary("student", limit=self.teacher_mix_min_student_episodes)
        if student["count"] < self.teacher_mix_min_student_episodes:
            return
        if self._stable_student_success is None:
            self._stable_student_success = student["success"]
            self._stable_student_oob = student["oob"]
            self._save_stable_checkpoint()
            return
        bad = (
            student["success"] < self._stable_student_success - self.teacher_mix_success_drop_threshold
            or student["oob"] > self._stable_student_oob + self.teacher_mix_oob_rise_threshold
        )
        self._teacher_mix_bad_windows = self._teacher_mix_bad_windows + 1 if bad else 0
        if self._teacher_mix_bad_windows >= 2 and iteration < self.teacher_mix_force_zero_iteration:
            self._teacher_mix_recovery_until = max(
                self._teacher_mix_recovery_until, iteration + 1 + self.teacher_mix_recovery_iterations
            )
            self._teacher_mix_bad_windows = 0
        if not bad:
            self._stable_student_success = student["success"]
            self._stable_student_oob = student["oob"]
            self._save_stable_checkpoint()
        if student["non_finite"] > 0.0 and self._last_stable_checkpoint and os.path.exists(self._last_stable_checkpoint):
            self.load(self._last_stable_checkpoint, load_optimizer=True)

    def _save_stable_checkpoint(self):
        if self.log_dir is None or self.disable_logs:
            return
        self._last_stable_checkpoint = os.path.join(self.log_dir, "model_stable.pt")
        self.save(self._last_stable_checkpoint)

    def save(self, path: str, infos=None):
        infos = dict(infos or {})
        infos["v4_teacher_mix_controller"] = {
            "recovery_until": self._teacher_mix_recovery_until,
            "bad_windows": self._teacher_mix_bad_windows,
            "stable_student_success": self._stable_student_success,
            "stable_student_oob": self._stable_student_oob,
            "last_stable_checkpoint": self._last_stable_checkpoint,
        }
        if isinstance(self.alg, HybridDistillationPPO):
            infos["hybrid_dagger_ppo"] = self.alg.checkpoint_metadata()
            infos["hybrid_training_state"] = self.alg.training_state()
            infos["hybrid_initialization"] = {
                "teacher_checkpoint": self.hybrid_teacher_init_source,
                "student_checkpoint": self.hybrid_student_init_source,
                "student_iteration_restored": bool(
                    self.cfg.get("student_init_restore_iteration", False)
                ),
                "optimizer": "fresh_on_dual_checkpoint_migration",
            }
            infos["hybrid_reward_contract"] = self._hybrid_reward_contract()
        if isinstance(
            self.alg.policy,
            StudentTeacherV2TestDualCameraDistillation,
        ):
            infos["dual_depth_input_contract"] = make_dual_depth_input_contract(
                self.alg.policy.depth_output_size
            )
            if self.alg.policy.student_tactile_image_obs_group is not None:
                infos["vtgen_tactile_input_contract"] = (
                    make_vtgen_tactile_input_contract(
                        self.alg.policy.vtgen_tactile_checkpoint_sha256,
                        tactile_feature_dim=self.alg.policy.tactile_feature_dim,
                    )
                )
        elif isinstance(self.alg.policy, StudentTeacherDepthDaggerV5):
            depth_range_m = (
                V2_TEST_POLICY_DEPTH_RANGE_M
                if isinstance(
                    self.alg.policy,
                    (
                        StudentTeacherV2TestDistillation,
                        StudentTeacherV2TestDistillationV4,
                    ),
                )
                else V5_POLICY_DEPTH_RANGE_M
            )
            infos["v5_input_contract"] = make_v5_input_contract(
                self.alg.policy.depth_output_size,
                depth_range_m=depth_range_m,
            )
        if isinstance(
            self.alg.policy,
            StudentTeacherV2TestDualCameraDistillation,
        ):
            infos["v2_distillation_action_contract"] = (
                make_v2_test_dual_camera_distillation_action_contract()
            )
        elif isinstance(self.alg.policy, StudentTeacherV2TestDistillation):
            infos["v2_distillation_action_contract"] = (
                make_v2_test_distillation_v5_action_contract()
            )
        elif isinstance(self.alg.policy, StudentTeacherV2TestDistillationV4):
            infos["v2_distillation_action_contract"] = (
                make_v2_test_distillation_action_contract()
            )
        elif isinstance(self.alg.policy, StudentTeacherV2Distillation):
            infos["v2_distillation_action_contract"] = make_v2_distillation_action_contract()
        if isinstance(self.alg.policy, StudentTeacherDepthDiffusionV6):
            policy = self.alg.policy
            diffusion = policy.student.diffusion_policy
            infos["v6_diffusion"] = {
                "student_init_source": policy.student_init_source,
                "teacher_init_source": policy.teacher_init_source,
                "horizon": policy.horizon,
                "n_action_steps": policy.n_action_steps,
                "num_inference_steps": diffusion.num_inference_steps,
                "num_train_timesteps": diffusion.noise_scheduler.config.num_train_timesteps,
                "beta_start": diffusion.noise_scheduler.config.beta_start,
                "beta_end": diffusion.noise_scheduler.config.beta_end,
                "beta_schedule": diffusion.noise_scheduler.config.beta_schedule,
                "prediction_type": diffusion.noise_scheduler.config.prediction_type,
                "clip_sample": diffusion.noise_scheduler.config.clip_sample,
                "ema_decay": policy.ema_decay,
                "student_tactile_obs_group": policy.student_tactile_obs_group,
                "student_tactile_image_obs_group": policy.student_tactile_image_obs_group,
                "student_tactile_valid_obs_group": policy.student_tactile_valid_obs_group,
                "tactile_input_dim": policy.student.tactile_input_dim,
                "tactile_feature_dim": policy.student.tactile_feature_dim,
            }
        return super().save(path, infos=infos)

    def load_student_init(self, path: str, map_location: str | None = None) -> int:
        """Warm-start the student according to the current task's declared mode."""

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Student initialization checkpoint does not exist: {path}")
        policy = self.alg.policy
        if (
            isinstance(policy, StudentTeacherV2TestDualCameraDistillation)
            and getattr(policy, "student_tactile_image_obs_group", None) is not None
        ):
            raise ValueError(
                "The predicted-tactile task does not support --student_init_checkpoint. "
                "Start a new student from a PPO teacher via --checkpoint, or resume an "
                "already tactile-enabled student via --resume and --checkpoint."
            )
        checkpoint = torch.load(
            path,
            weights_only=False,
            map_location=self.device if map_location is None else map_location,
        )
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Student initialization checkpoint has no model_state_dict: {path}")

        student_init_mode = self.cfg.get("student_init_mode")
        if student_init_mode == "full_v2_test_dual_camera":
            if not isinstance(policy, StudentTeacherV2TestDualCameraDistillation):
                raise TypeError(
                    "full_v2_test_dual_camera student initialization requires "
                    "StudentTeacherV2TestDualCameraDistillation."
                )

            contract_version = (
                v2_test_distillation_checkpoint_action_contract_version(checkpoint)
            )
            if contract_version != policy.action_contract_version:
                raise ValueError(
                    "student initialization checkpoint action contract does not "
                    "match the noise-prediction task: "
                    f"checkpoint={contract_version}, "
                    f"expected={policy.action_contract_version}"
                )
            checkpoint_depth_shape = dual_depth_checkpoint_shape(checkpoint)
            if checkpoint_depth_shape != policy.combined_depth_input_shape:
                raise ValueError(
                    "student initialization checkpoint depth input contract does "
                    "not match the noise-prediction task: "
                    f"checkpoint={checkpoint_depth_shape}, "
                    f"expected={policy.combined_depth_input_shape}"
                )

            if (
                isinstance(self.alg, HybridDistillationPPO)
                and not getattr(policy, "hybrid_teacher_rl_initialized", False)
            ):
                raise RuntimeError(
                    "Load model_21800.pt through --checkpoint before importing "
                    "the student checkpoint; the hybrid critic/std must come "
                    "from the PPO teacher."
                )
            state_dict = checkpoint["model_state_dict"]
            student_state = {
                key.removeprefix("student."): value
                for key, value in state_dict.items()
                if str(key).startswith("student.")
            }
            if not student_state:
                raise ValueError(
                    "student initialization checkpoint contains no student parameters"
                )
            normalizer_state = {
                key.removeprefix("student_obs_normalizer."): value
                for key, value in state_dict.items()
                if str(key).startswith("student_obs_normalizer.")
            }
            policy.student.load_state_dict(student_state, strict=True)
            policy.student_obs_normalizer.load_state_dict(
                normalizer_state,
                strict=True,
            )
            policy._hidden_state = None
            copied_count = len(student_state) + len(normalizer_state)
            if isinstance(self.alg, HybridDistillationPPO):
                schedule_iteration = self.current_learning_iteration
                if self.cfg.get("student_init_restore_iteration", False):
                    checkpoint_iteration = checkpoint.get("iter")
                    if (
                        isinstance(checkpoint_iteration, bool)
                        or not isinstance(checkpoint_iteration, int)
                        or checkpoint_iteration < 0
                    ):
                        raise ValueError(
                            "hybrid student initialization checkpoint must contain "
                            "a non-negative integer top-level iter"
                        )
                    self.current_learning_iteration = checkpoint_iteration
                    schedule_iteration = checkpoint_iteration
                # A complete pure-BC student has already satisfied the first
                # hybrid phase. Start it at critic ramp while retaining the
                # checkpoint iteration for environment curricula and logging.
                self.alg.set_schedule_origin(
                    schedule_iteration,
                    initial_relative_iteration=self.alg.critic_start_iteration,
                )
                self.alg.set_training_iteration(schedule_iteration)
                self.hybrid_student_init_source = os.path.abspath(path)
            print(
                "[INFO]: Warm-started complete v2-test stereo student "
                f"({copied_count} tensors; teacher/training state excluded): {path}"
            )
            return copied_count

        if not isinstance(self.alg.policy, StudentTeacherDepthDiffusionV6):
            raise TypeError(
                "--student_init_checkpoint requires either a task-declared "
                "student_init_mode or a V6 diffusion student."
            )
        checkpoint_depth_shape = v5_checkpoint_depth_shape(checkpoint)
        legacy_depth_shape = (2, *V5_LEGACY_DEPTH_OUTPUT_SIZE)
        if checkpoint_depth_shape != legacy_depth_shape:
            raise ValueError(
                "V6 can only warm-start from the legacy V5 depth contract: "
                f"checkpoint={checkpoint_depth_shape}, expected={legacy_depth_shape}"
            )
        copied = self.alg.policy.load_v5_backbone_state_dict(
            checkpoint["model_state_dict"], source=os.path.abspath(path)
        )
        print(f"[INFO]: Warm-started V6 from V5 student trunk ({len(copied)} tensors): {path}")
        return len(copied)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        fresh_v2_gravity_state = (
            dict(self._v2_gravity_curriculum_state)
            if self._v2_gravity_curriculum_state is not None
            else None
        )
        is_student_checkpoint = False
        if isinstance(self.alg.policy, StudentTeacherDepthDaggerV5):
            checkpoint = torch.load(path, weights_only=False, map_location="cpu")
            if (
                "model_state_dict" not in checkpoint
                and "generator_state_dict" in checkpoint
            ):
                raise ValueError(
                    "--checkpoint points to a VT-Gen generator checkpoint. "
                    "Use --vtgen_tactile_checkpoint for best_model.pth; "
                    "--checkpoint must be a PPO teacher or distilled student checkpoint."
                )
            state_dict = checkpoint.get("model_state_dict", {})
            is_student_checkpoint = isinstance(state_dict, dict) and any(
                str(key).startswith("student.") for key in state_dict
            )
            if is_student_checkpoint:
                if isinstance(self.alg, HybridDistillationPPO):
                    checkpoint_infos = checkpoint.get("infos")
                    metadata = (
                        checkpoint_infos.get("hybrid_dagger_ppo")
                        if isinstance(checkpoint_infos, dict)
                        else None
                    )
                    if metadata is None:
                        raise ValueError(
                            "This is a legacy MSE student checkpoint and cannot be "
                            "resumed directly by HybridDistillationPPO. Use "
                            "--checkpoint model_21800.pt together with "
                            "--student_init_checkpoint model_650.pt; this creates a "
                            "fresh optimizer and imports the teacher critic/std."
                        )
                    expected_metadata = self.alg.checkpoint_metadata()
                    if metadata != expected_metadata:
                        schema_version = (
                            metadata.get("schema_version")
                            if isinstance(metadata, dict)
                            else None
                        )
                        if schema_version in (1, 2):
                            raise ValueError(
                                "hybrid checkpoint schema 1/2 uses the retired fixed "
                                "BC-PPO weighting or gradient-balance controller and "
                                "cannot be resumed. Load the PPO teacher with "
                                "--checkpoint and a pure-BC student with "
                                "--student_init_checkpoint to start schema 3."
                            )
                        raise ValueError(
                            "hybrid checkpoint schema or loss schedule is incompatible: "
                            f"checkpoint={metadata!r}, expected={expected_metadata!r}"
                        )
                    reward_contract = checkpoint_infos.get(
                        "hybrid_reward_contract"
                    )
                    expected_reward_contract = self._hybrid_reward_contract()
                    if reward_contract != expected_reward_contract:
                        raise ValueError(
                            "hybrid checkpoint reward contract is incompatible: "
                            f"checkpoint={reward_contract!r}, "
                            f"expected={expected_reward_contract!r}"
                        )
                    self.alg.validate_training_state(
                        checkpoint_infos.get("hybrid_training_state")
                    )
                if isinstance(self.alg.policy, StudentTeacherV2Distillation):
                    infos = checkpoint.get("infos")
                    contract = (
                        infos.get("v2_distillation_action_contract")
                        if isinstance(infos, dict)
                        else None
                    )
                    if isinstance(
                        self.alg.policy,
                        StudentTeacherV2TestDualCameraDistillation,
                    ):
                        validate_v2_test_dual_camera_distillation_action_contract(
                            contract
                        )
                    elif isinstance(
                        self.alg.policy,
                        StudentTeacherV2TestDistillation,
                    ):
                        validate_v2_test_distillation_v5_action_contract(contract)
                    elif isinstance(
                        self.alg.policy,
                        StudentTeacherV2TestDistillationV4,
                    ):
                        validate_v2_test_distillation_action_contract(contract)
                    else:
                        validate_v2_distillation_action_contract(contract)
                is_dual_camera = isinstance(
                    self.alg.policy,
                    StudentTeacherV2TestDualCameraDistillation,
                )
                is_v2_test = isinstance(
                    self.alg.policy,
                    (
                        StudentTeacherV2TestDistillation,
                        StudentTeacherV2TestDistillationV4,
                    ),
                )
                if is_dual_camera:
                    checkpoint_depth_shape = dual_depth_checkpoint_shape(checkpoint)
                    expected_depth_shape = self.alg.policy.combined_depth_input_shape
                    if self.alg.policy.student_tactile_image_obs_group is not None:
                        infos = checkpoint.get("infos")
                        tactile_contract = (
                            infos.get("vtgen_tactile_input_contract")
                            if isinstance(infos, dict)
                            else None
                        )
                        validate_vtgen_tactile_input_contract(
                            tactile_contract,
                            checkpoint_sha256=(
                                self.alg.policy.vtgen_tactile_checkpoint_sha256
                            ),
                            tactile_feature_dim=self.alg.policy.tactile_feature_dim,
                        )
                else:
                    checkpoint_depth_shape = v5_checkpoint_depth_shape(
                        checkpoint,
                        expected_depth_range_m=(
                            V2_TEST_POLICY_DEPTH_RANGE_M
                            if is_v2_test
                            else V5_POLICY_DEPTH_RANGE_M
                        ),
                        allow_missing_contract=not is_v2_test,
                    )
                    expected_depth_shape = self.alg.policy.depth_input_shape
                if checkpoint_depth_shape != expected_depth_shape:
                    raise ValueError(
                        "resume checkpoint depth input contract does not match the task: "
                        f"checkpoint={checkpoint_depth_shape}, expected={expected_depth_shape}"
                    )
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        if isinstance(self.alg, HybridDistillationPPO):
            if is_student_checkpoint:
                initialization = (
                    infos.get("hybrid_initialization")
                    if isinstance(infos, dict)
                    else None
                )
                if isinstance(initialization, dict):
                    self.hybrid_teacher_init_source = initialization.get(
                        "teacher_checkpoint"
                    )
                    self.hybrid_student_init_source = initialization.get(
                        "student_checkpoint"
                    )
                self.alg.load_training_state(
                    infos.get("hybrid_training_state")
                )
                self.alg.set_training_iteration(self.current_learning_iteration)
            else:
                self.hybrid_teacher_init_source = os.path.abspath(path)
                if not getattr(
                    self.alg.policy,
                    "hybrid_teacher_rl_initialized",
                    False,
                ):
                    raise ValueError(
                        "PPO teacher checkpoint did not initialize the hybrid critic/std."
                    )
        if not is_student_checkpoint and fresh_v2_gravity_state is not None:
            # A PPO teacher initializes network weights only. Its training
            # curriculum state must never leak into a new student run.
            self._v2_gravity_curriculum_state = fresh_v2_gravity_state
            self._publish_v2_gravity_curriculum_state()
        if isinstance(self.alg.policy, StudentTeacherDepthDiffusionV6):
            v6_infos = infos.get("v6_diffusion") if isinstance(infos, dict) else None
            if isinstance(v6_infos, dict):
                self.alg.policy.student_init_source = str(
                    v6_infos.get("student_init_source", self.alg.policy.student_init_source)
                )
                self.alg.policy.teacher_init_source = str(
                    v6_infos.get("teacher_init_source", self.alg.policy.teacher_init_source)
                )
            else:
                self.alg.policy.teacher_init_source = os.path.abspath(path)
        if isinstance(infos, dict) and "v4_teacher_mix_controller" in infos:
            state = infos["v4_teacher_mix_controller"]
            self._teacher_mix_recovery_until = int(state.get("recovery_until", -1))
            self._teacher_mix_bad_windows = int(state.get("bad_windows", 0))
            self._stable_student_success = state.get("stable_student_success")
            self._stable_student_oob = state.get("stable_student_oob")
            self._last_stable_checkpoint = state.get("last_stable_checkpoint")
        return infos

    def _hybrid_reward_contract(self) -> dict[str, object]:
        contract = dict(HYBRID_STABLE_REWARD_CONTRACT)
        env = getattr(self.env, "unwrapped", self.env)
        env_cfg = getattr(env, "cfg", None)
        abnormal_robot = getattr(
            getattr(env_cfg, "terminations", None),
            "abnormal_robot",
            None,
        )
        params = getattr(abnormal_robot, "params", None)
        if not isinstance(params, dict) or "velocity_limit_scale" not in params:
            raise ValueError(
                "Hybrid DAgger-PPO environment is missing abnormal_robot "
                "velocity_limit_scale."
            )
        contract["abnormal_velocity_limit_scale"] = float(
            params["velocity_limit_scale"]
        )
        return contract

    def _mix_rollout_actions(self, student_actions, teacher_actions, probability: float):
        num_envs = int(student_actions.shape[0])
        mask = self._ensure_teacher_episode_mask(num_envs, probability, student_actions.device)
        actions = torch.where(mask.unsqueeze(-1), teacher_actions, student_actions)
        return actions, mask

    def _ensure_teacher_episode_mask(self, num_envs: int, probability: float, device):
        if self._teacher_episode_mask is None or self._teacher_episode_mask.shape[0] != num_envs:
            self._teacher_episode_mask = torch.rand(num_envs, device=device) < probability
        elif self._teacher_episode_mask.device != torch.device(device):
            self._teacher_episode_mask = self._teacher_episode_mask.to(device=device)
        return self._teacher_episode_mask

    def _resample_teacher_mask_for_dones(self, dones: torch.Tensor, probability: float):
        if self._teacher_episode_mask is None:
            return
        done_mask = dones.reshape(-1).to(dtype=torch.bool, device=self._teacher_episode_mask.device)
        done_ids = done_mask.nonzero(as_tuple=False).reshape(-1)
        if done_ids.numel() > 0:
            self._teacher_episode_mask[done_ids] = (
                torch.rand(done_ids.numel(), device=self._teacher_episode_mask.device) < probability
            )

    @staticmethod
    def _clone_fixed_observations(obs):
        if isinstance(obs, dict):
            return {key: DistillationRunner._clone_fixed_observations(value) for key, value in obs.items()}
        if torch.is_tensor(obs):
            return obs.detach().clone()
        return obs.detach().clone()

    def _log_fixed_observation(
        self,
        loss_dict: dict[str, float],
        it: int,
        start_iter: int,
        tot_iter: int,
        num_learning_iterations: int,
        learn_time: float,
        fixed_obs_size: int,
        batch_size: int,
        width: int = 80,
        pad: int = 35,
    ):
        sample_count = fixed_obs_size * self.gpu_world_size
        self.tot_timesteps += sample_count
        self.tot_time += learn_time
        fps = int(sample_count / max(learn_time, 1.0e-8))

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"FixedObs/{key}", value, it)
        self.writer.add_scalar("Loss/behavior", loss_dict["mse"], it)
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        title = f" Fixed-observation distillation {it}/{tot_iter} "
        log_string = (
            f"{'#' * width}\n"
            f"{title.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} samples/s (learning {learn_time:.3f}s)\n"
            f"{'Fixed observations:':>{pad}} {fixed_obs_size}\n"
            f"{'Mini-batch size:':>{pad}} {batch_size}\n"
        )
        for key, value in loss_dict.items():
            log_string += f"{key + ':':>{pad}} {value:.6f}\n"
        eta = self.tot_time / (it - start_iter + 1) * (start_iter + num_learning_iterations - it)
        log_string += (
            f"{'-' * width}\n"
            f"{'Total fixed samples:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {learn_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}\n"
        )
        print(log_string)
