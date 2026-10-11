# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V5-compatible recurrent depth student with a one-step diffusion action head."""

from __future__ import annotations

import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rsl_rl.modules.student_teacher_depth_dagger_v5 import (
    DepthMaskResNet18,
    V5_LEGACY_DEPTH_OUTPUT_SIZE,
)
from rsl_rl.modules.student_teacher_diffusion_policy import DiffusionLowdimStudent
from rsl_rl.networks import EmpiricalNormalization, MLP


class DextrahDepthRecurrentDiffusionStudent(nn.Module):
    """V5 visual/recurrent trunk followed by a conditional 1-D diffusion model."""

    def __init__(
        self,
        low_dim_obs_dim: int,
        action_dim: int,
        depth_input_channels: int = 2,
        visual_feature_dim: int = 128,
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 1,
        auxiliary_hidden_dim: int = 256,
        tactile_input_dim: int = 0,
        tactile_feature_dim: int = 64,
        horizon: int = 1,
        n_action_steps: int = 1,
        num_inference_steps: int = 4,
        obs_cond_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: list[int] | tuple[int, ...] = (128,),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        depth_encode_batch_size: int = 32,
        use_gradient_checkpointing: bool = True,
        use_bfloat16: bool = True,
    ):
        super().__init__()
        self.depth_encoder = DepthMaskResNet18(
            visual_feature_dim,
            input_channels=depth_input_channels,
        )
        fused_dim = int(low_dim_obs_dim) + int(visual_feature_dim)
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.memory = nn.LSTM(
            input_size=fused_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
        )
        self.memory_norm = nn.LayerNorm(rnn_hidden_dim)
        self.auxiliary_object_position = nn.Sequential(
            nn.Linear(rnn_hidden_dim, auxiliary_hidden_dim),
            nn.ELU(),
            nn.Linear(auxiliary_hidden_dim, 3),
        )

        self.tactile_input_dim = int(tactile_input_dim)
        self.tactile_feature_dim = int(tactile_feature_dim)
        if self.tactile_input_dim > 0:
            self.tactile_encoder = nn.Sequential(
                nn.Linear(self.tactile_input_dim, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Linear(128, self.tactile_feature_dim),
                nn.LayerNorm(self.tactile_feature_dim),
                nn.ELU(),
            )
        else:
            self.tactile_encoder = None
        # This adapter is present even when tactile observations are disabled, so
        # adding a tactile encoder later does not change the LSTM input shape.
        self.tactile_adapter = nn.Linear(self.tactile_feature_dim, fused_dim, bias=False)
        self.tactile_gate = nn.Parameter(torch.zeros(()))

        self.diffusion_policy = DiffusionLowdimStudent(
            obs_dim=rnn_hidden_dim,
            action_dim=action_dim,
            horizon=horizon,
            n_obs_steps=1,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
            obs_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
            oa_step_convention=True,
        )
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.rnn_num_layers = int(rnn_num_layers)
        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.action_dim = int(action_dim)
        self.depth_encode_batch_size = max(1, int(depth_encode_batch_size))
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_bfloat16 = bool(use_bfloat16)

    def initial_hidden(self, batch_size: int, device, dtype=torch.float32):
        shape = (self.rnn_num_layers, batch_size, self.rnn_hidden_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        outputs = []
        autocast_enabled = self.use_bfloat16 and depth.device.type == "cuda"
        for start in range(0, depth.shape[0], self.depth_encode_batch_size):
            depth_batch = depth[start : start + self.depth_encode_batch_size]

            def encode(batch):
                with torch.autocast(
                    device_type=batch.device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    return self.depth_encoder(batch)

            if self.training and self.use_gradient_checkpointing and torch.is_grad_enabled():
                encoded = checkpoint(encode, depth_batch, use_reentrant=False)
            else:
                encoded = encode(depth_batch)
            outputs.append(encoded.float())
        return torch.cat(outputs, dim=0)

    def _tactile_residual(
        self,
        tactile: torch.Tensor | None,
        tactile_valid: torch.Tensor | None,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if tactile is None or self.tactile_encoder is None:
            return torch.zeros(batch_size, self.fusion_norm.normalized_shape[0], device=device, dtype=dtype)
        features = self.tactile_encoder(tactile)
        if tactile_valid is not None:
            valid = tactile_valid.reshape(batch_size, -1).amin(dim=-1, keepdim=True).to(features.dtype)
            features = features * valid
        return self.tactile_adapter(features).to(dtype=dtype) * self.tactile_gate.to(dtype=dtype)

    def fused_features(
        self,
        low_dim_obs: torch.Tensor,
        depth: torch.Tensor,
        tactile: torch.Tensor | None = None,
        tactile_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_features = self.encode_depth(depth)
        base = torch.cat((low_dim_obs, visual_features), dim=-1)
        tactile_residual = self._tactile_residual(
            tactile,
            tactile_valid,
            base.shape[0],
            base.device,
            base.dtype,
        )
        return self.fusion_norm(base + tactile_residual)

    def forward_memory_step(
        self,
        low_dim_obs: torch.Tensor,
        depth: torch.Tensor,
        hidden_state=None,
        tactile: torch.Tensor | None = None,
        tactile_valid: torch.Tensor | None = None,
    ):
        if hidden_state is None:
            hidden_state = self.initial_hidden(low_dim_obs.shape[0], low_dim_obs.device, low_dim_obs.dtype)
        fused = self.fused_features(low_dim_obs, depth, tactile, tactile_valid).unsqueeze(0)
        recurrent, hidden_state = self.memory(fused, hidden_state)
        recurrent = self.memory_norm(recurrent.squeeze(0))
        return recurrent, self.auxiliary_object_position(recurrent), hidden_state

    def forward_memory_sequence(
        self,
        low_dim_obs: torch.Tensor,
        depth: torch.Tensor,
        dones: torch.Tensor,
        hidden_state=None,
        tactile: torch.Tensor | None = None,
        tactile_valid: torch.Tensor | None = None,
    ):
        """Encode T x B observations once and reset memory after terminal transitions."""

        sequence_length, batch_size = low_dim_obs.shape[:2]
        flat_low_dim = low_dim_obs.reshape(sequence_length * batch_size, -1)
        flat_depth = depth.reshape(sequence_length * batch_size, *depth.shape[2:])
        flat_tactile = None if tactile is None else tactile.reshape(sequence_length * batch_size, -1)
        flat_valid = None if tactile_valid is None else tactile_valid.reshape(sequence_length * batch_size, -1)
        fused = self.fused_features(flat_low_dim, flat_depth, flat_tactile, flat_valid).reshape(
            sequence_length, batch_size, -1
        )
        if hidden_state is None:
            hidden_state = self.initial_hidden(batch_size, low_dim_obs.device, low_dim_obs.dtype)

        recurrent_outputs = []
        for step in range(sequence_length):
            recurrent, hidden_state = self.memory(fused[step : step + 1], hidden_state)
            recurrent_outputs.append(recurrent.squeeze(0))
            if step < sequence_length - 1:
                keep = (~dones[step].reshape(-1).bool()).to(low_dim_obs.dtype).view(1, -1, 1)
                hidden_state = (hidden_state[0] * keep, hidden_state[1] * keep)
        recurrent = self.memory_norm(torch.stack(recurrent_outputs, dim=0))
        return recurrent, self.auxiliary_object_position(recurrent), hidden_state

    def sample_trajectory(
        self,
        recurrent_condition: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample one normalized action as a ``B x 1 x A`` diffusion tensor."""

        diffusion = self.diffusion_policy
        global_cond = diffusion.obs_cond_encoder(recurrent_condition)
        trajectory = torch.randn(
            recurrent_condition.shape[0],
            self.horizon,
            self.action_dim,
            device=recurrent_condition.device,
            dtype=torch.float32,
            generator=generator,
        )
        diffusion.noise_scheduler.set_timesteps(
            diffusion.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        )
        autocast_enabled = self.use_bfloat16 and trajectory.device.type == "cuda"
        for timestep in diffusion.noise_scheduler.timesteps:
            with torch.autocast(
                device_type=trajectory.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                model_output = diffusion.model(trajectory, timestep, global_cond=global_cond)
            trajectory = diffusion.noise_scheduler.step(
                model_output.float(), timestep, trajectory
            ).prev_sample
        return torch.nan_to_num(trajectory, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


class StudentTeacherDepthDiffusionV6(nn.Module):
    """V5 recurrent trunk with an EMA one-step diffusion action head."""

    is_recurrent = True
    uses_recurrent_diffusion = True
    uses_execution_mask = True
    supports_masked_rollout = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_low_dim_obs_group: str = "student_low_dim",
        student_depth_obs_group: str = "depth_image",
        student_tactile_image_obs_group: str | None = None,
        student_aux_obs_group: str = "student_aux",
        student_tactile_obs_group: str | None = None,
        student_tactile_valid_obs_group: str | None = None,
        tactile_input_dim: int = 0,
        tactile_feature_dim: int = 64,
        depth_input_channels: int = 2,
        teacher_obs_groups: list[str] | None = None,
        student_obs_normalization: bool = True,
        teacher_obs_normalization: bool = True,
        teacher_hidden_dims: list[int] | None = None,
        visual_feature_dim: int = 128,
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 1,
        auxiliary_hidden_dim: int = 256,
        auxiliary_loss_coefficient: float = 10.0,
        x0_loss_coefficient: float = 0.1,
        encoder_learning_rate: float = 1.0e-5,
        recurrent_learning_rate: float = 5.0e-5,
        diffusion_learning_rate: float = 1.0e-4,
        teacher_weight_min: float = 0.25,
        teacher_weight_max: float = 4.0,
        action_scale: list[float] | tuple[float, ...] | None = None,
        horizon: int = 1,
        n_action_steps: int = 1,
        num_inference_steps: int = 4,
        obs_cond_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: list[int] | tuple[int, ...] = (128,),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        ema_decay: float = 0.999,
        backbone_freeze_iterations: int = 100,
        depth_encode_batch_size: int = 32,
        use_gradient_checkpointing: bool = True,
        use_bfloat16: bool = True,
        inference_seed: int = 42,
        init_noise_std: float = 0.0,
        noise_std_type: str = "scalar",
        student_hidden_dims=None,
        actor_hidden_dims=None,
        activation: str = "elu",
        **kwargs,
    ):
        super().__init__()
        del init_noise_std, noise_std_type, student_hidden_dims, actor_hidden_dims, activation
        if kwargs:
            print(f"StudentTeacherDepthDiffusionV6 ignored unexpected arguments: {sorted(kwargs)}")
        if prediction_type != "epsilon":
            raise ValueError("V6 weighted epsilon and x0 losses require prediction_type='epsilon'.")
        if int(horizon) != 1 or int(n_action_steps) != 1:
            raise ValueError("V6 one-step control requires horizon=1 and n_action_steps=1.")

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.student_low_dim_obs_group = student_low_dim_obs_group
        self.student_depth_obs_group = student_depth_obs_group
        self.student_tactile_image_obs_group = student_tactile_image_obs_group
        self.student_aux_obs_group = student_aux_obs_group
        self.student_tactile_obs_group = student_tactile_obs_group
        self.student_tactile_valid_obs_group = student_tactile_valid_obs_group
        self.teacher_obs_groups = teacher_obs_groups or list(obs_groups["teacher"])
        self.auxiliary_loss_coefficient = float(auxiliary_loss_coefficient)
        self.x0_loss_coefficient = float(x0_loss_coefficient)
        self.encoder_learning_rate = float(encoder_learning_rate)
        self.recurrent_learning_rate = float(recurrent_learning_rate)
        self.diffusion_learning_rate = float(diffusion_learning_rate)
        self.teacher_weight_min = float(teacher_weight_min)
        self.teacher_weight_max = float(teacher_weight_max)
        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.num_actions = int(num_actions)
        self.ema_decay = float(ema_decay)
        self.backbone_freeze_iterations = int(backbone_freeze_iterations)
        self.training_iteration = 0
        self.inference_seed = int(inference_seed)

        if tuple(obs[student_low_dim_obs_group].shape[1:]) != (123,):
            raise ValueError(f"v6 requires student_low_dim [N,123], got {tuple(obs[student_low_dim_obs_group].shape)}")
        self.depth_observation_shape = (2, *V5_LEGACY_DEPTH_OUTPUT_SIZE)
        if tuple(obs[student_depth_obs_group].shape[1:]) != self.depth_observation_shape:
            raise ValueError(
                f"v6 preserves depth_image as [N,{self.depth_observation_shape[0]},"
                f"{self.depth_observation_shape[1]},{self.depth_observation_shape[2]}], "
                f"got {tuple(obs[student_depth_obs_group].shape)}"
            )
        tactile_image_channels = 0
        if student_tactile_image_obs_group is not None:
            tactile_image_shape = (1, *V5_LEGACY_DEPTH_OUTPUT_SIZE)
            if tuple(obs[student_tactile_image_obs_group].shape[1:]) != tactile_image_shape:
                raise ValueError(
                    f"v6 requires tactile_image [N,1,{tactile_image_shape[1]},"
                    f"{tactile_image_shape[2]}], got {tuple(obs[student_tactile_image_obs_group].shape)}"
                )
            tactile_image_channels = 1
        self.depth_input_shape = (
            self.depth_observation_shape[0] + tactile_image_channels,
            *V5_LEGACY_DEPTH_OUTPUT_SIZE,
        )
        if int(depth_input_channels) != self.depth_input_shape[0]:
            raise ValueError(
                f"depth_input_channels={depth_input_channels} does not match depth plus tactile images "
                f"({self.depth_input_shape[0]} channels)."
            )
        if tuple(obs[student_aux_obs_group].shape[1:]) != (3,):
            raise ValueError(f"v6 requires student_aux [N,3], got {tuple(obs[student_aux_obs_group].shape)}")
        if student_tactile_obs_group is None and int(tactile_input_dim) != 0:
            raise ValueError("tactile_input_dim must be 0 when student_tactile_obs_group is disabled.")
        if student_tactile_obs_group is not None:
            actual_tactile_dim = int(obs[student_tactile_obs_group].shape[-1])
            if actual_tactile_dim != int(tactile_input_dim):
                raise ValueError(
                    f"Configured tactile_input_dim={tactile_input_dim}, observation has {actual_tactile_dim}."
                )

        if action_scale is None:
            action_scale = [1.0] * self.num_actions
        action_scale_tensor = torch.as_tensor(action_scale, dtype=torch.float32)
        if action_scale_tensor.shape != (self.num_actions,):
            raise ValueError(f"v6 action_scale must contain {self.num_actions} values.")
        if not torch.isfinite(action_scale_tensor).all() or (action_scale_tensor <= 0.0).any():
            raise ValueError("v6 action_scale values must be finite and positive.")

        self.student = DextrahDepthRecurrentDiffusionStudent(
            low_dim_obs_dim=123,
            action_dim=self.num_actions,
            depth_input_channels=depth_input_channels,
            visual_feature_dim=visual_feature_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            auxiliary_hidden_dim=auxiliary_hidden_dim,
            tactile_input_dim=tactile_input_dim,
            tactile_feature_dim=tactile_feature_dim,
            horizon=horizon,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
            obs_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
            depth_encode_batch_size=depth_encode_batch_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_bfloat16=use_bfloat16,
        )
        self.ema_student = copy.deepcopy(self.student)
        self.ema_student.requires_grad_(False)
        self.ema_student.eval()

        self.student_obs_normalization = bool(student_obs_normalization)
        self.student_obs_normalizer = EmpiricalNormalization(123) if student_obs_normalization else nn.Identity()
        self.tactile_obs_normalizer = (
            EmpiricalNormalization(int(tactile_input_dim))
            if student_tactile_obs_group is not None and int(tactile_input_dim) > 0
            else nn.Identity()
        )
        num_teacher_obs = sum(int(obs[group].shape[-1]) for group in self.teacher_obs_groups)
        self.teacher = MLP(num_teacher_obs, self.num_actions, teacher_hidden_dims or [512, 256, 128], "elu")
        self.teacher_obs_normalization = bool(teacher_obs_normalization)
        self.teacher_obs_normalizer = (
            EmpiricalNormalization(num_teacher_obs) if teacher_obs_normalization else nn.Identity()
        )
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

        self.register_buffer("action_scale", action_scale_tensor)
        self.register_buffer("teacher_action_std", torch.ones(self.num_actions))
        self.register_buffer("teacher_action_weights", torch.ones(self.num_actions))
        self.register_buffer("student_action_std", torch.zeros(self.num_actions))
        self.register_buffer("ema_initialized", torch.tensor(True, dtype=torch.bool))
        self._hidden_state = None
        self._inference_generator = None
        self._last_action_mean = torch.zeros(1, self.num_actions)
        self.last_student_action_mask = None
        self.last_epsilon_loss = 0.0
        self.last_x0_loss = 0.0
        self.last_auxiliary_loss = 0.0
        self.last_teacher_target_clip_fraction = 0.0
        self.last_diffusion_sample_count = 0
        self.student_init_source = "random"
        self.teacher_init_source = "unloaded"
        self._backbone_is_frozen = False

    @property
    def action_mean(self):
        return self._last_action_mean

    @property
    def action_std(self):
        return self.student_action_std

    @property
    def entropy(self):
        return torch.zeros(self._last_action_mean.shape[0], device=self._last_action_mean.device)

    def _backbone_parameters(self):
        modules = (
            self.student.depth_encoder,
            self.student.fusion_norm,
            self.student.memory,
            self.student.memory_norm,
            self.student.auxiliary_object_position,
        )
        for module in modules:
            yield from module.parameters()

    def set_training_iteration(self, iteration: int):
        self.training_iteration = int(iteration)
        should_freeze = self.training_iteration < self.backbone_freeze_iterations
        if should_freeze != self._backbone_is_frozen:
            for parameter in self._backbone_parameters():
                parameter.requires_grad_(not should_freeze)
            self._backbone_is_frozen = should_freeze

    def optimizer_parameters(self):
        encoder_parameters = list(self.student.depth_encoder.parameters())
        recurrent_modules = (
            self.student.fusion_norm,
            self.student.memory,
            self.student.memory_norm,
            self.student.auxiliary_object_position,
        )
        recurrent_parameters = [p for module in recurrent_modules for p in module.parameters()]
        used = {id(parameter) for parameter in encoder_parameters + recurrent_parameters}
        diffusion_parameters = [
            parameter for parameter in self.student.parameters() if id(parameter) not in used
        ]
        return [
            {"params": encoder_parameters, "lr": self.encoder_learning_rate},
            {"params": recurrent_parameters, "lr": self.recurrent_learning_rate},
            {"params": diffusion_parameters, "lr": self.diffusion_learning_rate},
        ]

    def initialize_hidden_state(self, obs):
        batch_size = int(obs[self.student_low_dim_obs_group].shape[0])
        device = obs[self.student_low_dim_obs_group].device
        if self._hidden_state is None or self._hidden_state[0].shape[1] != batch_size:
            self._hidden_state = self.ema_student.initial_hidden(batch_size, device)
            self._inference_generator = torch.Generator(device=device)
            self._inference_generator.manual_seed(self.inference_seed)

    def get_hidden_states(self):
        return self._hidden_state, None

    def detach_hidden_states(self, dones=None):
        if self._hidden_state is not None:
            self._hidden_state = tuple(state.detach() for state in self._hidden_state)

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is not None:
            self._hidden_state = hidden_states[0] if len(hidden_states) == 2 else hidden_states
            return
        if dones is None:
            self._hidden_state = None
            self._inference_generator = None
            return
        if self._hidden_state is None:
            return
        done_mask = dones.reshape(-1).bool()
        keep = (~done_mask).to(self._hidden_state[0].dtype).view(1, -1, 1)
        self._hidden_state = (self._hidden_state[0] * keep, self._hidden_state[1] * keep)

    def get_teacher_obs(self, obs):
        return torch.cat([obs[group] for group in self.teacher_obs_groups], dim=-1)

    def evaluate(self, obs):
        teacher_obs = self.teacher_obs_normalizer(self.get_teacher_obs(obs))
        with torch.no_grad():
            return self.teacher(teacher_obs)

    def _student_inputs(self, obs, sequence: bool = False):
        low_dim = obs[self.student_low_dim_obs_group]
        if sequence:
            shape = low_dim.shape
            low_dim = self.student_obs_normalizer(low_dim.reshape(-1, shape[-1])).reshape(shape)
        else:
            low_dim = self.student_obs_normalizer(low_dim)
        depth = obs[self.student_depth_obs_group].float()
        if self.student_tactile_image_obs_group is not None:
            tactile_image = obs[self.student_tactile_image_obs_group].to(device=depth.device, dtype=depth.dtype)
            channel_dim = 2 if sequence else 1
            depth = torch.cat((depth, tactile_image), dim=channel_dim)
        tactile = None
        tactile_valid = None
        if self.student_tactile_obs_group is not None:
            tactile = obs[self.student_tactile_obs_group]
            if sequence:
                shape = tactile.shape
                tactile = self.tactile_obs_normalizer(tactile.reshape(-1, shape[-1])).reshape(shape)
            else:
                tactile = self.tactile_obs_normalizer(tactile)
            if self.student_tactile_valid_obs_group is not None:
                tactile_valid = obs[self.student_tactile_valid_obs_group]
        return low_dim, depth, tactile, tactile_valid

    def act(self, obs):
        mask = torch.ones(obs[self.student_low_dim_obs_group].shape[0], dtype=torch.bool, device=obs[self.student_low_dim_obs_group].device)
        fallback = torch.zeros(mask.shape[0], self.num_actions, device=mask.device)
        return self.act_masked(obs, mask, fallback)

    def act_inference(self, obs):
        return self.act(obs)

    def act_masked(self, obs, student_mask: torch.Tensor, fallback_actions: torch.Tensor):
        """Advance EMA memory for all envs and sample diffusion only for student episodes."""

        self.initialize_hidden_state(obs)
        low_dim, depth, tactile, tactile_valid = self._student_inputs(obs)
        recurrent, _, self._hidden_state = self.ema_student.forward_memory_step(
            low_dim,
            depth,
            self._hidden_state,
            tactile,
            tactile_valid,
        )
        student_mask = student_mask.reshape(-1).bool().to(recurrent.device)
        sample_ids = student_mask.nonzero(as_tuple=False).reshape(-1)
        actions = fallback_actions.detach().clone()
        if sample_ids.numel() > 0:
            normalized_action = self.ema_student.sample_trajectory(
                recurrent[sample_ids], generator=self._inference_generator
            )
            actions[sample_ids] = self.unnormalize_actions(normalized_action[:, 0])
        self.last_student_action_mask = student_mask
        self.last_diffusion_sample_count = int(sample_ids.numel())
        self._last_action_mean = actions
        return actions

    def normalize_actions(self, actions: torch.Tensor, clip: bool = True):
        normalized = actions / self.action_scale
        return normalized.clamp(-1.0, 1.0) if clip else normalized

    def unnormalize_actions(self, normalized_actions: torch.Tensor):
        return normalized_actions.clamp(-1.0, 1.0) * self.action_scale

    def _normalized_teacher_targets(self, teacher_actions: torch.Tensor):
        normalized = self.normalize_actions(teacher_actions, clip=False)
        self.last_teacher_target_clip_fraction = float(
            (normalized.abs() > 1.0).float().mean().detach().item()
        )
        return normalized.clamp(-1.0, 1.0)

    def _build_diffusion_windows(
        self,
        recurrent: torch.Tensor,
        normalized_actions: torch.Tensor,
        dones: torch.Tensor,
        execution_masks: torch.Tensor,
    ):
        del dones, execution_masks
        # Every recurrent state supervises only the teacher action at that same
        # frame. There is no future-action target and therefore no window that
        # can cross an episode boundary.
        sequence_length, num_envs, action_dim = normalized_actions.shape
        conditions = recurrent.reshape(sequence_length * num_envs, -1)
        targets = normalized_actions.reshape(sequence_length * num_envs, 1, action_dim)
        masks = torch.ones(
            sequence_length * num_envs,
            1,
            dtype=torch.bool,
            device=recurrent.device,
        )
        return conditions, targets, masks

    def compute_recurrent_diffusion_loss(
        self,
        obs,
        teacher_actions: torch.Tensor,
        dones: torch.Tensor,
        execution_masks: torch.Tensor,
        initial_hidden=None,
        diffusion_sample_minibatch_size: int = 128,
    ):
        low_dim, depth, tactile, tactile_valid = self._student_inputs(obs, sequence=True)
        recurrent, object_position, _ = self.student.forward_memory_sequence(
            low_dim,
            depth,
            dones,
            initial_hidden,
            tactile,
            tactile_valid,
        )
        auxiliary_loss = F.mse_loss(object_position.float(), obs[self.student_aux_obs_group].float())
        normalized_actions = self._normalized_teacher_targets(teacher_actions)
        windows = self._build_diffusion_windows(recurrent, normalized_actions, dones, execution_masks)
        conditions, trajectories, loss_mask = windows
        diffusion = self.student.diffusion_policy
        trajectories = trajectories.float()
        sample_batch_size = max(1, int(diffusion_sample_minibatch_size))
        weights = self.teacher_action_weights.view(1, 1, -1)
        epsilon_numerator = trajectories.new_zeros(())
        x0_numerator = trajectories.new_zeros(())
        denominator = trajectories.new_zeros(())
        alpha_cumprod = diffusion.noise_scheduler.alphas_cumprod.to(
            device=trajectories.device, dtype=torch.float32
        )
        autocast_enabled = self.student.use_bfloat16 and trajectories.device.type == "cuda"
        for start in range(0, trajectories.shape[0], sample_batch_size):
            end = min(start + sample_batch_size, trajectories.shape[0])
            condition_batch = conditions[start:end]
            trajectory_batch = trajectories[start:end]
            mask_batch = loss_mask[start:end]
            global_cond = diffusion.obs_cond_encoder(condition_batch)
            noise = torch.randn_like(trajectory_batch)
            timesteps = torch.randint(
                0,
                diffusion.noise_scheduler.config.num_train_timesteps,
                (trajectory_batch.shape[0],),
                device=trajectory_batch.device,
            ).long()
            noisy = diffusion.noise_scheduler.add_noise(trajectory_batch, noise, timesteps)
            expanded_mask = mask_batch.unsqueeze(-1).expand_as(noisy)
            with torch.autocast(
                device_type=noisy.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                prediction = diffusion.model(noisy, timesteps, global_cond=global_cond)
            prediction = prediction.float()
            float_mask = expanded_mask.float()
            batch_denominator = (float_mask * weights).sum()
            epsilon_numerator = epsilon_numerator + (
                ((prediction - noise) ** 2) * float_mask * weights
            ).sum()

            alpha = alpha_cumprod[timesteps].view(-1, 1, 1)
            predicted_x0 = (
                noisy - (1.0 - alpha).sqrt() * prediction
            ) / alpha.sqrt().clamp_min(1.0e-6)
            predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
            x0_element = F.smooth_l1_loss(predicted_x0, trajectory_batch, reduction="none")
            x0_numerator = x0_numerator + (x0_element * float_mask * weights).sum()
            denominator = denominator + batch_denominator

        denominator = denominator.clamp_min(1.0)
        epsilon_loss = epsilon_numerator / denominator
        x0_loss = x0_numerator / denominator
        total = (
            epsilon_loss
            + self.x0_loss_coefficient * x0_loss
            + self.auxiliary_loss_coefficient * auxiliary_loss
        )
        self.last_epsilon_loss = float(epsilon_loss.detach().item())
        self.last_x0_loss = float(x0_loss.detach().item())
        self.last_auxiliary_loss = float(auxiliary_loss.detach().item())
        self.last_diffusion_sample_count = int(conditions.shape[0])
        return total, epsilon_loss.detach(), x0_loss.detach(), auxiliary_loss.detach(), int(conditions.shape[0])

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            self.student_obs_normalizer.update(obs[self.student_low_dim_obs_group])
        if self.student_tactile_obs_group is not None and not isinstance(self.tactile_obs_normalizer, nn.Identity):
            self.tactile_obs_normalizer.update(obs[self.student_tactile_obs_group])

    @torch.no_grad()
    def update_ema(self):
        if not bool(self.ema_initialized.item()):
            self.ema_student.load_state_dict(self.student.state_dict())
            self.ema_initialized.fill_(True)
            return
        online_parameters = dict(self.student.named_parameters())
        ema_parameters = dict(self.ema_student.named_parameters())
        for name, parameter in online_parameters.items():
            ema_parameters[name].lerp_(parameter.detach(), 1.0 - self.ema_decay)
        online_buffers = dict(self.student.named_buffers())
        ema_buffers = dict(self.ema_student.named_buffers())
        for name, buffer in online_buffers.items():
            ema_buffers[name].copy_(buffer)

    def _set_teacher_action_std(self, action_std: torch.Tensor):
        action_std = action_std.detach().to(self.teacher_action_std).abs().clamp_min(1.0e-6)
        normalized_std = action_std / self.action_scale
        weights = normalized_std.reciprocal().square()
        weights = weights / weights.mean().clamp_min(1.0e-6)
        weights = weights.clamp(self.teacher_weight_min, self.teacher_weight_max)
        self.teacher_action_std.copy_(action_std)
        self.teacher_action_weights.copy_(weights)

    def load_v5_backbone_state_dict(self, state_dict: dict[str, torch.Tensor], source: str = "v5"):
        """Load the V5 trunk and zero-initialize a newly added saliency channel."""

        prefixes = (
            "student.depth_encoder.",
            "student.fusion_norm.",
            "student.memory.",
            "student.memory_norm.",
            "student.auxiliary_object_position.",
            "student_obs_normalizer.",
        )
        current = self.state_dict()
        copied = []
        for key, value in state_dict.items():
            if not key.startswith(prefixes) or key not in current:
                continue
            target = current[key]
            if target.shape == value.shape:
                current[key] = value.detach().to(target)
                copied.append(key)
            elif (
                key == "student.depth_encoder.backbone.conv1.weight"
                and value.ndim == 4
                and target.ndim == 4
                and value.shape[0] == target.shape[0]
                and value.shape[2:] == target.shape[2:]
                and value.shape[1] == 2
                and target.shape[1] == 3
            ):
                expanded = torch.zeros_like(target)
                expanded[:, :2].copy_(value.detach().to(target))
                current[key] = expanded
                copied.append(key)
        required_prefixes = prefixes[:5]
        missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in copied)]
        if missing:
            raise ValueError(f"V5 initialization checkpoint is missing compatible groups: {missing}")
        super().load_state_dict(current, strict=True)
        self.ema_student.load_state_dict(self.student.state_dict(), strict=True)
        self.ema_initialized.fill_(True)
        self.student_init_source = str(source)
        self._hidden_state = None
        return copied

    def load_state_dict(self, state_dict, strict=True):
        if any(key.startswith("actor.") for key in state_dict):
            teacher_state = {
                key.removeprefix("actor."): value
                for key, value in state_dict.items()
                if key.startswith("actor.")
            }
            normalizer_state = {
                key.removeprefix("actor_obs_normalizer."): value
                for key, value in state_dict.items()
                if key.startswith("actor_obs_normalizer.")
            }
            self.teacher.load_state_dict(teacher_state, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(normalizer_state, strict=strict)
            if "std" in state_dict:
                self._set_teacher_action_std(state_dict["std"])
            elif "log_std" in state_dict:
                self._set_teacher_action_std(state_dict["log_std"].exp())
            else:
                warnings.warn("Teacher checkpoint has no action std; V6 will use uniform action weights.")
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        if any(key.startswith("student.diffusion_policy.") for key in state_dict):
            current = self.state_dict()
            adapted = dict(state_dict)
            for key in (
                "student.depth_encoder.backbone.conv1.weight",
                "ema_student.depth_encoder.backbone.conv1.weight",
            ):
                source = adapted.get(key)
                target = current.get(key)
                if (
                    source is not None
                    and target is not None
                    and source.ndim == target.ndim == 4
                    and source.shape[0] == target.shape[0]
                    and source.shape[2:] == target.shape[2:]
                    and source.shape[1] == 2
                    and target.shape[1] == 3
                ):
                    expanded = torch.zeros_like(target)
                    expanded[:, :2].copy_(source.detach().to(target))
                    adapted[key] = expanded
            super().load_state_dict(adapted, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            self.ema_student.eval()
            self._hidden_state = None
            return True
        raise ValueError("state_dict is neither a PPO teacher nor a V6 checkpoint")

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()
        self.ema_student.eval()
        return self

    def diagnostics(self):
        return {
            "v6_epsilon_loss": self.last_epsilon_loss,
            "v6_x0_loss": self.last_x0_loss,
            "v6_auxiliary_object_position_loss": self.last_auxiliary_loss,
            "v6_teacher_target_clip_fraction": self.last_teacher_target_clip_fraction,
            "v6_diffusion_sample_count": float(self.last_diffusion_sample_count),
            "v6_tactile_gate": float(self.student.tactile_gate.detach().item()),
            "v6_backbone_frozen": float(self._backbone_is_frozen),
            "v6_action_scale_min": float(self.action_scale.min().item()),
            "v6_action_scale_max": float(self.action_scale.max().item()),
        }
