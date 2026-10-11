# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-student distillation module with a depth-conditioned diffusion student."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.modules.student_teacher_depth_dagger import DepthAug, DepthEncoder
from rsl_rl.modules.student_teacher_diffusion_policy import DiffusionLowdimStudent


class DepthDiffusionStudent(nn.Module):
    """Encode depth frames and feed low-dimensional features to a diffusion actor."""

    def __init__(
        self,
        low_dim_obs_dim: int,
        depth_image_shape: tuple[int, int],
        depth_feature_dim: int,
        action_dim: int,
        horizon: int = 8,
        n_obs_steps: int = 4,
        n_action_steps: int = 4,
        num_inference_steps: int = 8,
        obs_cond_dim: int | None = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: list[int] | tuple[int, ...] = (128, 256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        oa_step_convention: bool = True,
        depth_encode_batch_size: int = 4,
        depth_input_channels: int = 1,
    ):
        super().__init__()
        self.depth_encoder = DepthEncoder(
            depth_image_shape,
            out_features=depth_feature_dim,
            input_channels=depth_input_channels,
        )
        self.depth_encode_batch_size = int(depth_encode_batch_size)
        self.diffusion_policy = DiffusionLowdimStudent(
            obs_dim=low_dim_obs_dim + depth_feature_dim,
            action_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
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
            oa_step_convention=oa_step_convention,
        )

    def encode_step(self, low_dim_obs: torch.Tensor, depth_image: torch.Tensor) -> torch.Tensor:
        depth_features = self._encode_depth(depth_image)
        return torch.cat((low_dim_obs, depth_features), dim=-1)

    def encode_sequence(self, low_dim_obs_sequence: torch.Tensor, depth_image_sequence: torch.Tensor) -> torch.Tensor:
        batch_size, horizon = low_dim_obs_sequence.shape[:2]
        flat_depth = depth_image_sequence.reshape(
            batch_size * horizon,
            *depth_image_sequence.shape[2:],
        )
        depth_features = self._encode_depth(flat_depth).reshape(batch_size, horizon, -1)
        return torch.cat((low_dim_obs_sequence, depth_features), dim=-1)

    def _encode_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
        if self.depth_encode_batch_size <= 0 or depth_image.shape[0] <= self.depth_encode_batch_size:
            return self.depth_encoder(depth_image)
        return torch.cat(
            [self.depth_encoder(chunk) for chunk in depth_image.split(self.depth_encode_batch_size, dim=0)],
            dim=0,
        )

    def predict_action_chunks(self, low_dim_history: torch.Tensor, depth_history: torch.Tensor) -> torch.Tensor:
        obs_history = self.encode_sequence(low_dim_history, depth_history)
        return self.diffusion_policy.predict_action_chunks(obs_history)

    def compute_loss(
        self,
        low_dim_obs_sequence: torch.Tensor,
        depth_image_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_sequence = self.encode_sequence(low_dim_obs_sequence, depth_image_sequence)
        return self.diffusion_policy.compute_loss(obs_sequence, action_sequence, loss_mask=loss_mask)


class StudentTeacherDepthDiffusionPolicy(nn.Module):
    """Online DAgger distillation with a depth-conditioned diffusion-policy student."""

    is_recurrent = False
    uses_diffusion_policy = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_low_dim_obs_groups: list[str] | None = None,
        student_depth_obs_group: str = "depth_image",
        teacher_obs_groups: list[str] | None = None,
        use_teacher_obs_for_student: bool = False,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: list[int] | None = None,
        teacher_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        depth_feature_dim: int = 128,
        depth_min: float = 0.5,
        depth_max: float = 1.3,
        depth_output_size: tuple[int, int] = (120, 160),
        enable_depth_augmentation: bool = True,
        horizon: int = 8,
        n_obs_steps: int = 4,
        n_action_steps: int = 4,
        num_inference_steps: int = 8,
        obs_cond_dim: int | None = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: list[int] | tuple[int, ...] = (128, 256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        oa_step_convention: bool = True,
        depth_encode_batch_size: int = 4,
        depth_input_channels: int = 1,
        **kwargs,
    ):
        if teacher_hidden_dims is None:
            teacher_hidden_dims = [512, 256, 128]
        if kwargs:
            print(
                "StudentTeacherDepthDiffusionPolicy.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.student_depth_obs_group = student_depth_obs_group
        configured_student_low_dim_obs_groups = student_low_dim_obs_groups or [
            group for group in obs_groups["policy"] if group != student_depth_obs_group
        ]
        self.teacher_obs_groups = teacher_obs_groups or obs_groups["teacher"]
        self.use_teacher_obs_for_student = bool(use_teacher_obs_for_student)
        self.student_low_dim_obs_groups = (
            list(self.teacher_obs_groups)
            if self.use_teacher_obs_for_student
            else list(configured_student_low_dim_obs_groups)
        )
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.depth_output_size = tuple(depth_output_size)
        self.enable_depth_augmentation = bool(enable_depth_augmentation)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_actions = int(num_actions)
        self.depth_encode_batch_size = int(depth_encode_batch_size)
        self.depth_input_channels = int(depth_input_channels)
        self._depth_aug: DepthAug | None = None
        self._depth_aug_device: str | None = None

        num_student_low_dim_obs = 0
        for obs_group in self.student_low_dim_obs_groups:
            assert len(obs[obs_group].shape) == 2, "Depth diffusion low-dimensional observations must be 1D."
            num_student_low_dim_obs += obs[obs_group].shape[-1]

        depth_sample = self._extract_depth(obs)
        depth_height, depth_width = int(depth_sample.shape[-2]), int(depth_sample.shape[-1])
        if (depth_height, depth_width) != self.depth_output_size:
            raise ValueError(
                f"Depth observation shape {(depth_height, depth_width)} does not match "
                f"configured depth_output_size {self.depth_output_size}."
            )

        num_teacher_obs = 0
        for obs_group in self.teacher_obs_groups:
            assert len(obs[obs_group].shape) == 2, "Depth diffusion teacher observations must be 1D."
            num_teacher_obs += obs[obs_group].shape[-1]

        self.student = DepthDiffusionStudent(
            low_dim_obs_dim=num_student_low_dim_obs,
            depth_image_shape=self.depth_output_size,
            depth_feature_dim=depth_feature_dim,
            action_dim=num_actions,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
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
            oa_step_convention=oa_step_convention,
            depth_encode_batch_size=depth_encode_batch_size,
            depth_input_channels=self.depth_input_channels,
        )
        self.student_hidden_dims = student_hidden_dims

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_low_dim_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()
        print(
            "Depth diffusion student: "
            f"use_teacher_obs_for_student={self.use_teacher_obs_for_student}, "
            f"low_dim_obs_groups={self.student_low_dim_obs_groups}, "
            f"low_dim_obs={num_student_low_dim_obs}, depth_feature_dim={depth_feature_dim}, "
            f"action_dim={num_actions}, horizon={horizon}, n_obs_steps={n_obs_steps}, "
            f"n_action_steps={n_action_steps}, num_inference_steps={num_inference_steps}, "
            f"obs_cond_dim={obs_cond_dim}, down_dims={list(down_dims)}, "
            f"depth_encode_batch_size={depth_encode_batch_size}"
        )

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()
        print(f"Teacher MLP: {self.teacher}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=False)
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)), requires_grad=False)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution: Normal | None = None
        self._low_dim_history: torch.Tensor | None = None
        self._depth_history: torch.Tensor | None = None
        self._action_queue: torch.Tensor | None = None
        self._action_queue_lengths: torch.Tensor | None = None
        self._pending_history_reset: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def forward(self):
        raise NotImplementedError

    def reset(self, dones=None, hidden_states=None):
        if dones is None:
            self._low_dim_history = None
            self._depth_history = None
            self._action_queue = None
            self._action_queue_lengths = None
            self._pending_history_reset = None
            return

        dones = dones.reshape(-1).to(dtype=torch.bool)
        if dones.numel() == 0 or not dones.any():
            return
        if self._pending_history_reset is None or self._pending_history_reset.numel() != dones.numel():
            self._pending_history_reset = torch.zeros(dones.numel(), dtype=torch.bool, device=dones.device)
        self._pending_history_reset = self._pending_history_reset.to(device=dones.device)
        self._pending_history_reset |= dones
        if self._action_queue_lengths is not None:
            self._action_queue_lengths = self._action_queue_lengths.to(device=dones.device)
            self._action_queue_lengths[dones] = 0

    def act(self, obs):
        action = self._act_from_history(obs)
        self._update_distribution(action)
        return action

    def act_inference(self, obs):
        action = self._act_from_history(obs)
        self._update_distribution(action)
        return action

    def evaluate(self, obs):
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def compute_distillation_loss(self, obs_sequence: dict[str, torch.Tensor], privileged_action_sequence: torch.Tensor):
        low_dim_obs, depth_image = self.get_student_obs_sequence(obs_sequence)
        low_dim_obs = self.student_obs_normalizer(low_dim_obs)
        return self.student.compute_loss(low_dim_obs, depth_image, privileged_action_sequence)

    def get_student_obs(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        low_dim_obs = torch.cat([obs[obs_group] for obs_group in self.student_low_dim_obs_groups], dim=-1)
        depth_image = self._extract_depth(obs)
        if self.training and self.enable_depth_augmentation:
            depth_image = self._augment_depth(depth_image)
        else:
            depth_image = self._clip_depth(depth_image)
        return low_dim_obs, depth_image

    def get_student_obs_sequence(self, obs_sequence: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        low_dim_obs = torch.cat([obs_sequence[obs_group] for obs_group in self.student_low_dim_obs_groups], dim=-1)
        depth_image = self._extract_depth_sequence(obs_sequence)
        if self.training and self.enable_depth_augmentation:
            depth_image = self._augment_depth_sequence(depth_image)
        else:
            depth_image = self._clip_depth(depth_image)
        return low_dim_obs, depth_image

    def get_teacher_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.teacher_obs_groups], dim=-1)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        if hasattr(self.teacher_obs_normalizer, "eval"):
            self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            low_dim_obs = torch.cat([obs[obs_group] for obs_group in self.student_low_dim_obs_groups], dim=-1)
            self.student_obs_normalizer.update(low_dim_obs)

    def load_state_dict(self, state_dict, strict=True):
        if any(key.startswith("actor.") for key in state_dict.keys()):
            teacher_state_dict = {
                key.replace("actor.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("actor.")
            }
            teacher_obs_normalizer_state_dict = {
                key.replace("actor_obs_normalizer.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("actor_obs_normalizer.")
            }
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False

        if any(key.startswith("student.") for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True

        raise ValueError("state_dict does not contain actor (teacher) or student parameters")

    def _act_from_history(self, obs):
        low_dim_obs, depth_image = self.get_student_obs(obs)
        low_dim_obs = self.student_obs_normalizer(low_dim_obs)
        self._update_obs_history(low_dim_obs, depth_image)

        if self._action_queue is None or self._action_queue.shape[0] != low_dim_obs.shape[0]:
            self._action_queue = torch.zeros(
                low_dim_obs.shape[0],
                self.n_action_steps,
                self.num_actions,
                dtype=low_dim_obs.dtype,
                device=low_dim_obs.device,
            )
            self._action_queue_lengths = torch.zeros(low_dim_obs.shape[0], dtype=torch.long, device=low_dim_obs.device)
        elif self._action_queue_lengths is None or self._action_queue_lengths.numel() != low_dim_obs.shape[0]:
            self._action_queue_lengths = torch.zeros(low_dim_obs.shape[0], dtype=torch.long, device=low_dim_obs.device)
        else:
            self._action_queue = self._action_queue.to(device=low_dim_obs.device, dtype=low_dim_obs.dtype)
            self._action_queue_lengths = self._action_queue_lengths.to(device=low_dim_obs.device)

        env_ids = (self._action_queue_lengths <= 0).nonzero(as_tuple=False).reshape(-1)
        if env_ids.numel() > 0:
            with torch.no_grad():
                chunks = self.student.predict_action_chunks(
                    self._low_dim_history[env_ids],
                    self._depth_history[env_ids],
                )
            self._set_action_chunks(env_ids, chunks)

        action = self._action_queue[:, 0].clone()
        self._action_queue[:, :-1] = self._action_queue[:, 1:].clone()
        self._action_queue[:, -1] = 0
        self._action_queue_lengths = torch.clamp(self._action_queue_lengths - 1, min=0)
        return action

    def _update_obs_history(self, low_dim_obs: torch.Tensor, depth_image: torch.Tensor):
        num_envs = low_dim_obs.shape[0]
        if self._low_dim_history is None or self._low_dim_history.shape[0] != num_envs:
            self._low_dim_history = low_dim_obs.unsqueeze(1).repeat(1, self.n_obs_steps, 1)
            self._depth_history = depth_image.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
            self._pending_history_reset = torch.zeros(num_envs, dtype=torch.bool, device=low_dim_obs.device)
            return

        self._low_dim_history = self._low_dim_history.to(device=low_dim_obs.device, dtype=low_dim_obs.dtype)
        self._depth_history = self._depth_history.to(device=depth_image.device, dtype=depth_image.dtype)
        if self._pending_history_reset is not None:
            self._pending_history_reset = self._pending_history_reset.to(device=low_dim_obs.device)
            reset_ids = self._pending_history_reset.nonzero(as_tuple=False).reshape(-1)
            if reset_ids.numel() > 0:
                self._low_dim_history[reset_ids] = low_dim_obs[reset_ids].unsqueeze(1).repeat(1, self.n_obs_steps, 1)
                self._depth_history[reset_ids] = depth_image[reset_ids].unsqueeze(1).repeat(
                    1,
                    self.n_obs_steps,
                    1,
                    1,
                    1,
                )
                self._pending_history_reset[reset_ids] = False

        self._low_dim_history[:, :-1] = self._low_dim_history[:, 1:].clone()
        self._low_dim_history[:, -1] = low_dim_obs
        self._depth_history[:, :-1] = self._depth_history[:, 1:].clone()
        self._depth_history[:, -1] = depth_image

    def _set_action_chunks(self, env_ids: torch.Tensor, chunks: torch.Tensor):
        num_envs = self._low_dim_history.shape[0]
        if self._action_queue is None or self._action_queue.shape[1] != self.n_action_steps:
            new_queue = torch.zeros(
                num_envs,
                self.n_action_steps,
                self.num_actions,
                dtype=chunks.dtype,
                device=chunks.device,
            )
            if self._action_queue is not None and self._action_queue.shape[1] > 0:
                old_len = min(self._action_queue.shape[1], self.n_action_steps)
                new_queue[:, :old_len] = self._action_queue[:, :old_len]
            self._action_queue = new_queue
        if self._action_queue_lengths is None or self._action_queue_lengths.numel() != num_envs:
            self._action_queue_lengths = torch.zeros(num_envs, dtype=torch.long, device=chunks.device)
        self._action_queue[env_ids] = chunks
        self._action_queue_lengths[env_ids] = chunks.shape[1]

    def _update_distribution(self, action: torch.Tensor):
        if self.noise_std_type == "scalar":
            std = self.std.to(action.device).expand_as(action)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std.to(action.device)).expand_as(action)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(action, std)

    def _extract_depth(self, obs) -> torch.Tensor:
        depth = obs[self.student_depth_obs_group]
        if not torch.is_tensor(depth):
            if "head_depth" in depth:
                depth = depth["head_depth"]
            elif len(depth.keys()) == 1:
                depth = depth[next(iter(depth.keys()))]
            else:
                raise ValueError(
                    f"Depth observation group {self.student_depth_obs_group!r} must be a tensor or contain one tensor."
                )

        depth = depth.to(dtype=torch.float32)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        elif depth.ndim == 4 and depth.shape[1] != 1 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                f"Expected depth observation [N, 1, H, W] or [N, H, W, 1], got shape {tuple(depth.shape)}."
            )
        return self._clip_depth(depth)

    def _extract_depth_sequence(self, obs_sequence: dict[str, torch.Tensor]) -> torch.Tensor:
        depth = obs_sequence[self.student_depth_obs_group]
        if not torch.is_tensor(depth):
            if "head_depth" in depth:
                depth = depth["head_depth"]
            elif len(depth.keys()) == 1:
                depth = depth[next(iter(depth.keys()))]
            else:
                raise ValueError(
                    f"Depth observation group {self.student_depth_obs_group!r} must be a tensor or contain one tensor."
                )

        depth = depth.to(dtype=torch.float32)
        if depth.ndim == 4:
            depth = depth.unsqueeze(2)
        elif depth.ndim == 5 and depth.shape[2] != 1 and depth.shape[-1] == 1:
            depth = depth.permute(0, 1, 4, 2, 3)
        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError(
                "Expected depth sequence [B, T, 1, H, W] or [B, T, H, W, 1], "
                f"got shape {tuple(depth.shape)}."
            )
        return self._clip_depth(depth)

    def _clip_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        invalid = (depth <= 1.0e-8) | (depth < self.depth_min) | (depth > self.depth_max)
        return torch.where(invalid, torch.zeros_like(depth), depth)

    def _augment_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
        depth_nhw = depth_image[:, 0].contiguous()
        depths = depth_nhw.clone()
        depth_aug = self._get_depth_aug(depths.device)
        depth_aug.add_correlated_noise(
            depth_nhw,
            depths,
            sigma_s=1.0 / 2.0,
            sigma_d=1.0 / 6.0,
            d_min=self.depth_min,
            d_max=self.depth_max,
        )
        depth_aug.add_normal_noise(depths, sigma_theta=0.01, d_min=self.depth_min, d_max=self.depth_max)
        depth_aug.add_pixel_dropout_and_randu(
            depths,
            p_dropout=0.0125 / 4.0,
            p_randu=0.0125 / 4.0,
            d_min=self.depth_min,
            d_max=self.depth_max,
        )
        depth_aug.add_sticks(
            depths,
            p_stick=0.001 / 4.0,
            max_stick_len=18.0,
            max_stick_width=3.0,
            d_min=self.depth_min,
            d_max=self.depth_max,
        )
        return self._clip_depth(depths.unsqueeze(1))

    def _augment_depth_sequence(self, depth_sequence: torch.Tensor) -> torch.Tensor:
        batch_size, horizon = depth_sequence.shape[:2]
        flat_depth = depth_sequence.reshape(batch_size * horizon, *depth_sequence.shape[2:])
        if self.depth_encode_batch_size > 0 and flat_depth.shape[0] > self.depth_encode_batch_size:
            flat_depth = torch.cat(
                [self._augment_depth(chunk) for chunk in flat_depth.split(self.depth_encode_batch_size, dim=0)],
                dim=0,
            )
        else:
            flat_depth = self._augment_depth(flat_depth)
        return flat_depth.reshape(batch_size, horizon, *flat_depth.shape[1:])

    def _get_depth_aug(self, torch_device: torch.device) -> DepthAug:
        device = self._warp_device_from_torch(torch_device)
        if self._depth_aug is None or self._depth_aug_device != device:
            self._depth_aug = DepthAug(device)
            self._depth_aug_device = device
        return self._depth_aug

    @staticmethod
    def _warp_device_from_torch(device: torch.device) -> str:
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            return f"cuda:{index}"
        return device.type
