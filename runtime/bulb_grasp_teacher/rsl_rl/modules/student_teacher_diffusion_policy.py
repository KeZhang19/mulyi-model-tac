# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-student distillation module with a low-dimensional diffusion-policy student."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


def _ensure_local_diffusion_policy_on_path() -> None:
    """Allow importing the vendored diffusion_policy checkout from this repository."""

    repo_root = Path(__file__).resolve().parents[3]
    local_root = repo_root / "diffusion_policy"
    if (local_root / "diffusion_policy").is_dir() and str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))


def _load_diffusion_policy_components():
    _ensure_local_diffusion_policy_on_path()
    try:
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "StudentTeacherDiffusionPolicy requires the local diffusion_policy package. "
            "Install requirement.txt or run from the BrainCo repository root."
        ) from exc

    try:
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    except ModuleNotFoundError:
        DDIMScheduler = LocalDDIMScheduler
    return ConditionalUnet1D, DDIMScheduler


class LocalDDIMScheduler:
    """Small DDIM scheduler fallback used when diffusers is not installed."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
    ):
        self.config = SimpleNamespace(
            num_train_timesteps=int(num_train_timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            beta_schedule=str(beta_schedule),
            clip_sample=bool(clip_sample),
            set_alpha_to_one=bool(set_alpha_to_one),
            steps_offset=int(steps_offset),
            prediction_type=prediction_type,
        )
        self.clip_sample = bool(clip_sample)
        self.set_alpha_to_one = bool(set_alpha_to_one)
        self.steps_offset = int(steps_offset)
        if beta_schedule == "squaredcos_cap_v2":
            betas = self._betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported fallback beta_schedule: {beta_schedule}")
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long)
        self._step_ratio = 1

    @staticmethod
    def _betas_for_alpha_bar(num_train_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
        def alpha_bar(time_step: float) -> float:
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_train_timesteps):
            t1 = i / num_train_timesteps
            t2 = (i + 1) / num_train_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    def set_timesteps(self, num_inference_steps: int):
        num_inference_steps = int(num_inference_steps)
        self._step_ratio = max(self.config.num_train_timesteps // num_inference_steps, 1)
        timesteps = torch.arange(
            self.config.num_train_timesteps - 1,
            -1,
            -self._step_ratio,
            dtype=torch.long,
        )
        self.timesteps = timesteps[:num_inference_steps]

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt().flatten()
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]).sqrt().flatten()
        while sqrt_alpha_prod.ndim < original_samples.ndim:
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor):
        if torch.is_tensor(timestep):
            timestep = int(timestep.item())
        prev_timestep = timestep - self._step_ratio

        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod.to(device=sample.device, dtype=sample.dtype)
        )
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
        else:
            raise ValueError(f"Unsupported fallback prediction_type: {self.config.prediction_type}")

        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1, 1)

        pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * pred_epsilon
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        return SimpleNamespace(prev_sample=prev_sample)


class DiffusionLowdimStudent(nn.Module):
    """Small wrapper around diffusion_policy's low-dimensional ConditionalUnet1D actor."""

    def __init__(
        self,
        obs_dim: int,
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
    ):
        super().__init__()
        if horizon < n_obs_steps:
            raise ValueError(f"horizon ({horizon}) must be >= n_obs_steps ({n_obs_steps}).")
        if horizon < n_action_steps:
            raise ValueError(f"horizon ({horizon}) must be >= n_action_steps ({n_action_steps}).")

        ConditionalUnet1D, DDIMScheduler = _load_diffusion_policy_components()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.oa_step_convention = bool(oa_step_convention)

        raw_global_cond_dim = self.obs_dim * self.n_obs_steps
        if obs_cond_dim is None or int(obs_cond_dim) <= 0:
            self.obs_cond_encoder = nn.Identity()
            global_cond_dim = raw_global_cond_dim
        else:
            global_cond_dim = int(obs_cond_dim)
            self.obs_cond_encoder = nn.Sequential(
                nn.Linear(raw_global_cond_dim, global_cond_dim),
                nn.Mish(),
            )

        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=prediction_type,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def _action_slice(self) -> slice:
        start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
        end = start + self.n_action_steps
        if end > self.horizon:
            raise ValueError(
                f"Action slice [{start}:{end}] exceeds horizon {self.horizon}. "
                "Increase horizon or reduce n_obs_steps/n_action_steps."
            )
        return slice(start, end)

    def conditional_sample(self, global_cond: torch.Tensor) -> torch.Tensor:
        batch_size = global_cond.shape[0]
        trajectory = torch.randn(
            (batch_size, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
        )
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        condition_data = torch.zeros_like(trajectory)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(trajectory, timestep, global_cond=global_cond)
            trajectory = self.noise_scheduler.step(model_output, timestep, trajectory).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action_chunks(self, obs_history: torch.Tensor) -> torch.Tensor:
        if obs_history.shape[1] < self.n_obs_steps:
            raise ValueError(
                f"Expected at least {self.n_obs_steps} observation steps, got {obs_history.shape[1]}."
            )
        obs_cond = obs_history[:, -self.n_obs_steps :, :].reshape(obs_history.shape[0], -1)
        obs_cond = self.obs_cond_encoder(obs_cond)
        action_pred = self.conditional_sample(obs_cond)
        return action_pred[:, self._action_slice(), :]

    def compute_loss(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if obs_sequence.shape[1] != self.horizon:
            raise ValueError(f"Expected obs horizon {self.horizon}, got {obs_sequence.shape[1]}.")
        if action_sequence.shape[1] != self.horizon:
            raise ValueError(f"Expected action horizon {self.horizon}, got {action_sequence.shape[1]}.")

        global_cond = obs_sequence[:, : self.n_obs_steps, :].reshape(obs_sequence.shape[0], -1)
        global_cond = self.obs_cond_encoder(global_cond)
        trajectory = action_sequence
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        if loss_mask is not None:
            input_mask = loss_mask.to(device=noisy_trajectory.device, dtype=torch.bool)
            if input_mask.ndim == 2:
                input_mask = input_mask.unsqueeze(-1)
            if input_mask.shape[-1] == 1:
                input_mask = input_mask.expand_as(noisy_trajectory)
            # Unsupervised future targets must not leak into the supervised action through U-Net convolutions.
            noisy_trajectory = torch.where(input_mask, noisy_trajectory, torch.randn_like(noisy_trajectory))
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported diffusion prediction_type: {pred_type}")

        squared_error = (pred - target) ** 2
        if loss_mask is None:
            return squared_error.mean()
        loss_mask = loss_mask.to(device=squared_error.device, dtype=squared_error.dtype)
        if loss_mask.ndim == 2:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask.shape[-1] == 1:
            loss_mask = loss_mask.expand_as(squared_error)
        if loss_mask.shape != squared_error.shape:
            raise ValueError(
                f"Diffusion loss mask shape {tuple(loss_mask.shape)} does not match {tuple(squared_error.shape)}."
            )
        denominator = loss_mask.sum().clamp_min(1.0)
        return (squared_error * loss_mask).sum() / denominator


class StudentTeacherDiffusionPolicy(nn.Module):
    """Online distillation module with an MLP teacher and diffusion-policy student."""

    is_recurrent = False
    uses_diffusion_policy = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: list[int] | None = None,
        teacher_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
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
        **kwargs,
    ):
        if teacher_hidden_dims is None:
            teacher_hidden_dims = [512, 256, 128]
        if kwargs:
            print(
                "StudentTeacherDiffusionPolicy.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_actions = int(num_actions)

        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "Diffusion student only supports 1D observation groups."
            num_student_obs += obs[obs_group].shape[-1]

        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "Diffusion teacher only supports 1D observation groups."
            num_teacher_obs += obs[obs_group].shape[-1]

        self.student = DiffusionLowdimStudent(
            obs_dim=num_student_obs,
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
        )
        # Keep compatibility with old config fields. Diffusion does not use these MLP hidden dims.
        self.student_hidden_dims = student_hidden_dims

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()
        print(
            "Diffusion student: "
            f"obs_dim={num_student_obs}, action_dim={num_actions}, horizon={horizon}, "
            f"n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}, "
            f"num_inference_steps={num_inference_steps}, obs_cond_dim={obs_cond_dim}, "
            f"down_dims={list(down_dims)}"
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
        self._obs_history: torch.Tensor | None = None
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
            if self._obs_history is not None:
                self._obs_history = None
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
        student_obs = self.get_student_obs_sequence(obs_sequence)
        student_obs = self.student_obs_normalizer(student_obs)
        return self.student.compute_loss(student_obs, privileged_action_sequence)

    def get_student_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_student_obs_sequence(self, obs_sequence: dict[str, torch.Tensor]):
        return torch.cat([obs_sequence[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_teacher_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["teacher"]], dim=-1)

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
            self.student_obs_normalizer.update(self.get_student_obs(obs))

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
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        self._update_obs_history(student_obs)

        if self._action_queue is None or self._action_queue.shape[0] != student_obs.shape[0]:
            self._action_queue = torch.zeros(
                student_obs.shape[0],
                self.n_action_steps,
                self.num_actions,
                dtype=student_obs.dtype,
                device=student_obs.device,
            )
            self._action_queue_lengths = torch.zeros(student_obs.shape[0], dtype=torch.long, device=student_obs.device)
        elif self._action_queue_lengths is None or self._action_queue_lengths.numel() != student_obs.shape[0]:
            self._action_queue_lengths = torch.zeros(student_obs.shape[0], dtype=torch.long, device=student_obs.device)
        else:
            self._action_queue = self._action_queue.to(device=student_obs.device, dtype=student_obs.dtype)
            self._action_queue_lengths = self._action_queue_lengths.to(device=student_obs.device)

        env_ids = (self._action_queue_lengths <= 0).nonzero(as_tuple=False).reshape(-1)

        if env_ids.numel() > 0:
            with torch.no_grad():
                chunks = self.student.predict_action_chunks(self._obs_history[env_ids])
            self._set_action_chunks(env_ids, chunks)

        action = self._action_queue[:, 0].clone()
        self._action_queue[:, :-1] = self._action_queue[:, 1:].clone()
        self._action_queue[:, -1] = 0
        self._action_queue_lengths = torch.clamp(self._action_queue_lengths - 1, min=0)
        return action

    def _update_obs_history(self, student_obs: torch.Tensor):
        num_envs, obs_dim = student_obs.shape
        if self._obs_history is None or self._obs_history.shape[0] != num_envs:
            self._obs_history = student_obs.unsqueeze(1).repeat(1, self.n_obs_steps, 1)
            self._pending_history_reset = torch.zeros(num_envs, dtype=torch.bool, device=student_obs.device)
            return

        self._obs_history = self._obs_history.to(device=student_obs.device, dtype=student_obs.dtype)
        if self._pending_history_reset is not None:
            self._pending_history_reset = self._pending_history_reset.to(device=student_obs.device)
            reset_ids = self._pending_history_reset.nonzero(as_tuple=False).reshape(-1)
            if reset_ids.numel() > 0:
                self._obs_history[reset_ids] = student_obs[reset_ids].unsqueeze(1).repeat(1, self.n_obs_steps, 1)
                self._pending_history_reset[reset_ids] = False

        self._obs_history[:, :-1] = self._obs_history[:, 1:].clone()
        self._obs_history[:, -1] = student_obs

    def _set_action_chunks(self, env_ids: torch.Tensor, chunks: torch.Tensor):
        num_envs = self._obs_history.shape[0]
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
