"""Corrected depth-conditioned diffusion student for Flexiv DepthDagger v3."""

from __future__ import annotations

import copy

import torch

from rsl_rl.modules.student_teacher_depth_diffusion_policy import StudentTeacherDepthDiffusionPolicy
from rsl_rl.modules.student_teacher_diffusion_policy import DiffusionLowdimStudent


class StudentTeacherDepthDiffusionPolicyV3(StudentTeacherDepthDiffusionPolicy):
    """Diffusion student with action normalization, EMA, and consistent sequence masking."""

    uses_execution_mask = True
    preserve_rollout_history_on_update = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        action_normalizer_warmup_iterations: int = 10,
        action_range_margin: float = 0.05,
        ema_decay: float = 0.999,
        use_ema: bool = True,
        use_depth_encoder: bool = True,
        augmentation_start_iteration: int = 500,
        **kwargs,
    ):
        self.use_depth_encoder = bool(use_depth_encoder)
        self.action_normalizer_warmup_iterations = int(action_normalizer_warmup_iterations)
        self.action_range_margin = float(action_range_margin)
        self.ema_decay = float(ema_decay)
        self.use_ema = bool(use_ema)
        self.augmentation_start_iteration = int(augmentation_start_iteration)
        self.training_iteration = 0
        self.last_normalized_saturation_ratio = 0.0
        self.last_action_clip_ratio = 0.0
        diffusion_kwargs = {
            "horizon": int(kwargs.get("horizon", 8)),
            "n_obs_steps": int(kwargs.get("n_obs_steps", 5)),
            "n_action_steps": int(kwargs.get("n_action_steps", 1)),
            "num_inference_steps": int(kwargs.get("num_inference_steps", 8)),
            "obs_cond_dim": kwargs.get("obs_cond_dim", 256),
            "diffusion_step_embed_dim": int(kwargs.get("diffusion_step_embed_dim", 128)),
            "down_dims": kwargs.get("down_dims", [128, 256, 512]),
            "kernel_size": int(kwargs.get("kernel_size", 5)),
            "n_groups": int(kwargs.get("n_groups", 8)),
            "cond_predict_scale": bool(kwargs.get("cond_predict_scale", True)),
            "num_train_timesteps": int(kwargs.get("num_train_timesteps", 100)),
            "beta_start": float(kwargs.get("beta_start", 0.0001)),
            "beta_end": float(kwargs.get("beta_end", 0.02)),
            "beta_schedule": kwargs.get("beta_schedule", "squaredcos_cap_v2"),
            "clip_sample": bool(kwargs.get("clip_sample", True)),
            "prediction_type": kwargs.get("prediction_type", "epsilon"),
            "oa_step_convention": bool(kwargs.get("oa_step_convention", True)),
        }
        kwargs["depth_input_channels"] = 2
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        if not self.use_depth_encoder:
            num_low_dim_obs = sum(int(obs[group].shape[-1]) for group in self.student_low_dim_obs_groups)
            self.student = DiffusionLowdimStudent(
                obs_dim=num_low_dim_obs,
                action_dim=self.num_actions,
                **diffusion_kwargs,
            )

        self.ema_student = copy.deepcopy(self.student)
        self.ema_student.requires_grad_(False)
        self.ema_student.eval()

        self.register_buffer("action_observed_min", torch.full((self.num_actions,), torch.inf))
        self.register_buffer("action_observed_max", torch.full((self.num_actions,), -torch.inf))
        self.register_buffer("action_center", torch.zeros(self.num_actions))
        self.register_buffer("action_half_range", torch.ones(self.num_actions))
        self.register_buffer("action_normalizer_frozen", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))

    def set_training_iteration(self, iteration: int):
        self.training_iteration = int(iteration)
        if self.training_iteration >= self.action_normalizer_warmup_iterations:
            self.freeze_action_normalizer()
            if not bool(self.action_normalizer_frozen.item()):
                raise RuntimeError("V3 action normalizer warmup ended without finite teacher-action statistics.")

    def should_skip_distillation_update(self) -> bool:
        return not bool(self.action_normalizer_frozen.item())

    def optimizer_parameters(self):
        return self.student.parameters()

    @torch.no_grad()
    def observe_teacher_actions(self, actions: torch.Tensor):
        flat = actions.detach().reshape(-1, self.num_actions)
        if flat.numel() == 0:
            return
        if not bool(self.action_normalizer_frozen.item()):
            self.action_observed_min.copy_(torch.minimum(self.action_observed_min, flat.amin(dim=0)))
            self.action_observed_max.copy_(torch.maximum(self.action_observed_max, flat.amax(dim=0)))
        else:
            outside = (flat < (self.action_center - self.action_half_range)) | (
                flat > (self.action_center + self.action_half_range)
            )
            self.last_action_clip_ratio = float(outside.float().mean().item())

    @torch.no_grad()
    def freeze_action_normalizer(self):
        if bool(self.action_normalizer_frozen.item()):
            return
        if not torch.isfinite(self.action_observed_min).all() or not torch.isfinite(self.action_observed_max).all():
            return
        center = 0.5 * (self.action_observed_min + self.action_observed_max)
        half_range = 0.5 * (self.action_observed_max - self.action_observed_min)
        half_range = half_range.clamp_min(1.0e-4) * (1.0 + self.action_range_margin)
        self.action_center.copy_(center)
        self.action_half_range.copy_(half_range)
        self.action_normalizer_frozen.fill_(True)

    def normalize_actions(self, actions: torch.Tensor, clip: bool = True) -> torch.Tensor:
        normalized = (actions - self.action_center) / self.action_half_range.clamp_min(1.0e-6)
        return normalized.clamp(-1.0, 1.0) if clip else normalized

    def unnormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        return normalized_actions * self.action_half_range + self.action_center

    def action_error_metrics(self, student_actions: torch.Tensor, teacher_actions: torch.Tensor) -> dict[str, float]:
        normalized_error = self.normalize_actions(student_actions, clip=False) - self.normalize_actions(
            teacher_actions, clip=False
        )
        return {
            "normalized_action_mse": float(normalized_error.pow(2).mean().item()),
            "action_mae": float((student_actions - teacher_actions).abs().mean().item()),
        }

    def compute_distillation_loss(
        self,
        obs_sequence: dict[str, torch.Tensor],
        privileged_action_sequence: torch.Tensor,
        execution_mask_sequence: torch.Tensor | None = None,
    ):
        low_dim_obs, depth_image = self.get_student_obs_sequence(obs_sequence)
        low_dim_obs = self.student_obs_normalizer(low_dim_obs)
        normalized_actions = self.normalize_actions(privileged_action_sequence)
        loss_mask = self.build_sequence_loss_mask(execution_mask_sequence, normalized_actions.shape[:2])
        if self.use_depth_encoder:
            return self.student.compute_loss(low_dim_obs, depth_image, normalized_actions, loss_mask=loss_mask)
        return self.student.compute_loss(low_dim_obs, normalized_actions, loss_mask=loss_mask)

    def build_sequence_loss_mask(
        self,
        execution_mask_sequence: torch.Tensor | None,
        batch_time_shape: tuple[int, int],
    ) -> torch.Tensor | None:
        if execution_mask_sequence is None:
            return None
        execution = execution_mask_sequence.to(dtype=torch.bool)
        all_teacher = execution.all(dim=1)
        all_student = (~execution).all(dim=1)
        mask = torch.zeros(batch_time_shape, dtype=torch.bool, device=execution.device)
        mask[all_teacher] = True
        diffusion = self.student.diffusion_policy if self.use_depth_encoder else self.student
        current_index = self.n_obs_steps - 1 if diffusion.oa_step_convention else self.n_obs_steps
        mask[all_student, current_index] = True
        return mask

    @torch.no_grad()
    def update_ema(self):
        if not self.use_ema:
            return
        if not bool(self.ema_initialized.item()):
            self.ema_student.load_state_dict(self.student.state_dict())
            self.ema_initialized.fill_(True)
            return
        online_parameters = dict(self.student.named_parameters())
        ema_parameters = dict(self.ema_student.named_parameters())
        if online_parameters.keys() != ema_parameters.keys():
            raise RuntimeError("Online and EMA student parameter structures differ.")
        for name, parameter in online_parameters.items():
            ema_parameter = ema_parameters[name]
            ema_parameter.lerp_(parameter.detach(), 1.0 - self.ema_decay)
        online_buffers = dict(self.student.named_buffers())
        ema_buffers = dict(self.ema_student.named_buffers())
        if online_buffers.keys() != ema_buffers.keys():
            raise RuntimeError("Online and EMA student buffer structures differ.")
        for name, buffer in online_buffers.items():
            ema_buffers[name].copy_(buffer)

    def get_student_obs(self, obs) -> tuple[torch.Tensor, torch.Tensor | None]:
        low_dim_obs = torch.cat([obs[group] for group in self.student_low_dim_obs_groups], dim=-1)
        if not self.use_depth_encoder:
            return low_dim_obs, None
        depth_image = self._extract_depth(obs)
        if self.training and self.enable_depth_augmentation and self.training_iteration >= self.augmentation_start_iteration:
            depth_image = self._augment_normalized_depth(depth_image)
        return low_dim_obs, depth_image

    def get_student_obs_sequence(self, obs_sequence) -> tuple[torch.Tensor, torch.Tensor | None]:
        low_dim_obs = torch.cat([obs_sequence[group] for group in self.student_low_dim_obs_groups], dim=-1)
        if not self.use_depth_encoder:
            return low_dim_obs, None
        depth_image = self._extract_depth_sequence(obs_sequence)
        if self.training and self.enable_depth_augmentation and self.training_iteration >= self.augmentation_start_iteration:
            batch_size, horizon = depth_image.shape[:2]
            flat = depth_image.reshape(batch_size * horizon, *depth_image.shape[2:])
            flat = self._augment_normalized_depth(flat)
            depth_image = flat.reshape(batch_size, horizon, *flat.shape[1:])
        return low_dim_obs, depth_image

    def _extract_depth(self, obs) -> torch.Tensor:
        depth = obs[self.student_depth_obs_group]
        if not torch.is_tensor(depth):
            depth = depth["head_depth"] if "head_depth" in depth else depth[next(iter(depth.keys()))]
        depth = depth.to(dtype=torch.float32)
        if depth.ndim == 4 and depth.shape[1] != 2 and depth.shape[-1] == 2:
            depth = depth.permute(0, 3, 1, 2)
        if depth.ndim != 4 or depth.shape[1] != 2:
            raise ValueError(f"Expected v3 depth [N, 2, H, W], got {tuple(depth.shape)}.")
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (depth[:, 1:2] > 0.5).to(dtype=depth.dtype)
        return torch.cat((depth[:, :1].clamp(-1.0, 1.0) * mask, mask), dim=1)

    def _extract_depth_sequence(self, obs_sequence) -> torch.Tensor:
        depth = obs_sequence[self.student_depth_obs_group]
        if not torch.is_tensor(depth):
            depth = depth["head_depth"] if "head_depth" in depth else depth[next(iter(depth.keys()))]
        depth = depth.to(dtype=torch.float32)
        if depth.ndim == 5 and depth.shape[2] != 2 and depth.shape[-1] == 2:
            depth = depth.permute(0, 1, 4, 2, 3)
        if depth.ndim != 5 or depth.shape[2] != 2:
            raise ValueError(f"Expected v3 depth sequence [B, T, 2, H, W], got {tuple(depth.shape)}.")
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (depth[:, :, 1:2] > 0.5).to(dtype=depth.dtype)
        return torch.cat((depth[:, :, :1].clamp(-1.0, 1.0) * mask, mask), dim=2)

    def _augment_normalized_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
        normalized, mask = depth_image[:, :1], depth_image[:, 1:2]
        metric = (normalized + 1.0) * 0.5 * (self.depth_max - self.depth_min) + self.depth_min
        augmented = super()._augment_depth(metric * mask)
        new_mask = (augmented > 1.0e-8).to(dtype=augmented.dtype)
        normalized = 2.0 * (augmented.clamp(self.depth_min, self.depth_max) - self.depth_min) / (
            self.depth_max - self.depth_min
        ) - 1.0
        return torch.cat((normalized * new_mask, new_mask), dim=1)

    def _act_from_history(self, obs):
        low_dim_obs, depth_image = self.get_student_obs(obs)
        low_dim_obs = self.student_obs_normalizer(low_dim_obs)
        self._update_v3_history(low_dim_obs, depth_image)
        num_envs = low_dim_obs.shape[0]
        if self._action_queue is None or self._action_queue.shape[0] != num_envs:
            self._action_queue = torch.zeros(
                num_envs, self.n_action_steps, self.num_actions, device=low_dim_obs.device, dtype=low_dim_obs.dtype
            )
            self._action_queue_lengths = torch.zeros(num_envs, dtype=torch.long, device=low_dim_obs.device)
        env_ids = (self._action_queue_lengths <= 0).nonzero(as_tuple=False).reshape(-1)
        if env_ids.numel() > 0:
            inference_student = (
                self.ema_student if self.use_ema and bool(self.ema_initialized.item()) else self.student
            )
            with torch.no_grad():
                if self.use_depth_encoder:
                    chunks = inference_student.predict_action_chunks(
                        self._low_dim_history[env_ids], self._depth_history[env_ids]
                    )
                else:
                    chunks = inference_student.predict_action_chunks(self._low_dim_history[env_ids])
                self.last_normalized_saturation_ratio = float((chunks.abs() >= 0.999).float().mean().item())
                chunks = self.unnormalize_actions(chunks)
            self._set_action_chunks(env_ids, chunks)
        action = self._action_queue[:, 0].clone()
        self._action_queue[:, :-1] = self._action_queue[:, 1:].clone()
        self._action_queue[:, -1] = 0
        self._action_queue_lengths = torch.clamp(self._action_queue_lengths - 1, min=0)
        return action

    def _update_v3_history(self, low_dim_obs: torch.Tensor, depth_image: torch.Tensor | None):
        num_envs = low_dim_obs.shape[0]
        if self._low_dim_history is None or self._low_dim_history.shape[0] != num_envs:
            self._low_dim_history = low_dim_obs.unsqueeze(1).repeat(1, self.n_obs_steps, 1)
            if depth_image is not None:
                self._depth_history = depth_image.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
            self._pending_history_reset = torch.zeros(num_envs, dtype=torch.bool, device=low_dim_obs.device)
            return
        reset_ids = self._pending_history_reset.nonzero(as_tuple=False).reshape(-1)
        if reset_ids.numel() > 0:
            self._low_dim_history[reset_ids] = low_dim_obs[reset_ids].unsqueeze(1).repeat(1, self.n_obs_steps, 1)
            if depth_image is not None:
                self._depth_history[reset_ids] = depth_image[reset_ids].unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
            self._pending_history_reset[reset_ids] = False
        self._low_dim_history = torch.cat((self._low_dim_history[:, 1:], low_dim_obs.unsqueeze(1)), dim=1)
        if depth_image is not None:
            self._depth_history = torch.cat((self._depth_history[:, 1:], depth_image.unsqueeze(1)), dim=1)

    def train(self, mode=True):
        super().train(mode)
        self.ema_student.eval()
        return self

    def diagnostics(self) -> dict[str, float]:
        return {
            "action_normalizer_frozen": float(self.action_normalizer_frozen.item()),
            "action_scale_min": float(self.action_half_range.min().item()),
            "action_scale_max": float(self.action_half_range.max().item()),
            "action_clip_ratio": self.last_action_clip_ratio,
            "normalized_action_saturation_ratio": self.last_normalized_saturation_ratio,
            "ema_initialized": float(self.ema_initialized.item()),
        }
