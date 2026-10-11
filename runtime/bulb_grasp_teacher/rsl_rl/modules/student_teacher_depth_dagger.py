# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


_WARP_KERNELS = None


def _get_warp_kernels():
    """Lazily import Warp and define depth augmentation kernels."""

    global _WARP_KERNELS
    if _WARP_KERNELS is not None:
        return _WARP_KERNELS

    try:
        import warp as wp  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Depth augmentation for StudentTeacherDepthDagger requires the `warp` package. "
            "Install/run inside the Isaac Lab environment or disable enable_depth_augmentation."
        ) from exc
    wp.init()
    wp_torch = getattr(wp, "torch", None)
    if wp_torch is None or not hasattr(wp_torch, "from_torch"):
        wp_torch = wp

    globals()["wp"] = wp

    @wp.kernel
    def add_pixel_dropout_and_randu_kernel(
        depths: wp.array(dtype=float, ndim=3),
        rand_dropout: wp.array(dtype=float, ndim=3),
        rand_u: wp.array(dtype=float, ndim=3),
        rand_u_values: wp.array(dtype=float, ndim=3),
        p_dropout: float,
        p_randu: float,
        d_min: float,
        d_max: float,
        kernel_size: int,
        height: int,
        width: int,
        seed: int,
    ):
        batch_index, pixel_row, pixel_column = wp.tid()

        if rand_dropout[batch_index, pixel_row, pixel_column] <= p_dropout:
            depths[batch_index, pixel_row, pixel_column] = 0.0

        if rand_u[batch_index, pixel_row, pixel_column] <= p_randu:
            rand_depth = rand_u_values[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min
            depths[batch_index, pixel_row, pixel_column] = rand_depth

            for i in range(kernel_size):
                for j in range(kernel_size):
                    out_row = pixel_row + i
                    out_col = pixel_column + j
                    if out_row < height and out_col < width:
                        state = wp.rand_init(seed, batch_index + pixel_row + pixel_column + i + j)
                        if wp.randf(state) < 0.25:
                            depths[batch_index, out_row, out_col] = rand_depth

    @wp.kernel
    def add_sticks_kernel(
        depths: wp.array(dtype=float, ndim=3),
        rand_sticks: wp.array(dtype=float, ndim=3),
        rand_sticks_depths: wp.array(dtype=float, ndim=3),
        p_stick: float,
        max_stick_len: float,
        max_stick_width: float,
        height: int,
        width: int,
        d_min: float,
        d_max: float,
        seed: int,
    ):
        batch_index, pixel_row, pixel_column = wp.tid()

        if rand_sticks[batch_index, pixel_row, pixel_column] > p_stick:
            return

        rand_depth = rand_sticks_depths[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min
        stick_width = float(0.0)
        stick_len = float(0.0)
        stick_rot = float(0.0)

        for i in range(3):
            state = wp.rand_init(seed, batch_index + pixel_row + pixel_column + i)
            if i == 0:
                stick_width = wp.randf(state) * max_stick_width
            if i == 1:
                stick_len = wp.randf(state) * max_stick_len + 1.0
            if i == 2:
                stick_rot = wp.randf(state) * (3.14 * 2.0)

        for i in range(int(wp.rint(stick_len))):
            hor_coord = float(pixel_column + i)
            vert_coord = wp.floor(float(i) * wp.sin(stick_rot)) + float(pixel_row)
            base_row = int(vert_coord)
            base_col = int(hor_coord)

            if base_row >= 0 and base_row < height and base_col >= 0 and base_col < width:
                depths[batch_index, base_row, base_col] = rand_depth

            for j in range(1, int(max_stick_width)):
                out_row = base_row
                out_col = base_col
                if stick_rot > (3.14 / 4.0) and stick_rot < (3.0 * 3.14 / 4.0):
                    out_row = base_row + j
                elif stick_rot > (5.0 * 3.14 / 4.0) and stick_rot < (7.0 * 3.14 / 4.0):
                    out_row = base_row + j
                else:
                    out_col = base_col + j

                if out_row >= 0 and out_row < height and out_col >= 0 and out_col < width:
                    depths[batch_index, out_row, out_col] = rand_depth

    @wp.kernel
    def add_correlated_noise_kernel(
        depths: wp.array(dtype=float, ndim=3),
        rand_sigma_s_x: wp.array(dtype=float, ndim=3),
        rand_sigma_s_y: wp.array(dtype=float, ndim=3),
        rand_sigma_d: wp.array(dtype=float, ndim=3),
        height: int,
        width: int,
        d_min: float,
        d_max: float,
        noisy_depths: wp.array(dtype=float, ndim=3),
    ):
        batch_index, pixel_row, pixel_column = wp.tid()

        nx = rand_sigma_s_x[batch_index, pixel_row, pixel_column]
        ny = rand_sigma_s_y[batch_index, pixel_row, pixel_column]

        u = nx + float(pixel_column)
        v = ny + float(pixel_row)

        u0 = int(u)
        v0 = int(v)
        u1 = u0 + 1
        v1 = v0 + 1

        fu = u - float(u0)
        fv = v - float(v0)

        u0 = wp.max(0, wp.min(u0, width - 1))
        u1 = wp.max(0, wp.min(u1, width - 1))
        v0 = wp.max(0, wp.min(v0, height - 1))
        v1 = wp.max(0, wp.min(v1, height - 1))

        w_00 = (1.0 - fu) * (1.0 - fv)
        w_01 = (1.0 - fu) * fv
        w_10 = fu * (1.0 - fv)
        w_11 = fu * fv

        noisy_depth = (
            depths[batch_index, v0, u0] * w_00
            + depths[batch_index, v0, u1] * w_01
            + depths[batch_index, v1, u0] * w_10
            + depths[batch_index, v1, u1] * w_11
        )

        if noisy_depth <= 1.0e-8:
            noisy_depths[batch_index, pixel_row, pixel_column] = 0.0
            return

        baseline = float(35130.0)
        ref = float(8.0)
        denominator = baseline / noisy_depth + rand_sigma_d[batch_index, pixel_row, pixel_column] + 0.5
        noisy_depths[batch_index, pixel_row, pixel_column] = baseline / (wp.rint(denominator / ref) * ref)

    @wp.kernel
    def add_normal_noise_kernel(
        depths: wp.array(dtype=float, ndim=3),
        rand_sigma_theta: wp.array(dtype=float, ndim=3),
        cam_matrix: wp.mat44f,
        height: int,
        width: int,
        d_min: float,
        d_max: float,
    ):
        batch_index, pixel_row, pixel_column = wp.tid()

        if pixel_row == (height - 1) or pixel_column == (width - 1):
            return

        depth_center = depths[batch_index, pixel_row, pixel_column]
        depth_right = depths[batch_index, pixel_row, pixel_column + 1]
        depth_up = depths[batch_index, pixel_row + 1, pixel_column]
        if depth_center <= 1.0e-8 or depth_right <= 1.0e-8 or depth_up <= 1.0e-8:
            return

        uv = wp.vec4f(float(pixel_column), float(pixel_row), 1.0, 1.0)
        xyzw = wp.inverse(cam_matrix) * uv
        point_3d = wp.vec3f(
            depth_center * xyzw[0] / xyzw[3],
            depth_center * xyzw[1] / xyzw[3],
            depth_center * xyzw[2] / xyzw[3],
        )

        uv = wp.vec4f(float(pixel_column + 1), float(pixel_row), 1.0, 1.0)
        xyzw = wp.inverse(cam_matrix) * uv
        point_3d_01 = wp.vec3f(
            depth_right * xyzw[0] / xyzw[3],
            depth_right * xyzw[1] / xyzw[3],
            depth_right * xyzw[2] / xyzw[3],
        )

        uv = wp.vec4f(float(pixel_column), float(pixel_row + 1), 1.0, 1.0)
        xyzw = wp.inverse(cam_matrix) * uv
        point_3d_10 = wp.vec3f(
            depth_up * xyzw[0] / xyzw[3],
            depth_up * xyzw[1] / xyzw[3],
            depth_up * xyzw[2] / xyzw[3],
        )

        x_axis = wp.normalize(point_3d_01 - point_3d)
        y_axis = wp.normalize(point_3d_10 - point_3d)
        normal = wp.cross(x_axis, y_axis)
        point_3d = point_3d + 1000.0 * rand_sigma_theta[batch_index, pixel_row, pixel_column] * normal
        depths[batch_index, pixel_row, pixel_column] = -point_3d[2] / 1000.0

    _WARP_KERNELS = {
        "wp": wp,
        "wp_torch": wp_torch,
        "add_pixel_dropout_and_randu": add_pixel_dropout_and_randu_kernel,
        "add_sticks": add_sticks_kernel,
        "add_correlated_noise": add_correlated_noise_kernel,
        "add_normal_noise": add_normal_noise_kernel,
    }
    return _WARP_KERNELS


class DepthAug:
    """Warp implementation of the DEXTRAH depth augmentation sequence."""

    def __init__(
        self,
        device: str,
        cam_matrix_values: tuple[tuple[float, float, float, float], ...] | None = None,
    ):
        self.device = device
        self.seed = 42
        self.kernel_size = 2
        self.cam_matrix_values = cam_matrix_values or (
            (2.2460368, 0.0, 0.0, 0.0),
            (0.0, 2.9947157, 0.0, 0.0),
            (0.0, 0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0e-3, 0.0),
        )

    def _cam_matrix(self):
        kernels = _get_warp_kernels()
        wp = kernels["wp"]
        cam_matrix = wp.mat44f()
        for row in range(4):
            for col in range(4):
                cam_matrix[row, col] = self.cam_matrix_values[row][col]
        return cam_matrix

    def add_pixel_dropout_and_randu(self, depths, p_dropout, p_randu, d_min, d_max):
        kernels = _get_warp_kernels()
        wp = kernels["wp"]
        wp_torch = kernels["wp_torch"]
        batch_size, height, width = depths.shape
        rand_dropout = torch.rand(batch_size, height, width, device=depths.device)
        rand_u = torch.rand(batch_size, height, width, device=depths.device)
        rand_u_values = torch.rand(batch_size, height, width, device=depths.device)

        wp.launch(
            kernel=kernels["add_pixel_dropout_and_randu"],
            dim=[batch_size, height, width],
            inputs=[
                wp_torch.from_torch(depths),
                wp_torch.from_torch(rand_dropout),
                wp_torch.from_torch(rand_u),
                wp_torch.from_torch(rand_u_values),
                p_dropout,
                p_randu,
                d_min,
                d_max,
                self.kernel_size,
                height,
                width,
                self.seed,
            ],
            device=self.device,
        )

    def add_sticks(self, depths, p_stick, max_stick_len, max_stick_width, d_min, d_max):
        kernels = _get_warp_kernels()
        wp = kernels["wp"]
        wp_torch = kernels["wp_torch"]
        batch_size, height, width = depths.shape
        rand_stick = torch.rand(batch_size, height, width, device=depths.device)
        rand_stick_depths = torch.rand(batch_size, height, width, device=depths.device)

        wp.launch(
            kernel=kernels["add_sticks"],
            dim=[batch_size, height, width],
            inputs=[
                wp_torch.from_torch(depths),
                wp_torch.from_torch(rand_stick),
                wp_torch.from_torch(rand_stick_depths),
                p_stick,
                max_stick_len,
                max_stick_width,
                height,
                width,
                d_min,
                d_max,
                self.seed,
            ],
            device=self.device,
        )

    def add_correlated_noise(self, depths, noisy_depths, sigma_s, sigma_d, d_min, d_max):
        kernels = _get_warp_kernels()
        wp = kernels["wp"]
        wp_torch = kernels["wp_torch"]
        batch_size, height, width = depths.shape
        rand_sigma_s_x = sigma_s * torch.randn(batch_size, height, width, device=depths.device)
        rand_sigma_s_y = sigma_s * torch.randn(batch_size, height, width, device=depths.device)
        rand_sigma_d = sigma_d * torch.randn(batch_size, height, width, device=depths.device)

        wp.launch(
            kernel=kernels["add_correlated_noise"],
            dim=[batch_size, height, width],
            inputs=[
                wp_torch.from_torch(depths),
                wp_torch.from_torch(rand_sigma_s_x),
                wp_torch.from_torch(rand_sigma_s_y),
                wp_torch.from_torch(rand_sigma_d),
                height,
                width,
                d_min,
                d_max,
            ],
            outputs=[wp_torch.from_torch(noisy_depths)],
            device=self.device,
        )

    def add_normal_noise(self, depths, sigma_theta, d_min, d_max):
        kernels = _get_warp_kernels()
        wp = kernels["wp"]
        wp_torch = kernels["wp_torch"]
        batch_size, height, width = depths.shape
        rand_sigma_theta = sigma_theta * torch.randn(batch_size, height, width, device=depths.device)

        wp.launch(
            kernel=kernels["add_normal_noise"],
            dim=[batch_size, height, width],
            inputs=[
                wp_torch.from_torch(depths),
                wp_torch.from_torch(rand_sigma_theta),
                self._cam_matrix(),
                height,
                width,
                d_min,
                d_max,
            ],
            device=self.device,
        )


class DepthEncoder(nn.Module):
    """Small CNN encoder for single-channel depth images."""

    def __init__(self, image_shape: tuple[int, int], out_features: int = 128, input_channels: int = 1):
        super().__init__()
        self.input_channels = int(input_channels)
        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_output_size = self._get_conv_output_size(image_shape)
        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, out_features)
        self.out_features = out_features

    def _get_conv_output_size(self, image_shape: tuple[int, int]) -> int:
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, image_shape[0], image_shape[1])
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
        return x.flatten(1).shape[-1]

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(depth)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DepthDaggerStudent(nn.Module):
    """Student action model combining normalized low-dimensional state with depth features."""

    def __init__(
        self,
        low_dim_obs_dim: int,
        depth_image_shape: tuple[int, int],
        depth_feature_dim: int,
        num_actions: int,
        hidden_dims: list[int],
        activation: str,
        n_obs_steps: int = 1,
        depth_input_channels: int = 1,
    ):
        super().__init__()
        self.n_obs_steps = int(n_obs_steps)
        self.depth_encoder = DepthEncoder(
            depth_image_shape, out_features=depth_feature_dim, input_channels=depth_input_channels
        )
        self.last_depth_feature_mean = 0.0
        self.last_depth_feature_std = 0.0
        frame_feature_dim = low_dim_obs_dim + depth_feature_dim
        self.action_mlp = MLP(frame_feature_dim * self.n_obs_steps, num_actions, hidden_dims, activation)

    def forward(self, low_dim_obs: torch.Tensor, depth_image: torch.Tensor) -> torch.Tensor:
        if low_dim_obs.ndim != 3 or low_dim_obs.shape[1] != self.n_obs_steps:
            raise ValueError(
                f"Expected low-dimensional history [B, {self.n_obs_steps}, D], got {tuple(low_dim_obs.shape)}."
            )
        if depth_image.ndim != 5 or depth_image.shape[1] != self.n_obs_steps:
            raise ValueError(
                f"Expected depth history [B, {self.n_obs_steps}, 1, H, W], got {tuple(depth_image.shape)}."
            )

        batch_size = low_dim_obs.shape[0]
        flat_depth = depth_image.reshape(batch_size * self.n_obs_steps, *depth_image.shape[2:])
        depth_features = self.depth_encoder(flat_depth).reshape(batch_size, self.n_obs_steps, -1)
        self.last_depth_feature_mean = float(depth_features.detach().mean().item())
        self.last_depth_feature_std = float(depth_features.detach().std().item())
        frame_features = torch.cat((low_dim_obs, depth_features), dim=-1)
        return self.action_mlp(frame_features.reshape(batch_size, -1))


class StudentTeacherDepthDagger(nn.Module):
    """Teacher-student distillation policy with a depth-image student."""

    is_recurrent = False

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
        n_obs_steps: int = 1,
        depth_input_channels: int = 1,
        **kwargs,
    ):
        if student_hidden_dims is None:
            student_hidden_dims = [256, 256, 128]
        if teacher_hidden_dims is None:
            teacher_hidden_dims = [512, 256, 128]
        if kwargs:
            print(
                "StudentTeacherDepthDagger.__init__ got unexpected arguments, which will be ignored: "
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
        self.enable_depth_augmentation = enable_depth_augmentation
        self.n_obs_steps = int(n_obs_steps)
        self.depth_input_channels = int(depth_input_channels)
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be positive, got {self.n_obs_steps}.")
        self._depth_aug: DepthAug | None = None
        self._depth_aug_device: str | None = None
        self._low_dim_history: torch.Tensor | None = None
        self._depth_history: torch.Tensor | None = None
        self._pending_history_reset: torch.Tensor | None = None

        num_student_low_dim_obs = 0
        for obs_group in self.student_low_dim_obs_groups:
            assert len(obs[obs_group].shape) == 2, "DepthDagger low-dimensional student observations must be 1D."
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
            assert len(obs[obs_group].shape) == 2, "DepthDagger teacher observations must be 1D."
            num_teacher_obs += obs[obs_group].shape[-1]

        self.student = DepthDaggerStudent(
            low_dim_obs_dim=num_student_low_dim_obs,
            depth_image_shape=self.depth_output_size,
            depth_feature_dim=depth_feature_dim,
            num_actions=num_actions,
            hidden_dims=student_hidden_dims,
            activation=activation,
            n_obs_steps=self.n_obs_steps,
            depth_input_channels=self.depth_input_channels,
        )
        print(
            "DepthDagger student history: "
            f"use_teacher_obs_for_student={self.use_teacher_obs_for_student}, "
            f"low_dim_obs_groups={self.student_low_dim_obs_groups}, n_obs_steps={self.n_obs_steps}, "
            f"low_dim_obs_per_step={num_student_low_dim_obs}, "
            f"depth_per_step=({self.depth_input_channels}, {self.depth_output_size[0]}, {self.depth_output_size[1]}), "
            f"action_dim={num_actions}"
        )

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_low_dim_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()
        print(f"DepthDagger student: {self.student}")

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
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution: Normal | None = None
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

    def reset(self, dones=None, hidden_states=None):
        if dones is None:
            self._low_dim_history = None
            self._depth_history = None
            self._pending_history_reset = None
            return

        dones = dones.reshape(-1).to(dtype=torch.bool)
        if dones.numel() == 0 or not dones.any():
            return
        if self._pending_history_reset is None or self._pending_history_reset.numel() != dones.numel():
            self._pending_history_reset = torch.zeros(dones.numel(), dtype=torch.bool, device=dones.device)
        self._pending_history_reset = self._pending_history_reset.to(device=dones.device)
        self._pending_history_reset |= dones

    def forward(self):
        raise NotImplementedError

    def act(self, obs):
        low_dim_obs, depth_image = self.get_student_obs(obs)
        low_dim_obs = self.student_obs_normalizer(low_dim_obs)
        low_dim_history, depth_history = self._update_obs_history(low_dim_obs, depth_image)
        self.update_distribution(low_dim_history, depth_history)
        return self.distribution.sample()

    def act_inference(self, obs):
        low_dim_obs, depth_image = self.get_student_obs(obs)
        low_dim_obs = self.student_obs_normalizer(low_dim_obs)
        low_dim_history, depth_history = self._update_obs_history(low_dim_obs, depth_image)
        return self.student(low_dim_history, depth_history)

    def evaluate(self, obs):
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def update_distribution(self, low_dim_obs: torch.Tensor, depth_image: torch.Tensor):
        mean = self.student(low_dim_obs, depth_image)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def get_student_obs(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        low_dim_obs = torch.cat([obs[obs_group] for obs_group in self.student_low_dim_obs_groups], dim=-1)
        depth_image = self._extract_depth(obs)
        if self.training and self.enable_depth_augmentation:
            depth_image = self._augment_depth(depth_image)
        else:
            depth_image = self._clip_depth(depth_image)
        return low_dim_obs, depth_image

    def get_teacher_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.teacher_obs_groups], dim=-1)

    def _update_obs_history(
        self, low_dim_obs: torch.Tensor, depth_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = low_dim_obs.shape[0]
        if self._low_dim_history is None or self._low_dim_history.shape[0] != num_envs:
            self._low_dim_history = low_dim_obs.unsqueeze(1).repeat(1, self.n_obs_steps, 1)
            self._depth_history = depth_image.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
            self._pending_history_reset = torch.zeros(num_envs, dtype=torch.bool, device=low_dim_obs.device)
            return self._low_dim_history, self._depth_history

        low_dim_history = self._low_dim_history.to(device=low_dim_obs.device, dtype=low_dim_obs.dtype)
        depth_history = self._depth_history.to(device=depth_image.device, dtype=depth_image.dtype)
        if self._pending_history_reset is not None:
            self._pending_history_reset = self._pending_history_reset.to(device=low_dim_obs.device)
            reset_ids = self._pending_history_reset.nonzero(as_tuple=False).reshape(-1)
            if reset_ids.numel() > 0:
                low_dim_history = low_dim_history.clone()
                depth_history = depth_history.clone()
                low_dim_history[reset_ids] = low_dim_obs[reset_ids].unsqueeze(1).repeat(
                    1, self.n_obs_steps, 1
                )
                depth_history[reset_ids] = depth_image[reset_ids].unsqueeze(1).repeat(
                    1, self.n_obs_steps, 1, 1, 1
                )
                self._pending_history_reset[reset_ids] = False

        # Keep prior history tensors immutable until accumulated losses have
        # completed backward; gradient accumulation spans several timesteps.
        self._low_dim_history = torch.cat((low_dim_history[:, 1:], low_dim_obs.unsqueeze(1)), dim=1)
        self._depth_history = torch.cat((depth_history[:, 1:], depth_image.unsqueeze(1)), dim=1)
        return self._low_dim_history, self._depth_history

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            low_dim_obs = torch.cat([obs[obs_group] for obs_group in self.student_low_dim_obs_groups], dim=-1)
            self.student_obs_normalizer.update(low_dim_obs)

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
        elif depth.ndim == 4 and depth.shape[1] != self.depth_input_channels and depth.shape[-1] == self.depth_input_channels:
            depth = depth.permute(0, 3, 1, 2)
        if depth.ndim != 4 or depth.shape[1] != self.depth_input_channels:
            raise ValueError(
                f"Expected depth observation with {self.depth_input_channels} channels, got shape {tuple(depth.shape)}."
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

    def load_state_dict(self, state_dict, strict=True):
        """Load PPO teacher checkpoints or full depth-distillation checkpoints."""

        if any("actor" in key for key in state_dict.keys()):
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("actor."):
                    teacher_state_dict[key.replace("actor.", "")] = value
                if key.startswith("actor_obs_normalizer."):
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")


class StudentTeacherDepthDaggerV2(StudentTeacherDepthDagger):
    """DepthDagger MLP student using normalized depth plus a validity mask."""

    def __init__(self, *args, augmentation_start_iteration: int = 500, **kwargs):
        kwargs["depth_input_channels"] = 2
        self.augmentation_start_iteration = int(augmentation_start_iteration)
        self.training_iteration = 0
        self.last_depth_valid_ratio = 0.0
        self.last_valid_depth_mean = 0.0
        self.last_valid_depth_std = 0.0
        super().__init__(*args, **kwargs)

    def set_training_iteration(self, iteration: int):
        self.training_iteration = int(iteration)

    def _extract_depth(self, obs) -> torch.Tensor:
        depth = obs[self.student_depth_obs_group]
        if not torch.is_tensor(depth):
            if "head_depth" in depth:
                depth = depth["head_depth"]
            elif len(depth.keys()) == 1:
                depth = depth[next(iter(depth.keys()))]
            else:
                raise ValueError(f"Depth observation group {self.student_depth_obs_group!r} is ambiguous.")
        depth = depth.to(dtype=torch.float32)
        if depth.ndim == 4 and depth.shape[1] != 2 and depth.shape[-1] == 2:
            depth = depth.permute(0, 3, 1, 2)
        if depth.ndim != 4 or depth.shape[1] != 2:
            raise ValueError(f"Expected v2 depth observation [N, 2, H, W], got {tuple(depth.shape)}.")
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = depth[:, :1].clamp(-1.0, 1.0)
        mask = (depth[:, 1:2] > 0.5).to(dtype=normalized.dtype)
        return torch.cat((normalized * mask, mask), dim=1)

    def get_student_obs(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        low_dim_obs = torch.cat([obs[group] for group in self.student_low_dim_obs_groups], dim=-1)
        depth_image = self._extract_depth(obs)
        mask = depth_image[:, 1:2]
        valid_values = depth_image[:, :1][mask.bool()]
        self.last_depth_valid_ratio = float(mask.mean().item())
        if valid_values.numel() > 0:
            metric_values = (valid_values + 1.0) * 0.5 * (self.depth_max - self.depth_min) + self.depth_min
            self.last_valid_depth_mean = float(metric_values.mean().item())
            self.last_valid_depth_std = (
                float(metric_values.std().item()) if metric_values.numel() > 1 else 0.0
            )
        if (
            self.training
            and self.enable_depth_augmentation
            and self.training_iteration >= self.augmentation_start_iteration
        ):
            depth_image = self._augment_normalized_depth(depth_image)
        return low_dim_obs, depth_image

    def _augment_normalized_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
        normalized, mask = depth_image[:, :1], depth_image[:, 1:2]
        metric = (normalized + 1.0) * 0.5 * (self.depth_max - self.depth_min) + self.depth_min
        augmented = super()._augment_depth(metric * mask)
        new_mask = (augmented > 1.0e-8).to(dtype=augmented.dtype)
        normalized = 2.0 * (augmented.clamp(self.depth_min, self.depth_max) - self.depth_min) / (
            self.depth_max - self.depth_min
        ) - 1.0
        return torch.cat((normalized * new_mask, new_mask), dim=1)

    def diagnostics(self) -> dict[str, float]:
        grad_sq = 0.0
        for parameter in self.student.depth_encoder.parameters():
            if parameter.grad is not None:
                grad_sq += float(parameter.grad.detach().pow(2).sum().item())
        return {
            "depth_valid_ratio": self.last_depth_valid_ratio,
            "valid_depth_mean": self.last_valid_depth_mean,
            "valid_depth_std": self.last_valid_depth_std,
            "depth_feature_mean": self.student.last_depth_feature_mean,
            "depth_feature_std": self.student.last_depth_feature_std,
            "depth_encoder_grad_norm": grad_sq**0.5,
        }


class StudentTeacherDepthDaggerV4(StudentTeacherDepthDaggerV2):
    """Distribution-stable MLP student with normalized teacher-action targets.

    The visual variant reuses the v2 depth encoder.  For the privileged oracle,
    ``use_depth_encoder=False`` builds a low-dimensional MLP over the selected
    observation groups and skips all depth processing.
    """

    uses_execution_mask = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        action_normalizer_warmup_iterations: int = 10,
        action_normalizer_clip: float = 5.0,
        action_std_floor: float = 1.0e-3,
        use_depth_encoder: bool = True,
        **kwargs,
    ):
        self.use_depth_encoder = bool(use_depth_encoder)
        self.action_normalizer_warmup_iterations = int(action_normalizer_warmup_iterations)
        self.action_normalizer_clip = float(action_normalizer_clip)
        self.action_std_floor = float(action_std_floor)
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        if not self.use_depth_encoder:
            low_dim_obs_dim = sum(int(obs[group].shape[-1]) for group in self.student_low_dim_obs_groups)
            hidden_dims = list(kwargs.get("student_hidden_dims") or [256, 256, 128])
            activation = kwargs.get("activation", "elu")
            self.student = MLP(low_dim_obs_dim * self.n_obs_steps, num_actions, hidden_dims, activation)

        self.register_buffer("action_stats_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("action_stats_mean", torch.zeros(num_actions))
        self.register_buffer("action_stats_m2", torch.zeros(num_actions))
        self.register_buffer("action_stats_std", torch.ones(num_actions))
        self.register_buffer("action_normalizer_frozen", torch.tensor(False, dtype=torch.bool))

    @property
    def replay_obs_groups(self) -> list[str]:
        groups = list(self.student_low_dim_obs_groups)
        if self.use_depth_encoder:
            groups.append(self.student_depth_obs_group)
        return groups

    @torch.no_grad()
    def observe_teacher_actions(self, actions: torch.Tensor):
        if bool(self.action_normalizer_frozen.item()):
            return
        flat = actions.detach().reshape(-1, actions.shape[-1]).to(dtype=torch.float32)
        if flat.numel() == 0:
            return
        batch_count = int(flat.shape[0])
        batch_mean = flat.mean(dim=0)
        batch_m2 = ((flat - batch_mean) ** 2).sum(dim=0)
        count = int(self.action_stats_count.item())
        if count == 0:
            self.action_stats_mean.copy_(batch_mean)
            self.action_stats_m2.copy_(batch_m2)
            self.action_stats_count.fill_(batch_count)
            return
        total = count + batch_count
        delta = batch_mean - self.action_stats_mean
        self.action_stats_mean.add_(delta * (batch_count / total))
        self.action_stats_m2.add_(batch_m2 + delta.pow(2) * (count * batch_count / total))
        self.action_stats_count.fill_(total)

    @torch.no_grad()
    def freeze_action_normalizer(self):
        if bool(self.action_normalizer_frozen.item()) or int(self.action_stats_count.item()) < 2:
            return
        variance = self.action_stats_m2 / max(int(self.action_stats_count.item()) - 1, 1)
        self.action_stats_std.copy_(variance.sqrt().clamp_min(self.action_std_floor))
        self.action_normalizer_frozen.fill_(True)

    def set_training_iteration(self, iteration: int):
        super().set_training_iteration(iteration)
        if iteration >= self.action_normalizer_warmup_iterations:
            self.freeze_action_normalizer()

    def should_skip_distillation_update(self) -> bool:
        return not bool(self.action_normalizer_frozen.item())

    def normalize_actions(self, actions: torch.Tensor, clip: bool = True) -> torch.Tensor:
        normalized = (actions - self.action_stats_mean) / self.action_stats_std.clamp_min(self.action_std_floor)
        if clip:
            normalized = normalized.clamp(-self.action_normalizer_clip, self.action_normalizer_clip)
        return normalized

    def unnormalize_actions(self, normalized_actions: torch.Tensor, clip: bool = True) -> torch.Tensor:
        if clip:
            normalized_actions = normalized_actions.clamp(-self.action_normalizer_clip, self.action_normalizer_clip)
        return normalized_actions * self.action_stats_std + self.action_stats_mean

    def _prepare_student_inputs(self, obs) -> tuple[torch.Tensor, torch.Tensor | None]:
        low_dim_obs = torch.cat([obs[group] for group in self.student_low_dim_obs_groups], dim=-1)
        low_dim_obs = self.student_obs_normalizer(low_dim_obs)
        if self.use_depth_encoder:
            depth_image = self._extract_depth(obs)
            if (
                self.training
                and self.enable_depth_augmentation
                and self.training_iteration >= self.augmentation_start_iteration
            ):
                depth_image = self._augment_normalized_depth(depth_image)
            return self._update_obs_history(low_dim_obs, depth_image)
        return self._update_low_dim_history(low_dim_obs), None

    def _update_low_dim_history(self, low_dim_obs: torch.Tensor) -> torch.Tensor:
        num_envs = int(low_dim_obs.shape[0])
        if self._low_dim_history is None or self._low_dim_history.shape[0] != num_envs:
            self._low_dim_history = low_dim_obs.unsqueeze(1).repeat(1, self.n_obs_steps, 1)
            self._depth_history = None
            self._pending_history_reset = torch.zeros(num_envs, dtype=torch.bool, device=low_dim_obs.device)
            return self._low_dim_history
        history = self._low_dim_history.to(device=low_dim_obs.device, dtype=low_dim_obs.dtype)
        if self._pending_history_reset is not None:
            self._pending_history_reset = self._pending_history_reset.to(device=low_dim_obs.device)
            reset_ids = self._pending_history_reset.nonzero(as_tuple=False).reshape(-1)
            if reset_ids.numel() > 0:
                history = history.clone()
                history[reset_ids] = low_dim_obs[reset_ids].unsqueeze(1).repeat(1, self.n_obs_steps, 1)
                self._pending_history_reset[reset_ids] = False
        self._low_dim_history = torch.cat((history[:, 1:], low_dim_obs.unsqueeze(1)), dim=1)
        return self._low_dim_history

    def _student_normalized_actions(self, obs) -> torch.Tensor:
        low_dim_history, depth_history = self._prepare_student_inputs(obs)
        if self.use_depth_encoder:
            return self.student(low_dim_history, depth_history)
        return self.student(low_dim_history.reshape(low_dim_history.shape[0], -1))

    def act(self, obs):
        normalized_mean = self._student_normalized_actions(obs)
        mean = self.unnormalize_actions(normalized_mean)
        std = self.std.expand_as(mean) if self.noise_std_type == "scalar" else torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.unnormalize_actions(self._student_normalized_actions(obs))

    def compute_behavior_loss(self, obs, privileged_actions: torch.Tensor, loss_fn) -> torch.Tensor:
        predicted = self._student_normalized_actions(obs)
        targets = self.normalize_actions(privileged_actions)
        return loss_fn(predicted, targets)

    def diagnostics(self) -> dict[str, float]:
        diagnostics = super().diagnostics() if self.use_depth_encoder else {
            "depth_valid_ratio": 0.0,
            "valid_depth_mean": 0.0,
            "valid_depth_std": 0.0,
            "depth_feature_mean": 0.0,
            "depth_feature_std": 0.0,
            "depth_encoder_grad_norm": 0.0,
        }
        diagnostics.update(
            {
                "action_normalizer_count": float(self.action_stats_count.item()),
                "action_normalizer_std_mean": float(self.action_stats_std.mean().item()),
                "action_normalizer_frozen": float(self.action_normalizer_frozen.item()),
            }
        )
        return diagnostics
