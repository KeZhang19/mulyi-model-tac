#!/usr/bin/env python3
# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-student module for distillation where:

- Teacher: simple MLP policy (same as `StudentTeacher`)
- Student: Transformer policy implemented by `ActorCriticTransformer`

This is meant to be used with `DistillationRunner` / `Distillation` algorithm.
The distillation algorithm uses:
  - `policy.act(obs)` for rollout actions (student)
  - `policy.evaluate(obs)` as privileged_actions (teacher actions)
  - `policy.act_inference(obs)` during update for behavior cloning loss
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.modules.actor_critic_transformer import ActorCriticTransformer


class StudentTeacherTransformer(nn.Module):
    """Teacher-student distillation policy with Transformer student.

    Notes:
    - We keep the teacher as an MLP so it can load from legacy PPO MLP checkpoints
      (via the same "actor.*" -> teacher mapping as in `StudentTeacher`).
    - The student is an `ActorCriticTransformer`, but we only use its actor path for distillation.
      (It still constructs a critic internally; we point student "critic" obs_groups to the same set as "policy".)
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        # distillation-style knobs (same naming as `StudentTeacher`)
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: list[int] = None,
        teacher_hidden_dims: list[int] = None,
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        # transformer defaults (match our PPO transformer configs)
        transformer_model_dim: int = 128,
        transformer_num_layers: int = 3,
        transformer_num_heads: int = 4,
        transformer_dropout: float = 0.0,
        max_seq_len: int = 128,
        obs_history_tokens: int = 5,
        # passthrough/compat
        **kwargs,
    ):
        if student_hidden_dims is None:
            student_hidden_dims = [256, 256, 128]
        if teacher_hidden_dims is None:
            teacher_hidden_dims = [256, 256, 256]

        if kwargs:
            print(
                "StudentTeacherTransformer.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        super().__init__()

        self.loaded_teacher = False  # indicates if teacher has been loaded

        # --- compute obs sizes for teacher (MLP) ---
        self.obs_groups = obs_groups

        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "StudentTeacherTransformer only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]

        # --- teacher (MLP) ---
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        print(f"Teacher MLP: {self.teacher}")

        # --- student (Transformer) ---
        # ActorCriticTransformer expects obs_groups with keys "policy" and "critic".
        student_obs_groups = {
            "policy": obs_groups["policy"],
            "critic": obs_groups.get("critic", obs_groups["policy"]),
        }

        # We only need actor behavior, but ActorCriticTransformer builds both actor+critic.
        self.student = ActorCriticTransformer(
            obs=obs,
            obs_groups=student_obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=student_obs_normalization,
            critic_obs_normalization=False,
            actor_hidden_dims=student_hidden_dims,
            critic_hidden_dims=student_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            transformer_model_dim=transformer_model_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_dropout=transformer_dropout,
            max_seq_len=max_seq_len,
            obs_history_tokens=obs_history_tokens,
        )

        # Keep a distribution handle for compatibility with logging (mean/std/entropy)
        # This is driven by the student's distribution after calling `act()`.
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # --------------------------------------------------------------------------
    # Compatibility interface expected by `DistillationRunner` / `Distillation`
    # --------------------------------------------------------------------------

    @property
    def action_mean(self):
        return self.distribution.mean if self.distribution is not None else None

    @property
    def action_std(self):
        return self.distribution.stddev if self.distribution is not None else None

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1) if self.distribution is not None else None

    def reset(self, dones=None, hidden_states=None):
        # teacher: stateless
        # student: transformer has no cache in this implementation
        self.student.reset(dones=dones, hidden_states=hidden_states)

    def forward(self):
        raise NotImplementedError

    def act(self, obs):
        # Rollout actions from student (with exploration noise)
        actions = self.student.act(obs)
        # mirror distribution for logging convenience
        self.distribution = self.student.distribution
        return actions

    def act_inference(self, obs):
        # Deterministic student action (mean)
        return self.student.act_inference(obs)

    def evaluate(self, obs):
        # Privileged teacher action (no grad)
        obs_t = self.get_teacher_obs(obs)
        obs_t = self.teacher_obs_normalizer(obs_t)
        with torch.no_grad():
            return self.teacher(obs_t)

    def get_teacher_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["teacher"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        # ensure teacher stays in eval mode
        self.teacher.eval()
        if hasattr(self.teacher_obs_normalizer, "eval"):
            self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs):
        # Update only student's normalizer (same behavior as legacy StudentTeacher)
        self.student.update_normalization(obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load parameters for student + teacher.

        Supports:
        - PPO-trained teacher checkpoint: keys contain "actor." and "actor_obs_normalizer."
          We load those into teacher (MLP) and teacher_obs_normalizer.
        - Distillation checkpoint: keys contain "student." and "teacher."
          We load everything as-is.
        """

        # Case 1: loading teacher from RL training checkpoint (PPO)
        if any("actor" in key for key in state_dict.keys()):
            # rename keys to match teacher (MLP) and remove unrelated keys
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            teacher_keys = set(self.teacher.state_dict().keys())
            teacher_norm_keys = (
                set(self.teacher_obs_normalizer.state_dict().keys())
                if hasattr(self.teacher_obs_normalizer, "state_dict")
                else set()
            )

            for key, value in state_dict.items():
                if key.startswith("actor."):
                    mapped = key.replace("actor.", "")
                    if mapped in teacher_keys:
                        teacher_state_dict[mapped] = value
                elif key.startswith("actor_obs_normalizer."):
                    mapped = key.replace("actor_obs_normalizer.", "")
                    if mapped in teacher_norm_keys:
                        teacher_obs_normalizer_state_dict[mapped] = value

            missing = self.teacher.load_state_dict(teacher_state_dict, strict=False)
            if hasattr(self.teacher_obs_normalizer, "load_state_dict") and teacher_obs_normalizer_state_dict:
                self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=False)

            self.loaded_teacher = True
            self.teacher.eval()
            if hasattr(self.teacher_obs_normalizer, "eval"):
                self.teacher_obs_normalizer.eval()
            return False  # training does not resume

        # Case 2: loading from distillation checkpoint (resume)
        if any("student" in key for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            if hasattr(self.teacher_obs_normalizer, "eval"):
                self.teacher_obs_normalizer.eval()
            return True  # training resumes

        raise ValueError("state_dict does not contain actor (teacher) or student parameters")


