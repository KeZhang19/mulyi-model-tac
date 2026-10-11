# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_dual_head import ActorCriticDualHead
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_transformer import ActorCriticTransformer
from .actor_critic_transformerxl import ActorCriticTransformerXL
from .actor_critic_transformer_cache import ActorCriticTransformerCache
from .rnd import *
from .student_teacher import StudentTeacher
from .student_teacher_depth_dagger import (
    StudentTeacherDepthDagger,
    StudentTeacherDepthDaggerV2,
    StudentTeacherDepthDaggerV4,
)
from .student_teacher_depth_diffusion_policy import StudentTeacherDepthDiffusionPolicy
from .student_teacher_depth_diffusion_policy_v3 import StudentTeacherDepthDiffusionPolicyV3
from .student_teacher_depth_dagger_v5 import (
    DextrahStereoDepthTransformerStudent,
    StereoDepthTransformerEncoder,
    StudentTeacherDepthDaggerV5,
)
from .student_teacher_depth_diffusion_v6 import StudentTeacherDepthDiffusionV6
from .student_teacher_diffusion_policy import StudentTeacherDiffusionPolicy
from .student_teacher_recurrent import StudentTeacherRecurrent
from .student_teacher_transformer import StudentTeacherTransformer
from .student_teacher_v2_distillation import (
    StudentTeacherV2Distillation,
    StudentTeacherV2TestDualCameraDistillation,
    StudentTeacherV2TestDistillation,
    StudentTeacherV2TestDistillationV4,
)
from .symmetry import *

__all__ = [
    "ActorCritic",
    "ActorCriticDualHead",
    "ActorCriticRecurrent",
    "ActorCriticTransformer",
    "ActorCriticTransformerXL",
    "StudentTeacher",
    "StudentTeacherDepthDagger",
    "StudentTeacherDepthDaggerV2",
    "StudentTeacherDepthDaggerV4",
    "StudentTeacherDepthDaggerV5",
    "DextrahStereoDepthTransformerStudent",
    "StereoDepthTransformerEncoder",
    "StudentTeacherDepthDiffusionPolicy",
    "StudentTeacherDepthDiffusionPolicyV3",
    "StudentTeacherDepthDiffusionV6",
    "StudentTeacherDiffusionPolicy",
    "StudentTeacherRecurrent",
    "StudentTeacherTransformer",
    "StudentTeacherV2Distillation",
    "StudentTeacherV2TestDualCameraDistillation",
    "StudentTeacherV2TestDistillation",
    "StudentTeacherV2TestDistillationV4",
    "ActorCriticTransformerCache",
]
