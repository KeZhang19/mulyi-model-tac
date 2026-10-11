# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .hybrid_distillation_ppo import HybridDistillationPPO
from .ppo import PPO
from .ppo_dual_head import PPODualHead

__all__ = ["PPO", "PPODualHead", "Distillation", "HybridDistillationPPO"]
