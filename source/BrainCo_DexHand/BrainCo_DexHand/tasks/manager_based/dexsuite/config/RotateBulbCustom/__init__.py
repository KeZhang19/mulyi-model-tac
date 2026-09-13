# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the independently configurable Rotate-Bulb-Custom task."""

import gymnasium as gym

from . import agents


gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DexsuiteRevo3RotateBulbEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:DexsuiteRevo3RotateBulbEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteRevo3RotateBulbPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:DexsuiteRevo3RotateBulbDistillationRunnerCfg"
        ),
    },
)
