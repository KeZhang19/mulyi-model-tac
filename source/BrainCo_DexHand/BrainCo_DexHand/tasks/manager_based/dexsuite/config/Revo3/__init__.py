# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dexsuite Revo3 environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dexsuite_revo3_env_cfg_insert_d_peg:DexsuiteRevo3InsertDPegEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.dexsuite_revo3_env_cfg_insert_d_peg:DexsuiteRevo3InsertDPegEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg_insert_d_peg:DexsuiteRevo3InsertDPegPPORunnerCfg"
        ),
    },
)


gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.dexsuite_revo3_env_cfg_grasp:DexsuiteRevo3LiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteRevo3PPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DexsuiteRevo3DistillationRunnerCfg",
    },
)


gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dexsuite_revo3_env_cfg_rotate_bulb:DexsuiteRevo3RotateBulbEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.dexsuite_revo3_env_cfg_rotate_bulb:DexsuiteRevo3RotateBulbEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_rotate_bulb.yaml",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg_rotate_bulb:DexsuiteRevo3RotateBulbPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg_rotate_bulb:"
            "DexsuiteRevo3RotateBulbDistillationRunnerCfg"
        ),
    },
)
