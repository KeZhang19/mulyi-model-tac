"""Independent Flexiv D-peg insertion task using the Rotate-Bulb-Custom scene."""

import gymnasium as gym

from . import agents


gym.register(
    id="BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DexsuiteFlexivInsertDPegEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:DexsuiteFlexivInsertDPegEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteFlexivInsertDPegPPORunnerCfg",
    },
)
