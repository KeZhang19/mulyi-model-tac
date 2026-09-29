"""Open-hand approach, grasp, three-turn unscrew and randomized transport."""
import gymnasium as gym

gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Randomized-v0",
    entry_point=f"{__name__}.env_cfg:make_randomized_bulb_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RotateBulbRandomizedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:RotateBulbRandomizedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:RotateBulbRandomizedPPORunnerCfg",
    },
)
