"""Opt-in assembly curriculum; the original complete task remains registered."""
import gymnasium as gym

gym.register(
    id="BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0",
    entry_point=f"{__name__}.environment:make_curriculum_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:AssemblyCurriculumEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:AssemblyCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:AssemblyCurriculumRunnerCfg",
    },
)
