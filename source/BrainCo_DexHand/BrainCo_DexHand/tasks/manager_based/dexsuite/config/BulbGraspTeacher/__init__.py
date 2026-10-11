"""Independent port of the V2-test stable bulb grasp teacher."""
import gymnasium as gym

for task, cfg in (
    ("BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0", "BulbGraspTeacherEnvCfg"),
    ("BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-Play-v0", "BulbGraspTeacherPlayEnvCfg"),
):
    gym.register(
        id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{cfg}",
                "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:BulbGraspTeacherPPORunnerCfg"},
    )
