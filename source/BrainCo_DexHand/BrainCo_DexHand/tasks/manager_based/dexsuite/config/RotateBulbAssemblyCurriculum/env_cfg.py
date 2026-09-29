"""Opt-in stages with exactly the original assembly action/observation layout."""
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from ..RotateBulbCustom.assembly_env_cfg import DexsuiteRevo3BulbAssemblyEnvCfg, DexsuiteRevo3BulbAssemblyRewardCfg
from ..RotateBulbCustom.agents.rsl_rl_ppo_cfg import DexsuiteRevo3RotateBulbPPORunnerCfg
from ..RotateBulbCustom.assembly_reward_adapter import configure_assembly_rewards
from .core import CurriculumSettings, EPISODE_SECONDS
from .insertion import CurriculumInsertionReward
from . import runtime


@configclass
class AssemblyCurriculumEnvCfg(DexsuiteRevo3BulbAssemblyEnvCfg):
    bulb_curriculum: CurriculumSettings = CurriculumSettings()


@configclass
class AssemblyCurriculumRunnerCfg(DexsuiteRevo3RotateBulbPPORunnerCfg):
    experiment_name = "dexsuite_revo3_bulb_assembly_curriculum"
    save_interval = 100


def configure_curriculum(cfg):
    """Resolve after Hydra overrides, including startup/reset parameters and rewards."""
    s = cfg.bulb_curriculum
    s.validate()
    if not cfg.bulb_pure_collision or not cfg.bulb_assembly_task:
        raise ValueError("Curriculum requires the free rigid-body assembly task")
    cfg.task_episode_length_s = cfg.episode_length_s = EPISODE_SECONDS[s.stage]
    cfg.rewards = DexsuiteRevo3BulbAssemblyRewardCfg()
    if s.stage < 4:
        for name, term in vars(cfg.rewards).items():
            if term is not None and hasattr(term, "weight") and name not in ("action_l2", "action_rate_l2", "success"):
                term.weight = 5. if s.stage == 3 and name == "insertion_tracking" else 0.
        cfg.rewards.curriculum_progress = RewardTermCfg(func=runtime.curriculum_reward, weight=1.)
        if s.stage == 3:
            cfg.rewards.insertion_tracking.func = CurriculumInsertionReward
    cfg.terminations.curriculum_stage_complete = TerminationTermCfg(func=runtime.curriculum_stage_complete)
    cfg.terminations.curriculum_reset_failed = TerminationTermCfg(func=runtime.curriculum_reset_failed)
    cfg.events.reset_object.func = runtime.reset_curriculum_scene
    cfg.events.initialize_lamp = cfg.events.reset_object.replace(mode="startup")
    cfg.events.reset_robot_joints = None
    cfg.events.initialize_robot = None
    # All other original terminations, observations, actions and physical constraints are retained.
    configure_assembly_rewards(cfg)
