"""Free horizontal tabletop curriculum with an explicit state-only contract."""
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from ..RotateBulbCustom.assembly_env_cfg import DexsuiteRevo3BulbAssemblyEnvCfg, DexsuiteRevo3BulbAssemblyRewardCfg
from ..RotateBulbCustom.agents.rsl_rl_ppo_cfg import DexsuiteRevo3RotateBulbPPORunnerCfg
from ..RotateBulbCustom.assembly_reward_adapter import configure_assembly_rewards
from .core import CurriculumSettings, EPISODE_SECONDS
from .insertion import CurriculumInsertionReward
from . import runtime
from .rewards import CurriculumRewardAdapter, FloorReachReward, arm_path_cost, base_excursion_cost


@configclass
class AssemblyCurriculumEnvCfg(DexsuiteRevo3BulbAssemblyEnvCfg):
    bulb_curriculum: CurriculumSettings = CurriculumSettings()

    def __post_init__(self):
        super().__post_init__()
        # Avoid inherited large-scene pair reservations. Explicit Hydra values
        # still win, and contact/stack capacity is checked by the native probe.
        for name in ("gpu_found_lost_pairs_capacity", "gpu_found_lost_aggregate_pairs_capacity",
                     "gpu_total_aggregate_pairs_capacity"):
            setattr(self.sim.physx, name, 2**22)


@configclass
class AssemblyCurriculumRunnerCfg(DexsuiteRevo3RotateBulbPPORunnerCfg):
    experiment_name = "dexsuite_revo3_bulb_assembly_curriculum"
    save_interval = 100


def configure_curriculum(cfg):
    """Resolve after Hydra overrides, including startup/reset parameters and rewards."""
    s = cfg.bulb_curriculum
    s.validate()
    cfg.sim.dt = 1. / s.physics_hz
    cfg.decimation = s.physics_hz // 30
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.lazy_sensor_update = True
    cfg.observations.perception.object_point_cloud.params["visualize"] = False
    if not cfg.tactile_policy_enabled:
        for name in list(vars(cfg.scene)):
            if any(tag in name for tag in ("_tacmap_", "_hydroshear_", "_warpsdf_")):
                setattr(cfg.scene, name, None)
        cfg.observations.proprio.rl_ours_pressure = None
        cfg.observations.proprio.contact = None
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
    # Original terminations, bounded actions and physical constraints are retained.
    configure_assembly_rewards(cfg)
    if cfg.OLD_REWARD:
        raise ValueError("Curriculum v2 requires the physical assembly reward ledger")
    cfg.rewards.contact_stability.func = CurriculumRewardAdapter
    # Continuous proximity plus signed approach/retreat replaces first-record
    # only reach credit. Grasp/lift/hover/thread bonuses keep their predicates.
    cfg.rewards.reach_progress = RewardTermCfg(func=FloorReachReward, weight=1.)
    cfg.rewards.arm_path = RewardTermCfg(func=arm_path_cost, weight=-.08)
    cfg.rewards.base_excursion = RewardTermCfg(func=base_excursion_cost, weight=-.5)
