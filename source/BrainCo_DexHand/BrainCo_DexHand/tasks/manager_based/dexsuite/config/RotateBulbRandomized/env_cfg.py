"""Independent open-hand randomized bulb task; original tasks keep their defaults."""
from dataclasses import asdict
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..RotateBulbCustom.env_cfg import DexsuiteRevo3RotateBulbEnvCfg
from ..RotateBulbCustom.environment import make_rotate_bulb_env
from ..RotateBulbCustom.agents.rsl_rl_ppo_cfg import DexsuiteRevo3RotateBulbPPORunnerCfg
from . import mdp
from .progress_penalty import ProgressPenaltySettings
from .progress_penalty_terms import RandomizedProgressPenalty
from .phase_rewards import ReleasedContactGuidance, prerelease_arm_motion


@configclass
class RotateBulbRandomizedEnvCfg(DexsuiteRevo3RotateBulbEnvCfg):
    bulb_randomized_task: bool = True
    task_episode_length_s: float = 12.0
    randomized_workspace_path: str = str(mdp.BANK_PATH)
    pressure_pose_mode: str = "pose_only"
    unscrew_progress_penalty: dict = asdict(ProgressPenaltySettings())
    contact_transport_reward_version: int = 3

    def __post_init__(self):
        super().__post_init__()
        # Temporary timing profile: two physics updates per action at 30 Hz control.
        self.sim.dt = 1 / 60
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.scene.robot.init_state.joint_pos = {
            name: 0. if name.startswith("right_") else value
            for name, value in self.scene.robot.init_state.joint_pos.items()
        }
        self.events.reset_object.func = mdp.reset_randomized_lamp
        self.events.reset_object.params.update(x_range=(-.1, .1), y_range=(-.1, .1))
        self.events.initialize_lamp = self.events.reset_object.replace(mode="startup")
        self.commands.object_pose.class_type = mdp.RandomizedBulbPoseCommand
        self.commands.object_pose.ranges.pos_x = (-.15, .15)
        self.commands.object_pose.ranges.pos_y = (-.15, .15)
        self.commands.object_pose.ranges.pos_z = mdp.GOAL_Z_RANGE
        self.rewards.action_l2 = RewTerm(func=mdp.randomized_action_l2, weight=-.05)
        self.rewards.arm_motion = RewTerm(func=prerelease_arm_motion, weight=-.3)
        self.rewards.hand_far = RewTerm(func=mdp.randomized_hand_far, weight=-.3)
        self.rewards.randomized_grasp = RewTerm(func=mdp.randomized_grasp_reward, weight=0.)
        # Pay contact guidance only after detachment. Opposing contact earns
        # 1/s, a single side at most .25/s, capped at 4 points per episode.
        # Actual contact still gates physical turn progress before release.
        self.rewards.any_finger_contact = RewTerm(
            func=ReleasedContactGuidance, weight=1.,
            params={"force_scale": .5, "reward_rate": 1., "max_reward": 4.})
        self.rewards.position_tracking.weight = 40.0
        self.rewards.palm_above_support = RewTerm(
            func=mdp.RandomizedPalmAboveSupport, weight=1., params=dict(mdp.PALM_ABOVE_PARAMS))
        self.rewards.unscrew_regression = RewTerm(
            func=RandomizedProgressPenalty, weight=-1., params={"component": "regression"})
        self.rewards.unscrew_stall = RewTerm(
            func=RandomizedProgressPenalty, weight=-1., params={"component": "stall"})
        self.observations.policy.randomized_phase = ObsTerm(func=mdp.randomized_phase_observation)
        self.events.zz_initialize_tactile_contract.func = mdp.initialize_randomized_contract
        # Preserve the latest guided profile without changing Custom/Assembly.
        self.sim.physx.min_position_iteration_count = 24
        self.sim.physx.max_position_iteration_count = 24
        self.sim.physx.min_velocity_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 4
        for name, sensor in vars(self.scene).items():
            if "warpsdf" in name and sensor is not None:
                sensor.update_period = 1 / 30


@configclass
class RotateBulbRandomizedEnvCfg_PLAY(RotateBulbRandomizedEnvCfg):
    """PLAY shares the exact reset/goal distribution and observation contract."""


@configclass
class RotateBulbRandomizedPPORunnerCfg(DexsuiteRevo3RotateBulbPPORunnerCfg):
    experiment_name = "dexsuite_revo3_rotate_bulb_randomized"
    save_interval = 100


def make_randomized_bulb_env(cfg, **kwargs):
    if cfg.bulb_pure_collision or tuple(cfg.initial_screw_turns_range) != (3., 3.):
        raise ValueError("Randomized workspace is validated only for three-turn guided rotation")
    return make_rotate_bulb_env(cfg, **kwargs)
