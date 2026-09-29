"""Independent full tabletop bulb assembly configuration."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import assembly
from .assembly_guidance import AssemblyGuidanceCfg, AssemblyGuidanceReward, assembly_thread_downward
from .assembly_lift import AssemblyLiftCfg, AssemblyLiftReward
from .env_cfg import (
    DexsuiteRevo3RotateBulbEnvCfg,
    DexsuiteRevo3RotateBulbEnvCfg_PLAY,
    DexsuiteRevo3RotateBulbSceneCfg,
    reset_robot_to_grasp,
    rotate_bulb_action_l2_balanced,
    rotate_bulb_action_rate_l2_balanced,
    rotate_bulb_finger_forces,
    rotate_bulb_hand_shell_distance,
    rotate_bulb_non_finite,
)


def assembly_target_pose_b(env):
    """Return the fixed installation pose in the robot base frame.

    The legacy ``object_pose`` command is intentionally not used by this task:
    it carries a release/success metric and its reset path assumes the old
    guided screw state.  The target is derived from the measured socket frame
    and each episode's reset anchor instead.
    """
    support = env.scene["support"]
    robot = env.scene["robot"]
    target_x = assembly.ENTRY_ROOT_X_M - assembly.INSTALLATION_DEPTH_M
    offset = torch.zeros((env.num_envs, 3), device=env.device, dtype=support.data.root_pos_w.dtype)
    offset[:, 0] = target_x
    rel_quat = torch.zeros((env.num_envs, 4), device=env.device, dtype=support.data.root_quat_w.dtype)
    rel_quat[:, 0] = 1.0
    target_pos, target_quat = assembly.combine_frame_transforms(
        support.data.root_pos_w, support.data.root_quat_w, offset, rel_quat
    )
    target_pos_b, target_quat_b = assembly.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos, target_quat
    )
    return torch.cat((target_pos_b, target_quat_b), dim=-1)


class AssemblyCompletionReward(ManagerTermBase):
    """Pay once after the termination-side one-second hold is satisfied.

    ``assembly_success`` owns the hold timer.  Calling it from both the
    termination manager and the reward manager would advance that timer twice
    per control step, so this term only consumes the state flag set by the
    termination check.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False

    def __call__(self, env):
        state = getattr(env, "_bulb_assembly_state", None)
        if state is None:
            return torch.zeros(env.num_envs, device=env.device)
        reached = state.success
        credit = reached & ~self.paid
        self.paid |= reached
        return credit.float() / env.step_dt


class AssemblyReachProgress(ManagerTermBase):
    """Dense physical approach progress before the first finger contact."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_distance = torch.zeros(env.num_envs, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.previous_distance[ids] = 0.0
        self.initialized[ids] = False

    def __call__(self, env, distance_scale: float = 0.05):
        if not torch.isfinite(torch.as_tensor(distance_scale)) or distance_scale <= 0.0:
            raise ValueError("distance_scale must be positive and finite")
        distance = torch.nan_to_num(
            rotate_bulb_hand_shell_distance(env), nan=1.0, posinf=1.0, neginf=1.0
        )
        delta = (self.previous_distance - distance) / distance_scale
        value = torch.where(self.initialized, delta.clamp(-1.0, 1.0), torch.zeros_like(delta))
        self.previous_distance = distance
        self.initialized[:] = True
        state = getattr(env, "_bulb_assembly_state", None)
        if state is not None:
            value = torch.where(state.grasped, torch.zeros_like(value), value)
        return value


def assembly_hand_shell_distance(env) -> torch.Tensor:
    """Return the farthest fingertip distance to the bulb's grasp shell.

    This is the independent-task adaptation of dextac's ``object_ee_distance``:
    it keeps the bulb's measured local +X shell target rather than the bulb
    root, and uses the five Revo3 fingertip bodies.  The farthest fingertip is
    used so the policy cannot earn full approach credit while one finger stays
    far away.
    """
    robot = env.scene["robot"]
    ids = getattr(env, "_assembly_reach_tip_body_ids", None)
    if ids is None:
        names = (
            "right_thumb_tip_Link", "right_index_tip_Link", "right_middle_tip_Link",
            "right_ring_tip_Link", "right_little_tip_Link",
        )
        ids = [robot.body_names.index(name) for name in names]
        env._assembly_reach_tip_body_ids = ids
    bulb = env.scene["object"]
    offset = torch.zeros_like(bulb.data.root_pos_w)
    offset[:, 0] = 0.04
    shell, _ = assembly.combine_frame_transforms(
        bulb.data.root_pos_w, bulb.data.root_quat_w, offset,
    )
    tips = robot.data.body_pos_w[:, ids]
    return torch.linalg.vector_norm(tips - shell[:, None, :], dim=-1).amax(dim=-1)


def assembly_hand_shell_potential(env, std: float = 0.2) -> torch.Tensor:
    """Dense tanh proximity potential adapted from the dextac grasp task."""
    if not torch.isfinite(torch.as_tensor(std)) or std <= 0.0:
        raise ValueError("std must be positive and finite")
    distance = torch.nan_to_num(
        assembly_hand_shell_distance(env), nan=1.0, posinf=1.0, neginf=1.0,
    )
    potential = 1.0 - torch.tanh(distance / std)
    state = getattr(env, "_bulb_assembly_state", None)
    if state is not None:
        potential = torch.where(state.installed, torch.zeros_like(potential), potential)
    return potential


class AssemblyHandShellBestProgress(ManagerTermBase):
    """Pay only for new best hand-to-shell proximity, per environment."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.best = assembly_hand_shell_potential(env, **cfg.params).clone()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.best[ids] = assembly_hand_shell_potential(self._env, **self.cfg.params)[ids]

    def __call__(self, env, std: float = 0.2):
        potential = torch.nan_to_num(
            assembly_hand_shell_potential(env, std=std), nan=0.0, posinf=0.0, neginf=0.0,
        )
        delta = (potential - self.best).clamp_min(0.0)
        self.best = torch.maximum(self.best, potential)
        return delta


def assembly_any_finger_contact(env, threshold: float = 1.0) -> torch.Tensor:
    """Dextac grasp reward: any distal finger contacting the bulb above 1 N.

    Use bulb-filtered DIP contact forces, not total forces from the table or
    socket. Like the source grasp task, this has no phase or failure gate.
    DIP body contacts do not distinguish the pad face from the body's sides.
    """
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be positive and finite")
    return (rotate_bulb_finger_forces(env) > threshold).any(dim=-1).float()


def assembly_good_finger_contact(env, threshold: float = 1.0) -> torch.Tensor:
    """Dextac grasp reward: thumb plus at least one other finger on the bulb."""
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be positive and finite")
    contacts = rotate_bulb_finger_forces(env) > threshold
    return (contacts[:, 0] & contacts[:, 1:].any(dim=-1)).float()


@configclass
class DexsuiteRevo3BulbAssemblySceneCfg(DexsuiteRevo3RotateBulbSceneCfg):
    """Legacy scene plus an explicit bulb-to-support contact sensor."""

    bulb_support_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Lamp/Bulb",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Lamp/Support"],
        track_air_time=False,
    )


@configclass
class DexsuiteRevo3BulbAssemblyRewardCfg:
    action_l2 = RewTerm(func=rotate_bulb_action_l2_balanced, weight=-0.02)
    action_rate_l2 = RewTerm(func=rotate_bulb_action_rate_l2_balanced, weight=-0.01)
    reach_progress = RewTerm(func=AssemblyReachProgress, weight=1.0, params={"distance_scale": 0.05})
    # Dextac-style absolute proximity plus historical-best progress.  The
    # shell target is the measured grippable bulb surface, not the screw root.
    hand_shell_potential = RewTerm(func=assembly_hand_shell_potential, weight=1.0, params={"std": 0.2})
    hand_shell_best_progress = RewTerm(
        func=AssemblyHandShellBestProgress, weight=200.0, params={"std": 0.2},
    )
    # Temporary transplant from Dextac FlexivReorientRewardCfg. These are
    # contact reward rates (RewardManager multiplies by step_dt), not bonuses
    # for entering a phase. No grasp/lift/alignment prerequisite is applied.
    any_finger_contact = RewTerm(
        func=assembly_any_finger_contact, weight=0.7, params={"threshold": 1.0},
    )
    good_finger_contact = RewTerm(
        func=assembly_good_finger_contact, weight=2.0, params={"threshold": 1.0},
    )
    # Finite episode bonuses, already divided by step_dt by the term.
    lift_best_progress = RewTerm(func=AssemblyLiftReward, weight=1.0, params={"component": "progress"})
    lift_stable_once = RewTerm(func=AssemblyLiftReward, weight=1.0, params={"component": "stable"})
    position_tracking = RewTerm(func=AssemblyGuidanceReward, weight=5.0, params={"component": "position"})
    insertion_tracking = RewTerm(func=AssemblyGuidanceReward, weight=5.0, params={"component": "insertion"})
    thread_downward = RewTerm(func=assembly_thread_downward, weight=1.0)
    phase = RewTerm(func=assembly.assembly_phase_reward, weight=0.5)
    success = RewTerm(func=AssemblyCompletionReward, weight=20.0)


@configclass
class DexsuiteRevo3BulbAssemblyEnvCfg(DexsuiteRevo3RotateBulbEnvCfg):
    """Complete tabletop initialization; no legacy pre-engaged reset."""

    scene: DexsuiteRevo3BulbAssemblySceneCfg = DexsuiteRevo3BulbAssemblySceneCfg(
        num_envs=32, env_spacing=3, replicate_physics=False,
    )
    bulb_assembly_task: bool = True
    # Reward-only ablation switch, resolved by the task factory after Hydra.
    OLD_REWARD: bool = False
    # Assembly starts with a free bulb on the table, independent of Rotate-Bulb's default.
    bulb_pure_collision: bool = True
    assembly_guidance: AssemblyGuidanceCfg = AssemblyGuidanceCfg()
    assembly_lift: AssemblyLiftCfg = AssemblyLiftCfg()
    # Training-only reset curriculum knob.  The acceptance/default path keeps
    # the measured tabletop separation; staged training may start nearer the
    # reference grasp and increase the offset before formal evaluation.
    bulb_training_reset_offset_y: float = assembly.DESKTOP_OFFSET_Y_M
    # Independent world XY offsets around the nominal desktop bulb position.
    # Shared by training and PLAY; yaw is about world Z, in radians.
    bulb_reset_x_range: tuple[float, float] = (-0.05, 0.05)
    bulb_reset_y_range: tuple[float, float] = (-0.05, 0.05)
    bulb_reset_yaw_range: tuple[float, float] = (-math.pi / 2, math.pi / 2)

    def __post_init__(self):
        # Keep the complete legacy reward object while the parent resolves the
        # Revo3 robot, scene, sensors, and reset arguments.  The parent reads
        # fields such as rewards.success.params["rot_std"] during setup.
        super().__post_init__()
        # Start tabletop assembly with an open hand. Named defaults also feed
        # startup and full/partial reset; keep the seven arm joints unchanged.
        self.scene.robot.init_state.joint_pos = {
            name: 0.0 if name.startswith("right_") else position
            for name, position in self.scene.robot.init_state.joint_pos.items()
        }
        # Replace the reward graph only after parent setup.  None of these
        # terms reads the legacy release, unscrew, or command state.
        self.rewards = DexsuiteRevo3BulbAssemblyRewardCfg()

        # Reuse resolved table/support arguments, replacing only the reset
        # callable.  No legacy articulation/attachment event is used.
        self.events.reset_object = EventTerm(
            func=assembly.reset_bulb_assembly_on_table,
            mode="reset",
            params={
                **self.events.reset_object.params,
                "desktop_offset_y": self.bulb_training_reset_offset_y,
                "x_range": self.bulb_reset_x_range,
                "y_range": self.bulb_reset_y_range,
                "yaw_range": self.bulb_reset_yaw_range,
            },
        )
        self.events.initialize_lamp = self.events.reset_object.replace(mode="startup")
        self.events.release_bulb = None
        self.events.reset_robot_joints = EventTerm(func=reset_robot_to_grasp, mode="reset")
        self.events.initialize_robot = self.events.reset_robot_joints.replace(mode="startup")
        self.events.reset_robot_wrist_joint = None

        self.terminations.bulb_dropped = DoneTerm(func=assembly.assembly_bulb_dropped)
        self.terminations.lost_grasp = DoneTerm(func=assembly.assembly_lost_grasp)
        self.terminations.non_finite = DoneTerm(func=rotate_bulb_non_finite)
        self.terminations.success = DoneTerm(func=assembly.assembly_success, params={"hold_duration": assembly.HOLD_DURATION_S})
        # Remove the legacy command manager state and its success metrics.  The
        # target pose observation below is computed from the support and reset
        # anchor directly, so no command term is needed by this task.
        self.commands.object_pose = None
        self.observations.policy.target_object_pose_b = ObsTerm(func=assembly_target_pose_b)
        self.observations.policy.bulb_phase = ObsTerm(func=assembly.assembly_phase_observation)
        self.observations.policy.screw_progress = ObsTerm(func=assembly.assembly_screw_progress)
        self.episode_length_s = self.task_episode_length_s


@configclass
class DexsuiteRevo3BulbAssemblyEnvCfg_PLAY(DexsuiteRevo3BulbAssemblyEnvCfg):
    """Evaluation/play variant with the same complete initialization."""

    scene: DexsuiteRevo3BulbAssemblySceneCfg = DexsuiteRevo3BulbAssemblySceneCfg(
        num_envs=32, env_spacing=3, replicate_physics=False,
    )
