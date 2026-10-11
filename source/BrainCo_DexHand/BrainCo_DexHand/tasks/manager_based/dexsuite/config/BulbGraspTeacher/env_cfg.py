"""V2-test bulb teacher extended with a three-stage grasp/insert/screw curriculum.

The typed snapshot preserves the running teacher's complete manager configuration,
including term order, history order, noise, rewards, resets and physical settings.
Runtime MDP functions live beside this module, isolated from assembly tasks.
"""
from dataclasses import MISSING
from importlib import import_module
import inspect
import json
from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .curriculum import (
    BulbGraspCurriculumCommand, BulbGraspCurriculumDifficulty,
    curriculum_position_tracking_reward,
    bulb_down_orientation_penalty, bulb_thread_down_reward,
    reset_socket_from_target_ranges, update_grasp_curriculum,
    grasp_screw_time_out, stage_two_position_orientation_reward, stage_two_success_bonus,
)
from .insertion import match_socket_scale_to_bulb
from .screwing import carry_velocity_before_screwing, final_screw_success_bonus, screw_in_reward

ROOT = Path(__file__).resolve().parents[8]
PACKAGE = Path(__file__).resolve().parent


@configclass
class ConfigNode:
    """Config container for the teacher's custom groups and command ranges."""
    pass


def _resolve(name):
    module, symbol = name.split(":", 1)
    value = import_module(module)
    for part in symbol.split("."):
        value = getattr(value, part)
    return value


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$missing" in value:
        return MISSING
    if "$tuple" in value:
        return tuple(_decode(item) for item in value["$tuple"])
    if "$slice" in value:
        return slice(*value["$slice"])
    if "$callable" in value:
        return _resolve(value["$callable"])
    if "$type" in value:
        cls = ConfigNode if value["$type"] == "ConfigNode" else _resolve(value["$type"])
        fields = {key: _decode(item) for key, item in value["fields"].items()}
        accepted = inspect.signature(cls).parameters
        obj = cls(**{key: item for key, item in fields.items() if key in accepted})
        for key, item in fields.items():
            setattr(obj, key, item)
        return obj
    return {key: _decode(item) for key, item in value.items()}


@configclass
class BulbGraspTeacherEnvCfg(ManagerBasedRLEnvCfg):
    """28 actions; policy/proprio/perception = 195/745/960; 30 Hz control."""

    def __post_init__(self):
        snapshot = json.loads((PACKAGE / "teacher_snapshot.json").read_text())
        frozen = _decode(snapshot)
        for name, value in vars(frozen).items():
            setattr(self, name, value)
        robot = ROOT / "assets/rotate_bulb_custom/robot/usd/rizon4_training.usda"
        bulb = ROOT / "assets/bulb_grasp_teacher/bulb_insert.usd"
        points = ROOT / "assets/bulb_grasp_teacher/point_clouds_64"
        for path in (robot, bulb, points / "bulb.npy", points / "manifest.json"):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing teacher asset: {path}; run BulbGraspTeacher/build_asset.py "
                    "then build_insertion_asset.py"
                )
        self.scene.robot.spawn.usd_path = str(robot)
        self.scene.object.spawn.assets_cfg[0].usd_path = str(bulb)
        # Clear even the largest matching socket for every initial bulb attitude.
        initial_pos = self.scene.object.init_state.pos
        self.scene.object.init_state.pos = (*initial_pos[:2], initial_pos[2] + .01)
        self.observations.perception.object_point_cloud.params["offline_point_cloud_dir"] = str(points)
        # Keep the frozen source snapshot intact; all curriculum differences live here.
        socket = ROOT / "assets/bulb_grasp_teacher/socket.usd"
        if not socket.is_file():
            raise FileNotFoundError(f"Missing socket: {socket}; run BulbGraspTeacher/build_socket.py")
        table_top = self.scene.table.init_state.pos[2] + self.scene.table.spawn.size[2] / 2
        self.scene.socket = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Socket",
            spawn=sim_utils.UsdFileCfg(usd_path=str(socket)),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.30, 0.32, table_top + 0.044732484966516495),
                rot=(0.7071067811865476, 0.0, -0.7071067811865476, 0.0),
            ),
        )
        command = self.commands.object_pose
        command.class_type = BulbGraspCurriculumCommand
        command.socket_name = "socket"
        command.socket_mouth_x_m = -3.2485189876751974e-05
        command.socket_bottom_x_m = -0.044732484966516495
        command.stage1_height_m = 0.15
        command.position_tolerance_m = 0.05
        # Small point markers, independent of the task's success tolerances.
        for marker in ("position_far", "position_near"):
            command.goal_pose_visualizer_cfg.markers[marker].radius = .005
            command.curr_pose_visualizer_cfg.markers[marker].radius = .004
        geometry = json.loads((bulb.parent / "insertion_geometry.json").read_text())
        command.bulb_tip_offset_m = tuple(geometry["bulb_tip_offset_m"])
        command.source_thread_tip_x_m = geometry["source_thread_tip_x_m"]
        command.reference_extension_m = geometry["reference_extension_m"]
        command.reference_spin_rad = geometry["reference_spin_rad"]
        command.thread_pitch_m = geometry["pitch_m"]
        command.insertion_depth_m = .0055
        command.insertion_depth_range_m = (0.0, .010)
        command.shallow_insertion_radial_tolerance_m = .012
        # Final assembly alignment stays independent of the coarse entry window.
        command.insertion_radial_tolerance_m = .0005
        command.shallow_insertion_tilt_tolerance_deg = 30.0  # Strict <; no speed/hold gate.
        # These stability tolerances now apply only to final stage-three success.
        command.insertion_tilt_tolerance_deg = 5.0
        command.insertion_speed_limit_m_s = .02
        command.insertion_angular_speed_limit_rad_s = .5
        command.insertion_hold_s = .2
        command.screw_target_turns = 3.0
        command.screw_final_depth_tolerance_m = .001
        command.screw_engagement_radial_m = .002
        command.screw_engagement_tilt_deg = 10.0
        command.screw_helix_tolerance_m = .0015
        command.screw_max_measured_turn_rate = 3.0
        command.screw_reward_max_turn_rate = 1.0
        command.screw_goal_lead_turns = .125
        command.screw_force_scale_n = .5
        command.screw_regrasp_grace_s = .2
        command.screw_milestone_turns = .25
        command.screw_regression_deadband_turns = .01
        command.stage12_duration_s = 8.0
        command.stage3_duration_s = 6.0
        # Runs after the inherited bulb scale event, before PhysX cooks meshes.
        self.events.match_socket_scale = EventTermCfg(func=match_socket_scale_to_bulb, mode="prestartup")
        # Append after the teacher's table/robot/object reset events.
        self.events.reset_socket = EventTermCfg(func=reset_socket_from_target_ranges, mode="reset")
        self.rewards.palm_down_orientation = None
        self.rewards.bulb_down_orientation = RewardTermCfg(func=bulb_down_orientation_penalty, weight=-0.2)
        self.rewards.thread_down = RewardTermCfg(func=bulb_thread_down_reward, weight=5.0)
        self.rewards.position_tracking.func = curriculum_position_tracking_reward
        self.rewards.success.func = stage_two_position_orientation_reward
        self.rewards.success.params["orientation_floor"] = 0.2
        for name, component in (("carry_relative_lin_vel_l2", "linear"), ("carry_relative_ang_vel_l2", "angular")):
            term = getattr(self.rewards, name)
            term.func = carry_velocity_before_screwing
            term.params["component"] = component
        # Isaac integrates reward rates over dt. New physical turns pay 9/turn,
        # once-only quarter-turn milestones pay 3/turn: maximum 36 for 3 turns.
        self.rewards.screw_in_force = RewardTermCfg(func=screw_in_reward, weight=9., params={"component": "force"})
        self.rewards.screw_in_progress = RewardTermCfg(func=screw_in_reward, weight=3., params={"component": "progress"})
        self.rewards.screw_backtrack = RewardTermCfg(func=screw_in_reward, weight=-14., params={"component": "regression"})
        self.rewards.final_screw_success = RewardTermCfg(func=final_screw_success_bonus, weight=20.)
        self.rewards.stage2_success_bonus = RewardTermCfg(func=stage_two_success_bonus, weight=10.)
        # Promotion on the final stage-2 step must happen before timeout checks.
        original_terminations = self.terminations
        self.terminations = ConfigNode()
        self.terminations.curriculum_update = TerminationTermCfg(func=update_grasp_curriculum)
        for name, term in vars(original_terminations).items():
            setattr(self.terminations, name, term)
        self.terminations.time_out.func = grasp_screw_time_out
        self.curriculum.adr.func = BulbGraspCurriculumDifficulty
        self.episode_length_s = command.stage12_duration_s + command.stage3_duration_s
        self.bulb_grasp_curriculum_version = 10


@configclass
class BulbGraspTeacherPlayEnvCfg(BulbGraspTeacherEnvCfg):
    """Small scene with exactly the same MDP, history and action semantics."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
