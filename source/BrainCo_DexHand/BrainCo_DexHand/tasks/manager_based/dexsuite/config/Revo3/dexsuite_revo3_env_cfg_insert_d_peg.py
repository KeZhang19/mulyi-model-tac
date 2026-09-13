"""Free D-shaped peg insertion, starting from a physically calibrated grasp."""

from dataclasses import MISSING, asdict
import json
import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import CommandTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, subtract_frame_transforms

from ... import dexsuite_env_cfg_grasp_tianji as dexsuite
from ... import mdp
from ...mdp.d_peg_actions import DPegPregraspResidualActionCfg
from ...mdp.d_peg_insertion import (
    DPegInsertionState,
    d_peg_dropped,
    d_peg_insertion_phase,
    d_peg_insertion_progress,
    d_peg_nonfinite,
    d_peg_out_of_bounds,
    d_peg_success,
    validate_d_peg_geometry_metadata,
)
from ...mdp.d_peg_tactile import DPegPretrainedTactile, initialize_d_peg_tactile_contract
from .dexsuite_revo3_env_cfg_grasp import Revo3MixinCfg


_REPO_ROOT = Path(__file__).resolve().parents[8]
_ASSETS = _REPO_ROOT / "assets/d_peg_insertion"
_COMMON_SCENE = dexsuite.SceneCfg(num_envs=1, env_spacing=3)
_JOINT_NAMES = (
    *(f"Joint{i}_R" for i in range(1, 8)),
    "right_thumbcmp_roll_joint", "right_thumbcmr_roll_joint", "right_thumbmcp_roll_joint",
    "right_thumbpip_roll_joint", "right_thumbdip_roll_joint",
    *(f"right_{finger}{joint}_joint" for finger in ("index", "mid", "ring", "pinky")
      for joint in ("mcp_yaw", "mcp_roll", "pip_roll", "dip_roll")),
)


def load_d_peg_pregrasp(path):
    """Require the explicit, finite 28-joint calibration; never substitute an open hand."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported D-peg pregrasp schema: {path}")
    for field in ("robot_joint_positions", "robot_joint_targets"):
        if field not in data and field == "robot_joint_targets":
            data[field] = dict(data["robot_joint_positions"])
        values = data.get(field, {})
        if set(values) != set(_JOINT_NAMES) or not all(math.isfinite(float(v)) for v in values.values()):
            raise ValueError(f"{field} must contain all 28 finite robot joints: {path}")
        data[field] = {name: float(value) for name, value in values.items()}
    pose = data.get("peg_pose", {})
    position, quaternion = pose.get("position", []), pose.get("quaternion", [])
    if (len(position) != 3 or len(quaternion) != 4
            or not all(math.isfinite(float(v)) for v in [*position, *quaternion])):
        raise ValueError(f"Pregrasp requires a finite environment-local peg pose: {path}")
    if not math.isclose(sum(float(v) ** 2 for v in quaternion), 1.0, abs_tol=1e-5):
        raise ValueError(f"Pregrasp quaternion must be unit length (wxyz): {path}")
    return data


def reset_d_peg_pregrasp(env, env_ids=None):
    """Reset selected environments without simulation steps or rigid attachments."""
    ids = (torch.arange(env.num_envs, device=env.device, dtype=torch.long) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if ids.numel() == 0:
        return
    calibration = env.cfg._d_peg_pregrasp
    robot, peg, socket = env.scene["robot"], env.scene["object"], env.scene["socket"]
    joints = torch.tensor([calibration["robot_joint_positions"][name] for name in robot.joint_names],
                          device=env.device).expand(len(ids), -1).clone()
    targets = torch.tensor([calibration["robot_joint_targets"][name] for name in robot.joint_names],
                           device=env.device).expand(len(ids), -1).clone()
    zeros = torch.zeros_like(joints)
    robot.write_joint_state_to_sim(joints, zeros, env_ids=ids)
    robot.set_joint_position_target(targets, env_ids=ids)
    robot.set_joint_velocity_target(zeros, env_ids=ids)
    robot.set_joint_effort_target(zeros, env_ids=ids)
    pose = calibration["peg_pose"]
    peg_pose = torch.tensor([*pose["position"], *pose["quaternion"]], device=env.device)
    peg_pose = peg_pose.expand(len(ids), -1).clone()
    peg_pose[:, :3] += env.scene.env_origins[ids]
    peg.write_root_pose_to_sim(peg_pose, env_ids=ids)
    peg.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)

    socket_pose = socket.data.default_root_state[ids, :7].clone()
    socket_pose[:, :3] += env.scene.env_origins[ids]
    radius = float(env.cfg.socket_xy_randomization_m)
    socket_pose[:, :2] += torch.empty((len(ids), 2), device=env.device).uniform_(-radius, radius)
    angle = math.radians(float(env.cfg.socket_yaw_randomization_deg))
    yaw = torch.empty(len(ids), device=env.device).uniform_(-angle, angle)
    socket_pose[:, 3:] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    socket.write_root_pose_to_sim(socket_pose, env_ids=ids)
    socket.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)


class DPegInsertionPoseCommand(CommandTerm):
    """Final peg-tip pose derived from the stationary socket, in robot-root coordinates."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.socket = env.scene["socket"]
        self.pose_command_b = torch.zeros((env.num_envs, 7), device=env.device)
        self.pose_command_b[:, 3] = 1.0
        self.metrics["success"] = torch.zeros(env.num_envs, device=env.device)

    @property
    def command(self):
        return self.pose_command_b

    def _resample_command(self, env_ids):
        offset = torch.zeros((len(env_ids), 3), device=self.device)
        offset[:, 2] = self._env.cfg.socket_mouth_height_m - self._env.cfg.insertion_depth_m
        goal_w, _ = combine_frame_transforms(
            self.socket.data.root_pos_w[env_ids], self.socket.data.root_quat_w[env_ids], offset,
        )
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids], self.robot.data.root_quat_w[env_ids],
            goal_w, self.socket.data.root_quat_w[env_ids],
        )
        self.pose_command_b[env_ids] = torch.cat((pos_b, quat_b), dim=-1)

    def _update_metrics(self):
        self.metrics["success"] = d_peg_success(self._env).float()

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class DexsuiteRevo3InsertDPegSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_ASSETS / "peg.usd"), activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.10, 0.84)),
    )
    socket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSETS / "socket.usd"), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.10, 0.76)),
    )
    peg_socket_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object", filter_prim_paths_expr=["{ENV_REGEX_NS}/Socket"],
    )
    peg_table_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object", filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
    )
    table = _COMMON_SCENE.table.copy()
    marker_helper = _COMMON_SCENE.marker_helper.copy()
    plane = _COMMON_SCENE.plane.copy()
    # Use the same light intensity, with no remote texture dependency for local validation.
    sky_light = _COMMON_SCENE.sky_light.copy()


@configclass
class DexsuiteRevo3InsertDPegRewardCfg:
    insertion = RewTerm(func=DPegInsertionState, weight=1.0)
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)


def configure_d_peg_insertion(cfg):
    """Configure only the new insertion task after the shared robot/sensors are attached."""
    cfg._d_peg_geometry = asdict(validate_d_peg_geometry_metadata(
        cfg.geometry_metadata_path, cfg.insertion_depth_m, cfg.socket_mouth_height_m,
    ))
    cfg._d_peg_pregrasp = load_d_peg_pregrasp(cfg.pregrasp_path)
    cfg.scene.object.spawn.usd_path = cfg.peg_usd_path
    cfg.scene.socket.spawn.usd_path = cfg.socket_usd_path
    pose = cfg._d_peg_pregrasp["peg_pose"]
    cfg.scene.object.init_state.pos = tuple(pose["position"])
    cfg.scene.object.init_state.rot = tuple(pose["quaternion"])
    cfg.scene.table.init_state.pos = (0.55, 0.0, 0.38)
    cfg.scene.robot.init_state.pos = (1.0, 0.0, 0.766)
    cfg.scene.robot.init_state.rot = (0.7071068, 0.0, 0.0, -0.7071068)
    cfg.scene.robot.init_state.joint_pos = dict(cfg._d_peg_pregrasp["robot_joint_positions"])
    cfg.actions.action = DPegPregraspResidualActionCfg(
        asset_name="robot", joint_names=[".*"],
        scale={"Joint[1-7]_R": 0.1, "right_.*": 0.02},
    )
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.events.randomize_object_scale = None
    cfg.events.object_physics_material = None
    cfg.events.object_scale_mass = None
    cfg.events.variable_gravity = None
    # A calibrated initial grasp uses calibrated gains/materials, not the lift ADR.
    cfg.events.joint_stiffness_and_damping = None
    cfg.events.joint_friction = None
    cfg.events.robot_physics_material = None
    cfg.events.reset_robot_joints = None
    cfg.events.reset_robot_wrist_joint = None
    cfg.events.reset_table.params["pose_range"] = {}
    cfg.events.reset_object = EventTerm(func=reset_d_peg_pregrasp, mode="reset")
    cfg.events.initialize_d_peg = EventTerm(func=reset_d_peg_pregrasp, mode="startup")
    cfg.commands.object_pose.class_type = DPegInsertionPoseCommand
    cfg.commands.object_pose.position_only = False
    cfg.commands.object_pose.debug_vis = False
    cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    cfg.observations.policy.insertion_progress = ObsTerm(func=d_peg_insertion_progress)
    cfg.observations.policy.insertion_phase = ObsTerm(func=d_peg_insertion_phase)
    cfg.observations.perception.object_point_cloud.params["visualize"] = False
    component_cfg = dict(cfg.observations.proprio.rl_ours_hydroshear.params["component_cfg"])
    cfg.observations.proprio.rl_ours_tacmap_policy = None
    cfg.observations.proprio.rl_ours_hydroshear = None
    cfg.observations.proprio.rl_ours_taxim_rgb = None
    cfg.observations.proprio.pretrained_tactile = ObsTerm(
        func=DPegPretrainedTactile, params={"component_cfg": component_cfg},
    )
    cfg.events.zz_initialize_tactile_contract = EventTerm(func=initialize_d_peg_tactile_contract, mode="startup")
    for sensor in vars(cfg.scene).values():
        if getattr(sensor, "target_mesh_prim_path", None):
            sensor.target_mesh_prim_path = sensor.target_mesh_prim_path.replace("/Object", "/Object/TactileSurface")
        for target in getattr(sensor, "mesh_prim_paths", []):
            if hasattr(target, "prim_expr"):
                target.prim_expr = target.prim_expr.replace("/Object", "/Object/TactileSurface")
    cfg.terminations.object_out_of_bound = DoneTerm(
        func=d_peg_out_of_bounds,
        params={"in_bound_range": {"x": (0.1, 0.8), "y": (-0.35, 0.55), "z": (0.60, 1.35)}},
    )
    cfg.terminations.peg_dropped = DoneTerm(func=d_peg_dropped)
    cfg.terminations.non_finite = DoneTerm(func=d_peg_nonfinite)
    cfg.terminations.success = DoneTerm(func=d_peg_success)
    cfg.sim.gravity = (0.0, 0.0, -9.81)
    cfg.sim.dt = 1 / 240
    cfg.decimation = 8
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.min_position_iteration_count = 32
    cfg.sim.physx.min_velocity_iteration_count = 8
    cfg.sim.physx.gpu_collision_stack_size = 2**28
    cfg.episode_length_s = cfg.task_episode_length_s


@configclass
class DexsuiteRevo3InsertDPegEnvCfg(Revo3MixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    scene: DexsuiteRevo3InsertDPegSceneCfg = DexsuiteRevo3InsertDPegSceneCfg(
        num_envs=32, env_spacing=3, replicate_physics=False,
    )
    rewards: DexsuiteRevo3InsertDPegRewardCfg = DexsuiteRevo3InsertDPegRewardCfg()
    curriculum = None
    task_episode_length_s: float = 15.0
    socket_xy_randomization_m: float = 0.005
    socket_yaw_randomization_deg: float = 10.0
    socket_mouth_height_m: float = 0.05
    insertion_depth_m: float = 0.028
    peg_usd_path: str = str(_ASSETS / "peg.usd")
    socket_usd_path: str = str(_ASSETS / "socket.usd")
    geometry_metadata_path: str = str(_ASSETS / "metadata.json")
    pregrasp_path: str = str(_ASSETS / "pregrasp.json")
    tactile_policy_checkpoint: str = str(_REPO_ROOT / "runs/revo3_cross_modal_restoration_v1/best.pt")
    tactile_policy_enabled: bool = False
    tactile_encoder_chunk_size: int = 256
    tactile_taxim_chunk_size: int = 32
    d_peg_fast_taxim: bool = False
    tactile_reconstruction_diagnostics: bool = False

    def __post_init__(self):
        super().__post_init__()
        configure_d_peg_insertion(self)


@configclass
class DexsuiteRevo3InsertDPegEnvCfg_PLAY(DexsuiteRevo3InsertDPegEnvCfg):
    """Use the identical task distribution and observation contract for local play."""
