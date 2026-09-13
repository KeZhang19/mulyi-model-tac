"""D-shaped peg insertion with the Rotate-Bulb-Custom scene and Flexiv asset."""

from dataclasses import MISSING, asdict
import json
import math
import warnings
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
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from ..RotateBulbCustom import base_env_cfg as dexsuite
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
from ..RotateBulbCustom.robot_setup import Revo3MixinCfg
from ..RotateBulbCustom.robot_contract import GRASP_REFERENCE, ROBOT_ORIENTATION, ROBOT_POSITION


_REPO_ROOT = Path(__file__).resolve().parents[8]
_ASSETS = _REPO_ROOT / "assets/d_peg_insertion"
_COMMON_SCENE = dexsuite.SceneCfg(num_envs=1, env_spacing=3)
_JOINT_NAMES = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "right_thumb_CMP_joint", "right_thumb_CMR_joint", "right_thumb_MCP_joint",
    "right_thumb_PIP_joint", "right_thumb_DIP_joint", "right_index_MPR_joint",
    "right_index_MCP_joint", "right_index_PIP_joint", "right_index_DIP_joint",
    "right_middle_MPR_joint", "right_middle_MCP_joint", "right_middle_PIP_joint",
    "right_middle_DIP_joint", "right_ring_MPR_joint", "right_ring_MCP_joint",
    "right_ring_PIP_joint", "right_ring_DIP_joint", "right_little_MPR_joint",
    "right_little_MCP_joint", "right_little_PIP_joint", "right_little_DIP_joint",
)
_SOURCE_TO_CUSTOM = {
    **{f"Joint{i}_R": f"joint{i}" for i in range(1, 8)},
    "right_thumbcmp_roll_joint": "right_thumb_CMP_joint",
    "right_thumbcmr_roll_joint": "right_thumb_CMR_joint",
    "right_thumbmcp_roll_joint": "right_thumb_MCP_joint",
    "right_thumbpip_roll_joint": "right_thumb_PIP_joint",
    "right_thumbdip_roll_joint": "right_thumb_DIP_joint",
}
for _finger, _custom in (("index", "index"), ("mid", "middle"), ("ring", "ring"), ("pinky", "little")):
    for _old, _new in (("mcp_yaw", "MPR"), ("mcp_roll", "MCP"), ("pip_roll", "PIP"), ("dip_roll", "DIP")):
        _SOURCE_TO_CUSTOM[f"right_{_finger}{_old}_joint"] = f"right_{_custom}_{_new}_joint"



def load_d_peg_pregrasp(path):
    """Require the explicit, finite 28-joint calibration; never substitute an open hand."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported D-peg pregrasp schema: {path}")
    validation = data.get("validation", {})
    if validation.get("status") == "editor_candidate" and not validation.get("physics_validated", False):
        warnings.warn(
            f"D-peg pregrasp is an editor candidate and has not passed physics validation: {path}",
            UserWarning,
            stacklevel=2,
        )
    # Calibrations produced with the legacy Tianji asset use Joint*_R and
    # lower-case finger names. Translate them to the Rotate-Bulb-Custom
    # Rizon4/Revo3 inventory while retaining the calibrated values.
    for field in ("robot_joint_positions", "robot_joint_targets"):
        values = data.get(field)
        if isinstance(values, dict):
            data[field] = {_SOURCE_TO_CUSTOM.get(name, name): value for name, value in values.items()}
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
    socket_pose = data.get("socket_pose")
    if socket_pose is None:
        raise ValueError(f"Flexiv pregrasp requires an explicit socket_pose: {path}")
    socket_position, socket_quaternion = socket_pose.get("position", []), socket_pose.get("quaternion", [])
    if (len(socket_position) != 3 or len(socket_quaternion) != 4
            or not all(math.isfinite(float(v)) for v in [*socket_position, *socket_quaternion])):
        raise ValueError(f"socket_pose requires finite position and quaternion: {path}")
    if not math.isclose(sum(float(v) ** 2 for v in socket_quaternion), 1.0, abs_tol=1e-5):
        raise ValueError(f"socket_pose quaternion must be unit length (wxyz): {path}")
    data["socket_pose"] = {
        "position": [float(v) for v in socket_position],
        "quaternion": [float(v) for v in socket_quaternion],
    }
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
    yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    # Compose a world-frame yaw perturbation with the calibrated socket pose;
    # replacing the quaternion would silently discard non-identity calibration.
    socket_pose[:, 3:] = quat_mul(yaw_quat, socket_pose[:, 3:])
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


def _d_peg_in_bound_range(peg_position, socket_position, table_surface_z):
    """Return local bounds enclosing both calibrated bodies and table clearance."""
    return {
        "x": (min(float(peg_position[0]), float(socket_position[0])) - 0.35,
              max(float(peg_position[0]), float(socket_position[0])) + 0.35),
        "y": (min(float(peg_position[1]), float(socket_position[1])) - 0.45,
              max(float(peg_position[1]), float(socket_position[1])) + 0.45),
        "z": (float(table_surface_z) - 0.15,
              max(float(peg_position[2]), float(socket_position[2])) + 0.5),
    }


@configclass
class DexsuiteFlexivInsertDPegSceneCfg(InteractiveSceneCfg):
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
class DexsuiteFlexivInsertDPegRewardCfg:
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
    # Keep the table, plane and lighting placement identical to
    # Rotate-Bulb-Custom. Only the lamp assembly is replaced by the peg/socket.
    table = GRASP_REFERENCE["table"]
    cfg.scene.table.spawn.size = tuple(table["size_m"])
    cfg.scene.table.init_state.pos = tuple(table["position_m"])
    cfg.scene.table.init_state.rot = tuple(table["quaternion_wxyz"])
    cfg.scene.plane.init_state.pos = (
        0.0, 0.0, table["position_m"][2] - table["size_m"][2] / 2,
    )
    pose = cfg._d_peg_pregrasp["peg_pose"]
    cfg.scene.object.init_state.pos = tuple(pose["position"])
    cfg.scene.object.init_state.rot = tuple(pose["quaternion"])
    cfg.scene.robot.init_state.pos = ROBOT_POSITION
    cfg.scene.robot.init_state.rot = ROBOT_ORIENTATION
    cfg.scene.robot.init_state.joint_pos = dict(cfg._d_peg_pregrasp["robot_joint_positions"])
    socket_pose = cfg._d_peg_pregrasp["socket_pose"]
    cfg.scene.socket.init_state.pos = tuple(socket_pose["position"])
    cfg.scene.socket.init_state.rot = tuple(socket_pose["quaternion"])
    table_surface_z = float(table["position_m"][2] + table["size_m"][2] / 2)
    peg_xy = pose["position"][:2]
    socket_xy = socket_pose["position"][:2]
    cfg._d_peg_in_bound_range = _d_peg_in_bound_range(
        pose["position"], socket_pose["position"], table_surface_z,
    )
    cfg.actions.action = DPegPregraspResidualActionCfg(
        asset_name="robot", joint_names=[".*"],
        scale={"joint[1-7]": 0.1, "right_.*": 0.02},
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
        params={"in_bound_range": cfg._d_peg_in_bound_range},
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
class DexsuiteFlexivInsertDPegEnvCfg(Revo3MixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    scene: DexsuiteFlexivInsertDPegSceneCfg = DexsuiteFlexivInsertDPegSceneCfg(
        num_envs=32, env_spacing=3, replicate_physics=False,
    )
    rewards: DexsuiteFlexivInsertDPegRewardCfg = DexsuiteFlexivInsertDPegRewardCfg()
    task_id: str = "BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0"
    curriculum = None
    task_episode_length_s: float = 15.0
    socket_xy_randomization_m: float = 0.005
    socket_yaw_randomization_deg: float = 10.0
    socket_mouth_height_m: float = 0.05
    insertion_depth_m: float = 0.028
    peg_usd_path: str = str(_ASSETS / "peg.usd")
    socket_usd_path: str = str(_ASSETS / "socket.usd")
    geometry_metadata_path: str = str(_ASSETS / "metadata.json")
    pregrasp_path: str = str(_ASSETS / "pregrasp_flexiv.json")
    tactile_policy_checkpoint: str = str(_REPO_ROOT / "runs/revo3_cross_modal_restoration_v1/best.pt")
    tactile_policy_enabled: bool = False
    tactile_encoder_chunk_size: int = 256
    tactile_taxim_chunk_size: int = 32
    d_peg_fast_taxim: bool = False
    tactile_reconstruction_diagnostics: bool = False

    def __post_init__(self):
        super().__post_init__()
        # The custom Rizon4 training USD explicitly enables robot self-collision.
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        configure_d_peg_insertion(self)


@configclass
class DexsuiteFlexivInsertDPegEnvCfg_PLAY(DexsuiteFlexivInsertDPegEnvCfg):
    """Use the identical task distribution and observation contract for local play."""
