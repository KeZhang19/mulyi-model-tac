# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent Rotate-Bulb-Custom scene, rewards, reset and encoder settings."""

from dataclasses import MISSING
import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, subtract_frame_transforms

from . import base_env_cfg as dexsuite
from ... import mdp
from ...mdp.commands.pose_commands import ObjectUniformPoseCommand
from .tactile import RotateBulbPretrainedTactile, initialize_rotate_bulb_tactile_contract
from .robot_setup import Revo3MixinCfg
from .robot_contract import GRASP_REFERENCE


_LAMP_USD = str(
    Path(__file__).resolve().parents[8] / "assets/bulb/hiveboard_lamp/hiveboard_lamp_threaded.usd"
)
# Rotate the asset's screw axis (+X) upright (+Z). The support's original
# minimum X is -0.044732484966516495 m (threaded_metadata.json).
_LAMP_ROT = (0.7071067811865476, 0.0, -0.7071067811865476, 0.0)
_SUPPORT_BOTTOM_OFFSET = 0.044732484966516495
# Table-local XY. With the table centered at X=0.55, these retain the
# original environment-local socket region X=[0.43, 0.47], Y=[0.08, 0.12].
_LAMP_X_RANGE = (-0.12, -0.08)
_LAMP_Y_RANGE = (0.08, 0.12)
_SCREW_PITCH = 0.006
_SCREW_TRAVEL = 0.024
_INITIAL_SCREW_TURN = -2 * math.pi * _SCREW_TRAVEL / _SCREW_PITCH
# Extend the USD's 24.3 mm endpoint by exactly one pitch to preserve the thread
# phase. The 30.021 mm external thread then clears the mouth by ~0.28 mm.
_SCREW_EXIT_EXTENSION = 0.0243 + _SCREW_PITCH
_SCREW_RELEASE_TOLERANCE = 0.001
_COMMON_SCENE = dexsuite.SceneCfg(num_envs=1, env_spacing=3)
_PRETRAINED_TACTILE_CHECKPOINT = str(
    Path(__file__).resolve().parents[8] / "runs/revo3_cross_modal_restoration_v1/best.pt"
)


@sim_utils.clone
def spawn_tabletop_threaded_lamp(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Keep the USD's static socket movable together with its articulation at reset."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    # Support is a sibling of Base in the source USD, not an articulation link.
    # A kinematic rigid body keeps its concave thread collider intact and lets
    # reset move the socket through PhysX, including when Fabric is enabled.
    UsdGeom.Xform.Define(prim.GetStage(), prim_path + "/Support")
    sim_utils.define_rigid_body_properties(
        prim_path + "/Support",
        sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    )
    stage = prim.GetStage()
    # Keep the two native screw joints in a permanent articulation, but attach
    # the physical bulb through an EXCLUDED regular joint. Releasing it does
    # not change articulation topology or invalidate its tensor views.
    follower = UsdGeom.Xform.Define(stage, prim_path + "/ScrewFollower")
    follower.AddTranslateOp().Set(Gf.Vec3d(_SCREW_EXIT_EXTENSION, 0, 0))
    follower.AddOrientOp().Set(Gf.Quatf(0, 1, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(follower.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(follower.GetPrim())
    mass.CreateMassAttr(0.01)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5))
    PhysxSchema.PhysxRigidBodyAPI.Apply(follower.GetPrim()).CreateDisableGravityAttr(True)
    UsdPhysics.RevoluteJoint.Get(stage, prim_path + "/Joints/screw_turn").GetBody1Rel().SetTargets(
        [follower.GetPath()]
    )
    UsdPhysics.PrismaticJoint.Get(stage, prim_path + "/Joints/screw_slide").GetLocalPos0Attr().Set(
        Gf.Vec3f(_SCREW_EXIT_EXTENSION, 0, 0)
    )
    for name in ("AxialCarrier", "Bulb"):
        stage.GetPrimAtPath(prim_path + "/" + name).GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(_SCREW_EXIT_EXTENSION, 0, 0)
        )
    bulb_prim = stage.GetPrimAtPath(prim_path + "/Bulb")
    PhysxSchema.PhysxRigidBodyAPI.Apply(bulb_prim).CreateDisableGravityAttr(False)
    attachment = UsdPhysics.FixedJoint.Define(stage, prim_path + "/Joints/bulb_attachment")
    attachment.CreateBody0Rel().SetTargets([follower.GetPath()])
    attachment.CreateBody1Rel().SetTargets([Sdf.Path(prim_path + "/Bulb")])
    attachment.CreateLocalPos0Attr(Gf.Vec3f(0))
    attachment.CreateLocalPos1Attr(Gf.Vec3f(0))
    attachment.CreateLocalRot0Attr(Gf.Quatf(1))
    attachment.CreateLocalRot1Attr(Gf.Quatf(1))
    attachment.CreateExcludeFromArticulationAttr(True)
    attachment.CreateJointEnabledAttr(True)
    # The original shell shares the thread's low-friction material. Give the
    # gripping surfaces their own material without changing screw contacts.
    grip_material_path = prim_path + "/Materials/GripContact"
    grip_material = sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8)
    grip_material.func(grip_material_path, grip_material)
    for name in ("ShellA", "ShellB"):
        sim_utils.bind_physics_material(prim_path + "/Bulb/" + name, grip_material_path)
    # The tactile mesh loader resolves one mesh. Combine all three visible
    # bulb pieces in the bulb's local frame so it senses the shells as well as
    # the screw. This hidden sensing surface has no physics/collision API.
    points, counts, indices = [], [], []
    for name in ("ExternalThread", "ShellA", "ShellB"):
        mesh = UsdGeom.Mesh(prim.GetStage().GetPrimAtPath(prim_path + "/Bulb/" + name))
        indices.extend(int(index) + len(points) for index in mesh.GetFaceVertexIndicesAttr().Get())
        counts.extend(mesh.GetFaceVertexCountsAttr().Get())
        points.extend(mesh.GetPointsAttr().Get())
    surface = UsdGeom.Mesh.Define(prim.GetStage(), prim_path + "/Bulb/TactileSurface")
    surface.CreatePointsAttr(points)
    surface.CreateFaceVertexCountsAttr(counts)
    surface.CreateFaceVertexIndicesAttr(indices)
    surface.CreateSubdivisionSchemeAttr("none")
    surface.CreateVisibilityAttr("invisible")
    return prim


@configclass
class DexsuiteRevo3RotateBulbSceneCfg(InteractiveSceneCfg):
    """Spawn the assembly before creating views of the bulb and support bodies."""

    robot: ArticulationCfg = MISSING
    lamp = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Lamp",
        spawn=sim_utils.UsdFileCfg(usd_path=_LAMP_USD, func=spawn_tabletop_threaded_lamp),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.45, 0.1, 0.76 + _SUPPORT_BOTTOM_OFFSET), rot=_LAMP_ROT,
            # Start seated: positive rotation about world +Z unscrews upward
            # by 6 mm/revolution, with 24 mm available before the upper stop.
            joint_pos={"screw_turn": _INITIAL_SCREW_TURN, "screw_slide": -_SCREW_TRAVEL},
            joint_vel={".*": 0.0},
        ),
        # The hand supplies the torque; disable the demo's automatic position
        # drive, while retaining the native rotation/translation mimic joint.
        actuators={"passive_screw": ImplicitActuatorCfg(
            joint_names_expr=["screw_turn", "screw_slide"], stiffness=0.0, damping=0.0,
            # Isaac Sim 5.1 Coulomb friction is an effort (Nm for the turn).
            # Gravity alone supplies ~0.0019 Nm through this screw pitch.
            # Passive resistance holds the sampled depth until the hand turns it.
            friction={"screw_turn": 0.004, "screw_slide": 0.0},
            dynamic_friction={"screw_turn": 0.003, "screw_slide": 0.0},
            viscous_friction={"screw_turn": 0.0005, "screw_slide": 0.0},
        )},
    )
    # Existing observations, rewards and tactile models read the moving bulb,
    # rather than the fixed articulation base. These views spawn no extra USD.
    object = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Lamp/Bulb", spawn=None)
    support = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Lamp/Support", spawn=None)
    bulb_table_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Lamp/Bulb", filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
    )
    table = _COMMON_SCENE.table.copy()
    marker_helper = _COMMON_SCENE.marker_helper.copy()
    plane = _COMMON_SCENE.plane.copy()
    sky_light = _COMMON_SCENE.sky_light.copy()


def reset_lamp_on_table(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    table_half_height: float,
    support_bottom_offset: float = _SUPPORT_BOTTOM_OFFSET,
    support_pose: tuple[float, ...] | None = None,
):
    """Fix the support on the table and seat the bulb for upward unscrewing.

    An explicit environment-local support_pose restores the editor reference,
    independent of table rotation. Otherwise sample in the table's local frame.
    XY randomization happens only at startup/reset. During the episode the
    table and support remain kinematic and the articulation base stays fixed.
    Reset also reconnects a bulb which detached during the previous attempt.
    """
    lamp = env.scene["lamp"]
    # Read at reset so Hydra/from_dict overrides after __post_init__ also
    # affect startup and every subsequent (including partial) reset.
    initial_turns_range = env.cfg.initial_screw_turns_range
    if not 0 < initial_turns_range[0] <= initial_turns_range[1] <= 4.0:
        raise ValueError("initial_screw_turns_range must satisfy 0 < min <= max <= 4")
    support = env.scene["support"]
    table = env.scene["table"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # Capture the terminal physics state before overwriting the seated joints.
    # CommandManager.compute runs after resets and cannot see that last step.
    turn_metrics = getattr(env, "_bulb_turn_metrics_command", None)
    if turn_metrics is not None and hasattr(env, "_bulb_start_joints"):
        turn_metrics.record_turn_metrics(env_ids)

    root_state = lamp.data.default_root_state[env_ids].clone()
    offset = torch.zeros_like(root_state[:, :3])
    if support_pose is not None:
        root_state[:, :7] = torch.as_tensor(support_pose, device=env.device, dtype=root_state.dtype)
        root_state[:, :3] += env.scene.env_origins[env_ids]
    else:
        offset[:, 0].uniform_(*x_range)
        offset[:, 1].uniform_(*y_range)
        offset[:, 2] = table_half_height + support_bottom_offset
        # Table poses already contain each environment's origin.
        root_state[:, :3], root_state[:, 3:7] = combine_frame_transforms(
            table.data.root_pos_w[env_ids], table.data.root_quat_w[env_ids], offset, root_state[:, 3:7]
        )
    root_state[:, 7:] = 0.0
    support.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    support.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    lamp.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    lamp.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    joint_pos = lamp.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    turns = torch.empty(len(env_ids), device=env.device).uniform_(*initial_turns_range)
    turn_id, slide_id = (lamp.joint_names.index(name) for name in ("screw_turn", "screw_slide"))
    joint_pos[:, turn_id] = -2 * math.pi * turns
    joint_pos[:, slide_id] = -_SCREW_PITCH * turns
    lamp.reset(env_ids=env_ids)
    lamp.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # Clear targets as well as state, including a previous episode's torque.
    lamp.set_joint_position_target(joint_pos, env_ids=env_ids)
    lamp.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    lamp.set_joint_effort_target(torch.zeros_like(joint_vel), env_ids=env_ids)
    # Restore both independent bulb state and its mechanical attachment, only
    # for the resetting environments. Never teleport during an active episode.
    offset.zero_()
    offset[:, 0] = _SCREW_EXIT_EXTENSION + joint_pos[:, slide_id]
    angle = joint_pos[:, turn_id] + math.pi
    zero = torch.zeros_like(angle)
    bulb_pos, bulb_quat = combine_frame_transforms(
        root_state[:, :3], root_state[:, 3:7], offset, quat_from_euler_xyz(angle, zero, zero)
    )
    env.scene["object"].write_root_pose_to_sim(torch.cat((bulb_pos, bulb_quat), dim=-1), env_ids=env_ids)
    env.scene["object"].write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]), env_ids=env_ids)
    if not hasattr(env, "_bulb_released"):
        env._bulb_released = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._bulb_start_joints = lamp.data.default_joint_pos.clone()
        env._bulb_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._bulb_attachment_attrs = [
            env.sim.stage.GetPrimAtPath(f"/World/envs/env_{i}/Lamp/Joints/bulb_attachment").GetAttribute(
                "physics:jointEnabled"
            ) for i in range(env.num_envs)
        ]
    env._bulb_start_joints[env_ids] = joint_pos
    env._bulb_released[env_ids] = False
    env._bulb_success[env_ids] = False
    for index in env_ids.tolist():
        env._bulb_attachment_attrs[index].Set(True)


def release_unscrewed_bulb(env: ManagerBasedRLEnv, env_ids=None):
    """Release at the physical exit once; ordinary rigid-body dynamics take over."""
    turn_metrics = getattr(env, "_bulb_turn_metrics_command", None)
    if turn_metrics is not None:
        turn_metrics.record_turn_metrics()
    ready = rotate_bulb_screw_complete(env) & ~env._bulb_released
    for index in ready.nonzero(as_tuple=False).flatten().tolist():
        env._bulb_attachment_attrs[index].Set(False)
    env._bulb_released |= ready


class RotateBulbPoseCommand(ObjectUniformPoseCommand):
    """Sample a nearby goal relative to the fixed socket, in robot coordinates."""

    turn_metric_names = ("unscrew_turns_max", "unscrew_turns_final", "unscrew_reached_1_turn",
                         "unscrew_reached_2_turns", "unscrew_reached_3_turns")

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.turns = torch.zeros(self.num_envs, device=self.device)
        self.max_turns = torch.zeros_like(self.turns)
        self._turn_metrics_reset_step = -1
        env._bulb_turn_metrics_command = self

    def record_turn_metrics(self, env_ids=None):
        """Track physical progress independently of reward weights and contacts."""
        ids = slice(None) if env_ids is None else env_ids
        turns = rotate_bulb_unscrewed_turns(self._env)[ids]
        # After detachment the screw follower no longer measures bulb motion.
        self.turns[ids] = torch.where(self._env._bulb_released[ids], self.turns[ids], turns)
        self.max_turns[ids] = torch.maximum(self.max_turns[ids], self.turns[ids])

    def reset(self, env_ids=None):
        # Scene reset clears _bulb_success before CommandManager.reset. The
        # termination flag still contains this completed episode's outcome.
        metrics = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        metrics["success"] = self._env.termination_manager.get_term("success")[ids].float().mean().item()
        completed = self._env.episode_length_buf[ids] > 0
        peak = self.max_turns[ids][completed].clone()
        final = self.turns[ids][completed].clone()
        # Keep one sample per completed episode, including failures. RSL-RL
        # concatenates these vectors before averaging, so unequal reset batches
        # have the correct weight. Startup resets contribute no samples.
        values = (peak, final, *((peak >= n - 1e-5).float() for n in (1, 2, 3)))
        metrics.update(zip(self.turn_metric_names, values))
        self.turns[ids] = 0.0
        self.max_turns[ids] = 0.0
        self._turn_metrics_reset_step = self._env.common_step_counter
        return metrics

    def _resample_command(self, env_ids):
        support = self._env.scene["support"]
        offset = torch.empty((len(env_ids), 3), device=self.device)
        offset[:, 0].uniform_(*self.cfg.ranges.pos_x)
        offset[:, 1].uniform_(*self.cfg.ranges.pos_y)
        offset[:, 2].uniform_(*self.cfg.ranges.pos_z)
        goal_w = support.data.root_pos_w[env_ids] + offset
        goal_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids], self.robot.data.root_quat_w[env_ids], goal_w
        )
        self.pose_command_b[env_ids, :3] = goal_b
        self.pose_command_b[env_ids, 3:] = 0.0
        self.pose_command_b[env_ids, 3] = 1.0

    def _update_metrics(self):
        super()._update_metrics()
        self.record_turn_metrics()
        if self._turn_metrics_reset_step != self._env.common_step_counter:
            # Isaac Lab retains extras on steps without resets. Replace the
            # log dict (the runner retains earlier dicts) to avoid recounting
            # old episodes. Empty vectors preserve the logger's metric keys.
            log = dict(self._env.extras.get("log", {}))
            log.update({f"Metrics/object_pose/{name}": self.turns[:0].clone()
                        for name in self.turn_metric_names})
            self._env.extras["log"] = log
        success = getattr(self._env, "_bulb_success", torch.zeros(self.num_envs, device=self.device))
        self.metrics["success"] = success.float()
        self.metrics["released"] = getattr(
            self._env, "_bulb_released", torch.zeros(self.num_envs, device=self.device)
        ).float()
        self.success_visualizer.visualize(self.success_vis_asset.data.root_pos_w, marker_indices=success.int())


def rotate_bulb_dropped(env: ManagerBasedRLEnv, force_threshold: float = 0.5) -> torch.Tensor:
    """After release, table contact ends the attempt regardless of bulb orientation."""
    forces = env.scene.sensors["bulb_table_contact"].data.force_matrix_w
    contact = torch.linalg.vector_norm(forces, dim=-1).flatten(start_dim=1).max(dim=-1).values
    return env._bulb_released & (contact > force_threshold)


def rotate_bulb_phase_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose release and remaining time for the finite-horizon task."""
    released = getattr(env, "_bulb_released", torch.zeros(env.num_envs, device=env.device))
    time_left = (1.0 - env.episode_length_buf / env.max_episode_length).clamp(0.0, 1.0)
    return torch.stack((released.float(), time_left), dim=-1)


def reset_robot_to_grasp(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None):
    """Restore all reference joints and clear previous episode drive targets."""
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    position = robot.data.default_joint_pos[env_ids].clone()
    velocity = torch.zeros_like(position)
    robot.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
    robot.set_joint_position_target(position, env_ids=env_ids)
    robot.set_joint_velocity_target(velocity, env_ids=env_ids)
    robot.set_joint_effort_target(torch.zeros_like(position), env_ids=env_ids)


def configure_rotate_bulb_scene(cfg):
    """Apply the lamp-specific reset and sensor targets after the Revo3 mixin."""
    # Use actual saved world poses (metres, wxyz), including the table's yaw.
    # The robot asset config reads the same reference by joint name.
    table = GRASP_REFERENCE["table"]
    support = next(item for item in GRASP_REFERENCE["fixtures"] if item["name"] == "lamp_support")
    cfg.scene.table.spawn.size = tuple(table["size_m"])
    cfg.scene.table.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.45, 0.32, 0.22), metallic=0.0, roughness=0.7,
    )
    cfg.scene.table.init_state.pos = tuple(table["position_m"])
    cfg.scene.table.init_state.rot = tuple(table["quaternion_wxyz"])
    tabletop_z = table["surface_center_world_m"][2]
    # The editor's origin is the arm base, with the table extending below Z=0.
    cfg.scene.plane.init_state.pos = (0.0, 0.0, table["position_m"][2] - table["size_m"][2] / 2)
    cfg.viewer.eye = (0.6, 1.1, 0.85)
    cfg.viewer.lookat = (-0.4, 0.0, 0.18)
    cfg.events.randomize_object_scale = None
    cfg.events.object_physics_material = None
    cfg.events.object_scale_mass = None
    lower, upper = cfg.initial_screw_turns_range
    if not 0 < lower <= upper <= _SCREW_TRAVEL / _SCREW_PITCH:
        raise ValueError("initial_screw_turns_range must satisfy 0 < min <= max <= 4")
    cfg.scene.lamp.init_state.joint_pos = {
        "screw_turn": -2 * math.pi * (lower + upper) / 2,
        "screw_slide": -_SCREW_PITCH * (lower + upper) / 2,
    }
    cfg.events.reset_object = EventTerm(
        func=reset_lamp_on_table,
        mode="reset",
        params={
            "x_range": (0.0, 0.0),
            "y_range": (0.0, 0.0),
            "table_half_height": cfg.scene.table.spawn.size[2] / 2,
            "support_pose": tuple(support["position_m"] + support["quaternion_wxyz"]),
        },
    )
    # Also establish the seated pose when the environment is first created,
    # before the caller's first env.reset(). The source USD starts extended.
    cfg.events.initialize_lamp = cfg.events.reset_object.replace(mode="startup")
    cfg.events.release_bulb = EventTerm(
        func=release_unscrewed_bulb, mode="interval", interval_range_s=(0.0, 0.0),
    )
    cfg.events.reset_table.params["pose_range"] = {}
    cfg.events.reset_table.params["velocity_range"] = {}
    cfg.events.reset_root.params["pose_range"] = {}
    cfg.events.reset_root.params["velocity_range"] = {}
    cfg.events.reset_robot_joints = EventTerm(func=reset_robot_to_grasp, mode="reset")
    cfg.events.reset_robot_wrist_joint = None
    cfg.events.initialize_robot = cfg.events.reset_robot_joints.replace(mode="startup")
    # Use gravity from the first attempt. The inherited ADR promotes based on
    # distance alone and can also change global gravity during partial resets.
    cfg.curriculum = None
    cfg.events.variable_gravity = None
    cfg.sim.gravity = (0.0, 0.0, -9.81)
    cfg.commands.object_pose.class_type = RotateBulbPoseCommand
    cfg.commands.object_pose.ranges.pos_x = (-0.04, 0.04)
    cfg.commands.object_pose.ranges.pos_y = (-0.04, 0.04)
    cfg.commands.object_pose.ranges.pos_z = (0.12, 0.16)
    bulb_x, bulb_y, _ = GRASP_REFERENCE["object"]["position_m"]
    cfg.terminations.object_out_of_bound.params["in_bound_range"] = {
        "x": (bulb_x - 0.35, bulb_x + 0.30),
        "y": (bulb_y - 0.45, bulb_y + 0.45),
        "z": (tabletop_z - 0.11, tabletop_z + 0.54),
    }
    cfg.terminations.bulb_dropped = DoneTerm(func=rotate_bulb_dropped)
    cfg.scene.lamp.init_state.pos = tuple(support["position_m"])
    cfg.scene.lamp.init_state.rot = tuple(support["quaternion_wxyz"])
    for sensor in vars(cfg.scene).values():
        if hasattr(sensor, "filter_prim_paths_expr"):
            sensor.filter_prim_paths_expr = [
                path.replace("/Object", "/Lamp/Bulb") for path in sensor.filter_prim_paths_expr
            ]
        if getattr(sensor, "target_mesh_prim_path", None):
            sensor.target_mesh_prim_path = sensor.target_mesh_prim_path.replace(
                "/Object", "/Lamp/Bulb/TactileSurface"
            )
        for target in getattr(sensor, "mesh_prim_paths", []):
            if hasattr(target, "prim_expr"):
                target.prim_expr = target.prim_expr.replace("/Object", "/Lamp/Bulb/TactileSurface")
    # SDF thread contacts and native screw coupling use the asset's validated
    # solver settings. Keep the original 60 Hz policy/control rate.
    cfg.sim.dt = 1 / 240
    cfg.decimation = 4
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.min_position_iteration_count = 32
    cfg.sim.physx.min_velocity_iteration_count = 8
    # Reserve 256 MiB for collision stacks shared by this process's environments.
    cfg.sim.physx.gpu_collision_stack_size = 2**28
    # Use the task's episode duration; episode_length_s can also be overridden
    # after construction when evaluating a different screw depth.
    cfg.episode_length_s = cfg.task_episode_length_s
    cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.object_pose.position_only = True
    cfg.rewards.orientation_tracking = None
    cfg.rewards.success.params["rot_std"] = None
    # End a completed task before post-success action costs could favor dropping.
    # The termination and reward instances run the same strict timer once each
    # per control step; terminations run first and the reward is paid before reset.
    cfg.terminations.success = DoneTerm(func=RotateBulbSuccessTermination)
    # Orientation alone cannot distinguish zero from four full revolutions.
    cfg.observations.policy.screw_progress = ObsTerm(func=rotate_bulb_screw_progress)
    cfg.observations.policy.bulb_phase = ObsTerm(func=rotate_bulb_phase_state)
    # Keep pressure and state observations; replace the old independent image
    # ResNets and raw 3-D shear vector with jointly encoded RGB/Depth/2-D markers.
    component_cfg = dict(cfg.observations.proprio.rl_ours_hydroshear.params["component_cfg"])
    cfg.observations.proprio.rl_ours_tacmap_policy = None
    cfg.observations.proprio.rl_ours_hydroshear = None
    cfg.observations.proprio.rl_ours_taxim_rgb = None
    cfg.observations.proprio.pretrained_tactile = ObsTerm(
        func=RotateBulbPretrainedTactile, params={"component_cfg": component_cfg},
    )
    cfg.events.zz_initialize_tactile_contract = EventTerm(
        func=initialize_rotate_bulb_tactile_contract, mode="startup",
    )


def rotate_bulb_hand_contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Thumb plus at least one other fingertip contact for the Revo3 hand."""
    thumb_contact_sensor: ContactSensor = env.scene.sensors["right_thumbdip_roll_rubber_link_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["right_indexdip_roll_rubber_link_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["right_middip_roll_rubber_link_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["right_ringdip_roll_rubber_link_object_s"]
    little_contact_sensor: ContactSensor = env.scene.sensors["right_pinkydip_roll_rubber_link_object_s"]

    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    little_contact = little_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    little_contact_mag = torch.norm(little_contact, dim=-1)

    return (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold)
        | (middle_contact_mag > threshold)
        | (ring_contact_mag > threshold)
        | (little_contact_mag > threshold)
    )


def rotate_bulb_position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Transport shaping only after unscrewing, while holding the bulb."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import combine_frame_transforms

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(obj.data.root_pos_w - des_pos_w, dim=1)
    gate = env._bulb_released & rotate_bulb_hand_contacts(env, 1.0)
    return (1 - torch.tanh(distance / std)) * gate.float()


def rotate_bulb_screw_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Unwrapped angular and axial progress from the seated pose, in [0, 1]."""
    lamp = env.scene["lamp"]
    ids = [lamp.joint_names.index(name) for name in ("screw_turn", "screw_slide")]
    start = getattr(env, "_bulb_start_joints", lamp.data.default_joint_pos)[:, ids]
    progress = ((lamp.data.joint_pos[:, ids] - start) / (-start).clamp_min(1e-6)).clamp(0.0, 1.0)
    if hasattr(env, "_bulb_released"):
        progress = torch.where(env._bulb_released[:, None], 1.0, progress)
    return progress


def rotate_bulb_screw_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Both joints reach the exit, or the bulb has already been released."""
    return (rotate_bulb_screw_progress(env) >= 1.0 - _SCREW_RELEASE_TOLERANCE).all(dim=-1)


def rotate_bulb_unscrewed_turns(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Actual turns from this reset, limited by both angular and axial travel."""
    lamp = env.scene["lamp"]
    ids = [lamp.joint_names.index(name) for name in ("screw_turn", "screw_slide")]
    start = getattr(env, "_bulb_start_joints", lamp.data.default_joint_pos)[:, ids]
    travel = torch.minimum((lamp.data.joint_pos[:, ids] - start).clamp_min(0.0), (-start).clamp_min(0.0))
    # Do not use the phase observation here: it latches to 100% on release,
    # after which the empty follower can move independently of the bulb.
    return torch.minimum(travel[:, 0] / (2 * math.pi), travel[:, 1] / _SCREW_PITCH)


def rotate_bulb_contact_scores(env: ManagerBasedRLEnv, force_scale: float = 1.0) -> torch.Tensor:
    """Saturating [thumb, strongest other fingertip] contact scores in [0, 1]."""
    if not math.isfinite(force_scale) or force_scale <= 0.0:
        raise ValueError("force_scale must be finite and positive")
    forces = []
    for finger in ("thumb", "index", "mid", "ring", "pinky"):
        sensor = env.scene.sensors[f"right_{finger}dip_roll_rubber_link_object_s"]
        forces.append(sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3).norm(dim=-1).amax(dim=-1))
    scores = (torch.stack(forces, dim=-1) / force_scale).clamp(0.0, 1.0)
    return torch.stack((scores[:, 0], scores[:, 1:].amax(dim=-1)), dim=-1)


def rotate_bulb_grasp_shaping(env: ManagerBasedRLEnv, force_scale: float = 1.0) -> torch.Tensor:
    """Guide initial contact and opposing grasp continuously while unscrewing."""
    scores = rotate_bulb_contact_scores(env, force_scale)
    # One side alone earns at most 0.25; most credit requires both sides.
    quality = (scores.sum(dim=-1) + 2 * scores.min(dim=-1).values) / 4
    return quality * (~rotate_bulb_screw_complete(env)).float()


def rotate_bulb_any_finger_contact(env: ManagerBasedRLEnv, force_scale: float = 0.5) -> torch.Tensor:
    """Average the five independently saturated fingertip contacts with the bulb.

    Use the bulb-filtered force matrix, so touching the table or socket does not
    count. Each finger contributes up to one fifth of the score: with weight=0.1,
    each pays at most 0.02 per second. This continues after the grasp budget is used.
    """
    if not math.isfinite(force_scale) or force_scale <= 0.0:
        raise ValueError("force_scale must be finite and positive")
    forces = []
    for finger in ("thumb", "index", "mid", "ring", "pinky"):
        sensor = env.scene.sensors[f"right_{finger}dip_roll_rubber_link_object_s"]
        magnitude = sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3).norm(dim=-1)
        # Sanitize before saturation: an infinite force must not become full credit.
        magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        forces.append(magnitude.amax(dim=-1))
    # Saturate before averaging so excess force on one finger cannot earn the
    # shares belonging to the other four fingers.
    quality = (torch.stack(forces, dim=-1) / force_scale).clamp(0.0, 1.0).mean(dim=-1)
    return quality * (~rotate_bulb_failed(env)).float()


class RotateBulbUnscrewProgress(ManagerTermBase):
    """Pay per new physical turn, with bounded contact memory for regrasping."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.best = rotate_bulb_unscrewed_turns(env).clone()
        self.contact_memory = torch.zeros(env.num_envs, device=env.device)
        self.completion_paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        # RewardManager resets after the task has restored the seated joints.
        self.best[ids] = rotate_bulb_unscrewed_turns(self._env)[ids]
        self.contact_memory[ids] = 0.0
        self.completion_paid[ids] = False

    def __call__(
        self, env, contact_threshold: float = 1.0, completion_bonus: float = 0.5,
        regrasp_grace_s: float = 0.2,
    ):
        if not math.isfinite(regrasp_grace_s) or regrasp_grace_s < 0.0:
            raise ValueError("regrasp_grace_s must be finite and non-negative")
        progress = rotate_bulb_unscrewed_turns(env)
        progress = torch.where(env._bulb_released, self.best, progress)
        quality = rotate_bulb_contact_scores(env, contact_threshold).min(dim=-1).values
        if regrasp_grace_s > 0.0:
            recent = (self.contact_memory - env.step_dt / regrasp_grace_s).clamp_min(0.0)
            quality = torch.maximum(quality, recent)
        self.contact_memory = torch.where(env._bulb_released, 0.0, quality)
        contact = rotate_bulb_hand_contacts(env, contact_threshold)
        delta = (progress - self.best).clamp_min(0.0)
        # Always consume new physical progress, even without contact. Regrasp
        # can earn future turns, never reclaim uncredited travel or backtracking.
        self.best = torch.maximum(self.best, progress)
        completed = rotate_bulb_screw_complete(env) & contact & ~self.completion_paid
        self.completion_paid |= completed
        # Isaac Lab multiplies by step_dt. With weight=40 this pays 40 per
        # fully held turn at every initial depth, plus the same 20 at completion.
        return (delta * self.contact_memory + completion_bonus * completed.float()) / env.step_dt


class RotateBulbPhysicalTurnProgress(ManagerTermBase):
    """Pay only for new best effective turns, independently of grasp quality."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.best = rotate_bulb_unscrewed_turns(env).clone()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.best[ids] = rotate_bulb_unscrewed_turns(self._env)[ids]

    def __call__(self, env):
        turns = torch.where(env._bulb_released, self.best, rotate_bulb_unscrewed_turns(env))
        delta = (turns - self.best).clamp_min(0.0)
        self.best = torch.maximum(self.best, turns)
        return delta / env.step_dt


def rotate_bulb_reach_shell(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reach the grippable shell, 4 cm above the bulb's screw-frame origin."""
    bulb = env.scene["object"]
    offset = torch.zeros_like(bulb.data.root_pos_w)
    offset[:, 0] = 0.04  # Bulb local +X, following its pose in world space.
    shell_pos, _ = combine_frame_transforms(bulb.data.root_pos_w, bulb.data.root_quat_w, offset)
    tips = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
    distance = torch.linalg.vector_norm(tips - shell_pos[:, None, :], dim=-1).mean(dim=-1)
    # Once rotation is complete, the position rewards take over.
    return (1 - torch.tanh(distance / std)) * (~rotate_bulb_screw_complete(env)).float()


class RotateBulbReachProgress(ManagerTermBase):
    """Reward new best approach to the shell, with per-environment reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.best = rotate_bulb_reach_shell(env, **cfg.params).clone()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.best[ids] = rotate_bulb_reach_shell(self._env, **self.cfg.params)[ids]

    def __call__(self, env, std: float, asset_cfg: SceneEntityCfg):
        reach = rotate_bulb_reach_shell(env, std, asset_cfg)
        delta = (reach - self.best).clamp_min(0.0)
        self.best = torch.maximum(self.best, reach)
        return delta / env.step_dt


def rotate_bulb_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Read this step's physical failure flags before the manager resets them."""
    return env._bulb_released & (
        env.termination_manager.get_term("bulb_dropped")
        | env.termination_manager.get_term("object_out_of_bound")
    )


def rotate_bulb_bounded_credit(env, quality, paid, reward_rate, max_reward):
    """Integrate a reward rate in seconds, with a finite per-episode budget."""
    if not all(math.isfinite(x) and x >= 0.0 for x in (reward_rate, max_reward)):
        raise ValueError("reward_rate and max_reward must be finite and non-negative")
    quality = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    return torch.minimum(quality * reward_rate * env.step_dt, (max_reward - paid).clamp_min(0.0))


class RotateBulbGraspBudget(ManagerTermBase):
    """Bound pre-release contact guidance so waiting cannot dominate progress."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.paid[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, force_scale: float = 1.0, reward_rate: float = 0.25, max_reward: float = 1.0):
        credit = rotate_bulb_bounded_credit(
            env, rotate_bulb_grasp_shaping(env, force_scale), self.paid, reward_rate, max_reward,
        )
        self.paid += credit
        return credit / env.step_dt


def rotate_bulb_released_grasp_quality(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Soft opposing contact after release, allowing deliberate slow transport."""
    bulb = env.scene["object"]
    clear = bulb.data.root_pos_w[:, 2] - env.scene["support"].data.root_pos_w[:, 2] > 0.015
    linear_speed = bulb.data.root_lin_vel_w.norm(dim=-1)
    angular_speed = bulb.data.root_ang_vel_w.norm(dim=-1)
    # Translation up to 20 cm/s is not penalized; rapid spinning/throwing is.
    settled = torch.exp(-((linear_speed - 0.2).clamp_min(0.0) / 0.2).square()
                        - (angular_speed / 2.0).square())
    quality = rotate_bulb_contact_scores(env).min(dim=-1).values * settled
    return quality * (env._bulb_released & clear & ~rotate_bulb_failed(env)).float()


class RotateBulbReleasedStability(ManagerTermBase):
    """Pay at most two points for stable free-bulb contact, independent of Hz."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.paid[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, reward_rate: float = 0.5, max_reward: float = 2.0):
        credit = rotate_bulb_bounded_credit(
            env, rotate_bulb_released_grasp_quality(env), self.paid, reward_rate, max_reward,
        )
        self.paid += credit
        return credit / env.step_dt


class RotateBulbReleaseGrasp(ManagerTermBase):
    """A one-time milestone for retaining the bulb after physical detachment."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hold_time = torch.zeros(env.num_envs, device=env.device)
        self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.hold_time[ids] = 0.0
        self.paid[ids] = False

    def __call__(self, env, hold_duration: float = 0.3):
        if not math.isfinite(hold_duration) or hold_duration <= 0.0:
            raise ValueError("hold_duration must be finite and positive")
        bulb = env.scene["object"]
        clear = bulb.data.root_pos_w[:, 2] - env.scene["support"].data.root_pos_w[:, 2] > 0.015
        valid = env._bulb_released & clear & ~rotate_bulb_failed(env) & rotate_bulb_hand_contacts(env, 1.0)
        valid &= (bulb.data.root_lin_vel_w.norm(dim=-1) < 0.2)
        valid &= (bulb.data.root_ang_vel_w.norm(dim=-1) < 1.0)
        self.hold_time = torch.where(valid, self.hold_time + env.step_dt, 0.0)
        reached = valid & (self.hold_time + 1e-6 >= hold_duration)
        credit = reached & ~self.paid
        self.paid |= reached
        return credit.float() / env.step_dt


class RotateBulbTransportProgress(ManagerTermBase):
    """Dense post-release position tracking, following the v2-test reward."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def __call__(self, env, std: float, command_name: str, asset_cfg: SceneEntityCfg,
                 align_asset_cfg: SceneEntityCfg):
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("std must be finite and positive")
        robot = env.scene[asset_cfg.name]
        bulb = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        goal, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
        potential = 1.0 - torch.tanh((bulb.data.root_pos_w - goal).norm(dim=-1) / std)
        potential = torch.nan_to_num(potential, nan=0.0)
        # Keep the physical phase and failure gates, but return the current
        # tracking quality every step instead of only a new-best delta.  The
        # reward manager multiplies this raw [0, 1] value by step_dt.
        contact = rotate_bulb_contact_scores(env).min(dim=-1).values
        contact = torch.nan_to_num(contact, nan=0.0, posinf=0.0, neginf=0.0)
        released = getattr(
            env, "_bulb_released",
            torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        )
        gate = released & ~rotate_bulb_failed(env)
        return potential * contact * gate.float()


class RotateBulbFailurePenalty(ManagerTermBase):
    """Charge a single physical failure, even when drop and bounds overlap."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.paid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env):
        failed = rotate_bulb_failed(env)
        charge = failed & ~self.paid
        self.paid |= failed
        return charge.float() / env.step_dt


class RotateBulbStableSuccess(ManagerTermBase):
    """Reward/terminate after the object reaches the goal and stays there."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hold_time = torch.zeros(env.num_envs, device=env.device)
        self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.hold_time[ids] = 0.0
        self.paid[ids] = False

    def __call__(
        self, env, command_name: str, asset_cfg: SceneEntityCfg,
        align_asset_cfg: SceneEntityCfg, pos_std: float, rot_std: float | None = None,
        hold_duration: float = 0.3, min_clearance: float = 0.05,
        max_linear_speed: float = 0.05, max_angular_speed: float = 0.5,
        once: bool = False,
        as_termination: bool = False,
    ):
        # Keep the inherited arguments for config compatibility.  The relaxed
        # v2-test-style criterion uses post-release position only; contact,
        # clearance, and velocity are deliberately not hard gates.  Requiring
        # physical release prevents the attached socket state from counting
        # as a transport success.
        asset = env.scene[asset_cfg.name]
        bulb = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        goal, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3])
        close = torch.linalg.vector_norm(bulb.data.root_pos_w - goal, dim=-1) < pos_std
        released = getattr(
            env, "_bulb_released",
            torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        )
        valid = close & released
        if once or as_termination:
            valid &= ~rotate_bulb_failed(env)
        self.hold_time = torch.where(valid, self.hold_time + env.step_dt, 0.0)
        env._bulb_success = valid & (self.hold_time + 1e-6 >= hold_duration)
        if as_termination:
            return env._bulb_success
        if once:
            credit = env._bulb_success & ~self.paid
            self.paid |= env._bulb_success
            return credit.float() / env.step_dt
        return env._bulb_success.float()


class RotateBulbSuccessTermination(RotateBulbStableSuccess):
    """Use the reward manager's final parameters, including late Hydra overrides."""

    def __call__(self, env):
        params = env.reward_manager.get_term_cfg("success").params
        return super().__call__(env, **{**params, "once": False, "as_termination": True})


def rotate_bulb_orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Orientation tracking reward gated by Revo3 hand contact."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import quat_error_magnitude, quat_mul

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    quat_error = quat_error_magnitude(obj.data.root_quat_w, des_quat_w)
    return (1 - torch.tanh(quat_error / std)) * rotate_bulb_hand_contacts(env, 1.0).float()


@configclass
class DexsuiteRevo3RotateBulbRewardCfg:
    """Phase budgets with dense carry tracking and a position-only success hold.

    Event/progress terms divide by step_dt; reward-rate terms integrate seconds.
    Each new turn pays 0.5 plus 1.5 times grasp quality (at most 6 for three turns).
    Any-finger contact adds at most 0.1 per second (2 over the default 20 seconds).
    """

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.0001)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.0002)

    # Keep the descriptor for the shared Revo3 mixin's fingertip selection.
    # Leave the legacy proximity and opposing-contact occupancy terms disabled.
    fingers_to_object = RewTerm(
        func=rotate_bulb_reach_shell, params={"std": 0.2, "asset_cfg": SceneEntityCfg("robot")}, weight=0.0,
    )
    fingers_to_object_delta = RewTerm(
        func=RotateBulbReachProgress, params={"std": 0.2, "asset_cfg": SceneEntityCfg("robot")}, weight=1.0,
    )

    good_finger_contact = RewTerm(
        func=rotate_bulb_hand_contacts,
        weight=0.0,
        params={"threshold": 1.0},
    )

    # Persistent auxiliary guidance, independent of the bounded grasp reward.
    # RewardManager multiplies by step_dt: 0.02 per finger, 0.1 per second total.
    any_finger_contact = RewTerm(
        func=rotate_bulb_any_finger_contact, weight=0.1, params={"force_scale": 0.5},
    )

    grasp_shaping = RewTerm(
        func=RotateBulbGraspBudget, weight=1.0,
        params={"force_scale": 1.0, "reward_rate": 0.25, "max_reward": 1.0},
    )

    unscrew_turns = RewTerm(func=RotateBulbPhysicalTurnProgress, weight=0.5)

    unscrew_progress = RewTerm(
        func=RotateBulbUnscrewProgress, weight=1.5,
        params={"contact_threshold": 1.0, "completion_bonus": 0.0, "regrasp_grace_s": 0.2},
    )

    released_stability = RewTerm(
        func=RotateBulbReleasedStability, weight=1.0, params={"reward_rate": 0.5, "max_reward": 2.0},
    )
    release_grasp = RewTerm(func=RotateBulbReleaseGrasp, weight=2.0, params={"hold_duration": 0.3})

    position_tracking = RewTerm(
        func=RotateBulbTransportProgress,
        # Dense v2-test-style tracking. Weight 0.5 caps the integrated
        # carry shaping near ten points over a 20 s episode while retaining
        # a nonzero distance gradient every step.
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )
    position_tracking_fine = None

    orientation_tracking = None

    success = RewTerm(
        func=RotateBulbStableSuccess,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.03,
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
            "once": True,
            "hold_duration": 1.0,
        },
    )

    failure = RewTerm(func=RotateBulbFailurePenalty, weight=-6.0)


@configclass
class DexsuiteRevo3RotateBulbEnvCfg(Revo3MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Configuration for the Revo3 rotate-bulb environment (training)."""

    scene: DexsuiteRevo3RotateBulbSceneCfg = DexsuiteRevo3RotateBulbSceneCfg(
        num_envs=32,
        env_spacing=3,
        replicate_physics=False,
    )
    rewards: DexsuiteRevo3RotateBulbRewardCfg = DexsuiteRevo3RotateBulbRewardCfg()
    # Three turns place the bulb tip about 17.69 mm inside the socket mouth.
    initial_screw_turns_range: tuple[float, float] = (3.0, 3.0)
    task_episode_length_s: float = 20.0
    # Read by the observation term after Hydra overrides, before loading any model.
    tactile_policy_enabled: bool = False
    tactile_policy_checkpoint: str = _PRETRAINED_TACTILE_CHECKPOINT
    tactile_encoder_chunk_size: int = 256
    tactile_taxim_chunk_size: int = 32
    tactile_reconstruction_diagnostics: bool = False

    def __post_init__(self):
        super().__post_init__()
        configure_rotate_bulb_scene(self)


@configclass
class DexsuiteRevo3RotateBulbEnvCfg_PLAY(Revo3MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Configuration for the Revo3 rotate-bulb environment (evaluation/play)."""

    scene: DexsuiteRevo3RotateBulbSceneCfg = DexsuiteRevo3RotateBulbSceneCfg(
        num_envs=32,
        env_spacing=3,
        replicate_physics=False,
    )
    rewards: DexsuiteRevo3RotateBulbRewardCfg = DexsuiteRevo3RotateBulbRewardCfg()
    # Match the training task's initial engagement and available time.
    initial_screw_turns_range: tuple[float, float] = (3.0, 3.0)
    task_episode_length_s: float = 20.0
    # Must match the setting used to train the policy being evaluated.
    tactile_policy_enabled: bool = False
    tactile_policy_checkpoint: str = _PRETRAINED_TACTILE_CHECKPOINT
    tactile_encoder_chunk_size: int = 256
    tactile_taxim_chunk_size: int = 32
    tactile_reconstruction_diagnostics: bool = False

    def __post_init__(self):
        super().__post_init__()
        configure_rotate_bulb_scene(self)
