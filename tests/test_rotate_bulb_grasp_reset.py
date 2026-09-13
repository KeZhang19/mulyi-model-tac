"""Exercise the saved-grasp scene configuration and resets without Isaac Sim."""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ("source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/"
                 "dexsuite/config/RotateBulbCustom/env_cfg.py")
REFERENCE = json.loads((ROOT / "assets/rotate_bulb_custom/grasp_reference/simulation_state.json").read_text())
SUPPORT = next(item for item in REFERENCE["fixtures"] if item["name"] == "lamp_support")


def _rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[..., [1, 2, 3, 0]])


def _combine_frames(pos, quat, offset, child_quat):
    """Independent CPU substitute for Isaac's transform utility, using SciPy."""
    rotation = _rotation(quat)
    out_pos = np.asarray(pos) + rotation.apply(np.asarray(offset))
    out_quat = (rotation * _rotation(child_quat)).as_quat()[..., [3, 0, 1, 2]]
    return torch.as_tensor(out_pos, dtype=pos.dtype), torch.as_tensor(out_quat, dtype=quat.dtype)


def _euler_quat(roll, pitch, yaw):
    angles = np.stack((roll, pitch, yaw), axis=-1)
    return torch.as_tensor(Rotation.from_euler("xyz", angles).as_quat()[..., [3, 0, 1, 2]], dtype=roll.dtype)


class _ResetTerm(SimpleNamespace):
    def replace(self, **kwargs):
        result = copy.deepcopy(self)
        result.__dict__.update(kwargs)
        return result


@pytest.fixture
def reset_code():
    tree = ast.parse(CONFIG.read_text())
    constants = {"_SUPPORT_BOTTOM_OFFSET", "_SCREW_PITCH", "_SCREW_TRAVEL", "_SCREW_EXIT_EXTENSION"}
    definitions = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants
                                           for target in node.targets)
        or isinstance(node, ast.FunctionDef) and node.name in {
            "reset_lamp_on_table", "reset_robot_to_grasp", "configure_rotate_bulb_scene"}
    )]
    namespace = dict(torch=torch, math=math, ManagerBasedRLEnv=object,
                     combine_frame_transforms=_combine_frames, quat_from_euler_xyz=_euler_quat,
                     GRASP_REFERENCE=REFERENCE, EventTerm=_ResetTerm, DoneTerm=_ResetTerm, ObsTerm=_ResetTerm,
                     sim_utils=SimpleNamespace(PreviewSurfaceCfg=_ResetTerm))
    for name in ("release_unscrewed_bulb", "RotateBulbPoseCommand", "rotate_bulb_dropped",
                 "RotateBulbSuccessTermination",
                 "rotate_bulb_screw_progress", "rotate_bulb_phase_state", "RotateBulbPretrainedTactile",
                 "initialize_rotate_bulb_tactile_contract"):
        namespace[name] = object()
    # Only simulator integration types/imports are replaced. Execute the real
    # task reset and configuration bodies, with their actual screw constants.
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(CONFIG), "exec"), namespace)
    return SimpleNamespace(**namespace)


class _ResetAsset:
    def __init__(self, count):
        state = torch.arange(count * 13, dtype=torch.float64).reshape(count, 13) / 10 + 2
        state[:, 3:7] = torch.tensor([1., 0., 0., 0.])
        self.data = SimpleNamespace(root_state_w=state, root_pos_w=state[:, :3], root_quat_w=state[:, 3:7],
                                    default_root_state=state.clone())
        self.data.default_root_state[:, 3:7] = torch.tensor(SUPPORT["quaternion_wxyz"])
        # Reverse the USD joint order to catch accidental position-based indexing.
        self.joint_names = ["screw_slide", "screw_turn"]
        self.data.default_joint_pos = torch.tensor([[-0.024, -8 * math.pi]], dtype=torch.float64).repeat(count, 1)
        self.data.joint_pos = torch.full((count, 2), 7.0, dtype=torch.float64)
        self.data.joint_vel = torch.full((count, 2), 8.0, dtype=torch.float64)
        self.position_target = torch.full((count, 2), 9.0, dtype=torch.float64)
        self.velocity_target = torch.full((count, 2), 10.0, dtype=torch.float64)
        self.effort_target = torch.full((count, 2), 11.0, dtype=torch.float64)
        self.reset_ids = []

    def write_root_pose_to_sim(self, pose, *, env_ids):
        self.data.root_state_w[env_ids, :7] = pose

    def write_root_velocity_to_sim(self, velocity, *, env_ids):
        self.data.root_state_w[env_ids, 7:] = velocity

    def reset(self, *, env_ids):
        self.reset_ids.append(env_ids.tolist())

    def write_joint_state_to_sim(self, position, velocity, *, env_ids):
        self.data.joint_pos[env_ids] = position
        self.data.joint_vel[env_ids] = velocity

    def set_joint_position_target(self, target, *, env_ids):
        self.position_target[env_ids] = target

    def set_joint_velocity_target(self, target, *, env_ids):
        self.velocity_target[env_ids] = target

    def set_joint_effort_target(self, target, *, env_ids):
        self.effort_target[env_ids] = target


class _ResetScene(dict):
    pass


class _Attachment:
    def __init__(self):
        self.enabled = False
        self.calls = []

    def Set(self, value):
        self.enabled = value
        self.calls.append(value)


def _grasp_reset_env():
    count = 4
    scene = _ResetScene({name: _ResetAsset(count) for name in ("lamp", "support", "object", "table", "robot")})
    scene.env_origins = torch.tensor([[3., -6., .2], [-3., 9., -.3], [12., 6., 1.], [6., -9., 0.]],
                                     dtype=torch.float64)
    table = scene["table"].data.root_state_w
    table[:, :3] = scene.env_origins + torch.tensor(REFERENCE["table"]["position_m"], dtype=torch.float64)
    table[:, 3:7] = torch.tensor(REFERENCE["table"]["quaternion_wxyz"], dtype=torch.float64)
    robot = scene["robot"]
    robot.joint_names = list(reversed(REFERENCE["robot"]["joint_names"]))
    robot.data.default_joint_pos = torch.tensor([list(reversed(REFERENCE["robot"]["joint_positions_rad"]))],
                                                dtype=torch.float64).repeat(count, 1)
    base = REFERENCE["robot"]["base"]
    robot.data.default_root_state[:, :7] = torch.tensor(base["position_m"] + base["quaternion_wxyz"],
                                                       dtype=torch.float64)
    for field in ("joint_pos", "joint_vel"):
        setattr(robot.data, field, torch.full_like(robot.data.default_joint_pos, 15.))
    for field in ("position_target", "velocity_target", "effort_target"):
        setattr(robot, field, torch.full_like(robot.data.default_joint_pos, 16.))
    attributes = [_Attachment() for _ in range(count)]

    def get_prim(path):
        index = int(path.split("/env_")[1].split("/")[0])

        def get_attribute(name):
            assert name == "physics:jointEnabled"
            return attributes[index]

        return SimpleNamespace(GetAttribute=get_attribute)

    return SimpleNamespace(scene=scene, num_envs=count, device="cpu",
                           cfg=SimpleNamespace(initial_screw_turns_range=(3., 3.)),
                           sim=SimpleNamespace(stage=SimpleNamespace(GetPrimAtPath=get_prim)),
                           attachments=attributes)


def _saved_reset_kwargs():
    return dict(x_range=(0., 0.), y_range=(0., 0.), table_half_height=REFERENCE["table"]["size_m"][2] / 2,
                support_pose=tuple(SUPPORT["position_m"] + SUPPORT["quaternion_wxyz"]))


def _assert_saved_pose(asset, reference, origins, ids):
    expected = origins[ids] + torch.tensor(reference["position_m"], dtype=torch.float64)
    torch.testing.assert_close(asset.data.root_state_w[ids, :3], expected, rtol=0, atol=1e-7)
    # q and -q denote the same orientation; compare physical angular error.
    error = (_rotation(asset.data.root_state_w[ids, 3:7]).inv()
             * _rotation(reference["quaternion_wxyz"])).magnitude()
    assert np.max(error) < 2e-6


@pytest.mark.parametrize("selected", [None, [3, 1]])
def test_saved_grasp_reset_restores_selected_environments(reset_code, selected):
    env = _grasp_reset_env()
    ids = list(range(env.num_envs)) if selected is None else selected
    untouched = [i for i in range(env.num_envs) if i not in ids]
    if selected is not None:
        # Simulate a partly completed episode before resetting just two envs.
        env._bulb_released = torch.ones(env.num_envs, dtype=torch.bool)
        env._bulb_success = torch.ones(env.num_envs, dtype=torch.bool)
        env._bulb_start_joints = torch.full((env.num_envs, 2), 17., dtype=torch.float64)
        env._bulb_attachment_attrs = env.attachments
    previous = copy.deepcopy(env)
    reset_code.reset_lamp_on_table(env, selected, **_saved_reset_kwargs())
    for name, reference in (("lamp", SUPPORT), ("support", SUPPORT), ("object", REFERENCE["object"])):
        asset = env.scene[name]
        _assert_saved_pose(asset, reference, env.scene.env_origins, ids)
        assert not asset.data.root_state_w[ids, 7:].any()
        torch.testing.assert_close(asset.data.root_state_w[untouched], previous.scene[name].data.root_state_w[untouched])
    torch.testing.assert_close(env.scene["table"].data.root_state_w, previous.scene["table"].data.root_state_w)
    lamp = env.scene["lamp"]
    source = REFERENCE["assembly"]["source_configuration"]
    expected = torch.tensor([[source["screw_slide_m"], source["screw_turn_rad"]]], dtype=torch.float64).repeat(len(ids), 1)
    torch.testing.assert_close(lamp.data.joint_pos[ids], expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(lamp.position_target[ids], expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(env._bulb_start_joints[ids], expected, rtol=0, atol=1e-6)
    assert lamp.reset_ids == [ids]
    for field in ("joint_pos", "joint_vel"):
        torch.testing.assert_close(getattr(lamp.data, field)[untouched], getattr(previous.scene["lamp"].data, field)[untouched])
    for field in ("position_target", "velocity_target", "effort_target"):
        torch.testing.assert_close(getattr(lamp, field)[untouched], getattr(previous.scene["lamp"], field)[untouched])
    assert not lamp.data.joint_vel[ids].any()
    assert not lamp.velocity_target[ids].any()
    assert not lamp.effort_target[ids].any()
    assert not env._bulb_released[ids].any()
    assert not env._bulb_success[ids].any()
    for index, attachment in enumerate(env.attachments):
        assert attachment.enabled == (index in ids)
        assert attachment.calls == ([True] if index in ids else [])
    if untouched:
        assert env._bulb_released[untouched].all()
        assert env._bulb_success[untouched].all()
        torch.testing.assert_close(env._bulb_start_joints[untouched], previous._bulb_start_joints[untouched])


def test_grasp_reset_legacy_sampling_uses_rotated_table_frame(reset_code):
    env = _grasp_reset_env()
    args = _saved_reset_kwargs()
    args.pop("support_pose")
    args.update(x_range=(.12, .12), y_range=(-.08, -.08))
    reset_code.reset_lamp_on_table(env, [2], **args)
    table_rotation = _rotation(REFERENCE["table"]["quaternion_wxyz"])
    expected = (np.asarray(REFERENCE["table"]["position_m"])
                + table_rotation.apply([.12, -.08, .38 + reset_code._SUPPORT_BOTTOM_OFFSET])
                + np.asarray(env.scene.env_origins[2]))
    np.testing.assert_allclose(env.scene["support"].data.root_pos_w[2], expected, rtol=0, atol=1e-7)
    rotation = _rotation(env.scene["support"].data.root_quat_w[2])
    expected_rotation = table_rotation * _rotation(SUPPORT["quaternion_wxyz"])
    assert (rotation.inv() * expected_rotation).magnitude() < 1e-7


@pytest.mark.parametrize("depth", [(0., 3.), (-1., 1.), (3., 2.), (3., 4.01)])
def test_grasp_reset_rejects_invalid_screw_depth_before_writing(reset_code, depth):
    env = _grasp_reset_env()
    env.cfg.initial_screw_turns_range = depth
    before = copy.deepcopy(env)
    with pytest.raises(ValueError, match="initial_screw_turns_range"):
        reset_code.reset_lamp_on_table(env, None, **_saved_reset_kwargs())
    for name in env.scene:
        torch.testing.assert_close(env.scene[name].data.root_state_w, before.scene[name].data.root_state_w)
    assert not env.scene["lamp"].reset_ids
    assert not any(attribute.calls for attribute in env.attachments)


@pytest.mark.parametrize("selected", [None, [3, 1]])
def test_grasp_robot_reset_clears_previous_episode_targets(reset_code, selected):
    env = _grasp_reset_env()
    robot = env.scene["robot"]
    previous = copy.deepcopy(robot)
    ids = list(range(env.num_envs)) if selected is None else selected
    untouched = [i for i in range(env.num_envs) if i not in ids]
    reset_code.reset_robot_to_grasp(env, selected)
    _assert_saved_pose(robot, REFERENCE["robot"]["base"], env.scene.env_origins, ids)
    assert not robot.data.root_state_w[ids, 7:].any()
    torch.testing.assert_close(robot.data.joint_pos[ids], robot.data.default_joint_pos[ids])
    torch.testing.assert_close(robot.position_target[ids], robot.data.default_joint_pos[ids])
    assert not robot.data.joint_vel[ids].any()
    assert not robot.velocity_target[ids].any()
    assert not robot.effort_target[ids].any()
    for field in ("root_state_w", "joint_pos", "joint_vel"):
        torch.testing.assert_close(getattr(robot.data, field)[untouched], getattr(previous.data, field)[untouched])
    for field in ("position_target", "velocity_target", "effort_target"):
        torch.testing.assert_close(getattr(robot, field)[untouched], getattr(previous, field)[untouched])


def _reset_config(initial_turns):
    def asset():
        return SimpleNamespace(spawn=SimpleNamespace(size=(1., 1., 1.)), init_state=SimpleNamespace())

    sensor = SimpleNamespace(filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                             target_mesh_prim_path="{ENV_REGEX_NS}/Object",
                             mesh_prim_paths=[SimpleNamespace(prim_expr="{ENV_REGEX_NS}/Object")])
    events = SimpleNamespace(**{name: _ResetTerm(params={"pose_range": {"x": (-1, 1)},
                                                       "velocity_range": {"z": (0, 1)}})
                                for name in ("reset_table", "reset_root")})
    events.zz_revo3_touch_compliant_materials = _ResetTerm(mode="startup")
    events.zz_revo3_pressure_pad_collision_properties = _ResetTerm(mode="startup")
    return SimpleNamespace(
        initial_screw_turns_range=initial_turns, task_episode_length_s=30.,
        scene=SimpleNamespace(table=asset(), lamp=asset(), plane=asset(), tactile=sensor),
        viewer=SimpleNamespace(), events=events,
        sim=SimpleNamespace(physx=SimpleNamespace()),
        commands=SimpleNamespace(object_pose=SimpleNamespace(ranges=SimpleNamespace())),
        terminations=SimpleNamespace(object_out_of_bound=_ResetTerm(params={})),
        rewards=SimpleNamespace(success=_ResetTerm(params={})),
        observations=SimpleNamespace(policy=SimpleNamespace(), proprio=SimpleNamespace(
            rl_ours_hydroshear=_ResetTerm(params={"component_cfg": {"encoder": "retained"}}))),
    )


@pytest.mark.parametrize("class_name", ["DexsuiteRevo3RotateBulbEnvCfg", "DexsuiteRevo3RotateBulbEnvCfg_PLAY"])
def test_grasp_reset_configuration_restores_reference_at_startup_and_reset(reset_code, class_name):
    tree = ast.parse(CONFIG.read_text())
    config_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    default = next(node.value for node in config_class.body if isinstance(node, ast.AnnAssign)
                   and node.target.id == "initial_screw_turns_range")
    cfg = _reset_config(ast.literal_eval(default))
    reset_code.configure_rotate_bulb_scene(cfg)
    assert cfg.initial_screw_turns_range == (3., 3.)
    assert cfg.scene.table.spawn.size == tuple(REFERENCE["table"]["size_m"])
    assert cfg.scene.table.init_state.pos == tuple(REFERENCE["table"]["position_m"])
    assert cfg.scene.table.init_state.rot == tuple(REFERENCE["table"]["quaternion_wxyz"])
    assert cfg.scene.plane.init_state.pos[2] == pytest.approx(REFERENCE["table"]["position_m"][2] - .38)
    assert cfg.events.initialize_lamp.mode == "startup"
    assert cfg.events.reset_object.mode == "reset"
    assert cfg.events.initialize_robot.mode == "startup"
    assert cfg.events.reset_robot_joints.mode == "reset"
    assert cfg.events.initialize_robot.func is reset_code.reset_robot_to_grasp
    assert cfg.events.reset_robot_joints.func is reset_code.reset_robot_to_grasp
    assert cfg.events.reset_robot_wrist_joint is None
    # Collider instances are prepared during spawning; material callbacks keep
    # their normal startup ordering relative to other material randomization.
    assert cfg.events.zz_revo3_touch_compliant_materials.mode == "startup"
    assert cfg.events.zz_revo3_pressure_pad_collision_properties.mode == "startup"
    for event in (cfg.events.initialize_lamp, cfg.events.reset_object):
        env = _grasp_reset_env()
        env.cfg.initial_screw_turns_range = cfg.initial_screw_turns_range
        event.func(env, None, **event.params)
        _assert_saved_pose(env.scene["support"], SUPPORT, env.scene.env_origins, list(range(env.num_envs)))
        _assert_saved_pose(env.scene["object"], REFERENCE["object"], env.scene.env_origins, list(range(env.num_envs)))
    for name in ("reset_table", "reset_root"):
        assert getattr(cfg.events, name).params["pose_range"] == {}
        assert getattr(cfg.events, name).params["velocity_range"] == {}
    for name in ("randomize_object_scale", "object_physics_material", "object_scale_mass", "variable_gravity"):
        assert getattr(cfg.events, name) is None
    for axis, coordinate in zip("xyz", REFERENCE["object"]["position_m"]):
        lo, hi = cfg.terminations.object_out_of_bound.params["in_bound_range"][axis]
        assert lo < coordinate < hi, "The aligned bulb must not terminate immediately outside old scene bounds"
    assert cfg.scene.tactile.filter_prim_paths_expr == ["{ENV_REGEX_NS}/Lamp/Bulb"]
    assert cfg.scene.tactile.target_mesh_prim_path == "{ENV_REGEX_NS}/Lamp/Bulb/TactileSurface"
    assert cfg.scene.tactile.mesh_prim_paths[0].prim_expr == "{ENV_REGEX_NS}/Lamp/Bulb/TactileSurface"
    assert cfg.observations.proprio.pretrained_tactile.params["component_cfg"] == {"encoder": "retained"}
