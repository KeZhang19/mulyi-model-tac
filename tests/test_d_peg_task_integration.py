"""CPU validation of insertion calibration, reset isolation, targets and run contracts."""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from BrainCo_DexHand.tactile_representation.policy import file_sha256, validate_policy_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/Revo3"
CONFIG = CONFIG_ROOT / "dexsuite_revo3_env_cfg_insert_d_peg.py"
TACTILE = CONFIG_ROOT.parent.parent / "mdp/d_peg_tactile.py"


def _quat_multiply(a, b):
    scalar = a[:, :1] * b[:, :1] - (a[:, 1:] * b[:, 1:]).sum(-1, keepdim=True)
    vector = a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] + torch.cross(a[:, 1:], b[:, 1:], dim=-1)
    return torch.cat((scalar, vector), dim=-1)


def _rotate(q, v):
    return v + 2 * torch.cross(q[:, 1:], torch.cross(q[:, 1:], v, dim=-1) + q[:, :1] * v, dim=-1)


def _combine(p, q, offset):
    return p + _rotate(q, offset), q


def _subtract(p, q, target_p, target_q):
    inverse = q * torch.tensor([1., -1., -1., -1.])
    return _rotate(inverse, target_p - p), _quat_multiply(inverse, target_q)


def _yaw_quaternion(roll, pitch, yaw):
    return torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)


@pytest.fixture
def task_code():
    class Command:
        def __init__(self, cfg, env):
            self.cfg, self._env, self.device, self.metrics = cfg, env, env.device, {}

    tree = ast.parse(CONFIG.read_text())
    selected = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name in {"load_d_peg_pregrasp", "reset_d_peg_pregrasp"}
        or isinstance(node, ast.ClassDef) and node.name == "DPegInsertionPoseCommand"
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_JOINT_NAMES" for t in node.targets)
    )]
    ns = dict(torch=torch, json=json, math=math, Path=Path, CommandTerm=Command,
              combine_frame_transforms=_combine, subtract_frame_transforms=_subtract,
              quat_from_euler_xyz=_yaw_quaternion, d_peg_success=lambda env: torch.zeros(env.num_envs, dtype=torch.bool))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CONFIG), "exec"), ns)
    return ns


def _calibration(task_code):
    joints = {name: index / 100 for index, name in enumerate(task_code["_JOINT_NAMES"])}
    return dict(schema_version=1, robot_joint_positions=joints,
                robot_joint_targets={name: value + .02 for name, value in joints.items()},
                peg_pose=dict(position=[.45, .10, .84], quaternion=[1., 0., 0., 0.]))


def test_pregrasp_requires_complete_calibration_and_preserves_preload(task_code, tmp_path):
    source = _calibration(task_code)
    path = tmp_path / "pregrasp.json"
    path.write_text(json.dumps(source))
    assert task_code["load_d_peg_pregrasp"](path) == source
    source.pop("robot_joint_targets")
    path.write_text(json.dumps(source))
    loaded = task_code["load_d_peg_pregrasp"](path)
    assert loaded["robot_joint_targets"] == source["robot_joint_positions"]
    assert loaded["robot_joint_targets"] is not loaded["robot_joint_positions"]
    del source["robot_joint_positions"]["Joint7_R"]
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="28 finite"):
        task_code["load_d_peg_pregrasp"](path)
    with pytest.raises(FileNotFoundError):
        task_code["load_d_peg_pregrasp"](tmp_path / "missing.json")


@pytest.mark.parametrize("bad_pose", [dict(position=[0., 0., .84], quaternion=[2., 0., 0., 0.]),
                                      dict(position=[float("nan"), 0., .84], quaternion=[1., 0., 0., 0.])])
def test_pregrasp_rejects_nonphysical_pose(task_code, tmp_path, bad_pose):
    source = _calibration(task_code)
    source["peg_pose"] = bad_pose
    path = tmp_path / "pregrasp.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        task_code["load_d_peg_pregrasp"](path)


class _Body:
    def __init__(self, position):
        state = torch.zeros((3, 13))
        state[:, :3] = torch.tensor(position)
        state[:, 3] = 1.0
        self.data = SimpleNamespace(default_root_state=state.clone(), root_pos_w=state[:, :3], root_quat_w=state[:, 3:7])
        self.velocity = state[:, 7:]

    def write_root_pose_to_sim(self, pose, env_ids):
        self.data.root_pos_w[env_ids] = pose[:, :3]
        self.data.root_quat_w[env_ids] = pose[:, 3:]

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.velocity[env_ids] = velocity


class _Robot(_Body):
    def __init__(self, names):
        super().__init__([1., 0., .766])
        self.joint_names = names
        self.positions = torch.full((3, 28), -1.)
        self.targets = self.positions.clone()
        self.velocities = self.positions.clone()
        self.velocity_targets = self.positions.clone()
        self.efforts = self.positions.clone()

    def write_joint_state_to_sim(self, positions, velocities, env_ids):
        self.positions[env_ids], self.velocities[env_ids] = positions, velocities

    def set_joint_position_target(self, values, env_ids):
        self.targets[env_ids] = values

    def set_joint_velocity_target(self, values, env_ids):
        self.velocity_targets[env_ids] = values

    def set_joint_effort_target(self, values, env_ids):
        self.efforts[env_ids] = values


class _Scene(dict):
    pass


def test_partial_reset_preserves_other_environments_and_writes_preload(task_code):
    calibration = _calibration(task_code)
    scene = _Scene(robot=_Robot(list(reversed(task_code["_JOINT_NAMES"]))),
                   object=_Body([-.1, -.2, .3]), socket=_Body([.45, .10, .76]))
    scene.env_origins = torch.tensor([[0., 0., 0.], [3., 3., 0.], [6., 6., 0.]])
    env = SimpleNamespace(num_envs=3, device="cpu", scene=scene,
                          cfg=SimpleNamespace(_d_peg_pregrasp=calibration,
                                              socket_xy_randomization_m=.005, socket_yaw_randomization_deg=10.))
    torch.manual_seed(11)
    task_code["reset_d_peg_pregrasp"](env, [1])
    robot = scene["robot"]
    torch.testing.assert_close(robot.positions[[0, 2]], torch.full((2, 28), -1.))
    torch.testing.assert_close(robot.targets[[0, 2]], torch.full((2, 28), -1.))
    torch.testing.assert_close(robot.positions[1], torch.tensor([calibration["robot_joint_positions"][n] for n in robot.joint_names]))
    torch.testing.assert_close(robot.targets[1], robot.positions[1] + .02)
    torch.testing.assert_close(robot.velocities[1], torch.zeros(28))
    torch.testing.assert_close(scene["object"].data.root_pos_w[1], torch.tensor([3.45, 3.10, .84]))
    torch.testing.assert_close(scene["object"].data.root_pos_w[[0, 2]], torch.tensor([[-.1, -.2, .3]]).repeat(2, 1))
    socket_local = scene["socket"].data.root_pos_w[1] - scene.env_origins[1]
    assert torch.all((socket_local[:2] - torch.tensor([.45, .10])).abs() <= .005)
    assert float(socket_local[2]) == pytest.approx(.76)
    assert abs(float(scene["socket"].data.root_quat_w[1, 3])) <= math.sin(math.radians(5))


def test_command_uses_socket_orientation_final_tip_and_robot_frame(task_code):
    scene = _Scene(robot=_Robot(task_code["_JOINT_NAMES"]), socket=_Body([.45, .10, .76]))
    robot_q = torch.tensor([math.sqrt(.5), 0., 0., -math.sqrt(.5)])
    scene["robot"].data.root_quat_w[:] = robot_q
    yaw = torch.tensor([0., math.radians(10), -math.radians(10)])
    scene["socket"].data.root_quat_w[:] = _yaw_quaternion(yaw * 0, yaw * 0, yaw)
    env = SimpleNamespace(scene=scene, num_envs=3, device="cpu",
                          cfg=SimpleNamespace(socket_mouth_height_m=.05, insertion_depth_m=.028))
    command = task_code["DPegInsertionPoseCommand"](None, env)
    command._resample_command([0, 1, 2])
    world_p = scene["robot"].data.root_pos_w + _rotate(scene["robot"].data.root_quat_w, command.command[:, :3])
    world_q = _quat_multiply(scene["robot"].data.root_quat_w, command.command[:, 3:])
    torch.testing.assert_close(world_p, torch.tensor([[.45, .10, .782]]).repeat(3, 1), atol=1e-6, rtol=0)
    torch.testing.assert_close(world_q, scene["socket"].data.root_quat_w)


def test_contract_rejects_bulb_and_changed_task_assets(tmp_path):
    tree = ast.parse(TACTILE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    legacy = dict(task="BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0", observation_dim=1949,
                  projection_dim=256, history_length=1,
                  observation_groups={group: [dict(name="group", shape=[size])]
                                      for group, size in {"policy": 43, "proprio": 1714, "perception": 192}.items()})
    def initialize(env, env_ids=None):
        env.tactile_policy_contract = dict(legacy)
    ns = dict(initialize_rotate_bulb_tactile_contract=initialize, file_sha256=file_sha256, Path=Path,
              math=math, asdict=lambda value: value,
              validate_d_peg_geometry_metadata=lambda *_: {"insertion_depth_m": .028})
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TACTILE), "exec"), ns)
    asset = tmp_path / "asset"
    asset.write_bytes(b"first geometry")
    cfg = SimpleNamespace(peg_usd_path=asset, socket_usd_path=asset, geometry_metadata_path=asset,
                          pregrasp_path=asset, insertion_depth_m=.028, socket_mouth_height_m=.05,
                          socket_xy_randomization_m=.005, socket_yaw_randomization_deg=10.,
                          episode_length_s=30., actions=SimpleNamespace(action=SimpleNamespace(scale=.1)))
    action_schema = dict(schema_version=2, type="pregrasp_target_residual_position", scale=.1)
    env = SimpleNamespace(cfg=cfg, action_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(action_contract=lambda: dict(action_schema)),
    ))
    ns[function.name](env)
    first = env.tactile_policy_contract
    assert first["observation_dim"] == 1949
    assert first["task_schema_version"] == 3
    with pytest.raises(ValueError, match="task_schema_version"):
        validate_policy_contract(first, dict(first, task_schema_version=2))
    previous_d_peg = dict(first, task_schema_version=1)
    previous_d_peg.pop("action_schema")
    with pytest.raises(ValueError, match="task_schema_version|action_schema"):
        validate_policy_contract(first, previous_d_peg)
    changed_actions = dict(first, action_schema=dict(action_schema, type="current_joint_relative_position"))
    with pytest.raises(ValueError, match="action_schema"):
        validate_policy_contract(first, changed_actions)
    with pytest.raises(ValueError, match="task"):
        validate_policy_contract(first, legacy)
    asset.write_bytes(b"changed geometry")
    ns[function.name](env)
    with pytest.raises(ValueError, match="task_assets"):
        validate_policy_contract(env.tactile_policy_contract, first)
    legacy["projection_dim"] = 64
    with pytest.raises(ValueError, match="256 features"):
        ns[function.name](env)
