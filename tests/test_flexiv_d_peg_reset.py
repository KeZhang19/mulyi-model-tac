"""CPU behaviour checks for the Flexiv D-peg calibration/reset contract."""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/InsertDPegCustom/env_cfg.py"


def _quat_mul(a, b):
    scalar = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    vector = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return torch.cat((scalar, vector), dim=-1)


@pytest.fixture
def task_code():
    tree = ast.parse(CONFIG.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"load_d_peg_pregrasp", "reset_d_peg_pregrasp", "_d_peg_in_bound_range"}]
    ns = dict(torch=torch, json=json, math=math, Path=Path, warnings=__import__("warnings"),
              _SOURCE_TO_CUSTOM={}, _JOINT_NAMES=JOINTS,
              quat_mul=_quat_mul,
              quat_from_euler_xyz=lambda r, p, y: torch.stack((torch.cos(y / 2), torch.zeros_like(y),
                                                               torch.zeros_like(y), torch.sin(y / 2)), -1))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CONFIG), "exec"), ns)
    return ns


JOINTS = tuple(
    [f"joint{i}" for i in range(1, 8)]
    + [f"right_{finger}_{part}_joint" for finger in ("thumb", "index", "middle", "ring", "little")
       for part in (("CMP", "CMR", "MCP", "PIP", "DIP") if finger == "thumb" else ("MPR", "MCP", "PIP", "DIP"))]
)


def _calibration():
    values = {name: index / 100 for index, name in enumerate(JOINTS)}
    return dict(schema_version=1, robot_joint_positions=values,
                robot_joint_targets=dict(values),
                peg_pose={"position": [-.72, .04, .07031], "quaternion": [1., 0., 0., 0.]},
                socket_pose={"position": [-.72, .04, -.00969], "quaternion": [math.sqrt(.5), math.sqrt(.5), 0., 0.]},
                validation={"status": "editor_candidate", "physics_validated": False})


class _Body:
    def __init__(self, position, quaternion=(1., 0., 0., 0.)):
        state = torch.zeros((3, 13))
        state[:, :3] = torch.tensor(position)
        state[:, 3:7] = torch.tensor(quaternion)
        self.data = SimpleNamespace(default_root_state=state.clone(), root_pos_w=state[:, :3], root_quat_w=state[:, 3:7])
        self.velocity = state[:, 7:]

    def write_root_pose_to_sim(self, pose, env_ids):
        self.data.root_pos_w[env_ids] = pose[:, :3]
        self.data.root_quat_w[env_ids] = pose[:, 3:]

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.velocity[env_ids] = velocity


class _Robot(_Body):
    def __init__(self, names):
        super().__init__([0., 0., 0.])
        self.joint_names = list(names)
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
    def __init__(self, values):
        super().__init__(values)
        self.env_origins = values["env_origins"]


def test_loader_warns_for_editor_candidate_without_changing_pose(task_code, tmp_path):
    source = _calibration()
    path = tmp_path / "pregrasp.json"
    path.write_text(json.dumps(source))
    with pytest.warns(UserWarning, match="editor candidate"):
        loaded = task_code["load_d_peg_pregrasp"](path)
    assert loaded["peg_pose"] == source["peg_pose"]
    assert loaded["socket_pose"] == source["socket_pose"]


def test_bounds_enclose_calibrated_peg_socket_and_table(task_code):
    bounds = task_code["_d_peg_in_bound_range"]([-.72, .04, .07031], [-.72, .04, -.00969], -.00969)
    assert bounds["x"] == pytest.approx((-1.07, -.37))
    assert bounds["y"] == pytest.approx((-.41, .49))
    assert bounds["z"] == pytest.approx((-.15969, .57031))
    target_tip = (-.72, .04, .01231)  # socket mouth (-.00969 + .05) minus .028 depth
    for point in ((-.72, .04, .07031), (-.72, .04, -.00969), target_tip):
        assert all(bounds[axis][0] <= point[i] <= bounds[axis][1] for i, axis in enumerate(("x", "y", "z")))
    out_of_bounds = (bounds["x"][0] - 1e-3, target_tip[1], target_tip[2])
    assert not all(bounds[axis][0] <= out_of_bounds[i] <= bounds[axis][1]
                   for i, axis in enumerate(("x", "y", "z")))


def test_partial_reset_preserves_other_envs_and_composes_world_yaw(task_code):
    calibration = _calibration()
    robot = _Robot(reversed(JOINTS))
    socket = _Body([-.72, .04, -.00969], (math.sqrt(.5), math.sqrt(.5), 0., 0.))
    scene = {"robot": robot, "object": _Body([0., 0., 0.]), "socket": socket,
             "env_origins": torch.tensor([[0., 0., 0.], [3., 3., 0.], [6., 6., 0.]])}
    env = SimpleNamespace(num_envs=3, device="cpu", scene=_Scene(scene),
                          cfg=SimpleNamespace(_d_peg_pregrasp=calibration,
                                              socket_xy_randomization_m=0., socket_yaw_randomization_deg=10.))
    original = socket.data.root_quat_w.clone()
    torch.manual_seed(7)
    env.cfg.socket_yaw_randomization_deg = 0.
    task_code["reset_d_peg_pregrasp"](env, [1])
    # Zero randomization must restore the calibrated socket orientation exactly.
    assert torch.allclose(socket.data.root_quat_w[1], original[1], atol=1e-7, rtol=0)
    assert torch.allclose(socket.data.root_quat_w[0], original[0], atol=1e-7, rtol=0)

    env.cfg.socket_yaw_randomization_deg = 10.
    task_code["reset_d_peg_pregrasp"](env, [1])
    assert torch.all(robot.positions[[0, 2]] == -1)
    assert torch.all(robot.positions[1] == torch.tensor([calibration["robot_joint_positions"][n] for n in robot.joint_names]))
    assert torch.all(robot.targets[[0, 2]] == -1)
    assert torch.allclose(robot.targets[1], robot.positions[1])
    assert torch.all(robot.velocities[1] == 0)
    assert torch.allclose(scene["object"].data.root_pos_w[1], torch.tensor([2.28, 3.04, .07031]))
    assert torch.allclose(socket.data.root_pos_w[1], torch.tensor([2.28, 3.04, -.00969]))
    assert torch.allclose(socket.data.root_quat_w[0], original[0])
    assert torch.allclose(socket.data.root_quat_w[1].norm(), torch.tensor(1.), atol=1e-6)
    # q_new * conjugate(q_default) is the world-frame yaw perturbation.
    relative = _quat_mul(socket.data.root_quat_w[1:2],
                         torch.cat((original[1:2, :1], -original[1:2, 1:]), dim=-1))[0]
    assert abs(float(relative[1])) < 1e-6 and abs(float(relative[2])) < 1e-6
    assert abs(2 * math.atan2(abs(float(relative[3])), float(relative[0]))) <= math.radians(10) + 1e-6
