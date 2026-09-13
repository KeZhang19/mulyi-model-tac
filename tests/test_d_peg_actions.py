"""CPU checks for the calibrated action anchor, ordering and subset reset semantics."""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/d_peg_actions.py"
PREGRASP = json.loads((ROOT / "assets/d_peg_insertion/pregrasp.json").read_text())


class _PositionAction:
    """Minimal external Isaac affine action API; task methods are loaded unchanged."""

    def __init__(self, cfg, env):
        self.cfg, self.device, self.num_envs = cfg, env.device, env.num_envs
        self._joint_names = env.joint_names
        self.action_dim = len(self._joint_names)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._offset = 0.0

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self.cfg.scale + self._offset


@pytest.fixture
def action_type():
    tree = ast.parse(MODULE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "DPegPregraspResidualAction")
    ns = dict(JointPositionAction=_PositionAction, torch=torch, math=math)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(MODULE), "exec"), ns)
    return ns[cls.name]


def _env(names=None, calibration=None):
    return SimpleNamespace(device="cpu", num_envs=3,
                           joint_names=names or list(reversed(PREGRASP["robot_joint_targets"])),
                           cfg=SimpleNamespace(_d_peg_pregrasp=calibration or PREGRASP))


def test_zero_residual_retains_measured_preload_with_actual_joint_order(action_type):
    env = _env()
    term = action_type(SimpleNamespace(scale=.1, clip=None), env)
    targets = torch.tensor([PREGRASP["robot_joint_targets"][name] for name in env.joint_names])
    measured = torch.tensor([PREGRASP["robot_joint_positions"][name] for name in env.joint_names])
    assert float((targets - measured).abs().max()) > .2
    term.process_actions(torch.zeros(3, 28))
    torch.testing.assert_close(term._processed_actions, targets.expand(3, -1))
    # No measured joint state participates; reference remains exact after many
    # repeated control steps instead of accumulating a squeeze/random walk.
    for _ in range(120):
        term.process_actions(torch.full((3, 28), .15))
    torch.testing.assert_close(term._processed_actions, targets.expand(3, -1) + .015)
    term.process_actions(torch.zeros(3, 28))
    torch.testing.assert_close(term._processed_actions, targets.expand(3, -1))


@pytest.mark.parametrize("ids", [[1], torch.tensor([1]), slice(1, 2)])
def test_subset_action_reset_preserves_running_environments(action_type, ids):
    term = action_type(SimpleNamespace(scale=.1, clip=None), _env())
    actions = torch.arange(84).reshape(3, 28).float() / 100
    term.process_actions(actions)
    before = term._processed_actions.clone()
    term.reset(ids)
    torch.testing.assert_close(term._processed_actions[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(term._raw_actions[[0, 2]], actions[[0, 2]])
    torch.testing.assert_close(term._raw_actions[1], torch.zeros(28))
    torch.testing.assert_close(term._processed_actions[1], term._offset[1])
    term.reset()
    torch.testing.assert_close(term._processed_actions, term._offset)
    torch.testing.assert_close(term._raw_actions, torch.zeros(3, 28))


def test_action_contract_records_calibration_order_and_nonintegrating_reference(action_type):
    env = _env()
    term = action_type(SimpleNamespace(scale=.1, clip=None), env)
    contract = term.action_contract()
    assert contract["schema_version"] == 2
    assert contract["joint_order"] == env.joint_names
    assert contract["integrates_actions"] is False
    assert contract["clip"] is None
    torch.testing.assert_close(torch.tensor(contract["reference_targets_rad"]), term._offset[0])


def test_action_rejects_incomplete_calibration(action_type):
    with pytest.raises(ValueError, match="all 28"):
        action_type(SimpleNamespace(scale=.1, clip=None), _env(names=list(PREGRASP["robot_joint_targets"])[:27]))
    calibration = dict(PREGRASP, robot_joint_targets=dict(PREGRASP["robot_joint_targets"]))
    calibration["robot_joint_targets"]["Joint1_R"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        action_type(SimpleNamespace(scale=.1, clip=None), _env(calibration=calibration))
