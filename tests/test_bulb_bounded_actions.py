"""Check the real action implementation against a CPU articulation boundary."""

import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
SOURCE = Path(__file__).resolve().parents[1] / (
    "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom")


@pytest.fixture
def action_module(monkeypatch):
    """Only replace the simulator-backed base; exercise all task action methods."""
    class JointAction:
        def __init__(self, cfg, env):
            self.cfg, self._asset = cfg, env.scene["robot"]
            self.device, self.num_envs = "cpu", env.num_envs
            self._joint_names = list(self._asset.joint_names)
            self._joint_ids = slice(None)
            self._raw_actions = torch.zeros(env.num_envs, len(self._joint_names))
            self._processed_actions = torch.zeros_like(self._raw_actions)
            self._scale = torch.tensor([next(v for k, v in cfg.scale.items() if re.fullmatch(k, name))
                                        for name in self._joint_names])
            self._offset = cfg.offset

    configs = ModuleType("isaaclab.envs.mdp.actions.actions_cfg")
    configs.JointActionCfg = type("JointActionCfg", (), {})
    joints = ModuleType("isaaclab.envs.mdp.actions.joint_actions")
    joints.JointAction = JointAction
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda value: value
    monkeypatch.setitem(sys.modules, configs.__name__, configs)
    monkeypatch.setitem(sys.modules, joints.__name__, joints)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    package = ModuleType("_bounded_bulb_action_task")
    package.__path__ = [str(SOURCE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(package.__name__ + ".actions", SOURCE / "actions.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def make_action(module):
    names = list(reversed(module.JOINT_NAMES))  # Deliberately not arm-first.
    robot = SimpleNamespace(joint_names=names, data=SimpleNamespace(
        joint_pos=torch.zeros(2, 28), soft_joint_pos_limits=torch.tensor([-1., 1.]).expand(2, 28, 2).clone()))
    robot.submitted = []
    robot.set_joint_position_target = lambda targets, joint_ids: robot.submitted.append(targets.clone())
    env = SimpleNamespace(scene={"robot": robot}, num_envs=2, step_dt=1 / 60, cfg=SimpleNamespace(decimation=4))
    return module.BulbJointPositionAction(module.BulbJointPositionActionCfg(), env), robot


def test_bounded_targets_hold_for_all_substeps_and_map_by_name(action_module):
    action, robot = make_action(action_module)
    values = torch.full((2, 28), 1000.)
    action.process_actions(values)
    expected = torch.tensor([0.025 if name.startswith("joint") else 0.05 for name in robot.joint_names])
    torch.testing.assert_close(action.position_targets, expected.expand(2, 28))
    torch.testing.assert_close(action.processed_actions, action.position_targets)
    assert len(action.arm_action_indices) == 7 and len(action.hand_action_indices) == 21
    assert action.arm_action_indices != list(range(7))
    for _ in range(4):
        action.apply_actions()
        robot.data.joint_pos += 0.01
    assert all(torch.equal(target, robot.submitted[0]) for target in robot.submitted)
    action.process_actions(torch.zeros_like(values))
    torch.testing.assert_close(action.position_targets, robot.data.joint_pos)
    assert torch.equal(action.action_deltas, torch.zeros_like(values))
    contract = action.action_contract()
    assert contract["joint_names"] == robot.joint_names and contract["decimation"] == 4


def test_target_limits_partial_reset_and_nonfinite_rejection(action_module):
    action, robot = make_action(action_module)
    robot.data.joint_pos[0] = 0.99
    robot.data.joint_pos[1] = -0.99
    action.process_actions(torch.stack((torch.ones(28), -torch.ones(28))))
    torch.testing.assert_close(action.position_targets, torch.stack((torch.ones(28), -torch.ones(28))))
    untouched = action.position_targets[1].clone()
    robot.data.joint_pos[0] = 0.2
    action.reset(torch.tensor([0]))
    torch.testing.assert_close(action.position_targets[0], torch.full((28,), 0.2))
    torch.testing.assert_close(action.position_targets[1], untouched)
    assert action.action_deltas[0].count_nonzero() == 0
    assert action.action_deltas[1].count_nonzero() == 28
    with pytest.raises(ValueError, match="non-finite"):
        action.process_actions(torch.full((2, 28), float("nan")))
