"""Free-bulb measurements: full turns, tilted clearance and partial resets."""

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from scipy.spatial.transform import Rotation

from test_rotate_bulb_rewards import CUSTOM_CONFIG, reward_code


@pytest.fixture
def collision_code(monkeypatch):
    # Load the installed pure tensor math utility without launching Isaac Sim.
    spec = importlib.util.find_spec("isaaclab")
    if spec is None:
        pytest.skip("Isaac Lab tensor math is required")
    roots = [Path(p) for p in spec.submodule_search_locations]
    candidates = [p for root in roots for p in (root / "utils/math.py",
                   root / "source/isaaclab/isaaclab/utils/math.py")]
    source = next(p for p in candidates if p.is_file())
    math_spec = importlib.util.spec_from_file_location("_bulb_tensor_math", source)
    math_module = importlib.util.module_from_spec(math_spec)
    math_spec.loader.exec_module(math_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)
    source = CUSTOM_CONFIG.parent / "collision.py"
    spec = importlib.util.spec_from_file_location("_bulb_collision_measurements", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quat(turns):
    angles = torch.as_tensor(turns, dtype=torch.float64).numpy() * (2 * math.pi)
    return torch.tensor(Rotation.from_euler("x", angles).as_quat()[:, [3, 0, 1, 2]], dtype=torch.float32)


def _state(code, count=2):
    state = code.CollisionScrewState(count, "cpu", .006, [[-.030, -.014, -.014], [0, .014, .014]], 0.)
    pos = torch.tensor([[.012, 0., 0.]]).repeat(count, 1)
    state.reset(torch.arange(count), pos, _quat([0.] * count), torch.full((count,), 3.))
    return state, pos


def test_full_turns_are_unwrapped_and_duplicate_reads_do_not_add_progress(collision_code):
    state, start = _state(collision_code)
    released = torch.zeros(2, dtype=torch.bool)
    for step in range(25):
        turns = step / 8
        pos = start.clone()
        pos[0, 0] += turns * .006
        quat = _quat([turns, 0])
        state.measure(pos, quat, released)
        state.measure(pos, -quat, released)  # q and -q are the same physical pose.
    torch.testing.assert_close(state.effective_turns, torch.tensor([3., 0.]), atol=1e-5, rtol=0)
    torch.testing.assert_close(state.progress, torch.tensor([[1., 1.], [0., 0.]]), atol=1e-5, rtol=0)


def test_spin_without_lift_and_lift_without_spin_do_not_count_as_unscrewing(collision_code):
    state, pos = _state(collision_code)
    released = torch.zeros(2, dtype=torch.bool)
    pos[1, 0] += .018
    for step in range(1, 9):
        state.measure(pos, _quat([step / 8, 0]), released)
    assert state.effective_turns.tolist() == [0., 0.]
    assert state.progress[0, 0] > .3 and state.progress[0, 1] == 0
    torch.testing.assert_close(state.progress[1], torch.tensor([0., 1.]))


def test_reverse_rotation_reduces_progress_and_released_motion_is_frozen(collision_code):
    state, start = _state(collision_code)
    released = torch.zeros(2, dtype=torch.bool)
    for turns in (.25, .5, .75, 1., .75, .5):
        pos = start.clone()
        pos[:, 0] += turns * .006
        state.measure(pos, _quat([turns, turns]), released)
    torch.testing.assert_close(state.effective_turns, torch.tensor([.5, .5]), atol=1e-5, rtol=0)
    released[:] = True
    for turns in (.75, 1., 1.25):
        state.measure(pos + .1, _quat([turns, turns]), released)
    torch.testing.assert_close(state.effective_turns, torch.tensor([.5, .5]), atol=1e-5, rtol=0)
    state.reset(torch.tensor([0]), start[:1], _quat([0]), torch.tensor([3.]))
    torch.testing.assert_close(state.effective_turns, torch.tensor([0., .5]), atol=1e-5, rtol=0)
    assert state.angle[0] == 0 and state.angle[1] > 3.1


def test_clearance_accounts_for_tilt_and_does_not_require_fake_joint_progress(collision_code):
    state, pos = _state(collision_code)
    quat = torch.tensor(Rotation.from_euler("z", [90, 90], degrees=True).as_quat()[:, [3, 0, 1, 2]],
                        dtype=torch.float32)
    pos[:, 0] = torch.tensor([.013, .015])
    state.measure(pos, quat, torch.zeros(2, dtype=torch.bool))
    assert state.cleared.tolist() == [False, True]
    torch.testing.assert_close(state.clearance, torch.tensor([-.001, .001]), atol=1e-7, rtol=0)
    assert not state.effective_turns.any()


def test_phase_event_only_observes_separation_without_any_physics_writes(monkeypatch, collision_code):
    code = reward_code.__wrapped__(monkeypatch, SimpleNamespace(param=CUSTOM_CONFIG))
    state, pos = _state(collision_code)
    pos[1, 0] = .04
    quat = _quat([0., 0.])
    # No lamp articulation, attachment attributes or physical setter methods:
    # invoking a legacy state write or synthetic joint read fails this test.
    env = SimpleNamespace(
        cfg=SimpleNamespace(bulb_pure_collision=True), num_envs=2, device="cpu",
        _bulb_collision_state=state, _bulb_released=torch.zeros(2, dtype=torch.bool),
        scene={"object": SimpleNamespace(data=SimpleNamespace(root_pos_w=pos, root_quat_w=quat)),
               "support": SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3), root_quat_w=quat))},
    )
    before = pos.clone()
    code.release_unscrewed_bulb(env)
    assert env._bulb_released.tolist() == [False, True]
    assert code.rotate_bulb_screw_complete(env).tolist() == [False, True]
    assert code.rotate_bulb_screw_progress(env).shape == (2, 2)
    assert not code.rotate_bulb_unscrewed_turns(env).any()
    torch.testing.assert_close(pos, before)
    code.release_unscrewed_bulb(env)  # Idempotent phase latch.
    assert env._bulb_released.tolist() == [False, True]


def test_train_and_play_default_to_pure_collision():
    """Check the pure-collision default flag for rotation and its assembly subclass."""
    tree = ast.parse(CUSTOM_CONFIG.read_text())
    for name in ("DexsuiteRevo3RotateBulbEnvCfg", "DexsuiteRevo3RotateBulbEnvCfg_PLAY"):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        field = next(node for node in cls.body if isinstance(node, ast.AnnAssign)
                     and node.target.id == "bulb_pure_collision")
        assert ast.literal_eval(field.value) is False
    # A parent-default change must not constrain the independent tabletop task.
    assembly_tree = ast.parse((CUSTOM_CONFIG.parent / "assembly_env_cfg.py").read_text())
    cls = next(node for node in assembly_tree.body if isinstance(node, ast.ClassDef)
               and node.name == "DexsuiteRevo3BulbAssemblyEnvCfg")
    field = next(node for node in cls.body if isinstance(node, ast.AnnAssign)
                 and node.target.id == "bulb_pure_collision")
    assert ast.literal_eval(field.value) is True
