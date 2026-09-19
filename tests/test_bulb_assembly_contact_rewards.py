"""Exercise bulb-filtered contact rewards without starting Isaac Sim."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom"
FINGERS = ("thumb", "index", "mid", "ring", "pinky")


@pytest.fixture
def rewards():
    # Load the real functions, avoiding scene/Kit imports in a CPU test.
    namespace = {"torch": torch, "math": math, "ManagerBasedRLEnv": object}
    for filename, names in (
        ("env_cfg.py", {"rotate_bulb_finger_forces"}),
        ("assembly_env_cfg.py", {"assembly_any_finger_contact", "assembly_good_finger_contact"}),
    ):
        tree = ast.parse((CONFIG / filename).read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONFIG / filename), "exec"), namespace)
    return SimpleNamespace(**namespace)


def make_env(forces):
    values = torch.tensor(forces, dtype=torch.float32)
    sensors = {}
    for i, finger in enumerate(FINGERS):
        matrix = torch.zeros(values.shape[0], 1, 1, 3)
        matrix[:, 0, 0, 2] = values[:, i]
        sensors[f"right_{finger}dip_roll_rubber_link_object_s"] = SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=matrix,
            net_forces_w=torch.full((values.shape[0], 1, 3), 100.0),
        ))
    return SimpleNamespace(num_envs=values.shape[0], device="cpu", scene=SimpleNamespace(sensors=sensors))


def test_only_bulb_contact_and_thumb_opposition_qualify(rewards):
    env = make_env([
        [0, 0, 0, 0, 0],       # Large net forces from other objects do not count.
        [1, 1, 1, 1, 1],       # Source predicate is strictly greater than 1 N.
        [0, 2, 2, 0, 0],       # Two non-thumb fingers are not opposing contact.
        [2, 0, 0, 0, 0],
        [2, 0, 0, 0, 2],
        [100, 100, 100, 100, 100],
    ])
    torch.testing.assert_close(rewards.assembly_any_finger_contact(env), torch.tensor([0., 0., 1., 1., 1., 1.]))
    torch.testing.assert_close(rewards.assembly_good_finger_contact(env), torch.tensor([0., 0., 0., 0., 1., 1.]))


@pytest.mark.parametrize("finger", range(5))
def test_every_finger_resumes_credit_on_recontact_without_phase_state(rewards, finger):
    env = make_env([[0, 0, 0, 0, 0]])
    force = env.scene.sensors[f"right_{FINGERS[finger]}dip_roll_rubber_link_object_s"].data.force_matrix_w
    for value, expected in ((2., 1.), (0., 0.), (2., 1.)):
        force[..., 2] = value
        assert rewards.assembly_any_finger_contact(env).item() == expected


def test_invalid_or_missing_filtered_forces_do_not_earn_contact(rewards):
    env = make_env([[float("nan"), float("inf"), -float("inf"), 0., 2.]])
    assert rewards.assembly_any_finger_contact(env).item() == 1.
    assert rewards.assembly_good_finger_contact(env).item() == 0.
    env.scene.sensors["right_pinkydip_roll_rubber_link_object_s"].data.force_matrix_w = None
    assert rewards.assembly_any_finger_contact(env).item() == 0.


@pytest.mark.parametrize("threshold", [0., -1., float("nan"), float("inf")])
def test_bad_threshold_is_rejected(rewards, threshold):
    env = make_env([[0, 0, 0, 0, 0]])
    for function in (rewards.assembly_any_finger_contact, rewards.assembly_good_finger_contact):
        with pytest.raises(ValueError, match="threshold"):
            function(env, threshold=threshold)
