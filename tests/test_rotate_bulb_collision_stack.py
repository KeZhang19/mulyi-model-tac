"""Regression checks for late CLI overrides and the large-scene contact budget."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / (
    "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/"
    "config/RotateBulbCustom/environment.py"
)
spec = importlib.util.spec_from_file_location("_bulb_collision_environment", SOURCE)
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


@pytest.mark.parametrize("count, configured, expected", [
    (1, 2**28, 2**28),
    (32, 2**28, 2**28),
    (512, 2**28, 2**28),
    (1024, 2**28, 2**29),
    (4096, 2**28, 2**31),
    (4096, 2**30, 2**31),  # Old launch command must not reinstate the overflow.
    (8192, 2**28, 2**31),  # Required single-GPU scale remains representable.
    (4096, 3 * 2**30, 3 * 2**30),
])
def test_budget_handles_scene_scale_and_preserves_larger_reservations(count, configured, expected):
    assert environment.collision_stack_bytes(count, configured) == expected


@pytest.mark.parametrize("count, configured", [(0, 2**28), (-1, 2**28), (1.5, 2**28),
                                              (True, 2**28), (32, -1), (8192, 2**28), (32, 2**32)])
def test_invalid_or_unrepresentable_budget_fails_before_simulation(count, configured):
    with pytest.raises(ValueError):
        environment.collision_stack_bytes(count, configured)


def test_final_overrides_are_applied_before_environment_constructor(monkeypatch):
    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=32),
                          sim=SimpleNamespace(physx=SimpleNamespace(gpu_collision_stack_size=2**28)))
    # Simulate parse_env_cfg/CLI/Hydra updates made after config construction.
    cfg.scene.num_envs = 4096
    cfg.sim.physx.gpu_collision_stack_size = 2**30
    calls = []
    result = object()

    def construct(**kwargs):
        assert kwargs['cfg'].sim.physx.gpu_collision_stack_size > 1_251_133_104
        assert kwargs['cfg'] is cfg
        calls.append(kwargs)
        return result

    package = ModuleType('isaaclab.envs')
    package.ManagerBasedRLEnv = construct
    monkeypatch.setitem(sys.modules, 'isaaclab.envs', package)
    assert environment.make_rotate_bulb_env(cfg, render_mode='rgb_array') is result
    assert calls == [{'cfg': cfg, 'render_mode': 'rgb_array'}]
