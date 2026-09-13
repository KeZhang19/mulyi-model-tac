"""Exercise task config inheritance using Isaac Lab's real CPU configclass implementation."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/Revo3/agents/rsl_rl_ppo_cfg_insert_d_peg.py"


def test_ppo_policy_is_read_from_config_instance_and_copied(monkeypatch):
    specification = importlib.util.find_spec("isaaclab")
    if specification is None:
        pytest.skip("Isaac Lab is required for its real configclass semantics")
    candidates = [path for root in specification.submodule_search_locations for path in (
        Path(root) / "utils/configclass.py",
        Path(root) / "source/isaaclab/isaaclab/utils/configclass.py",
    )]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        pytest.fail("Installed Isaac Lab configclass source could not be located")
    # Import the unchanged CPU utility without executing Isaac's simulation-dependent
    # utils/__init__.py; relative dictionary/array/string utilities load normally.
    package_name = "_d_peg_cpu_isaac_utils"
    package = ModuleType(package_name)
    package.__path__ = [str(source.parent)]
    monkeypatch.setitem(sys.modules, package_name, package)
    module_spec = importlib.util.spec_from_file_location(f"{package_name}.configclass", source)
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_spec.name, module)
    module_spec.loader.exec_module(module)
    configclass = module.configclass

    @configclass
    class Policy:
        init_noise_std: float = 1.0
        actor_hidden_dims: list[int] = [512, 256, 128]

    @configclass
    class Algorithm:
        learning_rate: float = 1.0e-3
        gamma: float = .97  # Deliberately different to verify the task's explicit override.
        schedule: str = "adaptive"
        desired_kl: float = .01
        num_learning_epochs: int = 5

    @configclass
    class Parent:
        policy: Policy = Policy()
        algorithm: Algorithm = Algorithm()
        experiment_name: str = "dexsuite_revo3_rotate_bulb"
        save_interval: int = 250

    # This is the real dataclass behavior that an identity-decorator stub misses.
    assert not hasattr(Parent, "policy")
    tree = ast.parse(CONFIG.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {"__name__": __name__, "configclass": configclass,
                 "DexsuiteRevo3RotateBulbPPORunnerCfg": Parent}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(CONFIG), "exec"), namespace)
    task_type = namespace[cls.name]
    first, second, bulb = task_type(), task_type(), Parent()
    assert first.policy.init_noise_std == .15
    assert first.algorithm.learning_rate == 3e-5
    assert first.algorithm.gamma == .999
    assert first.algorithm.schedule == "fixed"
    assert first.algorithm.entropy_coef == .001
    assert first.algorithm.desired_kl == .01
    assert first.algorithm.num_learning_epochs == 5
    assert first.experiment_name == "dexsuite_revo3_insert_d_peg_v3"
    assert first.save_interval == 25
    assert bulb.save_interval == 250
    assert bulb.policy.init_noise_std == 1.0
    assert bulb.algorithm.gamma == .97
    first.policy.actor_hidden_dims.append(16)
    first.policy.init_noise_std = .2
    first.algorithm.learning_rate = 2e-5
    first.algorithm.gamma = .5
    assert second.policy.actor_hidden_dims == [512, 256, 128]
    assert second.policy.init_noise_std == .15
    assert bulb.policy.actor_hidden_dims == [512, 256, 128]
    assert second.algorithm.learning_rate == 3e-5
    assert second.algorithm.gamma == .999
    assert bulb.algorithm.learning_rate == 1e-3
    assert bulb.algorithm.gamma == .97
