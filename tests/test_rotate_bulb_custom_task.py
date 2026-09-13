"""Check registration, configuration isolation and the custom observation contract."""

import ast
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEX = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite"
CUSTOM = DEX / "config/RotateBulbCustom"
ORIGINAL = DEX / "config/Revo3"
TASK = "BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0"


def test_custom_task_is_registered_with_resolvable_local_config_entries(monkeypatch):
    gym = pytest.importorskip("gymnasium")
    # Execute the actual Dexsuite package initializer, including both registration
    # imports, without booting unrelated direct tasks or Isaac Sim extensions.
    monkeypatch.setattr(gym.envs.registration, "registry", {})
    name = "_custom_registration_dexsuite"
    spec = importlib.util.spec_from_file_location(name, DEX / "__init__.py")
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    custom = gym.spec(TASK)
    original = gym.spec("BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0")
    assert custom.entry_point == original.entry_point
    assert custom.kwargs.keys() == original.kwargs.keys()
    for key, entry in custom.kwargs.items():
        module, symbol = entry.split(":")
        assert ".config.RotateBulbCustom." in module
        assert entry != original.kwargs[key]
        relative = module.split(".config.RotateBulbCustom.", 1)[1]
        if symbol.endswith(".yaml"):
            assert (CUSTOM / relative.replace(".", "/") / symbol).is_file()
        else:
            path = CUSTOM / (relative.replace(".", "/") + ".py")
            definitions = ast.parse(path.read_text()).body
            assert any(isinstance(node, ast.ClassDef) and node.name == symbol for node in definitions)


def test_custom_configs_do_not_import_original_task_parameter_modules():
    forbidden = {
        "Revo3", "dexsuite_env_cfg_grasp_tianji", "dexsuite_revo3_env_cfg_grasp",
        "dexsuite_revo3_env_cfg_rotate_bulb", "tianji_revo3_right", "rotate_bulb_tactile",
    }
    for path in CUSTOM.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = set((node.module or "").split(".")) | {item.name for item in node.names}
                assert not names & forbidden, (path, ast.unparse(node))


@pytest.fixture
def cpu_agent_loader(monkeypatch):
    """Load real Isaac Lab config classes without its simulator package initializers."""
    spec = importlib.util.find_spec("isaaclab")
    if spec is None:
        pytest.skip("Isaac Lab's real CPU configuration utilities are required")
    roots = [Path(path) for path in spec.submodule_search_locations]
    candidates = [path for root in roots for path in (
        root / "utils/configclass.py", root / "source/isaaclab/isaaclab/utils/configclass.py",
    )]
    source = next((path for path in candidates if path.is_file()), None)
    assert source is not None, "Cannot locate installed Isaac Lab configclass"

    def package(name, path):
        module = ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, path):
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        monkeypatch.setitem(sys.modules, name, module)
        module_spec.loader.exec_module(module)
        return module

    # Private utility namespace keeps tests independent of simulator imports.
    package("_custom_cpu_isaac_utils", source.parent)
    configclass = load("_custom_cpu_isaac_utils.configclass", source).configclass
    utils = ModuleType("isaaclab.utils")
    utils.configclass = configclass
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)

    rl_candidates = [root / "source/isaaclab_rl/isaaclab_rl/rsl_rl" for root in roots]
    rl_spec = importlib.util.find_spec("isaaclab_rl")
    if rl_spec is not None:
        rl_candidates.extend(Path(path) / "rsl_rl" for path in rl_spec.submodule_search_locations)
    rl_root = next((path for path in rl_candidates if (path / "rl_cfg.py").is_file()), None)
    assert rl_root is not None, "Cannot locate installed Isaac Lab RSL-RL config sources"
    package("isaaclab_rl", rl_root.parent)
    rl = package("isaaclab_rl.rsl_rl", rl_root)
    for filename in ("rnd_cfg", "symmetry_cfg", "rl_cfg", "distillation_cfg"):
        module = load(f"isaaclab_rl.rsl_rl.{filename}", rl_root / f"{filename}.py")
        for name, value in vars(module).items():
            if name.startswith("RslRl"):
                setattr(rl, name, value)
    return load


@pytest.mark.parametrize("kind, class_name, network_fields", [
    ("ppo", "DexsuiteRevo3RotateBulbPPORunnerCfg", ("actor_hidden_dims", "critic_hidden_dims")),
    ("distillation", "DexsuiteRevo3RotateBulbDistillationRunnerCfg", ("student_hidden_dims", "teacher_hidden_dims")),
])
def test_agent_defaults_match_but_nested_mutations_are_independent(cpu_agent_loader, kind, class_name, network_fields):
    load = cpu_agent_loader
    original_type = getattr(load(f"_bulb_original_{kind}", ORIGINAL / f"agents/rsl_rl_{kind}_cfg_rotate_bulb.py"), class_name)
    custom_type = getattr(load(f"_bulb_custom_{kind}", CUSTOM / f"agents/rsl_rl_{kind}_cfg.py"), class_name)
    original, custom, another = original_type(), custom_type(), custom_type()
    before = original.to_dict()
    original_values, custom_values = original.to_dict(), custom.to_dict()
    assert original_values.pop("experiment_name") != custom_values.pop("experiment_name")
    assert original_values == custom_values
    for field in network_fields:
        getattr(custom.policy, field).append(16)
        assert getattr(another.policy, field) == getattr(original.policy, field)
    custom.algorithm.learning_rate = 2e-5
    custom.obs_groups["policy"].append("custom_state")
    assert another.algorithm.learning_rate == original.algorithm.learning_rate
    assert another.obs_groups == original.obs_groups
    assert original.to_dict() == before


def test_rl_games_defaults_match_but_have_an_independent_experiment():
    yaml = pytest.importorskip("yaml")
    original = yaml.safe_load((ORIGINAL / "agents/rl_games_ppo_cfg_rotate_bulb.yaml").read_text())
    custom = yaml.safe_load((CUSTOM / "agents/rl_games_ppo_cfg.yaml").read_text())
    assert original["params"]["config"].pop("name") != custom["params"]["config"].pop("name")
    assert original == custom


def test_custom_tactile_contract_identifies_new_task(tmp_path):
    marker = tmp_path / "marker_positions.npz"
    camera = tmp_path / "camera_ray_rectangles_320x240.json"
    background = tmp_path / "background.png"
    for path in (marker, camera, background):
        path.write_bytes(b"sensor-asset")
    cfg = {"hydroshear_marker_layout_path": str(marker), "taxim_rgb_background_path": str(background)}
    encoder = SimpleNamespace(
        projection_dim=16, bundle_sha256="encoder-hash",
        observation_contract=lambda **kwargs: dict(kwargs),
    )
    manager = SimpleNamespace(
        group_obs_dim={"policy": (20,), "proprio": (90,), "perception": (10,)},
        active_terms={name: ["state"] for name in ("policy", "proprio", "perception")},
        group_obs_term_dim={"policy": [(20,)], "proprio": [(90,)], "perception": [(10,)]},
    )
    env = SimpleNamespace(
        _rotate_bulb_tactile_term=SimpleNamespace(encoder=encoder, component_cfg=cfg),
        observation_manager=manager,
    )
    source = CUSTOM / "tactile.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "initialize_rotate_bulb_tactile_contract")
    namespace = {"Path": Path, "file_sha256": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    namespace[function.name](env)
    assert env.tactile_policy_contract["task"] == TASK
    assert env.tactile_policy_contract["state_dim"] == 40
    assert env.tactile_policy_contract["observation_dim"] == 120
