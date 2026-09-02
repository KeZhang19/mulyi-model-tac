from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_revo3_tactile_standalone.py"
REVO_CFG = (
    REPO_ROOT
    / "source"
    / "BrainCo_DexHand"
    / "BrainCo_DexHand"
    / "tasks"
    / "manager_based"
    / "dexsuite"
    / "config"
    / "Revo3"
    / "dexsuite_revo3_env_cfg_grasp.py"
)


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_standalone_benchmark_is_valid_python() -> None:
    ast.parse(_source(), filename=str(SCRIPT))


def test_standalone_benchmark_does_not_construct_gym_or_manager_runtime() -> None:
    tree = ast.parse(_source(), filename=str(SCRIPT))
    imported_modules: list[str] = []
    called_attributes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.append(node.func.attr)

    assert not any(name == "gym" or name.startswith("gymnasium") for name in imported_modules)
    assert "make" not in called_attributes
    assert "compute" not in called_attributes


def test_standalone_benchmark_uses_vectorized_interactive_scene() -> None:
    source = _source()
    assert "scene = InteractiveScene(source_cfg.scene)" in source
    assert "source_cfg.scene.num_envs = int(args_cli.num_envs)" in source
    assert '"env_steps_per_s": float(runtime.num_envs) * sim_hz' in source


def test_standalone_benchmark_computes_all_raw_tactile_modalities() -> None:
    source = _source()
    for function_name in (
        "ours_rl_pressure_obs",
        "ours_rl_tacmap_policy_obs",
        "ours_rl_taxim_rgb_obs",
        "ours_rl_hydroshear_obs",
    ):
        assert f"mdp.{function_name}(" in source
    assert "ours_rl_tacmap_resnet_obs" not in source
    assert "ours_rl_taxim_resnet_obs" not in source


def test_standalone_benchmark_exposes_non_rl_gpu_implementations() -> None:
    source = _source()
    tree = ast.parse(source, filename=str(SCRIPT))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "IMPLEMENTATION_CHOICES" for target in node.targets)
    )
    implementations = ast.literal_eval(assignment.value)

    assert implementations == ("ours", "tacmap", "fots", "tacsl", "hydroshear")
    for function_name in (
        "fots_baseline_rl_obs",
        "tacsl_baseline_rl_obs",
        "hydroshear_baseline_rl_obs",
    ):
        assert f"mdp.{function_name}(" in source
    assert '"rl_environment": False' in source


def test_standalone_reuses_connected_320_by_240_rl_tacmap_baseline() -> None:
    source = _source()
    config_source = REVO_CFG.read_text(encoding="utf-8")
    assert 'if implementation == "tacmap":' in source
    assert "mdp.tacmap_rl_obs(runtime, **kernel_params)" in source
    assert '"tacmap": "rl_tacmap"' in source
    assert 'elif tactile_implementation == "tacmap_baseline":' in config_source
    assert "func=mdp.tacmap_rl_obs" in config_source
    assert "TIANJI_TACMAP_BASELINE_RAY_ROWS = 240" in config_source
    assert "TIANJI_TACMAP_BASELINE_RAY_COLS = 240" in config_source
    assert '"tacmap_rows": TIANJI_TACMAP_BASELINE_RAY_ROWS' in config_source
    assert '"tacmap_cols": TIANJI_TACMAP_BASELINE_RAY_COLS' in config_source


def test_tacmap_baseline_uses_offline_surface_reference_and_only_object_raycasters() -> None:
    config_source = REVO_CFG.read_text(encoding="utf-8")
    reference_path = (
        REPO_ROOT
        / "tacmap"
        / "assets"
        / "tactilesensor_map"
        / "revo21_dv2"
        / "tacmap_surface_reference_240x240.npy"
    )
    reference = np.load(reference_path, allow_pickle=False)

    assert 'if tactile_implementation != "tacmap_baseline":' in config_source
    assert '"tacmap_surface_sensor_names": []' in config_source
    assert '"tacmap_surface_reference_npy": str(TIANJI_TACMAP_BASELINE_SURFACE_REFERENCE_NPY)' in config_source
    assert reference.shape == (5, 240, 240)
    assert reference.dtype == np.float32
    assert np.isfinite(reference).all()
    assert np.count_nonzero(reference) > 0


def test_standalone_baselines_toggle_the_four_rl_selection_flags() -> None:
    source = _source()
    expected = {
        "TACSL": "tacsl",
        "TACMAP": "tacmap",
        "HYDROSHEAR": "hydroshear",
        "FOTS": "fots",
    }
    for flag, implementation in expected.items():
        assert (
            f'revo_cfg.ENABLE_RL_{flag}_BASELINE_OBS = implementation == "{implementation}"'
            in source
        )


def test_standalone_benchmark_records_nvml_process_vram() -> None:
    source = _source()
    assert "class NvmlMemorySampler" in source
    assert '"process_current_peak_bytes"' in source
    assert '"process_growth_peak_bytes"' in source
    assert '"device_used_peak_bytes"' in source
    assert 'gpu["nvml"] = nvml_sampler.result()' in source


def test_standalone_benchmark_defines_visual_tactile_mode_presets() -> None:
    tree = ast.parse(_source(), filename=str(SCRIPT))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "BENCHMARK_MODE_MODALITIES"
    )
    assert assignment.value is not None
    presets = ast.literal_eval(assignment.value)

    assert presets == {
        "depth": ("depth",),
        "rgb": ("rgb",),
        "marker": ("marker",),
        "all": ("depth", "rgb", "marker"),
    }
    assert '"mode": selected_mode_name()' in _source()


def test_standalone_benchmark_uses_shallow_default_press_offsets() -> None:
    source = _source()
    assert 'parser.add_argument("--press-start-offset", type=float, default=0.035)' in source
    assert 'parser.add_argument("--press-end-offset", type=float, default=0.025)' in source


def test_standalone_visualization_is_optional_and_outside_timed_region() -> None:
    source = _source()
    assert '"--visualize"' in source
    assert "args_cli.headless = not bool(args_cli.visualize)" in source
    assert '"visualization_in_timed_region": False' in source

    tree = ast.parse(source, filename=str(SCRIPT))
    main_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    measured_append_line = next(
        node.lineno
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "measured_times"
    )
    visualization_line = next(
        node.lineno
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_visualization_image"
    )
    assert measured_append_line < visualization_line


def test_standalone_result_explicitly_records_runtime_boundary() -> None:
    source = _source()
    for field in (
        '"gym_environment": False',
        '"rl_environment": False',
        '"observation_manager": False',
        '"policy_inference": False',
        '"raw_tactile_only": True',
    ):
        assert field in source


def test_standalone_benchmark_supports_wall_clock_measurement() -> None:
    source = _source()
    assert '"--measure-seconds"' in source
    assert '"measurement_mode": "seconds" if args_cli.measure_seconds is not None else "steps"' in source
    assert '"measured_steps": len(step_times_s)' in source
    assert "measured_elapsed_s >= measurement_target_s" in source
