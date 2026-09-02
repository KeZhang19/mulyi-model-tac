from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_isaaclab_tacsl_sensor.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_official_tacsl_benchmark_is_valid_python() -> None:
    ast.parse(_source(), filename=str(SCRIPT))


def test_official_tacsl_benchmark_uses_isaaclab_contrib_sensor() -> None:
    source = _source()
    assert "from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg" in source
    assert '"sensor_class": "isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor"' in source
    assert '"official_sensor_class": True' in source
    assert '"rl_environment": False' in source


def test_official_tacsl_benchmark_has_force_rgb_and_all_modes() -> None:
    source = _source()
    assert 'choices=("force_field", "rgb", "all")' in source
    assert "enable_camera_tactile=str(args_cli.mode) in (\"rgb\", \"all\")" in source
    assert "enable_force_field=str(args_cli.mode) in (\"force_field\", \"all\")" in source
    assert "tactile_array_size=(20, 25)" in source
    assert "GELSIGHT_R15_CFG.image_height" in source


def test_official_tacsl_benchmark_measures_complete_steps() -> None:
    source = _source()
    assert "sim.step()" in source
    assert "scene.update(sim.get_physics_dt())" in source
    assert 'scene["tactile_sensor"].data' in source
    assert 'parser.add_argument("--measure-seconds", type=float, default=3.0)' in source
    assert '"sync_cuda_per_step": bool(args_cli.sync_cuda)' in source
