"""Static checks for D-peg runtime optimizations that preserve the RL contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / (
    "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/"
    "config/InsertDPegCustom/env_cfg.py"
)


def test_d_peg_removes_only_inactive_ray_sensor_families():
    source = ENV_CFG.read_text(encoding="utf-8")
    assert "TIANJI_TACMAP_SURFACE_SENSOR_NAMES" in source
    assert "TIANJI_TACMAP_OBJECT_SENSOR_NAMES" in source
    assert "TIANJI_HYDROSHEAR_MARKER_SURFACE_SENSOR_NAMES" in source
    assert "TIANJI_HYDROSHEAR_MARKER_OBJECT_SENSOR_NAMES" in source
    assert "delattr(cfg.scene, sensor_name)" in source
    # Pressure remains active because it is part of the 669-dimensional input.
    assert "TIANJI_PRESSURE_SENSOR_NAMES" in source
    assert "sensor_cfg.update_period = pressure_update_period" in source


def test_pressure_update_period_matches_control_step_without_changing_decimation():
    source = ENV_CFG.read_text(encoding="utf-8")
    assert "pressure_update_period = cfg.sim.dt * cfg.decimation" in source
    assert "cfg.sim.dt = 1 / 240" in source
    assert "cfg.decimation = 8" in source
    assert "cfg.observations.proprio.pretrained_tactile = ObsTerm" in source
