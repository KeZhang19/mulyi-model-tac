from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARPA_WORKER = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_sharpa_tacmap_depth.py"
COMPARISON = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_depth_sharpa_vs_ours.py"
SWEEP = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_depth_sharpa_vs_ours_sweep.py"


def test_depth_comparison_scripts_are_valid_python() -> None:
    for path in (SHARPA_WORKER, COMPARISON, SWEEP):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_sharpa_worker_reuses_official_multienvironment_classes() -> None:
    source = SHARPA_WORKER.read_text(encoding="utf-8")
    assert "from env import SharpaWaveInhandRotateTactileAlignEnv" in source
    assert "from env_cfg import SharpaWaveEnvCfg" in source
    assert "cfg.scene.num_envs = int(args_cli.num_envs)" in source
    assert "observation, *_ = env.step(actions)" in source
    assert 'observation.get("vbts_deform")' in source
    assert '"sensor_count": len(cfg.vbts_sensor)' in source


def test_comparison_runs_children_sequentially_for_three_seconds() -> None:
    source = COMPARISON.read_text(encoding="utf-8")
    assert 'parser.add_argument("--measure-seconds", type=float, default=3.0)' in source
    assert '"--implementation",' in source
    assert '"ours",' in source
    assert '"--mode",' in source
    assert '"depth",' in source
    assert '"sequential_children": True' in source
    assert "for name, command, cwd, result_path in child_commands(args, output):" in source


def test_comparison_records_speed_throughput_and_vram() -> None:
    source = COMPARISON.read_text(encoding="utf-8")
    for field in ("batch_hz", "env_steps_per_s", "process_peak_vram_gib"):
        assert f'"{field}"' in source
    for artifact in ("comparison.csv", "depth_comparison.png", "summary.json"):
        assert artifact in source


def test_sweep_uses_isolated_comparisons_and_records_failed_points() -> None:
    source = SWEEP.read_text(encoding="utf-8")
    assert "default=[16, 32, 64, 128, 256, 512, 1024]" in source
    assert 'parser.add_argument("--measure-seconds", type=float, default=3.0)' in source
    assert '"sequential_isolated_processes": True' in source
    assert "comparison_command(args, num_envs=num_envs, output=point_output)" in source
    assert 'point_summary.get("failures")' in source
    for artifact in (
        "aggregate.csv",
        "vram_scaling.png",
        "depth_only_vram_scaling.pdf",
        "depth_only_vram_scaling.png",
        "throughput_scaling.png",
        "summary.json",
    ):
        assert artifact in source
