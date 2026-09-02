from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_revo3_tactile_standalone_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("standalone_tactile_sweep", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_sweep_is_valid_python_and_uses_power_of_two_counts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(SCRIPT))
    module = _load_module()

    assert module.DEFAULT_ENV_COUNTS == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert module.BENCHMARK.name == "benchmark_revo3_tactile_standalone.py"
    assert "visualize_rl_tactile_obs.py" not in source


def test_standalone_sweep_defaults_to_three_measured_seconds(monkeypatch, tmp_path) -> None:
    module = _load_module()
    monkeypatch.setattr(
        "sys.argv",
        [str(SCRIPT), "--output", str(tmp_path / "results")],
    )

    args = module.parse_args()
    command = module.child_command(args, num_envs=8, result_path=tmp_path / "result.json")

    assert args.measure_seconds == 3.0
    assert args.measure_steps is None
    assert args.warmup_steps == 0
    assert "--measure-seconds" in command
    assert command[command.index("--measure-seconds") + 1] == "3"
    assert "--measure-steps" not in command


def test_standalone_sweep_aggregates_and_plots_three_scaling_metrics(tmp_path) -> None:
    module = _load_module()
    rows = []
    for repeat, batch_hz in ((1, 20.0), (2, 22.0)):
        rows.append(
            {
                "num_envs": 16,
                "repeat": repeat,
                "mean_step_ms": 1000.0 / batch_hz,
                "p95_step_ms": 55.0,
                "batch_hz": batch_hz,
                "env_steps_per_s": batch_hz * 16.0,
                "real_time_factor": batch_hz / 60.0,
                "process_peak_vram_gib": 4.0,
                "torch_peak_allocated_gib": 1.0,
                "torch_peak_reserved_gib": 1.2,
                "gpu": "test",
                "result": "test.json",
            }
        )

    aggregates = module.aggregate_rows(rows)
    plot_path = tmp_path / "scaling.png"
    module.plot_curves(plot_path, aggregates, failed_counts=[512])

    assert len(aggregates) == 1
    assert aggregates[0]["batch_hz_mean"] == 21.0
    assert aggregates[0]["env_steps_per_s_mean"] == 336.0
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
