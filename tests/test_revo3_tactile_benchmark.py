from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "scripts" / "rsl_rl" / "benchmark_revo3_tactile_obs.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("_revo3_tactile_benchmark", BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_child_command_enables_headless_synchronized_full_tactile_benchmark(tmp_path):
    module = load_benchmark_module()
    args = module.argparse.Namespace(
        python=Path(sys.executable),
        task="BrainCo-Dexsuite-Revo3-Right-Lift-v0",
        device="cuda:0",
        focus_finger="index",
        presser="square_4",
        presser_force_n=100.0,
        warmup_steps=200,
        measure_steps=1000,
        sync_cuda=True,
        extra_arg=[],
    )
    result_path = tmp_path / "result.json"
    command = module.child_command(args, num_envs=8, result_path=result_path)

    assert "--headless" in command
    assert "--no-local-ui" in command
    assert command[command.index("--num_envs") + 1] == "8"
    assert command[command.index("--benchmark-warmup-steps") + 1] == "200"
    assert command[command.index("--benchmark-steps") + 1] == "1000"
    assert "--benchmark-sync-cuda" in command
    assert command[command.index("--benchmark-output") + 1] == str(result_path)


def test_aggregate_rows_reports_population_statistics():
    module = load_benchmark_module()
    rows = [
        {
            "num_envs": 4,
            "mean_step_ms": 10.0,
            "p95_step_ms": 12.0,
            "sim_hz": 100.0,
            "env_steps_per_s": 400.0,
            "real_time_factor": 100.0 / 60.0,
            "peak_allocated_gb": 2.0,
            "peak_reserved_gb": 3.0,
        },
        {
            "num_envs": 4,
            "mean_step_ms": 20.0,
            "p95_step_ms": 24.0,
            "sim_hz": 50.0,
            "env_steps_per_s": 200.0,
            "real_time_factor": 50.0 / 60.0,
            "peak_allocated_gb": 4.0,
            "peak_reserved_gb": 5.0,
        },
    ]

    aggregate = module.aggregate_rows(rows)

    assert len(aggregate) == 1
    assert aggregate[0]["num_envs"] == 4
    assert aggregate[0]["successful_repeats"] == 2
    assert aggregate[0]["mean_step_ms_mean"] == pytest.approx(15.0)
    assert aggregate[0]["mean_step_ms_std"] == pytest.approx(5.0)
    assert aggregate[0]["env_steps_per_s_mean"] == pytest.approx(300.0)
