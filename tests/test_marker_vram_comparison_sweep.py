from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_marker_vram_comparison_sweep.py"


def _source() -> str:
    return SWEEP.read_text(encoding="utf-8")


def test_marker_vram_sweep_is_valid_python() -> None:
    ast.parse(_source(), filename=str(SWEEP))


def test_marker_vram_sweep_compares_requested_methods_to_4096() -> None:
    source = _source()
    assert 'DEFAULT_ENV_COUNTS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)' in source
    assert 'METHODS = ("ours", "tacsl", "fots", "hydroshear")' in source
    assert 'command.extend(("--mode", "marker"))' in source
    assert 'parser.add_argument("--measure-seconds", type=float, default=3.0)' in source


def test_marker_vram_sweep_marks_cuda_oom_at_gpu_capacity() -> None:
    source = _source()
    assert 'status="inferred_oom"' in source
    assert 'status in ("oom", "inferred_oom")' in source
    assert 'y = capacity' in source
    assert 'if status in ("oom", "inferred_oom"):\n                break' in source
    assert 'axis.axhline(capacity' in source
    assert 'stop_method_after_first_cuda_oom' in source
    assert '"--retry-timeouts"' in source
    assert 'prior_status == "timeout" and bool(args.retry_timeouts)' in source


def test_marker_vram_sweep_uses_isolated_non_rl_worker() -> None:
    source = _source()
    assert 'benchmark_revo3_tactile_standalone.py' in source
    assert '"sequential_isolated_processes": True' in source
    assert 'start_new_session=True' in source
    for artifact in (
        "aggregate.csv",
        "tactile_field_vram_scaling.png",
        "tactile_field_vram_scaling.pdf",
        "marker_throughput_scaling.png",
        "marker_throughput_scaling.pdf",
        "summary.json",
    ):
        assert artifact in source
