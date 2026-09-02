from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP = REPO_ROOT / "scripts" / "tactile_simulation" / "benchmark_ours_modalities_vram_sweep.py"


def _source() -> str:
    return SWEEP.read_text(encoding="utf-8")


def test_ours_modalities_sweep_is_valid_python() -> None:
    ast.parse(_source(), filename=str(SWEEP))


def test_ours_modalities_sweep_uses_requested_cumulative_variants() -> None:
    source = _source()
    assert 'DEFAULT_ENV_COUNTS = (16, 32, 64, 128, 256, 512, 1024, 2048)' in source
    assert '"depth": ("depth",)' in source
    assert '"depth_pressure": ("depth", "pressure")' in source
    assert '"depth_pressure_marker": ("depth", "pressure", "marker")' in source
    assert '"depth_pressure_marker_rgb": ("depth", "pressure", "marker", "rgb")' in source
    assert 'parser.add_argument("--measure-seconds", type=float, default=3.0)' in source
    assert '"--implementation",\n        "ours"' in source
    assert '"--modalities"' in source


def test_ours_modalities_sweep_stops_after_first_oom() -> None:
    source = _source()
    assert 'status="inferred_oom"' in source
    assert 'if status in ("oom", "inferred_oom"):' in source
    assert 'oom_seen = oom' in source
    assert '"stop_variant_after_first_cuda_oom": True' in source


def test_ours_modalities_sweep_writes_expected_artifacts() -> None:
    source = _source()
    for artifact in (
        "aggregate.csv",
        "cumulative_modalities_vram_scaling.png",
        "cumulative_modalities_vram_scaling.pdf",
        "cumulative_modalities_throughput_scaling.png",
        "cumulative_modalities_throughput_scaling.pdf",
        "summary.json",
    ):
        assert artifact in source
