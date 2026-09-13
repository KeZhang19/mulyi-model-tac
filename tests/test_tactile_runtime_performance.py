"""Regression coverage for batched tactile inference and long-run logging."""

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))

from BrainCo_DexHand.tactile_representation.simulation_utils import project_marker_flow
from BrainCo_DexHand.tactile_representation.models.network import RobustCrossModalTactileNetwork
from BrainCo_DexHand.tactile_representation.config import CrossModalTactileNetworkCfg
from integrate.curved_hydroshear_adapter import RevoCurvedHydroShearAdapter


@pytest.mark.parametrize("batch", [1, 7])
def test_batched_marker_projection_matches_collector_with_invalid_markers(batch):
    torch.manual_seed(17)
    count = 12
    motion = torch.randn(batch, count, 3) * 0.003
    row_axes = torch.randn_like(motion)
    col_axes = torch.randn_like(motion)
    valid = torch.rand(batch, count) > 0.3
    adapter = SimpleNamespace(
        cfg=SimpleNamespace(arrow_scale_px_per_m=247.3),
        _marker_uv_t=torch.rand(count, 2) * 200,
        _batched_marker_samples=lambda *args: (None, None, None, valid, row_axes, col_axes, None),
        _batched_uniform_marker_projection_axes=lambda *args: (row_axes, col_axes),
    )
    actual = project_marker_flow(adapter, motion, None, None, None, None, None, None)
    clean = torch.where(valid[..., None], motion, torch.zeros_like(motion))
    expected = torch.stack([
        RevoCurvedHydroShearAdapter._flow_from_displacement_t(
            adapter, adapter._marker_uv_t, clean[i], row_axes[i], col_axes[i],
        )
        for i in range(batch)
    ])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 0][~valid], actual[:, 1][~valid], rtol=0, atol=0)


@pytest.mark.parametrize("modalities", [("rgb",), ("depth",), ("marker",), ("rgb", "depth", "marker")])
def test_default_modality_mask_matches_explicit_mask(modalities):
    torch.manual_seed(29)
    cfg = CrossModalTactileNetworkCfg(
        image_height=32, image_width=48, marker_count=12, d_model=16,
        num_heads=2, image_base_channels=4, marker_transformer_layers=1,
        fusion_layers=1, decoder_base_channels=16, marker_summary_tokens=2,
        ffn_ratio=2, dropout=0.0,
    )
    model = RobustCrossModalTactileNetwork(cfg).eval()
    inputs = {
        "rgb": torch.rand(2, 3, 32, 48),
        "depth": torch.rand(2, 1, 32, 48),
        "marker": torch.rand(2, 12, 5),
    }
    inputs["marker"][..., 4] = 1
    selected = {k: v for k, v in inputs.items() if k in modalities}
    mask = torch.tensor([k in modalities for k in ("rgb", "depth", "marker")])
    with torch.inference_mode():
        expected = model.encode(**selected, modality_mask=mask)
        actual = model.encode(**selected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if modalities == ("marker",):
        with pytest.raises(ValueError, match="at least one valid modality"):
            model.encode(**selected, marker_valid_mask=torch.zeros(2, 12, dtype=torch.bool))
    else:
        with pytest.raises(ValueError, match="missing input tensors"):
            if len(modalities) < 3:
                model.encode(**selected, modality_mask=torch.ones(3, dtype=torch.bool))
            else:
                model.encode(rgb=inputs["rgb"], modality_mask=torch.ones(3, dtype=torch.bool))


def test_duration_logger_preserves_metrics_and_reports_days(capsys):
    class UpstreamRunner:
        def log(self, locs, width=80, pad=35):
            self.tot_time += locs["collection_time"] + locs["learn_time"]
            print("Mean reward: 12.34")
            print("    Time elapsed: 19:30:00")
            print("             ETA: 19:30:00")

    path = ROOT / "scripts/rsl_rl/duration_logging.py"
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    namespace = dict(
        OnPolicyRunner=UpstreamRunner, redirect_stdout=redirect_stdout,
        io=io, re=re, sys=sys,
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    runner = namespace["DurationLoggingOnPolicyRunner"]()
    runner.tot_time = 163.5 * 3600 - 10
    runner.log(dict(it=100, start_iter=100, num_learning_iterations=2, collection_time=9, learn_time=1))
    output = capsys.readouterr().out
    assert "Mean reward: 12.34" in output
    assert "Time elapsed: 6d 19:30:00" in output
    assert "ETA: 6d 19:30:00" in output
    runner.log(dict(it=101, start_iter=100, num_learning_iterations=2, collection_time=9, learn_time=1))
    assert "ETA: 00:00:00" in capsys.readouterr().out


@pytest.mark.parametrize("print_terminal", [True, False])
def test_duration_logger_supports_distributed_rsl_rl_fork(capsys, print_terminal):
    class UpstreamRunner:
        def log(self, locs, width=80, pad=35, print_terminal=True):
            self.tot_time += locs["collection_time"] + locs["learn_time"]
            self.metrics_written = True
            self.received_options = (width, pad, print_terminal)
            if print_terminal:
                print("Mean reward: 12.34")
                print("Time elapsed: 01:00:00")
                print("ETA: 01:00:00")

    path = ROOT / "scripts/rsl_rl/duration_logging.py"
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    namespace = dict(OnPolicyRunner=UpstreamRunner, redirect_stdout=redirect_stdout,
                     io=io, re=re, sys=sys)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    runner = namespace["DurationLoggingOnPolicyRunner"]()
    runner.tot_time = 90000 - 10
    runner.log(dict(it=0, start_iter=0, num_learning_iterations=2,
                    collection_time=9, learn_time=1), width=100, pad=30,
               print_terminal=print_terminal)
    assert runner.metrics_written and runner.tot_time == 90000
    assert runner.received_options == (100, 30, print_terminal)
    output = capsys.readouterr().out
    if print_terminal:
        assert "Mean reward: 12.34" in output
        assert "Time elapsed: 1d 01:00:00" in output
        assert "ETA: 1d 01:00:00" in output
    else:
        assert output == ""
