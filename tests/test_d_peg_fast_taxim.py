"""Real calibrated CPU rendering and task-local reset isolation checks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from BrainCo_DexHand.tactile_representation.simulation_utils import ensure_taxim_scatter


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calibrated():
    ensure_taxim_scatter()
    from integrate.tacex_rgb_adapter import RevoTacExRgbAdapter, RevoTacExRgbCfg
    runtime = load_module("d_peg_fast_runtime_test", ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/d_peg_tactile_runtime.py")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    cfg = RevoTacExRgbCfg(device="cpu", width=320, height=240)
    yield runtime, RevoTacExRgbAdapter, cfg
    torch.set_num_threads(previous_threads)


def depths():
    result = torch.zeros(3, 240, 320)
    result[1, 104:133, 141:180] = .0002
    yy, xx = torch.meshgrid(torch.arange(240), torch.arange(320), indexing="ij")
    result[2] = .0003 * (1 - ((xx - 175) / 27) ** 2 - ((yy - 97) / 18) ** 2).clamp_min(0)
    return result


def test_calibrated_rgb_depth_and_lazy_statistics_are_exact_and_instance_local(calibrated):
    runtime, adapter_type, cfg = calibrated
    baseline = adapter_type(cfg)
    candidate_base = adapter_type(cfg)
    class_normals = type(baseline._taxim)._TaximTorch__generate_normals
    fast = runtime.enable_d_peg_fast_taxim(candidate_base)
    assert runtime.enable_d_peg_fast_taxim(fast) is fast
    assert type(baseline._taxim)._TaximTorch__generate_normals is class_normals
    assert baseline._taxim._TaximTorch__generate_normals.__func__ is class_normals
    source = depths()
    reference = baseline.step(source)
    actual = fast.step(source)
    assert fast.first_contact_rgb_exact is True
    assert torch.equal(actual.tactile_rgb, reference.tactile_rgb)
    assert torch.equal(actual.depth_mm, reference.depth_mm)
    assert torch.equal(actual.taxim_height_map_mm, reference.taxim_height_map_mm)
    assert "max_depth_mm" not in actual.__dict__ and "active_pixels" not in actual.__dict__
    assert torch.equal(actual.max_depth_mm, reference.max_depth_mm)
    assert torch.equal(actual.active_pixels, reference.active_pixels)
    assert torch.equal(baseline.step(source).tactile_rgb, reference.tactile_rgb)
    assert torch.equal(fast.step(source).tactile_rgb, reference.tactile_rgb)


def test_background_does_not_consume_first_real_input_guard(calibrated):
    runtime, adapter_type, cfg = calibrated
    fast = runtime.enable_d_peg_fast_taxim(adapter_type(cfg))
    fast.step(torch.zeros(1, 240, 320))
    assert fast.first_contact_rgb_exact is None
    fast.step(depths()[1:2])
    assert fast.first_contact_rgb_exact is True


def test_opted_in_adapter_survives_same_step_partial_reset(calibrated):
    runtime, adapter_type, cfg = calibrated
    reset_test = load_module("d_peg_reset_test_helpers", ROOT / "tests/test_d_peg_tactile_reset.py")
    term, env, hydro = reset_test.runtime()
    env.cfg = SimpleNamespace(d_peg_fast_taxim=True)
    env._rl_ours_taxim_rgb_adapter = adapter_type(cfg)
    term.component_cfg = {}
    namespace = type(term).__call__.__wrapped__.__globals__
    namespace["tactile_obs"] = SimpleNamespace(_ours_taxim_rgb_adapter=lambda env, cfg: env._rl_ours_taxim_rgb_adapter)
    namespace["enable_d_peg_fast_taxim"] = runtime.enable_d_peg_fast_taxim
    first = term(env, {}).clone()
    installed = env._rl_ours_taxim_rgb_adapter
    term.reset(torch.tensor([1]))
    after = term(env, {})
    assert env._rl_ours_taxim_rgb_adapter is installed
    assert torch.equal(after[[0, 2]], first[[0, 2]])
    assert torch.all(hydro._batch_hydrosoft_forces[5:10] == 2)
    assert torch.equal(term(env, {}), after)
