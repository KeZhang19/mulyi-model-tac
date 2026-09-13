"""Pretrained checkpoint loading, sensor units/order and manager reset behavior."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_tactile_policy_encoder import aligned_towers, raw_inputs
from BrainCo_DexHand.tactile_representation.policy import (
    FrozenTactilePolicyEncoder, export_unaligned_sim_policy_encoder_bundle, file_sha256,
    load_sim_policy_encoder, marker_flow_to_features_tensor, prepare_policy_run,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/rotate_bulb_tactile.py"


def test_raw_pretrained_checkpoint_matches_exported_encoder(aligned_towers, tmp_path):
    _, sources, _, _ = aligned_towers
    source = tmp_path / "best.pt"
    torch.save(sources["sim"], source)
    bundle = tmp_path / "sim_policy_encoder.pt"
    export_unaligned_sim_policy_encoder_bundle(
        bundle, source=sources["sim"], source_checkpoint_sha256=file_sha256(source),
    )
    loaded = load_sim_policy_encoder(source, chunk_size=3)
    exported = load_sim_policy_encoder(bundle, chunk_size=4)
    sample = raw_inputs()
    torch.testing.assert_close(loaded(**sample), exported(**sample))
    for key, value in loaded.encoder.state_dict().items():
        torch.testing.assert_close(value, sources["sim"]["model_state"][key], rtol=0, atol=0)
    assert loaded.feature_mode == "unaligned_sim" and loaded.projection_dim == 16
    assert loaded.bundle_sha256 == loaded.source_checkpoint_sha256 == file_sha256(source)
    assert all(not p.requires_grad for p in loaded.parameters())
    loaded.train()
    assert not loaded.training and not loaded.encoder.training


def test_default_loader_rejects_missing_lfs_real_and_incomplete_weights(aligned_towers, tmp_path):
    _, sources, _, paths = aligned_towers
    with pytest.raises(FileNotFoundError):
        load_sim_policy_encoder(tmp_path / "absent.pt")
    pointer = tmp_path / "pointer.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS"):
        load_sim_policy_encoder(pointer)
    with pytest.raises(ValueError, match="tower"):
        load_sim_policy_encoder(paths["real"])
    broken = copy.deepcopy(sources["sim"])
    broken["model_state"].pop(next(iter(broken["model_state"])))
    torch.save(broken, tmp_path / "broken.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_sim_policy_encoder(tmp_path / "broken.pt")


@pytest.fixture
def runtime_code():
    class Term:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

    # Execute the production definitions with only the simulator integration
    # types replaced. Tests below supply deterministic sensor outputs.
    tree = ast.parse(RUNTIME.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    ns = dict(torch=torch, copy=copy, Path=Path, ManagerTermBase=Term,
              SceneEntityCfg=lambda name: name, file_sha256=file_sha256,
              load_sim_policy_encoder=load_sim_policy_encoder,
              marker_flow_to_features_tensor=marker_flow_to_features_tensor,
              _POLICY_FINGER_INDICES=(3, 2, 0, 1, 4))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(RUNTIME), "exec"), ns)
    return ns


def test_sensor_adapter_preserves_metric_depth_pixel_flow_and_finger_order(runtime_code):
    n = 2
    rgb = torch.arange(5.0)[None, :, None, None, None].expand(n, 5, 3, 240, 320) / 10
    depth = torch.arange(5.0)[None, :, None, None].expand(n, 5, 240, 320) * .0001
    flow = torch.zeros(n * 5, 2, 100, 2)
    flow[:, 0, :, 0] = torch.arange(n * 5)[:, None] * 10
    flow[:, 1] = flow[:, 0] + torch.tensor([2., -3.])
    valid = torch.ones(n * 5, 100, dtype=torch.bool)
    valid[3, 7] = False
    env = SimpleNamespace(num_envs=n, _brainco_rl_hydroshear_adapter=object(),
                          _rl_ours_dense_tacmap_depth_m=depth,
                          _rl_tacmap_penetration_m=torch.zeros(n, 5, 32, 24),
                          _rl_tacmap_surface_points_w=torch.zeros(n, 5, 32, 24, 3),
                          _rl_tacmap_surface_valid=torch.ones(n, 5, 32, 24, dtype=torch.bool))
    runtime_code["tactile_obs"] = SimpleNamespace(
        ours_rl_hydroshear_obs=lambda *_: torch.zeros(n, 1500),
        ours_rl_taxim_rgb_obs=lambda *_: rgb.flatten(1),
        _hydroshear_calibrated_marker_world=lambda *a, **kw: (None, None, None, valid),
    )
    runtime_code["project_marker_flow"] = lambda *a: flow
    sample = runtime_code["observe_rotate_bulb_tactile"](env, {
        "hydroshear_marker_layout_path": "layout.npz", "hydroshear_marker_finger_link_names": [],
    })
    order = [3, 2, 0, 1, 4]
    torch.testing.assert_close(sample["depth_m"], depth[:, order, None])
    torch.testing.assert_close(sample["rgb"], (rgb[:, order] * 255).round().to(torch.uint8))
    torch.testing.assert_close(sample["marker"][0, :, 0, 0], torch.tensor(order, dtype=torch.float32) * 10)
    torch.testing.assert_close(sample["marker"][0, :, 0, 2:4], torch.tensor([2., -3.]).expand(5, 2))
    assert not sample["marker_valid"][0, 0, 7] and not sample["marker"][0, 0, 7].any()


def test_manager_caches_once_per_step_and_resets_only_selected_shear_state(runtime_code):
    calls, reset_ids = [], []

    def observe(*args):
        calls.append(1)
        return {"feature": torch.ones(2, 5, 4)}

    def invalidate(env, ids):
        env._rl_ours_tactile_reset_epoch += 1

    runtime_code["observe_rotate_bulb_tactile"] = observe
    runtime_code["tactile_obs"] = SimpleNamespace(invalidate_ours_tactile_cache_on_reset=invalidate)
    env = SimpleNamespace(num_envs=2, common_step_counter=0, _rl_ours_tactile_reset_epoch=0,
                          cfg=SimpleNamespace(tactile_reconstruction_diagnostics=False),
                          _brainco_rl_hydroshear_adapter=SimpleNamespace(
                              _batch_state_valid=torch.ones(10, dtype=torch.bool),
                              _reset_hydrosoft_state=reset_ids.append))
    term = object.__new__(runtime_code["RotateBulbPretrainedTactile"])
    term._env, term._cache_key, term._cached = env, None, None
    term.component_cfg, term.encoder = {}, lambda **sample: sample["feature"]
    first = term(env, {})
    assert term(env, {}) is first and len(calls) == 1
    env.common_step_counter += 1
    term(env, {})
    assert len(calls) == 2
    term.reset(torch.tensor([1]))
    assert reset_ids == [5, 6, 7, 8, 9]
    term(env, {})
    assert len(calls) == 3
    # A task reset event also invalidates the cache within the same control step.
    invalidate(env, [0])
    term(env, {})
    assert len(calls) == 4
    reset_ids.clear()
    term.reset()
    assert reset_ids == list(range(10))


def test_manager_contract_serializes_numpy_dimensions_and_guards_resume(runtime_code, aligned_towers, tmp_path):
    _, _, _, paths = aligned_towers
    encoder = FrozenTactilePolicyEncoder(paths["sim"])
    marker = tmp_path / "marker.npz"
    marker.write_bytes(b"marker")
    marker.with_name("camera_ray_rectangles_320x240.json").write_text("{}")
    background = tmp_path / "background.png"
    background.write_bytes(b"background")
    cfg = {"hydroshear_marker_layout_path": str(marker), "taxim_rgb_background_path": str(background),
           "taxim_rgb_render_chunk_size": 8, "tactile_resnet_output_dim": 128, "tacmap_rows": 32}
    env = SimpleNamespace(_rotate_bulb_tactile_term=SimpleNamespace(encoder=encoder, component_cfg=cfg),
                          observation_manager=SimpleNamespace(
                              group_obs_dim={"policy": (np.int64(43),), "proprio": (np.int64(469),), "perception": (np.int64(192),)},
                              active_terms={"policy": ["state"], "proprio": ["joints", "pretrained_tactile"], "perception": ["cloud"]},
                              group_obs_term_dim={"policy": [(np.int64(43),)], "proprio": [(np.int64(434),), (np.int64(35),)], "perception": [(np.int64(192),)]},
                          ))
    env.unwrapped = env
    runtime_code["initialize_rotate_bulb_tactile_contract"](env)
    assert env.tactile_policy_contract["observation_dim"] == 704
    assert env.tactile_policy_contract["sensor_parameters"] == {"tacmap_rows": 32}
    assert json.loads(json.dumps(env.tactile_policy_contract)) == env.tactile_policy_contract
    prepare_policy_run(env, tmp_path / "run")
    prepare_policy_run(env, tmp_path / "resumed", resume_path=tmp_path / "run/model.pt")
    env.tactile_policy_contract["simulation_encoder_sha256"] = "different"
    with pytest.raises(ValueError, match="simulation_encoder_sha256"):
        prepare_policy_run(env, tmp_path / "bad", resume_path=tmp_path / "run/model.pt")
    with pytest.raises(FileNotFoundError, match="legacy"):
        prepare_policy_run(env, tmp_path / "bad", resume_path=tmp_path / "old_resnet/model.pt")


def test_taxim_fallback_reduces_repeated_indices_and_empty_inputs(monkeypatch):
    import builtins
    import sys
    from BrainCo_DexHand.tactile_representation.simulation_utils import ensure_taxim_scatter

    original_import = builtins.__import__

    def without_scatter(name, *args, **kwargs):
        if name == "torch_scatter":
            raise ModuleNotFoundError("missing", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scatter)
    # Record the original entry even when absent, so teardown removes the stub.
    monkeypatch.setitem(sys.modules, "torch_scatter", None)
    ensure_taxim_scatter()
    scatter = sys.modules["torch_scatter"].scatter_min
    out = torch.full((2, 4), torch.inf)
    result, _ = scatter(torch.tensor([[4., 2., 3.], [1., 8., 7.]]), torch.tensor([1, 1, 3]), out=out)
    torch.testing.assert_close(result, torch.tensor([[torch.inf, 2., torch.inf, 3.], [torch.inf, 1., torch.inf, 7.]]))
    empty, _ = scatter(torch.empty(2, 0), torch.empty(0, dtype=torch.long), dim_size=4)
    assert empty.shape == (2, 4) and empty.isinf().all()
