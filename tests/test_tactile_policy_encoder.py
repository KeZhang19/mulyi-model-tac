from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
sys.path.insert(0, str(ROOT / "scripts/tactile_representation"))

from BrainCo_DexHand.tactile_representation.policy import (
    FINGER_ORDER, FrozenTactilePolicyEncoder, TactileObservationHistory,
    build_tactile_encoder, export_policy_encoder_bundle, load_tactile_checkpoint,
    marker_flow_to_features_tensor, prepare_policy_run, validate_policy_contract,
)
from BrainCo_DexHand.tactile_representation.config import CrossModalTactileNetworkCfg
from BrainCo_DexHand.tactile_representation.tri_modal_config import TriModalCrossAutoencoderCfg
from BrainCo_DexHand.tactile_representation.models.latent_alignment import TactileLatentAlignmentNetwork
from BrainCo_DexHand.tactile_representation.training.data import NpzTactileDataset, TactileNormalization
from BrainCo_DexHand.tactile_representation.collection import marker_flow_to_features
from BrainCo_DexHand.tactile_representation.direct_calibration import prepare_direct_calibration
from test_tactile_representation_training import _tiny_dataset


@pytest.fixture
def aligned_towers(tmp_path):
    torch.manual_seed(23)
    kwargs = dict(image_height=32, image_width=48, marker_count=12, d_model=16, num_heads=2,
                  image_base_channels=4, marker_transformer_layers=1, decoder_base_channels=16,
                  marker_summary_tokens=2, ffn_ratio=2, dropout=0.0)
    cfgs = {
        "sim": CrossModalTactileNetworkCfg(**kwargs, fusion_layers=1, predicted_soft_restoration=True),
        "real": TriModalCrossAutoencoderCfg(**kwargs, cross_layers=1, cross_summary_tokens=2),
    }
    types = {"sim": "robust", "real": "tri_modal"}
    towers = {d: build_tactile_encoder(types[d], asdict(c)) for d, c in cfgs.items()}
    aligner = TactileLatentAlignmentNetwork(towers["sim"], towers["real"], latent_dim=16,
                                           projection_dim=7, projection_hidden_dim=11,
                                           projection_layernorm=True).eval()
    norm = TactileNormalization(32, 48, 12, depth_scale_m=0.004, marker_motion_scale_px=3.0)
    sources = {d: {"model_cfg": asdict(c), "normalization": asdict(norm),
                   "model_state": copy.deepcopy(towers[d].state_dict())} for d, c in cfgs.items()}
    # Mimic end-to-end alignment: final tower weights differ from source checkpoints.
    with torch.no_grad():
        for tower in towers.values():
            next(tower.parameters()).add_(0.2)
    alignment = {"model_state": aligner.state_dict(), "args": {
        "projection_dim": 7, "projection_hidden_dim": 11, "projection_layernorm": True,
        "sim_model_type": "robust", "real_model_type": "tri_modal",
    }}
    bundles = {}
    for domain in towers:
        bundles[domain] = tmp_path / f"{domain}.pt"
        export_policy_encoder_bundle(bundles[domain], alignment=alignment, source=sources[domain],
                                     domain=domain, network_type=types[domain], alignment_id="paired-run-23")
    return aligner, sources, alignment, bundles


def raw_inputs(batch=2):
    return {
        "rgb": torch.randint(0, 256, (batch, 5, 3, 32, 48), dtype=torch.uint8),
        "depth_m": torch.rand(batch, 5, 1, 32, 48) * 0.005,
        "marker": torch.cat((torch.rand(batch, 5, 12, 4) * 3, torch.ones(batch, 5, 12, 1)), dim=-1),
        "marker_valid": torch.ones(batch, 5, 12, dtype=torch.bool),
    }


def test_export_preserves_final_aligned_z_for_both_towers(aligned_towers):
    aligner, _, _, paths = aligned_towers
    sample = raw_inputs()
    loaded = {d: FrozenTactilePolicyEncoder(p, domain=d, chunk_size=3) for d, p in paths.items()}
    normalized = loaded["sim"].preprocess(**sample)
    with torch.no_grad():
        expected = aligner(normalized, normalized)
    for domain, encoder in loaded.items():
        actual = encoder(**sample)
        torch.testing.assert_close(actual.flatten(0, 1), expected[f"z_{domain}"], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual.norm(dim=-1), torch.ones(2, 5))
        assert not actual.requires_grad
        assert all(not p.requires_grad for p in encoder.parameters())
        encoder.train(True)
        assert all(not m.training for m in encoder.modules())
        encoder.chunk_size = 20
        torch.testing.assert_close(actual, encoder(**sample), atol=2e-6, rtol=2e-5)


def test_preprocessing_matches_actual_training_dataset(aligned_towers, tmp_path):
    _, _, _, paths = aligned_towers
    encoder = FrozenTactilePolicyEncoder(paths["sim"])
    root = tmp_path / "dataset"
    _tiny_dataset(root)
    dataset = NpzTactileDataset(root, episode_ids=[0], normalization=encoder.normalization, backend="npz")
    with np.load(root / "shards/shard_000000.npz") as shard:
        sample = {"rgb": torch.tensor(shard["rgb"][:1]), "depth_m": torch.tensor(shard["depth_m"][:1]),
                  "marker": torch.tensor(shard["marker_2d"][:1]), "marker_valid": torch.tensor(shard["marker_valid"][:1])}
    actual = encoder.preprocess(**sample)
    for runtime_name, data_name in (("rgb", "rgb"), ("depth", "depth"), ("marker", "marker"), ("marker_valid_mask", "marker_valid")):
        torch.testing.assert_close(actual[runtime_name][0], dataset[0][data_name])


def test_gpu_marker_features_match_collection_with_invalid_values():
    flow = torch.randn(2, 5, 2, 12, 2)
    flow[1, 3, 0, 2, 0] = float("nan")
    valid = torch.ones(2, 5, 12, dtype=torch.bool)
    valid[0, 2, 1] = False
    features, actual_valid = marker_flow_to_features_tensor(flow, valid)
    for b in range(2):
        for f in range(5):
            expected, expected_valid = marker_flow_to_features(flow[b, f].numpy(), valid[b, f].numpy())
            np.testing.assert_array_equal(features[b, f], expected)
            np.testing.assert_array_equal(actual_valid[b, f], expected_valid)


def test_policy_marker_projection_matches_existing_collector_renderer():
    import ast
    from integrate.curved_hydroshear_adapter import RevoCurvedHydroShearAdapter, RevoCurvedHydroShearCfg

    # Load the actual tensor helper without starting Isaac's scene imports.
    runtime_path = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/brainco/visuotactile_runtime.py"
    tree = ast.parse(runtime_path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "project_hydroshear_flow")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(runtime_path), "exec"), namespace)
    adapter = RevoCurvedHydroShearAdapter(RevoCurvedHydroShearCfg(
        width=8, height=8, marker_rows=2, marker_cols=2, marker_margin_x=1, marker_margin_y=1, device="cpu",
    ))
    y, x = torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing="ij")
    points = torch.stack((x * 0.001, y * 0.001, torch.zeros_like(x)), dim=-1)[None].repeat(2, 1, 1, 1)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    depth = torch.zeros(2, 8, 8)
    motion = torch.randn(2, 4, 3) * 0.0001
    actual = namespace["project_hydroshear_flow"](adapter, motion, depth, points, valid, None, None, None)
    expected = adapter.render_displacement_output(motion, depth, points, valid, render_marker_images=False).marker_flow
    torch.testing.assert_close(actual, expected)


def test_no_contact_and_all_invalid_markers_have_finite_latents(aligned_towers):
    *_, paths = aligned_towers
    sample = raw_inputs(1)
    sample["depth_m"].zero_()
    sample["marker"].fill_(float("nan"))
    sample["marker_valid"].zero_()
    for domain, path in paths.items():
        encoder = FrozenTactilePolicyEncoder(path)
        assert torch.isfinite(encoder(**sample)).all()
        recon = encoder.reconstruct(**sample)
        assert all(torch.isfinite(value).all() for value in recon.values())


def test_history_retains_finger_order_and_isolates_partial_reset():
    history = TactileObservationHistory(2, 152, 64, 4, "cpu")
    state = torch.zeros(2, 152)
    z = torch.arange(5.0)[None, :, None].expand(2, 5, 64)
    first = history.append(state, z).clone()
    assert first.shape == (2, 1888)
    torch.testing.assert_close(first[0, 152:472], z[0].flatten())
    history.append(state + 1, z + 1)
    other_before = history.frames[1].clone()
    history.reset([0])
    result = history.append(state + 2, z + 2).reshape(2, 4, 472)
    torch.testing.assert_close(result[0], result[0, -1:].expand(4, -1))
    torch.testing.assert_close(result[1, :-1], other_before[1:])
    assert FINGER_ORDER == ("little", "ring", "middle", "index", "thumb")


def test_missing_lfs_wrong_domain_and_bad_metadata_fail(aligned_towers, tmp_path):
    *_, paths = aligned_towers
    with pytest.raises(FileNotFoundError):
        FrozenTactilePolicyEncoder(tmp_path / "missing.pt")
    pointer = tmp_path / "pointer.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    with pytest.raises(ValueError, match="Git LFS"):
        load_tactile_checkpoint(pointer)
    with pytest.raises(ValueError, match="tower"):
        FrozenTactilePolicyEncoder(paths["real"], domain="sim")
    bundle = load_tactile_checkpoint(paths["sim"])
    bundle["normalization"]["marker_count"] = 99
    with pytest.raises(ValueError, match="disagree"):
        FrozenTactilePolicyEncoder(bundle)


def test_contract_prevents_different_alignment_and_legacy_resume(aligned_towers, tmp_path):
    from types import SimpleNamespace
    *_, paths = aligned_towers
    contract = FrozenTactilePolicyEncoder(paths["sim"]).observation_contract()
    env = SimpleNamespace(unwrapped=SimpleNamespace(tactile_policy_contract=contract))
    prepare_policy_run(env, tmp_path / "run")
    prepare_policy_run(env, tmp_path / "resumed", resume_path=tmp_path / "run/model_10.pt")
    saved = json.loads((tmp_path / "resumed/tactile_policy_contract.json").read_text())
    validate_policy_contract(contract, saved)
    saved["alignment_id"] = "different-training-run"
    with pytest.raises(ValueError, match="alignment_id"):
        validate_policy_contract(contract, saved)
    with pytest.raises(FileNotFoundError, match="legacy"):
        prepare_policy_run(env, tmp_path / "new", resume_path=tmp_path / "legacy/model.pt")


def test_real_tower_accepts_matching_sim_policy_contract(aligned_towers):
    *_, paths = aligned_towers
    sim = FrozenTactilePolicyEncoder(paths["sim"])
    real = FrozenTactilePolicyEncoder(paths["real"])
    saved = sim.observation_contract()
    saved["simulation_encoder_sha256"] = sim.bundle_sha256
    saved["simulation_sensor_assets"] = {"marker_layout": "sim-only-metadata"}
    real.validate_observation_contract(saved)
    sim.validate_observation_contract(saved)
    saved["alignment_id"] = "other-pair"
    with pytest.raises(ValueError, match="alignment_id"):
        real.validate_observation_contract(saved)


def test_direct_calibration_applies_fixed_mount_without_changing_camera_pixels(tmp_path):
    source = ROOT / "assets/revo21_right_touch/marker_positions/vitai_4fingers/marker_positions.npz"
    urdf = ROOT / "assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf"
    output = prepare_direct_calibration(source, urdf, tmp_path)
    with np.load(source) as original, np.load(output) as direct:
        np.testing.assert_array_equal(original["pixels_distorted"], direct["pixels_distorted"])
        offset = np.array([-0.00139151820394552, -0.000576118195242635, -0.0294848218597647])
        np.testing.assert_allclose(direct["middle_camera_origin_link_m"], original["middle_camera_origin_link_m"] + offset, atol=1e-8)
        np.testing.assert_allclose(direct["middle_ray_starts_link_m"], original["middle_ray_starts_link_m"] + offset, atol=1e-8)
        np.testing.assert_array_equal(direct["index_ray_starts_link_m"], original["index_ray_starts_link_m"])


def test_export_cli_supports_policy_and_original_exports(aligned_towers, tmp_path):
    from export_tactile_encoders import main
    _, sources, alignment, _ = aligned_towers
    for domain, source in sources.items():
        torch.save(source, tmp_path / f"source_{domain}.pt")
    torch.save(alignment, tmp_path / "alignment.pt")
    args = ["--sim-checkpoint", str(tmp_path / "source_sim.pt"), "--real-checkpoint", str(tmp_path / "source_real.pt"),
            "--output-dir", str(tmp_path / "exports")]
    main(args)
    assert (tmp_path / "exports/sim_encoder.pt").is_file()
    main(args + ["--alignment-checkpoint", str(tmp_path / "alignment.pt")])
    assert FrozenTactilePolicyEncoder(tmp_path / "exports/sim_policy_encoder.pt").projection_dim == 7
    with pytest.raises(FileExistsError):
        main(args + ["--alignment-checkpoint", str(tmp_path / "alignment.pt")])


def test_unaligned_sim_export_uses_pretrained_features_and_rejects_real_deployment(aligned_towers, tmp_path):
    from export_tactile_encoders import main
    _, sources, _, aligned_paths = aligned_towers
    source_path = tmp_path / "source.pt"
    torch.save(sources["sim"], source_path)
    args = ["--sim-checkpoint", str(source_path), "--unaligned-sim", "--output-dir", str(tmp_path / "baseline")]
    main(args)
    path = tmp_path / "baseline/sim_policy_encoder_unaligned.pt"
    baseline = FrozenTactilePolicyEncoder(path, domain="sim", chunk_size=3)
    source = build_tactile_encoder("robust", sources["sim"]["model_cfg"]).eval()
    source.load_state_dict(sources["sim"]["model_state"], strict=True)
    sample = raw_inputs()
    with torch.no_grad():
        expected = torch.nn.functional.normalize(source.encode(**baseline.preprocess(**sample)), dim=-1)
    actual = baseline(**sample)
    torch.testing.assert_close(actual.flatten(0, 1), expected, atol=2e-6, rtol=2e-5)
    assert actual.shape == (2, 5, 16)
    assert not actual.requires_grad and all(not p.requires_grad for p in baseline.parameters())
    contract = baseline.observation_contract(history_length=1)
    assert contract["alignment_id"] is None
    assert contract["feature"] == "normalized_h"
    assert contract["observation_dim"] == 232
    baseline.validate_observation_contract(contract, history_length=1)
    for domain in ("sim", "real"):
        with pytest.raises(ValueError, match="contract mismatch"):
            FrozenTactilePolicyEncoder(aligned_paths[domain]).validate_observation_contract(contract, history_length=1)
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(SystemExit):
        main(args + ["--alignment-checkpoint", "alignment.pt"])
    bundle = load_tactile_checkpoint(path)
    bundle["domain"] = "real"
    with pytest.raises(ValueError, match="simulation-only"):
        FrozenTactilePolicyEncoder(bundle)


def test_unaligned_sim_bundle_rejects_projection_and_source_mismatches(aligned_towers, tmp_path):
    from BrainCo_DexHand.tactile_representation.policy import export_unaligned_sim_policy_encoder_bundle
    _, sources, _, _ = aligned_towers
    path = tmp_path / "baseline.pt"
    export_unaligned_sim_policy_encoder_bundle(path, source=sources["sim"], source_checkpoint_sha256="a" * 64)
    bundle = load_tactile_checkpoint(path)
    bad = copy.deepcopy(bundle)
    bad["projection_cfg"]["projection_dim"] = 7
    with pytest.raises(ValueError, match="preserve"):
        FrozenTactilePolicyEncoder(bad)
    bad = copy.deepcopy(bundle)
    bad["projection_state"] = {"weight": torch.eye(16)}
    with pytest.raises(RuntimeError, match="Unexpected key"):
        FrozenTactilePolicyEncoder(bad)
    contract = FrozenTactilePolicyEncoder(bundle).observation_contract()
    bundle["source_checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source_checkpoint_sha256"):
        FrozenTactilePolicyEncoder(bundle).validate_observation_contract(contract)


@pytest.mark.parametrize("history_length", [1, 4])
@pytest.mark.parametrize("env_ids", [[], [0, 2], [0, 1, 2]])
def test_history_reset_preserves_pending_rollout_observations(history_length, env_ids):
    # RSL-RL stores an observation only after env.step(), which may reset history.
    with torch.inference_mode():
        history = TactileObservationHistory(3, 152, 8, history_length, "cpu")
        state = torch.arange(3 * 152, dtype=torch.float32).reshape(3, 152) + 1
        tactile = torch.arange(3 * 5 * 8, dtype=torch.float32).reshape(3, 5, 8) + 1
        history.append(state, tactile)
        published = history.append(state + 1, tactile + 1)
        cached = history.frames.flatten(1)
        expected = published.clone()

        history.reset(env_ids)
        # Match the delayed copy into PPO's rollout storage, after the reset.
        stored = torch.empty_like(published)
        stored.copy_(published)
        torch.testing.assert_close(stored, expected, rtol=0, atol=0)
        torch.testing.assert_close(cached, expected, rtol=0, atol=0)
        assert torch.count_nonzero(history.frames[env_ids]) == 0

        history.reset([1])
        history.append(state + 2, tactile + 2)
        torch.testing.assert_close(published, expected, rtol=0, atol=0)
        torch.testing.assert_close(cached, expected, rtol=0, atol=0)
