from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from BrainCo_DexHand.tactile_representation import (  # noqa: E402
    CrossModalTactileNetworkCfg,
    RobustCrossModalTactileNetwork,
)
from BrainCo_DexHand.tactile_representation.training import (  # noqa: E402
    CrossModalLossCfg,
    MODALITY_NAMES,
    NpzTactileDataset,
    TactileNormalization,
    build_mmap_cache,
    compute_cross_modal_objective,
    degrade_tactile_inputs,
    load_complete_manifest,
    manifest_sha256,
    restoration_loss,
    sample_degraded_modality_indices,
    split_episode_ids,
)


def _tiny_dataset(root: Path) -> tuple[dict, list[tuple[int, int]]]:
    root.mkdir(parents=True)
    (root / "shards").mkdir()
    height, width, markers = 32, 48, 12
    stages = ("force", "offset_u", "offset_v", "tilt", "slide")
    plan_pairs: list[tuple[int, int]] = []
    plans = []
    episode_id = 0
    for stage_index, stage in enumerate(stages):
        for condition_index in range(3):
            pair = []
            for presser in ("square_4", "cylinder_D4"):
                pair.append(episode_id)
                plans.append(
                    {
                        "episode_id": episode_id,
                        "presser": presser,
                        "target_force_n": float((5, 10, 20)[condition_index]),
                        "offset_u_m": 0.001 * condition_index if stage == "offset_u" else 0.0,
                        "offset_v_m": 0.001 * condition_index if stage == "offset_v" else 0.0,
                        "tilt_axis": "+u" if stage == "tilt" else "none",
                        "tilt_deg": float(condition_index * 5 if stage == "tilt" else 0),
                        "slide_axis": "+v",
                        "slide_distance_m": (
                            0.001 * condition_index if stage == "slide" else 0.0
                        ),
                        "baseline_steps": 1,
                        "press_steps": 1,
                        "slide_steps": 1 if stage == "slide" and condition_index else 0,
                        "hold_steps": 0,
                        "sweep_stage": stage,
                    }
                )
                episode_id += 1
            plan_pairs.append((pair[0], pair[1]))

    count = len(plans)
    rng = np.random.default_rng(3)
    valid = np.ones((count, markers), dtype=bool)
    valid[:, -2:] = False
    marker = np.zeros((count, markers, 5), dtype=np.float32)
    marker[..., 0] = np.linspace(2.0, width - 3.0, markers)
    marker[..., 1] = np.linspace(3.0, height - 4.0, markers)
    marker[..., 2:4] = rng.normal(0.0, 1.0, size=(count, markers, 2))
    marker[..., 4] = valid
    marker[~valid] = 0.0
    arrays = {
        "rgb": rng.integers(0, 256, size=(count, 3, height, width), dtype=np.uint8),
        "depth_m": rng.uniform(0.0, 0.0025, size=(count, 1, height, width)).astype(np.float32),
        "marker_2d": marker,
        "marker_valid": valid,
        "episode_id": np.arange(count, dtype=np.int32),
        "episode_step": np.zeros(count, dtype=np.int32),
        "env_id": np.arange(count, dtype=np.int16) % 4,
        "phase": np.ones(count, dtype=np.int8),
        "contact": np.ones(count, dtype=bool),
        "target_force_n": np.asarray([p["target_force_n"] for p in plans], dtype=np.float32),
        "presser_id": np.asarray([i % 2 for i in range(count)], dtype=np.int8),
        "sweep_stage": np.asarray([i // 6 for i in range(count)], dtype=np.int8),
    }
    shard_path = root / "shards" / "shard_000000.npz"
    np.savez_compressed(shard_path, **arrays)
    schema = {
        name: {"dtype": str(value.dtype), "sample_shape": list(value.shape[1:])}
        for name, value in arrays.items()
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "sample_count": count,
        "shard_count": 1,
        "schema": schema,
        "shards": [{"file": "shards/shard_000000.npz", "sample_count": count}],
        "metadata": {"episode_plans": plans},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, plan_pairs


def test_condition_grouped_episode_split_has_no_leakage(tmp_path):
    manifest, pairs = _tiny_dataset(tmp_path / "dataset")

    splits = split_episode_ids(manifest, seed=11)
    groups = {
        "train": set(splits.train),
        "validation": set(splits.validation),
        "test": set(splits.test),
    }

    assert set().union(*groups.values()) == set(range(30))
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])
    for left, right in pairs:
        assert any(left in episode_ids and right in episode_ids for episode_ids in groups.values())


def test_npz_dataset_normalizes_modalities_and_preserves_metadata(tmp_path):
    manifest, _ = _tiny_dataset(tmp_path / "dataset")
    root, loaded = load_complete_manifest(tmp_path / "dataset")
    splits = split_episode_ids(loaded, seed=5)
    norm = TactileNormalization.from_manifest(manifest, depth_scale_m=0.003)
    dataset = NpzTactileDataset(
        root, episode_ids=splits.train, normalization=norm, cache_size=1
    )

    sample = dataset[0]

    assert tuple(sample["rgb"].shape) == (3, 32, 48)
    assert tuple(sample["rgb_reference"].shape) == (3, 32, 48)
    assert tuple(sample["depth"].shape) == (1, 32, 48)
    assert tuple(sample["marker"].shape) == (12, 5)
    assert sample["rgb"].dtype == torch.float32
    assert 0.0 <= float(sample["rgb"].min()) <= float(sample["rgb"].max()) <= 1.0
    assert dataset.rgb_reference_fallback_episodes
    assert 0.0 <= float(sample["depth"].min()) <= float(sample["depth"].max()) <= 1.0
    assert torch.equal(sample["marker"][..., 4].bool(), sample["marker_valid"])
    assert torch.count_nonzero(sample["marker"][~sample["marker_valid"], :4]) == 0
    with np.load(root / "shards" / "shard_000000.npz", allow_pickle=False) as arrays:
        raw_marker = arrays["marker_2d"][int(sample["episode_id"])]
    assert torch.allclose(
        sample["marker"][..., 2:4],
        torch.from_numpy(raw_marker[..., 2:4] / norm.marker_motion_scale_px),
    )
    assert int(sample["episode_id"]) in splits.train
    assert len(manifest_sha256(root)) == 64
    assert dataset.backend == "npz"


def test_mmap_cache_matches_npz_backend_and_auto_selects_it(tmp_path):
    manifest, _ = _tiny_dataset(tmp_path / "dataset")
    root, loaded = load_complete_manifest(tmp_path / "dataset")
    splits = split_episode_ids(loaded, seed=5)
    norm = TactileNormalization.from_manifest(manifest, depth_scale_m=0.003)
    npz_dataset = NpzTactileDataset(
        root,
        episode_ids=splits.train,
        normalization=norm,
        cache_size=1,
        backend="npz",
    )

    cache_root = build_mmap_cache(root)
    assert cache_root == root / "mmap"
    assert build_mmap_cache(root) == cache_root
    mmap_dataset = NpzTactileDataset(
        root,
        episode_ids=splits.train,
        normalization=norm,
        cache_size=1,
        backend="auto",
    )

    assert mmap_dataset.backend == "mmap"
    assert mmap_dataset.episode_ids == npz_dataset.episode_ids
    assert len(mmap_dataset) == len(npz_dataset)
    for index in (0, len(mmap_dataset) - 1):
        expected = npz_dataset[index]
        actual = mmap_dataset[index]
        assert actual.keys() == expected.keys()
        for name in actual:
            assert torch.equal(actual[name], expected[name]), name


def test_required_mmap_backend_rejects_missing_or_stale_cache(tmp_path):
    manifest, _ = _tiny_dataset(tmp_path / "dataset")
    root, loaded = load_complete_manifest(tmp_path / "dataset")
    splits = split_episode_ids(loaded, seed=5)
    norm = TactileNormalization.from_manifest(manifest, depth_scale_m=0.003)

    with pytest.raises(FileNotFoundError, match="prepare_mmap_cache"):
        NpzTactileDataset(
            root,
            episode_ids=splits.train,
            normalization=norm,
            backend="mmap",
        )

    build_mmap_cache(root)
    manifest["metadata"]["cache_probe"] = True
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different dataset manifest"):
        NpzTactileDataset(
            root,
            episode_ids=splits.train,
            normalization=norm,
            backend="mmap",
        )


def test_incomplete_dataset_is_rejected(tmp_path):
    manifest, _ = _tiny_dataset(tmp_path / "dataset")
    manifest["status"] = "interrupted"
    (tmp_path / "dataset" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not complete"):
        load_complete_manifest(tmp_path / "dataset")


def test_degraded_modality_sampling_is_count_balanced():
    generator = torch.Generator().manual_seed(17)
    indices = sample_degraded_modality_indices(512, "cpu", generator=generator)
    counts = torch.bincount(indices, minlength=len(MODALITY_NAMES))

    assert set(indices.tolist()) == {0, 1, 2}
    assert int(counts.max() - counts.min()) <= 1


def test_learning_rate_schedule_supports_truly_fixed_mode():
    trainer = runpy.run_path(str(REPO_ROOT / "scripts/tactile_representation/train_cross_modal.py"))
    lr_lambda = trainer["_lr_lambda"]

    assert lr_lambda(0, schedule="fixed", epochs=1000, warmup_epochs=5) == 1.0
    assert lr_lambda(999, schedule="fixed", epochs=1000, warmup_epochs=5) == 1.0
    assert lr_lambda(0, schedule="cosine", epochs=1000, warmup_epochs=5) == 0.2
    assert lr_lambda(4, schedule="cosine", epochs=1000, warmup_epochs=5) == 1.0
    assert lr_lambda(999, schedule="cosine", epochs=1000, warmup_epochs=5) < 0.02
    with pytest.raises(ValueError, match="Unsupported learning-rate schedule"):
        lr_lambda(0, schedule="unknown", epochs=1000, warmup_epochs=0)


def _network_batch(batch_size: int = 3) -> tuple[dict[str, torch.Tensor], TactileNormalization]:
    norm = TactileNormalization(image_height=32, image_width=48, marker_count=12)
    valid = torch.ones(batch_size, 12, dtype=torch.bool)
    valid[:, -2:] = False
    marker = torch.rand(batch_size, 12, 5)
    marker[..., 2:4] = marker[..., 2:4] * 1.2 - 0.6
    marker[..., 4] = valid
    marker[..., :4].masked_fill_(~valid.unsqueeze(-1), 0.0)
    return (
        {
            "rgb": torch.rand(batch_size, 3, 32, 48),
            "depth": torch.rand(batch_size, 1, 32, 48),
            "marker": marker,
            "marker_valid": valid,
            "contact": torch.arange(batch_size) > 0,
        },
        norm,
    )


def test_degradation_keeps_all_inputs_and_changes_exactly_one_observation_per_row():
    batch, _ = _network_batch(batch_size=2)
    batch["marker_valid"][:] = False
    batch["marker_valid"][:, 0] = True
    batch["marker"][..., 4] = batch["marker_valid"]

    degraded, flags, quality, severity = degrade_tactile_inputs(
        batch,
        normalization=TactileNormalization(image_height=32, image_width=48, marker_count=12),
        generator=torch.Generator().manual_seed(2),
    )

    assert torch.equal(flags.sum(dim=1), torch.ones(2, dtype=torch.int64))
    assert bool((quality >= 0.0).all() and (quality <= 1.0).all())
    assert bool((quality.masked_select(~flags) == 1.0).all())
    assert bool((severity >= 0.15).all() and (severity <= 0.45).all())
    assert bool(degraded["marker_valid"].any(dim=1).all())


def test_clean_degradation_rows_preserve_every_observation_and_target_full_quality():
    batch, norm = _network_batch(batch_size=4)

    degraded, flags, quality, severity = degrade_tactile_inputs(
        batch,
        normalization=norm,
        clean_probability=1.0,
        generator=torch.Generator().manual_seed(19),
    )

    assert not bool(flags.any())
    assert torch.equal(quality, torch.ones_like(quality))
    assert torch.equal(severity, torch.zeros_like(severity))
    assert torch.equal(degraded["rgb"], batch["rgb"])
    assert torch.equal(degraded["depth"], batch["depth"])
    assert torch.equal(degraded["marker"], batch["marker"])
    assert torch.equal(degraded["marker_valid"], batch["marker_valid"])


def test_degradation_respects_explicit_severity_bounds():
    batch, norm = _network_batch(batch_size=64)

    _, _, quality, severity = degrade_tactile_inputs(
        batch,
        normalization=norm,
        min_severity=0.10,
        max_severity=0.20,
        generator=torch.Generator().manual_seed(31),
    )

    assert bool((severity >= 0.10).all() and (severity <= 0.20).all())
    assert torch.allclose(quality.min(dim=1).values, 1.0 - severity)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    ((-0.1, 0.2), (0.4, 0.2), (0.2, 1.1)),
)
def test_degradation_rejects_invalid_severity_bounds(minimum, maximum):
    batch, norm = _network_batch(batch_size=2)

    with pytest.raises(ValueError, match="severity bounds"):
        degrade_tactile_inputs(
            batch,
            normalization=norm,
            min_severity=minimum,
            max_severity=maximum,
        )


def test_default_degradation_is_mild_for_every_modality():
    batch, norm = _network_batch(batch_size=64)

    degraded_rgb, _, _, _ = degrade_tactile_inputs(
        batch,
        normalization=norm,
        degraded_modality=0,
        generator=torch.Generator().manual_seed(41),
    )
    degraded_depth, _, _, _ = degrade_tactile_inputs(
        batch,
        normalization=norm,
        degraded_modality=1,
        generator=torch.Generator().manual_seed(42),
    )
    degraded_marker, _, _, _ = degrade_tactile_inputs(
        batch,
        normalization=norm,
        degraded_modality=2,
        generator=torch.Generator().manual_seed(43),
    )

    rgb_mae = (degraded_rgb["rgb"] - batch["rgb"]).abs().mean()
    depth_mae = (degraded_depth["depth"] - batch["depth"]).abs().mean()
    common_marker_valid = batch["marker_valid"] & degraded_marker["marker_valid"]
    marker_epe_px = torch.linalg.vector_norm(
        (degraded_marker["marker"][..., 2:4] - batch["marker"][..., 2:4])
        * norm.marker_motion_scale_px,
        dim=-1,
    )[common_marker_valid].mean()
    marker_dropout_fraction = 1.0 - (
        degraded_marker["marker_valid"].sum() / batch["marker_valid"].sum()
    )

    assert 0.01 < float(rgb_mae) < 0.06
    assert 0.005 < float(depth_mae) < 0.05
    assert 1.5 < float(marker_epe_px) < 5.0
    assert float(marker_dropout_fraction) == 0.0


def test_marker_degradation_preserves_static_calibration_coordinates():
    batch, norm = _network_batch(batch_size=2)

    degraded, flags, _, _ = degrade_tactile_inputs(
        batch,
        normalization=norm,
        degraded_modality=2,
        generator=torch.Generator().manual_seed(8),
    )

    assert bool(flags[:, 2].all())
    assert torch.equal(degraded["marker"][..., :2], batch["marker"][..., :2])


def test_marker_restoration_targets_only_dx_dy():
    batch, norm = _network_batch(batch_size=2)
    output = {
        "rgb_recon": batch["rgb"].clone(),
        "depth_recon": batch["depth"].clone(),
        "marker_recon": batch["marker"][..., 2:4].clone(),
    }
    degradation_flags = torch.tensor([[False, False, True], [False, False, True]])

    _, metrics_before = restoration_loss(
        output, batch, degradation_flags, norm, CrossModalLossCfg()
    )
    batch_with_other_calibration = {name: value.clone() for name, value in batch.items()}
    batch_with_other_calibration["marker"][..., :2] = torch.rand_like(
        batch_with_other_calibration["marker"][..., :2]
    )
    _, metrics_after = restoration_loss(
        output,
        batch_with_other_calibration,
        degradation_flags,
        norm,
        CrossModalLossCfg(),
    )

    assert output["marker_recon"].shape[-1] == 2
    assert torch.allclose(metrics_before["loss/restore_marker"], torch.tensor(0.0))
    assert torch.equal(
        metrics_before["loss/restore_marker"], metrics_after["loss/restore_marker"]
    )


def test_marker_direction_loss_penalizes_reversed_visible_motion():
    batch, norm = _network_batch(batch_size=1)
    batch["marker"][..., 2] = 0.4
    batch["marker"][..., 3] = 0.0
    output = {
        "rgb_recon": batch["rgb"].clone(),
        "depth_recon": batch["depth"].clone(),
        "marker_recon": -batch["marker"][..., 2:4].clone(),
    }
    degradation_flags = torch.tensor([[False, False, True]])

    without_direction, _ = restoration_loss(
        output,
        batch,
        degradation_flags,
        norm,
        CrossModalLossCfg(marker_motion_weight=0.0, marker_direction_weight=0.0),
    )
    with_direction, metrics = restoration_loss(
        output,
        batch,
        degradation_flags,
        norm,
        CrossModalLossCfg(marker_motion_weight=0.0, marker_direction_weight=1.0),
    )

    assert torch.allclose(without_direction, torch.tensor(0.0))
    assert with_direction > without_direction
    assert torch.allclose(metrics["loss/restore_marker_direction"], torch.tensor(2.0))
    assert torch.allclose(metrics["metric/marker_direction_error_rate"], torch.tensor(1.0))


def test_rgb_reference_residual_loss_emphasizes_changed_pixels():
    batch, norm = _network_batch(batch_size=1)
    batch["rgb"].zero_()
    batch["rgb"][:, :, :8, :8] = 0.5
    reference = torch.zeros_like(batch["rgb"])
    output = {
        "rgb_recon": torch.zeros_like(batch["rgb"]),
        "depth_recon": batch["depth"].clone(),
        "marker_recon": batch["marker"][..., 2:4].clone(),
    }
    degradation_flags = torch.tensor([[True, False, False]])

    uniform_cfg = CrossModalLossCfg(rgb_residual_loss_weight=0.0)
    weighted_cfg = CrossModalLossCfg(
        rgb_residual_loss_weight=0.75,
        rgb_change_threshold=0.02,
        rgb_change_boost=4.0,
    )
    uniform_loss, uniform_metrics = restoration_loss(
        output, batch, degradation_flags, norm, uniform_cfg, rgb_reference=reference
    )
    weighted_loss, weighted_metrics = restoration_loss(
        output, batch, degradation_flags, norm, weighted_cfg, rgb_reference=reference
    )

    assert weighted_loss > uniform_loss
    assert weighted_metrics["metric/rgb_mae"] == uniform_metrics["metric/rgb_mae"]
    assert weighted_metrics["metric/rgb_change_mae"] > 0.0
    assert weighted_metrics["metric/rgb_residual_mae"] > uniform_metrics["metric/rgb_mae"]


def test_depth_structure_loss_reports_sparse_foreground_quality():
    batch, norm = _network_batch(batch_size=1)
    batch["depth"].zero_()
    batch["depth"][:, :, 8:12, 10:14] = 0.2
    output = {
        "rgb_recon": batch["rgb"].clone(),
        "depth_recon": torch.zeros_like(batch["depth"]),
        "marker_recon": batch["marker"][..., 2:4].clone(),
    }
    degradation_flags = torch.tensor([[False, True, False]])

    loss, metrics = restoration_loss(
        output,
        batch,
        degradation_flags,
        norm,
        CrossModalLossCfg(
            depth_structure_weight=1.0,
            depth_foreground_threshold=0.02,
            depth_mask_temperature=0.02,
        ),
    )

    assert torch.isfinite(loss)
    assert metrics["loss/restore_depth_structure"] > 0.0
    assert metrics["metric/depth_foreground_mae"] > 0.0
    assert torch.allclose(metrics["metric/depth_foreground_iou"], torch.tensor(0.0))


def test_restoration_objective_is_finite_and_backpropagates_through_all_heads():
    batch, norm = _network_batch()
    cfg = CrossModalTactileNetworkCfg(
        image_height=32,
        image_width=48,
        marker_count=12,
        d_model=64,
        num_heads=4,
        image_base_channels=8,
        decoder_base_channels=32,
        marker_transformer_layers=1,
        fusion_layers=1,
        ffn_ratio=2,
    )
    model = RobustCrossModalTactileNetwork(cfg).train()
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    loss, metrics = compute_cross_modal_objective(
        model,
        batch,
        teacher_model=teacher,
        normalization=norm,
        cfg=CrossModalLossCfg(),
        generator=torch.Generator().manual_seed(9),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(
        (
            "loss/total",
            "loss/restoration",
            "loss/quality",
            "metric/degraded_modality_accuracy",
        )
    ) <= set(metrics)
    assert torch.allclose(metrics["metric/modalities_per_restoration"], torch.tensor(2.0))
    gradients = (
        model.rgb_encoder.stem[0].weight.grad,
        model.depth_encoder.stem[0].weight.grad,
        model.marker_encoder.motion_mlp[0].weight.grad,
        model.reliability_heads["rgb"].mlp[0].weight.grad,
        model.fusion.layers[0].attention.query_projection.weight.grad,
        model.rgb_decoder.output_head.weight.grad,
        model.depth_decoder.output_head.weight.grad,
        model.marker_decoder.output_head[-1].weight.grad,
    )
    assert all(value is not None for value in gradients)
    assert all(torch.isfinite(value).all() for value in gradients)
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_clean_latent_sufficiency_loss_backpropagates_without_restoration_loss():
    batch, norm = _network_batch()
    cfg = CrossModalTactileNetworkCfg(
        image_height=32,
        image_width=48,
        marker_count=12,
        d_model=64,
        num_heads=4,
        image_base_channels=8,
        decoder_base_channels=32,
        marker_transformer_layers=1,
        fusion_layers=1,
        ffn_ratio=2,
        rgb_depth_spatial_skip=True,
        depth_rgb_spatial_skip=True,
        marker_static_context=True,
        marker_image_spatial_context=True,
        cross_modal_reliability=True,
        predicted_soft_restoration=True,
        detach_reliability_gate=True,
    )
    model = RobustCrossModalTactileNetwork(cfg).train()
    loss_cfg = CrossModalLossCfg(
        consistency_weight=0.0,
        quality_weight=0.0,
        clean_identity_weight=0.0,
        latent_sufficiency_weight=0.1,
    )

    loss, metrics = compute_cross_modal_objective(
        model,
        batch,
        normalization=norm,
        cfg=loss_cfg,
        clean_probability=1.0,
        generator=torch.Generator().manual_seed(10),
    )
    loss.backward()

    assert torch.allclose(metrics["loss/candidate_restoration"], torch.tensor(0.0))
    assert metrics["loss/latent_sufficiency"] > 0.0
    assert torch.allclose(
        loss,
        0.1 * metrics["loss/latent_sufficiency"],
        atol=1.0e-6,
    )
    assert metrics["latent/loss/restore_rgb"] > 0.0
    assert metrics["latent/loss/restore_depth"] > 0.0
    assert metrics["latent/loss/restore_marker"] > 0.0
    gradients = (
        model.rgb_encoder.stem[0].weight.grad,
        model.depth_encoder.stem[0].weight.grad,
        model.marker_encoder.motion_mlp[0].weight.grad,
        model.fusion.layers[0].attention.query_projection.weight.grad,
        model.rgb_decoder.output_head.weight.grad,
        model.depth_decoder.output_head.weight.grad,
        model.marker_decoder.output_head[-1].weight.grad,
    )
    assert all(value is not None for value in gradients)
    assert all(torch.isfinite(value).all() for value in gradients)


def test_rgb_degradation_restores_from_depth_and_marker():
    batch, norm = _network_batch(batch_size=1)
    cfg = CrossModalTactileNetworkCfg(
        image_height=32,
        image_width=48,
        marker_count=12,
        d_model=32,
        num_heads=4,
        image_base_channels=4,
        decoder_base_channels=16,
        marker_transformer_layers=1,
        fusion_layers=1,
        ffn_ratio=2,
    )
    model = RobustCrossModalTactileNetwork(cfg).train()

    loss, metrics = compute_cross_modal_objective(
        model,
        batch,
        normalization=norm,
        cfg=CrossModalLossCfg(),
        degraded_modality=0,
        generator=torch.Generator().manual_seed(21),
    )
    loss.backward()

    assert metrics["loss/restore_rgb"] > 0.0
    assert torch.allclose(metrics["metric/degraded_rgb_fraction"], torch.tensor(1.0))
    assert torch.allclose(metrics["metric/modalities_per_restoration"], torch.tensor(2.0))
    assert model.depth_encoder.stem[0].weight.grad is not None
    assert model.marker_encoder.motion_mlp[0].weight.grad is not None
    assert model.reliability_heads["rgb"].mlp[0].weight.grad is not None
    assert model.rgb_decoder.output_head.weight.grad is not None


def test_predicted_objective_never_converts_damage_label_into_input_mask():
    batch, norm = _network_batch(batch_size=3)
    cfg = CrossModalTactileNetworkCfg(
        image_height=32,
        image_width=48,
        marker_count=12,
        d_model=32,
        num_heads=4,
        image_base_channels=4,
        decoder_base_channels=16,
        marker_transformer_layers=1,
        fusion_layers=1,
        ffn_ratio=2,
        cross_modal_reliability=True,
        predicted_soft_restoration=True,
        detach_reliability_gate=True,
    )
    model = RobustCrossModalTactileNetwork(cfg).train()
    observed_masks = []
    original_forward = model.forward

    def tracked_forward(*args, **kwargs):
        observed_masks.append(kwargs["modality_mask"].detach().clone())
        return original_forward(*args, **kwargs)

    model.forward = tracked_forward
    loss, metrics = compute_cross_modal_objective(
        model,
        batch,
        normalization=norm,
        cfg=CrossModalLossCfg(),
        generator=torch.Generator().manual_seed(71),
    )
    loss.backward()

    assert observed_masks
    assert all(bool(mask.all()) for mask in observed_masks)
    assert torch.allclose(metrics["metric/modalities_per_restoration"], torch.tensor(3.0))
    assert "loss/candidate_restoration" in metrics
    assert "loss/end_to_end_restoration" in metrics
    assert "loss/oracle_upper_bound_restoration" not in metrics


def test_predicted_objective_trains_clean_no_restoration_behavior():
    batch, norm = _network_batch(batch_size=3)
    cfg = CrossModalTactileNetworkCfg(
        image_height=32,
        image_width=48,
        marker_count=12,
        d_model=32,
        num_heads=4,
        image_base_channels=4,
        decoder_base_channels=16,
        marker_transformer_layers=1,
        fusion_layers=1,
        ffn_ratio=2,
        cross_modal_reliability=True,
        predicted_soft_restoration=True,
        detach_reliability_gate=True,
    )
    model = RobustCrossModalTactileNetwork(cfg).train()
    for head in model.reliability_heads.values():
        for parameter in head.parameters():
            parameter.data.zero_()
        head.mlp[-1].bias.data.fill_(-8.0)

    loss, metrics = compute_cross_modal_objective(
        model,
        batch,
        normalization=norm,
        cfg=CrossModalLossCfg(clean_identity_weight=1.0),
        clean_probability=1.0,
        generator=torch.Generator().manual_seed(73),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.allclose(metrics["metric/clean_fraction"], torch.tensor(1.0))
    assert torch.allclose(
        metrics["metric/clean_false_restoration_rate"], torch.tensor(1.0)
    )
    assert torch.allclose(metrics["metric/degraded_rgb_fraction"], torch.tensor(0.0))
    assert torch.allclose(metrics["loss/candidate_restoration"], torch.tensor(0.0))
    assert metrics["loss/clean_identity"] > 0.0
    assert metrics["metric/clean_quality_mae"] > 0.9
    assert model.reliability_heads["rgb"].mlp[-1].bias.grad is not None


def test_oracle_mask_is_reported_only_as_an_explicit_upper_bound():
    batch, norm = _network_batch(batch_size=2)
    cfg = CrossModalTactileNetworkCfg(
        image_height=32,
        image_width=48,
        marker_count=12,
        d_model=32,
        num_heads=4,
        image_base_channels=4,
        decoder_base_channels=16,
        marker_transformer_layers=1,
        fusion_layers=1,
        ffn_ratio=2,
        cross_modal_reliability=True,
        predicted_soft_restoration=True,
        detach_reliability_gate=True,
    )
    model = RobustCrossModalTactileNetwork(cfg).eval()

    with torch.no_grad():
        _, metrics = compute_cross_modal_objective(
            model,
            batch,
            normalization=norm,
            cfg=CrossModalLossCfg(),
            degraded_modality=2,
            report_oracle_upper_bound=True,
            generator=torch.Generator().manual_seed(72),
        )

    assert "loss/oracle_upper_bound_restoration" in metrics
    assert torch.isfinite(metrics["loss/oracle_upper_bound_restoration"])
