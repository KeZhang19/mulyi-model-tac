from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_UTILS_PATH = (
    REPO_ROOT
    / "source"
    / "BrainCo_DexHand"
    / "BrainCo_DexHand"
    / "tactile_representation"
    / "collection.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_tactile_collection_utils", COLLECTION_UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_COLLECTION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COLLECTION
_SPEC.loader.exec_module(_COLLECTION)
NpzShardWriter = _COLLECTION.NpzShardWriter
batch_episode_plans = _COLLECTION.batch_episode_plans
make_cross_modal_sample = _COLLECTION.make_cross_modal_sample
make_episode_plans = _COLLECTION.make_episode_plans
make_sweep_episode_plans = _COLLECTION.make_sweep_episode_plans
marker_flow_to_features = _COLLECTION.marker_flow_to_features
merge_part_datasets = _COLLECTION.merge_part_datasets


def test_episode_plans_are_deterministic_varied_and_bounded():
    kwargs = dict(
        episode_count=9,
        seed=17,
        pressers=("square_4", "cylinder_D4", "ball_probe"),
        force_levels_n=(5.0, 20.0, 100.0),
        offset_u_range_m=0.003,
        offset_v_range_m=0.002,
        tilt_max_deg=15.0,
        tilt_probability=1.0,
        slide_max_m=0.004,
        slide_probability=1.0,
        baseline_steps=3,
        press_steps=10,
        slide_steps=4,
        hold_steps=2,
    )
    first = make_episode_plans(**kwargs)
    second = make_episode_plans(**kwargs)

    assert first == second
    assert {plan.presser for plan in first} == {"square_4", "cylinder_D4", "ball_probe"}
    assert {plan.target_force_n for plan in first} == {5.0, 20.0, 100.0}
    assert all(abs(plan.offset_u_m) <= 0.003 for plan in first)
    assert all(abs(plan.offset_v_m) <= 0.002 for plan in first)
    assert all(0.0 <= plan.tilt_deg <= 15.0 for plan in first)
    assert all(0.0 <= plan.slide_distance_m <= 0.004 for plan in first)


def _sweep_plans(episode_count=None):
    return make_sweep_episode_plans(
        episode_count=episode_count,
        pressers=("square_4", "cylinder_D4"),
        force_levels_n=(5.0, 10.0, 20.0, 40.0),
        reference_force_n=10.0,
        offset_u_range_m=0.003,
        offset_v_range_m=0.003,
        offset_step_m=0.001,
        tilt_max_deg=15.0,
        tilt_step_deg=5.0,
        slide_max_m=0.004,
        slide_step_m=0.001,
        baseline_steps=3,
        press_steps=10,
        slide_steps=4,
        hold_steps=2,
    )


def test_sweep_protocol_is_regular_balanced_and_one_factor_at_a_time():
    plans = _sweep_plans()

    assert len(plans) == 96
    assert [plan.episode_id for plan in plans] == list(range(96))
    assert [plan.presser for plan in plans[:8]] == ["square_4", "cylinder_D4"] * 4
    assert [plan.target_force_n for plan in plans[:8:2]] == [5.0, 10.0, 20.0, 40.0]

    offset_u = [plan for plan in plans if plan.presser == "square_4" and plan.sweep_stage == "offset_u"]
    offset_v = [plan for plan in plans if plan.presser == "square_4" and plan.sweep_stage == "offset_v"]
    np.testing.assert_allclose(
        [plan.offset_u_m for plan in offset_u],
        [-0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003],
    )
    np.testing.assert_allclose(
        [plan.offset_v_m for plan in offset_v],
        [-0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003],
    )
    assert all(plan.target_force_n == 10.0 for plan in offset_u + offset_v)
    assert all(plan.offset_v_m == 0.0 for plan in offset_u)
    assert all(plan.offset_u_m == 0.0 for plan in offset_v)

    tilt = [plan for plan in plans if plan.presser == "square_4" and plan.sweep_stage == "tilt"]
    slide = [plan for plan in plans if plan.presser == "square_4" and plan.sweep_stage == "slide"]
    assert [(plan.tilt_axis, plan.tilt_deg) for plan in tilt[:4]] == [
        ("none", 0.0),
        ("+u", 5.0),
        ("+u", 10.0),
        ("+u", 15.0),
    ]
    np.testing.assert_allclose(
        [plan.slide_distance_m for plan in slide[:5]],
        [0.0, 0.001, 0.002, 0.003, 0.004],
    )
    assert all(plan.offset_u_m == plan.offset_v_m == 0.0 for plan in tilt + slide)
    assert all(plan.slide_steps == 0 for plan in slide if plan.slide_distance_m == 0.0)
    assert all(plan.slide_steps == 4 for plan in slide if plan.slide_distance_m > 0.0)


def test_sweep_episode_limit_is_a_deterministic_prefix():
    limited = _sweep_plans(episode_count=11)
    full = _sweep_plans()

    assert limited == full[:11]


def test_sweep_rejects_non_regular_steps():
    with pytest.raises(ValueError, match="exact multiple"):
        make_sweep_episode_plans(
            episode_count=None,
            pressers=("square_4",),
            force_levels_n=(5.0,),
            reference_force_n=5.0,
            offset_u_range_m=0.003,
            offset_v_range_m=0.003,
            offset_step_m=0.002,
            tilt_max_deg=0.0,
            tilt_step_deg=5.0,
            slide_max_m=0.0,
            slide_step_m=0.001,
            baseline_steps=1,
            press_steps=1,
            slide_steps=0,
            hold_steps=0,
        )


def test_episode_batches_keep_schedules_compatible_and_cover_every_plan_once():
    plans = _sweep_plans()
    waves = batch_episode_plans(plans, batch_size=4)
    flattened = [plan for wave in waves for plan in wave]

    assert sorted(plan.episode_id for plan in flattened) == list(range(len(plans)))
    assert all(1 <= len(wave) <= 4 for wave in waves)
    assert all(len({plan.presser for plan in wave}) == 1 for wave in waves)
    assert all(
        len(
            {
                (plan.baseline_steps, plan.press_steps, plan.slide_steps, plan.hold_steps)
                for plan in wave
            }
        )
        == 1
        for wave in waves
    )


def test_marker_flow_is_packed_as_start_delta_and_validity():
    flow = np.asarray(
        [
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            [[13.0, 18.0], [31.0, 44.0], [np.nan, 61.0]],
        ],
        dtype=np.float32,
    )
    features, valid = marker_flow_to_features(flow, np.asarray([True, False, True]))

    np.testing.assert_allclose(features[0], [10.0, 20.0, 3.0, -2.0, 1.0])
    np.testing.assert_array_equal(features[1], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(features[2], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(valid, [True, False, False])


def _sample(value: int, marker_count: int = 4):
    rgb = np.full((3, 8, 12), value / 255.0, dtype=np.float32)
    depth = np.full((8, 12), value * 1.0e-6, dtype=np.float32)
    start = np.stack((np.arange(marker_count), np.arange(marker_count) + 10), axis=-1).astype(np.float32)
    flow = np.stack((start, start + 1.0), axis=0)
    return make_cross_modal_sample(
        rgb_chw=rgb,
        depth_m=depth,
        marker_flow=flow,
        marker_valid=np.ones(marker_count, dtype=bool),
        marker_displacement_3d_m=np.zeros((marker_count, 3), dtype=np.float32),
        scalar_fields={"episode_id": np.int32(value), "contact": np.bool_(value > 0)},
        expected_marker_count=marker_count,
    )


def test_sample_normalization_preserves_network_ready_shapes():
    sample = _sample(128)

    assert sample["rgb"].shape == (3, 8, 12)
    assert sample["rgb"].dtype == np.uint8
    assert sample["depth_m"].shape == (1, 8, 12)
    assert sample["marker_2d"].shape == (4, 5)
    assert sample["marker_valid"].shape == (4,)
    assert sample["marker_displacement_3d_m"].shape == (4, 3)


def test_npz_writer_shards_atomically_and_records_manifest(tmp_path: Path):
    output = tmp_path / "dataset"
    writer = NpzShardWriter(
        output,
        shard_size=2,
        compressed=True,
        metadata={"finger": "index"},
    )
    writer.append(_sample(1))
    writer.append(_sample(2))
    writer.append(_sample(3))
    writer.close()

    with (output / "manifest.json").open("r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    assert manifest["status"] == "complete"
    assert manifest["sample_count"] == 3
    assert manifest["shard_count"] == 2
    assert manifest["metadata"]["finger"] == "index"

    with np.load(output / "shards" / "shard_000000.npz", allow_pickle=False) as shard:
        assert shard["rgb"].shape == (2, 3, 8, 12)
        assert shard["depth_m"].shape == (2, 1, 8, 12)
        assert shard["marker_2d"].shape == (2, 4, 5)
        assert shard["episode_id"].shape == (2,)
        assert shard["contact"].shape == (2,)


def test_npz_writer_refuses_nonempty_output(tmp_path: Path):
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        NpzShardWriter(output, shard_size=2, compressed=False, metadata={})


def test_part_manifests_are_merged_without_copying_shards(tmp_path: Path):
    root = tmp_path / "dataset"
    part_dirs = [root / "parts" / "square_4", root / "parts" / "ball_probe"]
    for index, part in enumerate(part_dirs):
        writer = NpzShardWriter(part, shard_size=2, compressed=False, metadata={"part": index})
        writer.append(_sample(index + 1))
        writer.close()

    manifest_path = merge_part_datasets(root, part_dirs, metadata={"finger": "index"})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "complete"
    assert manifest["sample_count"] == 2
    assert manifest["shard_count"] == 2
    assert manifest["metadata"]["finger"] == "index"
    assert manifest["shards"][0]["file"].startswith("parts/square_4/shards/")
    assert all((root / shard["file"]).is_file() for shard in manifest["shards"])
