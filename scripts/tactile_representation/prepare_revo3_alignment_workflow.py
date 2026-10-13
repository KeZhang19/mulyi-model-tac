#!/usr/bin/env python3
"""Historical ordinal CLIP adapter; legacy real optical-flow input is disabled.

New real captures use ros_ws/src/py_force_ws/tools/prepare_tactile_source.py.
The shared convert_marker helper only accepts real data with an explicit DFS layout.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
from BrainCo_DexHand.tactile_representation.training.alignment_data import AlignmentData, file_digest
from BrainCo_DexHand.tactile_representation.training.data import TactileNormalization
from train_latent_alignment import _load_checkpoint, checkpoint_model_type

SERIAL = "WTUVL3197X2600106"


def resize_image(array, height, width):
    value = np.asarray(array)
    if value.ndim == 2:
        value = value[..., None]
    tensor = torch.from_numpy(value.copy()).permute(2, 0, 1).float()[None]
    return F.interpolate(tensor, (height, width), mode="bilinear", align_corners=True)[0].numpy()


def convert_marker(array, valid, source_hw, target_hw, real, *, dfs_layout=None):
    marker = np.asarray(array, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool).copy()
    if real:
        # Legacy 400-point optical-flow -> 10x10 interpolation is disabled.
        if dfs_layout is None:
            raise ValueError("Recorded DFS layout required; legacy 400-point optical-flow interpolation is disabled. Use ros_ws/src/py_force_ws/tools/prepare_tactile_source.py for recorded DFS runs.")
        sys.path.insert(0, str(ROOT / "calibration/tools"))
        from marker_training import pack_dfs_frame
        if list(source_hw) != dfs_layout["image_hw"]:
            raise ValueError("Source image differs from DFS reference")
        native_shape = tuple(dfs_layout["native_shape"])
        marker, valid = pack_dfs_frame(marker[:, :2].reshape(native_shape),
            (marker[:, :2] + marker[:, 2:4]).reshape(native_shape),
            valid.reshape(native_shape[:-1]), dfs_layout)
    if marker.shape != (100, 5):
        raise ValueError("This workflow adapter targets the original 100-point encoders")
    src_h, src_w = source_hw
    dst_h, dst_w = target_hw
    marker[:, [0, 2]] *= (dst_w - 1) / (src_w - 1)
    marker[:, [1, 3]] *= (dst_h - 1) / (src_h - 1)
    marker[:, 4] = valid
    marker[~valid] = 0
    if not np.isfinite(marker).all():
        raise ValueError("Nonfinite converted marker")
    return marker.astype(np.float32), valid


def export_revo3_alignment_workflow(args):
    if not args.workflow_test:
        raise ValueError("Random ordinal data requires explicit --workflow-test")
    root, out = args.source.resolve(), args.output.resolve()
    checkpoints = [_load_checkpoint(path) for path in (args.sim_checkpoint, args.real_checkpoint)]
    if [checkpoint_model_type(c) for c in checkpoints] != ["robust", "tri_modal"]:
        raise ValueError("This adapter targets the original robust/tri_modal towers")
    norms = [TactileNormalization(**c["normalization"]) for c in checkpoints]
    for c, norm in zip(checkpoints, norms):
        if (norm.image_height, norm.image_width, norm.marker_count) != (240, 320, 100):
            raise ValueError("Unexpected original normalization contract")
        if (c["model_cfg"]["image_height"], c["model_cfg"]["image_width"], c["model_cfg"]["marker_count"]) != (240, 320, 100):
            raise ValueError("Unexpected original model input contract")
    with (root / "pair_index.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 50 or any(row["camera_serial"] != SERIAL for row in rows):
        raise ValueError("Expected the 50-pair little-only bundle")
    session = json.loads((root / "session.json").read_text())
    if session["pair_valid"] is not False or session["training_ready"] is not False:
        raise ValueError("Unexpected source validity state")
    if out.exists():
        raise FileExistsError(out)
    source_hashes = {"pair_index.csv": file_digest(root / "pair_index.csv"),
                     "session.json": file_digest(root / "session.json")}
    data = {domain: {name: [] for name in ("rgb", "depth_m", "marker_2d", "marker_valid")}
            for domain in ("sim", "real")}
    records = []
    for index, row in enumerate(rows):
        sid = f"s{index+1:03d}"
        if row["sample_id"] != sid:
            raise ValueError("Unexpected sample order")
        paths = [f"paired_metadata/{SERIAL}/{sid}.json", row["real_arrays"]]
        paths += [f"sim_reference/arrays/{kind}/{sid}.npy" for kind in
                  ("rgb_uint8", "depth_m", "marker_uv_flow_px")]
        for name in paths:
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Source path escapes bundle")
            source_hashes[name] = file_digest(path)
        meta = json.loads((root / paths[0]).read_text())
        if (meta["camera_serial"] != SERIAL or meta["physical_finger"] != "little"
                or meta["pair_valid"] is not False or meta["training_ready"] is not False):
            raise ValueError("Invalid little-finger ordinal metadata")
        records.append({"pair_id": [index, 0], "sample_id": sid,
                        "pair_valid": False, "training_ready": False,
                        "source_session_id": session["started_utc"],
                        "reference_id": f"{session['started_utc']}:tick0-unverified",
                        "condition_id": None,
                        "sim_source": {"episode_id": meta["sim_episode_id"], "frame_id": meta["sim_frame_id"],
                                       "offset_xy_m": meta["sim_offset_xy_m"]},
                        "real_source": {"tick": meta["real_tick"], "camera_serial": SERIAL,
                                        "host_read_end_unix_ns": meta["host_read_end_unix_ns"],
                                        "position_m": None, "target_force_n": None, "contact": None},
                        "source_files": paths})
        for domain, norm in zip(("sim", "real"), norms):
            if domain == "real":
                with np.load(root / row["real_arrays"], allow_pickle=False) as source:
                    rgb, depth = source["warped_rgb"], source["depth_sdk"] * 0.006
                    marker, valid = convert_marker(source["marker_2d"], source["marker_finite"],
                                                   rgb.shape[:2], (240, 320), True)
            else:
                rgb = np.load(root / paths[2], allow_pickle=False)
                depth = np.load(root / paths[3], allow_pickle=False)
                raw = np.load(root / paths[4], allow_pickle=False)
                marker, valid = convert_marker(raw, raw[:, 4] > 0.5, rgb.shape[:2], (240, 320), False)
            data[domain]["rgb"].append(np.rint(resize_image(rgb, 240, 320)).clip(0, 255).astype(np.uint8))
            data[domain]["depth_m"].append(resize_image(depth, 240, 320).astype(np.float32))
            data[domain]["marker_2d"].append(marker)
            data[domain]["marker_valid"].append(valid)
    quality = {name: False for name in ("pair_valid", "training_ready", "geometry_validated",
                                       "reference_unloaded_confirmed", "independent_validation")}
    out.mkdir(parents=True, exist_ok=False)
    for domain, values in data.items():
        arrays = {name: np.stack(value) for name, value in values.items()}
        arrays.update(pair_id=np.asarray([r["pair_id"] for r in records], dtype=np.int64),
                      pair_valid=np.zeros(50, dtype=bool), training_ready=np.zeros(50, dtype=bool))
        domain_root = out / domain
        (domain_root / "shards").mkdir(parents=True)
        path = domain_root / "shards/shard_000000.npz"
        np.savez_compressed(path, **arrays)
        info = {"schema": "tactile_alignment_domain_npz_v1", "status": "complete", "domain": domain,
                "sample_count": 50, "fields": {name: {"dtype": str(value.dtype), "sample_shape": list(value.shape[1:])}
                                               for name, value in arrays.items()},
                "shard": {"file": "shards/shard_000000.npz", "sha256": file_digest(path)},
                "units": {"rgb": "uint8 0-255 CHW", "depth_m": "m; computational scale only for real",
                          "marker_2d": "target image pixels: u0,v0,du,dv,valid"}}
        (domain_root / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    manifest = {"schema": "tactile_alignment_pairs_v1", "status": "complete", "sample_count": 50,
                "association_kind": "ordinal_workflow_test", "finger": "little", "camera_serial": SERIAL,
                "quality": quality, "domains": {"sim": "sim/manifest.json", "real": "real/manifest.json"},
                "pairs": records,
                "splits": {"train": [r["pair_id"] for r in records[:40]],
                           "validation": [r["pair_id"] for r in records[40:]], "test": []},
                "split_note": "same continuous session; 40/10 internal workflow probe, no independent test",
                "transform": {"version": "revo3_little_ordinal_v1", "image_interpolation": "bilinear align_corners=true; RGB round to uint8",
                              "sim_source_hw": [32, 40], "target_hw": [240, 320],
                              "real_rgb": "SDK warped_rgb", "real_depth_m_per_sdk_unit": 0.006,
                              "real_marker": "20x20 regular row-major grid ->10x10 bilinear origins/flows",
                              "marker_validity": "finite source points; no tracking confidence claim"},
                "source_root_label": root.name, "source_sha256": source_hashes,
                "initial_checkpoint_sha256": [file_digest(p) for p in (args.sim_checkpoint, args.real_checkpoint)],
                "normalization_expected": [asdict(norm) for norm in norms]}
    if any(file_digest(root / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Source bundle changed during conversion")
    (out / "pairs.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    accepted = AlignmentData(out, *norms, workflow_test=True)
    print(json.dumps({"output": str(out), "pairs": 50, "train": len(accepted.dataset("train")),
                      "validation": len(accepted.dataset("validation")), "fingerprint": accepted.contract["fingerprint"],
                      "pair_valid": False, "training_ready": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow-test", action="store_true")
    parser.add_argument("--sim-checkpoint", type=Path, default=ROOT / "runs/revo3_cross_modal_restoration_v1/best.pt")
    parser.add_argument("--real-checkpoint", type=Path, default=ROOT / "runs/revo3_tri_modal_cross_v1/best.pt")
    export_revo3_alignment_workflow(parser.parse_args())
