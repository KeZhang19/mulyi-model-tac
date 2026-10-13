#!/usr/bin/env python3
"""Prepare isolated 180x320 thumb initialization.

The historical real-domain path below used 400 optical-flow points and is now
rejected by convert_marker. New real data uses the ROS recorded-DFS source path;
checkpoint initialization remains available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
from BrainCo_DexHand.tactile_representation.thumb_alignment import (
    THUMB_INPUT_CONTRACT, initialize_thumb_checkpoint, validate_thumb_checkpoint,
)
from prepare_revo3_alignment_workflow import convert_marker


def prepare_real_domain(source: Path, output: Path, checkpoint: Path) -> dict:
    """Preserve SDK RGB/Depth pixels; these records have no simulation pairs."""
    header = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_thumb_checkpoint(header)
    files = sorted((source / "real_arrays" / THUMB_INPUT_CONTRACT["camera_serial"]).glob("s*.npz"))
    if not files:
        raise ValueError("Expected recorded real_arrays/WTUVL2198X260001A/s*.npz")
    if output.exists():
        raise FileExistsError(output)
    values = {key: [] for key in ("rgb", "depth_m", "marker_2d", "marker_valid")}
    provenance = []
    for index, path in enumerate(files):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as z:
            rgb, depth = z["warped_rgb"], z["depth_sdk"]
            if rgb.shape != (180, 320, 3) or rgb.dtype != np.uint8 or depth.shape != (180, 320):
                raise ValueError(f"Expected native SDK thumb RGB/Depth 180x320: {path}")
            if not np.isfinite(depth).all():
                raise ValueError(f"Nonfinite SDK depth: {path}")
            marker, valid = convert_marker(z["marker_2d"], z["marker_finite"],
                                           (180, 320), (180, 320), True)
            values["rgb"].append(rgb.transpose(2, 0, 1).copy())
            values["depth_m"].append((depth[None] * .006).astype(np.float32))
            values["marker_2d"].append(marker)
            values["marker_valid"].append(valid)
            tick = int(z["source_capture_tick"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed: {path}")
        provenance.append({"domain_row": index, "source": str(path.resolve()),
                           "source_capture_tick": tick, "sha256": digest})
    arrays = {key: np.stack(value) for key, value in values.items()}
    count = len(files)
    arrays.update(pair_id=np.c_[np.arange(count), np.zeros(count)].astype(np.int64),
                  pair_valid=np.zeros(count, dtype=bool), training_ready=np.zeros(count, dtype=bool))
    output.mkdir(parents=True)
    (output / "shards").mkdir()
    shard = output / "shards/shard_000000.npz"
    np.savez_compressed(shard, **arrays)
    manifest = {"schema": "tactile_alignment_domain_npz_v1", "status": "complete", "domain": "real",
                "sample_count": count, "thumb_input_contract": THUMB_INPUT_CONTRACT,
                "fields": {k: {"sample_shape": list(v.shape[1:]), "dtype": str(v.dtype)} for k, v in arrays.items()},
                "shard": {"file": "shards/shard_000000.npz", "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()},
                "units": {"rgb": "uint8 RGB CHW; native SDK warp, no image resize",
                          "depth_m": "depth_sdk * 0.006; computational scale, physically unvalidated",
                          "marker_2d": "native pixels: u0,v0,du,dv,finite; 20x20 ->10x10 virtual queries"},
                "pair_id_note": "provisional domain row IDs only; no physical sim/real association",
                "pair_valid": False, "training_ready": False,
                "provenance": provenance}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"output": str(output), "sample_count": count, "image_hw": [180, 320],
            "image_resize": False, "pair_valid": False, "training_ready": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("checkpoints", help="Explicitly initialize two new thumb towers from old weights")
    init.add_argument("--sim-checkpoint", type=Path, default=ROOT / "runs/revo3_cross_modal_restoration_v1/best.pt")
    init.add_argument("--real-checkpoint", type=Path, default=ROOT / "runs/revo3_tri_modal_cross_v1/best.pt")
    init.add_argument("--output", type=Path, required=True)
    real = commands.add_parser("real-domain", help="Convert recorded thumb SDK samples without fabricating pairs")
    real.add_argument("--source", type=Path, required=True)
    real.add_argument("--checkpoint", type=Path, required=True)
    real.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "checkpoints":
        args.output.mkdir(parents=True, exist_ok=False)
        info = {domain: initialize_thumb_checkpoint(path, args.output / f"{domain}_thumb_init.pt")
                for domain, path in (("sim", args.sim_checkpoint), ("real", args.real_checkpoint))}
        (args.output / "initialization.json").write_text(json.dumps(info, indent=2) + "\n")
    else:
        info = prepare_real_domain(args.source, args.output, args.checkpoint)
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
