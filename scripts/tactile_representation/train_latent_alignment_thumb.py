#!/usr/bin/env python3
"""Thumb-only 180x320 entry point for the original CLIP training loop."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import train_latent_alignment as base
from BrainCo_DexHand.tactile_representation.thumb_alignment import (
    THUMB_INPUT_CONTRACT, validate_thumb_checkpoint,
)
from BrainCo_DexHand.tactile_representation.training.alignment_data import AlignmentData
from BrainCo_DexHand.tactile_representation.training.data import TactileNormalization


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        print(__doc__ + " Requires explicit thumb checkpoints, --alignment-dataset and a new --output.")
    args = base.parse_args(argv)
    for option in ("--sim-checkpoint", "--real-checkpoint", "--alignment-dataset", "--output"):
        if not any(value.split("=", 1)[0] == option for value in argv):
            raise ValueError(f"Thumb entry requires explicit {option}; use prepare_thumb_clip.py for initialization")
    pair_manifest = json.loads((args.alignment_dataset / "pairs.json").read_text())
    if (pair_manifest.get("finger") != "thumb"
            or pair_manifest.get("camera_serial") != THUMB_INPUT_CONTRACT["camera_serial"]):
        raise ValueError("Thumb entry requires a thumb-only pair manifest with the confirmed SDK serial")
    norms = []
    for path in (args.sim_checkpoint, args.real_checkpoint):
        source = base._load_checkpoint(path)
        validate_thumb_checkpoint(source)
        norms.append(TactileNormalization(**source["normalization"]))
        del source
    # Keep every existing pairing/split gate, including explicit workflow mode.
    data = AlignmentData(args.alignment_dataset, *norms, workflow_test=args.workflow_test)
    contract = dict(THUMB_INPUT_CONTRACT, data_fingerprint=data.contract["fingerprint"],
                    normalization=data.contract["normalization"], workflow_test=args.workflow_test)
    del data
    sidecar = args.output / "thumb_input_contract.json"
    if sidecar.exists() and json.loads(sidecar.read_text()) != contract:
        raise ValueError("Thumb run input contract differs from the saved run")
    base.main(argv)
    sidecar.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
