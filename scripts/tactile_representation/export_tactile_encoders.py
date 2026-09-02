#!/usr/bin/env python3
"""Export the two pretrained tactile towers as standalone checkpoints.

The exported files keep the original model configuration, normalization, and
episode split metadata, but contain only one encoder's ``model_state``.  They
can therefore be loaded by the alignment script or by downstream inference
code without carrying the other tower or an optimizer state.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from train_latent_alignment import (  # noqa: E402
    MODEL_TYPES,
    _build_encoder,
    _load_checkpoint,
    _save_encoder_checkpoint,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-checkpoint", type=Path, required=True)
    parser.add_argument("--real-checkpoint", type=Path, required=True)
    parser.add_argument("--sim-model-type", choices=MODEL_TYPES, default="robust")
    parser.add_argument("--real-model-type", choices=MODEL_TYPES, default="tri_modal")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/revo3_encoder_exports")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing exported files.",
    )
    return parser.parse_args(argv)


def _check_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing encoder export: {path}. "
            "Pass --overwrite to replace it."
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim_output = args.output_dir / "sim_encoder.pt"
    real_output = args.output_dir / "real_encoder.pt"
    _check_output(sim_output, args.overwrite)
    _check_output(real_output, args.overwrite)

    sim_checkpoint = _load_checkpoint(args.sim_checkpoint)
    real_checkpoint = _load_checkpoint(args.real_checkpoint)
    sim_encoder = _build_encoder(args.sim_model_type, sim_checkpoint)
    real_encoder = _build_encoder(args.real_model_type, real_checkpoint)

    _save_encoder_checkpoint(
        sim_output,
        encoder=sim_encoder,
        source_checkpoint=sim_checkpoint,
        source_checkpoint_path=args.sim_checkpoint,
        role="simulation",
    )
    _save_encoder_checkpoint(
        real_output,
        encoder=real_encoder,
        source_checkpoint=real_checkpoint,
        source_checkpoint_path=args.real_checkpoint,
        role="real",
    )
    print(f"saved_sim_encoder={sim_output}")
    print(f"saved_real_encoder={real_output}")


if __name__ == "__main__":
    main()
