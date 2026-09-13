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
    _save_encoder_checkpoint,
)
from BrainCo_DexHand.tactile_representation.policy import (  # noqa: E402
    export_policy_encoder_bundle,
    export_unaligned_sim_policy_encoder_bundle,
    file_sha256,
    load_tactile_checkpoint,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-checkpoint", type=Path, required=True)
    parser.add_argument("--real-checkpoint", type=Path, default=None)
    parser.add_argument("--unaligned-sim", action="store_true",
                        help="Export frozen simulation features with identity projection for a sim-only PPO baseline.")
    parser.add_argument("--alignment-checkpoint", type=Path, default=None,
                        help="Export final aligned encoders plus projection heads as policy bundles.")
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
    args = parser.parse_args(argv)
    if args.unaligned_sim:
        if args.alignment_checkpoint is not None or args.sim_model_type != "robust":
            parser.error("--unaligned-sim requires a robust simulation encoder and no --alignment-checkpoint")
    elif args.real_checkpoint is None:
        parser.error("--real-checkpoint is required unless --unaligned-sim is used")
    return args


def _check_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing encoder export: {path}. "
            "Pass --overwrite to replace it."
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.unaligned_sim:
        output = args.output_dir / "sim_policy_encoder_unaligned.pt"
        _check_output(output, args.overwrite)
        export_unaligned_sim_policy_encoder_bundle(
            output, source=load_tactile_checkpoint(args.sim_checkpoint),
            source_checkpoint_sha256=file_sha256(args.sim_checkpoint),
        )
        print(f"saved_unaligned_sim_policy_encoder={output}")
        return
    if args.alignment_checkpoint is not None:
        sim_output = args.output_dir / "sim_policy_encoder.pt"
        real_output = args.output_dir / "real_policy_encoder.pt"
        _check_output(sim_output, args.overwrite)
        _check_output(real_output, args.overwrite)
        alignment = load_tactile_checkpoint(args.alignment_checkpoint)
        sources = {
            "sim": load_tactile_checkpoint(args.sim_checkpoint),
            "real": load_tactile_checkpoint(args.real_checkpoint),
        }
        alignment_id = file_sha256(args.alignment_checkpoint)
        for domain, output, model_type in (
            ("sim", sim_output, args.sim_model_type),
            ("real", real_output, args.real_model_type),
        ):
            export_policy_encoder_bundle(
                output, alignment=alignment, source=sources[domain], domain=domain,
                network_type=model_type, alignment_id=alignment_id,
            )
            print(f"saved_{domain}_policy_encoder={output}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim_output = args.output_dir / "sim_encoder.pt"
    real_output = args.output_dir / "real_encoder.pt"
    _check_output(sim_output, args.overwrite)
    _check_output(real_output, args.overwrite)

    sim_checkpoint = load_tactile_checkpoint(args.sim_checkpoint)
    real_checkpoint = load_tactile_checkpoint(args.real_checkpoint)
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
