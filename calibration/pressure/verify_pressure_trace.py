#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from BrainCo_DexHand.force_map import (  # noqa: E402
    DENSE_REFERENCE_ACCEPTANCE_LAYERS,
    evaluate_pressure_trace_report,
    pressure_trace_report,
    validate_pressure_trace_v1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify saved SDF-like pressure tactile traces.")
    parser.add_argument("trace", type=Path, help="Path to a pressure trace .npz file.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional report output path.")
    parser.add_argument("--pressure-key", default="pressure_norm")
    parser.add_argument("--raw-pressure-key", default="pressure_raw_n")
    parser.add_argument("--penetration-key", default="penetration_m")
    parser.add_argument(
        "--reference-key",
        default=None,
        help="Optional dense model/benchmark reference array in the same .npz, e.g. tacmap_raw_m.",
    )
    parser.add_argument(
        "--reference-valid-mask-key",
        default=None,
        help="Optional alignment valid-mask array for the reference. Defaults to '<reference-key>_valid_mask' when present.",
    )
    parser.add_argument(
        "--reference-layer",
        default=None,
        help=(
            "Optional validation layer metadata: L0_analytic, L1_model_reference, "
            "L1_sparse_sanity, L2_offline_oracle, L3_real_calibration, or benchmark_reference. "
            "Inferred from --reference-key when omitted."
        ),
    )
    parser.add_argument(
        "--dense-reference-layer",
        action="append",
        default=None,
        help=(
            "Reference layer allowed to drive dense spatial acceptance. Can be repeated. "
            "This gates acceptance relative to the declared reference layer, not physical GT. "
            f"Defaults to {', '.join(DENSE_REFERENCE_ACCEPTANCE_LAYERS)}."
        ),
    )
    parser.add_argument(
        "--allow-reference-layer-override",
        action="store_true",
        help=(
            "Allow a manually supplied --reference-layer to override an inferred layer. "
            "Intended only for deliberate diagnostics."
        ),
    )
    parser.add_argument("--active-threshold", type=float, default=1.0e-6)
    parser.add_argument("--penetration-threshold", type=float, default=0.0)
    parser.add_argument("--reference-threshold", type=float, default=0.0)
    parser.add_argument("--precontact-leakage-threshold", type=float, default=0.001)
    parser.add_argument("--reference-iou-threshold", type=float, default=0.95)
    parser.add_argument("--reference-alignment-valid-fraction-threshold", type=float, default=None)
    parser.add_argument("--reference-contact-valid-fraction-threshold", type=float, default=1.0)
    parser.add_argument("--reference-invalid-contact-fraction-threshold", type=float, default=0.0)
    parser.add_argument("--centroid-error-threshold-px", type=float, default=1.5)
    parser.add_argument("--bbox-error-threshold-px", type=float, default=2.0)
    parser.add_argument("--depth-rmse-threshold-m", type=float, default=2.0e-5)
    parser.add_argument("--onset-error-threshold-frames", type=int, default=1)
    parser.add_argument("--offset-error-threshold-frames", type=int, default=1)
    parser.add_argument("--force-depth-spearman-threshold", type=float, default=0.95)
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with status 2 when any enabled threshold check fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = validate_pressure_trace_v1(args.trace)
    if not bool(contract["passed"]):
        payload = {
            "contract": contract,
            "report": {"available": False, "reason": "contract failed"},
            "evaluation": {
                "passed": False,
                "checks": [
                    {
                        "name": "contract.passed",
                        "value": False,
                        "threshold": True,
                        "passed": False,
                    }
                ],
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(text + "\n", encoding="utf-8")
        return 2 if args.fail_on_threshold else 0

    report = pressure_trace_report(
        args.trace,
        pressure_key=str(args.pressure_key),
        raw_pressure_key=str(args.raw_pressure_key),
        penetration_key=str(args.penetration_key),
        reference_key=args.reference_key,
        reference_valid_mask_key=args.reference_valid_mask_key,
        reference_layer=args.reference_layer,
        active_threshold=float(args.active_threshold),
        penetration_threshold=float(args.penetration_threshold),
        reference_threshold=float(args.reference_threshold),
    )
    evaluation = evaluate_pressure_trace_report(
        report,
        precontact_leakage_threshold=float(args.precontact_leakage_threshold),
        reference_iou_threshold=float(args.reference_iou_threshold),
        reference_alignment_valid_fraction_threshold=args.reference_alignment_valid_fraction_threshold,
        reference_contact_valid_fraction_threshold=args.reference_contact_valid_fraction_threshold,
        reference_invalid_contact_fraction_threshold=args.reference_invalid_contact_fraction_threshold,
        centroid_error_threshold_px=float(args.centroid_error_threshold_px),
        bbox_error_threshold_px=float(args.bbox_error_threshold_px),
        depth_rmse_threshold_m=float(args.depth_rmse_threshold_m),
        onset_error_threshold_frames=int(args.onset_error_threshold_frames),
        offset_error_threshold_frames=int(args.offset_error_threshold_frames),
        force_depth_spearman_threshold=float(args.force_depth_spearman_threshold),
        dense_reference_layers=(
            tuple(str(value) for value in args.dense_reference_layer)
            if args.dense_reference_layer is not None
            else DENSE_REFERENCE_ACCEPTANCE_LAYERS
        ),
        allow_reference_layer_override=bool(args.allow_reference_layer_override),
    )
    payload = {"contract": contract, "report": report, "evaluation": evaluation}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    if args.fail_on_threshold and (not bool(evaluation["passed"]) or not bool(contract["passed"])):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
