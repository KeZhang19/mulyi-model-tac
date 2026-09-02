#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from BrainCo_DexHand.force_map import pressure_sample_boundary_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose dense pressure/reference boundary errors using raw per-taxel samples."
    )
    parser.add_argument("trace", type=Path, help="Path to a pressure trace .npz file.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional full diagnostics JSON output path.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional per-frame diagnostics CSV output path.")
    parser.add_argument("--pressure-key", default="geometry_normal_ray_pressure_norm")
    parser.add_argument("--penetration-key", default="geometry_normal_ray_penetration_m")
    parser.add_argument("--reference-key", default="normal_ray_penetration_m")
    parser.add_argument("--sample-penetrations-key", default="geometry_normal_ray_sample_penetrations_m")
    parser.add_argument("--active-threshold", type=float, default=1.0e-6)
    parser.add_argument("--reference-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = pressure_sample_boundary_diagnostics(
        args.trace,
        pressure_key=str(args.pressure_key),
        penetration_key=str(args.penetration_key),
        reference_key=str(args.reference_key),
        sample_penetrations_key=str(args.sample_penetrations_key),
        active_threshold=float(args.active_threshold),
        reference_threshold=float(args.reference_threshold),
        top_k=int(args.top_k),
    )
    summary = diagnostics.get("summary", diagnostics)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        frames = diagnostics.get("frames", [])
        if frames:
            with args.out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(frames[0].keys()))
                writer.writeheader()
                writer.writerows(frames)
        else:
            args.out_csv.write_text("", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
