#!/usr/bin/env python3
"""Build a validated mmap cache for tactile representation training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from BrainCo_DexHand.tactile_representation.training import (  # noqa: E402
    build_mmap_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tactile NPZ shards once into read-only .npy memory maps."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/revo3_index_sweep_parallel_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()

    def report(done: int, total: int, path: Path) -> None:
        if done == 1 or done == total or done % 10 == 0:
            print(f"[mmap {done:03d}/{total:03d}] {path.name}", flush=True)

    cache_root = build_mmap_cache(args.dataset, progress=report)
    manifest_path = cache_root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    total_bytes = sum(
        (cache_root / field["file"]).stat().st_size
        for field in manifest["fields"].values()
    )
    print(
        "[done] "
        + json.dumps(
            {
                "cache": str(cache_root),
                "sample_count": int(manifest["sample_count"]),
                "size_gib": total_bytes / (1024**3),
                "seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
