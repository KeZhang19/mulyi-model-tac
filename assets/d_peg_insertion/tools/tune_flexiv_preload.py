#!/usr/bin/env python3
"""Create a conservative Flexiv D-peg drive-target preload candidate.

The editor export gives a useful measured joint pose, but its drive targets
are identical to that pose.  In the Isaac task that leaves no closing force
after the first contact settles.  This tool keeps the measured pose and
object/socket frames unchanged and adds bounded finger-only target offsets.
The result is intentionally left unvalidated until the Isaac zero-action
gravity-hold check is run.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PACKAGE / "pregrasp_flexiv.json"
DEFAULT_OUTPUT = PACKAGE / "pregrasp_flexiv_preload_candidate.json"

# Positive values are flexion/closing targets for the Rizon4/Revo3 hand.  The
# worst editor gap is the index DIP (18.43 mm), so it receives the largest
# bounded preload.  Ring/little already have physical contact in the editor
# pose and therefore use smaller offsets to avoid increasing self-collision.
PRELOAD_RAD = {
    "right_thumb_CMP_joint": 0.005,
    "right_thumb_CMR_joint": 0.015,
    "right_thumb_MCP_joint": 0.080,
    "right_thumb_PIP_joint": 0.050,
    "right_thumb_DIP_joint": 0.025,
    "right_index_MPR_joint": 0.020,
    "right_index_MCP_joint": 0.080,
    "right_index_PIP_joint": 0.060,
    "right_index_DIP_joint": 0.030,
    "right_middle_MPR_joint": 0.015,
    "right_middle_MCP_joint": 0.030,
    "right_middle_PIP_joint": 0.025,
    "right_middle_DIP_joint": 0.012,
    "right_ring_MPR_joint": 0.010,
    "right_ring_MCP_joint": 0.020,
    "right_ring_PIP_joint": 0.015,
    "right_ring_DIP_joint": 0.008,
    "right_little_MPR_joint": 0.008,
    "right_little_MCP_joint": 0.016,
    "right_little_PIP_joint": 0.012,
    "right_little_DIP_joint": 0.006,
}


def tune(source: Path = DEFAULT_INPUT, output: Path = DEFAULT_OUTPUT) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    positions = data.get("robot_joint_positions")
    if not isinstance(positions, dict) or len(positions) != 28:
        raise ValueError("source must contain all 28 robot_joint_positions")
    tuned = copy.deepcopy(data)
    targets = dict(positions)
    for name, delta in PRELOAD_RAD.items():
        if name not in positions:
            raise ValueError(f"missing finger joint: {name}")
        targets[name] = float(positions[name]) + delta
    tuned["robot_joint_targets"] = targets
    tuned["validated"] = False
    tuned["provenance"] = (
        "Editor pose retained; conservative finger-only drive preload added "
        "from contact-gap diagnostics; requires Flexiv Isaac gravity-hold validation."
    )
    tuned["targets_source"] = "editor_joint_positions_plus_bounded_finger_preload"
    validation = dict(tuned.get("validation", {}))
    validation["status"] = "preload_candidate"
    validation["physics_validated"] = False
    validation["trajectory_validated"] = False
    validation["dynamic_lift_validated"] = False
    validation["preload"] = {
        "units": "rad",
        "joint_offsets": PRELOAD_RAD,
        "arm_targets_unchanged": True,
        "static_editor_pose_unchanged": True,
        "reason": "close the fingers after contact settling; index receives the largest bounded offset",
    }
    tuned["validation"] = validation
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tuned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return tuned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = tune(args.source, args.output)
    print(json.dumps({"output": str(args.output), "status": result["validation"]["status"]}, ensure_ascii=False))
