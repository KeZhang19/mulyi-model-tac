#!/usr/bin/env python3
"""Validate a full-stage checkpoint and copy it for the original assembly task.

No model weights, normalizers, action transforms or observation slots change.
The opt-in curriculum task ID is replaced only after validating its provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
CURRICULUM_TASK = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0"
FULL_TASK = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0"


def export_checkpoint(checkpoint: Path, output: Path):
    import torch
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = (data.get("infos") or {}).get("bulb_curriculum")
    if not state or state["settings"]["stage"] != 4 or state["settings"]["fixed_bulb"]:
        raise ValueError("Only the free full-20-second stage may export to the original task")
    contract = state["bank"]["contract"]
    parent = ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom"
    for name, digest in contract["sources"].items():
        if hashlib.sha256((parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Original task source differs from training: {name}")
    action = json.loads((checkpoint.parent / "bulb_action_contract.json").read_text())
    tactile = json.loads((checkpoint.parent / "tactile_policy_contract.json").read_text())
    if (action["task"] != CURRICULUM_TASK or action["observation_dim"] != 669
            or action["action"] != contract["action"] or tactile != contract["observations"]
            or not contract["pure_collision"]):
        raise ValueError("Action/observation/physics contract mismatch")
    if hashlib.sha256((ROOT / "assets/rotate_bulb_assembly_curriculum/pregrasp.json").read_bytes()).hexdigest() != contract["pregrasp"]:
        raise ValueError("Curriculum asset provenance changed")
    action["task"] = FULL_TASK
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint, output / checkpoint.name)
    (output / "bulb_action_contract.json").write_text(json.dumps(action, indent=2) + "\n")
    (output / "tactile_policy_contract.json").write_text(json.dumps(tactile, indent=2) + "\n")
    if (checkpoint.parent / "params").is_dir():
        shutil.copytree(checkpoint.parent / "params", output / "training_params")
    (output / "curriculum_export.json").write_text(json.dumps({
        "source": str(checkpoint.resolve()), "target_task": FULL_TASK,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "weights_modified": False, "full_task_success_validated": False,
    }, indent=2) + "\n")
    return output / checkpoint.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint.resolve(), args.out.resolve()))


if __name__ == "__main__":
    main()
