#!/usr/bin/env python3
"""Prepare an audited D-peg resume seed without modifying the source run.

The normal training contract validator remains strict. Optional socket changes
require passing physical validation for that exact asset. This prepares files
only; it never starts or stops training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
from BrainCo_DexHand.tactile_representation.policy import (
    file_sha256, policy_contract_path, validate_policy_contract,
)

TASK = "BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0"
PHYSICS_CHECKS = {
    "aligned_reaches_28mm", "aligned_remains_stable", "offset_is_blocked",
    "wrong_yaw_is_blocked", "blocked_cases_have_contacts", "socket_stays_fixed",
    "no_joint_constraint",
}


def _physics_validation(path: Path, socket: Path, expected_peg_hash: str) -> dict:
    report = json.loads(path.read_text())
    probe = report.get("contact_probe", {})
    if (not isinstance(probe, dict) or probe.get("disable_peg_sleep", False) is not False
            or probe.get("additional_down_force_n_last_second_blocked_cases", 0) != 0):
        raise ValueError("Canonical physics validation must keep peg sleep and applied forces unchanged")
    checks = report.get("checks", {})
    if (report.get("passed") is not True or not PHYSICS_CHECKS.issubset(checks)
            or any(value is not True for value in checks.values())):
        raise ValueError("Socket physics validation must pass every required check")
    socket_hash = file_sha256(socket)
    recorded_hash = report.get("socket_sha256")
    reported_path = Path(report["socket_usd"]) if report.get("socket_usd") else None
    if recorded_hash is not None:
        if recorded_hash != socket_hash:
            raise ValueError("Physics report socket_sha256 does not match the selected socket")
        identity = "recorded_sha256"
    elif reported_path is not None and reported_path.is_file():
        if file_sha256(reported_path) != socket_hash:
            raise ValueError("Physics report socket_usd bytes do not match the selected socket")
        identity = "current_report_path_sha256"
    else:
        raise ValueError("Physics report needs socket_sha256 or an accessible socket_usd path")
    if reported_path is not None and reported_path.is_file() and file_sha256(reported_path) != socket_hash:
        raise ValueError("Physics report path has changed since its recorded socket hash")
    peg_hash = report.get("peg_sha256")
    reported_peg = Path(report["peg_usd"]) if report.get("peg_usd") else None
    if peg_hash is None and reported_peg is not None and reported_peg.is_file():
        peg_hash = file_sha256(reported_peg)
    if peg_hash != expected_peg_hash:
        raise ValueError("Physics report must identify the unchanged D-peg asset")
    return dict(path=str(path), sha256=file_sha256(path), socket_identity=identity,
                socket_sha256=socket_hash, peg_sha256=peg_hash, passed=True, checks=checks)


def _assert_same_tree(before, after, path="checkpoint"):
    if isinstance(before, torch.Tensor):
        if not isinstance(after, torch.Tensor) or before.dtype != after.dtype:
            raise ValueError(f"Changed tensor type at {path}")
        if not torch.isfinite(before).all():
            raise ValueError(f"Nonfinite checkpoint tensor at {path}")
        torch.testing.assert_close(before, after, rtol=0, atol=0, equal_nan=False)
    elif isinstance(before, dict):
        if not isinstance(after, dict) or before.keys() != after.keys():
            raise ValueError(f"Changed keys at {path}")
        for key in before:
            _assert_same_tree(before[key], after[key], f"{path}.{key}")
    elif isinstance(before, (tuple, list)):
        if type(before) is not type(after) or len(before) != len(after):
            raise ValueError(f"Changed sequence at {path}")
        for index, (left, right) in enumerate(zip(before, after)):
            _assert_same_tree(left, right, f"{path}[{index}]")
    elif type(before) is not type(after) or before != after:
        raise ValueError(f"Changed value at {path}")


def prepare_resume(checkpoint: Path, output_dir: Path, source_socket_usd: Path, *,
                   socket_usd: Path | None = None, validation_report: Path | None = None,
                   target_contract_path: Path | None = None, total_iterations: int = 15000) -> dict:
    if output_dir.is_symlink():
        raise FileExistsError(f"Refusing to use a symlink as a new run: {output_dir}")
    checkpoint, output_dir = checkpoint.resolve(), output_dir.resolve()
    source_socket_usd = source_socket_usd.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    source_contract_path = policy_contract_path(checkpoint).resolve()
    original_contract_bytes = source_contract_path.read_bytes()
    contract = json.loads(original_contract_bytes)
    if (contract.get("task") != TASK or contract.get("task_schema_version") != 2
            or contract.get("observation_dim") != 1949):
        raise ValueError("Only the approved 1949-observation D-peg v2 task can be resumed")
    asset_hashes = contract.get("task_assets", {})
    if set(asset_hashes) != {"peg", "socket", "geometry", "pregrasp"}:
        raise ValueError("Source contract must record all four D-peg task assets")
    if file_sha256(source_socket_usd) != asset_hashes["socket"]:
        raise ValueError("Source socket bytes disagree with the source checkpoint contract")
    source_hashes = {str(path): file_sha256(path) for path in (
        checkpoint, source_contract_path, source_socket_usd)}
    if source_hashes[str(source_contract_path)] != hashlib.sha256(original_contract_bytes).hexdigest():
        raise ValueError("Source contract changed while being read")
    target = deepcopy(contract)
    physics = None
    if socket_usd is not None:
        if validation_report is None:
            raise ValueError("--socket-usd requires --validation-report")
        socket_usd = socket_usd.resolve()
        physics = _physics_validation(validation_report.resolve(), socket_usd, asset_hashes["peg"])
        target["task_assets"]["socket"] = file_sha256(socket_usd)
    elif validation_report is not None:
        raise ValueError("--validation-report requires an explicit --socket-usd")
    if target_contract_path is not None:
        validate_policy_contract(target, json.loads(target_contract_path.read_text()))
    # Exact equality after replacing this one leaf makes every other field mandatory.
    restored_contract = deepcopy(target)
    restored_contract["task_assets"]["socket"] = contract["task_assets"]["socket"]
    validate_policy_contract(contract, restored_contract)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"model_state_dict", "optimizer_state_dict", "iter", "infos"}.issubset(payload):
        raise ValueError("Source must contain policy, optimizer, iteration and infos")
    iteration = payload["iter"]
    if type(iteration) is not int or iteration < 0 or iteration + 1 >= total_iterations:
        raise ValueError("Source iteration must leave at least one requested training update")
    if not payload["model_state_dict"] or not payload["optimizer_state_dict"].get("state"):
        raise ValueError("Policy and initialized optimizer state must be nonempty")
    groups = payload["optimizer_state_dict"].get("param_groups", [])
    rates = [float(group["lr"]) for group in groups]
    if not rates or any(not math.isfinite(rate) or rate <= 0 or rate != rates[0] for rate in rates):
        raise ValueError("Expected the same finite positive adaptive learning rate in every optimizer group")
    _assert_same_tree(payload, payload)
    next_iteration = iteration + 1
    migrated = dict(payload, iter=next_iteration)
    name = f"model_{next_iteration}.pt"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".d_peg_resume_", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        torch.save(migrated, stage / name)
        reloaded = torch.load(stage / name, map_location="cpu", weights_only=True)
        if reloaded["iter"] != next_iteration:
            raise ValueError("Staged checkpoint does not start at the next iteration")
        _assert_same_tree(payload, dict(reloaded, iter=iteration))
        (stage / "tactile_policy_contract.json").write_text(json.dumps(target, indent=2) + "\n")
        (stage / "source_tactile_policy_contract.json").write_bytes(original_contract_bytes)
        changed_socket = target["task_assets"]["socket"] != asset_hashes["socket"]
        manifest = {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "validated_socket_collision" if changed_socket else "geometry_only",
            "source": {"checkpoint": str(checkpoint), "checkpoint_sha256": source_hashes[str(checkpoint)],
                       "contract": str(source_contract_path), "contract_sha256": source_hashes[str(source_contract_path)],
                       "socket_usd": str(source_socket_usd), "socket_sha256": asset_hashes["socket"],
                       "completed_iteration": iteration},
            "target": {"checkpoint": str(output_dir / name), "checkpoint_sha256": file_sha256(stage / name),
                       "contract_sha256": file_sha256(stage / "tactile_policy_contract.json"),
                       "socket_usd": str(socket_usd or source_socket_usd),
                       "socket_sha256": target["task_assets"]["socket"], "next_iteration": next_iteration},
            "checkpoint_changes": {"iter": {"before": iteration, "after": next_iteration,
                "reason": "Installed RSL-RL saves last completed index and resumes at saved index; start at next update."}},
            "contract_changed_fields": ["task_assets.socket"] if changed_socket else [],
            "preserved": {"model_state_dict_exact": True, "optimizer_state_dict_exact": True,
                          "other_checkpoint_fields_exact": True,
                          "normalizer_keys": sorted(key for key in payload["model_state_dict"] if "normalizer" in key)},
            "physics_validation": physics, "optimizer_group_learning_rates": rates,
            "recommended_training": {"resume": True, "load_run": output_dir.name, "checkpoint": name,
                "additional_iterations": total_iterations - next_iteration,
                "agent_algorithm_learning_rate": rates[0],
                "required_hydra_override": f"agent.algorithm.learning_rate={rates[0]:.17g}"},
            "resume_scope": "Restores policy, observation normalizers and optimizer; environment/RNG state was not saved by RSL-RL.",
        }
        (stage / "resume_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        for path, expected in source_hashes.items():
            if file_sha256(path) != expected:
                raise ValueError(f"Source changed while preparing resume: {path}")
        if socket_usd is not None and file_sha256(socket_usd) != target["task_assets"]["socket"]:
            raise ValueError("Selected socket changed while preparing resume")
        if physics is not None and file_sha256(physics["path"]) != physics["sha256"]:
            raise ValueError("Physics report changed while preparing resume")
        # Exclusive creation also rejects a directory created by another process.
        output_dir.mkdir()
        try:
            for path in stage.iterdir():
                shutil.copy2(path, output_dir / path.name)
        except Exception:
            shutil.rmtree(output_dir)
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-socket-usd", type=Path, required=True)
    parser.add_argument("--socket-usd", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--target-contract", type=Path)
    parser.add_argument("--total-iterations", type=int, default=15000)
    args = parser.parse_args()
    manifest = prepare_resume(args.checkpoint, args.output_dir, args.source_socket_usd,
        socket_usd=args.socket_usd, validation_report=args.validation_report,
        target_contract_path=args.target_contract, total_iterations=args.total_iterations)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
