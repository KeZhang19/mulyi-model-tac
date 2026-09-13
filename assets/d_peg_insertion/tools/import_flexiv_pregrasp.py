#!/usr/bin/env python3
"""Import an editor's exact static D-peg pose without claiming physical calibration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/d_peg_insertion"
TASK = "BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0"


def import_flexiv_pregrasp(export_path: Path, assets: Path = ASSETS):
    """Validate identities/frames, archive the source, and update the task candidate."""
    export_path, assets = Path(export_path).resolve(), Path(assets).resolve()
    raw = export_path.read_bytes()
    export = json.loads(raw)
    if (export.get("schema"), export.get("schema_version")) != ("revo3_simulation_state", 1):
        raise ValueError("Expected a schema v1 revo3_simulation_state export")
    robot = export["robot"]
    scene_reference = json.loads((ROOT / "assets/rotate_bulb_custom/grasp_reference/simulation_state.json").read_text())
    names, values = robot["joint_names"], robot["joint_positions_rad"]
    if len(names) != 28 or len(set(names)) != 28 or set(names) != set(scene_reference["robot"]["joint_names"]):
        raise ValueError("Expected exactly the 28 Flexiv Rizon4 + Revo3 joint names")
    if len(values) != 28 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("Expected 28 finite joint positions")
    joints = dict(zip(names, values, strict=True))
    for joint in robot["joints"]:
        value = joints[joint["name"]]
        if value != export["state"]["qpos"][joint["qpos_index"]]:
            raise ValueError(f"Joint/qpos disagreement: {joint['name']}")
        if joint["limited"] and not joint["range_rad"][0] <= value <= joint["range_rad"][1]:
            raise ValueError(f"Joint outside editor limits: {joint['name']}")

    def pose(item):
        p, q = item["position_m"], item["quaternion_wxyz"]
        if item.get("frame") != "world" or len(p) != 3 or len(q) != 4 or not all(math.isfinite(float(x)) for x in [*p, *q]):
            raise ValueError("Expected finite world xyz + wxyz pose")
        if not math.isclose(sum(float(x) ** 2 for x in q), 1., abs_tol=1e-8):
            raise ValueError("Pose quaternion must be normalized")
        return dict(position=p, quaternion=q)

    # The isolated task deliberately retains the bulb editor's robot root and
    # table frame. A differently based export needs a reviewed scene migration.
    if pose(robot["base"]) != pose(scene_reference["robot"]["base"]):
        raise ValueError("Export robot base differs from the Flexiv task frame")
    for key in ("position_m", "quaternion_wxyz", "size_m"):
        if export["table"][key] != scene_reference["table"][key]:
            raise ValueError("Export table differs from the Flexiv task scene")
    if export["object"]["name"] != "peg" or export["object"]["mesh_scale_xyz"] != [1., 1., 1.]:
        raise ValueError("Expected the original metre-scale peg")
    fixtures = export["fixtures"]
    if len(fixtures) != 1 or fixtures[0]["name"] != "d_socket" or not fixtures[0]["fixed"]:
        raise ValueError("Expected one fixed d_socket fixture")
    assembly = export["assembly"]["source_configuration"]
    if assembly.get("task") != TASK:
        raise ValueError("Export belongs to another task")
    for name in ("peg.usd", "socket.usd", "metadata.json"):
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != assembly["source_sha256"][name]:
            raise ValueError(f"Export geometry differs from task asset: {name}")
    peg_pose, socket_pose = pose(export["object"]), pose(fixtures[0])
    optimization = export["validation"]["optimization"]
    if not all(math.isfinite(float(optimization[name])) for name in ("max_contact_distance_m", "max_penetration_m")):
        raise ValueError("Nonfinite optimization diagnostics")

    archive = assets / "flexiv_grasp_reference" / export_path.parent.name
    archived_json = archive / "simulation_state.json"
    if archived_json.exists() and archived_json.read_bytes() != raw:
        raise ValueError(f"Archive identity collision: {archived_json}")
    archive.mkdir(parents=True, exist_ok=True)
    archived_json.write_bytes(raw)
    if (export_path.parent / "README_CN.md").is_file():
        shutil.copyfile(export_path.parent / "README_CN.md", archive / "EDITOR_README_CN.md")
    digest = hashlib.sha256(raw).hexdigest()
    provenance = dict(source_path=str(export_path), source_sha256=digest,
                      imported_at_utc=datetime.now(timezone.utc).isoformat(),
                      model_source_path=str(export_path.parent / "scene.mjb"),
                      source_validation=export["validation"])
    (archive / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    source_reference = str(archived_json.relative_to(assets))
    candidate = dict(schema_version=1, validated=False, task=TASK,
                     provenance="User editor optimized static pose; exact joints and object/fixture poses; no PD preload calibration.",
                     robot_joint_positions=joints, robot_joint_targets=dict(joints),
                     targets_source="editor_joint_positions_without_drive_preload",
                     peg_pose=peg_pose, socket_pose=socket_pose,
                     source_grasp_reference=source_reference, source_grasp_reference_sha256=digest,
                     validation=dict(status="editor_candidate", physics_validated=False,
                                     source_export_pose_candidate_pass=bool(optimization["pose_candidate_pass"]),
                                     optimization=optimization))
    target = assets / "pregrasp_flexiv.json"
    backup = archive / "previous_pregrasp_flexiv.json"
    if target.is_file() and not backup.exists():
        shutil.copyfile(target, backup)
    pending = target.with_suffix(".json.tmp")
    pending.write_text(json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(target)
    return candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    result = import_flexiv_pregrasp(args.export)
    print(json.dumps(dict(pregrasp=str(ASSETS / "pregrasp_flexiv.json"), validation=result["validation"]), indent=2))
