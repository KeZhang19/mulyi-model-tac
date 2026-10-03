#!/usr/bin/env python3
"""Build a collision-screened approach to a free, horizontal tabletop bulb.

Uses measured collision vertices for reset height. No fixed bulb or scripted
controller is used at runtime. The sampled corridor is not dynamic validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def build():
    tools = runpy.run_path(str(ROOT / "scripts/diagnostics/build_bulb_randomized_workspace.py"))
    kin = tools["Kinematics"]()
    screen = tools["CollisionScreen"](kin)
    transform, overlap = tools["transform"], tools["overlap"]
    # Local +X is the bulb axis; world +Y is a horizontal nominal heading.
    quat = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
    rotation = transform([0., 0., 0.], quat)[:3, :3]
    vertices = np.concatenate([c.points for c in screen.bulb])
    top = float(screen.table[1][2])
    nominal = np.array([-.72, -.14, top - (vertices @ rotation.T)[:, 2].min() + .002])
    shell = nominal + rotation @ np.array([.04, 0., 0.])
    ref = kin.reference["object"]
    old_shell = transform(ref["position_m"], ref["quaternion_wxyz"]) @ np.array([.04, 0., 0., 1.])
    delta = shell - old_shell[:3]
    bulb = [c.placed(transform(nominal, quat)) for c in screen.bulb]
    support = [c.placed(screen.support_pose) for c in screen.support]

    def safe(q):
        poses = kin.fk(q)
        placed = [c.placed(poses[c.body]) for c in screen.robot]
        if any(overlap(placed[i], placed[j], margin=0.) for i, j in screen.pairs):
            return False
        return not any(overlap(g, screen.table, margin=0.) or
                       any(overlap(g, b, margin=0.) for b in bulb + support) for g in placed)

    selected = None
    for extra_z in (0., .005, .01, .02, .03, .04, .05, .06, .08):
        near_delta = delta + np.array([0., 0., extra_z])
        near = kin.solve(near_delta)
        middle = kin.solve(near_delta * .5)
        if near is None or middle is None:
            continue
        # Open fingers, with only thumb opposition retained for an approach.
        for opening in (.7, .5, .3, 0.):
            near[kin.hand] = kin.qref[kin.hand] * opening
            middle[kin.hand] = 0.
            far = kin.qzero.copy()
            path = [near + f * (middle - near) for f in np.linspace(0, 1, 21)]
            path += [middle + f * (far - middle) for f in np.linspace(0, 1, 21)]
            if all(safe(q) for q in path):
                selected = near.copy(), middle, far, opening, extra_z
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("No collision-screened floor pregrasp found")
    near, middle, far, opening, extra_z = selected
    old = json.loads((ROOT / "assets/rotate_bulb_assembly_curriculum/pregrasp.json").read_text())
    return {"schema": 1, "curriculum_version": 2, "joint_names": kin.names,
            "approach_joint_positions": [q.tolist() for q in (near, middle, far)],
            "bulb_position_m": nominal.tolist(), "bulb_quaternion_wxyz": quat,
            "bulb_collision_vertices": vertices.tolist(), "table_top_m": top,
            "opening_fraction": opening, "extra_z_m": extra_z,
            "screened_path_samples": len(path), "dynamic_grasp_verified": False,
            "joint1_path_range_deg": np.rad2deg([min(q[0] for q in path), max(q[0] for q in path)]).tolist(),
            "tip_shell_distances_m": {name: float(np.linalg.norm(pose[:3, 3] - shell))
                                      for name, pose in kin.fk(near).items() if "tip_Link" in name},
            "asset_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                             for p in old["asset_hashes"]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Select a new output path")
    result = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ("asset_hashes", "approach_joint_positions", "bulb_collision_vertices")}, indent=2))
