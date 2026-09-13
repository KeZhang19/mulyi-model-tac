#!/usr/bin/env python3
"""CPU-only ring/pinky search and real-mesh clearance checks for vertical insertion.

Only the non-gripping ring/pinky joint positions and targets are changed. The
robot root, arm, thumb/index/middle, peg pose and task geometry remain fixed.
The generated candidate still needs the separate physical gravity-hold check.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.spatial.transform import Rotation

from solve_pregrasp import Kinematics
from validate_assets import load_usd_modules


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT / "assets/bulb/tools"))
from tabletop_validation import body_meshes_local


def box_distance(points, center, half):
    q = np.abs(points - center) - half
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def cylinder_distance(points, center, radius, half_height):
    local = points - center
    q = np.c_[np.linalg.norm(local[:, :2], axis=1) - radius, np.abs(local[:, 2]) - half_height]
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def load_geometry(stage, kin):
    root = str(stage.GetDefaultPrim().GetPath())
    geometry = {}
    for name in kin.fk({}):
        if not stage.GetPrimAtPath(root + "/" + name).IsValid():
            continue
        pieces = body_meshes_local(stage, root + "/" + name)
        if pieces:
            geometry[name] = np.concatenate([np.r_[vertices, vertices[faces].mean(axis=1)]
                                             for _, vertices, faces in pieces])
    return geometry


def transformed(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def trajectory_report(data, kin, geometry, travel):
    root = np.eye(4)
    root[:3, :3] = Rotation.from_euler("z", -np.pi / 2).as_matrix()
    root[:3, 3] = [1, 0, .766]
    joints = data["robot_joint_positions"]
    start = kin.fk(joints, root)["base_link"]
    arm_names = [f"Joint{i}_R" for i in range(1, 8)]
    arm = np.array([joints[n] for n in arm_names])
    stations = []
    for down in np.linspace(0, travel, 16):
        target = start.copy(); target[2, 3] -= down

        def residual(q):
            actual = kin.fk(dict(joints) | dict(zip(arm_names, q)), root)["base_link"]
            return np.r_[(actual[:3, 3] - target[:3, 3]) * 10,
                         Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()]

        fit = least_squares(residual, arm, bounds=kin.bounds[:7].T, max_nfev=150)
        arm = fit.x
        poses = kin.fk(dict(joints) | dict(zip(arm_names, arm)), root)
        table, socket = [], []
        for name, local in geometry.items():
            if name == "Base_R":
                continue
            points = transformed(local, poses[name])
            table.append((float(points[:, 2].min() - .76), name))
            # Using the full socket block is conservative for fingers; the D
            # aperture is reserved for the peg and never used as finger clearance.
            distance = box_distance(points, np.array([.45, .1, .785]), np.array([.04, .04, .025]))
            socket.append((float(distance.min()), name))
        stations.append({"down_m": float(down), "ik_residual": float(np.linalg.norm(fit.fun)),
                         "table_minimum": sorted(table)[:5], "socket_minimum": sorted(socket)[:5]})
    return {"stations": stations,
            "min_table_clearance_m": min(s["table_minimum"][0][0] for s in stations),
            "min_socket_clearance_m": min(s["socket_minimum"][0][0] for s in stations),
            "max_ik_residual": max(s["ik_residual"] for s in stations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PACKAGE / "pregrasp_recentered.json")
    parser.add_argument("--output", type=Path, default=PACKAGE / "pregrasp_clearance_candidate.json")
    parser.add_argument("--report", type=Path, default=PACKAGE / "validation/clearance_candidate.json")
    parser.add_argument("--travel", type=float, default=.058)
    parser.add_argument("--margin", type=float, default=.003)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    Usd, *_ = load_usd_modules()
    kin = Kinematics()
    source_text = args.source.read_text()
    source = json.loads(source_text)
    data = copy.deepcopy(source)
    stage = Usd.Stage.Open(str(ROOT / "assets/urdf/Tianji_Revo3/urdf/Tianji_Revo3_Right_visual.usd"))
    geometry = load_geometry(stage, kin)
    root = np.eye(4)
    root[:3, :3] = Rotation.from_euler("z", -np.pi / 2).as_matrix()
    root[:3, 3] = [1, 0, .766]
    hand = kin.fk(data["robot_joint_positions"], root)["base_link"]
    search_results = {}
    if not args.check_only:
        for finger in ("ring", "pinky"):
            names = [f"right_{finger}{suffix}_joint" for suffix in ("mcp_yaw", "mcp_roll", "pip_roll", "dip_roll")]
            seed = np.array([data["robot_joint_positions"][n] for n in names])
            bounds = kin.bounds[[kin.names.index(n) for n in names]]
            local_geometry = {n: p[np.linspace(0, len(p) - 1, min(len(p), 250)).astype(int)]
                              for n, p in geometry.items() if n.startswith("right_" + finger)}
            swept_center = np.array([.45, .1, .785 + args.travel / 2])
            swept_half = np.array([.04, .04, .025 + args.travel / 2])

            def objective(q):
                poses = kin.fk(dict(data["robot_joint_positions"]) | dict(zip(names, q)), hand, hand_only=True)
                points = np.concatenate([transformed(p, poses[n]) for n, p in local_geometry.items()])
                obstacles = np.minimum(box_distance(points, swept_center, swept_half), points[:, 2] - .76 - args.travel)
                # These fingers should remain out of the grip and shaft.
                obstacles = np.minimum(obstacles, cylinder_distance(points, np.array([.45, .1, .90]), .025, .025))
                obstacles = np.minimum(obstacles, cylinder_distance(points, np.array([.45, .1, .8575]), .012, .0175))
                penetration = np.maximum(args.margin - obstacles, 0)
                return float(1e6 * (np.mean(penetration**2) + penetration.max()**2) + .0001 * np.sum((q - seed)**2))

            result = differential_evolution(objective, bounds, seed=617, popsize=9,
                                            maxiter=args.iterations, polish=True, tol=.0001, x0=seed)
            for name, value in zip(names, result.x):
                data["robot_joint_positions"][name] = float(value)
                data["robot_joint_targets"][name] = float(value)
            search_results[finger] = {"joint_names": names, "before": seed.tolist(),
                                      "after": result.x.tolist(), "objective": float(result.fun)}
            print(f"{finger}: {search_results[finger]}", flush=True)
    check = trajectory_report(data, kin, geometry, args.travel)
    passed = check["min_table_clearance_m"] > .001 and check["min_socket_clearance_m"] > .001 and check["max_ik_residual"] < .001
    report = {"passed": passed, "source": str(args.source),
              "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
              "method": "Actual USD mesh vertices and face centers; 16 bounded-IK stations; non-gripping finger search uses a conservative socket swept volume.",
              "travel_m": args.travel, "search": search_results, "trajectory": check,
              "physical_hold_revalidated": False, "self_collision_certified": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not args.check_only:
        # The candidate is a proposal; only the eight selected joint values and
        # their control targets differ from the input calibration.
        args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"passed": passed, "output": str(args.output), "report": str(args.report),
                      "min_table_clearance_m": check["min_table_clearance_m"],
                      "min_socket_clearance_m": check["min_socket_clearance_m"]}), flush=True)


if __name__ == "__main__":
    main()
