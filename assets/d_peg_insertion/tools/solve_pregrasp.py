#!/usr/bin/env python3
"""Solve a reproducible geometric seed; physical calibration is a separate step.

Run with the brainco Python (NumPy/SciPy required). The output is deliberately
marked unvalidated until validate_pregrasp.py has measured a gravity hold.
"""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/d_peg_insertion"


class Kinematics:
    def __init__(self):
        self.path = ROOT / "assets/urdf/Tianji_Revo3/urdf/Tianji_Revo3_Right.urdf"
        self.xml = ET.parse(self.path).getroot()
        self.joints = []
        self.names, self.bounds = [], []
        for joint in self.xml.findall("joint"):
            origin = joint.find("origin")
            matrix = np.eye(4)
            if origin is not None:
                matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.zeros(3)
            name = joint.get("name")
            movable = joint.get("type") != "fixed"
            if movable:
                self.names.append(name)
                limit = joint.find("limit")
                self.bounds.append([float(limit.get("lower")), float(limit.get("upper"))])
            self.joints.append((name, joint.find("parent").get("link"), joint.find("child").get("link"), matrix, axis, movable))
        self.bounds = np.array(self.bounds) + [1.e-4, -1.e-4]

    def fk(self, q, root=None, hand_only=False):
        q = dict(zip(self.names, q)) if not isinstance(q, dict) else q
        poses = {"base_link" if hand_only else "Base_R": np.eye(4) if root is None else root}
        pending = list(self.joints)
        while pending:
            progressed = False
            for joint in pending[:]:
                name, parent, child, matrix, axis, movable = joint
                if parent not in poses:
                    continue
                local = matrix.copy()
                if movable:
                    local[:3, :3] = matrix[:3, :3] @ Rotation.from_rotvec(axis * q.get(name, 0.)).as_matrix()
                poses[child] = poses[parent] @ local
                pending.remove(joint)
                progressed = True
            if not progressed:
                break
        return poses


def solve(output):
    kin = Kinematics()
    hand_names = [n for n in kin.names if n.startswith("right_")]
    hand_ids = [kin.names.index(n) for n in hand_names]
    bounds = kin.bounds[hand_ids]
    initial = np.array([.05 if "yaw" in n else .8 for n in hand_names])
    initial[hand_names.index("right_thumbcmp_roll_joint")] = 1.3
    initial[hand_names.index("right_thumbcmr_roll_joint")] = .8
    initial = np.clip(initial, bounds[:, 0]+1e-4, bounds[:, 1]-1e-4)
    # Tilt the handle axis in the palm frame so the wrist clears the table
    # throughout the full 58 mm insertion travel, not only at the reset height.
    angle = np.deg2rad(35)
    axis = np.array([0., np.cos(angle), -np.sin(angle)])
    radial = np.array([0., np.sin(angle), np.cos(angle)])
    center = np.array([.05, .03, .135])
    targets = {"thumb": center - .025*radial + .014*axis,
               "index": center + .025*radial + .005*axis,
               "mid": center + .025*radial - .018*axis}

    def residual(q):
        poses = kin.fk(dict(zip(hand_names, q)), hand_only=True)
        errors = []
        for name, goal in targets.items():
            pose = poses[f"right_{name}dip_roll_rubber_link"]
            surface = {"thumb": [.0,.033,-.0002], "index": [.009,.001,.021], "mid": [.010,.001,.050]}[name]
            point = pose[:3,:3] @ surface + pose[:3,3]
            errors.extend((point-goal)*100)
            normal = pose[:3, 1 if name == "thumb" else 0]
            inward = center-point
            inward -= axis * np.dot(axis,inward)
            inward /= np.linalg.norm(inward)
            errors.extend((normal-inward)*.3)
        errors.extend((q-initial)*.01)
        return errors

    result = least_squares(residual, initial, bounds=bounds.T, max_nfev=700)
    hand = dict(zip(hand_names, result.x))
    # Keep the two lower fingers outside the short handle and table sweep.
    for stem in ("ring", "pinky"):
        for suffix, value in (("mcp_yaw", -.1), ("mcp_roll", 1.2), ("pip_roll", 1.3), ("dip_roll", .8)):
            hand[f"right_{stem}{suffix}_joint"] = value
    tip = np.array([.45, .10, .84])
    grip_world = tip + [0, 0, .06]
    robot_root = np.eye(4)
    robot_root[:3, :3] = Rotation.from_euler("z", -np.pi/2).as_matrix()
    robot_root[:3, 3] = [1, 0, .766]
    # Choose the free wrist heading by multiple-start IK, keeping joints in limits.
    arm_names = [f"Joint{i}_R" for i in range(1, 8)]
    arm_bounds = kin.bounds[:7]
    rng = np.random.default_rng(731)
    candidates = []
    for yaw in np.linspace(-np.pi, np.pi, 13):
        rotation = Rotation.from_euler("z", yaw).as_matrix() @ np.array([[1,0,0],[0,0,-1],[0,1,0]]) @ Rotation.from_euler("x",angle).as_matrix()
        target = np.eye(4)
        target[:3, :3] = rotation
        target[:3, 3] = grip_world - rotation @ center

        def arm_residual(q):
            pose = kin.fk(dict(zip(arm_names, q)), robot_root)["base_link"]
            return np.r_[(pose[:3, 3]-target[:3, 3])*10,
                         Rotation.from_matrix(target[:3,:3].T @ pose[:3,:3]).as_rotvec()]

        for _ in range(4):
            seed = rng.uniform(arm_bounds[:,0]+.01, arm_bounds[:,1]-.01)
            solution = least_squares(arm_residual, seed, bounds=arm_bounds.T, max_nfev=180)
            if np.linalg.norm(solution.fun) < .001:
                poses = kin.fk(dict(zip(arm_names, solution.x)), robot_root)
                min_z = min(poses[f"Link{i}_R"][2,3] for i in range(2,8))
                if min_z > .85:
                    candidates.append((np.linalg.norm(solution.x)*.01 - min_z, solution.x, target, yaw))
    if not candidates:
        raise RuntimeError("No arm IK solution with tabletop clearance")
    _, arm, target, yaw = min(candidates, key=lambda x:x[0])
    positions = dict(zip(arm_names, arm)) | hand
    data = dict(schema_version=1, validated=False,
                provenance="URDF bounded least-squares seed; requires PhysX calibration",
                robot_joint_positions=positions, robot_joint_targets=positions.copy(),
                peg_pose=dict(position=tip.tolist(), quaternion=[1.,0.,0.,0.]),
                seed_hand_center=center.tolist(), seed_hand_axis=axis.tolist(), seed_hand_pose=target.tolist(),
                hand_fit_cost=float(result.cost), arm_heading_rad=float(yaw))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps(data, indent=2))


def recenter_calibration(source, output):
    """Move a physically found grasp as a whole, using bounded arm IK only."""
    kin = Kinematics()
    data = json.loads(source.read_text())
    root = np.eye(4)
    root[:3,:3] = Rotation.from_euler("z",-np.pi/2).as_matrix()
    root[:3,3] = [1,0,.766]
    old_hand = kin.fk(data["robot_joint_positions"],root)["base_link"]
    old_peg = np.eye(4)
    pose = data["peg_pose"]
    old_peg[:3,3] = pose["position"]
    old_peg[:3,:3] = Rotation.from_quat(np.array(pose["quaternion"])[[1,2,3,0]]).as_matrix()
    goal_peg = np.eye(4)
    goal_peg[:3,3] = [.45,.10,.84]
    goal_hand = goal_peg @ np.linalg.inv(old_peg) @ old_hand
    names = [f"Joint{i}_R" for i in range(1,8)]
    q0 = np.array([data["robot_joint_positions"][n] for n in names])
    def residual(q):
        p = kin.fk(dict(zip(names,q)),root)["base_link"]
        return np.r_[(p[:3,3]-goal_hand[:3,3])*10,
                     Rotation.from_matrix(goal_hand[:3,:3].T@p[:3,:3]).as_rotvec()]
    rng = np.random.default_rng(13)
    solutions = []
    for seed in [q0,*rng.uniform(kin.bounds[:7,0],kin.bounds[:7,1],size=(20,7))]:
        fit = least_squares(residual,seed,bounds=kin.bounds[:7].T,max_nfev=200)
        if np.linalg.norm(fit.fun)<1e-4:
            solutions.append(fit.x)
    if not solutions:
        raise RuntimeError("Recentered grasp unreachable")
    q = min(solutions,key=lambda q:np.linalg.norm(q-q0))
    for name,value,old in zip(names,q,q0):
        delta = data["robot_joint_targets"][name]-old
        data["robot_joint_positions"][name] = float(value)
        data["robot_joint_targets"][name] = float(value+delta)
    data["peg_pose"] = dict(position=[.45,.1,.84],quaternion=[1.,0.,0.,0.])
    data["validated"] = False
    data["provenance"] = "Gravity-calibrated finger grasp recentered with bounded arm IK; repeat restored-state hold before acceptance"
    output.write_text(json.dumps(data,indent=2)+"\n")
    print("Recentered calibration",output,"arm",q.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ASSETS / "pregrasp_seed.json")
    parser.add_argument("--recenter",type=Path)
    args = parser.parse_args()
    if args.recenter:
        recenter_calibration(args.recenter,args.output)
    else:
        solve(args.output)
