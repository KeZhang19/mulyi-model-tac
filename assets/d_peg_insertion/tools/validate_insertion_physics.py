#!/usr/bin/env python3
"""Validate real D-peg/socket contacts with force control in three small scenes.

This script creates no robot. The columns are aligned, X-offset 5 mm, and yaw
180 degrees. After an initial gravity check, a bounded wrench drives each free
peg toward 28 mm insertion. No pose/joint constraint advances the insertion.
Run this separately from other Isaac Sim processes on a small GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


PACKAGE = Path(__file__).resolve().parents[1]


def contact_force_peaks(full_rate_norms, sampled_norms):
    """Keep physical acceptance independent of the trajectory logging stride."""
    if not full_rate_norms or not sampled_norms:
        raise ValueError("Contact acceptance needs full-rate and trajectory samples")
    return {
        "maximum_socket_contact_n": max(full_rate_norms),
        "trajectory_sampled_maximum_socket_contact_n": max(sampled_norms),
        "full_rate_contact_sample_count": len(full_rate_norms),
        "trajectory_contact_sample_count": len(sampled_norms),
    }


def run(args):
    import math
    import numpy as np
    import torch
    from pxr import Usd, UsdPhysics, PhysxSchema
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

    asset_hashes = {
        "peg_sha256": hashlib.sha256((PACKAGE / "peg.usd").read_bytes()).hexdigest(),
        "socket_sha256": hashlib.sha256(args.socket_usd.read_bytes()).hexdigest(),
    }
    cfg = sim_utils.SimulationCfg(dt=1 / 240, device=args.device, gravity=(0., 0., -9.81))
    cfg.physx.solver_type = 1
    cfg.physx.min_position_iteration_count = 32
    cfg.physx.min_velocity_iteration_count = 8
    sim = sim_utils.SimulationContext(cfg)
    stage = sim.stage
    light = sim_utils.DomeLightCfg(intensity=1200, color=(.9, .93, 1.0))
    light.func("/World/Light", light)
    table = sim_utils.CuboidCfg(
        size=(1.0, .4, .05), collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.4, .45, .5)),
    )
    table.func("/World/Table", table, translation=(0, 0, .735))
    cases = (("aligned", -.3, 0., 0.), ("offset_5mm", 0., .005, 0.), ("yaw_180", .3, 0., math.pi))
    pegs, sockets, sensors, targets = [], [], [], []
    for name, x, offset, yaw in cases:
        root = f"/World/{name}"
        sim_utils.create_prim(root, "Xform")
        socket = RigidObject(RigidObjectCfg(
            prim_path=root + "/Socket", spawn=sim_utils.UsdFileCfg(usd_path=str(args.socket_usd)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0., .760)),
        ))
        quat = (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))
        peg = RigidObject(RigidObjectCfg(
            prim_path=root + "/Peg", spawn=sim_utils.UsdFileCfg(usd_path=str(PACKAGE / "peg.usd"), activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x + offset, 0., .840), rot=quat),
        ))
        sensor = ContactSensor(ContactSensorCfg(
            prim_path=root + "/Peg", update_period=0., history_length=1,
            filter_prim_paths_expr=[root + "/Socket"],
        ))
        pegs.append(peg); sockets.append(socket); sensors.append(sensor)
        targets.append(((x + offset, 0., .810 - .028), quat))
    joints = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert not joints, joints
    for socket, peg in zip(sockets, pegs):
        socket_root = stage.GetPrimAtPath(socket.cfg.prim_path)
        peg_root = stage.GetPrimAtPath(peg.cfg.prim_path)
        assert UsdPhysics.RigidBodyAPI(socket_root).GetKinematicEnabledAttr().Get()
        assert not UsdPhysics.RigidBodyAPI(peg_root).GetKinematicEnabledAttr().Get()
        assert not peg_root.GetAttribute("physxRigidBody:disableGravity").Get()
        assert abs(UsdPhysics.MassAPI(peg_root).GetMassAttr().Get() - .1) < 1e-6
        if args.disable_peg_sleep:
            PhysxSchema.PhysxRigidBodyAPI(peg_root).CreateSleepThresholdAttr().Set(0.0)
        colliders = [p for p in Usd.PrimRange(socket_root) if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(colliders) == args.expected_socket_colliders, len(colliders)
        assert all(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == args.socket_approximation for p in colliders)
    sim.set_camera_view((1.0, -1.4, 1.35), (0., 0., .81))
    sim.reset()
    for asset in (*pegs, *sockets):
        asset.reset()
    for sensor in sensors:
        sensor.reset()
    dt = sim.get_physics_dt()

    def advance():
        for asset in (*pegs, *sockets):
            asset.write_data_to_sim()
        sim.step(render=bool(args.render))
        for asset in (*pegs, *sockets):
            asset.update(dt)
        for sensor in sensors:
            sensor.update(dt, force_recompute=True)

    initial_poses = [peg.data.root_pose_w.clone() for peg in pegs]
    initial_socket_poses = [socket.data.root_pose_w.clone() for socket in sockets]
    initial_vz = [float(peg.data.root_lin_vel_w[0, 2]) for peg in pegs]
    # With 30 mm initial clearance, 50 ms free fall stays above the mouth.
    for _ in range(12):
        advance()
    gravity_errors = [abs(float(peg.data.root_lin_vel_w[0, 2]) - vz + 9.81 * 12 * dt)
                      for peg, vz in zip(pegs, initial_vz)]
    assert max(gravity_errors) < .04, gravity_errors
    records = [[] for _ in cases]
    full_rate_contacts = [[] for _ in cases]
    contact_probe = [[] for _ in cases]
    for step in range(round(args.seconds / dt)):
        applied_forces = []
        for index, (peg, (position, quat)) in enumerate(zip(pegs, targets)):
            target_position = torch.tensor(position, device=args.device).view(1, 3)
            target_quat = torch.tensor(quat, device=args.device).view(1, 4)
            error = target_position - peg.data.root_pos_w
            force = 120.0 * error - 4.0 * peg.data.root_lin_vel_w
            force[:, :2] = force[:, :2].clamp(-1.5, 1.5)
            target_vz = (5.0 * error[:, 2]).clamp(-.015, .015)
            force[:, 2] = .1 * 9.81 + (12.0 * (target_vz - peg.data.root_lin_vel_w[:, 2])).clamp(-1.5, 1.5)
            if args.probe_down_force > 0 and index > 0 and step * dt >= args.seconds - 1.0:
                force[:, 2] -= args.probe_down_force
            rotation_error = axis_angle_from_quat(quat_mul(target_quat, quat_conjugate(peg.data.root_quat_w)))
            torque = (.7 * rotation_error - .02 * peg.data.root_ang_vel_w).clamp(-.15, .15)
            peg.set_external_force_and_torque(force.view(1, 1, 3), torque.view(1, 1, 3), is_global=True)
            if args.contact_probe:
                applied_forces.append(force.clone())
        advance()
        # Contact impulses can alternate with the physics step. Sampling every
        # twelve steps can therefore report zero even during repeated contact.
        for index, sensor in enumerate(sensors):
            contact = sensor.data.force_matrix_w
            full_rate_contacts[index].append(float(torch.linalg.vector_norm(contact, dim=-1).max()))
        if args.contact_probe:
            for index, (peg, sensor) in enumerate(zip(pegs, sensors)):
                filtered = sensor.data.force_matrix_w.reshape(-1, 3).sum(dim=0)
                net = sensor.data.net_forces_w.reshape(-1, 3).sum(dim=0)
                contact_probe[index].append({
                    "time_s": (step + 1) * dt,
                    "filtered_force_w_n": filtered.cpu().tolist(),
                    "net_force_w_n": net.cpu().tolist(),
                    "filtered_force_norm_n": float(torch.linalg.vector_norm(filtered)),
                    "net_force_norm_n": float(torch.linalg.vector_norm(net)),
                    "position_w_m": peg.data.root_pos_w[0].cpu().tolist(),
                    "linear_velocity_w_m_s": peg.data.root_lin_vel_w[0].cpu().tolist(),
                    "applied_force_w_n": applied_forces[index][0].cpu().tolist(),
                })
        if step % 12 == 0:
            for index, (peg, sensor) in enumerate(zip(pegs, sensors)):
                records[index].append({
                    "time_s": (step + 1) * dt,
                    "depth_m": .810 - float(peg.data.root_pos_w[0, 2]),
                    "speed_m_s": float(torch.linalg.vector_norm(peg.data.root_lin_vel_w[0])),
                    "socket_contact_n": full_rate_contacts[index][-1],
                    "xy_error_m": float(torch.linalg.vector_norm(
                        peg.data.root_pos_w[0, :2] - torch.tensor(targets[index][0][:2], device=args.device))),
                })
    summary = {}
    for index, (name, *_rest) in enumerate(cases):
        depths = np.asarray([r["depth_m"] for r in records[index]])
        tail = depths[-10:]
        summary[name] = {
            "final_depth_m": float(depths[-1]), "maximum_depth_m": float(depths.max()),
            "final_half_second_depth_std_m": float(tail.std()),
            **contact_force_peaks(full_rate_contacts[index], [r["socket_contact_n"] for r in records[index]]),
            "max_xy_error_m": max(r["xy_error_m"] for r in records[index]),
            "gravity_velocity_error_m_s": gravity_errors[index],
            "socket_displacement_m": float(torch.linalg.vector_norm(
                sockets[index].data.root_pos_w - initial_socket_poses[index][:, :3])),
            "initial_tip_m": initial_poses[index][0, :3].cpu().tolist(),
            "final_tip_m": pegs[index].data.root_pos_w[0].cpu().tolist(),
        }
    checks = {
        "aligned_reaches_28mm": abs(summary["aligned"]["final_depth_m"] - .028) < .001,
        "aligned_remains_stable": summary["aligned"]["final_half_second_depth_std_m"] < .0005,
        "offset_is_blocked": summary["offset_5mm"]["maximum_depth_m"] < .010,
        "wrong_yaw_is_blocked": summary["yaw_180"]["maximum_depth_m"] < .010,
        "blocked_cases_have_contacts": all(summary[n]["maximum_socket_contact_n"] > .05 for n in ("offset_5mm", "yaw_180")),
        "socket_stays_fixed": all(r["socket_displacement_m"] < 1e-5 for r in summary.values()),
        "no_joint_constraint": not joints,
    }
    result = {"passed": all(checks.values()), "checks": checks, "cases": summary, "samples": records,
              "controller": "Bounded world wrench; 15 mm/s Z velocity servo; XY and orientation PD; no pose writes after spawn.",
              "device": args.device, "dt": dt, "seconds": args.seconds,
              "peg_usd": str(PACKAGE / "peg.usd"), "socket_usd": str(args.socket_usd),
              "socket_collider_count": args.expected_socket_colliders,
              "socket_approximation": args.socket_approximation,
              **asset_hashes,
              "contact_sampling": {
                  "acceptance_sample_period_s": dt,
                  "trajectory_sample_period_s": 12 * dt,
                  "acceptance_statistic": "maximum filtered socket force over every physics step",
                  "blocked_contact_threshold_n": .05,
                  "contact_force_threshold_changed": False,
              }}
    if args.contact_probe:
        result["contact_probe"] = {
            "disable_peg_sleep": args.disable_peg_sleep,
            "additional_down_force_n_last_second_blocked_cases": args.probe_down_force,
            "sample_period_s": dt,
            "summary": {name: {
                "maximum_filtered_force_n": max(row["filtered_force_norm_n"] for row in rows),
                "maximum_net_force_n": max(row["net_force_norm_n"] for row in rows),
                "nonzero_filtered_samples": sum(row["filtered_force_norm_n"] > 1e-6 for row in rows),
                "nonzero_net_samples": sum(row["net_force_norm_n"] > 1e-6 for row in rows),
                "sample_count": len(rows),
            } for (name, *_), rows in zip(cases, contact_probe)},
            "samples": dict((case[0], rows) for case, rows in zip(cases, contact_probe)),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    if args.scene_export:
        args.scene_export.parent.mkdir(parents=True, exist_ok=True)
        stage.Flatten().Export(str(args.scene_export))
    print(json.dumps({"passed": result["passed"], "checks": checks, "cases": summary}, indent=2), flush=True)
    assert result["passed"], f"Physical insertion check failed: {checks}; see {args.report}"


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--socket-usd", type=Path, default=PACKAGE / "socket.usd")
    parser.add_argument("--expected-socket-colliders", type=int, default=133)
    parser.add_argument("--socket-approximation", choices=("convexHull", "none"), default="convexHull")
    parser.add_argument("--contact-probe", action="store_true", help="Record filtered/net force, velocity and applied wrench at every physics step.")
    parser.add_argument("--disable-peg-sleep", action="store_true", help="Contact diagnostic: disable peg sleep; original default is unchanged.")
    parser.add_argument("--probe-down-force", type=float, default=0.0, help="Contact diagnostic: additional downward load (N) in the final second, blocked cases only.")
    parser.add_argument("--report", type=Path, default=PACKAGE / "insertion_physics_validation.json")
    parser.add_argument("--scene-export", type=Path)
    parser.add_argument("--render", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    options = parser.parse_args()
    if not 0.0 <= options.probe_down_force <= 1.5:
        parser.error("--probe-down-force must be between 0 and 1.5 N")
    if (options.disable_peg_sleep or options.probe_down_force > 0) and not options.contact_probe:
        parser.error("Contact diagnostic overrides require --contact-probe so the changed protocol is recorded")
    application = AppLauncher(options).app
    code = 0
    try:
        run(options)
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        application.close(wait_for_replicator=False, skip_cleanup=True)
    sys.exit(code)
