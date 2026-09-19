#!/usr/bin/env python3
"""Measure real nonadjacent arm contacts in two isolated Isaac environments.

Run the repaired and donor-filter controls in separate processes, on an idle
GPU. This diagnostic has no policy, lamp, trainer, or connection to a live run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "assets/rotate_bulb_custom/robot/usd/rizon4_training.usda"
REFERENCE = ROOT / "assets/rotate_bulb_custom/grasp_reference/simulation_state.json"
PROBE_PAIR = ("link2", "link6")
TARGET_ARM_RAD = (-1.199339939926037, 1.5170282748780095, -2.3218149636501213,
                  2.6663067767137427, 0.9774095278906656, 2.4564157509496445, -2.4160773351160225)
START_JOINT4_RAD = 2.2


def pose_matrix(position, quaternion_wxyz):
    import numpy as np
    from scipy.spatial.transform import Rotation

    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.roll(quaternion_wxyz, -1)).as_matrix()
    result[:3, 3] = position
    return result


def arm_geometry(stage):
    """Return actual convex collider vertices and hull planes in body frames."""
    import numpy as np
    from pxr import Usd, UsdGeom
    from scipy.spatial import ConvexHull

    cache = UsdGeom.XformCache()
    result = {}
    for body in ("base_link", *(f"link{i}" for i in range(1, 8))):
        root = stage.GetPrimAtPath("/flexiv_rizon4/" + body)
        points = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(prim.GetPath()):
                continue
            transform = np.asarray(cache.GetLocalToWorldTransform(prim)
                                   * cache.GetLocalToWorldTransform(root).GetInverse()).T
            vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            points.append(vertices @ transform[:3, :3].T + transform[:3, 3])
        points = np.vstack(points)
        hull = ConvexHull(points)
        result[body] = (points[hull.vertices], hull.equations)
    return result


def forward_kinematics(stage, angles):
    import numpy as np
    from pxr import UsdPhysics
    from scipy.spatial.transform import Rotation

    poses = {"base_link": np.eye(4)}
    pending = [UsdPhysics.Joint(prim) for prim in stage.Traverse()
               if prim.IsA(UsdPhysics.Joint) and prim.GetName() != "root_joint"]
    while pending:
        previous_count = len(pending)
        for joint in pending.copy():
            parent = joint.GetBody0Rel().GetTargets()[0].name
            child = joint.GetBody1Rel().GetTargets()[0].name
            if parent not in poses:
                continue
            frames = []
            for pos, quat in ((joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                              (joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())):
                frames.append(pose_matrix(pos, [quat.GetReal(), *quat.GetImaginary()]))
            motion = np.eye(4)
            if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
                axis = UsdPhysics.RevoluteJoint(joint).GetAxisAttr().Get() or "X"
                motion[:3, :3] = Rotation.from_rotvec(
                    np.eye(3)["XYZ".index(axis)] * angles[joint.GetPrim().GetName()]).as_matrix()
            poses[child] = poses[parent] @ frames[0] @ motion @ np.linalg.inv(frames[1])
            pending.remove(joint)
        if len(pending) == previous_count:
            raise ValueError("Robot joint graph is disconnected")
    return poses


def intersection_inradius(geometry, poses, pair=PROBE_PAIR):
    """Positive value proves intersecting hulls; this is not contact penetration."""
    import numpy as np
    from scipy.optimize import linprog

    equations = []
    for body in pair:
        _, local = geometry[body]
        transform = poses[body]
        normals = local[:, :3] @ transform[:3, :3].T
        equations.append(np.column_stack((normals, local[:, 3] - normals @ transform[:3, 3])))
    planes = np.vstack(equations)
    result = linprog([0, 0, 0, -1], A_ub=np.column_stack((planes[:, :3], np.ones(len(planes)))),
                     b_ub=-planes[:, 3], bounds=[(None, None)] * 4, method="highs")
    if not result.success:
        raise RuntimeError(f"Collision geometry query failed: {result.message}")
    return float(result.x[-1])


def fixture_geometry():
    from pxr import Usd

    stage = Usd.Stage.Open(str(ASSET))
    reference = json.loads(REFERENCE.read_text())["robot"]
    angles = dict(zip(reference["joint_names"], reference["joint_positions_rad"], strict=True))
    angles.update({f"joint{i + 1}": value for i, value in enumerate(TARGET_ARM_RAD)})
    geometry = arm_geometry(stage)
    target_overlap = intersection_inradius(geometry, forward_kinematics(stage, angles))
    angles["joint4"] = START_JOINT4_RAD
    start_overlap = intersection_inradius(geometry, forward_kinematics(stage, angles))
    if start_overlap >= -0.002 or target_overlap <= 0.004:
        raise AssertionError(f"Collision probe fixture changed: start={start_overlap}, target={target_overlap}")
    return stage, geometry, angles, {
        "pair": list(PROBE_PAIR), "start_joint4_rad": START_JOINT4_RAD,
        "target_arm_rad": list(TARGET_ARM_RAD),
        "start_intersection_inradius_m": start_overlap,
        "target_intersection_inradius_m": target_overlap,
    }


def run_probe(args, report):
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import UsdGeom

    sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.robot_setup import (
        _make_revo3_robot_cfg_with_touch_compliance,
    )
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.self_collision import (
        RESTORED_PAIRS, spawn_bulb_robot_with_self_collision,
    )

    _, geometry, angles, fixture = fixture_geometry()
    report["fixture"] = fixture
    report["legacy_filters"] = args.legacy_filters
    report["num_envs"] = 2
    dt = 1.0 / 240.0
    simulation = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device=args.device, gravity=(0.0, 0.0, 0.0),
        physx=sim_utils.PhysxCfg(solver_type=1, min_position_iteration_count=32,
                               min_velocity_iteration_count=8),
    ))
    for index in range(2):
        prim = simulation.stage.DefinePrim(f"/World/envs/env_{index}", "Xform")
        UsdGeom.Xformable(prim).AddTranslateOp().Set((index * 3.0, 0.0, 0.0))
    cfg = _make_revo3_robot_cfg_with_touch_compliance().replace(prim_path="/World/envs/env_.*/Robot")
    cfg.init_state.joint_pos = dict(angles)
    if not args.legacy_filters:
        cfg.spawn = cfg.spawn.replace(func=spawn_bulb_robot_with_self_collision)
    robot = Articulation(cfg)
    sensor = ContactSensor(ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link2",
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/link6"],
        update_period=0.0, history_length=0, track_air_time=False,
    ))
    # Explicitly inspect every composed instance before creating physics views.
    filtered_counts = []
    for index in range(2):
        root = f"/World/envs/env_{index}/Robot"
        count = 0
        for first, second in RESTORED_PAIRS:
            for owner, other in ((first, second), (second, first)):
                targets = simulation.stage.GetPrimAtPath(root + "/" + owner).GetRelationship(
                    "physics:filteredPairs").GetTargets()
                count += sum(str(target) == root + "/" + other for target in targets)
        filtered_counts.append(count)
    report["restored_pair_exclusions_per_env"] = filtered_counts
    assert filtered_counts == ([192, 192] if args.legacy_filters else [0, 0]), filtered_counts
    print(f"[probe] native physics initialization; exclusions={filtered_counts}", flush=True)
    simulation.reset()
    print("[probe] native physics ready", flush=True)
    positions = robot.data.default_joint_pos.clone()
    velocities = torch.zeros_like(positions)
    robot.write_joint_state_to_sim(positions, velocities)
    robot.set_joint_position_target(positions)
    robot.write_data_to_sim()
    robot.reset()
    sensor.reset()
    joint4 = robot.joint_names.index("joint4")
    body_ids = {name: robot.body_names.index(name) for name in PROBE_PAIR}
    initial = positions.clone()
    max_force = np.zeros(2)
    steady_initial_error = np.zeros(2)
    samples = []
    settle_s, ramp_s, hold_s = 0.5, 3.0, 1.0
    for step in range(round((settle_s + ramp_s + hold_s) / dt)):
        time_s = step * dt
        blend = min(1.0, max(0.0, (time_s - settle_s) / ramp_s))
        positions[:, joint4] = START_JOINT4_RAD + blend * (TARGET_ARM_RAD[3] - START_JOINT4_RAD)
        robot.set_joint_position_target(positions)
        robot.write_data_to_sim()
        simulation.step(render=False)
        robot.update(dt)
        sensor.update(dt, force_recompute=True)
        if not torch.isfinite(robot.data.joint_pos).all():
            raise AssertionError("Non-finite robot state during collision probe")
        force = sensor.data.force_matrix_w
        if force is None or force.shape[0] != 2:
            raise AssertionError(f"Expected two per-environment filtered contact matrices, got {force}")
        force_n = torch.linalg.vector_norm(force, dim=-1).flatten(start_dim=1).sum(-1).cpu().numpy()
        max_force = np.maximum(max_force, force_n)
        if time_s < settle_s:
            arm_ids = [robot.joint_names.index(f"joint{i}") for i in range(1, 8)]
            error = (robot.data.joint_pos[:, arm_ids] - initial[:, arm_ids]).abs().max(-1).values.cpu().numpy()
            steady_initial_error = np.maximum(steady_initial_error, error)
        if step % 24 == 0 or step == round((settle_s + ramp_s + hold_s) / dt) - 1:
            overlaps = []
            for index in range(2):
                poses = {}
                for body, body_id in body_ids.items():
                    state = robot.data.body_link_pose_w[index, body_id].cpu().numpy()
                    poses[body] = pose_matrix(state[:3], state[3:7])
                overlaps.append(intersection_inradius(geometry, poses))
            samples.append({"time_s": time_s, "force_n": force_n.tolist(),
                            "intersection_inradius_m": overlaps,
                            "joint4_rad": robot.data.joint_pos[:, joint4].cpu().tolist()})
        if step % 240 == 0:
            print(f"[probe] step={step} pair_force_n={force_n.tolist()}", flush=True)
    final_overlap = np.asarray(samples[-1]["intersection_inradius_m"])
    report.update(max_pair_contact_force_n=max_force.tolist(),
                  initial_max_arm_deviation_rad=steady_initial_error.tolist(),
                  final_intersection_inradius_m=final_overlap.tolist(), samples=samples)
    # The donor-filter control must really penetrate the same native meshes.
    # Repaired robots must both report that exact pair's contact and resist it.
    checks = {
        "initial_arm_stable": bool((steady_initial_error < 0.05).all()),
        "both_environments_have_expected_physics": bool(
            ((max_force < 0.05).all() and (final_overlap > 0.004).all()) if args.legacy_filters
            else ((max_force > 0.5).all() and (final_overlap < 0.001).all())
        ),
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    if not report["passed"]:
        raise AssertionError(f"Native self-collision acceptance failed: {checks}")


def main():
    import faulthandler
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-filters", action="store_true", help="Run unchanged donor-filter control")
    parser.add_argument("--report", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(45, repeat=True)
    app = AppLauncher(args).app
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False}
    try:
        run_probe(args, report)
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        faulthandler.cancel_dump_traceback_later()
        # No recorder or trainer state exists in this isolated probe.  Avoid
        # waiting on background renderer/replicator cleanup after the durable
        # JSON result has been written.
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
