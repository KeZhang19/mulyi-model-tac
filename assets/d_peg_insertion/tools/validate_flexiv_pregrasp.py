#!/usr/bin/env python3
"""Validate a Flexiv D-peg editor export against the registered Isaac task.

The validator performs two checks with two environments: deterministic startup,
full resets and a partial reset, followed by a two-second zero-action rollout.
The default path keeps the production observation graph enabled (with tactile
policy disabled); ``--geometry-only`` is an explicit reduced-observation
ablation and is reported as such.

This is a diagnostic only. It never edits the calibration JSON or starts PPO.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/d_peg_insertion"
DEFAULT_TASK = "BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0"
# The validator compares the imported calibration against the original editor
# export; the task calibration itself is not an editor ``simulation_state``.
DEFAULT_REFERENCE = ASSETS / "flexiv_grasp_reference/20260913_142053_9886dc87/simulation_state.json"


def _finite(value):
    """Recursively test tensors/containers for finite values."""
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(v) for v in value)
    return True


def _root_pose(asset):
    data = asset.data
    for name in ("root_link_pose_w", "root_pose_w"):
        if hasattr(data, name):
            return getattr(data, name)
    raise AttributeError(f"Asset {asset!r} has no root pose tensor")


def _sensor_for_finger(env, finger):
    aliases = {
        "thumb": ("right_thumbdip_roll_rubber_link_object_s",),
        "index": ("right_indexdip_roll_rubber_link_object_s",),
        "middle": ("right_middip_roll_rubber_link_object_s", "right_middledip_roll_rubber_link_object_s"),
        "ring": ("right_ringdip_roll_rubber_link_object_s",),
        "little": ("right_pinkydip_roll_rubber_link_object_s", "right_littledip_roll_rubber_link_object_s"),
    }
    for name in aliases[finger]:
        if name in env.scene.sensors:
            return name, env.scene.sensors[name]
    return None, None


def _force_tensor(sensor, num_envs):
    import torch
    force = sensor.data.force_matrix_w
    return force.reshape(num_envs, -1, 3).norm(dim=-1).amax(dim=-1)


def _compare_pose(report, name, actual, expected, origins, position_tol, rotation_tol):
    import torch
    actual = actual.to(dtype=torch.float64)
    origins = origins.to(device=actual.device, dtype=torch.float64)
    expected_pos = torch.as_tensor(expected["position_m"], device=actual.device, dtype=torch.float64)
    expected_quat = torch.as_tensor(expected["quaternion_wxyz"], device=actual.device, dtype=torch.float64)
    pos_err = (actual[:, :3] - origins - expected_pos).abs().amax(dim=-1)
    aq = actual[:, 3:7] / actual[:, 3:7].norm(dim=-1, keepdim=True)
    eq = expected_quat / expected_quat.norm()
    dot = (aq * eq).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    passed = bool(torch.isfinite(actual).all()) and bool((pos_err <= position_tol).all()) and bool((angle <= rotation_tol).all())
    report["checks"].append({
        "name": name,
        "max_position_error_m": float(pos_err.max()),
        "max_rotation_error_rad": float(angle.max()),
        "position_tolerance_m": position_tol,
        "rotation_tolerance_rad": rotation_tol,
        "passed": passed,
    })
    return passed


def _compare_tensor(report, name, actual, expected, tolerance):
    import torch
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected, device=actual.device, dtype=actual.dtype)
    error = (actual - expected).abs().max()
    passed = bool(torch.isfinite(actual).all()) and bool(error <= tolerance)
    report["checks"].append({"name": name, "max_absolute_error": float(error), "tolerance": tolerance, "passed": passed})
    return passed


def validate(args, report):
    import gymnasium as gym
    import torch
    from isaaclab.managers import ObservationTermCfg
    from isaaclab_tasks.utils import parse_env_cfg

    sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
    import BrainCo_DexHand.tasks  # noqa: F401

    ref_bytes = args.reference.read_bytes()
    reference = json.loads(ref_bytes)
    if reference.get("schema") != "revo3_simulation_state":
        raise ValueError("reference must be an editor revo3_simulation_state export")
    report.update(reference=str(args.reference.resolve()), reference_sha256=hashlib.sha256(ref_bytes).hexdigest(),
                  source_validation=reference.get("validation"), geometry_only=args.geometry_only,
                  full_task_observations=not args.geometry_only, tactile_policy_enabled=False,
                  checks=[], failed_checks=[])

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.tactile_policy_enabled = False
    cfg.socket_xy_randomization_m = 0.0
    cfg.socket_yaw_randomization_deg = 0.0
    if args.geometry_only:
        for group_name, group in list(vars(cfg.observations).items()):
            if group is None:
                continue
            if group_name != "proprio":
                setattr(cfg.observations, group_name, None)
                continue
            for term_name, term in list(vars(group).items()):
                if isinstance(term, ObservationTermCfg) and term_name not in {"joint_pos", "joint_vel"}:
                    setattr(group, term_name, None)
        cfg.events.zz_initialize_tactile_contract = None

    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        robot, peg, socket = env.scene["robot"], env.scene["object"], env.scene["socket"]
        table = env.scene["table"]
        origins = env.scene.env_origins
        expected_joints = dict(zip(reference["robot"]["joint_names"], reference["robot"]["joint_positions_rad"], strict=True))
        if set(robot.joint_names) != set(expected_joints):
            raise AssertionError(f"joint names differ: sim={robot.joint_names}, reference={list(expected_joints)}")
        joint_values = torch.tensor([expected_joints[n] for n in robot.joint_names], device=env.device, dtype=robot.data.joint_pos.dtype)
        # Verify that the task actually imported the editor export, rather than
        # merely matching a second copy of the reference in this validator.
        imported = cfg._d_peg_pregrasp
        # A calibrated grasp may deliberately drive toward a different target
        # than the restored measured joint pose.  Validate both contracts
        # independently: positions must match the editor export, while
        # targets come from the imported calibration (including preload).
        expected_targets = dict(imported["robot_joint_targets"])
        imported_joints = torch.tensor([imported["robot_joint_positions"][n] for n in robot.joint_names], device=env.device, dtype=torch.float64)
        export_joints = torch.tensor([expected_joints[n] for n in robot.joint_names], device=env.device, dtype=torch.float64)
        if not torch.allclose(imported_joints, export_joints, atol=args.joint_tolerance, rtol=0.0):
            raise AssertionError("pregrasp_flexiv.json joint_positions do not match the supplied editor export")
        for field, expected in (("peg_pose", reference["object"]), ("socket_pose", reference["fixtures"][0])):
            actual = imported[field]
            if max(abs(float(a) - float(b)) for a, b in zip(actual["position"], expected["position_m"])) > args.position_tolerance:
                raise AssertionError(f"imported {field} position does not match the supplied editor export")
            if max(abs(float(a) - float(b)) for a, b in zip(actual["quaternion"], expected["quaternion_wxyz"])) > args.rotation_tolerance:
                raise AssertionError(f"imported {field} quaternion does not match the supplied editor export")
        report["import_matches_reference"] = True
        report.update(action_dim=int(env.action_manager.total_action_dim), observation_spaces=str(env.observation_manager),
                      joint_names=list(robot.joint_names), sensor_names=list(env.scene.sensors))
        if env.action_manager.total_action_dim != 28:
            raise AssertionError(f"expected 28 actions, got {env.action_manager.total_action_dim}")

        def refresh():
            env.scene.write_data_to_sim()
            env.sim.physics_sim_view.update_articulations_kinematic()
            env.sim.forward()

        def state_check(phase, ids):
            refresh()
            ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
            ok = True
            for key, expected in (("robot", reference["robot"]["base"]), ("object", reference["object"]),
                                  ("socket", reference["fixtures"][0])):
                ok &= _compare_pose(report, f"{phase}.{key}", _root_pose(env.scene[key])[ids], expected,
                                    origins[ids], args.position_tolerance, args.rotation_tolerance)
            ok &= _compare_tensor(report, f"{phase}.robot.joint_positions_rad", robot.data.joint_pos[ids], joint_values.expand(len(ids), -1), args.joint_tolerance)
            ok &= _compare_tensor(report, f"{phase}.robot.joint_velocities_rad_s", robot.data.joint_vel[ids], torch.zeros_like(robot.data.joint_vel[ids]), args.joint_tolerance)
            target_values = torch.tensor([expected_targets[n] for n in robot.joint_names], device=env.device, dtype=robot.data.joint_pos_target.dtype)
            ok &= _compare_tensor(report, f"{phase}.robot.joint_position_targets_rad", robot.data.joint_pos_target[ids], target_values.expand(len(ids), -1), args.joint_tolerance)
            return bool(ok)

        obs, _ = env.reset(seed=args.seed)
        if not _finite(obs):
            raise AssertionError("startup observations contain non-finite values")
        ids = torch.arange(env.num_envs, device=env.device)
        state_check("startup", ids)
        for i in range(3):
            obs, _ = env.reset()
            if not _finite(obs):
                raise AssertionError(f"reset {i + 1} observations contain non-finite values")
            state_check(f"full_reset_{i + 1}", ids)

        selected, untouched = ids[::2], ids[1::2]
        for key in ("object", "socket"):
            pose = _root_pose(env.scene[key]).clone()
            pose[:, 0] += (ids.to(pose.dtype) + 1.0) * 0.003
            env.scene[key].write_root_pose_to_sim(pose)
        q = robot.data.joint_pos.clone(); q[:, 0] += 0.01
        robot.write_joint_state_to_sim(q, torch.zeros_like(q)); robot.set_joint_position_target(q)
        refresh()
        before_obj = _root_pose(env.scene["object"])[untouched].clone(); before_sock = _root_pose(env.scene["socket"])[untouched].clone()
        before_q = robot.data.joint_pos[untouched].clone()
        env._reset_idx(selected)
        state_check("partial_reset", selected)
        _compare_tensor(report, "partial_reset.untouched.object", _root_pose(env.scene["object"])[untouched], before_obj, 1e-7)
        _compare_tensor(report, "partial_reset.untouched.socket", _root_pose(env.scene["socket"])[untouched], before_sock, 1e-7)
        _compare_tensor(report, "partial_reset.untouched.robot.joints", robot.data.joint_pos[untouched], before_q, 1e-7)

        initial_peg = _root_pose(peg).clone(); initial_socket = _root_pose(socket).clone()
        finger_names, finger_sensors = [], []
        for finger in ("thumb", "index", "middle", "ring", "little"):
            name, sensor = _sensor_for_finger(env, finger)
            if sensor is not None:
                finger_names.append(finger); finger_sensors.append(sensor)
        if len(finger_sensors) < 2:
            raise AssertionError(f"fewer than two fingertip object contact sensors found: {finger_names}")
        min_forces = torch.full((env.num_envs, len(finger_sensors)), float("inf"), device=env.device)
        max_drift = torch.zeros(env.num_envs, device=env.device)
        max_socket_drift = torch.zeros_like(max_drift)
        max_angle = torch.zeros_like(max_drift)
        term_count = torch.zeros(env.num_envs, device=env.device)
        trunc_count = torch.zeros_like(term_count)
        obs_finite = True
        steps = round(args.seconds / env.step_dt)
        zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(steps):
            obs, _, terminated, truncated, _ = env.step(zero)
            obs_finite &= _finite(obs)
            term_count += terminated.float(); trunc_count += truncated.float()
            min_forces = torch.minimum(min_forces, torch.stack([_force_tensor(s, env.num_envs) for s in finger_sensors], dim=-1))
            max_drift = torch.maximum(max_drift, (_root_pose(peg)[:, :3] - initial_peg[:, :3]).norm(dim=-1))
            max_socket_drift = torch.maximum(max_socket_drift, (_root_pose(socket)[:, :3] - initial_socket[:, :3]).norm(dim=-1))
            aq, iq = _root_pose(peg)[:, 3:], initial_peg[:, 3:]
            dot = (aq / aq.norm(dim=-1, keepdim=True) * iq / iq.norm(dim=-1, keepdim=True)).sum(dim=-1).abs().clamp(0., 1.)
            max_angle = torch.maximum(max_angle, 2 * torch.acos(dot))
        contact_ok = (min_forces[:, 0] > args.contact_threshold) & (min_forces[:, 1:].amax(dim=-1) > args.contact_threshold)
        rollout_ok = (max_drift < args.max_drift) & (max_socket_drift < args.max_socket_drift) & (max_angle < args.max_angle_rad) & contact_ok & (term_count == 0) & (trunc_count == 0) & obs_finite
        report.update(zero_action={"seconds": args.seconds, "steps": steps, "finger_order": finger_names,
                    "minimum_finger_force_N": min_forces.cpu().tolist(), "max_peg_drift_m": max_drift.cpu().tolist(),
                    "max_socket_drift_m": max_socket_drift.cpu().tolist(), "max_peg_angle_deg": (max_angle * 180.0 / 3.141592653589793).cpu().tolist(),
                    "terminated_count": term_count.cpu().tolist(), "truncated_count": trunc_count.cpu().tolist(),
                    "observations_finite": obs_finite, "per_env_passed": rollout_ok.cpu().tolist()})
        report["failed_checks"] = [c for c in report["checks"] if not c["passed"]]
        reasons = []
        if bool((max_drift >= args.max_drift).any()):
            reasons.append("zero_action_peg_drift_exceeds_threshold")
        if bool((max_socket_drift >= args.max_socket_drift).any()):
            reasons.append("zero_action_socket_drift_exceeds_threshold")
        if bool((max_angle >= args.max_angle_rad).any()):
            reasons.append("zero_action_peg_rotation_exceeds_threshold")
        if bool((~contact_ok).any()):
            reasons.append("zero_action_fingertip_contact_below_threshold")
        if bool((term_count > 0).any()):
            reasons.append("zero_action_terminated")
        if bool((trunc_count > 0).any()):
            reasons.append("zero_action_truncated")
        if not obs_finite:
            reasons.append("zero_action_observation_nonfinite")
        report["failed"].extend(reasons)
        report["passed"] = not report["failed_checks"] and bool(rollout_ok.all())
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--report", type=Path, default=ASSETS / "validation/flexiv_pregrasp_runtime.json")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--position-tolerance", type=float, default=2e-4)
    parser.add_argument("--rotation-tolerance", type=float, default=2e-3)
    parser.add_argument("--joint-tolerance", type=float, default=2e-5)
    parser.add_argument("--contact-threshold", type=float, default=1.0)
    parser.add_argument("--max-drift", type=float, default=3e-3)
    parser.add_argument("--max-socket-drift", type=float, default=1e-5)
    parser.add_argument("--max-angle-rad", type=float, default=0.0872664626)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 2:
        parser.error("Use exactly two environments for the requested 2-env check")
    report = {"task": args.task, "created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False, "failed": [], "traceback": None}
    app = AppLauncher(args).app
    try:
        validate(args, report)
    except BaseException as exc:
        report["failed"] = [str(exc)]
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"FLEXIV PREGRASP {'PASS' if report['passed'] else 'FAIL'}: {args.report}", flush=True)
        try:
            app.app.post_quit(0 if report["passed"] else 1)
        except Exception:
            pass
        app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
