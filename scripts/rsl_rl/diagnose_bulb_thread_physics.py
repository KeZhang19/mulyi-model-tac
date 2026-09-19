#!/usr/bin/env python3
"""Diagnose the threaded bulb contact path with a pre-aligned rigid body.

This is a physics diagnostic, not a task-evaluation entrypoint.  Each
environment is reset normally and then the free bulb is placed once at the
measured first-entry pose.  During the trajectory the script applies only an
external wrench to the rigid bulb; it never writes a pose, phase, engagement,
installation, or success flag after that one-time diagnostic placement.

The forward and reverse torque sections answer a narrow physical question:
does a contacted bulb rotate in the expected insertion direction, accumulate
an axial advance close to the asset pitch, and retreat when the torque is
reversed?  The pre-aligned placement intentionally skips grasp/lift/alignment,
so a passing report cannot be used as complete-task acceptance evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import traceback


TASK = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0"
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "BrainCo_DexHand"


def _tolist(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _finite(*values) -> bool:
    """Check tensors and nested values without importing numpy."""
    import torch

    for value in values:
        if isinstance(value, dict):
            if not _finite(*value.values()):
                return False
        elif isinstance(value, (tuple, list)):
            if not _finite(*value):
                return False
        elif isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all().item()):
                return False
        elif isinstance(value, (float, int)):
            if value != value or abs(value) == float("inf"):
                return False
    return True


def _step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, extras = result
        return obs, reward, terminated | truncated, extras
    obs, reward, done, extras = result
    return obs, reward, done, extras


def _reset(env, seed: int | None = None):
    try:
        return env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        return env.reset()


def _contact_force(raw, name: str):
    """Return one maximum force magnitude per environment."""
    import torch

    sensor = raw.scene.sensors.get(name)
    force = None if sensor is None else getattr(sensor.data, "force_matrix_w", None)
    if force is None:
        return torch.zeros(raw.num_envs, device=raw.device), False
    force = torch.nan_to_num(force).reshape(raw.num_envs, -1, 3)
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=-1), True


def _relative_metrics(raw, assembly_module, angle_unwrapped, previous_raw):
    """Compute live relative translation, signed twist, and pitch residual."""
    import torch

    support, bulb = raw.scene["support"], raw.scene["object"]
    rel_pos, rel_quat = assembly_module.subtract_frame_transforms(
        support.data.root_pos_w,
        support.data.root_quat_w,
        bulb.data.root_pos_w,
        bulb.data.root_quat_w,
    )
    # The frozen asset's +X axis is the screw axis.  A quaternion projection is
    # sufficient here because the trajectory is deliberately axis-aligned;
    # the full relative quaternion is retained in the JSONL stream.
    raw_angle = 2.0 * torch.atan2(rel_quat[:, 1], rel_quat[:, 0])
    if previous_raw is not None:
        delta = torch.remainder(raw_angle - previous_raw + torch.pi, 2.0 * torch.pi) - torch.pi
        angle_unwrapped = angle_unwrapped + delta
    advance = assembly_module.ENTRY_ROOT_X_M - rel_pos[:, 0]
    expected = -angle_unwrapped * assembly_module.PITCH_M / (2.0 * torch.pi)
    residual = advance - expected
    return rel_pos, rel_quat, raw_angle, angle_unwrapped, advance, expected, residual


def _stabilization_wrench(
    raw,
    assembly_module,
    radial_stiffness: float,
    radial_damping: float,
    angular_stiffness: float,
    angular_damping: float,
):
    """Compute optional support-frame radial and axis-alignment feedback.

    The result is a wrench in world coordinates plus scalar diagnostics.  This
    helper only reads live rigid-body state.  It never writes pose, velocity,
    phase, engagement, installation, or success state.  All gains default to
    zero, so the default diagnostic remains the unassisted external-wrench
    trajectory used by the physics baseline.
    """
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    support, bulb = raw.scene["support"], raw.scene["object"]
    support_quat = support.data.root_quat_w
    axis_local = torch.zeros((raw.num_envs, 3), device=raw.device, dtype=support_quat.dtype)
    axis_local[:, 0] = 1.0
    target_axis_local = axis_local
    rel_pos, rel_quat = assembly_module.subtract_frame_transforms(
        support.data.root_pos_w,
        support_quat,
        bulb.data.root_pos_w,
        bulb.data.root_quat_w,
    )
    support_lin_vel = getattr(support.data, "root_lin_vel_w", torch.zeros_like(support.data.root_pos_w))
    support_ang_vel = getattr(support.data, "root_ang_vel_w", torch.zeros_like(support.data.root_pos_w))
    bulb_lin_vel = getattr(bulb.data, "root_lin_vel_w", torch.zeros_like(support.data.root_pos_w))
    bulb_ang_vel = getattr(bulb.data, "root_ang_vel_w", torch.zeros_like(support.data.root_pos_w))
    relative_lin_vel_local = assembly_module.quat_apply_inverse(
        support_quat, bulb_lin_vel - support_lin_vel
    )
    relative_ang_vel_local = assembly_module.quat_apply_inverse(
        support_quat, bulb_ang_vel - support_ang_vel
    )

    # Keep the support-frame screw axis component out of the radial spring.
    # The y/z terms pull the bulb centerline back to the support axis and
    # damp only radial motion.
    radial_error_local = rel_pos.clone()
    radial_error_local[:, 0] = 0.0
    radial_velocity_local = relative_lin_vel_local.clone()
    radial_velocity_local[:, 0] = 0.0
    radial_force_local = -radial_stiffness * radial_error_local - radial_damping * radial_velocity_local

    # For a body axis slightly tilted away from +X, axis x target gives a
    # restoring torque direction.  Project angular velocity perpendicular to
    # the live body axis so this diagnostic does not brake screw twist.
    body_axis_local = assembly_module.quat_apply(rel_quat, target_axis_local)
    axis_error_vector_local = torch.cross(body_axis_local, target_axis_local, dim=-1)
    tilt_velocity_local = relative_ang_vel_local - body_axis_local * (
        relative_ang_vel_local * body_axis_local
    ).sum(dim=-1, keepdim=True)
    angular_torque_local = angular_stiffness * axis_error_vector_local - angular_damping * tilt_velocity_local

    radial_force_w = assembly_module.quat_apply(support_quat, radial_force_local)
    angular_torque_w = assembly_module.quat_apply(support_quat, angular_torque_local)
    axis_error_rad = torch.acos(body_axis_local[:, 0].clamp(-1.0, 1.0))
    metrics = {
        "radial_error_m": torch.linalg.vector_norm(radial_error_local, dim=-1),
        "radial_speed_m_s": torch.linalg.vector_norm(radial_velocity_local, dim=-1),
        "radial_force_n": torch.linalg.vector_norm(radial_force_w, dim=-1),
        "axis_error_rad": axis_error_rad,
        "axis_tilt_speed_rad_s": torch.linalg.vector_norm(tilt_velocity_local, dim=-1),
        "axis_torque_nm": torch.linalg.vector_norm(angular_torque_w, dim=-1),
    }
    return radial_force_w, angular_torque_w, metrics


def _apply_wrench(
    raw,
    axis_w,
    torque_nm: float,
    axial_force_n: float,
    stabilization_force_w=None,
    stabilization_torque_w=None,
    gravity_compensation: bool = False,
):
    """Apply the external insertion wrench and optional diagnostic feedback."""
    import torch

    bulb = raw.scene["object"]
    body_count = max(1, int(getattr(bulb, "num_bodies", 1)))
    # Negative support +X is the measured insertion direction.  The caller
    # supplies a signed torque: forward is negative, reverse is positive.
    base_forces = -axis_w * axial_force_n
    base_torques = axis_w * torque_nm
    if gravity_compensation:
        gravity_w = torch.tensor((0.0, 0.0, -9.81), device=raw.device, dtype=axis_w.dtype)
        mass = getattr(getattr(bulb, "data", None), "root_mass", None)
        if mass is None:
            mass = torch.full((raw.num_envs,), 0.2, device=raw.device, dtype=axis_w.dtype)
        else:
            mass = torch.as_tensor(mass, device=raw.device, dtype=axis_w.dtype).reshape(raw.num_envs, -1).sum(dim=-1)
        base_forces = base_forces - mass[:, None] * gravity_w
    if stabilization_force_w is not None:
        base_forces = base_forces + stabilization_force_w
    if stabilization_torque_w is not None:
        base_torques = base_torques + stabilization_torque_w
    forces = base_forces[:, None, :].expand(raw.num_envs, body_count, 3).contiguous()
    torques = base_torques[:, None, :].expand(raw.num_envs, body_count, 3).contiguous()
    composer = getattr(bulb, "permanent_wrench_composer", None)
    if composer is not None and hasattr(composer, "set_forces_and_torques"):
        if hasattr(composer, "reset"):
            composer.reset()
        composer.set_forces_and_torques(forces=forces, torques=torques, is_global=True)
    elif hasattr(bulb, "set_external_force_and_torque"):
        bulb.set_external_force_and_torque(forces=forces, torques=torques, is_global=True)
    else:
        raise RuntimeError("RigidObject has no external wrench API")


def _clear_wrench(raw):
    """Remove the diagnostic wrench before closing the simulator."""
    import torch

    bulb = raw.scene["object"]
    body_count = max(1, int(getattr(bulb, "num_bodies", 1)))
    zeros = torch.zeros((raw.num_envs, body_count, 3), device=raw.device)
    composer = getattr(bulb, "permanent_wrench_composer", None)
    if composer is not None and hasattr(composer, "reset"):
        composer.reset()
    elif hasattr(bulb, "set_external_force_and_torque"):
        bulb.set_external_force_and_torque(forces=zeros, torques=zeros, is_global=True)


def _first_entry_pose(raw, assembly_module, phase_rad: float):
    """Write the one-time diagnostic placement and return the pose record.

    The packaged asset records a 180 degree external-thread phase at its
    extended pose. Starting at that phase tests the measured entry lead.
    """
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    support, bulb = raw.scene["support"], raw.scene["object"]
    offset = torch.zeros((raw.num_envs, 3), device=raw.device, dtype=support.data.root_pos_w.dtype)
    offset[:, 0] = assembly_module.ENTRY_ROOT_X_M
    identity = quat_from_euler_xyz(
        torch.full((raw.num_envs,), phase_rad, device=raw.device, dtype=support.data.root_quat_w.dtype),
        torch.zeros(raw.num_envs, device=raw.device, dtype=support.data.root_quat_w.dtype),
        torch.zeros(raw.num_envs, device=raw.device, dtype=support.data.root_quat_w.dtype),
    )
    bulb_pos, bulb_quat = assembly_module.combine_frame_transforms(
        support.data.root_pos_w, support.data.root_quat_w, offset, identity
    )
    bulb.write_root_pose_to_sim(torch.cat((bulb_pos, bulb_quat), dim=-1))
    bulb.write_root_velocity_to_sim(torch.zeros((raw.num_envs, 6), device=raw.device))
    # Refresh the cached object data once.  No pose or velocity writes occur
    # after this function returns; all later motion comes from PhysX.
    bulb.update(0.0)
    rel_pos, rel_quat = assembly_module.subtract_frame_transforms(
        support.data.root_pos_w, support.data.root_quat_w,
        bulb.data.root_pos_w, bulb.data.root_quat_w,
    )
    return {
        "relative_pos_m": _tolist(rel_pos),
        "relative_quat_wxyz": _tolist(rel_quat),
        "target_relative_x_m": float(assembly_module.ENTRY_ROOT_X_M),
        "phase_rad": phase_rad,
        "pose_writes": 1,
    }


def run_diagnosis(args, report, stream):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    import BrainCo_DexHand.tasks  # noqa: F401

    assembly_module = importlib.import_module(
        "BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.assembly"
    )
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.bulb_pure_collision = True
    report["config"] = {
        "task": args.task,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": args.device,
        "control_dt_s": float(cfg.sim.dt * cfg.decimation),
        "pure_collision": True,
        "forward_steps": args.forward_steps,
        "reverse_steps": args.reverse_steps,
        "torque_nm": args.torque_nm,
        "axial_force_n": args.axial_force_n,
        "radial_stiffness_n_m": args.radial_stiffness,
        "radial_damping_n_s_m": args.radial_damping,
        "angular_stiffness_nm_rad": args.angular_stiffness,
        "angular_damping_nm_s_rad": args.angular_damping,
        "entry_phase_rad": args.entry_phase_rad,
        "gravity_compensation": args.gravity_compensation,
        "stabilization_enabled": any(
            value > 0.0
            for value in (
                args.radial_stiffness,
                args.radial_damping,
                args.angular_stiffness,
                args.angular_damping,
            )
        ),
    }
    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    try:
        reset_obs = _reset(env, args.seed)
        report["reset_observation_finite"] = _finite(reset_obs[0] if isinstance(reset_obs, tuple) else reset_obs)
        report["diagnostic_only"] = {
            "prealigned_first_entry": True,
            "acceptance_eligible": False,
            "reason": "grasp, lift, and alignment are bypassed for this contact-path diagnosis",
        }
        report["first_entry_pose"] = _first_entry_pose(raw, assembly_module, args.entry_phase_rad)

        support = raw.scene["support"]
        axis_local = torch.zeros((raw.num_envs, 3), device=raw.device)
        axis_local[:, 0] = 1.0
        axis_w = assembly_module.quat_apply(support.data.root_quat_w, axis_local)
        action_dim = int(getattr(raw.action_manager.action, "shape", [raw.num_envs, 28])[-1])
        action = torch.zeros((raw.num_envs, action_dim), device=raw.device)
        previous_raw = None
        angle_unwrapped = torch.zeros(raw.num_envs, device=raw.device)
        finite = bool(report["reset_observation_finite"])
        rows = []
        stage_counts = {"forward": 0, "reverse": 0}
        force_sensor_available = False
        start_metrics = _relative_metrics(raw, assembly_module, angle_unwrapped, previous_raw)
        previous_raw = start_metrics[2].detach().clone()
        angle_unwrapped = start_metrics[3].detach().clone()
        for stage, steps, torque_sign in (
            ("forward", args.forward_steps, -1.0),
            ("reverse", args.reverse_steps, 1.0),
        ):
            for stage_step in range(steps):
                stabilization_force_w, stabilization_torque_w, stabilization_metrics = _stabilization_wrench(
                    raw,
                    assembly_module,
                    args.radial_stiffness,
                    args.radial_damping,
                    args.angular_stiffness,
                    args.angular_damping,
                )
                _apply_wrench(
                    raw,
                    axis_w,
                    torque_sign * args.torque_nm,
                    args.axial_force_n,
                    stabilization_force_w=stabilization_force_w,
                    stabilization_torque_w=stabilization_torque_w,
                    gravity_compensation=args.gravity_compensation,
                )
                obs, reward, done, extras = _step(env, action)
                metrics = _relative_metrics(raw, assembly_module, angle_unwrapped, previous_raw)
                rel_pos, rel_quat, raw_angle, angle_unwrapped, advance, expected, residual = metrics
                previous_raw = raw_angle.detach().clone()
                contact_force, available = _contact_force(raw, "bulb_support_contact")
                force_sensor_available = force_sensor_available or available
                values = (
                    obs,
                    reward,
                    done,
                    rel_pos,
                    rel_quat,
                    angle_unwrapped,
                    advance,
                    residual,
                    contact_force,
                    *stabilization_metrics.values(),
                )
                step_finite = _finite(*values)
                finite = finite and step_finite
                state = getattr(raw, "_bulb_assembly_state", None)
                row = {
                    "stage": stage,
                    "stage_step": stage_step + 1,
                    "global_step": len(rows) + 1,
                    "torque_sign": torque_sign,
                    "relative_pos_m": _tolist(rel_pos),
                    "relative_quat_wxyz": _tolist(rel_quat),
                    "relative_angle_rad": _tolist(angle_unwrapped),
                    "axial_advance_m": _tolist(advance),
                    "expected_pitch_advance_m": _tolist(expected),
                    "helical_residual_m": _tolist(residual),
                    "support_contact_force_n": _tolist(contact_force),
                    "stabilization_radial_error_m": _tolist(stabilization_metrics["radial_error_m"]),
                    "stabilization_radial_speed_m_s": _tolist(stabilization_metrics["radial_speed_m_s"]),
                    "stabilization_radial_force_n": _tolist(stabilization_metrics["radial_force_n"]),
                    "stabilization_axis_error_rad": _tolist(stabilization_metrics["axis_error_rad"]),
                    "stabilization_axis_tilt_speed_rad_s": _tolist(
                        stabilization_metrics["axis_tilt_speed_rad_s"]
                    ),
                    "stabilization_axis_torque_nm": _tolist(stabilization_metrics["axis_torque_nm"]),
                    "done": _tolist(done),
                    "state_phase": _tolist(getattr(state, "phase", None)),
                    "state_engaged": _tolist(getattr(state, "engaged", None)),
                    "state_installed": _tolist(getattr(state, "installed", None)),
                    "state_success": _tolist(getattr(state, "success", None)),
                    "finite": step_finite,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
                stage_counts[stage] += 1
        _clear_wrench(raw)

        def _tensor_rows(name):
            values = [row[name] for row in rows]
            return values[-1] if values else None

        forward_rows = [row for row in rows if row["stage"] == "forward"]
        reverse_rows = [row for row in rows if row["stage"] == "reverse"]
        forward_last = forward_rows[-1] if forward_rows else None
        reverse_first = reverse_rows[0] if reverse_rows else None
        reverse_last = reverse_rows[-1] if reverse_rows else None
        reverse_angle_delta = None
        reverse_advance_delta = None
        reverse_retreated = False
        if forward_last is not None and reverse_last is not None:
            reverse_angle_delta = [
                float(after) - float(before)
                for before, after in zip(
                    forward_last["relative_angle_rad"], reverse_last["relative_angle_rad"]
                )
            ]
            reverse_advance_delta = [
                float(after) - float(before)
                for before, after in zip(
                    forward_last["axial_advance_m"], reverse_last["axial_advance_m"]
                )
            ]
            # Reverse torque should increase the signed angle and reduce
            # insertion advance.  This is deliberately diagnostic output;
            # a failed retreat is evidence against stable threading, not a
            # reason to mark the pre-aligned trajectory successful.
            reverse_retreated = any(
                angle_delta > 0.05 and advance_delta < 0.0
                for angle_delta, advance_delta in zip(reverse_angle_delta, reverse_advance_delta)
            )

        report["trajectory"] = {
            "rows": len(rows),
            "stage_counts": stage_counts,
            "force_sensor_available": force_sensor_available,
            "finite": finite,
            "initial_relative_x_m": _tolist(start_metrics[0][:, 0]),
            "final_relative_x_m": _tensor_rows("relative_pos_m"),
            "final_angle_rad": _tensor_rows("relative_angle_rad"),
            "final_advance_m": _tensor_rows("axial_advance_m"),
            "final_helical_residual_m": _tensor_rows("helical_residual_m"),
            "max_contact_force_n": max(
                (max(float(v) for v in row["support_contact_force_n"]) for row in rows),
                default=0.0,
            ),
            "max_abs_helical_residual_m": max(
                (max(abs(float(v)) for v in row["helical_residual_m"]) for row in rows),
                default=0.0,
            ),
            "max_stabilization_radial_force_n": max(
                (max(float(v) for v in row["stabilization_radial_force_n"]) for row in rows),
                default=0.0,
            ),
            "max_stabilization_axis_torque_nm": max(
                (max(float(v) for v in row["stabilization_axis_torque_nm"]) for row in rows),
                default=0.0,
            ),
            "max_stabilization_radial_error_m": max(
                (max(float(v) for v in row["stabilization_radial_error_m"]) for row in rows),
                default=0.0,
            ),
            "max_stabilization_axis_error_rad": max(
                (max(float(v) for v in row["stabilization_axis_error_rad"]) for row in rows),
                default=0.0,
            ),
            "forward_advance_m": (
                _tensor_rows("axial_advance_m") if not rows else next(
                    (row["axial_advance_m"] for row in reversed(rows) if row["stage"] == "forward"), None
                )
            ),
            "reverse_exit_recorded": any(row["stage"] == "reverse" for row in rows),
            "reverse_first_angle_rad": None if reverse_first is None else reverse_first["relative_angle_rad"],
            "reverse_last_angle_rad": None if reverse_last is None else reverse_last["relative_angle_rad"],
            "reverse_angle_delta_rad": reverse_angle_delta,
            "reverse_advance_delta_m": reverse_advance_delta,
            "reverse_retreated": reverse_retreated,
        }
        report["checks"] = {
            "reset_observation_finite": bool(report["reset_observation_finite"]),
            "first_entry_pose_written_once": report["first_entry_pose"]["pose_writes"] == 1,
            "wrench_trajectory_complete": len(rows) == args.forward_steps + args.reverse_steps,
            "finite_trajectory": finite,
            "reverse_torque_exit_recorded": report["trajectory"]["reverse_exit_recorded"],
            "support_contact_sensor_available": force_sensor_available,
        }
        # A diagnostic passes when it produced finite physical measurements;
        # it deliberately does not require task success or a particular screw
        # advance because pre-alignment bypasses the learned manipulation.
        report["passed"] = all(report["checks"].values())
    finally:
        try:
            _clear_wrench(raw)
        except Exception:
            pass
        env.close()


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--forward-steps", type=int, default=240)
    parser.add_argument("--reverse-steps", type=int, default=120)
    parser.add_argument("--torque-nm", type=float, default=0.02)
    parser.add_argument("--axial-force-n", type=float, default=0.5)
    parser.add_argument(
        "--entry-phase-rad",
        type=float,
        default=math.pi,
        help="Measured external-thread phase at first entry; default is the asset's recorded pi phase.",
    )
    parser.add_argument(
        "--gravity-compensation",
        action="store_true",
        help="Apply a live equal-and-opposite bulb gravity wrench for diagnostic hand support.",
    )
    parser.add_argument(
        "--radial-stiffness",
        type=float,
        default=0.0,
        help="Optional diagnostic radial spring gain (N/m); zero preserves the baseline.",
    )
    parser.add_argument(
        "--radial-damping",
        type=float,
        default=0.0,
        help="Optional diagnostic radial damping gain (N s/m); zero preserves the baseline.",
    )
    parser.add_argument(
        "--angular-stiffness",
        type=float,
        default=0.0,
        help="Optional diagnostic axis-alignment torque gain (N m/rad); zero preserves the baseline.",
    )
    parser.add_argument(
        "--angular-damping",
        type=float,
        default=0.0,
        help="Optional diagnostic tilt damping gain (N m s/rad); zero preserves the baseline.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.forward_steps < 1 or args.reverse_steps < 1:
        parser.error("--num-envs, --forward-steps, and --reverse-steps must be positive")
    if args.torque_nm <= 0.0 or args.axial_force_n < 0.0:
        parser.error("--torque-nm must be positive and --axial-force-n non-negative")
    if any(
        value < 0.0
        for value in (args.radial_stiffness, args.radial_damping, args.angular_stiffness, args.angular_damping)
    ):
        parser.error("stabilization gains must be non-negative")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "bulb_thread_physics_diagnosis",
        "protocol_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "task": args.task,
        "passed": False,
    }
    report_path = args.out / "report.json"
    trajectory_path = args.out / "trajectory.jsonl"
    app = AppLauncher(args).app
    try:
        with trajectory_path.open("w", encoding="utf-8") as stream:
            run_diagnosis(args, report, stream)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"BULB THREAD DIAGNOSIS {'PASS' if report['passed'] else 'FAIL'}: {report_path}", flush=True)
        try:
            app.app.post_quit(0 if report["passed"] else 1)
        except Exception:
            pass
        app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
