#!/usr/bin/env python3
"""Measure the real Custom-bulb arm position response without PPO updates.

This probe keeps the existing guided Custom-bulb physics, including gravity,
collisions and termination terms.  It first replays the supplied policy for a
full 20 seconds, then resets the same environment seed and applies a constant
normalised command to one of the seven arm joints while retaining the policy's
hand commands.  Terminal state is captured by a termination-manager hook before
Isaac Lab resets the slot.  Torque values are explicitly labelled PD estimates:
implicit PhysX drives do not expose measured motor torque through Isaac Lab.

The output is diagnostic evidence only.  It is not an acceptance evaluation,
does not update a policy, and never writes a pose or attachment state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


TASK = "BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0"
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "BrainCo_DexHand"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_hash(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _as_bool(value, device):
    import torch

    return torch.as_tensor(value, device=device).detach().bool().flatten()


def _tolist(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def _parameter_hash(raw, action_term) -> str:
    digest = hashlib.sha256()
    robot = raw.scene["robot"]
    for name in ("joint_stiffness", "joint_damping", "joint_friction_coeff"):
        value = getattr(robot.data, name, None)
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
        else:
            value = value[:, action_term.joint_ids].detach().cpu().contiguous()
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _world_target(raw):
    from isaaclab.utils.math import combine_frame_transforms

    robot = raw.scene["robot"]
    command = raw.command_manager.get_command("object_pose")
    target, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    return target


def _contact_snapshot(raw):
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom import env_cfg as bulb

    try:
        contacts = bulb.rotate_bulb_contact_scores(raw)
        forces = bulb.rotate_bulb_finger_forces(raw)
    except (AttributeError, KeyError, RuntimeError, TypeError):
        contacts = forces = None
    return contacts, forces


class _PreResetTrace:
    """Record step state before ManagerBasedRLEnv resets terminated slots."""

    def __init__(self, raw, action_term, path: Path, scenario: str):
        self.raw = raw
        self.action_term = action_term
        self.path = path
        self.scenario = scenario
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8")
        self.step = 0
        self.current_actions = None
        self._original_compute = None
        self.terminal_count = 0
        self.reference_q = None
        self.last_q = None
        self.last_qd = None
        self.max_abs_dq = None

    def set_actions(self, actions):
        self.current_actions = actions.detach().clone()

    def _terms(self):
        manager = self.raw.termination_manager
        result = {
            name: _as_bool(manager.get_term(name), self.raw.device)
            for name in manager.active_terms
        }
        try:
            result["time_out"] = _as_bool(manager.time_outs, self.raw.device)
        except (AttributeError, RuntimeError, TypeError):
            pass
        return result

    def _snapshot(self, terms):
        import torch
        from isaaclab.utils.math import subtract_frame_transforms

        raw, action = self.raw, self.action_term
        robot, obj, support = (raw.scene[name] for name in ("robot", "object", "support"))
        joint_ids = action.joint_ids
        q = robot.data.joint_pos[:, joint_ids]
        qd = robot.data.joint_vel[:, joint_ids]
        target = getattr(robot.data, "joint_pos_target", None)
        if target is not None:
            target = target[:, joint_ids]
        else:
            target = action.position_targets
        vel_target = getattr(robot.data, "joint_vel_target", None)
        if vel_target is not None:
            vel_target = vel_target[:, joint_ids]
        else:
            vel_target = torch.zeros_like(q)
        effort_target = getattr(robot.data, "joint_effort_target", None)
        if effort_target is not None:
            effort_target = effort_target[:, joint_ids]
        else:
            effort_target = torch.zeros_like(q)
        stiffness = getattr(robot.data, "joint_stiffness", None)
        damping = getattr(robot.data, "joint_damping", None)
        friction = getattr(robot.data, "joint_friction_coeff", None)
        stiffness = None if stiffness is None else stiffness[:, joint_ids]
        damping = None if damping is None else damping[:, joint_ids]
        friction = None if friction is None else friction[:, joint_ids]
        pd = None
        if stiffness is not None and damping is not None:
            pd = stiffness * (target - q) + damping * (vel_target - qd) + effort_target
        # Isaac Lab's data names exist for explicit actuators.  For this task
        # the actuators are implicit, so these are normally None; retain them
        # when available but do not call them measured PhysX motor torque.
        computed = getattr(robot.data, "computed_torque", None)
        applied = getattr(robot.data, "applied_torque", None)
        if computed is not None:
            computed = computed[:, joint_ids]
        if applied is not None:
            applied = applied[:, joint_ids]
        actuator_applied = None
        # ImplicitActuator exposes the same PD estimate as ``applied_effort``
        # after clipping.  It is still an estimate (PhysX does not expose the
        # internal drive torque), but retaining it is useful for checking the
        # effort-limit branch and actuator-group mapping.
        try:
            global_ids = list(range(robot.data.joint_pos.shape[1])) if isinstance(joint_ids, slice) else [
                int(index) for index in torch.as_tensor(joint_ids).flatten().tolist()
            ]
            positions = {index: offset for offset, index in enumerate(global_ids)}
            actuator_applied = torch.zeros_like(q)
            assigned = torch.zeros(len(global_ids), dtype=torch.bool, device=raw.device)
            for actuator in robot.actuators.values():
                effort = getattr(actuator, "applied_effort", None)
                indices = getattr(actuator, "joint_indices", None)
                if effort is None or indices is None:
                    continue
                for local, global_index in enumerate(torch.as_tensor(indices).flatten().tolist()):
                    if int(global_index) in positions:
                        actuator_applied[:, positions[int(global_index)]] = effort[:, local]
                        assigned[positions[int(global_index)]] = True
            if not bool(assigned.all().item()):
                actuator_applied = None
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            actuator_applied = None
        rel_pos, rel_quat = subtract_frame_transforms(
            support.data.root_pos_w, support.data.root_quat_w,
            obj.data.root_pos_w, obj.data.root_quat_w,
        )
        target_w = _world_target(raw)
        contacts, forces = _contact_snapshot(raw)
        released = getattr(raw, "_bulb_released", None)
        row = {
            "scenario": self.scenario,
            "step": self.step,
            "seconds": float((self.step + 1) * raw.step_dt),
            "actions": _tolist(self.current_actions),
            "joint_pos_rad": _tolist(q),
            "joint_vel_rad_s": _tolist(qd),
            "joint_pos_target_rad": _tolist(target),
            "joint_vel_target_rad_s": _tolist(vel_target),
            "joint_effort_target_nm": _tolist(effort_target),
            "joint_stiffness": _tolist(stiffness),
            "joint_damping": _tolist(damping),
            "joint_static_friction_effort": _tolist(friction),
            "pd_effort_estimate_nm": _tolist(pd),
            "computed_torque_api_nm": _tolist(computed),
            "applied_torque_api_nm": _tolist(applied),
            "actuator_applied_effort_estimate_nm": _tolist(actuator_applied),
            "bulb_root_pos_w": _tolist(obj.data.root_pos_w),
            "bulb_root_lin_vel_w": _tolist(obj.data.root_lin_vel_w),
            "bulb_root_ang_vel_w": _tolist(obj.data.root_ang_vel_w),
            "bulb_relative_pos": _tolist(rel_pos),
            "bulb_relative_quat": _tolist(rel_quat),
            "target_pos_w": _tolist(target_w),
            "contact_scores": _tolist(contacts),
            "finger_forces_n": _tolist(forces),
            "released": _tolist(released),
            "termination": {name: _tolist(value) for name, value in terms.items()},
        }
        return row

    def on_compute(self, result):
        terms = self._terms()
        row = self._snapshot(terms)
        import torch
        q = torch.as_tensor(row["joint_pos_rad"], device=self.raw.device)
        qd = torch.as_tensor(row["joint_vel_rad_s"], device=self.raw.device)
        if self.reference_q is None:
            self.reference_q = q.clone()
        self.last_q = q.clone()
        self.last_qd = qd.clone()
        delta = (q - self.reference_q).abs()
        self.max_abs_dq = delta if self.max_abs_dq is None else torch.maximum(self.max_abs_dq, delta)
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        done = _as_bool(result, self.raw.device)
        self.terminal_count += int(done.sum().item())
        self.step += 1

    def install(self):
        self._original_compute = self.raw.termination_manager.compute
        original = self._original_compute

        def wrapped_compute(*args, **kwargs):
            result = original(*args, **kwargs)
            self.on_compute(result)
            return result

        self.raw.termination_manager.compute = wrapped_compute

    def close(self):
        if self._original_compute is not None:
            self.raw.termination_manager.compute = self._original_compute
        self.stream.close()


def _reset(raw, seed):
    import torch

    try:
        with torch.inference_mode():
            return raw.reset(seed=seed)
    except TypeError:
        with torch.inference_mode():
            return raw.reset()


def _run_scenario(raw, wrapped, policy, action_term, output_dir, name, seed, steps,
                  arm_joint=None, amplitude=0.0):
    import torch

    _reset(raw, seed)
    obs = wrapped.get_observations()
    trace = _PreResetTrace(raw, action_term, output_dir / name / "trace.jsonl", name)
    trace.install()
    initial_q = raw.scene["robot"].data.joint_pos[:, action_term.joint_ids].detach().clone()
    trace.reference_q = initial_q.clone()
    try:
        with torch.inference_mode():
            for _ in range(steps):
                actions = policy(obs)
                if arm_joint is not None:
                    actions = actions.clone()
                    actions[:, arm_joint] = amplitude
                trace.set_actions(actions)
                obs, _reward, _done, _extras = wrapped.step(actions)
    finally:
        trace.close()
    final_q = trace.last_q
    final_qd = trace.last_qd
    return {
        "scenario": name,
        "steps": steps,
        "arm_joint_index": arm_joint,
        "amplitude": amplitude,
        "startup_parameter_hash": _parameter_hash(raw, action_term),
        "initial_q_rad": _tolist(initial_q),
        "final_q_pre_reset_rad": _tolist(final_q),
        "final_qd_pre_reset_rad_s": _tolist(final_qd),
        "delta_q_pre_reset_rad": _tolist(final_q - initial_q) if final_q is not None else None,
        "max_abs_delta_q_pre_reset_rad": _tolist(trace.max_abs_dq),
        "terminal_events_captured": trace.terminal_count,
        "trace": str(output_dir / name / "trace.jsonl"),
    }


def run(args, report):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner
    import yaml

    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(ROOT / "scripts" / "rsl_rl"))
    import BrainCo_DexHand  # noqa: F401
    from BrainCo_DexHand.tactile_representation.policy import prepare_policy_run
    from bulb_training import configure_bulb_checkpoint, prepare_bulb_policy_run
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.agents.rsl_rl_ppo_cfg import (
        DexsuiteRevo3RotateBulbPPORunnerCfg,
    )

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as Wrapper
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    saved_env_path = args.checkpoint.parent / "params" / "env.yaml"
    if not saved_env_path.is_file():
        raise ValueError(f"checkpoint is missing saved environment parameters: {saved_env_path}")
    saved_env = yaml.unsafe_load(saved_env_path.read_text())
    if bool(saved_env.get("bulb_pure_collision")):
        raise ValueError("control response probe requires a guided checkpoint (bulb_pure_collision=false)")
    saved_ranges = saved_env.get("commands", {}).get("object_pose", {}).get("ranges", {})
    for name in ("pos_x", "pos_y", "pos_z"):
        values = saved_ranges.get(name)
        if values is not None:
            setattr(cfg.commands.object_pose.ranges, name, tuple(values))
    cfg.seed = args.seed
    cfg.bulb_pure_collision = False
    cfg.initial_screw_turns_range = (3.0, 3.0)
    cfg.tactile_policy_enabled = False
    cfg.tactile_reconstruction_diagnostics = False
    cfg.log_dir = str(args.out)
    agent_cfg = DexsuiteRevo3RotateBulbPPORunnerCfg()
    agent_cfg.device = args.device
    configure_bulb_checkpoint(agent_cfg, str(args.checkpoint), training=False)
    env = gym.make(TASK, cfg=cfg)
    runtime_dir = args.out / "runtime"
    prepare_policy_run(env, runtime_dir, resume_path=str(args.checkpoint))
    prepare_bulb_policy_run(env, runtime_dir, resume_path=str(args.checkpoint))
    wrapped = Wrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
    # No optimizer state is needed for a frozen response probe.
    runner.load(str(args.checkpoint), load_optimizer=False, map_location=args.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    raw = env.unwrapped
    policy_state = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    policy_hash_before = _state_hash(policy_state)
    action_term = raw.action_manager.get_term("action")
    steps_probe = max(1, math.ceil(args.probe_seconds / float(raw.step_dt)))
    steps_natural = max(1, math.ceil(args.natural_seconds / float(raw.step_dt)))
    report["runtime"] = {
        "task": TASK,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": args.device,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "physics_mode": "guided",
        "bulb_pure_collision": False,
        "checkpoint_env_ranges": {
            name: list(getattr(cfg.commands.object_pose.ranges, name))
            for name in ("pos_x", "pos_y", "pos_z")
        },
        "step_dt": float(raw.step_dt),
        "arm_action_indices": list(action_term.arm_action_indices),
        "hand_action_indices": list(action_term.hand_action_indices),
        "joint_names": list(action_term.joint_names),
        "delta_limits_rad": action_term.action_contract()["delta_limits_rad"],
        "acceptance_eligible": False,
    }
    report["scenarios"] = []
    report["scenarios"].append(_run_scenario(
        raw, wrapped, policy, action_term, args.out, "natural_20s", args.seed,
        steps_natural,
    ))
    for joint in action_term.arm_action_indices:
        for amplitude in (0.5, -0.5, 1.0, -1.0):
            label = f"arm_{action_term.joint_names[joint]}_{amplitude:+.1f}"
            report["scenarios"].append(_run_scenario(
                raw, wrapped, policy, action_term, args.out, label, args.seed,
                steps_probe, arm_joint=joint, amplitude=amplitude,
            ))
    report["policy_state_hash_before"] = policy_hash_before
    report["policy_state_hash_after"] = _state_hash(policy_state)
    report["policy_weights_unchanged"] = report["policy_state_hash_before"] == report["policy_state_hash_after"]
    parameter_hashes = {item["startup_parameter_hash"] for item in report["scenarios"]}
    report["startup_parameters_unchanged_across_resets"] = len(parameter_hashes) == 1
    report["startup_parameter_hashes"] = sorted(parameter_hashes)
    report["passed"] = True
    env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-seconds", type=float, default=0.75)
    parser.add_argument("--natural-seconds", type=float, default=20.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs not in (4, 8):
        parser.error("--num-envs must be 4 or 8")
    if not (0.5 <= args.probe_seconds <= 1.0):
        parser.error("--probe-seconds must be between 0.5 and 1.0")
    if args.natural_seconds < 20.0:
        parser.error("--natural-seconds must be at least 20 seconds")
    args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "rotate_bulb_custom_control_response_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False,
    }
    app = AppLauncher(args).app
    try:
        run(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        try:
            app.close(wait_for_replicator=False, skip_cleanup=True)
        except Exception:
            pass
    print(f"BULB CONTROL PROBE {'PASS' if report['passed'] else 'FAIL'}: {args.out / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
