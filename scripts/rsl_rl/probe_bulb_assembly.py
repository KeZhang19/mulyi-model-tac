#!/usr/bin/env python3
"""Small, read-only physics probe for the complete tabletop bulb task.

This is deliberately separate from training and evaluation.  It creates the
registered ``BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0`` task,
records the reset state and a zero-action rollout, then checks that a partial
reset clears only the selected environments (including assembly timers).
Every observation is written as JSON so a failed Kit startup still leaves a
useful report.  The probe does not load a checkpoint or alter task files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback


TASK = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0"
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "BrainCo_DexHand"
LAMP_ASSET_REL = Path("assets/bulb/hiveboard_lamp/hiveboard_lamp_threaded.usd")


def _tolist(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _finite(value):
    """Return false for any non-finite tensor/value without requiring numpy."""
    import torch

    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (float, int)):
        return value == value and abs(value) != float("inf")
    return True


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _module_record(module):
    path = Path(module.__file__).resolve()
    return {"module": module.__name__, "path": str(path), "sha256": _sha256(path)}


def _contact_for_sensor(raw, name: str):
    import torch

    sensor = raw.scene.sensors.get(name)
    if sensor is None or getattr(sensor.data, "force_matrix_w", None) is None:
        return torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device), None
    force = torch.nan_to_num(sensor.data.force_matrix_w)
    # ContactSensor layouts differ between Isaac Lab releases.  Flatten all
    # filter/body dimensions, retaining one force magnitude per environment.
    force = force.reshape(raw.num_envs, -1, 3)
    magnitude = torch.linalg.vector_norm(force, dim=-1).amax(dim=-1)
    return magnitude > 0.2, magnitude


def _asset_geometry(raw, cfg, report):
    """Read thread meshes, mouth, masses and table dimensions from live USD."""
    import torch

    asset_cfg = getattr(getattr(cfg.scene, "lamp", None), "spawn", None)
    configured = getattr(asset_cfg, "usd_path", None)
    path = Path(str(configured)).expanduser() if configured else ROOT / LAMP_ASSET_REL
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    result = {"configured_usd_path": str(configured), "usd_path": str(path), "exists": path.exists()}
    metadata = path.with_name("threaded_metadata.json")
    result["metadata_path"] = str(metadata)
    result["metadata_sha256"] = _sha256(metadata)
    if metadata.is_file():
        try:
            result["metadata"] = json.loads(metadata.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            result["metadata_error"] = str(exc)

    # The stage contains the cloned prim, so these values verify the actual
    # imported asset rather than relying solely on the checked-in metadata.
    try:
        from pxr import Usd, UsdPhysics

        stage = raw.sim.stage
        prefix = "/World/envs/env_0/Lamp"
        meshes = {
            "male_external_thread": prefix + "/Bulb/ExternalThreadCollider",
            "female_internal_thread": prefix + "/Support/InternalThreadCollider",
        }
        live = {}
        for name, prim_path in meshes.items():
            prim = stage.GetPrimAtPath(prim_path)
            points = prim.GetAttribute("points").Get() if prim.IsValid() else None
            if points:
                values = [[float(point[i]) for i in range(3)] for point in points]
                live[name] = {
                    "prim_path": prim_path,
                    "active": bool(prim.IsActive()),
                    "vertex_count": len(values),
                    "bounds_m": [
                        [min(point[i] for point in values) for i in range(3)],
                        [max(point[i] for point in values) for i in range(3)],
                    ],
                }
            else:
                live[name] = {"prim_path": prim_path, "active": bool(prim.IsActive()) if prim.IsValid() else False}
        if live.get("female_internal_thread", {}).get("bounds_m"):
            live["female_internal_thread"]["mouth_x_m"] = live["female_internal_thread"]["bounds_m"][1][0]
        result["live_meshes"] = live

        masses = []
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(prefix + "/Bulb"):
                continue
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass = prim.GetAttribute("physics:mass").Get()
                if mass is not None:
                    masses.append({"prim_path": str(prim.GetPath()), "mass_kg": float(mass)})
        result["live_masses"] = masses
    except Exception as exc:  # Kit/USD versions vary; retain the rest of report.
        result["live_usd_error"] = f"{type(exc).__name__}: {exc}"

    try:
        object_asset = raw.scene["object"]
        result["sim_default_mass_kg"] = _tolist(getattr(object_asset.data, "default_mass", None))
        result["sim_default_inertia_kg_m2"] = _tolist(getattr(object_asset.data, "default_inertia", None))
    except Exception as exc:
        result["sim_mass_error"] = f"{type(exc).__name__}: {exc}"

    table_cfg = cfg.scene.table.spawn
    size = tuple(float(x) for x in table_cfg.size)
    table_pos = raw.scene["table"].data.root_pos_w
    result["table_size_m"] = list(size)
    result["table_half_height_m"] = size[2] / 2.0
    result["table_root_z_m"] = _tolist(table_pos[:, 2])
    result["table_top_z_m"] = _tolist(table_pos[:, 2] + size[2] / 2.0)
    report["asset_geometry"] = result


def _lamp_joints(raw):
    """List active Lamp joints and enabled attachment constraints."""
    result = []
    try:
        from pxr import UsdPhysics

        for prim in raw.sim.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/World/envs/env_0/Lamp"):
                continue
            if not prim.IsA(UsdPhysics.Joint):
                continue
            attr = prim.GetAttribute("physics:jointEnabled")
            result.append({
                "path": path,
                "active": bool(prim.IsActive()),
                "joint_enabled": attr.Get() if attr.IsValid() else None,
                "type": str(prim.GetTypeName()) if prim.GetTypeName() else "",
            })
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "joints": []}
    return {"joints": result, "active_enabled_attachment": any(
        item["active"] and item["joint_enabled"] is True and "attachment" in item["path"]
        for item in result
    )}


def _state_record(raw):
    import torch

    state = getattr(raw, "_bulb_assembly_state", None)
    if state is None:
        return {"available": False}
    fields = {}
    for name in ("phase", "grasped", "lifted", "aligned", "engaged", "installed", "success",
                 "relative_angle", "best_advance", "helical_error", "last_contact", "table_top"):
        value = getattr(state, name, None)
        if value is not None:
            fields[name] = _tolist(value)
    fields["available"] = True
    for name in ("_assembly_hold_time", "_assembly_no_contact_time"):
        fields[name] = _tolist(getattr(raw, name, None))
    return fields


def _reset(env, seed: int | None = None):
    try:
        return env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        return env.reset()


def _step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, extras = result
        return obs, reward, terminated | truncated, extras
    obs, reward, done, extras = result
    return obs, reward, done, extras


def run_probe(args, report, step_stream):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    sys.path.insert(0, str(SOURCE))
    import BrainCo_DexHand.tasks  # noqa: F401

    env_module = importlib.import_module(
        "BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.environment"
    )
    cfg_module = importlib.import_module(
        "BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.assembly_env_cfg"
    )
    assembly_module = importlib.import_module(
        "BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.assembly"
    )
    report["imports"] = [_module_record(module) for module in (env_module, cfg_module, assembly_module)]
    spec = gym.spec(args.task)
    report["task_spec"] = {"id": args.task, "entry_point": spec.entry_point, "kwargs": spec.kwargs}

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.pure_collision:
        cfg.bulb_pure_collision = True
    report["config"] = {
        "task": args.task,
        "num_envs": args.num_envs,
        "device": args.device,
        "seed": args.seed,
        "episode_length_s": float(cfg.episode_length_s),
        "control_dt_s": float(cfg.sim.dt * cfg.decimation),
        "bulb_pure_collision": bool(getattr(cfg, "bulb_pure_collision", False)),
        "source_modules": report["imports"],
    }

    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    try:
        initial = _reset(env, args.seed)
        _asset_geometry(raw, cfg, report)
        report["lamp_joints"] = _lamp_joints(raw)
        report["initial_state"] = _state_record(raw)

        support = raw.scene["support"]
        bulb = raw.scene["object"]
        table = raw.scene["table"]
        relative_pos, relative_quat = assembly_module.subtract_frame_transforms(
            support.data.root_pos_w, support.data.root_quat_w,
            bulb.data.root_pos_w, bulb.data.root_quat_w,
        )
        report["initial_geometry"] = {
            "support_root_pose_w": _tolist(support.data.root_link_pose_w),
            "bulb_root_pose_w": _tolist(bulb.data.root_link_pose_w),
            "bulb_support_relative_pos_m": _tolist(relative_pos),
            "bulb_support_relative_quat_wxyz": _tolist(relative_quat),
            "table_top_z_m": _tolist(table.data.root_pos_w[:, 2] + cfg.scene.table.spawn.size[2] / 2),
            "bulb_above_table_m": _tolist(bulb.data.root_pos_w[:, 2] - (table.data.root_pos_w[:, 2] + cfg.scene.table.spawn.size[2] / 2)),
        }
        table_contact, table_force = _contact_for_sensor(raw, "bulb_table_contact")
        support_contact, support_force = _contact_for_sensor(raw, "bulb_support_contact")
        report["initial_contacts"] = {
            "table_contact": _tolist(table_contact), "table_force_n": _tolist(table_force),
            "support_contact": _tolist(support_contact), "support_force_n": _tolist(support_force),
        }

        # Initial state must be a desktop phase, detached from the socket, and
        # must not be able to claim success before one physics step.
        state = getattr(raw, "_bulb_assembly_state", None)
        initial_checks = {
            "state_available": state is not None,
            "phase_desktop": bool(state is not None and torch.all(state.phase == assembly_module.DESKTOP)),
            "not_grasped": bool(state is not None and not bool(state.grasped.any())),
            "not_lifted": bool(state is not None and not bool(state.lifted.any())),
            "not_engaged": bool(state is not None and not bool(state.engaged.any())),
            "not_installed": bool(state is not None and not bool(state.installed.any())),
            "not_success": bool(state is not None and not bool(state.success.any())),
            "no_enabled_attachment": not report["lamp_joints"].get("active_enabled_attachment", False),
        }
        report["initial_checks"] = initial_checks

        action_dim = int(getattr(raw.action_manager.action, "shape", [args.num_envs, 28])[-1])
        zero = torch.zeros((raw.num_envs, action_dim), device=raw.device)
        report["zero_action_dim"] = action_dim
        start_pos = bulb.data.root_pos_w.clone()
        positions = [start_pos.clone()]
        zero_success = []
        zero_done = []
        finite = bool(_finite(initial[0] if isinstance(initial, tuple) else initial))
        for step in range(args.steps):
            obs, reward, done, extras = _step(env, zero)
            finite = finite and _finite(obs) and _finite(reward)
            zero_success.append(_tolist(getattr(getattr(raw, "_bulb_assembly_state", None), "success", None)))
            zero_done.append(_tolist(done))
            positions.append(bulb.data.root_pos_w.clone())
            table_contact, table_force = _contact_for_sensor(raw, "bulb_table_contact")
            support_contact, support_force = _contact_for_sensor(raw, "bulb_support_contact")
            record = {
                "step": step + 1,
                "reward": _tolist(reward),
                "done": _tolist(done),
                "finite": bool(_finite(obs) and _finite(reward)),
                "state": _state_record(raw),
                "bulb_root_pos_w": _tolist(bulb.data.root_pos_w),
                "bulb_root_lin_vel_w": _tolist(bulb.data.root_lin_vel_w),
                "table_contact": _tolist(table_contact),
                "table_force_n": _tolist(table_force),
                "support_contact": _tolist(support_contact),
                "support_force_n": _tolist(support_force),
            }
            step_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            step_stream.flush()
        position_tensor = torch.stack(positions)
        displacement = torch.linalg.vector_norm(position_tensor[1:] - position_tensor[:-1], dim=-1)
        recent = position_tensor[max(0, len(position_tensor) - max(5, args.steps // 5)):]
        table_contacts, _ = _contact_for_sensor(raw, "bulb_table_contact")
        report["zero_action"] = {
            "steps": args.steps,
            "success_latched_any": any(bool(any(row)) if isinstance(row, list) else bool(row) for row in zero_success if row is not None),
            "done_any": any(bool(any(row)) if isinstance(row, list) else bool(row) for row in zero_done if row is not None),
            "finite": finite,
            "max_step_displacement_m": float(displacement.max().item()) if displacement.numel() else 0.0,
            "final_position_std_m": _tolist(recent.std(dim=0)),
            "final_table_contact": _tolist(table_contacts),
            "stable_tabletop_heuristic": bool(
                finite and displacement.numel() > 0 and float(displacement.max()) < 0.03
                # ``recent`` is [time, env, xyz]; keep the environment axis
                # when selecting the world-z coordinate.  The previous
                # ``recent[:, 2]`` indexed the environment axis and crashed
                # for the two-environment smoke probe.
                and float(recent[:, :, 2].std()) < 0.01 and bool(table_contacts.any())
            ),
        }

        # Partial reset isolation, including timers used by lost-grasp and the
        # one-second success dwell.  Set sentinels before resetting so a stale
        # timer cannot hide behind an absent attribute.
        ids = torch.arange(raw.num_envs, device=raw.device)
        selected, untouched = ids[::2], ids[1::2]
        if not hasattr(raw, "_assembly_hold_time"):
            raw._assembly_hold_time = torch.zeros(raw.num_envs, device=raw.device)
        if not hasattr(raw, "_assembly_no_contact_time"):
            raw._assembly_no_contact_time = torch.zeros(raw.num_envs, device=raw.device)
        raw._assembly_hold_time[:] = torch.arange(raw.num_envs, device=raw.device, dtype=torch.float32) + 0.25
        raw._assembly_no_contact_time[:] = torch.arange(raw.num_envs, device=raw.device, dtype=torch.float32) + 0.5
        hold_before, no_contact_before = raw._assembly_hold_time.clone(), raw._assembly_no_contact_time.clone()
        released_before = getattr(raw, "_bulb_released", torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)).clone()
        success_before = getattr(raw, "_bulb_success", torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)).clone()
        phase_before = raw._bulb_assembly_state.phase.clone() if hasattr(raw, "_bulb_assembly_state") else None
        raw._reset_idx(selected)
        hold_after, no_contact_after = raw._assembly_hold_time.clone(), raw._assembly_no_contact_time.clone()
        released_after = getattr(raw, "_bulb_released", torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device))
        success_after = getattr(raw, "_bulb_success", torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device))
        phase_after = raw._bulb_assembly_state.phase.clone() if hasattr(raw, "_bulb_assembly_state") else None
        report["partial_reset"] = {
            "selected_env_ids": _tolist(selected), "untouched_env_ids": _tolist(untouched),
            "selected_hold_zero": bool(torch.allclose(hold_after[selected], torch.zeros_like(hold_after[selected]), atol=1e-7)),
            "selected_no_contact_zero": bool(torch.allclose(no_contact_after[selected], torch.zeros_like(no_contact_after[selected]), atol=1e-7)),
            "untouched_hold_preserved": bool(torch.equal(hold_after[untouched], hold_before[untouched])),
            "untouched_no_contact_preserved": bool(torch.equal(no_contact_after[untouched], no_contact_before[untouched])),
            "untouched_phase_preserved": bool(phase_before is None or torch.equal(phase_after[untouched], phase_before[untouched])),
            "selected_released_cleared": bool(not released_after[selected].any()),
            "selected_success_cleared": bool(not success_after[selected].any()),
            "untouched_released_preserved": bool(torch.equal(released_after[untouched], released_before[untouched])),
            "untouched_success_preserved": bool(torch.equal(success_after[untouched], success_before[untouched])),
            "hold_before": _tolist(hold_before), "hold_after": _tolist(hold_after),
            "no_contact_before": _tolist(no_contact_before), "no_contact_after": _tolist(no_contact_after),
        }

        checks = dict(initial_checks)
        checks.update({
            "zero_action_not_success": not report["zero_action"]["success_latched_any"],
            "zero_action_finite": report["zero_action"]["finite"],
            "tabletop_stable": report["zero_action"]["stable_tabletop_heuristic"],
            "partial_selected_hold_cleared": report["partial_reset"]["selected_hold_zero"],
            "partial_selected_no_contact_cleared": report["partial_reset"]["selected_no_contact_zero"],
            "partial_untouched_hold_preserved": report["partial_reset"]["untouched_hold_preserved"],
            "partial_untouched_no_contact_preserved": report["partial_reset"]["untouched_no_contact_preserved"],
            "partial_untouched_phase_preserved": report["partial_reset"]["untouched_phase_preserved"],
            "partial_selected_released_cleared": report["partial_reset"]["selected_released_cleared"],
            "partial_selected_success_cleared": report["partial_reset"]["selected_success_cleared"],
            "partial_untouched_released_preserved": report["partial_reset"]["untouched_released_preserved"],
            "partial_untouched_success_preserved": report["partial_reset"]["untouched_success_preserved"],
        })
        report["checks"] = checks
        report["passed"] = all(checks.values())
    finally:
        env.close()


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--num_envs", "--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--pure-collision", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 2:
        parser.error("--steps must be positive and --num_envs must be at least 2")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "bulb_assembly_physics_probe",
        "protocol_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "task": args.task,
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "steps": args.steps,
        "num_envs": args.num_envs,
    }
    step_path = args.out / "steps.jsonl"
    report_path = args.out / "report.json"
    app = AppLauncher(args).app
    try:
        with step_path.open("w", encoding="utf-8") as stream:
            run_probe(args, report, stream)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"BULB ASSEMBLY PROBE {'PASS' if report['passed'] else 'FAIL'}: {report_path}", flush=True)
        try:
            app.app.post_quit(0 if report["passed"] else 1)
        except Exception:
            pass
        app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
