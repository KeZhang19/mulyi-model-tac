#!/usr/bin/env python3
"""Measure whether reset geometry and bounded actions can reach the tabletop bulb.

This read-only probe records live fingertip-to-shell distance under zero and
deterministic random bounded actions. It never writes a pose, attachment,
phase, engagement, installation, or success flag and is not an acceptance
evaluator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback


TASK = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0"
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "BrainCo_DexHand"


def _tolist(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def _finite(value):
    import torch

    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return True


def _reset(env, seed):
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def _step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, extras = result
        return obs, reward, terminated | truncated, extras
    return result


def _reach_metrics(raw, distance_fn):
    import torch

    distance = distance_fn(raw)
    robot = raw.scene["robot"]
    bulb = raw.scene["object"]
    names = ("right_thumb_tip_Link", "right_index_tip_Link", "right_middle_tip_Link",
             "right_ring_tip_Link", "right_little_tip_Link")
    ids = [robot.body_names.index(name) for name in names]
    bulb_axis = torch.zeros_like(bulb.data.root_pos_w)
    bulb_axis[:, 0] = 0.04
    from isaaclab.utils.math import combine_frame_transforms
    shell, _ = combine_frame_transforms(bulb.data.root_pos_w, bulb.data.root_quat_w, bulb_axis)
    tips = robot.data.body_pos_w[:, ids]
    tip_distance = torch.linalg.vector_norm(tips - shell[:, None, :], dim=-1)
    contact = torch.zeros(raw.num_envs, device=raw.device)
    for finger in ("thumb", "index", "mid", "ring", "pinky"):
        sensor = raw.scene.sensors.get(f"right_{finger}dip_roll_rubber_link_object_s")
        if sensor is not None and sensor.data.force_matrix_w is not None:
            force = torch.nan_to_num(sensor.data.force_matrix_w).reshape(raw.num_envs, -1, 3)
            contact = torch.maximum(contact, torch.linalg.vector_norm(force, dim=-1).amax(dim=-1))
    return {
        "best_pair_distance_m": _tolist(distance),
        "all_tip_distance_m": _tolist(tip_distance),
        "max_finger_contact_n": _tolist(contact),
        "bulb_root_pos_w": _tolist(bulb.data.root_pos_w),
        "robot_tip_pos_w": _tolist(tips),
    }


def run(args, report):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    sys.path.insert(0, str(SOURCE))
    import BrainCo_DexHand.tasks  # noqa: F401
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.env_cfg import (
        rotate_bulb_hand_shell_distance,
    )

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.bulb_pure_collision = True
    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    try:
        reset_obs = _reset(env, args.seed)
        report["reset_observation_finite"] = _finite(reset_obs[0] if isinstance(reset_obs, tuple) else reset_obs)
        report["config"] = {
            "task": args.task,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "device": args.device,
            "zero_steps": args.zero_steps,
            "random_steps": args.random_steps,
            "random_action_scale": args.random_action_scale,
            "acceptance_eligible": False,
        }
        report["initial"] = _reach_metrics(raw, rotate_bulb_hand_shell_distance)
        action_dim = int(raw.action_manager.action.shape[-1])
        zero = torch.zeros((raw.num_envs, action_dim), device=raw.device)
        best = torch.full((raw.num_envs,), float("inf"), device=raw.device)
        for _ in range(args.zero_steps):
            _step(env, zero)
            best = torch.minimum(best, rotate_bulb_hand_shell_distance(raw))
        generator = torch.Generator(device=raw.device)
        generator.manual_seed(args.seed + 1009)
        for _ in range(args.random_steps):
            action = torch.randn((raw.num_envs, action_dim), generator=generator, device=raw.device)
            action = action.clamp(-1.0, 1.0) * args.random_action_scale
            _step(env, action)
            best = torch.minimum(best, rotate_bulb_hand_shell_distance(raw))
        report["final"] = _reach_metrics(raw, rotate_bulb_hand_shell_distance)
        report["best_pair_distance_m"] = _tolist(best)
        report["finite"] = bool(torch.isfinite(best).all().item())
        report["passed"] = report["reset_observation_finite"] and report["finite"]
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--zero-steps", type=int, default=60)
    parser.add_argument("--random-steps", type=int, default=300)
    parser.add_argument("--random-action-scale", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "bulb_reachability_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False,
    }
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        run(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"BULB REACH PROBE {'PASS' if report['passed'] else 'FAIL'}: {args.out / 'report.json'}", flush=True)
    try:
        app.close(wait_for_replicator=False, skip_cleanup=True)
    except Exception:
        pass
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
