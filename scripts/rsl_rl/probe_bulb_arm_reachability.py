#!/usr/bin/env python3
"""Probe arm-only reachability of the tabletop bulb from the real reset.

This is a read-only diagnostic: it commands only bounded arm joint targets,
keeps all hand actions at zero, and never writes object poses or assembly
state.  A small cross-entropy search estimates whether the reset arm can move
the existing hand to the tabletop bulb without relying on a hidden guide.
The result is diagnostic-only and cannot count as a task success.
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


def _step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, extras = result
        return obs, reward, terminated | truncated, extras
    return result


def _reset(env, seed):
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def run(args, report):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    sys.path.insert(0, str(SOURCE))
    import BrainCo_DexHand.tasks  # noqa: F401
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.env_cfg import (
        rotate_bulb_hand_shell_distance,
    )

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.population)
    cfg.seed = args.seed
    cfg.bulb_pure_collision = True
    # Keep the physical scene and reset, but prevent this diagnostic's random
    # candidates from being replaced by task failure resets mid-trajectory.
    for name in ("bulb_dropped", "lost_grasp", "object_out_of_bound", "success"):
        if hasattr(cfg.terminations, name):
            setattr(cfg.terminations, name, None)
    cfg.episode_length_s = max(float(getattr(cfg, "episode_length_s", 20.0)), 60.0)
    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    try:
        _reset(env, args.seed)
        action_dim = int(raw.action_manager.action.shape[-1])
        term = raw.action_manager.get_term("action")
        arm_indices = list(term.arm_action_indices)
        hand_indices = list(term.hand_action_indices)
        zero = torch.zeros((args.population, action_dim), device=raw.device)
        initial = rotate_bulb_hand_shell_distance(raw).clone()
        generator = torch.Generator(device=raw.device)
        generator.manual_seed(args.seed + 4019)
        # Optimize a constant normalized arm direction over the horizon.  This
        # keeps the reachability search seven-dimensional and tests the actual
        # bounded incremental action contract over a useful time window.
        mean = torch.zeros((len(arm_indices),), device=raw.device)
        std = torch.full_like(mean, args.initial_std)
        best_score = torch.full((args.population,), float("inf"), device=raw.device)
        best_sequence = torch.zeros_like(mean)
        history = []
        for generation in range(args.generations):
            _reset(env, args.seed + generation + 1)
            sequence = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                (args.population, len(arm_indices)),
                generator=generator, device=raw.device,
            )
            sequence = sequence.clamp(-1.0, 1.0)
            for step in range(args.horizon):
                action = zero.clone()
                action[:, arm_indices] = sequence
                action[:, hand_indices] = 0.0
                _step(env, action)
            score = rotate_bulb_hand_shell_distance(raw)
            order = torch.argsort(score)
            elite_count = max(1, int(args.population * args.elite_fraction))
            elite = sequence[order[:elite_count]]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0, unbiased=False).clamp_min(args.min_std)
            improved = score < best_score
            best_score = torch.where(improved, score, best_score)
            if bool(score.min() <= best_score.min() + 1e-8):
                best_sequence = sequence[torch.argmin(score)].clone()
            history.append({
                "generation": generation,
                "best_m": float(score.min().item()),
                "median_m": float(score.median().item()),
                "elite_m": float(score[order[:elite_count]].mean().item()),
                "std_mean": float(std.mean().item()),
            })
        # Replay the best open-loop sequence from a fresh reset for a final
        # measurement, without changing any task-owned pose/state.
        _reset(env, args.seed + 10000)
        for step in range(args.horizon):
            action = zero.clone()
            action[:, arm_indices] = best_sequence
            _step(env, action)
        final = rotate_bulb_hand_shell_distance(raw)
        report.update({
            "config": {
                "task": args.task, "device": args.device, "seed": args.seed,
                "population": args.population, "horizon": args.horizon,
                "generations": args.generations, "initial_std": args.initial_std,
                "elite_fraction": args.elite_fraction, "acceptance_eligible": False,
                "arm_action_indices": arm_indices,
            },
            "initial_best_pair_distance_m": _tolist(initial),
            "final_best_pair_distance_m": _tolist(final),
            "search_history": history,
            "best_distance_m": float(final.min().item()),
            "finite": bool(torch.isfinite(final).all().item()),
            "passed": bool(torch.isfinite(final).all().item()),
        })
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--initial-std", type=float, default=0.35)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=97)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "bulb_arm_reachability_probe",
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
    args.out.joinpath("report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"BULB ARM REACH PROBE {'PASS' if report['passed'] else 'FAIL'}: {args.out / 'report.json'}", flush=True)
    try:
        app.close(wait_for_replicator=False, skip_cleanup=True)
    except Exception:
        pass
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
