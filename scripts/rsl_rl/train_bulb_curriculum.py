#!/usr/bin/env python3
"""Run the assembly stages through train.py, with independent promotion checks.

One process per stage keeps the Stage0 world fixture out of every free stage.
This launcher does not reserve GPUs or start any remote process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TASK = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0"
PHASES = (("stage0", 0, True), ("stage0_free", 0, False), ("stage1", 1, False),
          ("stage2", 2, False), ("stage3", 3, False), ("full", 4, False))
THRESHOLDS = (.9, .8, .85, .7)


def training_command(args, phase, directory, checkpoint=None, *, transfer=False):
    _, stage, fixed = phase
    cmd = [sys.executable, str(ROOT / "scripts/rsl_rl/train.py"), "--task", TASK,
           "--num_envs", str(args.num_envs), "--seed", str(args.seed), "--device", args.device,
           "--max_iterations", str(args.iterations_per_stage), "--logger", "tensorboard", "--headless",
           "--log_dir", str(directory), f"env.bulb_curriculum.stage={stage}",
           f"env.bulb_curriculum.fixed_bulb={str(fixed).lower()}",
           f"env.bulb_curriculum.transfer={str(transfer).lower()}"]
    if getattr(args, "kit_args", ""):
        cmd += ["--kit_args", args.kit_args]
    if checkpoint:
        cmd += ["--resume", "--curriculum_checkpoint", str(checkpoint)]
    return cmd + list(args.env_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kit_args", default="", help="Kit GPU/thread arguments shared by every training and evaluation process")
    parser.add_argument("--iterations-per-stage", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=100, help="Per independent validation seed")
    parser.add_argument("--eval-seeds", type=int, nargs=3, default=[401, 402, 403])
    parser.add_argument("--start", choices=[x[0] for x in PHASES], default="stage0")
    parser.add_argument("--checkpoint", type=Path, help="Resume the selected phase (not a stage bypass)")
    parser.add_argument("--dry-run", action="store_true", help="Print the first command without launching Isaac")
    parser.add_argument("env_args", nargs="*", help="Additional Hydra overrides forwarded to train and evaluator")
    args = parser.parse_args()
    if min(args.num_envs, args.iterations_per_stage, args.eval_episodes) < 1 or len(set(args.eval_seeds)) != 3:
        parser.error("Counts must be positive and validation seeds must be distinct")
    if args.start != "stage0" and args.checkpoint is None:
        parser.error("--start requires the selected phase's checkpoint; cannot bypass predecessor learning")
    if any(not x.startswith("env.") for x in args.env_args):
        parser.error("Only env.* overrides can be shared by training and independent evaluation")
    if any(x.startswith(("env.bulb_curriculum.", "env.seed", "env.episode_length_s", "env.task_episode_length_s")) for x in args.env_args):
        parser.error("Stage, seed, time budget and curriculum settings are owned by this launcher")
    start = [x[0] for x in PHASES].index(args.start)
    output = args.out.resolve()
    if args.dry_run:
        print(json.dumps(training_command(args, PHASES[start], output / PHASES[start][0], args.checkpoint), indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    process_env = dict(os.environ)
    process_env["PYTHONPATH"] = str(ROOT / "source/BrainCo_DexHand") + os.pathsep + process_env.get("PYTHONPATH", "")
    history = []
    for index, phase in enumerate(PHASES[start:], start):
        name, stage, fixed = phase
        directory = output / name
        cmd = training_command(args, phase, directory, checkpoint, transfer=index > start)
        (output / f"{name}_launch.json").write_text(json.dumps(cmd, indent=2) + "\n")
        subprocess.run(cmd, cwd=ROOT, env=process_env, check=True)
        status = json.loads((directory / "curriculum_status.json").read_text())
        history.append(status)
        (output / "curriculum_progress.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint = Path(status["checkpoint"])
        if stage == 4:
            print(f"Full 20-second training completed: {checkpoint}")
            return 0
        if not status["promotion_nominated"]:
            print(f"{name} reached its budget without promotion; checkpoint retained: {checkpoint}")
            return 2
        rates = []
        for seed in args.eval_seeds:
            evaluation = output / f"{name}_validation_{seed}"
            cmd = [sys.executable, str(ROOT / "scripts/rsl_rl/evaluate_bulb_curriculum.py"),
                   "--checkpoint", str(checkpoint), "--out", str(evaluation), "--episodes", str(args.eval_episodes),
                   "--seed", str(seed), "--device", args.device, "--num_envs", str(min(32, args.num_envs)), "--headless"]
            if args.kit_args:
                cmd += ["--kit_args", args.kit_args]
            cmd += list(args.env_args)
            subprocess.run(cmd, cwd=ROOT, env=process_env, check=True)
            report = json.loads((evaluation / "summary.json").read_text())
            rates.append(report["successes"] / report["episodes"])
        passed = sum(rates)/len(rates) >= THRESHOLDS[stage] and min(rates) >= THRESHOLDS[stage] - .1
        status["independent_validation"] = {"seeds": args.eval_seeds, "rates": rates, "passed": passed}
        (output / "curriculum_progress.json").write_text(json.dumps(history, indent=2) + "\n")
        if not passed:
            print(f"{name} independent validation failed; no stage advance, all evidence retained")
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
