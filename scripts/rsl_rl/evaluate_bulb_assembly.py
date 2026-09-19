"""Evaluate the complete tabletop bulb assembly task from a fixed checkpoint.

This evaluator intentionally creates a fresh Isaac environment for each seed.
It counts only complete vectorized episodes and reads the task's raw
``success`` termination (plus the ``AssemblyState`` ledger) rather than
inferring success from reward.  The assembly task's reset places a free bulb
on the table; this script does not call any pre-grasp, pre-alignment, or
pre-engagement helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


TASK = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0"
DEFAULT_SEEDS = (101, 202, 303)


def _as_bool_tensor(value, *, device):
    """Convert a manager term or done vector to a detached boolean tensor."""
    import torch

    if value is None:
        return torch.zeros(0, dtype=torch.bool, device=device)
    return torch.as_tensor(value, device=device).detach().bool().flatten()


def _safe_float(value, index: int, default: float = 0.0) -> float:
    """Read one scalar tensor element without leaking CUDA tensors to JSON."""
    try:
        return float(value[index].detach().item())
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return default


def _term_snapshot(raw) -> dict[str, list[bool]]:
    """Capture all active termination terms directly from the manager."""
    terms = {}
    manager = getattr(raw, "termination_manager", None)
    names = list(getattr(manager, "active_terms", ())) if manager is not None else []
    for name in names:
        try:
            terms[name] = _as_bool_tensor(manager.get_term(name), device=raw.device).cpu().tolist()
        except (AttributeError, RuntimeError, TypeError):
            # A custom IsaacLab build may expose active_terms before a term is
            # allocated. Keep the report explicit without stopping evaluation.
            terms[name] = []
    return terms


def _build_env_and_runner(args, seed: int, output: Path):
    """Build one fresh environment and load the bounded Bulb policy."""
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import BrainCo_DexHand  # noqa: F401  (registers the task)
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.agents.rsl_rl_ppo_cfg import (
        DexsuiteRevo3RotateBulbPPORunnerCfg,
    )
    from BrainCo_DexHand.tactile_representation.policy import prepare_policy_run
    from bulb_training import (
        bulb_inference_policy,
        configure_bulb_checkpoint,
        prepare_bulb_policy_run,
    )

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = seed
    cfg.sim.device = args.device
    if args.episode_length_s is not None:
        if args.episode_length_s <= 0:
            raise ValueError("--episode-length-s must be positive")
        cfg.episode_length_s = float(args.episode_length_s)
    # These are task configuration assertions, not curriculum aids. The
    # assembly config starts with a detached rigid bulb on the tabletop.
    cfg.bulb_pure_collision = True
    cfg.tactile_policy_enabled = False
    cfg.tactile_reconstruction_diagnostics = False
    if not getattr(cfg, "bulb_assembly_task", False):
        raise ValueError(f"{args.task} did not resolve to the independent assembly configuration")
    cfg.log_dir = str(output)

    agent_cfg = DexsuiteRevo3RotateBulbPPORunnerCfg()
    agent_cfg.device = args.device
    configure_bulb_checkpoint(agent_cfg, args.checkpoint, training=False)

    env = gym.make(args.task, cfg=cfg)
    # Validate both observation contracts against the fixed checkpoint before
    # constructing the runner. These helpers do not alter episode state.
    prepare_policy_run(env, output, resume_path=args.checkpoint)
    prepare_bulb_policy_run(env, output, resume_path=args.checkpoint)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint, map_location=args.device)
    policy = bulb_inference_policy(runner.get_inference_policy(device=env.unwrapped.device))
    return env, wrapped, runner, policy, cfg


def _evaluate_seed(args, seed: int, output: Path) -> dict:
    """Evaluate one seed until the requested number of complete episodes."""
    import torch

    env = wrapped = runner = None
    episodes: list[dict] = []
    try:
        env, wrapped, runner, policy, cfg = _build_env_and_runner(args, seed, output)
        raw = env.unwrapped
        raw.reset(seed=seed)
        obs = wrapped.get_observations()
        count = raw.num_envs
        max_phase = torch.zeros(count, dtype=torch.long, device=raw.device)
        max_advance = torch.zeros(count, device=raw.device)
        max_hold = torch.zeros(count, device=raw.device)
        contact_seen = torch.zeros(count, dtype=torch.bool, device=raw.device)
        episode_index = torch.zeros(count, dtype=torch.long, device=raw.device)

        # A no-termination policy still gets enough time for the configured
        # episode timeout. The margin catches staggered vectorized resets.
        episode_steps = math.ceil(float(cfg.episode_length_s) / float(cfg.sim.dt * cfg.decimation))
        max_steps = max(episode_steps + 1, math.ceil(args.episodes_per_seed / count) * episode_steps * 3)
        steps = 0
        # Capture terminal evidence before ManagerBasedRLEnv resets the scene.
        # Reading the state after step() loses the final hold sample.
        terminal_evidence = {}
        original_reset = raw._reset_idx

        def capture_before_reset(env_ids):
            state = raw._bulb_assembly_state
            for index in env_ids.tolist():
                terminal_evidence[index] = {
                    "phase": int(state.phase[index].item()),
                    "hold_s": float(raw._assembly_hold_time[index].item()),
                    "advance_m": float(state.best_advance[index].item()),
                    "support_contact": bool(state.last_contact[index]),
                }
            original_reset(env_ids)

        raw._reset_idx = capture_before_reset
        while len(episodes) < args.episodes_per_seed and steps < max_steps:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = wrapped.step(actions)
            steps += 1

            state = getattr(raw, "_bulb_assembly_state", None)
            if state is not None:
                max_phase = torch.maximum(max_phase, state.phase.detach())
                max_advance = torch.maximum(max_advance, state.best_advance.detach())
                hold = getattr(raw, "_assembly_hold_time", None)
                if hold is not None:
                    max_hold = torch.maximum(max_hold, hold.detach())
                contact_seen |= state.last_contact.detach().bool()

            done = _as_bool_tensor(dones, device=raw.device)
            if not bool(done.any()):
                continue
            try:
                success_term = _as_bool_tensor(
                    raw.termination_manager.get_term("success"), device=raw.device
                )
            except (AttributeError, RuntimeError, TypeError):
                success_term = torch.zeros(count, dtype=torch.bool, device=raw.device)
            term_snapshot = _term_snapshot(raw)
            for env_id in done.nonzero(as_tuple=False).flatten().tolist():
                if len(episodes) >= args.episodes_per_seed:
                    break
                episode_success = bool(success_term[env_id].item()) if env_id < len(success_term) else False
                terminal = terminal_evidence.pop(env_id, {})
                failures = any(
                    values[env_id] for name, values in term_snapshot.items()
                    if name not in ("success", "time_out") and env_id < len(values)
                )
                episode_success = episode_success and not failures and terminal.get("hold_s", 0) >= 1.0 - 1e-6
                episodes.append({
                    "episode": len(episodes),
                    "vector_env_id": env_id,
                    "episode_index_in_env": int(episode_index[env_id].item()),
                    "success": episode_success,
                    "success_source": "termination_manager.success",
                    "terminal_raw_state": terminal,
                    "max_phase": int(max_phase[env_id].item()),
                    "max_advance_m": _safe_float(max_advance, env_id),
                    "max_hold_duration_s": _safe_float(max_hold, env_id),
                    "contact_seen": bool(contact_seen[env_id].item()),
                    "termination_terms": {
                        name: bool(values[env_id]) if env_id < len(values) else False
                        for name, values in term_snapshot.items()
                    },
                })
                episode_index[env_id] += 1
                max_phase[env_id] = 0
                max_advance[env_id] = 0
                max_hold[env_id] = 0
                contact_seen[env_id] = False

            (output / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")

        successes = sum(int(item["success"]) for item in episodes)
        return {
            "seed": seed,
            "episodes_requested": args.episodes_per_seed,
            "episodes_completed": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes) if episodes else 0.0,
            "steps": steps,
            "max_steps": max_steps,
            "complete": len(episodes) == args.episodes_per_seed,
            "episodes": episodes,
        }
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Fixed bounded BulbActorCritic checkpoint (.pt).")
    parser.add_argument(
        "--out", "--output", dest="out", type=Path, required=True,
        help="New directory for the evaluation report.",
    )
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--num_envs", "--num-envs", type=int, default=32)
    parser.add_argument("--episodes-per-seed", "--episodes_per_seed", dest="episodes_per_seed", type=int, default=100)
    parser.add_argument(
        "--episode-length-s", type=float, default=None,
        help="Optional shortened episode duration for evaluator smoke tests; omit for the frozen task protocol.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes_per_seed < 1 or not args.seeds:
        parser.error("--num-envs, --episodes-per-seed and --seeds must be non-empty positive values")
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = _parse_args()
    # Isaac's SimulationContext cannot reliably recreate this sensor scene in
    # one process. Each seed therefore owns a separate simulator process.
    if len(args.seeds) > 1:
        per_seed = []
        for seed in args.seeds:
            child_out = args.out / f"seed_{seed}"
            command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", args.checkpoint,
                       "--out", str(child_out), "--seeds", str(seed), "--num_envs", str(args.num_envs),
                       "--episodes-per-seed", str(args.episodes_per_seed), "--device", args.device]
            if args.headless:
                command.append("--headless")
            if args.episode_length_s is not None:
                command.extend(["--episode-length-s", str(args.episode_length_s)])
            with (args.out / f"seed_{seed}.log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            path = child_out / "report.json"
            if not path.exists():
                raise RuntimeError(f"Seed {seed} exited {completed.returncode} without report; see {log.name}")
            child = json.loads(path.read_text())
            per_seed.extend(child.get("per_seed", []))
            print(f"[bulb-eval] completed seed={seed} exit={completed.returncode}", flush=True)
        total = sum(item["episodes_completed"] for item in per_seed)
        successes = sum(item["successes"] for item in per_seed)
        protocol = args.seeds == list(DEFAULT_SEEDS) and args.episodes_per_seed == 100 and args.episode_length_s is None
        passed = (protocol and len(per_seed) == 3 and all(item["complete"] for item in per_seed)
                  and successes / max(1, total) >= .30 and all(item["success_rate"] >= .20 for item in per_seed))
        report = {"schema": "bulb_assembly_independent_evaluation", "task": args.task,
                  "checkpoint": args.checkpoint,
                  "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  "formal_protocol": protocol, "passed": passed, "per_seed": per_seed,
                  "overall": {"episodes_completed": total, "successes": successes,
                              "success_rate": successes / max(1, total)}}
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return 0 if passed else 1
    app = AppLauncher(args).app
    report = {
        "schema": "bulb_assembly_independent_evaluation",
        "protocol_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "num_envs": args.num_envs,
        "episodes_per_seed": args.episodes_per_seed,
        "episode_length_s": args.episode_length_s,
        "seeds": args.seeds,
        "pregrasp_aid": False,
        "prealign_aid": False,
        "preengage_aid": False,
        "passed": False,
        "per_seed": [],
    }
    try:
        for seed in args.seeds:
            seed_out = args.out / f"seed_{seed}"
            seed_out.mkdir()
            result = _evaluate_seed(args, seed, seed_out)
            report["per_seed"].append(result)
            print(
                f"[bulb-eval] seed={seed} episodes={result['episodes_completed']}/"
                f"{args.episodes_per_seed} successes={result['successes']}",
                flush=True,
            )
        completed = [item for item in report["per_seed"] if item["complete"]]
        total_episodes = sum(item["episodes_completed"] for item in report["per_seed"])
        total_successes = sum(item["successes"] for item in report["per_seed"])
        report["overall"] = {
            "episodes_completed": total_episodes,
            "successes": total_successes,
            "success_rate": total_successes / total_episodes if total_episodes else 0.0,
            "all_seeds_complete": len(completed) == len(args.seeds),
        }
        # The acceptance threshold is recorded as an explicit protocol check;
        # this script never changes it or treats a partial seed as success.
        report["acceptance"] = {
            "overall_min_success_rate": 0.30,
            "per_seed_min_success_rate": 0.20,
            "hold_duration_s": 1.0,
            "overall_pass": bool(
                report["overall"]["all_seeds_complete"]
                and report["overall"]["success_rate"] >= 0.30
                and all(item["success_rate"] >= 0.20 for item in completed)
            ),
        }
        report["formal_protocol"] = (args.seeds == list(DEFAULT_SEEDS) and args.episodes_per_seed == 100
                                     and args.episode_length_s is None)
        report["passed"] = report["formal_protocol"] and report["acceptance"]["overall_pass"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        path = args.out / "report.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"BULB ASSEMBLY EVALUATION {'PASS' if report['passed'] else 'FAIL'}: {path}", flush=True)
        try:
            app.app.post_quit(0 if report["passed"] else 1)
        except Exception:
            pass
        app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
