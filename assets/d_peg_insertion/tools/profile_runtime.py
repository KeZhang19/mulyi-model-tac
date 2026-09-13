#!/usr/bin/env python3
"""Profile a standalone D-peg rollout, with a reusable action sequence for A/B runs.

Example (from the repository root, using the Isaac Lab Python environment)::

    python assets/d_peg_insertion/tools/profile_runtime.py --headless \
        --num_envs 512 --warmup 8 --steps 32 --device cuda:0 \
        --checkpoint /path/to/model_250.pt --output /tmp/peg-baseline
    python assets/d_peg_insertion/tools/profile_runtime.py --headless \
        --num_envs 512 --warmup 8 --steps 32 --device cuda:0 \
        --action-file /tmp/peg-baseline/actions.pt \
        --socket-usd /path/to/candidate.usd --output /tmp/peg-candidate

The first pass measures synchronized env.step wall time without component hooks.
The second pass resets and replays those actions with synchronized nested hooks.
Component inclusive times overlap; exclusive times partition the instrumented
env.step, and must not be compared directly with production asynchronous timings.
This script does not update policies or attach to a running training process.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]
TASK = "BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0"
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def summarize(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "total_s": 0.0}
    return {
        "count": len(ordered), "total_s": sum(ordered),
        "mean_s": statistics.mean(ordered), "median_s": statistics.median(ordered),
        "min_s": ordered[0], "max_s": ordered[-1],
        "p95_s": ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))],
    }


def load_saved_config(path):
    """Load Isaac dump_yaml output, allowing its explicit serialized slice tag."""
    import yaml

    class ConfigLoader(yaml.FullLoader):
        pass

    def construct_slice(loader, node):
        values = loader.construct_sequence(node, deep=True)
        if len(values) != 3 or any(value is not None and not isinstance(value, int) for value in values):
            raise ValueError(f"Invalid serialized scene-entity slice: {values!r}")
        return slice(*values)

    ConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", construct_slice)
    with Path(path).open() as stream:
        return yaml.load(stream, Loader=ConfigLoader)


def restore_saved_config(config, path):
    """Materialize optional scalar defaults, then use Isaac's typed restoration.

    Isaac from_dict rejects None -> int/string even for fields such as seed or
    log_dir. Preserve its callable resolution and nested config types, while
    handling those optional values explicitly. A missing typed config object is
    deliberately not reconstructed from an untyped mapping.
    """
    from collections.abc import Mapping
    import copy

    data = load_saved_config(path)
    materialized = []

    def prepare(target, values, namespace=""):
        for name, value in values.items():
            key = f"{namespace}/{name}"
            if isinstance(target, dict):
                if name not in target:
                    continue  # Let from_dict report the exact unknown key.
                previous = target[name]
            else:
                if not hasattr(target, name):
                    continue
                previous = getattr(target, name)
            if previous is None and value is not None:
                if isinstance(value, Mapping) or (
                    isinstance(value, (tuple, list)) and any(isinstance(item, Mapping) for item in value)
                ):
                    raise ValueError(f"Cannot restore absent typed config from mapping at {key}")
                if isinstance(target, dict):
                    target[name] = copy.deepcopy(value)
                else:
                    setattr(target, name, copy.deepcopy(value))
                materialized.append(key)
            elif isinstance(value, Mapping) and previous is not None:
                prepare(previous, value, key)
            elif isinstance(value, (list, tuple)) and isinstance(previous, (list, tuple)):
                for index, (old, new) in enumerate(zip(previous, value)):
                    if isinstance(new, Mapping) and old is not None:
                        prepare(old, new, f"{key}/{index}")

    prepare(config, data)
    config.from_dict(data)
    return materialized


class NestedProfiler:
    """Synchronous call-tree timers, with child time subtracted exactly once."""

    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.enabled = False
        self.stack = []
        self.samples = defaultdict(lambda: {"inclusive_s": [], "exclusive_s": []})
        self.restore = []
        self.installed = []

    @contextmanager
    def span(self, name):
        if not self.enabled:
            yield
            return
        self.synchronize()
        path = "/".join([frame["name"] for frame in self.stack] + [name])
        frame = {"name": name, "started": time.perf_counter(), "children_s": 0.0}
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - frame["started"]
            self.stack.pop()
            self.samples[path]["inclusive_s"].append(elapsed)
            self.samples[path]["exclusive_s"].append(elapsed - frame["children_s"])
            if self.stack:
                self.stack[-1]["children_s"] += elapsed

    def patch(self, owner, name, label=None):
        original = getattr(owner, name, None)
        if not callable(original):
            return

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.span(label or name):
                return original(*args, **kwargs)

        setattr(owner, name, wrapped)
        self.restore.append((owner, name, original))
        self.installed.append(label or name)

    def remove(self):
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)
        self.restore.clear()

    def report(self, steps):
        return {
            path: {
                "calls": len(values["inclusive_s"]),
                "inclusive": summarize(values["inclusive_s"]),
                "exclusive": summarize(values["exclusive_s"]),
                "inclusive_ms_per_control_step": 1000 * sum(values["inclusive_s"]) / steps,
                "exclusive_ms_per_control_step": 1000 * sum(values["exclusive_s"]) / steps,
            }
            for path, values in sorted(self.samples.items())
        }


def install_hooks(profiler, env):
    """Patch only this diagnostic process; term objects keep their reset methods."""
    from BrainCo_DexHand.tasks.manager_based.dexsuite.mdp import d_peg_insertion
    from BrainCo_DexHand.tasks.manager_based.dexsuite.mdp import observations as tactile
    from BrainCo_DexHand.tasks.manager_based.dexsuite.mdp import rotate_bulb_tactile

    for owner, names, prefix in (
        (env.sim, ("step", "render"), "sim"),
        (env.scene, ("write_data_to_sim", "update"), "scene"),
        (env.action_manager, ("process_action", "apply_action"), "actions"),
        (env.termination_manager, ("compute",), "terminations"),
        (env.reward_manager, ("compute",), "rewards"),
        (env.observation_manager, ("compute",), "observations"),
        (env.command_manager, ("compute",), "commands"),
        (env, ("_reset_idx",), "env"),
        (env._rotate_bulb_tactile_term.encoder, ("forward",), "tactile_encoder"),
    ):
        for name in names:
            profiler.patch(owner, name, f"{prefix}.{name}")

    # Wrapping term __call__ at class level leaves ManagerTermBase identity and
    # reset hooks intact. Plain observation functions can be wrapped on their cfg.
    wrapped_classes = set()
    manager = env.observation_manager
    for group, configs in manager._group_obs_term_cfgs.items():
        for term_name, config in zip(manager.active_terms[group], configs, strict=True):
            label = f"obs.{group}.{term_name}"
            if not isinstance(config.func, type) and hasattr(config.func, "reset"):
                cls = type(config.func)
                if cls not in wrapped_classes:
                    profiler.patch(cls, "__call__", label)
                    wrapped_classes.add(cls)
            else:
                profiler.patch(config, "func", label)

    for name in ("_geometry", "d_peg_geometry", "_snapshot"):
        profiler.patch(d_peg_insertion, name, f"geometry.{name}")
    profiler.patch(rotate_bulb_tactile, "observe_rotate_bulb_tactile", "tactile.inputs")
    profiler.patch(rotate_bulb_tactile, "project_marker_flow", "tactile.marker_projection")
    for name in (
        "ours_rl_hydroshear_obs", "hydroshear_rl_obs", "ours_rl_taxim_rgb_obs",
        "_ours_taxim_rgb_policy_from_local", "_ours_dense_tacmap_depth_from_local",
        "adaptive_local_tacmap_rl_obs", "tacmap_rl_obs", "_ours_tacmap_policy_components",
        "warpsdf_pressure_obs", "_hydroshear_marker_ray_measurements",
    ):
        profiler.patch(tactile, name, f"tactile.{name}")


def run(args, report):
    import gymnasium as gym
    import torch
    import BrainCo_DexHand.tasks  # noqa: F401
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from validate_training import finite_tree
    from BrainCo_DexHand.tasks.manager_based.dexsuite.mdp.d_peg_insertion import (
        d_peg_diagnostic_totals, d_peg_state_metrics,
    )
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.dexsuite_revo3_env_cfg_insert_d_peg import (
        DexsuiteRevo3InsertDPegEnvCfg,
    )
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.agents.rsl_rl_ppo_cfg_insert_d_peg import (
        DexsuiteRevo3InsertDPegPPORunnerCfg,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    cfg, agent = DexsuiteRevo3InsertDPegEnvCfg(), DexsuiteRevo3InsertDPegPPORunnerCfg()
    if args.env_yaml:
        report["env_optional_fields_materialized"] = restore_saved_config(cfg, args.env_yaml)
    if args.agent_yaml:
        report["agent_optional_fields_materialized"] = restore_saved_config(agent, args.agent_yaml)
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    cfg.log_dir = str(args.output)
    agent.device, agent.seed = args.device, args.seed
    if args.socket_usd:
        cfg.socket_usd_path = str(args.socket_usd)
        cfg.scene.socket.spawn.usd_path = str(args.socket_usd)
    if args.encoder_chunk_size is not None:
        cfg.tactile_encoder_chunk_size = args.encoder_chunk_size
    if args.taxim_chunk_size is not None:
        cfg.tactile_taxim_chunk_size = args.taxim_chunk_size
    if hasattr(cfg, "d_peg_fast_taxim"):
        cfg.d_peg_fast_taxim = args.fast_taxim
    elif args.fast_taxim:
        raise ValueError("--fast-taxim requires the task's optional d_peg_fast_taxim implementation")
    dump_yaml(str(args.output / "params/env.yaml"), cfg)
    dump_yaml(str(args.output / "params/agent.yaml"), agent)
    write_json(args.output / "params/runtime_overrides.json", {
        "encoder_channels_last": args.encoder_channels_last,
        "cudnn_benchmark": args.cudnn_benchmark,
        "fast_taxim": args.fast_taxim, "taxim_chunk_size": cfg.tactile_taxim_chunk_size,
        "precision": "float32", "allow_tf32": True,
    })
    env = gym.make(TASK, cfg=cfg)
    raw, profiler = env.unwrapped, None
    try:
        if args.encoder_channels_last:
            raw._rotate_bulb_tactile_term.encoder.encoder.to(memory_format=torch.channels_last)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        device = torch.device(raw.device)

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        report["configuration"] = {
            "task": TASK, "num_envs": raw.num_envs, "device": str(device), "seed": args.seed,
            "warmup": args.warmup, "steps": args.steps, "decimation": cfg.decimation,
            "physics_dt": raw.physics_dt, "control_dt": raw.step_dt,
            "rollout_steps_per_env": agent.num_steps_per_env,
            "socket_usd": cfg.socket_usd_path, "encoder_chunk_size": cfg.tactile_encoder_chunk_size,
            "encoder_channels_last": args.encoder_channels_last, "cudnn_benchmark": args.cudnn_benchmark,
            "fast_taxim": args.fast_taxim, "taxim_chunk_size": cfg.tactile_taxim_chunk_size,
            "action_mode": args.action_mode, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "replay_source": str(args.action_file) if args.action_file else None,
            "env_yaml": str(args.env_yaml) if args.env_yaml else None,
            "agent_yaml": str(args.agent_yaml) if args.agent_yaml else None,
        }
        report["tactile_policy_contract"] = raw.tactile_policy_contract
        report["source_files"] = {}
        source_paths = {"profile_runtime": Path(__file__).resolve()}
        for name in (
            "BrainCo_DexHand.tasks.manager_based.dexsuite.mdp.d_peg_insertion",
            "BrainCo_DexHand.tasks.manager_based.dexsuite.mdp.d_peg_tactile",
            "BrainCo_DexHand.tasks.manager_based.dexsuite.mdp.d_peg_tactile_runtime",
            "BrainCo_DexHand.tasks.manager_based.dexsuite.mdp.observations",
            "BrainCo_DexHand.tasks.manager_based.dexsuite.mdp.rotate_bulb_tactile",
            "BrainCo_DexHand.tactile_representation.policy",
            "isaaclab.envs.manager_based_rl_env",
        ):
            module = sys.modules.get(name)
            if module is not None and getattr(module, "__file__", None):
                source_paths[name] = Path(module.__file__).resolve()
        for name, path in source_paths.items():
            report["source_files"][name] = {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        if device.type == "cuda":
            report["gpu"] = {"name": torch.cuda.get_device_name(device),
                             "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory}
        replay = None
        if args.action_file:
            payload = torch.load(args.action_file, map_location="cpu", weights_only=True)
            replay = payload["actions"].to(device)
            expected = (args.warmup + args.steps, raw.num_envs, raw.action_manager.total_action_dim)
            if tuple(replay.shape) != expected:
                raise ValueError(f"Action replay shape {tuple(replay.shape)} != {expected}")
            if int(payload["seed"]) != args.seed or int(payload["warmup"]) != args.warmup:
                raise ValueError("Replay requires the original --seed and --warmup")
        policy, inference = None, None
        if args.checkpoint and replay is None:
            from duration_logging import DurationLoggingOnPolicyRunner
            from distributed_normalization import install_distributed_normalization_fix
            runner = DurationLoggingOnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=str(device))
            runner.load(str(args.checkpoint), load_optimizer=False, map_location=str(device))
            install_distributed_normalization_fix(runner)
            inference = runner.get_inference_policy(device=str(device))
            policy = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
            saved_contract_path = args.checkpoint.parent / "tactile_policy_contract.json"
            if saved_contract_path.is_file():
                from BrainCo_DexHand.tactile_representation.policy import validate_policy_contract
                actual = json.loads(json.dumps(raw.tactile_policy_contract))
                expected_contract = json.loads(saved_contract_path.read_text())
                if args.socket_usd:
                    actual["task_assets"]["socket"] = expected_contract["task_assets"]["socket"]
                    report["diagnostic_socket_contract_override"] = True
                validate_policy_contract(actual, expected_contract)
            report["checkpoint_iteration"] = int(runner.current_learning_iteration)

        @torch.inference_mode()
        def reset():
            torch.manual_seed(args.seed)
            observations, _ = env.reset(seed=args.seed)
            if policy is not None:
                policy.reset(torch.ones(raw.num_envs, device=device, dtype=torch.bool))
            return wrapped.get_observations()

        def select_action(observations, index):
            if replay is not None:
                return replay[index]
            if args.action_mode == "deterministic":
                return inference(observations)
            if args.action_mode == "sampled":
                return policy.act(observations)
            if args.action_mode == "random":
                return args.action_std * torch.randn((raw.num_envs, raw.action_manager.total_action_dim), device=device)
            return torch.zeros((raw.num_envs, raw.action_manager.total_action_dim), device=device)

        observations = reset()
        finite_tree(observations, "initial_observations")
        action_sequence, baseline_steps, action_times = [], [], []
        reward_mean, done_count = [], []
        with torch.inference_mode():
            for index in range(args.warmup + args.steps):
                synchronize()
                started = time.perf_counter()
                actions = select_action(observations, index)
                synchronize()
                action_time = time.perf_counter() - started
                action_sequence.append(actions.detach().clone())
                synchronize()
                started = time.perf_counter()
                observations, rewards, dones, _ = wrapped.step(actions)
                synchronize()
                step_time = time.perf_counter() - started
                # Diagnostic reductions stay outside the timed env.step span.
                finite_tree((observations, rewards, actions), f"baseline/{index}")
                if policy is not None:
                    policy.reset(dones)
                if index >= args.warmup:
                    baseline_steps.append(step_time)
                    action_times.append(action_time)
                    reward_mean.append(float(rewards.mean()))
                    done_count.append(int(dones.sum()))
                if (index + 1) % args.print_every == 0 or index + 1 == args.warmup + args.steps:
                    print(f"PROFILE_PROGRESS baseline {index + 1}/{args.warmup + args.steps} env_step_s={step_time:.6f}", flush=True)
        replay = torch.stack(action_sequence)
        action_cpu = replay.cpu().contiguous()
        torch.save({"actions": action_cpu, "seed": args.seed, "warmup": args.warmup}, args.output / "actions.pt")
        report["actions_sha256"] = hashlib.sha256(action_cpu.numpy().tobytes()).hexdigest()
        report["baseline"] = {
            "env_step": summarize(baseline_steps), "action_selection": summarize(action_times),
            "step_seconds": baseline_steps, "reward_mean_per_step": reward_mean,
            "completed_episodes_per_step": done_count,
            "estimated_collection_s_for_one_iteration": statistics.mean(baseline_steps) * agent.num_steps_per_env,
            "transitions_per_second": raw.num_envs / statistics.mean(baseline_steps),
        }
        write_json(args.output / "report.json", report)
        if args.profile:
            observations = reset()
            # Warm up the replay without hooks, then start component collection.
            with torch.inference_mode():
                for index in range(args.warmup):
                    observations, _, _, _ = wrapped.step(replay[index])
                profiler = NestedProfiler(synchronize)
                install_hooks(profiler, raw)
                profiler.enabled = True
                profile_rewards, profile_dones = [], []
                for index in range(args.steps):
                    with profiler.span("env.step"):
                        observations, rewards, dones, _ = wrapped.step(replay[args.warmup + index])
                    finite_tree((observations, rewards), f"components/{index}")
                    profile_rewards.append(float(rewards.mean()))
                    profile_dones.append(int(dones.sum()))
                    if (index + 1) % args.print_every == 0 or index + 1 == args.steps:
                        print(f"PROFILE_PROGRESS components {index + 1}/{args.steps}", flush=True)
                profiler.enabled = False
            report["components"] = profiler.report(args.steps)
            report["profile_replay"] = {
                "reward_mean_per_step": profile_rewards,
                "completed_episodes_per_step": profile_dones,
            }
            report["installed_hooks"] = profiler.installed
            total = report["components"]["env.step"]["inclusive"]["total_s"]
            partition = sum(item["exclusive"]["total_s"] for item in report["components"].values())
            report["exclusive_partition_error_s"] = partition - total
            report["instrumented_to_baseline_ratio"] = total / sum(baseline_steps)
        report["final_state"] = {
            name: {"root_pos_local_mean": (raw.scene[name].data.root_pos_w - raw.scene.env_origins).mean(0).cpu().tolist()}
            for name in ("object", "socket")
        }
        finite_tree({name: raw.scene[name].data.root_state_w for name in ("robot", "object", "socket")},
                    "final_root_states")
        finite_tree(raw.scene["robot"].data.joint_pos, "final_joint_positions")
        report["final_task_metrics"] = {
            name: {"mean": float(value.float().mean()), "max": float(value.float().max())}
            for name, value in d_peg_state_metrics(raw).items()
        }
        def serializable(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {name: serializable(child) for name, child in value.items()}
            return value

        report["completed_episode_diagnostics"] = serializable(d_peg_diagnostic_totals(raw))
        report["finite_outputs_verified"] = True
        adapter = getattr(raw, "_rl_ours_taxim_rgb_adapter", None)
        report["fast_taxim_runtime"] = {
            "enabled": args.fast_taxim,
            "adapter_class": type(adapter).__name__ if adapter is not None else None,
            "adapter_module": type(adapter).__module__ if adapter is not None else None,
            "first_contact_rgb_exact": getattr(adapter, "first_contact_rgb_exact", None),
            "checked_real_contact": getattr(adapter, "_checked_real_contact", None),
        }
        if args.fast_taxim and report["fast_taxim_runtime"]["first_contact_rgb_exact"] is not True:
            raise RuntimeError("Fast Taxim requires a verified exact RGB comparison on a real positive-depth batch")
        if args.save_observation_sample:
            # Keep the last returned observation and its cached tactile inputs
            # from exactly the same control step. Never recompute observations.
            sample_path = args.output / "observation_sample.pt"
            torch.save({
                "inputs": {name: value.detach().cpu() for name, value in raw.latest_tactile_inputs.items()},
                "observations": {name: value.detach().cpu() for name, value in observations.items()},
                "tactile_latent": raw.latest_tactile_latent.detach().cpu(),
                "contract": raw.tactile_policy_contract,
                "agent_config": agent.to_dict(),
                "num_actions": raw.action_manager.total_action_dim,
                "common_step_counter": int(raw.common_step_counter),
            }, sample_path)
            report["observation_sample"] = {"path": str(sample_path), "bytes": sample_path.stat().st_size,
                                            "scope": "One complete batch, saved outside timed sections"}
        if device.type == "cuda":
            report["gpu"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    finally:
        if profiler is not None:
            profiler.remove()
        env.close()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_envs", "--num-envs", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--action-file", type=Path)
    parser.add_argument("--action-mode", choices=("zero", "random", "deterministic", "sampled"))
    parser.add_argument("--action-std", type=float, default=0.15)
    parser.add_argument("--socket-usd", type=Path)
    parser.add_argument("--env-yaml", type=Path, help="Restore the saved training environment config before explicit overrides.")
    parser.add_argument("--agent-yaml", type=Path, help="Restore the saved runner config before explicit overrides.")
    parser.add_argument("--encoder-chunk-size", type=int)
    parser.add_argument("--taxim-chunk-size", type=int)
    parser.add_argument("--fast-taxim", action="store_true",
                        help="Enable the task's optional fast Taxim implementation for this diagnostic process.")
    parser.add_argument("--encoder-channels-last", action="store_true",
                        help="Diagnostic-only FP32 CNN memory-format experiment.")
    parser.add_argument("--cudnn-benchmark", action="store_true",
                        help="Diagnostic-only fixed-shape cuDNN autotuning experiment.")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-observation-sample", action="store_true",
                        help="Save one real tactile/observation batch after measurement for encoder/actor comparisons.")
    parser.add_argument("--print-every", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "logs/diagnostics" / f"d_peg_profile_{datetime.now():%Y%m%d_%H%M%S}")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.warmup < 0 or args.print_every <= 0:
        parser.error("num_envs, steps and print-every must be positive; warmup must be nonnegative")
    if args.encoder_chunk_size is not None and args.encoder_chunk_size <= 0:
        parser.error("--encoder-chunk-size must be positive")
    if args.taxim_chunk_size is not None and args.taxim_chunk_size <= 0:
        parser.error("--taxim-chunk-size must be positive")
    args.action_mode = args.action_mode or ("deterministic" if args.checkpoint else "zero")
    if args.action_mode in ("deterministic", "sampled") and not (args.checkpoint or args.action_file):
        parser.error("Policy action modes require --checkpoint or --action-file")
    for name in ("checkpoint", "action_file", "socket_usd", "env_yaml", "agent_yaml"):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser().resolve()
            if not path.is_file():
                parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
            setattr(args, name, path)
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
        "timing_semantics": "CUDA synchronized wall time. Nested inclusive values overlap; exclusive values partition env.step.",
        "scope": "Standalone rollout only, no PPO updates. Identical replay actions do not guarantee bitwise deterministic GPU physics.",
    }
    app, started = None, time.perf_counter()
    try:
        app = AppLauncher(args).app
        run(args, report)
        report["passed"] = True
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "report.json", report)
        print("D_PEG_RUNTIME_REPORT", args.output / "report.json", report["passed"], flush=True)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
