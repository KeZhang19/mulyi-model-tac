"""Independent, frozen evaluation for the existing Rotate-Bulb-Custom task.

The environment resets a terminated slot inside ``ManagerBasedRLEnv.step``.
This evaluator hooks ``TerminationManager.compute`` so terminal state and the
independent success calculation are captured before ``_reset_idx`` runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

TASK = "BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0"
FINAL_SEEDS = (31001, 41003, 51007)
PROTOCOL = Path(__file__).resolve().parents[2] / ".autonomy" / "bulb_custom" / "eval_protocol.json"


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


def _world_target(raw):
    from isaaclab.utils.math import combine_frame_transforms
    robot = raw.scene["robot"]
    command = raw.command_manager.get_command("object_pose")
    target, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    return target


class _TerminalCapture:
    """Capture pre-reset state and independent success for every vector slot."""

    def __init__(self, raw, output: Path, quotas: list[int]):
        import torch
        self.raw, self.output, self.quotas = raw, output, quotas
        output.mkdir(parents=True, exist_ok=True)
        self.count = [0] * raw.num_envs
        self.total, self.dt = 0, float(raw.step_dt)
        self.dwell = torch.zeros(raw.num_envs, device=raw.device)
        self.ever_released = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        self.max_turns = torch.zeros(raw.num_envs, device=raw.device)
        self.buffers: list[list[dict]] = [[] for _ in range(raw.num_envs)]
        self.current_actions = torch.zeros((raw.num_envs, raw.action_manager.total_action_dim), device=raw.device)
        self.terminal: list[dict] = []
        self.stream = (output / "episodes.jsonl").open("w", encoding="utf-8")
        self._attachment_attrs = []
        for env_id in range(raw.num_envs):
            prim = raw.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Lamp/Joints/bulb_attachment")
            self._attachment_attrs.append(None if not prim.IsValid() else prim.GetAttribute("physics:jointEnabled"))
        self._original_compute = None

    def set_actions(self, actions):
        self.current_actions = actions.detach().clone()

    def _terms(self):
        manager = self.raw.termination_manager
        result = {name: _as_bool(manager.get_term(name), self.raw.device) for name in manager.active_terms}
        try:
            result["time_out"] = _as_bool(manager.time_outs, self.raw.device)
        except (AttributeError, RuntimeError, TypeError):
            pass
        return result

    def _attachment(self, env_id: int):
        attr = self._attachment_attrs[env_id]
        if attr is None:
            return None
        value = attr.Get()
        return None if value is None else bool(value)

    def _snapshot(self, terms):
        import torch
        from isaaclab.utils.math import subtract_frame_transforms
        from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom import env_cfg as bulb

        robot, obj, support = (self.raw.scene[name] for name in ("robot", "object", "support"))
        rel_pos, rel_quat = subtract_frame_transforms(support.data.root_pos_w, support.data.root_quat_w,
                                                       obj.data.root_pos_w, obj.data.root_quat_w)
        raw_twist = 2.0 * torch.atan2(rel_quat[:, 1], rel_quat[:, 0])
        target = _world_target(self.raw)
        distance = (obj.data.root_pos_w - target).norm(dim=-1)
        released = getattr(self.raw, "_bulb_released", torch.zeros(self.raw.num_envs, dtype=torch.bool,
                                                                     device=self.raw.device)).detach().bool()
        self.ever_released |= released
        failed = torch.zeros(self.raw.num_envs, dtype=torch.bool, device=self.raw.device)
        for name in ("bulb_dropped", "object_out_of_bound", "lost_grasp", "non_finite", "time_out"):
            if name in terms:
                failed |= terms[name]
        valid = released & (distance < 0.03) & ~failed
        self.dwell = torch.where(valid, self.dwell + self.dt, torch.zeros_like(self.dwell))
        independent_success = valid & (self.dwell + 1e-6 >= 1.0)
        command = getattr(self.raw, "_bulb_turn_metrics_command", None)
        if command is not None:
            self.max_turns = torch.maximum(self.max_turns, command.max_turns.detach())
        try:
            contacts = bulb.rotate_bulb_contact_scores(self.raw)
            finger_forces = bulb.rotate_bulb_finger_forces(self.raw)
        except (AttributeError, RuntimeError, TypeError):
            contacts = torch.zeros((self.raw.num_envs, 2), device=self.raw.device)
            finger_forces = torch.zeros((self.raw.num_envs, 5), device=self.raw.device)
        try:
            joint_pos = self.raw.scene["lamp"].data.joint_pos
        except (KeyError, AttributeError, RuntimeError):
            joint_pos = torch.zeros((self.raw.num_envs, 0), device=self.raw.device)
        return {
            "root_pos_w": obj.data.root_pos_w.detach().cpu().tolist(),
            "root_quat_w": obj.data.root_quat_w.detach().cpu().tolist(),
            "root_lin_vel_w": obj.data.root_lin_vel_w.detach().cpu().tolist(),
            "root_ang_vel_w": obj.data.root_ang_vel_w.detach().cpu().tolist(),
            "relative_pos": rel_pos.detach().cpu().tolist(), "relative_quat": rel_quat.detach().cpu().tolist(),
            "raw_twist_rad": raw_twist.detach().cpu().tolist(), "raw_axial_m": rel_pos[:, 0].detach().cpu().tolist(),
            "robot_joint_pos": robot.data.joint_pos.detach().cpu().tolist(),
            "lamp_joint_pos": joint_pos.detach().cpu().tolist(), "contact_scores": contacts.detach().cpu().tolist(),
            "finger_forces": finger_forces.detach().cpu().tolist(),
            "target_pos_w": target.detach().cpu().tolist(), "distance_m": distance.detach().cpu().tolist(),
            "released": released.cpu().tolist(), "dwell_s": self.dwell.detach().cpu().tolist(),
            "independent_success": independent_success.cpu().tolist(),
            "attachment_enabled": [self._attachment(i) for i in range(self.raw.num_envs)],
            "terms": {name: values.cpu().tolist() for name, values in terms.items()},
        }

    def _append_step(self, snapshot, env_id):
        row = {key: value[env_id] for key, value in snapshot.items() if key != "terms"}
        row["actions"] = self.current_actions[env_id].detach().cpu().tolist()
        row["done"] = False
        self.buffers[env_id].append(row)

    def on_compute(self, result):
        """Called immediately after compute and before ManagerBasedRLEnv reset."""
        terms = self._terms()
        snapshot = self._snapshot(terms)
        reset = _as_bool(result, self.raw.device)
        for env_id in range(self.raw.num_envs):
            self._append_step(snapshot, env_id)
        for env_id in reset.nonzero(as_tuple=False).flatten().tolist():
            if self.count[env_id] >= self.quotas[env_id]:
                self.buffers[env_id].clear()
                self.dwell[env_id] = 0
                self.ever_released[env_id] = False
                self.max_turns[env_id] = 0
                continue
            values = snapshot["terms"]
            self.buffers[env_id][-1]["done"] = True
            episode = {
                "episode": self.total, "vector_env_id": env_id, "episode_index_in_env": self.count[env_id],
                "steps": len(self.buffers[env_id]), "task_success": bool(values.get("success", [False] * self.raw.num_envs)[env_id]),
                "independent_success": bool(snapshot["independent_success"][env_id]),
                "success": bool(snapshot["independent_success"][env_id]),
                "failure": not bool(snapshot["independent_success"][env_id]),
                "success_source": "raw_pre_reset_release_distance_dwell", "released": bool(self.ever_released[env_id]),
                "max_unscrew_turns": float(self.max_turns[env_id]),
                "target_distance_final_m": float(snapshot["distance_m"][env_id]),
                "hold_seconds_final": float(snapshot["dwell_s"][env_id]),
                "termination_terms": {name: bool(vals[env_id]) for name, vals in values.items()},
                "final_state": {key: val[env_id] for key, val in snapshot.items() if key != "terms"},
            }
            self.stream.write(json.dumps(episode, ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
            self.terminal.append(episode)
            if episode["independent_success"]:
                self._save_trajectory("success", self.buffers[env_id])
            elif not (self.output / "trajectory_failure.npz").exists():
                self._save_trajectory("failure", self.buffers[env_id])
            # Keep one trace per observed failure cause so the earliest drop
            # does not hide a later timeout after successful unscrewing.
            if not episode["independent_success"]:
                for reason in ("bulb_dropped", "lost_grasp", "object_out_of_bound", "non_finite", "time_out"):
                    if episode["termination_terms"].get(reason, False):
                        self._save_trajectory(reason, self.buffers[env_id])
                if episode["released"]:
                    self._save_trajectory("released_failure", self.buffers[env_id])
            self.count[env_id] += 1
            self.total += 1
            self.buffers[env_id].clear()
            self.dwell[env_id] = 0
            self.ever_released[env_id] = False
            self.max_turns[env_id] = 0

    def _save_trajectory(self, kind, rows):
        import numpy as np
        path = self.output / f"trajectory_{kind}.npz"
        if not rows or path.exists():
            return
        keys = ("actions", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "relative_pos",
                "relative_quat", "raw_twist_rad", "raw_axial_m", "robot_joint_pos", "lamp_joint_pos",
                "contact_scores", "finger_forces", "target_pos_w", "distance_m", "released", "dwell_s", "attachment_enabled", "done")
        np.savez_compressed(path,
                            **{key: np.asarray([row[key] for row in rows]) for key in keys})

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


def _build(args, seed: int, output: Path):
    import gymnasium as gym
    import BrainCo_DexHand  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    import yaml
    from BrainCo_DexHand.tactile_representation.policy import prepare_policy_run
    from bulb_training import configure_bulb_checkpoint, prepare_bulb_policy_run
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.agents.rsl_rl_ppo_cfg import DexsuiteRevo3RotateBulbPPORunnerCfg

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed, cfg.sim.device = seed, args.device
    cfg.bulb_pure_collision = args.physics_mode == "pure"
    cfg.initial_screw_turns_range = (3.0, 3.0)
    cfg.tactile_policy_enabled = False
    cfg.tactile_reconstruction_diagnostics = False
    cfg.log_dir = str(output)
    agent_cfg = DexsuiteRevo3RotateBulbPPORunnerCfg()
    agent_cfg.device = args.device
    configure_bulb_checkpoint(agent_cfg, args.checkpoint, training=False)
    params = Path(args.checkpoint).parent / "params"
    agent_params, env_params = params / "agent.yaml", params / "env.yaml"
    if not agent_params.is_file() or not env_params.is_file():
        raise ValueError(f"checkpoint must carry saved params/agent.yaml and env.yaml: {params}")
    # env.yaml is Isaac Lab's own dump and contains Python tuple/slice tags;
    # it is read only from the checkpoint directory selected by the operator.
    saved_agent, saved_env = yaml.safe_load(agent_params.read_text()), yaml.unsafe_load(env_params.read_text())
    if saved_agent.get("class_name") != "OnPolicyRunner" or saved_agent.get("policy", {}).get("class_name") != "BulbActorCritic":
        raise ValueError("checkpoint saved agent parameters are not the bounded Custom PPO configuration")
    if saved_agent.get("obs_groups", {}).get("policy") != ["policy", "proprio", "perception"]:
        raise ValueError("checkpoint saved observation groups do not match the frozen Custom policy")
    turns = saved_env.get("initial_screw_turns_range")
    if bool(saved_env.get("bulb_pure_collision")) != (args.physics_mode == "pure") or saved_env.get("task_episode_length_s") != 20.0 or tuple(turns or ()) != (3.0, 3.0):
        raise ValueError("checkpoint saved environment parameters do not match the selected 20-second physics mode")
    if saved_env.get("tactile_policy_enabled") is not False:
        raise ValueError("checkpoint saved tactile_policy_enabled must be false for this protocol")
    env = gym.make(args.task, cfg=cfg)
    prepare_policy_run(env, output, resume_path=args.checkpoint)
    prepare_bulb_policy_run(env, output, resume_path=args.checkpoint)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint, map_location=args.device)
    return env, wrapped, runner, runner.get_inference_policy(device=env.unwrapped.device), cfg, {
        "agent_yaml_sha256": _sha256(agent_params), "env_yaml_sha256": _sha256(env_params),
    }


def _evaluate_seed(args, seed: int, output: Path, runtime=None):
    import torch
    base, extra = divmod(args.episodes_per_seed, args.num_envs)
    quotas = [base + int(i < extra) for i in range(args.num_envs)]
    env = wrapped = capture = None
    owns_runtime = runtime is None
    try:
        if runtime is None:
            runtime = _build(args, seed, output)
        env, wrapped, runner, policy, cfg, params_hashes = runtime
        raw = env.unwrapped
        network = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        before_hash = _state_hash(network)
        raw.reset(seed=seed)
        obs = wrapped.get_observations()
        capture = _TerminalCapture(raw, output, quotas)
        capture.install()
        episode_steps = math.ceil(float(cfg.episode_length_s) / float(cfg.sim.dt * cfg.decimation))
        max_steps = max(episode_steps + 1, max(quotas, default=0) * episode_steps * 3)
        steps = 0
        while capture.total < args.episodes_per_seed and steps < max_steps:
            with torch.inference_mode():
                actions = policy(obs)
                capture.set_actions(actions)
                obs, _reward, _dones, _extras = wrapped.step(actions)
            steps += 1
        after_hash = _state_hash(network)
        episodes = capture.terminal
        successes = sum(int(e["success"]) for e in episodes)
        return {"seed": seed, "episodes_requested": args.episodes_per_seed, "episodes_completed": len(episodes),
                "successes": successes, "success_rate": successes / len(episodes) if episodes else 0.0,
                "steps": steps, "max_steps": max_steps, "complete": len(episodes) == args.episodes_per_seed,
                "slot_quotas": quotas, "weight_hash_before": before_hash, "weight_hash_after": after_hash,
                "weights_unchanged": before_hash == after_hash, "saved_params": params_hashes, "episodes": episodes}
    finally:
        if capture is not None:
            capture.close()
        if owns_runtime and wrapped is not None:
            wrapped.close()
        elif owns_runtime and env is not None:
            env.close()


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--num_envs", "--num-envs", type=int, default=32)
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS))
    parser.add_argument("--physics-mode", choices=("guided", "pure"), required=True,
                        help="Physics mode must be stated explicitly; only guided is acceptance mode.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes_per_seed < 1 or not args.seeds:
        parser.error("num_envs, episodes-per-seed and seeds must be positive")
    args.development_run = not (args.episodes_per_seed == 100 and tuple(args.seeds) == FINAL_SEEDS)
    args.checkpoint, args.out = str(Path(args.checkpoint).resolve()), args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = _parse()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_hash = _sha256(PROTOCOL)
    if protocol["task_id"] != args.task:
        raise SystemExit(f"protocol task mismatch: {protocol['task_id']} != {args.task}")
    frozen = protocol["acceptance"]
    if tuple(frozen["seeds"]) != FINAL_SEEDS or frozen["episodes_per_seed"] != 100:
        raise SystemExit("eval_protocol.json acceptance seeds/episode quota no longer match the frozen protocol")
    if protocol["initial_state"]["episode_length_seconds"] != 20.0:
        raise SystemExit("eval_protocol.json episode length is not the frozen 20 seconds")
    if "initial_screw_turns_range [3.0, 3.0]" not in protocol["initial_state"]["lamp"]:
        raise SystemExit("eval_protocol.json does not freeze the three-turn initial state")
    target = protocol["target"]
    if target["position_tolerance_m"] != 0.03 or target["hold_seconds"] != 1.0:
        raise SystemExit("eval_protocol.json target tolerance/hold no longer match the frozen predicate")
    if protocol["physics"]["bulb_pure_collision"] is not False:
        raise SystemExit("eval_protocol.json must freeze guided (bulb_pure_collision=false) acceptance")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    # Keep the launcher alive for the lifetime of the Kit application.  If the
    # temporary AppLauncher is immediately collected, Isaac Sim can close right
    # after startup before the environment is built (especially with another
    # Kit process running on a different GPU).
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    report = {"schema": "rotate_bulb_custom_independent_evaluation", "protocol_version": protocol["protocol_version"],
              "protocol_sha256": protocol_hash, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "task": args.task, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
              "num_envs": args.num_envs, "episodes_per_seed": args.episodes_per_seed, "seeds": args.seeds,
              "development_run": args.development_run, "physics_mode": args.physics_mode,
              "bulb_pure_collision": args.physics_mode == "pure", "initial_screw_turns": 3.0,
              "passed": False, "per_seed": []}
    runtime = None
    try:
        # Isaac Sim owns a process-wide SimulationContext. Build one environment
        # and reset it for each seed instead of destroying/recreating contexts.
        runtime = _build(args, args.seeds[0], args.out / "runtime")
        for seed in args.seeds:
            result = _evaluate_seed(args, seed, args.out / f"seed_{seed}", runtime=runtime)
            report["per_seed"].append(result)
            print(f"[bulb-custom-eval] seed={seed} episodes={result['episodes_completed']}/{args.episodes_per_seed} successes={result['successes']}", flush=True)
        total = sum(r["episodes_completed"] for r in report["per_seed"])
        wins = sum(r["successes"] for r in report["per_seed"])
        report["overall"] = {"episodes_completed": total, "successes": wins, "success_rate": wins / total if total else 0.0,
                              "all_seeds_complete": all(r["complete"] for r in report["per_seed"]),
                              "all_weights_unchanged": all(r["weights_unchanged"] for r in report["per_seed"])}
        protocol_match = (args.physics_mode == "guided" and tuple(args.seeds) == FINAL_SEEDS and args.episodes_per_seed == 100)
        report["protocol_acceptance_match"] = protocol_match
        report["passed"] = (not args.development_run and protocol_match and report["overall"]["all_seeds_complete"]
                            and report["overall"]["all_weights_unchanged"] and report["overall"]["success_rate"] >= 0.30
                            and all(r["success_rate"] >= 0.20 for r in report["per_seed"]))
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    finally:
        if runtime is not None:
            runtime[1].close()
        app.close()
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
