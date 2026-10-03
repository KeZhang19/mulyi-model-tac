#!/usr/bin/env python3
"""Finite native floor-reset, physical-step and throughput probe (no learning)."""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def probe(args, report):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_apply
    from pxr import UsdPhysics
    sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))
    import BrainCo_DexHand.tasks  # noqa: F401
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.RotateBulbCustom.assembly_lift import bulb_table_clearance
    task = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed, cfg.log_dir = 42, str(args.out)
    cfg.bulb_curriculum.fixed_bulb = False
    if hasattr(cfg.bulb_curriculum, "physics_hz"):
        cfg.bulb_curriculum.physics_hz = args.physics_hz
    else:
        cfg.sim.dt, cfg.decimation = 1 / args.physics_hz, args.physics_hz // 30
        cfg.sim.render_interval = cfg.decimation
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.tactile_policy_enabled = False
    start = time.perf_counter()
    env = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.snapshot else None).unwrapped
    report["construction_s"] = time.perf_counter() - start
    checks = {}
    try:
        obs, _ = env.reset()
        if args.snapshot:
            import imageio.v3 as iio
            for _ in range(3):
                frame = env.render()
            iio.imwrite(args.out / "initial_scene.png", frame)
        runtime = env._bulb_curriculum
        bulb, robot = env.scene["object"], env.scene["robot"]
        initial = bulb.data.root_state_w.clone()
        axis = quat_apply(initial[:, 3:7], initial.new_tensor([1., 0., 0.]).expand(args.num_envs, -1))
        checks["horizontal_axis"] = bool((axis[:, 2].abs() < 1e-5).all()) if not args.legacy else True
        checks["clearance_2mm"] = bool(torch.allclose(bulb_table_clearance(env, runtime.vertices),
                                                     initial.new_full((args.num_envs,), .002), atol=1e-5))
        path = bulb.root_physx_view.prim_paths[0]
        checks["free_rigid_body"] = not bool(UsdPhysics.RigidBodyAPI(
            env.scene.stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Get())
        checks["self_collision_enabled"] = cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True
        checks["encoder_absent"] = env._rotate_bulb_tactile_term.encoder is None
        checks["zero_tactile_columns"] = env._rotate_bulb_tactile_term._disabled_obs.shape[-1] == 0
        if not args.legacy:
            checks["no_pressure_sensors"] = not any("warpsdf" in name for name in env.scene.sensors)
            checks["369_state_observations"] = sum(obs[k].shape[-1] for k in ("policy", "proprio", "perception")) == 369
        checks["control_30hz"] = abs(env.step_dt - 1/30) < 1e-9
        actual_steps = [0]
        original_step = env.sim.step
        def counted(*a, **kw):
            actual_steps[0] += 1
            return original_step(*a, **kw)
        env.sim.step = counted
        untouched = robot.data.joint_pos[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        env.scene.write_data_to_sim()
        env.sim.forward()
        checks["reset_no_hidden_physics"] = actual_steps[0] == 0
        checks["partial_reset_independent"] = torch.equal(untouched, robot.data.joint_pos[1:])
        action = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device=env.device)
        durations = []
        free, total = torch.cuda.mem_get_info()
        peak_device_used = total - free
        finite = True
        for step in range(args.warmup + args.steps):
            # Same bounded action distribution for each physics candidate.
            if args.random_actions:
                action.uniform_(-.15, .15)
            torch.cuda.synchronize()
            start = time.perf_counter()
            obs, reward, terminated, truncated, _ = env.step(action)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= args.warmup:
                durations.append(elapsed)
            if step % 30 == 0:
                free, total = torch.cuda.mem_get_info()
                peak_device_used = max(peak_device_used, total - free)
            finite &= bool(torch.isfinite(reward).all() and all(torch.isfinite(x).all() for x in obs.values()))
        checks["finite_rollout"] = finite
        checks["exact_physics_step_count"] = actual_steps[0] == (args.warmup + args.steps) * cfg.decimation
        checks["fallen_not_supported_by_fixture"] = bool((initial[:, 2] - bulb.data.root_pos_w[:, 2]).max() > .0001)
        report.update(checks=checks, passed=all(checks.values()), num_envs=args.num_envs,
                      physics_hz=args.physics_hz, decimation=cfg.decimation,
                      step_dt=env.step_dt, actual_physics_steps=actual_steps[0],
                      mean_control_step_s=sum(durations)/len(durations),
                      transitions_per_s=args.num_envs*len(durations)/sum(durations),
                      max_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      max_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                      peak_device_used_bytes=peak_device_used,
                      initial_axis=axis[0].cpu().tolist(), initial_pose=initial[0, :7].cpu().tolist(),
                      final_clearance_min_m=float(bulb_table_clearance(env, runtime.vertices).min()),
                      observation_shapes={k: list(v.shape) for k, v in obs.items()},
                      sensors=list(env.scene.sensors), episodes=runtime.episodes,
                      collision_stack_bytes=cfg.sim.physx.gpu_collision_stack_size,
                      stage_successes=sum(r["success"] for r in runtime.finished_records),
                      learned_success_validated=False)
    finally:
        env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--physics-hz", type=int, choices=(120, 240), default=120)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.snapshot:
        args.enable_cameras = True
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"passed": False}
    launcher = AppLauncher(args)
    try:
        probe(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    # Finite tensors do not establish valid physics: PhysX can drop contacts
    # after a buffer overflow and keep stepping. Audit the native Kit log too.
    import carb.settings
    log_file = carb.settings.get_settings().get("/log/file")
    report["native_log_file"] = log_file
    if not log_file or not Path(log_file).is_file():
        report["native_log_error"] = "Native Kit log unavailable; physics integrity unverified"
        report["passed"] = False
    else:
        lines = Path(log_file).read_text(errors="replace").splitlines()
        errors = [line for line in lines if re.search(
            r"PhysX error|buffer overflow|contacts have been dropped|CUDA error|out of memory", line, re.I)]
        report["physics_log_errors"] = errors
        report.setdefault("checks", {})["no_physics_log_errors"] = not errors
        report["passed"] = report["passed"] and not errors
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(0 if report["passed"] else 1)
