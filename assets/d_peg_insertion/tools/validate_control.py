#!/usr/bin/env python3
"""Measure production env.step control and grasp before launching PPO."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/BrainCo_DexHand"))


def run(args):
    import gymnasium as gym
    import torch
    import BrainCo_DexHand.tasks  # noqa: F401
    from isaaclab.utils.math import quat_error_magnitude
    from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.dexsuite_revo3_env_cfg_insert_d_peg import DexsuiteRevo3InsertDPegEnvCfg

    cfg = DexsuiteRevo3InsertDPegEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.tactile_policy_enabled = args.tactile_policy_enabled
    if args.hand_scale is not None:
        if not 0 < args.hand_scale <= 1:
            raise ValueError("--hand-scale must be in (0, 1]")
        cfg.actions.action.scale = {"Joint[1-7]_R": .1, "right_.*": args.hand_scale}
    cfg.log_dir = str(args.output)
    env = gym.make("BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0", cfg=cfg).unwrapped
    report = {"passed": False, "mode": "production_env_step", "num_envs": args.num_envs,
              "noise_std": args.noise_std, "full_tactile": cfg.tactile_policy_enabled, "seconds": args.seconds,
              "action_scale": cfg.actions.action.scale}
    try:
        obs, _ = env.reset(seed=args.seed)
        peg, robot = env.scene["object"], env.scene["robot"]
        initial = peg.data.root_pose_w.clone()
        max_drift = torch.zeros(env.num_envs, device=env.device)
        max_angle = torch.zeros_like(max_drift)
        min_force = torch.full((env.num_envs, 5), torch.inf, device=env.device)
        held_frames = torch.zeros_like(max_drift)
        completed = torch.zeros_like(max_drift)
        history = []
        steps = round(args.seconds / env.step_dt)
        expected = torch.tensor([cfg._d_peg_pregrasp["robot_joint_targets"][n] for n in robot.joint_names], device=env.device)
        max_target_error = 0.0
        # Encoder construction must not shift the action sequence in an ablation.
        generator = torch.Generator(device=env.device).manual_seed(args.seed + 1000)
        actions = torch.randn((steps, env.num_envs, 28), device=env.device,
                              generator=generator) * args.noise_std
        torch.cuda.synchronize(env.device)
        started = time.monotonic()
        for step in range(steps):
            action = actions[step]
            with torch.inference_mode():
                obs, reward, terminated, truncated, _ = env.step(action)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
            completed += (terminated | truncated).float()
            force = torch.stack([env.scene[f"right_{f}dip_roll_rubber_link_object_s"].data.force_matrix_w.reshape(env.num_envs, -1, 3).norm(dim=-1).amax(-1)
                                 for f in ("thumb", "index", "mid", "ring", "pinky")], -1)
            min_force = torch.minimum(min_force, force)
            held = (force[:, 0] > 1) & (force[:, 1:].amax(-1) > 1)
            held_frames += held.float()
            max_drift = torch.maximum(max_drift, (peg.data.root_pos_w - initial[:, :3]).norm(dim=-1))
            max_angle = torch.maximum(max_angle, quat_error_magnitude(peg.data.root_quat_w, initial[:, 3:]))
            if args.noise_std == 0:
                max_target_error = max(max_target_error, float((robot.data.joint_pos_target - expected).abs().max()))
            if step % 15 == 0 or step == steps - 1:
                history.append({"step": step + 1, "held": held.cpu().tolist(), "force_N": force.cpu().tolist(),
                                "drift_mm": ((peg.data.root_pos_w-initial[:, :3]).norm(dim=-1)*1000).cpu().tolist()})
        per_env = (max_drift < .003) & (max_angle < torch.deg2rad(torch.tensor(5., device=env.device))) & (held_frames == steps) & (completed == 0)
        report.update(passed=bool(per_env.all()) and (args.noise_std != 0 or max_target_error < 1e-6),
                      acceptance="static_grasp_drift_and_continuous_contact; noisy run is diagnostic only",
                      per_env_passed=per_env.cpu().tolist(), max_drift_m=max_drift.cpu().tolist(),
                      max_angle_deg=torch.rad2deg(max_angle).cpu().tolist(), minimum_finger_force_N=min_force.cpu().tolist(),
                      held_fraction=(held_frames/steps).cpu().tolist(), completed_episodes=completed.cpu().tolist(),
                      max_zero_action_target_error_rad=max_target_error, history=history,
                      elapsed_seconds=time.monotonic()-started,
                      cuda_max_memory_allocated_bytes=torch.cuda.max_memory_allocated(env.device),
                      action_contract=env.tactile_policy_contract.get("action_schema"))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("D_PEG_CONTROL_REPORT", json.dumps({k:v for k,v in report.items() if k != "history"}), flush=True)
        if args.noise_std == 0 and not report["passed"]:
            raise RuntimeError("Production zero-residual controller failed the calibrated grasp acceptance")
    finally:
        env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=2.)
    parser.add_argument("--noise-std", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tactile-policy-enabled", action="store_true")
    parser.add_argument("--hand-scale", type=float, help="Override finger scale for a matched-action comparison")
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        run(args)
    except BaseException:
        import traceback
        import os
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()
