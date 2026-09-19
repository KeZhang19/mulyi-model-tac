#!/usr/bin/env python3
"""Capture actual assembly reset and settling frames without policy actions."""

import argparse
import json
from pathlib import Path
import sys


def capture_bulb_scene(args):
    import gymnasium as gym
    import torch
    from PIL import Image
    from isaaclab_tasks.utils import parse_env_cfg

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/BrainCo_DexHand"))
    import BrainCo_DexHand.tasks  # noqa: F401

    task = "BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = 101
    cfg.bulb_pure_collision = True
    cfg.viewer.resolution = (1280, 960)
    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    raw = env.unwrapped
    env.reset(seed=101)
    origin = raw.scene.env_origins[0].detach().cpu().numpy()
    views = {
        "overview": ((0.45, -1.1, 0.9), (-0.4, 0.0, 0.12)),
        "close": ((-1.06, -0.61, 0.38), (-0.71, -0.055, 0.075)),
        "top": ((-0.72, -0.05, 0.85), (-0.72, -0.05, 0.0)),
    }
    metadata = {"task": task, "seed": 101, "actions": "zero", "frames": []}
    try:
        for label, step_count in (("reset", 0), ("settled", 60)):
            for _ in range(step_count):
                env.step(torch.zeros_like(raw.action_manager.action))
            poses = {
                name: raw.scene[name].data.root_state_w[0].detach().cpu().tolist()
                for name in ("robot", "object", "support", "table")
            }
            for name, (eye, target) in views.items():
                raw.sim.set_camera_view(eye=(origin + eye).tolist(), target=(origin + target).tolist())
                frame = None
                for _ in range(12):
                    frame = env.render()
                if frame is None or frame.size == 0:
                    raise RuntimeError("Renderer returned no frame")
                output = args.out / f"{label}_{name}.png"
                Image.fromarray(frame).save(output)
                metadata["frames"].append({"file": output.name, "control_step": step_count,
                                           "poses": poses, "eye": list(eye), "target": list(target)})
                print(f"CAPTURED {output}", flush=True)
        (args.out / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    launcher = AppLauncher(args)
    try:
        capture_bulb_scene(args)
    finally:
        launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
