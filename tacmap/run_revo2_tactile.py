from __future__ import annotations

"""Run a Revo2 tactile press test with the generated TacMap maps."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parent


parser = argparse.ArgumentParser(description="Revo2 tactile TacMap cylinder press test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--press_info",
    type=str,
    default="assets/test_case/cylinder_D4_left_145_20250904193821_40_60.json",
    help="Press-info JSON. Its presser_name and presser_init_rot are reused.",
)
parser.add_argument("--finger", choices=("middle", "thumb"), default="middle", help="Revo2 touch link to test.")
parser.add_argument("--press-start-offset", type=float, default=0.025, help="Start offset along outward tactile normal.")
parser.add_argument("--press-end-offset", type=float, default=0.018, help="End offset along outward tactile normal.")
parser.add_argument("--press-steps", type=int, default=240, help="Number of policy steps in one press cycle.")
parser.add_argument(
    "--flip-axis",
    choices=("none", "x", "y", "z"),
    default="y",
    help="Apply a 180-degree local object flip after press_info rotation. Use none to keep the raw rotation.",
)
parser.add_argument("--hide-debug-viz", action="store_true", help="Disable USD point/normal debug visualization.")
parser.add_argument("--hide-normals", action="store_true", help="Disable USD normal debug visualization.")
parser.add_argument("--debug-viz-max-points", type=int, default=15000, help="Max tactile surface points to draw.")
parser.add_argument("--max-steps", type=int, default=0, help="Exit after this many env steps. Use 0 to run until closed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

from revo2_tactile_env import Revo2TactilePressEnv  # noqa: E402
from revo2_tactile_env_cfg import Revo2TactilePressEnvCfg  # noqa: E402
from vbts_viz_wrapper import VBTSVizWrapper  # noqa: E402


FINGER_MAPS = {
    "middle": {
        "touch_link": "right_middle_touch_link",
        "points": REPO_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_point.npy",
        "normals": REPO_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_normal.npy",
    },
    "thumb": {
        "touch_link": "right_thumb_touch_link",
        "points": REPO_ROOT / "assets" / "tactilesensor_map" / "right_thumb_touch_point.npy",
        "normals": REPO_ROOT / "assets" / "tactilesensor_map" / "right_thumb_touch_normal.npy",
    },
}


def apply_finger_cfg(env_cfg: Revo2TactilePressEnvCfg, finger: str):
    spec = FINGER_MAPS[finger]
    env_cfg.finger = finger
    env_cfg.touch_link = spec["touch_link"]
    env_cfg.points_npy = str(spec["points"])
    env_cfg.normals_npy = str(spec["normals"])
    env_cfg.contact_sensor[0].prim_path = f"/World/envs/env_.*/Robot/{spec['touch_link']}"
    env_cfg.vbts_sensor[0].prim_path = f"/World/envs/env_.*/Robot/{spec['touch_link']}"
    env_cfg.vbts_sensor[0].points_npy = str(spec["points"])
    env_cfg.vbts_sensor[0].normals_npy = str(spec["normals"])


def apply_press_info(env_cfg: Revo2TactilePressEnvCfg, press_info_path: str | None):
    if press_info_path is None:
        return
    path = Path(press_info_path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Press-info JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        press_info = json.load(f)

    presser_name = press_info.get("presser_name")
    if presser_name:
        usd_path = REPO_ROOT / "assets" / "presser" / f"{presser_name}.usd"
        if not usd_path.exists():
            raise FileNotFoundError(f"Presser USD not found: {usd_path}")
        env_cfg.presser_name = presser_name
        env_cfg.object_cfg.spawn.usd_path = str(usd_path)

    presser_init_rot = press_info.get("presser_init_rot")
    if presser_init_rot is not None:
        env_cfg.object_rot_in_touch_frame = tuple(float(x) for x in presser_init_rot)

    env_cfg.press_info = path.stem
    print(f"[INFO] press_info: {path}")
    print(f"[INFO] presser_name: {env_cfg.presser_name}")
    print(f"[INFO] object_rot_in_touch_frame: {env_cfg.object_rot_in_touch_frame}")


def flip_axis_to_quat(axis: str) -> tuple[float, float, float, float]:
    return {
        "none": (1.0, 0.0, 0.0, 0.0),
        "x": (0.0, 1.0, 0.0, 0.0),
        "y": (0.0, 0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 0.0, 1.0),
    }[axis]


def main():
    env_cfg = Revo2TactilePressEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    apply_finger_cfg(env_cfg, args_cli.finger)
    apply_press_info(env_cfg, args_cli.press_info)
    env_cfg.object_flip_quat_in_touch_frame = flip_axis_to_quat(args_cli.flip_axis)

    env_cfg.press_start_offset = float(args_cli.press_start_offset)
    env_cfg.press_end_offset = float(args_cli.press_end_offset)
    env_cfg.press_steps = int(args_cli.press_steps)
    env_cfg.vbts_sensor[0].debug_viz = not args_cli.hide_debug_viz
    env_cfg.vbts_sensor[0].debug_viz_normals = not args_cli.hide_normals
    env_cfg.vbts_sensor[0].debug_viz_max_points = int(args_cli.debug_viz_max_points)
    print(f"[INFO] object flip axis: {args_cli.flip_axis}, quat: {env_cfg.object_flip_quat_in_touch_frame}")

    env = Revo2TactilePressEnv(env_cfg)
    show_panel = not getattr(args_cli, "headless", False)
    env = VBTSVizWrapper(env, show=show_panel, env_idx=[0])

    obs = env.reset()
    print(f"[INFO] Revo2 tactile press test running on {args_cli.finger}.")
    step_count = 0
    while simulation_app.is_running():
        actions = torch.zeros(env_cfg.scene.num_envs, env_cfg.action_space, device=env.device)
        obs = env.step(actions)
        step_count += 1
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
