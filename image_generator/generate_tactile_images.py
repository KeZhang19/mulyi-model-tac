"""Generate paired TacMap and TacEx RGB tactile images.

Example:
    /path/to/IsaacLab/isaaclab.sh -p image_generator/generate_tactile_images.py \
        --workers 4 --samples 400 --save-every 5 --headless

Output:
    image_generator/output/tacmap/000000.png
    image_generator/output/rgb/000000.png
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


IMAGE_GENERATOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = IMAGE_GENERATOR_ROOT.parent
INTEGRATE_ROOT = REPO_ROOT / "integrate"
TACMAP_ROOT = REPO_ROOT / "tacmap"
DEFAULT_OUTPUT_DIR = IMAGE_GENERATOR_ROOT / "output"
DEFAULT_TACEX_RGB_ROOT = INTEGRATE_ROOT / "third_party" / "tacex_gpu_taxim"
DEFAULT_TACEX_RGB_CALIB_DIR = DEFAULT_TACEX_RGB_ROOT / "calibs" / "640x480"
DEFAULT_TACEX_RGB_SIM_DIR = DEFAULT_TACEX_RGB_ROOT / "sim"
REVO21_TACMAP_DIR = TACMAP_ROOT / "assets" / "tactilesensor_map" / "revo21_dv2"


_MIDDLE_DIP = {
    "touch_link": "right_middip_roll_rubber_link",
    "points": REVO21_TACMAP_DIR / "right_middip_roll_rubber_point.npy",
    "normals": REVO21_TACMAP_DIR / "right_middip_roll_rubber_normal.npy",
    "collision_paths": ("right_middip_roll_rubber_link/collisions",),
    "link_surface": {
        "ray_axis": "+x",
        "grid_u_axis": "+y",
        "grid_v_axis": "+z",
        "grid_u_size": 0.0202,
        "grid_v_size": 0.0182,
        "grid_center": (0.00159, 0.00103, 0.04757),
    },
}

for path in (INTEGRATE_ROOT, REPO_ROOT / "scripts" / "force_map" / "official_replay", TACMAP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


FINGER_MAPS = {
    "mid_dip": _MIDDLE_DIP,
    "middle": _MIDDLE_DIP,
    "index": {
        "touch_link": "right_indexdip_roll_rubber_link",
        "points": REVO21_TACMAP_DIR / "right_indexdip_roll_rubber_point.npy",
        "normals": REVO21_TACMAP_DIR / "right_indexdip_roll_rubber_normal.npy",
        "collision_paths": ("right_indexdip_roll_rubber_link/collisions",),
        "link_surface": {
            "ray_axis": "+x",
            "grid_u_axis": "+y",
            "grid_v_axis": "+z",
            "grid_u_size": 0.02068358,
            "grid_v_size": 0.01846460,
            "grid_center": (0.00105538, 0.00114480, 0.01765266),
        },
    },
    "ring": {
        "touch_link": "right_ringdip_roll_rubber_link",
        "points": REVO21_TACMAP_DIR / "right_ringdip_roll_rubber_point.npy",
        "normals": REVO21_TACMAP_DIR / "right_ringdip_roll_rubber_normal.npy",
        "collision_paths": ("right_ringdip_roll_rubber_link/collisions",),
        "link_surface": {
            "ray_axis": "+x",
            "grid_u_axis": "+y",
            "grid_v_axis": "+z",
            "grid_u_size": 0.02056884,
            "grid_v_size": 0.01818433,
            "grid_center": (0.00104103, -0.00038099, 0.01768645),
        },
    },
    "pinky": {
        "touch_link": "right_pinkydip_roll_rubber_link",
        "points": REVO21_TACMAP_DIR / "right_pinkydip_roll_rubber_point.npy",
        "normals": REVO21_TACMAP_DIR / "right_pinkydip_roll_rubber_normal.npy",
        "collision_paths": ("right_pinkydip_roll_rubber_link/collisions",),
        "link_surface": {
            "ray_axis": "+x",
            "grid_u_axis": "+y",
            "grid_v_axis": "+z",
            "grid_u_size": 0.02068569,
            "grid_v_size": 0.01843646,
            "grid_center": (0.00100531, -0.00114829, 0.01764967),
        },
    },
    "thumb": {
        "touch_link": "right_thumbdip_roll_rubber_link",
        "points": REVO21_TACMAP_DIR / "right_thumbdip_roll_rubber_point.npy",
        "normals": REVO21_TACMAP_DIR / "right_thumbdip_roll_rubber_normal.npy",
        "collision_paths": ("right_thumbdip_roll_rubber_link/collisions",),
        "link_surface": {
            "ray_axis": "+y",
            "grid_u_axis": "+x",
            "grid_v_axis": "+z",
            "grid_u_size": 0.01343826,
            "grid_v_size": 0.02319049,
            "grid_center": (-0.00072565, 0.02421566, -0.00022462),
        },
    },
}

PRESSER_SPECS = {
    "cylinder_D4": {
        "usd": TACMAP_ROOT / "assets" / "presser" / "cylinder_D4.usd",
        "default_flip_axis": "y",
        "default_extra_rot_axis": "none",
        "default_extra_rot_deg": 0.0,
        "rot": (
            0.7071067811801017,
            -3.019609190115596e-06,
            0.7071067811800986,
            -3.0196091876020132e-06,
        ),
    },
    "square_4": {
        "usd": TACMAP_ROOT / "assets" / "presser" / "square_4.usd",
        "default_flip_axis": "none",
        "default_extra_rot_axis": "y",
        "default_extra_rot_deg": -90.0,
        "rot": (
            1.0,
            1.4563845496008064e-15,
            -1.5916065870379613e-16,
            5.5510727714786566e-17,
        ),
    },
}

SLIDE_AXES = ("+x", "-x", "+y", "-y", "+z", "-z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run one environment with manual motion settings.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel Isaac processes.")
    parser.add_argument("--worker-id", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--index-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=200, help="Total paired images to save.")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after this many sim steps even if fewer images are saved.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--finger", choices=tuple(FINGER_MAPS), default="mid_dip")
    parser.add_argument("--presser", choices=("cylinder_D4", "square_4"), default="cylinder_D4")
    parser.add_argument("--press-start-offset", type=float, default=0.025)
    parser.add_argument("--press-end-offset", type=float, default=0.018)
    parser.add_argument("--press-steps", type=int, default=240)
    parser.add_argument("--press-slide-distance", type=float, default=0.005)
    parser.add_argument("--min-slide-distance", type=float, default=0.0)
    parser.add_argument("--max-slide-distance", type=float, default=None)
    parser.add_argument("--press-slide-steps", type=int, default=50)
    parser.add_argument("--local-offset-range", type=float, default=0.004)
    parser.add_argument("--test-local-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--test-slide-axis", choices=SLIDE_AXES, default="+y")
    parser.add_argument("--test-slide-distance", type=float, default=None)
    parser.add_argument("--tacmap-max-distance", type=float, default=0.015)
    parser.add_argument("--tacmap-gamma", type=float, default=1.0)
    parser.add_argument("--tacmap-view", choices=("raw", "quantized", "mask", "normalized"), default="quantized")
    parser.add_argument("--tacex-rgb-width", type=int, default=320)
    parser.add_argument("--tacex-rgb-height", type=int, default=240)
    parser.add_argument("--tacex-rgb-depth-scale", type=float, default=1.0)
    parser.add_argument("--tacex-rgb-with-shadow", action="store_true")
    parser.add_argument("--tacex-rgb-device", type=str, default=None)
    parser.add_argument("--tacex-rgb-calib-dir", type=str, default=str(DEFAULT_TACEX_RGB_CALIB_DIR))
    parser.add_argument("--tacex-rgb-sim-dir", type=str, default=str(DEFAULT_TACEX_RGB_SIM_DIR))

    # Forward standard IsaacLab launcher arguments when this process is a worker.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.test):
        args.workers = 1
        args.worker_id = 0
    if int(args.worker_id) < 0 and int(args.workers) > 1:
        run_parent(args)
        return
    run_worker(args)


def run_parent(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser()
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "tacmap").mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers))
    base = int(args.samples) // workers
    rem = int(args.samples) % workers
    procs: list[subprocess.Popen] = []
    index_start = 0
    for worker_id in range(workers):
        worker_samples = base + (1 if worker_id < rem else 0)
        if worker_samples <= 0:
            continue
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *parent_to_worker_args(args),
            "--workers",
            "1",
            "--worker-id",
            str(worker_id),
            "--samples",
            str(worker_samples),
            "--index-start",
            str(index_start),
        ]
        print(f"[parent] launch worker {worker_id}: {worker_samples} samples", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=os.environ.copy()))
        index_start += worker_samples

    failed = 0
    for proc in procs:
        failed += int(proc.wait() != 0)
    if failed:
        raise SystemExit(f"{failed} worker process(es) failed")
    print(f"[done] wrote paired images to {out_dir}", flush=True)


def parent_to_worker_args(args: argparse.Namespace) -> list[str]:
    keys = (
        "output_dir",
        "image_size",
        "seed",
        "finger",
        "presser",
        "press_start_offset",
        "press_end_offset",
        "press_steps",
        "press_slide_distance",
        "min_slide_distance",
        "max_slide_distance",
        "press_slide_steps",
        "local_offset_range",
        "test",
        "test_local_offset",
        "test_slide_axis",
        "test_slide_distance",
        "max_steps",
        "tacmap_max_distance",
        "tacmap_gamma",
        "tacmap_view",
        "save_every",
        "tacex_rgb_width",
        "tacex_rgb_height",
        "tacex_rgb_depth_scale",
        "tacex_rgb_device",
        "tacex_rgb_calib_dir",
        "tacex_rgb_sim_dir",
        "device",
        "headless",
    )
    out: list[str] = []
    for key in keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if value is None:
            continue
        if key == "headless":
            if bool(value):
                out.append("--headless")
            continue
        if key == "test":
            if bool(value):
                out.append("--test")
            continue
        if key == "test_local_offset":
            out.append("--test-local-offset")
            out.extend(str(v) for v in value)
            continue
        out.extend([f"--{key.replace('_', '-')}", str(value)])
    if bool(getattr(args, "tacex_rgb_with_shadow", False)):
        out.append("--tacex-rgb-with-shadow")
    return out


def run_worker(args: argparse.Namespace) -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from integrated_tactile_env import IntegratedTactileEnv, IntegratedTactileEnvCfg
    from tacex_rgb_adapter import RevoTacExRgbAdapter, RevoTacExRgbCfg

    worker_id = max(0, int(args.worker_id))
    cfg = IntegratedTactileEnvCfg(
        headless=bool(args.headless),
        device=str(args.device),
        enable_camera=False,
    )
    cfg.enable_press_motion = True
    cfg.enable_sample_point_view = False
    cfg.enable_tacmap = True
    cfg.tacmap_ray_mode = "link_surface"
    cfg.tacmap_max_distance = float(args.tacmap_max_distance)
    cfg.tacmap_link_surface_width = 240
    cfg.tacmap_link_surface_height = 240
    cfg.tacmap_link_surface_ray_axis = "+x"
    cfg.tacmap_link_surface_use_mean_normal = True
    cfg.tacmap_link_surface_grid_u_axis = "+y"
    cfg.tacmap_link_surface_grid_v_axis = "+z"
    cfg.tacmap_link_surface_grid_u_size = 0.0202
    cfg.tacmap_link_surface_grid_v_size = 0.0182
    cfg.tacmap_link_surface_grid_center = (0.00159, 0.00103, 0.04757)
    cfg.press_start_offset = float(args.press_start_offset)
    cfg.press_end_offset = float(args.press_end_offset)
    cfg.press_steps = int(args.press_steps)
    cfg.press_slide_distance = (
        float(args.test_slide_distance)
        if bool(args.test) and args.test_slide_distance is not None
        else float(args.press_slide_distance)
    )
    cfg.press_slide_steps = int(args.press_slide_steps)

    if bool(args.test):
        offset = tuple(float(v) for v in args.test_local_offset)
        slide_axis = str(args.test_slide_axis)
    else:
        offset, slide_axis = worker_motion(worker_id, int(args.seed), float(args.local_offset_range))
    rng = np.random.default_rng(int(args.seed) + worker_id * 9973)
    cfg.press_local_offset = offset
    cfg.press_slide_axis_l = axis_to_vector(slide_axis)
    apply_finger_cfg(cfg, str(args.finger))
    apply_presser_cfg(cfg, str(args.presser))
    cfg.press_object_flip_quat_in_touch_frame = flip_axis_to_quat(
        str(PRESSER_SPECS[str(args.presser)]["default_flip_axis"])
    )

    out_dir = Path(args.output_dir).expanduser()
    rgb_dir = out_dir / "rgb"
    tacmap_dir = out_dir / "tacmap"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    tacmap_dir.mkdir(parents=True, exist_ok=True)

    env = IntegratedTactileEnv(cfg, simulation_app=simulation_app)
    obs, _ = env.reset()
    rgb_adapter = RevoTacExRgbAdapter(
        RevoTacExRgbCfg(
            width=int(args.tacex_rgb_width),
            height=int(args.tacex_rgb_height),
            depth_scale=float(args.tacex_rgb_depth_scale),
            with_shadow=bool(args.tacex_rgb_with_shadow),
            device=str(args.tacex_rgb_device or args.device),
            calib_dir=str(args.tacex_rgb_calib_dir),
            taxim_sim_dir=str(args.tacex_rgb_sim_dir),
        )
    )

    total_to_save = int(args.samples)
    saved = 0
    skipped = 0
    step = 0
    global_start = int(getattr(args, "index_start", 0))
    max_steps = int(args.max_steps)
    if max_steps <= 0:
        max_steps = max(
            int(args.press_steps) + int(args.press_slide_steps) + 10,
            total_to_save * max(1, int(args.save_every)) * 20,
        )
    print(
        f"[worker {worker_id}] offset={offset} slide_axis={slide_axis} "
        f"slide_distance={cfg.press_slide_distance:g} target={total_to_save}",
        flush=True,
    )
    while saved < total_to_save and step < max_steps and simulation_app.is_running():
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        if not bool(args.test) and step % max(1, int(args.save_every)) == 0:
            offset, slide_axis, slide_distance = randomize_motion(env, cfg, rng, args)
        obs, *_ = env.step(action)
        if step % max(1, int(args.save_every)) == 0:
            tacmap_raw = obs.get("tacmap_raw")
            if tacmap_raw is not None:
                index = global_start + saved
                tacmap_img = resize_rgb(tacmap_to_rgb(
                    obs["tacmap"],
                    tacmap_raw,
                    str(args.tacmap_view),
                    float(args.tacmap_max_distance),
                    float(args.tacmap_gamma),
                ), int(args.image_size))
                tactile_rgb = resize_rgb(rgb_adapter.step(tacmap_raw).tactile_rgb[0], int(args.image_size))
                if save_pair(tacmap_img, tactile_rgb, tacmap_dir / f"{index:06d}.png", rgb_dir / f"{index:06d}.png"):
                    saved += 1
                else:
                    skipped += 1
                    if bool(args.test):
                        print(f"[worker {worker_id}] skipped blank image at step {step}", flush=True)
        step += 1

    env.close()
    simulation_app.close()
    print(f"[worker {worker_id}] saved {saved}/{total_to_save}, skipped={skipped}, steps={step}", flush=True)


def worker_motion(worker_id: int, seed: int, offset_range: float) -> tuple[tuple[float, float, float], str]:
    rng = np.random.default_rng(int(seed) + worker_id * 9973)
    offset = tuple(float(v) for v in rng.uniform(-offset_range, offset_range, size=3))
    offset = (offset[0], offset[1], 0.0)
    slide_axis = SLIDE_AXES[worker_id % len(SLIDE_AXES)]
    return offset, slide_axis


def randomize_motion(env, cfg, rng: np.random.Generator, args: argparse.Namespace) -> tuple[tuple[float, float, float], str, float]:
    import torch

    offset_range = float(args.local_offset_range)
    offset_xy = rng.uniform(-offset_range, offset_range, size=2)
    offset = (float(offset_xy[0]), float(offset_xy[1]), 0.0)
    slide_axis = str(rng.choice(SLIDE_AXES))
    max_distance = float(args.max_slide_distance) if args.max_slide_distance is not None else float(args.press_slide_distance)
    min_distance = min(float(args.min_slide_distance), max_distance)
    slide_distance = float(rng.uniform(min_distance, max_distance))

    cfg.press_local_offset = offset
    cfg.press_slide_axis_l = axis_to_vector(slide_axis)
    cfg.press_slide_distance = slide_distance

    device = getattr(env, "_device", str(args.device))
    env._press_local_offset_l = torch.tensor(offset, dtype=torch.float32, device=device)
    axis_t = torch.tensor(cfg.press_slide_axis_l, dtype=torch.float32, device=device)
    env._press_slide_axis_l = axis_t / torch.linalg.norm(axis_t).clamp_min(1.0e-8)
    return offset, slide_axis, slide_distance


def apply_finger_cfg(cfg, finger: str) -> None:
    spec = FINGER_MAPS[finger]
    cfg.press_touch_link = spec["touch_link"]
    cfg.press_points_npy = str(spec["points"])
    cfg.press_normals_npy = str(spec["normals"])
    cfg.touch_collision_paths = tuple(spec["collision_paths"])
    cfg.tactile_link_keywords = (str(spec["touch_link"]),)
    link_surface = spec["link_surface"]
    cfg.tacmap_link_surface_ray_axis = str(link_surface["ray_axis"])
    cfg.tacmap_link_surface_grid_u_axis = str(link_surface["grid_u_axis"])
    cfg.tacmap_link_surface_grid_v_axis = str(link_surface["grid_v_axis"])
    cfg.tacmap_link_surface_grid_u_size = float(link_surface["grid_u_size"])
    cfg.tacmap_link_surface_grid_v_size = float(link_surface["grid_v_size"])
    cfg.tacmap_link_surface_grid_center = tuple(float(value) for value in link_surface["grid_center"])


def apply_presser_cfg(cfg, presser: str) -> None:
    spec = PRESSER_SPECS[presser]
    usd_path = Path(spec["usd"])
    if not usd_path.is_file():
        raise FileNotFoundError(f"Presser USD not found: {usd_path}")
    cfg.press_object_usd_path = str(usd_path)
    extra_rot_axis = str(spec["default_extra_rot_axis"])
    extra_rot_deg = float(spec["default_extra_rot_deg"])
    cfg.press_object_rot_in_touch_frame = quat_mul(
        tuple(float(v) for v in spec["rot"]),
        axis_angle_to_quat(extra_rot_axis, extra_rot_deg),
    )


def tacmap_to_rgb(
    tacmap: np.ndarray,
    tacmap_raw: np.ndarray | None,
    view: str,
    display_max_m: float,
    gamma: float,
) -> np.ndarray:
    if view == "raw" and tacmap_raw is not None and len(tacmap_raw) > 0:
        grid = np.nan_to_num(np.asarray(tacmap_raw[0], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.clip(grid / max(1.0e-8, float(display_max_m)), 0.0, 1.0)
        gray = (np.power(norm, max(1.0e-6, float(gamma))) * 255.0).astype(np.uint8)
    elif view == "mask" and tacmap_raw is not None and len(tacmap_raw) > 0:
        grid = np.nan_to_num(np.asarray(tacmap_raw[0], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        gray = (grid > 0.0).astype(np.uint8) * 255
    elif view == "normalized":
        grid = np.nan_to_num(np.asarray(tacmap[0], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        vmax = float(grid.max())
        norm = grid / (vmax + 1.0e-8) if vmax > 0.0 else np.zeros_like(grid)
        gray = (np.power(np.clip(norm, 0.0, 1.0), max(1.0e-6, float(gamma))) * 255.0).astype(np.uint8)
    else:
        grid = np.nan_to_num(np.asarray(tacmap[0], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        gray = np.clip(grid, 0.0, 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    img = np.asarray(img, dtype=np.uint8)
    if int(size) <= 0 or img.shape[:2] == (int(size), int(size)):
        return img
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((int(size), int(size)), Image.BILINEAR), dtype=np.uint8)
    except ImportError:
        y_idx = np.linspace(0, img.shape[0] - 1, int(size)).round().astype(np.int64)
        x_idx = np.linspace(0, img.shape[1] - 1, int(size)).round().astype(np.int64)
        return img[y_idx][:, x_idx]


def save_image(img: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image

        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)
    except ImportError:
        np.save(path.with_suffix(".npy"), np.asarray(img, dtype=np.uint8))


def save_pair(tacmap_img: np.ndarray, rgb_img: np.ndarray, tacmap_path: Path, rgb_path: Path) -> bool:
    if is_black_or_white(tacmap_img) or is_black_or_white(rgb_img):
        return False
    save_image(tacmap_img, tacmap_path)
    save_image(rgb_img, rgb_path)
    return True


def is_black_or_white(img: np.ndarray) -> bool:
    img = np.asarray(img, dtype=np.uint8)
    return bool(np.all(img <= 1) or np.all(img >= 254))


def flip_axis_to_quat(axis: str) -> tuple[float, float, float, float]:
    return {
        "none": (1.0, 0.0, 0.0, 0.0),
        "x": (0.0, 1.0, 0.0, 0.0),
        "y": (0.0, 0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 0.0, 1.0),
    }[axis]


def axis_angle_to_quat(axis: str, degrees: float) -> tuple[float, float, float, float]:
    if axis == "none" or abs(float(degrees)) <= 1.0e-12:
        return (1.0, 0.0, 0.0, 0.0)
    half = math.radians(float(degrees)) * 0.5
    c = math.cos(half)
    s = math.sin(half)
    return {
        "x": (c, s, 0.0, 0.0),
        "y": (c, 0.0, s, 0.0),
        "z": (c, 0.0, 0.0, s),
    }[axis]


def axis_to_vector(axis: str) -> tuple[float, float, float]:
    sign = -1.0 if axis.startswith("-") else 1.0
    return {
        "x": (sign, 0.0, 0.0),
        "y": (0.0, sign, 0.0),
        "z": (0.0, 0.0, sign),
    }[axis[-1]]


def quat_mul(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


if __name__ == "__main__":
    main()
