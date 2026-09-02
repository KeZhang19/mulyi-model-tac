"""
Minimal example: replay a vt-refine trajectory in ALOHA tactile env + visualize tactile grids.

Usage:
    cd ~/repos/IsaacLab
    ./isaaclab.sh -p Isaacsim_tactile_env/reply_with_tactile.py \
      --dataset_npz Isaacsim_tactile_env/data/dataset_train.npz \
      --normalization_pth Isaacsim_tactile_env/data/dataset_normalizer.npz \
      --episode_idx 0 \
      --replay_key joint_states \
      --steps_per_frame 3
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

# --- SimulationApp MUST be created before any omni/isaaclab imports ---
from isaaclab.app import AppLauncher


REPLAY_ROOT = Path(__file__).resolve().parent


# -----------------------------
# Args (minimal)
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("points", "press", "replay"), default="points")
parser.add_argument("--dataset_npz", type=str, default=str(REPLAY_ROOT / "data" / "dataset_train.npz"))
parser.add_argument("--normalization_pth", type=str, default=str(REPLAY_ROOT / "data" / "dataset_normalizer.npz"))
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--replay_key", type=str, default="joint_states", choices=("joint_states", "actions"))
parser.add_argument("--steps_per_frame", type=int, default=3)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--finger", choices=("middle",), default="middle")
parser.add_argument("--press-start-offset", type=float, default=0.025)
parser.add_argument("--press-end-offset", type=float, default=0.018)
parser.add_argument("--press-steps", type=int, default=240)
parser.add_argument("--flip-axis", choices=("none", "x", "y", "z"), default="y")
parser.add_argument("--show-sample-points", action="store_true", help="Show tactile taxel sample points in the Isaac viewport.")
parser.add_argument("--show-sample-axes", action="store_true", help="Show the tactile patch frame axes in the Isaac viewport.")
parser.add_argument("--sample-point-radius", type=float, default=0.0015, help="Radius for tactile sample point markers.")
parser.add_argument("--num-rows", type=int, default=None, help="Override tactile force-map rows.")
parser.add_argument("--num-cols", type=int, default=None, help="Override tactile force-map columns.")
parser.add_argument("--point-distance", type=float, default=None, help="Override spacing between neighboring tactile taxels in meters.")
parser.add_argument("--normal-offset", type=float, default=None, help="Override tactile taxel offset along the patch normal in meters.")
parser.add_argument("--no-local-ui", action="store_true", help="Disable the Isaac/Omni UI tactile force-map panel.")
parser.add_argument("--tactile_scale", type=int, default=8)
parser.add_argument("--tactile-gamma", type=float, default=0.7, help="Visualization gamma. Values below 1 brighten weaker contacts.")
parser.add_argument("--tactile-active-threshold", type=float, default=1.0e-4, help="Force threshold used for active taxel counts.")
parser.add_argument("--print-active-taxels", action="store_true", help="Print active tactile taxel row/column coordinates.")
parser.add_argument("--print-active-every", type=int, default=30, help="Print active taxels every N simulation steps.")
parser.add_argument("--print-active-topk", type=int, default=20, help="Maximum number of active taxels to print per sensor.")
parser.add_argument("--show_cv", action="store_true", help="Show tactile maps in an OpenCV HighGUI window.")
parser.add_argument("--save_force_maps", action="store_true", help="Save tactile map images under output/force_maps.")
parser.add_argument("--save_every", type=int, default=30, help="Save one tactile map every N simulation steps.")
parser.add_argument("--live_web", action="store_true", help="Serve a live tactile map at http://localhost:<live_port>.")
parser.add_argument("--live_port", type=int, default=8090, help="Port for --live_web.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Now safe to import env ---
sys.path.insert(0, str(Path(__file__).parent))
from aloha_tactile_env import AlohaTactileEnv, AlohaTactileEnvCfg  # noqa: E402


SENSOR_LABELS = ["thumb", "index", "middle", "ring", "pinky"]
_LIVE_LOCK = threading.Lock()
_LIVE_IMAGE_RGB: np.ndarray | None = None
_LIVE_STATS = "waiting for tactile frames..."

FINGER_MAPS = {
    "middle": {
        "touch_link": "right_middle_touch_link",
        "points": REPLAY_ROOT.parents[2] / "tacmap" / "assets" / "tactilesensor_map" / "right_middle_touch_point.npy",
        "normals": REPLAY_ROOT.parents[2] / "tacmap" / "assets" / "tactilesensor_map" / "right_middle_touch_normal.npy",
    },
}


# -----------------------------
# Dataset utilities (minimal)
# -----------------------------
def _denorm(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return (x + 1.0) * 0.5 * (x_max - x_min) + x_min


def load_episode(npz_path: str, norm_path: str | None, *, key: str, episode_idx: int) -> np.ndarray:
    npz_path = os.path.expanduser(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    if key not in data:
        raise KeyError(f"Key '{key}' not in dataset. Have: {list(data.keys())}")
    if "traj_lengths" not in data:
        raise KeyError("Dataset missing 'traj_lengths'")

    vals = data[key].astype(np.float32)
    lengths = np.asarray(data["traj_lengths"], dtype=np.int64)
    if not (0 <= episode_idx < lengths.size):
        raise ValueError(f"episode_idx={episode_idx} out of range (num_episodes={lengths.size})")

    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    s = int(starts[episode_idx])
    L = int(lengths[episode_idx])
    ep = vals[s : s + L]

    if norm_path is None:
        return ep

    stats = torch.load(os.path.expanduser(norm_path), map_location="cpu")
    min_key = f"stats.{key}.min"
    max_key = f"stats.{key}.max"
    if min_key not in stats or max_key not in stats:
        raise KeyError(f"Normalization missing {min_key}/{max_key}")

    x_min = stats[min_key].detach().cpu().numpy().astype(np.float32)
    x_max = stats[max_key].detach().cpu().numpy().astype(np.float32)
    return _denorm(ep, x_min, x_max).astype(np.float32)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _rgb_to_bmp_bytes(img_rgb: np.ndarray) -> bytes:
    img = np.asarray(img_rgb, dtype=np.uint8)
    h, w, _ = img.shape
    row_stride = ((w * 3 + 3) // 4) * 4
    padding = row_stride - w * 3
    pixel_rows = []
    for row in img[::-1]:
        pixel_rows.append(row[:, ::-1].tobytes() + b"\x00" * padding)
    pixel_data = b"".join(pixel_rows)
    file_size = 14 + 40 + len(pixel_data)
    header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
    dib = struct.pack("<IIIHHIIIIII", 40, w, h, 1, 24, 0, len(pixel_data), 2835, 2835, 0, 0)
    return header + dib + pixel_data


def start_live_web_server(port: int):
    placeholder = np.zeros((64, 256, 3), dtype=np.uint8)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send(self, content_type: str, data: bytes):
            try:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # The browser may cancel an in-flight image request when refreshing quickly.
                self.close_connection = True

        def do_GET(self):
            if self.path.startswith("/force.bmp"):
                with _LIVE_LOCK:
                    img = placeholder if _LIVE_IMAGE_RGB is None else _LIVE_IMAGE_RGB.copy()
                self._send("image/bmp", _rgb_to_bmp_bytes(img))
                return

            if self.path.startswith("/stats"):
                with _LIVE_LOCK:
                    stats = _LIVE_STATS
                self._send("text/plain; charset=utf-8", stats.encode("utf-8"))
                return

            html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ALOHA Tactile Force Map</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
    main {{ padding: 16px; }}
    img {{ image-rendering: pixelated; max-width: 100%; border: 1px solid #444; }}
    pre {{ font-size: 18px; }}
  </style>
</head>
<body>
  <main>
    <pre id="stats">loading...</pre>
    <img id="force" src="/force.bmp" />
  </main>
  <script>
    async function refresh() {{
      document.getElementById("force").src = "/force.bmp?t=" + Date.now();
      document.getElementById("stats").textContent = await fetch("/stats?t=" + Date.now()).then(r => r.text());
    }}
    setInterval(refresh, 100);
    refresh();
  </script>
</body>
</html>"""
            self._send("text/html; charset=utf-8", html.encode("utf-8"))

    server = ThreadingHTTPServer(("0.0.0.0", int(port)), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[LIVE] Open http://localhost:{port} to view tactile force maps.", flush=True)
    return server


def update_live_web(img_rgb: np.ndarray, stats: str):
    if not args.live_web:
        return
    global _LIVE_IMAGE_RGB, _LIVE_STATS
    with _LIVE_LOCK:
        _LIVE_IMAGE_RGB = np.asarray(img_rgb, dtype=np.uint8).copy()
        _LIVE_STATS = stats


def show_image(name: str, img_rgb: np.ndarray) -> None:
    """Show image with OpenCV HighGUI when available."""
    if not args.show_cv or img_rgb is None:
        return
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    try:
        import cv2
    except ImportError as exc:
        print(f"[WARN] OpenCV is not installed, disabling --show_cv: {exc}", flush=True)
        args.show_cv = False
        return

    try:
        cv2.imshow(name, img_rgb[:, :, ::-1])
        cv2.waitKey(1)
    except Exception as exc:
        print(f"[WARN] OpenCV window display unavailable, disabling --show_cv: {exc}", flush=True)
        args.show_cv = False


def save_image(img_rgb: np.ndarray, step: int) -> None:
    if not args.save_force_maps:
        return
    save_every = max(1, int(args.save_every))
    if step % save_every != 0:
        return

    out_dir = REPLAY_ROOT / "output" / "force_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img_rgb).save(out_dir / f"force_map_{step:06d}.png")
    except ImportError:
        np.save(out_dir / f"force_map_{step:06d}.npy", img_rgb)


def tactile_strip(tactile: np.ndarray, *, scale: int, labels: list[str], gamma: float = 1.0) -> np.ndarray:
    """tactile: (S,H,W) -> RGB strip."""
    imgs = []
    gamma = max(1.0e-6, float(gamma))
    for i in range(tactile.shape[0]):
        grid = tactile[i]
        vmax = float(grid.max())
        norm = (grid / (vmax + 1e-8)) if vmax > 0 else np.zeros_like(grid)
        norm = np.power(np.clip(norm, 0.0, 1.0), gamma)
        rgb = _jet_colormap(norm)
        if scale != 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        imgs.append(rgb)
    return np.concatenate(imgs, axis=1) if imgs else np.zeros((1, 1, 3), dtype=np.uint8)


def tactile_stats(tactile: np.ndarray, labels: list[str], *, threshold: float) -> str:
    parts = []
    threshold = float(threshold)
    for i in range(tactile.shape[0]):
        label = labels[i] if i < len(labels) else f"S{i}"
        grid = np.asarray(tactile[i])
        active = int(np.count_nonzero(grid > threshold))
        total = int(grid.size)
        parts.append(
            f"{label}=max:{float(grid.max()):.4f} active:{active}/{total} mean:{float(grid.mean()):.4f}"
        )
    return ", ".join(parts)


def active_taxel_report(
    tactile: np.ndarray,
    labels: list[str],
    *,
    threshold: float,
    point_distance: float,
    topk: int,
) -> str:
    rows, cols = tactile.shape[-2], tactile.shape[-1]
    u = np.linspace(
        -point_distance * (rows + 1) / 2.0,
        +point_distance * (rows + 1) / 2.0,
        num=rows + 2,
        dtype=np.float64,
    )[1:-1]
    v = np.linspace(
        -point_distance * (cols + 1) / 2.0,
        +point_distance * (cols + 1) / 2.0,
        num=cols + 2,
        dtype=np.float64,
    )[1:-1]

    reports = []
    for sensor_idx in range(tactile.shape[0]):
        label = labels[sensor_idx] if sensor_idx < len(labels) else f"S{sensor_idx}"
        grid = np.asarray(tactile[sensor_idx])
        active = np.argwhere(grid > float(threshold))
        if active.size == 0:
            reports.append(f"{label}: none")
            continue

        values = grid[active[:, 0], active[:, 1]]
        order = np.argsort(values)[::-1][: max(1, int(topk))]
        cells = []
        for idx in order:
            r, c = int(active[idx, 0]), int(active[idx, 1])
            cells.append(
                f"(r={r},c={c},u={u[r] * 1000.0:+.1f}mm,v={v[c] * 1000.0:+.1f}mm,f={float(grid[r, c]):.4f})"
            )
        reports.append(f"{label}: " + " ".join(cells))
    return " | ".join(reports)


class _ForceMapPanel:
    def __init__(self, height: int, width: int, title: str = "Tactile Force Map"):
        import omni.ui as ui

        self._height = int(height)
        self._width = int(width)
        self._win = ui.Window(title, width=self._width + 24, height=self._height + 72)
        with self._win.frame:
            with ui.VStack(spacing=6):
                self._stats = ui.Label("max=0.0000")
                self._provider = ui.ByteImageProvider()
                ui.ImageWithProvider(self._provider, width=self._width, height=self._height)

    def update(self, img_rgb: np.ndarray, stats: str) -> None:
        rgb = np.ascontiguousarray(img_rgb[:, :, :3], dtype=np.uint8)
        alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=-1), dtype=np.uint8)
        self._provider.set_bytes_data(memoryview(rgba.reshape(-1)), (self._width, self._height))
        self._stats.text = stats


class ForceMapUIWrapper:
    def __init__(self, env, *, show: bool, scale: int, labels: list[str], gamma: float, active_threshold: float):
        self._env = env
        self._show = bool(show)
        self._scale = int(scale)
        self._labels = labels
        self._gamma = float(gamma)
        self._active_threshold = float(active_threshold)
        self._panel = None

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        self._update_panel(obs)
        return out

    def step(self, action):
        out = self._env.step(action)
        obs = out[0] if isinstance(out, tuple) else out
        self._update_panel(obs)
        return out

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _update_panel(self, obs: dict) -> None:
        if not self._show or not isinstance(obs, dict) or "tactile" not in obs:
            return
        tactile = obs["tactile"]
        img = tactile_strip(tactile, scale=self._scale, labels=self._labels, gamma=self._gamma)
        if self._panel is None:
            self._panel = _ForceMapPanel(img.shape[0], img.shape[1])
        stats = tactile_stats(tactile, self._labels, threshold=self._active_threshold)
        self._panel.update(img, stats)


def flip_axis_to_quat(axis: str) -> tuple[float, float, float, float]:
    return {
        "none": (1.0, 0.0, 0.0, 0.0),
        "x": (0.0, 1.0, 0.0, 0.0),
        "y": (0.0, 0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 0.0, 1.0),
    }[axis]


def apply_press_cfg(cfg: AlohaTactileEnvCfg, finger: str) -> None:
    spec = FINGER_MAPS[finger]
    cfg.press_touch_link = spec["touch_link"]
    cfg.press_points_npy = str(spec["points"])
    cfg.press_normals_npy = str(spec["normals"])


# -----------------------------
# Main
# -----------------------------
def main():
    # env config: keep minimal + match AppLauncher flags
    cfg = AlohaTactileEnvCfg(
        headless=bool(getattr(args, "headless", True)),
        device=str(getattr(args, "device", "cuda:0")),
        enable_camera=bool(getattr(args, "enable_cameras", False)),
    )
    cfg.enable_press_motion = args.mode == "press"
    cfg.enable_sample_point_view = args.mode == "points"
    if cfg.enable_press_motion or cfg.enable_sample_point_view:
        apply_press_cfg(cfg, args.finger)
    if args.num_rows is not None:
        cfg.num_rows = int(args.num_rows)
    if args.num_cols is not None:
        cfg.num_cols = int(args.num_cols)
    if args.point_distance is not None:
        cfg.point_distance = float(args.point_distance)
    if args.normal_offset is not None:
        cfg.normal_offset = float(args.normal_offset)
    cfg.debug_vis = bool(args.show_sample_points or args.show_sample_axes or cfg.enable_sample_point_view)
    cfg.debug_vis_show_all_taxels = bool(args.show_sample_points or cfg.enable_sample_point_view)
    cfg.debug_vis_show_axes = bool(args.show_sample_axes)
    cfg.debug_vis_point_radius = float(args.sample_point_radius)
    if cfg.enable_press_motion:
        cfg.press_start_offset = float(args.press_start_offset)
        cfg.press_end_offset = float(args.press_end_offset)
        cfg.press_steps = int(args.press_steps)
        cfg.press_object_flip_quat_in_touch_frame = flip_axis_to_quat(args.flip_axis)
        print(
            f"[INFO] press mode: finger={args.finger}, "
            f"offset={cfg.press_start_offset}->{cfg.press_end_offset}, steps={cfg.press_steps}",
            flush=True,
        )
    elif cfg.enable_sample_point_view:
        print(f"[INFO] points mode: finger={args.finger}, press motion disabled", flush=True)

    sensor_labels = [args.finger] if (cfg.enable_press_motion or cfg.enable_sample_point_view) else list(getattr(cfg, "tactile_sensor_labels", SENSOR_LABELS))

    if not cfg.headless:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(list(cfg.camera_eye), list(cfg.camera_target))
        except Exception as e:
            print(f"[WARN] Failed to set viewport camera view: {e}")

    env = AlohaTactileEnv(cfg, simulation_app=simulation_app)
    show_local_ui = (not bool(getattr(args, "headless", True))) and not bool(args.no_local_ui)
    env = ForceMapUIWrapper(
        env,
        show=show_local_ui,
        scale=int(args.tactile_scale),
        labels=sensor_labels,
        gamma=float(args.tactile_gamma),
        active_threshold=float(args.tactile_active_threshold),
    )
    env.reset()
    live_server = start_live_web_server(args.live_port) if args.live_web else None

    traj = None
    ep_len = 1 if args.mode == "points" else int(cfg.press_steps)
    spf = 1
    if args.mode == "replay":
        # normalization auto-guess (optional)
        norm = args.normalization_pth
        if norm is None:
            guess = str(Path(os.path.expanduser(args.dataset_npz)).parent / "normalization.pth")
            if os.path.isfile(guess):
                norm = guess

        traj = load_episode(args.dataset_npz, norm, key=args.replay_key, episode_idx=args.episode_idx)
        ep_len = traj.shape[0]
        spf = max(1, int(args.steps_per_frame))
        print(f"[INFO] episode={args.episode_idx} key={args.replay_key} len={ep_len} shape={traj.shape}")

    total = ep_len * spf
    if args.mode in ("points", "press") and args.max_steps <= 0:
        total = 10**12
    elif args.max_steps > 0:
        total = min(total, int(args.max_steps)) if args.mode == "replay" else int(args.max_steps)

    for step in range(total):
        if not simulation_app.is_running():
            break

        traj_idx = min(step // spf, ep_len - 1)
        if traj is None:
            action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        else:
            action = traj[traj_idx].copy()

        # keep your small tweak (optional)
        if args.mode == "replay" and action.shape[0] >= 16:
            action[14] -= 0.005
            action[15] += 0.005

        obs, *_ = env.step(action)
        tactile = obs["tactile"]  # (S,H,W)
        vis = tactile_strip(
            tactile,
            scale=int(args.tactile_scale),
            labels=sensor_labels,
            gamma=float(args.tactile_gamma),
        )
        stats = (
            f"[step {step:06d}] traj_idx={traj_idx:4d} "
            f"{tactile_stats(tactile, sensor_labels, threshold=float(args.tactile_active_threshold))}"
        )
        update_live_web(vis, stats)
        show_image("Tactile", vis)
        save_image(vis, step)

        if step % 30 == 0:
            print(stats, flush=True)
        if args.print_active_taxels and step % max(1, int(args.print_active_every)) == 0:
            print(
                "[taxels] "
                + active_taxel_report(
                    tactile,
                    sensor_labels,
                    threshold=float(args.tactile_active_threshold),
                    point_distance=float(cfg.point_distance),
                    topk=int(args.print_active_topk),
                ),
                flush=True,
            )

    env.close()
    if live_server is not None:
        live_server.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
