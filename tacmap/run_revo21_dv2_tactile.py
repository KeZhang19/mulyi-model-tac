from __future__ import annotations

"""Run a Revo21 DV2 tactile press test with TacMap."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parent
FINGER_CHOICES = ("mid_dip", "middle", "index", "ring", "pinky", "thumb")


parser = argparse.ArgumentParser(description="Revo21 DV2 tactile TacMap press test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--press_info",
    type=str,
    default=None,
    help="Optional press-info JSON to override presser name and initial rotation.",
)
parser.add_argument(
    "--finger",
    choices=FINGER_CHOICES,
    default="mid_dip",
    help="Touch link to test. mid_dip/middle and the other names select Revo21 DIP rubber pads.",
)
parser.add_argument("--press-start-offset", type=float, default=0.025)
parser.add_argument("--press-end-offset", type=float, default=0.018)
parser.add_argument("--press-steps", type=int, default=240)
parser.add_argument(
    "--press-local-offset",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.0, 0.0, 0.0),
    help="Local xyz offset in the touch-link frame, in meters, applied to the whole presser trajectory.",
)
parser.add_argument(
    "--flip-axis",
    choices=("none", "x", "y", "z"),
    default="y",
)
parser.add_argument("--hide-debug-viz", action="store_true")
parser.add_argument("--hide-normals", action="store_true")
parser.add_argument("--debug-viz-max-points", type=int, default=15000)
parser.add_argument("--max-steps", type=int, default=0)
parser.add_argument(
    "--tacmap-mode",
    choices=("linksurface", "surfacenormal"),
    default="linksurface",
    help="TacMap ray mode. linksurface uses a local link-frame grid; surfacenormal uses npy points/normals.",
)
parser.add_argument(
    "--lock-joints",
    action="store_true",
    help="Set very high joint stiffness so finger cannot be deflected by the presser.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402
import numpy as np  # noqa: E402

from isaaclab.sensors.ray_caster import patterns  # noqa: E402
from revo2_tactile_env import Revo2TactilePressEnv  # noqa: E402
from revo21_dv2_tactile_env_cfg import Revo21Dv2TactilePressEnvCfg  # noqa: E402
from tacmap_sensor.sharpa_tacmap_link_surface import SharpaTacmapLinkSurfaceCfg  # noqa: E402
from torch_jit_utils import deform_quantize  # noqa: E402
from vbts_viz_wrapper import VBTSVizWrapper  # noqa: E402


_TACMAP_DIR = REPO_ROOT / "assets" / "tactilesensor_map" / "revo21_dv2"

_MIDDLE_DIP = {
    "touch_link": "right_middip_roll_rubber_link",
    "points": _TACMAP_DIR / "right_middip_roll_rubber_point.npy",
    "normals": _TACMAP_DIR / "right_middip_roll_rubber_normal.npy",
    "link_surface": {
        "ray_axis": "+x",
        "grid_u_axis": "+y",
        "grid_v_axis": "+z",
        "grid_u_size": 0.0202,
        "grid_v_size": 0.0182,
        "grid_center": (0.00159, 0.00103, 0.04757),
    },
}

FINGER_MAPS = {
    "mid_dip": _MIDDLE_DIP,
    "middle": _MIDDLE_DIP,
    "index": {
        "touch_link": "right_indexdip_roll_rubber_link",
        "points": _TACMAP_DIR / "right_indexdip_roll_rubber_point.npy",
        "normals": _TACMAP_DIR / "right_indexdip_roll_rubber_normal.npy",
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
        "points": _TACMAP_DIR / "right_ringdip_roll_rubber_point.npy",
        "normals": _TACMAP_DIR / "right_ringdip_roll_rubber_normal.npy",
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
        "points": _TACMAP_DIR / "right_pinkydip_roll_rubber_point.npy",
        "normals": _TACMAP_DIR / "right_pinkydip_roll_rubber_normal.npy",
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
        "points": _TACMAP_DIR / "right_thumbdip_roll_rubber_point.npy",
        "normals": _TACMAP_DIR / "right_thumbdip_roll_rubber_normal.npy",
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


def apply_finger_cfg(env_cfg: Revo21Dv2TactilePressEnvCfg, finger: str):
    spec = FINGER_MAPS[finger]
    env_cfg.finger = finger
    env_cfg.touch_link = spec["touch_link"]
    env_cfg.points_npy = str(spec["points"])
    env_cfg.normals_npy = str(spec["normals"])
    env_cfg.contact_sensor[0].prim_path = f"/World/envs/env_.*/Robot/{spec['touch_link']}"
    env_cfg.vbts_sensor[0].prim_path = f"/World/envs/env_.*/Robot/{spec['touch_link']}"
    env_cfg.vbts_sensor[0].points_npy = str(spec["points"])
    env_cfg.vbts_sensor[0].normals_npy = str(spec["normals"])
    env_cfg.touch_collision_paths = [f"{spec['touch_link']}/collisions"]
    link_surface = spec["link_surface"]
    env_cfg.tacmap_link_surface_ray_axis = str(link_surface["ray_axis"])
    env_cfg.tacmap_link_surface_grid_u_axis = str(link_surface["grid_u_axis"])
    env_cfg.tacmap_link_surface_grid_v_axis = str(link_surface["grid_v_axis"])
    env_cfg.tacmap_link_surface_grid_u_size = float(link_surface["grid_u_size"])
    env_cfg.tacmap_link_surface_grid_v_size = float(link_surface["grid_v_size"])
    env_cfg.tacmap_link_surface_grid_center = tuple(float(value) for value in link_surface["grid_center"])


def mean_normal_from_npy(normals_npy: str) -> tuple[float, float, float] | None:
    normals = np.load(normals_npy).astype(np.float32).reshape(-1, 3)
    valid = np.isfinite(normals).all(axis=-1) & (np.linalg.norm(normals, axis=-1) > 1.0e-8)
    if not np.any(valid):
        return None
    normals = normals[valid]
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1.0e-12)
    vec = np.mean(normals, axis=0)
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return None
    vec = vec / norm
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def apply_tacmap_mode(env_cfg: Revo21Dv2TactilePressEnvCfg, mode: str):
    if mode == "surfacenormal":
        return

    if mode != "linksurface":
        raise ValueError(f"Unsupported TacMap mode: {mode}")

    base_cfg = env_cfg.vbts_sensor[0]
    ray_direction = env_cfg.tacmap_link_surface_ray_direction
    if ray_direction is None and bool(env_cfg.tacmap_link_surface_use_mean_normal):
        ray_direction = mean_normal_from_npy(base_cfg.normals_npy)

    common_kwargs = dict(
        prim_path=base_cfg.prim_path,
        update_period=base_cfg.update_period,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
        offset=SharpaTacmapLinkSurfaceCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["distance_along_normal", "distance_along_normal_raw"],
        points_npy=base_cfg.points_npy,
        normals_npy=base_cfg.normals_npy,
        resolution_step=base_cfg.resolution_step,
        max_distance=base_cfg.max_distance,
        correction_scale=base_cfg.correction_scale,
        debug_viz=base_cfg.debug_viz,
        debug_viz_max_points=base_cfg.debug_viz_max_points,
        debug_viz_normals=base_cfg.debug_viz_normals,
        image_width=max(1, int(env_cfg.tacmap_link_surface_width)),
        image_height=max(1, int(env_cfg.tacmap_link_surface_height)),
        ray_axis=str(env_cfg.tacmap_link_surface_ray_axis),
        ray_direction=ray_direction,
        grid_u_axis=str(env_cfg.tacmap_link_surface_grid_u_axis),
        grid_v_axis=str(env_cfg.tacmap_link_surface_grid_v_axis),
        grid_u_size=float(env_cfg.tacmap_link_surface_grid_u_size),
        grid_v_size=float(env_cfg.tacmap_link_surface_grid_v_size),
        grid_center=tuple(float(v) for v in env_cfg.tacmap_link_surface_grid_center),
    )
    surface_cfg = SharpaTacmapLinkSurfaceCfg(
        **common_kwargs,
        mesh_prim_paths=[
            SharpaTacmapLinkSurfaceCfg.RaycastTargetCfg(
                prim_expr=base_cfg.prim_path,
                track_mesh_transforms=True,
            )
        ],
        ray_hit_index=2,
        debug_viz_hits=False,
        debug_viz_link_surfaces=bool(env_cfg.tacmap_link_surface_debug_surfaces),
        debug_viz_rays=False,
        debug_viz_ray_length=float(env_cfg.tacmap_link_surface_debug_ray_length),
        debug_viz_ray_width=float(env_cfg.tacmap_link_surface_debug_ray_width),
    )
    object_cfg = SharpaTacmapLinkSurfaceCfg(
        **common_kwargs,
        mesh_prim_paths=base_cfg.mesh_prim_paths,
        ray_hit_index=1,
        debug_viz_hits=bool(env_cfg.tacmap_link_surface_debug_hits),
        debug_viz_hits_external_mask=True,
        debug_viz_rays=bool(env_cfg.tacmap_link_surface_debug_rays),
        debug_viz_ray_length=float(env_cfg.tacmap_link_surface_debug_ray_length),
        debug_viz_ray_width=float(env_cfg.tacmap_link_surface_debug_ray_width),
    )
    env_cfg.vbts_sensor = [surface_cfg, object_cfg]
    env_cfg.env_info["link_surface_ray_direction_resolved"] = ray_direction


class LinkSurfacePenetrationWrapper:
    """Convert integrated-style surface/object ray depths into one TacMap channel."""

    def __init__(self, env, surface_sensor_idx: int = 0, object_sensor_idx: int = 1):
        self._env = env
        self._surface_sensor_idx = int(surface_sensor_idx)
        self._object_sensor_idx = int(object_sensor_idx)
        self._rows = int(getattr(env.cfg, "tacmap_link_surface_height", 240))
        self._cols = int(getattr(env.cfg, "tacmap_link_surface_width", 240))

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        self._replace_tactile_obs(out)
        return out

    def step(self, action):
        out = self._env.step(action)
        self._replace_tactile_obs(out)
        return out

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _raw_depth(self, sensor_idx: int) -> torch.Tensor | None:
        if sensor_idx >= len(self._env._vbts_sensor):
            return None
        data = self._env._vbts_sensor[sensor_idx].data.output.get("distance_along_normal_raw")
        if data is None:
            return None
        return data.reshape(self._env.num_envs, self._rows, self._cols)

    def _penetration_image(self) -> torch.Tensor | None:
        surface = self._raw_depth(self._surface_sensor_idx)
        obj = self._raw_depth(self._object_sensor_idx)
        if surface is None or obj is None:
            return None
        valid = (surface > 0.0) & (obj > 0.0) & (obj <= surface)
        penetration = torch.where(valid, surface - obj, torch.zeros_like(surface))
        return deform_quantize(penetration.clone().reshape(self._env.num_envs, -1, 1)).reshape(
            self._env.num_envs, 1, self._rows, self._cols
        )

    def _replace_tactile_obs(self, out):
        obs = out[0] if isinstance(out, tuple) else out
        if not isinstance(obs, dict):
            return
        image = self._penetration_image()
        if image is not None:
            obs["vbts_deform"] = image


def apply_press_info(env_cfg: Revo21Dv2TactilePressEnvCfg, press_info_path: str | None):
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
    env_cfg = Revo21Dv2TactilePressEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    apply_finger_cfg(env_cfg, args_cli.finger)
    apply_tacmap_mode(env_cfg, args_cli.tacmap_mode)
    apply_press_info(env_cfg, args_cli.press_info)
    env_cfg.object_flip_quat_in_touch_frame = flip_axis_to_quat(args_cli.flip_axis)

    if args_cli.lock_joints:
        env_cfg.robot_cfg.actuators["revo21_hand"].stiffness = 500.0
        env_cfg.robot_cfg.actuators["revo21_hand"].damping = 50.0
        env_cfg.robot_cfg.actuators["revo21_hand"].effort_limit_sim = 100.0
        print("[INFO] --lock-joints: stiffness=500, damping=50, effort_limit=100")

    env_cfg.press_start_offset = float(args_cli.press_start_offset)
    env_cfg.press_end_offset = float(args_cli.press_end_offset)
    env_cfg.press_steps = int(args_cli.press_steps)
    env_cfg.press_local_offset = tuple(float(v) for v in args_cli.press_local_offset)
    if args_cli.tacmap_mode == "linksurface":
        for sensor_cfg in env_cfg.vbts_sensor:
            sensor_cfg.debug_viz = False
            sensor_cfg.debug_viz_normals = not args_cli.hide_normals
            sensor_cfg.debug_viz_max_points = int(args_cli.debug_viz_max_points)
        env_cfg.vbts_sensor[0].debug_viz = not args_cli.hide_debug_viz
        env_cfg.vbts_sensor[0].debug_viz_link_surfaces = (
            bool(env_cfg.tacmap_link_surface_debug_surfaces) and not args_cli.hide_debug_viz
        )
        env_cfg.vbts_sensor[1].debug_viz_hits = (
            bool(env_cfg.tacmap_link_surface_debug_hits) and not args_cli.hide_debug_viz
        )
        env_cfg.vbts_sensor[1].debug_viz_rays = (
            bool(env_cfg.tacmap_link_surface_debug_rays) and not args_cli.hide_debug_viz
        )
    else:
        env_cfg.vbts_sensor[0].debug_viz = not args_cli.hide_debug_viz
        env_cfg.vbts_sensor[0].debug_viz_normals = not args_cli.hide_normals
        env_cfg.vbts_sensor[0].debug_viz_max_points = int(args_cli.debug_viz_max_points)
    print(
        f"[INFO] finger={args_cli.finger} ({FINGER_MAPS[args_cli.finger]['touch_link']}), "
        f"tacmap_mode={args_cli.tacmap_mode}, flip_axis={args_cli.flip_axis}, "
        f"press_local_offset={env_cfg.press_local_offset}"
    )

    env = Revo2TactilePressEnv(env_cfg)
    if args_cli.tacmap_mode == "linksurface":
        env = LinkSurfacePenetrationWrapper(env)
    show_panel = not getattr(args_cli, "headless", False)
    env = VBTSVizWrapper(env, show=show_panel, env_idx=[0])

    obs = env.reset()
    print(f"[INFO] Revo21 DV2 tactile press test running on {args_cli.finger}.")
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
