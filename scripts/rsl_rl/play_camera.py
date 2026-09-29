"""Shared spectator camera for policy playback and checkpoint-free scene previews.

The camera is a single viewport render product, not a policy observation sensor.
Coordinates are in the selected environment, so batched environments work too.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# dextac-Depth-main: utils/v5_camera_pose.py, default_v2_test_primary_camera_pose_path
# for the ordinary v2-test dual-camera distillation train/PLAY tasks, without CLI
# extrinsics. Snapshot of Deploy/config_new_camera/
# camera_pose_sim_depth_from_training_rgb_candidate.yaml (T_S_D, ROS optical).
# Keep the values here so server playback does not depend on that checkout.
DEXTAC_EYE = (-1.0360810136455936, 0.0173613225619778, 0.5959323083288192)
DEXTAC_ROS_QUAT = (0.2794479451413978, -0.6307074363713034, 0.6575970433847331, -0.3027921798756468)
# ROS -> OpenGL: local 180 degree X rotation; quaternions are wxyz.
DEXTAC_OPENGL_QUAT = (-DEXTAC_ROS_QUAT[1], DEXTAC_ROS_QUAT[0], DEXTAC_ROS_QUAT[3], -DEXTAC_ROS_QUAT[2])
DEXTAC_FX = 426.0688171386719
DEXTAC_WIDTH = 848
# Earlier single-camera depth distillation: FlexivDepthCameraMixinCfg inherits
# these from dexsuite_flexiv_camera_env_cfg.py (already OpenGL, robot frame).
EARLY_EYE = (-1.38333, 0.04515, 0.72006)
EARLY_OPENGL_QUAT = (0.57009, 0.34556, -0.41395, -0.61987)
# _look_at_opengl_quat(EARLY_EYE, (0.4, 0.0, 0.1)) in the V5/V2 depth task.
V5_OPENGL_QUAT = (-0.5689242844014236, -0.40456187613925004, 0.4149341340930568, 0.5835105066384525)


def add_camera_args(parser):
    group = parser.add_argument_group("play camera")
    group.add_argument("--camera", choices=("auto", "dextac", "dextac-v5", "dextac-dual", "task"), default="auto",
                       help="auto/dextac: early single camera; dextac-v5: V5 single camera; dextac-dual: current dual primary; task: original viewer.")
    group.add_argument("--camera-eye", type=float, nargs=3, metavar=("X", "Y", "Z"),
                       help="Camera position in the selected environment (metres).")
    group.add_argument("--camera-lookat", type=float, nargs=3, metavar=("X", "Y", "Z"),
                       help="Camera target in the selected environment (metres).")
    group.add_argument("--camera-env", type=int, default=0, help="Environment index to record.")
    group.add_argument("--video-resolution", type=int, nargs=2, default=(1280, 960), metavar=("WIDTH", "HEIGHT"))
    group.add_argument("--screenshot", type=Path, help="Save a PNG of the initial scene and camera JSON alongside it.")
    group.add_argument("--camera-preview", action="store_true",
                       help="Render the reset scene and exit; no policy/checkpoint required.")
    group.add_argument("--video-dir", "--video_dir", type=Path, help="Override the video output directory.")


def validate_camera_args(parser, args):
    if args.video_length < 1:
        parser.error("--video_length must be positive")
    if min(args.video_resolution) < 1 or any(n % 2 for n in args.video_resolution):
        parser.error("--video-resolution requires positive even dimensions (H.264)")
    if args.camera_env < 0:
        parser.error("--camera-env must be nonnegative")
    for value in (args.camera_eye, args.camera_lookat):
        if value is not None and not all(math.isfinite(x) for x in value):
            parser.error("Camera coordinates must be finite")
    if args.camera_preview and args.video:
        parser.error("--camera-preview saves a reset screenshot; use --video with a checkpoint for policy playback")
    if args.screenshot is not None and args.screenshot.suffix.lower() != ".png":
        parser.error("--screenshot must end in .png")


def _multiply_quat(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j, w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def _rotate(q, xyz):
    q = tuple(x / math.sqrt(sum(v*v for v in q)) for x in q)
    return _multiply_quat(_multiply_quat(q, (0.0, *xyz)), (q[0], -q[1], -q[2], -q[3]))[1:]


def configure_play_camera(cfg, task, args):
    """Set the viewer before gym.make; return optional exact OpenGL orientation."""
    if args.camera_env >= cfg.scene.num_envs:
        raise ValueError(f"--camera-env {args.camera_env} is outside {cfg.scene.num_envs} environments")
    task = task.split(":")[-1]
    arm = task.endswith(("Rotate-Bulb-Custom-v0", "Bulb-Assembly-Custom-v0", "Insert-D-Peg-Custom-v0"))
    hand = task.startswith("BrainCo-Direct-Revo3-")
    rotation = None
    if args.camera.startswith("dextac") or (args.camera == "auto" and (arm or hand)):
        reference_eye = DEXTAC_EYE if args.camera == "dextac-dual" else EARLY_EYE
        rotation = DEXTAC_OPENGL_QUAT if args.camera == "dextac-dual" else EARLY_OPENGL_QUAT
        if args.camera == "dextac-v5":
            rotation = V5_OPENGL_QUAT
        if hand:
            # Same viewing direction, centred on the standalone hand/cube.
            target = cfg.object_cfg.init_state.pos
            forward = _rotate(rotation, (0.0, 0.0, -1.0))
            eye = tuple(t - 0.45*f for t, f in zip(target, forward))
        else:
            robot = getattr(cfg.scene, "robot", None)
            base_pos = robot.init_state.pos if robot is not None else (0.0, 0.0, 0.0)
            base_rot = robot.init_state.rot if robot is not None else (1.0, 0.0, 0.0, 0.0)
            eye = tuple(p + x for p, x in zip(base_pos, _rotate(base_rot, reference_eye)))
            rotation = _multiply_quat(base_rot, rotation)
            forward = _rotate(rotation, (0.0, 0.0, -1.0))
            target = tuple(p + f for p, f in zip(eye, forward))
        cfg.viewer.eye = eye
        cfg.viewer.lookat = target
        cfg.viewer.origin_type = "env"
    if args.camera_eye is not None or args.camera_lookat is not None:
        cfg.viewer.eye = tuple(args.camera_eye or cfg.viewer.eye)
        cfg.viewer.lookat = tuple(args.camera_lookat or cfg.viewer.lookat)
        cfg.viewer.origin_type = "env"
        rotation = None  # Isaac's look-at controller handles user-specified poses.
    if math.dist(cfg.viewer.eye, cfg.viewer.lookat) < 1e-6:
        raise ValueError("Camera position and target must be different")
    cfg.viewer.env_index = args.camera_env
    cfg.viewer.resolution = tuple(args.video_resolution)
    return rotation


def apply_play_camera(env, rotation, *, camera="auto"):
    """Preserve reference roll and field of view on the playback viewer camera."""
    if rotation is None:
        return
    reference_aperture = 13.0 * DEXTAC_WIDTH / DEXTAC_FX if camera in ("dextac-v5", "dextac-dual") else 20.955
    rotation = tuple(x / math.sqrt(sum(v*v for v in rotation)) for x in rotation)
    from pxr import Gf, UsdGeom

    raw = env.unwrapped
    viewer = raw.cfg.viewer
    origin = raw.scene.env_origins[viewer.env_index].detach().cpu().tolist()
    eye = tuple(p + x for p, x in zip(origin, viewer.eye))
    # Own a camera prim: Kit continually maintains the default viewport camera's
    # rotateXYZ ops, so replacing those ops races with its camera controller.
    camera_path = "/World/PlayCamera"
    camera = UsdGeom.Camera.Define(raw.sim.stage, camera_path)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*eye))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(rotation[0], Gf.Vec3d(*rotation[1:])))
    focal = 13.0
    aperture = reference_aperture
    width, height = viewer.resolution
    camera.GetFocalLengthAttr().Set(focal)
    camera.GetHorizontalApertureAttr().Set(aperture)
    camera.GetVerticalApertureAttr().Set(aperture * height / width)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    viewer.cam_prim_path = camera_path
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is not None:
        viewport.set_active_camera(camera_path)


def save_camera_preview(env, path, *, task, checkpoint=None, camera_preset="auto"):
    """Warm up RTX without advancing physics, then save a real RGB frame."""
    from PIL import Image
    from pxr import UsdGeom

    raw = env.unwrapped
    frame = None
    for _ in range(24):
        raw.sim.render()
        frame = raw.render()
    if frame is None or frame.size == 0 or frame.max() == 0:
        raise RuntimeError("Camera returned no RGB image after RTX warm-up")
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)
    camera = UsdGeom.Camera(raw.sim.stage.GetPrimAtPath(raw.cfg.viewer.cam_prim_path))
    matrix = UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(0)
    viewer = raw.cfg.viewer
    metadata = {
        "task": task, "checkpoint": checkpoint, "seed": raw.cfg.seed,
        "camera_preset": camera_preset,
        "scene": "reset, before policy actions", "resolution": list(viewer.resolution),
        "env_index": viewer.env_index, "origin_type": viewer.origin_type,
        "eye": list(viewer.eye), "lookat": list(viewer.lookat),
        "camera_to_world_row_matrix": [list(row) for row in matrix],
        "focal_length": camera.GetFocalLengthAttr().Get(),
        "horizontal_aperture": camera.GetHorizontalApertureAttr().Get(),
        "step_dt": raw.step_dt,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[INFO] Camera screenshot: {path}")
    return path
